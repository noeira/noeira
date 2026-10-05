"""Rule test cases, one family per compiled graph.

Shared by the MAX side (``harness.py``: gradcheck, golden dumps) and the
torch side (``golden_torch.py``, run in the ``act-ref`` env). This module
imports neither MAX nor torch: a case's ``max_fn`` receives a namespace with
the ``max.graph.ops`` functions and ``DType``; its ``torch_fn`` receives the
``torch`` module.

Shapes mix static extents (ints) and dim names. Within a family, a name is one
symbolic dim of the family's graph, so a single compile covers every binding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arg:
    shape: tuple
    domain: str = "normal"
    """normal | positive | away0 (|x| >= 0.2, off the kinks) | unit (|x| < 0.8)
    | int:<hi> (integer in [0, hi), never differentiated)."""
    diff: bool = True


@dataclass(frozen=True)
class Case:
    name: str
    args: tuple[Arg, ...]
    out: tuple
    """The output's shape: ints, dim names, or expressions over the names
    (``"n+2"``) where MAX derives a dim."""
    max_fn: Callable
    torch_fn: Callable | None = None
    eps: float = 1e-6
    """Finite-difference step (float64)."""
    rtol: float = 1e-6


@dataclass(frozen=True)
class Family:
    name: str
    cases: tuple[Case, ...]
    bindings: tuple[dict, ...]
    """Extents for the family's dim names; include size 1 and odd sizes."""


X = ("n", "m")


def _unary(name, domain, max_fn, torch_fn):
    return Case(name, (Arg(X, domain),), X, max_fn, torch_fn)


def _binary(name, a, b, out, max_fn, torch_fn, da="normal", db="normal"):
    return Case(name, (Arg(a, da), Arg(b, db)), out, max_fn, torch_fn)


ELEMENTWISE = Family(
    "elementwise",
    (
        _unary("negative", "normal", lambda o, x: o.negate(x), lambda t, x: -x),
        _unary("exp", "normal", lambda o, x: o.exp(x), lambda t, x: t.exp(x)),
        _unary("log", "positive", lambda o, x: o.log(x), lambda t, x: t.log(x)),
        _unary("log1p", "positive", lambda o, x: o.log1p(x), lambda t, x: t.log1p(x)),
        _unary("sqrt", "positive", lambda o, x: o.sqrt(x), lambda t, x: t.sqrt(x)),
        _unary("rsqrt", "positive", lambda o, x: o.rsqrt(x), lambda t, x: t.rsqrt(x)),
        _unary("tanh", "normal", lambda o, x: o.tanh(x), lambda t, x: t.tanh(x)),
        _unary("sigmoid", "normal", lambda o, x: o.sigmoid(x), lambda t, x: t.sigmoid(x)),
        _unary("relu", "away0", lambda o, x: o.relu(x), lambda t, x: t.relu(x)),
        _unary("abs", "away0", lambda o, x: o.abs(x), lambda t, x: t.abs(x)),
        _unary("sin", "normal", lambda o, x: o.sin(x), lambda t, x: t.sin(x)),
        _unary("cos", "normal", lambda o, x: o.cos(x), lambda t, x: t.cos(x)),
        _unary("erf", "normal", lambda o, x: o.erf(x), lambda t, x: t.erf(x)),
        _unary("atanh", "unit", lambda o, x: o.atanh(x), lambda t, x: t.atanh(x)),
        _unary("silu", "normal", lambda o, x: o.silu(x),
               lambda t, x: t.nn.functional.silu(x)),
        _unary("gelu", "normal", lambda o, x: o.gelu(x),
               lambda t, x: t.nn.functional.gelu(x)),
        _unary("gelu_tanh", "normal", lambda o, x: o.gelu(x, approximate="tanh"),
               lambda t, x: t.nn.functional.gelu(x, approximate="tanh")),
        _unary("gelu_quick", "normal", lambda o, x: o.gelu(x, approximate="quick"),
               lambda t, x: x * t.sigmoid(1.702 * x)),
        Case(
            "cast_f32_roundtrip",
            (Arg(X),),
            X,
            lambda o, x: o.cast(o.cast(x, o.DType.float32), o.DType.float64),
            lambda t, x: x.float().double(),
            eps=1e-2,  # through float32 rounding
            rtol=1e-3,
        ),
        _binary("add", X, X, X, lambda o, a, b: o.add(a, b), lambda t, a, b: a + b),
        _binary("add_bcast_row", X, (1, "m"), X,
                lambda o, a, b: o.add(a, b), lambda t, a, b: a + b),
        _binary("add_bcast_col", X, ("n", 1), X,
                lambda o, a, b: o.add(a, b), lambda t, a, b: a + b),
        _binary("add_bcast_rank", X, ("m",), X,
                lambda o, a, b: o.add(a, b), lambda t, a, b: a + b),
        _binary("sub_bcast_rank", ("m",), X, X,
                lambda o, a, b: o.sub(a, b), lambda t, a, b: a - b),
        _binary("mul", X, X, X, lambda o, a, b: o.mul(a, b), lambda t, a, b: a * b),
        _binary("mul_bcast_row", X, (1, "m"), X,
                lambda o, a, b: o.mul(a, b), lambda t, a, b: a * b),
        _binary("div", X, X, X, lambda o, a, b: o.div(a, b), lambda t, a, b: a / b,
                db="positive"),
        _binary("div_bcast_col", X, ("n", 1), X,
                lambda o, a, b: o.div(a, b), lambda t, a, b: a / b, db="positive"),
        _binary("pow", X, X, X, lambda o, a, b: o.pow(a, b), lambda t, a, b: a ** b,
                da="positive"),
        Case("pow_const", (Arg(X, "positive"),), X,
             lambda o, x: o.pow(x, 3.0), lambda t, x: x ** 3.0),
        _binary("maximum", X, X, X, lambda o, a, b: o.max(a, b),
                lambda t, a, b: t.maximum(a, b)),
        _binary("minimum_bcast_rank", X, ("m",), X, lambda o, a, b: o.min(a, b),
                lambda t, a, b: t.minimum(a, b)),
        Case(
            "where",
            (Arg(X), Arg(X), Arg(X, diff=False)),
            X,
            lambda o, a, b, c: o.where(o.greater(c, 0.0), a, b),
            lambda t, a, b, c: t.where(c > 0, a, b),
        ),
        Case(
            "where_bcast",
            (Arg(X), Arg(("m",)), Arg(X, diff=False)),
            X,
            lambda o, a, b, c: o.where(o.greater(c, 0.0), a, b),
            lambda t, a, b, c: t.where(c > 0, a, b),
        ),
    ),
    ({"n": 1, "m": 1}, {"n": 3, "m": 5}, {"n": 7, "m": 2}),
)


REDUCTION = Family(
    "reduction",
    (
        Case("sum_axis0", (Arg(X),), (1, "m"), lambda o, x: o.sum(x, axis=0),
             lambda t, x: x.sum(0, keepdim=True)),
        Case("sum_axis1", (Arg(X),), ("n", 1), lambda o, x: o.sum(x, axis=1),
             lambda t, x: x.sum(1, keepdim=True)),
        Case("sum_3d_mid", (Arg(("n", 3, "m")),), ("n", 1, "m"),
             lambda o, x: o.sum(x, axis=1), lambda t, x: x.sum(1, keepdim=True)),
        Case("mean_symbolic", (Arg(X),), ("n", 1), lambda o, x: o.mean(x, axis=1),
             lambda t, x: x.mean(1, keepdim=True)),
        Case("mean_static", (Arg(("n", 6)),), ("n", 1), lambda o, x: o.mean(x, axis=1),
             lambda t, x: x.mean(1, keepdim=True)),
        Case("max_axis1", (Arg(X),), ("n", 1), lambda o, x: o.max(x, axis=1),
             lambda t, x: x.amax(1, keepdim=True)),
        Case("min_axis0", (Arg(X),), (1, "m"), lambda o, x: o.min(x, axis=0),
             lambda t, x: x.amin(0, keepdim=True)),
    ),
    ({"n": 1, "m": 1}, {"n": 4, "m": 3}, {"n": 2, "m": 7}),
)


LINALG = Family(
    "linalg",
    (
        Case("mm", (Arg(("n", 4)), Arg((4, "p"))), ("n", "p"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
        Case("mm_symbolic_k", (Arg(("n", "p")), Arg(("p", "n"))), ("n", "n"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
        Case("bmm", (Arg(("b", "n", 4)), Arg(("b", 4, "p"))), ("b", "n", "p"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
        Case("bmm_2d_weight", (Arg(("b", "n", 4)), Arg((4, "p"))), ("b", "n", "p"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
        Case("bmm_2d_lhs", (Arg(("n", 4)), Arg(("b", 4, "p"))), ("b", "n", "p"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
        Case("bmm_batch1", (Arg(("b", "n", 4)), Arg((1, 4, "p"))), ("b", "n", "p"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
        Case("bmm_4d", (Arg(("b", 2, "n", 4)), Arg(("b", 2, 4, "n"))),
             ("b", 2, "n", "n"),
             lambda o, a, b: o.matmul(a, b), lambda t, a, b: a @ b),
    ),
    ({"n": 1, "p": 1, "b": 1}, {"n": 3, "p": 5, "b": 2}, {"n": 2, "p": 3, "b": 3}),
)


SHAPE = Family(
    "shape",
    (
        Case("reshape_split_dim", (Arg(("n", 6)),), ("n", 2, 3),
             lambda o, x: o.reshape(x, [x.shape[0], 2, 3]),
             lambda t, x: x.reshape(-1, 2, 3)),
        Case("reshape_merge_dims", (Arg(("n", 2, 3)),), ("n", 6),
             lambda o, x: o.reshape(x, [x.shape[0], 6]),
             lambda t, x: x.reshape(-1, 6)),
        Case("transpose", (Arg(X),), ("m", "n"), lambda o, x: o.transpose(x, 0, 1),
             lambda t, x: x.T),
        Case("permute_3d", (Arg(("n", 3, "m")),), ("m", "n", 3),
             lambda o, x: o.permute(x, [2, 0, 1]), lambda t, x: x.permute(2, 0, 1)),
        Case("broadcast_to_row", (Arg((1, "m")),), (3, "m"),
             lambda o, x: o.broadcast_to(x, [3, x.shape[1]]),
             lambda t, x: x.expand(3, -1)),
        Case("broadcast_to_rank", (Arg(("m",)),), (4, "m"),
             lambda o, x: o.broadcast_to(x, [4, x.shape[0]]),
             lambda t, x: x.expand(4, -1)),
        Case("concat_axis1", (Arg(("n", 2)), Arg(("n", 3))), ("n", 5),
             lambda o, a, b: o.concat([a, b], axis=1),
             lambda t, a, b: t.cat([a, b], 1)),
        Case("concat_axis0_symbolic", (Arg((2, "m")), Arg(("n", "m"))), ("n+2", "m"),
             lambda o, a, b: o.concat([a, b], axis=0),
             lambda t, a, b: t.cat([a, b], 0)),
        Case(
            "split_both_used",
            (Arg(("n", 5)),),
            ("n", 5),
            lambda o, x: (lambda p: o.concat([o.exp(p[1]), p[0]], axis=1))(
                o.split(x, [2, 3], axis=1)),
            lambda t, x: (lambda p: t.cat([t.exp(p[1]), p[0]], 1))(x.split([2, 3], 1)),
        ),
        Case("split_one_unused", (Arg(("n", 5)),), ("n", 3),
             lambda o, x: o.exp(o.split(x, [2, 3], axis=1)[1]),
             lambda t, x: t.exp(x.split([2, 3], 1)[1])),
        Case("slice_columns", (Arg(("n", 6)),), ("n", 3),
             lambda o, x: o.slice_tensor(x, [slice(None), slice(1, 4)]),
             lambda t, x: x[:, 1:4]),
        Case("slice_negative_rows", (Arg((5, "m")),), (3, "m"),
             lambda o, x: o.slice_tensor(x, [slice(-3, None), slice(None)]),
             lambda t, x: x[-3:]),
    ),
    ({"n": 1, "m": 1}, {"n": 3, "m": 5}, {"n": 4, "m": 2}),
)


INDEXING = Family(
    "indexing",
    (
        Case("gather_rows", (Arg((7, 4)), Arg(("n",), "int:7", diff=False)), ("n", 4),
             lambda o, w, i: o.gather(w, i, axis=0), lambda t, w, i: w[i]),
        Case("gather_embedding",
             (Arg((7, 4)), Arg(("n", "t"), "int:7", diff=False)), ("n", "t", 4),
             lambda o, w, i: o.gather(w, i, axis=0), lambda t, w, i: w[i]),
        Case("gather_axis1", (Arg((3, 7)), Arg(("n",), "int:7", diff=False)), (3, "n"),
             lambda o, w, i: o.gather(w, i, axis=1), lambda t, w, i: w[:, i]),
    ),
    # n > 7 draws repeated indices: the scatter must add, not overwrite.
    ({"n": 1, "t": 1}, {"n": 9, "t": 3}, {"n": 12, "t": 2}),
)


def _one_hot(o, y, classes):
    labels = o.constant(np.arange(classes), o.DType.int64, y.device)
    return o.cast(o.equal(o.unsqueeze(y, -1), labels), o.DType.float64)


def _causal_mask(o, n, device):
    return o.constant(np.tril(np.ones((n, n), dtype=bool)), o.DType.bool, device)


def _dropout_fixed_seed(o, x):
    """Inverted dropout with a constant seed: the same mask on every
    execution, so finite differences see a fixed linear map."""
    o.random.set_seed(7)
    draw = o.random.uniform(o.TensorType(o.DType.float32, x.shape, x.device))
    return x * o.cast(o.greater_equal(draw, 0.5), x.dtype) * 2.0


def _attention(o, q, k, v):
    n = int(q.shape[0])
    scores = o.matmul(q, o.transpose(k, 0, 1)) * (1.0 / np.sqrt(int(q.shape[1])))
    neg_inf = o.constant(-np.inf, o.DType.float64, q.device)
    return o.matmul(o.softmax(o.where(_causal_mask(o, n, q.device), scores, neg_inf)), v)


NN = Family(
    "nn",
    (
        Case("softmax", (Arg(X),), X, lambda o, x: o.softmax(x),
             lambda t, x: t.softmax(x, -1)),
        Case("logsoftmax", (Arg(X),), X, lambda o, x: o.logsoftmax(x),
             lambda t, x: t.log_softmax(x, -1)),
        Case("layer_norm", (Arg(("n", 6)), Arg((6,)), Arg((6,))), ("n", 6),
             lambda o, x, w, b: o.layer_norm(x, w, b, epsilon=1e-5),
             lambda t, x, w, b: t.nn.functional.layer_norm(x, (6,), w, b, eps=1e-5)),
        Case("layer_norm_3d", (Arg(("n", 3, 6)), Arg((6,)), Arg((6,))), ("n", 3, 6),
             lambda o, x, w, b: o.layer_norm(x, w, b, epsilon=1e-5),
             lambda t, x, w, b: t.nn.functional.layer_norm(x, (6,), w, b, eps=1e-5)),
        Case(
            "cross_entropy",
            (Arg(("n", 5)), Arg(("n",), "int:5", diff=False)),
            (1, 1),
            lambda o, z, y: -o.mean(
                o.sum(o.logsoftmax(z) * _one_hot(o, y, 5), axis=1), axis=0),
            lambda t, z, y: t.nn.functional.cross_entropy(z, y).reshape(1, 1),
        ),
        # No torch twin: torch draws other masks.
        Case("dropout_fixed_seed", (Arg(X),), X, _dropout_fixed_seed),
        Case(
            "causal_attention",
            (Arg((5, 4)), Arg((5, 4)), Arg((5, 4))),
            (5, 4),
            _attention,
            lambda t, q, k, v: t.nn.functional.scaled_dot_product_attention(
                q[None], k[None], v[None], is_causal=True)[0],
        ),
    ),
    ({"n": 1, "m": 1}, {"n": 4, "m": 5}, {"n": 3, "m": 2}),
)


# Several rules composed: fan-out (a value used twice), a broadcast bias in a
# layer, and a chain through every family.
COMPOSITE = Family(
    "composite",
    (
        Case("used_twice", (Arg(X),), X, lambda o, x: o.add(o.mul(x, x), x),
             lambda t, x: x * x + x),
        Case("used_thrice_through_exp", (Arg(X),), X,
             lambda o, x: o.mul(o.exp(x), o.sub(x, o.tanh(x))),
             lambda t, x: t.exp(x) * (x - t.tanh(x))),
        Case(
            "dense_layer",
            (Arg(("n", 4)), Arg((4, 3)), Arg((3,))),
            ("n", 3),
            lambda o, x, w, b: o.tanh(o.add(o.matmul(x, w), b)),
            lambda t, x, w, b: t.tanh(x @ w + b),
        ),
        Case(
            "two_layer_mean",
            (Arg(("n", 4)), Arg((4, 5)), Arg((5, 2))),
            (1, 1),
            lambda o, x, w1, w2: o.mean(
                o.mean(o.matmul(o.gelu(o.matmul(x, w1), approximate="tanh"), w2),
                       axis=0), axis=1),
            lambda t, x, w1, w2: (
                t.nn.functional.gelu(x @ w1, approximate="tanh") @ w2
            ).mean().reshape(1, 1),
        ),
    ),
    ({"n": 1, "m": 1}, {"n": 5, "m": 3}),
)


FAMILIES = (ELEMENTWISE, REDUCTION, LINALG, SHAPE, INDEXING, NN, COMPOSITE)
