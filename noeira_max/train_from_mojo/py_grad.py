"""The Python half of ``max_train``: differentiation, and the gate's reference.

A Mojo program builds its model, loss and optimizer into a ``max.graph``
graph through ``max_graph_gen``. Differentiation is the one step it hands to
Python: ``backward`` runs the autodiff prototype's transform
(``noeira_max/autodiff/transform.py``) over the ops the Mojo program emitted
since ``snapshot``, and emits their VJPs into the same graph. That is
``value_and_grad`` with the function call taken out: the Mojo side has already
emitted the forward pass.

The rest serves the gate (``train_mlp.mojo``): the problem's initial weights
and batch, and the same train step built and run by the Python prototype.
"""

from __future__ import annotations

import ctypes

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff import transform
from noeira_max.autodiff._ops import is_one, ones_like, sum_all
from noeira_max.autodiff.bench.mlp_step import mse
from noeira_max.autodiff.models import gpt, mlp
from noeira_max.autodiff.models.common import KERNELS, dense, layer_norm_kernel
from noeira_max.autodiff.optim import AdamW
from noeira_max.autodiff.train import CompiledStep, build_train_step
from noeira_max.graph_mojo.py_backend import describe


def snapshot(graph: Graph) -> set:
    """The ops already in ``graph``: everything emitted after this is what
    ``backward`` differentiates."""
    return transform._op_ids(graph)


def backward(graph: Graph, before: set, primals: list, loss) -> list:  # noqa: ANN001
    """Emits the gradient of the scalar ``loss`` with respect to ``primals``
    into ``graph``, from the ops emitted since ``before``. Returns one
    ``(value, describe(value))`` per primal, in their order.

    The same steps as ``transform.value_and_grad`` after its call of the
    function: the tape, the all-ones seed, the backward walk.
    """
    primals = list(primals)
    tape = transform._Tape(graph, before, inputs=primals, outputs=[loss])
    if not all(is_one(d) for d in loss.shape):
        raise TypeError(f"backward needs a scalar loss (every dim 1); got shape {loss.shape}")
    grads = tape.backward([ones_like(loss)])
    return [(g, describe(g)) for g in grads]


def graph_text(graph: Graph) -> str:
    return str(graph)


# -- the gate's problem and reference -------------------------------------


def problem(dims: list, batch: int, seed: int) -> dict:
    """The initial weights (``mlp.init``'s He-uniform, zero biases) and one
    batch, float32, drawn as ``bench/mlp_step.py`` draws them."""
    rng = np.random.default_rng(seed)
    init = mlp.init(rng, dims=tuple(dims), dtype=np.float32)
    x = rng.standard_normal((batch, dims[0])).astype(np.float32)
    y = rng.standard_normal((batch, dims[-1])).astype(np.float32)
    return {"init": init, "x": x, "y": y}


def address(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def array_at(addr: int, shape: list) -> np.ndarray:
    """A copy of the float32 array at ``addr`` (memory owned by the Mojo
    program)."""
    n = int(np.prod(shape))
    raw = (ctypes.c_float * n).from_address(addr)
    return np.ctypeslib.as_array(raw).reshape(shape).copy()


def reference(dims: list, batch: int, seed: int, steps: int, name: str, act: str,
              lr: float, device: str, init: dict | None = None) -> dict:
    """The same train step built by the Python prototype (``build_train_step``
    with ``mlp_step.py``'s MSE and ``AdamW``), compiled and run ``steps``
    times from the same weights (``problem``'s, or ``init``) and batch."""
    data = problem(dims, batch, seed)
    if init is not None:
        data["init"] = {k: np.array(v, np.float32) for k, v in init.items()}
    depth = len(dims) - 1
    return _run(lambda p, xv, yv: mse(p, xv, yv, depth, act), data, steps, name, lr, device)


def _run(loss_fn, data: dict, steps: int, name: str, lr: float, device: str) -> dict:  # noqa: ANN001
    dev_obj = Accelerator() if device == "gpu" else CPU()
    dev = DeviceRef.from_device(dev_obj)
    opt = AdamW(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    step = build_train_step(
        loss_fn, data["init"], opt,
        [TensorType(DType.from_numpy(data["x"].dtype), data["x"].shape, dev),
         TensorType(DType.from_numpy(data["y"].dtype), data["y"].shape, dev)],
        dev, name=name,
    )
    compiled = CompiledStep(step, data["init"], opt, dev_obj)
    x = Buffer.from_numpy(data["x"]).to(dev_obj)
    y = Buffer.from_numpy(data["y"]).to(dev_obj)
    losses = [float(compiled(x, y)[0].to_numpy().item()) for _ in range(steps)]
    return {
        "text": str(step.graph),
        "compile_s": compiled.compile_seconds,
        "losses": losses,
        "params": compiled.host_params(),
        "state": compiled.host_state(),
    }


def compare(name: str, got: np.ndarray, want: np.ndarray) -> tuple:
    """``(elements that differ, max relative difference)``."""
    got = np.asarray(got, np.float32).reshape(want.shape)
    differ = int(np.count_nonzero(got != want))
    scale = max(float(np.max(np.abs(want))), 1e-30)
    return differ, float(np.max(np.abs(got.astype(np.float64) - want))) / scale


def forward_error(got: np.ndarray, params: dict, x: np.ndarray, act: str, layers: int = 0) -> float:
    """The largest difference between ``got`` and the MLP's forward pass in
    float64 from ``params`` (its first ``layers`` layers; 0: all), relative
    to the output's largest value."""
    h = x.astype(np.float64)
    depth = len(params) // 2
    for i in range(layers or depth):
        h = h @ params[f"w{i}"].astype(np.float64) + params[f"b{i}"].astype(np.float64)
        if i < depth - 1:
            h = np.maximum(h, 0.0) if act == "relu" else np.tanh(h)
    got = np.asarray(got, np.float64).reshape(h.shape)
    return float(np.max(np.abs(got - h)) / max(float(np.max(np.abs(h))), 1e-30))


def to_list(array: np.ndarray) -> list:
    return np.asarray(array, np.float32).ravel().tolist()


def forward_change(before: dict, after: dict, x: np.ndarray, act: str) -> float:
    """How far training moved the MLP's output on ``x``, relative to its
    largest value: the scale a forward check on the trained weights must
    resolve."""
    def forward(params: dict) -> np.ndarray:
        h = x.astype(np.float64)
        depth = len(params) // 2
        for i in range(depth):
            h = h @ np.asarray(params[f"w{i}"], np.float64) + np.asarray(params[f"b{i}"], np.float64)
            if i < depth - 1:
                h = np.maximum(h, 0.0) if act == "relu" else np.tanh(h)
        return h
    a, b = forward(before), forward(after)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


# -- a model with a Mojo kernel pair: x -> dense -> LayerNorm (custom op) ->
# ReLU -> dense, MSE ---------------------------------------------------------

LN_EPS = 1e-5


def kernels_path() -> str:
    """The autodiff prototype's Mojo kernel package (LayerNorm, attention)."""
    return str(KERNELS)


def ln_problem(dims: list, batch: int, seed: int) -> dict:
    """Weights in the step's input order (w0, b0, g0, beta0, w1, b1), He
    uniform and zero biases, LayerNorm's gamma 1 and beta 0; and one batch."""
    i, h, o = dims
    rng = np.random.default_rng(seed)
    init = {
        "w0": rng.uniform(-np.sqrt(6 / i), np.sqrt(6 / i), (i, h)).astype(np.float32),
        "b0": np.zeros(h, np.float32),
        "g0": np.ones(h, np.float32),
        "beta0": np.zeros(h, np.float32),
        "w1": rng.uniform(-np.sqrt(6 / h), np.sqrt(6 / h), (h, o)).astype(np.float32),
        "b1": np.zeros(o, np.float32),
    }
    x = rng.standard_normal((batch, i)).astype(np.float32)
    y = rng.standard_normal((batch, o)).astype(np.float32)
    return {"init": init, "x": x, "y": y}


def ln_loss(p, x, y):  # noqa: ANN001, ANN201
    h = dense(x, p["w0"], p["b0"])
    h = layer_norm_kernel(h, p["g0"], p["beta0"], LN_EPS)
    h = ops.relu(h)
    d = dense(h, p["w1"], p["b1"]) - y
    return sum_all(d * d) * (1.0 / (int(y.shape[0]) * int(y.shape[1])))


def ln_reference(dims: list, batch: int, seed: int, steps: int, name: str, lr: float,
                 device: str) -> dict:
    return _run(ln_loss, ln_problem(dims, batch, seed), steps, name, lr, device)


# -- the GPT: noeira's LayerNorm and attention kernel pairs, a fixed batch --


def gpt_config(vocab: int, seq: int, dim: int, heads: int, layers: int) -> gpt.Config:
    return gpt.Config(vocab=vocab, seq=seq, dim=dim, heads=heads, layers=layers,
                      layer_norm="kernel", attention="kernel")


def gpt_problem(cfg: list, batch: int, seed: int) -> dict:
    """``gpt.init``'s weights, in its order, and one batch of token ids and
    next-token targets (random: the gate needs a batch, not a corpus)."""
    config = gpt_config(*cfg)
    rng = np.random.default_rng(seed)
    init = gpt.init(config, rng, dtype=np.float32)
    idx = rng.integers(0, config.vocab, (batch, config.seq)).astype(np.int64)
    targets = rng.integers(0, config.vocab, (batch, config.seq)).astype(np.int64)
    return {"init": init, "x": idx, "y": targets, "names": list(init),
            "shapes": [list(v.shape) for v in init.values()]}


def gpt_reference(cfg: list, batch: int, seed: int, steps: int, name: str, lr: float,
                  device: str) -> dict:
    config = gpt_config(*cfg)
    return _run(lambda p, i, t: gpt.loss(p, i, t, config), gpt_problem(cfg, batch, seed),
                steps, name, lr, device)


def sorted_order(names: list) -> list:
    """The indices of ``names`` in sorted order: the order in which the
    transform loads a dict of parameter buffers."""
    return sorted(range(len(names)), key=lambda i: names[i])
