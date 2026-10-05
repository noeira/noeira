"""Gradcheck harness: one compiled graph per family of cases.

For each case the graph computes ``L = sum(y * r)`` and its gradient with
respect to the case's differentiable inputs, through ``value_and_grad``.
``r`` is a graph input, so ``∇L = J^T r`` is the VJP the golden test compares
against ``torch.func.vjp``.

The check is a central directional difference in float64: along a random
direction ``v``, ``(L(x + εv) - L(x - εv)) / 2ε`` must match ``<∇L, v>``.
That costs two executions per direction for the whole family, rather than two
per input element; dims are symbolic, so one compile covers every binding.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff import value_and_grad
from noeira_max.autodiff._ops import sum_all

from .cases import Arg, Case, Family

# What a case's ``max_fn`` sees as its first argument.
OPS = types.SimpleNamespace(
    **{n: getattr(ops, n) for n in dir(ops) if not n.startswith("_")}
)
OPS.DType = DType

DIRECTIONS = 3


def resolve(shape: tuple, binding: dict) -> tuple[int, ...]:
    """A shape's extents under ``binding``; entries may be expressions."""
    return tuple(
        d if isinstance(d, int) else int(eval(d, {}, dict(binding)))
        for d in shape
    )


def declared(shape: tuple, tag: str) -> tuple:
    """A shape as a graph input can declare it: an expression (``"n+2"``) is
    not a dim name, so it becomes a fresh one, rebound to the op's output."""
    return tuple(
        d if isinstance(d, int) or d.isidentifier() else f"{tag}d{i}"
        for i, d in enumerate(shape)
    )


def sample(arg: Arg, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if arg.domain.startswith("int:"):
        return rng.integers(0, int(arg.domain[4:]), size=shape, dtype=np.int64)
    if arg.domain == "normal":
        return rng.standard_normal(shape)
    if arg.domain == "positive":
        return rng.uniform(0.5, 2.0, shape)
    if arg.domain == "away0":
        return rng.choice([-1.0, 1.0], shape) * rng.uniform(0.2, 1.5, shape)
    if arg.domain == "unit":
        return rng.uniform(-0.8, 0.8, shape)
    raise ValueError(f"unknown domain {arg.domain!r}")


def _dtype(arg: Arg) -> DType:
    return DType.int64 if arg.domain.startswith("int:") else DType.float64


@dataclass
class Failure:
    case: str
    binding: dict
    fd: float
    analytic: float
    error: float


class FamilyGraph:
    """A family's cases, built into one graph and compiled once."""

    def __init__(self, family: Family, session: InferenceSession | None = None):
        self.family = family
        dev = DeviceRef.CPU()
        types_ = []
        for i, case in enumerate(family.cases):
            types_ += [TensorType(_dtype(a), a.shape, dev) for a in case.args]
            types_.append(
                TensorType(DType.float64, declared(case.out, f"out{i}"), dev)
            )

        with Graph(f"gradcheck_{family.name}", input_types=types_) as graph:
            inputs = iter(graph.inputs)
            outputs = []
            for case in family.cases:
                xs = [next(inputs) for _ in case.args]
                r = next(inputs)
                outputs += self._loss_and_grads(case, xs, r)
            graph.output(*outputs)
        self.graph = graph

        session = session or InferenceSession(devices=[CPU()])
        start = time.perf_counter()
        self.model = session.load(graph)
        self.compile_seconds = time.perf_counter() - start

    @staticmethod
    def _loss_and_grads(case: Case, xs: list, r) -> list:
        """``L``, the op's output ``y``, then ``∇L`` per differentiable arg."""
        diff = [i for i, a in enumerate(case.args) if a.diff]

        def loss(*diff_values):
            full = list(xs)
            for i, value in zip(diff, diff_values):
                full[i] = value
            y = case.max_fn(OPS, *full)
            weight = r if list(r.shape) == list(y.shape) else ops.rebind(r, y.shape)
            return sum_all(y * weight), y

        (value, y), grads = value_and_grad(
            loss, argnums=tuple(range(len(diff))), has_aux=True
        )(*(xs[i] for i in diff))
        return [value, y, *grads]

    # -- execution ---------------------------------------------------------

    def inputs(self, binding: dict, seed: int) -> list[list[np.ndarray]]:
        """Per case: its arguments, then ``r``."""
        rng = np.random.default_rng(seed)
        per_case = []
        for case in self.family.cases:
            arrays = [sample(a, resolve(a.shape, binding), rng) for a in case.args]
            arrays.append(rng.standard_normal(resolve(case.out, binding)))
            per_case.append(arrays)
        return per_case

    def run(self, per_case: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        """Per case: ``L``, ``y``, then one gradient per differentiable arg."""
        flat = [Buffer.from_numpy(np.ascontiguousarray(a)) for c in per_case for a in c]
        out = [b.to_numpy() for b in self.model.execute(*flat)]
        per_case_out = []
        for case in self.family.cases:
            n = 2 + sum(a.diff for a in case.args)
            per_case_out.append(out[:n])
            out = out[n:]
        return per_case_out

    def gradcheck(self, binding: dict, seed: int = 0) -> tuple[list[Failure], dict]:
        """Returns the failures, and the worst relative error per case."""
        base_in = self.inputs(binding, seed)
        base_out = self.run(base_in)
        rng = np.random.default_rng(seed + 1)
        worst = {case.name: 0.0 for case in self.family.cases}
        failures = []
        for _ in range(DIRECTIONS):
            dirs = [
                [rng.standard_normal(a.shape) if arg.diff else None
                 for arg, a in zip(case.args, arrays)]
                for case, arrays in zip(self.family.cases, base_in)
            ]
            plus = self._shift(base_in, dirs, +1)
            minus = self._shift(base_in, dirs, -1)
            out_p, out_m = self.run(plus), self.run(minus)
            for case, d, base, p, m in zip(
                self.family.cases, dirs, base_out, out_p, out_m
            ):
                fd = (p[0].item() - m[0].item()) / (2 * case.eps)
                vs = [v for v in d if v is not None]
                terms = [g * v for g, v in zip(base[2:], vs)]
                analytic = float(sum(t.sum() for t in terms))
                scale = max(float(sum(np.abs(t).sum() for t in terms)), 1e-12)
                error = abs(fd - analytic) / scale
                worst[case.name] = max(worst[case.name], error)
                if not error <= case.rtol:  # also catches NaN
                    failures.append(Failure(case.name, binding, fd, analytic, error))
        return failures, worst

    def _shift(self, base_in, dirs, sign):
        shifted = []
        for case, arrays, d in zip(self.family.cases, base_in, dirs):
            shifted.append([
                a + sign * case.eps * v if v is not None else a
                for a, v in zip(arrays[:-1], d)
            ] + [arrays[-1]])
        return shifted
