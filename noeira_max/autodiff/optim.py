"""Optimizers that emit their update into the train-step graph (plan §4, M2).

Parameters and optimizer state are buffers of the graph, updated in place by
``buffer_store``. The step counter is one of those buffers, so the learning
rate schedule and Adam's bias correction are computed on the device from it,
never on the host (RFC §6.4, capture conditions C1 and C2).

The arithmetic follows PyTorch's single-tensor implementations op for op
(``lerp``, ``addcmul``, ``addcdiv``), so the float64 parity test can be tight.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue, ops

from ._ops import scalar


# -- schedules ---------------------------------------------------------------


@dataclass(frozen=True)
class Constant:
    def scale(self, it: TensorValue) -> TensorValue:
        return scalar(1.0, it)

    def reference(self, it: int) -> float:
        return 1.0


@dataclass(frozen=True)
class WarmupCosine:
    """The torch twin's ``lr_at(it)`` (``it`` counts completed steps, from 0):
    a linear warmup to 1, then a cosine down to ``min_scale`` at ``total``."""

    warmup: int
    total: int
    min_scale: float = 0.1

    def scale(self, it: TensorValue) -> TensorValue:
        warm = (it + 1.0) * (1.0 / self.warmup)
        progress = ops.min(
            (it - float(self.warmup)) * (1.0 / max(1, self.total - self.warmup)),
            scalar(1.0, it),
        )
        cosine = self.min_scale + (1.0 - self.min_scale) * 0.5 * (
            1.0 + ops.cos(progress * math.pi)
        )
        return ops.where(ops.greater(scalar(float(self.warmup), it), it), warm, cosine)

    def reference(self, it: int) -> float:
        if it < self.warmup:
            return (it + 1) / self.warmup
        progress = min(1.0, (it - self.warmup) / max(1, self.total - self.warmup))
        return self.min_scale + (1 - self.min_scale) * 0.5 * (1 + math.cos(math.pi * progress))


# -- shared pieces -------------------------------------------------------------


def global_norm(grads: Mapping[str, TensorValue]) -> TensorValue:
    """``sqrt(sum of every gradient's squared entries)``, as a ``[1]`` tensor."""
    total = None
    for g in grads.values():
        sq = ops.reshape(ops.sum(ops.reshape(g * g, [-1]), axis=0), [1])
        total = sq if total is None else total + sq
    return ops.sqrt(total)


def clip_by_global_norm(
    grads: Mapping[str, TensorValue], max_norm: float
) -> tuple[dict[str, TensorValue], TensorValue]:
    """``torch.nn.utils.clip_grad_norm_``: scales every gradient by
    ``min(1, max_norm / (norm + 1e-6))``. Returns the clipped gradients and the
    norm before clipping."""
    norm = global_norm(grads)
    coef = ops.min(max_norm / (norm + 1e-6), scalar(1.0, norm))
    return {k: g * coef for k, g in grads.items()}, norm


def _load(buffer: BufferValue | Tensor) -> TensorValue:
    """A buffer's current value: a ``max.graph`` buffer, or a
    ``max.experimental`` tensor declared with a ``BufferLayout``."""
    if isinstance(buffer, Tensor):
        return buffer.__tensorvalue__()
    return ops.buffer_load(buffer)


def _store(buffer: BufferValue | Tensor, value: TensorValue) -> None:
    if isinstance(buffer, Tensor):
        F.buffer_store(buffer, Tensor.from_graph_value(value))
    else:
        ops.buffer_store(buffer, value)


@dataclass(frozen=True)
class _Optimizer:
    lr: float
    schedule: Constant | WarmupCosine = field(default_factory=Constant)
    clip_norm: float | None = None

    def init(self, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        dtype = next(iter(params.values())).dtype
        return {"step": np.zeros(1, dtype)}

    def _begin(self, state, grads):
        """Reads the counter, clips, and returns ``(it, lr, grads, norm)``."""
        it = _load(state["step"])
        lr = self.lr * self.schedule.scale(it)
        norm = None
        if self.clip_norm is not None:
            grads, norm = clip_by_global_norm(grads, self.clip_norm)
        return it, lr, grads, norm


def _store_all(updates: list[tuple[BufferValue | Tensor, TensorValue]]) -> None:
    """Every store after every load.

    The order is load-bearing for compile time: MAX 26.6 compiles a graph
    whose buffer loads and stores alternate (load b1, store b1, load b2, ...)
    in time that explodes with the count (30 buffers: 25 s, 40: over 240 s),
    while all loads, then all stores, stays flat (170 buffers: 11 s). See
    tests/test_max_findings.py.
    """
    for buffer, value in updates:
        _store(buffer, value)


@dataclass(frozen=True)
class SGD(_Optimizer):
    def apply(self, params, state, grads) -> dict[str, TensorValue]:
        it, lr, grads, norm = self._begin(state, grads)
        updates = [(b, _load(b) - lr * grads[name]) for name, b in params.items()]
        _store_all(updates + [(state["step"], it + 1.0)])
        return {"lr": lr} | ({"grad_norm": norm} if norm is not None else {})


@dataclass(frozen=True)
class AdamW(_Optimizer):
    """``torch.optim.AdamW``, decoupled weight decay on the parameters that
    ``decay(name, shape)`` selects."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    decay: Callable[[str, tuple], bool] = lambda name, shape: True

    def init(self, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        state = super().init(params)
        for name, value in params.items():
            state[f"m.{name}"] = np.zeros_like(value)
            state[f"v.{name}"] = np.zeros_like(value)
        return state

    def apply(self, params, state, grads) -> dict[str, TensorValue]:
        it, lr, grads, norm = self._begin(state, grads)
        b1, b2 = self.betas
        t = it + 1.0
        bias1 = 1.0 - ops.pow(scalar(b1, t), t)
        bias2_sqrt = ops.sqrt(1.0 - ops.pow(scalar(b2, t), t))
        step_size = lr / bias1
        updates = []
        for name, buffer in params.items():
            g = grads[name]
            p = _load(buffer)
            m = _load(state[f"m.{name}"])
            v = _load(state[f"v.{name}"])
            if self.weight_decay and self.decay(name, tuple(int(d) for d in p.shape)):
                p = p * (1.0 - lr * self.weight_decay)
            m = m + (g - m) * (1.0 - b1)  # m.lerp_(g, 1 - b1)
            v = v * b2 + g * g * (1.0 - b2)  # v.mul_(b2).addcmul_(g, g, 1 - b2)
            denom = ops.sqrt(v) / bias2_sqrt + self.eps
            updates += [
                (state[f"m.{name}"], m),
                (state[f"v.{name}"], v),
                (buffer, p - step_size * (m / denom)),
            ]
        _store_all(updates + [(state["step"], t)])
        return {"lr": lr} | ({"grad_norm": norm} if norm is not None else {})
