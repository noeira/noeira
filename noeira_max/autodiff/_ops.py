"""Small builder helpers shared by the transform and the rules.

Public ``max.graph`` API only. Every helper works with symbolic dims: none
turns a dim into a Python int unless it is static.
"""

from __future__ import annotations

from typing import Any

from max.graph import StaticDim, TensorValue, ops


def scalar(value: float, like: TensorValue) -> TensorValue:
    """A scalar constant with ``like``'s dtype and device."""
    return ops.constant(value, like.dtype, like.device)


def zeros_like(like: TensorValue) -> TensorValue:
    return ops.broadcast_to(scalar(0, like), like.shape)


def ones_like(like: TensorValue) -> TensorValue:
    return ops.broadcast_to(scalar(1, like), like.shape)


def mask(condition: TensorValue, like: TensorValue) -> TensorValue:
    """A boolean condition as 0/1 in ``like``'s dtype."""
    return ops.cast(condition, like.dtype)


def is_one(dim: Any) -> bool:
    return isinstance(dim, StaticDim) and int(dim) == 1


def static_size(dim: Any) -> int | None:
    return int(dim) if isinstance(dim, StaticDim) else None


def unbroadcast(ct: TensorValue, like: TensorValue) -> TensorValue:
    """Sums ``ct`` over the axes along which ``like`` was broadcast.

    ``ct`` has the broadcast result's shape: as many or more leading axes than
    ``like``, and possibly a larger extent where ``like`` has size 1.
    """
    target = list(like.shape)
    if list(ct.shape) == target:
        return ct
    extra = ct.rank - len(target)
    if extra < 0:
        raise ValueError(
            f"cannot unbroadcast a cotangent of shape {ct.shape} to {target}"
        )
    for _ in range(extra):
        ct = ops.squeeze(ops.sum(ct, axis=0), 0)
    for axis, (have, want) in enumerate(zip(list(ct.shape), target)):
        if is_one(want) and not is_one(have):
            ct = ops.sum(ct, axis=axis)
    if list(ct.shape) != target:
        # Same extents under different symbolic names: assert it at run time.
        ct = ops.rebind(ct, target)
    return ct


def mean(x: TensorValue, axis: int) -> TensorValue:
    """A mean along ``axis`` (kept) with an exact scale.

    MAX 26.6's float64 ``ops.mean`` multiplies by ``1/n`` rounded to float32:
    ``mean(ones(3))`` is ``1.0000000298023224``, i.e. ``3 * float32(1/3)``.
    Rules use this instead, so a cotangent is as precise as its dtype.
    """
    total = ops.sum(x, axis=axis)
    n = static_size(x.shape[axis])
    if n is not None:
        return total * (1.0 / n)
    return total / ops.sum(ones_like(x), axis=axis)


def sum_all(x: TensorValue) -> TensorValue:
    """Sums every axis, keeping each as size 1."""
    for axis in range(x.rank):
        x = ops.sum(x, axis=axis)
    return x
