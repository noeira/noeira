"""Reductions. MAX keeps the reduced axis (size 1), so a cotangent broadcasts
straight back to the input's shape."""

from __future__ import annotations

from max.graph import ops

from .._ops import mask, ones_like, static_size
from ..registry import defvjp, nondiff

nondiff("rmo.mo.reduce.arg_max")


@defvjp("rmo.mo.reduce.add")
def _sum(ctx):
    x = ctx.inputs[0]
    return [ops.broadcast_to(ctx.ct, x.shape)]


@defvjp("rmo.mo.reduce.mean")
def _mean(ctx):
    x = ctx.inputs[0]
    axis = ctx.op.axis
    n = static_size(x.shape[axis])
    if n is not None:
        scaled = ctx.ct * (1.0 / n)
    else:  # a symbolic extent: count it in the graph
        scaled = ctx.ct / ops.sum(ones_like(x), axis=axis)
    return [ops.broadcast_to(scaled, x.shape)]


def _extremum(ctx):
    """Splits the cotangent evenly among the elements equal to the extremum
    (PyTorch's ``amax`` / ``amin`` convention)."""
    x = ctx.inputs[0]
    axis = ctx.op.axis
    hit = mask(ops.equal(x, ops.broadcast_to(ctx.outputs[0], x.shape)), x)
    share = ctx.ct / ops.sum(hit, axis=axis)
    return [ops.broadcast_to(share, x.shape) * hit]


defvjp("rmo.mo.reduce.max")(_extremum)
defvjp("rmo.mo.reduce.min")(_extremum)
