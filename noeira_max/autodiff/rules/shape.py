"""Shape ops: each backward is the matching inverse movement of the cotangent.

Several of these ops keep their parameters outside the typed properties: a
transpose's permutation and ``split``'s sizes are constant operands, and
``rmo.concat`` has no typed binding at all.
"""

from __future__ import annotations

import numpy as np
from max.graph import ops

from .. import _graph
from .._ops import static_size, unbroadcast, zeros_like
from ..registry import defvjp


@defvjp("rmo.reshape")
def _reshape(ctx):
    return [ops.reshape(ctx.ct, ctx.inputs[0].shape)]


@defvjp("rmo.mo.transpose")
def _transpose(ctx):
    perm = [int(p) for p in ctx.const(1)]
    inverse = [int(i) for i in np.argsort(perm)]
    return [ops.permute(ctx.ct, inverse), None]


@defvjp("rmo.broadcast_to")
def _broadcast_to(ctx):
    return [unbroadcast(ctx.ct, ctx.inputs[0])] + [None] * (len(ctx.inputs) - 1)


@defvjp("rmo.concat")
def _concat(ctx):
    """Cuts the cotangent back into the inputs' extents.

    ``ops.split`` takes static sizes only, and ``slice_tensor`` accepts a
    symbolic extent only as the output size of a slice with static bounds. So
    a symbolic extent is supported in the last piece only.
    """
    axis = ctx.attr("axis")
    g = ctx.ct
    sizes = [x.shape[axis] for x in ctx.inputs]
    static = [static_size(s) for s in sizes]
    if all(s is not None for s in static):
        pieces = ops.split(g, static, axis)
    elif all(s is not None for s in static[:-1]):
        pieces, offset = [], 0
        for size, dim in zip(static, sizes):
            stop = None if size is None else offset + size
            index = [slice(None)] * g.rank
            index[axis] = (slice(offset, stop), dim)
            pieces.append(ops.slice_tensor(g, index))
            offset = stop or offset
    else:
        raise NotImplementedError(
            "rmo.concat backward with a symbolic extent before the last "
            f"input (extents {sizes}): the builder cannot slice at a symbolic "
            "offset"
        )
    return [p if need else None for p, need in zip(pieces, ctx.needs)]


@defvjp("mo.split")
def _split(ctx):
    pieces = [
        ct if ct is not None else zeros_like(out)
        for ct, out in zip(ctx.cts, ctx.outputs)
    ]
    return [ops.concat(pieces, axis=ctx.op.axis), None]


@defvjp("rmo.slice")
def _slice(ctx):
    """Pads the cotangent back to the input's extent (unit steps only)."""
    x = ctx.inputs[0]
    out = ctx.outputs[0]
    starts = _graph.shape_attr(ctx.op, "starts")
    steps = _graph.shape_attr(ctx.op, "steps")
    paddings = []
    for axis, (start, step) in enumerate(zip(starts, steps)):
        if out.shape[axis] == x.shape[axis]:
            paddings += [0, 0]  # untouched axis, possibly symbolic
            continue
        extent, kept = static_size(x.shape[axis]), static_size(out.shape[axis])
        if step != 1 or start is None or extent is None or kept is None:
            raise NotImplementedError(
                "rmo.slice backward needs unit steps and static extents on "
                f"sliced axes (axis {axis}: start {start}, step {step})"
            )
        before = start + extent if start < 0 else start
        paddings += [before, extent - before - kept]
    return [ops.pad(ctx.ct, paddings)]
