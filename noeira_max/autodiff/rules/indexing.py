"""Indexing: a gather's backward scatters the cotangent back, adding where an
index repeats (an embedding row used twice in a batch)."""

from __future__ import annotations

from max.graph import ops

from .._ops import zeros_like
from ..registry import defvjp


@defvjp("rmo.mo.gather")
def _gather(ctx):
    table, indices = ctx.inputs
    axis = ctx.op.axis
    g = ctx.ct
    if axis != 0:
        # Move the gathered axis to the front, scatter, and move it back.
        perm = [axis] + [i for i in range(table.rank) if i != axis]
        inverse = [perm.index(i) for i in range(table.rank)]
        moved = ops.permute(table, perm)
        g = ops.permute(g, _gather_perm(table.rank, indices.rank, axis))
        dmoved = ops.scatter_nd_add(
            zeros_like(moved), g, ops.unsqueeze(indices, -1)
        )
        return [ops.permute(dmoved, inverse), None]
    dtable = ops.scatter_nd_add(zeros_like(table), g, ops.unsqueeze(indices, -1))
    return [dtable, None]


def _gather_perm(table_rank: int, index_rank: int, axis: int) -> list[int]:
    """The permutation taking gather's output layout (``table[:axis]``,
    ``indices``, ``table[axis+1:]``) to scatter's (``indices``,
    ``table[:axis]``, ``table[axis+1:]``)."""
    lead = list(range(axis))
    idx = list(range(axis, axis + index_rank))
    tail = list(range(axis + index_rank, table_rank - 1 + index_rank))
    return idx + lead + tail
