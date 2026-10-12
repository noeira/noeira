"""Indexing: a gather's backward scatters the cotangent back, adding where an
index repeats (an embedding row used twice in a batch)."""

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import TensorValue, ops

from .._ops import scalar, static_size, zeros_like
from ..registry import defvjp

# Elements of the table copies a spread scatter may allocate (16 MiB of
# float32); see ``_scatter_rows``.
SPREAD_BUDGET = 1 << 22


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
        return [ops.permute(_scatter_rows(moved, g, indices), inverse), None]
    return [_scatter_rows(table, g, indices), None]


def _scatter_rows(table: TensorValue, g: TensorValue, indices: TensorValue) -> TensorValue:
    """Zeros shaped like ``table`` plus the rows of ``g`` added at ``indices``.

    MAX's ``scatter_nd_add`` adds element by element with atomics, so its time
    grows with how often an index repeats. On the 5090 the GPT's [64, 256, 384]
    embedding cotangent takes 18.7 ms on Shakespeare text (65 characters, a
    space is ~15% of them) against 0.94 ms on uniform indices
    (``bench/kernel_probe.py``). Here position ``i`` adds into copy ``i mod k``
    of the table and the copies are summed: contention drops ``k``-fold.
    ``k`` is the largest power of two that divides the index count and keeps
    the copies within ``SPREAD_BUDGET``; a table too large for two copies, or a
    symbolic shape, takes the plain scatter.
    """
    n = math.prod(static_size(d) or 0 for d in indices.shape)
    size = math.prod(static_size(d) or 0 for d in table.shape)
    k = 1
    while n and size and n % (2 * k) == 0 and 2 * k * size <= SPREAD_BUDGET:
        k *= 2
    if k == 1:
        return ops.scatter_nd_add(zeros_like(table), g, ops.unsqueeze(indices, -1))
    copies = ops.range(0, k, 1, out_dim=k, device=table.device, dtype=DType.int64)
    copy = ops.reshape(ops.broadcast_to(ops.unsqueeze(copies, 0), [n // k, k]), [n])
    rows = ops.cast(ops.reshape(indices, [n]), DType.int64)
    spread = ops.scatter_nd_add(
        ops.broadcast_to(scalar(0, table), [k, *table.shape]),
        ops.reshape(g, [n, *table.shape[1:]]),
        ops.stack([copy, rows], axis=-1),
    )
    return ops.squeeze(ops.sum(spread, axis=0), 0)


def _gather_perm(table_rank: int, index_rank: int, axis: int) -> list[int]:
    """The permutation taking gather's output layout (``table[:axis]``,
    ``indices``, ``table[axis+1:]``) to scatter's (``indices``,
    ``table[:axis]``, ``table[axis+1:]``)."""
    lead = list(range(axis))
    idx = list(range(axis, axis + index_rank))
    tail = list(range(axis + index_rank, table_rank - 1 + index_rank))
    return idx + lead + tail
