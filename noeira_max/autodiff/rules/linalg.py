"""Matrix multiplication (2-D, batched, and broadcast over batch dims).

``rmo.matmul`` has no transpose flags, so every product with a transposed
operand is an explicit ``transpose`` followed by a ``matmul``: whether the
compiler folds the transpose into the GEMM is an M2 measurement (RFC K1).
"""

from __future__ import annotations

from max.graph import ops

from .._ops import unbroadcast
from ..registry import defvjp


def _t(x):
    return ops.transpose(x, -1, -2)


@defvjp("rmo.matmul")
def _matmul(ctx):
    a, b = ctx.inputs
    g = ctx.ct
    if a.rank < 2 or b.rank < 2:
        raise NotImplementedError("rmo.matmul with a rank-1 operand")

    da = db = None
    if ctx.needs[0]:
        da = unbroadcast(ops.matmul(g, _t(b)), a)
    if ctx.needs[1]:
        if b.rank == 2 and a.rank > 2:
            # A 2-D weight shared across batch dims: one GEMM over the
            # flattened batch, rather than a batched GEMM and a reduction.
            k, n = a.shape[-1], g.shape[-1]
            a2 = ops.reshape(a, [-1, k])
            g2 = ops.reshape(g, [-1, n])
            db = ops.matmul(_t(a2), g2)
        else:
            db = unbroadcast(ops.matmul(_t(a), g), b)
    return [da, db]
