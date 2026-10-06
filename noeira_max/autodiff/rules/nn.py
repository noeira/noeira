"""Fused NN ops, with composite rules.

``mo.reduce.layer_norm`` returns ``y`` only, so its rule recomputes the mean
and the reciprocal standard deviation from the input: a MAX op cannot hand
residuals to its backward. The kernel-backed rule (``rules/custom.py``) takes
them from the forward instead.
"""

from __future__ import annotations

from max.graph import ops

from .._ops import mean, unbroadcast
from ..registry import defvjp


@defvjp("rmo.mo.reduce.softmax")
def _softmax(ctx):
    y = ctx.outputs[0]
    g = ctx.ct
    return [y * (g - ops.sum(g * y, axis=ctx.op.axis))]


@defvjp("rmo.mo.reduce.logsoftmax")
def _logsoftmax(ctx):
    y = ctx.outputs[0]
    g = ctx.ct
    return [g - ops.exp(y) * ops.sum(g, axis=ctx.op.axis)]


@defvjp("mo.reduce.layer_norm")
def _layer_norm(ctx):
    """Normalisation over the last axis: ``y = xhat * gamma + beta``."""
    x, gamma, beta, _ = ctx.inputs
    eps = float(ctx.const(3))
    g = ctx.ct

    last = x.rank - 1
    centred = x - mean(x, last)
    rstd = ops.rsqrt(mean(centred * centred, last) + eps)
    xhat = centred * rstd

    dx = None
    if ctx.needs[0]:
        dxhat = g * gamma
        dx = rstd * (
            dxhat - mean(dxhat, last) - xhat * mean(dxhat * xhat, last)
        )
    dgamma = unbroadcast(g * xhat, gamma) if ctx.needs[1] else None
    dbeta = unbroadcast(g, beta) if ctx.needs[2] else None
    return [dx, dgamma, dbeta, None]
