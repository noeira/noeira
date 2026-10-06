"""Elementwise ops. Binary ops broadcast, so their cotangents are summed back
to each input's shape; unary ops reuse their output where it is cheaper."""

from __future__ import annotations

import math
from collections.abc import Callable

from max.graph import TensorValue, ops

from .._ops import mask, scalar, unbroadcast
from ..registry import RuleContext, defvjp, nondiff

nondiff(
    "rmo.greater",
    "rmo.greater_equal",
    "rmo.equal",
    "rmo.not_equal",
    "rmo.mo.floor",
)

_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_TANH_GELU_K = math.sqrt(2.0 / math.pi)
_TANH_GELU_C = 0.044715
_QUICK_GELU_A = 1.702


def _binary(
    ctx: RuleContext,
    da: Callable[[], TensorValue],
    db: Callable[[], TensorValue],
) -> list[TensorValue | None]:
    """Builds only the cotangents the transform needs, then unbroadcasts."""
    a, b = ctx.inputs
    return [
        unbroadcast(da(), a) if ctx.needs[0] else None,
        unbroadcast(db(), b) if ctx.needs[1] else None,
    ]


@defvjp("rmo.add")
def _add(ctx):
    return _binary(ctx, lambda: ctx.ct, lambda: ctx.ct)


@defvjp("rmo.sub")
def _sub(ctx):
    return _binary(ctx, lambda: ctx.ct, lambda: -ctx.ct)


@defvjp("rmo.mul")
def _mul(ctx):
    a, b = ctx.inputs
    return _binary(ctx, lambda: ctx.ct * b, lambda: ctx.ct * a)


@defvjp("rmo.div")
def _div(ctx):
    _, b = ctx.inputs
    out = ctx.outputs[0]
    return _binary(ctx, lambda: ctx.ct / b, lambda: -(ctx.ct * out) / b)


@defvjp("rmo.pow")
def _pow(ctx):
    a, b = ctx.inputs
    out = ctx.outputs[0]
    return _binary(
        ctx,
        lambda: ctx.ct * b * ops.pow(a, b - 1),
        lambda: ctx.ct * out * ops.log(a),
    )


def _tie_split(win: TensorValue, tie: TensorValue, like: TensorValue):
    """1 where this input won, 1/2 on a tie (as PyTorch's maximum/minimum)."""
    return mask(win, like) + mask(tie, like) * 0.5


@defvjp("rmo.max")
def _maximum(ctx):
    a, b = ctx.inputs
    tie = ops.equal(a, b)
    return _binary(
        ctx,
        lambda: ctx.ct * _tie_split(ops.greater(a, b), tie, ctx.ct),
        lambda: ctx.ct * _tie_split(ops.greater(b, a), tie, ctx.ct),
    )


@defvjp("rmo.min")
def _minimum(ctx):
    a, b = ctx.inputs
    tie = ops.equal(a, b)
    return _binary(
        ctx,
        lambda: ctx.ct * _tie_split(ops.greater(b, a), tie, ctx.ct),
        lambda: ctx.ct * _tie_split(ops.greater(a, b), tie, ctx.ct),
    )


@defvjp("rmo.select")
def _select(ctx):
    cond, x, y = ctx.inputs
    g = ctx.ct
    zero = scalar(0, g)
    return [
        None,
        unbroadcast(ops.where(cond, g, zero), x) if ctx.needs[1] else None,
        unbroadcast(ops.where(cond, zero, g), y) if ctx.needs[2] else None,
    ]


def _unary(name: str, derivative: Callable[[TensorValue, TensorValue], TensorValue]):
    """Registers ``g * derivative(x, out)`` as the VJP of a unary op."""

    @defvjp(name)
    def rule(ctx):
        return [ctx.ct * derivative(ctx.inputs[0], ctx.outputs[0])]

    return rule


@defvjp("rmo.mo.negative")
def _negative(ctx):
    return [-ctx.ct]


_unary("rmo.mo.exp", lambda x, out: out)
_unary("rmo.mo.log", lambda x, out: 1.0 / x)
_unary("rmo.mo.log1p", lambda x, out: 1.0 / (x + 1.0))
_unary("rmo.mo.sqrt", lambda x, out: 0.5 / out)
_unary("rmo.mo.rsqrt", lambda x, out: -0.5 * out * out * out)
_unary("rmo.mo.tanh", lambda x, out: 1.0 - out * out)
_unary("rmo.mo.sigmoid", lambda x, out: out * (1.0 - out))
_unary("rmo.mo.relu", lambda x, out: mask(ops.greater(x, scalar(0, x)), x))
_unary(
    "rmo.mo.abs",
    lambda x, out: mask(ops.greater(x, scalar(0, x)), x)
    - mask(ops.greater(scalar(0, x), x), x),
)
_unary("rmo.mo.sin", lambda x, out: ops.cos(x))
_unary("rmo.mo.cos", lambda x, out: -ops.sin(x))
_unary("rmo.mo.erf", lambda x, out: _TWO_OVER_SQRT_PI * ops.exp(-(x * x)))
_unary("rmo.mo.atanh", lambda x, out: 1.0 / (1.0 - x * x))


def _silu(x, out):
    s = ops.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _gelu(x, out):
    # d/dx [x Phi(x)] = Phi(x) + x phi(x)
    cdf = 0.5 * (1.0 + ops.erf(x / _SQRT_2))
    pdf = _INV_SQRT_2PI * ops.exp(-0.5 * x * x)
    return cdf + x * pdf


def _gelu_tanh(x, out):
    # out = x/2 (1 + tanh(u)),  u = k (x + c x^3)
    t = ops.tanh(_TANH_GELU_K * (x + _TANH_GELU_C * x * x * x))
    du = _TANH_GELU_K * (1.0 + 3.0 * _TANH_GELU_C * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


def _gelu_quick(x, out):
    # out = x sigmoid(a x)
    s = ops.sigmoid(_QUICK_GELU_A * x)
    return s + _QUICK_GELU_A * x * s * (1.0 - s)


_unary("rmo.mo.silu", _silu)
_unary("rmo.mo.gelu", _gelu)
_unary("rmo.mo.gelu_tanh", _gelu_tanh)
_unary("rmo.mo.gelu_quick", _gelu_quick)


@defvjp("mo.cast")
def _cast(ctx):
    # Only float -> float casts carry a cotangent (the transform never asks
    # for an integer input's).
    return [ops.cast(ctx.ct, ctx.inputs[0].dtype)]
