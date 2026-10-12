"""Custom ops: a rule per kernel symbol.

Every custom op is ``mo.custom``; the ``symbol`` attribute names the Mojo
kernel. ``defvjp_custom(symbol)`` registers that kernel's VJP. The ones here
pair a forward kernel with its backward kernel, the forward returning its
residuals as extra results and the VJP passing them on:

- LayerNorm: the residuals are the mean and rstd, which ``nn.py``'s
  composite rule recomputes;
- causal attention (noeira's fused attention): the residuals are the output
  and each row's log-sum-exp, so the backward never sees the score matrix.
"""

from __future__ import annotations

from collections.abc import Callable

from max.graph import TensorType, ops

from ..registry import Rule, defvjp

_BY_SYMBOL: dict[str, Rule] = {}


def defvjp_custom(symbol: str) -> Callable[[Rule], Rule]:
    """Registers the decorated function as the VJP of the custom op whose
    kernel is ``symbol``."""

    def register(rule: Rule) -> Rule:
        if symbol in _BY_SYMBOL:
            raise ValueError(f"a VJP for custom op '{symbol}' is already registered")
        _BY_SYMBOL[symbol] = rule
        return rule

    return register


@defvjp("mo.custom")
def _custom(ctx):
    symbol = ctx.attr("symbol")
    rule = _BY_SYMBOL.get(symbol)
    if rule is None:
        raise NotImplementedError(
            f"No VJP for custom op '{symbol}'. Register one with "
            f"@defvjp_custom('{symbol}')."
        )
    return rule(ctx)


# Chunks of rows whose partial dgamma / dbeta the backward kernel returns.
PARTIALS = 64


@defvjp_custom("noeira_layer_norm_fwd")
def _layer_norm_kernel(ctx):
    rows, gamma, beta, _ = ctx.inputs
    _, mean, rstd = ctx.outputs
    if ctx.cts[1] is not None or ctx.cts[2] is not None:
        raise NotImplementedError(
            "noeira_layer_norm_fwd: its mean and rstd are residuals; a graph "
            "that differentiates through them needs their cotangents too"
        )
    partial = TensorType(gamma.dtype, [PARTIALS, gamma.shape[0]], rows.device)
    dx, dgamma, dbeta = ops.custom(
        "noeira_layer_norm_bwd",
        device=rows.device,
        values=[ctx.ct, rows, gamma, mean, rstd],
        out_types=[rows.type, partial, partial],
    )
    # The kernel returns per-chunk partial sums; the graph finishes them.
    finish = lambda p: ops.reshape(ops.sum(p.tensor, axis=0), gamma.shape)  # noqa: E731
    return [dx.tensor, finish(dgamma), finish(dbeta), None]


@defvjp_custom("noeira_attention_fwd")
def _attention_kernel(ctx):
    (qkv,) = ctx.inputs
    o, lse = ctx.outputs
    if ctx.cts[1] is not None:
        raise NotImplementedError(
            "noeira_attention_fwd: its log-sum-exp is a residual; a graph that "
            "differentiates through it needs its cotangent too"
        )
    b, t, c3 = (int(d) for d in qkv.shape)
    heads = int(lse.shape[0]) // (b * t)
    dqkv, _ = ops.custom(
        "noeira_attention_bwd",
        device=qkv.device,
        values=[ctx.ct, qkv, o, lse],
        out_types=[qkv.type, lse.type],  # dqkv, and the D vector (workspace)
        parameters={"B": b, "H": heads, "S": t, "HD": c3 // 3 // heads},
    )
    return [dqkv.tensor]
