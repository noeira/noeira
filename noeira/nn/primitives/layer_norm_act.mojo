"""LayerNormAct[DIM, OP] — fused LayerNorm + elementwise activation.

`y = OP.forward(gamma * x_hat + beta)` in ONE kernel instead of two, and the
backward gates `grad_output` through `OP.backward` inside the two LayerNorm
backward kernels instead of a third launch ahead of them.

## Why

TD-MPC2's `NormedLinear = Sequential[Linear, LayerNorm, Mish]` is the most-run
composite in the framework — the encoder, dynamics, reward, Q and policy trunks
are all stacks of it. On an RTX 5090 walker profile its three epilogue kernels
were:

    layer_norm    21,600 x 5,295 ns  = 114.4 ms   3.8%
    elementwise   19,200 x 4,639 ns  =  89.1 ms   3.0%
    bias_add      21,600 x 2,857 ns  =  61.7 ms   2.2%

The activation kernel is pure overhead: it reads [B, DIM], applies one cheap
scalar op, and writes [B, DIM] back — a full round trip through memory for
work that fits in the register the normalized value already occupies. Folding
it into LayerNorm's store removes that traffic and ~19,200 launches per run,
and the backward's separate act-gate kernel goes with it.

## The trick that keeps it free

The backward needs `OP`'s cached value — the PRE-activation `z` for ops with
`owns_cache=False` (ReLU, Mish), the POST-activation `y` for those with
`owns_cache=True` (Tanh, Sigmoid). Caching either would cost another
[B, DIM] buffer and another write, giving back most of what the fusion won.

It is not needed: `z = gamma * x_hat + beta` and `x_hat` is ALREADY cached
(LayerNorm's backward needs it), so both backward kernels RECOMPUTE `z` from
`cache_xhat`, `gamma` and `beta` for a couple of FLOPs and no memory at all.
`y` likewise, via one extra `OP.forward_scalar`.

## Drop-in for `Sequential[..., LayerNorm[D], Act[D]]`

Same `Param` names (`gamma`, `beta`) and same sizes, and because the activation
child carried NO parameters, replacing the PAIR with this single module leaves
the `Sequential` child indices of every preceding module untouched:

    Sequential[Linear, LayerNorm, Mish]  ->  0.weight 0.bias 1.gamma 1.beta
    Sequential[Linear, LayerNormAct]     ->  0.weight 0.bias 1.gamma 1.beta

so existing checkpoints load unchanged. (Anything AFTER the pair does shift by
one index — there is nothing after it in `NormedLinear`, but check before
swapping this into a longer stack.)

⚠ GPU-only fusion. The CPU path runs the same math in a plain loop.
⚠ `LN_EPS`, `LN_TPB`, `LN_ACC` and the register-cache threshold are shared with
`layer_norm.mojo` — the numerics are meant to match it exactly, and
`tests/nn/test_layer_norm_act_parity.mojo` asserts that against the unfused
`LayerNorm` + activation pair.
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.utils.numerics import get_accum_type

from noeira.nn.constants import DT, CPU_SIMD_W
from noeira.nn.core.element_op import ElementOp
from ..core.tensor import Tensor, TensorImpl
from ..core.polyak import polyak_tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .layer_norm import LN_EPS, LN_TPB, LN_ACC, LN_REG_CAP
from noeira.core.tiles import RowMajor, Tile


@always_inline
def _act_cache_of[OP: ElementOp](z: Scalar[DT]) -> Scalar[DT]:
    """The value `OP.backward_scalar` expects as its `cache` argument.

    `owns_cache=True` ops (Tanh, Sigmoid) differentiate w.r.t. their OUTPUT;
    the rest (ReLU, Mish) w.r.t. their INPUT. Recomputed rather than stored —
    see the module docstring.
    """
    comptime if OP.owns_cache:
        return OP.forward_scalar(z)
    else:
        return z


def _ln_act_forward_kernel[
    BATCH: Int, DIM: Int, OP: ElementOp, ADT: DType = DT
](
    input: Tile[ADT, BATCH, DIM],
    output: Tile[ADT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
):
    """Mirror of `_layer_norm_forward_kernel` with `OP.forward` on the store."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + LN_TPB - 1) // LN_TPB
    comptime REG_CACHE = ELEMS <= LN_REG_CAP
    var inv_dim = Scalar[LN_ACC](1.0) / Scalar[LN_ACC](DIM)
    var my_sum = Scalar[LN_ACC](0)
    var my_sumsq = Scalar[LN_ACC](0)

    comptime if REG_CACHE:
        var slice = Array[Scalar[LN_ACC], ELEMS](fill=Scalar[LN_ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * LN_TPB
            if col < DIM:
                var x = rebind[Scalar[ADT]](input[b, col]).cast[LN_ACC]()
                slice[e] = x
                my_sum += x
                my_sumsq += x * x
        var mean_val = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_sum) * inv_dim
        )
        var ex2 = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_sumsq) * inv_dim
        )
        var var_val = ex2 - mean_val * mean_val
        if var_val < Scalar[LN_ACC](0):
            var_val = Scalar[LN_ACC](0)
        var inv_std = Scalar[LN_ACC](1.0) / sqrt(var_val + LN_EPS.cast[LN_ACC]())
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()

        comptime for e in range(ELEMS):
            var col = t + e * LN_TPB
            if col < DIM:
                var x_hat = (slice[e] - mean_val) * inv_std
                cache_xhat[b, col] = x_hat.cast[DT]()
                var g_d = rebind[Scalar[DT]](gamma[col]).cast[LN_ACC]()
                var bt_d = rebind[Scalar[DT]](beta[col]).cast[LN_ACC]()
                var z = (g_d * x_hat + bt_d).cast[DT]()
                output[b, col] = OP.forward_scalar(z).cast[ADT]()
    else:
        var idx = t
        while idx < DIM:
            var x = rebind[Scalar[ADT]](input[b, idx]).cast[LN_ACC]()
            my_sum += x
            my_sumsq += x * x
            idx += LN_TPB
        var mean_val = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_sum) * inv_dim
        )
        var ex2 = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_sumsq) * inv_dim
        )
        var var_val = ex2 - mean_val * mean_val
        if var_val < Scalar[LN_ACC](0):
            var_val = Scalar[LN_ACC](0)
        var inv_std = Scalar[LN_ACC](1.0) / sqrt(var_val + LN_EPS.cast[LN_ACC]())
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()
        idx = t
        while idx < DIM:
            var x = rebind[Scalar[ADT]](input[b, idx]).cast[LN_ACC]()
            var x_hat = (x - mean_val) * inv_std
            cache_xhat[b, idx] = x_hat.cast[DT]()
            var g_d = rebind[Scalar[DT]](gamma[idx]).cast[LN_ACC]()
            var bt_d = rebind[Scalar[DT]](beta[idx]).cast[LN_ACC]()
            var z = (g_d * x_hat + bt_d).cast[DT]()
            output[b, idx] = OP.forward_scalar(z).cast[ADT]()
            idx += LN_TPB


def _ln_act_backward_dx_kernel[
    BATCH: Int, DIM: Int, OP: ElementOp, ADT: DType = DT
](
    grad_output: Tile[ADT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
    grad_input: Tile[ADT, BATCH, DIM],
):
    """`_layer_norm_backward_dx_kernel` with the activation gate folded in.

    `go_ln = OP.backward(z, go)` where `z = gamma*x_hat + beta` is RECOMPUTED,
    not read — see the module docstring.
    """
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + LN_TPB - 1) // LN_TPB
    comptime REG_CACHE = ELEMS <= LN_REG_CAP
    var inv_dim = Scalar[LN_ACC](1.0) / Scalar[LN_ACC](DIM)
    var inv_std = rebind[Scalar[DT]](cache_inv_std[b]).cast[LN_ACC]()
    var my_g = Scalar[LN_ACC](0)
    var my_g_xhat = Scalar[LN_ACC](0)

    comptime if REG_CACHE:
        var g_s = Array[Scalar[LN_ACC], ELEMS](fill=Scalar[LN_ACC](0))
        var xh_s = Array[Scalar[LN_ACC], ELEMS](fill=Scalar[LN_ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * LN_TPB
            if col < DIM:
                var go_a = rebind[Scalar[ADT]](grad_output[b, col]).cast[DT]()
                var gm = rebind[Scalar[DT]](gamma[col])
                var bt = rebind[Scalar[DT]](beta[col])
                var xh = rebind[Scalar[DT]](cache_xhat[b, col])
                var z = gm * xh + bt
                var go = OP.backward_scalar(_act_cache_of[OP](z), go_a)
                var g = go.cast[LN_ACC]() * gm.cast[LN_ACC]()
                g_s[e] = g
                xh_s[e] = xh.cast[LN_ACC]()
                my_g += g
                my_g_xhat += g * xh.cast[LN_ACC]()
        var mean_g = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_g_xhat)
            * inv_dim
        )

        comptime for e in range(ELEMS):
            var col = t + e * LN_TPB
            if col < DIM:
                grad_input[b, col] = (
                    inv_std * (g_s[e] - mean_g - xh_s[e] * mean_g_xhat)
                ).cast[ADT]()
    else:
        var idx = t
        while idx < DIM:
            var go_a = rebind[Scalar[ADT]](grad_output[b, idx]).cast[DT]()
            var gm = rebind[Scalar[DT]](gamma[idx])
            var bt = rebind[Scalar[DT]](beta[idx])
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx])
            var z = gm * xh + bt
            var go = OP.backward_scalar(_act_cache_of[OP](z), go_a)
            var g = go.cast[LN_ACC]() * gm.cast[LN_ACC]()
            my_g += g
            my_g_xhat += g * xh.cast[LN_ACC]()
            idx += LN_TPB
        var mean_g = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=LN_TPB, broadcast=True](val=my_g_xhat)
            * inv_dim
        )
        idx = t
        while idx < DIM:
            var go_a = rebind[Scalar[ADT]](grad_output[b, idx]).cast[DT]()
            var gm = rebind[Scalar[DT]](gamma[idx])
            var bt = rebind[Scalar[DT]](beta[idx])
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx])
            var z = gm * xh + bt
            var go = OP.backward_scalar(_act_cache_of[OP](z), go_a)
            var g = go.cast[LN_ACC]() * gm.cast[LN_ACC]()
            grad_input[b, idx] = (
                inv_std * (g - mean_g - xh.cast[LN_ACC]() * mean_g_xhat)
            ).cast[ADT]()
            idx += LN_TPB


def _ln_act_backward_dparams_kernel[
    BATCH: Int, DIM: Int, OP: ElementOp, ADT: DType = DT
](
    grad_output: Tile[ADT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    """`_layer_norm_backward_dparams_kernel` with the same recomputed gate.

    ⚠ The gate is recomputed here rather than shared with the dx kernel: the
    two run as independent launches, so sharing would need a third [B, DIM]
    buffer and a third launch — exactly what this fusion exists to remove. The
    duplicated work is a handful of FLOPs per element.
    """
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var gm = rebind[Scalar[DT]](gamma[col])
    var bt = rebind[Scalar[DT]](beta[col])
    var my_dg: Scalar[DT] = 0.0
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        var go_a = rebind[Scalar[ADT]](grad_output[bi, col]).cast[DT]()
        var xh = rebind[Scalar[DT]](cache_xhat[bi, col])
        var go = OP.backward_scalar(_act_cache_of[OP](gm * xh + bt), go_a)
        my_dg += go * xh
        my_db += go
        bi += LN_TPB
    var total_dg = block.sum[block_size=LN_TPB, broadcast=False](val=my_dg)
    var total_db = block.sum[block_size=LN_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]
        grad_beta[col] = rebind[Scalar[DT]](grad_beta[col]) + total_db[0]


struct LayerNormAct[DIM_: Int, OP: ElementOp, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return String("LayerNormAct")

    var gamma: Param["gamma", False, Self.DIM_]
    var beta: Param["beta", False, Self.DIM_]
    var cache_xhat: Tensor      # [BATCH, DIM] — fp32
    var cache_inv_std: Tensor   # [BATCH] — fp32

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM_]()
        self.beta = Param["beta", False, Self.DIM_]()
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        m.gamma = Param["gamma", False, Self.DIM_].make[target](ctx)
        m.beta = Param["beta", False, Self.DIM_].make[target](ctx)
        # γ = 1, β = 0 — LayerNorm's init, NOT the Initializer's (which would
        # randomize a scale that must start at identity).
        for i in range(Self.DIM_):
            m.gamma.val.data[i] = Scalar[DT](1.0)
            m.beta.val.data[i] = Scalar[DT](0.0)
        comptime if target == "gpu":
            m.gamma.val.upload_resident(ctx.value())
            m.beta.val.upload_resident(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            outd.ensure(B * Self.DIM_)
            self.cache_xhat.ensure(B * Self.DIM_)
            self.cache_inv_std.ensure(B)
            # Reductions are SIMD, mirroring `LayerNorm`'s CPU forward — it
            # was already vectorized and this module must not regress it.
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.DIM_)
            var in_p = in0d.data.unsafe_ptr()
            for b in range(B):
                var row = b * Self.DIM_
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM_:
                    acc += in_p.unsafe_load[width=W](row + d)
                    d += W
                var sm = acc.reduce_add()
                while d < Self.DIM_:
                    sm += in_p[unsafe_offset=row + d]
                    d += 1
                var mean = sm * inv_dim
                var meanv = SIMD[DT, W](mean)
                var vacc = SIMD[DT, W](0)
                d = 0
                while d + W <= Self.DIM_:
                    var df = in_p.unsafe_load[width=W](row + d) - meanv
                    vacc += df * df
                    d += W
                var sv = vacc.reduce_add()
                while d < Self.DIM_:
                    var df = in_p[unsafe_offset=row + d] - mean
                    sv += df * df
                    d += 1
                var inv_std = Scalar[DT](1.0) / sqrt(sv * inv_dim + LN_EPS)
                self.cache_inv_std.data[b] = inv_std
                # ⚠ SIMD, not a scalar loop. `Elementwise`'s CPU path is
                # vectorized (`forward_simd[CPU_SIMD_W]`), so fusing the
                # activation in scalar form would make the CPU SLOWER than the
                # pair this module replaces — Mish is exp/log/tanh per element.
                var xp = in0d.data.unsafe_ptr()
                var op = outd.data.unsafe_ptr()
                var cp = self.cache_xhat.data.unsafe_ptr()
                var gp = self.gamma.val.data.unsafe_ptr()
                var bp = self.beta.val.data.unsafe_ptr()
                var j = 0
                while j + CPU_SIMD_W <= Self.DIM_:
                    var xh = (
                        xp.unsafe_load[width=CPU_SIMD_W](row + j) - mean
                    ) * inv_std
                    cp.unsafe_store(row + j, xh)
                    var z = gp.unsafe_load[width=CPU_SIMD_W](
                        j
                    ) * xh + bp.unsafe_load[width=CPU_SIMD_W](j)
                    op.unsafe_store(
                        row + j, Self.OP.forward_simd[CPU_SIMD_W](z)
                    )
                    j += CPU_SIMD_W
                while j < Self.DIM_:
                    var xh = (in0d.data[row + j] - mean) * inv_std
                    self.cache_xhat.data[row + j] = xh
                    var z = self.gamma.val.data[j] * xh + self.beta.val.data[j]
                    outd.data[row + j] = Self.OP.forward_scalar(z)
                    j += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_xhat.ensure_gpu(c, B * Self.DIM_)
            self.cache_inv_std.ensure_gpu(c, B)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime lb = RowMajor[B]
            comptime ld = RowMajor[Self.DIM_]
            c.enqueue_function[
                _ln_act_forward_kernel[B, Self.DIM_, Self.OP, Self.ADT]
            ](
                in0.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
                self.gamma.val.tt["gpu", Self.DIM_](),
                self.beta.val.tt["gpu", Self.DIM_](),
                self.cache_xhat.tt["gpu", B, Self.DIM_](),
                self.cache_inv_std.tt["gpu", B](),
                grid_dim=B,
                block_dim=LN_TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            gind.ensure(B * Self.DIM_)
            var inv_dim = Scalar[DT](1.0) / Scalar[DT](Self.DIM_)
            for b in range(B):
                var row = b * Self.DIM_
                var inv_std = self.cache_inv_std.data[b]
                var mean_g = Scalar[DT](0)
                var mean_g_xhat = Scalar[DT](0)
                for j in range(Self.DIM_):
                    var xh = self.cache_xhat.data[row + j]
                    var z = self.gamma.val.data[j] * xh + self.beta.val.data[j]
                    var go = Self.OP.backward_scalar(
                        _act_cache_of[Self.OP](z), god.data[row + j]
                    )
                    self.gamma.grd.data[j] += go * xh
                    self.beta.grd.data[j] += go
                    var g = go * self.gamma.val.data[j]
                    mean_g += g
                    mean_g_xhat += g * xh
                mean_g *= inv_dim
                mean_g_xhat *= inv_dim
                for j in range(Self.DIM_):
                    var xh = self.cache_xhat.data[row + j]
                    var z = self.gamma.val.data[j] * xh + self.beta.val.data[j]
                    var go = Self.OP.backward_scalar(
                        _act_cache_of[Self.OP](z), god.data[row + j]
                    )
                    var g = go * self.gamma.val.data[j]
                    gind.data[row + j] = inv_std * (
                        g - mean_g - xh * mean_g_xhat
                    )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime lb = RowMajor[B]
            comptime ld = RowMajor[Self.DIM_]
            c.enqueue_function[
                _ln_act_backward_dx_kernel[B, Self.DIM_, Self.OP, Self.ADT]
            ](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.gamma.val.tt["gpu", Self.DIM_](),
                self.beta.val.tt["gpu", Self.DIM_](),
                self.cache_xhat.tt["gpu", B, Self.DIM_](),
                self.cache_inv_std.tt["gpu", B](),
                gin.tt["gpu", B, Self.DIM_](),
                grid_dim=B,
                block_dim=LN_TPB,
            )
            c.enqueue_function[
                _ln_act_backward_dparams_kernel[B, Self.DIM_, Self.OP, Self.ADT]
            ](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.gamma.val.tt["gpu", Self.DIM_](),
                self.beta.val.tt["gpu", Self.DIM_](),
                self.cache_xhat.tt["gpu", B, Self.DIM_](),
                self.gamma.grd.tt["gpu", Self.DIM_](),
                self.beta.grd.tt["gpu", Self.DIM_](),
                grid_dim=Self.DIM_,
                block_dim=LN_TPB,
            )

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.DIM_](
            self.gamma.val, src.gamma.val, tau, ctx
        )
        polyak_tensor[target, Self.DIM_](self.beta.val, src.beta.val, tau, ctx)
