"""LayerNorm[DIM] — per-row (feature-axis) layer normalisation (storage surface).

Transformed from legacy `nn.primitives.LayerNorm` (surface-only change). γ/β are
`Param`s (decay=False); the per-row x̂ + inv_std cache is leaf-owned (output-
caching, no input aliasing). No running State, no train/eval split. The CPU SIMD
feature-axis loops and the three GPU kernels (forward / backward-dx 1-block-per-
row, backward-dparams 1-block-per-col) are carried over verbatim.

Backward (cache):  g = go·γ ; mean_g = mean_d(g) ; mean_g_xhat = mean_d(g·x̂)
    dx = inv_std·(g - mean_g - x̂·mean_g_xhat) ; dγ += Σ_b go·x̂ ; dβ += Σ_b go

bf16-FLOW (AMP "Step B"): `LayerNorm[DIM]` is fp32 (ACT_DT == DT, the legacy
NoAMP path, byte-identical); `LayerNorm[DIM, DType.bfloat16]` is fp32-INTERNAL
but flows its I/O activations at bf16. The mean/var/normalize stats, the affine
(γ/β are fp32 `Param` masters) and the whole `cache_xhat`/`cache_inv_std` cache
stay fp32 (LN_ACC) — only the I/O-activation kernel operands (`input`/`output`/
`grad_output`/`grad_input`) are parametrized by `ADT`: on READ each bf16 element
is cast→fp32 before computing; on WRITE the fp32 result is cast→bf16. bf16-flow
is GPU-only. The reductions stay in LN_ACC (fp32) so accuracy is unchanged.
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.utils.numerics import get_accum_type
from layout import Layout, LayoutTensor, TileTensor, row_major

from noeira.nn.constants import DT, CPU_SIMD_W
from ..core.tensor import Tensor, TensorImpl
from ..core.polyak import polyak_tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime LN_EPS: Scalar[DT] = 1e-5
"""Default epsilon. ⚠ SigLIP's vision tower uses `layer_norm_eps = 1e-6`; pass
`LayerNorm[DIM, EPS=…]` rather than relying on this."""
comptime LN_TPB: Int = 128
# Reductions run in the accumulation dtype (f32 for bf16 inputs; identity for
# DT=f32). ELEMS = per-thread feature slice; each thread reads its slice ONCE
# into registers, derives mean/var from raw moments (E[x²]−E[x]²) in a single
# read, then normalizes from registers — vs the legacy 3× input re-read.
comptime LN_ACC = get_accum_type[DT]()
comptime LN_REG_CAP = 8  # max per-thread slice (≈ DIM≤1024) to register-cache


# ── GPU kernels (single-pass register-cached forward; block-per-row) ────
# The I/O-ACTIVATION operands (`input`/`output`/`grad_output`/`grad_input`) are
# parametrized by `ADT` (the activation-flow dtype). The fp32 path runs them at
# DT (default `ADT = DT` reproduces the legacy leaf byte-for-byte); the bf16-flow
# path holds those operands at bfloat16, reading each element `.cast[LN_ACC]()`
# (UP to fp32) and writing fp32 results `.cast[ADT]()` (DOWN to bf16). The stats,
# the affine, and the cache (`cache_xhat`/`cache_inv_std`, fp32) are unchanged.
def _layer_norm_forward_kernel[
    BATCH: Int,
    DIM: Int,
    ADT: DType = DT,
    EPS: Scalar[DT] = LN_EPS,
    # ⚠ The kernel used to read the module constant `LN_EPS` instead of the
    # struct's `EPS`, so `LayerNorm[DIM, EPS=1e-12]` was 1e-12 on CPU and 1e-5
    # on GPU. Gate: tests/nn/test_layer_norm_eps_gpu.mojo.
](
    input: LayoutTensor[ADT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[ADT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + LN_TPB - 1) // LN_TPB
    # Register-cache the thread's feature slice only when it is small enough to
    # stay in registers (≤ LN_REG_CAP); else a spill would make it slower than
    # the 2-read raw-moments fallback (still better than the legacy 3 reads).
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
        var inv_std = Scalar[LN_ACC](1.0) / sqrt(var_val + EPS.cast[LN_ACC]())
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()

        comptime for e in range(ELEMS):
            var col = t + e * LN_TPB
            if col < DIM:
                var x_hat = (slice[e] - mean_val) * inv_std
                cache_xhat[b, col] = x_hat.cast[DT]()
                var g_d = rebind[Scalar[DT]](gamma[col]).cast[LN_ACC]()
                var bt_d = rebind[Scalar[DT]](beta[col]).cast[LN_ACC]()
                output[b, col] = (g_d * x_hat + bt_d).cast[ADT]()
    else:
        # 2-read raw-moments: stats pass (sum + Σx²) then normalize pass.
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
        var inv_std = Scalar[LN_ACC](1.0) / sqrt(var_val + EPS.cast[LN_ACC]())
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()
        idx = t
        while idx < DIM:
            var x = rebind[Scalar[ADT]](input[b, idx]).cast[LN_ACC]()
            var x_hat = (x - mean_val) * inv_std
            cache_xhat[b, idx] = x_hat.cast[DT]()
            var g_d = rebind[Scalar[DT]](gamma[idx]).cast[LN_ACC]()
            var bt_d = rebind[Scalar[DT]](beta[idx]).cast[LN_ACC]()
            output[b, idx] = (g_d * x_hat + bt_d).cast[ADT]()
            idx += LN_TPB


def _layer_norm_backward_dx_kernel[
    BATCH: Int,
    DIM: Int,
    ADT: DType = DT,
](
    grad_output: LayoutTensor[ADT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    grad_input: LayoutTensor[ADT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
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
        # Cache g=go·γ and x̂ once; the write pass reads no global memory.
        var g_s = Array[Scalar[LN_ACC], ELEMS](fill=Scalar[LN_ACC](0))
        var xh_s = Array[Scalar[LN_ACC], ELEMS](fill=Scalar[LN_ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * LN_TPB
            if col < DIM:
                var go = rebind[Scalar[ADT]](grad_output[b, col]).cast[LN_ACC]()
                var gm = rebind[Scalar[DT]](gamma[col]).cast[LN_ACC]()
                var xh = rebind[Scalar[DT]](cache_xhat[b, col]).cast[LN_ACC]()
                var g = go * gm
                g_s[e] = g
                xh_s[e] = xh
                my_g += g
                my_g_xhat += g * xh
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
            var go = rebind[Scalar[ADT]](grad_output[b, idx]).cast[LN_ACC]()
            var gm = rebind[Scalar[DT]](gamma[idx]).cast[LN_ACC]()
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[LN_ACC]()
            var g = go * gm
            my_g += g
            my_g_xhat += g * xh
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
            var go = rebind[Scalar[ADT]](grad_output[b, idx]).cast[LN_ACC]()
            var gm = rebind[Scalar[DT]](gamma[idx]).cast[LN_ACC]()
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[LN_ACC]()
            var g = go * gm
            grad_input[b, idx] = (
                inv_std * (g - mean_g - xh * mean_g_xhat)
            ).cast[ADT]()
            idx += LN_TPB


def _layer_norm_backward_dparams_kernel[
    BATCH: Int,
    DIM: Int,
    ADT: DType = DT,
](
    grad_output: LayoutTensor[ADT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    # γ/β grads accumulate into the fp32 master (`grad_gamma`/`grad_beta`); the
    # bf16 `grad_output` operand is read `.cast[DT]()` UP to fp32, the cache is
    # already fp32 → the whole accumulation is fp32.
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_dg: Scalar[DT] = 0.0
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        var go = rebind[Scalar[ADT]](grad_output[bi, col]).cast[DT]()
        var xh = rebind[Scalar[DT]](cache_xhat[bi, col])
        my_dg += go * xh
        my_db += go
        bi += LN_TPB
    var total_dg = block.sum[block_size=LN_TPB, broadcast=False](val=my_dg)
    var total_db = block.sum[block_size=LN_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]
        grad_beta[col] = rebind[Scalar[DT]](grad_beta[col]) + total_db[0]


struct LayerNorm[DIM_: Int, ADT: DType = DT, EPS: Scalar[DT] = LN_EPS](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype. `LayerNorm[DIM]` = fp32 (ACT_DT == DT, the legacy
    # NoAMP path, byte-identical); `LayerNorm[DIM, bfloat16]` flows its I/O
    # activations at bf16 while computing fp32 INTERNALLY (stats + affine + cache
    # all stay fp32; only the I/O-activation kernel operands cast at the bf16
    # boundary). bf16-flow is GPU-only.
    comptime ACT_DT = Self.ADT

    var gamma: Param["gamma", False, Self.DIM_]
    var beta: Param["beta", False, Self.DIM_]
    var cache_xhat: Tensor  # [BATCH, DIM] — fp32 (fp32-internal)
    var cache_inv_std: Tensor  # [BATCH] — fp32

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM_]()
        self.beta = Param["beta", False, Self.DIM_]()
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var ln = Self()
        ln.gamma = Param["gamma", False, Self.DIM_].make[target](ctx)
        ln.beta = Param["beta", False, Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            ln.gamma.val.data[k] = Scalar[DT](1.0)  # γ←1, β←0
        comptime if target != "cpu":
            var dctx = ctx.value()
            ln.gamma.val.upload(dctx)
            ln.beta.val.upload(dctx)
        return ln^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here, but the checker won't collapse the opaque
            # `Self.ACT_DT` to `DT` for unification vs the fp32 γ/β/cache views —
            # so rebind the activation refs (sound; the dtypes are equal here).
            # `TensorImpl[Self.ACT_DT]` ≡ `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.DIM_)
                self.cache_xhat.ensure(B * Self.DIM_)
                self.cache_inv_std.ensure(B)
                var inv_v = TileTensor(self.cache_inv_std.data, row_major[B]())
                var in_p = in0d.data.unsafe_ptr()
                var out_p = outd.data.unsafe_ptr()
                var g_p = self.gamma.val.data.unsafe_ptr()
                var b_p = self.beta.val.data.unsafe_ptr()
                var xh_p = self.cache_xhat.data.unsafe_ptr()
                comptime W = CPU_SIMD_W
                var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
                for b in range(B):
                    var row = b * Self.DIM_
                    var acc = SIMD[DT, W](0)
                    var d = 0
                    while d + W <= Self.DIM_:
                        acc += in_p.unsafe_load[width=W](row + d)
                        d += W
                    var s = acc.reduce_add()
                    while d < Self.DIM_:
                        s += in_p[unsafe_offset=row + d]
                        d += 1
                    var mean = s * inv_dim
                    var meanv = SIMD[DT, W](mean)
                    var vacc = SIMD[DT, W](0)
                    d = 0
                    while d + W <= Self.DIM_:
                        var diff = in_p.unsafe_load[width=W](row + d) - meanv
                        vacc += diff * diff
                        d += W
                    var sv = vacc.reduce_add()
                    while d < Self.DIM_:
                        var diff = in_p[unsafe_offset=row + d] - mean
                        sv += diff * diff
                        d += 1
                    var var_v = sv * inv_dim
                    var inv_std = Scalar[DT](1.0) / sqrt(var_v + Self.EPS)
                    inv_v[b] = inv_std
                    var isv = SIMD[DT, W](inv_std)
                    d = 0
                    while d + W <= Self.DIM_:
                        var xh = (in_p.unsafe_load[width=W](row + d) - meanv) * isv
                        xh_p.unsafe_store(row + d, xh)
                        out_p.unsafe_store(
                            row + d,
                            g_p.unsafe_load[width=W](d) * xh + b_p.unsafe_load[width=W](d),
                        )
                        d += W
                    while d < Self.DIM_:
                        var xh = (in_p[unsafe_offset=row + d] - mean) * inv_std
                        xh_p[unsafe_offset=row + d] = xh
                        out_p[unsafe_offset=row + d] = g_p[unsafe_offset=d] * xh + b_p[unsafe_offset=d]
                        d += 1
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.DIM_)
                self.cache_xhat.ensure_gpu(c, B * Self.DIM_)
                self.cache_inv_std.ensure_gpu(c, B)
                comptime l2d = Layout.row_major(B, Self.DIM_)
                comptime lb = Layout.row_major(B)
                comptime ld = Layout.row_major(Self.DIM_)
                c.enqueue_function[
                    _layer_norm_forward_kernel[B, Self.DIM_, Self.ADT, Self.EPS]
                ](
                    in0d.lt["gpu", l2d](),
                    outd.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", ld](),
                    self.beta.val.lt["gpu", ld](),
                    self.cache_xhat.lt["gpu", l2d](),
                    self.cache_inv_std.lt["gpu", lb](),
                    grid_dim=B,
                    block_dim=LN_TPB,
                )
        else:
            # ── bf16-flow path (GPU-only). Activations cast at the I/O boundary;
            #    stats + affine + cache stay fp32 (the leaf is fp32-internal). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow LayerNorm is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_xhat.ensure_gpu(c, B * Self.DIM_)
            self.cache_inv_std.ensure_gpu(c, B)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime lb = Layout.row_major(B)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[
                _layer_norm_forward_kernel[B, Self.DIM_, Self.ADT, Self.EPS]
            ](
                in0.lt["gpu", l2d](),
                out.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.beta.val.lt["gpu", ld](),
                self.cache_xhat.lt["gpu", l2d](),
                self.cache_inv_std.lt["gpu", lb](),
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
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here — rebind the activation refs (sound; see forward).
            ref god = rebind[Tensor](grad_output)
            ref gind = rebind[Tensor](gin)
            comptime if target == "cpu":
                gind.ensure(B * Self.DIM_)
                var inv_v = TileTensor(self.cache_inv_std.data, row_major[B]())
                var go_p = god.data.unsafe_ptr()
                var gi_p = gind.data.unsafe_ptr()
                var g_p = self.gamma.val.data.unsafe_ptr()
                var gg_p = self.gamma.grd.data.unsafe_ptr()
                var gb_p = self.beta.grd.data.unsafe_ptr()
                var xh_p = self.cache_xhat.data.unsafe_ptr()
                comptime W = CPU_SIMD_W
                var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
                for b in range(B):
                    var row = b * Self.DIM_
                    var inv_std = inv_v[b]
                    var acc_g = SIMD[DT, W](0)
                    var acc_gx = SIMD[DT, W](0)
                    var d = 0
                    while d + W <= Self.DIM_:
                        var g = go_p.unsafe_load[width=W](row + d) * g_p.unsafe_load[width=W](d)
                        acc_g += g
                        acc_gx += g * xh_p.unsafe_load[width=W](row + d)
                        d += W
                    var sum_g = acc_g.reduce_add()
                    var sum_g_xhat = acc_gx.reduce_add()
                    while d < Self.DIM_:
                        var g = go_p[unsafe_offset=row + d] * g_p[unsafe_offset=d]
                        sum_g += g
                        sum_g_xhat += g * xh_p[unsafe_offset=row + d]
                        d += 1
                    var mean_g = sum_g * inv_dim
                    var mean_g_xhat = sum_g_xhat * inv_dim
                    var mg = SIMD[DT, W](mean_g)
                    var mgx = SIMD[DT, W](mean_g_xhat)
                    var isv = SIMD[DT, W](inv_std)
                    d = 0
                    while d + W <= Self.DIM_:
                        var g = go_p.unsafe_load[width=W](row + d) * g_p.unsafe_load[width=W](d)
                        var xh = xh_p.unsafe_load[width=W](row + d)
                        gi_p.unsafe_store(row + d, isv * (g - mg - xh * mgx))
                        d += W
                    while d < Self.DIM_:
                        var g = go_p[unsafe_offset=row + d] * g_p[unsafe_offset=d]
                        var xh = xh_p[unsafe_offset=row + d]
                        gi_p[unsafe_offset=row + d] = inv_std * (g - mean_g - xh * mean_g_xhat)
                        d += 1
                    # dγ += go·x̂ ; dβ += go  (accumulated across the batch)
                    d = 0
                    while d + W <= Self.DIM_:
                        var go = go_p.unsafe_load[width=W](row + d)
                        gg_p.unsafe_store(
                            d,
                            gg_p.unsafe_load[width=W](d)
                            + go * xh_p.unsafe_load[width=W](row + d),
                        )
                        gb_p.unsafe_store(d, gb_p.unsafe_load[width=W](d) + go)
                        d += W
                    while d < Self.DIM_:
                        gg_p[unsafe_offset=d] = gg_p[unsafe_offset=d] + go_p[unsafe_offset=row + d] * xh_p[unsafe_offset=row + d]
                        gb_p[unsafe_offset=d] = gb_p[unsafe_offset=d] + go_p[unsafe_offset=row + d]
                        d += 1
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, B * Self.DIM_)
                comptime l2d = Layout.row_major(B, Self.DIM_)
                comptime lb = Layout.row_major(B)
                comptime ld = Layout.row_major(Self.DIM_)
                c.enqueue_function[
                    _layer_norm_backward_dx_kernel[B, Self.DIM_, Self.ADT]
                ](
                    god.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", ld](),
                    self.cache_xhat.lt["gpu", l2d](),
                    self.cache_inv_std.lt["gpu", lb](),
                    gind.lt["gpu", l2d](),
                    grid_dim=B,
                    block_dim=LN_TPB,
                )
                c.enqueue_function[
                    _layer_norm_backward_dparams_kernel[B, Self.DIM_, Self.ADT]
                ](
                    god.lt["gpu", l2d](),
                    self.cache_xhat.lt["gpu", l2d](),
                    self.gamma.grd.lt["gpu", ld](),
                    self.beta.grd.lt["gpu", ld](),
                    grid_dim=Self.DIM_,
                    block_dim=LN_TPB,
                )
        else:
            # ── bf16-flow path (GPU-only). I/O activations cast at the boundary;
            #    grad math + γ/β masters + cache stay fp32 (fp32-internal). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow LayerNorm is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime lb = Layout.row_major(B)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[
                _layer_norm_backward_dx_kernel[B, Self.DIM_, Self.ADT]
            ](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_xhat.lt["gpu", l2d](),
                self.cache_inv_std.lt["gpu", lb](),
                gin.lt["gpu", l2d](),
                grid_dim=B,
                block_dim=LN_TPB,
            )
            c.enqueue_function[
                _layer_norm_backward_dparams_kernel[B, Self.DIM_, Self.ADT]
            ](
                grad_output.lt["gpu", l2d](),
                self.cache_xhat.lt["gpu", l2d](),
                self.gamma.grd.lt["gpu", ld](),
                self.beta.grd.lt["gpu", ld](),
                grid_dim=Self.DIM_,
                block_dim=LN_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update the affine params toward `src`.

        ⚠⚠ This override was MISSING, and `Module.polyak_from` defaults to a
        NO-OP. Any online/target pair containing a `LayerNorm` therefore copied
        its `Linear` weights every Polyak step and left the target's
        `gamma`/`beta` frozen at their init forever, while the online ones
        drifted. Nothing raises; the target net is simply evaluated with a
        different output scale than the online net, which is exactly the kind
        of mis-scaled bootstrap that reads as a plausible loss curve.

        Found via an FB run whose backward net ends in `LayerNorm[D]`: the
        online gain had reached ~1.2 while the target's was still 1.0. Same
        family as the zero-series promotion bug where `hard_copy` missed
        BatchNorm.
        """
        polyak_tensor[target, Self.DIM_](
            self.gamma.val, src.gamma.val, tau, ctx
        )
        polyak_tensor[target, Self.DIM_](
            self.beta.val, src.beta.val, tau, ctx
        )
