"""RMSNorm[DIM] — root-mean-square normalization (storage surface).

Transformed from legacy `nn.primitives.RMSNorm` (surface-only change). No mean
subtraction, no β — one `Param` (γ, decay=False, init 1). The x·inv_rms + inv_rms
cache is leaf-owned. CPU SIMD feature-axis loops + 3 GPU kernels (forward /
backward-dx 1-block-per-row / backward-dgamma 1-block-per-col) carried verbatim.

Backward (cache):  R = Σ_d go·γ·n ; dx = inv_rms·(go·γ - n·R/DIM) ; dγ += Σ_b go·n
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.utils.numerics import get_accum_type
from layout import Layout, LayoutTensor, TileTensor, row_major

from noeira.nn.constants import DT, CPU_SIMD_W
from ..core.polyak import polyak_tensor
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime RMS_EPS: Scalar[DT] = 1e-4
"""Default epsilon. ⚠ It is NOT the value every model wants — SmolLM2 and Llama
use `rms_norm_eps = 1e-5`, a factor of ten smaller, and the difference is
visible in a parity check while being invisible to any shape or NaN test. Pass
`RMSNorm[DIM, EPS=…]` rather than relying on this."""
comptime RMS_TPB: Int = 128
# Reductions run in the accumulation dtype (f32 for bf16 inputs; identity for
# DT=f32). ELEMS = per-thread feature slice; each thread reads its slice ONCE
# into registers, computes Σx² in that single read, then normalizes from
# registers — vs the legacy 2× input re-read.
comptime RMS_ACC = get_accum_type[DT]()
comptime RMS_REG_CAP = 8  # max per-thread slice (≈ DIM≤1024) to register-cache


# ── GPU kernels (single-pass register-cached forward; block-per-row) ────
def _rms_norm_forward_kernel[
    BATCH: Int,
    DIM: Int,
    EPS: Scalar[DT] = RMS_EPS,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_rms: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + RMS_TPB - 1) // RMS_TPB
    # Register-cache the thread's feature slice only when small enough to stay
    # in registers (≤ RMS_REG_CAP); else 2-read fallback (still beats the
    # legacy 2 reads only marginally — kept for the no-spill guarantee).
    comptime REG_CACHE = ELEMS <= RMS_REG_CAP
    var inv_dim = Scalar[RMS_ACC](1.0) / Scalar[RMS_ACC](DIM)
    var my_sumsq = Scalar[RMS_ACC](0)

    comptime if REG_CACHE:
        var slice = Array[Scalar[RMS_ACC], ELEMS](fill=Scalar[RMS_ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * RMS_TPB
            if col < DIM:
                var x = rebind[Scalar[DT]](input[b, col]).cast[RMS_ACC]()
                slice[e] = x
                my_sumsq += x * x
        var mean2 = (
            block.sum[block_size=RMS_TPB, broadcast=True](val=my_sumsq)
            * inv_dim
        )
        var inv_rms = Scalar[RMS_ACC](1.0) / sqrt(
            mean2 + EPS.cast[RMS_ACC]()
        )
        if t == 0:
            cache_inv_rms[b] = inv_rms.cast[DT]()

        comptime for e in range(ELEMS):
            var col = t + e * RMS_TPB
            if col < DIM:
                var n = slice[e] * inv_rms
                cache_norm[b, col] = n.cast[DT]()
                var g = rebind[Scalar[DT]](gamma[col]).cast[RMS_ACC]()
                output[b, col] = (n * g).cast[DT]()
    else:
        var idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx]).cast[RMS_ACC]()
            my_sumsq += x * x
            idx += RMS_TPB
        var mean2 = (
            block.sum[block_size=RMS_TPB, broadcast=True](val=my_sumsq)
            * inv_dim
        )
        var inv_rms = Scalar[RMS_ACC](1.0) / sqrt(
            mean2 + EPS.cast[RMS_ACC]()
        )
        if t == 0:
            cache_inv_rms[b] = inv_rms.cast[DT]()
        idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx]).cast[RMS_ACC]()
            var n = x * inv_rms
            cache_norm[b, idx] = n.cast[DT]()
            var g = rebind[Scalar[DT]](gamma[idx]).cast[RMS_ACC]()
            output[b, idx] = (n * g).cast[DT]()
            idx += RMS_TPB


def _rms_norm_backward_dx_kernel[
    BATCH: Int,
    DIM: Int,
    EPS: Scalar[DT] = RMS_EPS,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_rms: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + RMS_TPB - 1) // RMS_TPB
    comptime REG_CACHE = ELEMS <= RMS_REG_CAP
    var inv_dim = Scalar[RMS_ACC](1.0) / Scalar[RMS_ACC](DIM)
    var inv_rms = rebind[Scalar[DT]](cache_inv_rms[b]).cast[RMS_ACC]()
    var my_r = Scalar[RMS_ACC](0)

    comptime if REG_CACHE:
        # Cache gg=go·γ and n once; the write pass reads no global memory.
        var gg_s = Array[Scalar[RMS_ACC], ELEMS](fill=Scalar[RMS_ACC](0))
        var n_s = Array[Scalar[RMS_ACC], ELEMS](fill=Scalar[RMS_ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * RMS_TPB
            if col < DIM:
                var go = rebind[Scalar[DT]](grad_output[b, col]).cast[RMS_ACC]()
                var gm = rebind[Scalar[DT]](gamma[col]).cast[RMS_ACC]()
                var n = rebind[Scalar[DT]](cache_norm[b, col]).cast[RMS_ACC]()
                var gg = go * gm
                gg_s[e] = gg
                n_s[e] = n
                my_r += gg * n
        var R = block.sum[block_size=RMS_TPB, broadcast=True](val=my_r)

        comptime for e in range(ELEMS):
            var col = t + e * RMS_TPB
            if col < DIM:
                grad_input[b, col] = (
                    inv_rms * (gg_s[e] - n_s[e] * R * inv_dim)
                ).cast[DT]()
    else:
        var idx = t
        while idx < DIM:
            var go = rebind[Scalar[DT]](grad_output[b, idx]).cast[RMS_ACC]()
            var gm = rebind[Scalar[DT]](gamma[idx]).cast[RMS_ACC]()
            var n = rebind[Scalar[DT]](cache_norm[b, idx]).cast[RMS_ACC]()
            my_r += go * gm * n
            idx += RMS_TPB
        var R = block.sum[block_size=RMS_TPB, broadcast=True](val=my_r)
        idx = t
        while idx < DIM:
            var go = rebind[Scalar[DT]](grad_output[b, idx]).cast[RMS_ACC]()
            var gm = rebind[Scalar[DT]](gamma[idx]).cast[RMS_ACC]()
            var n = rebind[Scalar[DT]](cache_norm[b, idx]).cast[RMS_ACC]()
            grad_input[b, idx] = (
                inv_rms * (go * gm - n * R * inv_dim)
            ).cast[DT]()
            idx += RMS_TPB


def _rms_norm_backward_dgamma_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_dg: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        var go = rebind[Scalar[DT]](grad_output[bi, col])
        var n = rebind[Scalar[DT]](cache_norm[bi, col])
        my_dg += go * n
        bi += RMS_TPB
    var total_dg = block.sum[block_size=RMS_TPB, broadcast=False](val=my_dg)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]


struct RMSNorm[DIM_: Int, EPS: Scalar[DT] = RMS_EPS](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var gamma: Param["gamma", False, Self.DIM_]
    var cache_norm: Tensor  # [BATCH, DIM]
    var cache_inv_rms: Tensor  # [BATCH]

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM_]()
        self.cache_norm = Tensor()
        self.cache_inv_rms = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var rn = Self()
        rn.gamma = Param["gamma", False, Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            rn.gamma.val.data[k] = Scalar[DT](1.0)
        comptime if target != "cpu":
            rn.gamma.val.upload(ctx.value())
        return rn^

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update toward `src` (target <- online). The `Module`
        default is a NO-OP, so without this override a target net containing
        this layer would keep its init values forever while the online copy
        trains (the `LayerNorm` / `Conv2D` bug class; audit 2026-10-07)."""
        polyak_tensor[target, Self.DIM_](self.gamma.val, src.gamma.val, tau, ctx)


    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            self.cache_norm.ensure(B * Self.DIM_)
            self.cache_inv_rms.ensure(B)
            var inv_v = TileTensor(self.cache_inv_rms.data, row_major[B]())
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var nm_p = self.cache_norm.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
            for b in range(B):
                var row = b * Self.DIM_
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM_:
                    var x = in_p.unsafe_load[width=W](row + d)
                    acc += x * x
                    d += W
                var sumsq = acc.reduce_add()
                while d < Self.DIM_:
                    var x = in_p[unsafe_offset=row + d]
                    sumsq += x * x
                    d += 1
                var mean2 = sumsq * inv_dim
                var inv_rms = Scalar[DT](1.0) / sqrt(mean2 + Self.EPS)
                inv_v[b] = inv_rms
                var irv = SIMD[DT, W](inv_rms)
                d = 0
                while d + W <= Self.DIM_:
                    var n = in_p.unsafe_load[width=W](row + d) * irv
                    nm_p.unsafe_store(row + d, n)
                    out_p.unsafe_store(row + d, n * g_p.unsafe_load[width=W](d))
                    d += W
                while d < Self.DIM_:
                    var n = in_p[unsafe_offset=row + d] * inv_rms
                    nm_p[unsafe_offset=row + d] = n
                    out_p[unsafe_offset=row + d] = n * g_p[unsafe_offset=d]
                    d += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_norm.ensure_gpu(c, B * Self.DIM_)
            self.cache_inv_rms.ensure_gpu(c, B)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime lb = Layout.row_major(B)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[_rms_norm_forward_kernel[B, Self.DIM_, Self.EPS]](
                in0.lt["gpu", l2d](),
                out.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_norm.lt["gpu", l2d](),
                self.cache_inv_rms.lt["gpu", lb](),
                grid_dim=B,
                block_dim=RMS_TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var inv_v = TileTensor(self.cache_inv_rms.data, row_major[B]())
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var gg_p = self.gamma.grd.data.unsafe_ptr()
            var nm_p = self.cache_norm.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
            for b in range(B):
                var row = b * Self.DIM_
                var inv_rms = inv_v[b]
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM_:
                    acc += (
                        go_p.unsafe_load[width=W](row + d)
                        * g_p.unsafe_load[width=W](d)
                        * nm_p.unsafe_load[width=W](row + d)
                    )
                    d += W
                var R = acc.reduce_add()
                while d < Self.DIM_:
                    R += go_p[unsafe_offset=row + d] * g_p[unsafe_offset=d] * nm_p[unsafe_offset=row + d]
                    d += 1
                var irv = SIMD[DT, W](inv_rms)
                var rscaled = SIMD[DT, W](R * inv_dim)
                d = 0
                while d + W <= Self.DIM_:
                    var go = go_p.unsafe_load[width=W](row + d)
                    var n = nm_p.unsafe_load[width=W](row + d)
                    gi_p.unsafe_store(
                        row + d, irv * (go * g_p.unsafe_load[width=W](d) - n * rscaled)
                    )
                    d += W
                while d < Self.DIM_:
                    var go = go_p[unsafe_offset=row + d]
                    gi_p[unsafe_offset=row + d] = inv_rms * (
                        go * g_p[unsafe_offset=d] - nm_p[unsafe_offset=row + d] * R * inv_dim
                    )
                    d += 1
                # dγ += go·n  (accumulated across the batch)
                d = 0
                while d + W <= Self.DIM_:
                    gg_p.unsafe_store(
                        d,
                        gg_p.unsafe_load[width=W](d)
                        + go_p.unsafe_load[width=W](row + d)
                        * nm_p.unsafe_load[width=W](row + d),
                    )
                    d += W
                while d < Self.DIM_:
                    gg_p[unsafe_offset=d] = gg_p[unsafe_offset=d] + go_p[unsafe_offset=row + d] * nm_p[unsafe_offset=row + d]
                    d += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime lb = Layout.row_major(B)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[_rms_norm_backward_dx_kernel[B, Self.DIM_, Self.EPS]](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_norm.lt["gpu", l2d](),
                self.cache_inv_rms.lt["gpu", lb](),
                gin.lt["gpu", l2d](),
                grid_dim=B,
                block_dim=RMS_TPB,
            )
            c.enqueue_function[_rms_norm_backward_dgamma_kernel[B, Self.DIM_]](
                grad_output.lt["gpu", l2d](),
                self.cache_norm.lt["gpu", l2d](),
                self.gamma.grd.lt["gpu", ld](),
                grid_dim=Self.DIM_,
                block_dim=RMS_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
