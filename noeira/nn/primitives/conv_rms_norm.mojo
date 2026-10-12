"""ConvRMSNorm[C, HW] — channel-wise RMSNorm for NCHW conv feature maps.

DreamerV3 normalizes conv activations OVER THE CHANNEL AXIS, per spatial
location (reference `rssm.py`: `Conv → Norm(rms) → act`, norm over the last
axis = channels). The plain `RMSNorm[DIM]` normalizes over the whole flat
feature map; here we instead treat a flat NCHW slab `[B, C*HW]` (element
`(b,c,p)` at `b*C*HW + c*HW + p`) as `B*HW` independent rows of length `C` and
RMS-normalize each over its `C` channels, with one shared γ of size `C`
(broadcast across all spatial positions, like the reference).

This is the strided-channel twin of `rms_norm.mojo` — same math, but the
reduction axis (C) has stride HW in NCHW, so it needs its own kernels rather
than reusing the contiguous-DIM ones. (In NHWC channels are contiguous and the
plain RMSNorm kernels could be reused via a [B*HW, C] reshape; we stay NCHW to
avoid the env/decoder/GIF pixel-order ripple.)

Backward (cache n, inv_rms):  R = Σ_c go·γ·n ; dx = inv_rms·(go·γ - n·R/C) ;
                              dγ += Σ_{b,p} go·n
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


comptime CRN_EPS: Scalar[DT] = 1e-4
comptime CRN_TPB: Int = 128
comptime CRN_ACC = get_accum_type[DT]()


# ── GPU kernels (block-per-(b,p); reduce over the C channels, stride HW) ────
def _conv_rms_forward_kernel[
    B: Int, C: Int, HW: Int
](
    input: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    cache_inv_rms: LayoutTensor[DT, Layout.row_major(B * HW), MutAnyOrigin],
):
    var row = Int(block_idx.x)  # 0 .. B*HW-1  → one spatial location of one b
    var t = Int(thread_idx.x)
    if row >= B * HW:
        return
    var b = row // HW
    var p = row % HW
    var inv_c = Scalar[CRN_ACC](1.0) / Scalar[CRN_ACC](C)
    var my_sumsq = Scalar[CRN_ACC](0)
    var c = t
    while c < C:
        var x = rebind[Scalar[DT]](input[b, c * HW + p]).cast[CRN_ACC]()
        my_sumsq += x * x
        c += CRN_TPB
    var mean2 = (
        block.sum[block_size=CRN_TPB, broadcast=True](val=my_sumsq) * inv_c
    )
    var inv_rms = Scalar[CRN_ACC](1.0) / sqrt(mean2 + CRN_EPS.cast[CRN_ACC]())
    if t == 0:
        cache_inv_rms[row] = inv_rms.cast[DT]()
    c = t
    while c < C:
        var x = rebind[Scalar[DT]](input[b, c * HW + p]).cast[CRN_ACC]()
        var n = x * inv_rms
        cache_norm[b, c * HW + p] = n.cast[DT]()
        var g = rebind[Scalar[DT]](gamma[c]).cast[CRN_ACC]()
        output[b, c * HW + p] = (n * g).cast[DT]()
        c += CRN_TPB


def _conv_rms_backward_dx_kernel[
    B: Int, C: Int, HW: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    cache_inv_rms: LayoutTensor[DT, Layout.row_major(B * HW), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
):
    var row = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if row >= B * HW:
        return
    var b = row // HW
    var p = row % HW
    var inv_c = Scalar[CRN_ACC](1.0) / Scalar[CRN_ACC](C)
    var inv_rms = rebind[Scalar[DT]](cache_inv_rms[row]).cast[CRN_ACC]()
    var my_r = Scalar[CRN_ACC](0)
    var c = t
    while c < C:
        var go = rebind[Scalar[DT]](grad_output[b, c * HW + p]).cast[CRN_ACC]()
        var gm = rebind[Scalar[DT]](gamma[c]).cast[CRN_ACC]()
        var n = rebind[Scalar[DT]](cache_norm[b, c * HW + p]).cast[CRN_ACC]()
        my_r += go * gm * n
        c += CRN_TPB
    var R = block.sum[block_size=CRN_TPB, broadcast=True](val=my_r)
    c = t
    while c < C:
        var go = rebind[Scalar[DT]](grad_output[b, c * HW + p]).cast[CRN_ACC]()
        var gm = rebind[Scalar[DT]](gamma[c]).cast[CRN_ACC]()
        var n = rebind[Scalar[DT]](cache_norm[b, c * HW + p]).cast[CRN_ACC]()
        grad_input[b, c * HW + p] = (
            inv_rms * (go * gm - n * R * inv_c)
        ).cast[DT]()
        c += CRN_TPB


def _conv_rms_backward_dgamma_kernel[
    B: Int, C: Int, HW: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(B, C * HW), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)  # one channel per block
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var my_dg: Scalar[DT] = 0.0
    var i = t
    while i < B * HW:
        var b = i // HW
        var p = i % HW
        var go = rebind[Scalar[DT]](grad_output[b, c * HW + p])
        var n = rebind[Scalar[DT]](cache_norm[b, c * HW + p])
        my_dg += go * n
        i += CRN_TPB
    var total_dg = block.sum[block_size=CRN_TPB, broadcast=False](val=my_dg)
    if t == 0:
        grad_gamma[c] = rebind[Scalar[DT]](grad_gamma[c]) + total_dg[0]


struct ConvRMSNorm[C_: Int, HW_: Int](Module):
    """Channel-wise RMSNorm over an NCHW conv map `[B, C_*HW_]` (γ size C_)."""

    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.C_ * Self.HW_)
    comptime OUT_DIM = Self.C_ * Self.HW_

    var gamma: Param["gamma", False, Self.C_]
    var cache_norm: Tensor  # [B, C*HW]
    var cache_inv_rms: Tensor  # [B*HW]

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.C_]()
        self.cache_norm = Tensor()
        self.cache_inv_rms = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var rn = Self()
        rn.gamma = Param["gamma", False, Self.C_].make[target](ctx)
        for k in range(Self.C_):
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
        polyak_tensor[target, Self.C_](self.gamma.val, src.gamma.val, tau, ctx)


    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime DIM = Self.C_ * Self.HW_
        comptime if target == "cpu":
            out.ensure(B * DIM)
            self.cache_norm.ensure(B * DIM)
            self.cache_inv_rms.ensure(B * Self.HW_)
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var nm_p = self.cache_norm.data.unsafe_ptr()
            var iv_p = self.cache_inv_rms.data.unsafe_ptr()
            var inv_c: Scalar[DT] = 1.0 / Float32(Self.C_)
            for b in range(B):
                var base = b * DIM
                for p in range(Self.HW_):
                    var sumsq: Scalar[DT] = 0.0
                    for c in range(Self.C_):
                        var x = in_p[unsafe_offset=base + c * Self.HW_ + p]
                        sumsq += x * x
                    var inv_rms = Scalar[DT](1.0) / sqrt(sumsq * inv_c + CRN_EPS)
                    iv_p[unsafe_offset=b * Self.HW_ + p] = inv_rms
                    for c in range(Self.C_):
                        var idx = base + c * Self.HW_ + p
                        var n = in_p[unsafe_offset=idx] * inv_rms
                        nm_p[unsafe_offset=idx] = n
                        out_p[unsafe_offset=idx] = n * g_p[unsafe_offset=c]
        else:
            var cc = ctx.value()
            out.ensure_gpu(cc, B * DIM)
            self.cache_norm.ensure_gpu(cc, B * DIM)
            self.cache_inv_rms.ensure_gpu(cc, B * Self.HW_)
            comptime l2d = Layout.row_major(B, DIM)
            comptime lr = Layout.row_major(B * Self.HW_)
            comptime lc = Layout.row_major(Self.C_)
            cc.enqueue_function[
                _conv_rms_forward_kernel[B, Self.C_, Self.HW_]
            ](
                in0.lt["gpu", l2d](),
                out.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", lc](),
                self.cache_norm.lt["gpu", l2d](),
                self.cache_inv_rms.lt["gpu", lr](),
                grid_dim=B * Self.HW_,
                block_dim=CRN_TPB,
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
        comptime DIM = Self.C_ * Self.HW_
        comptime if target == "cpu":
            gin.ensure(B * DIM)
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var gg_p = self.gamma.grd.data.unsafe_ptr()
            var nm_p = self.cache_norm.data.unsafe_ptr()
            var iv_p = self.cache_inv_rms.data.unsafe_ptr()
            var inv_c: Scalar[DT] = 1.0 / Float32(Self.C_)
            for b in range(B):
                var base = b * DIM
                for p in range(Self.HW_):
                    var inv_rms = iv_p[unsafe_offset=b * Self.HW_ + p]
                    var R: Scalar[DT] = 0.0
                    for c in range(Self.C_):
                        var idx = base + c * Self.HW_ + p
                        R += go_p[unsafe_offset=idx] * g_p[unsafe_offset=c] * nm_p[unsafe_offset=idx]
                    for c in range(Self.C_):
                        var idx = base + c * Self.HW_ + p
                        var go = go_p[unsafe_offset=idx]
                        var n = nm_p[unsafe_offset=idx]
                        gi_p[unsafe_offset=idx] = inv_rms * (go * g_p[unsafe_offset=c] - n * R * inv_c)
                        gg_p[unsafe_offset=c] = gg_p[unsafe_offset=c] + go * n
        else:
            var cc = ctx.value()
            gin.ensure_gpu(cc, B * DIM)
            comptime l2d = Layout.row_major(B, DIM)
            comptime lr = Layout.row_major(B * Self.HW_)
            comptime lc = Layout.row_major(Self.C_)
            cc.enqueue_function[
                _conv_rms_backward_dx_kernel[B, Self.C_, Self.HW_]
            ](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", lc](),
                self.cache_norm.lt["gpu", l2d](),
                self.cache_inv_rms.lt["gpu", lr](),
                gin.lt["gpu", l2d](),
                grid_dim=B * Self.HW_,
                block_dim=CRN_TPB,
            )
            cc.enqueue_function[
                _conv_rms_backward_dgamma_kernel[B, Self.C_, Self.HW_]
            ](
                grad_output.lt["gpu", l2d](),
                self.cache_norm.lt["gpu", l2d](),
                self.gamma.grd.lt["gpu", lc](),
                grid_dim=Self.C_,
                block_dim=CRN_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults.
