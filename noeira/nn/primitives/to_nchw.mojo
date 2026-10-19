"""ToNCHW[C, H, W, LAYOUT] — channels-last→channels-first latent adapter.

The channels_last migration flips only the conv TOWER (e.g. the EZv2 rep net) to
NHWC, while downstream consumers (dynamics / prediction / projector / MCTS) keep
the canonical NCHW `[C,H,W]`-flat latent. This module sits at that boundary: it
transposes its input from `Self.LAYOUT` to NCHW so the tower can be NHWC while
everything after stays NCHW with no changes.

  forward: out[b, c*H*W + hw] = in[b, off_LAYOUT(c, hw)]
  vjp:     gin[b, off_LAYOUT(c, hw)] = go[b, c*H*W + hw]
  where off_NCHW(c,hw)=c*H*W+hw (identity — passthrough copy, bit-identical to no
  adapter) and off_NHWC(c,hw)=hw*C+c (the [H,W,C]→[C,H,W] transpose).

Tiny by construction (the latent is small, e.g. 64·6·6); cost is negligible vs
the tower it unlocks. Appended to `Rep` via a comptime `LAYOUT` so NCHW configs
get a pure identity copy.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, LAYOUT_NCHW
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .conv2d import _out_off
from noeira.core.tiles import Tile


def _to_nchw_kernel[
    B: Int, C: Int, HW: Int, DIM: Int, LAYOUT: Int, ADT: DType = DT
](
    src: Tile[ADT, B * DIM],
    dst: Tile[ADT, B * DIM],
):
    # dst is NCHW [C,H,W]-flat; src is Self.LAYOUT. One thread per dst element.
    var idx = Int(global_idx.x)
    if idx >= B * DIM:
        return
    var b = idx // DIM
    var pos = idx % DIM           # = c*HW + hw  (NCHW output position)
    var c = pos // HW
    var hw = pos % HW
    dst[idx] = rebind[Scalar[ADT]](src[b * DIM + _out_off[LAYOUT, C, HW](c, hw)])


def _from_nchw_kernel[
    B: Int, C: Int, HW: Int, DIM: Int, LAYOUT: Int, ADT: DType = DT
](
    go: Tile[ADT, B * DIM],
    gin: Tile[ADT, B * DIM],
):
    # vjp: scatter NCHW grad back into Self.LAYOUT. One thread per NCHW element.
    var idx = Int(global_idx.x)
    if idx >= B * DIM:
        return
    var b = idx // DIM
    var pos = idx % DIM
    var c = pos // HW
    var hw = pos % HW
    gin[b * DIM + _out_off[LAYOUT, C, HW](c, hw)] = rebind[Scalar[ADT]](go[idx])


struct ToNCHW[
    C_: Int, H_: Int, W_: Int,
    LAYOUT: Int = LAYOUT_NCHW,
    ADT: DType = DT,
](Module):
    comptime ARITY = 1
    comptime DIM = Self.C_ * Self.H_ * Self.W_
    comptime HW = Self.H_ * Self.W_
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM
    comptime ACT_DT = Self.ADT

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            comptime if Self.LAYOUT == LAYOUT_NCHW:
                for k in range(N):
                    out.data[k] = in0.data[k]
            else:
                for b in range(B):
                    var base = b * Self.DIM
                    for c in range(Self.C_):
                        for hw in range(Self.HW):
                            out.data[base + c * Self.HW + hw] = in0.data[
                                base + _out_off[Self.LAYOUT, Self.C_, Self.HW](c, hw)
                            ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[
                _to_nchw_kernel[
                    B, Self.C_, Self.HW, Self.DIM, Self.LAYOUT, Self.ACT_DT
                ]
            ](
                in0.tt["gpu", N](),
                out.tt["gpu", N](),
                grid_dim=nblk,
                block_dim=TPB,
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
        comptime N = B * Self.DIM
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            comptime if Self.LAYOUT == LAYOUT_NCHW:
                for k in range(N):
                    gin.data[k] = grad_output.data[k]
            else:
                for b in range(B):
                    var base = b * Self.DIM
                    for c in range(Self.C_):
                        for hw in range(Self.HW):
                            gin.data[
                                base + _out_off[Self.LAYOUT, Self.C_, Self.HW](c, hw)
                            ] = grad_output.data[base + c * Self.HW + hw]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[
                _from_nchw_kernel[
                    B, Self.C_, Self.HW, Self.DIM, Self.LAYOUT, Self.ACT_DT
                ]
            ](
                grad_output.tt["gpu", N](),
                gin.tt["gpu", N](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults.
