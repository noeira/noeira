"""Upsample2x[C, H, W] — nearest-neighbor ×2 spatial upsample (NCHW, no params).

The DreamerV3 reference decoder at `strided: False` upsamples with
`x.repeat(2, -2).repeat(2, -3)` (nearest-neighbor) before each kernel-5
stride-1 conv (rssm.py Decoder). This is that repeat as a leaf Module:

    out[c, 2h+dh, 2w+dw] = in[c, h, w]        (dh, dw ∈ {0,1})

Backward: each input cell receives the SUM of its 2×2 output cells'
gradients. Pure architectural op — no params, no cache. CPU + GPU.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


def _up2x_fwd_kernel[
    B: Int, C: Int, H: Int, W: Int,
](
    inp: Tile[DT, B * C * H * W],
    dst: Tile[DT, B * C * (2 * H) * (2 * W)],
):
    """One thread per OUTPUT element; reads its source cell."""
    comptime OH = 2 * H
    comptime OW = 2 * W
    var i = Int(global_idx.x)
    if i < B * C * OH * OW:
        var ow = i % OW
        var oh = (i // OW) % OH
        var bc = i // (OH * OW)
        dst[i] = inp[bc * (H * W) + (oh // 2) * W + (ow // 2)]


def _up2x_bwd_kernel[
    B: Int, C: Int, H: Int, W: Int,
](
    gout: Tile[DT, B * C * (2 * H) * (2 * W)],
    gin: Tile[DT, B * C * H * W],
):
    """One thread per INPUT element; sums its 2×2 output cell grads."""
    comptime OH = 2 * H
    comptime OW = 2 * W
    var i = Int(global_idx.x)
    if i < B * C * H * W:
        var w = i % W
        var h = (i // W) % H
        var bc = i // (H * W)
        var base = bc * (OH * OW)
        var acc = Scalar[DT](0.0)
        for dh in range(2):
            for dw in range(2):
                acc += rebind[Scalar[DT]](
                    gout[base + (2 * h + dh) * OW + (2 * w + dw)]
                )
        gin[i] = acc


struct Upsample2x[C: Int, H: Int, W: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_FLAT: Int = Self.C * Self.H * Self.W
    comptime OUT_FLAT: Int = Self.C * (2 * Self.H) * (2 * Self.W)
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_FLAT)
    comptime OUT_DIM = Self.OUT_FLAT

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params) but accepted
        for Sequential.make[target, INIT] uniformity."""
        comptime assert target == "cpu" or target == "gpu", (
            "Upsample2x: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.C > 0 and Self.H > 0 and Self.W > 0, (
            "Upsample2x: C, H, W must be positive"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime OH = 2 * Self.H
        comptime OW = 2 * Self.W
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_FLAT)
            for b in range(B):
                for c in range(Self.C):
                    var src = (b * Self.C + c) * (Self.H * Self.W)
                    var dst = (b * Self.C + c) * (OH * OW)
                    for h in range(Self.H):
                        for w in range(Self.W):
                            var v = in0.data[src + h * Self.W + w]
                            var d0 = dst + (2 * h) * OW + 2 * w
                            out.data[d0] = v
                            out.data[d0 + 1] = v
                            out.data[d0 + OW] = v
                            out.data[d0 + OW + 1] = v
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            comptime n_blocks = (B * Self.OUT_FLAT + TPB - 1) // TPB
            comptime kernel = _up2x_fwd_kernel[B, Self.C, Self.H, Self.W]
            c.enqueue_function[kernel](
                in0.tt["gpu", B * Self.IN_FLAT](),
                out.tt["gpu", B * Self.OUT_FLAT](),
                grid_dim=n_blocks,
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime OH = 2 * Self.H
        comptime OW = 2 * Self.W
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_FLAT)
            for b in range(B):
                for c in range(Self.C):
                    var src = (b * Self.C + c) * (OH * OW)
                    var dst = (b * Self.C + c) * (Self.H * Self.W)
                    for h in range(Self.H):
                        for w in range(Self.W):
                            var s0 = src + (2 * h) * OW + 2 * w
                            gin.data[dst + h * Self.W + w] = (
                                grad_output.data[s0]
                                + grad_output.data[s0 + 1]
                                + grad_output.data[s0 + OW]
                                + grad_output.data[s0 + OW + 1]
                            )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            comptime n_blocks = (B * Self.IN_FLAT + TPB - 1) // TPB
            comptime kernel = _up2x_bwd_kernel[B, Self.C, Self.H, Self.W]
            c.enqueue_function[kernel](
                grad_output.tt["gpu", B * Self.OUT_FLAT](),
                gin.tt["gpu", B * Self.IN_FLAT](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
