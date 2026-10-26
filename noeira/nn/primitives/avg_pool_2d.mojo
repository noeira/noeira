"""AvgPool2D[C, K, S, P, H, W] — 2D average pooling on the storage surface.

Storage-surface port of legacy `nn.primitives.AvgPool2D` (the kernels + SIMD
math are carried VERBATIM — only the surface changed). `[B, C·H·W]` in,
`[B, C·OH·OW]` out where `OH = (H + 2P - K)//S + 1`, `OW = (W + 2P - K)//S + 1`.

Padding convention: `count_include_pad = True` (PyTorch default). Denominator is
always `K·K`; padded cells contribute 0 to the sum but still count in the
average. No params, no cache — backward broadcasts each output gradient
uniformly to its `K·K` input window with weight `1/(K·K)`; padded lanes never
receive gradient. The legacy two-phase / cache machinery dissolves because the
storage surface never needs the input in backward.

GPU layout: forward is output-indexed (1 thread per output cell); backward is
input-indexed (1 thread per input cell, looping over overlapping output windows)
— no atomics even for overlapping pool configurations.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB, LAYOUT_NCHW, LAYOUT_NHWC
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP

# Reuse the conv2d channels-first/last index helpers (one source of truth).
from .conv2d import _in_off, _in_decode, _out_off, _out_decode


def _avg_pool_2d_forward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int, ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: Tile[ADT, BATCH, IN_FLAT],
    output: Tile[ADT, BATCH, OUT_FLAT],
    inv_kk: Scalar[DT],
):
    # AMP §3 fp32-INTERNAL: I/O is the activation dtype (`ADT`) but the pooling
    # sum accumulates in fp32 (`DT`). ADT == DT (default) → byte-identical.
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * OUT_FLAT
    if idx >= total:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var c, os = _out_decode[LAYOUT, C, OH * OW](out_pos)
    var oh = os // OW
    var ow = os % OW

    var s: Scalar[DT] = 0.0
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            s += rebind[Scalar[ADT]](
                input[b, _in_off[LAYOUT, C, H, W](c, ih, iw)]
            ).cast[DT]()
    output[b, out_pos] = (s * inv_kk).cast[ADT]()


def _avg_pool_2d_backward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int, ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: Tile[ADT, BATCH, OUT_FLAT],
    grad_input: Tile[ADT, BATCH, IN_FLAT],
    inv_kk: Scalar[DT],
):
    # AMP §3 fp32-INTERNAL: I/O `ADT`, gradient distribution accumulated in fp32.
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * IN_FLAT
    if idx >= total:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var c, ih, iw = _in_decode[LAYOUT, C, H, W](in_pos)

    # Output windows that contain input cell (ih, iw).
    var oh_max_raw = ih + P
    var ow_max_raw = iw + P
    if oh_max_raw < 0 or ow_max_raw < 0:
        grad_input[b, in_pos] = Scalar[ADT](0.0)
        return
    var oh_top = oh_max_raw // S
    var ow_top = ow_max_raw // S
    var oh_bot_raw = ih + P - K + 1
    var ow_bot_raw = iw + P - K + 1
    var oh_bot: Int = 0
    if oh_bot_raw > 0:
        oh_bot = (oh_bot_raw + S - 1) // S
    var ow_bot: Int = 0
    if ow_bot_raw > 0:
        ow_bot = (ow_bot_raw + S - 1) // S
    if oh_top >= OH:
        oh_top = OH - 1
    if ow_top >= OW:
        ow_top = OW - 1

    var acc: Scalar[DT] = 0.0
    var oh = oh_bot
    while oh <= oh_top:
        var ow = ow_bot
        while ow <= ow_top:
            acc += rebind[Scalar[ADT]](
                grad_output[b, _out_off[LAYOUT, C, OH * OW](c, oh * OW + ow)]
            ).cast[DT]()
            ow += 1
        oh += 1
    grad_input[b, in_pos] = (acc * inv_kk).cast[ADT]()


struct AvgPool2D[
    C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](Module):
    comptime ARITY: Int = 1
    comptime OH: Int = (Self.H + 2 * Self.P - Self.K) // Self.S + 1
    comptime OW: Int = (Self.W + 2 * Self.P - Self.K) // Self.S + 1
    comptime IN_FLAT: Int = Self.C * Self.H * Self.W
    comptime OUT_FLAT: Int = Self.C * Self.OH * Self.OW
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_FLAT)
    comptime OUT_DIM = Self.OUT_FLAT
    # Activation-flow dtype (AMP §3 fp32-INTERNAL): AvgPool accepts/emits ACT_DT
    # but accumulates the pooling sum in fp32. ACT_DT == DT (default) →
    # byte-identical to the fp32 path.
    comptime ACT_DT = Self.ADT

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "AvgPool2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.K > 0 and Self.S > 0, (
            "AvgPool2D: K and S must be positive"
        )
        comptime assert Self.OH > 0 and Self.OW > 0, (
            "AvgPool2D: invalid spatial shape — check H/W/K/S/P"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        var inv_kk = Scalar[DT](1.0 / Float64(Self.K * Self.K))
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_FLAT)
            for b in range(B):
                var in_base = b * Self.IN_FLAT
                var out_base = b * Self.OUT_FLAT
                for c in range(Self.C):
                    for oh in range(Self.OH):
                        for ow in range(Self.OW):
                            # fp32 accumulator (AMP §3 fp32-internal).
                            var s: Scalar[DT] = 0.0
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    s += in0.data[
                                        in_base
                                        + _in_off[
                                            Self.LAYOUT, Self.C, Self.H, Self.W
                                        ](c, ih, iw)
                                    ].cast[DT]()
                            out.data[
                                out_base
                                + _out_off[
                                    Self.LAYOUT, Self.C, Self.OH * Self.OW
                                ](c, oh * Self.OW + ow)
                            ] = (s * inv_kk).cast[Self.ACT_DT]()
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            comptime total = B * Self.OUT_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _avg_pool_2d_forward_kernel[
                    B, Self.C, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                    Self.IN_FLAT, Self.OUT_FLAT, Self.ACT_DT, Self.LAYOUT,
                ]
            ](
                in0.tt["gpu", B, Self.IN_FLAT](),
                out.tt["gpu", B, Self.OUT_FLAT](),
                inv_kk,
                grid_dim=n_blocks, block_dim=TPB,
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
        var inv_kk = Scalar[DT](1.0 / Float64(Self.K * Self.K))
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_FLAT)
            # fp32 scatter accumulator (AMP §3 fp32-internal); cast to ACT_DT
            # at the end. ACT_DT == DT (default) → byte-identical.
            var acc = List[Scalar[DT]](length=B * Self.IN_FLAT, fill=0.0)
            for b in range(B):
                var in_base = b * Self.IN_FLAT
                var out_base = b * Self.OUT_FLAT
                for c in range(Self.C):
                    for oh in range(Self.OH):
                        for ow in range(Self.OW):
                            var go_val = (
                                grad_output.data[
                                    out_base
                                    + _out_off[
                                        Self.LAYOUT, Self.C, Self.OH * Self.OW
                                    ](c, oh * Self.OW + ow)
                                ].cast[DT]()
                                * inv_kk
                            )
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    acc[
                                        in_base
                                        + _in_off[
                                            Self.LAYOUT, Self.C, Self.H, Self.W
                                        ](c, ih, iw)
                                    ] += go_val
            for k in range(B * Self.IN_FLAT):
                gin.data[k] = acc[k].cast[Self.ACT_DT]()
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            comptime total = B * Self.IN_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _avg_pool_2d_backward_kernel[
                    B, Self.C, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                    Self.IN_FLAT, Self.OUT_FLAT, Self.ACT_DT, Self.LAYOUT,
                ]
            ](
                grad_output.tt["gpu", B, Self.OUT_FLAT](),
                gin.tt["gpu", B, Self.IN_FLAT](),
                inv_kk,
                grid_dim=n_blocks, block_dim=TPB,
            )

    # for_each_param / zero_grad / polyak_from inherit the Module defaults
    # (param-less → no-op).
