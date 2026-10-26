"""MaxPool2D[C, K, S, P, H, W] — 2D max-pooling on the storage surface.

Storage-surface port of legacy `nn.primitives.MaxPool2D` (kernels + SIMD math
carried VERBATIM — only the surface changed). `[B, C·H·W]` in, `[B, C·OH·OW]`
out where `OH = (H + 2P - K)//S + 1`, `OW = (W + 2P - K)//S + 1`.

No params, no leaf-owned cache: backward re-scans each pooling window through the
input slab to re-find the argmax. The storage surface passes `forward_input`
explicitly into `vjp` (invariant §3.1), so the legacy `_cached_input_ptr` /
two-phase machinery dissolves — `forward_input[0]` IS the input slab.

Tie-break: first lane in row-major (kh, kw) iteration order wins (PyTorch
convention). Padded (OOB) lanes contribute `-inf` so they never win nor receive
gradient. GPU backward is input-indexed (one thread per input cell, recomputing
argmax for each covering window) — pure single-writer per output cell, no race
even with overlapping pools.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from max.algorithm import parallelize
from noeira.nn.constants import DT, TPB, LAYOUT_NCHW, LAYOUT_NHWC
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP

# Reuse the conv2d channels-first/last index helpers (generic Int params; the
# C/OSP shapes map onto IC/OC/SO). One source of truth, already gated.
from .conv2d import _in_off, _in_decode, _out_off, _out_decode


comptime MP_NEG_INF: Scalar[DT] = -1.0e30


def _max_pool_2d_forward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: Tile[DT, BATCH, IN_FLAT],
    output: Tile[DT, BATCH, OUT_FLAT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * OUT_FLAT
    if idx >= total:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var c, os = _out_decode[LAYOUT, C, OH * OW](out_pos)
    var oh = os // OW
    var ow = os % OW

    var best: Scalar[DT] = MP_NEG_INF
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            var v = rebind[Scalar[DT]](
                input[b, _in_off[LAYOUT, C, H, W](c, ih, iw)]
            )
            if v > best:
                best = v
    output[b, out_pos] = best


def _max_pool_2d_backward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: Tile[DT, BATCH, OUT_FLAT],
    input: Tile[DT, BATCH, IN_FLAT],
    grad_input: Tile[DT, BATCH, IN_FLAT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * IN_FLAT
    if idx >= total:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var c, ih, iw = _in_decode[LAYOUT, C, H, W](in_pos)

    # Output positions whose receptive field covers (ih, iw):
    #     oh ∈ [ceil((ih + P - K + 1) / S), floor((ih + P) / S)] ∩ [0, OH-1]
    # Implemented with safe integer math (no floor of negative).
    var oh_max_raw = ih + P
    var ow_max_raw = iw + P
    if oh_max_raw < 0 or ow_max_raw < 0:
        grad_input[b, in_pos] = Scalar[DT](0.0)
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
            # Recompute argmax for window (oh, ow).
            var best: Scalar[DT] = MP_NEG_INF
            var best_ih: Int = -1
            var best_iw: Int = -1
            for kh in range(K):
                var win_ih = oh * S + kh - P
                if win_ih < 0 or win_ih >= H:
                    continue
                for kw in range(K):
                    var win_iw = ow * S + kw - P
                    if win_iw < 0 or win_iw >= W:
                        continue
                    var v = rebind[Scalar[DT]](
                        input[b, _in_off[LAYOUT, C, H, W](c, win_ih, win_iw)]
                    )
                    if v > best:
                        best = v
                        best_ih = win_ih
                        best_iw = win_iw
            if best_ih == ih and best_iw == iw:
                acc += rebind[Scalar[DT]](
                    grad_output[
                        b, _out_off[LAYOUT, C, OH * OW](c, oh * OW + ow)
                    ]
                )
            ow += 1
        oh += 1
    grad_input[b, in_pos] = acc


struct MaxPool2D[
    C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](Module):
    comptime ARITY: Int = 1
    comptime OH: Int = (Self.H + 2 * Self.P - Self.K) // Self.S + 1
    comptime OW: Int = (Self.W + 2 * Self.P - Self.K) // Self.S + 1
    comptime IN_FLAT: Int = Self.C * Self.H * Self.W
    comptime OUT_FLAT: Int = Self.C * Self.OH * Self.OW
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_FLAT)
    comptime OUT_DIM = Self.OUT_FLAT

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MaxPool2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.K > 0 and Self.S > 0, (
            "MaxPool2D: K and S must be positive"
        )
        comptime assert Self.OH > 0 and Self.OW > 0, (
            "MaxPool2D: invalid spatial shape — check H/W/K/S/P"
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
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_FLAT)
            # ⚠⚠ ONE CALL, 9.43 ms, 15.6% OF ACT's CPU FORWARD. Measured in
            # situ: ResNet18's single 3x3/2 pool over 2,457,600 inputs (two
            # cameras batched) is 5.5M taps of scalar work with an index
            # computation and two branches EACH, and it was the largest item
            # left after im2col was threaded. Splitting it over (batch,
            # channel) — each pair owning a disjoint output plane, so there is
            # nothing to synchronise — is the same treatment and the same
            # measured ~200 us launch cost applies.
            comptime TASKS = B * Self.C
            comptime TAPS = B * Self.C * Self.OH * Self.OW * Self.K * Self.K
            var _ip = in0.data.unsafe_ptr()
            var _op = out.data.unsafe_ptr()

            def _plane(t: Int) {imm}:
                var b = t // Self.C
                var c = t % Self.C
                var in_base = b * Self.IN_FLAT
                var out_base = b * Self.OUT_FLAT
                for oh in range(Self.OH):
                    for ow in range(Self.OW):
                        var best: Scalar[DT] = MP_NEG_INF
                        for kh in range(Self.K):
                            var ih = oh * Self.S + kh - Self.P
                            if ih < 0 or ih >= Self.H:
                                continue
                            for kw in range(Self.K):
                                var iw = ow * Self.S + kw - Self.P
                                if iw < 0 or iw >= Self.W:
                                    continue
                                var v = _ip[
                                    unsafe_offset = in_base
                                    + _in_off[
                                        Self.LAYOUT, Self.C, Self.H, Self.W
                                    ](c, ih, iw)
                                ]
                                if v > best:
                                    best = v
                        _op[
                            unsafe_offset = out_base
                            + _out_off[
                                Self.LAYOUT, Self.C, Self.OH * Self.OW
                            ](c, oh * Self.OW + ow)
                        ] = best

            # ⚠ THE GATE IS NOT A FINELY SWEPT ONE, and saying so is the point.
            # `parallelize`'s ~200 us fixed cost is measured (see conv2d.mojo's
            # table, where a blanket call made five shapes slower and one 20x
            # slower). ACT has exactly ONE pooling layer and it clears this
            # bound by more than 30x, so the thresholds are a floor that keeps
            # small pools off the thread pool, not a tuned crossover. A network
            # whose pools sit near it deserves its own measurement.
            comptime MP_PAR_MIN_TASKS = 16
            comptime MP_PAR_MIN_TAPS = 200_000
            comptime if TASKS >= MP_PAR_MIN_TASKS and TAPS >= MP_PAR_MIN_TAPS:
                parallelize(_plane, TASKS)
            else:
                for t in range(TASKS):
                    _plane(t)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            comptime total = B * Self.OUT_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _max_pool_2d_forward_kernel[
                    B, Self.C, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                    Self.IN_FLAT, Self.OUT_FLAT, Self.LAYOUT,
                ]
            ](
                in0.tt["gpu", B, Self.IN_FLAT](),
                out.tt["gpu", B, Self.OUT_FLAT](),
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_FLAT)
            # Zero-fill grad_input — we scatter argmax-only.
            for k in range(B * Self.IN_FLAT):
                gin.data[k] = Scalar[DT](0.0)
            for b in range(B):
                var in_base = b * Self.IN_FLAT
                var out_base = b * Self.OUT_FLAT
                for c in range(Self.C):
                    for oh in range(Self.OH):
                        for ow in range(Self.OW):
                            var best: Scalar[DT] = MP_NEG_INF
                            var best_idx: Int = -1
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    var idx = in_base + _in_off[
                                        Self.LAYOUT, Self.C, Self.H, Self.W
                                    ](c, ih, iw)
                                    var v = fin.data[idx]
                                    if v > best:
                                        best = v
                                        best_idx = idx
                            if best_idx >= 0:
                                gin.data[best_idx] += grad_output.data[
                                    out_base
                                    + _out_off[
                                        Self.LAYOUT, Self.C, Self.OH * Self.OW
                                    ](c, oh * Self.OW + ow)
                                ]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            comptime total = B * Self.IN_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _max_pool_2d_backward_kernel[
                    B, Self.C, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                    Self.IN_FLAT, Self.OUT_FLAT, Self.LAYOUT,
                ]
            ](
                grad_output.tt["gpu", B, Self.OUT_FLAT](),
                fin.tt["gpu", B, Self.IN_FLAT](),
                gin.tt["gpu", B, Self.IN_FLAT](),
                grid_dim=n_blocks, block_dim=TPB,
            )

    # for_each_param / zero_grad / polyak_from inherit the Module defaults
    # (param-less → no-op).
