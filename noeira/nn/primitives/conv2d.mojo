"""Conv2D[IC, OC, K, S, P, H, W] — 2D convolution on the storage surface.

The Conv2D de-risk for the storage migration (plan §2/§5: the ONE unproven
kernel). Reduction is identical to legacy `nn.primitives.Conv2D` — im2col +
`max_matmul` GEMM — but on the storage surface (`ref/mut Tensor`, `TensorRefs`,
`lt_gpu`), and the legacy's two-phase vjp split collapses to ONE `vjp` because
the storage surface passes `forward_input` (x) explicitly (invariant §3.1) — no
`_cached_input_ptr`, no param-before-input ordering hazard.

Layouts (flat trait order):
    input    [BATCH, IC·H·W]
    weight   [OC, IC·K·K]            (col index = (ic·K + kh)·K + kw)
    col      [BS, IC·K·K]            BS = BATCH·OH·OW   (im2col output)
    out      [BATCH, OC·OH·OW]

  forward:  col = im2col(x); out_packed[BS,OC] = col @ Wᵀ; scatter + bias.
  vjp:      d_bias = colsum(go); col = im2col(x); dW += goᵀ @ col;
            d_col = go_packed @ W; d_input = col2im(d_col).

CPU uses portable `max_matmul`-into-temp + accumulate (no Apple-cblas beta=1
special-case — correctness first; gated to tolerance vs a direct-conv ref). GPU
re-derives the legacy kernels here so `nn/storage` stays independent of the
legacy package (which gets deleted at the end of the migration).
"""

from noeira.nn.core.mm import mm, bmm
from noeira.nn.core.cublas_gemm import cublas_gemm, cublas_tf32, GEMM_PATH
from noeira.nn.core.cudnn_conv import (
    CONV_PATH,
    cudnn_conv_forward,
    cudnn_conv_backward_data,
    cudnn_conv_backward_filter,
)
from std.sys import CompilationTarget, has_nvidia_gpu_accelerator
from max.gpu import thread_idx, block_idx, block_dim, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from max.algorithm import parallelize
from std.os import getenv
from noeira.nn.core.splitk_gemm import (
    splitk_path_applies,
    decide_partitions,
    dispatch_splitk_gemm,
)
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from noeira.nn.constants import CPU_SIMD_W, DT, TPB, LAYOUT_NCHW, LAYOUT_NHWC
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor
from .linear import _cast_f2b_kernel, BF16, enqueue_bias_grad


comptime CONV_DW_TPB: Int = 128

# A/B knob: block size for the flat thread-per-element im2col/scatter launches.
# These are bandwidth-bound elementwise kernels over BS·COL / B·OUT_FLAT, so 128
# vs 256 only changes occupancy granularity (both are warp multiples) — prediction
# is ~0% on NVIDIA (cf. O1). Bumped to 256 to A/B against Modular's im2col block
# size; revert to TPB (128) if no NVIDIA gain. Apple is parity-only here.
comptime CONV_TPB: Int = 256


# ── Layout-2D index helpers (NCHW default / NHWC) ────────────────────────────
# ONE source of truth for the channels-first vs channels-last index math, shared
# by every CPU + GPU conv kernel. Pure integer arithmetic, @always_inline →
# device-safe. NCHW reproduces the pre-LAYOUT formulas exactly (bit-identical).
#   input  (ic, ih, iw): NCHW ic*H*W + ih*W + iw   | NHWC (ih*W+iw)*IC + ic
#   COL    (ic, kh, kw): NCHW (ic*K+kh)*K + kw     | NHWC (kh*K+kw)*IC + ic
#   output (oc, s)     : NCHW oc*SO + s            | NHWC s*OC + oc
@always_inline
def _in_off[
    LAYOUT: Int, IC: Int, H: Int, W: Int
](ic: Int, ih: Int, iw: Int) -> Int:
    comptime if LAYOUT == LAYOUT_NHWC:
        return (ih * W + iw) * IC + ic
    else:
        return ic * H * W + ih * W + iw


@always_inline
def _in_decode[
    LAYOUT: Int, IC: Int, H: Int, W: Int
](in_pos: Int) -> Tuple[Int, Int, Int]:
    """Inverse of `_in_off`: flat within-sample offset → (ic, ih, iw)."""
    comptime if LAYOUT == LAYOUT_NHWC:
        var ic = in_pos % IC
        var sp = in_pos // IC
        return (ic, sp // W, sp % W)
    else:
        var hw = H * W
        var rem = in_pos % hw
        return (in_pos // hw, rem // W, rem % W)


@always_inline
def _col_off[LAYOUT: Int, IC: Int, K: Int](ic: Int, kh: Int, kw: Int) -> Int:
    comptime if LAYOUT == LAYOUT_NHWC:
        return (kh * K + kw) * IC + ic
    else:
        return (ic * K + kh) * K + kw


@always_inline
def _col_decode[
    LAYOUT: Int, IC: Int, K: Int
](ck: Int) -> Tuple[Int, Int, Int]:
    """Inverse of `_col_off`: COL index → (ic, kh, kw)."""
    comptime if LAYOUT == LAYOUT_NHWC:
        var ic = ck % IC
        var khkw = ck // IC
        return (ic, khkw // K, khkw % K)
    else:
        var ic = ck // (K * K)
        var rem = ck % (K * K)
        return (ic, rem // K, rem % K)


@always_inline
def _out_off[LAYOUT: Int, OC: Int, SO: Int](oc: Int, s: Int) -> Int:
    comptime if LAYOUT == LAYOUT_NHWC:
        return s * OC + oc
    else:
        return oc * SO + s


@always_inline
def _out_decode[
    LAYOUT: Int, OC: Int, SO: Int
](out_pos: Int) -> Tuple[Int, Int]:
    """Inverse of `_out_off`: flat within-sample offset → (oc, s)."""
    comptime if LAYOUT == LAYOUT_NHWC:
        return (out_pos % OC, out_pos // OC)
    else:
        return (out_pos // SO, out_pos % SO)


# ── when is it worth putting im2col on more than one core? ───────────────
# `parallelize` here costs a FIXED ~200 us — measured, not assumed: `l3.down`
# takes 21 us serial and 221 us parallel, and `l4.down` 11 us against 241 us.
# That is enormous next to the small convolutions, so a blanket `parallelize`
# would make five of ResNet18's eleven shapes SLOWER, one of them by 20x.
#
# Serial vs parallel, best of 30, every distinct ResNet18 shape at 240x320:
#
#     shape     N        OH    serial   parallel   speedup
#     conv1     2822400  120   1921 us    341 us     5.63x
#     layer1    2764800   60   2515 us    512 us     4.91x
#     layer2    1382400   30   1243 us    292 us     4.26x
#     l2.0c1     691200   30    627 us    184 us     3.41x
#     layer3     691200   15    621 us    182 us     3.41x
#     l3.0c1     345600   15    312 us    116 us     2.69x
#     layer4     368640    8    359 us    372 us     0.97x   <- 8 tasks
#     l4.0c1     184320    8    177 us    299 us     0.59x   <- 8 tasks
#     l2.down     76800   30     38 us     42 us     0.90x   <- too little work
#     l3.down     38400   15     21 us    221 us     0.10x
#     l4.down     20480    8     11 us    241 us     0.05x
#
# TWO conditions, because neither alone separates the table: `layer4` has as
# much work per task as `layer2` and still loses, on task COUNT — eight tasks
# over eight cores leaves no slack for the slowest one. And `l2.down` has 30
# tasks and still loses, on total WORK.
comptime IM2COL_PAR_MIN_ROWS: Int = 16
"""At least two tasks per core on an 8-core box. Every OH=8 shape lost."""
comptime IM2COL_PAR_MIN_ELEMS: Int = 200_000
"""~300 us of serial work, i.e. enough to pay the ~200 us launch. The measured
boundary sits between `l2.down` (76,800, loses) and `l3.0c1` (345,600, wins
2.7x); 200,000 is the middle of that gap, not a fitted edge."""


@always_inline
def im2col_uses_threads[OH: Int, ELEMS: Int]() -> Bool:
    """The dispatch rule, in ONE place so the gate can report which path a
    shape takes instead of restating the arithmetic and drifting from it."""
    return OH >= IM2COL_PAR_MIN_ROWS and ELEMS >= IM2COL_PAR_MIN_ELEMS


# ── CPU im2col / col2im over List storage (no pointers, no origins) ──────
def _im2col_cpu[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](ref in_list: List[Scalar[DT]], in_off: Int, mut col_list: List[Scalar[DT]]):
    """x[IC·H·W] slab at `in_off` → col_list[OH·OW, COL] row-major. COL axis and
    input slab follow LAYOUT (NCHW default).

    ⚠⚠ **THE LARGEST SINGLE ITEM IN ACT's CPU FORWARD, AND IT COMPUTES
    NOTHING.** Measured IN SITU on an M1 Pro (timers inside this very loop,
    ResNet18 at 240x320, BATCH=1, summed over all 20 convolutions of both
    cameras) against an ~86 ms forward:

        im2col            38.0 ms   44%
        cblas GEMM        12.9 ms   15%   (Accelerate, and it threads itself)
        scatter + bias     3.5 ms    4%
        everything else   ~31 ms    36%   (BN, ReLU, pool, adds, transformer)

    Half of that 38 ms is ONE shape — `IC=64 K=3 S=1` at 60x80, layer1's four
    convolutions across two cameras, 18.5 ms.

    ⚠ THE COST IS NOT A BOUNDS CHECK, which is the intuitive suspicion after
    the stateful-tensor migration replaced pointers with `List`. The NCHW path
    below does **exactly the same `List` indexing** as the loop it replaced —
    same storage, same origins, no pointers — and hoisting the `iw` test out of
    the inner loop plus strength-reducing the addresses is worth **1.95 ms**
    in situ (39.9 -> 38.0). If the access itself carried a runtime check,
    restructuring the loop could not have bought even that.

    ⚠⚠ **AND 1.95 ms IS THE NUMBER, NOT THE 16 ms AN ISOLATED BENCHMARK SAID.**
    Calling this function in a tight loop on its own buffers reported the old
    version at 59.9 ms and the new at 43.8 — a 1.44x that DID NOT SURVIVE
    CONTACT WITH THE REAL FORWARD, where the same two versions are 39.9 and
    38.0. The isolated loop overstated the old cost by 50% and the improvement
    by 8x, and the end-to-end forward showed no change at all, which is what
    sent us back to instrument the real thing.

    The reason is that the two are not bottlenecked on the same resource. Alone
    and warm, this loop is limited by the instructions it issues, so removing a
    branch shows up. Inside the forward it alternates with a multi-threaded
    BLAS GEMM that churns the caches, and it moves ~90 MB per camera at an
    effective ~5 GB/s — nowhere near DRAM bandwidth, so it is bound by gather
    latency and by an inner `kw` run of about three elements, neither of which
    a hoisted branch touches.

    ⚠ SO THE LEVER WAS NOT FEWER INSTRUCTIONS — IT WAS MORE CORES, and that is
    what the threaded path above now does: **38.0 -> 17.0 ms in situ**, taking
    the whole ACT forward from 85.1 to 60.5 ms (-28.9%, minimum of 16
    interleaved runs per arm).

    ⚠ A DIRECT CONVOLUTION WAS TRIED AND LOSES BY 7.6x. On the dominant shape
    it must beat im2col+GEMM+scatter at 2.83 ms; a SIMD direct conv, verified
    against this path to 5.7e-6, took 22.5 ms (7 GMAC/s). The reason is not the
    quality of that kernel: `max_matmul` here runs 176.9 MMAC in 0.354 ms =
    **500 GMAC/s**, which is Accelerate on AMX across dispatch threads, while a
    NEON core peaks near 25.6 GMAC/s. Even a PERFECT single-threaded direct
    convolution would land at ~6.9 ms, still 2.4x slower. Trading a vendor
    kernel at 500 GMAC/s for a hand-written one never pays here, whatever
    memory traffic it saves. Do not re-litigate this without a new AMX story.

    ⚠ AND THE ISOLATED BENCH OVER-PROMISED AGAIN, in the same direction: it
    predicted 39.7 -> 10.9 ms (3.6x) for threading; in situ it is 2.2x. Alone,
    the threads get the whole machine; inside the forward they contend with
    Accelerate's own. Measure IN SITU. Twice now.

    ⚠ NHWC KEEPS THE ORIGINAL LOOP. Its COL axis is `(kh*K + kw)*IC + ic`, so
    the innermost `kw` run is strided by IC rather than contiguous and the same
    rewrite does not apply. It is also on no path measured here.
    """
    comptime CK = IC * K * K
    comptime if LAYOUT != LAYOUT_NCHW:
        for oh in range(OH):
            for ow in range(OW):
                var row_off = (oh * OW + ow) * CK
                for ic in range(IC):
                    for kh in range(K):
                        var ih = oh * S + kh - P
                        for kw in range(K):
                            var iw = ow * S + kw - P
                            var c_idx = row_off + _col_off[LAYOUT, IC, K](
                                ic, kh, kw
                            )
                            if ih < 0 or ih >= H or iw < 0 or iw >= W:
                                col_list[c_idx] = Scalar[DT](0)
                            else:
                                col_list[c_idx] = in_list[
                                    in_off
                                    + _in_off[LAYOUT, IC, H, W](ic, ih, iw)
                                ]
        return

    comptime HW = H * W
    comptime if K == 1:
        # ⚠ A SEPARATE PATH BECAUSE THE GENERAL ONE IS SLOWER HERE. With K == 1
        # there is no `kw` window to hoist anything out of, and computing its
        # bounds costs more than the branch it removes — measured 0.69-0.80x,
        # i.e. a REGRESSION, on the three 1x1 downsample convolutions. They are
        # only 0.13 ms of the forward, but a rewrite that makes a shape slower
        # should handle it rather than average it away.
        for oh in range(OH):
            var ih = oh * S - P
            var row0 = oh * OW * IC
            var inside_h = ih >= 0 and ih < H
            for ow in range(OW):
                var iw = ow * S - P
                var row_off = row0 + ow * IC
                if not inside_h or iw < 0 or iw >= W:
                    for ic in range(IC):
                        col_list[row_off + ic] = Scalar[DT](0)
                    continue
                var x_at = in_off + ih * W + iw
                for ic in range(IC):
                    col_list[row_off + ic] = in_list[x_at + ic * HW]
        return

    # ⚠ RAW POINTERS, NOT THE `List` REFS, and only for the threaded body.
    # `parallelize`'s closure cannot infer a capture convention for a `ref` /
    # `mut List` parameter, and this repo has a recorded case of a cross-thread
    # write through an owned `List` being FOLDED AWAY by the compiler
    # (`_a_mojo_owned_slab_folds_a_cross_thread_write_away`). Writing through
    # pointers obtained here, with each `oh` owning a disjoint span of `col`,
    # is the shape that was verified: 2,764,800 elements identical to the
    # serial answer, and the gate re-checks it on every run.
    var _xp = in_list.unsafe_ptr()
    var _cp = col_list.unsafe_ptr()

    def _row(oh: Int) {imm}:
        var row0 = oh * OW * CK
        for ow in range(OW):
            var row_off = row0 + ow * CK
            var iw0 = ow * S - P
            var kw_lo = 0 if iw0 >= 0 else -iw0
            if kw_lo > K:
                kw_lo = K
            var kw_hi = K if iw0 + K <= W else W - iw0
            if kw_hi > K:
                kw_hi = K
            if kw_hi < kw_lo:
                kw_hi = kw_lo
            for ic in range(IC):
                var c_ic = row_off + ic * K * K
                var x_ic = in_off + ic * HW
                for kh in range(K):
                    var ih = oh * S + kh - P
                    var c_base = c_ic + kh * K
                    if ih < 0 or ih >= H:
                        for kw in range(K):
                            _cp[unsafe_offset=c_base + kw] = Scalar[DT](0)
                        continue
                    var x_row = x_ic + ih * W + iw0
                    for kw in range(kw_lo):
                        _cp[unsafe_offset=c_base + kw] = Scalar[DT](0)
                    for kw in range(kw_lo, kw_hi):
                        _cp[unsafe_offset=c_base + kw] = _xp[
                            unsafe_offset = x_row + kw
                        ]
                    for kw in range(kw_hi, K):
                        _cp[unsafe_offset=c_base + kw] = Scalar[DT](0)

    comptime if im2col_uses_threads[OH, OH * OW * CK]():
        parallelize(_row, OH)
        return

    for oh in range(OH):
        var row0 = oh * OW * CK
        for ow in range(OW):
            var row_off = row0 + ow * CK
            var iw0 = ow * S - P
            # The `kw` values for which `iw0 + kw` lands inside [0, W).
            # ⚠ BOTH ENDS CLAMPED INTO [0, K]. A window falling ENTIRELY
            # outside the input — possible whenever P >= K, e.g. K=1 P=2 —
            # gives `kw_lo > K`, and an unclamped `range(kw_lo)` zero-fill
            # would run off the end of this row into the next one. The
            # prototype of this rewrite passed all 11 ResNet18 shapes WITHOUT
            # this clamp: the shapes a real model uses are not an adversarial
            # test of a bounds computation, which is why the gate carries
            # shapes no ResNet has.
            var kw_lo = 0 if iw0 >= 0 else -iw0
            if kw_lo > K:
                kw_lo = K
            var kw_hi = K if iw0 + K <= W else W - iw0
            if kw_hi > K:
                kw_hi = K
            if kw_hi < kw_lo:
                kw_hi = kw_lo
            for ic in range(IC):
                var c_ic = row_off + ic * K * K
                var x_ic = in_off + ic * HW
                for kh in range(K):
                    var ih = oh * S + kh - P
                    var c_base = c_ic + kh * K
                    if ih < 0 or ih >= H:
                        for kw in range(K):
                            col_list[c_base + kw] = Scalar[DT](0)
                        continue
                    # `x_row + kw` over [kw_lo, kw_hi) is a CONTIGUOUS run of
                    # the input row written to a contiguous run of the col row:
                    # no per-element index arithmetic and no branch.
                    var x_row = x_ic + ih * W + iw0
                    for kw in range(kw_lo):
                        col_list[c_base + kw] = Scalar[DT](0)
                    for kw in range(kw_lo, kw_hi):
                        col_list[c_base + kw] = in_list[x_row + kw]
                    for kw in range(kw_hi, K):
                        col_list[c_base + kw] = Scalar[DT](0)


def _col2im_cpu[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    ref d_col_list: List[Scalar[DT]],
    mut d_in_list: List[Scalar[DT]],
    in_off: Int,
):
    """Scatter-add d_col[OH·OW, COL] back into d_in_list[IC·H·W] at `in_off`
    (must be pre-zeroed). COL axis and input slab follow LAYOUT (NCHW default)."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                for kh in range(K):
                    var ih = oh * S + kh - P
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if iw < 0 or iw >= W:
                            continue
                        d_in_list[
                            in_off + _in_off[LAYOUT, IC, H, W](ic, ih, iw)
                        ] += d_col_list[
                            row_off + _col_off[LAYOUT, IC, K](ic, kh, kw)
                        ]


# ── GPU kernels (re-derived; args MutAnyOrigin = the GPU ABI boundary) ──
# The GEMM-flanking kernels are dtype-parametric on the ACTIVATION dtype (`ADT`):
# the fp32 path calls them with DT, the bf16-flow path with bfloat16 (im2col,
# scatter, col2im, transpose, pack are pure gather/copy — dtype-transparent).
def _im2col_kernel[
    BATCH: Int,
    IC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    OH: Int,
    OW: Int,
    IN_FLAT: Int,
    COL: Int,
    DCOL: Int,
    SO: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    col: LayoutTensor[ADT, Layout.row_major(BS, DCOL), MutAnyOrigin],
):
    """im2col into a row stride of `DCOL >= COL`, zero-filling `[COL, DCOL)`.

    `DCOL` is the K-alignment pad (`Conv2D.CPAD`); `DCOL == COL` is the
    unpadded case and the kernel is then exactly what it was. The zeros are
    what make the pad free: the forward GEMM contracts over `DCOL`, and a
    zero column contributes exactly 0 to every output — the same argument that
    made `Linear`'s K pad bit-identical."""
    var idx = Int(global_idx.x)
    if idx >= BS * DCOL:
        return
    var row = idx // DCOL
    var ck = idx % DCOL
    if ck >= COL:
        col[row, ck] = Scalar[ADT](0)
        return
    var b = row // SO
    var s = row % SO
    var oh = s // OW
    var ow = s % OW
    var ic, kh, kw = _col_decode[LAYOUT, IC, K](ck)
    var ih = oh * S + kh - P
    var iw = ow * S + kw - P
    if ih < 0 or ih >= H or iw < 0 or iw >= W:
        col[row, ck] = Scalar[ADT](0)
    else:
        col[row, ck] = rebind[Scalar[ADT]](
            input[b, _in_off[LAYOUT, IC, H, W](ic, ih, iw)]
        )


def _pad_w_cols_kernel[
    OC: Int, COL: Int, DCOL: Int, OCP: Int = OC
](
    src: LayoutTensor[DT, Layout.row_major(OC * COL), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(OCP * DCOL), MutAnyOrigin],
):
    """`W[OC, COL]` -> `W_pad[OCP, DCOL]`, zeros outside the source rectangle.

    Both axes of the forward GEMM: `DCOL` is the K pad (the contraction) and
    `OCP` is the N pad (the output width). Row-major with a WIDER row stride,
    so this is a 2-D copy, not a flat tail append — every row after the first
    moves. Twin of `linear.mojo`'s `_pad_2d_kernel`.

    The zero ROWS matter as much as the zero columns: `out_packed[:, oc]` for
    `oc >= OC` is computed against them and comes out 0, and the scatter never
    reads it."""
    var i = Int(global_idx.x)
    if i >= OCP * DCOL:
        return
    var r = i // DCOL
    var c = i % DCOL
    dst[i] = src[r * COL + c] if (r < OC and c < COL) else Scalar[DT](0)


def _accum_w_2d_kernel[
    OC: Int, COL: Int, DCOL: Int
](
    dst: LayoutTensor[DT, Layout.row_major(OC * COL), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(OC * DCOL), MutAnyOrigin],
):
    """`grad_w[oc, c] += dW_pad[oc, c]` where `src` has the WIDER row stride.

    ⚠ The flat `_accum_kernel` is WRONG the moment the dW GEMM writes a padded
    `[OC, DCOL]`: the strides differ, so a flat add folds each row's padding
    into the next row's leading weights — a wrong gradient that still trains.
    Same trap `linear.mojo::_accum_2d_kernel` documents."""
    var i = Int(global_idx.x)
    if i >= OC * COL:
        return
    var r = i // COL
    var c = i % COL
    dst[i] = rebind[Scalar[DT]](dst[i]) + rebind[Scalar[DT]](
        src[r * DCOL + c]
    )


def _scatter_bias_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
    OCP: Int = OC,
](
    out_packed: LayoutTensor[ADT, Layout.row_major(BS, OCP), MutAnyOrigin],
    bias: LayoutTensor[ADT, Layout.row_major(OC), MutAnyOrigin],
    output: LayoutTensor[ADT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
):
    """Scatter + bias, reading `out_packed` at a row stride of `OCP >= OC`.

    `OCP` is the N-alignment pad. The slice back to `OC` rides here and costs
    NO extra launch — channels `[OC, OCP)` are simply never read — exactly the
    way `linear.mojo`'s `_bias_add_slice_kernel` rides on its bias add.
    `OCP == OC` is the unpadded case and this is then the original kernel."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var oc, s = _out_decode[LAYOUT, OC, SO](out_pos)
    output[b, out_pos] = rebind[Scalar[ADT]](
        out_packed[b * SO + s, oc]
    ) + rebind[Scalar[ADT]](bias[oc])



def _fwd_oc1_matvec_kernel[
    BATCH: Int,
    SO: Int,
    COL: Int,
    DCOL: Int,
    BS: Int,
    ADT: DType = DT,
](
    col: LayoutTensor[ADT, Layout.row_major(BS, DCOL), MutAnyOrigin],
    weight: LayoutTensor[ADT, Layout.row_major(COL), MutAnyOrigin],
    bias: LayoutTensor[ADT, Layout.row_major(1), MutAnyOrigin],
    output: LayoutTensor[ADT, Layout.row_major(BATCH, SO), MutAnyOrigin],
):
    """OC==1 forward: `max_matmul[transpose_b=True]` on GPU SILENTLY
    MISCOMPUTES N=1 GEMMs (out[BS,1] = col @ Wᵀ; verified wrong on Metal,
    N>=2 exact — sibling of the documented N=1 abort in the CPU vjp path).
    Compute the matvec + bias directly instead; with OC=1, OUT_FLAT == SO
    and the LAYOUT scatter is the identity, so this fuses GEMM + scatter.
    Accumulates in fp32 (matches the GEMM's accumulation for bf16)."""
    var idx = Int(global_idx.x)
    if idx >= BS:
        return
    var acc: Scalar[DT] = 0.0
    for k in range(COL):
        acc += (
            rebind[Scalar[ADT]](col[idx, k]).cast[DT]()
            * rebind[Scalar[ADT]](weight[k]).cast[DT]()
        )
    acc += rebind[Scalar[ADT]](bias[0]).cast[DT]()
    output[idx // SO, idx % SO] = acc.cast[ADT]()


def _go_transpose_kernel[
    BATCH: Int,
    OC: Int,
    SO: Int,
    OUT_FLAT: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: LayoutTensor[
        ADT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    go_T: LayoutTensor[ADT, Layout.row_major(OC, BS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= OC * BS:
        return
    var oc = idx // BS
    var col = idx % BS
    var b = col // SO
    var s = col % SO
    go_T[oc, col] = rebind[Scalar[ADT]](
        grad_output[b, _out_off[LAYOUT, OC, SO](oc, s)]
    )


def _accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


def _wT_transpose_kernel[
    OC: Int, COL: Int, ADT: DType = DT
](
    w: LayoutTensor[ADT, Layout.row_major(OC, COL), MutAnyOrigin],
    wT: LayoutTensor[ADT, Layout.row_major(COL, OC), MutAnyOrigin],
):
    """O2: transpose weight `W[OC, COL]` → `Wᵀ[COL, OC]` so the input-grad GEMM
    can compute `d_colᵀ[COL, BS] = Wᵀ[COL, OC] @ goᵀ[OC, BS]` (reusing the `goᵀ`
    already built for dW — no `_go_pack` kernel, no `[BS,COL]` d_col). `W` is
    small (`OC·COL`), so the transpose is cheap relative to the col2im it
    coalesces. `max_matmul` rejects `transpose_a`, hence the explicit transpose."""
    var idx = Int(global_idx.x)
    if idx >= OC * COL:
        return
    var oc = idx // COL
    var c = idx % COL
    wT[c, oc] = rebind[Scalar[ADT]](w[oc, c])


def _dx_col2im_kernel[
    BATCH: Int,
    IC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    OH: Int,
    OW: Int,
    IN_FLAT: Int,
    COL: Int,
    SO: Int,
    BS: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    d_col: LayoutTensor[ADT, Layout.row_major(COL, BS), MutAnyOrigin],
    grad_input: LayoutTensor[
        ADT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin
    ],
):
    # O2: `d_col` is the TRANSPOSED layout `[COL, BS]` (vs the natural `[BS,COL]`
    # GEMM output). Adjacent threads differ in `iw` → `row` (col2im is a gather:
    # one input element ← up to K² d_col entries, each read exactly once, so no
    # cross-thread reuse and shared-mem tiling buys nothing — coalescing is the
    # only lever). In `[COL, BS]` the per-(kh,kw) read `d_col[col_idx, row]` has
    # adjacent threads at adjacent `row` → CONTIGUOUS, coalesced. Measured 1.59×
    # on the S=1 hot shape (NVIDIA); strided/`[BS,COL]` was stride-COL scattered.
    # O3: the `S==1` branch (the whole MuZero/EZv2 residual tower) drops the
    # `% S` / `// S` integer ops and a divergence source.
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * IN_FLAT:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var ic, ih, iw = _in_decode[LAYOUT, IC, H, W](in_pos)
    # fp32 accumulator (the col2im sum) even on the bf16-flow path; only the
    # written grad_input is bf16 (activation dtype).
    var acc: Scalar[DT] = 0
    comptime if S == 1:
        for kh in range(K):
            var oh = ih + P - kh
            if oh < 0 or oh >= OH:
                continue
            for kw in range(K):
                var ow = iw + P - kw
                if ow < 0 or ow >= OW:
                    continue
                var row = b * SO + oh * OW + ow
                var col_idx = _col_off[LAYOUT, IC, K](ic, kh, kw)
                acc += rebind[Scalar[ADT]](d_col[col_idx, row]).cast[DT]()
    else:
        for kh in range(K):
            var oh_num = ih + P - kh
            if oh_num < 0 or oh_num % S != 0:
                continue
            var oh = oh_num // S
            if oh >= OH:
                continue
            for kw in range(K):
                var ow_num = iw + P - kw
                if ow_num < 0 or ow_num % S != 0:
                    continue
                var ow = ow_num // S
                if ow >= OW:
                    continue
                var row = b * SO + oh * OW + ow
                var col_idx = _col_off[LAYOUT, IC, K](ic, kh, kw)
                acc += rebind[Scalar[ADT]](d_col[col_idx, row]).cast[DT]()
    grad_input[b, in_pos] = acc.cast[ADT]()


def _backward_db_kernel[
    BATCH: Int,
    OC: Int,
    OH: Int,
    OW: Int,
    OUT_FLAT: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: LayoutTensor[
        ADT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin
    ],
    grad_bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
):
    # grad_output is the ACTIVATION dtype (`ADT`); grad_bias is the FP32 master
    # grad. Each element casts to DT before the block-sum (fp32 accumulator).
    var oc = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if oc >= OC:
        return
    var so = OH * OW
    var n_eff = BATCH * so
    var my_acc: Scalar[DT] = 0
    var idx = t
    while idx < n_eff:
        var b = idx // so
        var s_pos = idx % so
        my_acc += rebind[Scalar[ADT]](
            grad_output[b, _out_off[LAYOUT, OC, OH * OW](oc, s_pos)]
        ).cast[DT]()
        idx += CONV_DW_TPB
    var total = block.sum[block_size=CONV_DW_TPB, broadcast=False](val=my_acc)
    if t == 0:
        grad_bias[oc] = rebind[Scalar[DT]](grad_bias[oc]) + total[0]


def _rowsum_kernel[ROWS: Int, COLS: Int](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(ROWS), MutAnyOrigin],
):
    """`dst[r] = Σ_c src[r, c]`, one block per row (coalesced along c).
    Stage 1 of the NCHW bias gradient: rows are (sample, channel), columns
    the contiguous OH·OW positions."""
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0
    var c = t
    while c < COLS:
        acc += rebind[Scalar[DT]](src[r, c])
        c += CONV_DW_TPB
    var total = block.sum[block_size=CONV_DW_TPB, broadcast=False](val=acc)
    if t == 0:
        dst[r] = total[0]


def _bias_add_out_kernel[
    BATCH: Int, OC: Int, SO: Int, OUT_FLAT: Int, LAYOUT: Int = LAYOUT_NCHW
](
    dst: LayoutTensor[DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
):
    """`dst[b, oc, s] += bias[oc]` in place, in either layout — the bias of
    the cuDNN forward, which writes the convolution alone."""
    var idx = Int(global_idx.x)
    if idx < BATCH * OUT_FLAT:
        var b = idx // OUT_FLAT
        var r = idx % OUT_FLAT
        var oc: Int
        comptime if LAYOUT == LAYOUT_NHWC:
            oc = r % OC
        else:
            oc = r // SO
        dst[b, r] = rebind[Scalar[DT]](dst[b, r]) + rebind[Scalar[DT]](bias[oc])


# ── Conv2D ────────────────────────────────────────────────────────────────

# NOTE: the conv dW's tile-config dispatch lives in
# `nn/core/splitk_gemm.mojo::dispatch_splitk_gemm`, shared with `Linear` and
# the other dW sites. The conv dW is `[OC, BS] @ [BS, CPAD]`, so the
# contraction is `batch * OH * OW` — far longer than the transformer dW's
# `batch * tokens`, and the M and N are an out-channel count and an im2col
# column count, both small. That makes the tile grid tiny (a ResNet18 stem at
# OC=64, CPAD=256 is TWO tiles) while K runs into the hundreds of thousands,
# which is the regime split-K exists for and the regime `select_config`'s
# partition cap handles worst.


struct Conv2D[
    IC_: Int,
    OC_: Int,
    K_: Int,
    S_: Int,
    P_: Int,
    H_: Int,
    W_: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](Module):
    comptime ARITY = 1
    comptime OH = (Self.H_ + 2 * Self.P_ - Self.K_) // Self.S_ + 1
    comptime OW = (Self.W_ + 2 * Self.P_ - Self.K_) // Self.S_ + 1
    comptime IN_FLAT = Self.IC_ * Self.H_ * Self.W_
    comptime OUT_FLAT = Self.OC_ * Self.OH * Self.OW
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_FLAT)
    comptime OUT_DIM = Self.OUT_FLAT
    comptime W_SIZE = Self.OC_ * Self.IC_ * Self.K_ * Self.K_
    comptime B_SIZE = Self.OC_
    comptime COL = Self.IC_ * Self.K_ * Self.K_
    comptime SO = Self.OH * Self.OW

    # ── K-alignment padding (GPU fp32) — the SAME defect `Linear` documents ──
    # `max_matmul` falls off a ~10x cliff when the CONTRACTION dim is
    # misaligned, on BOTH backends (Metal wants K%16, an RTX 5090 wants K%32,
    # so 32 satisfies both); and a narrow, unaligned OUTPUT width makes cuBLAS
    # pick a split-K path whose workspace is allocated, memset and freed on
    # EVERY call. Read the two comment blocks at `Linear.PAD_TO` /
    # `Linear.NEEDS_N_PAD` before changing anything here — `PAD_TO` is
    # calibrated by `benchmarks/bench_matmul_k_alignment.mojo`.
    #
    # This conv is im2col + GEMM, and `COL = IC*K*K` lands on BOTH bad axes:
    #
    #     forward   out[BS, OC]     = col[BS, COL] @ W[OC, COL]ᵀ    K = COL
    #     vjp dW    dW[OC, COL]     = goᵀ[OC, BS] @ col[BS, COL]    N = COL
    #     vjp dx    d_colᵀ[COL, BS] = Wᵀ[COL, OC] @ goᵀ[OC, BS]     (K = OC)
    #
    # so ONE pad on COL fixes the forward's contraction and the dW GEMM's
    # output width at once. `docs/GPU_STEP_PERF.md` ranks conv2d as suspect #1
    # for the allocation churn precisely because it had no padding at all.
    #
    # ⚠ Which convs this actually moves: 3x3 stages give COL = 576 / 1152 /
    # 2304 / 4608 and 1x1 downsamples give COL = IC — all already multiples of
    # 32, so a ResNet18 has exactly ONE offender, the 7x7x3 stem at COL = 147
    # (padded to 160). That stem is also the largest GEMM in the backbone
    # (BS = 307,200 at 240x320 batch 16), so it is worth its own pad. Elsewhere
    # in the repo COL = 25 (MNIST 5x5x1), 72 and 400 are the misaligned ones.
    #
    # ⚠ NOT the N=OC axis. `OC` is a multiple of 32 in every net here, and
    # padding it produces a WIDER out_packed that `_scatter_bias_kernel` would
    # have to slice — the asymmetry `Linear.NEEDS_N_PAD` spells out. Left
    # undone deliberately, not overlooked.
    #
    # ⚠ fp32 GPU only, matching `Linear`: the bf16-flow path keeps `COL`.
    # ⚠ 128, not 32, and the reason is the BACKWARD. `CPAD` is the im2col row
    # stride, so it is the forward GEMM's K *and* the dW GEMM's N:
    #
    #     forward   out[BS, OCPAD] = col[BS, CPAD] @ w[CPAD, OCPAD]
    #     backward  dW[OC, CPAD]   = goT[OC, BS]   @ col[BS, CPAD]
    #
    # The forward only needs `k % 32 == 0 and k >= 128`, which 32 satisfied
    # minimally. The dW needs `n % 128 == 0` or `multi_gemm_cond` fails and the
    # whole GEMM goes to the VENDOR fallback -- which allocates and memsets
    # 32 MB PER CALL (vendor/blas.mojo:780) and is therefore uncapturable, as
    # well as slow.
    #
    # Measured on a 5090 (benchmarks/bench_splitk_act_dw_sweep.mojo, `sweep_pad`),
    # net of the extra forward FLOPs the wider stride costs:
    #
    #   layer1 3x3  COL 576 -> 640 (+11% fwd)   dW -157.6us  fwd +13.0us
    #                                           NET -144.6us/call, x4 = +0.58 ms/step
    #   stem 7x7    COL 147 -> 256 (+60% fwd)   dW  -64.8us  fwd +69.1us
    #                                           NET   +4.3us/call, x2 = -0.009 ms/step
    #
    # So the stem is a wash on throughput and everything else is a win. It is
    # padded anyway, because at -0.009 ms/step (0.02% of the step) it buys the
    # removal of the last non-D2H CUDA-graph blocker in the model. Judge this
    # constant on capture first and throughput second.
    #
    # ⚠ AND JUDGE IT ON CAPTURE ALONE, BECAUSE THE THROUGHPUT CASE DID NOT
    # SURVIVE THE STEP. The per-call numbers above predicted +0.57 ms/step.
    # Measured end to end on the ACT step: 39.968 -> 39.909 ms, i.e. **0.06 ms**,
    # a tenth of the prediction. Isolated GEMM sweeps launch back-to-back with
    # one sync and so measure throughput under saturation; the real step already
    # runs the GPU ~88% busy, and removing GEMM time there does not compose 1:1.
    # Treat `sweep_pad`-style per-call deltas as an UPPER BOUND on what a step
    # will show, not an estimate of it.
    #
    # ⚠ `NOEIRA_SPLITK=0` no longer gives the pre-change baseline. This
    # constant is comptime, so the env var disables split-K while LEAVING THE
    # PADDING ON -- which is exactly the arm that measured 0.21x. The OFF arm
    # now reads ~50.5 ms against a true pre-change baseline of 44.0 ms. It is
    # still the right A/B for "is split-K helping", but it is NOT "before".
    #
    # ⚠ The dW win only exists WITH a tuned partition count. At MAX's own P=8
    # the padded GEMM is 0.21x -- a 5x REGRESSION versus the vendor path. The
    # pad and `splitk_gemm` are one change, not two.
    comptime KEEP_MAX = (
        CONV_PATH == "auto"
        and Self.IC_ == 1
        and Self.OC_ <= 16
        and Self.COL >= 25
        and Self.COL <= 64
    )
    """The one regime where neither cuBLAS nor cuDNN beat MAX's padded GEMMs
    (`benchmarks/bench_conv2d_paths_gpu.mojo`, RTX 5090): a single-input-
    channel conv with a 25-64 long contraction and <= 16 outputs (MNIST conv1:
    58 us vs 99 cuDNN, 116 im2col). A property of the layer, not of the
    batch, because it decides the padding below."""
    comptime CUB = has_nvidia_gpu_accelerator() and GEMM_PATH != "max" and GEMM_PATH != "convmax" and not Self.KEEP_MAX
    """NVIDIA fp32: the three GEMMs (forward, dW, d_col) run on `cublas_gemm`,
    UNPADDED: cuBLAS takes any shape without the per-call workspace that MAX's
    fallback allocates, so `CPAD == COL` and `OCPAD == OC_` below and none of
    the padding exists. MAX's padded GEMMs stay for Apple and for
    `-D NN_GEMM_PATH=max` (A/B runs)."""
    @staticmethod
    def use_cudnn[B: Int]() -> Bool:
        """NVIDIA fp32: run this layer's forward and backward through cuDNN
        (`cudnn_conv`) instead of im2col + GEMM. `NN_CONV_PATH` selects;
        `auto` is the per-shape rule below."""
        comptime if not has_nvidia_gpu_accelerator() or Self.ADT != DT or GEMM_PATH == "max" or GEMM_PATH == "convmax":
            return False
        comptime if CONV_PATH == "cudnn":
            return True
        comptime if CONV_PATH == "im2col":
            return False
        # `auto`: measured per shape on an RTX 5090 (80 shapes from every
        # agent and example; cuDNN restricted to its deterministic GEMM
        # algorithms, `cudnn_conv`). cuDNN's implicit GEMM wins where the
        # im2col matrix is large: stride-1 3x3 / 5x5 convs with >= 12 input
        # channels over >= 30k output positions, or heavy ones (contraction
        # >= 2304 at >= 9k positions); NHWC (cuDNN's layout); OC == 1 (the
        # im2col path's matvec: 4.4 ms vs 1.1 ms on DreamerV3's decoder
        # output); 16-channel 1x1 heads. Stems, stride-2 layers and small
        # spatial sizes stay on im2col + cuBLAS. With this rule no measured
        # shape is slower than MAX's padded path (median 1.48x faster).
        comptime BS = B * Self.SO
        comptime if Self.OC_ == 1:
            return True
        comptime if Self.KEEP_MAX:
            return False
        comptime if Self.K_ == 1 and Self.OC_ <= 16 and Self.IC_ >= 32:
            return True
        comptime if Self.LAYOUT == LAYOUT_NHWC and Self.IC_ >= 12:
            return True
        return (
            Self.S_ == 1 and Self.K_ >= 3 and Self.IC_ >= 12
            and (BS >= 30_000 or (BS >= 9_000 and Self.COL >= 2304))
        )

    comptime PAD_TO = 128
    comptime K_MIN = 128
    comptime CPAD = Self.COL if Self.CUB else (Self._round_up(Self.COL, Self.PAD_TO) if Self._round_up(
        Self.COL, Self.PAD_TO
    ) > Self.K_MIN else Self.K_MIN)
    comptime NEEDS_COL_PAD = Self.CPAD != Self.COL
    """The im2col row stride the fp32 GPU path uses — `COL` rounded up to
    `PAD_TO`, and exactly `COL` when it is already aligned (then every kernel
    below is byte-for-byte the unpadded one)."""
    # ── N-alignment padding on OC — the axis this block used to skip ─────
    # The forward GEMM's N is `OC_`, and the gate wants `n % 128 == 0`. The
    # note below said OC is "always a multiple of 32" and left it — true, and
    # the wrong test: 32 and 64 are multiples of 32 and both land on cuBLAS.
    # Measured on the ResNet18 stem, `[307200 x 160] @ [160 x N]`:
    #
    #     OC =  64   433.37 us   cutlass      ALLOCATES
    #     OC = 128   280.37 us   multistage   free
    #
    # 2x the FLOPs and 1.55x FASTER. The slice back to `OC_` rides on
    # `_scatter_bias_kernel` (channels `[OC_, OCPAD)` are never read), so it
    # costs no extra launch — the same trick `Linear`'s bias add uses.
    #
    # ⚠ OC_ == 1 is EXCLUDED: that path is `_fwd_oc1_matvec_kernel`, a direct
    # matvec that never calls `max_matmul`, so there is no dispatch gate to
    # satisfy and padding would only waste a buffer.
    #
    # ⚠ Only the FORWARD is fixed. `vjp`'s dW GEMM has N = `CPAD` and its
    # d_col GEMM has N = `BS` = BATCH*OH*OW — a batch-times-spatial size that
    # is not ours to pad. Those stay on the vendor path.
    comptime N_PAD_TO = 128
    comptime OCPAD = Self.OC_ if Self.CUB else (Self._round_up(
        Self.OC_, Self.N_PAD_TO
    ) if Self.OC_ != 1 else 1)
    comptime NEEDS_OC_PAD = Self.OCPAD != Self.OC_
    comptime NEEDS_W_PAD = Self.NEEDS_COL_PAD or Self.NEEDS_OC_PAD
    comptime WPAD_SIZE = Self.OCPAD * Self.CPAD
    """Size of the padded FORWARD weight `[OCPAD, CPAD]`."""
    comptime DWPAD_SIZE = Self.OC_ * Self.CPAD
    """Size of the `vjp` dW temp `[OC_, CPAD]` — a DIFFERENT rectangle.

    ⚠ These were one constant until `OCPAD` split them. The forward GEMM pads
    N (= `OC_`); the dW GEMM's N is `CPAD` and its M is the true `OC_`, so it
    must NOT be widened. Sharing the name silently oversized `dW_tmp` and fed
    `_accum_w_2d_kernel` a view of the wrong extent — the compiler caught it
    here, but the same shape of mistake with matching sizes would have been a
    wrong gradient that still trains."""

    @staticmethod
    def _round_up(v: Int, to: Int) -> Int:
        return ((v + to - 1) // to) * to
    # Activation-flow dtype (satisfies the Module trait). `Conv2D[...]` = fp32
    # (ACT_DT == DT, the legacy NoAMP path); `Conv2D[..., DType.bfloat16]` flows
    # activations at bf16 (the AMP "Step B" memory win). Master weights/grads/
    # bias STAY fp32 (`Param` is always `DT`); only the CACHED bf16 weight copy
    # (`w_bf`, version-gated) and bf16 bias (`b_a`) are low-precision.
    comptime ACT_DT = Self.ADT

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # GPU im2col + GEMM scratch (lazy, reused — capture-safe). fp32-path scratch
    # (`dW_tmp` is ALSO used by the bf16 path: the dW GEMM writes a FP32 output).
    var col_t: Tensor  # [BS, COL] (im2col col) reused as [COL, BS] d_colᵀ (O2)
    var outp_t: Tensor  # [BS, OC]   (out_packed; fwd only)
    var goT_t: Tensor  # [OC, BS]   (goᵀ — for dW AND d_colᵀ, O2)
    var dW_tmp: Tensor  # [OC, COL]  (fp32 dW temp; bf16-in → fp32-out GEMM)
    var sk_ws_fwd: Tensor
    """Split-K reduction workspace for the FORWARD GEMM, `[P, BS, OCPAD]`.

    ⚠ The forward splits for the SAME reason the dW does, and we missed it the
    first time round. Forward is `out[BS, OCPAD] = col[BS, CPAD] @ wᵀ`, so its
    contraction is `CPAD = IC * K * K` — 2304 for a 3x3 at IC=256, 4608 at
    IC=512. `select_config` partitions any K >= 2048, so ResNet18's layer3 and
    layer4 forwards all take MAX's split-K path and allocate `P * BS * OCPAD *
    4` bytes per call. That is what aborted the ACT CUDA-graph capture at
    18.75MB = 2 * 9600 * 256 * 4, on `MxNxK: 9600x256x2304`.

    Sized on the first eager forward, per `_decide_sk_p_fwd`. It is bigger than
    the dW workspace — M is BS, not OC — so it is only allocated when the
    forward actually splits."""
    var _sk_p_fwd: Int
    """Cached partition count for the FORWARD GEMM. -1 = undecided, 1 = do not
    split."""
    var sk_ws: Tensor
    """Split-K reduction workspace for the dW GEMM, `[P, OC_, CPAD]`.

    Same rationale as `Linear.sk_ws`: `linalg.matmul` allocates this per call
    and frees it again, which costs a cuMemAlloc/cuMemFree pair per GEMM and —
    being a SYNCHRONOUS driver allocation — makes the step uncapturable. It
    lives on the Module because a CUDA graph holds RAW POINTERS to every
    operand, so the workspace must outlive the LAST REPLAY. Sized by
    `ensure_gpu` on an eager step; never grows inside a capture region."""
    var _sk_p: Int
    """Cached split-K partition count for the dW GEMM. -1 = undecided, 1 = do
    not split (also what `NOEIRA_SPLITK=0` forces).

    Decided once: P sets `grid_dim`, which is baked into a captured graph."""
    var wT_t: Tensor  # [COL, OC]  (O2: fp32 Wᵀ for the d_colᵀ GEMM)
    # K-alignment scratch (lazy; fp32 GPU only, and only when `NEEDS_COL_PAD`).
    # `w_pad` is the zero-tailed [OC, CPAD] weight, re-padded only when the
    # optimizer bumped `weight.val.version` — once per optimizer step, not per
    # forward, exactly like `w_bf`. The padded `col` needs no separate buffer:
    # `col_t` is simply allocated at the wider stride and im2col zero-fills the
    # tail. `dW_tmp` grows to [OC, CPAD] and is accumulated into the [OC, COL]
    # master grad with a STRIDED add.
    var w_pad: Tensor
    var _w_pad_version: Int
    # bf16-flow compute scratch (lazy; used only when ACT_DT == bf16 and
    # target == "gpu"). `col_t_bf`/`outp_t_bf`/`goT_t_bf` are the bf16 activation
    # scratch (im2col col, out_packed/go_packed, goᵀ). `w_bf` is the CACHED bf16
    # weight: recast from `weight.val` only when the optimizer bumped its version
    # since the last cast (`_w_cast_version`), so the W cast happens ONCE per
    # optimizer step, not per forward. `b_a` is the cached bf16 bias.
    var col_t_bf: TensorImpl[Self.ADT]
    var outp_t_bf: TensorImpl[Self.ADT]
    var goT_t_bf: TensorImpl[Self.ADT]
    var w_bf: TensorImpl[Self.ADT]
    var b_a: TensorImpl[Self.ADT]
    var wT_bf: TensorImpl[Self.ADT]  # [COL, OC] (O2: bf16 Wᵀ for d_colᵀ GEMM)
    var _w_cast_version: Int  # `weight.val.version` at last bf16 weight cast
    # Capture mode (set via `set_attr["capture_recast"]`): when True, the bf16
    # weight recast is UNCONDITIONAL so the cast kernel is always recorded into a
    # CUDA graph and reads the live fp32 master on every replay — the version
    # gate would skip it on replay and serve STALE weights. Off → version-gated.
    var _force_recast: Bool
    # A2: device address of the input the forward last im2col'd into
    # `col_t`/`col_t_bf`. The vjp REUSES that col (skips its redundant im2col)
    # iff its `forward_input` is the SAME buffer; else it recomputes (safe
    # fallback = today's behavior, never wrong). Valid under the framework's
    # re-forward-before-vjp cache contract — the same one `BatchNorm2D.vjp`
    # already relies on for `cache_xhat` (conv sits next to BN in every block,
    # so wherever BN's single-field cache is valid at vjp, `col_t` is too).
    # 0 = no forward has populated the col yet.
    var _col_src_ptr: Int
    var db_rows: Tensor
    """[B·OC] per-(sample, channel) sums: stage 1 of the NCHW bias grad."""
    var db_part: Tensor
    """Chunk partials of `enqueue_bias_grad` (stage 2)."""

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.col_t = Tensor()
        self.outp_t = Tensor()
        self.goT_t = Tensor()
        self.dW_tmp = Tensor()
        self.sk_ws_fwd = Tensor()
        self._sk_p_fwd = -1
        self.sk_ws = Tensor()
        self._sk_p = -1
        self.wT_t = Tensor()
        self.w_pad = Tensor()
        self._w_pad_version = -1  # < any real version → first forward pads
        self.col_t_bf = TensorImpl[Self.ADT]()
        self.outp_t_bf = TensorImpl[Self.ADT]()
        self.goT_t_bf = TensorImpl[Self.ADT]()
        self.w_bf = TensorImpl[Self.ADT]()
        self.b_a = TensorImpl[Self.ADT]()
        self.wT_bf = TensorImpl[Self.ADT]()
        self._w_cast_version = -1  # < any real version → first forward casts
        self._force_recast = False
        self._col_src_ptr = 0
        self.db_rows = Tensor()
        self.db_part = Tensor()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "capture_recast":
            self._force_recast = value != Scalar[DT](0.0)

    def _ensure_w_pad(mut self, c: DeviceContext) raises:
        """Ensure `w_pad` is `weight.val` widened to `[OC, CPAD]` with zeros.

        Re-pads ONLY when the optimizer bumped `val.version` since the last
        pad, so training pays it once per step and inference once ever.

        ⚠ `_force_recast` (CUDA-graph capture) must make this UNCONDITIONAL for
        the same reason it does for the bf16 cast: the version gate would skip
        the pad on replay and the GEMM would read a STALE weight."""
        self.w_pad.ensure_gpu(c, Self.WPAD_SIZE)
        if self._force_recast or self.weight.val.version != self._w_pad_version:
            c.enqueue_function[
                _pad_w_cols_kernel[
                    Self.OC_, Self.COL, Self.CPAD, Self.OCPAD
                ]
            ](
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.w_pad.lt["gpu", Layout.row_major(Self.WPAD_SIZE)](),
                grid_dim=(Self.WPAD_SIZE + CONV_TPB - 1) // CONV_TPB,
                block_dim=CONV_TPB,
            )
            self._w_pad_version = self.weight.val.version

    def _w_col_buf(mut self, c: DeviceContext) raises -> DeviceBuffer[DT]:
        """The `[OCPAD, CPAD]` weight the fp32 forward GEMM contracts against:
        the padded copy when EITHER axis needs it, otherwise the master weight
        itself (no copy, no extra kernel — both pads are identities there)."""
        comptime if Self.NEEDS_W_PAD:
            self._ensure_w_pad(c)
            return self.w_pad.dev.value()
        else:
            return self.weight.val.dev.value()

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Ensure the cached bf16 weight `w_bf` reflects the current fp32
        `weight.val`. Recasts ONLY when the optimizer bumped `val.version` since
        the last cast (so the weight cast is ONCE per step, not per fwd/bwd).
        Shared by forward (the cast) and vjp (which REUSES it — no optimizer step
        intervenes between a fwd and its bwd)."""
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        if self._force_recast or self.weight.val.version != self._w_cast_version:
            c.enqueue_function[_cast_f2b_kernel](
                self.weight.val.dev.value(),
                self.w_bf.dev.value(),
                Int64(Self.W_SIZE),
                grid_dim=(Self.W_SIZE + 255) // 256,
                block_dim=256,
            )
            self._w_cast_version = self.weight.val.version

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var c = Self()
        c.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        c.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        # Receptive-field-scaled fan_in/fan_out for a conv weight [OC, IC, K, K]
        # so Kaiming/Xavier get the RIGHT bound. This previously called a fixed
        # `(k%7-3)*0.1` placeholder that IGNORED `INIT` — a degenerate, non-random
        # init that BatchNorm masked (CIFAR/ResNet) but BN-FREE conv nets (MuZero
        # spatial h/g/f) could not recover from. `INIT=Deterministic` reproduces
        # that exact pattern, so bit-parity gates are unchanged. `init_weight`/
        # `init_bias` upload to device on GPU.
        comptime fan_in = Self.IC_ * Self.K_ * Self.K_
        comptime fan_out = Self.OC_ * Self.K_ * Self.K_
        INIT.init_weight[target](c.weight.val, Self.W_SIZE, fan_in, fan_out, ctx)
        INIT.init_bias[target](c.bias.val, Self.B_SIZE, ctx)
        return c^

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
            # ACT_DT IS DT in this branch, but the compiler doesn't collapse the
            # opaque `Self.ACT_DT` param to `DT` for type-unification against the
            # fp32 weight/bias views — so rebind the activation refs (sound: the
            # dtypes are equal here). `TensorImpl[Self.ACT_DT]` ≡ `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.OUT_FLAT)
                # ⚠ UNINITIALISED ON PURPOSE. `_im2col_cpu` writes EVERY
                # element of this buffer — the gathered value inside the input,
                # an explicit 0 outside it — and `COL` is exactly `IC*K*K`, the
                # row width it fills (the padded `CPAD` stride belongs to the
                # GPU path, not this one). So the zero-fill was a memset of a
                # buffer about to be entirely overwritten. Measured 2.2 ms per
                # forward across both cameras — less than it looks, because the
                # allocator hands back zeroed pages cheaply, which is why this
                # is a one-line change and not a buffer-reuse rewrite.
                # ⚠ `out_b` KEEPS ITS ZERO-FILL. That one is `max_matmul`'s
                # destination, and whether a GEMM writes or accumulates is the
                # library's business, not ours to assume.
                var col = List[Scalar[DT]](
                    unsafe_uninit_length=Self.SO * Self.COL
                )
                var out_b = List[Scalar[DT]](
                    length=Self.OC_ * Self.SO, fill=Scalar[DT](0)
                )
                var w_tt = TileTensor(
                    self.weight.val.data, row_major[Self.OC_, Self.COL]()
                )
                for b in range(B):
                    _im2col_cpu[
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.LAYOUT,
                    ](in0d.data, b * Self.IN_FLAT, col)
                    var col_tt = TileTensor(col, row_major(Self.SO, Self.COL))
                    var out_b_tt = TileTensor(
                        out_b, row_major(Self.OC_, Self.SO)
                    )
                    # out_b[OC,SO] = W[OC,COL] @ col[SO,COL]ᵀ
                    max_matmul[transpose_b=True, target="cpu"](
                        out_b_tt, w_tt, col_tt, None
                    )
                    # scatter + bias broadcast into out.data[b*OUT_FLAT:] (out_b
                    # is the [OC,SO] GEMM result; the output offset follows LAYOUT)
                    var base = b * Self.OUT_FLAT
                    # ⚠ VECTORISED FOR NCHW, where `_out_off` is `oc*SO + s`:
                    # source and destination are BOTH contiguous in `s` and the
                    # bias is a per-CHANNEL constant, so this is a broadcast
                    # add over a contiguous run. It was the scatter half of
                    # `Conv2D::forward`'s 14.3% main-thread self time, scalar,
                    # 3.5 ms per forward. NHWC is `s*OC + oc` — strided, and a
                    # vector store would scatter across channels.
                    comptime if Self.LAYOUT == LAYOUT_NCHW:
                        var _obp = out_b.unsafe_ptr()
                        var _dst = outd.data.unsafe_ptr()
                        for oc in range(Self.OC_):
                            var bv = self.bias.val.data[oc]
                            var bvv = SIMD[DT, CPU_SIMD_W](bv)
                            var src0 = oc * Self.SO
                            var dst0 = base + oc * Self.SO
                            var k = 0
                            while k + CPU_SIMD_W <= Self.SO:
                                _dst.unsafe_store(
                                    dst0 + k,
                                    _obp.unsafe_load[width=CPU_SIMD_W](src0 + k)
                                    + bvv,
                                )
                                k += CPU_SIMD_W
                            while k < Self.SO:
                                _dst[unsafe_offset = dst0 + k] = (
                                    _obp[unsafe_offset = src0 + k] + bv
                                )
                                k += 1
                    else:
                        for oc in range(Self.OC_):
                            var bv = self.bias.val.data[oc]
                            for s in range(Self.SO):
                                outd.data[
                                    base
                                    + _out_off[
                                        Self.LAYOUT, Self.OC_, Self.SO
                                    ](oc, s)
                                ] = (out_b[oc * Self.SO + s] + bv)
            else:
                var c = ctx.value()
                comptime BS = B * Self.SO
                outd.ensure_gpu(c, B * Self.OUT_FLAT)
                comptime if Self.use_cudnn[B]():
                    cudnn_conv_forward[
                        B, Self.IC_, Self.OC_, Self.K_, Self.S_, Self.P_,
                        Self.H_, Self.W_, Self.LAYOUT == LAYOUT_NHWC,
                        cublas_tf32(BS, Self.OC_, Self.COL),
                    ](c, outd.dev.value(), in0d.dev.value(), self.weight.val.dev.value())
                    comptime nb_b = (B * Self.OUT_FLAT + CONV_TPB - 1) // CONV_TPB
                    c.enqueue_function[
                        _bias_add_out_kernel[B, Self.OC_, Self.SO, Self.OUT_FLAT, Self.LAYOUT]
                    ](
                        outd.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                        self.bias.val.lt["gpu", Layout.row_major(Self.OC_)](),
                        grid_dim=nb_b,
                        block_dim=CONV_TPB,
                    )
                    return
                # K-aligned im2col stride (== COL when already aligned).
                self.col_t.ensure_gpu(c, BS * Self.CPAD)
                self.outp_t.ensure_gpu(c, BS * Self.OCPAD)
                # (1) im2col → col[BS, CPAD] (tail columns zeroed)
                comptime nb_col = (BS * Self.CPAD + CONV_TPB - 1) // CONV_TPB
                c.enqueue_function[
                    _im2col_kernel[
                        B,
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.IN_FLAT,
                        Self.COL,
                        Self.CPAD,
                        Self.SO,
                        BS,
                        DT,
                        Self.LAYOUT,
                    ]
                ](
                    in0d.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                    self.col_t.lt["gpu", Layout.row_major(BS, Self.CPAD)](),
                    grid_dim=nb_col,
                    block_dim=CONV_TPB,
                )
                # A2: record the input buffer so this forward's col_t can be
                # reused by the matching vjp (skipping a redundant im2col).
                self._col_src_ptr = Int(in0d.dev.value().unsafe_ptr())
                comptime if Self.OC_ == 1:
                    # (2+3 fused) OC==1: max_matmul GPU miscomputes N=1 —
                    # direct matvec + bias (see _fwd_oc1_matvec_kernel).
                    comptime nb_mv = (BS + CONV_TPB - 1) // CONV_TPB
                    c.enqueue_function[
                        _fwd_oc1_matvec_kernel[
                            B, Self.SO, Self.COL, Self.CPAD, BS, DT
                        ]
                    ](
                        self.col_t.lt["gpu", Layout.row_major(BS, Self.CPAD)](),
                        self.weight.val.lt["gpu", Layout.row_major(Self.COL)](),
                        self.bias.val.lt["gpu", Layout.row_major(1)](),
                        outd.lt["gpu", Layout.row_major(B, Self.SO)](),
                        grid_dim=nb_mv,
                        block_dim=CONV_TPB,
                    )
                else:
                    # (2) out_packed[BS,OC] = col[BS,CPAD] @ W[OC,CPAD]ᵀ.
                    # The padded columns are zero on BOTH operands, so they
                    # contribute exactly 0 — the result is the unpadded GEMM's,
                    # bit for bit, at an aligned contraction length.
                    comptime if Self.CUB:
                        # out[BS, OC] = col[BS, COL] @ W[OC, COL]ᵀ, unpadded.
                        cublas_gemm[False, True, cublas_tf32(BS, Self.OC_, Self.COL)](
                            c, self.outp_t.dev.value(), self.col_t.dev.value(),
                            self.weight.val.dev.value(),
                            BS, Self.OC_, Self.COL, 0.0,
                        )
                    else:
                        var w_buf = self._w_col_buf(c)
                        var col_tt = TileTensor(
                            self.col_t.dev.value(), row_major[BS, Self.CPAD]()
                        )
                        var w_tt = TileTensor(
                            w_buf, row_major[Self.OCPAD, Self.CPAD]()
                        )
                        var outp_tt = TileTensor(
                            self.outp_t.dev.value(), row_major[BS, Self.OCPAD]()
                        )
                        # Same treatment as the dW, and for the same reason:
                        # once `CPAD >= 2048` MAX partitions K here too and
                        # allocates its reduction workspace per call, which is a
                        # capture blocker. See `sk_ws_fwd`.
                        comptime if splitk_path_applies[c.default_device_info]():
                            if self._sk_p_fwd < 0:
                                self._decide_sk_p_fwd(BS, c)
                            if self._sk_p_fwd > 1:
                                dispatch_splitk_gemm[transpose_b=True](
                                    outp_tt, col_tt, w_tt,
                                    BS, Self.OCPAD, Self.CPAD,
                                    self._sk_p_fwd, self.sk_ws_fwd, c,
                                )
                            else:
                                mm[transpose_b=True, A0=BS, A1=Self.CPAD, B0=Self.OCPAD, B1=Self.CPAD, O0=BS, O1=Self.OCPAD](
                                    self.outp_t.dev.value(), self.col_t.dev.value(), w_buf, c
                                )
                        else:
                            mm[transpose_b=True, A0=BS, A1=Self.CPAD, B0=Self.OCPAD, B1=Self.CPAD, O0=BS, O1=Self.OCPAD](
                                self.outp_t.dev.value(), self.col_t.dev.value(), w_buf, c
                            )
                    # (3) scatter → output[B, OC·SO] + bias
                    comptime nb_sc = (
                        B * Self.OUT_FLAT + CONV_TPB - 1
                    ) // CONV_TPB
                    c.enqueue_function[
                        _scatter_bias_kernel[
                            B,
                            Self.OC_,
                            Self.SO,
                            Self.OUT_FLAT,
                            BS,
                            DT,
                            Self.LAYOUT,
                            Self.OCPAD,
                        ]
                    ](
                        self.outp_t.lt["gpu", Layout.row_major(BS, Self.OCPAD)](),
                        self.bias.val.lt["gpu", Layout.row_major(Self.OC_)](),
                        outd.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                        grid_dim=nb_sc,
                        block_dim=CONV_TPB,
                    )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert target == "gpu", "bf16-flow Conv2D is GPU-only"
            var c = ctx.value()
            comptime BS = B * Self.SO
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            self.col_t_bf.ensure_gpu(c, BS * Self.COL)
            self.outp_t_bf.ensure_gpu(c, BS * Self.OC_)
            # x (in0) is ALREADY bf16 — no input cast. W: cached bf16 (recast
            # only on a version bump). bias: cheap per-forward DT→bf16 cast.
            self._ensure_w_bf(c)
            self.b_a.ensure_gpu(c, Self.B_SIZE)
            c.enqueue_function[_cast_f2b_kernel](
                self.bias.val.dev.value(),
                self.b_a.dev.value(),
                Int64(Self.B_SIZE),
                grid_dim=(Self.B_SIZE + 255) // 256,
                block_dim=256,
            )
            # (1) im2col → col[BS, COL] (bf16-in → bf16-out)
            comptime nb_col = (BS * Self.COL + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _im2col_kernel[
                    B,
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.IN_FLAT,
                    Self.COL,
                    Self.COL,  # DCOL: bf16 flow keeps the unpadded stride
                    Self.SO,
                    BS,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                in0.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                self.col_t_bf.lt["gpu", Layout.row_major(BS, Self.COL)](),
                grid_dim=nb_col,
                block_dim=CONV_TPB,
            )
            # A2: record the input buffer for col_t_bf reuse in the matching vjp.
            self._col_src_ptr = Int(in0.dev.value().unsafe_ptr())
            comptime if Self.OC_ == 1:
                # (2+3 fused) OC==1: max_matmul GPU miscomputes N=1 —
                # direct matvec + bias (see _fwd_oc1_matvec_kernel).
                comptime nb_mv = (BS + CONV_TPB - 1) // CONV_TPB
                c.enqueue_function[
                    _fwd_oc1_matvec_kernel[
                        B, Self.SO, Self.COL, Self.COL, BS, Self.ADT
                    ]
                ](
                    self.col_t_bf.lt["gpu", Layout.row_major(BS, Self.COL)](),
                    self.w_bf.lt["gpu", Layout.row_major(Self.COL)](),
                    self.b_a.lt["gpu", Layout.row_major(1)](),
                    out.lt["gpu", Layout.row_major(B, Self.SO)](),
                    grid_dim=nb_mv,
                    block_dim=CONV_TPB,
                )
            else:
                # (2) out_packed[BS,OC] = col[BS,COL] @ W[OC,COL]ᵀ — bf16-in →
                # bf16-out GEMM (fp32 accumulation is automatic).
                mm[transpose_b=True, A0=BS, A1=Self.COL, B0=Self.OC_, B1=Self.COL, O0=BS, O1=Self.OC_](
                    self.outp_t_bf.dev.value(), self.col_t_bf.dev.value(), self.w_bf.dev.value(), c
                )
                # (3) scatter → output[B, OC·SO] + bf16 bias
                comptime nb_sc = (B * Self.OUT_FLAT + CONV_TPB - 1) // CONV_TPB
                c.enqueue_function[
                    _scatter_bias_kernel[
                        B,
                        Self.OC_,
                        Self.SO,
                        Self.OUT_FLAT,
                        BS,
                        Self.ADT,
                        Self.LAYOUT,
                    ]
                ](
                    self.outp_t_bf.lt["gpu", Layout.row_major(BS, Self.OC_)](),
                    self.b_a.lt["gpu", Layout.row_major(Self.OC_)](),
                    out.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                    grid_dim=nb_sc,
                    block_dim=CONV_TPB,
                )

    def _decide_sk_p_fwd(mut self, BS: Int, ctx: DeviceContext) raises:
        """Decide the FORWARD GEMM's partition count, once, and cache it.

        `out[BS, OCPAD] = col[BS, CPAD] @ wᵀ[OCPAD, CPAD]`, so M=BS, N=OCPAD,
        K=CPAD and `transpose_b=True`. N is the PADDED out-channel count, so
        `multi_gemm_cond`'s `n % 128` holds by construction; what decides is
        whether `CPAD >= 2048`, i.e. a 3x3 conv at IC >= 228.

        Decided on the first forward — an EAGER step, before any capture — and
        never again: P sets `grid_dim`, which is baked into a captured graph.
        """
        # `NOEIRA_SPLITK_FWD=0` pins the FORWARD to plain `max_matmul` while
        # leaving the dW routing alone. Deliberately separate from
        # `NOEIRA_SPLITK`: that one also disables the dW split-K and, because
        # `PAD_TO` is comptime, leaves the padding on — which is the arm
        # measured at 0.21x, so it is not a usable "before" for anything. This
        # switch isolates exactly one change in one binary, which is the only
        # honest way to A/B it against a baseline recorded from a build that
        # has since moved.
        if getenv("NOEIRA_SPLITK_FWD", "1") == "0":
            self._sk_p_fwd = 1
            return
        self._sk_p_fwd = decide_partitions[transpose_b=True](
            BS, Self.OCPAD, Self.CPAD, ctx
        )
        if self._sk_p_fwd > 1:
            self.sk_ws_fwd.ensure_gpu(ctx, self._sk_p_fwd * BS * Self.OCPAD)

    def _decide_sk_p(mut self, BS: Int, ctx: DeviceContext) raises:
        """Decide the conv dW GEMM's partition count, once, and cache it.

        ⚠ The conv dW fails `multi_gemm_cond` often, which is why
        `decide_partitions` checks it before anything else: `CPAD` rounds the
        im2col column count to 32, not 128, so a ResNet18 stem's N=160 goes to
        cuBLAS and substituting the multistage kernel would be a wrong answer.
        `PAD_TO = 128` is what keeps the common shapes on the right side of it.
        """
        self._sk_p = decide_partitions(Self.OC_, Self.CPAD, BS, ctx)
        if self._sk_p > 1:
            self.sk_ws.ensure_gpu(ctx, self._sk_p * Self.OC_ * Self.CPAD)

    def _enqueue_db[B: Int](mut self, c: DeviceContext, mut god: Tensor) raises:
        """`grad_bias += Σ_{b, s} grad_output[b, oc, s]`, deterministic, in two
        passes. The old `_backward_db_kernel` gave each channel ONE block
        walking B·OH·OW values: 16 blocks on a 170-SM card for a ResNet stem,
        ~118 us at batch 100 x 32x32 — longer than cuDNN's whole dgrad.
        NCHW: per-(b, oc) row sums over the contiguous OH·OW, then the
        [B, OC] column sum. NHWC: grad_output already is [B·OH·OW, OC]."""
        comptime if Self.LAYOUT == LAYOUT_NHWC:
            enqueue_bias_grad[DT](
                c, god.dev.value(), self.bias.grd.dev.value(), B * Self.SO, Self.OC_, self.db_part
            )
        else:
            self.db_rows.ensure_gpu(c, B * Self.OC_)
            c.enqueue_function[_rowsum_kernel[B * Self.OC_, Self.SO]](
                god.lt["gpu", Layout.row_major(B * Self.OC_, Self.SO)](),
                self.db_rows.lt["gpu", Layout.row_major(B * Self.OC_)](),
                grid_dim=B * Self.OC_,
                block_dim=CONV_DW_TPB,
            )
            enqueue_bias_grad[DT](
                c, self.db_rows.dev.value(), self.bias.grd.dev.value(), B, Self.OC_, self.db_part
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
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
          # ── fp32 path (legacy NoAMP, byte-identical) ──
          # ACT_DT IS DT here — rebind the activation refs (sound; see forward).
          ref find = rebind[Tensor](fin)
          ref gind = rebind[Tensor](gin)
          ref god = rebind[Tensor](grad_output)
          comptime if target == "cpu":
            gind.ensure(B * Self.IN_FLAT)
            for k in range(B * Self.IN_FLAT):
                gind.data[k] = Scalar[DT](0)
            var col = List[Scalar[DT]](
                length=Self.SO * Self.COL, fill=Scalar[DT](0)
            )
            var d_col = List[Scalar[DT]](
                length=Self.SO * Self.COL, fill=Scalar[DT](0)
            )
            var w_tt = TileTensor(
                self.weight.val.data, row_major[Self.OC_, Self.COL]()
            )
            # Apple-fp32: fused cblas paths (beta=1 dW-accumulate + TRANSPOSE
            # d_col), matching legacy Conv2D. Elsewhere: portable max_matmul
            # into a temp + add (one extra W_SIZE pass) and an explicit
            # transpose (max_matmul rejects transpose_a).
            comptime IS_APPLE_F32 = (
                CompilationTarget.is_macos() and DT == DType.float32
            )
            for b in range(B):
                var go_base = b * Self.OUT_FLAT
                # d_bias[oc] += Σ_s go[oc, s]
                for oc in range(Self.OC_):
                    var acc: Scalar[DT] = 0
                    for s in range(Self.SO):
                        acc += god.data[
                            go_base
                            + _out_off[Self.LAYOUT, Self.OC_, Self.SO](oc, s)
                        ]
                    self.bias.grd.data[oc] += acc
                # rebuild col_b = im2col(x_b)
                _im2col_cpu[
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.LAYOUT,
                ](find.data, b * Self.IN_FLAT, col)
                # The Apple-cblas fused path reinterprets grad_output memory as a
                # contiguous [OC, SO] matrix — only valid for NCHW. NHWC falls back
                # to the portable gather-into-[OC,SO] path below.
                comptime if IS_APPLE_F32 and Self.LAYOUT == LAYOUT_NCHW:
                    var cblas = get_cblas_f32_function()
                    var go_b_p = god.data.unsafe_ptr().unsafe_offset(go_base)
                    # dW += go_b[OC,SO] @ col_b[SO,COL]  (beta=1, no temp)
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.NO_TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.OC_),
                        Int32(Self.COL),
                        Int32(Self.SO),
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](go_b_p),
                        Int32(Self.SO),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            col.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                        Float32(1.0),
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            self.weight.grd.data.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                    )
                    # d_col[SO,COL] = go_bᵀ[SO,OC] @ W[OC,COL]  (beta=0, A^T)
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.SO),
                        Int32(Self.COL),
                        Int32(Self.OC_),
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](go_b_p),
                        Int32(Self.SO),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            self.weight.val.data.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                        Float32(0.0),
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            d_col.unsafe_ptr()
                        ),
                        Int32(Self.COL),
                    )
                else:
                    # gather grad_output (LAYOUT-ordered) into packed [OC, SO]
                    var go_b = List[Scalar[DT]](
                        length=Self.OC_ * Self.SO, fill=Scalar[DT](0)
                    )
                    for oc in range(Self.OC_):
                        for s in range(Self.SO):
                            go_b[oc * Self.SO + s] = god.data[
                                go_base
                                + _out_off[Self.LAYOUT, Self.OC_, Self.SO](oc, s)
                            ]
                    comptime if Self.COL == 1:
                        # Degenerate N=COL=1 GEMM (a 1x1 conv with IC=1 — the
                        # MuZero action-embedding). `max_matmul` aborts on N=1 on
                        # the non-Apple path, so compute the two matrix-vector
                        # products directly. col / d_col are [SO, 1] == [SO];
                        # W is [OC, 1] == weight[oc].
                        #   dW[oc]  += Σ_s go[oc,s]·col[s]
                        #   d_col[s] = Σ_oc go[oc,s]·W[oc]
                        for oc in range(Self.OC_):
                            var acc = Scalar[DT](0)
                            for s in range(Self.SO):
                                acc += go_b[oc * Self.SO + s] * col[s]
                            self.weight.grd.data[oc] += acc
                        for s in range(Self.SO):
                            var acc = Scalar[DT](0)
                            for oc in range(Self.OC_):
                                acc += go_b[oc * Self.SO + s] * self.weight.val.data[oc]
                            d_col[s] = acc
                    else:
                        var col_tt = TileTensor(
                            col, row_major(Self.SO, Self.COL)
                        )
                        var go_b_tt = TileTensor(
                            go_b, row_major(Self.OC_, Self.SO)
                        )
                        # dW += go_b[OC,SO] @ col[SO,COL]
                        var dw_tmp = List[Scalar[DT]](
                            length=Self.W_SIZE, fill=Scalar[DT](0)
                        )
                        var dw_tmp_tt = TileTensor(
                            dw_tmp, row_major(Self.OC_, Self.COL)
                        )
                        max_matmul[target="cpu"](dw_tmp_tt, go_b_tt, col_tt, None)
                        for k in range(Self.W_SIZE):
                            self.weight.grd.data[k] += dw_tmp[k]
                        # d_col[SO,COL] = go_bᵀ[SO,OC] @ W[OC,COL]
                        var go_b_T = List[Scalar[DT]](
                            length=Self.SO * Self.OC_, fill=Scalar[DT](0)
                        )
                        for s in range(Self.SO):
                            for oc in range(Self.OC_):
                                go_b_T[s * Self.OC_ + oc] = go_b[
                                    oc * Self.SO + s
                                ]
                        var go_b_T_tt = TileTensor(
                            go_b_T, row_major(Self.SO, Self.OC_)
                        )
                        var d_col_tt = TileTensor(
                            d_col, row_major(Self.SO, Self.COL)
                        )
                        max_matmul[target="cpu"](
                            d_col_tt, go_b_T_tt, w_tt, None
                        )
                # col2im → grad_input_b (scatter-add)
                _col2im_cpu[
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.LAYOUT,
                ](d_col, gind.data, b * Self.IN_FLAT)
          else:
            var c = ctx.value()
            comptime BS = B * Self.SO
            gind.ensure_gpu(c, B * Self.IN_FLAT)
            comptime if Self.use_cudnn[B]():
                comptime NHWC = Self.LAYOUT == LAYOUT_NHWC
                comptime TF = cublas_tf32(BS, Self.OC_, Self.COL)
                cudnn_conv_backward_filter[
                    B, Self.IC_, Self.OC_, Self.K_, Self.S_, Self.P_,
                    Self.H_, Self.W_, NHWC, TF,
                ](c, self.weight.grd.dev.value(), find.dev.value(), god.dev.value())
                self._enqueue_db[B](c, god)
                cudnn_conv_backward_data[
                    B, Self.IC_, Self.OC_, Self.K_, Self.S_, Self.P_,
                    Self.H_, Self.W_, NHWC, TF,
                ](c, gind.dev.value(), self.weight.val.dev.value(), god.dev.value())
                return
            self.col_t.ensure_gpu(c, BS * Self.CPAD)
            self.goT_t.ensure_gpu(c, Self.OC_ * BS)
            # [OC, CPAD] when padding — the dW GEMM's OUTPUT width is COL, the
            # axis that picks the split-K workspace path on cuBLAS.
            self.dW_tmp.ensure_gpu(c, Self.DWPAD_SIZE)
            # (1) col = im2col(x) — A2: REUSE the forward's col_t when this vjp's
            # forward_input is the buffer the forward im2col'd; else recompute
            # (safe fallback). See `_col_src_ptr`.
            if (
                self._col_src_ptr == 0
                or Int(find.dev.value().unsafe_ptr()) != self._col_src_ptr
            ):
                comptime nb_col = (BS * Self.CPAD + TPB - 1) // TPB
                c.enqueue_function[
                    _im2col_kernel[
                        B,
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.IN_FLAT,
                        Self.COL,
                        Self.CPAD,
                        Self.SO,
                        BS,
                        DT,
                        Self.LAYOUT,
                    ]
                ](
                    find.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                    self.col_t.lt["gpu", Layout.row_major(BS, Self.CPAD)](),
                    grid_dim=nb_col,
                    block_dim=TPB,
                )
            # (2) goᵀ[OC,BS] = transpose(grad_output)
            comptime nb_got = (Self.OC_ * BS + TPB - 1) // TPB
            c.enqueue_function[
                _go_transpose_kernel[
                    B,
                    Self.OC_,
                    Self.SO,
                    Self.OUT_FLAT,
                    BS,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                god.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.goT_t.lt["gpu", Layout.row_major(Self.OC_, BS)](),
                grid_dim=nb_got,
                block_dim=TPB,
            )
            # (3) grad_w += goᵀ @ col
            comptime if Self.CUB:
                # One cuBLAS call, β = 1 straight into the master grad: no
                # temporary, no accumulate kernel, no padding.
                cublas_gemm[False, False, cublas_tf32(BS, Self.OC_, Self.COL)](
                    c, self.weight.grd.dev.value(), self.goT_t.dev.value(),
                    self.col_t.dev.value(),
                    Self.OC_, Self.COL, BS, 1.0,
                )
            else:
                # (3) dW_tmp = goᵀ @ col → accumulate into grad_w
                var goT_tt = TileTensor(
                    self.goT_t.dev.value(), row_major[Self.OC_, BS]()
                )
                var col_tt = TileTensor(
                    self.col_t.dev.value(), row_major[BS, Self.CPAD]()
                )
                var dW_tmp_tt = TileTensor(
                    self.dW_tmp.dev.value(), row_major[Self.OC_, Self.CPAD]()
                )
                # ── dW: split-K on OUR workspace, or plain matmul ──────────
                # `[OC, BS] @ [BS, CPAD]`: K is `batch * OH * OW`, so this is the
                # longest-K GEMM in the model and the one `select_config`
                # under-partitions worst (a ResNet18 stem is TWO tiles, so MAX's
                # P=8 puts 16 blocks on a 170-SM card). Inert unless MAX's own
                # dispatch would have reached a partitioned `multistage_gemm`.
                comptime if splitk_path_applies[c.default_device_info]():
                    if self._sk_p < 0:
                        self._decide_sk_p(BS, c)
                    if self._sk_p > 1:
                        dispatch_splitk_gemm(
                            dW_tmp_tt, goT_tt, col_tt,
                            Self.OC_, Self.CPAD, BS,
                            self._sk_p, self.sk_ws, c,
                        )
                    else:
                        mm[A0=Self.OC_, A1=BS, B0=BS, B1=Self.CPAD, O0=Self.OC_, O1=Self.CPAD](
                            self.dW_tmp.dev.value(), self.goT_t.dev.value(), self.col_t.dev.value(), c
                        )
                else:
                    mm[A0=Self.OC_, A1=BS, B0=BS, B1=Self.CPAD, O0=Self.OC_, O1=Self.CPAD](
                        self.dW_tmp.dev.value(), self.goT_t.dev.value(), self.col_t.dev.value(), c
                    )
                # ⚠ STRIDED accumulate: dW comes back `[OC, CPAD]` and the master
                # grad is `[OC, COL]`. A flat add folds each row's padding into the
                # next row's leading weights — see `_accum_w_2d_kernel`.
                comptime nb_acc = (Self.W_SIZE + TPB - 1) // TPB
                comptime if Self.NEEDS_COL_PAD:
                    c.enqueue_function[
                        _accum_w_2d_kernel[Self.OC_, Self.COL, Self.CPAD]
                    ](
                        self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                        self.dW_tmp.lt["gpu", Layout.row_major(Self.DWPAD_SIZE)](),
                        grid_dim=nb_acc,
                        block_dim=TPB,
                    )
                else:
                    c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                        self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                        self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                        grid_dim=nb_acc,
                        block_dim=TPB,
                    )
            # (4) d_bias — two-pass, deterministic (`_enqueue_db`)
            self._enqueue_db[B](c, god)
            # (5) d_input (O2): d_colᵀ[COL,BS] = Wᵀ[COL,OC] @ goᵀ[OC,BS] reusing
            # the goᵀ from (2) — NO `_go_pack` kernel — then the coalesced col2im
            # reading the transposed d_colᵀ. col_t (free after the dW GEMM) is
            # reused as the [COL, BS] d_colᵀ buffer.
            comptime if Self.CUB:
                # d_colᵀ[COL, BS] = W[OC, COL]ᵀ @ goᵀ[OC, BS]: cuBLAS takes the
                # transpose, so no Wᵀ copy.
                cublas_gemm[True, False, cublas_tf32(BS, Self.OC_, Self.COL)](
                    c, self.col_t.dev.value(), self.weight.val.dev.value(),
                    self.goT_t.dev.value(),
                    Self.COL, BS, Self.OC_, 0.0,
                )
            else:
                self.wT_t.ensure_gpu(c, Self.COL * Self.OC_)
                comptime nb_wt = (Self.OC_ * Self.COL + TPB - 1) // TPB
                c.enqueue_function[_wT_transpose_kernel[Self.OC_, Self.COL]](
                    self.weight.val.lt["gpu", Layout.row_major(Self.OC_, Self.COL)](),
                    self.wT_t.lt["gpu", Layout.row_major(Self.COL, Self.OC_)](),
                    grid_dim=nb_wt,
                    block_dim=TPB,
                )
                mm[A0=Self.COL, A1=Self.OC_, B0=Self.OC_, B1=BS, O0=Self.COL, O1=BS](
                    self.col_t.dev.value(), self.wT_t.dev.value(), self.goT_t.dev.value(), c
                )
            comptime nb_dx = (B * Self.IN_FLAT + CONV_DW_TPB - 1) // CONV_DW_TPB
            c.enqueue_function[
                _dx_col2im_kernel[
                    B,
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.IN_FLAT,
                    Self.COL,
                    Self.SO,
                    BS,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                self.col_t.lt["gpu", Layout.row_major(Self.COL, BS)](),
                gind.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                grid_dim=nb_dx,
                block_dim=CONV_DW_TPB,
            )
            # col_t now holds d_colᵀ, not this input's im2col: a second vjp
            # on the same forward must recompute it, not reuse it (A2).
            self._col_src_ptr = 0
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert target == "gpu", "bf16-flow Conv2D is GPU-only"
            var c = ctx.value()
            comptime BS = B * Self.SO
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            self.col_t_bf.ensure_gpu(c, BS * Self.COL)
            self.goT_t_bf.ensure_gpu(c, Self.OC_ * BS)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)  # fp32 dW temp
            self._ensure_w_bf(c)  # cached bf16 weight (reused from forward)
            # (1) col = im2col(x): x (fin) ALREADY bf16 → bf16 col. A2: REUSE the
            # forward's col_t_bf when fin is the buffer the forward im2col'd; else
            # recompute (safe fallback). See `_col_src_ptr`.
            if (
                self._col_src_ptr == 0
                or Int(fin.dev.value().unsafe_ptr()) != self._col_src_ptr
            ):
                comptime nb_col = (BS * Self.COL + TPB - 1) // TPB
                c.enqueue_function[
                    _im2col_kernel[
                        B,
                        Self.IC_,
                        Self.K_,
                        Self.S_,
                        Self.P_,
                        Self.H_,
                        Self.W_,
                        Self.OH,
                        Self.OW,
                        Self.IN_FLAT,
                        Self.COL,
                        Self.COL,  # DCOL: bf16 flow keeps the unpadded stride
                        Self.SO,
                        BS,
                        Self.ADT,
                        Self.LAYOUT,
                    ]
                ](
                    fin.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                    self.col_t_bf.lt["gpu", Layout.row_major(BS, Self.COL)](),
                    grid_dim=nb_col,
                    block_dim=TPB,
                )
            # (2) goᵀ[OC,BS] = transpose(grad_output): bf16 go → bf16 goᵀ.
            comptime nb_got = (Self.OC_ * BS + TPB - 1) // TPB
            c.enqueue_function[
                _go_transpose_kernel[
                    B,
                    Self.OC_,
                    Self.SO,
                    Self.OUT_FLAT,
                    BS,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.goT_t_bf.lt["gpu", Layout.row_major(Self.OC_, BS)](),
                grid_dim=nb_got,
                block_dim=TPB,
            )
            # (3) dW_tmp = goᵀ @ col → bf16-in, FP32-out GEMM → accumulate into
            # the fp32 master grad.
            # ⚠ NOT routed through split-K, deliberately. This is the
            # bf16-flow dW (bf16 operands, fp32 output). It would work the same
            # way, but no gate builds bf16 Conv2Ds and a mistake here would be
            # silent. Route it when there is a bf16 arm in the gate, not before
            # — same call as `Linear`'s bf16 dW site.
            mm[A0=Self.OC_, A1=BS, B0=BS, B1=Self.COL, O0=Self.OC_, O1=Self.COL](
                self.dW_tmp.dev.value(), self.goT_t_bf.dev.value(), self.col_t_bf.dev.value(), c
            )
            comptime nb_acc = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=nb_acc,
                block_dim=TPB,
            )
            # (4) d_bias — bf16 go → FP32 master grad (fp32 accumulator).
            c.enqueue_function[
                _backward_db_kernel[
                    B,
                    Self.OC_,
                    Self.OH,
                    Self.OW,
                    Self.OUT_FLAT,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_FLAT)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OC_)](),
                grid_dim=Self.OC_,
                block_dim=CONV_DW_TPB,
            )
            # (5) d_input (O2): d_colᵀ[COL,BS] = Wᵀ_bf[COL,OC] @ goᵀ[OC,BS]
            # reusing goᵀ from (2) — NO `_go_pack` kernel. All bf16 (grad_input
            # flows bf16); Wᵀ_bf transposes the forward's cached bf16 weight.
            # col_t_bf (free after the dW GEMM) is reused as the [COL,BS] d_colᵀ.
            self.wT_bf.ensure_gpu(c, Self.COL * Self.OC_)
            comptime nb_wt = (Self.OC_ * Self.COL + TPB - 1) // TPB
            c.enqueue_function[
                _wT_transpose_kernel[Self.OC_, Self.COL, Self.ADT]
            ](
                self.w_bf.lt["gpu", Layout.row_major(Self.OC_, Self.COL)](),
                self.wT_bf.lt["gpu", Layout.row_major(Self.COL, Self.OC_)](),
                grid_dim=nb_wt,
                block_dim=TPB,
            )
            mm[A0=Self.COL, A1=Self.OC_, B0=Self.OC_, B1=BS, O0=Self.COL, O1=BS](
                self.col_t_bf.dev.value(), self.wT_bf.dev.value(), self.goT_t_bf.dev.value(), c
            )
            comptime nb_dx = (B * Self.IN_FLAT + CONV_DW_TPB - 1) // CONV_DW_TPB
            c.enqueue_function[
                _dx_col2im_kernel[
                    B,
                    Self.IC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.H_,
                    Self.W_,
                    Self.OH,
                    Self.OW,
                    Self.IN_FLAT,
                    Self.COL,
                    Self.SO,
                    BS,
                    Self.ADT,
                    Self.LAYOUT,
                ]
            ](
                self.col_t_bf.lt["gpu", Layout.row_major(Self.COL, BS)](),
                gin.lt["gpu", Layout.row_major(B, Self.IN_FLAT)](),
                grid_dim=nb_dx,
                block_dim=CONV_DW_TPB,
            )
            # col_t_bf now holds d_colᵀ (see the fp32 path): invalidate A2.
            self._col_src_ptr = 0

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update weight + bias toward `src` (target ← online). Required
        for use as a target net (pixel-obs critics use Conv2D); the Module
        default is a no-op which would silently freeze the target (see
        LinearReLU polyak bug, Stage-5)."""
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )
        # ⚠ `polyak_tensor` writes `weight.val` IN PLACE without bumping
        # `val.version`, so the padded (`w_pad`) and bf16 (`w_bf`) weight
        # caches would keep serving the PRE-SYNC weight: a target conv net
        # frozen at its init weights. `Linear` / `LinearAct` learned this from
        # a DQN smoke (eval 200 -> 9); this one was missing it, and Rainbow on
        # pixel Pong stopped learning on the padded MAX path (1M steps: mean
        # return -20.9 vs +16 once fixed). Gate: `test_conv2d_gpu`
        # (`check_polyak`).
        self._w_pad_version = -1
        self._w_cast_version = -1

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
