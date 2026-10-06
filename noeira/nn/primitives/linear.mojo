"""Linear — Module conformer (CPU + GPU). y = x @ W + b via max_matmul.

Each forward/vjp branches `comptime if target == "cpu"` (tracked `TileTensor`
over `.data`) `else` (device `LayoutTensor` via `lt_gpu` + a naive kernel).
The storage surface (`ref/mut Tensor`, `TensorRefs`) is identical on both
targets; the only GPU erasure is the kernel-arg `MutAnyOrigin`. Params are
`Param` (two `Tensor`s, cpu+dev).

bf16-FLOW (AMP "Step B"): `Linear[IN, OUT]` is fp32 (unchanged), while
`Linear[IN, OUT, DType.bfloat16]` is a bf16-flow linear whose ACTIVATIONS are
STORED and FLOW at bf16 (`ACT_DT == bfloat16`) — there is NO per-call x/out cast
(the legacy "cast-around" AMP tax is gone). Master weights/grads/bias STAY fp32
(`Param` is always `DT`); only a CACHED bf16 weight copy (`w_bf`, version-gated)
and a cached bf16 bias (`b_a`) are low-precision. The fp32 (ACT_DT == DT) path is
byte-for-byte the legacy NoAMP path; the bf16 path is GPU-only (cblas/CPU matmul
is fp32-only).

LIFETIME NOTE: a pack subscript (`inputs[k]`) returns a TEMPORARY ref. Building
a view from `inputs[k].data` directly and using it LATER dangles (the temporary
dies at the end of the statement; a later op clobbers the stack). So each body
first binds the element to a named `ref` (`ref in0 = inputs[0]`) that lives for
the whole function, then builds views from that.
"""

from noeira.nn.core.mm import mm, mm_bias, bmm
from noeira.nn.core.cublas_gemm import cublas_gemm, CUBLAS_BWD, cublas_fwd, cublas_tf32
from std.sys import has_nvidia_gpu_accelerator
from std.sys import CompilationTarget
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
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

from noeira.nn.constants import DT, CPU_SIMD_W
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


# ── kernels (non-GEMM ops; the three matmuls go through max_matmul) ──────
# `_bias_add_kernel` is dtype-parametric: the fp32 path calls it with DT, the
# bf16-flow path with bfloat16 (activation + cached bf16 bias both at ADT).
def _bias_add_kernel[
    ADT: DType = DT
](
    o: Pointer[Scalar[ADT], MutAnyOrigin],
    bias: Pointer[Scalar[ADT], MutAnyOrigin],
    b_arg: Int64,
    out_arg: Int64,
):
    """`o[b, j] += bias[j]` over a row-major [B, OUT]; dims at RUNTIME so one
    module serves every shape (docs/COMPILE_TIME_PROFILING.md §3.9)."""
    var n_out = Int(out_arg)
    var idx = Int(global_idx.x)
    if idx < Int(b_arg) * n_out:
        o[unsafe_offset=idx] += bias[unsafe_offset=idx % n_out]


# Naive grad_w transpose (one thread/elem, strided read). Still used by
# linear_relu / noisy_linear; linear + linear_act use the tiled B1' kernel below.
def _transpose_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
):
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var idx = Int(global_idx.x)
    if idx < rows * cols:
        var r = idx // cols
        var c = idx % cols
        dst[unsafe_offset=c * rows + r] = src[unsafe_offset=idx]


comptime _T_TILE = 32
comptime _T_BR = 8  # 32x8 BLOCK_ROWS, 4 elems/thread (B1')


# Tiled grad_w transpose: 32x8 BLOCK_ROWS shared-mem tile (B1'). Coalesces the
# strided src read the naive kernel suffers; bench (RTX 5090) showed up to 1.65x
# on skinny grad_w (large ROWS·COLS), neutral on square. Launch with
# grid_dim=(ceildiv(COLS,32), ceildiv(ROWS,32)), block_dim=(32, 8).
# Dtype-parametric (`ADT`): the fp32 path transposes a DT activation; the bf16
# path transposes the bf16 forward-input directly (its source is already bf16).
def _transpose_tiled_kernel[
    ADT: DType = DT
](
    src: Pointer[Scalar[ADT], MutAnyOrigin],
    dst: Pointer[Scalar[ADT], MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
):
    var ROWS = Int(rows_arg)
    var COLS = Int(cols_arg)
    var tile = LayoutTensor[
        ADT,
        Layout.row_major(_T_TILE, _T_TILE + 1),  # +1 pad → no bank conflicts
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var cy = Int(block_idx.y) * _T_TILE  # tile origin in ROWS
    var cx = Int(block_idx.x) * _T_TILE  # tile origin in COLS
    var tx = Int(thread_idx.x)  # [0, _T_TILE)
    var ty = Int(thread_idx.y)  # [0, _T_BR)

    var c = cx + tx
    comptime for r in range(0, _T_TILE, _T_BR):
        var rr = cy + ty + r
        if rr < ROWS and c < COLS:
            tile[ty + r, tx] = src[unsafe_offset=rr * COLS + c]
    barrier()

    var r2 = cy + tx  # dst col (coalesced, stride 1)
    comptime for r in range(0, _T_TILE, _T_BR):
        var c2 = cx + ty + r  # dst row
        if r2 < ROWS and c2 < COLS:
            dst[unsafe_offset=(c2) * ROWS + (r2)] = rebind[Scalar[ADT]](tile[tx, ty + r])


def _accum_kernel(
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        dst[unsafe_offset=i] += src[unsafe_offset=i]


# grad_b += colsum(go): `enqueue_bias_grad`. Dtype-parametric on the
# grad_output activation (`ADT`): the bf16 path reads a bf16 `go` and
# accumulates into the FP32 `gb` (each element cast to DT before summing).
comptime _GB_COLS = 32
comptime _GB_ROWS = 8
comptime _GB_CHUNK = 1024
"""Rows per partial sum of the bias gradient."""


def _lin_gb_partial_kernel[
    ADT: DType = DT
](
    go: Pointer[Scalar[ADT], MutAnyOrigin],
    part: Pointer[Scalar[DT], MutAnyOrigin],
    b_arg: Int64,
    out_arg: Int64,
):
    """`part[c, j] = sum of go[b, j] over the rows of chunk c` (`_GB_CHUNK`
    rows). Block (`_GB_COLS`, `_GB_ROWS`): lanes along j (coalesced), the
    `_GB_ROWS` row-strided sums reduced in a fixed order in shared memory."""
    var n_out = Int(out_arg)
    var n_b = Int(b_arg)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var j = Int(block_idx.x) * _GB_COLS + tx
    var r0 = Int(block_idx.y) * _GB_CHUNK
    var r1 = min(r0 + _GB_CHUNK, n_b)
    var acc = LayoutTensor[
        DT, Layout.row_major(_GB_ROWS, _GB_COLS), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var s: Scalar[DT] = 0
    if j < n_out:
        var r = r0 + ty
        while r < r1:
            s += go[unsafe_offset=r * n_out + j].cast[DT]()
            r += _GB_ROWS
    acc[ty, tx] = s
    barrier()
    if ty == 0 and j < n_out:
        var t: Scalar[DT] = 0
        comptime for k in range(_GB_ROWS):
            t += rebind[Scalar[DT]](acc[k, tx])
        part[unsafe_offset=Int(block_idx.y) * n_out + j] = t


def _lin_gb_final_kernel(
    part: Pointer[Scalar[DT], MutAnyOrigin],
    gb: Pointer[Scalar[DT], MutAnyOrigin],
    chunks_arg: Int64,
    out_arg: Int64,
):
    """`gb[j] += sum_c part[c, j]`, chunks in order."""
    var n_out = Int(out_arg)
    var j = Int(global_idx.x)
    if j < n_out:
        var s: Scalar[DT] = 0
        for c in range(Int(chunks_arg)):
            s += part[unsafe_offset=c * n_out + j]
        gb[unsafe_offset=j] += s


def enqueue_bias_grad[ADT: DType](
    c: DeviceContext,
    go: DeviceBuffer[ADT],
    gb: DeviceBuffer[DT],
    B: Int,
    OUT: Int,
    mut part: Tensor,
) raises:
    """`gb += colsum(go)` for a [B, OUT] grad_output, deterministic.

    Replaces `_lin_gb_kernel`'s one thread per column, which gave the whole
    reduction OUT threads (a few blocks) each walking all B rows: on the
    LeWM encoder (B = 131,584 token rows) it was 39 % of a training step's
    GPU time on a 5090, 0.58 s of 1.47. Two passes now: chunk partials over
    a (OUT / 32, B / 1024) grid, then one thread per column adds the chunks
    in order. `part` is the caller's scratch, sized lazily (stable after the
    first step, so capture-safe)."""
    var chunks = (B + _GB_CHUNK - 1) // _GB_CHUNK
    part.ensure_gpu(c, chunks * OUT)
    c.enqueue_function[_lin_gb_partial_kernel[ADT]](
        go,
        part.dev.value(),
        Int64(B),
        Int64(OUT),
        grid_dim=((OUT + _GB_COLS - 1) // _GB_COLS, chunks),
        block_dim=(_GB_COLS, _GB_ROWS),
    )
    c.enqueue_function[_lin_gb_final_kernel](
        part.dev.value(),
        gb,
        Int64(chunks),
        Int64(OUT),
        grid_dim=(OUT + 255) // 256,
        block_dim=256,
    )


comptime BF16 = DType.bfloat16


def _cast_f2b_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[BF16], MutAnyOrigin],
    n_arg: Int64,
):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        dst[unsafe_offset=i] = src[unsafe_offset=i].cast[BF16]()


def _cast_b2f_kernel(
    src: Pointer[Scalar[BF16], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        dst[unsafe_offset=i] = src[unsafe_offset=i].cast[DT]()


def _pad_cols_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    src_cols_arg: Int64,
    dst_cols_arg: Int64,
):
    """[ROWS, SRC_COLS] -> [ROWS, DST_COLS], zero beyond SRC_COLS. Serves both
    K-alignment pads (the activation, and the weight as the ROWS=1 flat
    case)."""
    var src_cols = Int(src_cols_arg)
    var dst_cols = Int(dst_cols_arg)
    var i = Int(global_idx.x)
    if i < Int(rows_arg) * dst_cols:
        var r = i // dst_cols
        var c = i % dst_cols
        if c < src_cols:
            dst[unsafe_offset=i] = src[unsafe_offset=r * src_cols + c]
        else:
            dst[unsafe_offset=i] = Scalar[DT](0)


def _pad_2d_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src_rows_arg: Int64,
    src_cols_arg: Int64,
    dst_rows_arg: Int64,
    dst_cols_arg: Int64,
):
    """`dst[r, c] = src[r, c]` inside the source rectangle, 0 outside."""
    var src_rows = Int(src_rows_arg)
    var src_cols = Int(src_cols_arg)
    var dst_cols = Int(dst_cols_arg)
    var i = Int(global_idx.x)
    if i < Int(dst_rows_arg) * dst_cols:
        var r = i // dst_cols
        var c = i % dst_cols
        if r < src_rows and c < src_cols:
            dst[unsafe_offset=i] = src[unsafe_offset=r * src_cols + c]
        else:
            dst[unsafe_offset=i] = Scalar[DT](0)


def _bias_add_slice_kernel(
    ypad: Pointer[Scalar[DT], MutAnyOrigin],
    bias: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    b_arg: Int64,
    out_arg: Int64,
    n_pad_arg: Int64,
):
    """`dst[b, j] = ypad[b, j] + bias[j]`, reading the padded [B, N_PAD]."""
    var n_out = Int(out_arg)
    var n_pad = Int(n_pad_arg)
    var i = Int(global_idx.x)
    if i < Int(b_arg) * n_out:
        var b = i // n_out
        var j = i % n_out
        dst[unsafe_offset=i] = ypad[unsafe_offset=b * n_pad + j] + bias[unsafe_offset=j]


def _slice_cols_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    dst_cols_arg: Int64,
    src_cols_arg: Int64,
):
    """The inverse of `_pad_cols_kernel`: [ROWS, SRC_COLS] -> [ROWS, DST_COLS].
    """
    var dst_cols = Int(dst_cols_arg)
    var src_cols = Int(src_cols_arg)
    var i = Int(global_idx.x)
    if i < Int(rows_arg) * dst_cols:
        var r = i // dst_cols
        var cc = i % dst_cols
        dst[unsafe_offset=i] = src[unsafe_offset=r * src_cols + cc]


def _accum_2d_kernel(
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    src_cols_arg: Int64,
):
    """`dst[r, c] += src[r, c]` where src has a wider row stride (padded dW
    into the logical master grad)."""
    var cols = Int(cols_arg)
    var src_cols = Int(src_cols_arg)
    var i = Int(global_idx.x)
    if i < Int(rows_arg) * cols:
        var r = i // cols
        var cc = i % cols
        dst[unsafe_offset=i] += src[unsafe_offset=r * src_cols + cc]


# ── Linear ─────────────────────────────────────────────────────────────

# NOTE: the dW GEMM's tile-config dispatch and its partition-count decision
# both live in `nn/core/splitk_gemm.mojo` (`dispatch_splitk_gemm` /
# `decide_partitions`). They used to be written out here and were then copied
# verbatim into `Conv2D`; a rule whose failure mode is a silent wrong gradient
# gets exactly one home.


@always_inline
def _gemm_bkn[
    B: Int, K: Int, N: Int
](
    mut dst: DeviceBuffer[DT],
    x: DeviceBuffer[DT],
    w: DeviceBuffer[DT],
    c: DeviceContext,
) raises:
    """`dst[B,N] = x[B,K] @ w[K,N]` through MAX's matmul.

    The views carry RUNTIME extents, so every shape shares one set of
    dispatch kernels instead of instantiating all ~10 candidates per shape
    (docs/COMPILE_TIME_PROFILING.md §3.8: 1101 -> 456 Metal modules on the
    ACT trainer). Measured on Apple: runtime extents are also faster at every
    training batch (296 -> 163 us at [64x512]@[512x512]) but lose the gemv
    specialisation at B == 1 (55 -> 90 us), which is the acting path, so
    that one shape keeps its static views."""
    mm[A0=B, A1=K, B0=K, B1=N, O0=B, O1=N](dst, x, w, c)


struct Linear[IN_: Int, OUT_: Int, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_

    # ── K-alignment padding (GPU fp32 forward) ───────────────────────────
    # `max_matmul` falls off a ~10x cliff when the CONTRACTION dim is
    # misaligned, on BOTH backends — Metal wants K%16, an RTX 5090 wants K%32,
    # so 32 satisfies both (it costs Metal ~4% against its own optimum and is
    # worth 7-24x on the 5090). Measured by
    # `benchmarks/bench_matmul_k_alignment.mojo`; run it before changing PAD_TO.
    #
    #     [268 x 518] @ [518 x 512]   1356 us (Metal)    206 us (5090)
    #     [268 x 544] @ [544 x 512]    150 us            27 us
    #
    # `IN_` is a concatenation width in most of the repo — TD-MPC2's
    # `ZA = LATENT + ACT` = 518, a SAC critic's `obs|act`, a two-hot `BINS`
    # vector — so it lands on an arbitrary K constantly.
    #
    # ⚠ The padding is INTERNAL. `IN_` and the checkpointed `weight` Param stay
    # at their logical size, so this changes no on-disk format and no existing
    # checkpoint. The alternative (widening the caller's concat) would have
    # changed every TD-MPC2 net shape and invalidated every checkpoint.
    #
    # ⚠ The gate has TWO K terms and this pad must satisfy BOTH: `k % 32 == 0`
    # (PAD_TO) AND `k >= 128` (K_MIN). The floor is not decoration — it is the
    # whole fix for `Linear[6, 256]`, whose K=32 is a perfectly good multiple
    # of 32 and still lands on cuBLAS. Measured at [960 x K] @ [K x 256]:
    #
    #     K =  32 / 64 / 96   288 / 285 / 286 us   cutlass   ALLOCATES
    #     K = 128 / 160 / 192   8.6 / 10.1 / 11.9 us  multistage  free
    #
    # ⚠ THAT LAST PARENTHETICAL WAS RIGHT ABOUT THE FORWARD AND WRONG ABOUT
    # THE LAYER, AND IT COST A 3-7x. `K_PAD` is used on TWO axes:
    #
    #     forward      y_pad[B, N_PAD]  = x_pad[B, K_PAD] @ w_pad
    #                  K = K_PAD  -> the FLOOR test, satisfied by K_MIN
    #     grad_input   gi_pad[B, K_PAD] = go_pad[B, N_PAD] @ w_padᵀ
    #                  N = K_PAD  -> the MODULUS test, NOT satisfied by K_MIN
    #     grad_weight  dW_pad[K_PAD, N_PAD] = cT_pad @ go_pad
    #                  M = K_PAD  -> untested; M only sets the tile grid
    #
    # So 160 and 192 are free in the FORWARD and land on cuBLAS in the
    # BACKWARD, along with TD-MPC2's 518 -> 544. `PAD_TO = 128` makes K_PAD a
    # multiple of 128, which satisfies both tests at once and subsumes K_MIN.
    # Measured (`benchmarks/bench_linear_kpad_modulus.mojo`, 5090, all three
    # GEMMs of one layer, floor-128 -> modulus-128):
    #
    #     IN_=518 B=256   450.7 -> 100.6 us   4.48x   grad_input 392 -> 36 us
    #     IN_=518 B=960   473.8 -> 145.1 us   3.27x
    #     IN_=160 B=256   415.7 ->  59.6 us   6.98x
    #     IN_=192 B=960   470.1 -> 104.3 us   4.51x
    #     three controls (K_PAD already a 128-multiple)   0.93x .. 1.07x
    #
    # The forward does get SLOWER (38.2 -> 44.2 us at 544 -> 640, +16% of K);
    # it is repaid ~10x over by grad_input leaving the 32MB-per-call vendor
    # path. The controls bracket the noise at +/-7.4%, so the smallest real
    # win is 3x outside it.
    #
    # ⚠ ONE PADDING CONSTANT, TWO GEMMS, TWO DIFFERENT TESTS — and only one
    # was checked. The same shape of miss put `Conv2D`'s forward on MAX's
    # allocating split-K path (2e67eb83) and left `LinearAct` with no K floor
    # at all. Grep for every GEMM a pad constant reaches before trusting it.
    #
    # ⚠ NVIDIA, since the backward moved to cuBLAS (`cublas_gemm`): neither
    # backward GEMM reads K_PAD any more — they run on the unpadded operands —
    # so only the FORWARD's terms remain: k % 32 == 0 and k >= 128. PAD_TO is
    # 32 there (a 192-wide ViT layer is no longer copied to 256 every
    # forward). Apple keeps 128: its backward still runs on the padded shapes.
    comptime PAD_TO = 32 if has_nvidia_gpu_accelerator() else 128
    comptime K_MIN = 128
    comptime K_PAD = Self._round_up(Self.IN_, Self.PAD_TO) if Self._round_up(
        Self.IN_, Self.PAD_TO
    ) > Self.K_MIN else Self.K_MIN
    comptime NEEDS_PAD = Self.K_PAD != Self.IN_

    # ── N-alignment padding — a DIFFERENT defect from the K one ──────────
    # A narrow, unaligned OUTPUT width makes cuBLAS pick a split-K path, and
    # split-K needs a workspace that is allocated, memset and freed on EVERY
    # call. Measured on an RTX 5090 with `benchmarks/bench_matmul_alloc_probe.mojo`
    # at [2144x512]@[512xN]:
    #
    #     N=101   cutlass + splitKreduce   9.7 us of GPU work   211.7 us wall
    #     N=128   multistage_gemm         24.8 us of GPU work    25.5 us wall
    #
    # N=101 does LESS GPU work and costs 8.3x more wall-clock; ~202 us of it is
    # cuMemAlloc (87 us) + a 33.5 MB memset + cuMemFree (96 us). Across a walker
    # profile that was 21,464 alloc/free pairs = 67% of ALL CUDA API time, more
    # than the entire GPU kernel time of the same run, and 67 GB of memset.
    # `N = BINS = 101` feeds the reward head, all five Q heads and termination.
    #
    # ⚠ THIS IS NOT THE K FIX. Zero-padding K is invisible — the extra columns
    # contribute exactly 0, which is why that change came out bit-identical.
    # Padding N produces a WIDER OUTPUT, so `OUT_DIM` must keep its logical
    # value and the result has to be sliced back. The slice rides on the bias
    # add (`_bias_add_slice_kernel`), so it costs no extra launch.
    #
    # ⚠ NVIDIA-only in effect: Metal has no split-K path and measures N=101 and
    # N=128 within 3%. The padding is harmless there (a few extra MFLOPs).
    #
    # ⚠⚠ ALIGNMENT IS NOT THE CRITERION — WIDTH IS. This rounding is NOT
    # sufficient for a small `OUT_`, and the comment above overstates what it
    # buys. Measured on an RTX 5090 by `bench_matmul_alloc_act_shapes.mojo`,
    # same M and same K, only N changing:
    #
    #     [960 x 256] @ [256 x  32]   291.60 us   cutlass      ALLOCATES
    #     [960 x 256] @ [256 x 256]    14.87 us   multistage   free
    #
    # **N = 32 is a multiple of 32 and still allocates.** 8x the FLOPs at
    # N=256 and 19.6x FASTER, the difference being one cuMemAlloc + memset +
    # cuMemFree per call. N=101 -> 128 fixed TD-MPC2's head because 128 is WIDE
    # ENOUGH, not because it is aligned. So ACT's `ahat` (`Linear[256, 6]`,
    # padded to 32) and `latent_proj` (N = 64) are BOTH still on the allocating
    # path under this rule.
    #
    # ANSWERED — and it is not a threshold, it is a line of MAX source.
    # `matmul/gpu/__init__.mojo:591` gates its own (non-allocating) kernel on
    #
    #     multi_gemm_cond = m > 1 and n % 128 == 0 and k % 32 == 0 and k >= 128
    #
    # (on a 5090; the H100/AMD disjuncts are False). Fail it and control falls
    # through to the VENDOR BLAS fallback — cuBLAS, a cutlass kernel, and a
    # workspace allocated + memset + freed per call. 29/29 against measurement.
    #
    # So the two axes obey DIFFERENT tests, which is why one "pad to X" rule
    # never fit: N wants `% 128`, K wants `% 32` AND a `>= 128` FLOOR. Both
    # constants here are therefore half right — `PAD_TO = 32` is exactly the K
    # term (TD-MPC2's 518 -> 544 was this line, not a tensor-core cliff), and
    # N=101 -> 128 worked because 128 % 128 == 0, not because 128 is wide.
    #
    # Both terms are now honoured below: `K_MIN = 128` and `N_PAD_TO = 128`.
    #
    # ⚠ The pad is UNCONDITIONAL, and that is a judgement about scale, not an
    # oversight. It costs FLOPs, and it only pays while the extra work is
    # cheaper than the ~277 us the allocator charges per call. Break-even for a
    # 3x FLOP increase is ~4.6 GFLOP unpadded, i.e. M around 560,000 for a
    # 64x64 `Linear` — two orders beyond anything here (ACT's widest is
    # M = 2592, TD-MPC2's 2144). A `Linear` fed a genuinely huge M with tiny
    # IN_/OUT_ would want a comptime opt-out; none exists in this repo today.
    #
    # ⚠ These are HARDCODED MAX CONSTANTS, not hardware properties — the source
    # says "Hard coded this condition to 128 for now. TODO: Need to find a
    # better dispatch strategy." This padding is pinned to a MAX VERSION
    # (v26.5), not to a GPU. Re-run `benchmarks/bench_matmul_alloc_threshold`
    # after a MAX upgrade; if that TODO ever lands, the right pad may be none.
    # The K terms carry no device check at all, so this is not 5090-specific;
    # only H100-with-bfloat16 (`n % 8`) and AMD (`n % 4`) relax the N term, and
    # B200 never reaches this gate.
    #
    # ⚠ 128, NOT `PAD_TO`. The gate's N term is `n % 128 == 0`, so rounding to
    # 32 lands on the ALLOCATING side — which is what `Linear[256, 6]` (padded
    # to 32) was doing. Measured at [960 x 256] @ [256 x N]:
    #
    #     N = 32 / 48 / 64 / 96 / 160 / 192   ~285-293 us   cutlass  ALLOCATES
    #     N = 128 / 256                       14.3 / 14.9 us  multistage  free
    #
    # N=160 and N=192 are WIDER than 128 and still allocate: this is a
    # modulus, not a threshold.
    comptime N_PAD_TO = 128
    comptime N_PAD = Self._round_up(Self.OUT_, Self.N_PAD_TO)
    comptime NEEDS_N_PAD = Self.N_PAD != Self.OUT_
    # NVIDIA fp32: the forward GEMM runs on `cublas_gemm` (unpadded, one GEMM
    # + one bias kernel) except in the three regimes where MAX's measured
    # faster (`cublas_fwd`: batch 1 on an aligned shape, batch >= 16384 with a
    # mid-size K, a 2^25-element weight at batch <= 32), which keep the MAX
    # paths below. Where MAX needs padding its multistage GEMM runs a 128-wide
    # tile over what is, for an RL trunk, an 8- or 64-wide layer (8 -> 64 does
    # 32x the useful work), plus the x_pad / w_pad copies. `NN_GEMM_PATH`
    # (`cublas_gemm.mojo`) overrides for A/B runs.
    @staticmethod
    def use_cublas_fwd[B: Int]() -> Bool:
        return cublas_fwd(B, Self.OUT_, Self.IN_, Self.NEEDS_PAD or Self.NEEDS_N_PAD)
    comptime WPAD_SIZE = Self.K_PAD * Self.N_PAD

    @staticmethod
    def _round_up(v: Int, to: Int) -> Int:
        return ((v + to - 1) // to) * to

    # Activation-flow dtype (satisfies the Module trait). `Linear[IN, OUT]` =
    # fp32 (ACT_DT == DT, the legacy path); `Linear[IN, OUT, bfloat16]` flows
    # activations at bf16 (the AMP "Step B" memory win).
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return String("Linear")

    comptime B_SIZE = Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # grad_w scratch (lazy): cacheᵀ [IN, B] + dW_tmp [IN, OUT] for the
    # transpose + max_matmul + accumulate path (max_matmul rejects transpose_a).
    # Both STAY fp32 (master grad is fp32; the bf16 dW GEMM writes a fp32 output).
    var cacheT: Tensor
    var dW_tmp: Tensor
    # bf16-flow compute scratch (lazy; used only when ACT_DT == bf16 and
    # target == "gpu"). Master weights/grads/bias stay fp32 (`Param`); only these
    # low-precision copies are bf16. `w_bf` is the CACHED bf16 weight: recast from
    # `weight.val` only when the optimizer bumped `weight.val.version` since the
    # last cast (tracked by `_w_cast_version`), so the W cast happens ONCE per
    # optimizer step rather than every forward. `b_a` is the cached bf16 bias (the
    # tiny per-forward bias cast). `cacheT_bf` is the transposed bf16 fwd-input
    # (backward grad_w). No `x_bf`/`o_bf`/`go_bf`: activations ALREADY flow at bf16
    # (no input/output/grad_output cast — the whole point of bf16-flow).
    var w_bf: TensorImpl[Self.ADT]
    var b_a: TensorImpl[Self.ADT]  # cached bf16 bias (forward bias-add)
    var cacheT_bf: TensorImpl[Self.ADT]  # transposed-x bf16 (backward grad_w)
    var _w_cast_version: Int  # `weight.val.version` at last bf16 weight cast
    # K-alignment scratch (lazy; GPU fp32 forward only, and only when
    # `NEEDS_PAD`). `w_pad` is the zero-tailed [K_PAD, OUT_] weight, re-padded
    # only when the optimizer bumped `weight.val.version` — once per optimizer
    # step, not per forward, exactly like `w_bf` above. `x_pad` is the
    # zero-tailed [B, K_PAD] activation, which DOES have to be rebuilt every
    # forward because the input is a fresh upstream tensor each time.
    var w_pad: Tensor
    var x_pad: Tensor
    var y_pad: Tensor
    """[B, N_PAD] GEMM destination when the OUTPUT width is padded; sliced back
    into `out` by the fused bias add. Unused when `NEEDS_N_PAD` is False."""
    # ── vjp scratch, same padding rationale as the forward ────────────────
    # The BACKWARD GEMMs are unaligned wherever the forward's were, on the
    # opposite axes: `grad_w = xᵀ @ go` has N = OUT_, and `grad_input =
    # go @ Wᵀ` has K = OUT_ and N = IN_. On an RTX 5090 that put 12,200 cutlass
    # `..._tn_align1` / `..._nn_align1` kernels on a workspace-allocating path —
    # 12,500 cuMemAlloc + 33 MB memset + cuMemFree pairs, one per call.
    var go_pad: Tensor
    """[B, N_PAD] zero-padded grad_output — feeds BOTH backward GEMMs."""
    var cT_pad: Tensor
    """[K_PAD, B] zero-row-padded transpose of the forward input."""
    var dW_pad: Tensor
    """[K_PAD, N_PAD] padded grad_w, accumulated into the master grad with a
    STRIDED add (`_accum_2d_kernel`) because its row stride is N_PAD."""
    var gi_pad: Tensor
    """[B, K_PAD] padded grad_input, sliced back to [B, IN_]."""
    var gb_part: Tensor
    """Bias-gradient chunk partials (`enqueue_bias_grad`), [B / 1024, OUT_]."""
    var sk_ws: Tensor
    """Split-K reduction workspace for the dW GEMM, `[P, K_PAD, N_PAD]`.

    `linalg.matmul` allocates this per call and frees it again
    (`matmul/gpu/__init__.mojo:1845`/`:1915`), which is a cuMemAlloc/cuMemFree
    pair per GEMM and — being a SYNCHRONOUS driver allocation — makes any step
    containing a split-K GEMM impossible to capture into a CUDA graph. Owning
    it here removes both. It lives on the Module deliberately: a CUDA graph
    holds RAW POINTERS to every operand, so the workspace has to outlive the
    LAST REPLAY, and the Module outlives the trainer's graph. Sized by
    `ensure_gpu` on an eager step; it must never grow inside a capture region."""
    var _sk_p: Int
    """Cached split-K partition count for the dW GEMM. -1 = not yet decided,
    1 = do not split (also what `NOEIRA_SPLITK=0` forces).

    Decided ONCE and cached because it sets `grid_dim`, which is baked into a
    captured graph — a P that drifted between replays would be a wrong launch."""
    var _w_pad_version: Int
    # Capture mode (set via `set_attr["capture_recast"]`): when True, the bf16
    # weight recast in `_ensure_w_bf` is UNCONDITIONAL so the cast kernel is
    # always recorded into a CUDA graph and reads the live fp32 master on every
    # replay — the version gate would skip it on replay and serve STALE weights.
    # Off → the version-gated fast path (one cast per optimizer step).
    var _force_recast: Bool

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.cacheT = Tensor()
        self.dW_tmp = Tensor()
        self.w_bf = TensorImpl[Self.ADT]()
        self.b_a = TensorImpl[Self.ADT]()
        self.cacheT_bf = TensorImpl[Self.ADT]()
        self._w_cast_version = -1  # < any real version → first forward casts
        self._force_recast = False
        self.w_pad = Tensor()
        self.x_pad = Tensor()
        self.y_pad = Tensor()
        self.go_pad = Tensor()
        self.cT_pad = Tensor()
        self.dW_pad = Tensor()
        self.sk_ws = Tensor()
        self._sk_p = -1
        self.gi_pad = Tensor()
        self.gb_part = Tensor()
        self._w_pad_version = -1  # < any real version → first forward pads

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "capture_recast":
            self._force_recast = value != Scalar[DT](0.0)

    def _decide_sk_p(mut self, B: Int, ctx: DeviceContext) raises:
        """Decide the dW GEMM's partition count, once, and cache it.

        Called on the first backward — an EAGER step, before any capture — and
        never again, because `grid_dim` derives from P and is baked into a
        captured graph. A P that drifted between replays would be a wrong
        launch, not a slow one.

        The dW GEMM's M and N are the dims of the branch that will run: the
        PADDED site uses [K_PAD, N_PAD], the unpadded one [IN_, OUT_]. They
        coincide unless padding is active, and `select_config` is sensitive to
        both, so pick them the same way the launch will.
        """
        comptime M_DW = Self.K_PAD if Self.NEEDS_PAD or Self.NEEDS_N_PAD else Self.IN_
        comptime N_DW = Self.N_PAD if Self.NEEDS_PAD or Self.NEEDS_N_PAD else Self.OUT_
        self._sk_p = decide_partitions(M_DW, N_DW, B, ctx)
        if self._sk_p > 1:
            # Size the workspace now, on an eager step. It can never grow
            # inside a capture region, and P is fixed from here.
            self.sk_ws.ensure_gpu(ctx, self._sk_p * M_DW * N_DW)

    # NOTE: `dispatch_splitk_gemm` is a FREE function, not a method. `dWp_v`
    # is a view borrowed from `self.dW_pad`, so passing it alongside `mut self`
    # aliases a mutable borrow of the whole struct with a reference embedded in
    # one of its fields, and the borrow checker rejects it. Handing the
    # workspace over explicitly keeps the two borrows on separate fields.

    def _ensure_w_pad(mut self, c: DeviceContext) raises:
        """Ensure `w_pad` is `weight.val` with `K_PAD - IN_` rows of zeros
        appended. Re-pads ONLY when the optimizer bumped `val.version` since the
        last pad, so training pays it once per step and inference once ever.

        ⚠ `_force_recast` (CUDA-graph capture) must make this UNCONDITIONAL for
        the same reason it does for the bf16 cast: the version gate would skip
        the pad on replay and the GEMM would read a STALE weight.
        """
        self.w_pad.ensure_gpu(c, Self.WPAD_SIZE)
        if self._force_recast or self.weight.val.version != self._w_pad_version:
            # [IN_, OUT_] -> [K_PAD, N_PAD]. This must be the 2-D pad, not a
            # flat tail copy: once N is padded the row STRIDE changes from
            # `OUT_` to `N_PAD`, so every row moves, not just the appended ones.
            c.enqueue_function[_pad_2d_kernel](
                self.weight.val.dev.value(),
                self.w_pad.dev.value(),
                Int64(Self.IN_),
                Int64(Self.OUT_),
                Int64(Self.K_PAD),
                Int64(Self.N_PAD),
                grid_dim=(Self.WPAD_SIZE + 255) // 256,
                block_dim=256,
            )
            self._w_pad_version = self.weight.val.version

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Ensure the cached bf16 weight `w_bf` reflects the current fp32
        `weight.val`. Recasts ONLY when the optimizer bumped `val.version` since
        the last cast (so the weight cast is ONCE per step, not per fwd/bwd).
        Shared by forward (the cast) and vjp (which REUSES it — no optimizer step
        intervenes between a fwd and its bwd, so the forward's cast is valid).
        """
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        if (
            self._force_recast
            or self.weight.val.version != self._w_cast_version
        ):
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
        var l = Self()
        l.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        l.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        INIT.init_weight[target](
            l.weight.val, Self.W_SIZE, Self.IN_, Self.OUT_, ctx
        )
        INIT.init_bias[target](l.bias.val, Self.B_SIZE, ctx)
        return l^

    def release_buffers(mut self):
        """The forward / vjp activations and scratch: `cacheT` (the input,
        transposed, for grad_w), its bf16 twin, and the padded copies
        `x_pad` / `y_pad` / `go_pad` / `cT_pad` / `gi_pad`. Kept: the params,
        the version-gated weight caches (`w_pad`, `w_bf`, `b_a`), the
        weight-sized `dW_pad` / `dW_tmp`, and `sk_ws`, sized ONCE with the
        split-K decision."""
        self.cacheT.release()
        self.cacheT_bf.release()
        self.x_pad.release()
        self.y_pad.release()
        self.go_pad.release()
        self.cT_pad.release()
        self.gi_pad.release()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        # y = x @ W (max_matmul), then += bias.
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT in this branch, but the compiler doesn't collapse the
            # opaque `Self.ACT_DT` parameter to `DT` for type-unification against
            # the fp32 weight/bias views — so rebind the activation refs (sound:
            # the dtypes are equal here). `TensorImpl[Self.ACT_DT]` ≡ `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.OUT_)
                # ── Apple fp32: ONE cblas_sgemm, as `vjp` already does ──────
                # `max_matmul[target="cpu"]` collapses at small M. Measured on
                # an M1 Pro:
                #
                #     [1x512]@[512x512]   max_matmul 300.8 us (1.7 GFLOPS)
                #                         cblas        7.8 us ( 67 GFLOPS)
                #
                # — 38x, and B=1 is exactly the single-env acting path. At the
                # MPPI batch the two are comparable ([268x518]@[518x512]:
                # 146.8 vs 155.8 us), so this is a pure win at small B and a
                # wash at large B. `vjp` has called cblas directly on this path
                # since it was written (see `IS_APPLE_F32` below); only
                # `forward` was left on the generic dispatch.
                comptime IS_APPLE_F32 = (
                    CompilationTarget.is_macos() and DT == DType.float32
                )
                comptime if IS_APPLE_F32:
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.NO_TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(B),  # M
                        Int32(Self.OUT_),  # N
                        Int32(Self.IN_),  # K
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            in0d.data.unsafe_ptr()
                        ),
                        Int32(Self.IN_),  # lda
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            self.weight.val.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),  # ldb
                        Float32(0.0),  # beta: overwrite, bias added below
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            outd.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),  # ldc
                    )
                else:
                    var x_v = TileTensor(in0d.data, row_major[B, Self.IN_]())
                    var w_v = TileTensor(
                        self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
                    )
                    var out_v = TileTensor(outd.data, row_major[B, Self.OUT_]())
                    max_matmul[target="cpu"](out_v, x_v, w_v, None)
                # bias add, SIMD over the row (the scalar double loop was part
                # of the non-GEMM cost that dominates CPU trunks).
                var op = outd.data.unsafe_ptr()
                var bp = self.bias.val.data.unsafe_ptr()
                for b in range(B):
                    var row = b * Self.OUT_
                    var j = 0
                    while j + CPU_SIMD_W <= Self.OUT_:
                        op.unsafe_store(
                            row + j,
                            op.unsafe_load[width=CPU_SIMD_W](row + j)
                            + bp.unsafe_load[width=CPU_SIMD_W](j),
                        )
                        j += CPU_SIMD_W
                    while j < Self.OUT_:
                        op[unsafe_offset=row + j] = (
                            op[unsafe_offset=row + j] + bp[unsafe_offset=j]
                        )
                        j += 1
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.OUT_)
                var bl = self.bias.val.dev.value()
                # Zero-padding K leaves the dot products EXACTLY unchanged (the
                # appended columns are 0); only the GEMM's tiling — and hence
                # its fp32 reduction ORDER — moves, which can shift a result by
                # an ulp. Padding N adds columns nothing ever reads.
                comptime if Self.use_cublas_fwd[B]():
                    cublas_gemm[False, False, cublas_tf32(B, Self.OUT_, Self.IN_)](
                        c, outd.dev.value(), in0d.dev.value(),
                        self.weight.val.dev.value(),
                        B, Self.OUT_, Self.IN_, 0.0,
                    )
                    c.enqueue_function[_bias_add_kernel[DT]](
                        outd.dev.value(),
                        bl,
                        Int64(B),
                        Int64(Self.OUT_),
                        grid_dim=(B * Self.OUT_ + 255) // 256,
                        block_dim=256,
                    )
                elif Self.NEEDS_PAD or Self.NEEDS_N_PAD:
                    self._ensure_w_pad(c)
                    # The activation only needs a copy when K is padded; when
                    # only N is padded, K_PAD == IN_ and `in0d` is already the
                    # right shape.
                    comptime if Self.NEEDS_PAD:
                        self.x_pad.ensure_gpu(c, B * Self.K_PAD)
                        c.enqueue_function[_pad_cols_kernel](
                            in0d.dev.value(),
                            self.x_pad.dev.value(),
                            Int64(B),
                            Int64(Self.IN_),
                            Int64(Self.K_PAD),
                            grid_dim=(B * Self.K_PAD + 255) // 256,
                            block_dim=256,
                        )
                    comptime if Self.NEEDS_N_PAD:
                        # GEMM into the widened destination, then slice back to
                        # `OUT_` — fused with the bias add, so no extra launch.
                        self.y_pad.ensure_gpu(c, B * Self.N_PAD)
                        mm[A0=B, A1=Self.K_PAD, B0=Self.K_PAD, B1=Self.N_PAD, O0=B, O1=Self.N_PAD](
                            self.y_pad.dev.value(), self.x_pad.dev.value() if Self.NEEDS_PAD else in0d.dev.value(), self.w_pad.dev.value(), c
                        )
                        c.enqueue_function[_bias_add_slice_kernel](
                            self.y_pad.dev.value(),
                            bl,
                            outd.dev.value(),
                            Int64(B),
                            Int64(Self.OUT_),
                            Int64(Self.N_PAD),
                            grid_dim=(B * Self.OUT_ + 255) // 256,
                            block_dim=256,
                        )
                    else:
                        comptime if has_nvidia_gpu_accelerator():
                            # Bias in the GEMM epilogue (`mm_bias`): one launch.
                            mm_bias[A0=B, A1=Self.K_PAD, B0=Self.K_PAD, B1=Self.N_PAD, O0=B, O1=Self.OUT_](
                                outd.dev.value(), self.x_pad.dev.value() if Self.NEEDS_PAD else in0d.dev.value(), self.w_pad.dev.value(), bl, c
                            )
                        else:
                            mm[A0=B, A1=Self.K_PAD, B0=Self.K_PAD, B1=Self.N_PAD, O0=B, O1=Self.OUT_](
                                outd.dev.value(), self.x_pad.dev.value() if Self.NEEDS_PAD else in0d.dev.value(), self.w_pad.dev.value(), c
                            )
                            c.enqueue_function[_bias_add_kernel[DT]](
                                outd.dev.value(),
                                bl,
                                Int64(B),
                                Int64(Self.OUT_),
                                grid_dim=(B * Self.OUT_ + 255) // 256,
                                block_dim=256,
                            )
                else:
                    comptime if has_nvidia_gpu_accelerator():
                        mm_bias[A0=B, A1=Self.IN_, B0=Self.IN_, B1=Self.OUT_, O0=B, O1=Self.OUT_](
                            outd.dev.value(), in0d.dev.value(),
                            self.weight.val.dev.value(), bl, c,
                        )
                    else:
                        _gemm_bkn[B, Self.IN_, Self.OUT_](
                            outd.dev.value(),
                            in0d.dev.value(),
                            self.weight.val.dev.value(),
                            c,
                        )
                        c.enqueue_function[_bias_add_kernel[DT]](
                            outd.dev.value(),
                            bl,
                            Int64(B),
                            Int64(Self.OUT_),
                            grid_dim=(B * Self.OUT_ + 255) // 256,
                            block_dim=256,
                        )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert target == "gpu", "bf16-flow Linear is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
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
            # bf16-in → bf16-out GEMM (fp32 accumulation is automatic).
            mm[A0=B, A1=Self.IN_, B0=Self.IN_, B1=Self.OUT_, O0=B, O1=Self.OUT_](
                out.dev.value(), in0.dev.value(), self.w_bf.dev.value(), c
            )
            var ol = out.dev.value()
            var bl = self.b_a.dev.value()
            c.enqueue_function[_bias_add_kernel[Self.ADT]](
                ol,
                bl,
                Int64(B),
                Int64(Self.OUT_),
                grid_dim=(B * Self.OUT_ + 255) // 256,
                block_dim=256,
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
                gind.ensure(B * Self.IN_)
                self.cacheT.ensure(Self.IN_ * B)
                self.dW_tmp.ensure(Self.W_SIZE)
                var x_v = TileTensor(find.data, row_major[B, Self.IN_]())
                var go_v = TileTensor(god.data, row_major[B, Self.OUT_]())
                var gi_v = TileTensor(gind.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
                )
                var gw_v = TileTensor(
                    self.weight.grd.data, row_major[Self.IN_, Self.OUT_]()
                )
                var gb_v = TileTensor(
                    self.bias.grd.data, row_major[Self.OUT_]()
                )
                # grad_b += colsum(go)
                for b in range(B):
                    for j in range(Self.OUT_):
                        gb_v[j] += go_v[b, j]
                # grad_w += xᵀ @ go. Apple-fp32: ONE fused cblas_sgemm (TRANSPOSE
                # A=x, beta=1 → no transpose buffer, no temp, no accumulate loop —
                # matches legacy Linear). Else: transpose into cacheT + max_matmul
                # + add.
                comptime IS_APPLE_F32 = (
                    CompilationTarget.is_macos() and DT == DType.float32
                )
                comptime if IS_APPLE_F32:
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.IN_),
                        Int32(Self.OUT_),
                        Int32(B),
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            find.data.unsafe_ptr()
                        ),
                        Int32(Self.IN_),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            god.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                        Float32(1.0),
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            self.weight.grd.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                    )
                else:
                    var cT_v = TileTensor(
                        self.cacheT.data, row_major[Self.IN_, B]()
                    )
                    for b in range(B):
                        for k in range(Self.IN_):
                            cT_v[k, b] = x_v[b, k]
                    var dW_v = TileTensor(
                        self.dW_tmp.data, row_major[Self.IN_, Self.OUT_]()
                    )
                    max_matmul[target="cpu"](dW_v, cT_v, go_v, None)
                    for k in range(Self.IN_):
                        for j in range(Self.OUT_):
                            gw_v[k, j] += dW_v[k, j]
                # grad_input = go @ Wᵀ
                max_matmul[transpose_b=True, target="cpu"](
                    gi_v, go_v, w_v, None
                )
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, B * Self.IN_)
                comptime if CUBLAS_BWD:
                    # NVIDIA: both GEMMs through cuBLAS on the UNPADDED
                    # operands (`cublas_gemm`): grad_w += xᵀ @ go (β = 1, no
                    # transposed copy, no temporary, no accumulate kernel) and
                    # grad_input = go @ Wᵀ. cuBLAS takes any shape without the
                    # per-call workspace MAX's fallback allocates, so the
                    # backward needs none of the K / N padding below.
                    var gol = god.dev.value()
                    var gbl = self.bias.grd.dev.value()
                    enqueue_bias_grad[DT](c, gol, gbl, B, Self.OUT_, self.gb_part)
                    cublas_gemm[True, False, cublas_tf32(B, Self.OUT_, Self.IN_)](
                        c, self.weight.grd.dev.value(), find.dev.value(),
                        god.dev.value(),
                        Self.IN_, Self.OUT_, B, 1.0,
                    )
                    cublas_gemm[False, True, cublas_tf32(B, Self.OUT_, Self.IN_)](
                        c, gind.dev.value(), god.dev.value(),
                        self.weight.val.dev.value(),
                        B, Self.IN_, Self.OUT_, 0.0,
                    )
                else:
                    self.cacheT.ensure_gpu(c, Self.IN_ * B)
                    self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
                    # grad_b += colsum(go)
                    var gol = god.dev.value()
                    var gbl = self.bias.grd.dev.value()
                    enqueue_bias_grad[DT](c, gol, gbl, B, Self.OUT_, self.gb_part)
                    # grad_w += cacheᵀ @ go: transpose x → cacheT (B1' tiled, fp32),
                    # then the two GEMMs (grad_w + grad_x), then the grad_w accumulate.
                    var xl = find.dev.value()
                    var cTl = self.cacheT.dev.value()
                    c.enqueue_function[_transpose_tiled_kernel[DT]](
                        xl,
                        cTl,
                        Int64(B),
                        Int64(Self.IN_),
                        grid_dim=(
                            (Self.IN_ + _T_TILE - 1) // _T_TILE,
                            (B + _T_TILE - 1) // _T_TILE,
                        ),
                        block_dim=(_T_TILE, _T_BR),
                    )
                    comptime if Self.NEEDS_PAD or Self.NEEDS_N_PAD:
                        # Both backward GEMMs run on the PADDED shapes, reusing the
                        # forward's `w_pad` ([K_PAD, N_PAD]) so no extra weight copy
                        # is needed. The zero tails contribute exactly 0 to every
                        # dot product, so the gradients are unchanged up to fp32
                        # reduction order.
                        self._ensure_w_pad(c)
                        # go: [B, OUT_] -> [B, N_PAD]
                        comptime if Self.NEEDS_N_PAD:
                            self.go_pad.ensure_gpu(c, B * Self.N_PAD)
                            c.enqueue_function[_pad_cols_kernel](
                                god.dev.value(),
                                self.go_pad.dev.value(),
                                Int64(B),
                                Int64(Self.OUT_),
                                Int64(Self.N_PAD),
                                grid_dim=(B * Self.N_PAD + 255) // 256,
                                block_dim=256,
                            )
                        var gop_v = TileTensor(
                            self.go_pad.dev.value() if Self.NEEDS_N_PAD else god.dev.value(),
                            row_major[B, Self.N_PAD](),
                        )
                        # cacheT: [IN_, B] -> [K_PAD, B]  (append zero ROWS)
                        comptime if Self.NEEDS_PAD:
                            self.cT_pad.ensure_gpu(c, Self.K_PAD * B)
                            c.enqueue_function[_pad_2d_kernel](
                                self.cacheT.dev.value(),
                                self.cT_pad.dev.value(),
                                Int64(Self.IN_),
                                Int64(B),
                                Int64(Self.K_PAD),
                                Int64(B),
                                grid_dim=(Self.K_PAD * B + 255) // 256,
                                block_dim=256,
                            )
                        var cTp_v = TileTensor(
                            self.cT_pad.dev.value() if Self.NEEDS_PAD else self.cacheT.dev.value(),
                            row_major[Self.K_PAD, B](),
                        )
                        # grad_w = cacheTᵀ @ go   ->  [K_PAD, N_PAD]
                        self.dW_pad.ensure_gpu(c, Self.WPAD_SIZE)
                        var dWp_v = TileTensor(
                            self.dW_pad.dev.value(),
                            row_major[Self.K_PAD, Self.N_PAD](),
                        )
                        # ── grad_w: split-K on OUR workspace, or plain matmul ──
                        # This is the GEMM MODULAR_MATMUL_ALLOC_REPORT.md
                        # Measurement 4 caught allocating per call: K is
                        # `batch * tokens`, so it is long-K and skinny by
                        # construction, which is exactly when `select_config`
                        # partitions K. Routing it through our own workspace
                        # removes the alloc/free pair AND unblocks CUDA-graph
                        # capture of the whole step.
                        #
                        # Inert unless MAX's own dispatch would have reached
                        # `multistage_gemm` with a partitioned K (see
                        # `splitk_path_applies`: not H100, not sm_100 Blackwell,
                        # not AMD/Apple). Otherwise this is byte-for-byte the
                        # previous behaviour.
                        comptime if splitk_path_applies[c.default_device_info]():
                            if self._sk_p < 0:
                                self._decide_sk_p(B, c)
                            if self._sk_p > 1:
                                dispatch_splitk_gemm(
                                    dWp_v,
                                    cTp_v,
                                    gop_v,
                                    Self.K_PAD,
                                    Self.N_PAD,
                                    B,
                                    self._sk_p,
                                    self.sk_ws,
                                    c,
                                )
                            else:
                                mm[A0=Self.K_PAD, A1=B, B0=B, B1=Self.N_PAD, O0=Self.K_PAD, O1=Self.N_PAD](
                                    self.dW_pad.dev.value(), self.cT_pad.dev.value() if Self.NEEDS_PAD else self.cacheT.dev.value(), self.go_pad.dev.value() if Self.NEEDS_N_PAD else god.dev.value(), c
                                )
                        else:
                            mm[A0=Self.K_PAD, A1=B, B0=B, B1=Self.N_PAD, O0=Self.K_PAD, O1=Self.N_PAD](
                                    self.dW_pad.dev.value(), self.cT_pad.dev.value() if Self.NEEDS_PAD else self.cacheT.dev.value(), self.go_pad.dev.value() if Self.NEEDS_N_PAD else god.dev.value(), c
                                )
                        # grad_input = go @ w_padᵀ  ->  [B, K_PAD]
                        # ⚠ The GEMM DESTINATION must be a mutable view, so this
                        # cannot use the `... if NEEDS_PAD else ...` form the
                        # read-only operands above use — a ternary yields an
                        # immutable origin and `max_matmul`'s `c` rejects it.
                        comptime if Self.NEEDS_PAD:
                            self.gi_pad.ensure_gpu(c, B * Self.K_PAD)
                            mm[transpose_b=True, A0=B, A1=Self.N_PAD, B0=Self.K_PAD, B1=Self.N_PAD, O0=B, O1=Self.K_PAD](
                                self.gi_pad.dev.value(), self.go_pad.dev.value() if Self.NEEDS_N_PAD else god.dev.value(), self.w_pad.dev.value(), c
                            )
                            c.enqueue_function[_slice_cols_kernel](
                                self.gi_pad.dev.value(),
                                gind.dev.value(),
                                Int64(B),
                                Int64(Self.IN_),
                                Int64(Self.K_PAD),
                                grid_dim=(B * Self.IN_ + 255) // 256,
                                block_dim=256,
                            )
                        else:
                            # K_PAD == IN_ here, so this writes `gind` directly.
                            mm[transpose_b=True, A0=B, A1=Self.N_PAD, B0=Self.K_PAD, B1=Self.N_PAD, O0=B, O1=Self.K_PAD](
                                gind.dev.value(), self.go_pad.dev.value() if Self.NEEDS_N_PAD else god.dev.value(), self.w_pad.dev.value(), c
                            )
                        # ⚠ STRIDED accumulate: dW_pad's row stride is N_PAD, the
                        # master grad's is OUT_. A flat `_accum_kernel` would fold
                        # the padded columns into the next row's gradient.
                        c.enqueue_function[_accum_2d_kernel](
                            self.weight.grd.dev.value(),
                            # PREFIX view: dW_pad is [K_PAD, N_PAD] but only its
                            # first IN_ rows carry gradient — the rest correspond to
                            # the zero-padded contraction rows. Row-major makes
                            # those first IN_ rows contiguous from offset 0.
                            self.dW_pad.dev.value(),
                            Int64(Self.IN_),
                            Int64(Self.OUT_),
                            Int64(Self.N_PAD),
                            grid_dim=(Self.W_SIZE + 255) // 256,
                            block_dim=256,
                        )
                    else:
                        var dW_v = TileTensor(
                            self.dW_tmp.dev.value(),
                            row_major[Self.IN_, Self.OUT_](),
                        )
                        var cT_v = TileTensor(
                            self.cacheT.dev.value(), row_major[Self.IN_, B]()
                        )
                        var go_v = TileTensor(
                            god.dev.value(), row_major[B, Self.OUT_]()
                        )
                        # Same split-K routing as the padded branch above. This is
                        # the branch an ALIGNED Linear takes (K_PAD == IN_ and
                        # N_PAD == OUT_), which is most of them — the first version
                        # of this change patched only the padded site and the gate
                        # caught it as `split shapes: 0`.
                        comptime if splitk_path_applies[c.default_device_info]():
                            if self._sk_p < 0:
                                self._decide_sk_p(B, c)
                            if self._sk_p > 1:
                                dispatch_splitk_gemm(
                                    dW_v,
                                    cT_v,
                                    go_v,
                                    Self.IN_,
                                    Self.OUT_,
                                    B,
                                    self._sk_p,
                                    self.sk_ws,
                                    c,
                                )
                            else:
                                mm[A0=Self.IN_, A1=B, B0=B, B1=Self.OUT_, O0=Self.IN_, O1=Self.OUT_](
                                    self.dW_tmp.dev.value(), self.cacheT.dev.value(), god.dev.value(), c
                                )
                        else:
                            mm[A0=Self.IN_, A1=B, B0=B, B1=Self.OUT_, O0=Self.IN_, O1=Self.OUT_](
                                self.dW_tmp.dev.value(), self.cacheT.dev.value(), god.dev.value(), c
                            )
                        mm[transpose_b=True, A0=B, A1=Self.OUT_, B0=Self.IN_, B1=Self.OUT_, O0=B, O1=Self.IN_](
                            gind.dev.value(), god.dev.value(), self.weight.val.dev.value(), c
                        )
                        # grad_w += dW (accumulate into the fp32 master grad)
                        c.enqueue_function[_accum_kernel](
                            self.weight.grd.dev.value(),
                            self.dW_tmp.dev.value(),
                            Int64(Self.W_SIZE),
                            grid_dim=(Self.W_SIZE + 255) // 256,
                            block_dim=256,
                        )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert target == "gpu", "bf16-flow Linear is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            self.cacheT_bf.ensure_gpu(c, Self.IN_ * B)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # grad_b += colsum(go): bf16 go → fp32 master grad (fp32 accumulator).
            var gol = grad_output.dev.value()
            var gbl = self.bias.grd.dev.value()
            enqueue_bias_grad[Self.ADT](c, gol, gbl, B, Self.OUT_, self.gb_part)
            # grad_w += cacheᵀ @ go. `fin`/`go` are ALREADY bf16 (no cast).
            # Transpose the bf16 fwd-input directly → bf16 cacheT_bf, then a
            # bf16-in → FP32-out GEMM into the fp32 dW_tmp, then accumulate into
            # the fp32 master grad. W reuses the forward's cached cast.
            self._ensure_w_bf(c)
            var xl = fin.dev.value()
            var cTl = self.cacheT_bf.dev.value()
            c.enqueue_function[_transpose_tiled_kernel[Self.ADT]](
                xl,
                cTl,
                Int64(B),
                Int64(Self.IN_),
                grid_dim=(
                    (Self.IN_ + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )
            # grad_w = cacheT_bfᵀ-form @ go → fp32 dW (bf16-in, fp32-out).
            # ⚠ NOT routed through split-K, deliberately. bf16 operands with an
            # fp32 output would work the same way, but nothing gates it yet —
            # `tests/nn/test_linear_splitk_dw_gpu.mojo` builds fp32 Linears.
            # Route it when that gate grows a bf16 arm, not before.
            mm[A0=Self.IN_, A1=B, B0=B, B1=Self.OUT_, O0=Self.IN_, O1=Self.OUT_](
                self.dW_tmp.dev.value(), self.cacheT_bf.dev.value(), grad_output.dev.value(), c
            )
            # grad_x = go @ Wᵀ → bf16 gin (bf16-in, bf16-out — gin flows at bf16).
            mm[transpose_b=True, A0=B, A1=Self.OUT_, B0=Self.IN_, B1=Self.OUT_, O0=B, O1=Self.IN_](
                gin.dev.value(), grad_output.dev.value(), self.w_bf.dev.value(), c
            )
            # grad_w += dW (accumulate into the fp32 master grad)
            var gwl = self.weight.grd.dev.value()
            var dWl = self.dW_tmp.dev.value()
            c.enqueue_function[_accum_kernel](
                gwl,
                dWl,
                Int64(Self.W_SIZE),
                grid_dim=(Self.W_SIZE + 255) // 256,
                block_dim=256,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `weight` + `bias` Params).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )
        # ⚠ `polyak_tensor` writes `weight.val` IN PLACE and does NOT bump
        # `val.version` — so both derived weight caches below would keep
        # serving the PRE-SYNC weight forever. That is invisible in a forward
        # numerics test (which never syncs) and shows up only as a target
        # network frozen at its init weights: `tests/deep_agents/
        # test_storage_dqn_gpu_smoke.mojo` went from eval 200 to eval 9.
        # Invalidate both caches so the next forward rebuilds them.
        self._w_pad_version = -1
        self._w_cast_version = -1
