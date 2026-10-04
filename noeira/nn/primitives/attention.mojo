"""ScaledDotProductAttention[DIM, N_HEADS, SEQ_LEN, CAUSAL, USE_MAX_KERNELS].

Multi-head scaled dot-product attention as a single nn leaf, on the STORAGE
surface (transformed from legacy `nn.primitives.attention` — surface-only change;
the QKᵀ/softmax/attn·V kernels + the bmm pack/softmax/unpack glue are carried
over VERBATIM). Input is the per-token concatenated `[Q ‖ K ‖ V]` (each DIM-wide),
laid out per sample as `[all-Q tokens | all-K tokens | all-V tokens]`:

    IN_DIM  = SEQ_LEN * DIM * 3        (offsets: Q@0, K@SEQ·DIM, V@2·SEQ·DIM)
    OUT_DIM = SEQ_LEN * DIM

No params. Caches are leaf-owned `Tensor` fields (output-caching — backward reads
only the cache + grad_output, never the forward input slab; storage surface
passes `forward_input` explicitly but this leaf doesn't need it). The cache holds
[Q | K | V | scores] per sample:

    CACHE_SIZE = 3*SEQ_LEN*DIM + N_HEADS*SEQ_LEN*SEQ_LEN

`head_dim = DIM // N_HEADS`, `scale = 1/sqrt(head_dim)`. `CAUSAL=True` bounds each
query i's key loop to j ≤ i. Softmax computed with the standard max-shift.

GPU path: `USE_MAX_KERNELS=True` (default) → batched-GEMM attention (tensor
cores); `False` → serial per-(b,h) custom kernels. Bit-identical (the flag only
changes speed). CPU path mirrors the bmm forward/backward (BLAS GEMMs + scalar
softmax). Unlike legacy, scratch slabs are separate owned `Tensor` fields (one
buffer per slab) instead of a single pointer-sliced scratch — no `mptr`.
"""

from noeira.nn.core.mm import mm, bmm
from noeira.nn.core.mm_tiled import bmm_tiled
from std.math import exp, sqrt
from max.gpu import thread_idx, block_idx, block_dim, global_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.bmm import batched_matmul

from noeira.nn.constants import DT, TPB
from noeira.nn.random.hash_mask import hash_keep, new_dropout_seed
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — custom per-(b,h) path. One block per (b,h); threads stride
# over rows (fwd / dQ) or (j,d) pairs (dV / dK). Carried VERBATIM from the
# legacy leaf. ALL math is fp32 (DT) — Metal has no Float64; the per-sample
# cache stays fp32. bf16-FLOW (AMP "Step B"): the I/O ACTIVATION operands
# (`input`/`output`/`grad_output`/`grad_input`) are parametrized by `ADT` — on
# READ each bf16 element is cast→fp32 before computing; on WRITE the fp32 result
# is cast→bf16. The cache and the QKᵀ/softmax/attn·V math are unchanged fp32.
# The default `ADT = DT` reproduces the legacy fp32 leaf byte-for-byte.
# ──────────────────────────────────────────────────────────────────────


def _attn_fwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int, ADT: DType = DT,
](
    output: LayoutTensor[ADT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    # Step 1: cache this head's Q/K/V slice (for backward).
    var n_qkv = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_qkv:
        var i = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        cache.ptr[unsafe_offset=b * CACHE_SIZE + i * DIM + h_off + d] = rebind[Scalar[ADT]](
            input.ptr[unsafe_offset=b * IN_DIM + i * DIM + h_off + d]
        ).cast[DT]()
        cache.ptr[unsafe_offset=b * CACHE_SIZE + K_OFF + i * DIM + h_off + d] = rebind[
            Scalar[ADT]
        ](input.ptr[unsafe_offset=b * IN_DIM + K_OFF + i * DIM + h_off + d]).cast[DT]()
        cache.ptr[unsafe_offset=b * CACHE_SIZE + V_OFF + i * DIM + h_off + d] = rebind[
            Scalar[ADT]
        ](input.ptr[unsafe_offset=b * IN_DIM + V_OFF + i * DIM + h_off + d]).cast[DT]()
        idx0 += bs

    # Step 2: per-row attention; each thread strides over query rows i.
    var i = tid
    while i < SEQ:
        var j_end = SEQ
        comptime if CAUSAL:
            j_end = i + 1

        var max_score = Scalar[DT](-1e30)
        for j in range(j_end):
            var s = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var q = rebind[Scalar[ADT]](
                    input.ptr[unsafe_offset=b * IN_DIM + i * DIM + h_off + d]
                ).cast[DT]()
                var k = rebind[Scalar[ADT]](
                    input.ptr[unsafe_offset=b * IN_DIM + K_OFF + j * DIM + h_off + d]
                ).cast[DT]()
                s += q * k
            s *= scale
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[unsafe_offset=aidx] = s
            if s > max_score:
                max_score = s

        var sum_exp = Scalar[DT](0)
        for j in range(j_end):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var e = exp(rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) - max_score)
            cache.ptr[unsafe_offset=aidx] = e
            sum_exp += e

        var inv_sum = Scalar[DT](1) / sum_exp
        for j in range(j_end):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[unsafe_offset=aidx] = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * inv_sum

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(j_end):
                var aidx = (
                    b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                )
                var v = rebind[Scalar[ADT]](
                    input.ptr[unsafe_offset=b * IN_DIM + V_OFF + j * DIM + h_off + d]
                ).cast[DT]()
                acc += rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * v
            output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d] = acc.cast[ADT]()
        i += bs


def _attn_zero_grad_kernel[
    BATCH: Int, IN_DIM: Int, ADT: DType = DT
](
    grad_input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * IN_DIM:
        grad_input.ptr[unsafe_offset=idx] = Scalar[ADT](0)


def _attn_dV_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    V_OFF: Int, ATTN_OFF: Int, ADT: DType = DT,
](
    grad_input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    grad_output: LayoutTensor[
        ADT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dV[j, h_off+d] = Σ_i attn[i,j] * grad_out[i, h_off+d]. Causal: i ≥ j.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var i_start = 0
        comptime if CAUSAL:
            i_start = j
        var acc = Scalar[DT](0)
        for i in range(i_start, SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var go = rebind[Scalar[ADT]](
                grad_output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d]
            ).cast[DT]()
            acc += rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * go
        var dv_idx = b * IN_DIM + V_OFF + j * DIM + h_off + d
        grad_input.ptr[unsafe_offset=dv_idx] = (
            rebind[Scalar[ADT]](grad_input.ptr[unsafe_offset=dv_idx]).cast[DT]() + acc
        ).cast[ADT]()
        idx0 += bs


def _attn_dscore_dQ_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int, ADT: DType = DT,
](
    grad_input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    grad_output: LayoutTensor[
        ADT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # Per row i: dot_sum, then d_score (overwrites cache.attn), then dQ.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    var i = tid
    while i < SEQ:
        var j_end = SEQ
        comptime if CAUSAL:
            j_end = i + 1

        var dot_sum = Scalar[DT](0)
        for j in range(j_end):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[ADT]](
                    grad_output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d]
                ).cast[DT]()
                var v = rebind[Scalar[DT]](
                    cache.ptr[unsafe_offset=b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            dot_sum += rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * d_attn

        for j in range(j_end):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[ADT]](
                    grad_output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d]
                ).cast[DT]()
                var v = rebind[Scalar[DT]](
                    cache.ptr[unsafe_offset=b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var attn_w = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx])
            cache.ptr[unsafe_offset=aidx] = attn_w * (d_attn - dot_sum) * scale

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(j_end):
                var aidx = (
                    b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                )
                var d_score = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx])
                var k = rebind[Scalar[DT]](
                    cache.ptr[unsafe_offset=b * CACHE_SIZE + K_OFF + j * DIM + h_off + d]
                )
                acc += d_score * k
            var dq_idx = b * IN_DIM + i * DIM + h_off + d
            grad_input.ptr[unsafe_offset=dq_idx] = (
                rebind[Scalar[ADT]](grad_input.ptr[unsafe_offset=dq_idx]).cast[DT]() + acc
            ).cast[ADT]()
        i += bs


def _attn_dK_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, CACHE_SIZE: Int, K_OFF: Int, ATTN_OFF: Int,
    ADT: DType = DT,
](
    grad_input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dK[j, h_off+d] = Σ_i d_score[i,j] * Q[i, h_off+d]. Reads d_score from
    # cache.attn (dscore_dQ kernel overwrote it). Causal: i ≥ j.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var i_start = 0
        comptime if CAUSAL:
            i_start = j
        var acc = Scalar[DT](0)
        for i in range(i_start, SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var d_score = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx])
            var q = rebind[Scalar[DT]](
                cache.ptr[unsafe_offset=b * CACHE_SIZE + i * DIM + h_off + d]
            )
            acc += d_score * q
        var dk_idx = b * IN_DIM + K_OFF + j * DIM + h_off + d
        grad_input.ptr[unsafe_offset=dk_idx] = (
            rebind[Scalar[ADT]](grad_input.ptr[unsafe_offset=dk_idx]).cast[DT]() + acc
        ).cast[ADT]()
        idx0 += bs


# ──────────────────────────────────────────────────────────────────────
# BMM fast path — batched-GEMM attention behind USE_MAX_KERNELS. Single-
# launch batched matmuls (tensor cores) for QKᵀ and attn·V, plus
# pack/softmax/unpack glue. Carried VERBATIM. Packed layout: (BH, SEQ,
# HEAD_DIM); scores layout: (BH, SEQ, SEQ).
# ──────────────────────────────────────────────────────────────────────


def _attn_pack_qkv_fwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, CACHE_SIZE: Int, PACKED: Int, ADT: DType = DT,
](
    packed_q: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_k: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_v: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
    input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime pack_elems = BATCH * SEQ * DIM
    if idx >= pack_elems:
        return
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var col = h * HEAD_DIM + d
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    var qv = rebind[Scalar[ADT]](input.ptr[unsafe_offset=b * IN_DIM + t * DIM + col]).cast[DT]()
    var kv = rebind[Scalar[ADT]](
        input.ptr[unsafe_offset=b * IN_DIM + KOFF + t * DIM + col]
    ).cast[DT]()
    var vv = rebind[Scalar[ADT]](
        input.ptr[unsafe_offset=b * IN_DIM + VOFF + t * DIM + col]
    ).cast[DT]()
    cache.ptr[unsafe_offset=b * CACHE_SIZE + t * DIM + col] = qv
    cache.ptr[unsafe_offset=b * CACHE_SIZE + KOFF + t * DIM + col] = kv
    cache.ptr[unsafe_offset=b * CACHE_SIZE + VOFF + t * DIM + col] = vv
    packed_q.ptr[unsafe_offset=pidx] = qv
    packed_k.ptr[unsafe_offset=pidx] = kv
    packed_v.ptr[unsafe_offset=pidx] = vv


def attn_warp_rows_grid(rows: Int) -> Int:
    """Blocks of `TPB` for a one-warp-per-row kernel over `rows` rows. A whole
    number of warps per block keeps the early return warp-uniform (a warp
    reduction with a lane missing is undefined)."""
    comptime assert TPB % WARP_SIZE == 0, "attention: TPB must be whole warps"
    return (rows * WARP_SIZE + TPB - 1) // TPB


def _attn_softmax_warp_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int, CAUSAL: Bool,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    scores: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    """The stable softmax of the per-(b,h)-block kernel it replaced (scale, max, causal zeros,
    weights mirrored into cache.attn) with one WARP per (b, h, i) row, lanes
    striding the row: coalesced, and BH x SEQ warps instead of BH blocks with
    a thread walking each row three times. The old kernel was the attention's
    largest non-GEMM cost in LeWM training (0.15 s of a 0.91 s step, batch
    128 on a 5090; `docs/CROSS_ATTENTION_OPTIMIZATION.md` §2.2, §4.7).
    Launch `attn_warp_rows_grid(BH * SEQ)` blocks of `TPB`."""
    var idx = Int(global_idx.x)
    var row = idx // WARP_SIZE
    var lane = idx % WARP_SIZE
    if row >= BH * SEQ:
        return
    var bh = row // SEQ
    var i = row % SEQ
    var b = bh // N_HEADS
    var h = bh % N_HEADS
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var row_off = bh * SEQ * SEQ + i * SEQ
    var cache_row = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ
    var j_end = SEQ
    comptime if CAUSAL:
        j_end = i + 1
    var mx = Scalar[DT](-1e30)
    var j = lane
    while j < j_end:
        mx = max(mx, rebind[Scalar[DT]](scores.ptr[unsafe_offset=row_off + j]) * scale)
        j += WARP_SIZE
    mx = warp.max(mx)
    var part = Scalar[DT](0)
    j = lane
    while j < j_end:
        part += exp(rebind[Scalar[DT]](scores.ptr[unsafe_offset=row_off + j]) * scale - mx)
        j += WARP_SIZE
    var inv = Scalar[DT](1) / warp.sum(part)
    j = lane
    while j < SEQ:
        var w = Scalar[DT](0)
        if j < j_end:
            w = exp(rebind[Scalar[DT]](scores.ptr[unsafe_offset=row_off + j]) * scale - mx) * inv
        scores.ptr[unsafe_offset=row_off + j] = w
        cache.ptr[unsafe_offset=cache_row + j] = w
        j += WARP_SIZE


def _attn_softmax_jvp_warp_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    dscore: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    dattn: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    """`_attn_softmax_jvp_kernel` (dscore = scale·a·(dattn − Σ a·dattn)), one
    WARP per row; the dot is a warp reduction. Same launch as the softmax."""
    var idx = Int(global_idx.x)
    var row = idx // WARP_SIZE
    var lane = idx % WARP_SIZE
    if row >= BH * SEQ:
        return
    var bh = row // SEQ
    var i = row % SEQ
    var b = bh // N_HEADS
    var h = bh % N_HEADS
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var row_off = bh * SEQ * SEQ + i * SEQ
    var cache_row = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ
    var part = Scalar[DT](0)
    var j = lane
    while j < SEQ:
        part += rebind[Scalar[DT]](cache.ptr[unsafe_offset=cache_row + j]) * rebind[
            Scalar[DT]
        ](dattn.ptr[unsafe_offset=row_off + j])
        j += WARP_SIZE
    var s = warp.sum(part)
    j = lane
    while j < SEQ:
        var a = rebind[Scalar[DT]](cache.ptr[unsafe_offset=cache_row + j])
        var da = rebind[Scalar[DT]](dattn.ptr[unsafe_offset=row_off + j])
        dscore.ptr[unsafe_offset=row_off + j] = scale * a * (da - s)
        j += WARP_SIZE


def _attn_drop_kernel[SEQ: Int, SCORES: Int, TRANSPOSED: Bool](
    buf: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    seed: UInt64,
    ctr: UInt64,
    p: Float32,
    scale: Scalar[DT],
):
    """Attention dropout on a (BH, SEQ, SEQ) slab: × 1/(1-p) where
    `hash_keep(seed, ctr, bh·S² + i·S + j)`, else 0. TRANSPOSED: the slab is
    laid out (bh, j, i) — the index is mapped back so the mask is the
    forward's (`nn/random/hash_mask.mojo`)."""
    var idx = Int(global_idx.x)
    if idx >= SCORES:
        return
    var orig = idx
    comptime if TRANSPOSED:
        var i = idx % SEQ
        var j = (idx // SEQ) % SEQ
        var bh = idx // (SEQ * SEQ)
        orig = bh * SEQ * SEQ + i * SEQ + j
    var v = rebind[Scalar[DT]](buf.ptr[unsafe_offset=idx]) * scale
    if not hash_keep(seed, ctr, UInt64(orig), p):
        v = Scalar[DT](0)
    buf.ptr[unsafe_offset=idx] = v


def _attn_packed_t_kernel[BH: Int, SEQ: Int, HEAD_DIM: Int, PACKED: Int](
    dst: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
):
    """`dst[bh, d, t] = src[bh, t, d]`: a packed head (BH, SEQ, HEAD_DIM) to
    the (BH, HEAD_DIM, SEQ) operand `bmm_tiled` takes for a ·ᵀ product. One
    thread per SOURCE element: the reads coalesce, the writes stride."""
    var idx = Int(global_idx.x)
    if idx >= BH * SEQ * HEAD_DIM:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var t = rem % SEQ
    var bh = rem // SEQ
    dst.ptr[unsafe_offset=bh * HEAD_DIM * SEQ + d * SEQ + t] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset=idx]
    )


def _attn_unpack_out_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    OUT_DIM: Int, PACKED: Int, ADT: DType = DT,
](
    output: LayoutTensor[ADT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    packed_out: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BATCH * SEQ * DIM
    if idx >= n:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    output.ptr[unsafe_offset=b * OUT_DIM + t * DIM + h * HEAD_DIM + d] = rebind[Scalar[DT]](
        packed_out.ptr[unsafe_offset=pidx]
    ).cast[ADT]()


def _attn_pack_in_bwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int, PACKED: Int, ADT: DType = DT,
](
    packed_dout: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_q: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_k: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_v: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    grad_output: LayoutTensor[ADT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BATCH * SEQ * DIM
    if idx >= n:
        return
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var col = h * HEAD_DIM + d
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    packed_dout.ptr[unsafe_offset=pidx] = rebind[Scalar[ADT]](
        grad_output.ptr[unsafe_offset=b * OUT_DIM + t * DIM + col]
    ).cast[DT]()
    packed_q.ptr[unsafe_offset=pidx] = rebind[Scalar[DT]](cache.ptr[unsafe_offset=b * CACHE_SIZE + t * DIM + col])
    packed_k.ptr[unsafe_offset=pidx] = rebind[Scalar[DT]](
        cache.ptr[unsafe_offset=b * CACHE_SIZE + KOFF + t * DIM + col]
    )
    packed_v.ptr[unsafe_offset=pidx] = rebind[Scalar[DT]](
        cache.ptr[unsafe_offset=b * CACHE_SIZE + VOFF + t * DIM + col]
    )


def _attn_softmax_jvp_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    dscore: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    dattn: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dscore = scale * a * (dattn - sum_k a_k*dattn_k). Causal masking is
    # implicit: cache.attn[i,j>i]=0 → dscore[i,j>i]=0.
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var bh_off = blk * SEQ * SEQ
    var cache_attn_base = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ
    var i = tid
    while i < SEQ:
        var row_off = bh_off + i * SEQ
        var cache_row = cache_attn_base + i * SEQ
        var s = Scalar[DT](0)
        for j in range(SEQ):
            s += rebind[Scalar[DT]](cache.ptr[unsafe_offset=cache_row + j]) * rebind[
                Scalar[DT]
            ](dattn.ptr[unsafe_offset=row_off + j])
        for j in range(SEQ):
            var a = rebind[Scalar[DT]](cache.ptr[unsafe_offset=cache_row + j])
            var da = rebind[Scalar[DT]](dattn.ptr[unsafe_offset=row_off + j])
            dscore.ptr[unsafe_offset=row_off + j] = scale * a * (da - s)
        i += bs


def _attn_transpose_from_cache_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    attn_T: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BH * SEQ * SEQ
    if idx >= n:
        return
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var i = idx % SEQ
    var rem = idx // SEQ
    var j = rem % SEQ
    var bh = rem // SEQ
    var b = bh // N_HEADS
    var h = bh % N_HEADS
    var src = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
    attn_T.ptr[unsafe_offset=bh * SEQ * SEQ + j * SEQ + i] = rebind[Scalar[DT]](
        cache.ptr[unsafe_offset=src]
    )


def _attn_transpose_scores_kernel[
    SEQ: Int, SCORES: Int, BH: Int,
](
    dst: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BH * SEQ * SEQ
    if idx >= n:
        return
    var i = idx % SEQ
    var rem = idx // SEQ
    var j = rem % SEQ
    var bh = rem // SEQ
    dst.ptr[unsafe_offset=bh * SEQ * SEQ + j * SEQ + i] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset=bh * SEQ * SEQ + i * SEQ + j]
    )


def _attn_unpack_grad_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, PACKED: Int, ADT: DType = DT,
](
    grad_input: LayoutTensor[ADT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dQ: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    dK: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    dV: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BATCH * SEQ * DIM
    if idx >= n:
        return
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var col = h * HEAD_DIM + d
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    grad_input.ptr[unsafe_offset=b * IN_DIM + t * DIM + col] = rebind[Scalar[DT]](
        dQ.ptr[unsafe_offset=pidx]
    ).cast[ADT]()
    grad_input.ptr[unsafe_offset=b * IN_DIM + KOFF + t * DIM + col] = rebind[Scalar[DT]](
        dK.ptr[unsafe_offset=pidx]
    ).cast[ADT]()
    grad_input.ptr[unsafe_offset=b * IN_DIM + VOFF + t * DIM + col] = rebind[Scalar[DT]](
        dV.ptr[unsafe_offset=pidx]
    ).cast[ADT]()


struct ScaledDotProductAttention[
    DIM: Int,
    N_HEADS: Int,
    SEQ_LEN: Int,
    CAUSAL: Bool = False,
    USE_MAX_KERNELS: Bool = True,
    ADT: DType = DT,
    P_DROP: Float64 = 0.0,
](Module):
    """`P_DROP` > 0: dropout on the attention WEIGHTS while training (torch's
    `scaled_dot_product_attention(dropout_p=…)`): A·V uses a·m/(1-p) while the
    cache keeps a for the softmax JVP; the mask is a counter hash
    (`hash_mask.mojo`), redrawn in the vjp. Switched by `set_attr["dropout"]`
    (ON by default); fp32 + the bmm path only. 0 compiles it out."""
    comptime ARITY: Int = 1
    # Activation-flow dtype (satisfies the Module trait). `ScaledDotProduct
    # Attention[D, H, S]` = fp32 (ACT_DT == DT, the legacy path, byte-identical);
    # `…[D, H, S, …, bfloat16]` flows its I/O activations at bf16 (the AMP "Step
    # B" memory win) while computing fp32 INTERNALLY: the cache + QKᵀ/softmax/
    # attn·V all stay fp32; only the I/O-activation kernel operands cast at the
    # bf16 boundary (read→fp32, write→bf16). bf16-flow is GPU-only.
    comptime ACT_DT = Self.ADT
    comptime HEAD_DIM: Int = Self.DIM // Self.N_HEADS
    comptime IN_DIMS = Array[Int, 1](fill=Self.SEQ_LEN * Self.DIM * 3)
    # `Array` is not `ImplicitlyCopyable` (Mojo 1.0): indexing the comptime
    # `IN_DIMS` from a runtime context would materialize the whole array.
    comptime IN_DIM0 = Self.SEQ_LEN * Self.DIM * 3
    comptime OUT_DIM = Self.SEQ_LEN * Self.DIM
    # Cache offsets (per sample).
    comptime K_OFF: Int = Self.SEQ_LEN * Self.DIM
    comptime V_OFF: Int = 2 * Self.SEQ_LEN * Self.DIM
    comptime ATTN_OFF: Int = 3 * Self.SEQ_LEN * Self.DIM
    comptime CACHE_SIZE: Int = (
        3 * Self.SEQ_LEN * Self.DIM
        + Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
    )

    # Cache (leaf-owned, output-caching) — [BATCH, CACHE_SIZE], lazy.
    var cache: Tensor
    # BMM scratch slabs (separate owned Tensors — one buffer per slab, vs the
    # legacy single pointer-sliced scratch). Lazily sized; GPU-only in practice
    # (CPU path uses local Lists). 4 packed slabs + 2 scores slabs (fwd uses 4
    # packed + 1 scores; bwd recycles them, see _vjp_gpu_bmm aliasing comments).
    var sp0: Tensor  # packed slot 0
    var sp1: Tensor  # packed slot 1
    var sp2: Tensor  # packed slot 2
    var sp3: Tensor  # packed slot 3
    var ss0: Tensor  # scores slot 0
    var ss1: Tensor  # scores slot 1
    var sp4: Tensor  # packed slot 4: Vᵀ for the backward's dout·Vᵀ
    # attention dropout (P_DROP > 0): see the struct docstring
    var drop_on: Bool
    var drop_seed: UInt64
    var drop_ctr: UInt64
    var drop_ctr_fwd: UInt64
    var drop_drew: Bool

    def __init__(out self):
        comptime assert (
            Self.DIM % Self.N_HEADS == 0
        ), "ScaledDotProductAttention: DIM must be divisible by N_HEADS"
        self.cache = Tensor()
        self.sp0 = Tensor()
        self.sp1 = Tensor()
        self.sp2 = Tensor()
        self.sp3 = Tensor()
        self.ss0 = Tensor()
        self.ss1 = Tensor()
        self.sp4 = Tensor()
        self.drop_on = True
        self.drop_seed = new_dropout_seed()
        self.drop_ctr = 0
        self.drop_ctr_fwd = 0
        self.drop_drew = False

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`dropout` (0/1): the attention dropout (P_DROP > 0 only)."""
        comptime if ATTR == "dropout":
            self.drop_on = value != Scalar[DT](0)

    def _drop_begin(mut self):
        """At each forward: draw a mask iff P_DROP > 0 and switched on."""
        self.drop_drew = Self.P_DROP > 0.0 and self.drop_on
        if self.drop_drew:
            self.drop_ctr_fwd = self.drop_ctr
            self.drop_ctr += 1

    @staticmethod
    def _drop_gpu[TRANSPOSED: Bool, BATCH: Int](
        mut buf: Tensor, seed: UInt64, ctr: UInt64, c: DeviceContext
    ) raises:
        comptime SCORES = BATCH * Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
        c.enqueue_function[_attn_drop_kernel[Self.SEQ_LEN, SCORES, TRANSPOSED]](
            buf.lt["gpu", Layout.row_major(SCORES)](),
            seed, ctr, Float32(Self.P_DROP),
            Scalar[DT](1.0 / (1.0 - Self.P_DROP)),
            grid_dim=(SCORES + TPB - 1) // TPB, block_dim=TPB,
        )

    @always_inline
    def _drop_scale(self, idx: Int) -> Scalar[DT]:
        """CPU: the mask value (0 or 1/(1-p)) of score element `idx`."""
        if hash_keep(self.drop_seed, self.drop_ctr_fwd, UInt64(idx), Float32(Self.P_DROP)):
            return Scalar[DT](1.0 / (1.0 - Self.P_DROP))
        return Scalar[DT](0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ScaledDotProductAttention: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error(
                    "ScaledDotProductAttention.make[target='gpu']: ctx required"
                )
        return Self()

    def _ensure_scratch_gpu[BATCH: Int](mut self, c: DeviceContext) raises:
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BATCH * Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
        self.sp0.ensure_gpu(c, PACKED)
        self.sp1.ensure_gpu(c, PACKED)
        self.sp2.ensure_gpu(c, PACKED)
        self.sp3.ensure_gpu(c, PACKED)
        self.ss0.ensure_gpu(c, SCORES)
        self.ss1.ensure_gpu(c, SCORES)
        self.sp4.ensure_gpu(c, PACKED)

    # ----- Forward ---------------------------------------------------------

    def release_buffers(mut self):
        """The forward cache (scores + q/k/v) and the packed-head / scores
        scratch slots — all `ensure`d by every forward and vjp."""
        self.cache.release()
        self.sp0.release()
        self.sp1.release()
        self.sp2.release()
        self.sp3.release()
        self.ss0.release()
        self.ss1.release()
        self.sp4.release()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.P_DROP > 0.0:
            comptime assert Self.ACT_DT == DT and Self.USE_MAX_KERNELS, (
                "attention dropout: fp32 bmm path only"
            )
        self._drop_begin()
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # The CPU helpers are fp32-only (`Tensor`) → rebind the activation
            # refs (sound: ACT_DT IS DT here, the compiler just won't collapse
            # the opaque param). The GPU helpers are ACT_DT-generic → pass the
            # activations directly (no rebind).
            comptime if target == "cpu":
                ref in0d = rebind[Tensor](in0)
                ref outd = rebind[Tensor](out)
                self._forward_cpu[B](in0d, outd)
            else:
                var c = ctx.value()
                out.ensure_gpu(c, B * Self.OUT_DIM)
                self.cache.ensure_gpu(c, B * Self.CACHE_SIZE)
                comptime if Self.USE_MAX_KERNELS:
                    self._forward_gpu_bmm[B](in0, out, c)
                else:
                    self._forward_gpu_custom[B](in0, out, c)
        else:
            # ── bf16-flow path (GPU-only). Activations cast at the I/O boundary;
            #    cache + QKᵀ/softmax/attn·V stay fp32 (the leaf is fp32-internal).
            comptime assert (
                target == "gpu"
            ), "bf16-flow ScaledDotProductAttention is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            self.cache.ensure_gpu(c, B * Self.CACHE_SIZE)
            comptime if Self.USE_MAX_KERNELS:
                self._forward_gpu_bmm[B](in0, out, c)
            else:
                self._forward_gpu_custom[B](in0, out, c)

    def _forward_gpu_custom[
        B: Int
    ](
        mut self,
        mut in0: TensorImpl[Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime lay_in = Layout.row_major(B, Self.IN_DIM0)
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime kernel = _attn_fwd_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[kernel](
            out.lt["gpu", lay_out](),
            in0.lt["gpu", lay_in](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=B * Self.N_HEADS, block_dim=TPB,
        )

    def _forward_gpu_bmm[
        B: Int
    ](
        mut self,
        mut in0: TensorImpl[Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        self._ensure_scratch_gpu[B](c)

        # Forward uses pq=sp0, pk=sp1, pv=sp2, pout=sp3, scores=ss0.
        comptime lay_in = Layout.row_major(B, Self.IN_DIM0)
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime lay_p = Layout.row_major(PACKED)
        comptime lay_s = Layout.row_major(SCORES)

        # 1. pack QKV → (BH, SEQ, HEAD_DIM) + write cache.
        comptime pelems = B * Self.SEQ_LEN * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime pack_k = _attn_pack_qkv_fwd_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIM0, Self.CACHE_SIZE, PACKED, Self.ADT,
        ]
        c.enqueue_function[pack_k](
            self.sp0.lt["gpu", lay_p](),
            self.sp1.lt["gpu", lay_p](),
            self.sp2.lt["gpu", lay_p](),
            self.cache.lt["gpu", lay_c](),
            in0.lt["gpu", lay_in](),
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. scores = Q @ Kᵀ  (BH, SEQ, SEQ): Kᵀ materialised in sp3 (free
        # until step 4), then the tiled product — MAX's `bmm` takes its naive
        # kernel at HEAD_DIM 64 (no multistage GEMM below k = 128).
        comptime pt_k = _attn_packed_t_kernel[BH, Self.SEQ_LEN, Self.HEAD_DIM, PACKED]
        c.enqueue_function[pt_k](
            self.sp3.lt["gpu", lay_p](), self.sp1.lt["gpu", lay_p](),
            grid_dim=pblocks, block_dim=TPB,
        )
        bmm_tiled[BH=BH, M=Self.SEQ_LEN, N=Self.SEQ_LEN, K=Self.HEAD_DIM](
            self.ss0.dev.value(), self.sp0.dev.value(), self.sp3.dev.value(), c
        )

        # 3. scale + stable softmax in-place; mirror into cache.attn.
        comptime sm_k = _attn_softmax_warp_kernel[
            B, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM, Self.CAUSAL,
            Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[sm_k](
            self.ss0.lt["gpu", lay_s](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=attn_warp_rows_grid(BH * Self.SEQ_LEN), block_dim=TPB,
        )

        # 3b. attention dropout on the A·V operand (cache.attn keeps a).
        comptime if Self.P_DROP > 0.0:
            if self.drop_drew:
                Self._drop_gpu[False, B](self.ss0, self.drop_seed, self.drop_ctr_fwd, c)

        # 4. packed_out = attn @ V.
        bmm_tiled[BH=BH, M=Self.SEQ_LEN, N=Self.HEAD_DIM, K=Self.SEQ_LEN](
            self.sp3.dev.value(), self.ss0.dev.value(), self.sp2.dev.value(), c
        )

        # 5. unpack → output.
        comptime up_blocks = (pelems + TPB - 1) // TPB
        comptime up_k = _attn_unpack_out_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.OUT_DIM, PACKED, Self.ADT,
        ]
        c.enqueue_function[up_k](
            out.lt["gpu", lay_out](),
            self.sp3.lt["gpu", lay_p](),
            grid_dim=up_blocks, block_dim=TPB,
        )

    def _forward_cpu[B: Int](mut self, mut in0: Tensor, mut out: Tensor) raises:
        # Mirror the GPU bmm forward (pack → QKᵀ bmm → scalar softmax+mask →
        # attn·V bmm → unpack) with target="cpu" (Apple-Accelerate). The 2
        # GEMMs are BLAS; softmax + causal mask stay scalar. Cache layout
        # [Q|K|V|scores] is identical to the GPU path.
        out.ensure(B * Self.OUT_DIM)
        self.cache.ensure(B * Self.CACHE_SIZE)
        comptime IN = Self.IN_DIM0
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SD = Self.SEQ_LEN * Self.DIM
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(Self.HEAD_DIM)))

        ref ip = in0.data
        ref op = out.data
        ref cp = self.cache.data

        # Local scratch (separate Lists, no pointer slicing).
        var pq = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pout = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var sc = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. Cache Q/K/V and pack into (BH, SEQ, HEAD_DIM).
        for b in range(B):
            for i in range(SD):
                cp[b * C + i] = ip[b * IN + i]
                cp[b * C + Self.K_OFF + i] = ip[b * IN + Self.K_OFF + i]
                cp[b * C + Self.V_OFF + i] = ip[b * IN + Self.V_OFF + i]
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * Self.HEAD_DIM
                for t in range(Self.SEQ_LEN):
                    for d in range(Self.HEAD_DIM):
                        var col = h_off + d
                        var pidx = bh * Self.SEQ_LEN * Self.HEAD_DIM + t * Self.HEAD_DIM + d
                        pq[pidx] = ip[b * IN + t * Self.DIM + col]
                        pk[pidx] = ip[b * IN + Self.K_OFF + t * Self.DIM + col]
                        pv[pidx] = ip[b * IN + Self.V_OFF + t * Self.DIM + col]

        # 2. scores = Q @ Kᵀ  (BH, SEQ, SEQ).
        var scores_tt = TileTensor(
            sc, row_major[BH, Self.SEQ_LEN, Self.SEQ_LEN]()
        )
        var pq_tt = TileTensor(pq, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        var pk_tt = TileTensor(pk, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[transpose_b=True, target="cpu"](
            scores_tt, pq_tt, pk_tt
        )

        # 3. scale + stable softmax (scalar); mirror weights into cache.attn
        #    and the `sc` scores buffer (zeroed in the causal upper triangle).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sc_base = bh * Self.SEQ_LEN * Self.SEQ_LEN
                var cache_base = (
                    b * C + Self.ATTN_OFF + h * Self.SEQ_LEN * Self.SEQ_LEN
                )
                for i in range(Self.SEQ_LEN):
                    var j_end = Self.SEQ_LEN
                    comptime if Self.CAUSAL:
                        j_end = i + 1
                    var row = sc_base + i * Self.SEQ_LEN
                    var crow = cache_base + i * Self.SEQ_LEN
                    var mx = Scalar[DT](-1e30)
                    for j in range(j_end):
                        var s = sc[row + j] * scale
                        sc[row + j] = s
                        if s > mx:
                            mx = s
                    var se = Scalar[DT](0)
                    for j in range(j_end):
                        var e = exp(sc[row + j] - mx)
                        sc[row + j] = e
                        se += e
                    var inv = Scalar[DT](1) / se
                    for j in range(j_end):
                        var w = sc[row + j] * inv
                        sc[row + j] = w
                        cp[crow + j] = w
                    comptime if Self.CAUSAL:
                        for j in range(i + 1, Self.SEQ_LEN):
                            sc[row + j] = Scalar[DT](0)
                            cp[crow + j] = Scalar[DT](0)

        # 3b. attention dropout on the A·V operand (cache.attn keeps a).
        comptime if Self.P_DROP > 0.0:
            if self.drop_drew:
                for k in range(SCORES):
                    sc[k] = sc[k] * self._drop_scale(k)

        # 4. packed_out = attn @ V  (BH, SEQ, HEAD_DIM).
        var pout_tt = TileTensor(
            pout, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        var pv_tt = TileTensor(pv, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[target="cpu"](pout_tt, scores_tt, pv_tt)

        # 5. unpack packed_out → output.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * Self.HEAD_DIM
                for t in range(Self.SEQ_LEN):
                    for d in range(Self.HEAD_DIM):
                        var pidx = bh * Self.SEQ_LEN * Self.HEAD_DIM + t * Self.HEAD_DIM + d
                        op[b * OUT + t * Self.DIM + h_off + d] = pout[pidx]
        _ = pq^
        _ = pk^
        _ = pv^
        _ = pout^
        _ = sc^

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # forward_input unused — this leaf is output-caching (reads only the
        # cache + grad_output).
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # CPU helpers are fp32-only (`Tensor`) → rebind (sound, ACT_DT IS DT);
            # GPU helpers are ACT_DT-generic → pass activations directly.
            comptime if target == "cpu":
                ref god = rebind[Tensor](grad_output)
                ref gind = rebind[Tensor](gin)
                self._vjp_cpu[B](god, gind)
            else:
                var c = ctx.value()
                gin.ensure_gpu(c, B * Self.IN_DIM0)
                comptime if Self.USE_MAX_KERNELS:
                    self._vjp_gpu_bmm[B](grad_output, gin, c)
                else:
                    self._vjp_gpu_custom[B](grad_output, gin, c)
        else:
            # ── bf16-flow path (GPU-only). I/O activations cast at the boundary;
            #    cache + grad math stay fp32 (fp32-internal). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow ScaledDotProductAttention is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_DIM0)
            comptime if Self.USE_MAX_KERNELS:
                self._vjp_gpu_bmm[B](grad_output, gin, c)
            else:
                self._vjp_gpu_custom[B](grad_output, gin, c)

    def _vjp_gpu_custom[
        B: Int
    ](
        mut self,
        mut grad_output: TensorImpl[Self.ACT_DT],
        mut gin: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime lay_in = Layout.row_major(B, Self.IN_DIM0)
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime grid_bh = B * Self.N_HEADS
        # 1) zero grad_input.
        comptime zk = _attn_zero_grad_kernel[B, Self.IN_DIM0, Self.ADT]
        comptime zn = (B * Self.IN_DIM0 + TPB - 1) // TPB
        c.enqueue_function[zk](
            gin.lt["gpu", lay_in](), grid_dim=zn, block_dim=TPB
        )
        # 2) dV (reads attn weights — must precede dscore_dQ overwrite).
        comptime dvk = _attn_dV_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE,
            Self.V_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[dvk](
            gin.lt["gpu", lay_in](),
            grad_output.lt["gpu", lay_out](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=grid_bh, block_dim=TPB,
        )
        # 3) dscore + dQ (overwrites cache.attn with d_score).
        comptime dqk = _attn_dscore_dQ_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[dqk](
            gin.lt["gpu", lay_in](),
            grad_output.lt["gpu", lay_out](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=grid_bh, block_dim=TPB,
        )
        # 4) dK (reads d_score from cache.attn).
        comptime dkk = _attn_dK_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIM0, Self.CACHE_SIZE,
            Self.K_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[dkk](
            gin.lt["gpu", lay_in](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=grid_bh, block_dim=TPB,
        )

    def _vjp_gpu_bmm[
        B: Int
    ](
        mut self,
        mut grad_output: TensorImpl[Self.ACT_DT],
        mut gin: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        self._ensure_scratch_gpu[B](c)

        # Scratch aliasing (slabs recycled once their producer's last read is
        # enqueued — safe: kernels on the stream run in order). Map:
        #   sp0: pdout  → (step7) dK      sp1: pq     → (step8) dQ
        #   sp2: pk                        sp3: pv     → (step5) dV
        #   ss0: dattn  → (step4) attn_T → (step6) dscore_T
        #   ss1: dscore
        comptime lay_in = Layout.row_major(B, Self.IN_DIM0)
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime lay_p = Layout.row_major(PACKED)
        comptime lay_s = Layout.row_major(SCORES)

        comptime pelems = B * SL * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime sblocks = (SCORES + TPB - 1) // TPB

        # 1. pack dout + cache Q/K/V → (BH, SEQ, HEAD_DIM).
        comptime pin_k = _attn_pack_in_bwd_kernel[
            B, Self.DIM, Self.N_HEADS, SL, HD,
            Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE, PACKED, Self.ADT,
        ]
        c.enqueue_function[pin_k](
            self.sp0.lt["gpu", lay_p](),
            self.sp1.lt["gpu", lay_p](),
            self.sp2.lt["gpu", lay_p](),
            self.sp3.lt["gpu", lay_p](),
            grad_output.lt["gpu", lay_out](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. dattn(ss0) = dout @ Vᵀ: Vᵀ(sp4), then the tiled product.
        comptime pt_k = _attn_packed_t_kernel[BH, SL, HD, PACKED]
        c.enqueue_function[pt_k](
            self.sp4.lt["gpu", lay_p](), self.sp3.lt["gpu", lay_p](),
            grid_dim=pblocks, block_dim=TPB,
        )
        bmm_tiled[BH=BH, M=SL, N=SL, K=HD](
            self.ss0.dev.value(), self.sp0.dev.value(), self.sp4.dev.value(), c
        )

        # 2b. dropout: d(a) = d(a·m/(1-p)) · m/(1-p).
        comptime if Self.P_DROP > 0.0:
            if self.drop_drew:
                Self._drop_gpu[False, B](self.ss0, self.drop_seed, self.drop_ctr_fwd, c)

        # 3. softmax jvp → dscore(ss1)  (reads dattn(ss0)).
        comptime jvp_k = _attn_softmax_jvp_warp_kernel[
            B, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[jvp_k](
            self.ss1.lt["gpu", lay_s](),
            self.ss0.lt["gpu", lay_s](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=attn_warp_rows_grid(BH * SL), block_dim=TPB,
        )

        # 4. attn_T(ss0) = transpose(cache.attn)  (ss0 free — dattn consumed).
        comptime tac_k = _attn_transpose_from_cache_kernel[
            B, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[tac_k](
            self.ss0.lt["gpu", lay_s](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 4b. dropout: dV takes the DROPPED weights (a·m/(1-p))ᵀ.
        comptime if Self.P_DROP > 0.0:
            if self.drop_drew:
                Self._drop_gpu[True, B](self.ss0, self.drop_seed, self.drop_ctr_fwd, c)

        # 5. dV(sp3) = attn_T(ss0) @ dout(sp0)  (sp3 free — pv last read step 2).
        bmm_tiled[BH=BH, M=SL, N=HD, K=SL](
            self.sp3.dev.value(), self.ss0.dev.value(), self.sp0.dev.value(), c
        )

        # 6. dscore_T(ss0) = transpose(dscore(ss1)) (ss0 free — attn_T read s5).
        comptime ts_k = _attn_transpose_scores_kernel[SL, SCORES, BH]
        c.enqueue_function[ts_k](
            self.ss0.lt["gpu", lay_s](),
            self.ss1.lt["gpu", lay_s](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 7. dK(sp0) = dscore_T(ss0) @ Q(sp1)  (sp0 free — pdout last read s5).
        bmm_tiled[BH=BH, M=SL, N=HD, K=SL](
            self.sp0.dev.value(), self.ss0.dev.value(), self.sp1.dev.value(), c
        )

        # 8. dQ(sp1) = dscore(ss1) @ K(sp2)  (sp1 free — pq last read step 7).
        bmm_tiled[BH=BH, M=SL, N=HD, K=SL](
            self.sp1.dev.value(), self.ss1.dev.value(), self.sp2.dev.value(), c
        )

        # 9. unpack dQ(sp1)/dK(sp0)/dV(sp3) → grad_input.
        comptime ug_k = _attn_unpack_grad_kernel[
            B, Self.DIM, Self.N_HEADS, SL, HD, Self.IN_DIM0, PACKED, Self.ADT,
        ]
        c.enqueue_function[ug_k](
            gin.lt["gpu", lay_in](),
            self.sp1.lt["gpu", lay_p](),
            self.sp0.lt["gpu", lay_p](),
            self.sp3.lt["gpu", lay_p](),
            grid_dim=pblocks, block_dim=TPB,
        )

    def _vjp_cpu[
        B: Int
    ](mut self, mut grad_output: Tensor, mut gin: Tensor) raises:
        # Mirror the GPU bmm backward (pack → dattn=dout·Vᵀ bmm → scalar
        # softmax-JVP → transposes → dV/dK/dQ bmms → unpack) with target="cpu".
        gin.ensure(B * Self.IN_DIM0)
        comptime IN = Self.IN_DIM0
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * SL * Self.DIM
        comptime SCORES = BH * SL * SL
        var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(Self.HEAD_DIM)))

        ref gop = grad_output.data
        ref gip = gin.data
        ref cp = self.cache.data

        # Local scratch (separate Lists, no pointer slicing).
        var pdout = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pq = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dV = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dK = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dQ = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dattn = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var attn_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. pack dout + cache Q/K/V → (BH, SEQ, HEAD_DIM).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * HD
                for t in range(SL):
                    for d in range(HD):
                        var col = h_off + d
                        var pidx = bh * SL * HD + t * HD + d
                        pdout[pidx] = gop[b * OUT + t * Self.DIM + col]
                        pq[pidx] = cp[b * C + t * Self.DIM + col]
                        pk[pidx] = cp[b * C + Self.K_OFF + t * Self.DIM + col]
                        pv[pidx] = cp[b * C + Self.V_OFF + t * Self.DIM + col]

        # 2. dattn = dout @ Vᵀ  (BH, SEQ, SEQ).
        var dattn_tt = TileTensor(dattn, row_major[BH, SL, SL]())
        var pdout_tt = TileTensor(pdout, row_major[BH, SL, HD]())
        var pv_tt = TileTensor(pv, row_major[BH, SL, HD]())
        batched_matmul[transpose_b=True, target="cpu"](
            dattn_tt, pdout_tt, pv_tt
        )

        # 2b. dropout: d(a) = d(a·m/(1-p)) · m/(1-p).
        comptime if Self.P_DROP > 0.0:
            if self.drop_drew:
                for k in range(SCORES):
                    dattn[k] = dattn[k] * self._drop_scale(k)

        # 3. softmax jvp (scalar): dscore = scale*a*(dattn - Σ_k a_k*dattn_k).
        #    a is cache.attn (causal upper-triangle already zeroed) → dscore is
        #    automatically zero where masked. Also build attn_T / dscore_T.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sc_base = bh * SL * SL
                var cache_base = b * C + Self.ATTN_OFF + h * SL * SL
                for i in range(SL):
                    var row = sc_base + i * SL
                    var crow = cache_base + i * SL
                    var s = Scalar[DT](0)
                    for j in range(SL):
                        s += cp[crow + j] * dattn[row + j]
                    for j in range(SL):
                        var a = cp[crow + j]
                        dscore[row + j] = scale * a * (dattn[row + j] - s)
                        var a_used = a
                        comptime if Self.P_DROP > 0.0:
                            if self.drop_drew:
                                a_used = a * self._drop_scale(row + j)
                        attn_T[sc_base + j * SL + i] = a_used
                        dscore_T[sc_base + j * SL + i] = dscore[row + j]

        # 4. dV = attn_T @ dout.
        var attnT_tt = TileTensor(attn_T, row_major[BH, SL, SL]())
        var dV_tt = TileTensor(dV, row_major[BH, SL, HD]())
        batched_matmul[target="cpu"](dV_tt, attnT_tt, pdout_tt)

        # 5. dK = dscore_T @ Q.
        var dscoreT_tt = TileTensor(dscore_T, row_major[BH, SL, SL]())
        var pq_tt = TileTensor(pq, row_major[BH, SL, HD]())
        var dK_tt = TileTensor(dK, row_major[BH, SL, HD]())
        batched_matmul[target="cpu"](dK_tt, dscoreT_tt, pq_tt)

        # 6. dQ = dscore @ K.
        var dscore_tt = TileTensor(dscore, row_major[BH, SL, SL]())
        var pk_tt = TileTensor(pk, row_major[BH, SL, HD]())
        var dQ_tt = TileTensor(dQ, row_major[BH, SL, HD]())
        batched_matmul[target="cpu"](dQ_tt, dscore_tt, pk_tt)

        # 7. unpack dQ/dK/dV → grad_input.
        for i in range(B * IN):
            gip[i] = 0.0
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * HD
                for t in range(SL):
                    for d in range(HD):
                        var col = h_off + d
                        var pidx = bh * SL * HD + t * HD + d
                        gip[b * IN + t * Self.DIM + col] = dQ[pidx]
                        gip[b * IN + Self.K_OFF + t * Self.DIM + col] = dK[pidx]
                        gip[b * IN + Self.V_OFF + t * Self.DIM + col] = dV[pidx]
        _ = pdout^
        _ = pq^
        _ = pk^
        _ = pv^
        _ = dV^
        _ = dK^
        _ = dQ^
        _ = dattn^
        _ = dscore^
        _ = attn_T^
        _ = dscore_T^

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (no Param fields → no-op).
