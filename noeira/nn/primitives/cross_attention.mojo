"""CrossAttention[DIM, N_HEADS, Q_LEN, KV_LEN, MASKED] — multi-head attention
with SEPARATE query and key/value streams.

The gap this fills. `ScaledDotProductAttention` is ARITY=1 over a packed
`[Q|K|V]` of ONE sequence length, produced by a single `Linear[dim, 3*dim]` on
one input — so it can only ever do self-attention where q, k and v are three
projections of the same tensor. `DecoderBlock` is a degenerate single-KV-token
case built for LeWM. DETR/ACT needs two things neither can express:

  1. **Cross-attention proper** — `Q_LEN` queries (ACT: the k=100 action
     queries) attending to a `KV_LEN` memory (ACT: 2 + camera tokens).
  2. **Positional embedding on q and k but NOT v**, re-added at every layer
     (`transformer.py:with_pos_embed`). Even at `Q_LEN == KV_LEN` the existing
     self-attention leaf cannot do this, because it derives all three
     projections from one tensor.

Setting `Q_LEN == KV_LEN` recovers DETR self-attention, so ONE leaf serves both
the encoder and both attentions of the decoder layer.

## Signature

    inputs   q  [BATCH, Q_LEN  * DIM]
             k  [BATCH, KV_LEN * DIM]
             v  [BATCH, KV_LEN * DIM]
             m  [BATCH, KV_LEN]         (MASKED=True only)
    output      [BATCH, Q_LEN  * DIM]

    out[b,i] = concat_h( softmax_j( q·kᵀ/sqrt(HEAD_DIM) + bias(m) ) · v )

`m` is a **key padding mask, 1.0 = attend / 0.0 = ignore** — the same polarity
as `ACTDataset`'s `valid`, and the INVERSE of torch's `key_padding_mask` (where
True means *ignore*). It is converted to an additive `MASK_NEG` bias internally.
Per-SAMPLE, which is what `MaskedAttention` cannot do: that leaf owns ONE
`[SEQ, SEQ]` bias shared by the whole batch, right for a causal mask and wrong
for padding, where each row of the batch pads at a different length.

⚠ A fully-masked query row would divide by zero. Rows are renormalized by a
floored denominator and produce a zero context vector instead of NaN. ACT never
gets there — `cls` and `qpos` are always unmasked (`detr_vae.py:96`) — but a
NaN that only appears on a short episode is not a failure worth discovering
during a training run.

No params. Backward is the standard attention VJP; `q`/`k`/`v` come back
through `forward_input` (the `Module` contract, as `Linear.vjp` relies on), so
only the softmax weights are cached.

CPU + GPU. The GPU path mirrors `attention.mojo`'s `_forward_gpu_bmm` /
`_vjp_gpu_bmm` structure — pack to head-major, two `batched_matmul`s around a
scalar softmax, unpack — with the packed slabs sized separately for the query
and key/value streams, and per-sample masking folded into the softmax kernel.
"""

from noeira.nn.core.mm import mm, bmm
from noeira.nn.core.mm_tiled import bmm_tiled
from std.collections import Array
from std.math import exp, exp2, fma, sqrt
from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major, Idx
from noeira.core.tiles import RowMajor, Tile, shared_tile
from linalg.bmm import batched_matmul
from std.sys.info import default_accelerator
from nn.attention.gpu.mha import flash_attention_dispatch
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import TileTensorMHAOperand

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime XATTN_MASK_NEG: Scalar[DT] = Scalar[DT](-1.0e30)
"""Additive bias for a masked key. Matches `masked_attention.mojo:MASK_NEG`;
large enough that `exp(s - max)` underflows to 0 in fp32, finite so that a row
which is masked EVERYWHERE still has a defined maximum."""

comptime XATTN_DENOM_FLOOR: Scalar[DT] = Scalar[DT](1.0e-30)
"""Softmax denominator floor. Only reachable when every key of a row is
masked; turns a NaN into a zero context vector."""


# ══════════════════════════════════════════════════════════════════════════
# GPU kernels
# ══════════════════════════════════════════════════════════════════════════
# Token-major <-> head-major repacking, the masked softmax, and its JVP. The
# two matmuls are `batched_matmul`, as in `attention.mojo`. Everything is
# indexed off (BH, LEN, HEAD_DIM), where LEN is Q_LEN for the query stream and
# KV_LEN for the key/value stream — the single structural difference from the
# equal-length self-attention leaf.


def _xa_pack_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, LEN: Int, HEAD_DIM: Int, PACKED: Int
](
    packed: Tile[DT, PACKED],
    src: Tile[DT, BATCH, LEN * DIM],
):
    """token-major `(BATCH, LEN, DIM)` -> head-major `(BH, LEN, HEAD_DIM)`."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LEN * DIM:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % LEN
    var b = rem2 // LEN
    var bh = b * N_HEADS + h
    packed.ptr[unsafe_offset=bh * LEN * HEAD_DIM + t * HEAD_DIM + d] = rebind[
        Scalar[DT]
    ](src.ptr[unsafe_offset=b * LEN * DIM + t * DIM + h * HEAD_DIM + d])


def _xa_unpack_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, LEN: Int, HEAD_DIM: Int, PACKED: Int
](
    dst: Tile[DT, BATCH, LEN * DIM],
    packed: Tile[DT, PACKED],
):
    """head-major -> token-major (the inverse of `_xa_pack_kernel`)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LEN * DIM:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % LEN
    var b = rem2 // LEN
    var bh = b * N_HEADS + h
    dst.ptr[unsafe_offset=b * LEN * DIM + t * DIM + h * HEAD_DIM + d] = rebind[
        Scalar[DT]
    ](packed.ptr[unsafe_offset=bh * LEN * HEAD_DIM + t * HEAD_DIM + d])


def _xa_pack_kt_kernel[
    BATCH: Int, DIM: Int, NH: Int, KL: Int, HD: Int, PK: Int
](
    dst: Tile[DT, PK],
    src: Tile[DT, BATCH, KL * DIM],
):
    """token-major `(BATCH, KL, DIM)` -> `(BH, HD, KL)`, the score matmul's B.

    Packing and transposing in ONE pass. `_xa_pack_kernel` + a transpose reads
    and writes the slab twice for the same result (bit-identical: both are
    copies); on the Orin that is 0.234 ms against 0.150 at SigLIP's shape, and
    1.8x at ACT's encoder shape."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= PK:
        return
    var j = idx % KL
    var r = idx // KL
    var d = r % HD
    var bh = r // HD
    var b = bh // NH
    var h = bh % NH
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset=b * KL * DIM + j * DIM + h * HD + d]
    )


def _xa_row_stats_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HEAD_DIM: Int,
    MASKED: Bool, SCORES: Int, ROWS: Int,
](
    scores: Tile[DT, SCORES],
    stats: Tile[DT, ROWS * 2],
    mask: Tile[DT, BATCH, KL],
):
    """Softmax pass 1: one thread per (b, h, i) — the row max and 1/denominator.

    Reads only; `scores` keeps the raw Q.Kt product, which pass 2 re-reads."""
    var r = Int(block_dim.x * block_idx.x + thread_idx.x)
    if r >= ROWS:
        return
    var b = r // (N_HEADS * QL)
    var base = r * KL
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HEAD_DIM))
    var mx = XATTN_MASK_NEG
    for j in range(KL):
        var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=base + j]) * scale
        comptime if MASKED:
            if rebind[Scalar[DT]](
                mask.ptr[unsafe_offset=b * KL + j]
            ) < Scalar[DT](0.5):
                sv = XATTN_MASK_NEG
        if sv > mx:
            mx = sv
    var se = Scalar[DT](0)
    for j in range(KL):
        var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=base + j]) * scale
        comptime if MASKED:
            if rebind[Scalar[DT]](
                mask.ptr[unsafe_offset=b * KL + j]
            ) < Scalar[DT](0.5):
                sv = XATTN_MASK_NEG
        se += exp(sv - mx)
    var denom = se if se > XATTN_DENOM_FLOOR else XATTN_DENOM_FLOOR
    stats.ptr[unsafe_offset=2 * r] = mx
    stats.ptr[unsafe_offset=2 * r + 1] = Scalar[DT](1) / denom


def _xa_element_softmax_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HEAD_DIM: Int,
    MASKED: Bool, SCORES: Int, ROWS: Int,
](
    scores: Tile[DT, SCORES],
    stats: Tile[DT, ROWS * 2],
    mask: Tile[DT, BATCH, KL],
    attn: Tile[DT, SCORES],
):
    """Softmax pass 2: one thread per (b, h, i, j), straight into the cache.

    `_xa_softmax_kernel` did the whole softmax with one block per (b, h) and
    its threads striding query rows — 53% of a SigLIP layer on the Orin, its
    only parallelism 12 blocks of 128 threads for 12.6 M weights. Two passes
    over the scores cost one extra read and buy a thread per weight: 20.2 ms
    -> 4.1 ms at SigLIP, and 4.8 -> 0.55 at ACT's encoder shape.

    The weights go to the cache ONLY; the scores slab keeps the raw product,
    which nothing downstream reads (`self.attn` is the backward's input)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SCORES:
        return
    var j = idx % KL
    var r = idx // KL
    comptime if MASKED:
        var b = r // (N_HEADS * QL)
        if rebind[Scalar[DT]](
            mask.ptr[unsafe_offset=b * KL + j]
        ) < Scalar[DT](0.5):
            attn.ptr[unsafe_offset=idx] = Scalar[DT](0)
            return
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HEAD_DIM))
    var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=idx]) * scale
    var e = exp(sv - rebind[Scalar[DT]](stats.ptr[unsafe_offset=2 * r]))
    attn.ptr[unsafe_offset=idx] = e * rebind[Scalar[DT]](
        stats.ptr[unsafe_offset=2 * r + 1]
    )


def _xa_softmax_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HEAD_DIM: Int,
    MASKED: Bool, ATTN_SIZE: Int, SCORES: Int, BH: Int,
](
    scores: Tile[DT, SCORES],
    attn: Tile[DT, BATCH, ATTN_SIZE],
    mask: Tile[DT, BATCH, KL],
):
    """One block per (b, h); threads stride over query rows.

    Scale, apply the per-sample key mask, stable softmax in place, and mirror
    the weights into the cache. Identical arithmetic to the CPU path, including
    the floored denominator for a fully-masked row.

    ⚠ NO LONGER THE FORWARD'S SOFTMAX — replaced by `_xa_row_stats_kernel` +
    `_xa_element_softmax_kernel`, which are 4.95x faster at SigLIP's shape on
    the Orin. Kept as the reference `cross_attention_stages_bench.mojo` times
    and bit-compares the shipped pair against.
    """
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HEAD_DIM))
    var bh_off = blk * QL * KL
    var attn_base = b * ATTN_SIZE + h * QL * KL
    var i = tid
    while i < QL:
        var row = bh_off + i * KL
        var arow = attn_base + i * KL
        var mx = XATTN_MASK_NEG
        for j in range(KL):
            var sv = rebind[Scalar[DT]](
                scores.ptr[unsafe_offset=row + j]
            ) * scale
            comptime if MASKED:
                if rebind[Scalar[DT]](
                    mask.ptr[unsafe_offset=b * KL + j]
                ) < Scalar[DT](0.5):
                    sv = XATTN_MASK_NEG
            scores.ptr[unsafe_offset=row + j] = sv
            if sv > mx:
                mx = sv
        var se = Scalar[DT](0)
        for j in range(KL):
            var e = exp(
                rebind[Scalar[DT]](scores.ptr[unsafe_offset=row + j]) - mx
            )
            scores.ptr[unsafe_offset=row + j] = e
            se += e
        var denom = se if se > XATTN_DENOM_FLOOR else XATTN_DENOM_FLOOR
        var inv = Scalar[DT](1) / denom
        for j in range(KL):
            var w = rebind[Scalar[DT]](
                scores.ptr[unsafe_offset=row + j]
            ) * inv
            comptime if MASKED:
                if rebind[Scalar[DT]](
                    mask.ptr[unsafe_offset=b * KL + j]
                ) < Scalar[DT](0.5):
                    w = Scalar[DT](0)
            scores.ptr[unsafe_offset=row + j] = w
            attn.ptr[unsafe_offset=arow + j] = w
        i += bs


def _xa_softmax_jvp_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HEAD_DIM: Int,
    ATTN_SIZE: Int, SCORES: Int, BH: Int,
](
    dscore: Tile[DT, SCORES],
    dattn: Tile[DT, SCORES],
    attn: Tile[DT, BATCH, ATTN_SIZE],
):
    """`dscore = scale * a * (dattn - sum_k a_k * dattn_k)`.

    Masked columns have `a == 0`, so their dscore is zero without a second mask
    read — the same property the CPU path relies on.
    """
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HEAD_DIM))
    var bh_off = blk * QL * KL
    var attn_base = b * ATTN_SIZE + h * QL * KL
    var i = tid
    while i < QL:
        var row = bh_off + i * KL
        var arow = attn_base + i * KL
        var sdot = Scalar[DT](0)
        for j in range(KL):
            sdot += rebind[Scalar[DT]](attn.ptr[unsafe_offset=arow + j]) * (
                rebind[Scalar[DT]](dattn.ptr[unsafe_offset=row + j])
            )
        for j in range(KL):
            var a = rebind[Scalar[DT]](attn.ptr[unsafe_offset=arow + j])
            dscore.ptr[unsafe_offset=row + j] = (
                scale
                * a
                * (
                    rebind[Scalar[DT]](dattn.ptr[unsafe_offset=row + j])
                    - sdot
                )
            )
        i += bs


def _xa_transpose_k_kernel[BH: Int, KL: Int, HD: Int, PK: Int](
    out_t: Tile[DT, PK],
    k: Tile[DT, PK],
):
    """packed keys `(BH, KL, HD)` -> `(BH, HD, KL)`, contiguous.

    The forward's score matmul reads this instead of `bmm[transpose_b=True]`
    on the packed keys. Same matmul, same bits (`cross_attention_bench`: 0 of
    12.6 M cached weights differ at SigLIP's shape), but the transposed call is
    2.06x slower on the Orin (78.3 -> 38.0 ms per SigLIP layer)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= PK:
        return
    var j = idx % KL
    var rem = idx // KL
    var d = rem % HD
    var bh = rem // HD
    out_t.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        k.ptr[unsafe_offset=bh * KL * HD + j * HD + d]
    )


def _xa_transpose_attn_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, ATTN_SIZE: Int, SCORES: Int
](
    out_t: Tile[DT, SCORES],
    attn: Tile[DT, BATCH, ATTN_SIZE],
):
    """cache `(BH, QL, KL)` -> `(BH, KL, QL)`."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SCORES:
        return
    var j = idx % KL
    var rem = idx // KL
    var i = rem % QL
    var bh = rem // QL
    var b = bh // N_HEADS
    var h = bh % N_HEADS
    out_t.ptr[unsafe_offset=bh * KL * QL + j * QL + i] = rebind[Scalar[DT]](
        attn.ptr[unsafe_offset=b * ATTN_SIZE + h * QL * KL + i * KL + j]
    )


def _xa_transpose_scores_kernel[QL: Int, KL: Int, SCORES: Int](
    out_t: Tile[DT, SCORES],
    src: Tile[DT, SCORES],
):
    """`(BH, QL, KL)` -> `(BH, KL, QL)`."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SCORES:
        return
    var j = idx % KL
    var rem = idx // KL
    var i = rem % QL
    var bh = rem // QL
    out_t.ptr[unsafe_offset=bh * KL * QL + j * QL + i] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset=idx]
    )


def _xa_zero_kernel[N: Int](
    g: Tile[DT, N]
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < N:
        g.ptr[unsafe_offset=i] = Scalar[DT](0.0)


# +--------------------------------------------------------------------------+ #
# | Fused forward (inference only) — no scores, no cache, no packs
# +--------------------------------------------------------------------------+ #
#
# ⚠⚠ WHY A SECOND FORWARD. At SigLIP's shape (12 heads x 1024 x 1024) every
# layer writes 12.6 M scores (50 MB), reads them for the softmax, writes 12.6 M
# weights and reads those again for A.V — ~250 MB of LPDDR traffic per layer on
# a board with ~100 GB/s to spend, on top of three pack/unpack passes. After
# the two-pass softmax and the tiled matmuls (§2 of the optimisation notes)
# that path is 8.6 ms per layer, 206 ms of a SmolVLA query, and it is all
# memory traffic: the arithmetic is 3.2 GFLOP, 1.3 ms at the Orin's fp32 peak.
#
# The fused kernel keeps the softmax ONLINE (a running max and sum per query
# row, rescaled when the max rises) and never materialises a score: one block
# per (batch, head, 64-query tile), one thread per query row holding its q and
# its 64-wide context accumulator in registers, K and V streamed through
# shared memory 32 keys at a time and read as broadcasts. It reads q/k/v in
# their token-major layout directly and writes the output the same way, so
# the three pack kernels and the unpack go too.
#
# ⚠ IT BREAKS THIS PRIMITIVE'S CONTRACT ON PURPOSE. `self.attn` must hold the
# softmax weights after a forward because `vjp` reads them; this path writes
# no weights, so it is OFF by default and `vjp` after a fused forward RAISES.
# The switch is `set_attr["fused_attention"](1.0)`, reachable through every
# container (`ComputeGraph`/`Sequential`/`Repeat`/`Residual`/`Tokenwise` all
# forward `set_attr`), and the SmolVLA deploy turns it on for the frozen
# vision tower — the fine-tune leaves it off, so its vision cache keeps the
# two-pass kernel's bits and existing caches stay valid.
#
# ⚠ NOT BIT-IDENTICAL to the two-pass path: the denominator accumulates in
# key order with rescales, the two-pass one sums exp(s - max) once. Both are
# gated against the CPU leaf in `test_cross_attention_gpu_shapes.mojo` and
# against float64 in `benchmarks/cross_attention_bench.mojo` (variant F).
#
# ⚠⚠ ON NVIDIA, AT HEAD DIM 64, UNMASKED, THE FUSED FLAG ROUTES TO MAX'S OWN
# FLASH ATTENTION (`nn.attention.gpu.mha`, the FA2 kernel) — see
# `xa_fused_routes_to_max` and `_forward_gpu_max_fa`. Our token-major slab
# IS the BSHD layout it takes, so it gets our buffers with no pack. Orin, the
# SigLIP shape (B1, 12 heads, 1024 x 1024, HD 64), 21 Sep 2026:
#
#     two-pass path (A)                 8.42 ms
#     our fused kernel, row N           4.13
#     MAX `flash_attention` on its own 34.5      (naive fallback: its depth-64
#                                                 admission is by GPU NAME —
#                                                 A100/H100/B200 — and the
#                                                 Orin is `OrinNano`)
#     MAX FA2 kernel, dispatch forced   1.26      6.7x   <- routed
#
# It runs both matmuls on the tensor cores as TF32 (16x8x8 mma, 10 mantissa
# bits per operand): 3.7e-3 std units against float64 at that shape, where
# the fp32 kernels sit at ~7e-6. That is the band the network's GEMMs
# already run in on CUDA, and the gates carry a TF32 band for exactly the
# shapes this routes (`test_cross_attention_gpu_shapes.mojo`, bench row F).
# `XA_FUSED_VIA_MAX = False` puts our kernel back everywhere for a bisect;
# the deploy's `--no-fused-vision` is the runtime fallback.

comptime XA_FUSED_BQ: Int = 128
"""Query rows per block of the fused forward."""
comptime XA_FUSED_BK: Int = 32
"""Keys per shared-memory tile of the fused forward."""
comptime XA_FUSED_R: Int = 1
"""Query rows per THREAD (shipped default).

`benchmarks/cross_attention_bench.mojo` ranks the (R, SPLIT, KU, DOT_FMA, EXP2) grid on the Orin; the defaults are
its row N, the best at SigLIP's shape on 20 Sep 2026:

    two-pass path (A)                      8.45 ms
    R4 S4 KU1, while-loop loader           6.97      (the first shipped cut)
    R1 S2 KU1                              6.27
    R4 S4 KU1, unrolled loader             5.57
    R4 S4 KU4                              4.50
    R4 S4 KU4 + FMA dot                    4.41
    R4 S4 KU4 + FMA + exp2                 4.25
    R1 S1 KU4 + FMA + exp2   (SHIPS)       4.11      2.06x

Each knob bought its share and none was the bound alone: the kernel is
issue-bound, and what is left is a register-tiled inner loop."""
comptime XA_FUSED_KU: Int = 4
"""Keys per inner step (shipped default): scores for KU keys are formed and
shuffled together before one softmax update, which breaks the per-key
dependency chain (load -> dot -> 2 shuffles -> exp -> accumulate) that a
few resident warps cannot hide."""
comptime XA_FUSED_DOT_FMA: Bool = True
"""Dot product as a chain of fused multiply-adds (HW instructions) instead
of a multiply and a reduce tree (2*HW - 1). Shipped default follows the
board."""
comptime XA_FUSED_EXP2: Bool = True
"""Softmax in base 2: scores pre-scaled by log2(e) once, weights by `exp2`
(one MUFU op on NVIDIA, against ~25 instructions for the libm-accurate
`exp`). Mathematically the same softmax; the rounding differs and the
float64 gate in `benchmarks/cross_attention_bench.mojo` is the band."""
comptime XA_LOG2E: Scalar[DT] = Scalar[DT](1.4426950408889634)
comptime XA_FUSED_VIA_MAX: Bool = True
"""Route the fused forward to MAX's FA2 kernel where `xa_fused_routes_to_max`
admits it. Off = our `_xa_fused_kernel` everywhere (the bisect switch)."""


def xa_fused_routes_to_max[MASKED: Bool, HD: Int]() -> Bool:
    """Whether the fused forward at this shape is MAX's FA2 kernel: NVIDIA,
    no key-padding mask (MAX would need a `MaterializedMask` we do not build)
    and head dim 64 (its fp32 config pads the head dim to 64 and its own
    table admits nothing narrower on Ampere). The gates and the benchmark
    read this to pick the TF32 band for exactly these shapes."""
    comptime if not XA_FUSED_VIA_MAX:
        return False
    else:
        return default_accelerator().is_nvidia_gpu() and (not MASKED) and HD == 64


def _xa_fused_split[HD: Int]() -> Int:
    """Lanes per query-row group (shipped default): ONE — every warp
    instruction then serves 32 rows and no softmax bookkeeping is
    replicated; the Orin preferred it to 2 and 4 (row N vs J/M above)."""
    return 1


def xa_fused_block[R: Int, SPLIT: Int]() -> Int:
    """Threads per block of `_xa_fused_kernel`: BQ/R row groups x lanes."""
    return (XA_FUSED_BQ // R) * SPLIT


def _xa_fused_kernel[
    B: Int, DIM: Int, NH: Int, QL: Int, KL: Int, HD: Int, MASKED: Bool,
    R: Int, SPLIT: Int, KU: Int, DOT_FMA: Bool = False, EXP2: Bool = False,
](
    q: Tile[DT, B, QL * DIM],
    k: Tile[DT, B, KL * DIM],
    v: Tile[DT, B, KL * DIM],
    m: Tile[DT, B, KL],
    dst: Tile[DT, B, QL * DIM],
):
    """grid (ceil(QL/BQ), NH, B), block `xa_fused_block[R, SPLIT]()`. `m` is
    read only when MASKED (the unmasked instantiation is handed any tensor
    of that layout).

    Thread layout: `SPLIT` adjacent lanes share R consecutive query rows,
    each lane owning HD/SPLIT of the head for all of them (q and context in
    registers). Per inner step KU keys' K halves are fetched, the R x KU
    partial dot products meet their partners in log2(SPLIT) shuffle steps
    each, then each row does ONE max update over its KU scores and KU
    weighted V accumulations. A masked key (or a tile-tail slot past KL) is
    a key with weight ZERO, kept out of the max — the two-pass path's
    explicit zero weight — so a fully masked row ends at l = 0 and a zero
    context.

    ⚠ EVERY LANE RUNS THE WHOLE LOOP, dead rows included (their loads are
    clamped to a real row and only their store is skipped): the shuffle is
    a warp-synchronous instruction and a partner lane that skipped the loop
    would leave it undefined.
    """
    comptime BQ = XA_FUSED_BQ
    comptime BK = XA_FUSED_BK
    comptime HW = HD // SPLIT
    comptime NT = (BQ // R) * SPLIT
    comptime assert HD & (HD - 1) == 0, (
        "_xa_fused_kernel: HEAD_DIM must be a power of two (a SIMD width)"
    )
    comptime assert SPLIT == 1 or SPLIT == 2 or SPLIT == 4, (
        "_xa_fused_kernel: SPLIT is 1, 2 or 4 lanes"
    )
    comptime assert BK % KU == 0 and BQ % R == 0, (
        "_xa_fused_kernel: KU must divide BK and R must divide BQ"
    )
    comptime TILE = BK * HD

    var ks = shared_tile[DT, TILE]()
    var vs = shared_tile[DT, TILE]()
    var ms = shared_tile[DT, BK]()

    var qt = Int(block_idx.x)
    var h = Int(block_idx.y)
    var b = Int(block_idx.z)
    var tid = Int(thread_idx.x)
    var part = tid % SPLIT
    var r0 = qt * BQ + (tid // SPLIT) * R
    # With EXP2 the scores carry log2(e) as well, so the softmax's exp(x)
    # becomes exp2(x') with x' = x * log2(e) — the same weights.
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    comptime if EXP2:
        scale = scale * XA_LOG2E
    var qbase = b * (QL * DIM) + h * HD + part * HW

    var qr = Array[SIMD[DT, HW], R](fill=SIMD[DT, HW](0))
    var o = Array[SIMD[DT, HW], R](fill=SIMD[DT, HW](0))
    var mx = Array[Scalar[DT], R](fill=XATTN_MASK_NEG)
    var l = Array[Scalar[DT], R](fill=Scalar[DT](0))
    comptime for r in range(R):
        var i = r0 + r
        if i >= QL:
            i = QL - 1
        qr[r] = q.ptr.unsafe_load[width=HW](qbase + i * DIM)

    for j0 in range(0, KL, BK):
        # Tile load: element e strided by the block, so a warp reads 32
        # consecutive floats of one key row — coalesced — and stores them
        # to the same offsets in shared memory.
        # Unrolled at compile time so every global load of the tile is
        # issued before the first store waits on one: as a `while` loop the
        # 32 loads per thread were a chain of ~500-cycle LPDDR latencies
        # under a barrier, the same in every configuration of this kernel.
        comptime for it in range((TILE + NT - 1) // NT):
            var e = it * NT + tid
            if e < TILE:
                var jj = e // HD
                var d = e % HD
                var j = j0 + jj
                if j < KL:
                    var src = b * (KL * DIM) + j * DIM + h * HD + d
                    ks.ptr[unsafe_offset=e] = rebind[Scalar[DT]](
                        k.ptr[unsafe_offset=src]
                    )
                    vs.ptr[unsafe_offset=e] = rebind[Scalar[DT]](
                        v.ptr[unsafe_offset=src]
                    )
        comptime if MASKED:
            if tid < BK:
                var j = j0 + tid
                ms.ptr[unsafe_offset=tid] = (
                    rebind[Scalar[DT]](m.ptr[unsafe_offset=b * KL + j])
                    if j < KL else Scalar[DT](0)
                )
        barrier()
        var nk = KL - j0
        if nk > BK:
            nk = BK
        for jb in range(0, nk, KU):
            # ── scores for KU keys x R rows, then the partner reduction ──
            var sc = Array[Scalar[DT], R * KU](fill=Scalar[DT](0))
            var ok = Array[Bool, KU](fill=False)
            comptime for u in range(KU):
                var jj = jb + u
                var okay = jj < nk
                comptime if MASKED:
                    if okay and rebind[Scalar[DT]](
                        ms.ptr[unsafe_offset=jj]
                    ) < Scalar[DT](0.5):
                        okay = False
                ok[u] = okay
                var jr = jj if jj < BK else BK - 1
                var kh = ks.ptr.unsafe_load[width=HW](jr * HD + part * HW)
                comptime for r in range(R):
                    comptime if DOT_FMA:
                        var acc = Scalar[DT](0)
                        comptime for d in range(HW):
                            acc = fma(qr[r][d], kh[d], acc)
                        sc[r * KU + u] = acc
                    else:
                        sc[r * KU + u] = (qr[r] * kh).reduce_add()
            comptime if SPLIT >= 2:
                comptime for n in range(R * KU):
                    sc[n] = sc[n] + warp.shuffle_xor(sc[n], UInt32(1))
            comptime if SPLIT >= 4:
                comptime for n in range(R * KU):
                    sc[n] = sc[n] + warp.shuffle_xor(sc[n], UInt32(2))
            # ── one softmax update per row over its KU scores ────────────
            comptime for r in range(R):
                var mnew = mx[r]
                comptime for u in range(KU):
                    var sv = sc[r * KU + u] * scale
                    sc[r * KU + u] = sv
                    if ok[u] and sv > mnew:
                        mnew = sv
                if mnew > mx[r]:
                    # exp(old - new) is 0 on the first live key (old is
                    # MASK_NEG), which is the zero start.
                    var c: Scalar[DT]
                    comptime if EXP2:
                        c = exp2(mx[r] - mnew)
                    else:
                        c = exp(mx[r] - mnew)
                    l[r] = l[r] * c
                    o[r] = o[r] * c
                    mx[r] = mnew
            comptime for u in range(KU):
                var jj = jb + u
                var jr = jj if jj < BK else BK - 1
                var vh = vs.ptr.unsafe_load[width=HW](jr * HD + part * HW)
                comptime for r in range(R):
                    var pw = Scalar[DT](0)
                    if ok[u]:
                        comptime if EXP2:
                            pw = exp2(sc[r * KU + u] - mx[r])
                        else:
                            pw = exp(sc[r * KU + u] - mx[r])
                    l[r] = l[r] + pw
                    o[r] = o[r] + pw * vh
        barrier()

    # Floored like the two-pass path: a fully masked row has l = 0 and
    # yields a zero context, not a NaN.
    comptime for r in range(R):
        var i = r0 + r
        if i < QL:
            var inv = Scalar[DT](0)
            if l[r] > XATTN_DENOM_FLOOR:
                inv = Scalar[DT](1.0) / l[r]
            dst.ptr.unsafe_store(qbase + i * DIM, o[r] * inv)


struct CrossAttention[
    DIM: Int,
    N_HEADS: Int,
    Q_LEN: Int,
    KV_LEN: Int,
    MASKED: Bool = False,
](Module):
    comptime ARITY: Int = 4 if Self.MASKED else 3
    comptime HEAD_DIM: Int = Self.DIM // Self.N_HEADS
    comptime Q_DIM: Int = Self.Q_LEN * Self.DIM
    comptime KV_DIM: Int = Self.KV_LEN * Self.DIM
    comptime IN_DIMS = _xattn_in_dims[
        Self.ARITY, Self.Q_DIM, Self.KV_DIM, Self.KV_LEN
    ]()
    comptime OUT_DIM: Int = Self.Q_DIM

    comptime ATTN_SIZE: Int = Self.N_HEADS * Self.Q_LEN * Self.KV_LEN
    """Per-sample softmax weights — the only thing worth caching. q/k/v come
    back through `forward_input`."""

    var attn: Tensor

    # GPU scratch. Three query-width slabs, three key/value-width slabs and two
    # score slabs — enough for the backward pass to hold pdout, pq, pk, pv, dQ,
    # dK, dV and two of {dattn, dscore, attn_T, dscore_T} at once. Lazily sized;
    # unused on CPU (which uses local Lists, as the self-attention leaf does).
    var sq0: Tensor
    var sq1: Tensor
    var sq2: Tensor
    var sk0: Tensor
    var sk1: Tensor
    var sk2: Tensor
    var ss0: Tensor
    var ss1: Tensor
    var sst: Tensor
    """(max, 1/denominator) per softmax row — the forward's two-pass softmax."""
    var fused: Bool
    """GPU forward through `_xa_fused_kernel`: no scores, no packs and NO
    `attn` cache — inference only, `vjp` raises. `set_attr["fused_attention"]`.
    The CPU path ignores it."""

    def __init__(out self):
        comptime assert Self.DIM % Self.N_HEADS == 0, (
            "CrossAttention: DIM must be divisible by N_HEADS"
        )
        comptime assert Self.Q_LEN > 0 and Self.KV_LEN > 0, (
            "CrossAttention: sequence lengths must be positive"
        )
        self.attn = Tensor()
        self.sq0 = Tensor()
        self.sq1 = Tensor()
        self.sq2 = Tensor()
        self.sk0 = Tensor()
        self.sk1 = Tensor()
        self.sk2 = Tensor()
        self.ss0 = Tensor()
        self.ss1 = Tensor()
        self.sst = Tensor()
        self.fused = False

    def __init__(out self, *, deinit move: Self):
        self.attn = move.attn^
        self.sq0 = move.sq0^
        self.sq1 = move.sq1^
        self.sq2 = move.sq2^
        self.sk0 = move.sk0^
        self.sk1 = move.sk1^
        self.sk2 = move.sk2^
        self.ss0 = move.ss0^
        self.ss1 = move.ss1^
        self.sst = move.sst^
        self.fused = move.fused

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`fused_attention` != 0 selects the fused GPU forward (see the
        kernel's header for what that gives up). Other attrs are ignored."""
        comptime if ATTR == "fused_attention":
            self.fused = value != Scalar[DT](0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "CrossAttention: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error("CrossAttention.make[target='gpu']: ctx required")
        return Self()

    def _ensure_scratch_gpu[B: Int](mut self, c: DeviceContext) raises:
        comptime PQ = B * Self.N_HEADS * Self.Q_LEN * Self.HEAD_DIM
        comptime PK = B * Self.N_HEADS * Self.KV_LEN * Self.HEAD_DIM
        comptime SC = B * Self.N_HEADS * Self.Q_LEN * Self.KV_LEN
        self.sq0.ensure_gpu(c, PQ)
        self.sq1.ensure_gpu(c, PQ)
        self.sq2.ensure_gpu(c, PQ)
        self.sk0.ensure_gpu(c, PK)
        self.sk1.ensure_gpu(c, PK)
        self.sk2.ensure_gpu(c, PK)
        self.ss0.ensure_gpu(c, SC)
        self.ss1.ensure_gpu(c, SC)
        self.sst.ensure_gpu(c, B * Self.N_HEADS * Self.Q_LEN * 2)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target != "cpu":
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            if self.fused:
                # No cache: the 50 MB slab at SigLIP's shape is never sized.
                comptime if xa_fused_routes_to_max[Self.MASKED, Self.HEAD_DIM]():
                    self._forward_gpu_max_fa[B](inputs, out, c)
                else:
                    self._forward_gpu_fused[B](inputs, out, c)
                return
            self.attn.ensure_gpu(c, B * Self.ATTN_SIZE)
            self._forward_gpu[B](inputs, out, c)
            return

        out.ensure(B * Self.OUT_DIM)
        self.attn.ensure(B * Self.ATTN_SIZE)

        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PACK_Q = BH * QL * HD
        comptime PACK_KV = BH * KL * HD
        comptime SCORES = BH * QL * KL
        var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))

        ref qp = inputs[0].data
        ref kp = inputs[1].data
        ref vp = inputs[2].data
        ref op = out.data
        ref ap = self.attn.data

        var pq = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var pout = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var sc = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. token-major -> head-major (BH, LEN, HEAD_DIM).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        pq[bh * QL * HD + t * HD + d] = qp[
                            b * Self.Q_DIM + t * Self.DIM + hoff + d
                        ]
                for t in range(KL):
                    for d in range(HD):
                        var src = b * Self.KV_DIM + t * Self.DIM + hoff + d
                        var dst = bh * KL * HD + t * HD + d
                        pk[dst] = kp[src]
                        pv[dst] = vp[src]

        # 2. scores = Q @ Kᵀ  (BH, Q_LEN, KV_LEN).
        var sc_tt = TileTensor(sc, row_major[BH, QL, KL]())
        var pq_tt = TileTensor(pq, row_major[BH, QL, HD]())
        var pk_tt = TileTensor(pk, row_major[BH, KL, HD]())
        batched_matmul[transpose_b=True, target="cpu"](sc_tt, pq_tt, pk_tt)

        # 3. scale, additive key-padding bias, stable softmax. Weights land in
        #    BOTH `sc` (for the attn·V product) and the cache (for backward).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sbase = bh * QL * KL
                var abase = b * Self.ATTN_SIZE + h * QL * KL
                for i in range(QL):
                    var row = sbase + i * KL
                    var arow = abase + i * KL
                    var mx = XATTN_MASK_NEG
                    for j in range(KL):
                        var s = sc[row + j] * scale
                        comptime if Self.MASKED:
                            if inputs[3].data[b * KL + j] < Scalar[DT](0.5):
                                s = XATTN_MASK_NEG
                        sc[row + j] = s
                        if s > mx:
                            mx = s
                    var se = Scalar[DT](0)
                    for j in range(KL):
                        var e = exp(sc[row + j] - mx)
                        sc[row + j] = e
                        se += e
                    # Floored: a row whose every key is masked would otherwise
                    # be 0/0. exp(MASK_NEG - MASK_NEG) == 1 for such a row, so
                    # the floor is only load-bearing under fp underflow, but a
                    # NaN that surfaces one training run in ten is not a
                    # tradeoff worth taking for one comparison.
                    var denom = se if se > XATTN_DENOM_FLOOR else (
                        XATTN_DENOM_FLOOR
                    )
                    var inv = Scalar[DT](1) / denom
                    for j in range(KL):
                        var w = sc[row + j] * inv
                        comptime if Self.MASKED:
                            if inputs[3].data[b * KL + j] < Scalar[DT](0.5):
                                w = Scalar[DT](0)
                        sc[row + j] = w
                        ap[arow + j] = w

        # 4. pout = attn @ V  (BH, Q_LEN, HEAD_DIM).
        var pout_tt = TileTensor(pout, row_major[BH, QL, HD]())
        var pv_tt = TileTensor(pv, row_major[BH, KL, HD]())
        batched_matmul[target="cpu"](pout_tt, sc_tt, pv_tt)

        # 5. head-major -> token-major.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        op[b * Self.Q_DIM + t * Self.DIM + hoff + d] = pout[
                            bh * QL * HD + t * HD + d
                        ]
        _ = pq^
        _ = pk^
        _ = pv^
        _ = pout^
        _ = sc^

    # ── Backward ─────────────────────────────────────────────────────────

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if self.fused:
            raise Error(
                "CrossAttention.vjp after a FUSED forward: the softmax weights"
                " were never materialised. `set_attr[\"fused_attention\"](0)`"
                " before any forward whose gradient you need — the fused path"
                " is inference-only."
            )
        comptime if target != "cpu":
            self._vjp_gpu[B](
                forward_input, grad_output, grad_inputs, ctx.value()
            )
            return

        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PACK_Q = BH * QL * HD
        comptime PACK_KV = BH * KL * HD
        comptime SCORES = BH * QL * KL
        var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))

        ref gq = grad_inputs[0]
        ref gk = grad_inputs[1]
        ref gv = grad_inputs[2]
        gq.ensure(B * Self.Q_DIM)
        gk.ensure(B * Self.KV_DIM)
        gv.ensure(B * Self.KV_DIM)

        ref qp = forward_input[0].data
        ref kp = forward_input[1].data
        ref vp = forward_input[2].data
        ref gop = grad_output.data
        ref ap = self.attn.data

        var pdout = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var pq = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var dQ = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var dK = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var dV = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var dattn = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var attn_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. pack grad_output + q/k/v into head-major.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        var idx = bh * QL * HD + t * HD + d
                        var col = b * Self.Q_DIM + t * Self.DIM + hoff + d
                        pdout[idx] = gop[col]
                        pq[idx] = qp[col]
                for t in range(KL):
                    for d in range(HD):
                        var idx = bh * KL * HD + t * HD + d
                        var col = b * Self.KV_DIM + t * Self.DIM + hoff + d
                        pk[idx] = kp[col]
                        pv[idx] = vp[col]

        # 2. dattn = dout @ Vᵀ  (BH, Q_LEN, KV_LEN).
        var dattn_tt = TileTensor(dattn, row_major[BH, QL, KL]())
        var pdout_tt = TileTensor(pdout, row_major[BH, QL, HD]())
        var pv_tt = TileTensor(pv, row_major[BH, KL, HD]())
        batched_matmul[transpose_b=True, target="cpu"](
            dattn_tt, pdout_tt, pv_tt
        )

        # 3. softmax JVP: dscore = scale * a * (dattn - Σ_k a_k·dattn_k).
        #    Masked columns have a == 0, so dscore is zero there without a
        #    second mask read — the mask's own gradient is zero (it is data,
        #    not a parameter) and grad_inputs[3] is filled with zeros below.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sbase = bh * QL * KL
                var abase = b * Self.ATTN_SIZE + h * QL * KL
                for i in range(QL):
                    var row = sbase + i * KL
                    var arow = abase + i * KL
                    var s = Scalar[DT](0)
                    for j in range(KL):
                        s += ap[arow + j] * dattn[row + j]
                    for j in range(KL):
                        var a = ap[arow + j]
                        var ds = scale * a * (dattn[row + j] - s)
                        dscore[row + j] = ds
                        attn_T[sbase + j * QL + i] = a
                        dscore_T[sbase + j * QL + i] = ds

        # 4. dV = attnᵀ @ dout   (BH, KV_LEN, HEAD_DIM).
        var attnT_tt = TileTensor(attn_T, row_major[BH, KL, QL]())
        var dV_tt = TileTensor(dV, row_major[BH, KL, HD]())
        batched_matmul[target="cpu"](dV_tt, attnT_tt, pdout_tt)

        # 5. dK = dscoreᵀ @ Q    (BH, KV_LEN, HEAD_DIM).
        var dscoreT_tt = TileTensor(dscore_T, row_major[BH, KL, QL]())
        var pq_tt = TileTensor(pq, row_major[BH, QL, HD]())
        var dK_tt = TileTensor(dK, row_major[BH, KL, HD]())
        batched_matmul[target="cpu"](dK_tt, dscoreT_tt, pq_tt)

        # 6. dQ = dscore @ K     (BH, Q_LEN, HEAD_DIM).
        var dscore_tt = TileTensor(dscore, row_major[BH, QL, KL]())
        var pk_tt = TileTensor(pk, row_major[BH, KL, HD]())
        var dQ_tt = TileTensor(dQ, row_major[BH, QL, HD]())
        batched_matmul[target="cpu"](dQ_tt, dscore_tt, pk_tt)

        # 7. unpack.
        for i in range(B * Self.Q_DIM):
            gq.data[i] = Scalar[DT](0)
        for i in range(B * Self.KV_DIM):
            gk.data[i] = Scalar[DT](0)
            gv.data[i] = Scalar[DT](0)
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        gq.data[
                            b * Self.Q_DIM + t * Self.DIM + hoff + d
                        ] = dQ[bh * QL * HD + t * HD + d]
                for t in range(KL):
                    for d in range(HD):
                        var col = b * Self.KV_DIM + t * Self.DIM + hoff + d
                        var idx = bh * KL * HD + t * HD + d
                        gk.data[col] = dK[idx]
                        gv.data[col] = dV[idx]

        comptime if Self.MASKED:
            # The mask is data, not a differentiable input. Zero rather than
            # leave the caller's reused grad slot holding a previous node's
            # values — an accumulating graph would add them in.
            ref gm = grad_inputs[3]
            gm.ensure(B * KL)
            for i in range(B * KL):
                gm.data[i] = Scalar[DT](0)

        _ = pdout^
        _ = pq^
        _ = pk^
        _ = pv^
        _ = dQ^
        _ = dK^
        _ = dV^
        _ = dattn^
        _ = dscore^
        _ = attn_T^
        _ = dscore_T^

    # ── GPU bodies ───────────────────────────────────────────────────────

    def _forward_gpu_fused[
        B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        c: DeviceContext,
    ) raises:
        """One launch, straight from the token-major inputs to the
        token-major output. Nothing is packed and nothing is cached."""
        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime lay_q = RowMajor[B, Self.Q_DIM]
        comptime lay_kv = RowMajor[B, Self.KV_DIM]
        comptime lay_m = RowMajor[B, KL]
        comptime SPLIT = _xa_fused_split[Self.HEAD_DIM]()
        comptime kern = _xa_fused_kernel[
            B, Self.DIM, Self.N_HEADS, QL, KL, Self.HEAD_DIM, Self.MASKED,
            XA_FUSED_R, SPLIT, XA_FUSED_KU, XA_FUSED_DOT_FMA, XA_FUSED_EXP2,
        ]
        comptime qtiles = (QL + XA_FUSED_BQ - 1) // XA_FUSED_BQ
        comptime blk = xa_fused_block[XA_FUSED_R, SPLIT]()
        ref q = inputs[0]
        ref k = inputs[1]
        ref v = inputs[2]
        comptime if Self.MASKED:
            ref m = inputs[3]
            c.enqueue_function[kern](
                q.tt["gpu", B, Self.Q_DIM](), k.tt["gpu", B, Self.KV_DIM](),
                v.tt["gpu", B, Self.KV_DIM](), m.tt["gpu", B, KL](),
                out.tt["gpu", B, Self.Q_DIM](),
                grid_dim=(qtiles, Self.N_HEADS, B),
                block_dim=blk,
            )
        else:
            # The mask slot is never read; hand it the output as a stand-in
            # of the right layout, as `_forward_gpu` does with the cache.
            c.enqueue_function[kern](
                q.tt["gpu", B, Self.Q_DIM](), k.tt["gpu", B, Self.KV_DIM](),
                v.tt["gpu", B, Self.KV_DIM](),
                rebind[TileTensor[DT, lay_m, MutAnyOrigin]](
                    out.tt["gpu", B, KL]()
                ),
                out.tt["gpu", B, Self.Q_DIM](),
                grid_dim=(qtiles, Self.N_HEADS, B),
                block_dim=blk,
            )

    def _forward_gpu_max_fa[
        B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        c: DeviceContext,
    ) raises:
        """MAX's FA2 flash attention over our slabs UNCHANGED: `[B, L*DIM]`
        with the heads contiguous inside DIM is the BSHD it takes, so the
        views below reinterpret, they do not copy. `flash_attention_dispatch`
        rather than `flash_attention` because the latter's depth table admits
        depth 64 by GPU name and falls back to a naive two-BMM kernel on the
        Orin (34.5 ms against 1.26 ms). No cache, `vjp` raises, as for the
        other fused path. TF32 matmuls — see the header."""
        comptime if not xa_fused_routes_to_max[Self.MASKED, Self.HEAD_DIM]():
            raise Error(
                "CrossAttention: MAX's flash attention is not routed at this"
                " shape (NVIDIA, unmasked, head dim 64 only)"
            )
        else:
            comptime H = Self.N_HEADS
            comptime HD = Self.HEAD_DIM
            comptime QL = Self.Q_LEN
            comptime KL = Self.KV_LEN
            # ⚠ `.unsafe_ptr()` severs the borrow: the inputs are the
            # caller's tensors and `out` is resident; both outlive the
            # launch, which the caller synchronises before reading.
            var qp = inputs[0].dev.value().unsafe_ptr().as_imm().as_unsafe_any_origin()
            var kp = inputs[1].dev.value().unsafe_ptr().as_imm().as_unsafe_any_origin()
            var vp = inputs[2].dev.value().unsafe_ptr().as_imm().as_unsafe_any_origin()
            var q_tt = TileTensor(qp, row_major((B, QL, Idx[H], Idx[HD])))
            var k_tt = TileTensor(kp, row_major((B, KL, Idx[H], Idx[HD])))
            var v_tt = TileTensor(vp, row_major((B, KL, Idx[H], Idx[HD])))
            var o_tt = TileTensor(
                out.dev.value(), row_major((B, QL, Idx[H], Idx[HD]))
            )
            var scale = Float32(1.0 / sqrt(Float64(HD)))
            flash_attention_dispatch[
                kv_num_heads=H,
                ragged=False,
                sink=False,
                _is_flash_attention_applicable=True,
                _is_cache_length_accurate=True,
                _use_valid_length=False,
            ](
                o_tt,
                q_tt,
                TileTensorMHAOperand(k_tt),
                TileTensorMHAOperand(v_tt),
                NullMask(),
                QL,
                KL,
                scale,
                False,
                c,
            )

    def _forward_gpu[
        B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        c: DeviceContext,
    ) raises:
        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PQ = BH * QL * HD
        comptime PK = BH * KL * HD
        comptime SC = BH * QL * KL
        self._ensure_scratch_gpu[B](c)

        comptime lay_q = RowMajor[B, Self.Q_DIM]
        comptime lay_kv = RowMajor[B, Self.KV_DIM]
        comptime lay_pq = RowMajor[PQ]
        comptime lay_pk = RowMajor[PK]
        comptime lay_s = RowMajor[SC]
        comptime lay_a = RowMajor[B, Self.ATTN_SIZE]
        comptime lay_m = RowMajor[B, KL]

        ref q = inputs[0]
        ref k = inputs[1]
        ref v = inputs[2]

        # 1. pack q -> sq0, k -> sk0, v -> sk1  (head-major).
        comptime qblocks = (B * QL * Self.DIM + TPB - 1) // TPB
        comptime kblocks = (B * KL * Self.DIM + TPB - 1) // TPB
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            self.sq0.tt["gpu", PQ](),
            q.tt["gpu", B, Self.Q_DIM](),
            grid_dim=qblocks, block_dim=TPB,
        )
        # k packs STRAIGHT INTO Kt (BH, HD, KL) — one pass, not pack + transpose.
        c.enqueue_function[
            _xa_pack_kt_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk0.tt["gpu", PK](),
            k.tt["gpu", B, Self.KV_DIM](),
            grid_dim=(PK + TPB - 1) // TPB, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk1.tt["gpu", PK](),
            v.tt["gpu", B, Self.KV_DIM](),
            grid_dim=kblocks, block_dim=TPB,
        )

        # 2. scores(ss0) = Q @ Kt   (BH, QL, KL). Kt is already contiguous, so
        #    NOT `bmm[transpose_b=True]`: that call is 2.06x slower on the Orin
        #    for the same bits (`cross_attention_bench.mojo`). And not `bmm`
        #    at all: both attention products miss MAX's multistage GEMM and
        #    take its vendor path — `bmm_tiled` is 3.92x here on the board.
        bmm_tiled[BH=BH, M=QL, N=KL, K=HD](
            self.ss0.dev.value(), self.sq0.dev.value(), self.sk0.dev.value(), c
        )

        # 3. scale + mask + stable softmax -> the cache, in two passes: row
        #    stats, then one thread per weight. `_xa_element_softmax_kernel`
        #    says why the one-block-per-(b,h) kernel was worth replacing. The
        #    mask slot is only read when MASKED; the unmasked instantiation
        #    still needs SOME tensor for the parameter, so the cache is passed
        #    as an inert stand-in rather than allocating a dummy.
        comptime ROWS = BH * QL
        comptime lay_st = RowMajor[ROWS * 2]
        comptime st = _xa_row_stats_kernel[
            B, Self.N_HEADS, QL, KL, HD, Self.MASKED, SC, ROWS
        ]
        comptime el = _xa_element_softmax_kernel[
            B, Self.N_HEADS, QL, KL, HD, Self.MASKED, SC, ROWS
        ]
        comptime rowblocks = (ROWS + TPB - 1) // TPB
        comptime elblocks = (SC + TPB - 1) // TPB
        comptime if Self.MASKED:
            ref m = inputs[3]
            c.enqueue_function[st](
                self.ss0.tt["gpu", SC](),
                self.sst.tt["gpu", ROWS * 2](),
                m.tt["gpu", B, KL](),
                grid_dim=rowblocks, block_dim=TPB,
            )
            c.enqueue_function[el](
                self.ss0.tt["gpu", SC](),
                self.sst.tt["gpu", ROWS * 2](),
                m.tt["gpu", B, KL](),
                self.attn.tt["gpu", SC](),
                grid_dim=elblocks, block_dim=TPB,
            )
        else:
            c.enqueue_function[st](
                self.ss0.tt["gpu", SC](),
                self.sst.tt["gpu", ROWS * 2](),
                rebind[TileTensor[DT, lay_m, MutAnyOrigin]](
                    self.attn.tt["gpu", B, KL]()
                ),
                grid_dim=rowblocks, block_dim=TPB,
            )
            c.enqueue_function[el](
                self.ss0.tt["gpu", SC](),
                self.sst.tt["gpu", ROWS * 2](),
                rebind[TileTensor[DT, lay_m, MutAnyOrigin]](
                    self.attn.tt["gpu", B, KL]()
                ),
                self.attn.tt["gpu", SC](),
                grid_dim=elblocks, block_dim=TPB,
            )

        # 4. pout(sq1) = attn @ V(sk1). The weights only exist in the cache
        #    now — ss0 still holds the raw, unscaled scores.
        bmm_tiled[BH=BH, M=QL, N=HD, K=KL](
            self.sq1.dev.value(), self.attn.dev.value(), self.sk1.dev.value(), c
        )

        # 5. unpack -> token-major output.
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            out.tt["gpu", B, Self.Q_DIM](),
            self.sq1.tt["gpu", PQ](),
            grid_dim=qblocks, block_dim=TPB,
        )

    def _vjp_gpu[
        B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        c: DeviceContext,
    ) raises:
        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PQ = BH * QL * HD
        comptime PK = BH * KL * HD
        comptime SC = BH * QL * KL
        self._ensure_scratch_gpu[B](c)

        comptime lay_q = RowMajor[B, Self.Q_DIM]
        comptime lay_kv = RowMajor[B, Self.KV_DIM]
        comptime lay_pq = RowMajor[PQ]
        comptime lay_pk = RowMajor[PK]
        comptime lay_s = RowMajor[SC]
        comptime lay_a = RowMajor[B, Self.ATTN_SIZE]

        ref q = forward_input[0]
        ref k = forward_input[1]
        ref v = forward_input[2]
        ref gq = grad_inputs[0]
        ref gk = grad_inputs[1]
        ref gv = grad_inputs[2]
        gq.ensure_gpu(c, B * Self.Q_DIM)
        gk.ensure_gpu(c, B * Self.KV_DIM)
        gv.ensure_gpu(c, B * Self.KV_DIM)

        comptime qblocks = (B * QL * Self.DIM + TPB - 1) // TPB
        comptime kblocks = (B * KL * Self.DIM + TPB - 1) // TPB
        comptime sblocks = (SC + TPB - 1) // TPB

        # Slab map. sq0=pdout, sq1=pq, sq2=dQ, sk0=pk (-> dK), sk1=pv, sk2=dV,
        # ss0/ss1 = the score-shaped temporaries. Only sk0 is reused, and only
        # after its last read (step 7), so no ordering hazard: kernels on one
        # stream run in issue order.
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            self.sq0.tt["gpu", PQ](),
            grad_output.tt["gpu", B, Self.Q_DIM](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            self.sq1.tt["gpu", PQ](),
            q.tt["gpu", B, Self.Q_DIM](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk0.tt["gpu", PK](),
            k.tt["gpu", B, Self.KV_DIM](),
            grid_dim=kblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk1.tt["gpu", PK](),
            v.tt["gpu", B, Self.KV_DIM](),
            grid_dim=kblocks, block_dim=TPB,
        )


        # 2. dattn(ss0) = dout @ Vt   (BH, QL, KL).
        bmm[transpose_b=True, A0=BH, A1=QL, A2=HD, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=KL](
            self.ss0.dev.value(), self.sq0.dev.value(), self.sk1.dev.value(), c
        )

        # 3. dscore(ss1) = softmax JVP.
        c.enqueue_function[
            _xa_softmax_jvp_kernel[
                B, Self.N_HEADS, QL, KL, HD, Self.ATTN_SIZE, SC, BH
            ]
        ](
            self.ss1.tt["gpu", SC](),
            self.ss0.tt["gpu", SC](),
            self.attn.tt["gpu", B, Self.ATTN_SIZE](),
            grid_dim=BH, block_dim=TPB,
        )

        # 4. attn_T(ss0) = transpose(cache)  — ss0 free, dattn consumed.
        c.enqueue_function[
            _xa_transpose_attn_kernel[
                B, Self.N_HEADS, QL, KL, Self.ATTN_SIZE, SC
            ]
        ](
            self.ss0.tt["gpu", SC](),
            self.attn.tt["gpu", B, Self.ATTN_SIZE](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 5. dV(sk2) = attn_T(ss0) @ dout(sq0)   (BH, KL, HD).
        bmm[A0=BH, A1=KL, A2=QL, B0=BH, B1=QL, B2=HD, O0=BH, O1=KL, O2=HD](
            self.sk2.dev.value(), self.ss0.dev.value(), self.sq0.dev.value(), c
        )

        # 6. dQ(sq2) = dscore(ss1) @ K(sk0)  — BEFORE sk0 is recycled for dK.
        bmm[A0=BH, A1=QL, A2=KL, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=HD](
            self.sq2.dev.value(), self.ss1.dev.value(), self.sk0.dev.value(), c
        )

        # 7. dscore_T(ss0) = transpose(dscore)  — ss0 free, attn_T read at 5.
        c.enqueue_function[_xa_transpose_scores_kernel[QL, KL, SC]](
            self.ss0.tt["gpu", SC](),
            self.ss1.tt["gpu", SC](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 8. dK(sk0) = dscore_T(ss0) @ Q(sq1)  — sk0 free, pk read at 6.
        bmm[A0=BH, A1=KL, A2=QL, B0=BH, B1=QL, B2=HD, O0=BH, O1=KL, O2=HD](
            self.sk0.dev.value(), self.ss0.dev.value(), self.sq1.dev.value(), c
        )

        # 9. unpack.
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            gq.tt["gpu", B, Self.Q_DIM](),
            self.sq2.tt["gpu", PQ](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            gk.tt["gpu", B, Self.KV_DIM](),
            self.sk0.tt["gpu", PK](),
            grid_dim=kblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            gv.tt["gpu", B, Self.KV_DIM](),
            self.sk2.tt["gpu", PK](),
            grid_dim=kblocks, block_dim=TPB,
        )

        comptime if Self.MASKED:
            # The mask is data, not a parameter. Zeroed rather than left alone:
            # the caller's grad slot is reused across nodes.
            ref gm = grad_inputs[3]
            gm.ensure_gpu(c, B * KL)
            c.enqueue_function[_xa_zero_kernel[B * KL]](
                gm.tt["gpu", B * KL](),
                grid_dim=(B * KL + TPB - 1) // TPB, block_dim=TPB,
            )

    # for_each_param / for_each_state / zero_grad inherit the Module
    # reflection defaults — no `Param` fields.


# ── comptime helper ──────────────────────────────────────────────────────
# `IN_DIMS` is `Array[Int, ARITY]` and ARITY varies with MASKED, so the
# array cannot be written as one literal. Mirrors `concat.mojo`'s `_total_dim`.


def _xattn_in_dims[
    ARITY: Int, Q_DIM: Int, KV_DIM: Int, KV_LEN: Int
]() -> Array[Int, ARITY]:
    var a = Array[Int, ARITY](fill=KV_DIM)
    a[0] = Q_DIM
    comptime if ARITY == 4:
        a[3] = KV_LEN
    return a^


# ── aliases ──────────────────────────────────────────────────────────────

comptime SelfAttentionPos[
    DIM: Int, N_HEADS: Int, SEQ: Int
] = CrossAttention[DIM, N_HEADS, SEQ, SEQ, False]
"""DETR self-attention: `q = k = x + pos`, `v = x`. The positional embedding
reaching q and k but not v is why this cannot be `ScaledDotProductAttention`."""

comptime SelfAttentionPosMasked[
    DIM: Int, N_HEADS: Int, SEQ: Int
] = CrossAttention[DIM, N_HEADS, SEQ, SEQ, True]
"""As above with a per-sample key padding mask — ACT's CVAE encoder, whose
action chunk is zero-padded past the end of an episode."""
