# +--------------------------------------------------------------------------+ #
# | SmolVLA — attention with a static [Q_LEN, KV_LEN] block mask
# +--------------------------------------------------------------------------+ #
"""The one attention shape the denoising step needs and no existing leaf has.

    q [B, Q_LEN*DIM]   k,v [B, KV_LEN*DIM]   mask [Q_LEN, KV_LEN] additive
    out[b,i] = concat_h( softmax_j( q·kᵀ/sqrt(HEAD_DIM) + mask[i,j] ) · v )

`nn` already ships three attention leaves and none fits:

  * `ScaledDotProductAttention` — packed `[Q|K|V]` of ONE sequence.
  * `MaskedAttention` — has the static 2-D mask, but packed and `Q_LEN == KV_LEN`.
    Using it would mean recomputing the prefix's queries, which is what the KV
    cache exists to avoid.
  * `CrossAttention` — separate q/k/v and `Q_LEN != KV_LEN`, but its mask is a
    per-sample KEY PADDING mask `[B, KV_LEN]`. A key is masked for every query or
    for none, so it cannot express causality.

The denoising step needs `Q_LEN != KV_LEN` **and** a query-dependent mask: the 50
action queries attend to the whole cached prefix but only causally among
themselves.

⚠ **Only the SELF (even) layers need this.** A cross (odd) layer's suffix
attends to the entire prefix with nothing masked, so it can use `CrossAttention`
unmasked. Reaching for a mask there would be harmless and pointless; reaching
for `CrossAttention` in the even case would be silently wrong.

## The backward

`vjp` produces dQ, dK and dV. The leaf is still not a `Module` — `Module`'s
`vjp` takes one packed `TensorRefs` whose members share a dimension, and here
q is `[Q_LEN, DIM]` while k and v are `[KV_LEN, DIM]`. It stays hand-driven,
like `DecoderLayerWeights`.

⚠ **It RECOMPUTES the softmax rather than caching them in the forward**, which
is the opposite of `MaskedAttention` and deliberate. One `SmolVLADenoise`
instance drives all 16 layers through ONE leaf instance, so a forward-time
cache would be overwritten fifteen times before any backward read it — the
leaf would silently return the last layer's gradient for every layer. Being
stateless across the forward/backward boundary is what makes one instance
reusable, and it is also flash-attention's own trade: the probability matrix
never has to exist for the whole forward.

The three gradient kernels cannot all be written from one thread map — dQ
reduces over j, dK and dV reduce over i — so `vjp` materialises the
probabilities into leaf scratch first and then runs three race-free passes:

    probs   one thread per (b, h, i)      writes p[b,h,i,j]
    dV      one block  per (b, h)         dv[j] = Σ_i p[i,j]·g[i]
    dscore  one block  per (b, h)         OVERWRITES p with ds, then dQ
    dK      one block  per (b, h)         dk[j] = Σ_i ds[i,j]·q[i]

⚠ **Those four are order-dependent** — dV reads p, the third pass destroys it.
They are enqueued on one stream, which orders them; splitting them across
streams would not.

⚠ **The mask is applied exactly once**, in the probability pass: a masked (i,j)
gets p = 0, so its ds is 0 and it contributes nothing to dK or dV without any
kernel downstream re-reading the mask. A fully-masked row keeps the forward's
floored denominator, so it yields a zero gradient rather than a NaN.

⚠ **dQ, dK and dV are OVERWRITTEN, not accumulated.** Each is written exactly
once by exactly one pass, so no caller has to zero them first. A driver that
needs to sum gradients from two paths must add them itself.

⚠ A fully-masked query row would divide by zero. Rows are renormalised by a
floored denominator and produce a zero context vector instead of NaN — the same
guard, and the same constant, as `cross_attention.mojo`.
"""

from std.math import exp, sqrt
from max.gpu import global_idx, thread_idx, block_idx, block_dim, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import RowMajor, Tile


comptime BA_MASK_NEG: Scalar[DT] = Scalar[DT](-1.0e30)
comptime BA_DENOM_FLOOR: Scalar[DT] = Scalar[DT](1.0e-30)
"""One warp: the block size the row softmax was measured at on the Orin."""


# ⚠⚠ THE FORWARD IS THREE ELEMENT-INDEXED KERNELS, NOT ONE ROW-PER-THREAD ONE,
# AND THAT IS A MEASUREMENT. The previous `_ba_kernel` launched one thread per
# (batch, head, query) with every key and every head-dim inside the thread. Its
# own note said "a tiled BMM is not worth its complexity until this shows up in
# a profile". It showed up: nsys on the Orin put it at 160 calls per SmolVLA
# query, 1.08 s of 3.24 s, at 0.29% of the board's fp32 peak.
#
# `benchmarks/smolvla_block_attention_bench.mojo` then separated the causes in
# one run on the board (A = that kernel):
#
#     block 128 -> 32 (occupancy)            1.18-1.23x
#     branchless, no `continue`              1.27-1.36x   (NOT mask divergence:
#                                                          larger on the UNMASKED
#                                                          cross shape)
#     one-pass online softmax                no gain over branchless
#     THESE THREE KERNELS                    6.10x self, 6.55x cross
#
# The scores are tiny at this model's shapes — 15 heads x 50 queries x 185 keys
# = 138 750 floats, 555 KB — so materialising them costs nothing, and it lets
# each kernel be indexed by the element it WRITES: adjacent lanes then touch
# adjacent addresses. The row-per-thread shape is the one this repo already
# recorded as slow (`_row_per_thread_kernels_are_uncoalesced_and_tiny_grid`).
#
# ⚠ NOT BIT-IDENTICAL to the old kernel — the context is now a sum of
# normalised weights rather than a normalised sum. Held to ~2.5e-6 std units of
# a float64 reference on the board, and to `test_block_attention.mojo`'s
# GPU-vs-CPU band, which also exercises B = 2 (the benchmark is B = 1).
#
# ⚠⚠ SECOND PASS (20 Sep 2026): element-indexed was necessary, not sufficient.
# `benchmarks/smolvla_denoise_stages_bench.mojo` on the Orin put these three
# kernels at 16.8 ms of a 27.8 ms denoising step — 1.05 ms per layer for 18 M
# MACs, 1.2% of the board's fp32 peak — while the expert's linears ran at 38%.
# The scores kernel's lanes were adjacent in j, so each lane read its OWN key
# row at a stride of DIM floats: 32 cache lines per warp per head-dim step,
# 64 steps, 4 336 warps — the K reads alone were ~280 MB of L2 sector traffic
# for a 710 KB matrix. And the softmax was one THREAD per row: 750 threads on
# the whole board, three serial passes of 185 loads each, at stride KL.
#
# Now K is packed once per call into `kt[b, h, d, j]` (contiguous in j, the
# same trick as `cross_attention.mojo` §2.1 "Kt contiguous"), so the scores
# kernel's warp reads 128 B per step instead of 32 lines; and the two row
# kernels (the softmax here, `_ba_dscore_kernel` in the backward) are one WARP
# per row — lanes stride the row by 32, coalesced, and the reductions are
# warp reductions. The context kernel was already coalesced (lanes adjacent
# in d) and is unchanged.
#
# ⚠ The row reductions are now warp-tree sums, not serial ones: the
# probabilities and dS move at the last bit. `test_block_attention.mojo`
# (1e-5 vs the torch-gated CPU path) and `test_block_attention_vjp.mojo`
# (central differences) are the bands; the old serial kernels are not a
# reference.


def ba_warp_rows_grid(rows: Int) -> Int:
    """Blocks of `TPB` for a one-warp-per-row kernel over `rows` rows
    (`_ba_softmax_kernel`, `_ba_dscore_kernel`). `TPB % WARP_SIZE == 0` is
    what makes those kernels' early return warp-uniform — a warp reduction
    with a lane missing is undefined — and it is asserted here, at the one
    place the grid is formed."""
    comptime assert TPB % WARP_SIZE == 0, (
        "block_attention: TPB must be a whole number of warps"
    )
    return (rows * WARP_SIZE + TPB - 1) // TPB


def _ba_pack_kt_kernel[
    BATCH: Int, D: Int, NH: Int, KL: Int, H: Int
](
    k: Tile[DT, BATCH, KL * D],
    kt: Tile[DT, BATCH * NH * H * KL],
):
    """`kt[b, h, d, j] = k[b, j, h*H + d]` — one thread per INPUT element, so
    the reads are the coalesced side and the strided side is the writes,
    which do not stall a warp."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * KL * D:
        return
    var c = idx % D
    var r = idx // D
    var j = r % KL
    var b = r // KL
    var h = c // H
    var d = c % H
    kt.ptr[unsafe_offset = ((b * NH + h) * H + d) * KL + j] = rebind[
        Scalar[DT]
    ](k.ptr[unsafe_offset = idx])


def _ba_scores_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int
](
    q: Tile[DT, BATCH, Q * D],
    kt: Tile[DT, BATCH * NH * H * KL],
    mask: Tile[DT, Q * KL],
    scores: Tile[DT, BATCH * NH * Q * KL],
):
    """One thread per (b, h, i, j): scaled score plus the additive mask.
    `kt` is `_ba_pack_kt_kernel`'s `[b, h, d, j]`: adjacent lanes (adjacent j)
    read adjacent addresses at every d."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q * KL:
        return
    var j = idx % KL
    var r = idx // KL
    var i = r % Q
    var r2 = r // Q
    var h = r2 % NH
    var b = r2 // NH
    var qb = b * (Q * D) + i * D + h * H
    var kb = ((b * NH + h) * H) * KL + j
    var s = Scalar[DT](0)
    for d in range(H):
        s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
            Scalar[DT]
        ](kt.ptr[unsafe_offset = kb + d * KL])
    scores.ptr[unsafe_offset = idx] = s * (
        Scalar[DT](1.0) / sqrt(Scalar[DT](H))
    ) + rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])


def _ba_softmax_kernel[BATCH: Int, NH: Int, Q: Int, KL: Int](
    scores: Tile[DT, BATCH * NH * Q * KL],
):
    """One WARP per (b, h, i): stable softmax over one CONTIGUOUS row, in
    place. Lanes stride the row by `WARP_SIZE`, so every pass is coalesced;
    the max and the denominator are warp reductions. Launch with
    `ba_warp_rows_grid(rows)` blocks of `TPB`.

    ⚠ A FULLY MASKED ROW gives zero weights, not NaN: every score is ~-1e30, so
    the running max never rises above half of `BA_MASK_NEG` and the row is
    zeroed — the same zero context the CPU path produces.
    """
    var idx = Int(global_idx.x)
    var row = idx // WARP_SIZE
    var lane = idx % WARP_SIZE
    if row >= BATCH * NH * Q:
        return
    var base = row * KL
    var mx = BA_MASK_NEG
    var j = lane
    while j < KL:
        mx = max(mx, rebind[Scalar[DT]](scores.ptr[unsafe_offset = base + j]))
        j += WARP_SIZE
    mx = warp.max(mx)
    var part = Scalar[DT](0)
    j = lane
    while j < KL:
        part += exp(
            rebind[Scalar[DT]](scores.ptr[unsafe_offset = base + j]) - mx
        )
        j += WARP_SIZE
    var denom = warp.sum(part)
    var inv = Scalar[DT](0)
    if mx > BA_MASK_NEG * Scalar[DT](0.5) and denom > BA_DENOM_FLOOR:
        inv = Scalar[DT](1.0) / denom
    j = lane
    while j < KL:
        scores.ptr[unsafe_offset = base + j] = exp(
            rebind[Scalar[DT]](scores.ptr[unsafe_offset = base + j]) - mx
        ) * inv
        j += WARP_SIZE


def _ba_context_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int
](
    probs: Tile[DT, BATCH * NH * Q * KL],
    v: Tile[DT, BATCH, KL * D],
    dst: Tile[DT, BATCH, Q * D],
):
    """One thread per (b, h, i, d): the d-th coordinate of one context vector."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q * H:
        return
    var d = idx % H
    var r = idx // H
    var i = r % Q
    var r2 = r // Q
    var h = r2 % NH
    var b = r2 // NH
    var pbase = ((b * NH + h) * Q + i) * KL
    var acc = Scalar[DT](0)
    for j in range(KL):
        acc += rebind[Scalar[DT]](probs.ptr[unsafe_offset = pbase + j]) * rebind[
            Scalar[DT]
        ](v.ptr[unsafe_offset = b * (KL * D) + j * D + h * H + d])
    dst.ptr[unsafe_offset = b * (Q * D) + i * D + h * H + d] = acc


# +--------------------------------------------------------------------------+ #
# | Backward
# +--------------------------------------------------------------------------+ #
#
# ⚠⚠ SIX ELEMENT-INDEXED KERNELS, FOR THE SAME REASON THE FORWARD IS THREE.
# The first backward shipped as four kernels launched ONE BLOCK PER (b, h) with
# the threads striding over rows: at this model's shapes that is 15 blocks and
# 50 live threads each, every thread walking ~36 k serial MACs through
# uncoalesced reads. `SMOLVLA_PROFILE` on a 5090 put it at 33.6 ms of a
# 48.8 ms expert backward — 2.1 ms per layer for 15 x 50 x 190 x 64 MACs,
# about 0.1 % of the card. The shape this repo already recorded as slow
# (`_row_per_thread_kernels_are_uncoalesced_and_tiny_grid`), twice over.
#
# Now every kernel is indexed by the element it WRITES, over the whole grid:
#
#     probs   = softmax(scale q.k + mask)      the forward's own two kernels
#     dP      = g . vᵀ                          one thread per (b,h,i,j)
#     dS      = p (dP − Σ_j p dP) scale         one thread per (b,h,i), in place
#     dQ      = dS . k                          one thread per (b,h,i,d)
#     dK      = dSᵀ . q                         one thread per (b,h,j,d)
#     dV      = pᵀ . g                          one thread per (b,h,j,d)
#
# `dS` lives in its own scratch (`dscore`), not over `probs`, so dV can run
# whenever it likes and there is no ordering the stream has to be trusted with.
#
# ⚠ The probabilities are the FORWARD's kernels re-run, not a transcription of
# them: `_ba_scores_kernel` then `_ba_softmax_kernel`, into `probs`. A backward
# taken against a different softmax is the gradient of a different function,
# and two copies of one softmax drift (`_a_rule_written_inline_twice_drifts`).
# They have to be re-run at all because ONE instance serves all sixteen layers
# and `fwd_scores` holds only the last one's.


def _ba_dp_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    grad_out: Tile[DT, BATCH, QL * DIM],
    v: Tile[DT, BATCH, KL * DIM],
    dscore: Tile[DT, BATCH, N_HEADS * QL * KL],
):
    """dP[i, j] = g[i] . v[j] — one thread per (b, h, i, j)."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * N_HEADS * QL * KL:
        return
    var j = idx % KL
    var r = idx // KL
    var i = r % QL
    var r2 = r // QL
    var h = r2 % N_HEADS
    var b = r2 // N_HEADS
    var gb = b * (QL * DIM) + i * DIM + h * HD
    var vb = b * (KL * DIM) + j * DIM + h * HD
    var acc = Scalar[DT](0)
    for d in range(HD):
        acc += rebind[Scalar[DT]](
            grad_out.ptr[unsafe_offset = gb + d]
        ) * rebind[Scalar[DT]](v.ptr[unsafe_offset = vb + d])
    dscore.ptr[unsafe_offset = idx] = acc


def _ba_dscore_kernel[BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int](
    probs: Tile[DT, BATCH, N_HEADS * QL * KL],
    dscore: Tile[DT, BATCH, N_HEADS * QL * KL],
):
    """Softmax backward over one CONTIGUOUS row, in place over dP:

        dot     = Σ_j p[i,j] dP[i,j]
        dS[i,j] = p[i,j] (dP[i,j] − dot) scale

    One WARP per (b, h, i), lanes striding the row, `dot` a warp reduction;
    launch with `ba_warp_rows_grid(rows)` blocks of `TPB`. A masked key has
    p = 0 and so dS = 0 exactly, which is what makes dK and dV of that key
    exactly zero downstream; a fully masked row has every p = 0 and yields
    an all-zero dS, not a NaN.
    """
    var idx = Int(global_idx.x)
    var row = idx // WARP_SIZE
    var lane = idx % WARP_SIZE
    if row >= BATCH * N_HEADS * QL:
        return
    var base = row * KL
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var part = Scalar[DT](0)
    var j = lane
    while j < KL:
        part += rebind[Scalar[DT]](probs.ptr[unsafe_offset = base + j]) * rebind[
            Scalar[DT]
        ](dscore.ptr[unsafe_offset = base + j])
        j += WARP_SIZE
    var dot = warp.sum(part)
    j = lane
    while j < KL:
        var p = rebind[Scalar[DT]](probs.ptr[unsafe_offset = base + j])
        var dp = rebind[Scalar[DT]](dscore.ptr[unsafe_offset = base + j])
        dscore.ptr[unsafe_offset = base + j] = p * (dp - dot) * scale
        j += WARP_SIZE


def _ba_dq_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    dscore: Tile[DT, BATCH, N_HEADS * QL * KL],
    k: Tile[DT, BATCH, KL * DIM],
    dq: Tile[DT, BATCH, QL * DIM],
):
    """dQ[i, d] = Σ_j dS[i,j] k[j, d] — one thread per (b, h, i, d)."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * N_HEADS * QL * HD:
        return
    var d = idx % HD
    var r = idx // HD
    var i = r % QL
    var r2 = r // QL
    var h = r2 % N_HEADS
    var b = r2 // N_HEADS
    var sb = ((b * N_HEADS + h) * QL + i) * KL
    var kb = b * (KL * DIM) + h * HD + d
    var acc = Scalar[DT](0)
    for j in range(KL):
        acc += rebind[Scalar[DT]](dscore.ptr[unsafe_offset = sb + j]) * rebind[
            Scalar[DT]
        ](k.ptr[unsafe_offset = kb + j * DIM])
    dq.ptr[unsafe_offset = b * (QL * DIM) + i * DIM + h * HD + d] = acc


def _ba_dkv_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    coef: Tile[DT, BATCH, N_HEADS * QL * KL],
    src: Tile[DT, BATCH, QL * DIM],
    dst: Tile[DT, BATCH, KL * DIM],
):
    """dst[j, d] = Σ_i coef[i,j] src[i, d] — one thread per (b, h, j, d).

    ONE kernel for dK and dV, because they are one formula: dK is it with
    (dS, q), dV with (p, g). Two copies of one rule drift.
    """
    var idx = Int(global_idx.x)
    if idx >= BATCH * N_HEADS * KL * HD:
        return
    var d = idx % HD
    var r = idx // HD
    var j = r % KL
    var r2 = r // KL
    var h = r2 % N_HEADS
    var b = r2 // N_HEADS
    var cb = (b * N_HEADS + h) * QL * KL + j
    var sb = b * (QL * DIM) + h * HD + d
    var acc = Scalar[DT](0)
    for i in range(QL):
        acc += rebind[Scalar[DT]](
            coef.ptr[unsafe_offset = cb + i * KL]
        ) * rebind[Scalar[DT]](src.ptr[unsafe_offset = sb + i * DIM])
    dst.ptr[unsafe_offset = b * (KL * DIM) + j * DIM + h * HD + d] = acc


struct BlockCrossAttention[
    DIM: Int, N_HEADS: Int, Q_LEN: Int, KV_LEN: Int
](Movable):
    comptime HD: Int = Self.DIM // Self.N_HEADS
    comptime QN: Int = Self.Q_LEN * Self.DIM
    comptime KN: Int = Self.KV_LEN * Self.DIM
    comptime MASK_N: Int = Self.Q_LEN * Self.KV_LEN
    comptime PN: Int = Self.N_HEADS * Self.Q_LEN * Self.KV_LEN

    var mask: Tensor
    var is_gpu: Bool
    # Backward scratch: the probability matrix and, separately, dP then dS in
    # place over it. Lazily sized by `vjp`, and NOT touched by `forward` — the
    # forward stays allocation-free and one instance stays reusable across the
    # sixteen layers a driver runs through it.
    var probs: Tensor
    var dscore: Tensor
    var kt: Tensor
    """`_ba_pack_kt_kernel`'s `[b, h, d, j]` copy of K, rebuilt by every GPU
    forward and by every vjp's probability refill: one instance serves all
    sixteen layers, so nothing from the forward can be relied on."""
    var fwd_scores: Tensor
    """The GPU forward's materialised scores, then probabilities, in place.
    Sized on the first call and reused: `ensure_gpu` allocates only on a GROW,
    so the forward stays allocation-free across the sixteen layers and ten
    steps one instance serves. Not shared with `probs`, which `vjp` owns."""

    def __init__(out self):
        comptime assert Self.DIM % Self.N_HEADS == 0, (
            "BlockCrossAttention: DIM must be divisible by N_HEADS"
        )
        self.mask = Tensor()
        self.is_gpu = False
        self.probs = Tensor()
        self.dscore = Tensor()
        self.kt = Tensor()
        self.fwd_scores = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.mask = move.mask^
        self.is_gpu = move.is_gpu
        self.probs = move.probs^
        self.dscore = move.dscore^
        self.kt = move.kt^
        self.fwd_scores = move.fwd_scores^

    @staticmethod
    def make[
        target: StaticString
    ](
        ref mask: List[Scalar[DT]], ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        if len(mask) != Self.MASK_N:
            raise Error(
                "BlockCrossAttention: mask must be Q_LEN*KV_LEN = "
                + String(Self.MASK_N) + ", got " + String(len(mask))
            )
        var a = Self()
        a.mask = Tensor.alloc(Self.MASK_N)
        for i in range(Self.MASK_N):
            a.mask.data[i] = mask[i]
        comptime if target != "cpu":
            a.mask.upload(ctx.value())
            a.is_gpu = True
        return a^

    def forward[
        target: StaticString, B: Int
    ](
        mut self, mut q: Tensor, mut k: Tensor, mut v: Tensor,
        mut out: Tensor, ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            out.ensure(B * Self.QN)
            var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](Self.HD))
            for b in range(B):
                for h in range(Self.N_HEADS):
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var mx = BA_MASK_NEG
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var s = Scalar[DT](0)
                            for d in range(Self.HD):
                                s += q.data[qb + d] * k.data[kb + d]
                            s = s * scale + m
                            if s > mx:
                                mx = s
                        var denom = Scalar[DT](0)
                        for d in range(Self.HD):
                            out.data[qb + d] = Scalar[DT](0)
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var s = Scalar[DT](0)
                            for d in range(Self.HD):
                                s += q.data[qb + d] * k.data[kb + d]
                            var w = exp(s * scale + m - mx)
                            denom += w
                            for d in range(Self.HD):
                                out.data[qb + d] += w * v.data[kb + d]
                        var inv = Scalar[DT](0)
                        if denom > BA_DENOM_FLOOR:
                            inv = Scalar[DT](1.0) / denom
                        for d in range(Self.HD):
                            out.data[qb + d] *= inv
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.QN)
            self.fwd_scores.ensure_gpu(c, B * Self.PN)
            self.kt.ensure_gpu(c, B * Self.KN)
            comptime n_scores = B * Self.PN
            comptime n_rows = B * Self.N_HEADS * Self.Q_LEN
            comptime n_ctx = B * Self.N_HEADS * Self.Q_LEN * Self.HD
            comptime n_k = B * Self.KN
            c.enqueue_function[
                _ba_pack_kt_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.KV_LEN, Self.HD
                ]
            ](
                k.tt["gpu", B, Self.KN](),
                self.kt.tt["gpu", B * Self.KN](),
                grid_dim=(n_k + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_scores_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                q.tt["gpu", B, Self.QN](),
                self.kt.tt["gpu", B * Self.KN](),
                self.mask.tt["gpu", Self.MASK_N](),
                self.fwd_scores.tt["gpu", B * Self.PN](),
                grid_dim=(n_scores + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_softmax_kernel[B, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN]
            ](
                self.fwd_scores.tt["gpu", B * Self.PN](),
                grid_dim=ba_warp_rows_grid(n_rows),
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_context_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.fwd_scores.tt["gpu", B * Self.PN](),
                v.tt["gpu", B, Self.KN](),
                out.tt["gpu", B, Self.QN](),
                grid_dim=(n_ctx + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int
    ](
        mut self,
        mut q: Tensor, mut k: Tensor, mut v: Tensor,
        mut grad_out: Tensor,
        mut dq: Tensor, mut dk: Tensor, mut dv: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`dQ`, `dK` and `dV` from the forward's own inputs and its output
        gradient.

        `q`, `k` and `v` must be the SAME tensors the forward saw — this leaf
        keeps nothing from the forward and recomputes the softmax from them.
        Passing a later layer's activations would produce a plausible,
        finite, wrong gradient, so a driver has to save them per layer.

        The three outputs are OVERWRITTEN. See this file's header.
        """
        comptime if target == "cpu":
            dq.ensure(B * Self.QN)
            dk.ensure(B * Self.KN)
            dv.ensure(B * Self.KN)
            self.probs.ensure(B * Self.PN)
            for i in range(B * Self.QN):
                dq.data[i] = Scalar[DT](0)
            for i in range(B * Self.KN):
                dk.data[i] = Scalar[DT](0)
                dv.data[i] = Scalar[DT](0)

            var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](Self.HD))
            for b in range(B):
                for h in range(Self.N_HEADS):
                    # ── the forward's softmax, in the forward's arithmetic ──
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var pb = (
                            b * Self.PN + h * Self.Q_LEN * Self.KV_LEN
                            + i * Self.KV_LEN
                        )
                        var mx = BA_MASK_NEG
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            self.probs.data[pb + j] = Scalar[DT](0)
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var sc = Scalar[DT](0)
                            for d in range(Self.HD):
                                sc += q.data[qb + d] * k.data[kb + d]
                            sc = sc * scale + m
                            if sc > mx:
                                mx = sc
                        var denom = Scalar[DT](0)
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var sc = Scalar[DT](0)
                            for d in range(Self.HD):
                                sc += q.data[qb + d] * k.data[kb + d]
                            var w = exp(sc * scale + m - mx)
                            denom += w
                            self.probs.data[pb + j] = w
                        var inv = Scalar[DT](0)
                        if denom > BA_DENOM_FLOOR:
                            inv = Scalar[DT](1.0) / denom
                        for j in range(Self.KV_LEN):
                            self.probs.data[pb + j] *= inv

                    # ── the gradient. Float64 accumulators: a reduction over
                    #    KV_LEN (185 in the self case) of terms that cancel
                    #    around `dot` is where fp32 loses the most. ───────────
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var pb = (
                            b * Self.PN + h * Self.Q_LEN * Self.KV_LEN
                            + i * Self.KV_LEN
                        )
                        var dot = Float64(0)
                        for j in range(Self.KV_LEN):
                            var pj = Float64(self.probs.data[pb + j])
                            if pj == 0.0:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var da = Float64(0)
                            for d in range(Self.HD):
                                da += Float64(grad_out.data[qb + d]) * Float64(
                                    v.data[kb + d]
                                )
                            dot += pj * da
                        for j in range(Self.KV_LEN):
                            var pj = Float64(self.probs.data[pb + j])
                            if pj == 0.0:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var da = Float64(0)
                            for d in range(Self.HD):
                                da += Float64(grad_out.data[qb + d]) * Float64(
                                    v.data[kb + d]
                                )
                            var ds = pj * (da - dot) * Float64(scale)
                            for d in range(Self.HD):
                                dq.data[qb + d] = dq.data[qb + d] + Scalar[DT](
                                    ds * Float64(k.data[kb + d])
                                )
                                dk.data[kb + d] = dk.data[kb + d] + Scalar[DT](
                                    ds * Float64(q.data[qb + d])
                                )
                                dv.data[kb + d] = dv.data[kb + d] + Scalar[DT](
                                    pj * Float64(grad_out.data[qb + d])
                                )
        else:
            var c = ctx.value()
            dq.ensure_gpu(c, B * Self.QN)
            dk.ensure_gpu(c, B * Self.KN)
            dv.ensure_gpu(c, B * Self.KN)
            self.probs.ensure_gpu(c, B * Self.PN)
            self.dscore.ensure_gpu(c, B * Self.PN)

            comptime lay_q = RowMajor[B, Self.QN]
            comptime lay_k = RowMajor[B, Self.KN]
            comptime lay_p = RowMajor[B, Self.PN]
            comptime lay_pf = RowMajor[B * Self.PN]
            comptime n_scores = B * Self.PN
            comptime n_rows = B * Self.N_HEADS * Self.Q_LEN
            comptime n_q = B * Self.N_HEADS * Self.Q_LEN * Self.HD
            comptime n_kv = B * Self.N_HEADS * Self.KV_LEN * Self.HD

            # ── probs: the forward's three kernels, into `probs` ─────────
            self.kt.ensure_gpu(c, B * Self.KN)
            c.enqueue_function[
                _ba_pack_kt_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.KV_LEN, Self.HD
                ]
            ](
                k.tt["gpu", B, Self.KN](),
                self.kt.tt["gpu", B * Self.KN](),
                grid_dim=(B * Self.KN + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_scores_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                q.tt["gpu", B, Self.QN](),
                self.kt.tt["gpu", B * Self.KN](),
                self.mask.tt["gpu", Self.MASK_N](),
                self.probs.tt["gpu", B * Self.PN](),
                grid_dim=(n_scores + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_softmax_kernel[B, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN]
            ](
                self.probs.tt["gpu", B * Self.PN](),
                grid_dim=ba_warp_rows_grid(n_rows),
                block_dim=TPB,
            )
            # ── dV = pᵀ g: reads probs only, so it can go first ──────────
            c.enqueue_function[
                _ba_dkv_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.probs.tt["gpu", B, Self.PN](),
                grad_out.tt["gpu", B, Self.QN](),
                dv.tt["gpu", B, Self.KN](),
                grid_dim=(n_kv + TPB - 1) // TPB,
                block_dim=TPB,
            )
            # ── dP, then dS in place over it ─────────────────────────────
            c.enqueue_function[
                _ba_dp_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                grad_out.tt["gpu", B, Self.QN](),
                v.tt["gpu", B, Self.KN](),
                self.dscore.tt["gpu", B, Self.PN](),
                grid_dim=(n_scores + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_dscore_kernel[
                    B, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.probs.tt["gpu", B, Self.PN](),
                self.dscore.tt["gpu", B, Self.PN](),
                grid_dim=ba_warp_rows_grid(n_rows),
                block_dim=TPB,
            )
            # ── dQ = dS k,  dK = dSᵀ q ───────────────────────────────────
            c.enqueue_function[
                _ba_dq_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.dscore.tt["gpu", B, Self.PN](),
                k.tt["gpu", B, Self.KN](),
                dq.tt["gpu", B, Self.QN](),
                grid_dim=(n_q + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_dkv_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.dscore.tt["gpu", B, Self.PN](),
                q.tt["gpu", B, Self.QN](),
                dk.tt["gpu", B, Self.KN](),
                grid_dim=(n_kv + TPB - 1) // TPB,
                block_dim=TPB,
            )
