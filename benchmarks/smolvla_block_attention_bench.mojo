# +--------------------------------------------------------------------------+ #
# | SmolVLA's expert attention, one kernel at a time — WHY is it 0.29% of peak?
# +--------------------------------------------------------------------------+ #
"""Five variants of the denoising step's attention, at its real shapes.

    pixi run -e jetson smolvla-attn-bench-jetson          # the board
    pixi run -e apple mojo run -I . benchmarks/smolvla_block_attention_bench.mojo

⚠⚠ WHY THIS EXISTS. nsys on the Orin: the expert's `_ba_kernel` is 160 calls
per query, 1.08 s of a 3.24 s query, at **0.29% of the board's fp32 peak** —
26x WORSE per MAC than MAX's own naive batched matmul on the same board. A
fused attention kernel should be the fast one. Something about THIS kernel is
pathological, and the fix depends on which thing. One run of this file
separates the hypotheses, instead of one guess per round trip to the board.

Each variant changes EXACTLY ONE thing relative to the one above it:

    A  the row-per-thread kernel, block 128     the baseline (was shipped)
    B  same kernel, block 32 (one warp)         occupancy        BIT-IDENTICAL
    C  branchless: no `continue`, `max`          lane divergence  BIT-IDENTICAL
    D  one pass, online softmax (C's launch)     duplicated q.k   tolerance
    E  pack Kt -> scores -> softmax -> A.V      launch shape     tolerance
       (SHIPS; coalesced K + warp-per-row softmax since 20 Sep)

⚠ READING THE RESULT. The CROSS shape has no mask at all (every prefix key is
visible), so masking cannot diverge the lanes there. If C beats B on SELF but
not on CROSS, divergence is the cause. If B beats A on both, occupancy is.
If only E wins, it is the row-per-thread launch shape itself — the pattern this
repo already recorded as slow (`_row_per_thread_kernels_are_uncoalesced_and_tiny_grid`).

⚠ B AND C MUST BE BIT-IDENTICAL TO A, and the program says so if they are not:
a "speed-up" that changes the numbers is a different kernel. D and E change the
order of floating-point operations and are held to a tolerance against a
FLOAT64 reference instead — never against A, which is itself float32 and would
make the gate measure A's rounding.

⚠ The masks are the policy's own (`smolvla_ar` + `att_2d_mask`), not a stand-in:
the self layers are CAUSAL within the 50-step chunk, which is exactly the
masking that could diverge lanes. A synthetic all-visible mask would hide it.
"""

from std.math import exp, sqrt
from max.gpu import global_idx
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.smolvla.attn_mask import att_2d_mask, smolvla_ar
from noeira.deep_agents.smolvla.block_attention import (
    BA_DENOM_FLOOR,
    BA_MASK_NEG,
    _ba_context_kernel,
    _ba_pack_kt_kernel,
    _ba_scores_kernel,
    _ba_softmax_kernel,
    ba_warp_rows_grid,
)


# ── the expert's real shape (policy: N_CAM 2, IMG_TOK 64, N_LANG 6, CHUNK 50) ─
comptime B = 1
comptime DIM = 960
comptime HEADS = 15
comptime HD = DIM // HEADS
comptime QL = 50
comptime N_IMG = 2 * 64
comptime N_LANG = 6
comptime P = N_IMG + N_LANG + 1
comptime KL_SELF = P + QL
comptime KL_CROSS = P

comptime WARP = 32
comptime WARMUP = 3
comptime REPS = 20

comptime ORIN_FP32_PEAK_GFLOPS = 2.0 * 1024.0 * 1.173
"""1024 CUDA cores at 1.173 GHz, 2 FLOPs per core-cycle (fp32 FMA). The %
column is only meaningful on the Orin; elsewhere read the GFLOPS column."""

comptime SEED = 1234
comptime GATE_STD_UNITS = 1.0e-3


# ═══════════════════════════════════════════════════════════════════════════
# C — branchless: no `continue`, `max` instead of `if s > mx`
# ═══════════════════════════════════════════════════════════════════════════
#
# A masked key has `m = BA_MASK_NEG = -1e30`, so its score is -1e30: never the
# max (unless the whole row is masked, guarded below), and `exp(-1e30 - mx)`
# underflows to EXACTLY 0.0 — adding `0.0 * v` changes nothing. So removing the
# `continue` removes the control flow without changing a single bit.


def _c_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int
](
    q: Tile[DT, BATCH, Q * D],
    k: Tile[DT, BATCH, KL * D],
    v: Tile[DT, BATCH, KL * D],
    mask: Tile[DT, Q * KL],
    dst: Tile[DT, BATCH, Q * D],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q:
        return
    var i = idx % Q
    var r = idx // Q
    var h = r % NH
    var b = r // NH
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](H))
    var qb = b * (Q * D) + i * D + h * H

    var mx = BA_MASK_NEG
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        var kb = b * (KL * D) + j * D + h * H
        var s = Scalar[DT](0)
        for d in range(H):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        mx = max(mx, s * scale + m)

    var denom = Scalar[DT](0)
    for d in range(H):
        dst.ptr[unsafe_offset = qb + d] = Scalar[DT](0)
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        var kb = b * (KL * D) + j * D + h * H
        var s = Scalar[DT](0)
        for d in range(H):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        var w = exp(s * scale + m - mx)
        denom += w
        for d in range(H):
            dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
                dst.ptr[unsafe_offset = qb + d]
            ) + w * rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])

    var inv = Scalar[DT](1.0) / (denom if denom > BA_DENOM_FLOOR else Scalar[DT](1.0))
    if denom <= BA_DENOM_FLOOR:
        inv = Scalar[DT](0)
    for d in range(H):
        dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
            dst.ptr[unsafe_offset = qb + d]
        ) * inv


# ═══════════════════════════════════════════════════════════════════════════
# D — one pass: online normaliser (Milakov & Gimelshein) with the value sum
# ═══════════════════════════════════════════════════════════════════════════
#
# A and C compute every q.k TWICE: once to find the max for a stable softmax,
# again for the weights. Here the max is tracked ONLINE: when a new maximum
# arrives, the running denominator and running context are rescaled by
# exp(old_max - new_max) < 1, so every exponent stays <= 0 and nothing
# overflows. A running max of 185 values changes ~ln(185) ~ 5 times, so the
# rescale branch is RARE — ~5 rescales of 64 floats replace 185 dot products.


def _d_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int
](
    q: Tile[DT, BATCH, Q * D],
    k: Tile[DT, BATCH, KL * D],
    v: Tile[DT, BATCH, KL * D],
    mask: Tile[DT, Q * KL],
    dst: Tile[DT, BATCH, Q * D],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q:
        return
    var i = idx % Q
    var r = idx // Q
    var h = r % NH
    var b = r // NH
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](H))
    var qb = b * (Q * D) + i * D + h * H

    for d in range(H):
        dst.ptr[unsafe_offset = qb + d] = Scalar[DT](0)
    var mx = Scalar[DT](-3.0e38)
    var denom = Scalar[DT](0)
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        var kb = b * (KL * D) + j * D + h * H
        var s = Scalar[DT](0)
        for d in range(H):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        s = s * scale + m
        if s > mx:
            var ratio = exp(mx - s)
            denom = denom * ratio + Scalar[DT](1)
            for d in range(H):
                dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
                    dst.ptr[unsafe_offset = qb + d]
                ) * ratio + rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])
            mx = s
        else:
            var w = exp(s - mx)
            denom += w
            for d in range(H):
                dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
                    dst.ptr[unsafe_offset = qb + d]
                ) + w * rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])

    # A fully masked row has mx ~ -1e30: zero context, as A does.
    var inv = Scalar[DT](0)
    if mx > BA_MASK_NEG * Scalar[DT](0.5) and denom > BA_DENOM_FLOOR:
        inv = Scalar[DT](1.0) / denom
    for d in range(H):
        dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
            dst.ptr[unsafe_offset = qb + d]
        ) * inv


# ═══════════════════════════════════════════════════════════════════════════
# A — the row-per-thread kernel the forward USED to launch
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠ KEPT HERE, NOT IN THE LIBRARY. `block_attention.mojo` now launches E; this
# is the historical baseline every other variant is measured against, and this
# benchmark is its only consumer. E, on the other hand, is IMPORTED from the
# library below — so the benchmark keeps measuring the code that ships rather
# than a copy that could drift from it.


def _ba_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    q: Tile[DT, BATCH, QL * DIM],
    k: Tile[DT, BATCH, KL * DIM],
    v: Tile[DT, BATCH, KL * DIM],
    mask: Tile[DT, QL * KL],
    dst: Tile[DT, BATCH, QL * DIM],
):
    """One thread per (batch, head, query). Q_LEN is 50 and N_HEADS 15 here, so
    the row-per-thread map is a few hundred threads — small, but the work per
    row is KL*HD and the alternative (a tiled BMM) is not worth its complexity
    until this shows up in a profile."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * N_HEADS * QL:
        return
    var i = idx % QL
    var r = idx // QL
    var h = r % N_HEADS
    var b = r // N_HEADS
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var qb = b * (QL * DIM) + i * DIM + h * HD

    # pass 1: max over masked scores
    var mx = BA_MASK_NEG
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        if m <= BA_MASK_NEG:
            continue
        var kb = b * (KL * DIM) + j * DIM + h * HD
        var s = Scalar[DT](0)
        for d in range(HD):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        s = s * scale + m
        if s > mx:
            mx = s

    # pass 2: exponentiate, accumulate the context vector
    var denom = Scalar[DT](0)
    for d in range(HD):
        dst.ptr[unsafe_offset = qb + d] = Scalar[DT](0)
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        if m <= BA_MASK_NEG:
            continue
        var kb = b * (KL * DIM) + j * DIM + h * HD
        var s = Scalar[DT](0)
        for d in range(HD):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        var w = exp(s * scale + m - mx)
        denom += w
        for d in range(HD):
            dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
                dst.ptr[unsafe_offset = qb + d]
            ) + w * rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])

    var inv = Scalar[DT](1.0) / (denom if denom > BA_DENOM_FLOOR else Scalar[DT](1.0))
    if denom <= BA_DENOM_FLOOR:
        inv = Scalar[DT](0)   # fully masked row -> zero context, not NaN
    for d in range(HD):
        dst.ptr[unsafe_offset = qb + d] = rebind[Scalar[DT]](
            dst.ptr[unsafe_offset = qb + d]
        ) * inv


# ═══════════════════════════════════════════════════════════════════════════
# host
# ═══════════════════════════════════════════════════════════════════════════


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed * 2 + 1)

    def __init__(out self, *, deinit move: Self):
        self.s = move.s

    def next_unit(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(Float64((self.s >> 40) & UInt64(0xFFFFFF)) / 16777216.0)


def _random(n: Int, mut rng: Lcg, amp: Float32) -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT]((rng.next_unit() * 2.0 - 1.0) * amp)
    return t^


def _reference_f64(
    ref q: Tensor, ref k: Tensor, ref v: Tensor, ref mask: List[Scalar[DT]], kl: Int
) -> List[Float64]:
    """The same attention in FLOAT64, as plainly as it can be written."""
    var out = List[Float64](length=B * QL * DIM, fill=0.0)
    var scale = 1.0 / sqrt(Float64(HD))
    for b in range(B):
        for h in range(HEADS):
            for i in range(QL):
                var scores = List[Float64](length=kl, fill=0.0)
                var mx = -1.0e300
                var any = False
                for j in range(kl):
                    var m = Float64(mask[i * kl + j])
                    if m <= Float64(BA_MASK_NEG):
                        scores[j] = -1.0e300
                        continue
                    var s = 0.0
                    for d in range(HD):
                        s += Float64(q.data[b * QL * DIM + i * DIM + h * HD + d]) * Float64(
                            k.data[b * kl * DIM + j * DIM + h * HD + d]
                        )
                    scores[j] = s * scale + m
                    if scores[j] > mx:
                        mx = scores[j]
                    any = True
                if not any:
                    continue
                var denom = 0.0
                for j in range(kl):
                    if scores[j] > -1.0e299:
                        denom += exp(scores[j] - mx)
                for j in range(kl):
                    if scores[j] <= -1.0e299:
                        continue
                    var w = exp(scores[j] - mx) / denom
                    for d in range(HD):
                        out[b * QL * DIM + i * DIM + h * HD + d] += w * Float64(
                            v.data[b * kl * DIM + j * DIM + h * HD + d]
                        )
    return out^


struct Row(Copyable, Movable):
    var name: String
    var best_ms: Float64
    var median_ms: Float64
    var nominal_macs: Float64
    var bits_vs_a: Int
    var max_rel_vs_ref: Float64

    def __init__(
        out self, name: String, best_ms: Float64, median_ms: Float64,
        nominal_macs: Float64, bits_vs_a: Int, max_rel_vs_ref: Float64,
    ):
        self.name = name
        self.best_ms = best_ms
        self.median_ms = median_ms
        self.nominal_macs = nominal_macs
        self.bits_vs_a = bits_vs_a
        self.max_rel_vs_ref = max_rel_vs_ref


def _fmt(x: Float64, digits: Int) -> String:
    var p = 1.0
    for _ in range(digits):
        p *= 10.0
    var v = Float64(Int(x * p + (0.5 if x >= 0.0 else -0.5))) / p
    return String(v)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out^


def _median(mut xs: List[Float64]) -> Float64:
    for a in range(len(xs)):
        for c in range(a + 1, len(xs)):
            if xs[c] < xs[a]:
                var t = xs[a]
                xs[a] = xs[c]
                xs[c] = t
    return xs[len(xs) // 2]


def _compare(
    ref got: Tensor, ref base: List[Scalar[DT]], ref refv: List[Float64]
) -> Tuple[Int, Float64]:
    """(elements not bit-identical to A, max |got - ref| IN STD UNITS of ref).

    ⚠⚠ STD UNITS, NOT RELATIVE ERROR — and the first version of this file got
    that wrong. A context coordinate is a weighted sum of SIGNED values and
    often lands near zero, so dividing by |ref| turned float32's ordinary
    ~1e-6 rounding into a "0.6% error" on EVERY variant, the shipped kernel
    included. A gate that fails the reference implementation is measuring its
    own metric. Scaling by the output's RMS asks the question that matters:
    is the error small against the size of the thing computed?
    """
    var bits = 0
    var worst = 0.0
    var ss = 0.0
    for n in range(len(refv)):
        ss += refv[n] * refv[n]
    var rms = sqrt(ss / Float64(len(refv)))
    if rms < 1.0e-30:
        rms = 1.0
    for n in range(len(refv)):
        var g = got.data[n]
        if g != base[n]:
            bits += 1
        var e = abs(Float64(g) - refv[n])
        if e > worst:
            worst = e
    return (bits, worst / rms)


comptime _ba_kernel_self = _ba_kernel[B, DIM, HEADS, QL, KL_SELF, HD]
comptime _ba_kernel_cross = _ba_kernel[B, DIM, HEADS, QL, KL_CROSS, HD]
comptime _c_self = _c_kernel[B, DIM, HEADS, QL, KL_SELF, HD]
comptime _c_cross = _c_kernel[B, DIM, HEADS, QL, KL_CROSS, HD]
comptime _d_self = _d_kernel[B, DIM, HEADS, QL, KL_SELF, HD]
comptime _d_cross = _d_kernel[B, DIM, HEADS, QL, KL_CROSS, HD]
comptime _e_pack_self = _ba_pack_kt_kernel[B, DIM, HEADS, KL_SELF, HD]
comptime _e_pack_cross = _ba_pack_kt_kernel[B, DIM, HEADS, KL_CROSS, HD]
comptime _e_scores_self = _ba_scores_kernel[B, DIM, HEADS, QL, KL_SELF, HD]
comptime _e_scores_cross = _ba_scores_kernel[B, DIM, HEADS, QL, KL_CROSS, HD]
comptime _e_softmax_self = _ba_softmax_kernel[B, HEADS, QL, KL_SELF]
comptime _e_softmax_cross = _ba_softmax_kernel[B, HEADS, QL, KL_CROSS]
comptime _e_context_self = _ba_context_kernel[B, DIM, HEADS, QL, KL_SELF, HD]
comptime _e_context_cross = _ba_context_kernel[B, DIM, HEADS, QL, KL_CROSS, HD]


def run_shape(
    name: String, kl: Int, mut ctx: DeviceContext
) raises -> List[Row]:
    """All five variants at one KV length, checked and timed."""
    var ar = smolvla_ar(N_IMG, N_LANG, 1, QL)
    var mask = att_2d_mask(ar, P, P + QL, 0, kl)
    var visible = 0
    for n in range(len(mask)):
        if mask[n] > BA_MASK_NEG:
            visible += 1

    var rng = Lcg(SEED + kl)
    var q = _random(B * QL * DIM, rng, 1.0)
    var k = _random(B * kl * DIM, rng, 1.0)
    var v = _random(B * kl * DIM, rng, 1.0)
    var refv = _reference_f64(q, k, v, mask, kl)
    var mask_t = Tensor.alloc(QL * kl)
    for n in range(QL * kl):
        mask_t.data[n] = mask[n]
    q.upload(ctx)
    k.upload(ctx)
    v.upload(ctx)
    mask_t.upload(ctx)

    print("")
    print(
        "── " + name + ": Q " + String(QL) + " x KV " + String(kl) + ", "
        + String(HEADS) + " heads x " + String(HD) + ", "
        + String(visible) + "/" + String(QL * kl) + " keys visible ──"
    )

    var rows = List[Row]()
    # ⚠ A's output as its OWN list, not a second reference into a list of
    # tensors — Mojo refuses two references into one container in one call.
    var base_a = List[Scalar[DT]]()
    comptime QN = QL * DIM
    comptime ROWS = B * HEADS * QL
    var nominal = Float64(ROWS) * Float64(3 * kl * HD)
    var probs = Tensor()
    probs.ensure_gpu(ctx, B * HEADS * QL * kl)
    var kt = Tensor()
    kt.ensure_gpu(ctx, B * kl * DIM)

    for variant in range(5):
        var out = Tensor()
        out.ensure_gpu(ctx, B * QN)
        var times = List[Float64]()
        var macs = nominal
        if variant == 3:
            macs = Float64(ROWS) * Float64(2 * kl * HD)
        for rep in range(WARMUP + REPS):
            var t0 = perf_counter_ns()
            if variant <= 1:
                var tpb = TPB if variant == 0 else WARP
                if kl == KL_SELF:
                    ctx.enqueue_function[_ba_kernel_self](
                        q.tt["gpu", B, QN](),
                        k.tt["gpu", B, KL_SELF * DIM](),
                        v.tt["gpu", B, KL_SELF * DIM](),
                        mask_t.tt["gpu", QL * KL_SELF](),
                        out.tt["gpu", B, QN](),
                        grid_dim=(ROWS + tpb - 1) // tpb, block_dim=tpb,
                    )
                else:
                    ctx.enqueue_function[_ba_kernel_cross](
                        q.tt["gpu", B, QN](),
                        k.tt["gpu", B, KL_CROSS * DIM](),
                        v.tt["gpu", B, KL_CROSS * DIM](),
                        mask_t.tt["gpu", QL * KL_CROSS](),
                        out.tt["gpu", B, QN](),
                        grid_dim=(ROWS + tpb - 1) // tpb, block_dim=tpb,
                    )
            elif variant == 2 or variant == 3:
                if kl == KL_SELF:
                    if variant == 2:
                        ctx.enqueue_function[_c_self](
                            q.tt["gpu", B, QN](),
                            k.tt["gpu", B, KL_SELF * DIM](),
                            v.tt["gpu", B, KL_SELF * DIM](),
                            mask_t.tt["gpu", QL * KL_SELF](),
                            out.tt["gpu", B, QN](),
                            grid_dim=(ROWS + WARP - 1) // WARP, block_dim=WARP,
                        )
                    else:
                        ctx.enqueue_function[_d_self](
                            q.tt["gpu", B, QN](),
                            k.tt["gpu", B, KL_SELF * DIM](),
                            v.tt["gpu", B, KL_SELF * DIM](),
                            mask_t.tt["gpu", QL * KL_SELF](),
                            out.tt["gpu", B, QN](),
                            grid_dim=(ROWS + WARP - 1) // WARP, block_dim=WARP,
                        )
                else:
                    if variant == 2:
                        ctx.enqueue_function[_c_cross](
                            q.tt["gpu", B, QN](),
                            k.tt["gpu", B, KL_CROSS * DIM](),
                            v.tt["gpu", B, KL_CROSS * DIM](),
                            mask_t.tt["gpu", QL * KL_CROSS](),
                            out.tt["gpu", B, QN](),
                            grid_dim=(ROWS + WARP - 1) // WARP, block_dim=WARP,
                        )
                    else:
                        ctx.enqueue_function[_d_cross](
                            q.tt["gpu", B, QN](),
                            k.tt["gpu", B, KL_CROSS * DIM](),
                            v.tt["gpu", B, KL_CROSS * DIM](),
                            mask_t.tt["gpu", QL * KL_CROSS](),
                            out.tt["gpu", B, QN](),
                            grid_dim=(ROWS + WARP - 1) // WARP, block_dim=WARP,
                        )
            else:
                var n_sc = B * HEADS * QL * kl
                var n_cx = B * HEADS * QL * HD
                if kl == KL_SELF:
                    ctx.enqueue_function[_e_pack_self](
                        k.tt["gpu", B, KL_SELF * DIM](),
                        kt.tt["gpu", B * KL_SELF * DIM](),
                        grid_dim=(B * KL_SELF * DIM + TPB - 1) // TPB, block_dim=TPB,
                    )
                    ctx.enqueue_function[_e_scores_self](
                        q.tt["gpu", B, QN](),
                        kt.tt["gpu", B * KL_SELF * DIM](),
                        mask_t.tt["gpu", QL * KL_SELF](),
                        probs.tt["gpu", B * HEADS * QL * KL_SELF](),
                        grid_dim=(n_sc + TPB - 1) // TPB, block_dim=TPB,
                    )
                    ctx.enqueue_function[_e_softmax_self](
                        probs.tt["gpu", B * HEADS * QL * KL_SELF](),
                        grid_dim=ba_warp_rows_grid(ROWS), block_dim=TPB,
                    )
                    ctx.enqueue_function[_e_context_self](
                        probs.tt["gpu", B * HEADS * QL * KL_SELF](),
                        v.tt["gpu", B, KL_SELF * DIM](),
                        out.tt["gpu", B, QN](),
                        grid_dim=(n_cx + TPB - 1) // TPB, block_dim=TPB,
                    )
                else:
                    ctx.enqueue_function[_e_pack_cross](
                        k.tt["gpu", B, KL_CROSS * DIM](),
                        kt.tt["gpu", B * KL_CROSS * DIM](),
                        grid_dim=(B * KL_CROSS * DIM + TPB - 1) // TPB, block_dim=TPB,
                    )
                    ctx.enqueue_function[_e_scores_cross](
                        q.tt["gpu", B, QN](),
                        kt.tt["gpu", B * KL_CROSS * DIM](),
                        mask_t.tt["gpu", QL * KL_CROSS](),
                        probs.tt["gpu", B * HEADS * QL * KL_CROSS](),
                        grid_dim=(n_sc + TPB - 1) // TPB, block_dim=TPB,
                    )
                    ctx.enqueue_function[_e_softmax_cross](
                        probs.tt["gpu", B * HEADS * QL * KL_CROSS](),
                        grid_dim=ba_warp_rows_grid(ROWS), block_dim=TPB,
                    )
                    ctx.enqueue_function[_e_context_cross](
                        probs.tt["gpu", B * HEADS * QL * KL_CROSS](),
                        v.tt["gpu", B, KL_CROSS * DIM](),
                        out.tt["gpu", B, QN](),
                        grid_dim=(n_cx + TPB - 1) // TPB, block_dim=TPB,
                    )
            # ⚠ SYNCHRONISE BEFORE STOPPING THE CLOCK: the enqueue returns
            # immediately, and timing it measures nothing.
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if rep >= WARMUP:
                times.append(dt)
        out.download(ctx)
        var best = times[0]
        for t in times:
            if t < best:
                best = t
        if variant == 0:
            for n in range(B * QN):
                base_a.append(out.data[n])
        var cmp = _compare(out, base_a, refv)
        var names: List[String] = [
            String("A  row-per-thread (was shipped)"),
            String("B  same kernel, block 32"),
            String("C  branchless, block 32"),
            String("D  one-pass softmax"),
            String("E  pack+3 kernels (SHIPS now)"),
        ]
        rows.append(
            Row(names[variant], best, _median(times), macs, cmp[0], cmp[1])
        )
    return rows^


def _check(ref rows: List[Row], shape: String) -> Bool:
    """True if a variant changed the numbers — its timing is then not a result."""
    var bad = False
    for v in range(1, 3):
        if rows[v].bits_vs_a != 0:
            print("  ⚠⚠ " + shape + " " + rows[v].name + " changed "
                  + String(rows[v].bits_vs_a)
                  + " elements — it was meant to be bit-identical to A.")
            bad = True
    for v in range(5):
        # 1e-3 std units: float32 accumulation over ~185 terms lands near
        # 1e-5, and every real defect tried (wrong head offset, dropped mask,
        # unnormalised softmax) lands at O(0.1..10). Checked by mutation.
        if rows[v].max_rel_vs_ref > GATE_STD_UNITS:
            print("  ⚠⚠ " + shape + " " + rows[v].name
                  + " disagrees with the float64 reference by "
                  + String(rows[v].max_rel_vs_ref) + " std units.")
            bad = True
    return bad


def _report(ref rows: List[Row]):
    print(
        "   " + _pad(String("variant"), 28) + _pad(String("best ms"), 10)
        + _pad(String("median"), 10) + _pad(String("vs A"), 8)
        + _pad(String("GFLOPS"), 9) + _pad(String("% Orin"), 9)
        + _pad(String("bits != A"), 11) + "max err vs f64 (std units)"
    )
    for r in rows:
        var gflops = 2.0 * r.nominal_macs / (r.best_ms / 1e3) / 1e9
        print(
            "   " + _pad(r.name, 28) + _pad(_fmt(r.best_ms, 3), 10)
            + _pad(_fmt(r.median_ms, 3), 10)
            + _pad(_fmt(rows[0].best_ms / r.best_ms, 2) + "x", 8)
            + _pad(_fmt(gflops, 1), 9)
            + _pad(_fmt(100.0 * gflops / ORIN_FP32_PEAK_GFLOPS, 2), 9)
            + _pad(String(r.bits_vs_a), 11) + String(r.max_rel_vs_ref)
        )




def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    var ctx = DeviceContext()
    print("=" * 96)
    print("SmolVLA expert attention — five variants, " + String(ctx.name()))
    print("=" * 96)
    var self_rows = run_shape(String("SELF layer (causal chunk)"), KL_SELF, ctx)
    _report(self_rows)
    var cross_rows = run_shape(String("CROSS layer (prefix only, no mask)"), KL_CROSS, ctx)
    _report(cross_rows)

    print("")
    var bad = _check(self_rows, String("SELF"))
    if _check(cross_rows, String("CROSS")):
        bad = True
    if bad:
        raise Error("a variant produced different numbers — its timing is not a result")
    print("  all variants agree: B and C bit-identical to A, every variant within 1e-3 std units of float64")
    print("")
    print("  READ: B vs A = occupancy.  C vs B on SELF but not CROSS = lane divergence.")
    print("        D vs C = the duplicated q.k pass.   E = the row-per-thread launch shape.")
