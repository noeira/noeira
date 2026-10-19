# +--------------------------------------------------------------------------+ #
# | CrossAttention's GPU forward, stage by stage — and candidates per stage
# +--------------------------------------------------------------------------+ #
"""Where the 38 ms of one SigLIP attention layer goes on the Orin, and what
replaces the biggest stage.

    pixi run -e jetson cross-attn-stages-bench-jetson
    pixi run -e apple mojo run -I . benchmarks/cross_attention_stages_bench.mojo

⚠ K1 AND S3 SHIPPED (`cross_attention.mojo`, 18 Sep) after this benchmark
measured them on the Orin. The eight stages below are still the PRE-PORT
pipeline — `_xa_softmax_kernel` and pack-then-transpose are kept in the tree as
the reference these two are timed and bit-compared against — so the stage sum
is now LARGER than the whole shipped forward, by exactly what they bought.

⚠⚠ WHY STAGES, NOT WHOLE VARIANTS. `cross_attention_bench.mojo` ranked four
whole forwards, and the Metal ranking inverted on the board: the element-
indexed forward that won on the M1 lost 2.8x on the Orin. A whole-forward A/B
cannot say which stage moved. So this times each stage of the SHIPPED forward
alone (synchronised on both sides, best of REPS), then times candidates for a
stage against that stage, on the same inputs:

    pack K + transpose  vs  K1  pack token-major K straight into Kt
    softmax (shipped: one block per (b,h), threads striding rows, TPB=128,
             writes the scores slab AND mirrors the cache)
                        vs  S1  one thread per row, block 32, cache only
                        vs  S2  one thread per row, block 128, cache only
                        vs  S3  a row pass for (max, 1/denom), then ONE THREAD
                                PER ELEMENT for the weights, cache only

⚠ EACH CANDIDATE IS CHECKED AGAINST THE STAGE IT REPLACES, bits and std units,
on pre-poisoned destinations (a fresh buffer can be handed memory still holding
the previous answer). S1/S2 keep the shipped arithmetic, stored between passes,
so they should be bit-identical; S3 recomputes `s*scale - mx` in one expression,
which a GPU compiler may contract — its bit count says whether it did.

⚠ The ACT shapes are here because the primitive is shared: a softmax that wins
at 1024x1024 and loses at 62x62 x 16 is not shippable as-is.
"""

from std.math import exp, sqrt
from max.gpu import global_idx
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.mm import bmm
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.cross_attention import (
    CrossAttention,
    XATTN_DENOM_FLOOR,
    XATTN_MASK_NEG,
    _xa_pack_kernel,
    _xa_softmax_kernel,
    _xa_transpose_k_kernel,
    _xa_unpack_kernel,
)


comptime WARMUP = 3
comptime REPS = 10
comptime GATE_STD_UNITS = 1.0e-4
comptime POISON = 12345.0
comptime ROW_BLOCK_SMALL = 32


# ═══════════════════════════════════════════════════════════════════════════
# candidates
# ═══════════════════════════════════════════════════════════════════════════


def _xk_pack_kt_kernel[
    BATCH: Int, DIM: Int, NH: Int, KL: Int, HD: Int, PK: Int
](
    dst: Tile[DT, PK],
    src: Tile[DT, BATCH, KL * DIM],
):
    """K1: token-major `(B, KL, DIM)` -> `(BH, HD, KL)` in one pass."""
    var idx = Int(global_idx.x)
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


def _xs_row_softmax_kernel[
    BATCH: Int, NH: Int, QL: Int, KL: Int, HD: Int, MASKED: Bool, SC: Int
](
    scores: Tile[DT, SC],
    attn: Tile[DT, SC],
    mask: Tile[DT, BATCH, KL],
):
    """S1/S2: one thread per (b,h,i). The shipped kernel's three passes and its
    arithmetic, stored between passes, written to the cache only."""
    var r = Int(global_idx.x)
    if r >= BATCH * NH * QL:
        return
    var b = r // (NH * QL)
    var base = r * KL
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var mx = XATTN_MASK_NEG
    for j in range(KL):
        var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=base + j]) * scale
        comptime if MASKED:
            if rebind[Scalar[DT]](mask.ptr[unsafe_offset=b * KL + j]) < Scalar[DT](0.5):
                sv = XATTN_MASK_NEG
        attn.ptr[unsafe_offset=base + j] = sv
        if sv > mx:
            mx = sv
    var se = Scalar[DT](0)
    for j in range(KL):
        var e = exp(rebind[Scalar[DT]](attn.ptr[unsafe_offset=base + j]) - mx)
        attn.ptr[unsafe_offset=base + j] = e
        se += e
    var denom = se if se > XATTN_DENOM_FLOOR else XATTN_DENOM_FLOOR
    var inv = Scalar[DT](1) / denom
    for j in range(KL):
        var w = rebind[Scalar[DT]](attn.ptr[unsafe_offset=base + j]) * inv
        comptime if MASKED:
            if rebind[Scalar[DT]](mask.ptr[unsafe_offset=b * KL + j]) < Scalar[DT](0.5):
                w = Scalar[DT](0)
        attn.ptr[unsafe_offset=base + j] = w


def _xs_row_stats_kernel[
    BATCH: Int, NH: Int, QL: Int, KL: Int, HD: Int, MASKED: Bool, SC: Int, ROWS: Int
](
    scores: Tile[DT, SC],
    stats: Tile[DT, ROWS * 2],
    mask: Tile[DT, BATCH, KL],
):
    """S3, pass 1: per row, the max scaled score and 1/denominator. Reads only."""
    var r = Int(global_idx.x)
    if r >= ROWS:
        return
    var b = r // (NH * QL)
    var base = r * KL
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var mx = XATTN_MASK_NEG
    for j in range(KL):
        var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=base + j]) * scale
        comptime if MASKED:
            if rebind[Scalar[DT]](mask.ptr[unsafe_offset=b * KL + j]) < Scalar[DT](0.5):
                sv = XATTN_MASK_NEG
        if sv > mx:
            mx = sv
    var se = Scalar[DT](0)
    for j in range(KL):
        var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=base + j]) * scale
        comptime if MASKED:
            if rebind[Scalar[DT]](mask.ptr[unsafe_offset=b * KL + j]) < Scalar[DT](0.5):
                sv = XATTN_MASK_NEG
        se += exp(sv - mx)
    var denom = se if se > XATTN_DENOM_FLOOR else XATTN_DENOM_FLOOR
    stats.ptr[unsafe_offset=2 * r] = mx
    stats.ptr[unsafe_offset=2 * r + 1] = Scalar[DT](1) / denom


def _xs_element_softmax_kernel[
    BATCH: Int, NH: Int, QL: Int, KL: Int, HD: Int, MASKED: Bool, SC: Int, ROWS: Int
](
    scores: Tile[DT, SC],
    stats: Tile[DT, ROWS * 2],
    mask: Tile[DT, BATCH, KL],
    attn: Tile[DT, SC],
):
    """S3, pass 2: one thread per (b,h,i,j)."""
    var idx = Int(global_idx.x)
    if idx >= SC:
        return
    var j = idx % KL
    var r = idx // KL
    comptime if MASKED:
        var b = r // (NH * QL)
        if rebind[Scalar[DT]](mask.ptr[unsafe_offset=b * KL + j]) < Scalar[DT](0.5):
            attn.ptr[unsafe_offset=idx] = Scalar[DT](0)
            return
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var sv = rebind[Scalar[DT]](scores.ptr[unsafe_offset=idx]) * scale
    var e = exp(sv - rebind[Scalar[DT]](stats.ptr[unsafe_offset=2 * r]))
    attn.ptr[unsafe_offset=idx] = e * rebind[Scalar[DT]](
        stats.ptr[unsafe_offset=2 * r + 1]
    )


# ═══════════════════════════════════════════════════════════════════════════
# host
# ═══════════════════════════════════════════════════════════════════════════


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed * 2 + 1)

    def __init__(out self, *, deinit move: Self):
        self.s = move.s

    def unit(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(Float64((self.s >> 40) & UInt64(0xFFFFFF)) / 16777216.0)


def _fmt(x: Float64, digits: Int) -> String:
    var p = 1.0
    for _ in range(digits):
        p *= 10.0
    return String(Float64(Int(x * p + 0.5)) / p)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out^


def _compare(mut got: Tensor, mut want: Tensor, n: Int) -> Tuple[Int, Float64]:
    """(bits different, max|got-want| / rms(want)). Both already downloaded."""
    var bits = 0
    var worst = 0.0
    var ss = 0.0
    for i in range(n):
        var w = Float64(want.data[i])
        var g = Float64(got.data[i])
        if got.data[i] != want.data[i]:
            bits += 1
        worst = max(worst, abs(g - w))
        ss += w * w
    var rms = sqrt(ss / Float64(n))
    if rms < 1.0e-30:
        rms = 1.0
    return (bits, worst / rms)


def run_shape[
    B: Int, DIM: Int, H: Int, QL: Int, KL: Int, MASKED: Bool
](name: String, mut ctx: DeviceContext) raises -> Bool:
    comptime HD = DIM // H
    comptime BH = B * H
    comptime QN = B * QL * DIM
    comptime KN = B * KL * DIM
    comptime PQ = BH * QL * HD
    comptime PK = BH * KL * HD
    comptime SC = BH * QL * KL
    comptime ROWS = BH * QL
    comptime ATTN_SIZE = H * QL * KL
    comptime XA = CrossAttention[DIM, H, QL, KL, MASKED]

    comptime lay_q = RowMajor[B, QL * DIM]
    comptime lay_kv = RowMajor[B, KL * DIM]
    comptime lay_pq = RowMajor[PQ]
    comptime lay_pk = RowMajor[PK]
    comptime lay_s = RowMajor[SC]
    comptime lay_a = RowMajor[B, ATTN_SIZE]
    comptime lay_m = RowMajor[B, KL]
    comptime lay_st = RowMajor[ROWS * 2]
    comptime qblocks = (QN + TPB - 1) // TPB
    comptime kblocks = (KN + TPB - 1) // TPB

    var rng = Lcg(B * 7919 + QL * 131 + KL)
    var q = Tensor.alloc(QN)
    var k = Tensor.alloc(KN)
    var v = Tensor.alloc(KN)
    for n in range(QN):
        q.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    for n in range(KN):
        k.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
        v.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    var m = Tensor.alloc(B * KL)
    for b in range(B):
        var valid = KL - (b % 7)
        if valid < 1:
            valid = 1
        for j in range(KL):
            m.data[b * KL + j] = Scalar[DT](1.0) if (not MASKED or j < valid) else Scalar[DT](0.0)
    q.upload(ctx)
    k.upload(ctx)
    v.upload(ctx)
    m.upload(ctx)

    var sq = Tensor()
    var sk = Tensor()
    var skt = Tensor()
    var skt1 = Tensor()
    var sv = Tensor()
    var raw = Tensor()      # Q.Kt, unscaled — every softmax reads a copy of this
    var ss = Tensor()       # the shipped softmax works IN PLACE on this
    var pout = Tensor()
    var out = Tensor()
    var attn0 = Tensor()    # shipped cache
    var attn1 = Tensor()
    var attn2 = Tensor()
    var attn3 = Tensor()
    var stats = Tensor()
    sq.ensure_gpu(ctx, PQ)
    sk.ensure_gpu(ctx, PK)
    skt.ensure_gpu(ctx, PK)
    skt1.ensure_gpu(ctx, PK)
    sv.ensure_gpu(ctx, PK)
    raw.ensure_gpu(ctx, SC)
    ss.ensure_gpu(ctx, SC)
    pout.ensure_gpu(ctx, PQ)
    out.ensure_gpu(ctx, QN)
    attn0.ensure_gpu(ctx, SC)
    attn1.ensure_gpu(ctx, SC)
    attn2.ensure_gpu(ctx, SC)
    attn3.ensure_gpu(ctx, SC)
    stats.ensure_gpu(ctx, ROWS * 2)
    for t in [skt1.dev.value(), attn0.dev.value(), attn1.dev.value(),
              attn2.dev.value(), attn3.dev.value()]:
        t.enqueue_fill(Scalar[DT](POISON))

    # The shipped pipeline once, to have realistic scores in `raw`.
    ctx.enqueue_function[_xa_pack_kernel[B, DIM, H, QL, HD, PQ]](
        sq.tt["gpu", PQ](), q.tt["gpu", B, QL * DIM](), grid_dim=qblocks, block_dim=TPB)
    ctx.enqueue_function[_xa_pack_kernel[B, DIM, H, KL, HD, PK]](
        sk.tt["gpu", PK](), k.tt["gpu", B, KL * DIM](), grid_dim=kblocks, block_dim=TPB)
    ctx.enqueue_function[_xa_transpose_k_kernel[BH, KL, HD, PK]](
        skt.tt["gpu", PK](), sk.tt["gpu", PK](),
        grid_dim=(PK + TPB - 1) // TPB, block_dim=TPB)
    ctx.enqueue_function[_xa_pack_kernel[B, DIM, H, KL, HD, PK]](
        sv.tt["gpu", PK](), v.tt["gpu", B, KL * DIM](), grid_dim=kblocks, block_dim=TPB)
    bmm[A0=BH, A1=QL, A2=HD, B0=BH, B1=HD, B2=KL, O0=BH, O1=QL, O2=KL](
        raw.dev.value(), sq.dev.value(), skt.dev.value(), ctx)
    ctx.synchronize()

    print("")
    print(
        "── " + name + ": B " + String(B) + ", Q " + String(QL) + " x KV "
        + String(KL) + ", " + String(H) + " heads x " + String(HD)
        + (", MASKED" if MASKED else "") + " ──"
    )
    print("   " + _pad(String("stage"), 42) + _pad(String("best ms"), 22)
          + _pad(String("share"), 8) + "bits != shipped / std u.")

    comptime N_STAGES = 12
    var names: List[String] = [
        String("pack Q"),
        String("pack K (pre-port)"),
        String("transpose K -> Kt (pre-port)"),
        String("pack V"),
        String("bmm Q.Kt"),
        String("softmax, pre-port (block per (b,h))"),
        String("bmm A.V"),
        String("unpack"),
        String("K1  pack K into Kt  [SHIPPED]"),
        String("S1  row thread, block 32"),
        String("S2  row thread, block 128"),
        String("S3  row stats + element thread  [SHIPPED]"),
    ]
    var best = List[Float64](length=N_STAGES, fill=1.0e30)
    var mod = XA.make["gpu", Deterministic](Optional(ctx))
    var fwd_best = 1.0e30

    for rep in range(WARMUP + REPS):
        # whole shipped forward, for the "sum of stages vs whole" line
        ctx.synchronize()
        var tf = perf_counter_ns()
        comptime if MASKED:
            mod.forward["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[4, MutAnyOrigin](q, k, v, m)
                ), out, ctx,
            )
        else:
            mod.forward["gpu", B](
                rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                    TensorRefs[3, MutAnyOrigin](q, k, v)
                ), out, ctx,
            )
        ctx.synchronize()
        var dtf = Float64(perf_counter_ns() - tf) / 1e6
        if rep >= WARMUP and dtf < fwd_best:
            fwd_best = dtf

        for s in range(N_STAGES):
            if s == 5:
                # fresh raw scores for the in-place kernel, outside the clock
                ctx.enqueue_copy(ss.dev.value(), raw.dev.value())
            ctx.synchronize()
            var t0 = perf_counter_ns()
            if s == 0:
                ctx.enqueue_function[_xa_pack_kernel[B, DIM, H, QL, HD, PQ]](
                    sq.tt["gpu", PQ](), q.tt["gpu", B, QL * DIM](),
                    grid_dim=qblocks, block_dim=TPB)
            elif s == 1:
                ctx.enqueue_function[_xa_pack_kernel[B, DIM, H, KL, HD, PK]](
                    sk.tt["gpu", PK](), k.tt["gpu", B, KL * DIM](),
                    grid_dim=kblocks, block_dim=TPB)
            elif s == 2:
                ctx.enqueue_function[_xa_transpose_k_kernel[BH, KL, HD, PK]](
                    skt.tt["gpu", PK](), sk.tt["gpu", PK](),
                    grid_dim=(PK + TPB - 1) // TPB, block_dim=TPB)
            elif s == 3:
                ctx.enqueue_function[_xa_pack_kernel[B, DIM, H, KL, HD, PK]](
                    sv.tt["gpu", PK](), v.tt["gpu", B, KL * DIM](),
                    grid_dim=kblocks, block_dim=TPB)
            elif s == 4:
                bmm[A0=BH, A1=QL, A2=HD, B0=BH, B1=HD, B2=KL, O0=BH, O1=QL, O2=KL](
                    raw.dev.value(), sq.dev.value(), skt.dev.value(), ctx)
            elif s == 5:
                ctx.enqueue_function[
                    _xa_softmax_kernel[B, H, QL, KL, HD, MASKED, ATTN_SIZE, SC, BH]
                ](ss.tt["gpu", SC](), attn0.tt["gpu", B, ATTN_SIZE](),
                  m.tt["gpu", B, KL](), grid_dim=BH, block_dim=TPB)
            elif s == 6:
                bmm[A0=BH, A1=QL, A2=KL, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=HD](
                    pout.dev.value(), attn0.dev.value(), sv.dev.value(), ctx)
            elif s == 7:
                ctx.enqueue_function[_xa_unpack_kernel[B, DIM, H, QL, HD, PQ]](
                    out.tt["gpu", B, QL * DIM](), pout.tt["gpu", PQ](),
                    grid_dim=qblocks, block_dim=TPB)
            elif s == 8:
                ctx.enqueue_function[_xk_pack_kt_kernel[B, DIM, H, KL, HD, PK]](
                    skt1.tt["gpu", PK](), k.tt["gpu", B, KL * DIM](),
                    grid_dim=(PK + TPB - 1) // TPB, block_dim=TPB)
            elif s == 9 or s == 10:
                var blk = ROW_BLOCK_SMALL if s == 9 else TPB
                comptime kr = _xs_row_softmax_kernel[B, H, QL, KL, HD, MASKED, SC]
                if s == 9:
                    ctx.enqueue_function[kr](
                        raw.tt["gpu", SC](), attn1.tt["gpu", SC](),
                        m.tt["gpu", B, KL](),
                        grid_dim=(ROWS + blk - 1) // blk, block_dim=blk)
                else:
                    ctx.enqueue_function[kr](
                        raw.tt["gpu", SC](), attn2.tt["gpu", SC](),
                        m.tt["gpu", B, KL](),
                        grid_dim=(ROWS + blk - 1) // blk, block_dim=blk)
            else:
                ctx.enqueue_function[
                    _xs_row_stats_kernel[B, H, QL, KL, HD, MASKED, SC, ROWS]
                ](raw.tt["gpu", SC](), stats.tt["gpu", ROWS * 2](),
                  m.tt["gpu", B, KL](),
                  grid_dim=(ROWS + ROW_BLOCK_SMALL - 1) // ROW_BLOCK_SMALL,
                  block_dim=ROW_BLOCK_SMALL)
                ctx.enqueue_function[
                    _xs_element_softmax_kernel[B, H, QL, KL, HD, MASKED, SC, ROWS]
                ](raw.tt["gpu", SC](), stats.tt["gpu", ROWS * 2](),
                  m.tt["gpu", B, KL](), attn3.tt["gpu", SC](),
                  grid_dim=(SC + TPB - 1) // TPB, block_dim=TPB)
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if rep >= WARMUP and dt < best[s]:
                best[s] = dt

    # ── correctness of every candidate against the stage it replaces ──────
    skt.download(ctx)
    skt1.download(ctx)
    attn0.download(ctx)
    attn1.download(ctx)
    attn2.download(ctx)
    attn3.download(ctx)
    var ok = True
    var verdict = List[String](length=N_STAGES, fill=String(""))
    var k1 = _compare(skt1, skt, PK)
    verdict[8] = String(k1[0]) + "/" + String(PK) + "  " + String(k1[1])
    if k1[1] > GATE_STD_UNITS:
        ok = False
        verdict[8] += "  ⚠⚠ DISAGREES"
    var cands = List[Int]()
    cands.append(9)
    cands.append(10)
    cands.append(11)
    for c in cands:
        var r: Tuple[Int, Float64] = (0, 0.0)
        if c == 9:
            r = _compare(attn1, attn0, SC)
        elif c == 10:
            r = _compare(attn2, attn0, SC)
        else:
            r = _compare(attn3, attn0, SC)
        verdict[c] = String(r[0]) + "/" + String(SC) + "  " + String(r[1])
        if r[1] > GATE_STD_UNITS:
            ok = False
            verdict[c] += "  ⚠⚠ DISAGREES"

    var shipped_sum = 0.0
    for s in range(8):
        shipped_sum += best[s]
    for s in range(N_STAGES):
        if s == 8:
            print("   " + "-" * 40 + " candidates")
        var share = _fmt(100.0 * best[s] / shipped_sum, 1) + "%" if s < 8 else String("")
        var vs = String("")
        if s == 8:
            vs = "  (" + _fmt((best[1] + best[2]) / best[8], 2) + "x)"
        elif s >= 9:
            vs = "  (" + _fmt(best[5] / best[s], 2) + "x)"
        print("   " + _pad(names[s], 42) + _pad(_fmt(best[s], 3) + vs, 22)
              + _pad(share, 8) + verdict[s])
    print("   sum of the eight stages " + _fmt(shipped_sum, 3)
          + " ms  vs whole shipped forward " + _fmt(fwd_best, 3)
          + " ms  (the forward now runs K1 + S3, so it is FASTER than the sum)")
    return ok


def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    var ctx = DeviceContext()
    print("=" * 100)
    print("CrossAttention forward stages — " + String(ctx.name()))
    print("=" * 100)
    var ok = True
    if not run_shape[1, 768, 12, 1024, 1024, False](String("SigLIP self-attention"), ctx):
        ok = False
    if not run_shape[16, 256, 8, 162, 162, False](String("ACT encoder self (train)"), ctx):
        ok = False
    if not run_shape[1, 256, 8, 60, 162, False](String("ACT decoder cross (deploy)"), ctx):
        ok = False
    if not run_shape[16, 256, 8, 62, 62, True](String("ACT CVAE encoder, masked (train)"), ctx):
        ok = False
    print("")
    if not ok:
        raise Error("a candidate disagrees with the stage it replaces — its timing is not a result")
    print("  every candidate within 1e-4 std units of the shipped stage")
