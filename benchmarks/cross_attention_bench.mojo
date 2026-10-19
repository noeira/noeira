# +--------------------------------------------------------------------------+ #
# | CrossAttention's GPU forward — sixteen variants, at SigLIP's shape AND ACT's
# +--------------------------------------------------------------------------+ #
"""Which CrossAttention forward to ship, measured at the shapes that use it.

    pixi run -e jetson cross-attn-bench-jetson
    pixi run -e apple mojo run -I . benchmarks/cross_attention_bench.mojo

⚠⚠ WHY THIS EXISTS. nsys on the Orin: SmolVLA's two SigLIP towers spend ~1.88 s
of a 2.33 s query in this primitive's attention — the Q.Kt matmul through MAX's
naive batched matmul (49 ms/layer, 1.4% of fp32 peak), the softmax (20 ms), and
the A.V matmul (8.7 ms). The same method that took the expert's attention 6.1x
on the board (`smolvla_block_attention_bench.mojo`) is the obvious candidate.

⚠⚠ BUT THIS PRIMITIVE IS SHARED, and that changes what "wins" means. ACT uses
it four ways — encoder self-attention, decoder self and cross, and the CVAE
encoder, its ONLY masked user — and ACT runs at 30 Hz on the board today. So a
variant is only shippable if it is faster at SigLIP's shape AND no slower at
ACT's, at deploy batch (1) AND training batch (16). One run answers both.

⚠ ONE MORE REASON NOT TO ASSUME: the element-indexed kernels reached only
~1.9% of the Orin's peak on the expert. MAX's naive matmul already does 7.7% on
SigLIP's A.V. Replacing a 7.7% stage with a 1.9% one would be a regression that
a "same method worked last time" argument would ship. Hence C, which keeps the
matmul for A.V:

    A  the shipped forward: pack, Kt CONTIGUOUS, bmm Q.Kt, softmax + cache, bmm A.V
    B  the pre-port forward: same, but bmm Q.Kt TRANSPOSED on the packed keys
    C  element-indexed scores + row softmax INTO THE CACHE, bmm A.V
    D  element-indexed everything, no pack, no unpack

⚠ THE CACHE IS PART OF THE CONTRACT. The GPU backward reads the softmax weights
from `self.attn`, laid out [b, h, i, j] — so every variant is checked on its
cached weights as well as its output, against a FLOAT64 reference, in std
units. C and D write the scores straight into that layout, so the weights are
materialised once instead of into a scratch slab and then copied.

⚠⚠ THE VERDICT, 17 Sep, before the port (best ms; A was then the transposed
path, now B). The Metal ranking did NOT carry to the board:

    shape                       Orin: old    Kt contig.  C       D      | 5090: old  Kt contig.  C       D
    SigLIP  B1 1024x1024        78.3         38.0        107.7   105.0  | 5.57       4.75        2.67    2.76
    ACT enc self  B16 162       12.9         7.68        14.7    14.3   | 0.410      0.309       0.393   0.381
    ACT dec cross B1 60x162     0.335        0.225       0.392   0.376  | 0.085      0.091       0.038   0.030
    ACT CVAE masked B16 62      1.85         1.09        2.22    2.14   | 0.092      0.079       0.072   0.066

On the Orin the element-indexed variants LOSE at every shape and the whole
cost was the transposed matmul — so "Kt contiguous" shipped: faster at every
Orin shape, bit-identical everywhere. On a 5090 C/D are a further ~1.8x at
SigLIP; that is a separate, per-device decision, not taken here.

⚠ ROWS P AND Q ARE MAX'S OWN FLASH ATTENTION (`nn.attention.gpu.mha`, the
package ships it and it imports). Our `[B, L*DIM]` slab with the heads
contiguous inside DIM IS the BSHD layout it takes, so the rows hand it our
buffers with no pack. Its fp32 path runs BOTH matmuls on the tensor cores as
TF32 (`get_mma_shape[float32, float32]` = 16x8x8, 10 mantissa bits per
operand), so those two rows are gated in the GEMM path's band, not the fp32
kernels'. P is what `flash_attention` picks by itself: its admission test
admits depth 64 on A100 / H100 / B200 BY NAME, the Orin resolves to the
`OrinNano` entry and a 5090 to its own, so on both boards P is MAX's naive
two-BMM fallback. Q calls `flash_attention_dispatch` with the admission test
overridden — the FA2 kernel itself, which is what a depth-64 Ampere part
actually runs. NVIDIA only; unmasked, head dim 64 only (its config pads the
head dim to 64, and the masked user needs a `MaterializedMask` we do not build).
"""

from std.math import exp, sqrt
from max.gpu import global_idx
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major, Idx
from noeira.core.tiles import RowMajor, Tile
from std.sys.info import default_accelerator
from nn.attention.gpu.mha import flash_attention, flash_attention_dispatch
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import TileTensorMHAOperand

from noeira.nn.constants import DT, TPB
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.mm import bmm
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.cross_attention import (
    CrossAttention,
    XATTN_MASK_NEG,
    _xa_pack_kernel,
    _xa_softmax_kernel,
    _xa_unpack_kernel,
    _xa_fused_kernel,
    xa_fused_block,
    xa_fused_routes_to_max,
    XA_FUSED_BQ,
)
from noeira.deep_agents.smolvla.block_attention import (
    _ba_context_kernel,
    _ba_softmax_kernel,
    ba_warp_rows_grid,
)


comptime WARMUP = 2
comptime REPS = 8
comptime GATE_STD_UNITS = 1.0e-3
comptime GATE_TF32_STD_UNITS = 5.0e-2
"""The band for MAX's rows: TF32 operands (10 mantissa bits) in both matmuls.
The network's GEMMs already run in that band on CUDA (the Metal-vs-CUDA
memory: 8.8e-08 on Metal, 1.25e-03 on the 5090, weights bit-identical)."""
comptime POISON = 12345.0
"""Far outside any softmax weight (0..1) or context value (|v| <= 1 here)."""


# ═══════════════════════════════════════════════════════════════════════════
# the two new kernels (softmax and context are block_attention's, already gated)
# ═══════════════════════════════════════════════════════════════════════════


def _xe_scores_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int, MASKED: Bool
](
    q: Tile[DT, BATCH, Q * D],
    k: Tile[DT, BATCH, KL * D],
    mask: Tile[DT, BATCH, KL],
    attn: Tile[DT, BATCH * NH * Q * KL],
):
    """One thread per (b, h, i, j), written straight into the cache layout.

    The key-padding mask is PER SAMPLE — `m[b, j]`, 1.0 attend / 0.0 ignore —
    which is this primitive's mask, not block_attention's `[Q, KV]` one.
    """
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
    var kb = b * (KL * D) + j * D + h * H
    var s = Scalar[DT](0)
    for d in range(H):
        s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
            Scalar[DT]
        ](k.ptr[unsafe_offset = kb + d])
    s = s * (Scalar[DT](1.0) / sqrt(Scalar[DT](H)))
    comptime if MASKED:
        if rebind[Scalar[DT]](mask.ptr[unsafe_offset = b * KL + j]) < Scalar[DT](0.5):
            s = XATTN_MASK_NEG
    attn.ptr[unsafe_offset = idx] = s


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


def _std_units(ref got: Tensor, ref refv: List[Float64]) -> Float64:
    """max |got - ref| / rms(ref). See `smolvla_block_attention_bench.mojo`
    for why relative-to-|ref| is the wrong question here."""
    var ss = 0.0
    for n in range(len(refv)):
        ss += refv[n] * refv[n]
    var rms = sqrt(ss / Float64(len(refv)))
    if rms < 1.0e-30:
        rms = 1.0
    var worst = 0.0
    for n in range(len(refv)):
        var e = abs(Float64(got.data[n]) - refv[n])
        if e > worst:
            worst = e
    return worst / rms


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


def _launch_fused[
    B: Int, DIM: Int, H: Int, QL: Int, KL: Int, HD: Int, MASKED: Bool,
    R: Int, S: Int, KU: Int, DOT_FMA: Bool = False, EXP2: Bool = False,
](
    mut q: Tensor, mut k: Tensor, mut v: Tensor, mut m: Tensor,
    mut out: Tensor, mut ctx: DeviceContext,
) raises:
    """One fused-kernel configuration, exactly as `_forward_gpu_fused`
    launches the shipped one."""
    comptime lay_q = RowMajor[B, QL * DIM]
    comptime lay_kv = RowMajor[B, KL * DIM]
    comptime lay_m = RowMajor[B, KL]
    comptime qtiles = (QL + XA_FUSED_BQ - 1) // XA_FUSED_BQ
    ctx.enqueue_function[
        _xa_fused_kernel[
            B, DIM, H, QL, KL, HD, MASKED, R, S, KU, DOT_FMA, EXP2
        ]
    ](
        q.tt["gpu", B, QL * DIM](), k.tt["gpu", B, KL * DIM](), v.tt["gpu", B, KL * DIM](),
        m.tt["gpu", B, KL](), out.tt["gpu", B, QL * DIM](),
        grid_dim=(qtiles, H, B), block_dim=xa_fused_block[R, S](),
    )


def _launch_max_fa[
    B: Int, DIM: Int, H: Int, QL: Int, KL: Int, HD: Int, FORCE: Bool
](
    mut q: Tensor, mut k: Tensor, mut v: Tensor, mut out: Tensor,
    mut ctx: DeviceContext,
) raises:
    """MAX's flash attention over our slabs UNCHANGED — `[B, L*DIM]` with the
    heads contiguous inside DIM is the BSHD it wants, so the views below are
    reinterpretations, not copies. FORCE=False is `flash_attention`'s own
    choice (naive two-BMM on any part its depth table does not name);
    FORCE=True is `flash_attention_dispatch` with the admission test
    overridden: the FA2 kernel, TF32 tensor-core matmuls, online softmax."""
    comptime if not default_accelerator().is_nvidia_gpu():
        raise Error("MAX's flash attention rows run on NVIDIA only")
    else:
        # ⚠ `.unsafe_ptr()` severs the borrow; the Tensors are the caller's
        # live locals and outlive the synchronised launch.
        var qp = q.dev.value().unsafe_ptr().as_imm().as_unsafe_any_origin()
        var kp = k.dev.value().unsafe_ptr().as_imm().as_unsafe_any_origin()
        var vp = v.dev.value().unsafe_ptr().as_imm().as_unsafe_any_origin()
        var q_tt = TileTensor(qp, row_major((B, QL, Idx[H], Idx[HD])))
        var k_tt = TileTensor(kp, row_major((B, KL, Idx[H], Idx[HD])))
        var v_tt = TileTensor(vp, row_major((B, KL, Idx[H], Idx[HD])))
        var o_tt = TileTensor(
            out.dev.value(), row_major((B, QL, Idx[H], Idx[HD]))
        )
        var scale = Float32(1.0 / sqrt(Float64(HD)))
        comptime if FORCE:
            flash_attention_dispatch[
                kv_num_heads=H,
                ragged=False,
                sink=False,
                _is_flash_attention_applicable=True,
                _is_cache_length_accurate=True,
                _use_valid_length=False,
            ](
                o_tt.to_layout_tensor(),
                q_tt.to_layout_tensor(),
                TileTensorMHAOperand(k_tt),
                TileTensorMHAOperand(v_tt),
                NullMask(),
                QL,
                KL,
                scale,
                False,
                ctx,
            )
        else:
            flash_attention(o_tt, q_tt, k_tt, v_tt, NullMask(), scale, ctx)


def run_shape[
    B: Int, DIM: Int, H: Int, QL: Int, KL: Int, MASKED: Bool
](name: String, mut ctx: DeviceContext) raises -> Bool:
    """All four variants at one shape. Returns False if any disagrees."""
    comptime HD = DIM // H
    comptime BH = B * H
    comptime QN = B * QL * DIM
    comptime KN = B * KL * DIM
    comptime PQ = BH * QL * HD
    comptime PK = BH * KL * HD
    comptime SC = BH * QL * KL
    comptime ATTN_SIZE = H * QL * KL
    comptime XA = CrossAttention[DIM, H, QL, KL, MASKED]

    var rng = Lcg(B * 7919 + QL * 131 + KL)
    var q = Tensor.alloc(QN)
    var k = Tensor.alloc(KN)
    var v = Tensor.alloc(KN)
    for n in range(QN):
        q.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    for n in range(KN):
        k.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
        v.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    # Per-sample valid prefix, at least 1 key — ACT's CVAE never masks cls/qpos.
    var m = Tensor.alloc(B * KL)
    for b in range(B):
        var valid = KL - (b % 7)
        if valid < 1:
            valid = 1
        for j in range(KL):
            m.data[b * KL + j] = Scalar[DT](1.0) if (not MASKED or j < valid) else Scalar[DT](0.0)

    # ── float64 reference: output AND the cached weights ─────────────────
    var ref_out = List[Float64](length=QN, fill=0.0)
    var ref_attn = List[Float64](length=SC, fill=0.0)
    var scale = 1.0 / sqrt(Float64(HD))
    for b in range(B):
        for h in range(H):
            for i in range(QL):
                var row = List[Float64](length=KL, fill=0.0)
                var mx = -1.0e300
                var any = False
                for j in range(KL):
                    if Float64(m.data[b * KL + j]) < 0.5:
                        row[j] = -1.0e300
                        continue
                    var s = 0.0
                    for d in range(HD):
                        s += Float64(q.data[b * QL * DIM + i * DIM + h * HD + d]) * Float64(
                            k.data[b * KL * DIM + j * DIM + h * HD + d]
                        )
                    row[j] = s * scale
                    if row[j] > mx:
                        mx = row[j]
                    any = True
                if not any:
                    continue
                var denom = 0.0
                for j in range(KL):
                    if row[j] > -1.0e299:
                        denom += exp(row[j] - mx)
                for j in range(KL):
                    if row[j] <= -1.0e299:
                        continue
                    var w = exp(row[j] - mx) / denom
                    ref_attn[((b * H + h) * QL + i) * KL + j] = w
                    for d in range(HD):
                        ref_out[b * QL * DIM + i * DIM + h * HD + d] += w * Float64(
                            v.data[b * KL * DIM + j * DIM + h * HD + d]
                        )

    q.upload(ctx)
    k.upload(ctx)
    v.upload(ctx)
    m.upload(ctx)

    # scratch shared by B, C, D
    var sq = Tensor()
    var sk = Tensor()
    var sv = Tensor()
    var ss = Tensor()
    var pout = Tensor()
    sq.ensure_gpu(ctx, PQ)
    sk.ensure_gpu(ctx, PK)
    sv.ensure_gpu(ctx, PK)
    ss.ensure_gpu(ctx, SC)
    pout.ensure_gpu(ctx, PQ)

    comptime lay_q = RowMajor[B, QL * DIM]
    comptime lay_kv = RowMajor[B, KL * DIM]
    comptime lay_pq = RowMajor[PQ]
    comptime lay_pk = RowMajor[PK]
    comptime lay_s = RowMajor[SC]
    comptime lay_a = RowMajor[B, ATTN_SIZE]
    comptime lay_m = RowMajor[B, KL]
    comptime qblocks = (QN + TPB - 1) // TPB
    comptime kblocks = (KN + TPB - 1) // TPB

    print("")
    print(
        "── " + name + ": B " + String(B) + ", Q " + String(QL) + " x KV "
        + String(KL) + ", " + String(H) + " heads x " + String(HD)
        + (", MASKED" if MASKED else "") + " ──"
    )
    print(
        "   " + _pad(String("variant"), 36) + _pad(String("best ms"), 10)
        + _pad(String("vs A"), 8) + _pad(String("out (std u.)"), 24)
        + _pad(String("cache (std u.)"), 24) + _pad(String("out bits != A"), 16)
        + "cache bits != A"
    )

    var names: List[String] = [
        String("A  shipped (Kt contiguous, bmm)"),
        String("B  pre-port: bmm Q.Kt transposed"),
        String("C  scores+softmax in cache, bmm A.V"),
        String("D  element-indexed, no pack"),
        String("F  FUSED, as forward ships it"),
        String("G  fused R2 S4 KU1 (occupancy)"),
        String("H  fused R4 S4 KU4 (ILP across keys)"),
        String("I  fused R2 S4 KU4 (both)"),
        String("J  fused R1 S2 KU1 (the first cut)"),
        String("K  fused R1 S1 KU1 (no shuffles)"),
        String("L  fused R4 S4 KU4 + FMA dot"),
        String("M  fused R4 S4 KU4 + FMA + exp2"),
        String("N  fused R1 S1 KU4 + FMA + exp2"),
        String("O  fused R2 S1 KU4 + FMA + exp2"),
        String("P  MAX flash_attention, its own pick"),
        String("Q  MAX FA2 kernel forced (TF32 mma)"),
    ]
    # MAX's rows: NVIDIA, unmasked, head dim 64 (see the header).
    comptime MAX_ROWS = default_accelerator().is_nvidia_gpu() and not MASKED and HD == 64
    comptime N_VARIANTS = 16 if MAX_ROWS else 14
    var ok = True
    var best_a = 0.0
    # ⚠ A's output and cache as their OWN lists. Without a bit count, rows that
    # all print the same error could be four genuinely identical results OR four
    # reads of one buffer — opposite conclusions, the same printout.
    var a_out = List[Scalar[DT]]()
    var a_attn = List[Scalar[DT]]()
    for variant in range(N_VARIANTS):
        var mod = XA.make["gpu", Deterministic](Optional(ctx))
        if variant == 4:
            mod.set_attr["fused_attention"](Scalar[DT](1.0))
        var out = Tensor()
        out.ensure_gpu(ctx, QN)
        var attn = Tensor()
        attn.ensure_gpu(ctx, B * ATTN_SIZE)
        # ⚠⚠ PRE-POISONED, AND THE BIT COUNT IS MEANINGLESS WITHOUT IT. Every
        # variant reports 0 bits different from A. A fresh device buffer can be
        # handed the memory A just freed, STILL HOLDING A's ANSWER — so a
        # variant whose kernel silently wrote nowhere would match A exactly.
        # A sentinel no attention output can produce makes an unwritten
        # element fail loudly instead.
        out.dev.value().enqueue_fill(Scalar[DT](POISON))
        attn.dev.value().enqueue_fill(Scalar[DT](POISON))
        var best = 1.0e30
        for rep in range(WARMUP + REPS):
            var t0 = perf_counter_ns()
            if variant == 0 or variant == 4:
                # ⚠ `rebind` BRIDGES A SYMBOLIC ARITY. `forward` takes
                # `TensorRefs[4 if MASKED else 3]`, which a literal 4 or 3
                # does not unify with while MASKED is still a parameter; inside
                # each branch the two ARE the same type once instantiated.
                # `cross_attention.mojo` bridges its own mask slot the same way.
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
            elif variant >= 14:
                comptime if MAX_ROWS:
                    if variant == 14:
                        _launch_max_fa[B, DIM, H, QL, KL, HD, False](
                            q, k, v, out, ctx
                        )
                    else:
                        _launch_max_fa[B, DIM, H, QL, KL, HD, True](
                            q, k, v, out, ctx
                        )
            elif variant >= 5:
                # ⚠ THE (R, SPLIT, KU) GRID, launched directly. F is the
                # shipped default through `forward`; these separate the two
                # hypotheses for its Orin number (6.9 ms, 1.22x): fewer
                # registers => more resident warps (R2), or independent
                # work per step to hide the per-key chain (KU4).
                if variant == 5:
                    _launch_fused[B, DIM, H, QL, KL, HD, MASKED, 2, 4, 1](
                        q, k, v, m, out, ctx
                    )
                elif variant == 6:
                    _launch_fused[B, DIM, H, QL, KL, HD, MASKED, 4, 4, 4](
                        q, k, v, m, out, ctx
                    )
                elif variant == 7:
                    _launch_fused[B, DIM, H, QL, KL, HD, MASKED, 2, 4, 4](
                        q, k, v, m, out, ctx
                    )
                elif variant == 8:
                    _launch_fused[B, DIM, H, QL, KL, HD, MASKED, 1, 2, 1](
                        q, k, v, m, out, ctx
                    )
                # ⚠ THE SECOND ROUND. The Orin put F..J all at 6.1-7.0 ms: the
                # bound is what they SHARE — instructions per row-key (a
                # multiply + reduce tree for the dot, ~25 for libm exp, the
                # softmax bookkeeping replicated across the SPLIT lanes) —
                # and these vary exactly those.
                elif variant == 9:
                    _launch_fused[B, DIM, H, QL, KL, HD, MASKED, 1, 1, 1](
                        q, k, v, m, out, ctx
                    )
                elif variant == 10:
                    _launch_fused[
                        B, DIM, H, QL, KL, HD, MASKED, 4, 4, 4, True, False
                    ](q, k, v, m, out, ctx)
                elif variant == 11:
                    _launch_fused[
                        B, DIM, H, QL, KL, HD, MASKED, 4, 4, 4, True, True
                    ](q, k, v, m, out, ctx)
                elif variant == 12:
                    _launch_fused[
                        B, DIM, H, QL, KL, HD, MASKED, 1, 1, 4, True, True
                    ](q, k, v, m, out, ctx)
                else:
                    _launch_fused[
                        B, DIM, H, QL, KL, HD, MASKED, 2, 1, 4, True, True
                    ](q, k, v, m, out, ctx)
            else:
                if variant == 1 or variant == 2:
                    # V head-major, for the A.V matmul.
                    ctx.enqueue_function[
                        _xa_pack_kernel[B, DIM, H, KL, HD, PK]
                    ](sv.tt["gpu", PK](), v.tt["gpu", B, KL * DIM](),
                      grid_dim=kblocks, block_dim=TPB)
                if variant == 1:
                    ctx.enqueue_function[
                        _xa_pack_kernel[B, DIM, H, QL, HD, PQ]
                    ](sq.tt["gpu", PQ](), q.tt["gpu", B, QL * DIM](),
                      grid_dim=qblocks, block_dim=TPB)
                    ctx.enqueue_function[
                        _xa_pack_kernel[B, DIM, H, KL, HD, PK]
                    ](sk.tt["gpu", PK](), k.tt["gpu", B, KL * DIM](),
                      grid_dim=kblocks, block_dim=TPB)
                    bmm[transpose_b=True, A0=BH, A1=QL, A2=HD, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=KL](
                        ss.dev.value(), sq.dev.value(), sk.dev.value(), ctx
                    )
                    ctx.enqueue_function[
                        _xa_softmax_kernel[B, H, QL, KL, HD, MASKED, ATTN_SIZE, SC, BH]
                    ](ss.tt["gpu", SC](), attn.tt["gpu", B, ATTN_SIZE](),
                      m.tt["gpu", B, KL](), grid_dim=BH, block_dim=TPB)
                    bmm[A0=BH, A1=QL, A2=KL, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=HD](
                        pout.dev.value(), ss.dev.value(), sv.dev.value(), ctx
                    )
                else:
                    ctx.enqueue_function[
                        _xe_scores_kernel[B, DIM, H, QL, KL, HD, MASKED]
                    ](q.tt["gpu", B, QL * DIM](), k.tt["gpu", B, KL * DIM](),
                      m.tt["gpu", B, KL](), attn.tt["gpu", SC](),
                      grid_dim=(SC + TPB - 1) // TPB, block_dim=TPB)
                    ctx.enqueue_function[_ba_softmax_kernel[B, H, QL, KL]](
                        attn.tt["gpu", SC](),
                        grid_dim=ba_warp_rows_grid(BH * QL),
                        block_dim=TPB,
                    )
                    if variant == 2:
                        bmm[A0=BH, A1=QL, A2=KL, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=HD](
                            pout.dev.value(), attn.dev.value(), sv.dev.value(), ctx
                        )
                    else:
                        ctx.enqueue_function[_ba_context_kernel[B, DIM, H, QL, KL, HD]](
                            attn.tt["gpu", SC](), v.tt["gpu", B, KL * DIM](),
                            out.tt["gpu", B, QL * DIM](),
                            grid_dim=(PQ + TPB - 1) // TPB, block_dim=TPB,
                        )
                if variant == 1 or variant == 2:
                    ctx.enqueue_function[
                        _xa_unpack_kernel[B, DIM, H, QL, HD, PQ]
                    ](out.tt["gpu", B, QL * DIM](), pout.tt["gpu", PQ](),
                      grid_dim=qblocks, block_dim=TPB)
            # ⚠ SYNCHRONISE BEFORE STOPPING THE CLOCK.
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if rep >= WARMUP and dt < best:
                best = dt
        out.download(ctx)
        var cache_err = 0.0
        if variant == 0:
            mod.attn.download(ctx)
            cache_err = _std_units(mod.attn, ref_attn)
        elif variant >= 4:
            # The fused path writes no cache — by design, not by omission;
            # `test_cross_attention_gpu_shapes.mojo` checks it stays untouched.
            cache_err = 0.0
        else:
            attn.download(ctx)
            cache_err = _std_units(attn, ref_attn)
        var out_err = _std_units(out, ref_out)
        var out_bits = 0
        var attn_bits = 0
        if variant == 0:
            best_a = best
            for n in range(QN):
                a_out.append(out.data[n])
            for n in range(SC):
                a_attn.append(mod.attn.data[n])
        elif variant >= 4:
            for n in range(QN):
                if out.data[n] != a_out[n]:
                    out_bits += 1
            attn_bits = -1
        else:
            for n in range(QN):
                if out.data[n] != a_out[n]:
                    out_bits += 1
            for n in range(SC):
                if attn.data[n] != a_attn[n]:
                    attn_bits += 1
        var flag = String("")
        # Row F is whatever `forward` ships: on NVIDIA at head dim 64,
        # unmasked, that is MAX's FA2 kernel and its band is TF32's.
        var tf32 = variant >= 14 or (
            variant == 4 and xa_fused_routes_to_max[MASKED, HD]()
        )
        var band = GATE_TF32_STD_UNITS if tf32 else GATE_STD_UNITS
        if out_err > band or cache_err > band:
            flag = "   ⚠⚠ DISAGREES"
            ok = False
        elif tf32:
            flag = "   (TF32 band " + String(GATE_TF32_STD_UNITS) + ")"
        print(
            "   " + _pad(names[variant], 36) + _pad(_fmt(best, 3), 10)
            + _pad(_fmt(best_a / best, 2) + "x", 8)
            + _pad(String(out_err), 24)
            + _pad(String(cache_err) if variant < 4 else String("n/a (none written)"), 24)
            + _pad(String(out_bits) + "/" + String(QN), 16)
            + (String(attn_bits) + "/" + String(SC) if variant < 4 else String("n/a"))
            + flag
        )
    return ok


def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    var ctx = DeviceContext()
    print("=" * 100)
    print("CrossAttention forward — sixteen variants, " + String(ctx.name()))
    print("=" * 100)
    var ok = True
    # SigLIP-B/16 @ 512: the target. B=1 — one tower call per camera.
    if not run_shape[1, 768, 12, 1024, 1024, False](String("SigLIP self-attention"), ctx):
        ok = False
    # ACT, training batch and deploy batch, every way it uses the primitive.
    if not run_shape[16, 256, 8, 162, 162, False](String("ACT encoder self (train)"), ctx):
        ok = False
    if not run_shape[1, 256, 8, 60, 162, False](String("ACT decoder cross (deploy)"), ctx):
        ok = False
    if not run_shape[16, 256, 8, 62, 62, True](String("ACT CVAE encoder, masked (train)"), ctx):
        ok = False
    print("")
    if not ok:
        raise Error("a variant produced different numbers — its timing is not a result")
    print(
        "  every variant within 1e-3 std units of float64 (MAX's TF32 rows: "
        + String(GATE_TF32_STD_UNITS) + "), output AND cached weights"
    )
