"""Fused (flash) attention kernels for `ScaledDotProductAttention`, fp32.

FlashAttention-2 in float32 on CUDA cores: the forward pass never writes the
[B, H, S, S] score matrix. It walks the key / value tiles with an online
softmax and keeps, per query row, only the output and its log-sum-exp L (in
log2 units). The backward recomputes P = exp2(S·scale·log2e − L) tile by tile:

  - `_flash_d_kernel`       D_i = Σ_d dO·O, one warp per row;
  - `_flash_bwd_dq_kernel`  one block per QUERY tile, loops over key tiles: dQ;
  - `_flash_bwd_dkdv_kernel` one block per KEY tile, loops over query tiles:
                            dK, dV.

Each output element is owned by exactly one thread of one block, with no
atomics: the result is deterministic, so an eager run and a CUDA-graph replay
stay bit-identical. The price is computing S and dP in both backward kernels.

Layouts (per sample, heads interleaved in DIM):
  input  `IL=False`: the leaf's [Q tokens | K tokens | V tokens], element
         (i, h, d) of Q at i·DIM + h·HD + d, K at +S·DIM, V at +2·S·DIM;
         `IL=True`: the QKV projection's own token-major output, token i at
         i·3·DIM with q at +0, k at +DIM, v at +2·DIM (no `QKVToMajor`);
  output / grad_output [tokens], (i, h, d) at i·DIM + h·HD + d;
  grad_input = input's layout. Every element of the output and of the grad
  input is written (rows < S), so neither needs clearing.

Register tiling (the SGEMM scheme). A score tile is TQ x TK; thread t owns
the 4 rows ty·4 + i and the 4 columns tx + (TK/4)·j (ty = t / (TK/4),
tx = t % (TK/4)), so the TK/4 lanes of a row group are consecutive and reduce
the softmax statistics with shuffles. Every shared-memory operand is read as
a 128-bit vector: the products are 4 x 4 outer products of 4-wide slices, 16
FMAs per 8 loads along the contraction — the first version (one output per
thread, scalar loads) issued ~1.25 shared loads per FMA and ran 3.3x off the
shared-memory bound. The P·V / dS·K products give each thread the same 4 rows
and the head-dim columns (tx + (TK/4)·q)·4 + e. Tile rows are padded to an
odd multiple of 16 bytes (HD + 4, TK + 4 floats), so the 8 lanes of a
128-bit load phase hit 8 different bank groups and same-row lanes broadcast.
The next K/V (Q/dO) tile is fetched into registers while the current one is
computed. Tiles fit 48 KB of static shared memory on NVIDIA (fwd 64x32,
backward 32x32) and Apple's 32 KB at half the size.
"""

from std.math import exp2, log2, sqrt
from std.sys import size_of
from std.sys import has_nvidia_gpu_accelerator
from max.gpu import thread_idx, block_idx, global_idx, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from std.sys.defines import get_defined_string
from .flash_attention_mma import (
    mma_eligible, MMA_TQ, MMA_TK, MMA_WARPS,
    _mma_fwd_kernel, _mma_dq_kernel, _mma_dkdv_kernel,
)


comptime _NV = has_nvidia_gpu_accelerator()
comptime FWD_TQ = 64 if _NV else 32
comptime FWD_TK = 32 if _NV else 16
comptime DQ_TQ = 32 if _NV else 16
comptime DQ_TK = 32 if _NV else 16
comptime KV_TK = 32 if _NV else 16
comptime KV_TQ = 32 if _NV else 16

comptime ATTN_PATH = get_defined_string["NN_ATTN_PATH", "auto"]()
"""`auto`: bf16 activations on NVIDIA run the tensor-core kernels
(`flash_attention_mma.mojo`); `simt`: the fp32 CUDA-core kernels below for
every dtype (the path before, for A/B runs)."""


def _use_mma[ADT: DType, HD: Int, H: Int]() -> Bool:
    return _NV and ATTN_PATH != "simt" and mma_eligible[ADT, HD, H]()


comptime _NEG = Scalar[DT](-1e30)
comptime _LOG2E = Scalar[DT](1.4426950408889634)

comptime _Shared[R: Int, C: Int] = LayoutTensor[
    DT, Layout.row_major(R, C), MutAnyOrigin,
    address_space=AddressSpace.SHARED, alignment=16,
]


def flash_eligible[HD: Int, P_DROP: Float64]() -> Bool:
    """The fused path's domain: no attention dropout, HD ≤ 64 and a multiple
    of every tile's key width (the head-dim columns split evenly over the
    lanes of a row group)."""
    return (
        P_DROP == 0.0
        and HD <= 64
        and HD % FWD_TK == 0
        and HD % DQ_TK == 0
        and HD % KV_TQ == 0
    )


@always_inline
def _put4[N: Int, OFF: Int](mut dst: SIMD[DT, N], v: SIMD[DT, 4]):
    """dst[OFF + e] = v[e], element by element (Metal cannot lower the
    `llvm.vector.insert` that `SIMD.insert` emits at these widths)."""
    comptime for e in range(4):
        dst[OFF + e] = v[e]


@always_inline
def _get4[N: Int, OFF: Int](src: SIMD[DT, N]) -> SIMD[DT, 4]:
    """src[OFF : OFF + 4], element by element (see `_put4`)."""
    return SIMD[DT, 4](src[OFF], src[OFF + 1], src[OFF + 2], src[OFF + 3])


@always_inline
def _grp_max[CG: Int](v: Scalar[DT]) -> Scalar[DT]:
    """Max over the CG consecutive lanes of a row group (CG = 4 or 8)."""
    var x = v
    comptime if CG >= 8:
        x = max(x, warp.shuffle_xor(x, UInt32(4)))
    x = max(x, warp.shuffle_xor(x, UInt32(2)))
    return max(x, warp.shuffle_xor(x, UInt32(1)))


@always_inline
def _grp_sum[CG: Int](v: Scalar[DT]) -> Scalar[DT]:
    """Sum over the CG consecutive lanes of a row group (CG = 4 or 8)."""
    var x = v
    comptime if CG >= 8:
        x = x + warp.shuffle_xor(x, UInt32(4))
    x = x + warp.shuffle_xor(x, UInt32(2))
    return x + warp.shuffle_xor(x, UInt32(1))


@always_inline
def _fetch[
    ADT: DType, R: Int, HD: Int, DIM: Int, S: Int, NT: Int
](
    src: Pointer[Scalar[ADT], MutAnyOrigin], base: Int, row0: Int, t: Int
) -> SIMD[DT, 4 * (R * HD // 4 // NT)]:
    """Thread t's share of the R x HD tile at rows row0.. of `src` (row stride
    DIM), as fp32 4-vectors in registers; rows ≥ S are zero."""
    comptime C4 = HD // 4
    comptime CH = R * C4 // NT
    comptime assert CH * NT == R * C4, "tile chunks must split over threads"
    comptime AL = 4 * size_of[Scalar[ADT]]()
    var regs = SIMD[DT, 4 * (R * HD // 4 // NT)](0)
    comptime for ch in range(CH):
        var idx = t + ch * NT
        var rr = idx // C4
        var gi = row0 + rr
        if gi < S:
            var v = src.unsafe_load[width=4, alignment=AL](
                base + gi * DIM + (idx - rr * C4) * 4
            )
            _put4[4 * (R * HD // 4 // NT), 4 * ch](regs, v.cast[DT]())
    return regs


@always_inline
def _stash[
    R: Int, HD: Int, NT: Int
](tile: _Shared[R, HD + 4], regs: SIMD[DT, 4 * (R * HD // 4 // NT)], t: Int):
    """Store a `_fetch`ed share into the (padded) shared tile."""
    comptime C4 = HD // 4
    comptime CH = R * C4 // NT
    comptime for ch in range(CH):
        var idx = t + ch * NT
        var rr = idx // C4
        tile.aligned_store[4](
            rr, (idx - rr * C4) * 4, _get4[4 * (R * HD // 4 // NT), 4 * ch](regs)
        )


@always_inline
def _qk[
    TQ: Int, TK: Int, HD: Int
](
    a: _Shared[TQ, HD + 4], b: _Shared[TK, HD + 4], r0: Int, tx: Int
) -> SIMD[DT, 16]:
    """The thread's 4 x 4 block of a·bᵀ: rows r0 + i, columns tx + (TK/4)·j,
    element [i·4 + j]."""
    comptime CG = TK // 4
    var acc = SIMD[DT, 16](0)
    for d4 in range(HD // 4):
        var av = SIMD[DT, 16]()
        var bv = SIMD[DT, 16]()
        comptime for i in range(4):
            _put4[16, 4 * i](av, a.aligned_load[4](r0 + i, d4 * 4))
            _put4[16, 4 * i](bv, b.aligned_load[4](tx + CG * i, d4 * 4))
        comptime for i in range(4):
            comptime for j in range(4):
                comptime for e in range(4):
                    acc[i * 4 + j] += av[i * 4 + e] * bv[j * 4 + e]
    return acc


@always_inline
def _pv[
    TQ: Int, TK: Int, HD: Int
](
    mut o: SIMD[DT, 4 * (HD // (TK // 4))],
    p: _Shared[TQ, TK + 4],
    v: _Shared[TK, HD + 4],
    r0: Int,
    tx: Int,
):
    """o[i·DPT + q·4 + e] += Σ_c p[r0 + i, c] · v[c, (tx + CG·q)·4 + e]."""
    comptime CG = TK // 4
    comptime DPT = HD // CG
    comptime NC = DPT // 4
    for c4 in range(TK // 4):
        var pv = SIMD[DT, 16]()
        comptime for i in range(4):
            _put4[16, 4 * i](pv, p.aligned_load[4](r0 + i, c4 * 4))
        comptime for cc in range(4):
            comptime for q in range(NC):
                var vv = v.aligned_load[4](c4 * 4 + cc, (tx + CG * q) * 4)
                comptime for i in range(4):
                    comptime for e in range(4):
                        o[i * DPT + q * 4 + e] += pv[i * 4 + cc] * vv[e]


def _flash_fwd_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, TQ: Int, TK: Int,
    IL: Bool = False,
](
    inp: Pointer[Scalar[ADT], MutAnyOrigin],
    outp: Pointer[Scalar[ADT], MutAnyOrigin],
    o_cache: Pointer[Scalar[DT], MutAnyOrigin],
    lse: Pointer[Scalar[DT], MutAnyOrigin],
):
    """O = softmax(Q·Kᵀ·scale)·V for one (query tile, b·h); O (fp32) also into
    `o_cache`, L = m + log2 l (log2 units) into `lse` [B·H·S]."""
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM  # input token stride
    comptime KO = DIM if IL else S * DIM  # K, then V, offset in the input
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime NT = TQ * TK // 16
    comptime CG = TK // 4
    comptime DPT = HD // CG
    comptime NC = DPT // 4
    comptime AL = 4 * size_of[Scalar[ADT]]()
    var qt = Int(block_idx.x)
    var bh = Int(block_idx.y)
    var b = bh // H
    var h = bh - b * H
    var t = Int(thread_idx.x)
    var ty = t // CG
    var tx = t - ty * CG
    var r0 = ty * 4
    var scale2 = Scalar[DT](Float32(1.0) / sqrt(Float32(HD))) * _LOG2E

    var Qs = _Shared[TQ, HD + 4].stack_allocation()
    var Ks = _Shared[TK, HD + 4].stack_allocation()
    var Vs = _Shared[TK, HD + 4].stack_allocation()
    var Ps = _Shared[TQ, TK + 4].stack_allocation()

    var base = b * IN_DIM + h * HD
    _stash[TQ, HD, NT](
        Qs, _fetch[ADT, TQ, HD, IS, S, NT](inp, base, qt * TQ, t), t
    )
    comptime N_KT = (S + TK - 1) // TK
    var kt_end = N_KT
    comptime if CAUSAL:
        kt_end = min(N_KT, (qt * TQ + TQ - 1) // TK + 1)
    var kreg = _fetch[ADT, TK, HD, IS, S, NT](inp, base + KO, 0, t)
    var vreg = _fetch[ADT, TK, HD, IS, S, NT](inp, base + VO, 0, t)

    var m = SIMD[DT, 4](_NEG)
    var l = SIMD[DT, 4](0)
    var o = SIMD[DT, 4 * DPT](0)
    for kt in range(kt_end):
        barrier()
        _stash[TK, HD, NT](Ks, kreg, t)
        _stash[TK, HD, NT](Vs, vreg, t)
        barrier()
        if kt + 1 < kt_end:
            kreg = _fetch[ADT, TK, HD, IS, S, NT](
                inp, base + KO, (kt + 1) * TK, t
            )
            vreg = _fetch[ADT, TK, HD, IS, S, NT](
                inp, base + VO, (kt + 1) * TK, t
            )
        var s = _qk[TQ, TK, HD](Qs, Ks, r0, tx)
        comptime for i in range(4):
            var row = qt * TQ + r0 + i
            var mx = _NEG
            comptime for j in range(4):
                var col = kt * TK + tx + CG * j
                var ok = col < S
                comptime if CAUSAL:
                    ok = ok and col <= row
                var sv = s[i * 4 + j] * scale2 if ok else _NEG
                s[i * 4 + j] = sv
                mx = max(mx, sv)
            var m_new = max(m[i], _grp_max[CG](mx))
            var alpha = exp2(m[i] - m_new)
            var psum = Scalar[DT](0)
            comptime for j in range(4):
                var p = Scalar[DT](0)
                if s[i * 4 + j] > _NEG:
                    p = exp2(s[i * 4 + j] - m_new)
                psum += p
                Ps[r0 + i, tx + CG * j] = p
            l[i] = l[i] * alpha + _grp_sum[CG](psum)
            m[i] = m_new
            comptime for k in range(DPT):
                o[i * DPT + k] *= alpha
        barrier()
        _pv[TQ, TK, HD](o, Ps, Vs, r0, tx)
    comptime for i in range(4):
        var row = qt * TQ + r0 + i
        if row < S:
            var inv = Scalar[DT](1) / l[i]
            var ob = b * OUT_DIM + row * DIM + h * HD
            comptime for q in range(NC):
                var v = _get4[4 * DPT, i * DPT + q * 4](o) * inv
                var off = ob + (tx + CG * q) * 4
                outp.unsafe_store[alignment=AL](off, v.cast[ADT]())
                o_cache.unsafe_store[alignment=16](off, v)
            if tx == 0:
                lse[unsafe_offset=bh * S + row] = m[i] + log2(l[i])


def _flash_d_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, ROWS: Int
](
    dout: Pointer[Scalar[ADT], MutAnyOrigin],
    o_cache: Pointer[Scalar[DT], MutAnyOrigin],
    dvec: Pointer[Scalar[DT], MutAnyOrigin],
):
    """D[b·h·S + i] = Σ_d dO·O, one warp per row (ROWS = B·H·S)."""
    comptime DIM = H * HD
    var idx = Int(global_idx.x)
    var row = idx // WARP_SIZE
    var lane = idx - row * WARP_SIZE
    if row >= ROWS:
        return
    var bh = row // S
    var i = row - bh * S
    var b = bh // H
    var h = bh - b * H
    var base = b * S * DIM + i * DIM + h * HD
    var part = Scalar[DT](0)
    var d = lane
    while d < HD:
        part += dout[unsafe_offset=base + d].cast[DT]() * o_cache[unsafe_offset=base + d]
        d += WARP_SIZE
    var tot = warp.sum(part)
    if lane == 0:
        dvec[unsafe_offset=row] = tot


def _flash_bwd_dq_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, TQ: Int, TK: Int,
    IL: Bool = False,
](
    inp: Pointer[Scalar[ADT], MutAnyOrigin],
    dout: Pointer[Scalar[ADT], MutAnyOrigin],
    lse: Pointer[Scalar[DT], MutAnyOrigin],
    dvec: Pointer[Scalar[DT], MutAnyOrigin],
    gin: Pointer[Scalar[ADT], MutAnyOrigin],
):
    """dQ = Σ over key tiles of dS·K·scale, dS = P ∘ (dO·Vᵀ − D), for one
    (query tile, b·h)."""
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM  # input token stride
    comptime KO = DIM if IL else S * DIM  # K, then V, offset in the input
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime NT = TQ * TK // 16
    comptime CG = TK // 4
    comptime DPT = HD // CG
    comptime NC = DPT // 4
    comptime AL = 4 * size_of[Scalar[ADT]]()
    var qt = Int(block_idx.x)
    var bh = Int(block_idx.y)
    var b = bh // H
    var h = bh - b * H
    var t = Int(thread_idx.x)
    var ty = t // CG
    var tx = t - ty * CG
    var r0 = ty * 4
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HD)))
    var scale2 = scale * _LOG2E

    var Qs = _Shared[TQ, HD + 4].stack_allocation()
    var dOs = _Shared[TQ, HD + 4].stack_allocation()
    var Ks = _Shared[TK, HD + 4].stack_allocation()
    var Vs = _Shared[TK, HD + 4].stack_allocation()
    var dSs = _Shared[TQ, TK + 4].stack_allocation()

    var base = b * IN_DIM + h * HD
    var obase = b * OUT_DIM + h * HD
    _stash[TQ, HD, NT](
        Qs, _fetch[ADT, TQ, HD, IS, S, NT](inp, base, qt * TQ, t), t
    )
    _stash[TQ, HD, NT](
        dOs, _fetch[ADT, TQ, HD, DIM, S, NT](dout, obase, qt * TQ, t), t
    )
    var L = SIMD[DT, 4](0)
    var D = SIMD[DT, 4](0)
    comptime for i in range(4):
        var row = qt * TQ + r0 + i
        if row < S:
            L[i] = lse[unsafe_offset=bh * S + row]
            D[i] = dvec[unsafe_offset=bh * S + row]
    comptime N_KT = (S + TK - 1) // TK
    var kt_end = N_KT
    comptime if CAUSAL:
        kt_end = min(N_KT, (qt * TQ + TQ - 1) // TK + 1)
    var kreg = _fetch[ADT, TK, HD, IS, S, NT](inp, base + KO, 0, t)
    var vreg = _fetch[ADT, TK, HD, IS, S, NT](inp, base + VO, 0, t)

    var dq = SIMD[DT, 4 * DPT](0)
    for kt in range(kt_end):
        barrier()
        _stash[TK, HD, NT](Ks, kreg, t)
        _stash[TK, HD, NT](Vs, vreg, t)
        barrier()
        if kt + 1 < kt_end:
            kreg = _fetch[ADT, TK, HD, IS, S, NT](
                inp, base + KO, (kt + 1) * TK, t
            )
            vreg = _fetch[ADT, TK, HD, IS, S, NT](
                inp, base + VO, (kt + 1) * TK, t
            )
        var s = _qk[TQ, TK, HD](Qs, Ks, r0, tx)
        var dp = _qk[TQ, TK, HD](dOs, Vs, r0, tx)
        comptime for i in range(4):
            var row = qt * TQ + r0 + i
            comptime for j in range(4):
                var col = kt * TK + tx + CG * j
                var ok = col < S and row < S
                comptime if CAUSAL:
                    ok = ok and col <= row
                var ds = Scalar[DT](0)
                if ok:
                    ds = exp2(s[i * 4 + j] * scale2 - L[i]) * (dp[i * 4 + j] - D[i])
                dSs[r0 + i, tx + CG * j] = ds
        barrier()
        _pv[TQ, TK, HD](dq, dSs, Ks, r0, tx)
    comptime for i in range(4):
        var row = qt * TQ + r0 + i
        if row < S:
            var gb = b * IN_DIM + row * IS + h * HD
            comptime for q in range(NC):
                var v = _get4[4 * DPT, i * DPT + q * 4](dq) * scale
                gin.unsafe_store[alignment=AL](gb + (tx + CG * q) * 4, v.cast[ADT]())


def _flash_bwd_dkdv_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, TQ: Int, TK: Int,
    IL: Bool = False,
](
    inp: Pointer[Scalar[ADT], MutAnyOrigin],
    dout: Pointer[Scalar[ADT], MutAnyOrigin],
    lse: Pointer[Scalar[DT], MutAnyOrigin],
    dvec: Pointer[Scalar[DT], MutAnyOrigin],
    gin: Pointer[Scalar[ADT], MutAnyOrigin],
):
    """dK = Σ dSᵀ·Q·scale and dV = Σ Pᵀ·dO for one (key tile of TK rows,
    b·h), over the query tiles of TQ rows (from the key tile on, when causal).

    Score phase: thread rows = queries, columns = keys (the forward mapping);
    P and dS are stored TRANSPOSED (`Pt[key, query]`). Accumulation phase:
    thread kg = t / DG owns the 4 keys kg·4 + i and the head-dim columns
    (t % DG + DG·q)·4 + e."""
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM  # input token stride
    comptime KO = DIM if IL else S * DIM  # K, then V, offset in the input
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime NT = TQ * TK // 16
    comptime CG = TK // 4
    comptime DG = NT // (TK // 4)
    comptime DPK = HD // DG
    comptime NCK = DPK // 4
    comptime AL = 4 * size_of[Scalar[ADT]]()
    comptime assert DPK * DG == HD and NCK * 4 == DPK, "dK/dV columns split"
    var kt = Int(block_idx.x)
    var bh = Int(block_idx.y)
    var b = bh // H
    var h = bh - b * H
    var t = Int(thread_idx.x)
    var ty = t // CG
    var tx = t - ty * CG
    var r0 = ty * 4
    var kg = t // DG
    var dg = t - kg * DG
    var k0 = kg * 4
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HD)))
    var scale2 = scale * _LOG2E

    var Ks = _Shared[TK, HD + 4].stack_allocation()
    var Vs = _Shared[TK, HD + 4].stack_allocation()
    var Qs = _Shared[TQ, HD + 4].stack_allocation()
    var dOs = _Shared[TQ, HD + 4].stack_allocation()
    var Pt = _Shared[TK, TQ + 4].stack_allocation()
    var dSt = _Shared[TK, TQ + 4].stack_allocation()

    var base = b * IN_DIM + h * HD
    var obase = b * OUT_DIM + h * HD
    _stash[TK, HD, NT](
        Ks, _fetch[ADT, TK, HD, IS, S, NT](inp, base + KO, kt * TK, t), t
    )
    _stash[TK, HD, NT](
        Vs,
        _fetch[ADT, TK, HD, IS, S, NT](inp, base + VO, kt * TK, t),
        t,
    )
    comptime N_QT = (S + TQ - 1) // TQ
    var qt0 = 0
    comptime if CAUSAL:
        qt0 = (kt * TK) // TQ
    var qreg = _fetch[ADT, TQ, HD, IS, S, NT](inp, base, qt0 * TQ, t)
    var oreg = _fetch[ADT, TQ, HD, DIM, S, NT](dout, obase, qt0 * TQ, t)

    var dk = SIMD[DT, 4 * DPK](0)
    var dv = SIMD[DT, 4 * DPK](0)
    for qt in range(qt0, N_QT):
        barrier()
        _stash[TQ, HD, NT](Qs, qreg, t)
        _stash[TQ, HD, NT](dOs, oreg, t)
        barrier()
        if qt + 1 < N_QT:
            qreg = _fetch[ADT, TQ, HD, IS, S, NT](inp, base, (qt + 1) * TQ, t)
            oreg = _fetch[ADT, TQ, HD, DIM, S, NT](
                dout, obase, (qt + 1) * TQ, t
            )
        var s = _qk[TQ, TK, HD](Qs, Ks, r0, tx)
        var dp = _qk[TQ, TK, HD](dOs, Vs, r0, tx)
        comptime for i in range(4):
            var row = qt * TQ + r0 + i
            var Li = Scalar[DT](0)
            var Di = Scalar[DT](0)
            if row < S:
                Li = lse[unsafe_offset=bh * S + row]
                Di = dvec[unsafe_offset=bh * S + row]
            comptime for j in range(4):
                var col = kt * TK + tx + CG * j
                var ok = col < S and row < S
                comptime if CAUSAL:
                    ok = ok and col <= row
                var p = Scalar[DT](0)
                if ok:
                    p = exp2(s[i * 4 + j] * scale2 - Li)
                Pt[tx + CG * j, r0 + i] = p
                dSt[tx + CG * j, r0 + i] = p * (dp[i * 4 + j] - Di)
        barrier()
        for r4 in range(TQ // 4):
            var pv = SIMD[DT, 16]()
            var sv = SIMD[DT, 16]()
            comptime for i in range(4):
                _put4[16, 4 * i](pv, Pt.aligned_load[4](k0 + i, r4 * 4))
                _put4[16, 4 * i](sv, dSt.aligned_load[4](k0 + i, r4 * 4))
            comptime for rr in range(4):
                comptime for q in range(NCK):
                    var dov = dOs.aligned_load[4](r4 * 4 + rr, (dg + DG * q) * 4)
                    var qv = Qs.aligned_load[4](r4 * 4 + rr, (dg + DG * q) * 4)
                    comptime for i in range(4):
                        comptime for e in range(4):
                            dv[i * DPK + q * 4 + e] += pv[i * 4 + rr] * dov[e]
                            dk[i * DPK + q * 4 + e] += sv[i * 4 + rr] * qv[e]
    comptime for i in range(4):
        var key = kt * TK + k0 + i
        if key < S:
            var gb = b * IN_DIM + key * IS + h * HD
            comptime for q in range(NCK):
                var off = gb + (dg + DG * q) * 4
                var kv = _get4[4 * DPK, i * DPK + q * 4](dk) * scale
                gin.unsafe_store[alignment=AL](off + KO, kv.cast[ADT]())
                gin.unsafe_store[alignment=AL](
                    off + VO,
                    _get4[4 * DPK, i * DPK + q * 4](dv).cast[ADT](),
                )


def flash_forward[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, BH: Int,
    IL: Bool = False,
](
    c: DeviceContext,
    inp: DeviceBuffer[ADT],
    outp: DeviceBuffer[ADT],
    o_cache: DeviceBuffer[DT],
    lse: DeviceBuffer[DT],
) raises:
    """Enqueue the forward: `outp` [B, S·DIM], `o_cache` (fp32, same) and
    `lse` [BH·S] (log2 units)."""
    comptime if _use_mma[ADT, HD, H]():
        c.enqueue_function[_mma_fwd_kernel[ADT, H, S, HD, CAUSAL, IL]](
            inp, outp, o_cache, lse,
            grid_dim=((S + MMA_TQ - 1) // MMA_TQ, BH),
            block_dim=MMA_WARPS * WARP_SIZE,
        )
        return
    c.enqueue_function[
        _flash_fwd_kernel[ADT, H, S, HD, CAUSAL, FWD_TQ, FWD_TK, IL]
    ](
        inp, outp, o_cache, lse,
        grid_dim=((S + FWD_TQ - 1) // FWD_TQ, BH),
        block_dim=FWD_TQ * FWD_TK // 16,
    )


def flash_backward[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, BH: Int,
    IL: Bool = False,
](
    c: DeviceContext,
    inp: DeviceBuffer[ADT],
    dout: DeviceBuffer[ADT],
    o_cache: DeviceBuffer[DT],
    lse: DeviceBuffer[DT],
    dvec: DeviceBuffer[DT],
    gin: DeviceBuffer[ADT],
) raises:
    """Enqueue D = Σ dO·O, then dQ, then dK / dV into `gin` (the input's
    layout, every element written)."""
    comptime ROWS = BH * S
    c.enqueue_function[_flash_d_kernel[ADT, H, S, HD, ROWS]](
        dout, o_cache, dvec,
        grid_dim=(ROWS * WARP_SIZE + TPB - 1) // TPB, block_dim=TPB,
    )
    comptime if _use_mma[ADT, HD, H]():
        c.enqueue_function[_mma_dq_kernel[ADT, H, S, HD, CAUSAL, IL]](
            inp, dout, lse, dvec, gin,
            grid_dim=((S + MMA_TQ - 1) // MMA_TQ, BH),
            block_dim=MMA_WARPS * WARP_SIZE,
        )
        c.enqueue_function[_mma_dkdv_kernel[ADT, H, S, HD, CAUSAL, IL]](
            inp, dout, lse, dvec, gin,
            grid_dim=((S + MMA_TK - 1) // MMA_TK, BH),
            block_dim=MMA_WARPS * WARP_SIZE,
        )
        return
    c.enqueue_function[
        _flash_bwd_dq_kernel[ADT, H, S, HD, CAUSAL, DQ_TQ, DQ_TK, IL]
    ](
        inp, dout, lse, dvec, gin,
        grid_dim=((S + DQ_TQ - 1) // DQ_TQ, BH),
        block_dim=DQ_TQ * DQ_TK // 16,
    )
    c.enqueue_function[
        _flash_bwd_dkdv_kernel[ADT, H, S, HD, CAUSAL, KV_TQ, KV_TK, IL]
    ](
        inp, dout, lse, dvec, gin,
        grid_dim=((S + KV_TK - 1) // KV_TK, BH),
        block_dim=KV_TQ * KV_TK // 16,
    )
