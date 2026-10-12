"""Fused attention on the tensor cores, for bf16 activations on NVIDIA.

The same FlashAttention-2 as `flash_attention.mojo` — same entry points,
buffers, layouts and outputs (O at the activation dtype, its copy `o_cache`
— here bf16, as FA2, where the fp32 kernels keep fp32 — and the log-sum-exp
L in log2 units) — with every product on
`mma.sync.m16n8k16` (bf16 operands, fp32 accumulators) instead of fp32 FMAs.
The fp32 kernels ran the bf16 GPT's attention at 3.6 ms a step against
PyTorch's FA2 0.8 ms.

  - `_mma_fwd_kernel`   one block per 64-query tile, 4 warps x 16 rows. Q
                        lives in registers as A fragments; per 64-key tile,
                        S = Q·Kᵀ, the online softmax in fp32, O += P·V.
  - `_mma_dq_kernel`    one block per query tile, loops over key tiles:
                        dQ = scale · Σ dS·K, dS = P ⊙ (dP − D), dP = dO·Vᵀ.
  - `_mma_dkdv_kernel`  one block per 64-key tile, loops over query tiles:
                        dV = Σ Pᵀ·dO, dK = scale · Σ dSᵀ·Q.

D = Σ dO·O is computed by the dQ kernel (which holds dO's fragments) and
handed to the dK/dV kernel through `dvec`. As the fp32 kernels, no
atomics: each output element is written by one thread of one block, so an
eager step and its CUDA-graph replay stay bit-identical.

Fragments (PTX m16n8k16, lane l, g = l / 4, q = l % 4; checked on an RTX
5090): A [16 x 16] row-major holds (g, 2q..2q+1), (g+8, 2q..), (g, 2q+8..),
(g+8, 2q+8..); B [16(k) x 8(n)] holds (k = 2q..2q+1, n = g) then k + 8;
C [16 x 8] holds (g, 2q..2q+1), (g+8, 2q..2q+1). So an accumulator tile's
two 8-column halves ARE the A fragment of the next product (P·V, dS·K):
no shared-memory round trip. B operands come from row-major K, V, Q, dO
tiles in shared memory through `ldmatrix.x4` (two n tiles per load): plain
for the products that contract over the head dim, `.trans` for those that
contract over tokens — no transposed copies (whose element-wise stores were
~16-way bank conflicts: 11.21 -> 11.00 ms a GPT step). Shared rows are
padded by 8 elements (16 bytes): an `ldmatrix`'s 8 row addresses hit 32
distinct banks.

Numerics: products in bf16 x bf16 -> fp32 (as PyTorch's FA2); P and dS are
rounded to bf16 as the A operands of the second products, everything else
(scores, softmax statistics, accumulators) is fp32.
"""

from std.math import exp2, log2, sqrt
from std.memory import stack_allocation
from max.gpu import thread_idx, block_idx, lane_id, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.memory import (
    AddressSpace, async_copy, async_copy_commit_group, async_copy_wait_group,
)
from max.gpu.primitives import warp
from max.gpu.compute.mma import mma, ld_matrix

from noeira.nn.constants import DT


comptime BF = DType.bfloat16
comptime F32 = DType.float32
comptime MMA_TQ = 64
comptime MMA_TK = 64
comptime MMA_WARPS = 4
comptime _PAD = 8
comptime _NEG = Scalar[F32](-1e30)
comptime _LOG2E = Scalar[F32](1.4426950408889634)


def mma_eligible[ADT: DType, HD: Int, H: Int]() -> Bool:
    """bf16 activations, a head dim the 16-wide k steps and 8-wide n tiles
    cover, and 16-byte aligned rows (8-element vector loads)."""
    return ADT == BF and HD % 16 == 0 and HD <= 128 and (H * HD) % 8 == 0


comptime _SPtr = Pointer[Scalar[BF], MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def _smem[N: Int]() -> _SPtr:
    return stack_allocation[
        N, Scalar[BF], address_space=AddressSpace.SHARED, alignment=16
    ]()


@always_inline
def _ldm_b(p: _SPtr, ld: Int, n0: Int, k0: Int, l: Int) -> SIMD[BF, 8]:
    """`ldmatrix.x4`: the B fragments of n tiles n0..n0+7 and n0+8..n0+15 at
    k0..k0+15, from a row-major [n][k] tile (row stride `ld`). Lane l gives
    the row address of matrix l / 8: (n0 + (m / 2)·8 + l % 8, k0 + (m % 2)·8).
    Returns frag(n0) in [0:4], frag(n0 + 8) in [4:8]."""
    var m = l // 8
    var row = n0 + (m // 2) * 8 + (l - m * 8)
    return ld_matrix[8](p.unsafe_offset(row * ld + k0 + (m - (m // 2) * 2) * 8))


@always_inline
def _ldm_bt(p: _SPtr, ld: Int, k0: Int, n0: Int, l: Int) -> SIMD[BF, 8]:
    """`ldmatrix.x4.trans`: the B fragments of n tiles n0.. and n0+8.. at
    k0..k0+15, from a row-major [k][n] tile — the transposed read, no
    transposed copy. Matrix m = l / 8 sits at (k0 + (m % 2)·8 + l % 8,
    n0 + (m / 2)·8). Returns frag(n0) in [0:4], frag(n0 + 8) in [4:8]."""
    var m = l // 8
    var row = k0 + (m - (m // 2) * 2) * 8 + (l - m * 8)
    return ld_matrix[8, transpose=True](p.unsafe_offset(row * ld + n0 + (m // 2) * 8))


@always_inline
def _lo(v: SIMD[BF, 8]) -> SIMD[BF, 4]:
    return SIMD[BF, 4](v[0], v[1], v[2], v[3])


@always_inline
def _hi(v: SIMD[BF, 8]) -> SIMD[BF, 4]:
    return SIMD[BF, 4](v[4], v[5], v[6], v[7])


@always_inline
def _afrag_from_c(c0: SIMD[F32, 4], c1: SIMD[F32, 4]) -> SIMD[BF, 8]:
    """The A fragment of a 16 x 16 tile whose two 8-column halves are the
    accumulator fragments c0 (columns 0..7) and c1 (8..15)."""
    return SIMD[BF, 8](
        c0[0].cast[BF](), c0[1].cast[BF](), c0[2].cast[BF](), c0[3].cast[BF](),
        c1[0].cast[BF](), c1[1].cast[BF](), c1[2].cast[BF](), c1[3].cast[BF](),
    )


@always_inline
def _afrag_global[
    ADT: DType
](src: Pointer[Scalar[ADT], MutAnyOrigin], base: Int, stride: Int, r: Int, r8: Int, valid_r: Bool, valid_r8: Bool, k0: Int) -> SIMD[BF, 8]:
    """A fragment of rows r (g) and r8 (g + 8), columns k0 + 2q.. / +8, read
    from global memory (row stride `stride`); invalid rows read as zero."""
    var o = SIMD[BF, 8](0)
    if valid_r:
        var a = src.unsafe_load[width=2, alignment=4](base + r * stride + k0)
        var b = src.unsafe_load[width=2, alignment=4](base + r * stride + k0 + 8)
        o[0] = rebind[Scalar[BF]](a[0])
        o[1] = rebind[Scalar[BF]](a[1])
        o[4] = rebind[Scalar[BF]](b[0])
        o[5] = rebind[Scalar[BF]](b[1])
    if valid_r8:
        var a = src.unsafe_load[width=2, alignment=4](base + r8 * stride + k0)
        var b = src.unsafe_load[width=2, alignment=4](base + r8 * stride + k0 + 8)
        o[2] = rebind[Scalar[BF]](a[0])
        o[3] = rebind[Scalar[BF]](a[1])
        o[6] = rebind[Scalar[BF]](b[0])
        o[7] = rebind[Scalar[BF]](b[1])
    return o


@always_inline
def _stage[
    ADT: DType, R: Int, HD: Int, NT: Int
](
    src: Pointer[Scalar[ADT], MutAnyOrigin],
    base: Int,
    stride: Int,
    row0: Int,
    S: Int,
    dst: _SPtr,
    t: Int,
):
    """Copy the R x HD tile at rows row0.. (row stride `stride`) into shared
    memory, row-major [R][HD + PAD]. Rows >= S are zero. 16-byte loads and
    stores."""
    comptime C8 = HD // 8
    comptime CH = (R * C8 + NT - 1) // NT
    comptime for ch in range(CH):
        var idx = t + ch * NT
        if idx < R * C8:
            var rr = idx // C8
            var c0 = (idx - rr * C8) * 8
            var v = SIMD[BF, 8](0)
            if row0 + rr < S:
                v = rebind[SIMD[BF, 8]](
                    src.unsafe_load[width=8, alignment=16](
                        base + (row0 + rr) * stride + c0
                    )
                )
            dst.unsafe_store[alignment=16](rr * (HD + _PAD) + c0, v)


@always_inline
def _stage_async[
    ADT: DType, R: Int, HD: Int, NT: Int
](
    src: Pointer[Scalar[ADT], MutAnyOrigin],
    base: Int,
    stride: Int,
    row0: Int,
    S: Int,
    dst: _SPtr,
    t: Int,
):
    """`_stage` with `cp.async` (16 bytes per copy, L1 bypassed): the copies
    complete in the background — `async_copy_commit_group` /
    `async_copy_wait_group` fence them. Rows >= S are zero-filled by the copy
    itself (src_size 0; the address is clamped to row 0, never read)."""
    comptime C8 = HD // 8
    comptime CH = (R * C8 + NT - 1) // NT
    var g = rebind[Pointer[Scalar[BF], MutAnyOrigin]](src).unsafe_address_space_cast[
        AddressSpace.GLOBAL
    ]()
    comptime for ch in range(CH):
        var idx = t + ch * NT
        if idx < R * C8:
            var rr = idx // C8
            var c0 = (idx - rr * C8) * 8
            var ok = row0 + rr < S
            var row = row0 + rr if ok else 0
            async_copy[16](
                g.unsafe_offset(base + row * stride + c0),
                dst.unsafe_offset(rr * (HD + _PAD) + c0),
                src_size=Int32(16 if ok else 0),
            )


@always_inline
def _row_max4(x: Scalar[F32]) -> Scalar[F32]:
    """Max over the 4 lanes (q = 0..3) that share a fragment row."""
    var y = max(x, warp.shuffle_xor(x, UInt32(1)))
    return max(y, warp.shuffle_xor(y, UInt32(2)))


@always_inline
def _row_sum4(x: Scalar[F32]) -> Scalar[F32]:
    var y = x + warp.shuffle_xor(x, UInt32(1))
    return y + warp.shuffle_xor(y, UInt32(2))


def _mma_fwd_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, IL: Bool
](
    inp: Pointer[Scalar[ADT], MutAnyOrigin],
    outp: Pointer[Scalar[ADT], MutAnyOrigin],
    o_cache: Pointer[Scalar[ADT], MutAnyOrigin],
    lse: Pointer[Scalar[DT], MutAnyOrigin],
):
    comptime TQ = MMA_TQ
    comptime TK = MMA_TK
    comptime NT = MMA_WARPS * WARP_SIZE
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM
    comptime KO = DIM if IL else S * DIM
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime KS = HD // 16  # k steps over the head dim
    comptime NK = TK // 8  # n tiles over the keys
    comptime ND = HD // 8  # n tiles over the head dim
    var qt = Int(block_idx.x)
    var bh = Int(block_idx.y)
    var b = bh // H
    var h = bh - b * H
    var t = Int(thread_idx.x)
    var w = t // WARP_SIZE
    var l = t - w * WARP_SIZE
    var g = l // 4
    var q = l - g * 4
    var scale2 = Scalar[F32](1.0) / sqrt(Scalar[F32](HD)) * _LOG2E
    var base = b * IN_DIM + h * HD
    var r_a = qt * TQ + w * 16 + g  # this lane's two rows
    var r_b = r_a + 8

    # Double-buffered K / V: tile kt + 1 is copied (cp.async) while kt computes.
    var Ks0 = _smem[TK * (HD + _PAD)]()
    var Ks1 = _smem[TK * (HD + _PAD)]()
    var Vs0 = _smem[TK * (HD + _PAD)]()
    var Vs1 = _smem[TK * (HD + _PAD)]()
    _stage_async[ADT, TK, HD, NT](inp, base + KO, IS, 0, S, Ks0, t)
    _stage_async[ADT, TK, HD, NT](inp, base + VO, IS, 0, S, Vs0, t)
    async_copy_commit_group()

    var qf = Array[SIMD[BF, 8], KS](fill=SIMD[BF, 8](0))
    comptime for ks in range(KS):
        qf[ks] = _afrag_global[ADT](
            inp, base, IS, r_a, r_b, r_a < S, r_b < S, ks * 16 + q * 2
        )

    comptime N_KT = (S + TK - 1) // TK
    var kt_end = N_KT
    comptime if CAUSAL:
        kt_end = min(N_KT, (qt * TQ + TQ - 1) // TK + 1)

    var m_a = _NEG
    var m_b = _NEG
    var l_a = Scalar[F32](0)
    var l_b = Scalar[F32](0)
    var o = Array[SIMD[F32, 4], ND](fill=SIMD[F32, 4](0))
    for kt in range(kt_end):
        var even = (kt & 1) == 0
        var Ks = Ks0 if even else Ks1
        var Vs = Vs0 if even else Vs1
        if kt + 1 < kt_end:
            _stage_async[ADT, TK, HD, NT](inp, base + KO, IS, (kt + 1) * TK, S, Ks1 if even else Ks0, t)
            _stage_async[ADT, TK, HD, NT](inp, base + VO, IS, (kt + 1) * TK, S, Vs1 if even else Vs0, t)
            async_copy_commit_group()
            async_copy_wait_group(1)
        else:
            async_copy_wait_group(0)
        barrier()
        var s = Array[SIMD[F32, 4], NK](fill=SIMD[F32, 4](0))
        comptime for np in range(NK // 2):
            comptime for ks in range(KS):
                var bb = _ldm_b(Ks, HD + _PAD, np * 16, ks * 16, l)
                mma(s[2 * np], qf[ks], _lo(bb), s[2 * np])
                mma(s[2 * np + 1], qf[ks], _hi(bb), s[2 * np + 1])
        # Online softmax (log2 units), rows r_a (elements 0, 1) / r_b (2, 3).
        var mx_a = _NEG
        var mx_b = _NEG
        comptime for nt in range(NK):
            comptime for e in range(4):
                var col = kt * TK + nt * 8 + q * 2 + (e & 1)
                var row = r_a if e < 2 else r_b
                var ok = col < S
                comptime if CAUSAL:
                    ok = ok and col <= row
                var sv = s[nt][e] * scale2 if ok else _NEG
                s[nt][e] = sv
                if e < 2:
                    mx_a = max(mx_a, sv)
                else:
                    mx_b = max(mx_b, sv)
        var mn_a = max(m_a, _row_max4(mx_a))
        var mn_b = max(m_b, _row_max4(mx_b))
        var al_a = exp2(m_a - mn_a)
        var al_b = exp2(m_b - mn_b)
        var ps_a = Scalar[F32](0)
        var ps_b = Scalar[F32](0)
        comptime for nt in range(NK):
            comptime for e in range(4):
                var p = Scalar[F32](0)
                if s[nt][e] > _NEG:
                    p = exp2(s[nt][e] - (mn_a if e < 2 else mn_b))
                s[nt][e] = p
                if e < 2:
                    ps_a += p
                else:
                    ps_b += p
        l_a = l_a * al_a + _row_sum4(ps_a)
        l_b = l_b * al_b + _row_sum4(ps_b)
        m_a = mn_a
        m_b = mn_b
        comptime for dt in range(ND):
            o[dt][0] *= al_a
            o[dt][1] *= al_a
            o[dt][2] *= al_b
            o[dt][3] *= al_b
        # O += P·V: P's accumulator halves are the A fragments.
        comptime for kk in range(TK // 16):
            var pa = _afrag_from_c(s[2 * kk], s[2 * kk + 1])
            comptime for dp in range(ND // 2):
                var bb = _ldm_bt(Vs, HD + _PAD, kk * 16, dp * 16, l)
                mma(o[2 * dp], pa, _lo(bb), o[2 * dp])
                mma(o[2 * dp + 1], pa, _hi(bb), o[2 * dp + 1])
        barrier()  # every warp is done with this buffer before it is refilled

    var inv_a = Scalar[F32](1) / l_a
    var inv_b = Scalar[F32](1) / l_b
    comptime for dt in range(ND):
        var col = h * HD + dt * 8 + q * 2
        if r_a < S:
            var off = b * OUT_DIM + r_a * DIM + col
            var v = SIMD[F32, 2](o[dt][0], o[dt][1]) * inv_a
            outp.unsafe_store[alignment=4](off, v.cast[ADT]())
            o_cache.unsafe_store[alignment=4](off, v.cast[ADT]())
        if r_b < S:
            var off = b * OUT_DIM + r_b * DIM + col
            var v = SIMD[F32, 2](o[dt][2], o[dt][3]) * inv_b
            outp.unsafe_store[alignment=4](off, v.cast[ADT]())
            o_cache.unsafe_store[alignment=4](off, v.cast[ADT]())
    if q == 0:
        if r_a < S:
            lse[unsafe_offset=bh * S + r_a] = rebind[Scalar[DT]](m_a + log2(l_a))
        if r_b < S:
            lse[unsafe_offset=bh * S + r_b] = rebind[Scalar[DT]](m_b + log2(l_b))


def _mma_dq_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, IL: Bool
](
    inp: Pointer[Scalar[ADT], MutAnyOrigin],
    dout: Pointer[Scalar[ADT], MutAnyOrigin],
    o_cache: Pointer[Scalar[ADT], MutAnyOrigin],
    lse: Pointer[Scalar[DT], MutAnyOrigin],
    dvec: Pointer[Scalar[DT], MutAnyOrigin],
    gin: Pointer[Scalar[ADT], MutAnyOrigin],
):
    """Also computes D = Σ_d dO·O for its rows (from the dO fragments it
    already holds and the bf16 O, as FA2) and writes it to `dvec` for the
    dK/dV kernel — no separate D kernel."""
    comptime TQ = MMA_TQ
    comptime TK = MMA_TK
    comptime NT = MMA_WARPS * WARP_SIZE
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM
    comptime KO = DIM if IL else S * DIM
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime KS = HD // 16
    comptime NK = TK // 8
    comptime ND = HD // 8
    var qt = Int(block_idx.x)
    var bh = Int(block_idx.y)
    var b = bh // H
    var h = bh - b * H
    var t = Int(thread_idx.x)
    var w = t // WARP_SIZE
    var l = t - w * WARP_SIZE
    var g = l // 4
    var q = l - g * 4
    var scale = Scalar[F32](1.0) / sqrt(Scalar[F32](HD))
    var scale2 = scale * _LOG2E
    var base = b * IN_DIM + h * HD
    var obase = b * OUT_DIM + h * HD
    var r_a = qt * TQ + w * 16 + g
    var r_b = r_a + 8

    # Double-buffered K / V: tile kt + 1 is copied (cp.async) while kt computes.
    var Ks0 = _smem[TK * (HD + _PAD)]()
    var Ks1 = _smem[TK * (HD + _PAD)]()
    var Vs0 = _smem[TK * (HD + _PAD)]()
    var Vs1 = _smem[TK * (HD + _PAD)]()
    _stage_async[ADT, TK, HD, NT](inp, base + KO, IS, 0, S, Ks0, t)
    _stage_async[ADT, TK, HD, NT](inp, base + VO, IS, 0, S, Vs0, t)
    async_copy_commit_group()

    var qf = Array[SIMD[BF, 8], KS](fill=SIMD[BF, 8](0))
    var df = Array[SIMD[BF, 8], KS](fill=SIMD[BF, 8](0))
    comptime for ks in range(KS):
        qf[ks] = _afrag_global[ADT](inp, base, IS, r_a, r_b, r_a < S, r_b < S, ks * 16 + q * 2)
        df[ks] = _afrag_global[ADT](dout, obase, DIM, r_a, r_b, r_a < S, r_b < S, ks * 16 + q * 2)
    var L_a = Scalar[F32](0)
    var L_b = Scalar[F32](0)
    # D = Σ_d dO·O: this lane's 4·KS columns of each row, then the 4 lanes.
    var pd_a = Scalar[F32](0)
    var pd_b = Scalar[F32](0)
    comptime for ks in range(KS):
        var c0 = obase + ks * 16 + q * 2
        if r_a < S:
            var o0 = o_cache.unsafe_load[width=2, alignment=4](c0 + r_a * DIM).cast[F32]()
            var o1 = o_cache.unsafe_load[width=2, alignment=4](c0 + r_a * DIM + 8).cast[F32]()
            pd_a += (df[ks][0].cast[F32]() * rebind[Scalar[F32]](o0[0])
                + df[ks][1].cast[F32]() * rebind[Scalar[F32]](o0[1])
                + df[ks][4].cast[F32]() * rebind[Scalar[F32]](o1[0])
                + df[ks][5].cast[F32]() * rebind[Scalar[F32]](o1[1]))
        if r_b < S:
            var o0 = o_cache.unsafe_load[width=2, alignment=4](c0 + r_b * DIM).cast[F32]()
            var o1 = o_cache.unsafe_load[width=2, alignment=4](c0 + r_b * DIM + 8).cast[F32]()
            pd_b += (df[ks][2].cast[F32]() * rebind[Scalar[F32]](o0[0])
                + df[ks][3].cast[F32]() * rebind[Scalar[F32]](o0[1])
                + df[ks][6].cast[F32]() * rebind[Scalar[F32]](o1[0])
                + df[ks][7].cast[F32]() * rebind[Scalar[F32]](o1[1]))
    var D_a = _row_sum4(pd_a)
    var D_b = _row_sum4(pd_b)
    if r_a < S:
        L_a = rebind[Scalar[F32]](lse[unsafe_offset=bh * S + r_a])
        if q == 0:
            dvec[unsafe_offset=bh * S + r_a] = rebind[Scalar[DT]](D_a)
    if r_b < S:
        L_b = rebind[Scalar[F32]](lse[unsafe_offset=bh * S + r_b])
        if q == 0:
            dvec[unsafe_offset=bh * S + r_b] = rebind[Scalar[DT]](D_b)

    comptime N_KT = (S + TK - 1) // TK
    var kt_end = N_KT
    comptime if CAUSAL:
        kt_end = min(N_KT, (qt * TQ + TQ - 1) // TK + 1)

    var dq = Array[SIMD[F32, 4], ND](fill=SIMD[F32, 4](0))
    for kt in range(kt_end):
        var even = (kt & 1) == 0
        var Ks = Ks0 if even else Ks1
        var Vs = Vs0 if even else Vs1
        if kt + 1 < kt_end:
            _stage_async[ADT, TK, HD, NT](inp, base + KO, IS, (kt + 1) * TK, S, Ks1 if even else Ks0, t)
            _stage_async[ADT, TK, HD, NT](inp, base + VO, IS, (kt + 1) * TK, S, Vs1 if even else Vs0, t)
            async_copy_commit_group()
            async_copy_wait_group(1)
        else:
            async_copy_wait_group(0)
        barrier()
        var s = Array[SIMD[F32, 4], NK](fill=SIMD[F32, 4](0))
        var dp = Array[SIMD[F32, 4], NK](fill=SIMD[F32, 4](0))
        comptime for np in range(NK // 2):
            comptime for ks in range(KS):
                var bk = _ldm_b(Ks, HD + _PAD, np * 16, ks * 16, l)
                var bv = _ldm_b(Vs, HD + _PAD, np * 16, ks * 16, l)
                mma(s[2 * np], qf[ks], _lo(bk), s[2 * np])
                mma(s[2 * np + 1], qf[ks], _hi(bk), s[2 * np + 1])
                mma(dp[2 * np], df[ks], _lo(bv), dp[2 * np])
                mma(dp[2 * np + 1], df[ks], _hi(bv), dp[2 * np + 1])
        # dS = P ⊙ (dP − D), P = exp2(S·scale2 − L).
        comptime for nt in range(NK):
            comptime for e in range(4):
                var col = kt * TK + nt * 8 + q * 2 + (e & 1)
                var row = r_a if e < 2 else r_b
                var ok = col < S and row < S
                comptime if CAUSAL:
                    ok = ok and col <= row
                var ds = Scalar[F32](0)
                if ok:
                    var p = exp2(s[nt][e] * scale2 - (L_a if e < 2 else L_b))
                    ds = p * (dp[nt][e] - (D_a if e < 2 else D_b))
                s[nt][e] = ds
        comptime for kk in range(TK // 16):
            var da = _afrag_from_c(s[2 * kk], s[2 * kk + 1])
            comptime for dq2 in range(ND // 2):
                var bb = _ldm_bt(Ks, HD + _PAD, kk * 16, dq2 * 16, l)
                mma(dq[2 * dq2], da, _lo(bb), dq[2 * dq2])
                mma(dq[2 * dq2 + 1], da, _hi(bb), dq[2 * dq2 + 1])
        barrier()

    comptime for dt in range(ND):
        var col = h * HD + dt * 8 + q * 2
        if r_a < S:
            var v = SIMD[F32, 2](dq[dt][0], dq[dt][1]) * scale
            gin.unsafe_store[alignment=4](b * IN_DIM + r_a * IS + col, v.cast[ADT]())
        if r_b < S:
            var v = SIMD[F32, 2](dq[dt][2], dq[dt][3]) * scale
            gin.unsafe_store[alignment=4](b * IN_DIM + r_b * IS + col, v.cast[ADT]())


def _mma_dkdv_kernel[
    ADT: DType, H: Int, S: Int, HD: Int, CAUSAL: Bool, IL: Bool
](
    inp: Pointer[Scalar[ADT], MutAnyOrigin],
    dout: Pointer[Scalar[ADT], MutAnyOrigin],
    lse: Pointer[Scalar[DT], MutAnyOrigin],
    dvec: Pointer[Scalar[DT], MutAnyOrigin],
    gin: Pointer[Scalar[ADT], MutAnyOrigin],
):
    comptime TQ = MMA_TQ
    comptime TK = MMA_TK
    comptime NT = MMA_WARPS * WARP_SIZE
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM
    comptime KO = DIM if IL else S * DIM
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime KS = HD // 16
    comptime NQ = TQ // 8  # n tiles over the queries
    comptime ND = HD // 8
    var kt = Int(block_idx.x)
    var bh = Int(block_idx.y)
    var b = bh // H
    var h = bh - b * H
    var t = Int(thread_idx.x)
    var w = t // WARP_SIZE
    var l = t - w * WARP_SIZE
    var g = l // 4
    var q = l - g * 4
    var scale = Scalar[F32](1.0) / sqrt(Scalar[F32](HD))
    var scale2 = scale * _LOG2E
    var base = b * IN_DIM + h * HD
    var obase = b * OUT_DIM + h * HD
    var k_a = kt * TK + w * 16 + g  # this lane's two keys
    var k_b = k_a + 8

    var Qs0 = _smem[TQ * (HD + _PAD)]()
    var Qs1 = _smem[TQ * (HD + _PAD)]()
    var Ds0 = _smem[TQ * (HD + _PAD)]()
    var Ds1 = _smem[TQ * (HD + _PAD)]()
    var Lsh = stack_allocation[TQ, Scalar[F32], address_space=AddressSpace.SHARED]()
    var Dsh = stack_allocation[TQ, Scalar[F32], address_space=AddressSpace.SHARED]()

    var kf = Array[SIMD[BF, 8], KS](fill=SIMD[BF, 8](0))
    var vf = Array[SIMD[BF, 8], KS](fill=SIMD[BF, 8](0))
    comptime for ks in range(KS):
        kf[ks] = _afrag_global[ADT](inp, base + KO, IS, k_a, k_b, k_a < S, k_b < S, ks * 16 + q * 2)
        vf[ks] = _afrag_global[ADT](inp, base + VO, IS, k_a, k_b, k_a < S, k_b < S, ks * 16 + q * 2)

    comptime N_QT = (S + TQ - 1) // TQ
    var qt0 = 0
    comptime if CAUSAL:
        qt0 = (kt * TK) // TQ

    var dk = Array[SIMD[F32, 4], ND](fill=SIMD[F32, 4](0))
    var dv = Array[SIMD[F32, 4], ND](fill=SIMD[F32, 4](0))
    _stage_async[ADT, TQ, HD, NT](inp, base, IS, qt0 * TQ, S, Qs0, t)
    _stage_async[ADT, TQ, HD, NT](dout, obase, DIM, qt0 * TQ, S, Ds0, t)
    async_copy_commit_group()
    for qt in range(qt0, N_QT):
        var even = ((qt - qt0) & 1) == 0
        var Qs = Qs0 if even else Qs1
        var Ds = Ds0 if even else Ds1
        if qt + 1 < N_QT:
            _stage_async[ADT, TQ, HD, NT](inp, base, IS, (qt + 1) * TQ, S, Qs1 if even else Qs0, t)
            _stage_async[ADT, TQ, HD, NT](dout, obase, DIM, (qt + 1) * TQ, S, Ds1 if even else Ds0, t)
            async_copy_commit_group()
        if t < TQ:
            var r = qt * TQ + t
            var lv = Scalar[F32](0)
            var dvv = Scalar[F32](0)
            if r < S:
                lv = rebind[Scalar[F32]](lse[unsafe_offset=bh * S + r])
                dvv = rebind[Scalar[F32]](dvec[unsafe_offset=bh * S + r])
            Lsh[unsafe_offset=t] = lv
            Dsh[unsafe_offset=t] = dvv
        if qt + 1 < N_QT:
            async_copy_wait_group(1)
        else:
            async_copy_wait_group(0)
        barrier()
        # Sᵀ = K·Qᵀ and dPᵀ = V·dOᵀ: rows = this warp's keys, cols = queries.
        var s = Array[SIMD[F32, 4], NQ](fill=SIMD[F32, 4](0))
        var dp = Array[SIMD[F32, 4], NQ](fill=SIMD[F32, 4](0))
        comptime for np in range(NQ // 2):
            comptime for ks in range(KS):
                var bq = _ldm_b(Qs, HD + _PAD, np * 16, ks * 16, l)
                var bd = _ldm_b(Ds, HD + _PAD, np * 16, ks * 16, l)
                mma(s[2 * np], kf[ks], _lo(bq), s[2 * np])
                mma(s[2 * np + 1], kf[ks], _hi(bq), s[2 * np + 1])
                mma(dp[2 * np], vf[ks], _lo(bd), dp[2 * np])
                mma(dp[2 * np + 1], vf[ks], _hi(bd), dp[2 * np + 1])
        # Pᵀ (kept in s) and dSᵀ (in dp).
        comptime for nt in range(NQ):
            comptime for e in range(4):
                var cl = nt * 8 + q * 2 + (e & 1)
                var col = qt * TQ + cl  # query
                var key = k_a if e < 2 else k_b
                var ok = col < S and key < S
                comptime if CAUSAL:
                    ok = ok and key <= col
                var p = Scalar[F32](0)
                var ds = Scalar[F32](0)
                if ok:
                    p = exp2(s[nt][e] * scale2 - Lsh[unsafe_offset=cl])
                    ds = p * (dp[nt][e] - Dsh[unsafe_offset=cl])
                s[nt][e] = p
                dp[nt][e] = ds
        # dV += Pᵀ·dO, dK += dSᵀ·Q.
        comptime for kk in range(TQ // 16):
            var pa = _afrag_from_c(s[2 * kk], s[2 * kk + 1])
            var da = _afrag_from_c(dp[2 * kk], dp[2 * kk + 1])
            comptime for dd in range(ND // 2):
                var bdo = _ldm_bt(Ds, HD + _PAD, kk * 16, dd * 16, l)
                var bq = _ldm_bt(Qs, HD + _PAD, kk * 16, dd * 16, l)
                mma(dv[2 * dd], pa, _lo(bdo), dv[2 * dd])
                mma(dv[2 * dd + 1], pa, _hi(bdo), dv[2 * dd + 1])
                mma(dk[2 * dd], da, _lo(bq), dk[2 * dd])
                mma(dk[2 * dd + 1], da, _hi(bq), dk[2 * dd + 1])
        barrier()  # Lsh / Dsh and this buffer are rewritten next iteration

    comptime for dt in range(ND):
        var col = h * HD + dt * 8 + q * 2
        if k_a < S:
            var o = b * IN_DIM + k_a * IS + col
            var kv = SIMD[F32, 2](dk[dt][0], dk[dt][1]) * scale
            var vv = SIMD[F32, 2](dv[dt][0], dv[dt][1])
            gin.unsafe_store[alignment=4](o + KO, kv.cast[ADT]())
            gin.unsafe_store[alignment=4](o + VO, vv.cast[ADT]())
        if k_b < S:
            var o = b * IN_DIM + k_b * IS + col
            var kv = SIMD[F32, 2](dk[dt][2], dk[dt][3]) * scale
            var vv = SIMD[F32, 2](dv[dt][2], dv[dt][3])
            gin.unsafe_store[alignment=4](o + KO, kv.cast[ADT]())
            gin.unsafe_store[alignment=4](o + VO, vv.cast[ADT]())
