"""Fused attention on the tensor cores in TF32, for fp32 activations on
NVIDIA.

The fp32 twin of `flash_attention_mma.mojo` (bf16): the same FlashAttention-2
— entry points, buffers, layouts, outputs (O, its fp32 cache `o_cache`, the
log-sum-exp L in log2 units), deterministic (no atomics), D = Σ dO·O fused
into the dQ kernel, `cp.async` double buffering — with every product on
`mma.sync.m16n8k8.tf32` (fp32 accumulators) instead of fp32 FMAs on the CUDA
cores. TF32 is the precision `cublas_tf32` already gives every GEMM of 2^24+
multiply-adds (an attention layer's products are far past it).

  - `_tf32_fwd_kernel`   one block per 64-query tile, 4 warps x 16 rows, Q in
                         registers; per 32-key tile S = Q·Kᵀ, online softmax,
                         O += P·V.
  - `_tf32_dq_kernel`    D, then dQ = scale · Σ dS·K over key tiles.
  - `_tf32_dkdv_kernel`  one block per 64-key tile: dV = Σ Pᵀ·dO, dK = scale ·
                         Σ dSᵀ·Q over query tiles.

Fragments (PTX m16n8k8.tf32, lane l, g = l / 4, q = l % 4; checked on an RTX
5090): A [16 x 8] holds (g, q), (g+8, q), (g, q+4), (g+8, q+4); B [8(k) x
8(n)] holds (k = q, n = g), (k = q+4, n = g); C [16 x 8] holds (g, 2q..2q+1),
(g+8, 2q..2q+1). The accumulator's adjacent column pair does not match A's
(q, q+4) columns, so every product contracts over a PERMUTED k: within each
8-wide k step, k-slot q is element 2q and k-slot q+4 is element 2q+1 — on
both operands, so the sum is unchanged. Then:
  - an accumulator tile IS the next product's A fragment, (c0, c2, c1, c3);
  - a B fragment over the head dim (K, V, Q, dO rows: [n][k] tiles) is ONE
    8-byte load, (T[n][2q], T[n][2q+1]);
  - a B fragment over the tokens ([k][n] tiles) is two column loads,
    (T[2q][n], T[2q+1][n]).
Shared rows are HD + 8 floats (≡ 8 mod 32): the row-pair loads hit distinct
banks per half-warp, the column loads are 2-way.

TF32 conversion: the tensor core TRUNCATES an fp32 operand to TF32
(measured: 1 + 0.75·2^-10 enters as 1.0), which would shrink every softmax
weight towards zero. Operands are rounded first (`_tf32`, half a TF32 ulp
added to the bits, CUTLASS's round_half_ulp_truncate).
"""

from std.math import exp2, log2, sqrt
from std.memory import stack_allocation, bitcast
from max.gpu import thread_idx, block_idx, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.memory import (
    AddressSpace, async_copy, async_copy_commit_group, async_copy_wait_group,
)
from max.gpu.compute.mma import mma

from noeira.nn.constants import DT
from .flash_attention_mma import _row_max4, _row_sum4


comptime F32 = DType.float32
comptime TF_TQ = 64  # outer tile: 4 warps x 16 rows
comptime TF_TI = 32  # inner tile (keys / queries per loop step)
comptime TF_WARPS = 4
comptime _NEG = Scalar[F32](-1e30)
comptime _LOG2E = Scalar[F32](1.4426950408889634)


def tf32_eligible[ADT: DType, HD: Int, H: Int]() -> Bool:
    """fp32 activations, a head dim of whole 8-wide k steps, 16-byte rows."""
    return ADT == F32 and HD % 8 == 0 and HD <= 128 and (H * HD) % 4 == 0


comptime _SPtr = Pointer[Scalar[F32], MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def _smem[N: Int]() -> _SPtr:
    return stack_allocation[
        N, Scalar[F32], address_space=AddressSpace.SHARED, alignment=16
    ]()


@always_inline
def _tf32(v: SIMD[F32, 4]) -> SIMD[F32, 4]:
    """Round to TF32 (the tensor core truncates the low 13 bits)."""
    return bitcast[F32, 4](bitcast[DType.uint32, 4](v) + UInt32(0x1000))


@always_inline
def _tf32(v: SIMD[F32, 2]) -> SIMD[F32, 2]:
    return bitcast[F32, 2](bitcast[DType.uint32, 2](v) + UInt32(0x1000))


@always_inline
def _a_from_c(c: SIMD[F32, 4]) -> SIMD[F32, 4]:
    """The A fragment (permuted k) of an accumulator tile."""
    return _tf32(SIMD[F32, 4](c[0], c[2], c[1], c[3]))


@always_inline
def _b_row(p: _SPtr, ld: Int, n: Int, k0: Int) -> SIMD[F32, 2]:
    """B from a [n][k] tile: (T[n][k0], T[n][k0 + 1]), k0 = step·8 + 2q."""
    return _tf32(p.unsafe_load[width=2, alignment=8](n * ld + k0))


@always_inline
def _b_col(p: _SPtr, ld: Int, k0: Int, n: Int) -> SIMD[F32, 2]:
    """B from a [k][n] tile: (T[k0][n], T[k0 + 1][n]), k0 = step·8 + 2q."""
    return _tf32(
        SIMD[F32, 2](p[unsafe_offset=k0 * ld + n], p[unsafe_offset=(k0 + 1) * ld + n])
    )


@always_inline
def _a_global(
    src: Pointer[Scalar[F32], MutAnyOrigin],
    base: Int, stride: Int, r: Int, r8: Int, ok_r: Bool, ok_r8: Bool, k0: Int,
) -> SIMD[F32, 4]:
    """A fragment (permuted k) of rows r, r + 8 at columns k0, k0 + 1 (k0 =
    step·8 + 2q), from global memory; invalid rows read as zero."""
    var x = SIMD[F32, 2](0)
    var y = SIMD[F32, 2](0)
    if ok_r:
        x = src.unsafe_load[width=2, alignment=8](base + r * stride + k0)
    if ok_r8:
        y = src.unsafe_load[width=2, alignment=8](base + r8 * stride + k0)
    return _tf32(SIMD[F32, 4](x[0], y[0], x[1], y[1]))


@always_inline
def _stage_async[
    R: Int, HD: Int, NT: Int
](
    src: Pointer[Scalar[F32], MutAnyOrigin],
    base: Int, stride: Int, row0: Int, S: Int, dst: _SPtr, t: Int,
):
    """R x HD fp32 rows into a [R][HD + 8] shared tile with `cp.async`
    (16 bytes = 4 floats a copy); rows >= S zero-filled by the copy."""
    comptime C4 = HD // 4
    comptime CH = (R * C4 + NT - 1) // NT
    var g = src.unsafe_address_space_cast[AddressSpace.GLOBAL]()
    comptime for ch in range(CH):
        var idx = t + ch * NT
        if idx < R * C4:
            var rr = idx // C4
            var c0 = (idx - rr * C4) * 4
            var ok = row0 + rr < S
            var row = row0 + rr if ok else 0
            async_copy[16](
                g.unsafe_offset(base + row * stride + c0),
                dst.unsafe_offset(rr * (HD + 8) + c0),
                src_size=Int32(16 if ok else 0),
            )


def _tf32_fwd_kernel[
    H: Int, S: Int, HD: Int, CAUSAL: Bool, IL: Bool
](
    inp: Pointer[Scalar[F32], MutAnyOrigin],
    outp: Pointer[Scalar[F32], MutAnyOrigin],
    o_cache: Pointer[Scalar[F32], MutAnyOrigin],
    lse: Pointer[Scalar[F32], MutAnyOrigin],
):
    comptime TQ = TF_TQ
    comptime TK = TF_TI
    comptime NT = TF_WARPS * WARP_SIZE
    comptime LD = HD + 8
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM
    comptime KO = DIM if IL else S * DIM
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime KS = HD // 8
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
    var scale2 = Scalar[F32](1.0) / sqrt(Scalar[F32](HD)) * _LOG2E
    var base = b * IN_DIM + h * HD
    var r_a = qt * TQ + w * 16 + g
    var r_b = r_a + 8

    var Ks0 = _smem[TK * LD]()
    var Ks1 = _smem[TK * LD]()
    var Vs0 = _smem[TK * LD]()
    var Vs1 = _smem[TK * LD]()
    _stage_async[TK, HD, NT](inp, base + KO, IS, 0, S, Ks0, t)
    _stage_async[TK, HD, NT](inp, base + VO, IS, 0, S, Vs0, t)
    async_copy_commit_group()

    var qf = Array[SIMD[F32, 4], KS](fill=SIMD[F32, 4](0))
    comptime for ks in range(KS):
        qf[ks] = _a_global(inp, base, IS, r_a, r_b, r_a < S, r_b < S, ks * 8 + q * 2)

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
            _stage_async[TK, HD, NT](inp, base + KO, IS, (kt + 1) * TK, S, Ks1 if even else Ks0, t)
            _stage_async[TK, HD, NT](inp, base + VO, IS, (kt + 1) * TK, S, Vs1 if even else Vs0, t)
            async_copy_commit_group()
            async_copy_wait_group(1)
        else:
            async_copy_wait_group(0)
        barrier()
        var s = Array[SIMD[F32, 4], NK](fill=SIMD[F32, 4](0))
        comptime for nt in range(NK):
            comptime for ks in range(KS):
                mma(s[nt], qf[ks], _b_row(Ks, LD, nt * 8 + g, ks * 8 + q * 2), s[nt])
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
        comptime for kk in range(NK):
            var pa = _a_from_c(s[kk])
            comptime for dt in range(ND):
                mma(o[dt], pa, _b_col(Vs, LD, kk * 8 + q * 2, dt * 8 + g), o[dt])
        barrier()

    var inv_a = Scalar[F32](1) / l_a
    var inv_b = Scalar[F32](1) / l_b
    comptime for dt in range(ND):
        var col = h * HD + dt * 8 + q * 2
        if r_a < S:
            var off = b * OUT_DIM + r_a * DIM + col
            var v = SIMD[F32, 2](o[dt][0], o[dt][1]) * inv_a
            outp.unsafe_store[alignment=8](off, v)
            o_cache.unsafe_store[alignment=8](off, v)
        if r_b < S:
            var off = b * OUT_DIM + r_b * DIM + col
            var v = SIMD[F32, 2](o[dt][2], o[dt][3]) * inv_b
            outp.unsafe_store[alignment=8](off, v)
            o_cache.unsafe_store[alignment=8](off, v)
    if q == 0:
        if r_a < S:
            lse[unsafe_offset=bh * S + r_a] = m_a + log2(l_a)
        if r_b < S:
            lse[unsafe_offset=bh * S + r_b] = m_b + log2(l_b)


def _tf32_dq_kernel[
    H: Int, S: Int, HD: Int, CAUSAL: Bool, IL: Bool
](
    inp: Pointer[Scalar[F32], MutAnyOrigin],
    dout: Pointer[Scalar[F32], MutAnyOrigin],
    o_cache: Pointer[Scalar[F32], MutAnyOrigin],
    lse: Pointer[Scalar[F32], MutAnyOrigin],
    dvec: Pointer[Scalar[F32], MutAnyOrigin],
    gin: Pointer[Scalar[F32], MutAnyOrigin],
):
    """D = Σ dO·O for its rows (written to `dvec` for dK/dV), then dQ."""
    comptime TQ = TF_TQ
    comptime TK = TF_TI
    comptime NT = TF_WARPS * WARP_SIZE
    comptime LD = HD + 8
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM
    comptime KO = DIM if IL else S * DIM
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime KS = HD // 8
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

    var Ks0 = _smem[TK * LD]()
    var Ks1 = _smem[TK * LD]()
    var Vs0 = _smem[TK * LD]()
    var Vs1 = _smem[TK * LD]()
    _stage_async[TK, HD, NT](inp, base + KO, IS, 0, S, Ks0, t)
    _stage_async[TK, HD, NT](inp, base + VO, IS, 0, S, Vs0, t)
    async_copy_commit_group()

    var qf = Array[SIMD[F32, 4], KS](fill=SIMD[F32, 4](0))
    var df = Array[SIMD[F32, 4], KS](fill=SIMD[F32, 4](0))
    var pd_a = Scalar[F32](0)
    var pd_b = Scalar[F32](0)
    comptime for ks in range(KS):
        var k0 = ks * 8 + q * 2
        qf[ks] = _a_global(inp, base, IS, r_a, r_b, r_a < S, r_b < S, k0)
        # dO unrounded for D; rounded below for the MMA.
        var x = SIMD[F32, 2](0)
        var y = SIMD[F32, 2](0)
        if r_a < S:
            x = dout.unsafe_load[width=2, alignment=8](obase + r_a * DIM + k0)
            var oa = o_cache.unsafe_load[width=2, alignment=8](obase + r_a * DIM + k0)
            pd_a += x[0] * oa[0] + x[1] * oa[1]
        if r_b < S:
            y = dout.unsafe_load[width=2, alignment=8](obase + r_b * DIM + k0)
            var ob = o_cache.unsafe_load[width=2, alignment=8](obase + r_b * DIM + k0)
            pd_b += y[0] * ob[0] + y[1] * ob[1]
        df[ks] = _tf32(SIMD[F32, 4](x[0], y[0], x[1], y[1]))
    var D_a = _row_sum4(pd_a)
    var D_b = _row_sum4(pd_b)
    var L_a = Scalar[F32](0)
    var L_b = Scalar[F32](0)
    if r_a < S:
        L_a = lse[unsafe_offset=bh * S + r_a]
        if q == 0:
            dvec[unsafe_offset=bh * S + r_a] = D_a
    if r_b < S:
        L_b = lse[unsafe_offset=bh * S + r_b]
        if q == 0:
            dvec[unsafe_offset=bh * S + r_b] = D_b

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
            _stage_async[TK, HD, NT](inp, base + KO, IS, (kt + 1) * TK, S, Ks1 if even else Ks0, t)
            _stage_async[TK, HD, NT](inp, base + VO, IS, (kt + 1) * TK, S, Vs1 if even else Vs0, t)
            async_copy_commit_group()
            async_copy_wait_group(1)
        else:
            async_copy_wait_group(0)
        barrier()
        var s = Array[SIMD[F32, 4], NK](fill=SIMD[F32, 4](0))
        var dp = Array[SIMD[F32, 4], NK](fill=SIMD[F32, 4](0))
        comptime for nt in range(NK):
            comptime for ks in range(KS):
                mma(s[nt], qf[ks], _b_row(Ks, LD, nt * 8 + g, ks * 8 + q * 2), s[nt])
                mma(dp[nt], df[ks], _b_row(Vs, LD, nt * 8 + g, ks * 8 + q * 2), dp[nt])
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
        comptime for kk in range(NK):
            var da = _a_from_c(s[kk])
            comptime for dt in range(ND):
                mma(dq[dt], da, _b_col(Ks, LD, kk * 8 + q * 2, dt * 8 + g), dq[dt])
        barrier()

    comptime for dt in range(ND):
        var col = h * HD + dt * 8 + q * 2
        if r_a < S:
            gin.unsafe_store[alignment=8](
                b * IN_DIM + r_a * IS + col, SIMD[F32, 2](dq[dt][0], dq[dt][1]) * scale
            )
        if r_b < S:
            gin.unsafe_store[alignment=8](
                b * IN_DIM + r_b * IS + col, SIMD[F32, 2](dq[dt][2], dq[dt][3]) * scale
            )


def _tf32_dkdv_kernel[
    H: Int, S: Int, HD: Int, CAUSAL: Bool, IL: Bool
](
    inp: Pointer[Scalar[F32], MutAnyOrigin],
    dout: Pointer[Scalar[F32], MutAnyOrigin],
    lse: Pointer[Scalar[F32], MutAnyOrigin],
    dvec: Pointer[Scalar[F32], MutAnyOrigin],
    gin: Pointer[Scalar[F32], MutAnyOrigin],
):
    comptime TK = TF_TQ  # this block's keys: 4 warps x 16
    comptime TQ = TF_TI  # queries per loop step
    comptime NT = TF_WARPS * WARP_SIZE
    comptime LD = HD + 8
    comptime DIM = H * HD
    comptime IN_DIM = 3 * S * DIM
    comptime OUT_DIM = S * DIM
    comptime IS = 3 * DIM if IL else DIM
    comptime KO = DIM if IL else S * DIM
    comptime VO = 2 * DIM if IL else 2 * S * DIM
    comptime KS = HD // 8
    comptime NQ = TQ // 8
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
    var k_a = kt * TK + w * 16 + g
    var k_b = k_a + 8

    var Qs0 = _smem[TQ * LD]()
    var Qs1 = _smem[TQ * LD]()
    var Ds0 = _smem[TQ * LD]()
    var Ds1 = _smem[TQ * LD]()
    var Lsh = stack_allocation[TQ, Scalar[F32], address_space=AddressSpace.SHARED]()
    var Dsh = stack_allocation[TQ, Scalar[F32], address_space=AddressSpace.SHARED]()

    var kf = Array[SIMD[F32, 4], KS](fill=SIMD[F32, 4](0))
    var vf = Array[SIMD[F32, 4], KS](fill=SIMD[F32, 4](0))
    comptime for ks in range(KS):
        kf[ks] = _a_global(inp, base + KO, IS, k_a, k_b, k_a < S, k_b < S, ks * 8 + q * 2)
        vf[ks] = _a_global(inp, base + VO, IS, k_a, k_b, k_a < S, k_b < S, ks * 8 + q * 2)

    comptime N_QT = (S + TQ - 1) // TQ
    var qt0 = 0
    comptime if CAUSAL:
        qt0 = (kt * TK) // TQ

    var dk = Array[SIMD[F32, 4], ND](fill=SIMD[F32, 4](0))
    var dv = Array[SIMD[F32, 4], ND](fill=SIMD[F32, 4](0))
    _stage_async[TQ, HD, NT](inp, base, IS, qt0 * TQ, S, Qs0, t)
    _stage_async[TQ, HD, NT](dout, obase, DIM, qt0 * TQ, S, Ds0, t)
    async_copy_commit_group()
    for qt in range(qt0, N_QT):
        var even = ((qt - qt0) & 1) == 0
        var Qs = Qs0 if even else Qs1
        var Ds = Ds0 if even else Ds1
        if qt + 1 < N_QT:
            _stage_async[TQ, HD, NT](inp, base, IS, (qt + 1) * TQ, S, Qs1 if even else Qs0, t)
            _stage_async[TQ, HD, NT](dout, obase, DIM, (qt + 1) * TQ, S, Ds1 if even else Ds0, t)
            async_copy_commit_group()
        if t < TQ:
            var r = qt * TQ + t
            var lv = Scalar[F32](0)
            var dvv = Scalar[F32](0)
            if r < S:
                lv = lse[unsafe_offset=bh * S + r]
                dvv = dvec[unsafe_offset=bh * S + r]
            Lsh[unsafe_offset=t] = lv
            Dsh[unsafe_offset=t] = dvv
        if qt + 1 < N_QT:
            async_copy_wait_group(1)
        else:
            async_copy_wait_group(0)
        barrier()
        var s = Array[SIMD[F32, 4], NQ](fill=SIMD[F32, 4](0))
        var dp = Array[SIMD[F32, 4], NQ](fill=SIMD[F32, 4](0))
        comptime for nt in range(NQ):
            comptime for ks in range(KS):
                mma(s[nt], kf[ks], _b_row(Qs, LD, nt * 8 + g, ks * 8 + q * 2), s[nt])
                mma(dp[nt], vf[ks], _b_row(Ds, LD, nt * 8 + g, ks * 8 + q * 2), dp[nt])
        comptime for nt in range(NQ):
            comptime for e in range(4):
                var cl = nt * 8 + q * 2 + (e & 1)
                var col = qt * TQ + cl
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
        comptime for kk in range(NQ):
            var pa = _a_from_c(s[kk])
            var da = _a_from_c(dp[kk])
            comptime for dt in range(ND):
                mma(dv[dt], pa, _b_col(Ds, LD, kk * 8 + q * 2, dt * 8 + g), dv[dt])
                mma(dk[dt], da, _b_col(Qs, LD, kk * 8 + q * 2, dt * 8 + g), dk[dt])
        barrier()

    comptime for dt in range(ND):
        var col = h * HD + dt * 8 + q * 2
        if k_a < S:
            var o = b * IN_DIM + k_a * IS + col
            gin.unsafe_store[alignment=8](o + KO, SIMD[F32, 2](dk[dt][0], dk[dt][1]) * scale)
            gin.unsafe_store[alignment=8](o + VO, SIMD[F32, 2](dv[dt][0], dv[dt][1]))
        if k_b < S:
            var o = b * IN_DIM + k_b * IS + col
            gin.unsafe_store[alignment=8](o + KO, SIMD[F32, 2](dk[dt][2], dk[dt][3]) * scale)
            gin.unsafe_store[alignment=8](o + VO, SIMD[F32, 2](dv[dt][2], dv[dt][3]))
