"""A GEMM on Apple's simdgroup matrix hardware (8 x 8 MMA), written from scratch.

    o[M, N] (+)= op(a) @ op(b)      op(a) = a[M, K] or a[K, M]ᵀ (TA)
                                    op(b) = b[K, N] or b[N, K]ᵀ (TB)

THE ROUTE TO THE HARDWARE. Metal Shading Language's `simdgroup_load`,
`simdgroup_multiply_accumulate`, `simdgroup_store` and
`make_filled_simdgroup_matrix` compile to calls of four AIR functions (read
off `xcrun metal -S -emit-llvm` of a small MSL kernel of our own):

    <64 x float> air.simdgroup_matrix_8x8_load.v64f32.p{1,3}f32(ptr, <2 x i64> dims, <2 x i64> strides, <2 x i64> origin)
    <64 x float> air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float)
    <64 x float> air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(a, b, c)   # a·b + c
    void         air.simdgroup_matrix_8x8_store.v64f32.p{1,3}f32(m, ptr, dims, strides, origin)

`dims` = (8, 8); `strides` = (1, row stride) for row-major, (row stride, 1)
for the transposed load; p1 = device, p3 = threadgroup memory. They are not
LLVM intrinsics (`llvm_intrinsic` cannot find them) but external functions
the Metal backend lowers, so `external_call` reaches them. A matrix is an
opaque `<64 x float>` per thread (`SIMD[f32, 64]`); the hardware keeps two
elements of it per lane.

DESIGN. 128 threads (4 simdgroups, 2 x 2) per threadgroup; `sg_mm` picks:

- DIRECT kernels (`sg_direct_kernel_RAxCB`), when M, N are tile multiples
  and K a multiple of 8: each simdgroup holds RA x CB 8 x 8 accumulators and
  loads its fragments straight from device memory, the tape's transposes
  being the hardware's transposed load. No staging, no barrier. Plain
  products: 4 x 8 (64 x 128 tile) from 32 tiles up; aᵀ·b: 2 x 8 (32 x 128);
  a·bᵀ and small outputs: 4 x 4 (64 x 64).
- The STAGED kernel (`sg_mm_kernel`) otherwise: a 64 x 64 tile, simdgroup
  (r, c) owning a 32 x 32 quarter as 4 x 4 accumulators. Each 16-deep
  k-slab of op(a) (64 x 16) and op(b) (16 x 64) is staged in threadgroup
  memory in canonical row-major form — the transposes are absorbed by the
  staging loads, which walk the global array along its contiguous axis —
  zero-filled past M, N, K. Interior tiles store fragments straight to `o`
  (with `ACC`, the accumulators start as `o`'s own fragments); edge tiles go
  through threadgroup memory and a bounds-checked copy.

Each output's sum runs k = 0, 1, … in hardware order: deterministic, run to
run and launcher to launcher.

⚠ THE ACCUMULATORS ARE NAMED LOCALS, not an `Array[M64, 16]`. In an
array (memory) each `<64 x float>` was materialised whole per thread, and
the kernel ran at 0.04 TFLOP/s — 20x slower than the scalar tiled kernel.
As SSA values the backend keeps each in two registers per lane.
"""

from std.ffi import external_call
from std.memory import stack_allocation
from max.gpu import thread_idx, block_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from max.gpu.host import DeviceBuffer
from noeira.nn.constants import DT
from .backend import Dev

comptime P = Pointer[Scalar[DT], MutAnyOrigin]
comptime M64 = SIMD[DType.float32, 64]
comptime I2 = SIMD[DType.int64, 2]

comptime SG_T = 64  # output tile edge
comptime SG_K = 16  # k-slab
comptime SG_THREADS = 128  # 4 simdgroups of 32
comptime SG_PAD = 4  # threadgroup row padding (floats)
comptime SG_AS = SG_K + SG_PAD  # A slab row stride: [64][16 + pad]
comptime SG_BS = SG_T + SG_PAD  # B slab row stride: [16][64 + pad]
comptime SG_OS = SG_T + SG_PAD  # edge-tile output row stride
comptime SG_WIDE_MIN = 64 * 128 * 32
"""Plain products take the 64 x 128 tile from 32 tiles of it up."""


@always_inline
def _ld_g(p: P, off: Int, stride: Int) -> M64:
    var g = (p + off).unsafe_address_space_cast[AddressSpace.GLOBAL]()
    return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", M64](
        g, I2(8, 8), I2(1, Int64(stride)), I2(0, 0)
    )


@always_inline
def _ld_gt(p: P, off: Int, stride: Int) -> M64:
    """The transposed load: element (r, c) from p[off + c * stride + r]."""
    var g = (p + off).unsafe_address_space_cast[AddressSpace.GLOBAL]()
    return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", M64](
        g, I2(8, 8), I2(Int64(stride), 1), I2(0, 0)
    )


@always_inline
def _st_g(m: M64, p: P, off: Int, stride: Int):
    var g = (p + off).unsafe_address_space_cast[AddressSpace.GLOBAL]()
    external_call["air.simdgroup_matrix_8x8_store.v64f32.p1f32", NoneType](
        m, g, I2(8, 8), I2(1, Int64(stride)), I2(0, 0)
    )


@always_inline
def _ld_s[o: Origin](
    p: Pointer[Scalar[DT], o, address_space=AddressSpace.SHARED],
    off: Int,
    stride: Int,
) -> M64:
    return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", M64](
        p + off, I2(8, 8), I2(1, Int64(stride)), I2(0, 0)
    )


@always_inline
def _st_s[o: Origin](
    m: M64,
    p: Pointer[Scalar[DT], o, address_space=AddressSpace.SHARED],
    off: Int,
    stride: Int,
):
    external_call["air.simdgroup_matrix_8x8_store.v64f32.p3f32", NoneType](
        m, p + off, I2(8, 8), I2(1, Int64(stride)), I2(0, 0)
    )


@always_inline
def _mma(a: M64, b: M64, c: M64) -> M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        M64,
    ](a, b, c)


@always_inline
def _zero() -> M64:
    return external_call["air.simdgroup_matrix_8x8_init_filled.v64f32.f32", M64](
        Float32(0)
    )


def sg_direct_kernel_4x4[TA: Bool, TB: Bool, ACC: Bool](
    o: P, a: P, b: P, m_: Int64, n_: Int64, k_: Int64
):
    """Direct form, no edges: M % 64 == 0, N % 64 == 0, K % 8 == 0
    (the caller checks). 2 x 2 simdgroups, each 32 x 32 outputs as
    4 x 4 accumulators; fragments loaded straight from device memory."""
    var m = Int(m_)
    var n = Int(n_)
    var kk = Int(k_)
    var sg = Int(thread_idx.x) // 32
    var r0 = Int(block_idx.y) * 64 + (sg // 2) * 32
    var c0 = Int(block_idx.x) * 64 + (sg % 2) * 32
    var c0_0 = _zero()
    var c0_1 = _zero()
    var c0_2 = _zero()
    var c0_3 = _zero()
    var c1_0 = _zero()
    var c1_1 = _zero()
    var c1_2 = _zero()
    var c1_3 = _zero()
    var c2_0 = _zero()
    var c2_1 = _zero()
    var c2_2 = _zero()
    var c2_3 = _zero()
    var c3_0 = _zero()
    var c3_1 = _zero()
    var c3_2 = _zero()
    var c3_3 = _zero()
    comptime if ACC:
        c0_0 = _ld_g(o, (r0 + 0) * n + c0 + 0, n)
        c0_1 = _ld_g(o, (r0 + 0) * n + c0 + 8, n)
        c0_2 = _ld_g(o, (r0 + 0) * n + c0 + 16, n)
        c0_3 = _ld_g(o, (r0 + 0) * n + c0 + 24, n)
        c1_0 = _ld_g(o, (r0 + 8) * n + c0 + 0, n)
        c1_1 = _ld_g(o, (r0 + 8) * n + c0 + 8, n)
        c1_2 = _ld_g(o, (r0 + 8) * n + c0 + 16, n)
        c1_3 = _ld_g(o, (r0 + 8) * n + c0 + 24, n)
        c2_0 = _ld_g(o, (r0 + 16) * n + c0 + 0, n)
        c2_1 = _ld_g(o, (r0 + 16) * n + c0 + 8, n)
        c2_2 = _ld_g(o, (r0 + 16) * n + c0 + 16, n)
        c2_3 = _ld_g(o, (r0 + 16) * n + c0 + 24, n)
        c3_0 = _ld_g(o, (r0 + 24) * n + c0 + 0, n)
        c3_1 = _ld_g(o, (r0 + 24) * n + c0 + 8, n)
        c3_2 = _ld_g(o, (r0 + 24) * n + c0 + 16, n)
        c3_3 = _ld_g(o, (r0 + 24) * n + c0 + 24, n)
    var k = 0
    while k < kk:
        var a0: M64
        comptime if TA:
            a0 = _ld_gt(a, k * m + r0 + 0, m)
        else:
            a0 = _ld_g(a, (r0 + 0) * kk + k, kk)
        var a1: M64
        comptime if TA:
            a1 = _ld_gt(a, k * m + r0 + 8, m)
        else:
            a1 = _ld_g(a, (r0 + 8) * kk + k, kk)
        var a2: M64
        comptime if TA:
            a2 = _ld_gt(a, k * m + r0 + 16, m)
        else:
            a2 = _ld_g(a, (r0 + 16) * kk + k, kk)
        var a3: M64
        comptime if TA:
            a3 = _ld_gt(a, k * m + r0 + 24, m)
        else:
            a3 = _ld_g(a, (r0 + 24) * kk + k, kk)
        var b0: M64
        comptime if TB:
            b0 = _ld_gt(b, (c0 + 0) * kk + k, kk)
        else:
            b0 = _ld_g(b, k * n + c0 + 0, n)
        var b1: M64
        comptime if TB:
            b1 = _ld_gt(b, (c0 + 8) * kk + k, kk)
        else:
            b1 = _ld_g(b, k * n + c0 + 8, n)
        var b2: M64
        comptime if TB:
            b2 = _ld_gt(b, (c0 + 16) * kk + k, kk)
        else:
            b2 = _ld_g(b, k * n + c0 + 16, n)
        var b3: M64
        comptime if TB:
            b3 = _ld_gt(b, (c0 + 24) * kk + k, kk)
        else:
            b3 = _ld_g(b, k * n + c0 + 24, n)
        c0_0 = _mma(a0, b0, c0_0)
        c0_1 = _mma(a0, b1, c0_1)
        c0_2 = _mma(a0, b2, c0_2)
        c0_3 = _mma(a0, b3, c0_3)
        c1_0 = _mma(a1, b0, c1_0)
        c1_1 = _mma(a1, b1, c1_1)
        c1_2 = _mma(a1, b2, c1_2)
        c1_3 = _mma(a1, b3, c1_3)
        c2_0 = _mma(a2, b0, c2_0)
        c2_1 = _mma(a2, b1, c2_1)
        c2_2 = _mma(a2, b2, c2_2)
        c2_3 = _mma(a2, b3, c2_3)
        c3_0 = _mma(a3, b0, c3_0)
        c3_1 = _mma(a3, b1, c3_1)
        c3_2 = _mma(a3, b2, c3_2)
        c3_3 = _mma(a3, b3, c3_3)
        k += 8
    _st_g(c0_0, o, (r0 + 0) * n + c0 + 0, n)
    _st_g(c0_1, o, (r0 + 0) * n + c0 + 8, n)
    _st_g(c0_2, o, (r0 + 0) * n + c0 + 16, n)
    _st_g(c0_3, o, (r0 + 0) * n + c0 + 24, n)
    _st_g(c1_0, o, (r0 + 8) * n + c0 + 0, n)
    _st_g(c1_1, o, (r0 + 8) * n + c0 + 8, n)
    _st_g(c1_2, o, (r0 + 8) * n + c0 + 16, n)
    _st_g(c1_3, o, (r0 + 8) * n + c0 + 24, n)
    _st_g(c2_0, o, (r0 + 16) * n + c0 + 0, n)
    _st_g(c2_1, o, (r0 + 16) * n + c0 + 8, n)
    _st_g(c2_2, o, (r0 + 16) * n + c0 + 16, n)
    _st_g(c2_3, o, (r0 + 16) * n + c0 + 24, n)
    _st_g(c3_0, o, (r0 + 24) * n + c0 + 0, n)
    _st_g(c3_1, o, (r0 + 24) * n + c0 + 8, n)
    _st_g(c3_2, o, (r0 + 24) * n + c0 + 16, n)
    _st_g(c3_3, o, (r0 + 24) * n + c0 + 24, n)


def sg_direct_kernel_4x8[TA: Bool, TB: Bool, ACC: Bool](
    o: P, a: P, b: P, m_: Int64, n_: Int64, k_: Int64
):
    """Direct form, no edges: M % 64 == 0, N % 128 == 0, K % 8 == 0
    (the caller checks). 2 x 2 simdgroups, each 32 x 64 outputs as
    4 x 8 accumulators; fragments loaded straight from device memory."""
    var m = Int(m_)
    var n = Int(n_)
    var kk = Int(k_)
    var sg = Int(thread_idx.x) // 32
    var r0 = Int(block_idx.y) * 64 + (sg // 2) * 32
    var c0 = Int(block_idx.x) * 128 + (sg % 2) * 64
    var c0_0 = _zero()
    var c0_1 = _zero()
    var c0_2 = _zero()
    var c0_3 = _zero()
    var c0_4 = _zero()
    var c0_5 = _zero()
    var c0_6 = _zero()
    var c0_7 = _zero()
    var c1_0 = _zero()
    var c1_1 = _zero()
    var c1_2 = _zero()
    var c1_3 = _zero()
    var c1_4 = _zero()
    var c1_5 = _zero()
    var c1_6 = _zero()
    var c1_7 = _zero()
    var c2_0 = _zero()
    var c2_1 = _zero()
    var c2_2 = _zero()
    var c2_3 = _zero()
    var c2_4 = _zero()
    var c2_5 = _zero()
    var c2_6 = _zero()
    var c2_7 = _zero()
    var c3_0 = _zero()
    var c3_1 = _zero()
    var c3_2 = _zero()
    var c3_3 = _zero()
    var c3_4 = _zero()
    var c3_5 = _zero()
    var c3_6 = _zero()
    var c3_7 = _zero()
    comptime if ACC:
        c0_0 = _ld_g(o, (r0 + 0) * n + c0 + 0, n)
        c0_1 = _ld_g(o, (r0 + 0) * n + c0 + 8, n)
        c0_2 = _ld_g(o, (r0 + 0) * n + c0 + 16, n)
        c0_3 = _ld_g(o, (r0 + 0) * n + c0 + 24, n)
        c0_4 = _ld_g(o, (r0 + 0) * n + c0 + 32, n)
        c0_5 = _ld_g(o, (r0 + 0) * n + c0 + 40, n)
        c0_6 = _ld_g(o, (r0 + 0) * n + c0 + 48, n)
        c0_7 = _ld_g(o, (r0 + 0) * n + c0 + 56, n)
        c1_0 = _ld_g(o, (r0 + 8) * n + c0 + 0, n)
        c1_1 = _ld_g(o, (r0 + 8) * n + c0 + 8, n)
        c1_2 = _ld_g(o, (r0 + 8) * n + c0 + 16, n)
        c1_3 = _ld_g(o, (r0 + 8) * n + c0 + 24, n)
        c1_4 = _ld_g(o, (r0 + 8) * n + c0 + 32, n)
        c1_5 = _ld_g(o, (r0 + 8) * n + c0 + 40, n)
        c1_6 = _ld_g(o, (r0 + 8) * n + c0 + 48, n)
        c1_7 = _ld_g(o, (r0 + 8) * n + c0 + 56, n)
        c2_0 = _ld_g(o, (r0 + 16) * n + c0 + 0, n)
        c2_1 = _ld_g(o, (r0 + 16) * n + c0 + 8, n)
        c2_2 = _ld_g(o, (r0 + 16) * n + c0 + 16, n)
        c2_3 = _ld_g(o, (r0 + 16) * n + c0 + 24, n)
        c2_4 = _ld_g(o, (r0 + 16) * n + c0 + 32, n)
        c2_5 = _ld_g(o, (r0 + 16) * n + c0 + 40, n)
        c2_6 = _ld_g(o, (r0 + 16) * n + c0 + 48, n)
        c2_7 = _ld_g(o, (r0 + 16) * n + c0 + 56, n)
        c3_0 = _ld_g(o, (r0 + 24) * n + c0 + 0, n)
        c3_1 = _ld_g(o, (r0 + 24) * n + c0 + 8, n)
        c3_2 = _ld_g(o, (r0 + 24) * n + c0 + 16, n)
        c3_3 = _ld_g(o, (r0 + 24) * n + c0 + 24, n)
        c3_4 = _ld_g(o, (r0 + 24) * n + c0 + 32, n)
        c3_5 = _ld_g(o, (r0 + 24) * n + c0 + 40, n)
        c3_6 = _ld_g(o, (r0 + 24) * n + c0 + 48, n)
        c3_7 = _ld_g(o, (r0 + 24) * n + c0 + 56, n)
    var k = 0
    while k < kk:
        var a0: M64
        comptime if TA:
            a0 = _ld_gt(a, k * m + r0 + 0, m)
        else:
            a0 = _ld_g(a, (r0 + 0) * kk + k, kk)
        var a1: M64
        comptime if TA:
            a1 = _ld_gt(a, k * m + r0 + 8, m)
        else:
            a1 = _ld_g(a, (r0 + 8) * kk + k, kk)
        var a2: M64
        comptime if TA:
            a2 = _ld_gt(a, k * m + r0 + 16, m)
        else:
            a2 = _ld_g(a, (r0 + 16) * kk + k, kk)
        var a3: M64
        comptime if TA:
            a3 = _ld_gt(a, k * m + r0 + 24, m)
        else:
            a3 = _ld_g(a, (r0 + 24) * kk + k, kk)
        var b0: M64
        comptime if TB:
            b0 = _ld_gt(b, (c0 + 0) * kk + k, kk)
        else:
            b0 = _ld_g(b, k * n + c0 + 0, n)
        var b1: M64
        comptime if TB:
            b1 = _ld_gt(b, (c0 + 8) * kk + k, kk)
        else:
            b1 = _ld_g(b, k * n + c0 + 8, n)
        var b2: M64
        comptime if TB:
            b2 = _ld_gt(b, (c0 + 16) * kk + k, kk)
        else:
            b2 = _ld_g(b, k * n + c0 + 16, n)
        var b3: M64
        comptime if TB:
            b3 = _ld_gt(b, (c0 + 24) * kk + k, kk)
        else:
            b3 = _ld_g(b, k * n + c0 + 24, n)
        var b4: M64
        comptime if TB:
            b4 = _ld_gt(b, (c0 + 32) * kk + k, kk)
        else:
            b4 = _ld_g(b, k * n + c0 + 32, n)
        var b5: M64
        comptime if TB:
            b5 = _ld_gt(b, (c0 + 40) * kk + k, kk)
        else:
            b5 = _ld_g(b, k * n + c0 + 40, n)
        var b6: M64
        comptime if TB:
            b6 = _ld_gt(b, (c0 + 48) * kk + k, kk)
        else:
            b6 = _ld_g(b, k * n + c0 + 48, n)
        var b7: M64
        comptime if TB:
            b7 = _ld_gt(b, (c0 + 56) * kk + k, kk)
        else:
            b7 = _ld_g(b, k * n + c0 + 56, n)
        c0_0 = _mma(a0, b0, c0_0)
        c0_1 = _mma(a0, b1, c0_1)
        c0_2 = _mma(a0, b2, c0_2)
        c0_3 = _mma(a0, b3, c0_3)
        c0_4 = _mma(a0, b4, c0_4)
        c0_5 = _mma(a0, b5, c0_5)
        c0_6 = _mma(a0, b6, c0_6)
        c0_7 = _mma(a0, b7, c0_7)
        c1_0 = _mma(a1, b0, c1_0)
        c1_1 = _mma(a1, b1, c1_1)
        c1_2 = _mma(a1, b2, c1_2)
        c1_3 = _mma(a1, b3, c1_3)
        c1_4 = _mma(a1, b4, c1_4)
        c1_5 = _mma(a1, b5, c1_5)
        c1_6 = _mma(a1, b6, c1_6)
        c1_7 = _mma(a1, b7, c1_7)
        c2_0 = _mma(a2, b0, c2_0)
        c2_1 = _mma(a2, b1, c2_1)
        c2_2 = _mma(a2, b2, c2_2)
        c2_3 = _mma(a2, b3, c2_3)
        c2_4 = _mma(a2, b4, c2_4)
        c2_5 = _mma(a2, b5, c2_5)
        c2_6 = _mma(a2, b6, c2_6)
        c2_7 = _mma(a2, b7, c2_7)
        c3_0 = _mma(a3, b0, c3_0)
        c3_1 = _mma(a3, b1, c3_1)
        c3_2 = _mma(a3, b2, c3_2)
        c3_3 = _mma(a3, b3, c3_3)
        c3_4 = _mma(a3, b4, c3_4)
        c3_5 = _mma(a3, b5, c3_5)
        c3_6 = _mma(a3, b6, c3_6)
        c3_7 = _mma(a3, b7, c3_7)
        k += 8
    _st_g(c0_0, o, (r0 + 0) * n + c0 + 0, n)
    _st_g(c0_1, o, (r0 + 0) * n + c0 + 8, n)
    _st_g(c0_2, o, (r0 + 0) * n + c0 + 16, n)
    _st_g(c0_3, o, (r0 + 0) * n + c0 + 24, n)
    _st_g(c0_4, o, (r0 + 0) * n + c0 + 32, n)
    _st_g(c0_5, o, (r0 + 0) * n + c0 + 40, n)
    _st_g(c0_6, o, (r0 + 0) * n + c0 + 48, n)
    _st_g(c0_7, o, (r0 + 0) * n + c0 + 56, n)
    _st_g(c1_0, o, (r0 + 8) * n + c0 + 0, n)
    _st_g(c1_1, o, (r0 + 8) * n + c0 + 8, n)
    _st_g(c1_2, o, (r0 + 8) * n + c0 + 16, n)
    _st_g(c1_3, o, (r0 + 8) * n + c0 + 24, n)
    _st_g(c1_4, o, (r0 + 8) * n + c0 + 32, n)
    _st_g(c1_5, o, (r0 + 8) * n + c0 + 40, n)
    _st_g(c1_6, o, (r0 + 8) * n + c0 + 48, n)
    _st_g(c1_7, o, (r0 + 8) * n + c0 + 56, n)
    _st_g(c2_0, o, (r0 + 16) * n + c0 + 0, n)
    _st_g(c2_1, o, (r0 + 16) * n + c0 + 8, n)
    _st_g(c2_2, o, (r0 + 16) * n + c0 + 16, n)
    _st_g(c2_3, o, (r0 + 16) * n + c0 + 24, n)
    _st_g(c2_4, o, (r0 + 16) * n + c0 + 32, n)
    _st_g(c2_5, o, (r0 + 16) * n + c0 + 40, n)
    _st_g(c2_6, o, (r0 + 16) * n + c0 + 48, n)
    _st_g(c2_7, o, (r0 + 16) * n + c0 + 56, n)
    _st_g(c3_0, o, (r0 + 24) * n + c0 + 0, n)
    _st_g(c3_1, o, (r0 + 24) * n + c0 + 8, n)
    _st_g(c3_2, o, (r0 + 24) * n + c0 + 16, n)
    _st_g(c3_3, o, (r0 + 24) * n + c0 + 24, n)
    _st_g(c3_4, o, (r0 + 24) * n + c0 + 32, n)
    _st_g(c3_5, o, (r0 + 24) * n + c0 + 40, n)
    _st_g(c3_6, o, (r0 + 24) * n + c0 + 48, n)
    _st_g(c3_7, o, (r0 + 24) * n + c0 + 56, n)


def sg_direct_kernel_2x8[TA: Bool, TB: Bool, ACC: Bool](
    o: P, a: P, b: P, m_: Int64, n_: Int64, k_: Int64
):
    """Direct form, no edges: M % 32 == 0, N % 128 == 0, K % 8 == 0
    (the caller checks). 2 x 2 simdgroups, each 16 x 64 outputs as
    2 x 8 accumulators; fragments loaded straight from device memory."""
    var m = Int(m_)
    var n = Int(n_)
    var kk = Int(k_)
    var sg = Int(thread_idx.x) // 32
    var r0 = Int(block_idx.y) * 32 + (sg // 2) * 16
    var c0 = Int(block_idx.x) * 128 + (sg % 2) * 64
    var c0_0 = _zero()
    var c0_1 = _zero()
    var c0_2 = _zero()
    var c0_3 = _zero()
    var c0_4 = _zero()
    var c0_5 = _zero()
    var c0_6 = _zero()
    var c0_7 = _zero()
    var c1_0 = _zero()
    var c1_1 = _zero()
    var c1_2 = _zero()
    var c1_3 = _zero()
    var c1_4 = _zero()
    var c1_5 = _zero()
    var c1_6 = _zero()
    var c1_7 = _zero()
    comptime if ACC:
        c0_0 = _ld_g(o, (r0 + 0) * n + c0 + 0, n)
        c0_1 = _ld_g(o, (r0 + 0) * n + c0 + 8, n)
        c0_2 = _ld_g(o, (r0 + 0) * n + c0 + 16, n)
        c0_3 = _ld_g(o, (r0 + 0) * n + c0 + 24, n)
        c0_4 = _ld_g(o, (r0 + 0) * n + c0 + 32, n)
        c0_5 = _ld_g(o, (r0 + 0) * n + c0 + 40, n)
        c0_6 = _ld_g(o, (r0 + 0) * n + c0 + 48, n)
        c0_7 = _ld_g(o, (r0 + 0) * n + c0 + 56, n)
        c1_0 = _ld_g(o, (r0 + 8) * n + c0 + 0, n)
        c1_1 = _ld_g(o, (r0 + 8) * n + c0 + 8, n)
        c1_2 = _ld_g(o, (r0 + 8) * n + c0 + 16, n)
        c1_3 = _ld_g(o, (r0 + 8) * n + c0 + 24, n)
        c1_4 = _ld_g(o, (r0 + 8) * n + c0 + 32, n)
        c1_5 = _ld_g(o, (r0 + 8) * n + c0 + 40, n)
        c1_6 = _ld_g(o, (r0 + 8) * n + c0 + 48, n)
        c1_7 = _ld_g(o, (r0 + 8) * n + c0 + 56, n)
    var k = 0
    while k < kk:
        var a0: M64
        comptime if TA:
            a0 = _ld_gt(a, k * m + r0 + 0, m)
        else:
            a0 = _ld_g(a, (r0 + 0) * kk + k, kk)
        var a1: M64
        comptime if TA:
            a1 = _ld_gt(a, k * m + r0 + 8, m)
        else:
            a1 = _ld_g(a, (r0 + 8) * kk + k, kk)
        var b0: M64
        comptime if TB:
            b0 = _ld_gt(b, (c0 + 0) * kk + k, kk)
        else:
            b0 = _ld_g(b, k * n + c0 + 0, n)
        var b1: M64
        comptime if TB:
            b1 = _ld_gt(b, (c0 + 8) * kk + k, kk)
        else:
            b1 = _ld_g(b, k * n + c0 + 8, n)
        var b2: M64
        comptime if TB:
            b2 = _ld_gt(b, (c0 + 16) * kk + k, kk)
        else:
            b2 = _ld_g(b, k * n + c0 + 16, n)
        var b3: M64
        comptime if TB:
            b3 = _ld_gt(b, (c0 + 24) * kk + k, kk)
        else:
            b3 = _ld_g(b, k * n + c0 + 24, n)
        var b4: M64
        comptime if TB:
            b4 = _ld_gt(b, (c0 + 32) * kk + k, kk)
        else:
            b4 = _ld_g(b, k * n + c0 + 32, n)
        var b5: M64
        comptime if TB:
            b5 = _ld_gt(b, (c0 + 40) * kk + k, kk)
        else:
            b5 = _ld_g(b, k * n + c0 + 40, n)
        var b6: M64
        comptime if TB:
            b6 = _ld_gt(b, (c0 + 48) * kk + k, kk)
        else:
            b6 = _ld_g(b, k * n + c0 + 48, n)
        var b7: M64
        comptime if TB:
            b7 = _ld_gt(b, (c0 + 56) * kk + k, kk)
        else:
            b7 = _ld_g(b, k * n + c0 + 56, n)
        c0_0 = _mma(a0, b0, c0_0)
        c0_1 = _mma(a0, b1, c0_1)
        c0_2 = _mma(a0, b2, c0_2)
        c0_3 = _mma(a0, b3, c0_3)
        c0_4 = _mma(a0, b4, c0_4)
        c0_5 = _mma(a0, b5, c0_5)
        c0_6 = _mma(a0, b6, c0_6)
        c0_7 = _mma(a0, b7, c0_7)
        c1_0 = _mma(a1, b0, c1_0)
        c1_1 = _mma(a1, b1, c1_1)
        c1_2 = _mma(a1, b2, c1_2)
        c1_3 = _mma(a1, b3, c1_3)
        c1_4 = _mma(a1, b4, c1_4)
        c1_5 = _mma(a1, b5, c1_5)
        c1_6 = _mma(a1, b6, c1_6)
        c1_7 = _mma(a1, b7, c1_7)
        k += 8
    _st_g(c0_0, o, (r0 + 0) * n + c0 + 0, n)
    _st_g(c0_1, o, (r0 + 0) * n + c0 + 8, n)
    _st_g(c0_2, o, (r0 + 0) * n + c0 + 16, n)
    _st_g(c0_3, o, (r0 + 0) * n + c0 + 24, n)
    _st_g(c0_4, o, (r0 + 0) * n + c0 + 32, n)
    _st_g(c0_5, o, (r0 + 0) * n + c0 + 40, n)
    _st_g(c0_6, o, (r0 + 0) * n + c0 + 48, n)
    _st_g(c0_7, o, (r0 + 0) * n + c0 + 56, n)
    _st_g(c1_0, o, (r0 + 8) * n + c0 + 0, n)
    _st_g(c1_1, o, (r0 + 8) * n + c0 + 8, n)
    _st_g(c1_2, o, (r0 + 8) * n + c0 + 16, n)
    _st_g(c1_3, o, (r0 + 8) * n + c0 + 24, n)
    _st_g(c1_4, o, (r0 + 8) * n + c0 + 32, n)
    _st_g(c1_5, o, (r0 + 8) * n + c0 + 40, n)
    _st_g(c1_6, o, (r0 + 8) * n + c0 + 48, n)
    _st_g(c1_7, o, (r0 + 8) * n + c0 + 56, n)


def sg_mm_kernel[TA: Bool, TB: Bool, ACC: Bool](
    o: P, a: P, b: P, m_: Int64, n_: Int64, k_: Int64
):
    var m = Int(m_)
    var n = Int(n_)
    var kk = Int(k_)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var sr = (sg // 2) * 32  # this simdgroup's rows inside the tile
    var sc = (sg % 2) * 32  # and columns
    var row0 = Int(block_idx.y) * SG_T
    var col0 = Int(block_idx.x) * SG_T
    var interior = row0 + SG_T <= m and col0 + SG_T <= n

    var As = stack_allocation[
        SG_T * SG_AS, Scalar[DT], address_space=AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        SG_K * SG_BS, Scalar[DT], address_space=AddressSpace.SHARED
    ]()

    var c00 = _zero()
    var c01 = _zero()
    var c02 = _zero()
    var c03 = _zero()
    var c10 = _zero()
    var c11 = _zero()
    var c12 = _zero()
    var c13 = _zero()
    var c20 = _zero()
    var c21 = _zero()
    var c22 = _zero()
    var c23 = _zero()
    var c30 = _zero()
    var c31 = _zero()
    var c32 = _zero()
    var c33 = _zero()
    comptime if ACC:
        if interior:
            c00 = _ld_g(o, (row0 + sr + 0) * n + col0 + sc + 0, n)
            c01 = _ld_g(o, (row0 + sr + 0) * n + col0 + sc + 8, n)
            c02 = _ld_g(o, (row0 + sr + 0) * n + col0 + sc + 16, n)
            c03 = _ld_g(o, (row0 + sr + 0) * n + col0 + sc + 24, n)
            c10 = _ld_g(o, (row0 + sr + 8) * n + col0 + sc + 0, n)
            c11 = _ld_g(o, (row0 + sr + 8) * n + col0 + sc + 8, n)
            c12 = _ld_g(o, (row0 + sr + 8) * n + col0 + sc + 16, n)
            c13 = _ld_g(o, (row0 + sr + 8) * n + col0 + sc + 24, n)
            c20 = _ld_g(o, (row0 + sr + 16) * n + col0 + sc + 0, n)
            c21 = _ld_g(o, (row0 + sr + 16) * n + col0 + sc + 8, n)
            c22 = _ld_g(o, (row0 + sr + 16) * n + col0 + sc + 16, n)
            c23 = _ld_g(o, (row0 + sr + 16) * n + col0 + sc + 24, n)
            c30 = _ld_g(o, (row0 + sr + 24) * n + col0 + sc + 0, n)
            c31 = _ld_g(o, (row0 + sr + 24) * n + col0 + sc + 8, n)
            c32 = _ld_g(o, (row0 + sr + 24) * n + col0 + sc + 16, n)
            c33 = _ld_g(o, (row0 + sr + 24) * n + col0 + sc + 24, n)

    if interior and kk % 8 == 0:
        # DIRECT: fragments straight from device memory (the transposes are
        # the hardware's transposed load), no staging, no barrier.
        var k = 0
        while k < kk:
            var a0: M64
            comptime if TA:
                a0 = _ld_gt(a, k * m + row0 + sr + 0, m)
            else:
                a0 = _ld_g(a, (row0 + sr + 0) * kk + k, kk)
            var a1: M64
            comptime if TA:
                a1 = _ld_gt(a, k * m + row0 + sr + 8, m)
            else:
                a1 = _ld_g(a, (row0 + sr + 8) * kk + k, kk)
            var a2: M64
            comptime if TA:
                a2 = _ld_gt(a, k * m + row0 + sr + 16, m)
            else:
                a2 = _ld_g(a, (row0 + sr + 16) * kk + k, kk)
            var a3: M64
            comptime if TA:
                a3 = _ld_gt(a, k * m + row0 + sr + 24, m)
            else:
                a3 = _ld_g(a, (row0 + sr + 24) * kk + k, kk)
            var b0: M64
            comptime if TB:
                b0 = _ld_gt(b, (col0 + sc + 0) * kk + k, kk)
            else:
                b0 = _ld_g(b, k * n + col0 + sc + 0, n)
            var b1: M64
            comptime if TB:
                b1 = _ld_gt(b, (col0 + sc + 8) * kk + k, kk)
            else:
                b1 = _ld_g(b, k * n + col0 + sc + 8, n)
            var b2: M64
            comptime if TB:
                b2 = _ld_gt(b, (col0 + sc + 16) * kk + k, kk)
            else:
                b2 = _ld_g(b, k * n + col0 + sc + 16, n)
            var b3: M64
            comptime if TB:
                b3 = _ld_gt(b, (col0 + sc + 24) * kk + k, kk)
            else:
                b3 = _ld_g(b, k * n + col0 + sc + 24, n)
            c00 = _mma(a0, b0, c00)
            c01 = _mma(a0, b1, c01)
            c02 = _mma(a0, b2, c02)
            c03 = _mma(a0, b3, c03)
            c10 = _mma(a1, b0, c10)
            c11 = _mma(a1, b1, c11)
            c12 = _mma(a1, b2, c12)
            c13 = _mma(a1, b3, c13)
            c20 = _mma(a2, b0, c20)
            c21 = _mma(a2, b1, c21)
            c22 = _mma(a2, b2, c22)
            c23 = _mma(a2, b3, c23)
            c30 = _mma(a3, b0, c30)
            c31 = _mma(a3, b1, c31)
            c32 = _mma(a3, b2, c32)
            c33 = _mma(a3, b3, c33)
            k += 8
        _st_g(c00, o, (row0 + sr + 0) * n + col0 + sc + 0, n)
        _st_g(c01, o, (row0 + sr + 0) * n + col0 + sc + 8, n)
        _st_g(c02, o, (row0 + sr + 0) * n + col0 + sc + 16, n)
        _st_g(c03, o, (row0 + sr + 0) * n + col0 + sc + 24, n)
        _st_g(c10, o, (row0 + sr + 8) * n + col0 + sc + 0, n)
        _st_g(c11, o, (row0 + sr + 8) * n + col0 + sc + 8, n)
        _st_g(c12, o, (row0 + sr + 8) * n + col0 + sc + 16, n)
        _st_g(c13, o, (row0 + sr + 8) * n + col0 + sc + 24, n)
        _st_g(c20, o, (row0 + sr + 16) * n + col0 + sc + 0, n)
        _st_g(c21, o, (row0 + sr + 16) * n + col0 + sc + 8, n)
        _st_g(c22, o, (row0 + sr + 16) * n + col0 + sc + 16, n)
        _st_g(c23, o, (row0 + sr + 16) * n + col0 + sc + 24, n)
        _st_g(c30, o, (row0 + sr + 24) * n + col0 + sc + 0, n)
        _st_g(c31, o, (row0 + sr + 24) * n + col0 + sc + 8, n)
        _st_g(c32, o, (row0 + sr + 24) * n + col0 + sc + 16, n)
        _st_g(c33, o, (row0 + sr + 24) * n + col0 + sc + 24, n)
        return

    # STAGED (edge tiles, or K not a multiple of 8)
    var k0 = 0
    while k0 < kk:
        # stage op(a)[row0:+64, k0:+16] and op(b)[k0:+16, col0:+64]
        comptime for q in range(SG_T * SG_K // SG_THREADS):
            var e = tid + q * SG_THREADS
            var ia: Int
            var ka: Int
            comptime if TA:  # a is [K, M]: walk along m
                ka = e // SG_T
                ia = e % SG_T
            else:  # a is [M, K]: walk along k
                ia = e // SG_K
                ka = e % SG_K
            var gi = row0 + ia
            var gk = k0 + ka
            var av: Scalar[DT] = 0
            if gi < m and gk < kk:
                comptime if TA:
                    av = a[unsafe_offset=gk * m + gi]
                else:
                    av = a[unsafe_offset=gi * kk + gk]
            As[unsafe_offset=ia * SG_AS + ka] = av
            var jb: Int
            var kb: Int
            comptime if TB:  # b is [N, K]: walk along k
                jb = e // SG_K
                kb = e % SG_K
            else:  # b is [K, N]: walk along n
                kb = e // SG_T
                jb = e % SG_T
            var gj = col0 + jb
            var gk2 = k0 + kb
            var bv: Scalar[DT] = 0
            if gj < n and gk2 < kk:
                comptime if TB:
                    bv = b[unsafe_offset=gj * kk + gk2]
                else:
                    bv = b[unsafe_offset=gk2 * n + gj]
            Bs[unsafe_offset=kb * SG_BS + jb] = bv
        barrier()
        comptime for k8 in range(SG_K // 8):
            var a0 = _ld_s(As, (sr + 0) * SG_AS + 8 * k8, SG_AS)
            var a1 = _ld_s(As, (sr + 8) * SG_AS + 8 * k8, SG_AS)
            var a2 = _ld_s(As, (sr + 16) * SG_AS + 8 * k8, SG_AS)
            var a3 = _ld_s(As, (sr + 24) * SG_AS + 8 * k8, SG_AS)
            var b0 = _ld_s(Bs, (8 * k8) * SG_BS + sc + 0, SG_BS)
            var b1 = _ld_s(Bs, (8 * k8) * SG_BS + sc + 8, SG_BS)
            var b2 = _ld_s(Bs, (8 * k8) * SG_BS + sc + 16, SG_BS)
            var b3 = _ld_s(Bs, (8 * k8) * SG_BS + sc + 24, SG_BS)
            c00 = _mma(a0, b0, c00)
            c01 = _mma(a0, b1, c01)
            c02 = _mma(a0, b2, c02)
            c03 = _mma(a0, b3, c03)
            c10 = _mma(a1, b0, c10)
            c11 = _mma(a1, b1, c11)
            c12 = _mma(a1, b2, c12)
            c13 = _mma(a1, b3, c13)
            c20 = _mma(a2, b0, c20)
            c21 = _mma(a2, b1, c21)
            c22 = _mma(a2, b2, c22)
            c23 = _mma(a2, b3, c23)
            c30 = _mma(a3, b0, c30)
            c31 = _mma(a3, b1, c31)
            c32 = _mma(a3, b2, c32)
            c33 = _mma(a3, b3, c33)
        barrier()
        k0 += SG_K

    if interior:
        _st_g(c00, o, (row0 + sr + 0) * n + col0 + sc + 0, n)
        _st_g(c01, o, (row0 + sr + 0) * n + col0 + sc + 8, n)
        _st_g(c02, o, (row0 + sr + 0) * n + col0 + sc + 16, n)
        _st_g(c03, o, (row0 + sr + 0) * n + col0 + sc + 24, n)
        _st_g(c10, o, (row0 + sr + 8) * n + col0 + sc + 0, n)
        _st_g(c11, o, (row0 + sr + 8) * n + col0 + sc + 8, n)
        _st_g(c12, o, (row0 + sr + 8) * n + col0 + sc + 16, n)
        _st_g(c13, o, (row0 + sr + 8) * n + col0 + sc + 24, n)
        _st_g(c20, o, (row0 + sr + 16) * n + col0 + sc + 0, n)
        _st_g(c21, o, (row0 + sr + 16) * n + col0 + sc + 8, n)
        _st_g(c22, o, (row0 + sr + 16) * n + col0 + sc + 16, n)
        _st_g(c23, o, (row0 + sr + 16) * n + col0 + sc + 24, n)
        _st_g(c30, o, (row0 + sr + 24) * n + col0 + sc + 0, n)
        _st_g(c31, o, (row0 + sr + 24) * n + col0 + sc + 8, n)
        _st_g(c32, o, (row0 + sr + 24) * n + col0 + sc + 16, n)
        _st_g(c33, o, (row0 + sr + 24) * n + col0 + sc + 24, n)
        return
    # edge tile: through threadgroup memory, then a bounds-checked copy
    var Os = stack_allocation[
        SG_T * SG_OS, Scalar[DT], address_space=AddressSpace.SHARED
    ]()
    _st_s(c00, Os, (sr + 0) * SG_OS + sc + 0, SG_OS)
    _st_s(c01, Os, (sr + 0) * SG_OS + sc + 8, SG_OS)
    _st_s(c02, Os, (sr + 0) * SG_OS + sc + 16, SG_OS)
    _st_s(c03, Os, (sr + 0) * SG_OS + sc + 24, SG_OS)
    _st_s(c10, Os, (sr + 8) * SG_OS + sc + 0, SG_OS)
    _st_s(c11, Os, (sr + 8) * SG_OS + sc + 8, SG_OS)
    _st_s(c12, Os, (sr + 8) * SG_OS + sc + 16, SG_OS)
    _st_s(c13, Os, (sr + 8) * SG_OS + sc + 24, SG_OS)
    _st_s(c20, Os, (sr + 16) * SG_OS + sc + 0, SG_OS)
    _st_s(c21, Os, (sr + 16) * SG_OS + sc + 8, SG_OS)
    _st_s(c22, Os, (sr + 16) * SG_OS + sc + 16, SG_OS)
    _st_s(c23, Os, (sr + 16) * SG_OS + sc + 24, SG_OS)
    _st_s(c30, Os, (sr + 24) * SG_OS + sc + 0, SG_OS)
    _st_s(c31, Os, (sr + 24) * SG_OS + sc + 8, SG_OS)
    _st_s(c32, Os, (sr + 24) * SG_OS + sc + 16, SG_OS)
    _st_s(c33, Os, (sr + 24) * SG_OS + sc + 24, SG_OS)
    barrier()
    comptime for q in range(SG_T * SG_T // SG_THREADS):
        var e = tid + q * SG_THREADS
        var i = e // SG_T
        var j = e % SG_T
        var gi = row0 + i
        var gj = col0 + j
        if gi < m and gj < n:
            var v = Os[unsafe_offset=i * SG_OS + j]
            comptime if ACC:
                o[unsafe_offset=gi * n + gj] += v
            else:
                o[unsafe_offset=gi * n + gj] = v


def sg_mm[TA: Bool, TB: Bool, ACC: Bool](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT],
    m: Int, n: Int, k: Int,
) raises:
    """`o[m, n] (+)= op(a) @ op(b)` on the simdgroup kernels.

    No edges (M, N multiples of the tile, K of 8): a DIRECT kernel, picked
    by form (bench_tape_gemm.mojo, M1 Pro): op(b) transposed -> 4 x 4 per
    simdgroup (64 x 64 tiles); op(a) transposed -> 2 x 8 (32 x 128);
    plain -> 4 x 8 (64 x 128). Otherwise the STAGED kernel (edge tiles,
    any K)."""
    if k % 8 == 0 and m % 64 == 0 and n % 64 == 0:
        comptime if TB:
            ctx.enqueue_function[sg_direct_kernel_4x4[TA, TB, ACC]](
                o, a, b, Int64(m), Int64(n), Int64(k),
                grid_dim=(n // 64, m // 64), block_dim=(SG_THREADS, 1),
            )
            return
        else:
            # 4 x 8 / 2 x 8 tiles are 64 x 128 / 32 x 128: on a small output
            # they leave too few threadgroups (SAC 256 x 256: 8), 4 x 4 wins.
            if n % 128 == 0 and (TA or m * n >= SG_WIDE_MIN):
                comptime if TA:
                    ctx.enqueue_function[sg_direct_kernel_2x8[TA, TB, ACC]](
                        o, a, b, Int64(m), Int64(n), Int64(k),
                        grid_dim=(n // 128, m // 32), block_dim=(SG_THREADS, 1),
                    )
                else:
                    ctx.enqueue_function[sg_direct_kernel_4x8[TA, TB, ACC]](
                        o, a, b, Int64(m), Int64(n), Int64(k),
                        grid_dim=(n // 128, m // 64), block_dim=(SG_THREADS, 1),
                    )
                return
            ctx.enqueue_function[sg_direct_kernel_4x4[TA, TB, ACC]](
                o, a, b, Int64(m), Int64(n), Int64(k),
                grid_dim=(n // 64, m // 64), block_dim=(SG_THREADS, 1),
            )
            return
    ctx.enqueue_function[sg_mm_kernel[TA, TB, ACC]](
        o, a, b, Int64(m), Int64(n), Int64(k),
        grid_dim=((n + SG_T - 1) // SG_T, (m + SG_T - 1) // SG_T),
        block_dim=(SG_THREADS, 1),
    )
