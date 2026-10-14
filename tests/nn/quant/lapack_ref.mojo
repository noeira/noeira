"""Accelerate's LAPACK (macOS) as the reference for `noeira/nn/quant/dense.mojo`:
`dpotrf_` (Cholesky) and `dtrtri_` (triangular inverse), for
`tests/nn/quant/test_cholesky.mojo` and `benchmarks/quant/bench_cholesky.mojo`.

LAPACK is column-major. A row-major symmetric matrix read column-major is the
same matrix, and LAPACK's upper factor U (Uᵀ U = H) read back row-major is
our lower L (L Lᵀ = H): so 'U' below, on our row-major arrays, means L.
"""

from std.ffi import OwnedDLHandle
from std.sys import CompilationTarget

comptime _ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"


def _p[T: DType](mut v: List[Scalar[T]]) -> Pointer[Scalar[T], MutAnyOrigin]:
    return rebind[Pointer[Scalar[T], MutAnyOrigin]](v.unsafe_ptr())


def lapack_potrf_lower(ref h: List[Float64], n: Int) raises -> List[Float64]:
    """L (row-major, lower, upper triangle zeroed) with L Lᵀ = h, by dpotrf_."""
    comptime if CompilationTarget.is_macos():
        var a = h.copy()
        var uplo: List[Int8] = [Int8(ord("U"))]
        var nn: List[Int32] = [Int32(n)]
        var info: List[Int32] = [Int32(0)]
        var lib = OwnedDLHandle(_ACCELERATE)
        # `get_function`'s parameter is the symbol's RETURN type.
        var f = lib.get_function[Int32]("dpotrf_")
        _ = f(_p(uplo), _p(nn), _p(a), _p(nn), _p(info))
        if info[0] != 0:
            raise Error("dpotrf_ info " + String(info[0]))
        for i in range(n):
            for j in range(i + 1, n):
                a[i * n + j] = 0
        return a^
    else:
        raise Error("lapack_potrf_lower: Accelerate is macOS only")


def lapack_trtri_lower(ref l: List[Float64], n: Int) raises -> List[Float64]:
    """L⁻¹ (row-major, lower) by dtrtri_."""
    comptime if CompilationTarget.is_macos():
        var a = l.copy()
        var uplo: List[Int8] = [Int8(ord("U"))]
        var diag: List[Int8] = [Int8(ord("N"))]
        var nn: List[Int32] = [Int32(n)]
        var info: List[Int32] = [Int32(0)]
        var lib = OwnedDLHandle(_ACCELERATE)
        var f = lib.get_function[Int32]("dtrtri_")
        _ = f(_p(uplo), _p(diag), _p(nn), _p(a), _p(nn), _p(info))
        if info[0] != 0:
            raise Error("dtrtri_ info " + String(info[0]))
        return a^
    else:
        raise Error("lapack_trtri_lower: Accelerate is macOS only")
