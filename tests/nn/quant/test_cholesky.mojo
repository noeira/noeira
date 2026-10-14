"""Gates for `noeira/nn/quant/dense.mojo`: the threaded GEMM, the blocked fp64
Cholesky, the triangular inverse and GPTQ's inverse-Hessian factor.

    [1] gemm_acc against a triple loop (fp64 and fp32, odd shapes, strides,
        alpha, and `lower`'s tiles)
    [2] cholesky_f64: ‖L Lᵀ − H‖_F / ‖H‖_F < 1e-12 across the 128 block edges
    [3] tri_inv_lower_f64: ‖L X − I‖_max
    [4] hinv_upper_f64: U upper with a positive diagonal, ‖Uᵀ U H − I‖_max
    [5] macOS: L and L⁻¹ against Accelerate's LAPACK (dpotrf_, dtrtri_)
    [6] a matrix that is not positive definite raises

    pixi run mojo run -I . tests/nn/quant/test_cholesky.mojo
"""

from std.math import sqrt, abs
from std.random import random_float64, seed
from std.sys import CompilationTarget
from std.testing import assert_true

from noeira.nn.quant.dense import (
    gemm_acc,
    cholesky_f64,
    tri_inv_lower_f64,
    hinv_upper_f64,
)
from tests.nn.quant.lapack_ref import lapack_potrf_lower, lapack_trtri_lower


def _ptr[dt: DType](mut v: List[Scalar[dt]]) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return rebind[Pointer[Scalar[dt], MutAnyOrigin]](v.unsafe_ptr())


def _rand[dt: DType](n: Int) -> List[Scalar[dt]]:
    var v = List[Scalar[dt]](capacity=n)
    for _ in range(n):
        v.append(Scalar[dt](random_float64() - 0.5))
    return v^


def spd(n: Int, ridge: Float64) -> List[Float64]:
    """G Gᵀ / n + ridge · I, G an n × n uniform matrix: a full-rank but
    poorly conditioned Gram matrix, as a Hessian is."""
    var g = _rand[DType.float64](n * n)
    var gt = List[Float64](length=n * n, fill=0)
    for i in range(n):
        for j in range(n):
            gt[j * n + i] = g[i * n + j]
    var h = List[Float64](length=n * n, fill=0)
    gemm_acc[DType.float64](
        n, n, n, 1.0 / Float64(n), _ptr(g), n, _ptr(gt), n, _ptr(h), n
    )
    for i in range(n):
        h[i * n + i] += ridge
    return h^


def _gemm_case[dt: DType](M: Int, N: Int, K: Int, lower: Bool) raises -> Float64:
    var lda = K + 3
    var ldb = N + 5
    var ldc = N + 1
    var a = _rand[dt](M * lda)
    var b = _rand[dt](K * ldb)
    var c = _rand[dt](M * ldc)
    var ref_ = List[Float64](capacity=M * ldc)
    for i in range(M * ldc):
        ref_.append(Float64(c[i]))
    var alpha = Scalar[dt](-0.75)
    for i in range(M):
        for j in range(N):
            var acc = Float64(0)
            for k in range(K):
                acc += Float64(a[i * lda + k]) * Float64(b[k * ldb + j])
            ref_[i * ldc + j] += Float64(alpha) * acc
    gemm_acc[dt](M, N, K, alpha, _ptr(a), lda, _ptr(b), ldb, _ptr(c), ldc, lower)
    var err = Float64(0)
    for i in range(M):
        for j in range(ldc):
            if lower and j > i:
                continue
            err = max(err, abs(Float64(c[i * ldc + j]) - ref_[i * ldc + j]))
    return err / sqrt(Float64(K))


def _lower_err(ref l: List[Float64], ref h: List[Float64], n: Int) -> Float64:
    """‖L Lᵀ − H‖_F / ‖H‖_F, and checks nothing above the diagonal."""
    var num = Float64(0)
    var den = Float64(0)
    for i in range(n):
        for j in range(i + 1):
            var s = Float64(0)
            for k in range(j + 1):
                s += l[i * n + k] * l[j * n + k]
            var d = s - h[i * n + j]
            var w = 1.0 if i == j else 2.0
            num += w * d * d
            den += w * h[i * n + j] * h[i * n + j]
    return sqrt(num / den)


def _upper_is_zero(ref l: List[Float64], n: Int) -> Bool:
    for i in range(n):
        for j in range(i + 1, n):
            if l[i * n + j] != 0:
                return False
    return True


def _inv_err(ref l: List[Float64], ref x: List[Float64], n: Int) -> Float64:
    """max |L X − I|."""
    var e = Float64(0)
    for i in range(n):
        for j in range(n):
            var s = Float64(0)
            for k in range(n):
                s += l[i * n + k] * x[k * n + j]
            e = max(e, abs(s - (1.0 if i == j else 0.0)))
    return e


def main() raises:
    seed(7)
    print("[1] gemm_acc against a triple loop (max |diff| / sqrt(K))")
    for lower in [False, True]:
        for shape in [(1, 1, 1), (5, 13, 7), (67, 300, 129), (130, 270, 513)]:
            var M, N, K = shape
            var e64 = _gemm_case[DType.float64](M, N, K, lower)
            var e32 = _gemm_case[DType.float32](M, N, K, lower)
            print("    ", M, "x", N, "x", K, "lower" if lower else "full ",
                  " fp64 ", e64, "  fp32 ", e32)
            assert_true(e64 < 1e-14 and e32 < 1e-6, "gemm_acc differs from the triple loop")

    print("[2] cholesky_f64: ||L L^T - H||_F / ||H||_F")
    var sizes = [1, 2, 7, 127, 128, 129, 300, 600]
    for n in sizes:
        var h = spd(n, 1e-3)
        var l = h.copy()
        cholesky_f64(l, n)
        var e = _lower_err(l, h, n)
        print("    n =", n, " residual", e)
        assert_true(e < 1e-12, "the Cholesky residual is above 1e-12")
        assert_true(_upper_is_zero(l, n), "L has values above its diagonal")

    print("[3] tri_inv_lower_f64: max |L X - I|")
    for n in [1, 7, 129, 300]:
        var l = spd(n, 0.1)
        cholesky_f64(l, n)
        var x = tri_inv_lower_f64(l, n)
        var e = _inv_err(l, x, n)
        print("    n =", n, " max |L X - I|", e)
        assert_true(e < 1e-11, "L X is not the identity")
        assert_true(_upper_is_zero(x, n), "L^-1 has values above its diagonal")

    print("[4] hinv_upper_f64: U upper, diag > 0, max |U^T U H - I|")
    for n in [1, 7, 129, 300]:
        var h = spd(n, 0.1)
        var u = hinv_upper_f64(h.copy(), n)
        var ok = True
        for i in range(n):
            ok = ok and u[i * n + i] > 0
            for j in range(i):
                ok = ok and u[i * n + j] == 0
        # (U^T U) H
        var e = Float64(0)
        var uu = List[Float64](length=n * n, fill=0)
        for i in range(n):
            for j in range(n):
                var s = Float64(0)
                for k in range(min(i, j) + 1):
                    s += u[k * n + i] * u[k * n + j]
                uu[i * n + j] = s
        for i in range(n):
            for j in range(n):
                var s = Float64(0)
                for k in range(n):
                    s += uu[i * n + k] * h[k * n + j]
                e = max(e, abs(s - (1.0 if i == j else 0.0)))
        print("    n =", n, " max |U^T U H - I|", e)
        assert_true(ok, "U is not upper triangular with a positive diagonal")
        assert_true(e < 1e-10, "U^T U is not H^-1")

    comptime if CompilationTarget.is_macos():
        print("[5] against Accelerate's LAPACK: max |ours - LAPACK| / max |LAPACK|")
        for n in [7, 129, 600]:
            var h = spd(n, 1e-3)
            var l = h.copy()
            cholesky_f64(l, n)
            var ll = lapack_potrf_lower(h, n)
            var x = tri_inv_lower_f64(l, n)
            var xl = lapack_trtri_lower(ll, n)
            var dl = Float64(0)
            var ml = Float64(0)
            var dx = Float64(0)
            var mx = Float64(0)
            for i in range(n * n):
                dl = max(dl, abs(l[i] - ll[i]))
                ml = max(ml, abs(ll[i]))
                dx = max(dx, abs(x[i] - xl[i]))
                mx = max(mx, abs(xl[i]))
            print("    n =", n, " L ", dl / ml, "  L^-1 ", dx / mx)
            assert_true(dl / ml < 1e-12, "L differs from dpotrf_")
            assert_true(dx / mx < 1e-9, "L^-1 differs from dtrtri_")
    else:
        print("[5] skipped: Accelerate is macOS only")

    print("[6] not positive definite raises")
    for kind in range(2):
        var n = 200
        var h = spd(n, 1e-3)
        if kind == 0:
            h[150 * n + 150] = -1.0
        else:
            for i in range(n):
                h[i * n + i] -= 10.0
        var raised = False
        try:
            cholesky_f64(h, n)
        except e:
            raised = True
            print("    case", kind, "raised:", e)
        assert_true(raised, "cholesky_f64 accepted a matrix that is not positive definite")
    print("ALL PASS")
