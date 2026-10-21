"""Dense host linear algebra for GPTQ: a threaded GEMM, a blocked fp64
Cholesky, the inverse of a lower-triangular matrix, and the upper Cholesky
factor of H⁻¹ that GPTQ's column loop reads.

Everything is row-major on the host. These run off the hot path, once per
quantized linear, on matrices up to Kev's 12288 × 12288 `down_proj` Hessian;
`benchmarks/quant/bench_cholesky.mojo` times them and
`tests/nn/quant/test_cholesky.mojo` gates them against LAPACK.

## The GEMM

`gemm_acc[dt]`: c[M, N] += alpha · a[M, K] @ b[K, N], threaded over 64 × 256
tiles of c, each tile a 4 × 8 register micro-kernel over K in chunks of 256.
It is generic in the dtype: fp64 for the factorizations, fp32 for GPTQ's lazy
block update. `lower = True` skips the tiles entirely above c's diagonal (the
Cholesky's trailing update only needs its lower triangle).

## The factor GPTQ reads

GPTQ wants U, upper triangular, with Uᵀ U = H⁻¹ (`torch.linalg.cholesky(
cholesky_inverse(chol(H)), upper=True)` in the reference: three cubic steps).
With J the exchange matrix (J reverses the indices), J H J = L' L'ᵀ gives

    H⁻¹ = J L'⁻ᵀ L'⁻¹ J = (J L'⁻¹ J)ᵀ (J L'⁻¹ J),

and J L'⁻¹ J is upper triangular with a positive diagonal, so it IS U by the
uniqueness of the Cholesky factor. Two cubic steps of n³/6 multiply-adds
each instead of three, and J M J on a row-major n × n array is reversing the
flat array, in place.
"""

from std.math import sqrt
from std.sys import simd_width_of
from max.algorithm import parallelize


comptime _MR = 4
comptime _NR = 8
comptime _MB = 64
comptime _NB = 256
comptime _KC = 256
# Below this many multiply-adds a GEMM runs on the calling thread:
# `parallelize` costs a fixed ~200 us (nn/primitives/conv2d.mojo).
comptime _SERIAL_FMAS = 1 << 21
comptime CHOL_BLOCK = 128


@always_inline
def _ceildiv(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


@always_inline
def _tile[
    dt: DType
](
    i0: Int,
    i1: Int,
    j0: Int,
    j1: Int,
    K: Int,
    alpha: Scalar[dt],
    a: Pointer[Scalar[dt], MutAnyOrigin],
    lda: Int,
    b: Pointer[Scalar[dt], MutAnyOrigin],
    ldb: Int,
    c: Pointer[Scalar[dt], MutAnyOrigin],
    ldc: Int,
):
    """c[i0:i1, j0:j1] += alpha · a[i0:i1, :K] @ b[:K, j0:j1]."""
    for k0 in range(0, K, _KC):
        var k1 = min(k0 + _KC, K)
        var i = i0
        while i + _MR <= i1:
            var j = j0
            while j + _NR <= j1:
                var c0 = SIMD[dt, _NR](0)
                var c1 = SIMD[dt, _NR](0)
                var c2 = SIMD[dt, _NR](0)
                var c3 = SIMD[dt, _NR](0)
                for k in range(k0, k1):
                    var bv = b.unsafe_load[width=_NR](k * ldb + j)
                    c0 += a[unsafe_offset = i * lda + k] * bv
                    c1 += a[unsafe_offset = (i + 1) * lda + k] * bv
                    c2 += a[unsafe_offset = (i + 2) * lda + k] * bv
                    c3 += a[unsafe_offset = (i + 3) * lda + k] * bv
                var o = i * ldc + j
                c.unsafe_store(o, c.unsafe_load[width=_NR](o) + alpha * c0)
                o += ldc
                c.unsafe_store(o, c.unsafe_load[width=_NR](o) + alpha * c1)
                o += ldc
                c.unsafe_store(o, c.unsafe_load[width=_NR](o) + alpha * c2)
                o += ldc
                c.unsafe_store(o, c.unsafe_load[width=_NR](o) + alpha * c3)
                j += _NR
            for jj in range(j, j1):
                for r in range(_MR):
                    var acc = Scalar[dt](0)
                    for k in range(k0, k1):
                        acc += (
                            a[unsafe_offset = (i + r) * lda + k]
                            * b[unsafe_offset = k * ldb + jj]
                        )
                    c[unsafe_offset = (i + r) * ldc + jj] += alpha * acc
            i += _MR
        for ii in range(i, i1):
            var j = j0
            while j + _NR <= j1:
                var acc = SIMD[dt, _NR](0)
                for k in range(k0, k1):
                    acc += a[unsafe_offset = ii * lda + k] * b.unsafe_load[
                        width=_NR
                    ](k * ldb + j)
                var o = ii * ldc + j
                c.unsafe_store(o, c.unsafe_load[width=_NR](o) + alpha * acc)
                j += _NR
            for jj in range(j, j1):
                var acc = Scalar[dt](0)
                for k in range(k0, k1):
                    acc += a[unsafe_offset = ii * lda + k] * b[
                        unsafe_offset = k * ldb + jj
                    ]
                c[unsafe_offset = ii * ldc + jj] += alpha * acc


def gemm_acc_serial[
    dt: DType
](
    M: Int,
    N: Int,
    K: Int,
    alpha: Scalar[dt],
    a: Pointer[Scalar[dt], MutAnyOrigin],
    lda: Int,
    b: Pointer[Scalar[dt], MutAnyOrigin],
    ldb: Int,
    c: Pointer[Scalar[dt], MutAnyOrigin],
    ldc: Int,
):
    """`gemm_acc` on the calling thread, for callers that are themselves
    one task of a `parallelize`."""
    for j0 in range(0, N, _NB):
        _tile[dt](0, M, j0, min(j0 + _NB, N), K, alpha, a, lda, b, ldb, c, ldc)


def gemm_acc[
    dt: DType
](
    M: Int,
    N: Int,
    K: Int,
    alpha: Scalar[dt],
    a: Pointer[Scalar[dt], MutAnyOrigin],
    lda: Int,
    b: Pointer[Scalar[dt], MutAnyOrigin],
    ldb: Int,
    c: Pointer[Scalar[dt], MutAnyOrigin],
    ldc: Int,
    lower: Bool = False,
):
    """`c[M, N] += alpha · a[M, K] @ b[K, N]`, row-major with row strides
    `lda`, `ldb`, `ldc`. With `lower`, the 64 × 256 tiles of c lying wholly
    above its diagonal (every col > every row) are skipped; the others are
    computed whole, so c's upper triangle near the diagonal is written too."""
    if M <= 0 or N <= 0 or K <= 0:
        return
    var mt = _ceildiv(M, _MB)
    var nt = _ceildiv(N, _NB)

    def _one(t: Int) {imm}:
        var i0 = (t // nt) * _MB
        var j0 = (t % nt) * _NB
        var i1 = min(i0 + _MB, M)
        if lower and j0 > i1 - 1:
            return
        _tile[dt](i0, i1, j0, min(j0 + _NB, N), K, alpha, a, lda, b, ldb, c, ldc)

    if M * N * K < _SERIAL_FMAS:
        for t in range(mt * nt):
            _one(t)
        return
    parallelize(_one, mt * nt)


@always_inline
def _dot(
    x: Pointer[Float64, MutAnyOrigin],
    y: Pointer[Float64, MutAnyOrigin],
    n: Int,
) -> Float64:
    comptime W = simd_width_of[DType.float64]() * 4
    var acc = SIMD[DType.float64, W](0)
    var i = 0
    while i + W <= n:
        acc += x.unsafe_load[width=W](i) * y.unsafe_load[width=W](i)
        i += W
    var s = acc.reduce_add()
    while i < n:
        s += x[unsafe_offset=i] * y[unsafe_offset=i]
        i += 1
    return s


def _reverse(mut v: List[Float64]):
    var n = len(v)
    for i in range(n // 2):
        var t = v[i]
        v[i] = v[n - 1 - i]
        v[n - 1 - i] = t


def cholesky_f64(mut a: List[Float64], n: Int) raises:
    """In place: `a` (n × n row-major, symmetric positive definite; only its
    lower triangle is read) becomes L, lower triangular, with L Lᵀ = a; its
    upper triangle is set to 0. Raises at the first non-positive pivot.

    Right-looking and blocked by 128: factor the diagonal block, solve the
    panel below it row by row (threaded), then subtract the panel's outer
    product from the trailing matrix with `gemm_acc` (threaded; lower tiles
    only)."""
    if len(a) < n * n:
        raise Error("cholesky_f64: the matrix holds fewer than n * n values")
    var p = rebind[Pointer[Float64, MutAnyOrigin]](a.unsafe_ptr())
    var pt = List[Float64](length=CHOL_BLOCK * n, fill=0)
    var tp = rebind[Pointer[Float64, MutAnyOrigin]](pt.unsafe_ptr())
    for k0 in range(0, n, CHOL_BLOCK):
        var k1 = min(k0 + CHOL_BLOCK, n)
        var bs = k1 - k0
        for j in range(k0, k1):
            var rj = p.unsafe_offset(j * n + k0)
            var s = p[unsafe_offset = j * n + j] - _dot(rj, rj, j - k0)
            if not (s > 0):
                raise Error(
                    "cholesky_f64: not positive definite (pivot "
                    + String(j)
                    + " is "
                    + String(s)
                    + ")"
                )
            var d = sqrt(s)
            p[unsafe_offset = j * n + j] = d
            for i in range(j + 1, k1):
                var o = i * n + j
                p[unsafe_offset=o] = (
                    p[unsafe_offset=o] - _dot(p.unsafe_offset(i * n + k0), rj, j - k0)
                ) / d
        if k1 == n:
            break
        var m = n - k1

        def _rows(t: Int) {imm}:
            var r1 = min(k1 + (t + 1) * 64, n)
            for i in range(k1 + t * 64, r1):
                var ri = p.unsafe_offset(i * n + k0)
                for j in range(k0, k1):
                    var o = i * n + j
                    p[unsafe_offset=o] = (
                        p[unsafe_offset=o] - _dot(ri, p.unsafe_offset(j * n + k0), j - k0)
                    ) / p[unsafe_offset = j * n + j]
                    # `ri` sees the new value: column j of row i is now L's.

        if m * bs * bs < _SERIAL_FMAS:
            for t in range(_ceildiv(m, 64)):
                _rows(t)
        else:
            parallelize(_rows, _ceildiv(m, 64))
        # The panel transposed, [bs, m], so the trailing update is an NN GEMM.
        for i in range(k1, n):
            for j in range(k0, k1):
                tp[unsafe_offset = (j - k0) * m + (i - k1)] = p[
                    unsafe_offset = i * n + j
                ]
        gemm_acc[DType.float64](
            m,
            m,
            bs,
            -1.0,
            p.unsafe_offset(k1 * n + k0),
            n,
            tp,
            m,
            p.unsafe_offset(k1 * n + k1),
            n,
            lower=True,
        )
    for i in range(n):
        for j in range(i + 1, n):
            p[unsafe_offset = i * n + j] = 0


def tri_inv_lower_f64(ref l: List[Float64], n: Int) -> List[Float64]:
    """X = L⁻¹ for L lower triangular (n × n row-major, nonzero diagonal;
    its upper triangle is not read). X is lower triangular.

    Blocked forward substitution of L X = I by 128 rows: block row k is
    solved against L's diagonal block (threaded over column chunks), then
    subtracted from every row below with `gemm_acc`. n³/6 multiply-adds."""
    var x = List[Float64](length=n * n, fill=0)
    for i in range(n):
        x[i * n + i] = 1.0
    var lp = rebind[Pointer[Float64, MutAnyOrigin]](l.unsafe_ptr())
    var xp = rebind[Pointer[Float64, MutAnyOrigin]](x.unsafe_ptr())
    comptime CW = 256
    comptime W = simd_width_of[DType.float64]() * 2
    for k0 in range(0, n, CHOL_BLOCK):
        var k1 = min(k0 + CHOL_BLOCK, n)
        var bs = k1 - k0

        def _cols(t: Int) {imm}:
            var c0 = t * CW
            var c1 = min(c0 + CW, k1)
            for i in range(k0, k1):
                var ri = i * n
                for q in range(k0, i):
                    var f = lp[unsafe_offset = i * n + q]
                    if f == 0:
                        continue
                    var rq = q * n
                    var j = c0
                    while j + W <= c1:
                        xp.unsafe_store(
                            ri + j,
                            xp.unsafe_load[width=W](ri + j)
                            - f * xp.unsafe_load[width=W](rq + j),
                        )
                        j += W
                    while j < c1:
                        xp[unsafe_offset = ri + j] -= f * xp[unsafe_offset = rq + j]
                        j += 1
                var d = lp[unsafe_offset = i * n + i]
                for j in range(c0, c1):
                    xp[unsafe_offset = ri + j] = xp[unsafe_offset = ri + j] / d

        if k1 * bs * bs < _SERIAL_FMAS:
            for t in range(_ceildiv(k1, CW)):
                _cols(t)
        else:
            parallelize(_cols, _ceildiv(k1, CW))
        if k1 < n:
            gemm_acc[DType.float64](
                n - k1,
                k1,
                bs,
                -1.0,
                lp.unsafe_offset(k1 * n + k0),
                n,
                xp.unsafe_offset(k0 * n),
                n,
                xp.unsafe_offset(k1 * n),
                n,
            )
    return x^


def hinv_upper_f64(var h: List[Float64], n: Int) raises -> List[Float64]:
    """U, upper triangular with Uᵀ U = h⁻¹, for h symmetric positive
    definite (n × n row-major). GPTQ's inverse-Hessian factor; the module
    docstring derives the two-step route (Cholesky of J h J, invert, J · J)."""
    _reverse(h)
    cholesky_f64(h, n)
    var u = tri_inv_lower_f64(h, n)
    _reverse(u)
    return u^
