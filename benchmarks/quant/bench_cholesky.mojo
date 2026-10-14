"""The fp64 factorizations GPTQ runs once per linear, at Hessian sizes.

    pixi run mojo run -I . benchmarks/quant/bench_cholesky.mojo [n ...]

Default sizes 2560 (SmolVLA's `down_proj` input), 4096 and 12288 (Kev-9B's
`down_proj`, where the reference's MLX CPU Cholesky made a run take ~10 h).
Per size, on a GPTQ-like Hessian H = Xᵀ X / T + 1 % of its mean diagonal
(X is T = n / 2 rows with per-column scales spread over 30x, so H is rank
deficient before the damping, as a calibration Hessian is):

    gemm      `gemm_acc` fp64 throughput, from building H
    chol      `cholesky_f64`                     n³/6 multiply-adds
    inv       `tri_inv_lower_f64`                n³/6
    hinv      `hinv_upper_f64` = GPTQ's factor (two reversals + the two)
    lapack    Accelerate's dpotrf_ + dtrtri_ on the same H (macOS)

and the gate's residual ‖L Lᵀ − H‖_F / ‖H‖_F, which costs a full n³ GEMM
(skipped above 8192 unless BENCH_CHOL_FULL_RESIDUAL=1; a probe residual
‖L Lᵀ v − H v‖ / (‖H‖_F ‖v‖) over 4 random v replaces it).
"""

from std.math import sqrt
from std.os import getenv
from std.random import random_float64, seed
from std.sys import CompilationTarget, argv
from std.time import perf_counter_ns

from noeira.nn.quant.dense import (
    gemm_acc,
    cholesky_f64,
    tri_inv_lower_f64,
    hinv_upper_f64,
)
from tests.nn.quant.lapack_ref import lapack_potrf_lower, lapack_trtri_lower


def _p(mut v: List[Float64]) -> Pointer[Float64, MutAnyOrigin]:
    return rebind[Pointer[Float64, MutAnyOrigin]](v.unsafe_ptr())


def _s(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def hessian(n: Int) -> List[Float64]:
    var t = n // 2
    var xt = List[Float64](length=n * t, fill=0)  # [n, t] = Xᵀ
    for i in range(n):
        var scale = 1.0 + 29.0 * random_float64()
        for r in range(t):
            xt[i * t + r] = scale * (random_float64() - 0.5)
    var x = List[Float64](length=t * n, fill=0)
    for i in range(n):
        for r in range(t):
            x[r * n + i] = xt[i * t + r]
    var h = List[Float64](length=n * n, fill=0)
    var t0 = perf_counter_ns()
    gemm_acc[DType.float64](
        n, n, t, 1.0 / Float64(t), _p(xt), t, _p(x), n, _p(h), n
    )
    var sec = _s(t0)
    print("  gemm      ", sec, "s  (", Float64(2 * n) * Float64(n) * Float64(t) / sec / 1e9,
          "GFLOP/s fp64)")
    var mean = Float64(0)
    for i in range(n):
        mean += h[i * n + i]
    mean /= Float64(n)
    for i in range(n):
        h[i * n + i] += 0.01 * mean
    return h^


def frob(ref h: List[Float64]) -> Float64:
    var s = Float64(0)
    for v in h:
        s += v * v
    return sqrt(s)


def full_residual(mut l: List[Float64], ref h: List[Float64], n: Int) -> Float64:
    var lt = List[Float64](length=n * n, fill=0)
    for i in range(n):
        for j in range(i + 1):
            lt[j * n + i] = l[i * n + j]
    var c = List[Float64](length=n * n, fill=0)
    for i in range(n * n):
        c[i] = -h[i]
    gemm_acc[DType.float64](n, n, n, 1.0, _p(l), n, _p(lt), n, _p(c), n)
    return frob(c) / frob(h)


def probe_residual(ref l: List[Float64], ref h: List[Float64], n: Int) -> Float64:
    var worst = Float64(0)
    var hf = frob(h)
    for _ in range(4):
        var v = List[Float64](capacity=n)
        var vn = Float64(0)
        for _ in range(n):
            var r = random_float64() - 0.5
            v.append(r)
            vn += r * r
        var w = List[Float64](length=n, fill=0)  # Lᵀ v
        for i in range(n):
            for j in range(i + 1):
                w[j] += l[i * n + j] * v[i]
        var e = Float64(0)
        for i in range(n):
            var a = Float64(0)
            var b = Float64(0)
            for j in range(i + 1):
                a += l[i * n + j] * w[j]
            for j in range(n):
                b += h[i * n + j] * v[j]
            e += (a - b) * (a - b)
        worst = max(worst, sqrt(e) / (hf * sqrt(vn)))
    return worst


def main() raises:
    seed(3)
    var sizes = List[Int]()
    var args = argv()
    for i in range(1, len(args)):
        sizes.append(Int(String(args[i])))
    if len(sizes) == 0:
        sizes = [2560, 4096, 12288]
    var full_always = getenv("BENCH_CHOL_FULL_RESIDUAL") == "1"
    for n in sizes:
        print("n =", n, " (", Float64(n * n * 8) / 1e9, "GB per copy)")
        var h = hessian(n)
        var l = h.copy()
        var t0 = perf_counter_ns()
        cholesky_f64(l, n)
        var t_chol = _s(t0)
        t0 = perf_counter_ns()
        var x = tri_inv_lower_f64(l, n)
        var t_inv = _s(t0)
        _ = x^
        var cubic = Float64(n) * Float64(n) * Float64(n) / 6.0
        print("  chol      ", t_chol, "s  (", 2 * cubic / t_chol / 1e9, "GFLOP/s)")
        print("  inv       ", t_inv, "s  (", 2 * cubic / t_inv / 1e9, "GFLOP/s)")
        if n <= 8192 or full_always:
            t0 = perf_counter_ns()
            var r = full_residual(l, h, n)
            print("  residual   ||L L^T - H||_F / ||H||_F =", r, " (", _s(t0), "s)")
        else:
            print("  residual   probe ||L L^T v - H v|| / (||H||_F ||v||) =",
                  probe_residual(l, h, n))
        _ = l^
        t0 = perf_counter_ns()
        var u = hinv_upper_f64(h.copy(), n)
        print("  hinv      ", _s(t0), "s  (GPTQ's factor, end to end)")
        _ = u^
        comptime if CompilationTarget.is_macos():
            t0 = perf_counter_ns()
            var ll = lapack_potrf_lower(h, n)
            var t_p = _s(t0)
            t0 = perf_counter_ns()
            var xl = lapack_trtri_lower(ll, n)
            var t_t = _s(t0)
            print("  lapack     dpotrf_", t_p, "s  dtrtri_", t_t, "s  (Accelerate)")
            _ = xl^
