"""Benchmark CPU matmul: naive triple loop vs `linalg.matmul[target="cpu"]`.

The current `MatMul.eval` (noeira/nn/autodiff/primitives/matmul.mojo) uses a
naive triple-loop on CPU. The GPU path on NVIDIA already routes through
`linalg.matmul` (the optimized GEMM / vendor BLAS). This bench checks whether
the same `linalg.matmul` is also faster on CPU — it accepts `target="cpu"` and
delegates to a BLAS-style implementation.

Run:
    pixi run mojo run -I . benchmarks/benchmark_matmul_cpu.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT as dtype
from noeira.core.fmt import fit


@always_inline
def naive_matmul[
    M: Int, N: Int, K: Int
](
    a: Tile[dtype, M, K],
    b: Tile[dtype, K, N],
    c: Tile[dtype, M, N],
):
    """Body of MatMul.eval (without input caching). c = a @ b."""
    for i in range(M):
        for j in range(N):
            var acc: c.ElementType = 0
            for k in range(K):
                acc += a[i, k] * b[k, j]
            c[i, j] = acc


def bench[M: Int, K: Int, N: Int, WARMUP: Int, ITERS: Int]() raises:
    var label = "[" + String(M) + "x" + String(K) + "] @ [" + String(
        K
    ) + "x" + String(N) + "]"
    print("─" * 72)
    print(label)
    print("─" * 72)

    var a_buf = alloc[Scalar[dtype]](M * K)
    var b_buf = alloc[Scalar[dtype]](K * N)
    var c_naive_buf = alloc[Scalar[dtype]](M * N)
    var c_blas_buf = alloc[Scalar[dtype]](M * N)

    for i in range(M * K):
        a_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    for i in range(K * N):
        b_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    for i in range(M * N):
        c_naive_buf[i] = 0
        c_blas_buf[i] = 0

    var a_lt = tile[M, K](a_buf)
    var b_lt = tile[K, N](b_buf)
    var c_naive_lt = tile[M, N](c_naive_buf)
    var c_blas_lt = tile[M, N](c_blas_buf)

    # ── Warmup ──
    for _ in range(WARMUP):
        max_matmul[target="cpu"](
            c_blas_lt, a_lt, b_lt, None
        )

    # Run naive once (will be reused for verification; only time-bench at the
    # smaller sizes where it isn't impossibly slow).
    naive_matmul[M, N, K](a_lt, b_lt, c_naive_lt)

    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)

    # ── Bench naive (skip for very large sizes — O(MNK) becomes painful) ──
    comptime work = Int64(M) * Int64(N) * Int64(K)
    comptime skip_naive = work > Int64(512) * 512 * 512
    comptime small_naive = work <= Int64(256) * 256 * 256
    var naive_ms: Float64 = 0.0
    var naive_gflops: Float64 = 0.0
    comptime if not skip_naive:
        comptime n_naive = ITERS if small_naive else max(1, ITERS // 4)
        var t0 = perf_counter_ns()
        for _ in range(n_naive):
            naive_matmul[M, N, K](a_lt, b_lt, c_naive_lt)
        var t1 = perf_counter_ns()
        naive_ms = Float64(t1 - t0) / 1e6 / Float64(n_naive)
        naive_gflops = flops / (naive_ms * 1e6)

    # ── Bench linalg.matmul[target="cpu"] ──
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        max_matmul[target="cpu"](
            c_blas_lt, a_lt, b_lt, None
        )
    var t3 = perf_counter_ns()
    var blas_ms = Float64(t3 - t2) / 1e6 / Float64(ITERS)
    var blas_gflops = flops / (blas_ms * 1e6)

    # ── Verify correctness (only if naive was actually run) ──
    var max_abs_diff: Float64 = 0.0
    var max_rel_diff: Float64 = 0.0
    comptime if not skip_naive:
        for i in range(M * N):
            var x = Float64(c_naive_buf[i])
            var y = Float64(c_blas_buf[i])
            var d = abs(x - y)
            if d > max_abs_diff:
                max_abs_diff = d
            var denom = max(abs(x), abs(y))
            if denom > 1e-6:
                var r = d / denom
                if r > max_rel_diff:
                    max_rel_diff = r

    # ── Report ──
    comptime if not skip_naive:
        print(
            "  naive          : "
            + fit(String(naive_ms), 10)
            + " ms   "
            + fit(String(naive_gflops), 7)
            + " GFLOPS"
        )
    else:
        print("  naive          : skipped (too slow)")
    print(
        "  linalg[cpu]    : "
        + fit(String(blas_ms), 10)
        + " ms   "
        + fit(String(blas_gflops), 7)
        + " GFLOPS"
    )
    comptime if not skip_naive:
        var speedup = naive_ms / blas_ms
        print("  speedup        : " + fit(String(speedup), 6) + "x")
        print(
            "  max |diff|     : "
            + fit(String(max_abs_diff), 12)
            + "   max rel diff: "
            + fit(String(max_rel_diff), 12)
        )
    print()

    a_buf.free()
    b_buf.free()
    c_naive_buf.free()
    c_blas_buf.free()


def main() raises:
    seed(42)
    print("=" * 72)
    print(" CPU MatMul: naive triple-loop vs linalg.matmul[target=\"cpu\"]")
    print("=" * 72)
    print()

    # Small RL-layer sizes (closer to what nn primitives actually run)
    bench[64, 128, 64, 5, 200]()
    bench[64, 256, 128, 5, 100]()
    bench[64, 512, 256, 5, 50]()
    bench[256, 256, 256, 3, 50]()
    bench[512, 512, 512, 3, 20]()
    # Bigger sizes (naive auto-skipped past 512^3)
    bench[1024, 1024, 1024, 3, 10]()
    bench[2048, 2048, 2048, 2, 5]()

    print("=" * 72)
    print(" Benchmark complete.")
    print("=" * 72)
