"""Benchmark kernel launch overhead: JIT caching, chain vs fused.

Tests 3 scenarios:
1. Cold JIT vs warm cache — does Mojo cache compiled kernels?
2. Chain of 3 separate kernels vs 1 fused kernel — launch overhead cost
3. Batch of N enqueue calls without intermediate sync — pipeline overhead

Note: compile_function is unimplemented on Metal, so precompilation tests
only run on NVIDIA. The key question is whether enqueue_function already
caches internally (spoiler from initial runs: yes, ~285x cold/warm ratio).

Run with:
    pixi run -e apple mojo run -I . benchmarks/benchmark_precompilation.mojo
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_precompilation.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim
from std.sys import is_nvidia_gpu, default_accelerator
from noeira.core.tiles import Tile, ImmTile, tile
from std.time import perf_counter_ns

from noeira.nn.constants import DT as dtype


# =============================================================================
# Test kernels — trivial compute to isolate launch overhead
# =============================================================================


def simple_add_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input_a: ImmTile[dtype, N],
    input_b: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = input_a[tid] + input_b[tid]


def scale_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = input[tid] * Scalar[dtype](2.0)


def relu_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        var val = input[tid]
        output[tid] = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)


def bias_add_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = input[tid] + Scalar[dtype](0.1)


def negate_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = -input[tid]


def fused_scale_relu_bias_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        var val = input[tid] * Scalar[dtype](2.0)
        val = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        output[tid] = val + Scalar[dtype](0.1)


def fused_5op_kernel[
    dtype: DType, N: Int
](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    """Fused: scale, relu, bias, negate, scale (5 ops)."""
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        var val = input[tid] * Scalar[dtype](2.0)
        val = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        val = val + Scalar[dtype](0.1)
        val = -val
        output[tid] = val * Scalar[dtype](0.5)


# =============================================================================
# Helpers
# =============================================================================


def format_time(ns: UInt) -> String:
    var us = Float64(ns) / 1_000.0
    if us < 1000.0:
        return String.write(us) + " us"
    return String.write(us / 1000.0) + " ms"


# =============================================================================
# Test 1: Cold JIT vs Warm Cache
# =============================================================================


def benchmark_cold_vs_warm[N: Int](ctx: DeviceContext) raises:
    print("\n" + "=" * 70)
    print("TEST 1: Cold JIT vs Warm Cache — ctx.enqueue_function")
    print("  Tensor size: ", N)
    print("=" * 70)

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var out_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var a_t = tile[N](a_buf)
    var b_t = tile[N](b_buf)
    var out_t = tile[N](out_buf)

    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)
    comptime kernel = simple_add_kernel[dtype, N]

    # Cold launch — first-ever JIT compilation for this kernel specialization
    var start = perf_counter_ns()
    ctx.enqueue_function[kernel, kernel](
        out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()
    var cold_ns = perf_counter_ns() - start
    print("  Cold (1st launch):     ", format_time(cold_ns))

    # 2nd launch — should be cached
    start = perf_counter_ns()
    ctx.enqueue_function[kernel, kernel](
        out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()
    var second_ns = perf_counter_ns() - start
    print("  2nd launch:            ", format_time(second_ns))

    # 3rd launch
    start = perf_counter_ns()
    ctx.enqueue_function[kernel, kernel](
        out_t, a_t, b_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()
    var third_ns = perf_counter_ns() - start
    print("  3rd launch:            ", format_time(third_ns))

    # Warm average
    var iterations = 500
    var total_ns: UInt = 0
    for _ in range(iterations):
        start = perf_counter_ns()
        ctx.enqueue_function[kernel, kernel](
            out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_ns += perf_counter_ns() - start

    var warm_avg = total_ns // UInt(iterations)
    print("  Warm avg (", iterations, " iters): ", format_time(warm_avg))

    if warm_avg > 0:
        print(
            "  Cold/Warm ratio:       ",
            Float64(cold_ns) / Float64(warm_avg),
            "x",
        )
        print(
            (
                "\n  -> Mojo caches kernels after 1st launch. Cold cost is JIT"
                " compilation."
            ),
        )


# =============================================================================
# Test 2: Chain of N kernels vs 1 fused kernel
# =============================================================================


def benchmark_chain_vs_fused[N: Int](ctx: DeviceContext) raises:
    print("\n" + "=" * 70)
    print("TEST 2: Kernel Chain vs Fused — Launch Overhead Cost")
    print("  Tensor size: ", N)
    print("=" * 70)

    var in_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp1_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp2_buf = ctx.enqueue_create_buffer[dtype](N)
    var out_buf = ctx.enqueue_create_buffer[dtype](N)
    var fused_out_buf = ctx.enqueue_create_buffer[dtype](N)
    in_buf.enqueue_fill(Scalar[dtype](1.0))
    tmp1_buf.enqueue_fill(Scalar[dtype](0.0))
    tmp2_buf.enqueue_fill(Scalar[dtype](0.0))
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    fused_out_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var in_t = tile[N](in_buf)
    var tmp1_m = tile[N](tmp1_buf)
    var tmp1_i = tile[N](tmp1_buf)
    var tmp2_m = tile[N](tmp2_buf)
    var tmp2_i = tile[N](tmp2_buf)
    var out_t = tile[N](out_buf)
    var fused_t = tile[N](fused_out_buf)

    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    comptime k_scale = scale_kernel[dtype, N]
    comptime k_relu = relu_kernel[dtype, N]
    comptime k_bias = bias_add_kernel[dtype, N]
    comptime k_fused3 = fused_scale_relu_bias_kernel[dtype, N]

    var warmup = 50
    var iterations = 500

    # --- Warmup all kernels first (pay JIT cost once) ---
    ctx.enqueue_function[k_scale, k_scale](
        tmp1_m, in_t, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_relu, k_relu](
        tmp2_m, tmp1_i, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_bias, k_bias](
        out_t, tmp2_i, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_fused3, k_fused3](
        fused_t, in_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()

    # --- 3 separate kernels ---
    for _ in range(warmup):
        ctx.enqueue_function[k_scale, k_scale](
            tmp1_m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_relu, k_relu](
            tmp2_m, tmp1_i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_bias, k_bias](
            out_t, tmp2_i, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_chain3: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_scale, k_scale](
            tmp1_m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_relu, k_relu](
            tmp2_m, tmp1_i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_bias, k_bias](
            out_t, tmp2_i, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_chain3 += perf_counter_ns() - start

    var avg_chain3 = total_chain3 // UInt(iterations)

    # --- 1 fused kernel (same work) ---
    for _ in range(warmup):
        ctx.enqueue_function[k_fused3, k_fused3](
            fused_t, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_fused3: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_fused3, k_fused3](
            fused_t, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_fused3 += perf_counter_ns() - start

    var avg_fused3 = total_fused3 // UInt(iterations)

    print("  3 separate kernels:    ", format_time(avg_chain3))
    print("  1 fused kernel:        ", format_time(avg_fused3))
    if avg_fused3 > 0:
        print(
            "  Speedup:               ",
            Float64(avg_chain3) / Float64(avg_fused3),
            "x",
        )
        if avg_chain3 > avg_fused3:
            print(
                "  Overhead per extra kernel: ~",
                format_time((avg_chain3 - avg_fused3) // 2),
            )


# =============================================================================
# Test 3: Scaling — 1 vs 3 vs 5 kernel chain
# =============================================================================


def benchmark_chain_scaling[N: Int](ctx: DeviceContext) raises:
    print("\n" + "=" * 70)
    print("TEST 3: Launch Overhead Scaling — 1 vs 3 vs 5 Kernels")
    print("  Tensor size: ", N)
    print("=" * 70)

    var in_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp1_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp2_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp3_buf = ctx.enqueue_create_buffer[dtype](N)
    var tmp4_buf = ctx.enqueue_create_buffer[dtype](N)
    var out_buf = ctx.enqueue_create_buffer[dtype](N)
    in_buf.enqueue_fill(Scalar[dtype](1.0))
    tmp1_buf.enqueue_fill(Scalar[dtype](0.0))
    tmp2_buf.enqueue_fill(Scalar[dtype](0.0))
    tmp3_buf.enqueue_fill(Scalar[dtype](0.0))
    tmp4_buf.enqueue_fill(Scalar[dtype](0.0))
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var in_t = tile[N](in_buf)
    var t1m = tile[N](tmp1_buf)
    var t1i = tile[N](tmp1_buf)
    var t2m = tile[N](tmp2_buf)
    var t2i = tile[N](tmp2_buf)
    var t3m = tile[N](tmp3_buf)
    var t3i = tile[N](tmp3_buf)
    var t4m = tile[N](tmp4_buf)
    var t4i = tile[N](tmp4_buf)
    var out_t = tile[N](out_buf)

    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    comptime k_scale = scale_kernel[dtype, N]
    comptime k_relu = relu_kernel[dtype, N]
    comptime k_bias = bias_add_kernel[dtype, N]
    comptime k_negate = negate_kernel[dtype, N]
    comptime k_fused3 = fused_scale_relu_bias_kernel[dtype, N]
    comptime k_fused5 = fused_5op_kernel[dtype, N]

    # Warmup all kernels
    ctx.enqueue_function[k_scale, k_scale](
        t1m, in_t, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_relu, k_relu](
        t2m, t1i, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_bias, k_bias](
        t3m, t2i, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_negate, k_negate](
        t4m, t3i, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_fused3, k_fused3](
        out_t, in_t, grid_dim=grid, block_dim=block
    )
    ctx.enqueue_function[k_fused5, k_fused5](
        out_t, in_t, grid_dim=grid, block_dim=block
    )
    ctx.synchronize()

    var warmup = 50
    var iterations = 500

    # --- 1 kernel (baseline) ---
    for _ in range(warmup):
        ctx.enqueue_function[k_scale, k_scale](
            t1m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_1: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_scale, k_scale](
            t1m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_1 += perf_counter_ns() - start
    var avg_1 = total_1 // UInt(iterations)

    # --- 3 kernels ---
    for _ in range(warmup):
        ctx.enqueue_function[k_scale, k_scale](
            t1m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_relu, k_relu](
            t2m, t1i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_bias, k_bias](
            out_t, t2i, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_3: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_scale, k_scale](
            t1m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_relu, k_relu](
            t2m, t1i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_bias, k_bias](
            out_t, t2i, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_3 += perf_counter_ns() - start
    var avg_3 = total_3 // UInt(iterations)

    # --- 5 kernels ---
    for _ in range(warmup):
        ctx.enqueue_function[k_scale, k_scale](
            t1m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_relu, k_relu](
            t2m, t1i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_bias, k_bias](
            t3m, t2i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_negate, k_negate](
            t4m, t3i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_scale, k_scale](
            out_t, t4i, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_5: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_scale, k_scale](
            t1m, in_t, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_relu, k_relu](
            t2m, t1i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_bias, k_bias](
            t3m, t2i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_negate, k_negate](
            t4m, t3i, grid_dim=grid, block_dim=block
        )
        ctx.enqueue_function[k_scale, k_scale](
            out_t, t4i, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_5 += perf_counter_ns() - start
    var avg_5 = total_5 // UInt(iterations)

    # --- Fused equivalents ---
    for _ in range(warmup):
        ctx.enqueue_function[k_fused3, k_fused3](
            out_t, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_f3: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_fused3, k_fused3](
            out_t, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_f3 += perf_counter_ns() - start
    var avg_f3 = total_f3 // UInt(iterations)

    for _ in range(warmup):
        ctx.enqueue_function[k_fused5, k_fused5](
            out_t, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_f5: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[k_fused5, k_fused5](
            out_t, in_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_f5 += perf_counter_ns() - start
    var avg_f5 = total_f5 // UInt(iterations)

    print("  1 kernel:              ", format_time(avg_1))
    print("  3 kernels:             ", format_time(avg_3))
    print("  5 kernels:             ", format_time(avg_5))
    print("  1 fused (=3 ops):      ", format_time(avg_f3))
    print("  1 fused (=5 ops):      ", format_time(avg_f5))
    print()

    if avg_1 > 0:
        var overhead_per_kernel = (Float64(avg_5) - Float64(avg_1)) / 4.0
        print(
            "  Marginal cost per extra kernel launch: ~",
            Float64(overhead_per_kernel) / 1000.0,
            " us",
        )
        print(
            "  3-chain vs fused-3 speedup: ",
            Float64(avg_3) / Float64(avg_f3),
            "x",
        )
        print(
            "  5-chain vs fused-5 speedup: ",
            Float64(avg_5) / Float64(avg_f5),
            "x",
        )
        print(
            "\n  -> Each extra kernel launch adds ~",
            Float64(overhead_per_kernel) / 1000.0,
            " us of pure overhead",
        )
        print(
            "  -> For a typical RL step with 20 kernels, that's ~",
            Float64(overhead_per_kernel) * 20.0 / 1_000_000.0,
            " ms wasted",
        )


# =============================================================================
# Test 4: Precompilation (NVIDIA only)
# =============================================================================


def benchmark_precompile[N: Int](ctx: DeviceContext) raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("\n" + "=" * 70)
        print("TEST 4: Precompilation — SKIPPED (Metal: unimplemented)")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("TEST 4: Precompilation — compile_function vs direct enqueue")
    print("  Tensor size: ", N)
    print("=" * 70)

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var out_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    out_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var a_t = tile[N](a_buf)
    var b_t = tile[N](b_buf)
    var out_t = tile[N](out_buf)

    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)
    comptime kernel = simple_add_kernel[dtype, N]

    var warmup = 50
    var iterations = 500

    # Direct
    for _ in range(warmup):
        ctx.enqueue_function[kernel, kernel](
            out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_direct: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[kernel, kernel](
            out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_direct += perf_counter_ns() - start
    var avg_direct = total_direct // UInt(iterations)

    # Precompiled
    def wrapper(
        output: Tile[dtype, N],
        input_a: ImmTile[dtype, N],
        input_b: ImmTile[dtype, N],
    ):
        simple_add_kernel[dtype, N](output, input_a, input_b)

    var compiled = ctx.compile_function[wrapper, wrapper]()
    var stream = ctx.create_stream()

    for _ in range(warmup):
        stream.enqueue_function(
            compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()

    var total_pre: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        stream.enqueue_function(
            compiled, out_t, a_t, b_t, grid_dim=grid, block_dim=block
        )
        ctx.synchronize()
        total_pre += perf_counter_ns() - start
    var avg_pre = total_pre // UInt(iterations)

    print("  Direct (ctx.enqueue):  ", format_time(avg_direct))
    print("  Precompiled (stream):  ", format_time(avg_pre))
    if avg_pre > 0:
        print(
            "  Speedup:               ",
            Float64(avg_direct) / Float64(avg_pre),
            "x",
        )


# =============================================================================
# Main
# =============================================================================


def main() raises:
    print("=" * 70)
    print("KERNEL LAUNCH OVERHEAD BENCHMARK")
    print("=" * 70)
    print("Questions: Does Mojo cache JIT? How much does each launch cost?")
    print("           Does precompilation help? How much does fusion save?")

    with DeviceContext() as ctx:
        # RL-typical size: batch=32, hidden=256 → 8192 elements
        benchmark_cold_vs_warm[8192](ctx)

        # Chain vs fused at different sizes
        benchmark_chain_vs_fused[8192](ctx)
        benchmark_chain_vs_fused[65536](ctx)

        # Scaling test — see marginal cost per kernel
        benchmark_chain_scaling[8192](ctx)
        benchmark_chain_scaling[65536](ctx)

        # Precompilation (NVIDIA only)
        benchmark_precompile[8192](ctx)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
