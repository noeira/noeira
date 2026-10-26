"""GPU vs CPU Analysis for Deep RL.

This benchmark provides a comprehensive analysis of when GPU makes sense for DQN training.

Key findings from previous benchmarks:
1. Kernel launch overhead: 400-700 μs on Apple Silicon
2. Small network forward pass: ~50 μs on CPU, ~4000+ μs on GPU
3. GPU loses for small networks due to:
   - Kernel launch overhead >> computation time
   - Memory bandwidth limits for small networks
   - Poor GPU utilization with small batches

This benchmark explores:
1. Batch size impact on GPU efficiency
2. Network size impact on GPU efficiency
3. Break-even point analysis

Run with:
    pixi run -e apple mojo run -I . examples/benchmark_gpu_analysis.mojo
"""

from std.time import perf_counter_ns
from std.math import sqrt
from std.random import random_float64, seed

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from noeira.core.tiles import Tile, ImmTile, tile
from noeira.core.fmt import fit


# =============================================================================
# CPU Forward Pass (baseline)
# =============================================================================


def cpu_forward_batch[
    OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int, BATCH: Int
](
    obs: List[Float32],
    W1: List[Float32],
    W2: List[Float32],
    mut h: List[Float32],
    mut output: List[Float32],
):
    """CPU batch forward pass."""
    for b in range(BATCH):
        # Hidden layer
        for j in range(HIDDEN_DIM):
            var sum_val: Float32 = 0
            for k in range(OBS_DIM):
                sum_val += obs[b * OBS_DIM + k] * W1[k * HIDDEN_DIM + j]
            h[j] = sum_val if sum_val > 0 else 0

        # Output layer
        for j in range(OUT_DIM):
            var sum_val: Float32 = 0
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * W2[k * OUT_DIM + j]
            output[b * OUT_DIM + j] = sum_val


def benchmark_cpu[
    OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int, BATCH: Int
](num_iters: Int) -> Float64:
    """Benchmark CPU forward pass."""
    var W1 = List[Float32](capacity=OBS_DIM * HIDDEN_DIM)
    var h = List[Float32](capacity=HIDDEN_DIM)
    var W2 = List[Float32](capacity=HIDDEN_DIM * OUT_DIM)
    var obs = List[Float32](capacity=BATCH * OBS_DIM)
    var output = List[Float32](capacity=BATCH * OUT_DIM)

    for i in range(OBS_DIM * HIDDEN_DIM):
        W1.append(Float32((random_float64() - 0.5) * 0.1))
    for i in range(HIDDEN_DIM):
        h.append(Float32(0))
    for i in range(HIDDEN_DIM * OUT_DIM):
        W2.append(Float32((random_float64() - 0.5) * 0.1))
    for i in range(BATCH * OBS_DIM):
        obs.append(Float32(random_float64()))
    for i in range(BATCH * OUT_DIM):
        output.append(Float32(0))

    # Warm up
    for _ in range(100):
        cpu_forward_batch[OBS_DIM, HIDDEN_DIM, OUT_DIM, BATCH](
            obs, W1, W2, h, output
        )

    var start = perf_counter_ns()
    for _ in range(num_iters):
        cpu_forward_batch[OBS_DIM, HIDDEN_DIM, OUT_DIM, BATCH](
            obs, W1, W2, h, output
        )
    var end = perf_counter_ns()

    var dummy: Float32 = 0
    for i in range(BATCH * OUT_DIM):
        dummy += output[i]
    if dummy == Float32.MAX:
        print("dummy")

    return Float64(end - start) / Float64(num_iters)


# =============================================================================
# GPU Forward Pass Kernel
# =============================================================================


def gpu_forward_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    OUT_DIM: Int,
    TPB: Int,
](
    output: Tile[dtype, BATCH * OUT_DIM],
    obs: ImmTile[dtype, BATCH * OBS_DIM],
    W1: ImmTile[dtype, OBS_DIM * HIDDEN_DIM],
    W2: ImmTile[dtype, HIDDEN_DIM * OUT_DIM],
):
    """GPU forward pass - one thread per batch element."""
    var batch_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if batch_idx >= BATCH:
        return

    # Compute forward pass for this batch element
    for j in range(OUT_DIM):
        var out_val: Scalar[dtype] = 0

        for hid in range(HIDDEN_DIM):
            var h_val: Scalar[dtype] = 0
            for k in range(OBS_DIM):
                h_val += rebind[Scalar[dtype]](
                    obs[batch_idx * OBS_DIM + k]
                ) * rebind[Scalar[dtype]](W1[k * HIDDEN_DIM + hid])
            if h_val < Scalar[dtype](0):
                h_val = Scalar[dtype](0)
            out_val += h_val * rebind[Scalar[dtype]](W2[hid * OUT_DIM + j])

        output[batch_idx * OUT_DIM + j] = out_val


def benchmark_gpu[
    OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int, BATCH: Int
](ctx: DeviceContext, num_iters: Int) raises -> Float64:
    """Benchmark GPU forward pass."""
    comptime dtype = DType.float32
    comptime TPB = 64

    var obs_gpu = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    var W1_gpu = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var W2_gpu = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * OUT_DIM)
    var out_gpu = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)

    with obs_gpu.map_to_host() as host:
        for i in range(BATCH * OBS_DIM):
            host[i] = Scalar[dtype](random_float64())
    with W1_gpu.map_to_host() as host:
        for i in range(OBS_DIM * HIDDEN_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with W2_gpu.map_to_host() as host:
        for i in range(HIDDEN_DIM * OUT_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    out_gpu.enqueue_fill(0)
    ctx.synchronize()

    var obs_t = tile[BATCH * OBS_DIM](obs_gpu)
    var W1_t = tile[OBS_DIM * HIDDEN_DIM](W1_gpu)
    var W2_t = tile[HIDDEN_DIM * OUT_DIM](W2_gpu)
    var out_t = tile[BATCH * OUT_DIM](out_gpu)

    comptime num_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = gpu_forward_kernel[
        dtype, BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM, TPB
    ]

    for _ in range(100):
        ctx.enqueue_function[kernel, kernel](
            out_t,
            obs_t,
            W1_t,
            W2_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[kernel, kernel](
            out_t,
            obs_t,
            W1_t,
            W2_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / Float64(num_iters)


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("GPU vs CPU Analysis for Deep RL")
    print("=" * 70)
    print()

    var num_iters = 1000

    # Test configuration: DQN-like network
    comptime OBS_DIM = 8
    comptime OUT_DIM = 4

    print("=" * 70)
    print("ANALYSIS 1: Impact of Hidden Layer Size")
    print("=" * 70)
    print(
        "Fixed: obs_dim="
        + String(OBS_DIM)
        + ", out_dim="
        + String(OUT_DIM)
        + ", batch=64"
    )
    print()
    print("Hidden | CPU (us) | GPU (us) | GPU/CPU ratio")
    print("-" * 50)

    with DeviceContext() as ctx:
        # Test different hidden sizes
        var cpu_h64 = benchmark_cpu[OBS_DIM, 64, OUT_DIM, 64](num_iters)
        var gpu_h64 = benchmark_gpu[OBS_DIM, 64, OUT_DIM, 64](ctx, num_iters)
        print(
            "  64   | "
            + fit(String(cpu_h64 / 1000), 8)
            + " | "
            + fit(String(gpu_h64 / 1000), 8)
            + " | "
            + fit(String(gpu_h64 / cpu_h64), 5)
        )

        var cpu_h128 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 64](num_iters)
        var gpu_h128 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 64](ctx, num_iters)
        print(
            " 128   | "
            + fit(String(cpu_h128 / 1000), 8)
            + " | "
            + fit(String(gpu_h128 / 1000), 8)
            + " | "
            + fit(String(gpu_h128 / cpu_h128), 5)
        )

        var cpu_h256 = benchmark_cpu[OBS_DIM, 256, OUT_DIM, 64](num_iters)
        var gpu_h256 = benchmark_gpu[OBS_DIM, 256, OUT_DIM, 64](ctx, num_iters)
        print(
            " 256   | "
            + fit(String(cpu_h256 / 1000), 8)
            + " | "
            + fit(String(gpu_h256 / 1000), 8)
            + " | "
            + fit(String(gpu_h256 / cpu_h256), 5)
        )

        var cpu_h512 = benchmark_cpu[OBS_DIM, 512, OUT_DIM, 64](num_iters)
        var gpu_h512 = benchmark_gpu[OBS_DIM, 512, OUT_DIM, 64](ctx, num_iters)
        print(
            " 512   | "
            + fit(String(cpu_h512 / 1000), 8)
            + " | "
            + fit(String(gpu_h512 / 1000), 8)
            + " | "
            + fit(String(gpu_h512 / cpu_h512), 5)
        )

        print()
        print("=" * 70)
        print("ANALYSIS 2: Impact of Batch Size")
        print("=" * 70)
        print(
            "Fixed: obs_dim="
            + String(OBS_DIM)
            + ", hidden=128, out_dim="
            + String(OUT_DIM)
        )
        print()
        print("Batch  | CPU (us) | GPU (us) | GPU/CPU ratio | GPU throughput")
        print("-" * 70)

        var cpu_b32 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 32](num_iters)
        var gpu_b32 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 32](ctx, num_iters)
        print(
            "   32  | "
            + fit(String(cpu_b32 / 1000), 8)
            + " | "
            + fit(String(gpu_b32 / 1000), 8)
            + " | "
            + fit(String(gpu_b32 / cpu_b32), 5)
            + " | "
            + String(Int(32 * 1e9 / gpu_b32))
            + " obs/s"
        )

        var cpu_b64 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 64](num_iters)
        var gpu_b64 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 64](ctx, num_iters)
        print(
            "   64  | "
            + fit(String(cpu_b64 / 1000), 8)
            + " | "
            + fit(String(gpu_b64 / 1000), 8)
            + " | "
            + fit(String(gpu_b64 / cpu_b64), 5)
            + " | "
            + String(Int(64 * 1e9 / gpu_b64))
            + " obs/s"
        )

        var cpu_b128 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 128](num_iters)
        var gpu_b128 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 128](ctx, num_iters)
        print(
            "  128  | "
            + fit(String(cpu_b128 / 1000), 8)
            + " | "
            + fit(String(gpu_b128 / 1000), 8)
            + " | "
            + fit(String(gpu_b128 / cpu_b128), 5)
            + " | "
            + String(Int(128 * 1e9 / gpu_b128))
            + " obs/s"
        )

        var cpu_b256 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 256](num_iters)
        var gpu_b256 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 256](ctx, num_iters)
        print(
            "  256  | "
            + fit(String(cpu_b256 / 1000), 8)
            + " | "
            + fit(String(gpu_b256 / 1000), 8)
            + " | "
            + fit(String(gpu_b256 / cpu_b256), 5)
            + " | "
            + String(Int(256 * 1e9 / gpu_b256))
            + " obs/s"
        )

        var cpu_b512 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 512](num_iters)
        var gpu_b512 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 512](ctx, num_iters)
        print(
            "  512  | "
            + fit(String(cpu_b512 / 1000), 8)
            + " | "
            + fit(String(gpu_b512 / 1000), 8)
            + " | "
            + fit(String(gpu_b512 / cpu_b512), 5)
            + " | "
            + String(Int(512 * 1e9 / gpu_b512))
            + " obs/s"
        )

        var cpu_b1024 = benchmark_cpu[OBS_DIM, 128, OUT_DIM, 1024](num_iters)
        var gpu_b1024 = benchmark_gpu[OBS_DIM, 128, OUT_DIM, 1024](
            ctx, num_iters
        )
        print(
            " 1024  | "
            + fit(String(cpu_b1024 / 1000), 8)
            + " | "
            + fit(String(gpu_b1024 / 1000), 8)
            + " | "
            + fit(String(gpu_b1024 / cpu_b1024), 5)
            + " | "
            + String(Int(1024 * 1e9 / gpu_b1024))
            + " obs/s"
        )

        print()
        print("=" * 70)
        print("CONCLUSIONS")
        print("=" * 70)
        print()
        print("For DQN on Apple Silicon (Metal):")
        print()
        print("1. KERNEL LAUNCH OVERHEAD is the main bottleneck (~400-700 us)")
        print(
            "   - This is 10-100x the actual computation time for small"
            " networks"
        )
        print()
        print("2. CPU is faster for typical DQN configurations:")
        print("   - 8 -> 128 -> 4 network with batch 64")
        print("   - GPU would need batch > 1000 or hidden > 512 to break even")
        print()
        print("3. GPU makes sense only when:")
        print("   - Network is large (hidden > 512, or multiple layers > 2)")
        print("   - Batch is large (> 1000 for small networks)")
        print("   - Multiple training steps batched before sync")
        print()
        print("4. For RL action selection (single obs, online):")
        print("   - ALWAYS use CPU - GPU overhead is insurmountable")
        print()
        print("5. Recommended hybrid approach:")
        print("   - Environment stepping: CPU (very fast, ~27M steps/sec)")
        print("   - Action selection: CPU (low latency required)")
        print("   - Experience collection: CPU buffer")
        print("   - Training: GPU only if network is large, else CPU")
        print()
        print("=" * 70)
