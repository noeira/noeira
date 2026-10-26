"""Benchmark: CPU vs GPU neural network forward pass.

This isolates the neural network forward pass to identify where GPU overhead comes from.

Tests:
1. CPU forward pass (single obs, like action selection)
2. CPU batched forward pass
3. GPU batched forward pass
4. Kernel launch overhead

Run with:
    pixi run -e apple mojo run -I . examples/benchmark_forward_pass.mojo
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
# CPU Forward Pass
# =============================================================================


struct CPUNetwork[obs_dim: Int, hidden_dim: Int, out_dim: Int]:
    """Simple CPU neural network for benchmarking."""

    comptime W1_SIZE = Self.obs_dim * Self.hidden_dim
    comptime W2_SIZE = Self.hidden_dim * Self.out_dim

    var W1: Array[Float32, Self.W1_SIZE]
    var b1: Array[Float32, Self.hidden_dim]
    var W2: Array[Float32, Self.W2_SIZE]
    var b2: Array[Float32, Self.out_dim]

    def __init__(out self):
        self.W1 = Array[Float32, Self.W1_SIZE](fill=0)
        self.b1 = Array[Float32, Self.hidden_dim](fill=0)
        self.W2 = Array[Float32, Self.W2_SIZE](fill=0)
        self.b2 = Array[Float32, Self.out_dim](fill=0)

        # Xavier init
        var std1 = sqrt(2.0 / Float64(Self.obs_dim + Self.hidden_dim))
        var std2 = sqrt(2.0 / Float64(Self.hidden_dim + Self.out_dim))

        for i in range(Self.W1_SIZE):
            self.W1[i] = Float32((random_float64() - 0.5) * 2 * std1)
        for i in range(Self.hidden_dim):
            self.b1[i] = 0
        for i in range(Self.W2_SIZE):
            self.W2[i] = Float32((random_float64() - 0.5) * 2 * std2)
        for i in range(Self.out_dim):
            self.b2[i] = 0

    def forward_single(
        self,
        obs: Array[Float32, Self.obs_dim],
    ) -> Array[Float32, Self.out_dim]:
        """Single observation forward pass."""
        # Layer 1: ReLU(obs @ W1 + b1)
        var h = Array[Float32, Self.hidden_dim](fill=0)
        for j in range(Self.hidden_dim):
            var sum_val: Float32 = self.b1[j]
            for k in range(Self.obs_dim):
                sum_val += obs[k] * self.W1[k * Self.hidden_dim + j]
            h[j] = sum_val if sum_val > 0 else 0

        # Layer 2: h @ W2 + b2
        var out = Array[Float32, Self.out_dim](fill=0)
        for j in range(Self.out_dim):
            var sum_val: Float32 = self.b2[j]
            for k in range(Self.hidden_dim):
                sum_val += h[k] * self.W2[k * Self.out_dim + j]
            out[j] = sum_val

        return out^

    def forward_batch[
        batch_size: Int
    ](
        self,
        obs: Array[Float32, batch_size * Self.obs_dim],
        mut output: Array[Float32, batch_size * Self.out_dim],
    ):
        """Batched forward pass on CPU."""
        for b in range(batch_size):
            # Layer 1
            var h = Array[Float32, Self.hidden_dim](fill=0)
            for j in range(Self.hidden_dim):
                var sum_val: Float32 = self.b1[j]
                for k in range(Self.obs_dim):
                    sum_val += (
                        obs[b * Self.obs_dim + k]
                        * self.W1[k * Self.hidden_dim + j]
                    )
                h[j] = sum_val if sum_val > 0 else 0

            # Layer 2
            for j in range(Self.out_dim):
                var sum_val: Float32 = self.b2[j]
                for k in range(Self.hidden_dim):
                    sum_val += h[k] * self.W2[k * Self.out_dim + j]
                output[b * Self.out_dim + j] = sum_val


# =============================================================================
# GPU Forward Pass Kernel
# =============================================================================


def forward_kernel_simple[
    dtype: DType,
    batch_size: Int,
    obs_dim: Int,
    hidden_dim: Int,
    out_dim: Int,
    TPB: Int,
](
    output: Tile[dtype, batch_size, out_dim],
    obs: ImmTile[dtype, batch_size, obs_dim],
    W1: ImmTile[dtype, obs_dim, hidden_dim],
    b1: ImmTile[dtype, hidden_dim],
    W2: ImmTile[dtype, hidden_dim, out_dim],
    b2: ImmTile[dtype, out_dim],
):
    """Simple GPU forward pass - one thread per output element."""
    var global_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total_outputs = batch_size * out_dim

    if global_idx >= total_outputs:
        return

    var b = global_idx // out_dim  # Batch index
    var j = global_idx % out_dim  # Output index

    # Compute hidden layer activations for this batch element
    var h = Array[Scalar[dtype], hidden_dim](fill=Scalar[dtype](0))

    for hid in range(hidden_dim):
        var sum_val = rebind[Scalar[dtype]](b1[hid])
        for k in range(obs_dim):
            sum_val += rebind[Scalar[dtype]](obs[b, k]) * rebind[Scalar[dtype]](
                W1[k, hid]
            )
        h[hid] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

    # Compute output for this element
    var out_val = rebind[Scalar[dtype]](b2[j])
    for k in range(hidden_dim):
        out_val += h[k] * rebind[Scalar[dtype]](W2[k, j])

    output[b, j] = out_val


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_cpu_single[
    obs_dim: Int, hidden_dim: Int, out_dim: Int
](num_iters: Int) -> Float64:
    """Benchmark single-observation CPU forward pass."""
    var net = CPUNetwork[obs_dim, hidden_dim, out_dim]()

    var obs = Array[Float32, obs_dim](fill=0)
    for i in range(obs_dim):
        obs[i] = Float32(random_float64())

    # Warm up
    for _ in range(100):
        var out = net.forward_single(obs)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        var out = net.forward_single(obs)
    var end = perf_counter_ns()

    var elapsed_sec = Float64(end - start) / 1e9
    return Float64(num_iters) / elapsed_sec


def benchmark_cpu_batch[
    obs_dim: Int, hidden_dim: Int, out_dim: Int, batch_size: Int
](num_iters: Int) -> Float64:
    """Benchmark batched CPU forward pass."""
    var net = CPUNetwork[obs_dim, hidden_dim, out_dim]()

    var obs = Array[Float32, batch_size * obs_dim](fill=0)
    for i in range(batch_size * obs_dim):
        obs[i] = Float32(random_float64())
    var output = Array[Float32, batch_size * out_dim](fill=0)

    # Warm up
    for _ in range(100):
        net.forward_batch[batch_size](obs, output)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        net.forward_batch[batch_size](obs, output)
    var end = perf_counter_ns()

    var elapsed_sec = Float64(end - start) / 1e9
    return Float64(num_iters) / elapsed_sec


def benchmark_gpu_batch[
    obs_dim: Int, hidden_dim: Int, out_dim: Int, batch_size: Int
](ctx: DeviceContext, num_iters: Int) raises -> Float64:
    """Benchmark GPU forward pass."""
    comptime dtype = DType.float32
    comptime TPB = 64
    comptime W1_SIZE = obs_dim * hidden_dim
    comptime W2_SIZE = hidden_dim * out_dim

    # Allocate GPU buffers
    var obs_gpu = ctx.enqueue_create_buffer[dtype](batch_size * obs_dim)
    var W1_gpu = ctx.enqueue_create_buffer[dtype](W1_SIZE)
    var b1_gpu = ctx.enqueue_create_buffer[dtype](hidden_dim)
    var W2_gpu = ctx.enqueue_create_buffer[dtype](W2_SIZE)
    var b2_gpu = ctx.enqueue_create_buffer[dtype](out_dim)
    var output_gpu = ctx.enqueue_create_buffer[dtype](batch_size * out_dim)

    # Initialize
    with obs_gpu.map_to_host() as host:
        for i in range(batch_size * obs_dim):
            host[i] = Scalar[dtype](random_float64())
    with W1_gpu.map_to_host() as host:
        var std = sqrt(2.0 / Float64(obs_dim + hidden_dim))
        for i in range(W1_SIZE):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std)
    b1_gpu.enqueue_fill(0)
    with W2_gpu.map_to_host() as host:
        var std = sqrt(2.0 / Float64(hidden_dim + out_dim))
        for i in range(W2_SIZE):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std)
    b2_gpu.enqueue_fill(0)
    output_gpu.enqueue_fill(0)
    ctx.synchronize()

    # Create layout tensors
    var obs_t = tile[batch_size, obs_dim](obs_gpu)
    var W1_t = tile[obs_dim, hidden_dim](W1_gpu)
    var b1_t = tile[hidden_dim](b1_gpu)
    var W2_t = tile[hidden_dim, out_dim](W2_gpu)
    var b2_t = tile[out_dim](b2_gpu)
    var output_t = tile[batch_size, out_dim](output_gpu)

    comptime total_outputs = batch_size * out_dim
    comptime num_blocks = (total_outputs + TPB - 1) // TPB
    comptime kernel = forward_kernel_simple[
        dtype, batch_size, obs_dim, hidden_dim, out_dim, TPB
    ]

    # Warm up
    for _ in range(100):
        ctx.enqueue_function[kernel, kernel](
            output_t,
            obs_t,
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[kernel, kernel](
            output_t,
            obs_t,
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    var elapsed_sec = Float64(end - start) / 1e9
    return Float64(num_iters) / elapsed_sec


def benchmark_kernel_launch_overhead(
    ctx: DeviceContext, num_iters: Int
) raises -> Float64:
    """Measure pure kernel launch overhead with a trivial kernel."""
    comptime dtype = DType.float32
    comptime size = 64

    var buf = ctx.enqueue_create_buffer[dtype](size)
    buf.enqueue_fill(0)
    ctx.synchronize()

    var buf_t = tile[size](buf)

    # Simple kernel that does almost nothing
    def trivial_kernel(
        buf: Tile[dtype, size]
    ):
        var i = Int(thread_idx.x)
        if i < size:
            buf[i] = rebind[Scalar[dtype]](buf[i]) + Scalar[dtype](1.0)

    # Warm up
    for _ in range(100):
        ctx.enqueue_function[trivial_kernel, trivial_kernel](
            buf_t, grid_dim=(1,), block_dim=(64,)
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[trivial_kernel, trivial_kernel](
            buf_t, grid_dim=(1,), block_dim=(64,)
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    var elapsed_ns = Float64(end - start)
    return elapsed_ns / Float64(num_iters)  # ns per kernel launch


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("Neural Network Forward Pass Benchmark: CPU vs GPU")
    print("=" * 70)
    print()

    # Network config matching DQN: 8 -> 128 -> 128 -> 4
    comptime obs_dim = 8
    comptime hidden_dim = 128
    comptime out_dim = 4

    var num_iters = 10000

    print(
        "Network: "
        + String(obs_dim)
        + " -> "
        + String(hidden_dim)
        + " -> "
        + String(out_dim)
    )
    print("Iterations: " + String(num_iters))
    print()

    # CPU single observation (action selection scenario)
    print("-" * 70)
    print("CPU: Single observation forward pass (action selection)")
    print("-" * 70)
    var cpu_single = benchmark_cpu_single[obs_dim, hidden_dim, out_dim](
        num_iters
    )
    print("  " + String(Int(cpu_single)) + " forward passes/sec")
    print("  " + fit(String(1e9 / cpu_single), 6) + " ns per forward pass")
    print()

    # CPU batched
    print("-" * 70)
    print("CPU: Batched forward pass")
    print("-" * 70)

    var cpu_b64 = benchmark_cpu_batch[obs_dim, hidden_dim, out_dim, 64](
        num_iters
    )
    print(
        "  Batch 64:  "
        + String(Int(cpu_b64))
        + " batches/sec ("
        + String(Int(cpu_b64 * 64))
        + " obs/sec)"
    )

    var cpu_b256 = benchmark_cpu_batch[obs_dim, hidden_dim, out_dim, 256](
        num_iters
    )
    print(
        "  Batch 256: "
        + String(Int(cpu_b256))
        + " batches/sec ("
        + String(Int(cpu_b256 * 256))
        + " obs/sec)"
    )

    print()

    # GPU benchmarks
    with DeviceContext() as ctx:
        print("-" * 70)
        print("GPU: Kernel launch overhead (trivial kernel)")
        print("-" * 70)
        var launch_overhead_ns = benchmark_kernel_launch_overhead(
            ctx, num_iters
        )
        print("  " + fit(String(launch_overhead_ns), 8) + " ns per kernel launch")
        print(
            "  Max "
            + String(Int(1e9 / launch_overhead_ns))
            + " kernel launches/sec"
        )
        print()

        print("-" * 70)
        print("GPU: Batched forward pass")
        print("-" * 70)

        var gpu_b64 = benchmark_gpu_batch[obs_dim, hidden_dim, out_dim, 64](
            ctx, num_iters
        )
        print("  Batch 64:  " + String(Int(gpu_b64)) + " batches/sec")

        var gpu_b256 = benchmark_gpu_batch[obs_dim, hidden_dim, out_dim, 256](
            ctx, num_iters
        )
        print("  Batch 256: " + String(Int(gpu_b256)) + " batches/sec")

        print()
        print("-" * 70)
        print("Summary: GPU vs CPU speedup")
        print("-" * 70)
        print("  Batch 64:  GPU is " + fit(String(gpu_b64 / cpu_b64), 4) + "x CPU")
        print(
            "  Batch 256: GPU is " + fit(String(gpu_b256 / cpu_b256), 4) + "x CPU"
        )

        print()
        print("=" * 70)
        print("Analysis:")
        print(
            "  - Single forward pass on CPU: ~"
            + fit(String(1e6 / cpu_single), 4)
            + " microseconds"
        )
        print(
            "  - Kernel launch overhead: ~"
            + fit(String(launch_overhead_ns / 1000), 4)
            + " microseconds"
        )
        print("  - If kernel launch > forward pass time, GPU loses!")
        print("=" * 70)
