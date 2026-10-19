"""Benchmark: CUDA Graph replay vs direct kernel dispatch.

Captures a 5-kernel chain (simulating a mini forward pass) and
benchmarks graph replay against direct dispatch.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_cuda_graph.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, ImmTile, tile
from noeira.nn.constants import DT as dtype
from noeira.cuda import CUDAGraph


def add_kernel[N: Int](
    output: Tile[dtype, N],
    a: ImmTile[dtype, N],
    b: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = a[tid] + b[tid]


def scale_kernel[N: Int](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = input[tid] * Scalar[dtype](2.0)


def relu_kernel[N: Int](
    data: Tile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        var v = data.ptr[tid]
        data.ptr[tid] = v if v > Scalar[dtype](0.0) else Scalar[dtype](0.0)


def main() raises:
    print("=== CUDA Graph Benchmark ===\n")

    var ctx = DeviceContext()
    ctx.synchronize()

    # Buffers
    comptime N = 8192
    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    var d_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var a_t = tile[N](a_buf)
    var b_t = tile[N](b_buf)
    var c_t = tile[N](c_buf)
    var c_i = tile[N](c_buf)
    var d_t = tile[N](d_buf)
    var d_i = tile[N](d_buf)

    comptime k_add = add_kernel[N]
    comptime k_scale = scale_kernel[N]
    comptime k_relu = relu_kernel[N]

    # 5-kernel chain: add → scale → relu → scale → relu
    def run_chain() raises:
        ctx.enqueue_function[k_add, k_add](c_t, a_t, b_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](d_t, c_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](d_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](c_t, d_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](c_t, grid_dim=grid, block_dim=block)

    # Warmup (also discovers the Mojo stream for the interceptor)
    run_chain()
    ctx.synchronize()
    with c_buf.map_to_host() as h:
        print("Direct dispatch result: c[0] =", h[0], "(expected 12.0)")

    # --- Capture using CUDAGraph API ---
    var graph = CUDAGraph(ctx)

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    graph.begin_capture()
    run_chain()
    graph.end_capture()

    print("Captured graph with", graph.num_nodes(), "nodes")

    # Verify replay
    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    graph.replay()
    with c_buf.map_to_host() as h:
        print("Graph replay result: c[0] =", h[0], "(expected 12.0)")
        if h[0] != Scalar[dtype](12.0):
            print("MISMATCH")
            return

    # --- Benchmark ---
    print("\n--- Benchmark: 5 kernels x 1000 iterations ---\n")

    var warmup = 100
    var iters = 1000

    for _ in range(warmup):
        run_chain()
        ctx.synchronize()

    var total_direct: UInt = 0
    for _ in range(iters):
        var start = perf_counter_ns()
        run_chain()
        ctx.synchronize()
        total_direct += perf_counter_ns() - start

    for _ in range(warmup):
        graph.replay()

    var total_graph: UInt = 0
    for _ in range(iters):
        var start = perf_counter_ns()
        graph.replay()
        total_graph += perf_counter_ns() - start

    var avg_direct = Float64(total_direct // UInt(iters)) / 1000.0
    var avg_graph = Float64(total_graph // UInt(iters)) / 1000.0

    print("  Direct dispatch (5 kernels): ", avg_direct, " us")
    print("  Graph replay (5 kernels):    ", avg_graph, " us")
    if avg_graph > 0.0:
        print("  Speedup:                     ", avg_direct / avg_graph, "x")
        print("  Savings per call:            ", avg_direct - avg_graph, " us")

    print("\n=== Done ===")
