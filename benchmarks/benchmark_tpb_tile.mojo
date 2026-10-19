"""Benchmark for TPB (Threads Per Block) and TILE sizes on Apple Silicon.

Tests different configurations to find optimal GPU parameters for Metal.

Run with:
    pixi run -e apple mojo run benchmarks/benchmark_tpb_tile.mojo
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64
from std.math import sin, sqrt
from std.memory import Pointer

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from noeira.core.tiles import Tile, tile, shared_tile

from noeira.nn.constants import DT as dtype
from noeira.core.fmt import fit


# =============================================================================
# Benchmark Configuration
# =============================================================================

comptime BATCH = 1024
comptime IN_DIM = 64
comptime HIDDEN_DIM = 256
comptime OUT_DIM = 64
comptime WARMUP_ITERS = 10
comptime BENCHMARK_ITERS = 100


# =============================================================================
# Matmul Kernels with Different TILE Sizes
# =============================================================================


def matmul_kernel_tile8[
    M: Int, K: Int, N: Int
](
    A: Tile[dtype, M, K],
    B: Tile[dtype, K, N],
    C: Tile[dtype, M, N],
):
    """Tiled matmul with TILE=8."""
    comptime TILE = 8
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var row = Int(block_idx.y * TILE + UInt(local_row))
    var col = Int(block_idx.x * TILE + UInt(local_col))

    if row >= M or col >= N:
        return

    # Shared memory tiles
    var a_shared = shared_tile[dtype, TILE, TILE]()

    var b_shared = shared_tile[dtype, TILE, TILE]()

    var acc: Scalar[dtype] = 0.0

    for t in range((K + TILE - 1) // TILE):
        # Load A tile
        var a_col = t * TILE + local_col
        if row < M and a_col < K:
            a_shared[local_row, local_col] = A[row, a_col]
        else:
            a_shared[local_row, local_col] = 0

        # Load B tile
        var b_row = t * TILE + local_row
        if b_row < K and col < N:
            b_shared[local_row, local_col] = B[b_row, col]
        else:
            b_shared[local_row, local_col] = 0

        barrier()

        # Compute

        comptime for k in range(TILE):
            acc += rebind[Scalar[dtype]](a_shared[local_row, k]) * rebind[
                Scalar[dtype]
            ](b_shared[k, local_col])

        barrier()

    C[row, col] = acc


def matmul_kernel_tile16[
    M: Int, K: Int, N: Int
](
    A: Tile[dtype, M, K],
    B: Tile[dtype, K, N],
    C: Tile[dtype, M, N],
):
    """Tiled matmul with TILE=16."""
    comptime TILE = 16
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var row = Int(block_idx.y * TILE + UInt(local_row))
    var col = Int(block_idx.x * TILE + UInt(local_col))

    if row >= M or col >= N:
        return

    var a_shared = shared_tile[dtype, TILE, TILE]()

    var b_shared = shared_tile[dtype, TILE, TILE]()

    var acc: Scalar[dtype] = 0.0

    for t in range((K + TILE - 1) // TILE):
        var a_col = t * TILE + local_col
        if row < M and a_col < K:
            a_shared[local_row, local_col] = A[row, a_col]
        else:
            a_shared[local_row, local_col] = 0

        var b_row = t * TILE + local_row
        if b_row < K and col < N:
            b_shared[local_row, local_col] = B[b_row, col]
        else:
            b_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[Scalar[dtype]](a_shared[local_row, k]) * rebind[
                Scalar[dtype]
            ](b_shared[k, local_col])

        barrier()

    C[row, col] = acc


def matmul_kernel_tile32[
    M: Int, K: Int, N: Int
](
    A: Tile[dtype, M, K],
    B: Tile[dtype, K, N],
    C: Tile[dtype, M, N],
):
    """Tiled matmul with TILE=32."""
    comptime TILE = 32
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var row = Int(block_idx.y * TILE + UInt(local_row))
    var col = Int(block_idx.x * TILE + UInt(local_col))

    if row >= M or col >= N:
        return

    var a_shared = shared_tile[dtype, TILE, TILE]()

    var b_shared = shared_tile[dtype, TILE, TILE]()

    var acc: Scalar[dtype] = 0.0

    for t in range((K + TILE - 1) // TILE):
        var a_col = t * TILE + local_col
        if row < M and a_col < K:
            a_shared[local_row, local_col] = A[row, a_col]
        else:
            a_shared[local_row, local_col] = 0

        var b_row = t * TILE + local_row
        if b_row < K and col < N:
            b_shared[local_row, local_col] = B[b_row, col]
        else:
            b_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[Scalar[dtype]](a_shared[local_row, k]) * rebind[
                Scalar[dtype]
            ](b_shared[k, local_col])

        barrier()

    C[row, col] = acc


# =============================================================================
# Elementwise Kernels with Different TPB Sizes
# =============================================================================


def relu_kernel[
    SIZE: Int, TPB: Int
](
    x: Tile[dtype, SIZE],
    y: Tile[dtype, SIZE],
):
    """ReLU kernel with parameterized TPB."""
    var idx = Int(block_idx.x * UInt(TPB) + thread_idx.x)
    if idx >= SIZE:
        return
    var val = rebind[Scalar[dtype]](x[idx])
    y[idx] = val if val > 0 else Scalar[dtype](0)


def add_kernel[
    SIZE: Int, TPB: Int
](
    a: Tile[dtype, SIZE],
    b: Tile[dtype, SIZE],
    c: Tile[dtype, SIZE],
):
    """Add kernel with parameterized TPB."""
    var idx = Int(block_idx.x * UInt(TPB) + thread_idx.x)
    if idx >= SIZE:
        return
    c[idx] = rebind[Scalar[dtype]](a[idx]) + rebind[Scalar[dtype]](b[idx])


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_matmul_tile[TILE: Int](ctx: DeviceContext) raises -> Float64:
    """Benchmark matmul with specific TILE size."""
    comptime M = BATCH
    comptime K = IN_DIM
    comptime N = HIDDEN_DIM

    # Allocate buffers
    var a_buf = ctx.enqueue_create_buffer[dtype](M * K)
    var b_buf = ctx.enqueue_create_buffer[dtype](K * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](M * N)

    # Initialize with random data on host
    var a_data = Array[Scalar[dtype], M * K](uninitialized=True)
    var b_data = Array[Scalar[dtype], K * N](uninitialized=True)

    for i in range(M * K):
        a_data[i] = Scalar[dtype](random_float64() * 2 - 1)
    for i in range(K * N):
        b_data[i] = Scalar[dtype](random_float64() * 2 - 1)

    ctx.enqueue_copy(a_buf, a_data.unsafe_ptr())
    ctx.enqueue_copy(b_buf, b_data.unsafe_ptr())
    ctx.synchronize()

    var a = tile[M, K](a_buf.unsafe_ptr())
    var b = tile[K, N](b_buf.unsafe_ptr())
    var c = tile[M, N](c_buf.unsafe_ptr())

    comptime BLOCKS_X = (N + TILE - 1) // TILE
    comptime BLOCKS_Y = (M + TILE - 1) // TILE

    # Warmup and benchmark
    comptime if TILE == 8:

        @always_inline
        def kernel8(
            a: Tile[dtype, M, K],
            b: Tile[dtype, K, N],
            c: Tile[dtype, M, N],
        ):
            matmul_kernel_tile8[M, K, N](a, b, c)

        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[kernel8, kernel8](
                a,
                b,
                c,
                grid_dim=(BLOCKS_X, BLOCKS_Y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()

        var start = perf_counter_ns()
        for _ in range(BENCHMARK_ITERS):
            ctx.enqueue_function[kernel8, kernel8](
                a,
                b,
                c,
                grid_dim=(BLOCKS_X, BLOCKS_Y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()
        var end = perf_counter_ns()

        return Float64(end - start) / 1_000_000.0 / BENCHMARK_ITERS

    elif TILE == 16:

        @always_inline
        def kernel16(
            a: Tile[dtype, M, K],
            b: Tile[dtype, K, N],
            c: Tile[dtype, M, N],
        ):
            matmul_kernel_tile16[M, K, N](a, b, c)

        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[kernel16, kernel16](
                a,
                b,
                c,
                grid_dim=(BLOCKS_X, BLOCKS_Y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()

        var start = perf_counter_ns()
        for _ in range(BENCHMARK_ITERS):
            ctx.enqueue_function[kernel16, kernel16](
                a,
                b,
                c,
                grid_dim=(BLOCKS_X, BLOCKS_Y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()
        var end = perf_counter_ns()

        return Float64(end - start) / 1_000_000.0 / BENCHMARK_ITERS

    else:  # TILE == 32

        @always_inline
        def kernel32(
            a: Tile[dtype, M, K],
            b: Tile[dtype, K, N],
            c: Tile[dtype, M, N],
        ):
            matmul_kernel_tile32[M, K, N](a, b, c)

        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[kernel32, kernel32](
                a,
                b,
                c,
                grid_dim=(BLOCKS_X, BLOCKS_Y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()

        var start = perf_counter_ns()
        for _ in range(BENCHMARK_ITERS):
            ctx.enqueue_function[kernel32, kernel32](
                a,
                b,
                c,
                grid_dim=(BLOCKS_X, BLOCKS_Y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()
        var end = perf_counter_ns()

        return Float64(end - start) / 1_000_000.0 / BENCHMARK_ITERS


def benchmark_elementwise_tpb[TPB: Int](ctx: DeviceContext) raises -> Float64:
    """Benchmark elementwise ops with specific TPB."""
    comptime SIZE = BATCH * HIDDEN_DIM
    comptime BLOCKS = (SIZE + TPB - 1) // TPB

    # Allocate buffers
    var a_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var b_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var c_buf = ctx.enqueue_create_buffer[dtype](SIZE)

    # Initialize
    var a_data = Array[Scalar[dtype], SIZE](uninitialized=True)
    var b_data = Array[Scalar[dtype], SIZE](uninitialized=True)

    for i in range(SIZE):
        a_data[i] = Scalar[dtype](random_float64() * 2 - 1)
        b_data[i] = Scalar[dtype](random_float64() * 2 - 1)

    ctx.enqueue_copy(a_buf, a_data.unsafe_ptr())
    ctx.enqueue_copy(b_buf, b_data.unsafe_ptr())
    ctx.synchronize()

    var a = tile[SIZE](a_buf.unsafe_ptr())
    var b = tile[SIZE](b_buf.unsafe_ptr())
    var c = tile[SIZE](c_buf.unsafe_ptr())

    @always_inline
    def relu_wrapper(
        x: Tile[dtype, SIZE],
        y: Tile[dtype, SIZE],
    ):
        relu_kernel[SIZE, TPB](x, y)

    @always_inline
    def add_wrapper(
        a: Tile[dtype, SIZE],
        b: Tile[dtype, SIZE],
        c: Tile[dtype, SIZE],
    ):
        add_kernel[SIZE, TPB](a, b, c)

    # Warmup
    for _ in range(WARMUP_ITERS):
        ctx.enqueue_function[relu_wrapper, relu_wrapper](
            a,
            c,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[add_wrapper, add_wrapper](
            a,
            b,
            c,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(BENCHMARK_ITERS):
        ctx.enqueue_function[relu_wrapper, relu_wrapper](
            a,
            c,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[add_wrapper, add_wrapper](
            a,
            b,
            c,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    # Return time per op pair (ReLU + Add)
    return Float64(end - start) / 1_000_000.0 / BENCHMARK_ITERS


def benchmark_combined[
    TILE: Int, TPB: Int
](ctx: DeviceContext) raises -> Float64:
    """Benchmark combined matmul + elementwise pipeline."""
    comptime M = BATCH
    comptime K = IN_DIM
    comptime N = HIDDEN_DIM
    comptime SIZE = M * N

    # Allocate all buffers
    var a_buf = ctx.enqueue_create_buffer[dtype](M * K)
    var w_buf = ctx.enqueue_create_buffer[dtype](K * N)
    var out_buf = ctx.enqueue_create_buffer[dtype](M * N)
    var relu_buf = ctx.enqueue_create_buffer[dtype](M * N)

    # Initialize
    var a_data = Array[Scalar[dtype], M * K](uninitialized=True)
    var w_data = Array[Scalar[dtype], K * N](uninitialized=True)

    for i in range(M * K):
        a_data[i] = Scalar[dtype](random_float64() * 2 - 1)
    for i in range(K * N):
        w_data[i] = Scalar[dtype](random_float64() * 2 - 1)

    ctx.enqueue_copy(a_buf, a_data.unsafe_ptr())
    ctx.enqueue_copy(w_buf, w_data.unsafe_ptr())
    ctx.synchronize()

    var a = tile[M, K](a_buf.unsafe_ptr())
    var w = tile[K, N](w_buf.unsafe_ptr())
    var out = tile[M, N](out_buf.unsafe_ptr())
    var relu_out = tile[SIZE](relu_buf.unsafe_ptr())
    var out_flat = tile[SIZE](out_buf.unsafe_ptr())

    comptime MATMUL_BLOCKS_X = (N + TILE - 1) // TILE
    comptime MATMUL_BLOCKS_Y = (M + TILE - 1) // TILE
    comptime ELEM_BLOCKS = (SIZE + TPB - 1) // TPB

    @always_inline
    def matmul_wrapper(
        a: Tile[dtype, M, K],
        b: Tile[dtype, K, N],
        c: Tile[dtype, M, N],
    ):
        comptime if TILE == 8:
            matmul_kernel_tile8[M, K, N](a, b, c)
        elif TILE == 16:
            matmul_kernel_tile16[M, K, N](a, b, c)
        else:
            matmul_kernel_tile32[M, K, N](a, b, c)

    @always_inline
    def relu_wrapper(
        x: Tile[dtype, SIZE],
        y: Tile[dtype, SIZE],
    ):
        relu_kernel[SIZE, TPB](x, y)

    # Warmup
    for _ in range(WARMUP_ITERS):
        ctx.enqueue_function[matmul_wrapper, matmul_wrapper](
            a,
            w,
            out,
            grid_dim=(MATMUL_BLOCKS_X, MATMUL_BLOCKS_Y),
            block_dim=(TILE, TILE),
        )
        ctx.enqueue_function[relu_wrapper, relu_wrapper](
            out_flat,
            relu_out,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(BENCHMARK_ITERS):
        ctx.enqueue_function[matmul_wrapper, matmul_wrapper](
            a,
            w,
            out,
            grid_dim=(MATMUL_BLOCKS_X, MATMUL_BLOCKS_Y),
            block_dim=(TILE, TILE),
        )
        ctx.enqueue_function[relu_wrapper, relu_wrapper](
            out_flat,
            relu_out,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / 1_000_000.0 / BENCHMARK_ITERS


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)

    print("=" * 70)
    print("TPB and TILE Benchmark for Apple Silicon")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Batch size:       " + String(BATCH))
    print("  Input dim:        " + String(IN_DIM))
    print("  Hidden dim:       " + String(HIDDEN_DIM))
    print("  Output dim:       " + String(OUT_DIM))
    print("  Warmup iters:     " + String(WARMUP_ITERS))
    print("  Benchmark iters:  " + String(BENCHMARK_ITERS))
    print()

    var ctx = DeviceContext()

    # =========================================================================
    # TILE Benchmark (Matmul)
    # =========================================================================
    print("-" * 70)
    print(
        "TILE Size Benchmark (Matmul: "
        + String(BATCH)
        + "x"
        + String(IN_DIM)
        + " @ "
        + String(IN_DIM)
        + "x"
        + String(HIDDEN_DIM)
        + ")"
    )
    print("-" * 70)

    var tile8_time = benchmark_matmul_tile[8](ctx)
    print("  TILE=8:  " + fit(String(tile8_time), 8) + " ms/iter")

    var tile16_time = benchmark_matmul_tile[16](ctx)
    print("  TILE=16: " + fit(String(tile16_time), 8) + " ms/iter")

    var tile32_time = benchmark_matmul_tile[32](ctx)
    print("  TILE=32: " + fit(String(tile32_time), 8) + " ms/iter")

    var best_tile = 8
    var best_tile_time = tile8_time
    if tile16_time < best_tile_time:
        best_tile = 16
        best_tile_time = tile16_time
    if tile32_time < best_tile_time:
        best_tile = 32
        best_tile_time = tile32_time

    print()
    print(
        "  Best TILE: "
        + String(best_tile)
        + " ("
        + fit(String(best_tile_time), 8)
        + " ms)"
    )
    print()

    # =========================================================================
    # TPB Benchmark (Elementwise)
    # =========================================================================
    print("-" * 70)
    print(
        "TPB Size Benchmark (Elementwise: "
        + String(BATCH * HIDDEN_DIM)
        + " elements)"
    )
    print("-" * 70)

    var tpb32_time = benchmark_elementwise_tpb[32](ctx)
    print("  TPB=32:  " + fit(String(tpb32_time), 8) + " ms/iter")

    var tpb64_time = benchmark_elementwise_tpb[64](ctx)
    print("  TPB=64:  " + fit(String(tpb64_time), 8) + " ms/iter")

    var tpb128_time = benchmark_elementwise_tpb[128](ctx)
    print("  TPB=128: " + fit(String(tpb128_time), 8) + " ms/iter")

    var tpb256_time = benchmark_elementwise_tpb[256](ctx)
    print("  TPB=256: " + fit(String(tpb256_time), 8) + " ms/iter")

    var tpb512_time = benchmark_elementwise_tpb[512](ctx)
    print("  TPB=512: " + fit(String(tpb512_time), 8) + " ms/iter")

    var best_tpb = 32
    var best_tpb_time = tpb32_time
    if tpb64_time < best_tpb_time:
        best_tpb = 64
        best_tpb_time = tpb64_time
    if tpb128_time < best_tpb_time:
        best_tpb = 128
        best_tpb_time = tpb128_time
    if tpb256_time < best_tpb_time:
        best_tpb = 256
        best_tpb_time = tpb256_time
    if tpb512_time < best_tpb_time:
        best_tpb = 512
        best_tpb_time = tpb512_time

    print()
    print(
        "  Best TPB: "
        + String(best_tpb)
        + " ("
        + fit(String(best_tpb_time), 8)
        + " ms)"
    )
    print()

    # =========================================================================
    # Combined Benchmark (Matmul + ReLU pipeline)
    # =========================================================================
    print("-" * 70)
    print("Combined Benchmark (Matmul + ReLU)")
    print("-" * 70)
    print()
    print("Testing TILE x TPB combinations...")
    print()

    # Test key combinations
    var combo_8_64 = benchmark_combined[8, 64](ctx)
    var combo_8_128 = benchmark_combined[8, 128](ctx)
    var combo_8_256 = benchmark_combined[8, 256](ctx)
    var combo_16_64 = benchmark_combined[16, 64](ctx)
    var combo_16_128 = benchmark_combined[16, 128](ctx)
    var combo_16_256 = benchmark_combined[16, 256](ctx)
    var combo_32_64 = benchmark_combined[32, 64](ctx)
    var combo_32_128 = benchmark_combined[32, 128](ctx)
    var combo_32_256 = benchmark_combined[32, 256](ctx)

    print("           TPB=64     TPB=128    TPB=256")
    print(
        "  TILE=8:  "
        + fit(String(combo_8_64), 8)
        + "   "
        + fit(String(combo_8_128), 8)
        + "   "
        + fit(String(combo_8_256), 8)
    )
    print(
        "  TILE=16: "
        + fit(String(combo_16_64), 8)
        + "   "
        + fit(String(combo_16_128), 8)
        + "   "
        + fit(String(combo_16_256), 8)
    )
    print(
        "  TILE=32: "
        + fit(String(combo_32_64), 8)
        + "   "
        + fit(String(combo_32_128), 8)
        + "   "
        + fit(String(combo_32_256), 8)
    )
    print()

    # Find best combo
    var combos = List[Tuple[Int, Int, Float64]]()
    combos.append((8, 64, combo_8_64))
    combos.append((8, 128, combo_8_128))
    combos.append((8, 256, combo_8_256))
    combos.append((16, 64, combo_16_64))
    combos.append((16, 128, combo_16_128))
    combos.append((16, 256, combo_16_256))
    combos.append((32, 64, combo_32_64))
    combos.append((32, 128, combo_32_128))
    combos.append((32, 256, combo_32_256))

    var best_combo_tile = 16
    var best_combo_tpb = 256
    var best_combo_time = combo_16_256

    for i in range(len(combos)):
        if combos[i][2] < best_combo_time:
            best_combo_tile = combos[i][0]
            best_combo_tpb = combos[i][1]
            best_combo_time = combos[i][2]

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Best for Matmul:      TILE=" + String(best_tile))
    print("Best for Elementwise: TPB=" + String(best_tpb))
    print(
        "Best Combined:        TILE="
        + String(best_combo_tile)
        + ", TPB="
        + String(best_combo_tpb)
    )
    print()
    print("Current defaults in nn/constants.mojo:")
    print("  TILE = 16")
    print("  TPB = 256")
    print()

    if best_combo_tile != 16 or best_combo_tpb != 256:
        print("RECOMMENDATION: Update constants to:")
        print("  comptime TILE = " + String(best_combo_tile))
        print("  comptime TPB = " + String(best_combo_tpb))
        var improvement = (combo_16_256 - best_combo_time) / combo_16_256 * 100
        print("  Expected improvement: " + fit(String(improvement), 5) + "%")
    else:
        print("Current defaults are optimal for this configuration!")

    print()
    print("=" * 70)
