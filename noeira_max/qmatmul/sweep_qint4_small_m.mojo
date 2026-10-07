# +--------------------------------------------------------------------------+ #
# | MAX int4 GEMM tile configs at small M, on Kev-9B's shapes (the Orin)
# +--------------------------------------------------------------------------+ #
"""`matmul_gpu_qint4` picks a tuned `MatmulConfig` only for Llama-3-8B's
static (K, N); every Kev-9B shape but 4096x4096 runs the default 128x128x32
tile, which wastes most of an M = 48 tile (4-6 TF/s against 13.7 at M = 218,
bench_qint4_orin.mojo). This sweeps MAX's OWN small-M configs (taken from
its tables for 4096x6144 / 14336 and 14336x4096) through `multistage_gemm_q`
on Kev's shapes. One config per build, chosen at compile time:

    for c in 0 1 2 3 4 5 6 7 8; do
      pixi run -e jetson mojo build -I . --target-accelerator sm_87 -D CFG=$c \\
          noeira_max/qmatmul/sweep_qint4_small_m.mojo -o build/sweep_q4_$c && ./build/sweep_q4_$c
    done

A config that does not instantiate fails its own build only (MAX's tuned
4096x4096 set does not, see bench_qint4_orin.mojo).
"""

from std.memory import bitcast
from std.random import rand, random_float64, seed
from std.sys import get_defined_int
from std.time import perf_counter_ns
from std.utils.index import Index

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.utils_gpu import MatmulConfig

from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0, multistage_gemm_q


comptime GS = 32
comptime GB = 2 + GS // 2
comptime ITERS = 20
comptime CFG = get_defined_int["CFG", 0]()

# (block M, N, K), (warp M, N, K), pipeline stages, warp-K partitions — MAX's own tables
comptime BT = [Index(128, 128, 32), Index(64, 64, 32), Index(64, 64, 32), Index(32, 64, 32), Index(32, 64, 32),
               Index(16, 64, 32), Index(32, 64, 128), Index(64, 128, 32), Index(128, 128, 32)]
comptime WT = [Index(64, 64, 32), Index(64, 64, 32), Index(64, 64, 32), Index(32, 64, 32), Index(32, 64, 32),
               Index(16, 64, 32), Index(16, 64, 32), Index(64, 64, 32), Index(64, 64, 32)]
comptime STAGES = [5, 4, 5, 3, 4, 5, 3, 4, 3]
comptime WK = [1, 4, 4, 4, 4, 4, 4, 4, 2]

comptime CONFIG = MatmulConfig[DType.bfloat16, DType.uint8, DType.bfloat16, True](
    block_tile_shape=BT[CFG],
    warp_tile_shape=WT[CFG],
    num_pipeline_stages=STAGES[CFG],
    num_k_partitions=1,
    num_warp_k_partitions=WK[CFG],
)


def q4_0_value(blk: UnsafePointer[UInt8, _], j: Int) -> Float32:
    var u = UInt16(blk[0]) | (UInt16(blk[1]) << 8)
    var d = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](u)).cast[DType.float32]()
    var byte = blk[2 + (j % 16)]
    var nib = (byte & 0xF) if j < 16 else (byte >> 4)
    return (Float32(Int(nib)) - 8.0) * d[0]


def run_shape[N: Int, K: Int](ctx: DeviceContext, count: Int, mut total: List[Float64]) raises:
    comptime KB = (K // GS) * GB
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
    ctx.synchronize()
    var bp = b_h.unsafe_ptr()
    rand[DType.uint8](bp, N * KB, min=0, max=255)
    for blk in range(N * (K // GS)):
        var u = bitcast[DType.uint16, 1](SIMD[DType.float16, 1](Float16(0.001 + 0.001 * random_float64())))
        bp[blk * GB] = UInt8(u & 0xFF)
        bp[blk * GB + 1] = UInt8(u >> 8)
    var b_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    var bpk_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    ctx.enqueue_copy(b_d, b_h)
    var b_tt = TileTensor(b_d, row_major(Coord(Idx[N], Idx[KB])))
    var bpk_tt = TileTensor(bpk_d, row_major(Coord(Idx[N], Idx[KB])))
    gpu_qint4_repack_Q4_0["gpu"](b_tt.as_immut(), bpk_tt, ctx)
    ctx.synchronize()

    var line = String(t"N={N} K={K} x{count}")
    for i in range(3):
        var M = [48, 218, 42][i]
        var a_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
        var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
        ctx.synchronize()
        rand[DType.bfloat16](a_h.unsafe_ptr(), M * K)
        var a_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
        var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
        ctx.enqueue_copy(a_d, a_h)
        var a_tt = TileTensor(a_d, row_major(Coord(M, Idx[K])))
        var c_tt = TileTensor(c_d, row_major(Coord(M, Idx[N])))
        multistage_gemm_q[group_size=GS, pack_factor=8, config=CONFIG](
            c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(), bpk_tt.as_immut().to_layout_tensor(), CONFIG, ctx
        )
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()
        # 2 columns against the host dequantize-and-multiply
        var err = Float64(0)
        var mag = Float64(0)
        for n in range(2):
            for m in range(M):
                var acc = Float64(0)
                for g in range(K // GS):
                    var blk = bp + (n * (K // GS) + g) * GB
                    for j in range(GS):
                        acc += Float64(a_h.unsafe_ptr()[m * K + g * GS + j].cast[DType.float32]()) * Float64(q4_0_value(blk, j))
                err = max(err, abs(Float64(c_h.unsafe_ptr()[m * N + n].cast[DType.float32]()) - acc))
                mag = max(mag, abs(acc))
        for _ in range(3):
            multistage_gemm_q[group_size=GS, pack_factor=8, config=CONFIG](
                c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(), bpk_tt.as_immut().to_layout_tensor(), CONFIG, ctx
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            multistage_gemm_q[group_size=GS, pack_factor=8, config=CONFIG](
                c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(), bpk_tt.as_immut().to_layout_tensor(), CONFIG, ctx
            )
        ctx.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 1e3 / ITERS
        total[i] += Float64(count) * us
        line += String(t"  M={M} {Int(us)} us") + (" OK" if err / mag < 0.02 else " WRONG")
    print(line)
    _ = b_h^  # the reference reads through `bp`: keep the host buffer alive past its device copy


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime bt = BT[CFG]
    comptime wt = WT[CFG]
    comptime st: Int = STAGES[CFG]
    comptime wk: Int = WK[CFG]
    print("CFG", CFG, "block", bt[0], bt[1], bt[2], " warp", wt[0], wt[1], wt[2], " stages", st, " warp-K", wk)
    var total: List[Float64] = [0.0, 0.0, 0.0]
    run_shape[4096, 12288](ctx, 32, total)
    run_shape[12288, 4096](ctx, 64, total)
    run_shape[8192, 4096](ctx, 32, total)
    run_shape[4096, 4096](ctx, 56, total)
    run_shape[1024, 4096](ctx, 16, total)
    print("CFG", CFG, "per decision: M=48", Int(total[0] / 1e3), "ms  M=218", Int(total[1] / 1e3), "ms  M=42", Int(total[2] / 1e3), "ms  total", Int((total[0] + total[1] + total[2]) / 1e3), "ms")
