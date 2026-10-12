# +--------------------------------------------------------------------------+ #
# | MAX's int4 tensor-core GEMM on the Orin, at Kev-9B's shapes
# +--------------------------------------------------------------------------+ #
"""`quantization.qmatmul_gpu.matmul_gpu_qint4` (MAX 26.6): bf16 activations,
4-bit weights (GGUF Q4_0 here, group 32, repacked by `gpu_qint4_repack_Q4_0`),
dequantized in shared memory and fed to a multistage tensor-core GEMM. Its
tile configs are tuned for Llama-3-8B shapes (4096x4096 / 6144 / 14336 and
14336x4096); Kev's other shapes take the default 128x128x32, 5-stage config.

!! The TUNED 4096x4096 configs do not instantiate in MAX 26.6 — not for sm_87, not for sm_80 either
("copy_from should move data of the same size, getting dst size 1 and src size 4" in
multistage_qgemm_kernel). 4096x4096 therefore calls `multistage_gemm_q` with the default config below.

Same shapes, M and per-decision weighting as the MLX table in
noeira-docs/ORIN_KEV_SPEEDUP.md (state 48 + rows 218 and 42; 248 linears).
Correctness: 8 output columns against a host dequantize-and-multiply.

    pixi run -e jetson build-jetson noeira_max/qmatmul/bench_qint4_orin.mojo -o build/bench_qint4_orin
"""

from std.memory import bitcast
from std.random import rand, random_float64, seed
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major

from linalg.utils_gpu import MatmulConfig
from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0, matmul_gpu_qint4, multistage_gemm_q
from std.utils.index import Index


comptime GS = 32  # Q4_0 group
comptime GB = 2 + GS // 2  # Q4_0 block bytes: fp16 scale + 16 bytes of nibbles
comptime ITERS = 20


def q4_0_value(blk: UnsafePointer[UInt8, _], j: Int) -> Float32:
    """Element j (0..31) of one Q4_0 block: (nibble - 8) * scale."""
    var u = UInt16(blk[0]) | (UInt16(blk[1]) << 8)  # little-endian fp16 scale
    var d = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](u)).cast[DType.float32]()
    var byte = blk[2 + (j % 16)]
    var nib = (byte & 0xF) if j < 16 else (byte >> 4)
    return (Float32(Int(nib)) - 8.0) * d[0]


comptime DEFAULT_Q = MatmulConfig[DType.bfloat16, DType.uint8, DType.bfloat16, True](
    block_tile_shape=Index(128, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=5,
    num_k_partitions=1,
    num_warp_k_partitions=1,
)


def bench_shape[N: Int, K: Int](ctx: DeviceContext, count: Int, mut total: List[Float64]) raises:
    comptime KB = (K // GS) * GB
    # ── weights: valid Q4_0 blocks, scales ~1e-3 ───────────────────────────
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
    ctx.synchronize()
    var bp = b_h.unsafe_ptr()
    rand[DType.uint8](bp, N * KB, min=0, max=255)  # integer rand defaults to [0, 1)
    for blk in range(N * (K // GS)):
        var sc = SIMD[DType.float16, 1](Float16(0.001 + 0.001 * random_float64()))
        var u = bitcast[DType.uint16, 1](sc)
        bp[blk * GB] = UInt8(u & 0xFF)
        bp[blk * GB + 1] = UInt8(u >> 8)
    var b_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    var bpk_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    ctx.enqueue_copy(b_d, b_h)
    var b_tt = TileTensor(b_d, row_major(Coord(Idx[N], Idx[KB])))
    var bpk_tt = TileTensor(bpk_d, row_major(Coord(Idx[N], Idx[KB])))
    gpu_qint4_repack_Q4_0["gpu"](b_tt.as_immut(), bpk_tt, ctx)
    ctx.synchronize()

    print(t"N={N} K={K} (x{count})")
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

        comptime if N == 4096 and K == 4096:  # tuned configs do not compile (header)
            multistage_gemm_q[group_size=GS, pack_factor=8, config=DEFAULT_Q](
                c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(), bpk_tt.as_immut().to_layout_tensor(), DEFAULT_Q, ctx
            )
        else:
            matmul_gpu_qint4[GS, "gpu"](c_tt, a_tt.as_immut(), bpk_tt.as_immut(), ctx)
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()
        # 8 columns against the host dequantize-and-multiply
        var err = Float64(0)
        var mag = Float64(0)
        for n in range(8):
            for m in range(M):
                var acc = Float64(0)
                for g in range(K // GS):
                    var blk = bp + (n * (K // GS) + g) * GB
                    for j in range(GS):
                        acc += Float64(a_h.unsafe_ptr()[m * K + g * GS + j].cast[DType.float32]()) * Float64(q4_0_value(blk, j))
                var got = Float64(c_h.unsafe_ptr()[m * N + n].cast[DType.float32]())
                err = max(err, abs(got - acc))
                mag = max(mag, abs(acc))

        for _ in range(3):
            comptime if N == 4096 and K == 4096:  # tuned configs do not compile (header)
                multistage_gemm_q[group_size=GS, pack_factor=8, config=DEFAULT_Q](
                    c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(), bpk_tt.as_immut().to_layout_tensor(), DEFAULT_Q, ctx
                )
            else:
                matmul_gpu_qint4[GS, "gpu"](c_tt, a_tt.as_immut(), bpk_tt.as_immut(), ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            comptime if N == 4096 and K == 4096:  # tuned configs do not compile (header)
                multistage_gemm_q[group_size=GS, pack_factor=8, config=DEFAULT_Q](
                    c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(), bpk_tt.as_immut().to_layout_tensor(), DEFAULT_Q, ctx
                )
            else:
                matmul_gpu_qint4[GS, "gpu"](c_tt, a_tt.as_immut(), bpk_tt.as_immut(), ctx)
        ctx.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 1e3 / ITERS
        total[i] += Float64(count) * us
        var tf = 2.0 * Float64(M) * Float64(N) * Float64(K) / (us * 1e-6) / 1e12
        print("   M=", M, " ", Int(us), "us/call ", Float64(Int(tf * 100)) / 100, "TF/s  rel err", err / mag)
    # ⚠ `bp` is a raw pointer into b_h: without this the host buffer dies at its last use (the copy to the device)
    # and the reference reads whatever the next host allocation (the activations) put there.
    _ = b_h^


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    print("MAX matmul_gpu_qint4 (Q4_0, group 32) at Kev-9B shapes;", ctx.name())
    var total: List[Float64] = [0.0, 0.0, 0.0]
    bench_shape[4096, 12288](ctx, 32, total)  # mlp.down_proj
    bench_shape[12288, 4096](ctx, 64, total)  # mlp.gate_proj + up_proj
    bench_shape[8192, 4096](ctx, 32, total)  # linear_attn.in_proj_qkv + self_attn.q_proj
    bench_shape[4096, 4096](ctx, 56, total)  # in_proj_z + out_proj + o_proj
    bench_shape[1024, 4096](ctx, 16, total)  # k_proj + v_proj
    print("per decision (248 linears, passes M=48/218/42):", Int((total[0] + total[1] + total[2]) / 1e3), "ms")
