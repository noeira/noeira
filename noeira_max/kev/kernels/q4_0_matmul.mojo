# +--------------------------------------------------------------------------+ #
# | noeira_q4_0_matmul — MAX's int4 tensor-core GEMM, tile picked by M
# +--------------------------------------------------------------------------+ #
"""`c[M, N] (bf16) = a[M, K] (bf16) @ dequant(b)^T` for Q4_0 weights already
repacked by MAX's `GGUF_gpu_repack_q4_0` op.

Why not MAX's own `qmatmul_b4_g32` graph op: it dispatches through
`matmul_gpu_qint4`, whose TUNED 4096x4096 configs do not instantiate in MAX
26.6 (block K 128 over warp K 32 — sm_87 and sm_80 alike), and in a graph K
and N are static, so every 4096x4096 linear would hit them. Its other shapes
take a 128x128 tile that wastes most of a small-M call. This op calls the
same `multistage_gemm_q` with MAX's own 64x128x32 / warp-K 4 config for
M <= 64 (1.8x the default on the Orin at M = 48) and the default above
(noeira_max/qmatmul/sweep_qint4_small_m.mojo, noeira-docs/ORIN_KEV_SPEEDUP.md).
"""

import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from std.utils.index import Index

from linalg.utils_gpu import MatmulConfig
from quantization.qmatmul_gpu import multistage_gemm_q


comptime SMALL_M = MatmulConfig[DType.bfloat16, DType.uint8, DType.bfloat16, True](
    block_tile_shape=Index(64, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=4,
    num_k_partitions=1,
    num_warp_k_partitions=4,
)
comptime LARGE_M = MatmulConfig[DType.bfloat16, DType.uint8, DType.bfloat16, True](
    block_tile_shape=Index(128, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=5,
    num_k_partitions=1,
    num_warp_k_partitions=1,
)


@extensibility.register("noeira_q4_0_matmul")
struct NoeiraQ4_0Matmul:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        var cl = c.to_tile_tensor[.int64]().to_layout_tensor()
        var al = a.to_tile_tensor[.int64]().to_layout_tensor()
        var bl = b.to_tile_tensor[.int64]().to_layout_tensor()
        if a.dim_size(0) <= 64:
            multistage_gemm_q[group_size=32, pack_factor=8, config=SMALL_M](cl, al, bl, SMALL_M, ctx)
        else:
            multistage_gemm_q[group_size=32, pack_factor=8, config=LARGE_M](cl, al, bl, LARGE_M, ctx)
