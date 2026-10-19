"""A6 go/no-go: does padding the conv-GEMM N (=OC) to 128 unlock tensor cores
net of the extra FLOPs? (bf16, ResNet-20 conv shapes.)

The conv im2col GEMM is `out[M, OC] = col[M, COL] @ W[OC, COL]ᵀ` (M = BS·OH·OW,
COL = IC·K²). The matmul dispatch only routes to the tiled tensor-core kernel
when `N%128==0 AND K%32==0 AND K>=128 AND M>1` — and conv OC ∈ {16,32,64} fails
`N%128`, so conv GEMMs run as fp32/CUDA-core fallbacks even in bf16. A6 pads
N=OC→128 (zero-padded weight) to flip the dispatch; the cost is 128/OC× the GEMM
FLOPs. This bench times bf16 GEMM at N=OC (unpadded) vs N=128 (padded) for the
real ResNet-20 conv shapes so we know if padding is a NET win BEFORE changing the
shared conv path.

Tensor-core eligibility ALSO needs K=COL to satisfy `K%32==0 && K>=128`:
  stem  [3->16]  COL=27   -> K<128, never eligible (padding won't help)
  16ch  [16->16] COL=144  -> 144%32=16, NOT eligible even padded
  32ch  [32->32] COL=288  -> eligible after N->128 (4x FLOPs: 32->128)
  64ch  [64->64] COL=576  -> eligible after N->128 (2x FLOPs: 64->128)
So expect a win only on 32/64ch (and net only if tensor-core speed > the FLOP mult).

NVIDIA-only for real numbers (Apple has no bf16 tensor cores -> padded is just
~2-4x slower from the extra FLOPs, no offset; on Apple this only checks it runs).

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/amp/a6_conv_gemm_pad_bench.mojo
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.core.tensor import Tensor, TensorImpl

comptime BF16 = DType.bfloat16
comptime WARM = 10
comptime R = 100


# One conv-GEMM shape: bf16 `out[M,N] = col[M,K] @ W[N,K]ᵀ`, timed at N=OC vs N=128.
# Operands are device-allocated (content irrelevant to GEMM timing).
def bench_shape[M: Int, OC: Int, COL: Int, label: StaticString](
    ctx: DeviceContext
) raises:
    comptime K = COL
    comptime NPAD = 128
    # bf16 operands (content irrelevant to timing).
    var col = TensorImpl[BF16]()
    col.ensure_gpu(ctx, M * K)
    var w = TensorImpl[BF16]()
    w.ensure_gpu(ctx, OC * K)
    var wp = TensorImpl[BF16]()
    wp.ensure_gpu(ctx, NPAD * K)
    var o = TensorImpl[BF16]()
    o.ensure_gpu(ctx, M * OC)
    var op = TensorImpl[BF16]()
    op.ensure_gpu(ctx, M * NPAD)

    var colv = TileTensor(col.dev.value(), row_major[M, K]())
    var wv = TileTensor(w.dev.value(), row_major[OC, K]())
    var wpv = TileTensor(wp.dev.value(), row_major[NPAD, K]())
    var ov = TileTensor(o.dev.value(), row_major[M, OC]())
    var opv = TileTensor(op.dev.value(), row_major[M, NPAD]())

    comptime elig = (K % 32 == 0) and (K >= 128)
    # ---- unpadded N=OC ----
    for _ in range(WARM):
        max_matmul[transpose_b=True, target="gpu"](ov, colv, wv, ctx)
    ctx.synchronize()
    var u0 = perf_counter_ns()
    for _ in range(R):
        max_matmul[transpose_b=True, target="gpu"](ov, colv, wv, ctx)
    ctx.synchronize()
    var u_us = Float64(perf_counter_ns() - u0) / Float64(R) / 1000.0
    # ---- padded N=128 ----
    for _ in range(WARM):
        max_matmul[transpose_b=True, target="gpu"](opv, colv, wpv, ctx)
    ctx.synchronize()
    var p0 = perf_counter_ns()
    for _ in range(R):
        max_matmul[transpose_b=True, target="gpu"](opv, colv, wpv, ctx)
    ctx.synchronize()
    var p_us = Float64(perf_counter_ns() - p0) / Float64(R) / 1000.0

    print(
        label, " M=", M, " OC=", OC, " COL=", COL,
        " | K-gate eligible=", elig,
        " | N=OC:", u_us, "us  N=128:", p_us, "us  speedup:",
        u_us / p_us, "x  (FLOP mult", Float64(NPAD) / Float64(OC), "x)",
    )


def main() raises:
    var ctx = DeviceContext()
    comptime BATCH = 100
    print("A6 conv-GEMM pad bench (bf16). speedup>1 => padding to 128 WINS.")
    print("=" * 70)
    # ResNet-20 conv shapes (BATCH=100, forward im2col GEMM).
    bench_shape[BATCH * 32 * 32, 16, 16 * 9, "16ch@32x32"](ctx)  # K=144, not elig
    bench_shape[BATCH * 16 * 16, 32, 32 * 9, "32ch@16x16"](ctx)  # K=288, elig
    bench_shape[BATCH * 8 * 8, 64, 64 * 9, "64ch@8x8  "](ctx)    # K=576, elig
    # A bonus wide shape (the EZv2 dynamics 3x3 [80->64], COL=720) for reference.
    bench_shape[BATCH * 6 * 6, 64, 80 * 9, "ez-dyn3x3 "](ctx)    # K=720, elig
    print("=" * 70)
    print("If 32/64ch speedup>1 on NVIDIA, A6 (pad OC->128) is worth wiring into")
    print("conv2d (gated on OC<128 && COL%32==0 && COL>=128).")
