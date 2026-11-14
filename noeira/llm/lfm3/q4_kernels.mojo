# +--------------------------------------------------------------------------+ #
# | LFM3 4-bit weights (llama.cpp's Q4_0) — the decode GEMV
# +--------------------------------------------------------------------------+ #
"""Weights in llama.cpp's Q4_0 (docs/NN_MODEL_PORTS_PLAN.md §12.8, S2): blocks of
32 weights, each block one scale `d` and 32 four-bit values, `w = d * (q - 8)`.
Liquid ships LFM3-1.7B trained for it (`LFM3-1.7B-QAD-Q4_0.gguf`).

ON THE DEVICE the block's two parts are apart, so that a lane's load is one
aligned 16 bytes: `qs [n, k / 2]` uint8, where byte `i` of block `b` holds
weight `32 b + i` in its low half and weight `32 b + 16 + i` in its high half
(the file's order), and `d [n, k / 32]` fp32.

`q4_gemv_kernel` is `q8_gemv_kernel`'s shape: a warp per 4 weight rows, a lane
per block. The `- 8` is not taken per weight: `sum (q - 8) x = sum q x - 8 sum
x`, and a block's `sum x` is the same for the 4 rows.

⚠ MEASURED 2026-11-14, AND NOT FASTER: NOTHING IS BUILT ON THIS KERNEL.
`benchmarks/bench_q4_gemv_gpu.mojo`, the 153 linears of one decoded token on
the Orin: int8 16.3 ms, 4-bit 17.1 ms for one row of activations (21.6 / 29.5
for two, 39.7 / 58.6 for four). The int8 GEMV runs at the memory's bandwidth
(1.47 GB in 16 ms); 4-bit reads 0.92 GB but pays the unpacking on the CUDA
cores, and that arithmetic, not the bytes, is then its bound. On this GPU
4-bit weights would save memory (0.55 GB), not time. Kept with its bench as
the record of that answer, and as the starting point if a cheaper unpacking
is found.
"""

from max.gpu import thread_idx, block_idx
from max.gpu.primitives import warp

from noeira.nn.constants import DT
from .kernels import Ptr


comptime UPtr = Pointer[Scalar[DType.uint8], MutAnyOrigin]
comptime Q4_BLOCK: Int = 32
"""Weights a scale covers."""
comptime Q4_RPW: Int = 4
"""Weight rows a warp computes (the activations it loads serve all of them)."""


def q4_gemv_kernel[NS: Int](x: Ptr, qs: UPtr, d: Ptr, y: Ptr, n_n: Int64, k_n: Int64):
    """`y[s, r] = sum_b d[r, b] * sum_{j in block b} (q[r, j] - 8) x[s, j]` for
    the `NS` rows of `x [NS, k]`. `k` a multiple of 32; `n` a multiple of 4.
    Grid `ceil(n / (8 Q4_RPW))`, 256 threads a block."""
    var tid = Int(thread_idx.x)
    var row0 = (Int(block_idx.x) * 8 + (tid >> 5)) * Q4_RPW
    var lane = tid & 31
    var n = Int(n_n)
    var k = Int(k_n)
    var kb = k // 2
    var nb = k // Q4_BLOCK
    var acc = SIMD[DT, Q4_RPW * NS](0)
    if row0 < n:
        var b = lane
        while b < nb:
            # the NS activation rows' block: its low and high 16 values, and their sum
            var xlo = SIMD[DT, 16 * NS](0)
            var xhi = SIMD[DT, 16 * NS](0)
            var xsum = SIMD[DT, NS](0)
            comptime for c in range(NS):
                var lo = x.unsafe_load[width=16, alignment=16](c * k + b * Q4_BLOCK)
                var hi = x.unsafe_load[width=16, alignment=16](c * k + b * Q4_BLOCK + 16)
                xlo = xlo.insert[offset=c * 16](lo)
                xhi = xhi.insert[offset=c * 16](hi)
                xsum[c] = (lo + hi).reduce_add()
            comptime for r in range(Q4_RPW):
                var by = qs.unsafe_load[width=16, alignment=16]((row0 + r) * kb + b * 16)
                var wlo = (by & SIMD[DType.uint8, 16](15)).cast[DT]()
                var whi = (by >> SIMD[DType.uint8, 16](4)).cast[DT]()
                var dd = d[unsafe_offset=(row0 + r) * nb + b]
                comptime for c in range(NS):
                    var s = ((wlo * xlo.slice[16, offset=c * 16]()).reduce_add()
                             + (whi * xhi.slice[16, offset=c * 16]()).reduce_add())
                    acc[r * NS + c] += dd * (s - Scalar[DT](8) * xsum[c])
            b += 32
    comptime for r in range(Q4_RPW):
        comptime for c in range(NS):
            var tot = warp.sum(acc[r * NS + c])
            if lane == 0 and row0 < n:
                y[unsafe_offset=c * n + row0 + r] = tot
