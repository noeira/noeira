"""W8A8's kernels (`noeira/llm/lfm3/int8_kernels.mojo`) and the int8 GEMM
(`cublas_gemm_i8`) against the host. NVIDIA only (cuBLAS); the Orin.

    pixi run -e jetson build-jetson -D LFM3_BF16=1 tests/llm/test_lfm3_w8a8_gpu.mojo -o build/test_w8a8

On random rows with a few outlier columns (as an LLM's activations have):

1. `q8_act_kernel`: each row's scale is `max |x| / 127` exactly, each value
   `round(x / scale)` exactly (the host's fp32 arithmetic).
2. `q8_rms_kernel`: the norm's output within 1e-5 of the host's, and its
   int8 row from that output, exactly as 1.
3. The chain `q8_act` -> `cublas_gemm_i8` -> `q8_out_kernel`: every int32 sum
   equal to the host's, and every output within 1e-6 (relative to the row's
   largest) of the host's `sum * sx[t] * sw[n]` — the code computes what it
   says.
4. What the quantization costs: the chain against the fp32 product of the
   unquantized operands, relative error over the whole output under 0.1.
   Measured 0.064 on these rows (the Orin, 2026-11-16): one channel in 509 at
   40x the others sets each row's scale, and the rest land on a few levels —
   why d1 keeps the linears whose inputs look like that in bf16
   (`Lfm3Text.load`'s `int8_operator_out`). The bound catches a gross error
   (a scale on the wrong axis is off by its whole size).
5. `q8_swiglu_kernel` (the fused MLP, `Lfm3Text.mlp_w8a8`) against the
   unfused chain on the device — `q8_out` of w1's and w3's int32 sums,
   `swiglu_kernel`, `q8_act` — every int8 value and every scale EQUAL: the
   same floats in the same order.
"""

from std.math import sqrt
from std.memory import bitcast
from std.random import random_float64, seed as set_seed
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm_i8
from noeira.llm.lfm3.kernels import TPB, swiglu_kernel
from noeira.llm.lfm3.int8_kernels import (
    q8_quantize_kernel, q8_act_kernel, q8_rms_kernel, q8_out_kernel, Q8_OUT_W, q8_swiglu_kernel,
)


comptime T = 37
comptime K = 2048
comptime N = 96
comptime FF = 1540
"""The MLP's width in 5: a multiple of 4, not of the 1024 values a block's pass covers."""
comptime EPS: Float32 = 1e-5


def bf16_bits(v: Float32) -> UInt16:
    return bitcast[DType.uint16, 1](v.cast[DType.bfloat16]())


def bits_f32(b: UInt16) -> Float32:
    return bitcast[DType.float32, 1](UInt32(b) << 16)


def host_q(x: Float32, inv: Float32) -> Int:
    var v = (x * inv).__round__()
    return Int(min(max(v, Float32(-127)), Float32(127)))


def main() raises:
    var ctx = DeviceContext()
    set_seed(11)
    var fails = 0
    # ── operands: x [T, K] with outlier columns, w [N, K] bf16, the norm's weight [K] bf16
    var hx = ctx.enqueue_create_host_buffer[DT](T * K)
    var hw = ctx.enqueue_create_host_buffer[DType.uint16](N * K)
    var hg = ctx.enqueue_create_host_buffer[DType.uint16](K)
    ctx.synchronize()
    for r in range(T):
        for j in range(K):
            var v = Float32(random_float64(-1.0, 1.0))
            if j % 509 == 3:
                v *= 40.0  # an outlier channel
            if r == 5:
                v = 0.0    # a zero row: scale 0, q 0
            hx[r * K + j] = v
    for i in range(N * K):
        hw[i] = bf16_bits(Float32(random_float64(-0.05, 0.05)))
    for j in range(K):
        hg[j] = bf16_bits(Float32(random_float64(0.5, 1.5)))
    var x = ctx.enqueue_create_buffer[DT](T * K)
    var w = ctx.enqueue_create_buffer[DType.uint16](N * K)
    var g = ctx.enqueue_create_buffer[DType.uint16](K)
    ctx.enqueue_copy(x, hx)
    ctx.enqueue_copy(w, hw)
    ctx.enqueue_copy(g, hg)
    var wq = ctx.enqueue_create_buffer[DType.int8](N * K)
    var sw = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_function[q8_quantize_kernel](w, wq, sw, Int64(K), Int64(127), grid_dim=N, block_dim=TPB)

    # ── 1. q8_act_kernel
    var xq = ctx.enqueue_create_buffer[DType.int8](T * K)
    var sx = ctx.enqueue_create_buffer[DT](T)
    ctx.enqueue_function[q8_act_kernel](x, xq, sx, Int64(K), grid_dim=T, block_dim=TPB)
    var hxq = ctx.enqueue_create_host_buffer[DType.int8](T * K)
    var hsx = ctx.enqueue_create_host_buffer[DT](T)
    ctx.enqueue_copy(hxq, xq)
    ctx.enqueue_copy(hsx, sx)
    ctx.synchronize()
    var bad_s = 0
    var bad_q = 0
    for r in range(T):
        var m = Float32(0)
        for j in range(K):
            m = max(m, abs(hx[r * K + j]))
        var s = m / Float32(127)
        if hsx[r] != s:
            bad_s += 1
        var inv = Float32(1) / s if s > 0 else Float32(0)
        for j in range(K):
            if Int(hxq[r * K + j]) != host_q(hx[r * K + j], inv):
                bad_q += 1
    print("1. q8_act: scales differ", bad_s, "| values differ", bad_q, "of", T * K)
    if bad_s != 0 or bad_q != 0:
        print("  FAIL q8_act_kernel")
        fails += 1

    # ── 2. q8_rms_kernel
    var y = ctx.enqueue_create_buffer[DT](T * K)
    var yq = ctx.enqueue_create_buffer[DType.int8](T * K)
    var sy = ctx.enqueue_create_buffer[DT](T)
    ctx.enqueue_function[q8_rms_kernel](x, y, yq, sy, g, Int64(K), EPS, grid_dim=T, block_dim=TPB)
    var hy = ctx.enqueue_create_host_buffer[DT](T * K)
    var hyq = ctx.enqueue_create_host_buffer[DType.int8](T * K)
    var hsy = ctx.enqueue_create_host_buffer[DT](T)
    ctx.enqueue_copy(hy, y)
    ctx.enqueue_copy(hyq, yq)
    ctx.enqueue_copy(hsy, sy)
    ctx.synchronize()
    var worst_y = 0.0
    var bad_rs = 0
    var bad_rq = 0
    for r in range(T):
        var ss = 0.0
        for j in range(K):
            ss += Float64(hx[r * K + j]) * Float64(hx[r * K + j])
        var inv = 1.0 / sqrt(ss / Float64(K) + Float64(EPS))
        var m = Float32(0)
        for j in range(K):
            var want = Float64(bits_f32(hg[j])) * Float64(hx[r * K + j]) * inv
            worst_y = max(worst_y, abs(Float64(hy[r * K + j]) - want) / max(abs(want), 1.0))
            m = max(m, abs(hy[r * K + j]))
        var s = m / Float32(127)
        if hsy[r] != s:
            bad_rs += 1
        var qi = Float32(1) / s if s > 0 else Float32(0)
        for j in range(K):
            if Int(hyq[r * K + j]) != host_q(hy[r * K + j], qi):
                bad_rq += 1
    print("2. q8_rms: norm error", worst_y, "| scales differ", bad_rs, "| values differ", bad_rq)
    if worst_y > 1e-5 or bad_rs != 0 or bad_rq != 0:
        print("  FAIL q8_rms_kernel")
        fails += 1

    # ── 3. the chain
    var acc = ctx.enqueue_create_buffer[DType.int32](T * N)
    cublas_gemm_i8(ctx, acc, xq, wq, T, N, K)
    var out = ctx.enqueue_create_buffer[DT](T * N)
    ctx.enqueue_function[q8_out_kernel](
        acc, sx, sw, out, Int64(N), grid_dim=((N + Q8_OUT_W * TPB - 1) // (Q8_OUT_W * TPB), T), block_dim=TPB,
    )
    var hacc = ctx.enqueue_create_host_buffer[DType.int32](T * N)
    var hout = ctx.enqueue_create_host_buffer[DT](T * N)
    var hwq = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    var hsw = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(hacc, acc)
    ctx.enqueue_copy(hout, out)
    ctx.enqueue_copy(hwq, wq)
    ctx.enqueue_copy(hsw, sw)
    ctx.synchronize()
    var bad_acc = 0
    var worst_out = 0.0
    var err2 = 0.0
    var ref2 = 0.0
    for r in range(T):
        var row_top = 0.0
        var want = List[Float64](length=N, fill=0.0)
        for c in range(N):
            var s = 0
            var f = 0.0
            for j in range(K):
                s += Int(hxq[r * K + j]) * Int(hwq[c * K + j])
                f += Float64(hx[r * K + j]) * Float64(bits_f32(hw[c * K + j]))
            if Int(hacc[r * N + c]) != s:
                bad_acc += 1
            want[c] = Float64(s) * Float64(hsx[r]) * Float64(hsw[c])
            row_top = max(row_top, abs(want[c]))
            err2 += (Float64(hout[r * N + c]) - f) ** 2
            ref2 += f * f
        for c in range(N):
            worst_out = max(worst_out, abs(Float64(hout[r * N + c]) - want[c]) / max(row_top, 1e-30))
    print("3. chain: int32 sums differ", bad_acc, "of", T * N, "| output error", worst_out)
    if bad_acc != 0 or worst_out > 1e-6:
        print("  FAIL the int8 chain")
        fails += 1
    var rel = sqrt(err2 / ref2)
    print("4. against the fp32 product: relative error", rel)
    if rel > 0.1:
        print("  FAIL W8A8 error")
        fails += 1

    # ── 5. the fused SwiGLU against the unfused chain
    var hw1 = ctx.enqueue_create_host_buffer[DType.uint16](FF * K)
    var hw3 = ctx.enqueue_create_host_buffer[DType.uint16](FF * K)
    ctx.synchronize()
    for i in range(FF * K):
        hw1[i] = bf16_bits(Float32(random_float64(-0.05, 0.05)))
        hw3[i] = bf16_bits(Float32(random_float64(-0.05, 0.05)))
    var w1 = ctx.enqueue_create_buffer[DType.uint16](FF * K)
    var w3 = ctx.enqueue_create_buffer[DType.uint16](FF * K)
    ctx.enqueue_copy(w1, hw1)
    ctx.enqueue_copy(w3, hw3)
    var q1 = ctx.enqueue_create_buffer[DType.int8](FF * K)
    var q3 = ctx.enqueue_create_buffer[DType.int8](FF * K)
    var s1 = ctx.enqueue_create_buffer[DT](FF)
    var s3 = ctx.enqueue_create_buffer[DT](FF)
    ctx.enqueue_function[q8_quantize_kernel](w1, q1, s1, Int64(K), Int64(127), grid_dim=FF, block_dim=TPB)
    ctx.enqueue_function[q8_quantize_kernel](w3, q3, s3, Int64(K), Int64(127), grid_dim=FF, block_dim=TPB)
    var acc1 = ctx.enqueue_create_buffer[DType.int32](T * FF)
    var acc3 = ctx.enqueue_create_buffer[DType.int32](T * FF)
    cublas_gemm_i8(ctx, acc1, yq, q1, T, FF, K)
    cublas_gemm_i8(ctx, acc3, yq, q3, T, FF, K)
    var og = ((FF + Q8_OUT_W * TPB - 1) // (Q8_OUT_W * TPB), T)
    var a1 = ctx.enqueue_create_buffer[DT](T * FF)
    var a3 = ctx.enqueue_create_buffer[DT](T * FF)
    ctx.enqueue_function[q8_out_kernel](acc1, sy, s1, a1, Int64(FF), grid_dim=og, block_dim=TPB)
    ctx.enqueue_function[q8_out_kernel](acc3, sy, s3, a3, Int64(FF), grid_dim=og, block_dim=TPB)
    ctx.enqueue_function[swiglu_kernel](a1, a3, a1, Int64(T * FF), grid_dim=(T * FF + TPB - 1) // TPB, block_dim=TPB)
    var hq_ref = ctx.enqueue_create_buffer[DType.int8](T * FF)
    var hs_ref = ctx.enqueue_create_buffer[DT](T)
    ctx.enqueue_function[q8_act_kernel](a1, hq_ref, hs_ref, Int64(FF), grid_dim=T, block_dim=TPB)
    var hq = ctx.enqueue_create_buffer[DType.int8](T * FF)
    var hs = ctx.enqueue_create_buffer[DT](T)
    ctx.enqueue_function[q8_swiglu_kernel](acc1, acc3, sy, s1, s3, hq, hs, Int64(FF), grid_dim=T, block_dim=TPB)
    var h_hq_ref = ctx.enqueue_create_host_buffer[DType.int8](T * FF)
    var h_hq = ctx.enqueue_create_host_buffer[DType.int8](T * FF)
    var h_hs_ref = ctx.enqueue_create_host_buffer[DT](T)
    var h_hs = ctx.enqueue_create_host_buffer[DT](T)
    ctx.enqueue_copy(h_hq_ref, hq_ref)
    ctx.enqueue_copy(h_hq, hq)
    ctx.enqueue_copy(h_hs_ref, hs_ref)
    ctx.enqueue_copy(h_hs, hs)
    ctx.synchronize()
    var bad_hs = 0
    var bad_hq = 0
    var nonzero = 0
    for r in range(T):
        if h_hs[r] != h_hs_ref[r]:
            bad_hs += 1
        if h_hs[r] > 0:
            nonzero += 1
    for i in range(T * FF):
        if h_hq[i] != h_hq_ref[i]:
            bad_hq += 1
    print("5. fused SwiGLU vs the unfused chain: scales differ", bad_hs, "| values differ", bad_hq, "of", T * FF,
          "| rows with a nonzero scale", nonzero)
    if bad_hs != 0 or bad_hq != 0 or nonzero < T - 1:
        print("  FAIL q8_swiglu_kernel")
        fails += 1
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
