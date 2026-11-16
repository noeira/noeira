# +--------------------------------------------------------------------------+ #
# | W8A8 go / no-go: the int8 tensor-core GEMM against bf16 on LFM3's shapes
# +--------------------------------------------------------------------------+ #
"""L5 (docs/NN_MODEL_PORTS_PLAN.md §12.8.7): d1 is prefill only, and its
linears run as bf16 GEMMs (`cublas_gemm_lp`). Ampere's int8 tensor cores have
twice bf16's peak. Before any kernel or quantization work, this measures what
cuBLAS's int8 GEMM (`cublas_gemm_i8`, int32 out) gives on the shapes of one
LFM3 layer, against the bf16 GEMM the backbone runs today.

    pixi run -e jetson build-jetson -D LFM3_BF16=1 benchmarks/bench_w8a8_gemm_gpu.mojo -o build/bench_w8a8
    ./build/bench_w8a8 [reps] [t ...]

First an exactness check (int32 sums against the host's, random int8). Then,
per row count `t` and linear `[n, k]`: the median ms of the bf16 GEMM (fp32
out), the int8 GEMM (int32 out), and the int8 weight dequantized to bf16 (the
reply model's chunk path pays that before its bf16 GEMM, `_linear_q8`).
Last, the sum over d1's 24 layers (13 conv, 11 attention) for each `t`.
"""

from std.random import random_si64, seed as set_seed
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm_lp, cublas_gemm_i8
from noeira.llm.lfm3.int8_kernels import q8_dequant_bf16_kernel
from noeira.llm.lfm3.kernels import TPB


def median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def exact_check(ctx: DeviceContext) raises:
    """`cublas_gemm_i8` against the host on random int8 values, every sum equal."""
    set_seed(7)
    var m = 37
    var n = 64
    var k = 2048
    var ha = ctx.enqueue_create_host_buffer[DType.int8](m * k)
    var hb = ctx.enqueue_create_host_buffer[DType.int8](n * k)
    ctx.synchronize()
    for i in range(m * k):
        ha[i] = Int8(Int(random_si64(-127, 127)))
    for i in range(n * k):
        hb[i] = Int8(Int(random_si64(-127, 127)))
    var a = ctx.enqueue_create_buffer[DType.int8](m * k)
    var b = ctx.enqueue_create_buffer[DType.int8](n * k)
    var y = ctx.enqueue_create_buffer[DType.int32](m * n)
    ctx.enqueue_copy(a, ha)
    ctx.enqueue_copy(b, hb)
    cublas_gemm_i8(ctx, y, a, b, m, n, k)
    var hy = ctx.enqueue_create_host_buffer[DType.int32](m * n)
    ctx.enqueue_copy(hy, y)
    ctx.synchronize()
    var bad = 0
    for r in range(m):
        for c in range(n):
            var s = 0
            for j in range(k):
                s += Int(ha[r * k + j]) * Int(hb[c * k + j])
            if Int(hy[r * n + c]) != s:
                bad += 1
    print("exact check [" + String(m) + "x" + String(n) + "] K=" + String(k) + ": " + String(bad) + " sums differ")
    if bad != 0:
        raise Error("cublas_gemm_i8 is not exact")


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 7
    var ts = List[Int]()
    for i in range(2, len(args)):
        ts.append(Int(String(args[i])))
    if len(ts) == 0:
        ts = [64, 327, 819, 836, 1228]
    var ctx = DeviceContext()
    print("device: " + ctx.name())
    exact_check(ctx)
    # one layer's linears, [n, k]: the conv's in and out, attention's q / o, k / v, the MLP's w1 / w3 and w2
    var names: List[String] = ["conv_in", "conv_out", "q_o", "kv", "w1_w3", "w2"]
    var ns: List[Int] = [6144, 2048, 2048, 512, 7680, 2048]
    var ks: List[Int] = [2048, 2048, 2048, 2048, 2048, 7680]
    # calls of each per forward: 13 conv layers (in, out), 11 attention (q, o; k, v), 24 MLPs (w1, w3; w2)
    var per: List[Int] = [13, 13, 22, 22, 48, 24]
    var top = 1
    for t in ts:
        top = max(top, t)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](top * 7680)
    var wb = ctx.enqueue_create_buffer[DType.bfloat16](7680 * 2048)
    var yf = ctx.enqueue_create_buffer[DT](top * 7680)
    var xq = ctx.enqueue_create_buffer[DType.int8](top * 7680)
    var wq = ctx.enqueue_create_buffer[DType.int8](7680 * 2048)
    var yi = ctx.enqueue_create_buffer[DType.int32](top * 7680)
    var sc = ctx.enqueue_create_buffer[DT](7680)
    xb.enqueue_fill(BFloat16(0.01))
    wb.enqueue_fill(BFloat16(0.01))
    xq.enqueue_fill(Int8(3))
    wq.enqueue_fill(Int8(-5))
    sc.enqueue_fill(Float32(0.01))
    ctx.synchronize()
    print("t\tlinear\tbf16_ms\tint8_ms\tratio\tdequant_ms\tbf16_TF\tint8_TOPS")
    for t in ts:
        var sum_b = 0.0
        var sum_i = 0.0
        var sum_d = 0.0
        for i in range(len(ns)):
            var n = ns[i]
            var k = ks[i]
            var gb = List[Float64]()
            var gi = List[Float64]()
            var gd = List[Float64]()
            for r in range(reps + 1):
                var t0 = perf_counter_ns()
                for _ in range(10):
                    cublas_gemm_lp[False, True, DType.bfloat16, DT](ctx, yf, xb, wb, t, n, k, 0.0)
                ctx.synchronize()
                var t1 = perf_counter_ns()
                for _ in range(10):
                    cublas_gemm_i8(ctx, yi, xq, wq, t, n, k)
                ctx.synchronize()
                var t2 = perf_counter_ns()
                for _ in range(10):
                    ctx.enqueue_function[q8_dequant_bf16_kernel](
                        wq, sc, wb, Int64(k), grid_dim=((k // 16 + TPB - 1) // TPB, n), block_dim=TPB,
                    )
                ctx.synchronize()
                var t3 = perf_counter_ns()
                if r > 0:
                    gb.append(Float64(t1 - t0) / 1e7)
                    gi.append(Float64(t2 - t1) / 1e7)
                    gd.append(Float64(t3 - t2) / 1e7)
            var mb = median(gb^)
            var mi = median(gi^)
            var md = median(gd^)
            var flop = 2.0 * Float64(t) * Float64(n) * Float64(k)
            print(
                String(t) + "\t" + names[i] + "\t" + String(round(mb, 3)) + "\t" + String(round(mi, 3)) + "\t"
                + String(round(mb / mi, 2)) + "\t" + String(round(md, 3)) + "\t" + String(round(flop / mb / 1e9, 1))
                + "\t" + String(round(flop / mi / 1e9, 1))
            )
            sum_b += mb * Float64(per[i])
            sum_i += mi * Float64(per[i])
            sum_d += md * Float64(per[i])
        print(
            String(t) + "\tFORWARD\t" + String(round(sum_b, 1)) + "\t" + String(round(sum_i, 1)) + "\t"
            + String(round(sum_b / sum_i, 2)) + "\t" + String(round(sum_d, 1)) + "\t\t"
        )
