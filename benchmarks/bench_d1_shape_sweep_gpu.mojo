# +--------------------------------------------------------------------------+ #
# | d1's forward and its GEMMs against the token count: where the cliffs are
# +--------------------------------------------------------------------------+ #
"""Alone on the Orin, d1 answered requests of 814-819 tokens in 444-462 ms and
of 833-836 tokens in 341-349 ms (`bench_d1_replay_gpu`, docs/NN_MODEL_PORTS_PLAN.md
§12.8.7, L1): about 100 ms more for FEWER tokens. This finds which token
counts are slow, and which operation makes them so.

    pixi run -e nvidia mojo build -I . -D LFM3_BF16=1 benchmarks/bench_d1_shape_sweep_gpu.mojo -o build/bench_d1_sweep
    ./build/bench_d1_sweep [from] [to] [step] [reps] [d1 dir]

1. `forward`: d1's backbone over `t` tokens (one id repeated, no groups: a
   plain causal row; the time does not depend on the ids), synchronised.
2. `gemm`: each linear shape of one layer, `[t, k] @ W[n, k]^T`, bf16 operands
   and fp32 out through `cublas_gemm_lp` as the forward calls it, the median of
   `reps` timed in a batch of 10.
Printed as TSV: t, then the median ms of each column.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm_lp
from noeira.llm.d1.system_one import D1


def median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def time_gemm(ctx: DeviceContext, yb: DeviceBuffer[DT], xb: DeviceBuffer[DType.bfloat16], wb: DeviceBuffer[DType.bfloat16],
              m: Int, n: Int, k: Int, split: Int, reps: Int) raises -> Float64:
    """`[m, k] @ W[n, k]^T` as one call (`split` 0) or as rows [0, split) then [split, m), median ms of a batch of 10."""
    var g = List[Float64]()
    for r in range(reps + 1):
        var t0 = perf_counter_ns()
        for _ in range(10):
            if split <= 0 or split >= m:
                cublas_gemm_lp[False, True, DType.bfloat16, DT](ctx, yb, xb, wb, m, n, k, 0.0)
            else:
                cublas_gemm_lp[False, True, DType.bfloat16, DT](ctx, yb, xb, wb, split, n, k, 0.0)
                var x2 = xb.create_sub_buffer[DType.bfloat16](split * k, (m - split) * k)
                var y2 = yb.create_sub_buffer[DT](split * n, (m - split) * n)
                cublas_gemm_lp[False, True, DType.bfloat16, DT](ctx, y2, x2, wb, m - split, n, k, 0.0)
        ctx.synchronize()
        if r > 0:
            g.append(Float64(perf_counter_ns() - t0) / 1e7)
    return median(g^)


def split_mode() raises:
    """`split`: the w1/w3 shape in its slow bands, one call against two."""
    var ctx = DeviceContext()
    var n = 7680
    var k = 2048
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](2048 * k)
    var wb = ctx.enqueue_create_buffer[DType.bfloat16](n * k)
    var yb = ctx.enqueue_create_buffer[DT](2048 * n)
    xb.enqueue_fill(BFloat16(0.01))
    wb.enqueue_fill(BFloat16(0.01))
    ctx.synchronize()
    print("m\tone\tat_odd256\thalves\tat_256\tat_128")
    var ms: List[Int] = [288, 304, 320, 776, 800, 816, 819, 832, 836, 848, 1228, 1296, 1312, 1344, 1488, 1808, 1840]
    for m in ms:
        var base = ((m - 256) // 512) * 512 + 256      # the odd multiple of 256 at or below m
        var h = ((m // 2 + 7) // 8) * 8
        print(String(m) + "\t" + String(round(time_gemm(ctx, yb, xb, wb, m, n, k, 0, 5), 3))
              + "\t" + String(round(time_gemm(ctx, yb, xb, wb, m, n, k, base, 5), 3))
              + "\t" + String(round(time_gemm(ctx, yb, xb, wb, m, n, k, h, 5), 3))
              + "\t" + String(round(time_gemm(ctx, yb, xb, wb, m, n, k, (m // 256) * 256, 5), 3))
              + "\t" + String(round(time_gemm(ctx, yb, xb, wb, m, n, k, (m // 128) * 128, 5), 3)))


def main() raises:
    var args = argv()
    if len(args) > 1 and String(args[1]) == "split":
        split_mode()
        return
    var lo = Int(String(args[1])) if len(args) > 1 else 760
    var hi = Int(String(args[2])) if len(args) > 2 else 900
    var step = Int(String(args[3])) if len(args) > 3 else 4
    var reps = Int(String(args[4])) if len(args) > 4 else 5
    var dir = String(args[5]) if len(args) > 5 else getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
    var d1 = D1.load(DeviceContext(), dir)
    var ctx = d1.lm.ctx
    # one layer's linears, [n, k]: the conv's in and out, attention's q / kv / o, the MLP's w1 / w3 and w2
    var ns: List[Int] = [6144, 2048, 2048, 512, 7680, 2048]
    var ks: List[Int] = [2048, 2048, 2048, 2048, 2048, 7680]
    var names: List[String] = ["conv_in", "conv_out", "q_o", "kv", "w1_w3", "w2"]
    var top = hi + 64
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](top * 7680)
    var wb = ctx.enqueue_create_buffer[DType.bfloat16](7680 * 2048)
    var yb = ctx.enqueue_create_buffer[DT](top * 7680)
    xb.enqueue_fill(BFloat16(0.01))
    wb.enqueue_fill(BFloat16(0.01))
    ctx.synchronize()
    var head = String("t\tforward")
    for nm in names:
        head += "\t" + nm
    print(head)
    var t = lo
    while t <= hi:
        var ids = List[Int](length=t, fill=1000)
        var none = List[Int]()
        var fw = List[Float64]()
        for r in range(reps + 1):
            var taps = List[DeviceBuffer[DT]]()
            var t0 = perf_counter_ns()
            _ = d1.lm.forward(ids, none, 0, taps)
            ctx.synchronize()
            if r > 0:
                fw.append(Float64(perf_counter_ns() - t0) / 1e6)
        var line = String(t) + "\t" + String(round(median(fw^), 1))
        for i in range(len(ns)):
            var g = List[Float64]()
            for r in range(reps + 1):
                var t0 = perf_counter_ns()
                for _ in range(10):
                    cublas_gemm_lp[False, True, DType.bfloat16, DT](ctx, yb, xb, wb, t, ns[i], ks[i], 0.0)
                ctx.synchronize()
                if r > 0:
                    g.append(Float64(perf_counter_ns() - t0) / 1e7)
            line += "\t" + String(round(median(g^), 3))
        print(line)
        t += step
