"""QLinear (Q4_0) against the fp32 Linear it replaces, at SmolVLA's text shapes.

    pixi run -e jetson mojo run -I . --target-accelerator sm_87 \\
        benchmarks/quant/bench_qlinear_smolvla.mojo          # the Orin
    pixi run -e nvidia mojo run -I . benchmarks/quant/bench_qlinear_smolvla.mojo
    pixi run -e apple  mojo run -I . benchmarks/quant/bench_qlinear_smolvla.mojo

Per shape (q / o 960 -> 960, k / v 960 -> 320, gate / up 960 -> 2560, down
2560 -> 960) and T (135: the prefill's tokens at batch 1; 16), the forward of

    fp32     the frozen `Linear[IN, OUT]` the text tower runs today
    q4       `QLinear[IN, OUT]` end to end (cast + pad, int4 GEMM, slice +
             cast on NVIDIA; dequantize + fp32 GEMM elsewhere), `NN_Q4_CFG`
    gemm_*   on NVIDIA, the int4 GEMM alone in each of the four tile configs
             (`qlinear.mojo`), to pick `NN_Q4_CFG`

and grad_input (`vjp`) of both. Each sample is CALLS calls between two
synchronizations; the time is the median sample over CALLS.

Weight bytes per layer, for the memory side: fp32 4 B per weight (a host and
a device copy in noeira), Q4_0 0.5625 B, padded for MAX's repack.
"""

from std.sys import has_nvidia_gpu_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.quant.qlinear import (
    QLinear, Q4_CFG, Q4_PATH, Q4_SMALL, Q4_SMALL2, Q4_SMALL4, Q4_LARGE,
)

comptime CALLS = 10
comptime SAMPLES = 7
comptime INT4 = has_nvidia_gpu_accelerator() and Q4_PATH != "deq"


def _median(mut ts: List[Float64]) -> Float64:
    sort(ts)
    return ts[len(ts) // 2]


def row[IN: Int, OUT: Int, T: Int](ctx: DeviceContext, label: String) raises:
    var c = Optional(ctx)
    var lin = Linear[IN, OUT].make["gpu", Deterministic](c)
    lin.freeze()
    var w = List[Float32](capacity=OUT * IN)
    for i in range(OUT * IN):
        w.append(Float32(((i * 7919) % 2003) - 1001) * 1e-4)
    var ql = QLinear[IN, OUT].make["gpu", Deterministic](c)
    ql.set_weight["gpu"](w, c)
    var x = Tensor.alloc(T * IN)
    for i in range(T * IN):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT]((i * 7 % 37) - 18)
    x.upload(ctx)
    var g = Tensor.alloc(T * OUT)
    for i in range(T * OUT):
        g.data[i] = Scalar[DT](0.01) * Scalar[DT]((i * 5 % 29) - 14)
    g.upload(ctx)
    var y = Tensor()
    var gx = Tensor()

    var t_lin = List[Float64]()
    var t_q = List[Float64]()
    var b_lin = List[Float64]()
    var b_q = List[Float64]()
    var t_cfg = List[List[Float64]]()
    for _ in range(4):
        t_cfg.append(List[Float64]())
    # warm-up: every path once
    lin.forward["gpu", T](TensorRefs[1](x), y, c)
    ql.forward["gpu", T](TensorRefs[1](x), y, c)
    lin.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
    ql.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
    # small4 needs 102,400 B of shared memory per block: the Orin has 164 KB,
    # consumer Blackwell's per-block opt-in is 101,376 B and refuses it.
    var small4 = ctx.default_device_info.shared_memory_per_multiprocessor > 102400
    comptime if INT4:
        ql._gemm_q4[T, Q4_SMALL](ctx)
        ql._gemm_q4[T, Q4_SMALL2](ctx)
        if small4:
            ql._gemm_q4[T, Q4_SMALL4](ctx)
        ql._gemm_q4[T, Q4_LARGE](ctx)
    ctx.synchronize()
    for _ in range(SAMPLES):
        var t0 = perf_counter_ns()
        for _ in range(CALLS):
            lin.forward["gpu", T](TensorRefs[1](x), y, c)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for _ in range(CALLS):
            ql.forward["gpu", T](TensorRefs[1](x), y, c)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(CALLS):
            lin.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
        ctx.synchronize()
        var t3 = perf_counter_ns()
        for _ in range(CALLS):
            ql.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
        ctx.synchronize()
        var t4 = perf_counter_ns()
        t_lin.append(Float64(t1 - t0) / 1e3 / CALLS)
        t_q.append(Float64(t2 - t1) / 1e3 / CALLS)
        b_lin.append(Float64(t3 - t2) / 1e3 / CALLS)
        b_q.append(Float64(t4 - t3) / 1e3 / CALLS)
        comptime if INT4:
            var s0 = perf_counter_ns()
            for _ in range(CALLS):
                ql._gemm_q4[T, Q4_SMALL](ctx)
            ctx.synchronize()
            var s1 = perf_counter_ns()
            for _ in range(CALLS):
                ql._gemm_q4[T, Q4_SMALL2](ctx)
            ctx.synchronize()
            var s2 = perf_counter_ns()
            if small4:
                for _ in range(CALLS):
                    ql._gemm_q4[T, Q4_SMALL4](ctx)
                ctx.synchronize()
            var s3 = perf_counter_ns()
            for _ in range(CALLS):
                ql._gemm_q4[T, Q4_LARGE](ctx)
            ctx.synchronize()
            var s4 = perf_counter_ns()
            t_cfg[0].append(Float64(s1 - s0) / 1e3 / CALLS)
            t_cfg[1].append(Float64(s2 - s1) / 1e3 / CALLS)
            t_cfg[2].append(Float64(s3 - s2) / 1e3 / CALLS)
            t_cfg[3].append(Float64(s4 - s3) / 1e3 / CALLS)
    var fl = _median(t_lin)
    var fq = _median(t_q)
    var bl = _median(b_lin)
    var bq = _median(b_q)
    var line = String("  ") + label + " [" + String(IN) + " -> " + String(OUT) + "] T " + String(T)
    line += "   fwd fp32 " + String(Int(fl)) + " us, q4 " + String(Int(fq)) + " us (x" + String(fq / fl)[byte=0:4] + ")"
    line += "   dx fp32 " + String(Int(bl)) + " us, q4 " + String(Int(bq)) + " us (x" + String(bq / bl)[byte=0:4] + ")"
    print(line)
    comptime if INT4:
        print("      int4 GEMM alone: small", Int(_median(t_cfg[0])), "us | small2",
              Int(_median(t_cfg[1])), "us | small4",
              String(Int(_median(t_cfg[2]))) + " us" if small4 else String("(refused here)"),
              "| large", Int(_median(t_cfg[3])), "us")


def main() raises:
    var ctx = DeviceContext()
    print("QLinear (Q4_0) vs fp32 Linear, SmolVLA text shapes, on", ctx.name())
    print("  int4 GEMM:", INT4, "| NN_Q4_CFG =", Q4_CFG, "| median of", SAMPLES,
          "samples of", CALLS, "calls")
    row[960, 960, 135](ctx, String("q / o   "))
    row[960, 320, 135](ctx, String("k / v   "))
    row[960, 2560, 135](ctx, String("gate/up "))
    row[2560, 960, 135](ctx, String("down    "))
    row[960, 960, 16](ctx, String("q / o   "))
    row[2560, 960, 16](ctx, String("down    "))
    var fp32_b = 4 * (2 * 960 * 960 + 2 * 960 * 320 + 3 * 960 * 2560)
    var q4_b = (
        2 * 1024 * 1024 + 2 * 384 * 1024 + 2 * 2560 * 1024 + 1024 * 3072
    ) * 9 // 16
    print("  weight bytes per text layer: fp32", fp32_b, "(one copy) | Q4_0 padded",
          q4_b, "(x" + String(Float64(fp32_b) / Float64(q4_b))[byte=0:4] + " smaller)")
