"""What a LoRA adapter costs per linear, next to the frozen base it adapts.

    pixi run -e apple  mojo run -I . benchmarks/peft/bench_lora_overhead.mojo
    pixi run -e nvidia mojo run -I . benchmarks/peft/bench_lora_overhead.mojo

A frozen `Linear[IN, OUT]` (the LoRA regime: no dW) against the same layer plus
a trainable `LoRAAdapter` of rank R, at SmolVLA's text-tower shapes
(q/o 960 -> 960, gate/up 960 -> 2560, down 2560 -> 960), R in {8, 16, 32, 64}
and T in {64, 960, 7680} tokens per call (one observation's
expert suffix, one prefix, a batch of eight prefixes).

    forward   base `forward`       vs  base `forward` + `forward_add`
    backward  base `vjp` (dX only) vs  base `vjp` + `vjp_add` (dX, dA, dB)

Each sample is CALLS calls between two synchronizations, and the reported
time is the median sample over CALLS: a synchronized single launch on Metal
costs a ~250 us round trip that would otherwise swamp the T = 64 rows.

The adapter's arithmetic is R / OUT + R / IN of the base's forward, and its
backward adds dA and dB (two GEMMs contracting over T) to the frozen base's
single dX GEMM, so the ratio is expected to grow with R and to be worst where
the base is cheapest. On NVIDIA this runs the adapter's generic path (MAX's
matmul at N = R_PAD), not a cuBLAS one: read its rows as that path's cost.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.peft.lora import LoRAAdapter

comptime CALLS = 10
comptime SAMPLES = 7


def _median(mut ts: List[Float64]) -> Float64:
    sort(ts)
    return ts[len(ts) // 2]


def row[IN: Int, OUT: Int, R: Int, T: Int](ctx: DeviceContext) raises:
    var c = Optional(ctx)
    var base = Linear[IN, OUT].make["gpu", Deterministic](c)
    base.freeze()
    var a = LoRAAdapter[IN, OUT, R].make["gpu"](c)
    for j in range(R):
        for o in range(OUT):
            a.lora_B.val.data[j * OUT + o] = Scalar[DT](
                Float64(((j * 5 + o * 3) % 7) - 3) * 0.01
            )
    a.lora_B.val.upload_resident(ctx)
    var x = Tensor.alloc(T * IN)
    for i in range(T * IN):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT]((i * 7 % 37) - 18)
    var g = Tensor.alloc(T * OUT)
    for i in range(T * OUT):
        g.data[i] = Scalar[DT](0.003) * Scalar[DT]((i * 5 % 23) - 11)
    x.upload(ctx)
    g.upload(ctx)
    var y = Tensor.alloc_gpu(ctx, T * OUT)
    var gx = Tensor.alloc_gpu(ctx, T * IN)

    # warm up both paths (allocations, padded weight copies, kernel loads)
    base.forward["gpu", T](TensorRefs[1](x), y, c)
    a.forward_add["gpu", T](TensorRefs[1](x), y, c)
    base.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
    a.vjp_add["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
    ctx.synchronize()

    var tf = List[Float64]()
    var tfa = List[Float64]()
    var tb = List[Float64]()
    var tba = List[Float64]()
    for _ in range(SAMPLES):
        var t0 = perf_counter_ns()
        for _ in range(CALLS):
            base.forward["gpu", T](TensorRefs[1](x), y, c)
        ctx.synchronize()
        tf.append(Float64(perf_counter_ns() - t0) / 1e3 / Float64(CALLS))
        t0 = perf_counter_ns()
        for _ in range(CALLS):
            base.forward["gpu", T](TensorRefs[1](x), y, c)
            a.forward_add["gpu", T](TensorRefs[1](x), y, c)
        ctx.synchronize()
        tfa.append(Float64(perf_counter_ns() - t0) / 1e3 / Float64(CALLS))
        t0 = perf_counter_ns()
        for _ in range(CALLS):
            base.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
        ctx.synchronize()
        tb.append(Float64(perf_counter_ns() - t0) / 1e3 / Float64(CALLS))
        t0 = perf_counter_ns()
        for _ in range(CALLS):
            base.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
            a.vjp_add["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
        ctx.synchronize()
        tba.append(Float64(perf_counter_ns() - t0) / 1e3 / Float64(CALLS))
    var f = _median(tf)
    var fa = _median(tfa)
    var b = _median(tb)
    var ba = _median(tba)
    print(
        "  ", IN, "->", OUT, "  R ", R, "  T ", T,
        "   fwd ", f, " -> ", fa, " us (x", fa / f, ")",
        "   bwd ", b, " -> ", ba, " us (x", ba / b, ")", sep="",
    )


def shape[IN: Int, OUT: Int](ctx: DeviceContext) raises:
    comptime for ti in range(3):
        comptime T = 64 if ti == 0 else (960 if ti == 1 else 7680)
        row[IN, OUT, 8, T](ctx)
        row[IN, OUT, 16, T](ctx)
        row[IN, OUT, 32, T](ctx)
        row[IN, OUT, 64, T](ctx)


def main() raises:
    with DeviceContext() as ctx:
        print("LoRA adapter cost per linear (frozen base), median of", SAMPLES,
              "samples of", CALLS, "calls")
        shape[960, 960](ctx)
        shape[960, 2560](ctx)
        shape[2560, 960](ctx)
