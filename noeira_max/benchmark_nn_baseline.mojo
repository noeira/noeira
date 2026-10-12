"""nn native-Mojo MLP inference baseline — the "why incorporate MAX?" comparison.

Runs the SAME MLP shapes as `benchmark_interop.mojo` through the storage nn GPU
forward pass, so MAX's "MAX device compute" line can be compared directly against
nn's on-device compute. Same device, same dims, same steady-state (input already
on device) regime.

The honest framing for a Mojo caller:
  * MAX path  : compute + H2D + D2H + Python glue (+ ~0 interop floor)   [see interop bench]
  * nn path   : compute only — data is already in Mojo GPU buffers, no Python, no transfer.
So nn's *delivered* latency to a Mojo caller is just the number below; MAX must overcome
its transfer + Python-glue tax (~hundreds of us, see benchmark_interop.mojo) to win.

This file is pure nn (no Python / no max.engine), so plain `mojo run` is fine:
  pixi run -e apple  mojo run -I . noeira_max/benchmark_nn_baseline.mojo
  pixi run -e nvidia mojo run -I . noeira_max/benchmark_nn_baseline.mojo
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs


def f2(x: Float64) -> String:
    var neg = x < 0.0
    var v = -x if neg else x
    var scaled = Int(v * 100.0 + 0.5)
    var whole = scaled // 100
    var frac = scaled % 100
    var fs = String(frac)
    if frac < 10:
        fs = "0" + fs
    var s = String(whole) + "." + fs
    return "-" + s if neg else s


def bench_nn[
    IN: Int, H1: Int, H2: Int, OUT: Int, BATCH: Int
](ctx: DeviceContext, name: String, iters: Int) raises:
    # MLP: Linear+ReLU(IN->H1) -> Linear+ReLU(H1->H2) -> Linear(H2->OUT).
    comptime MLP = Sequential[
        Linear[IN, H1],
        ReLU[H1],
        Linear[H1, H2],
        ReLU[H2],
        Linear[H2, OUT],
    ]
    var net = MLP.make["gpu", Kaiming](Optional(ctx))

    # Resident device input/output (steady-state: data already on device).
    var x = Tensor.alloc(BATCH * IN)
    for i in range(BATCH * IN):
        x.data[i] = Scalar[DT](0.1)
    x.upload(ctx)
    var y = Tensor.alloc(BATCH * OUT)

    # warmup
    for _ in range(50):
        net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
    ctx.synchronize()
    var per = Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0  # us

    # Per call, synchronised: what a caller that needs the output waits (the
    # protocol of the C-API columns (d)/(e) in capi_mojo/bench/bench_capi.mojo).
    var samples = List[Int]()
    for _ in range(iters):
        var t = perf_counter_ns()
        net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
        ctx.synchronize()
        samples.append(Int(perf_counter_ns() - t))
    sort(samples)
    var synced = Float64(samples[len(samples) // 2]) / 1000.0

    var params = (IN * H1 + H1) + (H1 * H2 + H2) + (H2 * OUT + OUT)
    print(
        "  "
        + name
        + ":  "
        + String(IN)
        + "->"
        + String(H1)
        + "->"
        + String(H2)
        + "->"
        + String(OUT)
        + " batch="
        + String(BATCH)
        + " params="
        + String(params)
        + "   nn forward = "
        + f2(per)
        + " us/call (synced per call: median "
        + f2(synced)
        + " us)"
    )


def main() raises:
    var ctx = DeviceContext()
    var iters = 2000
    print("nn native GPU MLP inference baseline (on-device compute, no Python/transfer)")
    print("  (compare 'nn forward us/call' vs MAX 'device compute' in benchmark_interop)")
    print("")
    bench_nn[17, 256, 256, 6, 1](ctx, "actor-b1", iters)
    bench_nn[17, 256, 256, 6, 64](ctx, "actor-b64", iters)
    bench_nn[17, 256, 256, 6, 1024](ctx, "actor-b1024", iters)
    bench_nn[256, 512, 512, 64, 1](ctx, "wide-b1", iters)
    bench_nn[256, 512, 512, 64, 1024](ctx, "wide-b1024", iters)
