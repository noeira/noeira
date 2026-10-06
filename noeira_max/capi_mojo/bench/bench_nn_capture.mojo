"""Column (g) of the MLP table: noeira's nn with CUDA graph capture, the
like-for-like partner of MAX's captured column (e), next to the uncaptured
column (f) measured in the same run.

The five MLPs of `benchmark_nn_baseline.mojo` (Linear, ReLU, Linear, ReLU,
Linear). The forward is captured once with `noeira.cuda.CUDAGraph` and
replayed on Mojo's own stream, the one an eager forward uses:

- synchronised: per call, the forward (or a replay), then
  `ctx.synchronize()`; the median of CALLS calls.
- pipelined: CALLS calls back to back and one synchronisation.

A replay's output is checked against an eager forward's, with the output
zeroed in between so that the replay must write it.

The capture is noeira's stream capture (`noeira/cuda/graph.mojo`), the one
its RL steps use. MAX's `DeviceGraph` recording (`device_graph.mojo`) is not
safe here: these layers take MAX's vendor GEMM, whose per-call scratch a
recorded node would keep pointing at. Stream capture needs noeira's CUDA
interposer preloaded, which `pixi run` does on Linux, and loads it by a path
relative to the main checkout's root, so run it from there:

    cd <main checkout> && pixi run -e default mojo run -I . \\
        <this worktree>/noeira_max/capi_mojo/bench/bench_nn_capture.mojo
"""

from std.math import sin
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.cuda import CUDAGraph
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.linear import Linear

comptime WARMUP = 100
comptime CALLS = 1000


def median_us(mut samples: List[Int]) -> Float64:
    sort(samples)
    return Float64(samples[len(samples) // 2]) / 1000.0


def bench[
    IN: Int, H1: Int, H2: Int, OUT: Int, BATCH: Int
](ctx: DeviceContext, name: String) raises:
    comptime MLP = Sequential[
        Linear[IN, H1], ReLU[H1], Linear[H1, H2], ReLU[H2], Linear[H2, OUT]
    ]
    var net = MLP.make["gpu", Kaiming](Optional(ctx))
    var x = Tensor.alloc(BATCH * IN)
    for i in range(BATCH * IN):
        x.data[i] = Scalar[DT](sin(Float64(i) * 0.37))
    x.upload(ctx)
    var y = Tensor.alloc(BATCH * OUT)

    # (f) eager. The warmup also allocates every device buffer, so that the
    # capture below records no allocation.
    for _ in range(WARMUP):
        net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
    ctx.synchronize()
    y.download(ctx)
    var want = y.data.copy()
    var eager = List[Int]()
    for _ in range(CALLS):
        var t = perf_counter_ns()
        net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
        ctx.synchronize()
        eager.append(Int(perf_counter_ns() - t))
    var t0 = perf_counter_ns()
    for _ in range(CALLS):
        net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
    ctx.synchronize()
    var eager_pipelined = Float64(perf_counter_ns() - t0) / Float64(CALLS) / 1000.0

    # (g) captured once, replayed on Mojo's stream.
    var graph = CUDAGraph(ctx)
    if graph.is_disabled():
        raise Error("capture is disabled: run under `pixi run` (the CUDA interposer must be preloaded)")
    graph.begin_capture()
    net.forward["gpu", BATCH](TensorRefs[1](x), y, Optional(ctx))
    graph.end_capture()
    for i in range(BATCH * OUT):
        y.data[i] = 0
    y.upload_resident(ctx)  # the same device buffer, zeroed
    graph.replay_on_mojo_stream()
    ctx.synchronize()
    y.download(ctx)
    var scale = Float64(0)
    var worst = Float64(0)
    for i in range(BATCH * OUT):
        scale = max(scale, abs(Float64(want[i])))
        worst = max(worst, abs(Float64(y.data[i]) - Float64(want[i])))
    for _ in range(WARMUP):
        graph.replay_on_mojo_stream()
    ctx.synchronize()
    var captured = List[Int]()
    for _ in range(CALLS):
        var t = perf_counter_ns()
        graph.replay_on_mojo_stream()
        ctx.synchronize()
        captured.append(Int(perf_counter_ns() - t))
    t0 = perf_counter_ns()
    for _ in range(CALLS):
        graph.replay_on_mojo_stream()
    ctx.synchronize()
    var captured_pipelined = Float64(perf_counter_ns() - t0) / Float64(CALLS) / 1000.0

    print(
        "RESULT {\"shape\": \"" + name + "\", \"dims\": \"" + String(IN) + "->" + String(H1)
        + "->" + String(H2) + "->" + String(OUT) + "\", \"batch\": " + String(BATCH)
        + ", \"graph_nodes\": " + String(graph.num_nodes())
        + ", \"f_eager_us\": " + String(median_us(eager))
        + ", \"f_eager_pipelined_us\": " + String(eager_pipelined)
        + ", \"g_captured_us\": " + String(median_us(captured))
        + ", \"g_captured_pipelined_us\": " + String(captured_pipelined)
        + ", \"replay_vs_eager_max_abs_diff\": " + String(worst)
        + ", \"output_scale\": " + String(scale) + "}"
    )


def main() raises:
    var ctx = DeviceContext()
    print("(f) noeira nn eager and (g) captured, us per call; median of", CALLS, "synchronised calls")
    bench[17, 256, 256, 6, 1](ctx, "actor-b1")
    bench[17, 256, 256, 6, 64](ctx, "actor-b64")
    bench[17, 256, 256, 6, 1024](ctx, "actor-b1024")
    bench[256, 512, 512, 64, 1](ctx, "wide-b1")
    bench[256, 512, 512, 64, 1024](ctx, "wide-b1024")
