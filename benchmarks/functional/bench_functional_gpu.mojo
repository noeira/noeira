"""Functional mode vs the comptime nn ON THE GPU, same network.

One TRAIN step = zero grads, forward, MSE, backward (no optimizer); one ACT
step = forward only at batch 1. Inputs are already on the device in both
arms (`Tape.input_device`: one device copy). Each round enqueues `iters`
steps and synchronises once, so this is device THROUGHPUT, launches
pipelined. Rounds interleave the two arms; each reports its best.

    pixi run -e apple mojo build -I . benchmarks/functional/bench_functional_gpu.mojo -o /tmp/bench_fn_gpu
    /tmp/bench_fn_gpu
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.element_op import ElementOp
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.relu_op import ReLUOp
from noeira.nn.primitives.ops.tanh_op import TanhOp
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse import mse_backward
from noeira.nn.functional import Tape, FTensor
import noeira.nn.functional.F as F


comptime ROUNDS = 5


def _bench[
    B: Int, IN: Int, H: Int, OUT: Int, OP: ElementOp, TRAIN: Bool
](ctx: DeviceContext, name: String, iters: Int) raises:
    comptime RELU = OP.is_relu
    comptime NET = Sequential[
        Linear[IN, H], Elementwise[H, OP], Linear[H, H], Elementwise[H, OP],
        Linear[H, OUT],
    ]
    seed(1)
    var net = NET.make["gpu", Deterministic](ctx)
    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    for k in range(B * IN):
        x.data[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for k in range(B * OUT):
        y.data[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    x.upload(ctx)
    y.upload(ctx)
    var out = Tensor.alloc_gpu(ctx, B * OUT)
    var go = Tensor.alloc_gpu(ctx, B * OUT)
    var gx = Tensor.alloc_gpu(ctx, B * IN)

    var tape = Tape(ctx)
    var w1 = tape.param(IN, H, net.children[0].weight.val.data.copy())
    var b1 = tape.param(1, H, net.children[0].bias.val.data.copy())
    var w2 = tape.param(H, H, net.children[2].weight.val.data.copy())
    var b2 = tape.param(1, H, net.children[2].bias.val.data.copy())
    var w3 = tape.param(H, OUT, net.children[4].weight.val.data.copy())
    var b3 = tape.param(1, OUT, net.children[4].bias.val.data.copy())

    var best_c = Float64(1e30)
    var best_f = Float64(1e30)
    for round in range(ROUNDS + 1):  # round 0 = warmup
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(iters):
            comptime if TRAIN:
                net.zero_grad["gpu"](ctx)
            net.forward["gpu", B](TensorRefs[1](x), out, ctx)
            comptime if TRAIN:
                mse_backward["gpu", B, OUT](out, y, go, ctx)
                net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for _ in range(iters):
            tape.reset()
            comptime if TRAIN:
                tape.zero_grad()
            var fx = tape.input_device(B, IN, x.dev.value())
            var h = F.linear(fx, w1, b1)
            h = h.relu() if RELU else h.tanh()
            h = F.linear(h, w2, b2)
            h = h.relu() if RELU else h.tanh()
            var pred = F.linear(h, w3, b3)
            comptime if TRAIN:
                var loss = F.mse(pred, tape.input_device(B, OUT, y.dev.value()))
                loss.backward()
        tape.synchronize()
        var t2 = perf_counter_ns()
        if round > 0:
            best_c = min(best_c, Float64(t1 - t0) / Float64(iters) / 1000.0)
            best_f = min(best_f, Float64(t2 - t1) / Float64(iters) / 1000.0)
    print(
        name, ": comptime ", best_c, " us   functional ", best_f,
        " us   ratio ", best_f / best_c, "   (", tape.num_nodes(), " nodes)",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name(), "— best of", ROUNDS, "interleaved rounds, per step")
    _bench[1, 23, 256, 1, ReLUOp, False](ctx, "ACT   B=1    23-256-256-1 relu fwd   ", 500)
    _bench[64, 8, 64, 4, TanhOp, True](ctx, "PPO   B=64   8-64-64-4 tanh train   ", 200)
    _bench[256, 23, 256, 1, ReLUOp, True](ctx, "SAC   B=256  23-256-256-1 relu train", 200)
    _bench[4096, 64, 512, 64, ReLUOp, True](ctx, "BIG   B=4096 64-512-512-64 relu train", 50)
