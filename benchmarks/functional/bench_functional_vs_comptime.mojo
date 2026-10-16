"""Functional mode vs the comptime nn, same network, same CPU kernels.

One TRAIN step = zero grads, forward, MSE, backward (no optimizer).
One ACT step = forward only, batch 1 (the single-env acting path).
Rounds are interleaved (comptime, functional, comptime, …) and each arm
reports its best round, so a slow phase of the machine hits both.

    pixi run -e apple mojo build -I . benchmarks/functional/bench_functional_vs_comptime.mojo -o /tmp/bench_fn
    /tmp/bench_fn
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.core.element_op import ElementOp
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.relu_op import ReLUOp
from noeira.nn.primitives.ops.tanh_op import TanhOp
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse import mse_forward, mse_backward
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F


comptime ROUNDS = 7


def _rand(n: Int) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    return out^


def _bench[
    B: Int, IN: Int, H: Int, OUT: Int, OP: ElementOp, TRAIN: Bool
](name: String, iters: Int) raises:
    comptime RELU = OP.is_relu
    comptime NET = Sequential[
        Linear[IN, H], Elementwise[H, OP], Linear[H, H], Elementwise[H, OP],
        Linear[H, OUT],
    ]
    seed(1)
    var net = NET.make["cpu", Deterministic]()
    var xs = _rand(B * IN)
    var ys = _rand(B * OUT)
    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    for k in range(B * IN):
        x.data[k] = xs[k]
    for k in range(B * OUT):
        y.data[k] = ys[k]
    var out = Tensor.alloc(B * OUT)
    var go = Tensor.alloc(B * OUT)
    var gx = Tensor.alloc(B * IN)

    var tape = Tape()
    var w1 = tape.param(IN, H, net.children[0].weight.val.data.copy())
    var b1 = tape.param(1, H, net.children[0].bias.val.data.copy())
    var w2 = tape.param(H, H, net.children[2].weight.val.data.copy())
    var b2 = tape.param(1, H, net.children[2].bias.val.data.copy())
    var w3 = tape.param(H, OUT, net.children[4].weight.val.data.copy())
    var b3 = tape.param(1, OUT, net.children[4].bias.val.data.copy())

    var sink: Float64 = 0
    var best_c = Float64(1e30)
    var best_f = Float64(1e30)
    for round in range(ROUNDS + 1):  # round 0 = warmup, not scored
        var t0 = perf_counter_ns()
        for _ in range(iters):
            comptime if TRAIN:
                net.zero_grad["cpu"](None)
            net.forward["cpu", B](TensorRefs[1](x), out)
            comptime if TRAIN:
                mse_backward["cpu", B, OUT](out, y, go)
                net.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx))
        var t1 = perf_counter_ns()
        sink += Float64(out.data[0])
        for _ in range(iters):
            tape.reset()
            comptime if TRAIN:
                tape.zero_grad()
            var fx = tape.input(B, IN, xs)
            var h = F.linear(fx, w1, b1)
            h = h.relu() if RELU else h.tanh()
            h = F.linear(h, w2, b2)
            h = h.relu() if RELU else h.tanh()
            var pred = F.linear(h, w3, b3)
            comptime if TRAIN:
                var loss = F.mse(pred, tape.input(B, OUT, ys))
                loss.backward()
            sink += Float64(pred.value(0))
        var t2 = perf_counter_ns()
        if round > 0:
            best_c = min(best_c, Float64(t1 - t0) / Float64(iters) / 1000.0)
            best_f = min(best_f, Float64(t2 - t1) / Float64(iters) / 1000.0)
    print(
        name, ": comptime ", best_c, " us   functional ", best_f,
        " us   ratio ", best_f / best_c, "   (sink ", sink != 0, ")", sep="",
    )


def main() raises:
    print("best of", ROUNDS, "interleaved rounds, per step, CPU")
    _bench[1, 23, 256, 1, ReLUOp, False]("ACT   B=1   23-256-256-1 relu fwd   ", 4000)
    _bench[64, 8, 64, 4, TanhOp, True]("PPO   B=64  8-64-64-4 tanh train   ", 2000)
    _bench[256, 23, 256, 1, ReLUOp, True]("SAC   B=256 23-256-256-1 relu train", 300)
    _bench[8, 4, 16, 2, TanhOp, True]("TINY  B=8   4-16-16-2 tanh train   ", 20000)
