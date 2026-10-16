"""Functional mode vs the comptime nn ON THE GPU, same network.

One TRAIN step = zero grads, forward, MSE, backward (no optimizer); one ACT
step = forward only at batch 1. Inputs are already on the device in both
arms (`Tape.input_device`: one device copy). Each round enqueues `iters`
steps and synchronises once, so this is device THROUGHPUT, launches
pipelined. Rounds interleave the arms; each reports its best.

The functional arm uses the FUSED `F.linear(x, w, b, act)` (one node per
layer). On NVIDIA two more arms replay the same steps from a CUDA graph:
the comptime one through `CUDAGraph`, the functional one through
`tape.capture_begin/end/replay` — captured once after an eager warm-up.

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
from std.sys import has_nvidia_gpu_accelerator
from noeira.cuda.graph import CUDAGraph
from noeira.nn.optimizer.adam import Adam
from noeira.nn.functional import Tape, FTensor
import noeira.nn.functional.F as F


comptime ROUNDS = 5


def _bench[
    B: Int, IN: Int, H: Int, OUT: Int, OP: ElementOp, TRAIN: Bool,
    ADAM: Bool = False,
](ctx: DeviceContext, name: String, iters: Int) raises:
    """`ADAM`: the step ends with nn's Adam, ADOPTED (arena mode) in both
    arms so its bias correction advances on the device under replay."""
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
    var opt_c = Adam(lr=1e-5)
    var opt_f = Adam(lr=1e-5)
    comptime if ADAM:
        opt_c.adopt["gpu"](net, ctx)
        opt_f.adopt["gpu"](tape, ctx)

    def _ct_step(mut net: NET, ctx: DeviceContext) raises {mut x, mut y, mut out, mut go, mut gx, mut opt_c}:
        comptime if TRAIN:
            net.zero_grad["gpu"](ctx)
        net.forward["gpu", B](TensorRefs[1](x), out, ctx)
        comptime if TRAIN:
            mse_backward["gpu", B, OUT](out, y, go, ctx)
            net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), ctx)
        comptime if ADAM:
            opt_c.step["gpu"](net, ctx)

    def _fn_step(tape: Tape) raises {mut x, mut y, w1, b1, w2, b2, w3, b3, mut opt_f, ctx}:
        tape.reset()
        comptime if TRAIN:
            tape.zero_grad()
        var fx = tape.input_device(B, IN, x.dev.value())
        var h: FTensor
        comptime if RELU:
            h = F.linear(F.linear(fx, w1, b1, "relu"), w2, b2, "relu")
        else:
            h = F.linear(F.linear(fx, w1, b1, "tanh"), w2, b2, "tanh")
        var pred = F.linear(h, w3, b3)
        comptime if TRAIN:
            var loss = F.mse(pred, tape.input_device(B, OUT, y.dev.value()))
            loss.backward()
        comptime if ADAM:
            var t = tape
            opt_f.step["gpu"](t, ctx)

    # Warm-up (allocations, cuBLASLt plans), then capture on NVIDIA.
    _ct_step(net, ctx)
    _fn_step(tape)
    ctx.synchronize()
    comptime GRAPHS = has_nvidia_gpu_accelerator()
    var g_ct = CUDAGraph(ctx)
    comptime if GRAPHS:
        g_ct.begin_capture()
        _ct_step(net, ctx)
        g_ct.end_capture()
        tape.capture_begin()
        _fn_step(tape)
        tape.capture_end()
        if not g_ct.is_captured() or not tape.captured():
            raise Error("capture failed")

    var best = List[Float64](length=4, fill=1e30)
    for round in range(ROUNDS + 1):  # round 0 = warmup
        for arm in range(4 if GRAPHS else 2):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for _ in range(iters):
                if arm == 0:
                    _ct_step(net, ctx)
                elif arm == 1:
                    _fn_step(tape)
                elif arm == 2:
                    g_ct.replay_on_mojo_stream()
                else:
                    tape.replay()
            ctx.synchronize()
            var us = Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0
            if round > 0:
                best[arm] = min(best[arm], us)
    print(name, ":  eager comptime ", best[0], " us  functional ", best[1],
          " us  (x", best[1] / best[0], ")", sep="")
    comptime if GRAPHS:
        print("      graph  comptime ", best[2], " us  functional ", best[3],
              " us  (x", best[3] / best[2], ")", sep="")


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name(), "— best of", ROUNDS, "interleaved rounds, per step; functional = fused F.linear")
    _bench[1, 23, 256, 1, ReLUOp, False](ctx, "ACT   B=1    23-256-256-1 relu fwd   ", 500)
    _bench[64, 8, 64, 4, TanhOp, True](ctx, "PPO   B=64   8-64-64-4 tanh train   ", 200)
    _bench[256, 23, 256, 1, ReLUOp, True](ctx, "SAC   B=256  23-256-256-1 relu train", 200)
    _bench[4096, 64, 512, 64, ReLUOp, True](ctx, "BIG   B=4096 64-512-512-64 relu train", 50)
    print("with nn's Adam (adopted) in the step:")
    _bench[64, 8, 64, 4, TanhOp, True, True](ctx, "PPO   B=64   train + Adam            ", 200)
    _bench[256, 23, 256, 1, ReLUOp, True, True](ctx, "SAC   B=256  train + Adam            ", 200)
