"""The functional tape on noeira's Metal launcher vs on MAX, and the comptime nn.

The workloads of `bench_functional_gpu.mojo` (same networks, same steps):
ACT B=1 forward, PPO B=64 and SAC B=256 train (+ Adam), B=4096 64-512-512-64
train. Arms:

  comptime     the comptime nn through MAX (`Sequential[Linear, …]`, nn's
               Adam adopted)
  tape MAX     the tape on MAX: MAX's matmul, nn's Adam adopted (as
               bench_functional_gpu runs it)
  tape MAX own the tape on MAX with its own GEMM and `TapeAdam` — exactly the
               kernels of the metal_rt arms, so (tape MAX own) − (metal_rt)
               is the launcher alone
  rt every-16 / rt every-64 / rt adaptive
               the tape on `Tape.metal_rt` (policy COMMIT_EVERY_N, N = 16 or
               64; COMMIT_ADAPTIVE, N = 16, idle age 20 us)

Loop shapes: `end` (iters steps, one sync: throughput, as
bench_functional_gpu), `read/20` (loss read every 20 steps), `read/1` (every
step: the RL loop). A "read" is `loss.item()` (ACT: the output's
`item()`); on the comptime arm a 1-float copy of the output + synchronize.
On metal_rt a read is the only sync; nothing else flushes.

Rounds interleave the arms; each cell is the MEDIAN us/step over ROUNDS
rounds, with the min-max spread in brackets. `--arm K` (0-5) builds and
times ONE arm: MAX's per-launch cost grows with every live MAX buffer, so in
the interleaved run the MAX arms also pay for the other arms' tapes (run
each arm in its own process for the fair numbers).

    pixi run -e apple mojo build -I . benchmarks/functional/bench_functional_metal_rt.mojo -o /tmp/bench_fn_rt
    /tmp/bench_fn_rt
"""

from std.time import perf_counter_ns
from std.sys import argv
from std.random import seed, random_float64
from max.gpu.host import DeviceContext, DeviceBuffer

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
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor, TapeAdam
from noeira.experimental.metal_rt.runtime import COMMIT_EVERY_N, COMMIT_ADAPTIVE
import noeira.experimental.functional.F as F


comptime ROUNDS = 7
comptime NARMS = 6


def _median(var v: List[Float64]) -> Float64:
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


def _fmt(x: Float64) -> String:
    var r = Int(x * 10.0 + 0.5)
    return String(r // 10) + "." + String(r % 10)


def _bench[
    B: Int, IN: Int, H: Int, OUT: Int, OP: ElementOp, TRAIN: Bool,
    ADAM: Bool = False,
](ctx: DeviceContext, name: String, iters: Int, only: Int) raises:
    """`only` >= 0: build and time that ONE arm (MAX's per-launch cost grows
    with every live MAX buffer, so other arms' tapes would slow it down)."""
    comptime RELU = OP.is_relu
    comptime NET = Sequential[
        Linear[IN, H], Elementwise[H, OP], Linear[H, H], Elementwise[H, OP],
        Linear[H, OUT],
    ]
    seed(1)
    var ct_on = only < 0 or only == 0
    var wnet = NET.make["cpu", Deterministic]()  # the weights every arm starts from
    var net = NET.make["gpu", Deterministic](ctx) if ct_on else NET.make["cpu", Deterministic]()
    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    for k in range(B * IN):
        x.data[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for k in range(B * OUT):
        y.data[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    var out = Tensor.alloc(B * OUT)
    var go = Tensor.alloc(B * OUT)
    var gx = Tensor.alloc(B * IN)
    var host1 = List[Scalar[DT]](length=1, fill=0)
    var opt_c = Adam(lr=1e-5)
    if ct_on:
        x.upload(ctx)
        y.upload(ctx)
        out = Tensor.alloc_gpu(ctx, B * OUT)
        go = Tensor.alloc_gpu(ctx, B * OUT)
        gx = Tensor.alloc_gpu(ctx, B * IN)
        comptime if ADAM:
            opt_c.adopt["gpu"](net, ctx)

    # tape arms: 1 MAX, 2 MAX own kernels, 3-5 metal_rt
    # (an arm not built is a CPU tape: no device memory, never stepped)
    var tapes = List[Tape]()
    tapes.append(Tape(ctx) if only < 0 or only == 1 else Tape())
    tapes.append(Tape(ctx, own_kernels=True) if only < 0 or only == 2 else Tape())
    tapes.append(Tape.metal_rt(policy=COMMIT_EVERY_N, batch_n=16) if only < 0 or only == 3 else Tape())
    tapes.append(Tape.metal_rt(policy=COMMIT_EVERY_N, batch_n=64) if only < 0 or only == 4 else Tape())
    tapes.append(Tape.metal_rt(policy=COMMIT_ADAPTIVE, batch_n=16, idle_age_us=20) if only < 0 or only == 5 else Tape())
    var ps = List[List[FTensor]]()
    var xb = List[DeviceBuffer[DT]]()
    var yb = List[DeviceBuffer[DT]]()
    var nopt = Adam(lr=1e-5)
    var topts = List[TapeAdam]()
    for a in range(len(tapes)):
        var t = tapes[a]
        var p = List[FTensor]()
        p.append(t.param(IN, H, wnet.children[0].weight.val.data.copy()))
        p.append(t.param(1, H, wnet.children[0].bias.val.data.copy()))
        p.append(t.param(H, H, wnet.children[2].weight.val.data.copy()))
        p.append(t.param(1, H, wnet.children[2].bias.val.data.copy()))
        p.append(t.param(H, OUT, wnet.children[4].weight.val.data.copy()))
        p.append(t.param(1, OUT, wnet.children[4].bias.val.data.copy()))
        ps.append(p^)
        if t.on_gpu():
            var bx = t.device_buffer(B * IN)
            var by = t.device_buffer(B * OUT)
            t.upload(bx, x.data)
            t.upload(by, y.data)
            xb.append(bx)
            yb.append(by)
        else:
            xb.append(ctx.enqueue_create_buffer[DT](1))  # placeholder
            yb.append(ctx.enqueue_create_buffer[DT](1))
        topts.append(TapeAdam(lr=1e-5))
    comptime if ADAM:
        if tapes[0].on_gpu():
            var t0 = tapes[0]
            nopt.adopt["gpu"](t0, ctx)

    def _ct_step(do_read: Bool) raises {mut net, mut x, mut y, mut out, mut go, mut gx, mut opt_c, mut host1, ctx}:
        comptime if TRAIN:
            net.zero_grad["gpu"](ctx)
        net.forward["gpu", B](TensorRefs[1](x), out, ctx)
        comptime if TRAIN:
            mse_backward["gpu", B, OUT](out, y, go, ctx)
            net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), ctx)
        comptime if ADAM:
            opt_c.step["gpu"](net, ctx)
        if do_read:
            ctx.enqueue_copy(host1.unsafe_ptr(), out.dev.value().create_sub_buffer[DT](0, 1))
            ctx.synchronize()

    def _fn_step(a: Int, do_read: Bool) raises {mut tapes, ps, xb, yb, mut nopt, mut topts, ctx}:
        var tape = tapes[a]
        ref p = ps[a]
        tape.reset()
        comptime if TRAIN:
            tape.zero_grad()
        var fx = tape.input_device(B, IN, xb[a])
        var h: FTensor
        comptime if RELU:
            h = F.linear(F.linear(fx, p[0], p[1], "relu"), p[2], p[3], "relu")
        else:
            h = F.linear(F.linear(fx, p[0], p[1], "tanh"), p[2], p[3], "tanh")
        var pred = F.linear(h, p[4], p[5])
        comptime if TRAIN:
            var loss = F.mse(pred, tape.input_device(B, OUT, yb[a]))
            loss.backward()
            comptime if ADAM:
                if a == 0:
                    var t = tape
                    nopt.step["gpu"](t, ctx)
                else:
                    topts[a].step(tape)
            if do_read:
                _ = loss.item()
        else:
            if do_read:
                _ = pred.value(0)

    def _sync(a: Int) raises {tapes, ctx}:
        if a == 0:
            ctx.synchronize()
        else:
            tapes[a - 1].synchronize()

    # warm-up: allocations, pipelines
    for _ in range(3):
        if ct_on:
            _ct_step(False)
        for a in range(len(tapes)):
            if tapes[a].on_gpu():
                _fn_step(a, False)
    ctx.synchronize()
    for a in range(len(tapes)):
        tapes[a].synchronize()

    var shapes = List[Int]()
    shapes.append(0)
    if TRAIN:
        shapes.append(20)
    shapes.append(1)
    for si in range(len(shapes)):
        var every = shapes[si]
        var times = List[List[Float64]]()
        for _ in range(NARMS):
            times.append(List[Float64]())
        for round in range(ROUNDS + 1):  # round 0 = warm-up
            for arm in range(NARMS):
                if only >= 0 and arm != only:
                    continue
                _sync(arm)
                var t0 = perf_counter_ns()
                for i in range(iters):
                    var rd = every > 0 and i % every == every - 1
                    if arm == 0:
                        _ct_step(rd)
                    else:
                        _fn_step(arm - 1, rd)
                _sync(arm)
                var us = Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0
                if round > 0:
                    times[arm].append(us)
        var shape_name = String("end") if every == 0 else String("read/", every)
        var row = String("| ") + name + " | " + shape_name
        for arm in range(NARMS):
            if only >= 0 and arm != only:
                row += " | -"
                continue
            var lo = Float64(1e30)
            var hi = Float64(0)
            for v in times[arm]:
                lo = min(lo, v)
                hi = max(hi, v)
            row += " | " + _fmt(_median(times[arm].copy())) + " [" + _fmt(lo) + "-" + _fmt(hi) + "]"
        print(row + " |")
    for a in range(2, len(tapes)):
        if not tapes[a].on_gpu():
            continue
        var st = tapes[a].launcher_stats()
        print("    metal_rt arm", a + 1, ": launches", st[0], " command buffers", st[1],
              " (", _fmt(Float64(st[0]) / Float64(max(st[1], 1))), "launches / cb )")


def main() raises:
    var only = -1
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--arm" and i + 1 < len(args):
            only = Int(args[i + 1])
    var ctx = DeviceContext()
    print("GPU:", ctx.name(), "— us/step, median of", ROUNDS, "interleaved rounds [min-max]")
    print("| workload | loop | comptime (MAX) | tape MAX | tape MAX own kernels | rt every-16 | rt every-64 | rt adaptive-16 age 20 |")
    print("|---|---|---|---|---|---|---|---|")
    _bench[1, 23, 256, 1, ReLUOp, False](ctx, "ACT B=1 23-256-256-1 fwd", 500, only)
    _bench[64, 8, 64, 4, TanhOp, True](ctx, "PPO B=64 8-64-64-4 train", 200, only)
    _bench[256, 23, 256, 1, ReLUOp, True](ctx, "SAC B=256 23-256-256-1 train", 200, only)
    _bench[4096, 64, 512, 64, ReLUOp, True](ctx, "BIG B=4096 64-512-512-64 train", 40, only)
    _bench[64, 8, 64, 4, TanhOp, True, True](ctx, "PPO B=64 train + Adam", 200, only)
    _bench[256, 23, 256, 1, ReLUOp, True, True](ctx, "SAC B=256 train + Adam", 200, only)
