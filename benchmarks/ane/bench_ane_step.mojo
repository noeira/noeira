"""Step 3 of the ANE spike: one training step (forward, backward, Adam) of an
MLP — the functional tape on the CPU, the ANE step (`AneGrad`) with Adam on
the host, and the tape on noeira's Metal launcher (`Tape.metal_rt`) with
`TapeAdam`. The loss is read every step in every arm (the RL loop; an ANE
evaluation is synchronous anyway). One fixed batch, already in place
(device buffers on metal_rt, the input surfaces on the ANE).

A fourth arm, "ANE + GPU Adam" (`AneAdamGpu`), keeps fp32 masters on the GPU:
per step the ANE evaluates, then ONE metal_rt command buffer reads the fp16
gradients from the output surfaces, runs Adam and writes fp16 weights into
the input surfaces; one sync. Its breakdown: ANE eval / GPU command buffer
including the sync.

Workloads: `bench_functional_metal_rt.mojo`'s three training networks, and
a wider one that suits the ANE's matmul better. Each cell is the median
us/step over ROUNDS rounds of `iters` steps [min-max]. The ANE column's
breakdown is the mean over every timed step: fp32 -> fp16 parameter writes,
the evaluation, gradient reads (fp16 -> fp32, / scale), and Adam.

    pixi run -e apple mojo build -I . benchmarks/ane/bench_ane_step.mojo -o /tmp/bench_ane
    /tmp/bench_ane
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64
from std.sys import argv
from max.gpu.host import DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor, TapeAdam
from noeira.experimental.metal_rt.runtime import COMMIT_EVERY_N
import noeira.experimental.functional.F as F
from noeira.experimental.ane.runtime import ANE
from noeira.experimental.ane.step import AneGrad
from noeira.experimental.ane.gpu_adam import AneAdamGpu


comptime ROUNDS = 5


def _median(var v: List[Float64]) -> Float64:
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


def _cell(v: List[Float64]) -> String:
    var lo = v[0]
    var hi = v[0]
    for x in v:
        lo = min(lo, x)
        hi = max(hi, x)
    return String(_r(_median(v.copy())), " [", _r(lo), "-", _r(hi), "]")


def _r(x: Float64) -> String:
    var r = Int(x + 0.5)
    return String(r)


struct W(Copyable):
    var name: String
    var b: Int
    var din: Int
    var h: Int
    var dout: Int
    var relu: Bool
    var iters: Int
    var cpu: Bool

    def __init__(out self, name: String, b: Int, din: Int, h: Int, dout: Int, relu: Bool,
                 iters: Int, cpu: Bool = True):
        self.name = name
        self.b = b
        self.din = din
        self.h = h
        self.dout = dout
        self.relu = relu
        self.iters = iters
        self.cpu = cpu


def params(w: W, tape: Tape, init: List[List[Scalar[DT]]]) raises -> List[FTensor]:
    var p = List[FTensor]()
    var shapes: List[Tuple[Int, Int]] = [(w.din, w.h), (1, w.h), (w.h, w.h), (1, w.h), (w.h, w.dout), (1, w.dout)]
    for k in range(6):
        p.append(tape.param(shapes[k][0], shapes[k][1], init[k].copy()))
    return p^


def forward(w: W, p: List[FTensor], x: FTensor) raises -> FTensor:
    var act: StaticString = "relu" if w.relu else "tanh"
    var h = F.linear(F.linear(x, p[0], p[1], act), p[2], p[3], act)
    return F.linear(h, p[4], p[5])


def bench(a: ANE, w: W, dump: String = "") raises:
    seed(1)
    var init = List[List[Scalar[DT]]]()
    var shapes: List[Tuple[Int, Int]] = [(w.din, w.h), (1, w.h), (w.h, w.h), (1, w.h), (w.h, w.dout), (1, w.dout)]
    for k in range(6):
        var n = shapes[k][0] * shapes[k][1]
        var sc = (6.0 / Float64(shapes[k][0] + shapes[k][1])) ** 0.5 if k % 2 == 0 else 0.0
        var v = List[Scalar[DT]]()
        for _ in range(n):
            v.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * sc))
        init.append(v^)
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for _ in range(w.b * w.din):
        x.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    for _ in range(w.b * w.dout):
        y.append(Scalar[DT](random_float64() * 2.0 - 1.0))

    # CPU tape
    var t_cpu = Tape()
    var p_cpu = params(w, t_cpu, init)
    var o_cpu = Adam(lr=1e-5)
    # ANE
    var t_ane = Tape()
    var p_ane = params(w, t_ane, init)
    var o_ane = Adam(lr=1e-5)
    var xa = t_ane.input(w.b, w.din, x.copy())
    var ya = t_ane.input(w.b, w.dout, y.copy())
    var g = AneGrad(a, t_ane, F.mse(forward(w, p_ane, xa), ya), [xa, ya], dump=dump)
    t_ane.reset()
    g.set_data(a, 0, Pointer(to=x[0]))
    g.set_data(a, 1, Pointer(to=y[0]))
    # ANE + GPU Adam (its own tape and program)
    var t_ag = Tape()
    var p_ag = params(w, t_ag, init)
    var xg = t_ag.input(w.b, w.din, x.copy())
    var yg = t_ag.input(w.b, w.dout, y.copy())
    var gg = AneGrad(a, t_ag, F.mse(forward(w, p_ag, xg), yg), [xg, yg])
    t_ag.reset()
    var ag = AneAdamGpu(a, gg^, t_ag, lr=1e-5)
    ag.set_data(a, 0, Pointer(to=x[0]))
    ag.set_data(a, 1, Pointer(to=y[0]))
    # metal_rt tape
    var t_rt = Tape.metal_rt(policy=COMMIT_EVERY_N, batch_n=16)
    var p_rt = params(w, t_rt, init)
    var o_rt = TapeAdam(lr=1e-5)
    var bx = t_rt.device_buffer(w.b * w.din)
    var by = t_rt.device_buffer(w.b * w.dout)
    t_rt.upload(bx, x)
    t_rt.upload(by, y)

    var t_adam = 0
    var scale = g.start_scale()
    var nonfinite = 0

    def step(arm: Int) raises {mut t_cpu, p_cpu, mut o_cpu, mut t_ane, mut o_ane, mut g, a,
                                mut t_rt, p_rt, mut o_rt, bx, by, x, y, w, mut t_adam, scale, mut nonfinite,
                                mut ag}:
        if arm == 0:
            t_cpu.reset()
            t_cpu.zero_grad()
            var loss = F.mse(forward(w, p_cpu, t_cpu.input(w.b, w.din, x.copy())),
                             t_cpu.input(w.b, w.dout, y.copy()))
            loss.backward()
            var t = t_cpu
            o_cpu.step["cpu"](t, None)
            _ = loss.item()
        elif arm == 1:
            var l = g.run(a, t_ane, scale)
            if l != l:
                nonfinite += 1
            var t0 = perf_counter_ns()
            var t = t_ane
            o_ane.step["cpu"](t, None)
            t_adam += Int(perf_counter_ns() - t0)
        elif arm == 3:
            var l = ag.step(a, scale)
            if l != l:
                nonfinite += 1
        else:
            t_rt.reset()
            t_rt.zero_grad()
            var loss = F.mse(forward(w, p_rt, t_rt.input_device(w.b, w.din, bx)),
                             t_rt.input_device(w.b, w.dout, by))
            loss.backward()
            o_rt.step(t_rt)
            _ = loss.item()

    for arm in range(4):  # warm-up
        if arm == 0 and not w.cpu:
            continue
        for _ in range(5):
            step(arm)
    g.t_write = 0
    g.t_eval = 0
    g.t_read = 0
    t_adam = 0
    ag.t_eval = 0
    ag.t_gpu = 0
    ag.steps = 0
    var res = List[List[Float64]]()
    for _ in range(4):
        res.append(List[Float64]())
    for _ in range(ROUNDS):
        for arm in range(4):
            if arm == 0 and not w.cpu:
                continue
            var t0 = perf_counter_ns()
            for _ in range(w.iters):
                step(arm)
            res[arm].append(Float64(perf_counter_ns() - t0) / 1e3 / Float64(w.iters))
    var n = Float64(ROUNDS * w.iters) * 1e3
    var flop = 6.0 * Float64(w.b) * Float64(w.din * w.h + w.h * w.h + w.h * w.dout)
    var ane_med = _median(res[1].copy())
    var ns = Float64(ag.steps) * 1e3
    print(
        "|", w.name, "|", _cell(res[0]) if w.cpu else String("—"), "|", _cell(res[1]), "|", _cell(res[3]),
        "|", _cell(res[2]),
        "|", _r(Float64(g.t_write) / n), "/", _r(Float64(g.t_eval) / n), "/", _r(Float64(g.t_read) / n),
        "/", _r(Float64(t_adam) / n),
        "|", _r(Float64(ag.t_eval) / ns), "/", _r(Float64(ag.t_gpu) / ns),
        "|", flop / (Float64(g.t_eval) / n * 1e-6) / 1e12, "|",
    )
    _ = ane_med
    if nonfinite > 0:
        print("    (", w.name, ":", nonfinite, "ANE steps had a non-finite gradient)")
    g.free(a)
    ag.free(a)


def main() raises:
    var a = ANE()
    print("us/step (forward + backward + Adam, loss read every step), median of", ROUNDS, "rounds [min-max]")
    print("| workload | CPU tape | ANE + host Adam | ANE + GPU Adam | metal_rt tape | host-Adam arm: write / eval / read / Adam | GPU-Adam arm: eval / GPU cb incl. sync | ANE eval TFLOP/s |")
    print("|---|---|---|---|---|---|---|---|")
    var args = argv()
    if len(args) > 2 and args[1] == "--dump":  # write the BIG program's MIL, nothing else
        bench(a, W("BIG B=4096 64-512-512-64 relu", 4096, 64, 512, 64, True, 20), String(args[2]))
        return
    bench(a, W("PPO B=64 8-64-64-4 tanh", 64, 8, 64, 4, False, 200))
    bench(a, W("SAC B=256 23-256-256-1 relu", 256, 23, 256, 1, True, 200))
    bench(a, W("BIG B=4096 64-512-512-64 relu", 4096, 64, 512, 64, True, 20))
    bench(a, W("WIDE B=2048 512-2048-2048-512 relu", 2048, 512, 2048, 512, True, 10, cpu=False))
