"""Step 2 of the ANE spike: the functional tape's forward + backward as ONE
program on the Apple Neural Engine (`noeira.experimental.ane.emit`), Adam on
the host.

1. Gradients: the ANE step's loss and every parameter gradient against the
   fp32 CPU tape's, at the same weights, on three batches. fp16 everywhere on
   the ANE, so the bound is a relative one: max |g_ane - g| / max |g| per
   parameter.
   a. MLP 17 -> 64 relu -> 64 tanh -> 6, MSE, batch 64
   b. every other rank-2 op the emitter takes
   c. MUTANT (tanh's backward 1 + y²) must FAIL (a)'s bound.
2. Training: fit sin(x) on [-pi, pi], 1 -> 64 tanh -> 64 tanh -> 1, batch
   256, Adam 3e-3, the same init and batches, CPU tape vs the ANE step
   (dynamic loss scale from 32768: halve and skip on a non-finite gradient,
   double after 500 good steps). The ANE run must reach the CPU run's loss
   (within a factor) and its curve is printed.

Every gradient is scaled by the seed (an input): fp16's smallest normal is
6.1e-5, and at a 1024 scale the every-op graph's first-layer gradients
lost 10% to underflow (1.1% at 32768); section 0 shows the fp32 gradients
themselves move by < 7e-4 under fp16-sized input noise there, so the error
is range, not conditioning.

    pixi run -e apple mojo run -I . tests/experimental/ane/test_ane_step.mojo
"""

from std.math import pi, sin
from std.random import seed, random_float64
from std.time import perf_counter_ns
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F
from noeira.experimental.ane.runtime import ANE, Program
from noeira.experimental.ane.emit import emit_grad_step, GradStep


comptime GRAD_TOL = 3e-2


struct Net(Copyable):
    var b: Int
    var din: Int
    var h: Int
    var dout: Int
    var kind: Int
    """0 = MLP relu/tanh + MSE; 1 = the every-op graph; 2 = sine fit."""

    def __init__(out self, b: Int, din: Int, h: Int, dout: Int, kind: Int):
        self.b = b
        self.din = din
        self.h = h
        self.dout = dout
        self.kind = kind

    def params(self, tape: Tape) raises -> List[FTensor]:
        seed(7)
        var p = List[FTensor]()
        p.append(tape.param[Xavier](self.din, self.h, name="w1"))
        p.append(tape.zeros(1, self.h, name="b1"))
        p.append(tape.param[Xavier](self.h, self.h, name="w2"))
        p.append(tape.zeros(1, self.h, name="b2"))
        p.append(tape.param[Xavier](self.h, self.dout, name="w3"))
        p.append(tape.zeros(1, self.dout, name="b3"))
        # non-zero biases, so their gradients are checked off zero too
        for k in [1, 3, 5]:
            for i in range(p[k].size()):
                p[k].set_value(i, Scalar[DT](0.1 * random_float64() - 0.05))
        return p^

    def loss(self, p: List[FTensor], x: FTensor, y: FTensor) raises -> FTensor:
        if self.kind == 0:
            var h = F.linear(x, p[0], p[1], "relu")
            h = F.linear(h, p[2], p[3], "tanh")
            return F.mse(F.linear(h, p[4], p[5]), y)
        if self.kind == 2:
            var h = F.linear(x, p[0], p[1], "tanh")
            h = F.linear(h, p[2], p[3], "tanh")
            return F.mse(F.linear(h, p[4], p[5]), y)
        # every other op, rank 2 (tape_to_mo's gate graph)
        var z = x @ p[0] + p[1]                                  # matmul, add (broadcast)
        var a = z.sigmoid() * Scalar[DT](2.0) + Scalar[DT](0.5)  # scale, shift
        var c = (a.exp() + Scalar[DT](1.0)).log()                # exp, log
        var d = (c * c + Scalar[DT](1.0)).sqrt() / (a + Scalar[DT](1.0))  # mul, sqrt, div
        var e = (d @ p[2] - p[3]).tanh()                         # sub (broadcast)
        var f = (Scalar[DT](1.0) - e) / Scalar[DT](4.0)          # rsubc, divc
        var pred = (f @ p[4]) * p[5]                             # mul (broadcast)
        var r = (pred - y).relu()
        var t = ((x.T() @ x) / Scalar[DT](100.0)).sum_rows()      # transpose, sum_rows
        return r.mean() + (t * t).sum() * Scalar[DT](1e-4) + (pred * pred).mean()


def batch(net: Net, step: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    seed(1000 + step)
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    if net.kind == 2:
        for _ in range(net.b):
            var v = (random_float64() * 2.0 - 1.0) * pi
            x.append(Scalar[DT](v))
            y.append(Scalar[DT](sin(v)))
        return (x^, y^)
    for _ in range(net.b * net.din):
        x.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    for _ in range(net.b * net.dout):
        y.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    return (x^, y^)


struct AneStep(Movable):
    """The compiled step and where its data goes."""

    var step: GradStep
    var prog: Program
    var x_arg: Int
    var y_arg: Int
    var grad: List[Scalar[DT]]

    def __init__(out self, a: ANE, net: Net, mut tape: Tape, p: List[FTensor],
                 mutant: Bool = False) raises:
        var bt = batch(net, 0)
        var xb = tape.input(net.b, net.din, bt[0].copy())
        var yb = tape.input(net.b, net.dout, bt[1].copy())
        self.step = emit_grad_step(tape, net.loss(p, xb, yb), mutant=mutant)
        self.prog = Program(a, self.step.text)
        self.x_arg = -1
        self.y_arg = -1
        for k in range(len(self.step.args)):
            if self.step.args[k] == xb.id:
                self.x_arg = k
            elif self.step.args[k] == yb.id:
                self.y_arg = k
        if self.x_arg < 0 or self.y_arg < 0:
            raise Error("the batch inputs are not program arguments")
        if len(self.prog.in_layout) != len(self.step.args) + 1:
            raise Error("program inputs != tape arguments + seed")
        self.grad = List[Scalar[DT]]()
        tape.reset()

    def run(mut self, a: ANE, mut tape: Tape, x: List[Scalar[DT]], y: List[Scalar[DT]],
            seed_v: Float32) raises -> Float64:
        """Loss; the gradients (divided by the seed) land in the tape's
        parameter grads. Returns NaN if any gradient is not finite."""
        ref td = tape.data[]
        for k in range(len(self.step.args)):
            var id = self.step.args[k]
            if k == self.x_arg:
                self.prog.write(a, k, Pointer(to=x[0]))
            elif k == self.y_arg:
                self.prog.write(a, k, Pointer(to=y[0]))
            else:
                self.prog.write(a, k, Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(td.val_ptr(id))))
        var s: List[Float32] = [seed_v]
        self.prog.write(a, len(self.step.args), Pointer(to=s[0]))
        self.prog.run(a)
        var l: List[Float32] = [0.0]
        self.prog.read(a, 0, Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(l.unsafe_ptr())))
        var finite = True
        for k in range(len(self.step.grads)):
            var id = self.step.grads[k]
            var gp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(td.grad_ptr(id)))
            self.prog.read(a, k + 1, gp, 1.0 / seed_v)
            var n = self.step.shapes[id][0] * self.step.shapes[id][1]
            for i in range(n):
                var v = gp[unsafe_offset=i]
                if v != v or abs(v) > 1e30:
                    finite = False
        _ = s^
        return Float64(l[0]) if finite else _nan()


def _nan() -> Float64:
    var z = Float64(0)
    return z / z


def cpu_grads(net: Net, mut tape: Tape, p: List[FTensor], step: Int) raises -> Tuple[Float64, List[List[Scalar[DT]]]]:
    var bt = batch(net, step)
    tape.reset()
    tape.zero_grad()
    var loss = net.loss(p, tape.input(net.b, net.din, bt[0].copy()),
                        tape.input(net.b, net.dout, bt[1].copy()))
    loss.backward()
    var g = List[List[Scalar[DT]]]()
    for k in range(len(p)):
        g.append(p[k].grads())
    var l = Float64(loss.item())
    tape.reset()
    return (l, g^)


def sensitivity(name: String, net: Net) raises -> List[Float64]:
    """How far the fp32 tape's gradients move when every parameter and input
    value is perturbed by a random relative 2^-11 (one fp16 rounding): the
    least error an fp16 evaluation can have, per parameter (worst of three
    batches, max-normalised like the gate)."""
    var tape = Tape()
    var p = net.params(tape)
    var worst = List[Float64](length=len(p), fill=0)
    for s in range(3):
        var ref_ = cpu_grads(net, tape, p, s)
        var keep = List[List[Scalar[DT]]]()
        seed(77 + s)
        for k in range(len(p)):
            var v = p[k].values()
            keep.append(v.copy())
            for i in range(len(v)):
                p[k].set_value(i, v[i] * Scalar[DT](1.0 + (random_float64() * 2.0 - 1.0) * 2.0 ** -11))
        var bt = batch(net, s)
        var x = bt[0].copy()
        for i in range(len(x)):
            x[i] *= Scalar[DT](1.0 + (random_float64() * 2.0 - 1.0) * 2.0 ** -11)
        tape.reset()
        tape.zero_grad()
        var loss = net.loss(p, tape.input(net.b, net.din, x), tape.input(net.b, net.dout, bt[1].copy()))
        loss.backward()
        for k in range(len(p)):
            var gk = p[k].grads()
            var mx = Float64(0)
            var e = Float64(0)
            for i in range(len(gk)):
                mx = max(mx, abs(Float64(ref_[1][k][i])))
                e = max(e, abs(Float64(gk[i]) - Float64(ref_[1][k][i])))
            worst[k] = max(worst[k], e / max(mx, 1e-30))
        tape.reset()
        for k in range(len(p)):
            for i in range(len(keep[k])):
                p[k].set_value(i, keep[k][i])
    var line = String("     ", name, ": fp32 gradient moves under 2^-11 input noise:")
    for k in range(len(p)):
        line += String(" ", tape.data[].nodes[k].name, "=", worst[k])
    print(line)
    return worst^


def gate_grads(a: ANE, name: String, net: Net, mutant: Bool) raises -> Float64:
    """Worst relative gradient error over three batches (and prints them).
    Each batch runs at the largest loss scale (a power of two from 32768
    down) whose gradients stay finite, as a dynamic scaler settles."""
    var tape = Tape()
    var p = net.params(tape)
    var st = AneStep(a, net, tape, p, mutant)
    print("  ", name, ": compile+load", st.prog.compile_ms, "ms,", len(st.step.grads), "gradients")
    var worst = Float64(0)
    for s in range(3):
        var ref_ = cpu_grads(net, tape, p, s)
        var bt = batch(net, s)
        var seed_v = Float32(32768.0)
        var l = st.run(a, tape, bt[0], bt[1], seed_v)
        while l != l and seed_v > 1.0:
            seed_v *= 0.5
            l = st.run(a, tape, bt[0], bt[1], seed_v)
        var lerr = abs(l - ref_[0]) / max(abs(ref_[0]), 1e-30)
        var line = String("     batch ", s, " (scale ", Int(seed_v), "): loss ", l, " vs ", ref_[0],
                          " (rel ", lerr, ") | grad err")
        for k in range(len(st.step.grads)):
            var id = st.step.grads[k]
            var gp = tape.data[].grad_ptr(id)
            ref want = ref_[1][id]
            var mx = Float64(0)
            var e = Float64(0)
            for i in range(len(want)):
                mx = max(mx, abs(Float64(want[i])))
                e = max(e, abs(Float64(gp[unsafe_offset=i]) - Float64(want[i])))
            var rel = e / max(mx, 1e-30)
            worst = max(worst, rel)
            line += String(" ", tape.data[].nodes[id].name, "=", rel)
        worst = max(worst, lerr)
        print(line)
    st.prog.free(a)
    return worst


def train(a: ANE, net: Net, steps: Int, on_ane: Bool, mut curve: List[Float64]) raises -> Float64:
    """Mean step time (us) over the steps after the first 20."""
    var tape = Tape()
    var p = net.params(tape)
    var opt = Adam(lr=Scalar[DT](3e-3))
    var st = Optional[AneStep]()
    if on_ane:
        st = AneStep(a, net, tape, p)
    var scale = Float32(32768.0)
    var good = 0
    var skipped = 0
    var t_sum = Float64(0)
    for s in range(steps):
        var bt = batch(net, s)
        var t0 = perf_counter_ns()
        var l: Float64
        if on_ane:
            l = st.value().run(a, tape, bt[0], bt[1], scale)
            if l != l:  # an overflowed gradient: skip the step, halve the scale
                scale *= 0.5
                skipped += 1
                good = 0
                continue
            good += 1
            if good == 500 and scale < 32768.0:
                scale *= 2.0
                good = 0
        else:
            tape.reset()
            tape.zero_grad()
            var loss = net.loss(p, tape.input(net.b, net.din, bt[0].copy()),
                                tape.input(net.b, net.dout, bt[1].copy()))
            loss.backward()
            l = Float64(loss.item())
        var t = tape
        opt.step["cpu"](t, None)
        if s >= 20:
            t_sum += Float64(perf_counter_ns() - t0) / 1e3
        curve.append(l)
    if on_ane:
        print("     loss scale at the end", scale, ", skipped steps", skipped)
        st.value().prog.free(a)
    return t_sum / Float64(steps - 20)


def tail_mean(c: List[Float64], n: Int) -> Float64:
    var s = Float64(0)
    for i in range(len(c) - n, len(c)):
        s += c[i]
    return s / Float64(n)


def main() raises:
    var a = ANE()
    var ok = True
    print("0. conditioning (fp32 tape only)")
    _ = sensitivity("a. MLP", Net(64, 17, 64, 6, 0))
    _ = sensitivity("b. every-op", Net(64, 17, 64, 6, 1))
    print("1. gradients, ANE (fp16) vs CPU tape (fp32), bound", GRAD_TOL)
    var e = gate_grads(a, "a. MLP 17-64relu-64tanh-6", Net(64, 17, 64, 6, 0), False)
    ok = ok and e < GRAD_TOL
    print("     worst", e, "OK" if e < GRAD_TOL else "FAIL")
    e = gate_grads(a, "b. every-op graph", Net(64, 17, 64, 6, 1), False)
    ok = ok and e < GRAD_TOL
    print("     worst", e, "OK" if e < GRAD_TOL else "FAIL")
    e = gate_grads(a, "c. MUTANT tanh bwd", Net(64, 17, 64, 6, 0), True)
    ok = ok and e >= GRAD_TOL
    print("     worst", e, "caught (OK)" if e >= GRAD_TOL else "NOT CAUGHT (FAIL)")

    comptime STEPS = 2000
    print("2. fit sin(x), 1-64tanh-64tanh-1, batch 256, Adam 3e-3,", STEPS, "steps")
    var net = Net(256, 1, 64, 1, 2)
    var c_cpu = List[Float64]()
    var c_ane = List[Float64]()
    var us_cpu = train(a, net, STEPS, False, c_cpu)
    var us_ane = train(a, net, STEPS, True, c_ane)
    var line = String("     step:")
    for s in [0, 100, 250, 500, 1000, 1500, STEPS - 1]:
        if s < len(c_cpu) and s < len(c_ane):
            line += String(" ", s, ": ", c_cpu[s], " / ", c_ane[s], ";")
    print(line, "(CPU / ANE)")
    var f_cpu = tail_mean(c_cpu, 100)
    var f_ane = tail_mean(c_ane, 100)
    var reach = f_ane < 2.0 * f_cpu + 1e-4
    ok = ok and reach
    print("     final loss (mean of the last 100): CPU", f_cpu, "ANE", f_ane, "OK" if reach else "FAIL")
    print("     us per step: CPU tape", us_cpu, "| ANE step + host Adam", us_ane)
    print("ALL OK" if ok else "FAILED")
    if not ok:
        raise Error("test_ane_step failed")
