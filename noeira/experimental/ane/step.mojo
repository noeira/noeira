"""`AneGrad`: a functional tape's gradient step run on the Neural Engine.

    var a = ANE()
    var x = tape.input(B, IN, xs)          # ONE eager forward, to record
    var y = tape.input(B, OUT, ys)
    var g = AneGrad(a, tape, F.mse(net(x), y), [x, y])   # emit + compile
    tape.reset()
    # each step:
    g.set_data(a, 0, x_ptr); g.set_data(a, 1, y_ptr)
    var loss = g.run(a, tape, scale)   # the parameters' grads are now set
    opt.step["cpu"](tape, None)        # any host optimizer

`run` writes every parameter's current value (fp32 -> fp16), evaluates the
program, and writes each gradient / (`scale` * `step.grad_div`) into the
tape's parameter grads. It returns NaN when a gradient is not finite (the
scale is too large: halve it and skip the step). `start_scale()` is a
first scale that keeps the effective one (scale * grad_div) at 32768. `t_write`, `t_eval`, `t_read` accumulate the time of
each part (ns).
"""

from std.time import perf_counter_ns
from noeira.experimental.functional import Tape, FTensor
from .runtime import ANE, Program
from .emit import emit_grad_step, GradStep


def _nan() -> Float64:
    var z = Float64(0)
    return z / z


struct AneGrad(Movable):
    var step: GradStep
    var prog: Program
    var data_args: List[Int]
    """The program argument each data tensor (in `data` order) feeds."""
    var t_write: Int
    var t_eval: Int
    var t_read: Int

    def __init__(out self, a: ANE, tape: Tape, loss: FTensor, data: List[FTensor],
                 mutant: Bool = False, dump: String = "") raises:
        """`dump`: write the MIL text there (to bisect a refused program
        with `probe_mil.mojo`)."""
        self.step = emit_grad_step(tape, loss, mutant=mutant)
        if dump != "":
            with open(dump, "w") as f:
                f.write(self.step.text)
        self.prog = Program(a, self.step.text)
        if len(self.prog.in_layout) != len(self.step.args) + 1:
            raise Error("AneGrad: program inputs != tape arguments + seed")
        self.data_args = List[Int]()
        for d in data:
            var k = -1
            for j in range(len(self.step.args)):
                if self.step.args[j] == d.id:
                    k = j
            if k < 0:
                raise Error("AneGrad: a data tensor is not a program argument")
            self.data_args.append(k)
        self.t_write = 0
        self.t_eval = 0
        self.t_read = 0

    def set_data(self, a: ANE, k: Int, src: Pointer[Float32, _]) raises:
        self.prog.write(a, self.data_args[k], src)

    def run(mut self, a: ANE, mut tape: Tape, scale: Float32) raises -> Float64:
        ref td = tape.data[]
        var t0 = perf_counter_ns()
        for k in range(len(self.step.args)):
            var id = self.step.args[k]
            if id < td.n_params:
                self.prog.write(a, k, Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(td.val_ptr(id))))
        var s: List[Float32] = [scale]
        self.prog.write(a, len(self.step.args), Pointer(to=s[0]))
        var t1 = perf_counter_ns()
        self.prog.run(a)
        var t2 = perf_counter_ns()
        var l: List[Float32] = [0.0]
        self.prog.read(a, 0, Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(l.unsafe_ptr())))
        var finite = True
        for k in range(len(self.step.grads)):
            var id = self.step.grads[k]
            var gp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(td.grad_ptr(id)))
            self.prog.read(a, k + 1, gp, Float32(1.0 / (Float64(scale) * self.step.grad_div)))
            for i in range(self.step.shapes[id][0] * self.step.shapes[id][1]):
                var v = gp[unsafe_offset=i]
                if v != v or abs(v) > 1e30:
                    finite = False
        var t3 = perf_counter_ns()
        self.t_write += Int(t1 - t0)
        self.t_eval += Int(t2 - t1)
        self.t_read += Int(t3 - t2)
        _ = s^
        return Float64(l[0]) if finite else _nan()

    def start_scale(self) -> Float32:
        var s = Float64(32768.0)
        while s * self.step.grad_div > 32768.0 and s > 2.0 ** -14:
            s *= 0.5
        return Float32(s)

    def free(self, a: ANE) raises:
        self.prog.free(a)
