"""Functional mode (`noeira/experimental/functional/`) — gates.

1. PARITY vs the comptime nn: `Sequential[Linear, ReLU|Tanh, Linear]` + MSE,
   forward and backward, against the same network written with operators on
   a tape (`F.linear(x, w1, b1).relu()` …). Same weights, same input. The
   comptime path is the oracle; each row prints how many elements differ
   bit for bit and the largest absolute difference.
2. GRADIENT CHECK of every op against central finite differences, through
   broadcasting (a `[1, C]` and a `[1, 1]` operand) — with a MUTANT that
   scales one analytic gradient by 1.1 and must FAIL, so a check that
   cannot fail is caught.
3. REPLAY: after `reset`, a second step records the same number of nodes,
   allocates no new node, and reproduces the first step's loss and grads.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_mode.mojo
"""

from std.random import seed, random_float64

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse import mse_forward, mse_backward
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F


comptime B = 5
comptime D = 7
comptime H = 13
comptime O = 3


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


struct Diff:
    var n: Int
    var differ: Int
    var max_abs: Float64
    var max_ref: Float64
    """Largest |oracle value|: zero would make the row vacuous (two zero
    gradients agree bit for bit)."""

    def __init__(out self):
        self.n = 0
        self.differ = 0
        self.max_abs = 0
        self.max_ref = 0

    def add(mut self, a: Scalar[DT], b: Scalar[DT]):
        self.n += 1
        if a != b:
            self.differ += 1
        var d = abs(Float64(a) - Float64(b))
        if d > self.max_abs:
            self.max_abs = d
        if abs(Float64(b)) > self.max_ref:
            self.max_ref = abs(Float64(b))

    def report(self, name: String) -> Bool:
        var ok = self.n > 0 and self.max_ref > 0 and self.max_abs < 1e-5
        print(
            "  ", name, ": ", self.n, " compared (max |ref| ", self.max_ref,
            "), ", self.differ, " not bit-identical, max |d| = ", self.max_abs,
            "  ", "OK" if ok else "FAIL", sep="",
        )
        return ok


def _parity[ACT: Int]() raises -> Bool:
    """ACT 0 = ReLU, 1 = Tanh."""
    comptime NET0 = Sequential[Linear[D, H], ReLU[H], Linear[H, O]]
    comptime NET1 = Sequential[Linear[D, H], Tanh[H], Linear[H, O]]
    seed(7)
    var w1 = _rand(D * H, 0.5)
    var b1 = _rand(H, 0.2)
    var w2 = _rand(H * O, 0.5)
    var b2 = _rand(O, 0.2)
    var xs = _rand(B * D, 1.0)
    var ys = _rand(B * O, 1.0)

    # ── the comptime oracle ──
    var x = Tensor.alloc(B * D)
    var y = Tensor.alloc(B * O)
    for k in range(B * D):
        x.data[k] = xs[k]
    for k in range(B * O):
        y.data[k] = ys[k]
    var out = Tensor.alloc(B * O)
    var go = Tensor.alloc(B * O)
    var gx = Tensor.alloc(B * D)
    var ref_loss: Scalar[DT]
    var rw1 = List[Scalar[DT]]()
    var rb1 = List[Scalar[DT]]()
    var rw2 = List[Scalar[DT]]()
    var rb2 = List[Scalar[DT]]()

    comptime if ACT == 0:
        var net = NET0.make["cpu", Deterministic]()
        for k in range(D * H):
            net.children[0].weight.val.data[k] = w1[k]
        for k in range(H):
            net.children[0].bias.val.data[k] = b1[k]
        for k in range(H * O):
            net.children[2].weight.val.data[k] = w2[k]
        for k in range(O):
            net.children[2].bias.val.data[k] = b2[k]
        net.zero_grad["cpu"](None)
        net.forward["cpu", B](TensorRefs[1](x), out)
        ref_loss = mse_forward[B, O](out, y)
        mse_backward["cpu", B, O](out, y, go)
        net.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx))
        rw1 = net.children[0].weight.grd.data.copy()
        rb1 = net.children[0].bias.grd.data.copy()
        rw2 = net.children[2].weight.grd.data.copy()
        rb2 = net.children[2].bias.grd.data.copy()
    else:
        var net = NET1.make["cpu", Deterministic]()
        for k in range(D * H):
            net.children[0].weight.val.data[k] = w1[k]
        for k in range(H):
            net.children[0].bias.val.data[k] = b1[k]
        for k in range(H * O):
            net.children[2].weight.val.data[k] = w2[k]
        for k in range(O):
            net.children[2].bias.val.data[k] = b2[k]
        net.zero_grad["cpu"](None)
        net.forward["cpu", B](TensorRefs[1](x), out)
        ref_loss = mse_forward[B, O](out, y)
        mse_backward["cpu", B, O](out, y, go)
        net.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx))
        rw1 = net.children[0].weight.grd.data.copy()
        rb1 = net.children[0].bias.grd.data.copy()
        rw2 = net.children[2].weight.grd.data.copy()
        rb2 = net.children[2].bias.grd.data.copy()

    # ── the same network, written as code on a tape ──
    var tape = Tape()
    var fw1 = tape.param(D, H, w1)
    var fb1 = tape.param(1, H, b1)
    var fw2 = tape.param(H, O, w2)
    var fb2 = tape.param(1, O, b2)
    var fx = tape.input(B, D, xs, requires_grad=True)
    var fy = tape.input(B, O, ys)
    var h = F.linear(fx, fw1, fb1)
    h = h.relu() if ACT == 0 else h.tanh()
    var pred = h @ fw2 + fb2
    var loss = F.mse(pred, fy)
    loss.backward()

    print("parity vs Sequential[Linear, ", "ReLU" if ACT == 0 else "Tanh",
          ", Linear] + MSE  (B=", B, " D=", D, " H=", H, " O=", O, ")", sep="")
    var ok = True
    var d_out = Diff()
    for k in range(B * O):
        d_out.add(pred.value(k), out.data[k])
    ok = d_out.report("forward      ") and ok
    var d_loss = Diff()
    d_loss.add(loss.item(), ref_loss)
    ok = d_loss.report("loss         ") and ok
    var d = Diff()
    for k in range(D * H):
        d.add(fw1.grad(k), rw1[k])
    ok = d.report("dW1          ") and ok
    d = Diff()
    for k in range(H):
        d.add(fb1.grad(k), rb1[k])
    ok = d.report("db1          ") and ok
    d = Diff()
    for k in range(H * O):
        d.add(fw2.grad(k), rw2[k])
    ok = d.report("dW2          ") and ok
    d = Diff()
    for k in range(O):
        d.add(fb2.grad(k), rb2[k])
    ok = d.report("db2          ") and ok
    d = Diff()
    for k in range(B * D):
        d.add(fx.grad(k), gx.data[k])
    ok = d.report("dx           ") and ok
    return ok


# ── 2. finite differences over every op ─────────────────────────────────
def _expr(tape: Tape, a: FTensor, r: FTensor, s: FTensor, w: FTensor, which: Int) raises -> FTensor:
    """One scalar expression per op. `a` [3, 4], `r` a [1, 4] row, `s` a
    [1, 1] scalar, `w` [4, 2]; every case reduces to [1, 1]."""
    if which == 0:
        return (a + r).sum()
    elif which == 1:
        return (r - a * s).sum()
    elif which == 2:
        return (a * r).mean()
    elif which == 3:
        return (a / (r * r + 1.0)).sum()
    elif which == 4:
        return ((a @ w).tanh() ** 2).sum()
    elif which == 5:
        return (2.0 - a.T() @ a).sum_rows().sum()
    elif which == 6:
        return (a.relu() * 3.0 + a.sigmoid() / 2.0).sum()
    elif which == 7:
        return (a.gelu() + a.gelu_tanh() + a.mish() + a.swish() + a.elu()).sum()
    else:
        return (-(a - 0.5) * (a + r)).mean()


def _grad_check(mutant: Bool) raises -> Bool:
    comptime NCASE = 9
    comptime EPS: Float64 = 1e-2
    var names: List[String] = [
        "a + r (row bcast)", "r - a*s (scalar bcast)", "a*r mean", "a / (r*r+1)",
        "(a@w).tanh()**2", "2 - a.T@a, sum_rows", "relu*3 + sigmoid/2",
        "gelu+gelu_tanh+mish+swish+elu", "-(a-.5)*(a+r) mean",
    ]
    var all_ok = True
    for which in range(NCASE):
        seed(100 + which)
        var av = _rand(12, 1.0)
        if which == 6:
            # ReLU's kink: an input within EPS of 0 makes the central
            # difference straddle it (seen: a = 0.0057, fd 2.48 vs the
            # correct 3.12). Keep every input >= 0.1 away from it.
            for k in range(12):
                av[k] += Scalar[DT](0.1) if av[k] >= 0 else Scalar[DT](-0.1)
        var rv = _rand(4, 1.0)
        var sv = _rand(1, 1.0)
        var wv = _rand(8, 1.0)
        var tape = Tape()
        var a = tape.param(3, 4, av)
        var r = tape.param(1, 4, rv)
        var s = tape.param(1, 1, sv)
        var w = tape.param(4, 2, wv)
        var f = _expr(tape, a, r, s, w, which)
        f.backward()
        var worst: Float64 = 0
        var worst_at = String("")
        var params = tape.params()
        for p in range(len(params)):
            var t = params[p]
            for k in range(t.size()):
                var g = Float64(t.grad(k))
                if mutant and p == 0 and k == 0:
                    g *= 1.1
                var keep = t.value(k)
                t.set_value(k, keep + Scalar[DT](EPS))
                tape.reset()
                var fp = Float64(_expr(tape, a, r, s, w, which).item())
                t.set_value(k, keep - Scalar[DT](EPS))
                tape.reset()
                var fm = Float64(_expr(tape, a, r, s, w, which).item())
                t.set_value(k, keep)
                var fd = (fp - fm) / (2.0 * EPS)
                var err = abs(fd - g) / max(1.0, abs(fd))
                if err > worst:
                    worst = err
                    worst_at = String("param ", p, " elem ", k, " value ",
                                      keep, " fd ", fd, " analytic ", g)
        var ok = worst < 5e-3
        if not mutant:
            print("  ", names[which], ": worst rel err ", worst, "  ",
                  "OK" if ok else "FAIL", sep="")
            if not ok:
                print("    worst at", worst_at)
        all_ok = all_ok and ok
    return all_ok


# ── 3. replay ───────────────────────────────────────────────────────────
def _replay() raises -> Bool:
    seed(3)
    var tape = Tape()
    var w = tape.param(4, 8, _rand(32, 0.5))
    var b = tape.zeros(1, 8)
    var v = tape.param(8, 1, _rand(8, 0.5))
    var xs = _rand(16 * 4, 1.0)
    var ys = _rand(16, 1.0)
    var losses = List[Scalar[DT]]()
    var nodes = List[Int]()
    var slots = List[Int]()
    var g0 = List[Scalar[DT]]()
    for step in range(2):
        tape.reset()
        tape.zero_grad()
        var x = tape.input(16, 4, xs)
        var y = tape.input(16, 1, ys)
        var loss = F.mse((x @ w + b).tanh() @ v, y)
        loss.backward()
        losses.append(loss.item())
        nodes.append(tape.num_nodes())
        slots.append(len(tape.data[].nodes))
        g0.append(w.grad(5))
    var ok = (
        losses[0] == losses[1] and nodes[0] == nodes[1]
        and slots[0] == slots[1] and g0[0] == g0[1]
    )
    print("  2 steps: loss ", losses[0], " / ", losses[1], ", nodes ",
          nodes[0], " / ", nodes[1], ", node slots ", slots[0], " / ",
          slots[1], "  ", "OK" if ok else "FAIL", sep="")
    return ok


def main() raises:
    var ok = True
    ok = _parity[0]() and ok
    ok = _parity[1]() and ok
    print("gradient check vs central differences (eps 1e-2, fp32)")
    ok = _grad_check(False) and ok
    var mutant_caught = not _grad_check(True)
    print("  mutant (one grad x1.1) caught:", mutant_caught)
    ok = mutant_caught and ok
    print("replay after reset")
    ok = _replay() and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional mode gates failed")
