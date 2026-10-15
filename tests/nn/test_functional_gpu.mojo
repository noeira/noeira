"""Functional mode on the GPU — gates.

The CPU tape is the oracle here (it is itself gated bit-identical against
the comptime nn and against finite differences, `test_functional_mode`).

1. OPS: the nine expressions of the CPU gate, value and every gradient,
   GPU tape vs CPU tape.
2. MATMUL SHAPES: `((a @ b) * w).sum()` over shapes that take each GEMM
   route — MAX's matmul (no dimension 1) and the naive kernel (a dimension
   of 1: an acting row, a `[B, 1]` head, a `[*, 1]` input) — value, da, db.
3. PARITY vs the comptime nn ON THE GPU: `Sequential[Linear, Tanh, Linear]`
   + MSE, the same network on a GPU tape.
4. MUTANT: one CPU gradient nudged by 1e-3 must fail the comparison.
5. REPLAY: two GPU steps after `reset`, same loss, no new node slot.

Tolerance: |gpu - cpu| <= 1e-4 · max(1, |cpu|) — GPU sums run in another
order and Metal's transcendentals are not libm's.

Run: pixi run -e apple mojo run -I . tests/nn/test_functional_gpu.mojo
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse import mse_forward, mse_backward
from noeira.nn.functional import Tape, FTensor
import noeira.nn.functional.F as F


comptime TOL: Float64 = 1e-4


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


struct Cmp:
    var n: Int
    var differ: Int
    var worst: Float64
    var max_ref: Float64

    def __init__(out self):
        self.n = 0
        self.differ = 0
        self.worst = 0
        self.max_ref = 0

    def add(mut self, got: Scalar[DT], want: Scalar[DT]):
        self.n += 1
        if got != want:
            self.differ += 1
        var w = Float64(want)
        var e = abs(Float64(got) - w) / max(1.0, abs(w))
        if e > self.worst:
            self.worst = e
        if abs(w) > self.max_ref:
            self.max_ref = abs(w)

    def add_all(mut self, got: List[Scalar[DT]], want: List[Scalar[DT]]):
        for k in range(len(want)):
            self.add(got[k], want[k])

    def ok(self) -> Bool:
        return self.n > 0 and self.max_ref > 0 and self.worst <= TOL

    def line(self) -> String:
        return String(
            self.n, " compared, ", self.differ, " not bit-identical, worst rel ",
            self.worst, "  ", "OK" if self.ok() else "FAIL",
        )


# ── 1. ops ───────────────────────────────────────────────────────────────
def _expr(a: FTensor, r: FTensor, s: FTensor, w: FTensor, which: Int) raises -> FTensor:
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
    elif which == 8:
        return (-(a - 0.5) * (a + r)).mean()
    elif which == 9:
        # A DEAD branch: recorded, requires grad, never reaches the loss.
        # The GPU backward no longer zero-fills, so it must skip it.
        var dead = (a.T() @ a * s).relu()
        _ = dead
        return (a * r).sum()
    elif which == 10:
        # One node used three times: written once, then accumulated twice.
        var h = (a @ w).tanh()
        return (h * h + h).sum() + (a * r).sum()
    else:
        # Case 9 with its branch LIVE: the warm-up for every GPU case. It
        # leaves slots 4-7 holding real gradients at case 9's shapes, so a
        # backward that walked the dead branch would read them.
        var live = (a.T() @ a * s).relu()
        return live.sum() + (a * r).sum()


def _run_expr(
    tape: Tape, which: Int, warm: Int = -1
) raises -> List[List[Scalar[DT]]]:
    """Value first, then each parameter's gradient. `warm` >= 0 runs that
    expression's step first on the same tape, then `reset` + `zero_grad`:
    the measured step then REUSES node slots whose buffers hold the warm
    step's values and gradients — stale memory, where a fresh tape's
    buffers happen to read as zeros. ⚠ A slot only GROWS: a warm node smaller
    than the measured one is reallocated, fresh and zero, and hides the
    hazard again — so the warm-up (case 11) has case 9's shapes."""
    seed(100 + which)
    var a = tape.param(3, 4, _rand(12, 1.0))
    var r = tape.param(1, 4, _rand(4, 1.0))
    var s = tape.param(1, 1, _rand(1, 1.0))
    var w = tape.param(4, 2, _rand(8, 1.0))
    if warm >= 0:
        _expr(a, r, s, w, warm).backward()
        tape.reset()
        tape.zero_grad()
    var f = _expr(a, r, s, w, which)
    f.backward()
    var grads = List[List[Scalar[DT]]]()
    grads.append(f.values())
    grads.append(a.grads())
    grads.append(r.grads())
    grads.append(s.grads())
    grads.append(w.grads())
    return grads^


def _ops(ctx: DeviceContext, mutant: Bool) raises -> Bool:
    var names: List[String] = [
        "a + r (row bcast)", "r - a*s (scalar bcast)", "a*r mean",
        "a / (r*r+1)", "(a@w).tanh()**2", "2 - a.T@a, sum_rows",
        "relu*3 + sigmoid/2", "gelu+gelu_tanh+mish+swish+elu",
        "-(a-.5)*(a+r) mean", "dead branch", "node used 3x",
    ]
    var all_ok = True
    for which in range(11):
        var want = _run_expr(Tape(), which)
        var got = _run_expr(Tape(ctx), which, warm=11)
        if mutant:
            want[1][0] += Scalar[DT](1e-3)
        var c = Cmp()
        for t in range(len(want)):
            c.add_all(got[t], want[t])
        if not mutant:
            print("  ", names[which], ": ", c.line(), sep="")
            all_ok = all_ok and c.ok()
        else:
            all_ok = all_ok and not c.ok()  # the nudge must show in EVERY case
    return all_ok


# ── 2. matmul shapes ─────────────────────────────────────────────────────
def _mm_case(tape: Tape, m: Int, k: Int, n: Int) raises -> List[List[Scalar[DT]]]:
    seed(m * 10007 + k * 101 + n)
    var a = tape.param(m, k, _rand(m * k, 1.0))
    var b = tape.param(k, n, _rand(k * n, 1.0))
    var w = tape.input(m, n, _rand(m * n, 1.0))
    var f = ((a @ b) * w).sum()
    f.backward()
    var res = List[List[Scalar[DT]]]()
    res.append(f.values())
    res.append(a.grads())
    res.append(b.grads())
    return res^


def _matmul_shapes(ctx: DeviceContext) raises -> Bool:
    var shapes: List[List[Int]] = [
        [5, 7, 13], [64, 8, 64], [64, 64, 4], [33, 65, 17], [16, 32, 32],
        [1, 23, 256], [256, 256, 1], [64, 1, 32], [3, 1, 4], [1, 1, 1],
    ]
    var all_ok = True
    for i in range(len(shapes)):
        var m = shapes[i][0]
        var k = shapes[i][1]
        var n = shapes[i][2]
        var want = _mm_case(Tape(), m, k, n)
        var got = _mm_case(Tape(ctx), m, k, n)
        var c = Cmp()
        for t in range(3):
            c.add_all(got[t], want[t])
        var route = "naive" if (m == 1 or k == 1 or n == 1) else "MAX  "
        print("  [", m, "x", k, "] @ [", k, "x", n, "]  ", route, "  ", c.line(), sep="")
        all_ok = all_ok and c.ok()
    return all_ok


# ── 3. parity vs the comptime nn on the GPU ──────────────────────────────
def _parity(ctx: DeviceContext) raises -> Bool:
    comptime B = 200  # > ROW_CHUNK: the bias gradient takes the chunked path
    comptime D = 7
    comptime H = 48
    comptime O = 3
    comptime NET = Sequential[Linear[D, H], Tanh[H], Linear[H, O]]
    seed(7)
    var w1 = _rand(D * H, 0.5)
    var b1 = _rand(H, 0.2)
    var w2 = _rand(H * O, 0.5)
    var b2 = _rand(O, 0.2)
    var xs = _rand(B * D, 1.0)
    var ys = _rand(B * O, 1.0)

    var net = NET.make["gpu", Deterministic](ctx)
    for k in range(D * H):
        net.children[0].weight.val.data[k] = w1[k]
    for k in range(H):
        net.children[0].bias.val.data[k] = b1[k]
    for k in range(H * O):
        net.children[2].weight.val.data[k] = w2[k]
    for k in range(O):
        net.children[2].bias.val.data[k] = b2[k]
    net.children[0].weight.val.upload(ctx)
    net.children[0].bias.val.upload(ctx)
    net.children[2].weight.val.upload(ctx)
    net.children[2].bias.val.upload(ctx)
    net.zero_grad["gpu"](ctx)
    var x = Tensor.alloc(B * D)
    var y = Tensor.alloc(B * O)
    for k in range(B * D):
        x.data[k] = xs[k]
    for k in range(B * O):
        y.data[k] = ys[k]
    x.upload(ctx)
    y.upload(ctx)
    var out = Tensor.alloc_gpu(ctx, B * O)
    var go = Tensor.alloc_gpu(ctx, B * O)
    var gx = Tensor.alloc_gpu(ctx, B * D)
    net.forward["gpu", B](TensorRefs[1](x), out, ctx)
    mse_backward["gpu", B, O](out, y, go, ctx)
    net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), ctx)
    out.download(ctx)
    net.children[0].weight.grd.download(ctx)
    net.children[0].bias.grd.download(ctx)
    net.children[2].weight.grd.download(ctx)
    net.children[2].bias.grd.download(ctx)
    var ref_loss = mse_forward[B, O](out, y)

    var tape = Tape(ctx)
    var fw1 = tape.param(D, H, w1)
    var fb1 = tape.param(1, H, b1)
    var fw2 = tape.param(H, O, w2)
    var fb2 = tape.param(1, O, b2)
    var fx = tape.input(B, D, xs)
    var pred = F.linear(F.linear(fx, fw1, fb1).tanh(), fw2, fb2)
    var loss = F.mse(pred, tape.input(B, O, ys))
    loss.backward()

    var all_ok = True
    var rows: List[String] = ["forward", "loss", "dW1", "db1", "dW2", "db2"]
    for r in range(6):
        var c = Cmp()
        if r == 0:
            var p = pred.values()
            for k in range(B * O):
                c.add(p[k], out.data[k])
        elif r == 1:
            c.add(loss.item(), ref_loss)
        elif r == 2:
            c.add_all(fw1.grads(), net.children[0].weight.grd.data)
        elif r == 3:
            c.add_all(fb1.grads(), net.children[0].bias.grd.data)
        elif r == 4:
            c.add_all(fw2.grads(), net.children[2].weight.grd.data)
        else:
            c.add_all(fb2.grads(), net.children[2].bias.grd.data)
        print("  ", rows[r], ": ", c.line(), sep="")
        all_ok = all_ok and c.ok()
    return all_ok


# ── 5. replay ────────────────────────────────────────────────────────────
def _replay(ctx: DeviceContext) raises -> Bool:
    seed(3)
    var tape = Tape(ctx)
    var w = tape.param(4, 8, _rand(32, 0.5))
    var b = tape.zeros(1, 8)
    var v = tape.param(8, 1, _rand(8, 0.5))
    var xs = _rand(16 * 4, 1.0)
    var ys = _rand(16, 1.0)
    var losses = List[Scalar[DT]]()
    var slots = List[Int]()
    for _ in range(2):
        tape.reset()
        tape.zero_grad()
        var loss = F.mse(
            (tape.input(16, 4, xs) @ w + b).tanh() @ v, tape.input(16, 1, ys)
        )
        loss.backward()
        losses.append(loss.item())
        slots.append(len(tape.data[].nodes))
    var ok = losses[0] == losses[1] and slots[0] == slots[1]
    print("  2 steps: loss ", losses[0], " / ", losses[1], ", node slots ",
          slots[0], " / ", slots[1], "  ", "OK" if ok else "FAIL", sep="")
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var ok = True
    print("ops: GPU tape (after a warm-up step on the same tape) vs CPU tape")
    ok = _ops(ctx, False) and ok
    var caught = _ops(ctx, True)
    print("  mutant (one CPU grad + 1e-3) caught in all 11 cases:", caught)
    ok = caught and ok
    print("matmul shapes: GPU tape vs CPU tape")
    ok = _matmul_shapes(ctx) and ok
    print("parity vs comptime Sequential[Linear, Tanh, Linear] + MSE, both on the GPU")
    ok = _parity(ctx) and ok
    print("replay after reset (GPU)")
    ok = _replay(ctx) and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional GPU gates failed")
