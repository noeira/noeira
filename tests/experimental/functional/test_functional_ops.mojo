"""More ops on the functional tape — gates (CPU and GPU): exp, log, sqrt,
clamp, softmax, log-softmax, cross-entropy, dropout.

1. GRADIENT CHECK (CPU) against central differences, with a mutant that
   must fail. Clamp's inputs are kept 0.05 away from its bounds (a central
   difference straddling a kink is wrong, not the gradient).
2. GPU vs CPU, values and gradients, after a warm-up step; dropout's mask
   must be the SAME on both (one integer hash).
3. PROPERTIES: softmax rows sum to 1; log_softmax == log(softmax); the
   fused cross-entropy == the composite −Σ(log_softmax · onehot) / rows,
   value and gradient; dropout drops ≈ p, the same seed gives the same
   mask and another seed another, `training=False` is the identity; an
   out-of-range target raises.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_ops.mojo
"""

from std.math import sqrt as fsqrt
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F


comptime NCASE = 8
comptime TOL_FD: Float64 = 5e-3
comptime TOL_GPU: Float64 = 1e-4


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


def _names() -> List[String]:
    return [
        "exp", "log(x*x + 1)", "sqrt(x*x + 0.5)", "clamp(-0.3, 0.4)",
        "softmax [2,3,4]", "log_softmax [2,3,4]", "cross_entropy [2,3,4] vs [2,3]",
        "dropout(0.3, seed 7)",
    ]


def _a_values() -> List[Scalar[DT]]:
    seed(13)
    var v = _rand(24, 1.0)
    for k in range(24):  # away from clamp's kinks at -0.3 and 0.4
        if abs(Float64(v[k]) + 0.3) < 0.05 or abs(Float64(v[k]) - 0.4) < 0.05:
            v[k] += Scalar[DT](0.12)
    return v^


def _expr(tape: Tape, a: FTensor, which: Int) raises -> FTensor:
    seed(99)
    var C = tape.input([2, 3, 4], _rand(24, 1.0))
    var A3 = a.reshape([2, 3, 4])
    if which == 0:
        return ((a * 0.5).exp().reshape([2, 3, 4]) * C).sum()
    elif which == 1:
        return (a * a + 1.0).log().sum()
    elif which == 2:
        return (a * a + 0.5).sqrt().sum()
    elif which == 3:
        return (a.clamp(-0.3, 0.4).reshape([2, 3, 4]) * C).sum()
    elif which == 4:
        return (A3.softmax() * C).sum()
    elif which == 5:
        return (A3.log_softmax() * C).sum()
    elif which == 6:
        var tg = tape.input([2, 3], [0, 3, 1, 2, 2, 0])
        return F.cross_entropy(A3 * 2.0, tg)
    else:
        return (F.dropout(A3, 0.3, seed=7) * C).sum()


# ── 1. finite differences ────────────────────────────────────────────────
def _grad_check(mutant: Bool) raises -> Bool:
    var names = _names()
    var all_ok = True
    for which in range(NCASE):
        var tape = Tape()
        var a = tape.param(6, 4, _a_values(), name="a")
        _expr(tape, a, which).backward()
        var g_all = a.grads()
        var worst: Float64 = 0
        var nonzero = 0
        comptime EPS: Float64 = 1e-2
        for k in range(24):
            var g = Float64(g_all[k])
            if mutant and k == 0:
                g = g * 1.1 + 0.05
            var keep = a.value(k)
            a.set_value(k, keep + Scalar[DT](EPS))
            tape.reset()
            var fp = Float64(_expr(tape, a, which).item())
            a.set_value(k, keep - Scalar[DT](EPS))
            tape.reset()
            var fm = Float64(_expr(tape, a, which).item())
            a.set_value(k, keep)
            var fd = (fp - fm) / (2.0 * EPS)
            if g != 0.0:
                nonzero += 1
            worst = max(worst, abs(fd - g) / max(1.0, abs(fd)))
        var ok = worst < TOL_FD and nonzero > 0
        if not mutant:
            print("    ", names[which], ": ", nonzero, " nonzero, worst rel err ",
                  worst, "  ", "OK" if ok else "FAIL", sep="")
            all_ok = all_ok and ok
        else:
            all_ok = all_ok and not ok  # every case must see the nudge
    return all_ok


# ── 2. GPU vs CPU ────────────────────────────────────────────────────────
def _run(tape: Tape, which: Int, warm: Bool) raises -> List[List[Scalar[DT]]]:
    var a = tape.param(6, 4, _a_values(), name="a")
    if warm:
        _expr(tape, a, (which + 1) % NCASE).backward()
        tape.reset()
        tape.zero_grad()
    var f = _expr(tape, a, which)
    f.backward()
    var out = List[List[Scalar[DT]]]()
    out.append(f.values())
    out.append(a.grads())
    return out^


def _gpu_vs_cpu(ctx: DeviceContext) raises -> Bool:
    var names = _names()
    var all_ok = True
    for which in range(NCASE):
        var want = _run(Tape(), which, False)
        var got = _run(Tape(ctx), which, True)
        var worst: Float64 = 0
        var differ = 0
        var n = 0
        for t in range(2):
            for k in range(len(want[t])):
                n += 1
                if got[t][k] != want[t][k]:
                    differ += 1
                var w = Float64(want[t][k])
                worst = max(worst, abs(Float64(got[t][k]) - w) / max(1.0, abs(w)))
        var ok = worst <= TOL_GPU
        if which == 7:  # dropout: the masks must match exactly
            var zeros_match = True
            for k in range(24):
                zeros_match = zeros_match and ((got[1][k] == 0) == (want[1][k] == 0))
            ok = ok and zeros_match
        print("    ", names[which], ": ", n, " compared, ", differ,
              " not bit-identical, worst rel ", worst, "  ", "OK" if ok else "FAIL", sep="")
        all_ok = all_ok and ok
    return all_ok


# ── 3. properties ────────────────────────────────────────────────────────
def _check(name: String, ok: Bool) -> Bool:
    print("    ", name, ": ", "OK" if ok else "FAIL", sep="")
    return ok


def _props(ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    seed(21)
    var zv = _rand(4 * 5, 3.0)
    var tape = Tape(ctx.value()) if ctx else Tape()
    var z = tape.param(4, 5, zv, name="z")
    var sm = z.softmax().values()
    var sums_ok = True
    for r in range(4):
        var s: Float64 = 0
        for j in range(5):
            s += Float64(sm[r * 5 + j])
        sums_ok = sums_ok and abs(s - 1.0) < 1e-5
    ok = _check("softmax rows sum to 1", sums_ok) and ok
    var ls = z.log_softmax().values()
    var lsm = z.softmax().log().values()
    var ls_ok = True
    for k in range(20):
        ls_ok = ls_ok and abs(Float64(ls[k]) - Float64(lsm[k])) < 1e-5
    ok = _check("log_softmax == log(softmax)", ls_ok) and ok

    # fused cross-entropy vs the composite
    var tgt: List[Scalar[DT]] = [1, 4, 0, 2]
    var onehot = List[Scalar[DT]](length=20, fill=0)
    for r in range(4):
        onehot[r * 5 + Int(tgt[r])] = 1
    tape.reset()
    tape.zero_grad()
    var fused = F.cross_entropy(z, tape.input([4], tgt.copy()))
    fused.backward()
    var gf = z.grads()
    var vf = fused.item()
    tape.reset()
    tape.zero_grad()
    var comp = (-(z.log_softmax() * tape.input(4, 5, onehot)).sum(1)).mean()
    comp.backward()
    var gc = z.grads()
    var x_ok = abs(Float64(vf) - Float64(comp.item())) < 1e-5
    for k in range(20):
        x_ok = x_ok and abs(Float64(gf[k]) - Float64(gc[k])) < 1e-5
    ok = _check("cross_entropy == −Σ(log_softmax·onehot)/rows, value and grad", x_ok) and ok

    # dropout statistics
    tape.reset()
    comptime N = 10000
    var ones = List[Scalar[DT]](length=N, fill=1)
    var x = tape.input(100, 100, ones)
    var d1 = F.dropout(x, 0.25, seed=1).values()
    var d1b = F.dropout(x, 0.25, seed=1).values()
    var d2 = F.dropout(x, 0.25, seed=2).values()
    var dropped = 0
    var same = True
    var other = False
    var mean: Float64 = 0
    for k in range(N):
        if d1[k] == 0:
            dropped += 1
        same = same and d1[k] == d1b[k]
        other = other or d1[k] != d2[k]
        mean += Float64(d1[k])
    var frac = Float64(dropped) / Float64(N)
    var sigma = fsqrt(0.25 * 0.75 / Float64(N))
    ok = _check(String("dropout drops ", frac, " (p 0.25, within 4 sigma)"),
                abs(frac - 0.25) < 4.0 * sigma) and ok
    ok = _check(String("dropout keeps the mean (", mean / Float64(N), ")"),
                abs(mean / Float64(N) - 1.0) < 0.05) and ok
    ok = _check("same seed, same mask; another seed, another", same and other) and ok
    var ident = F.dropout(x, 0.25, seed=1, training=False)
    ok = _check("training=False is the identity", ident.id == x.id) and ok
    if not ctx:
        var bad = False
        try:
            _ = F.cross_entropy(z, tape.input([4], [0, 1, 5, 2]))
        except:
            bad = True
        ok = _check("an out-of-range target raises (CPU)", bad) and ok
    return ok


def main() raises:
    var ok = True
    print("gradient check vs central differences (CPU, eps 1e-2)")
    ok = _grad_check(False) and ok
    var caught = _grad_check(True)
    print("    mutant (one gradient x1.1 + 0.05) caught in every case:", caught)
    ok = caught and ok
    var ctx = DeviceContext()
    print("GPU vs CPU (", ctx.name(), ", after a warm-up step)", sep="")
    ok = _gpu_vs_cpu(ctx) and ok
    print("properties, CPU")
    ok = _props(None) and ok
    print("properties, GPU")
    ok = _props(ctx) and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional ops gates failed")
