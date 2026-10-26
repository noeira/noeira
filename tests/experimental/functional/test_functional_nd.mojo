"""N-d tensors on the functional tape — gates (CPU and GPU).

1. GRADIENT CHECK (CPU) of every N-d op against central differences, with
   a mutant (one gradient x1.1) that must fail: broadcasting at rank 3 and
   4, batched matmul, permute / transpose / T, sum / mean / max along an
   axis (keepdim or not), narrow, concat (a node with itself too), gather
   with repeated indices, rank-3 input to `F.linear`, rank 1.
2. GPU vs CPU: the same expressions on a GPU tape after a warm-up step on
   the same tape (stale buffers), values and every gradient.
3. SHAPE ERRORS raise: batch mismatch, reshape size, concat, broadcast,
   an out-of-range index.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_nd.mojo
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F


comptime NCASE = 10
comptime TOL_FD: Float64 = 5e-3
comptime TOL_GPU: Float64 = 1e-4


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


def _names() -> List[String]:
    return [
        "[2,3,4] + [3,1] (rank-3 broadcast)",
        "[2,3,2,2] * [2,1,1,2] (rank-4 broadcast)",
        "[2,3,4] @ [2,4,5], tanh (batched matmul)",
        "permute([2,0,1]) ** 2 * C",
        "A @ A.T() (batched, T of rank 3)",
        "sum(1) * mean(2) as rank 1",
        "max(2) ** 2 + max(0, keepdim)",
        "concat(narrow, narrow, 2); concat(A, A, 0)",
        "gather [5,3] at [2,3] (repeats)",
        "F.linear([2,3,4], w, b, tanh); reshape([-1])",
    ]


def _expr(tape: Tape, p: List[FTensor], which: Int) raises -> FTensor:
    """`p`: p0 [6,4], p1 [3,1], p2 [2,2], p3 [8,5], p4 [5,3], p5 [4,5],
    p6 [1,5] — all parameters."""
    var A = p[0].reshape([2, 3, 4])
    if which == 0:
        return (A + p[1]).sum()
    elif which == 1:
        var A4 = p[0].reshape([2, 3, 2, 2])
        var M = p[2].reshape([2, 1, 1, 2])
        return ((A4 * M) * A4).mean()
    elif which == 2:
        return (A @ p[3].reshape([2, 4, 5])).tanh().sum()
    elif which == 3:
        seed(77)
        var C = tape.input([4, 2, 3], _rand(24, 1.0))
        return ((A.permute([2, 0, 1]) ** 2) * C).sum()
    elif which == 4:
        return (A @ A.T()).sum()
    elif which == 5:
        return (A.sum(1).sum(1) * A.mean(2).sum(1)).sum()
    elif which == 6:
        return (A.max(2) ** 2).sum() + A.max(0, keepdim=True).sum()
    elif which == 7:
        var c1 = F.concat(A.narrow(2, 1, 2), A.narrow(2, 0, 1), axis=2)
        var c2 = F.concat(A, A, axis=0)
        return (c1 ** 2).sum() + (c2 * c2).mean()
    elif which == 8:
        var idx = tape.input([2, 3], [0, 4, 2, 4, 1, 0])
        var G = F.gather(p[4], idx)
        return (G * G).sum() + G.sum(2).sum()
    else:
        var y = F.linear(A, p[5], p[6], "tanh")
        var flat = y.reshape([-1])
        return (flat * flat).sum() + y.mean()


def _make(tape: Tape, values: List[List[Scalar[DT]]]) raises -> List[FTensor]:
    var shapes: List[List[Int]] = [[6, 4], [3, 1], [2, 2], [8, 5], [5, 3], [4, 5], [1, 5]]
    var p = List[FTensor]()
    for i in range(len(shapes)):
        p.append(tape.param(shapes[i][0], shapes[i][1], values[i].copy()))
    return p^


def _values() -> List[List[Scalar[DT]]]:
    seed(5)
    var v = List[List[Scalar[DT]]]()
    var sizes: List[Int] = [24, 3, 4, 40, 15, 20, 5]
    for i in range(len(sizes)):
        v.append(_rand(sizes[i], 1.0))
    return v^


# ── 1. finite differences ────────────────────────────────────────────────
def _grad_check(mutant: Bool) raises -> Bool:
    var names = _names()
    var all_ok = True
    for which in range(NCASE):
        var tape = Tape()
        var p = _make(tape, _values())
        var f = _expr(tape, p, which)
        f.backward()
        var worst: Float64 = 0
        var checked = 0
        comptime EPS: Float64 = 1e-2
        for q in range(len(p)):
            var t = p[q]
            var g_all = t.grads()
            for k in range(t.size()):
                var g = Float64(g_all[k])
                if mutant and q == 0 and k == 0:
                    g *= 1.1
                var keep = t.value(k)
                t.set_value(k, keep + Scalar[DT](EPS))
                tape.reset()
                var fp = Float64(_expr(tape, p, which).item())
                t.set_value(k, keep - Scalar[DT](EPS))
                tape.reset()
                var fm = Float64(_expr(tape, p, which).item())
                t.set_value(k, keep)
                var fd = (fp - fm) / (2.0 * EPS)
                if fd != 0.0 or g != 0.0:
                    checked += 1
                var err = abs(fd - g) / max(1.0, abs(fd))
                if err > worst:
                    worst = err
        var ok = worst < TOL_FD and checked > 0
        if not mutant:
            print("    ", names[which], ": ", checked, " nonzero derivatives, worst rel err ",
                  worst, "  ", "OK" if ok else "FAIL", sep="")
            all_ok = all_ok and ok
        elif which == 0 or which == 2:
            # the mutant touches p0[0], which every case reads
            all_ok = all_ok and not ok
    return all_ok


# ── 2. GPU vs CPU ────────────────────────────────────────────────────────
def _run(tape: Tape, which: Int, warm: Bool) raises -> List[List[Scalar[DT]]]:
    var p = _make(tape, _values())
    if warm:
        _expr(tape, p, (which + 3) % NCASE).backward()
        tape.reset()
        tape.zero_grad()
    var f = _expr(tape, p, which)
    f.backward()
    var out = List[List[Scalar[DT]]]()
    out.append(f.values())
    for q in range(len(p)):
        out.append(p[q].grads())
    return out^


def _gpu_vs_cpu(ctx: DeviceContext) raises -> Bool:
    var names = _names()
    var all_ok = True
    for which in range(NCASE):
        var want = _run(Tape(), which, False)
        var got = _run(Tape(ctx), which, True)
        var worst: Float64 = 0
        var n = 0
        var differ = 0
        for t in range(len(want)):
            for k in range(len(want[t])):
                n += 1
                if got[t][k] != want[t][k]:
                    differ += 1
                var w = Float64(want[t][k])
                worst = max(worst, abs(Float64(got[t][k]) - w) / max(1.0, abs(w)))
        var ok = worst <= TOL_GPU
        print("    ", names[which], ": ", n, " compared, ", differ,
              " not bit-identical, worst rel ", worst, "  ", "OK" if ok else "FAIL", sep="")
        all_ok = all_ok and ok
    return all_ok


# ── 3. shape errors ──────────────────────────────────────────────────────
def _raises(which: Int) -> Bool:
    try:
        var t = Tape()
        var a = t.input([2, 3, 4], _rand(24, 1.0))
        if which == 0:
            _ = a @ t.input([3, 4, 5], _rand(60, 1.0))
        elif which == 1:
            _ = a.reshape([5, 5])
        elif which == 2:
            _ = F.concat(a, t.input([2, 2, 3], _rand(12, 1.0)), axis=2)
        elif which == 3:
            _ = a + t.input([3, 3], _rand(9, 1.0))
        else:
            var table = t.input(4, 2, _rand(8, 1.0))
            _ = F.gather(table, t.input([2], [1, 7]))
    except:
        return True
    return False


def main() raises:
    var ok = True
    print("gradient check vs central differences (CPU, eps 1e-2)")
    ok = _grad_check(False) and ok
    var caught = _grad_check(True)
    print("    mutant (one gradient x1.1) caught:", caught)
    ok = caught and ok
    var ctx = DeviceContext()
    print("GPU vs CPU (", ctx.name(), ", after a warm-up step on the same tape)", sep="")
    ok = _gpu_vs_cpu(ctx) and ok
    print("shape errors raise")
    var labels: List[String] = ["batch mismatch", "reshape size", "concat off-axis", "broadcast", "index out of range"]
    for i in range(5):
        var r = _raises(i)
        print("    ", labels[i], ": ", "OK" if r else "FAIL", sep="")
        ok = r and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional N-d gates failed")
