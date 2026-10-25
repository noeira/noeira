"""The functional tape on noeira's own Metal launcher (`Tape.metal_rt`) — gates.

(a) BIT-IDENTITY. The same training run (forward, MSE, backward, `TapeAdam`)
    on a MAX tape with the tape's own kernels (`Tape(ctx, own_kernels=True)`:
    MAX's launch path, the tape's GEMM, the same Adam kernel) and on a
    metal_rt tape under three commit policies: the forward output and every
    parameter gradient of step 0, the loss at every read, and every parameter
    after the last step must agree BIT FOR BIT. Same kernels in the same
    order: any difference is a launcher bug (a missed ordering, a wrong
    argument binding, a lost upload).
(b) TOLERANCE vs the tape's normal MAX path (MAX's matmul, nn's Adam ADOPTED,
    as `bench_functional_gpu` runs it): only the GEMMs' summation order
    differs. Bounds, relative to the largest reference magnitude: forward and
    step-0 gradients 1e-5, losses 1e-4, parameters after the run 1e-3 (Adam
    turns a last-bit gradient difference near zero into up to a whole `lr`
    step, so parameters are compared in units of the run's total `lr`).
(c) REFUSALS: a metal_rt tape raises on a module op and on nn's optimizer.

Three networks: the PPO and SAC MLPs of `bench_functional_gpu` (fused
`F.linear`), and an unfused graph (matmul, broadcast add, tanh, softmax,
sigmoid, log, transpose, sum_rows, mean, pow) whose 256 x 256 products take
the tiled GEMM.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_metal_rt.mojo
     … --stress N   the SAC every-16 comparison N times (race hunting)

Negative control: the same runs with no ordering between command buffers
(`ORDER_NONE`) must differ from MAX (reported, not gated: it is a race).
"""

from std.memory import bitcast
from std.random import seed, random_float64
from std.sys import exit, argv
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.linear import Linear
from noeira.nn.core.initializer import Deterministic
from noeira.experimental.functional import Tape, FTensor, TapeAdam
from std.memory import ArcPointer, OwnedPointer
from noeira.experimental.functional.tape import TapeData
from noeira.experimental.metal_rt.runtime import (
    MetalRT,
    ORDER_NONE,
    COMMIT_EACH,
    COMMIT_EVERY_N,
    COMMIT_ADAPTIVE,
)
import noeira.experimental.functional.F as F

comptime STEPS = 8
comptime LR: Scalar[DT] = 1e-3


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


@fieldwise_init
struct Net(Copyable, Movable):
    var kind: Int  # 0 fused MLP, 1 unfused op graph
    var B: Int
    var IN: Int
    var H: Int
    var OUT: Int
    var act: StaticString
    var name: String


struct Data(Movable):
    var p: List[List[Scalar[DT]]]  # w1 b1 w2 b2 w3 b3
    var xs: List[List[Scalar[DT]]]
    var ys: List[List[Scalar[DT]]]

    def __init__(out self, n: Net):
        seed(7)
        self.p = List[List[Scalar[DT]]]()
        var dims = [n.IN, n.H, n.H, n.OUT]
        for l in range(3):
            self.p.append(_rand(dims[l] * dims[l + 1], 1.0 / Float64(dims[l]) ** 0.5))
            self.p.append(_rand(dims[l + 1], 0.1))
        self.xs = List[List[Scalar[DT]]]()
        self.ys = List[List[Scalar[DT]]]()
        for _ in range(STEPS):
            self.xs.append(_rand(n.B * n.IN, 1.0))
            self.ys.append(_rand(n.B * n.OUT, 1.0))


struct Run(Movable):
    var fwd: List[Scalar[DT]]
    var grads: List[Scalar[DT]]
    var losses: List[Scalar[DT]]
    var params: List[Scalar[DT]]
    var commits: Int

    def __init__(out self):
        self.fwd = List[Scalar[DT]]()
        self.grads = List[Scalar[DT]]()
        self.losses = List[Scalar[DT]]()
        self.params = List[Scalar[DT]]()
        self.commits = 0


def _forward(n: Net, x: FTensor, ps: List[FTensor]) raises -> FTensor:
    if n.kind == 0:
        var h = F.linear(F.linear(x, ps[0], ps[1], n.act), ps[2], ps[3], n.act)
        return F.linear(h, ps[4], ps[5])
    var h = (x @ ps[0] + ps[1]).tanh()
    var g = h @ ps[2] + ps[3]
    var s = g.softmax() * 2.0 + (g.sigmoid() + 1.0).log()
    var pred = s @ ps[4] + ps[5]
    # a second use of h through a [H, B] @ [B, H] product
    var reg = (h.T() @ h).sum_rows().mean() * 0.001
    return pred + reg


def run(
    tape: Tape, n: Net, d: Data, read_every: Int, nn_adam: Bool
) raises -> Run:
    var dims = [n.IN, n.H, n.H, n.OUT]
    var ps = List[FTensor]()
    for l in range(3):
        ps.append(tape.param(dims[l], dims[l + 1], d.p[2 * l].copy(), name=String("w", l)))
        ps.append(tape.param(1, dims[l + 1], d.p[2 * l + 1].copy(), name=String("b", l)))
    var opt = TapeAdam(lr=LR)
    var nopt = Adam(lr=LR)
    # nn's Adam must launch on the TAPE's context: a second DeviceContext is
    # another queue, unordered with the tape's (a race this gate once hit).
    var ctx = tape.data[].ctx.value()
    if nn_adam:
        var t = tape
        nopt.adopt["gpu"](t, ctx)
    var r = Run()
    for step in range(STEPS):
        tape.reset()
        tape.zero_grad()
        var x = tape.input(n.B, n.IN, d.xs[step])
        var y = tape.input(n.B, n.OUT, d.ys[step])
        var pred = _forward(n, x, ps)
        var loss = ((pred - y) ** 2).mean()
        loss.backward()
        if step == 0:
            r.fwd = pred.values()
            for k in range(len(ps)):
                r.grads.extend(ps[k].grads())
        if step % read_every == read_every - 1:
            r.losses.append(loss.item())
        if nn_adam:
            var t = tape
            nopt.step["gpu"](t, ctx)
        else:
            opt.step(tape)
    for k in range(len(ps)):
        r.params.extend(ps[k].values())
    r.commits = tape.launcher_stats()[1]
    return r^


def first_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Int:
    if len(a) != len(b):
        return 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return i
    return -1


def rel_err(got: List[Scalar[DT]], ref_: List[Scalar[DT]]) -> Float64:
    """max |got - ref| / max |ref|."""
    var md: Float64 = 0
    var mr: Float64 = 1e-30
    for i in range(len(ref_)):
        md = max(md, abs(Float64(got[i]) - Float64(ref_[i])))
        mr = max(mr, abs(Float64(ref_[i])))
    return md / mr


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var nets = List[Net]()
    nets.append(Net(0, 64, 8, 64, 4, "tanh", "PPO 8-64-64-4 tanh, B=64"))
    nets.append(Net(0, 256, 23, 256, 1, "relu", "SAC 23-256-256-1 relu, B=256"))
    nets.append(Net(1, 256, 23, 256, 4, "", "unfused graph 23-256-256-4, B=256"))
    var pols = [COMMIT_EACH, COMMIT_EVERY_N, COMMIT_ADAPTIVE]
    var pol_names = ["each", "every-16", "adaptive-16 age 20us"]
    var fails = 0
    var gates = 0
    var stress = 0
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--stress" and i + 1 < len(args):
            stress = Int(args[i + 1])
    if stress > 0:
        # `--stress N`: the SAC every-16 comparison N times (race hunting)
        ref n = nets[1]
        var d = Data(n)
        var ref_ = run(Tape(ctx, own_kernels=True), n, d, 1, False)
        var bad = 0
        for k in range(stress):
            var got = run(Tape.metal_rt(policy=COMMIT_EVERY_N, batch_n=16), n, d, 1, False)
            var dl = first_diff(got.losses, ref_.losses)
            var dp = first_diff(got.params, ref_.params)
            if dl >= 0 or dp >= 0:
                bad += 1
                print("  run", k, "differs: loss", dl, "param", dp)
        print("STRESS:", bad, "of", stress, "runs differ")
        if bad != 0:
            exit(1)
        return

    print("(a) metal_rt tape vs MAX tape with the tape's own kernels: bit for bit")
    for ni in range(len(nets)):
        ref n = nets[ni]
        var d = Data(n)
        for re in [1, 3]:
            var ref_ = run(Tape(ctx, own_kernels=True), n, d, re, False)
            for p in range(len(pols)):
                var tape = Tape.metal_rt(
                    policy=pols[p], batch_n=16, idle_age_us=20 if pols[p] == COMMIT_ADAPTIVE else 0
                )
                var got = run(tape, n, d, re, False)
                var df = first_diff(got.fwd, ref_.fwd)
                var dg = first_diff(got.grads, ref_.grads)
                var dl = first_diff(got.losses, ref_.losses)
                var dp = first_diff(got.params, ref_.params)
                var ok = df < 0 and dg < 0 and dl < 0 and dp < 0
                gates += 1
                if not ok:
                    fails += 1
                print("  ", n.name, "| loss read every", re, "|", pol_names[p],
                      "| cbs", got.commits, "| fwd", len(got.fwd), "grads", len(got.grads),
                      "losses", len(got.losses), "params", len(got.params),
                      "OK" if ok else String("FAIL (first diff fwd ", df, " grad ", dg, " loss ", dl, " param ", dp, ")"))

    print("(b) metal_rt tape vs the tape's normal MAX path (MAX matmul, nn's Adam adopted)")
    for ni in range(len(nets)):
        ref n = nets[ni]
        var d = Data(n)
        var ref_ = run(Tape(ctx), n, d, 1, True)
        var got = run(Tape.metal_rt(), n, d, 1, False)
        var ef = rel_err(got.fwd, ref_.fwd)
        var eg = rel_err(got.grads, ref_.grads)
        var el = rel_err(got.losses, ref_.losses)
        var ep: Float64 = 0
        for i in range(len(ref_.params)):
            ep = max(ep, abs(Float64(got.params[i]) - Float64(ref_.params[i])))
        ep = ep / (Float64(LR) * Float64(STEPS))
        var ok = ef <= 1e-5 and eg <= 1e-5 and el <= 1e-4 and ep <= 1e-3
        gates += 1
        if not ok:
            fails += 1
        print("  ", n.name, "| fwd", ef, "| grads", eg, "| losses", el,
              "| params (in run-lr units)", ep, "| identical params",
              first_diff(got.params, ref_.params) < 0, "OK" if ok else "FAIL")

    print("negative control: metal_rt with NO ordering between command buffers (ORDER_NONE)")
    var seen = False
    for ni in range(len(nets)):
        ref n = nets[ni]
        var d = Data(n)
        var ref_ = run(Tape(ctx, own_kernels=True), n, d, 1, False)
        var bad = Tape()
        bad.data = ArcPointer(TapeData(
            DeviceContext(),
            Optional(OwnedPointer(MetalRT(policy=COMMIT_EACH, order=ORDER_NONE))),
            True,
        ))
        var got = run(bad, n, d, 1, False)
        var differs = first_diff(got.params, ref_.params) >= 0 or first_diff(got.losses, ref_.losses) >= 0
        seen = seen or differs
        print("  ", n.name, "| differs from MAX:", differs)
    print("   the gate", "SEES the missing ordering" if seen else "did NOT see it this run (not a failure: a race)")

    print("(c) refusals on a metal_rt tape")
    var t = Tape.metal_rt()
    _ = t.param(4, 4, _rand(16, 1.0), name="w")
    var refused_adam = False
    try:
        var nopt = Adam(lr=LR)
        var tt = t
        nopt.step["gpu"](tt, ctx)
    except e:
        refused_adam = "metal_rt" in String(e)
    var refused_mod = False
    try:
        _ = t.module[4](Linear[4, 4].make["gpu", Deterministic](ctx), name="lin")
    except e:
        refused_mod = "metal_rt" in String(e)
    gates += 2
    if not refused_adam:
        fails += 1
    if not refused_mod:
        fails += 1
    print("   nn Adam refused:", refused_adam, "| module op refused:", refused_mod)

    print("GATES:", gates - fails, "/", gates)
    if fails != 0:
        print("FAILED")
        exit(1)
    print("ALL OK")
