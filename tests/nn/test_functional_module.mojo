"""Comptime nn modules on the functional tape (`tape.module[B](m)`) — gates,
on the CPU and on the GPU.

1. PARITY: `Sequential[Linear, LayerNorm, Tanh, Linear]` + MSE against the
   same modules called eagerly — `l2(ln(l1(x)).tanh())`. Forward, loss,
   every parameter gradient (LayerNorm's gamma/beta included) and dx.
2. ACCUMULATION: a residual block `h + l3(h)`, whose module input also
   feeds the skip, so the module's input gradient must be ADDED to the
   skip's — against the same block in native ops (`h @ w3 + b3`).
3. OPTIMIZER: `Adam.step(tape)` (the tape is `ParamWalkable`) against
   `Adam.step(sequential)` after (1); and against native parameters after
   (2); then 60 Adam steps of training through the tape.
4. MISUSE raises: a module called twice before `reset`, a wrong batch.

Run: pixi run -e apple mojo run -I . tests/nn/test_functional_module.mojo
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.activations import Tanh
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse import mse_forward, mse_backward
from noeira.nn.optimizer.adam import Adam
from noeira.nn.functional import Tape, FTensor, FModule
import noeira.nn.functional.F as F


comptime B = 96
comptime D = 7
comptime H = 40
comptime O = 3
comptime TOL: Float64 = 1e-5


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
    var first_bad: String

    def __init__(out self):
        self.n = 0
        self.differ = 0
        self.worst = 0
        self.max_ref = 0
        self.first_bad = String("")

    def add(mut self, got: Scalar[DT], want: Scalar[DT]):
        self.n += 1
        if got != want:
            self.differ += 1
        var w = Float64(want)
        var e = abs(Float64(got) - w) / max(1.0, abs(w))
        if e > TOL and self.first_bad.byte_length() < 300:
            self.first_bad += String(" [#", self.n - 1, " got ", got, " want ", want, "]")
        if e > self.worst:
            self.worst = e
        if abs(w) > self.max_ref:
            self.max_ref = abs(w)

    def add_n(mut self, got: List[Scalar[DT]], want: List[Scalar[DT]], n: Int):
        for k in range(n):
            self.add(got[k], want[k])

    def report(self, name: String, tol: Float64 = TOL) -> Bool:
        var ok = self.n > 0 and self.max_ref > 0 and self.worst <= tol
        print("    ", name, ": ", self.n, " compared, ", self.differ,
              " not bit-identical, worst rel ", self.worst, "  ",
              "OK" if ok else "FAIL", sep="")
        if not ok and self.first_bad.byte_length() > 0:
            print("      first mismatches:", self.first_bad)
        return ok


def _host[target: StaticString](mut t: Tensor, ctx: Optional[DeviceContext]) raises -> List[Scalar[DT]]:
    comptime if target == "gpu":
        t.download(ctx.value())
    return t.data.copy()


def _set[target: StaticString](
    mut t: Tensor, vals: List[Scalar[DT]], ctx: Optional[DeviceContext]
) raises:
    for k in range(len(vals)):
        t.data[k] = vals[k]
    comptime if target == "gpu":
        t.upload(ctx.value())


def _perturb_linear[target: StaticString, IN: Int, OUT: Int](
    mut l: Linear[IN, OUT], salt: Int, ctx: Optional[DeviceContext]
) raises:
    seed(salt)
    _set[target](l.weight.val, _rand(IN * OUT, 0.4), ctx)
    _set[target](l.bias.val, _rand(OUT, 0.2), ctx)


def _perturb_ln[target: StaticString, N: Int](
    mut ln: LayerNorm[N], salt: Int, ctx: Optional[DeviceContext]
) raises:
    seed(salt)
    var g = _rand(N, 0.3)
    for k in range(N):
        g[k] += 1.0
    _set[target](ln.gamma.val, g, ctx)
    _set[target](ln.beta.val, _rand(N, 0.2), ctx)


def _tape(ctx: Optional[DeviceContext]) -> Tape:
    if ctx:
        return Tape(ctx.value())
    return Tape()


# ── 1 + 3a. parity with Sequential, then one Adam step on each ──────────
def _parity[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime NET = Sequential[Linear[D, H], LayerNorm[H], Tanh[H], Linear[H, O]]
    seed(11)
    var xs = _rand(B * D, 1.0)
    var ys = _rand(B * O, 1.0)

    var net = NET.make[target, Deterministic](ctx)
    _perturb_linear[target](net.children[0], 1, ctx)
    _perturb_ln[target](net.children[1], 2, ctx)
    _perturb_linear[target](net.children[3], 3, ctx)
    net.zero_grad[target](ctx)
    var x = Tensor.alloc(B * D)
    var y = Tensor.alloc(B * O)
    for k in range(B * D):
        x.data[k] = xs[k]
    for k in range(B * O):
        y.data[k] = ys[k]
    var out: Tensor
    var go: Tensor
    var gx: Tensor
    comptime if target == "gpu":
        x.upload(ctx.value())
        y.upload(ctx.value())
        out = Tensor.alloc_gpu(ctx.value(), B * O)
        go = Tensor.alloc_gpu(ctx.value(), B * O)
        gx = Tensor.alloc_gpu(ctx.value(), B * D)
    else:
        out = Tensor.alloc(B * O)
        go = Tensor.alloc(B * O)
        gx = Tensor.alloc(B * D)
    net.forward[target, B](TensorRefs[1](x), out, ctx)
    mse_backward[target, B, O](out, y, go, ctx)
    net.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gx), ctx)
    var ref_out = _host[target](out, ctx)
    var ref_loss = mse_forward[B, O](out, y)
    var ref_gx = _host[target](gx, ctx)

    var tape = _tape(ctx)
    var l1 = tape.module[B](Linear[D, H].make[target, Deterministic](ctx))
    var ln = tape.module[B](LayerNorm[H].make[target, Deterministic](ctx))
    var l2 = tape.module[B](Linear[H, O].make[target, Deterministic](ctx))
    _perturb_linear[target](l1.get(), 1, ctx)
    _perturb_ln[target](ln.get(), 2, ctx)
    _perturb_linear[target](l2.get(), 3, ctx)
    tape.zero_grad()
    var fx = tape.input(B, D, xs, requires_grad=True)
    var pred = l2(ln(l1(fx)).tanh())
    var loss = F.mse(pred, tape.input(B, O, ys))
    loss.backward()

    var ok = True
    var c = Cmp()
    c.add_n(pred.values(), ref_out, B * O)
    ok = c.report("forward") and ok
    c = Cmp()
    c.add(loss.item(), ref_loss)
    ok = c.report("loss") and ok
    c = Cmp()
    c.add_n(_host[target](l1.get().weight.grd, ctx), _host[target](net.children[0].weight.grd, ctx), D * H)
    ok = c.report("dW1") and ok
    c = Cmp()
    c.add_n(_host[target](l1.get().bias.grd, ctx), _host[target](net.children[0].bias.grd, ctx), H)
    ok = c.report("db1") and ok
    c = Cmp()
    c.add_n(_host[target](ln.get().gamma.grd, ctx), _host[target](net.children[1].gamma.grd, ctx), H)
    ok = c.report("dgamma") and ok
    c = Cmp()
    c.add_n(_host[target](ln.get().beta.grd, ctx), _host[target](net.children[1].beta.grd, ctx), H)
    ok = c.report("dbeta") and ok
    c = Cmp()
    c.add_n(_host[target](l2.get().weight.grd, ctx), _host[target](net.children[3].weight.grd, ctx), H * O)
    ok = c.report("dW2") and ok
    c = Cmp()
    c.add_n(fx.grads(), ref_gx, B * D)
    ok = c.report("dx") and ok

    # One Adam step on each: the tape walks its modules' parameters.
    var opt_ref = Adam(lr=1e-2)
    var opt = Adam(lr=1e-2)
    opt_ref.step[target](net, ctx)
    opt.step[target](tape, ctx)
    c = Cmp()
    c.add_n(_host[target](l1.get().weight.val, ctx), _host[target](net.children[0].weight.val, ctx), D * H)
    c.add_n(_host[target](ln.get().gamma.val, ctx), _host[target](net.children[1].gamma.val, ctx), H)
    c.add_n(_host[target](l2.get().bias.val, ctx), _host[target](net.children[3].bias.val, ctx), O)
    ok = c.report("after Adam.step(tape) vs Adam.step(net)") and ok
    return ok


# ── 2 + 3b. accumulation through a module input; Adam on native params ──
def _residual[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    seed(21)
    var xs = _rand(B * D, 1.0)
    var ys = _rand(B * O, 1.0)
    seed(5)
    var w3v = _rand(H * H, 0.3)
    var b3v = _rand(H, 0.1)

    # A: the skip block as a module.
    var ta = _tape(ctx)
    var a1 = ta.module[B](Linear[D, H].make[target, Deterministic](ctx))
    var a3 = ta.module[B](Linear[H, H].make[target, Deterministic](ctx))
    var a2 = ta.module[B](Linear[H, O].make[target, Deterministic](ctx))
    _perturb_linear[target](a1.get(), 1, ctx)
    _set[target](a3.get().weight.val, w3v, ctx)
    _set[target](a3.get().bias.val, b3v, ctx)
    _perturb_linear[target](a2.get(), 3, ctx)
    ta.zero_grad()
    var ha = a1(ta.input(B, D, xs))
    var pa = a2((ha + a3(ha)).tanh())
    var la = F.mse(pa, ta.input(B, O, ys))
    la.backward()

    # B: the same block in native ops.
    var tb = _tape(ctx)
    var w3 = tb.param(H, H, w3v)
    var b3 = tb.param(1, H, b3v)
    var b1m = tb.module[B](Linear[D, H].make[target, Deterministic](ctx))
    var b2m = tb.module[B](Linear[H, O].make[target, Deterministic](ctx))
    _perturb_linear[target](b1m.get(), 1, ctx)
    _perturb_linear[target](b2m.get(), 3, ctx)
    tb.zero_grad()
    var hb = b1m(tb.input(B, D, xs))
    var pb = b2m((hb + (hb @ w3 + b3)).tanh())
    var lb = F.mse(pb, tb.input(B, O, ys))
    lb.backward()

    var ok = True
    var c = Cmp()
    c.add_n(pa.values(), pb.values(), B * O)
    ok = c.report("forward") and ok
    c = Cmp()
    c.add_n(_host[target](a1.get().weight.grd, ctx), _host[target](b1m.get().weight.grd, ctx), D * H)
    c.add_n(_host[target](a1.get().bias.grd, ctx), _host[target](b1m.get().bias.grd, ctx), H)
    ok = c.report("dW1, db1 (skip + module input, accumulated)") and ok
    c = Cmp()
    c.add_n(_host[target](a3.get().weight.grd, ctx), w3.grads(), H * H)
    c.add_n(_host[target](a3.get().bias.grd, ctx), b3.grads(), H)
    ok = c.report("module dW3, db3 vs native w3, b3 grads") and ok

    var oa = Adam(lr=1e-2)
    var ob = Adam(lr=1e-2)
    oa.step[target](ta, ctx)
    ob.step[target](tb, ctx)
    c = Cmp()
    c.add_n(_host[target](a3.get().weight.val, ctx), w3.values(), H * H)
    c.add_n(_host[target](a3.get().bias.val, ctx), b3.values(), H)
    ok = c.report("after Adam.step: module W3, b3 vs native w3, b3") and ok
    var moved = False
    var w3_now = w3.values()
    for k in range(H * H):
        if w3_now[k] != w3v[k]:
            moved = True
    print("    native params moved by Adam.step(tape):", moved)
    return ok and moved


# ── 3c. training, 4. misuse ──────────────────────────────────────────────
def _train_and_misuse[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    seed(31)
    var xs = _rand(B * D, 1.0)
    var ys = List[Scalar[DT]]()
    for i in range(B):
        for j in range(O):
            ys.append(Scalar[DT](0.5) * xs[i * D + j] - Scalar[DT](0.3) * xs[i * D + j + 1])
    var tape = _tape(ctx)
    var scale = tape.param(1, O, List[Scalar[DT]](length=O, fill=1.0))
    var l1 = tape.module[B](Linear[D, H].make[target, Deterministic](ctx))
    var ln = tape.module[B](LayerNorm[H].make[target, Deterministic](ctx))
    var l2 = tape.module[B](Linear[H, O].make[target, Deterministic](ctx))
    _perturb_linear[target](l1.get(), 1, ctx)
    _perturb_linear[target](l2.get(), 3, ctx)
    var opt = Adam(lr=1e-2)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(61):
        tape.reset()
        tape.zero_grad()
        var x = tape.input(B, D, xs)
        var loss = F.mse(l2(ln(l1(x)).tanh()) * scale, tape.input(B, O, ys))
        loss.backward()
        opt.step[target](tape, ctx)
        if step == 0:
            first = loss.item()
        last = loss.item()
    var trained = last < first * 0.5
    print("    60 Adam steps (modules + a native scale): loss", first, "->",
          last, " ", "OK" if trained else "FAIL")

    tape.reset()
    var x = tape.input(B, D, xs)
    _ = l1(x)
    var twice = False
    try:
        _ = l1(x)
    except:
        twice = True
    var wrong_batch = False
    try:
        _ = l1(tape.input(B - 1, D, xs))
    except:
        wrong_batch = True
    print("    second call before reset raises:", twice,
          "  wrong batch raises:", wrong_batch)
    return trained and twice and wrong_batch


def _all[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    print("  parity: Sequential[Linear, LayerNorm, Tanh, Linear] vs modules on a tape (B=", B, ")", sep="")
    ok = _parity[target](ctx) and ok
    print("  residual h + l3(h): module l3 vs native ops")
    ok = _residual[target](ctx) and ok
    print("  training and misuse")
    ok = _train_and_misuse[target](ctx) and ok
    return ok


def main() raises:
    var ok = True
    print("CPU")
    ok = _all["cpu"](None) and ok
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    ok = _all["gpu"](ctx) and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional module gates failed")
