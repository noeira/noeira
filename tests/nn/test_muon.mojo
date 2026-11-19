"""Muon (`nn/optimizer/muon.mojo`) against torch.

    uv run --no-project --python 3.12 --with torch --with numpy \\
        python tools/nn/muon_oracle.py --out /tmp/muon_oracle
    pixi run mojo run -I . tests/nn/test_muon.mojo /tmp/muon_oracle
    # the same gate with the model on the device (NVIDIA: the cuBLAS path; Metal: the host fallback)
    pixi run -e nvidia mojo build -I . -D MUON_GPU=1 tests/nn/test_muon.mojo -o t && ./t /tmp/muon_oracle

The reference is torch's own update code run in float64
(`tools/nn/muon_oracle.py`); `torch.optim.Muon` itself runs its Newton–Schulz
products in bfloat16 and sits 1.5 to 4 % of a step away from that reference.

WHAT EACH SECTION CAN CATCH
===========================
[1] three steps on a [12, 20] matrix (Gram matrix on the IN side), weight
    decay 0: every step within 1e-4 of the step's largest move of the float64
    reference, and within 6 % of torch's bfloat16 result.
[2] the same on a [24, 10] matrix (Gram matrix on the OUT side) with weight
    decay 0.1.
[3] the check has power: without Nesterov the SECOND step is far from the
    reference (at the first step the two updates are both proportional to
    the gradient, so they coincide after normalisation); without its weight
    decay the [24, 10] matrix is far from its reference.
[4] sharing a model with Adam: Muon zeroes the registered matrix's gradient
    and touches nothing else; an Adam step over the whole model after it
    leaves that matrix to the bit, at the first and at a later step, while
    the other params move.
[5] a name that is not in the model raises; a wrong shape raises.
"""

from std.math import abs
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext

from noeira.io.npy import load_npy_2d_f64
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.muon import Muon
from noeira.nn.primitives.linear import Linear


comptime LR = 1e-2
comptime GPU = get_defined_int["MUON_GPU", 0]() != 0
comptime TARGET: StaticString = "gpu" if GPU else "cpu"
comptime ModelA = Sequential[Linear[12, 20], Linear[20, 4]]
comptime ModelB = Sequential[Linear[24, 10], Linear[10, 4]]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


struct _Put(ParamVisitor):
    """Write `vals` into the param `target_name` (its value, or its gradient);
    with `fill_others`, every other param's gradient becomes 1."""
    var target_name: String
    var vals: List[Float64]
    var into_grad: Bool
    var fill_others: Bool
    var hit: Int

    def __init__(out self, target_name: String, vals: List[Float64], into_grad: Bool, fill_others: Bool = False):
        self.target_name = target_name.copy()
        self.vals = vals.copy()
        self.into_grad = into_grad
        self.fill_others = fill_others
        self.hit = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        grad.ensure(N)
        if name == self.target_name:
            if len(self.vals) != N:
                raise Error("_Put: " + name + " holds " + String(N) + ", given " + String(len(self.vals)))
            for i in range(N):
                if self.into_grad:
                    grad.data[i] = Scalar[DT](self.vals[i])
                else:
                    param.data[i] = Scalar[DT](self.vals[i])
            if not self.into_grad:
                param.version += 1
            self.hit += 1
        elif self.fill_others and self.into_grad:
            for i in range(N):
                grad.data[i] = 1
        comptime if target == "gpu":
            grad.upload_resident(ctx.value())
            if name == self.target_name and not self.into_grad:
                param.upload_resident(ctx.value())


struct _Get(ParamVisitor):
    """Every param's values (or gradients), by walk order, with its name."""
    var grads: Bool
    var names: List[String]
    var out: List[List[Float64]]

    def __init__(out self, grads: Bool):
        self.grads = grads
        self.names = List[String]()
        self.out = List[List[Float64]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            if self.grads:
                grad.download(ctx.value())
            else:
                param.download(ctx.value())
        var l = List[Float64](length=N, fill=0.0)
        for i in range(N):
            l[i] = Float64(grad.data[i]) if self.grads else Float64(param.data[i])
        self.names.append(name.copy())
        self.out.append(l^)


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, abs(a[i] - b[i]))
    return m


def _three_steps[M: Module, IN: Int, OUT: Int](
    tag: String, wd: Float64, nesterov: Bool, dump: String, ctx: Optional[DeviceContext],
) raises -> List[List[Float64]]:
    """The model's first weight after each of the oracle's three steps."""
    var model = M.make[TARGET, Kaiming](ctx)
    var w0 = load_npy_2d_f64(dump + "/" + tag + "_w0.npy", OUT)
    var put = _Put(String("0.weight"), w0.values, False)
    model.for_each_param[TARGET](put, ctx)
    var muon = Muon(Scalar[DT](LR), 0.95, nesterov, wd)
    muon.add(String("0.weight"), IN, OUT)
    var res = List[List[Float64]]()
    for k in range(1, 4):
        var g = load_npy_2d_f64(dump + "/" + tag + "_g" + String(k) + ".npy", OUT)
        var pg = _Put(String("0.weight"), g.values, True)
        model.for_each_param[TARGET](pg, ctx)
        muon.step[TARGET](model, ctx)
        var get = _Get(False)
        model.for_each_param[TARGET](get, ctx)
        res.append(get.out[0].copy())
    return res^


def _against_oracle[M: Module, IN: Int, OUT: Int](tag: String, wd: Float64, dump: String, mut t: Tally, ctx: Optional[DeviceContext]) raises:
    var ours = _three_steps[M, IN, OUT](tag, wd, True, dump, ctx)
    var prev = load_npy_2d_f64(dump + "/" + tag + "_w0.npy", OUT).values.copy()
    for k in range(1, 4):
        var f64 = load_npy_2d_f64(dump + "/" + tag + "_w" + String(k) + "_f64.npy", OUT).values.copy()
        var tor = load_npy_2d_f64(dump + "/" + tag + "_w" + String(k) + "_torch.npy", OUT).values.copy()
        var move = _maxdiff(f64, prev)
        var e64 = _maxdiff(ours[k - 1], f64)
        var etor = _maxdiff(ours[k - 1], tor)
        t.truth(
            e64 <= 1e-4 * move,
            "step " + String(k) + ": " + String(e64 / move) + " of the largest move (" + String(move) + ") from torch's algorithm in float64",
        )
        t.truth(etor <= 0.06 * move, "step " + String(k) + ": " + String(etor / move) + " of it from torch.optim.Muon (bfloat16)")
        prev = f64.copy()


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/muon_oracle")
    var t = Tally()
    var ctx: Optional[DeviceContext] = None
    comptime if GPU:
        ctx = DeviceContext()
    print("target:", TARGET)

    print("\n[1] [12, 20], weight decay 0")
    _against_oracle[ModelA, 12, 20](String("a"), 0.0, dump, t, ctx)

    print("\n[2] [24, 10], weight decay 0.1")
    _against_oracle[ModelB, 24, 10](String("b"), 0.1, dump, t, ctx)

    print("\n[3] the check has power")
    var plain = _three_steps[ModelA, 12, 20](String("a"), 0.0, False, dump, ctx)
    var w0a = load_npy_2d_f64(dump + "/a_w0.npy", 20).values.copy()
    var f1 = load_npy_2d_f64(dump + "/a_w1_f64.npy", 20).values.copy()
    var f2 = load_npy_2d_f64(dump + "/a_w2_f64.npy", 20).values.copy()
    var move1 = _maxdiff(f1, w0a)
    var move2 = _maxdiff(f2, f1)
    t.truth(_maxdiff(plain[0], f1) <= 1e-4 * move1, "without Nesterov the first step is the same (both updates are proportional to the gradient)")
    t.truth(_maxdiff(plain[1], f2) > 0.01 * move2, "and the second is " + String(_maxdiff(plain[1], f2) / move2) + " of a move away")
    var nodecay = _three_steps[ModelB, 24, 10](String("b"), 0.0, True, dump, ctx)
    var fb3 = load_npy_2d_f64(dump + "/b_w3_f64.npy", 10).values.copy()
    var fb2 = load_npy_2d_f64(dump + "/b_w2_f64.npy", 10).values.copy()
    t.truth(_maxdiff(nodecay[2], fb3) > 0.01 * _maxdiff(fb3, fb2), "without its weight decay the [24, 10] result is " + String(_maxdiff(nodecay[2], fb3) / _maxdiff(fb3, fb2)) + " of a move away")

    print("\n[4] sharing a model with Adam")
    var model = ModelA.make[TARGET, Kaiming](ctx)
    var muon = Muon(Scalar[DT](LR))
    muon.add(String("0.weight"), 12, 20)
    var adam = Adam(lr=Scalar[DT](LR))
    var g1 = load_npy_2d_f64(dump + "/a_g1.npy", 20)
    var shared_ok = True
    var others_moved = True
    for round in range(2):
        var pg = _Put(String("0.weight"), g1.values, True, True)
        model.for_each_param[TARGET](pg, ctx)
        var before = _Get(False)
        model.for_each_param[TARGET](before, ctx)
        muon.step[TARGET](model, ctx)
        var mid = _Get(False)
        model.for_each_param[TARGET](mid, ctx)
        var gmid = _Get(True)
        model.for_each_param[TARGET](gmid, ctx)
        if round == 0:
            var moved = _maxdiff(mid.out[0], before.out[0]) > 0.0
            var gzero = True
            for i in range(len(gmid.out[0])):
                gzero = gzero and gmid.out[0][i] == 0.0
            var rest_same = True
            var rest_grads = True
            for p in range(1, len(mid.out)):
                rest_same = rest_same and _maxdiff(mid.out[p], before.out[p]) == 0.0
                for i in range(len(gmid.out[p])):
                    rest_grads = rest_grads and gmid.out[p][i] == 1.0
            t.truth(moved and gzero, "Muon moves the registered matrix (" + mid.names[0] + ") and zeroes its gradient")
            t.truth(rest_same and rest_grads and len(mid.out) == 4, "and leaves the other " + String(len(mid.out) - 1) + " params and their gradients alone")
        adam.step[TARGET](model, ctx)
        var after = _Get(False)
        model.for_each_param[TARGET](after, ctx)
        shared_ok = shared_ok and _maxdiff(after.out[0], mid.out[0]) == 0.0
        for p in range(1, len(after.out)):
            others_moved = others_moved and _maxdiff(after.out[p], mid.out[p]) > 0.0
    t.truth(shared_ok, "an Adam step over the whole model leaves that matrix to the bit, at both steps")
    t.truth(others_moved, "while every other param moves under Adam")

    print("\n[5] wrong registrations")
    var bad = Muon(Scalar[DT](LR))
    bad.add(String("7.weight"), 12, 20)
    var refused = False
    try:
        bad.step[TARGET](model, ctx)
    except e:
        refused = True
        print("   ", e)
    t.truth(refused, "a name that is not in the model raises")
    var bad2 = Muon(Scalar[DT](LR))
    bad2.add(String("0.weight"), 12, 21)
    var refused2 = False
    try:
        bad2.step[TARGET](model, ctx)
    except e:
        refused2 = True
        print("   ", e)
    t.truth(refused2, "a shape that is not the param's raises")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_muon: " + String(t.fails) + " failed")
