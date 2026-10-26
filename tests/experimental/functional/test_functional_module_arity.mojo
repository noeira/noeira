"""Comptime modules with several inputs and several batch sizes on the
functional tape — gates.

1. `Modulate` (3 inputs: x, scale, shift) vs the same arithmetic in native
   ops, `x * (scale + 1) + shift`: value and the gradient of each input.
2. `GRUCell` (2 inputs, with parameters) through the tape vs the module
   called directly: output, both input gradients, every parameter
   gradient; and central differences through the tape for x and h.
3. `Concat[A, B]` vs `F.concat`; `Add(x, x)` — one node passed twice —
   vs `x + x`.
4. One `Linear` serving batches 8 and 3 (`tape.module[8, 3]`), each vs the
   module called directly at that batch; batch 5 raises; a wrong number of
   inputs raises.
5. GPU: every tape computation above on a GPU tape (after a warm-up step,
   so gradients slots are stale) vs the CPU tape.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_module_arity.mojo
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.modulate import Modulate
from noeira.nn.primitives.gru_cell import GRUCell
from noeira.nn.primitives.concat import Concat
from noeira.nn.primitives.add import Add
from noeira.experimental.functional import Tape, FTensor, FModule
import noeira.experimental.functional.F as F


comptime B = 6
comptime D = 5
comptime H = 4
comptime TOL: Float64 = 1e-5


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


def _tape(ctx: Optional[DeviceContext]) -> Tape:
    if ctx:
        return Tape(ctx.value())
    return Tape()


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

    def add_n(mut self, got: List[Scalar[DT]], want: List[Scalar[DT]]):
        for k in range(len(want)):
            self.n += 1
            if got[k] != want[k]:
                self.differ += 1
            var w = Float64(want[k])
            self.worst = max(self.worst, abs(Float64(got[k]) - w) / max(1.0, abs(w)))
            self.max_ref = max(self.max_ref, abs(w))

    def report(self, name: String, tol: Float64 = TOL) -> Bool:
        var ok = self.n > 0 and self.max_ref > 0 and self.worst <= tol
        print("    ", name, ": ", self.n, " compared, ", self.differ,
              " not bit-identical, worst rel ", self.worst, "  ",
              "OK" if ok else "FAIL", sep="")
        return ok


# ── the tape computations (CPU or GPU), each returning value + grads ─────
def _modulate(ctx: Optional[DeviceContext], native: Bool) raises -> List[List[Scalar[DT]]]:
    seed(1)
    var tape = _tape(ctx)
    var xv = _rand(B * D, 1.0)
    var sv = _rand(B * D, 0.5)
    var hv = _rand(B * D, 0.5)
    var cv = _rand(B * D, 1.0)
    var m = tape.module[B](_modulate_make(ctx), name="mod")
    for step in range(2):  # step 0 = warm-up (stale gradient slots on the GPU)
        tape.reset()
        var x = tape.input(B, D, xv, requires_grad=True)
        var s = tape.input(B, D, sv, requires_grad=True)
        var sh = tape.input(B, D, hv, requires_grad=True)
        var y = (x * (s + 1.0) + sh) if native else m(x, s, sh)
        var f = (y * tape.input(B, D, cv)).sum()
        f.backward()
        if step == 1:
            var out = List[List[Scalar[DT]]]()
            out.append(y.values())
            out.append(x.grads())
            out.append(s.grads())
            out.append(sh.grads())
            return out^
    raise Error("unreachable")


def _modulate_make(ctx: Optional[DeviceContext]) raises -> Modulate[D]:
    if ctx:
        return Modulate[D].make["gpu", Deterministic](ctx)
    return Modulate[D].make["cpu", Deterministic](ctx)


def _gru_make(ctx: Optional[DeviceContext]) raises -> GRUCell[D, H]:
    if ctx:
        return GRUCell[D, H].make["gpu", Deterministic](ctx)
    return GRUCell[D, H].make["cpu", Deterministic](ctx)


def _gru_tape(ctx: Optional[DeviceContext]) raises -> List[List[Scalar[DT]]]:
    seed(2)
    var xv = _rand(B * D, 1.0)
    var hv = _rand(B * H, 1.0)
    var cv = _rand(B * H, 1.0)
    var tape = _tape(ctx)
    var g = tape.module[B](_gru_make(ctx), name="gru")
    var out = List[List[Scalar[DT]]]()
    for step in range(2):
        tape.reset()
        tape.zero_grad()
        var x = tape.input(B, D, xv, requires_grad=True)
        var h = tape.input(B, H, hv, requires_grad=True)
        var y = g(x, h)
        (y * tape.input(B, H, cv)).sum().backward()
        if step == 1:
            out.append(y.values())
            out.append(x.grads())
            out.append(h.grads())
            var params = List[Scalar[DT]]()
            for np in _param_grads(g, ctx):
                params.append(np)
            out.append(params^)
    return out^


def _param_grads(mut g: FModule[GRUCell[D, H]], ctx: Optional[DeviceContext]) raises -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    ref m = g.get()
    _append_grads(m.W_ih.grd, out, ctx)
    _append_grads(m.W_hh.grd, out, ctx)
    _append_grads(m.b_ih.grd, out, ctx)
    _append_grads(m.b_hh.grd, out, ctx)
    return out^


def _append_grads(mut t: Tensor, mut out: List[Scalar[DT]], ctx: Optional[DeviceContext]) raises:
    if ctx:
        t.download(ctx.value())
    for k in range(t.n):
        out.append(t.data[k])


def _gru_direct() raises -> List[List[Scalar[DT]]]:
    """The same GRU step, the module called directly (CPU)."""
    seed(2)
    var xv = _rand(B * D, 1.0)
    var hv = _rand(B * H, 1.0)
    var cv = _rand(B * H, 1.0)
    var m = GRUCell[D, H].make["cpu", Deterministic]()
    m.zero_grad["cpu"](None)
    var x = Tensor.alloc(B * D)
    var h = Tensor.alloc(B * H)
    for k in range(B * D):
        x.data[k] = xv[k]
    for k in range(B * H):
        h.data[k] = hv[k]
    # A multi-input module takes ONE TensorRefs pack, whose entries share an
    # origin: nn's own idiom is a TensorPack.
    var ins = TensorPack[2]()
    ins.tensors[0] = x^
    ins.tensors[1] = h^
    var gin = TensorPack[2]()
    gin.tensors[0] = Tensor.alloc(B * D)
    gin.tensors[1] = Tensor.alloc(B * H)
    var y = Tensor.alloc(B * H)
    m.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), y)
    var go = Tensor.alloc(B * H)
    for k in range(B * H):
        go.data[k] = cv[k]
    m.vjp["cpu", B](TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](gin[0], gin[1]))
    var out = List[List[Scalar[DT]]]()
    out.append(y.data.copy())
    out.append(gin.tensors[0].data.copy())
    out.append(gin.tensors[1].data.copy())
    var params = List[Scalar[DT]]()
    _append_grads(m.W_ih.grd, params, None)
    _append_grads(m.W_hh.grd, params, None)
    _append_grads(m.b_ih.grd, params, None)
    _append_grads(m.b_hh.grd, params, None)
    out.append(params^)
    return out^


def _gru_fd() raises -> Float64:
    """Central differences through the tape for x and h (CPU)."""
    seed(2)
    var xv = _rand(B * D, 1.0)
    var hv = _rand(B * H, 1.0)
    var cv = _rand(B * H, 1.0)
    var tape = Tape()
    var g = tape.module[B](GRUCell[D, H].make["cpu", Deterministic]())
    var x = tape.param(B, D, xv, name="x")
    var h = tape.param(B, H, hv, name="h")
    def _f() raises {mut tape, g, x, h, cv} -> Float64:
        tape.reset()
        return Float64((g(x, h) * tape.input(B, H, cv)).sum().item())
    tape.zero_grad()
    tape.reset()
    (g(x, h) * tape.input(B, H, cv)).sum().backward()
    var worst: Float64 = 0
    comptime EPS: Float64 = 1e-2
    for which in range(2):
        var t = x if which == 0 else h
        var ga = t.grads()
        for k in range(t.size()):
            var keep = t.value(k)
            t.set_value(k, keep + Scalar[DT](EPS))
            var fp = _f()
            t.set_value(k, keep - Scalar[DT](EPS))
            var fm = _f()
            t.set_value(k, keep)
            var fd = (fp - fm) / (2.0 * EPS)
            worst = max(worst, abs(fd - Float64(ga[k])) / max(1.0, abs(fd)))
    return worst


def _concat_add(ctx: Optional[DeviceContext], native: Bool) raises -> List[List[Scalar[DT]]]:
    seed(3)
    var av = _rand(B * D, 1.0)
    var bv = _rand(B * H, 1.0)
    var cv = _rand(B * (D + H), 1.0)
    var dv = _rand(B * D, 1.0)
    var tape = _tape(ctx)
    var cat = tape.module[B](_concat_make(ctx), name="cat")
    var add = tape.module[B](_add_make(ctx), name="add")
    var out = List[List[Scalar[DT]]]()
    for step in range(2):
        tape.reset()
        var a = tape.input(B, D, av, requires_grad=True)
        var b = tape.input(B, H, bv, requires_grad=True)
        var y = F.concat(a, b, axis=-1) if native else cat(a, b)
        var z = (a + a) if native else add(a, a)
        ((y * tape.input(B, D + H, cv)).sum() + (z * tape.input(B, D, dv)).sum()).backward()
        if step == 1:
            out.append(y.values())
            out.append(z.values())
            out.append(a.grads())
            out.append(b.grads())
    return out^


def _concat_make(ctx: Optional[DeviceContext]) raises -> Concat[D, H]:
    if ctx:
        return Concat[D, H].make["gpu", Deterministic](ctx)
    return Concat[D, H].make["cpu", Deterministic](ctx)


def _add_make(ctx: Optional[DeviceContext]) raises -> Add[D]:
    if ctx:
        return Add[D].make["gpu", Deterministic](ctx)
    return Add[D].make["cpu", Deterministic](ctx)


def _lin_make(ctx: Optional[DeviceContext]) raises -> Linear[D, H]:
    if ctx:
        return Linear[D, H].make["gpu", Deterministic](ctx)
    return Linear[D, H].make["cpu", Deterministic](ctx)


def _batches(ctx: Optional[DeviceContext]) raises -> List[List[Scalar[DT]]]:
    """One Linear, batches 8 then 3: output, input grad, weight grad."""
    seed(4)
    var x8 = _rand(8 * D, 1.0)
    var x3 = _rand(3 * D, 1.0)
    var tape = _tape(ctx)
    var lin = tape.module[8, 3](_lin_make(ctx), name="lin")
    var out = List[List[Scalar[DT]]]()
    for step in range(2):
        var rows = 8 if step == 0 else 3
        tape.reset()
        tape.zero_grad()
        var x = tape.input(rows, D, x8 if step == 0 else x3, requires_grad=True)
        var y = lin(x)
        (y * y).sum().backward()
        out.append(y.values())
        out.append(x.grads())
        var gw = List[Scalar[DT]]()
        _append_grads(lin.get().weight.grd, gw, ctx)
        out.append(gw^)
    return out^


def _lin_direct[R: Int](xv: List[Scalar[DT]]) raises -> List[List[Scalar[DT]]]:
    var m = Linear[D, H].make["cpu", Deterministic]()
    m.zero_grad["cpu"](None)
    var x = Tensor.alloc(R * D)
    for k in range(R * D):
        x.data[k] = xv[k]
    var y = Tensor.alloc(R * H)
    m.forward["cpu", R](TensorRefs[1](x), y)
    var go = Tensor.alloc(R * H)
    for k in range(R * H):
        go.data[k] = y.data[k] * 2.0
    var gx = Tensor.alloc(R * D)
    m.vjp["cpu", R](TensorRefs[1](x), go, TensorRefs[1](gx))
    var out = List[List[Scalar[DT]]]()
    out.append(y.data.copy())
    out.append(gx.data.copy())
    out.append(m.weight.grd.data.copy())
    return out^


def _cmp_lists(name: String, got: List[List[Scalar[DT]]], want: List[List[Scalar[DT]]], tol: Float64 = TOL) -> Bool:
    var c = Cmp()
    for t in range(len(want)):
        c.add_n(got[t], want[t])
    return c.report(name, tol)


def _misuse() -> Bool:
    var b5 = False
    var arity = False
    try:
        var tape = Tape()
        var lin = tape.module[8, 3](Linear[D, H].make["cpu", Deterministic](), name="lin")
        _ = lin(tape.input(5, D, _rand(5 * D, 1.0)))
    except e:
        b5 = True
        print("      (batch 5:", e, ")")
    try:
        var tape = Tape()
        var m = tape.module[B](Modulate[D].make["cpu", Deterministic](), name="mod")
        var x = tape.input(B, D, _rand(B * D, 1.0))
        _ = m(x, x)
    except:
        arity = True
    print("    batch 5 on a module serving 8 and 3 raises:", b5)
    print("    two inputs to a three-input module raise:", arity)
    return b5 and arity


def main() raises:
    var ok = True
    print("CPU")
    ok = _cmp_lists("Modulate(x, scale, shift) vs x * (scale + 1) + shift",
                    _modulate(None, False), _modulate(None, True)) and ok
    ok = _cmp_lists("GRUCell(x, h) on the tape vs called directly",
                    _gru_tape(None), _gru_direct()) and ok
    var fd = _gru_fd()
    print("    GRUCell through the tape vs central differences (x and h): worst rel err ",
          fd, "  ", "OK" if fd < 5e-3 else "FAIL", sep="")
    ok = fd < 5e-3 and ok
    ok = _cmp_lists("Concat[A, B] vs F.concat; Add(x, x) vs x + x",
                    _concat_add(None, False), _concat_add(None, True)) and ok
    var got = _batches(None)
    seed(4)
    var x8 = _rand(8 * D, 1.0)
    var x3 = _rand(3 * D, 1.0)
    var want = _lin_direct[8](x8)
    var want3 = _lin_direct[3](x3)
    for t in range(3):
        want.append(want3[t].copy())
    ok = _cmp_lists("one Linear at batch 8 then 3 vs direct calls", got, want) and ok
    ok = _misuse() and ok

    var ctx = DeviceContext()
    print("GPU:", ctx.name(), "vs the CPU tape")
    ok = _cmp_lists("Modulate", _modulate(ctx, False), _modulate(None, False), 1e-4) and ok
    ok = _cmp_lists("GRUCell", _gru_tape(ctx), _gru_tape(None), 1e-4) and ok
    ok = _cmp_lists("Concat; Add(x, x)", _concat_add(ctx, False), _concat_add(None, False), 1e-4) and ok
    ok = _cmp_lists("Linear at batch 8 then 3", _batches(ctx), _batches(None), 1e-4) and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional module arity/batch gates failed")
