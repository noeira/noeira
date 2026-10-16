"""Functional tape — safety and basics: stale handles, names and
checkpoints, no_grad. On the CPU and the GPU.

1. STALE HANDLES: an activation recorded before `reset()` raises on every
   use (its slot now holds another node); parameters survive `reset`; a
   handle recorded after `reset` works.
2. NAMES: duplicates, `.` in a name and a clash with a default name raise;
   the walk is in NAME order (`named_params`); a checkpoint saved from one
   tape loads into a tape that creates the same names in ANOTHER order,
   bit for bit — and a tape with a renamed parameter refuses it.
3. NO_GRAD: the same values as with gradients; nothing requires grad, no
   gradient buffer and no saved pre-activation is allocated; `backward`
   raises; a module may be called twice; the mode is restored on exit.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_safety.mojo
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic, Kaiming
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.nn.core.named_params import named_params
from noeira.nn.primitives.linear import Linear
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor, FModule
import noeira.experimental.functional.F as F


comptime B = 8
comptime D = 5
comptime H = 6


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


def _tape(ctx: Optional[DeviceContext]) -> Tape:
    if ctx:
        return Tape(ctx.value())
    return Tape()


def _check(name: String, ok: Bool) -> Bool:
    print("    ", name, ": ", "OK" if ok else "FAIL", sep="")
    return ok


def _raises_values(t: FTensor) -> Bool:
    try:
        _ = t.values()
    except:
        return True
    return False


# ── 1. stale handles ─────────────────────────────────────────────────────
def _stale(ctx: Optional[DeviceContext]) raises -> Bool:
    seed(1)
    var tape = _tape(ctx)
    var w = tape.param(D, H, _rand(D * H, 0.5), name="w")
    var x = tape.input(B, D, _rand(B * D, 1.0))
    var h = (x @ w).tanh()
    var before = h.values()
    tape.reset()
    var x2 = tape.input(B, D, _rand(B * D, 1.0))
    var h2 = (x2 @ w).relu()  # reuses h's slot
    var ok = True
    ok = _check("stale activation raises on values()", _raises_values(h)) and ok
    var op_raises = False
    try:
        _ = h + h2
    except:
        op_raises = True
    ok = _check("stale activation raises in an op", op_raises) and ok
    var bwd_raises = False
    try:
        h.sum().backward()
    except:
        bwd_raises = True
    ok = _check("stale activation raises in backward", bwd_raises) and ok
    ok = _check("a parameter survives reset", not _raises_values(w)) and ok
    ok = _check("a handle made after reset works", not _raises_values(h2)) and ok
    # The control: without the check, `h` would have read `h2` — same slot.
    _ = before
    ok = _check("(control) the stale handle's slot is the new node's", h.id == h2.id) and ok
    return ok


# ── 2. names, walk order, checkpoint ─────────────────────────────────────
def _names(ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    var t = _tape(ctx)
    _ = t.param(2, 2, _rand(4, 1.0), name="a")
    var dup = False
    try:
        _ = t.param(2, 2, _rand(4, 1.0), name="a")
    except:
        dup = True
    ok = _check("a duplicate name raises", dup) and ok
    var dot = False
    try:
        _ = t.param(2, 2, _rand(4, 1.0), name="enc.weight")
    except:
        dot = True
    ok = _check("a `.` in a name raises", dot) and ok
    var clash2 = False
    try:
        var t2 = _tape(ctx)
        _ = t2.param(1, 1, _rand(1, 1.0), name="p1")
        _ = t2.param(1, 1, _rand(1, 1.0))  # p0
        _ = t2.param(1, 1, _rand(1, 1.0))  # p1: taken
    except:
        clash2 = True
    ok = _check("a default name that clashes with a given one raises", clash2) and ok
    return ok


def _build[target: StaticString](
    ctx: Optional[DeviceContext], order: Int, salt: Int, rename: Bool
) raises -> Tape:
    """w_in [D, H], enc (Linear[H, H] module), b_out [1, H]: created in
    order 0 (w_in, enc, b_out) or 1 (b_out, w_in, enc), values from `salt`."""
    seed(salt)
    var tape = _tape(ctx)
    var wv = _rand(D * H, 0.5)
    var bv = _rand(H, 0.5)
    var bname = "b_renamed" if rename else "b_out"
    if order == 0:
        _ = tape.param(D, H, wv, name="w_in")
        _ = tape.module[B](Linear[H, H].make[target, Kaiming](ctx), name="enc")
        _ = tape.param(1, H, bv, name=bname)
    else:
        _ = tape.param(1, H, bv, name=bname)
        _ = tape.param(D, H, wv, name="w_in")
        _ = tape.module[B](Linear[H, H].make[target, Kaiming](ctx), name="enc")
    return tape


def _native_by_name(tape: Tape) raises -> List[Scalar[DT]]:
    """The tape's own parameters' values, concatenated in NAME order — the
    same for two tapes holding the same named values in any creation order."""
    var names = tape.param_names()
    var order = List[Int]()
    for i in range(len(names)):
        order.append(i)
    for a in range(1, len(order)):
        var k = a
        while k > 0 and names[order[k]] < names[order[k - 1]]:
            var tmp = order[k]
            order[k] = order[k - 1]
            order[k - 1] = tmp
            k -= 1
    var out = List[Scalar[DT]]()
    for s in range(len(order)):
        var p = tape._wrap(order[s]).values()
        for k in range(len(p)):
            out.append(p[k])
    return out^


def _ckpt[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    var a = _build[target](ctx, 0, 11, False)
    var names = List[String]()
    for np in named_params[target](a, ctx):
        names.append(np.name)
    var want_order: List[String] = ["b_out", "enc.weight", "enc.bias", "w_in"]
    var same = len(names) == len(want_order)
    if same:
        for i in range(len(names)):
            same = same and names[i] == want_order[i]
    var shown = String("")
    for i in range(len(names)):
        shown += names[i] + " "
    ok = _check(String("walk in name order: ", shown), same) and ok

    var path = String("/tmp/noeira_functional_ckpt_", target, ".bin")
    save_params[target](a, path, ctx)
    var b = _build[target](ctx, 1, 99, False)  # other order, other values
    var pre_a = _native_by_name(a)
    var pre_b = _native_by_name(b)
    var pre_differ = 0
    for k in range(min(len(pre_a), len(pre_b))):
        if pre_a[k] != pre_b[k]:
            pre_differ += 1
    ok = _check(String("(control) before loading, ", pre_differ, " native values differ"),
                pre_differ > 0) and ok
    _ = load_params[target](b, path, ctx)
    var va = _native_by_name(a)
    var vb = _native_by_name(b)
    var eq = len(va) == len(vb) and len(va) > 0
    var differ = 0
    for k in range(min(len(va), len(vb))):
        if va[k] != vb[k]:
            differ += 1
    # The module's weights too, through its own Param (entry 0 on both).
    var ma = FModule[Linear[H, H]](a.data, 0)
    var mb = FModule[Linear[H, H]](b.data, 0)
    comptime if target == "gpu":
        ma.get().weight.val.download(ctx.value())
        mb.get().weight.val.download(ctx.value())
    for k in range(H * H):
        if ma.get().weight.val.data[k] != mb.get().weight.val.data[k]:
            differ += 1
    ok = _check(
        String("checkpoint loads into a tape built in another order: ",
               len(va) + H * H, " values, ", differ, " differ"),
        eq and differ == 0,
    ) and ok
    var c = _build[target](ctx, 0, 5, True)
    var refused = False
    try:
        _ = load_params[target](c, path, ctx)
    except e:
        refused = True
        print("      (renamed tape refused:", e, ")")
    ok = _check("a tape with a renamed parameter refuses the checkpoint", refused) and ok
    return ok


# ── 3. no_grad ───────────────────────────────────────────────────────────
def _no_grad[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    seed(3)
    var ok = True
    var wv = _rand(D * H, 0.5)
    var bv = _rand(H, 0.2)
    var xs = _rand(B * D, 1.0)

    var tg = _tape(ctx)
    var w = tg.param(D, H, wv, name="w")
    var b = tg.param(1, H, bv, name="b")
    var y_grad = F.linear(tg.input(B, D, xs), w, b, "relu").values()

    var tn = _tape(ctx)  # a FRESH tape: every slot allocated under no_grad
    var w2 = tn.param(D, H, wv, name="w")
    var b2 = tn.param(1, H, bv, name="b")
    var m = tn.module[B](Linear[H, H].make[target, Kaiming](ctx), name="m")
    var y = w2  # placeholder: a `with` body is not a definite assignment
    var twice_ok = True
    with tn.no_grad():
        y = F.linear(tn.input(B, D, xs), w2, b2, "relu")
        try:
            _ = m(y)
            _ = m(y)  # twice: fine without a backward
        except:
            twice_ok = False
    var yv = y.values()
    var same = len(yv) == len(y_grad)
    for k in range(len(yv)):
        same = same and yv[k] == y_grad[k]
    ok = _check("no_grad values == gradient-mode values", same) and ok
    ref node = tn.data[].nodes[y.id]
    ok = _check("no_grad node: no grad flag, no grad buffer, no saved z",
                not node.rg and node.grad.n == 0 and node.aux.n == 0) and ok
    var bwd = False
    try:
        y.sum().backward()
    except:
        bwd = True
    ok = _check("backward of a no_grad result raises", bwd) and ok
    ok = _check("a module called twice under no_grad", twice_ok) and ok
    ok = _check("gradients back on after the `with`", tn.grad_enabled()) and ok
    tn.reset()
    var twice_raises = False
    try:
        var x = tn.input(B, D, xs)
        var h = F.linear(x, w2, b2, "relu")
        _ = m(h)
        _ = m(h)
    except:
        twice_raises = True
    ok = _check("a module called twice WITH gradients raises", twice_raises) and ok
    return ok


def _all[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    print("  stale handles")
    ok = _stale(ctx) and ok
    print("  names")
    ok = _names(ctx) and ok
    print("  walk order and checkpoint")
    ok = _ckpt[target](ctx) and ok
    print("  no_grad")
    ok = _no_grad[target](ctx) and ok
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
        raise Error("functional safety gates failed")
