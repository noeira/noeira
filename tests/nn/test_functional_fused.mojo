"""Fused `F.linear(x, w, b, act)` and CUDA-graph capture of a tape step —
gates, on the CPU and on the GPU.

1. FUSED vs UNFUSED: one `OP_LINEAR` node against `(x @ w + b).act()`
   (three nodes), every activation, value and the gradients of w, b, x.
2. FUSED vs the comptime nn: `Linear` / `LinearAct[ReLU|Tanh]` handed to the
   tape (`tape.module`) against `F.linear` with the same weights.
3. CAPTURE (NVIDIA): an MLP step captured once, then replayed on a refilled
   input buffer, against an eager tape run on that input. Off NVIDIA the
   capture is unavailable and the row says so (not a pass).

Run: pixi run -e default mojo run -I . tests/nn/test_functional_fused.mojo
"""

from std.random import seed, random_float64
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_act import LinearAct
from noeira.nn.primitives.ops.relu_op import ReLUOp
from noeira.nn.primitives.ops.tanh_op import TanhOp
from noeira.nn.functional import Tape, FTensor, FModule
import noeira.nn.functional.F as F


comptime B = 96
comptime D = 13
comptime H = 40
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

    def __init__(out self):
        self.n = 0
        self.differ = 0
        self.worst = 0
        self.max_ref = 0

    def add_n(mut self, got: List[Scalar[DT]], want: List[Scalar[DT]], n: Int):
        for k in range(n):
            self.n += 1
            if got[k] != want[k]:
                self.differ += 1
            var w = Float64(want[k])
            var e = abs(Float64(got[k]) - w) / max(1.0, abs(w))
            if e > self.worst:
                self.worst = e
            if abs(w) > self.max_ref:
                self.max_ref = abs(w)

    def report(self, name: String) -> Bool:
        var ok = self.n > 0 and self.max_ref > 0 and self.worst <= TOL
        print("    ", name, ": ", self.n, " compared, ", self.differ,
              " not bit-identical, worst rel ", self.worst, "  ",
              "OK" if ok else "FAIL", sep="")
        return ok


def _tape(ctx: Optional[DeviceContext]) -> Tape:
    if ctx:
        return Tape(ctx.value())
    return Tape()


def _act(h: FTensor, act: StaticString) raises -> FTensor:
    if act == "none":
        return h
    elif act == "relu":
        return h.relu()
    elif act == "tanh":
        return h.tanh()
    elif act == "sigmoid":
        return h.sigmoid()
    elif act == "gelu":
        return h.gelu()
    return h.mish()


def _fused_case[act: StaticString](
    ctx: Optional[DeviceContext], fused: Bool
) raises -> List[List[Scalar[DT]]]:
    seed(3)
    var tape = _tape(ctx)
    var w = tape.param(D, H, _rand(D * H, 0.4))
    var b = tape.param(1, H, _rand(H, 0.3))
    var x = tape.input(B, D, _rand(B * D, 1.0), requires_grad=True)
    var r = tape.input(B, H, _rand(B * H, 1.0))
    var h: FTensor
    if fused:
        comptime if act == "none":
            h = F.linear(x, w, b)
        else:
            h = F.linear(x, w, b, act)
    else:
        h = _act(x @ w + b, act)
    var f = (h * r).sum()
    f.backward()
    var out = List[List[Scalar[DT]]]()
    out.append(h.values())
    out.append(w.grads())
    out.append(b.grads())
    out.append(x.grads())
    return out^


def _fused_vs_unfused[act: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var got = _fused_case[act](ctx, True)
    var want = _fused_case[act](ctx, False)
    var c = Cmp()
    for t in range(4):
        c.add_n(got[t], want[t], len(want[t]))
    return c.report(String("F.linear(…, ", act, ") vs (x @ w + b).", act))


def _vs_module[act: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    """The same layer as a comptime module on a tape and as `F.linear`."""
    seed(5)
    var wv = _rand(D * H, 0.4)
    var bv = _rand(H, 0.3)
    var xs = _rand(B * D, 1.0)
    var rs = _rand(B * H, 1.0)

    var tf = _tape(ctx)
    var w = tf.param(D, H, wv)
    var b = tf.param(1, H, bv)
    var x = tf.input(B, D, xs, requires_grad=True)
    var hf: FTensor
    comptime if act == "none":
        hf = F.linear(x, w, b)
    else:
        hf = F.linear(x, w, b, act)
    (hf * tf.input(B, H, rs)).sum().backward()

    var tm = _tape(ctx)
    var xm = tm.input(B, D, xs, requires_grad=True)
    var hm: FTensor
    var gw: List[Scalar[DT]]
    var gb: List[Scalar[DT]]
    if ctx:
        comptime if act == "none":
            var m = tm.module[B](Linear[D, H].make["gpu", Deterministic](ctx))
            _load["gpu"](m.get().weight.val, wv, ctx)
            _load["gpu"](m.get().bias.val, bv, ctx)
            hm = m(xm)
            (hm * tm.input(B, H, rs)).sum().backward()
            gw = _dl["gpu"](m.get().weight.grd, ctx)
            gb = _dl["gpu"](m.get().bias.grd, ctx)
        elif act == "relu":
            var m = tm.module[B](LinearAct[D, H, ReLUOp].make["gpu", Deterministic](ctx))
            _load["gpu"](m.get().weight.val, wv, ctx)
            _load["gpu"](m.get().bias.val, bv, ctx)
            hm = m(xm)
            (hm * tm.input(B, H, rs)).sum().backward()
            gw = _dl["gpu"](m.get().weight.grd, ctx)
            gb = _dl["gpu"](m.get().bias.grd, ctx)
        else:
            var m = tm.module[B](LinearAct[D, H, TanhOp].make["gpu", Deterministic](ctx))
            _load["gpu"](m.get().weight.val, wv, ctx)
            _load["gpu"](m.get().bias.val, bv, ctx)
            hm = m(xm)
            (hm * tm.input(B, H, rs)).sum().backward()
            gw = _dl["gpu"](m.get().weight.grd, ctx)
            gb = _dl["gpu"](m.get().bias.grd, ctx)
    else:
        comptime if act == "none":
            var m = tm.module[B](Linear[D, H].make["cpu", Deterministic](ctx))
            _load["cpu"](m.get().weight.val, wv, ctx)
            _load["cpu"](m.get().bias.val, bv, ctx)
            hm = m(xm)
            (hm * tm.input(B, H, rs)).sum().backward()
            gw = m.get().weight.grd.data.copy()
            gb = m.get().bias.grd.data.copy()
        elif act == "relu":
            var m = tm.module[B](LinearAct[D, H, ReLUOp].make["cpu", Deterministic](ctx))
            _load["cpu"](m.get().weight.val, wv, ctx)
            _load["cpu"](m.get().bias.val, bv, ctx)
            hm = m(xm)
            (hm * tm.input(B, H, rs)).sum().backward()
            gw = m.get().weight.grd.data.copy()
            gb = m.get().bias.grd.data.copy()
        else:
            var m = tm.module[B](LinearAct[D, H, TanhOp].make["cpu", Deterministic](ctx))
            _load["cpu"](m.get().weight.val, wv, ctx)
            _load["cpu"](m.get().bias.val, bv, ctx)
            hm = m(xm)
            (hm * tm.input(B, H, rs)).sum().backward()
            gw = m.get().weight.grd.data.copy()
            gb = m.get().bias.grd.data.copy()
    var c = Cmp()
    c.add_n(hf.values(), hm.values(), B * H)
    c.add_n(w.grads(), gw, D * H)
    c.add_n(b.grads(), gb, H)
    c.add_n(x.grads(), xm.grads(), B * D)
    var name = "Linear" if act == "none" else String("LinearAct[", act, "]")
    return c.report(String("F.linear vs ", name, " module"))


def _load[target: StaticString](
    mut t: Tensor, vals: List[Scalar[DT]], ctx: Optional[DeviceContext]
) raises:
    for k in range(len(vals)):
        t.data[k] = vals[k]
    comptime if target == "gpu":
        t.upload(ctx.value())


def _dl[target: StaticString](
    mut t: Tensor, ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    comptime if target == "gpu":
        t.download(ctx.value())
    return t.data.copy()


# ── 3. capture ───────────────────────────────────────────────────────────
def _mlp_step(
    tape: Tape, src: Tensor, ys: Tensor,
    w1: FTensor, b1: FTensor, w2: FTensor, b2: FTensor,
) raises -> FTensor:
    tape.reset()
    tape.zero_grad()
    var x = tape.input_device(B, D, src.dev.value())
    var h = F.linear(x, w1, b1, "relu")
    var loss = F.mse(F.linear(h, w2, b2), tape.input_device(B, 1, ys.dev.value()))
    loss.backward()
    return loss


def _capture(ctx: DeviceContext) raises -> Bool:
    seed(9)
    var wv1 = _rand(D * H, 0.4)
    var bv1 = _rand(H, 0.2)
    var wv2 = _rand(H, 0.4)
    var bv2 = _rand(1, 0.2)
    var xa = _rand(B * D, 1.0)
    var xb = _rand(B * D, 1.0)
    var yv = _rand(B, 1.0)

    def _dev(vals: List[Scalar[DT]], ctx: DeviceContext) raises -> Tensor:
        var t = Tensor.alloc(len(vals))
        for k in range(len(vals)):
            t.data[k] = vals[k]
        t.upload(ctx)
        return t^

    var src = _dev(xa, ctx)
    var ys = _dev(yv, ctx)
    var tape = Tape(ctx)
    var w1 = tape.param(D, H, wv1)
    var b1 = tape.param(1, H, bv1)
    var w2 = tape.param(H, 1, wv2)
    var b2 = tape.param(1, 1, bv2)
    _ = _mlp_step(tape, src, ys, w1, b1, w2, b2)  # eager: allocates
    tape.capture_begin()
    var loss = _mlp_step(tape, src, ys, w1, b1, w2, b2)
    tape.capture_end()
    if not tape.captured():
        print("    capture unavailable on this device (NVIDIA only) — not tested")
        return True
    # Refill the input buffer IN PLACE, replay twice (grads are zeroed
    # inside the step, so two replays = one).
    for k in range(B * D):
        src.data[k] = xb[k]
    src.upload_resident(ctx)
    tape.replay()
    tape.replay()
    var got_loss = loss.item()
    var got_w1 = w1.grads()
    var got_b2 = b2.grads()

    var ref_t = Tape(ctx)
    var srcb = _dev(xb, ctx)
    var r1 = ref_t.param(D, H, wv1)
    var rb1 = ref_t.param(1, H, bv1)
    var r2 = ref_t.param(H, 1, wv2)
    var rb2 = ref_t.param(1, 1, bv2)
    var ref_loss = _mlp_step(ref_t, srcb, ys, r1, rb1, r2, rb2)
    var c = Cmp()
    c.add_n([got_loss], [ref_loss.item()], 1)
    c.add_n(got_w1, r1.grads(), D * H)
    c.add_n(got_b2, rb2.grads(), 1)
    var ok = c.report("captured step replayed on a refilled input vs eager")
    var stale = Cmp()
    var eager_a = Tape(ctx)
    var sa = _dev(xa, ctx)
    var a1 = eager_a.param(D, H, wv1)
    var ab1 = eager_a.param(1, H, bv1)
    var a2 = eager_a.param(H, 1, wv2)
    var ab2 = eager_a.param(1, 1, bv2)
    var la = _mlp_step(eager_a, sa, ys, a1, ab1, a2, ab2)
    var differs = la.item() != got_loss
    print("    the replay used the NEW input (loss differs from the captured one):", differs)
    return ok and differs


def _all(ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    print("  fused vs unfused")
    ok = _fused_vs_unfused["none"](ctx) and ok
    ok = _fused_vs_unfused["relu"](ctx) and ok
    ok = _fused_vs_unfused["tanh"](ctx) and ok
    ok = _fused_vs_unfused["sigmoid"](ctx) and ok
    ok = _fused_vs_unfused["gelu"](ctx) and ok
    ok = _fused_vs_unfused["mish"](ctx) and ok
    print("  fused vs comptime modules")
    ok = _vs_module["none"](ctx) and ok
    ok = _vs_module["relu"](ctx) and ok
    ok = _vs_module["tanh"](ctx) and ok
    return ok


def main() raises:
    var ok = True
    print("CPU")
    ok = _all(None) and ok
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    ok = _all(ctx) and ok
    print("  capture")
    ok = _capture(ctx) and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional fused/capture gates failed")
