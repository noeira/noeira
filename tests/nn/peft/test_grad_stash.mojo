"""`suspend_grads` / `resume_grads`, and a frozen `LayerNorm`.

    pixi run mojo run -I . tests/nn/peft/test_grad_stash.mojo
    pixi run -e apple  mojo run -I . tests/nn/peft/test_grad_stash.mojo --gpu
    pixi run -e nvidia mojo run -I . tests/nn/peft/test_grad_stash.mojo --gpu

The stash moves a model's gradient buffers out for one backward pass, so its
`Linear` and `LayerNorm` layers form grad_input only. The model is a BFM-Zero
tower in miniature (LayerNorm -> Linear -> Mish, a residual block of the same,
a Linear head), because that is the network the FB actor step backpropagates
through.

  [1] grad_input with the gradients stashed is BIT-IDENTICAL to the full vjp
  [2] while stashed, every param reads frozen (empty gradient)
  [3] after `resume_grads` the gradients hold exactly what they held before
      the stashed vjp: it neither wrote into them nor lost them
  [4] the restored buffers are live: zero_grad + a full vjp forms the same
      gradients again, and they are nonzero, so [3] compares something
  [5] a LayerNorm with γ frozen and β trainable is refused
"""

from std.math import abs
from std.sys import argv
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.grad_stash import suspend_grads, resume_grads
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.residual import Residual
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.activations import Mish

comptime IN = 37
comptime H = 64
comptime OUT = 5
comptime B = 33

comptime Tower = Sequential[
    LayerNorm[IN], Linear[IN, H], Mish[H],
    Residual[Sequential[LayerNorm[H], Linear[H, H], Mish[H]]],
    Linear[H, OUT],
]


struct _Grads(ParamVisitor):
    """Every gradient, downloaded, in walk order; empty ones counted."""

    var vals: List[List[Scalar[DT]]]
    var empty: Int

    def __init__(out self):
        self.vals = List[List[Scalar[DT]]]()
        self.empty = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var g = List[Scalar[DT]](capacity=N)
        if grad.n == 0:
            self.empty += 1
        else:
            comptime if target == "gpu":
                grad.download(ctx.value())
                ctx.value().synchronize()
            for i in range(N):
                g.append(grad.data[i])
        self.vals.append(g^)


def _grads[target: StaticString](mut m: Tower, ctx: Optional[DeviceContext]) raises -> _Grads:
    var r = _Grads()
    m.for_each_param[target](r, ctx)
    return r^


def _same_grads(ref a: _Grads, ref b: _Grads) -> Bool:
    if len(a.vals) != len(b.vals):
        return False
    for k in range(len(a.vals)):
        if len(a.vals[k]) != len(b.vals[k]):
            return False
        for j in range(len(a.vals[k])):
            if a.vals[k][j] != b.vals[k][j]:
                return False
    return True


def _read[target: StaticString](mut t: Tensor, ctx: Optional[DeviceContext]) raises:
    comptime if target == "gpu":
        t.download(ctx.value())
        ctx.value().synchronize()


def _mismatches(a: Tensor, b: Tensor, n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if a.data[i] != b.data[i]:
            bad += 1
    return bad


def check[target: StaticString](ctx: Optional[DeviceContext]) raises:
    var m = Tower.make[target, Deterministic](ctx)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.02) * Scalar[DT]((i * 7 % 37) - 18)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.003) * Scalar[DT]((i * 5 % 23) - 11)
    var y = Tensor.alloc(B * OUT)
    var gia = Tensor.alloc(B * IN)
    var gib = Tensor.alloc(B * IN)
    comptime if target == "gpu":
        var c = ctx.value()
        x.upload(c)
        go.upload(c)
        y = Tensor.alloc_gpu(c, B * OUT)
        gia = Tensor.alloc_gpu(c, B * IN)
        gib = Tensor.alloc_gpu(c, B * IN)

    call_forward[target, B](m, TensorRefs[1, MutAnyOrigin](x), y, ctx)
    m.zero_grad[target](ctx)
    call_vjp[target, B](m, TensorRefs[1, MutAnyOrigin](x), go, TensorRefs[1, MutAnyOrigin](gia), ctx)
    var g_full = _grads[target](m, ctx)
    var n_params = len(g_full.vals)

    # the stashed pass, on top of the gradients the full pass just formed
    var held = suspend_grads[target](m, ctx)
    var g_while = _grads[target](m, ctx)
    call_vjp[target, B](m, TensorRefs[1, MutAnyOrigin](x), go, TensorRefs[1, MutAnyOrigin](gib), ctx)
    resume_grads[target](m, held^, ctx)
    var g_after = _grads[target](m, ctx)

    _read[target](gia, ctx)
    _read[target](gib, ctx)
    var bad_gi = _mismatches(gia, gib, B * IN)

    m.zero_grad[target](ctx)
    call_vjp[target, B](m, TensorRefs[1, MutAnyOrigin](x), go, TensorRefs[1, MutAnyOrigin](gia), ctx)
    var g_again = _grads[target](m, ctx)
    var mag = Float64(0)
    for k in range(len(g_full.vals)):
        for j in range(len(g_full.vals[k])):
            mag = max(mag, abs(Float64(g_full.vals[k][j])))

    print(
        "  ", target, "  params ", n_params,
        "  [1] grad_input mismatches ", bad_gi,
        "  [2] empty while stashed ", g_while.empty, "/", n_params,
        "  [3] untouched ", _same_grads(g_full, g_after),
        "  [4] live ", _same_grads(g_full, g_again), " max|g| ", mag,
        sep="",
    )
    assert_true(bad_gi == 0, "stashed grad_input differs from the full vjp's")
    assert_true(g_while.empty == n_params, "a param kept its gradient while stashed")
    assert_true(g_after.empty == 0, "resume_grads left a param frozen")
    assert_true(_same_grads(g_full, g_after), "the stashed vjp changed the gradients")
    assert_true(_same_grads(g_full, g_again), "the restored gradients are not live")
    assert_true(mag > 0.0, "every gradient is zero: [3] compared nothing")


def check_half_frozen_ln() raises:
    var ln = LayerNorm[IN].make["cpu", Deterministic]()
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT](i % 11)
    var y = Tensor.alloc(B * IN)
    var gi = Tensor.alloc(B * IN)
    var go = Tensor.alloc(B * IN)
    for i in range(B * IN):
        go.data[i] = Scalar[DT](0.001) * Scalar[DT](i % 5)
    ln.forward["cpu", B](TensorRefs[1](x), y)
    ln.gamma.freeze()
    var raised = False
    try:
        ln.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi))
    except:
        raised = True
    print("  [5] LayerNorm with only gamma frozen refused:", raised)
    assert_true(raised, "a half-frozen LayerNorm was accepted")


def main() raises:
    print("suspend_grads / resume_grads, frozen LayerNorm")
    check["cpu"](None)
    check_half_frozen_ln()
    var gpu = False
    for a in argv():
        if a == "--gpu":
            gpu = True
    if gpu:
        var ctx = DeviceContext()
        check["gpu"](Optional(ctx))
    print("PASSED")
