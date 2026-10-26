"""Optimizers on a model with a frozen layer (CPU): nothing moves it, nothing
else changes.

    pixi run mojo run -I . tests/nn/peft/test_freeze_optimizers.mojo

The CPU twin of `test_freeze_optimizers_gpu.mojo`, same legs and the same
reference (an identical model whose middle layer stays trainable, its gradient
zeroed after every backward and its values restored after every step), over
the CPU optimizer paths: per-parameter Adam, SGD and DreamerOpt (each with
weight decay where it has one), and Adam behind the global-norm clip, which on
the CPU runs the per-parameter clip visitors. Before the frozen-parameter skips,
every one of these read the frozen layer's empty gradient and aborted.
"""

from std.math import abs
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.sgd import SGD
from noeira.nn.optimizer.dreamer_opt import DreamerOpt

comptime D = 6
comptime H = 10
comptime H2 = 12
comptime O = 5
comptime B = 7
comptime K = 5
comptime NET = Sequential[Linear[D, H], Linear[H, H2], Linear[H2, O]]
comptime MID = Linear[H, H2]


def _feed(mut x: Tensor, mut go: Tensor, step: Int):
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4


def _get(ref t: Tensor, n: Int) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def _put(mut t: Tensor, ref vals: List[Scalar[DT]]):
    for i in range(len(vals)):
        t.data[i] = vals[i]
    t.version += 1


def _cmp(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Tuple[Int, Float64]:
    var bad = 0
    var rel = Float64(0)
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
            rel = max(
                rel,
                abs(Float64(a[i]) - Float64(b[i])) / max(abs(Float64(b[i])), 1e-6),
            )
    return (bad, rel)


def _run[OPT: Int, CLIP: Bool](label: String) raises:
    """OPT: 0 Adam, 1 SGD, 2 DreamerOpt."""
    var a = NET.make["cpu", Deterministic]()
    var r = NET.make["cpu", Deterministic]()
    a.children[1].freeze()
    var w0 = _get(a.children[1].weight.val, MID.W_SIZE)
    var b0 = _get(a.children[1].bias.val, MID.B_SIZE)
    var rw0 = _get(r.children[1].weight.val, MID.W_SIZE)
    var rb0 = _get(r.children[1].bias.val, MID.B_SIZE)
    var adam_a = Adam(lr=1e-2, wd=0.1)
    var adam_r = Adam(lr=1e-2, wd=0.1)
    var sgd_a = SGD(lr=1e-2, wd=0.1)
    var sgd_r = SGD(lr=1e-2, wd=0.1)
    var dr_a = DreamerOpt(lr=1e-2)
    var dr_r = DreamerOpt(lr=1e-2)
    var norm_a = Scalar[DT](0)
    for step in range(K):
        var x = Tensor.alloc(B * D)
        var go = Tensor.alloc(B * O)
        _feed(x, go, step)
        var ya = Tensor.alloc(B * O)
        var yr = Tensor.alloc(B * O)
        var ga = Tensor.alloc(B * D)
        var gr = Tensor.alloc(B * D)
        a.zero_grad["cpu"](None)
        r.zero_grad["cpu"](None)
        a.forward["cpu", B](TensorRefs[1](x), ya)
        r.forward["cpu", B](TensorRefs[1](x), yr)
        a.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](ga))
        r.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gr))
        r.children[1].weight.zero_grad["cpu"](None)
        r.children[1].bias.zero_grad["cpu"](None)
        comptime if CLIP:
            norm_a = adam_a.clip_grads["cpu"](a, Scalar[DT](0.05))
            _ = adam_r.clip_grads["cpu"](r, Scalar[DT](0.05))
        comptime if OPT == 0:
            adam_a.step["cpu"](a)
            adam_r.step["cpu"](r)
        elif OPT == 1:
            sgd_a.step["cpu"](a)
            sgd_r.step["cpu"](r)
        else:
            dr_a.step["cpu"](a)
            dr_r.step["cpu"](r)
        _put(r.children[1].weight.val, rw0)
        _put(r.children[1].bias.val, rb0)

    var moved = (
        _cmp(_get(a.children[1].weight.val, MID.W_SIZE), w0)[0]
        + _cmp(_get(a.children[1].bias.val, MID.B_SIZE), b0)[0]
    )
    var empty = (
        a.children[1].weight.grd.n == 0 and a.children[1].bias.grd.n == 0
        and a.children[1].weight.m.n == 0 and a.children[1].weight.v.n == 0
    )
    var c0 = _cmp(
        _get(a.children[0].weight.val, Linear[D, H].W_SIZE),
        _get(r.children[0].weight.val, Linear[D, H].W_SIZE),
    )
    var l2a = _get(a.children[2].weight.val, Linear[H2, O].W_SIZE)
    var c2 = _cmp(l2a, _get(r.children[2].weight.val, Linear[H2, O].W_SIZE))
    var fresh = NET.make["cpu", Deterministic]()
    var trained = _cmp(l2a, _get(fresh.children[2].weight.val, Linear[H2, O].W_SIZE))[1]
    print(
        "  ", label, ":  [1] frozen values moved ", moved,
        "  [2] no grad/moments ", empty,
        "  [3] trainable vs reference: mismatches ", c0[0] + c2[0],
        ", max rel ", max(c0[1], c2[1]), "  [4] pre-clip norm ", norm_a, sep="",
    )
    assert_true(moved == 0, "an optimizer moved a frozen parameter")
    assert_true(empty, "a frozen parameter got a gradient or moments back")
    assert_true(trained > 1e-3, "the trainable layers did not train: [3] is vacuous")
    comptime if CLIP:
        assert_true(norm_a > Scalar[DT](0.05), "the clip never engaged")
        assert_true(max(c0[1], c2[1]) < 1e-5, "clipped training drifted")
    else:
        assert_true(c0[0] + c2[0] == 0, "freezing changed the trainable updates")


def main() raises:
    print("optimizers on a model with its middle layer frozen (CPU)")
    _run[0, False]("Adam")
    _run[1, False]("SGD")
    _run[2, False]("DreamerOpt")
    _run[0, True]("Adam, global clip 0.05")
    print("PASSED")
