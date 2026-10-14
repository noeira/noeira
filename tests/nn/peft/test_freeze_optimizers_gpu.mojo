"""Optimizers on a model with a frozen layer (GPU): nothing moves it, nothing
else changes.

    pixi run -e apple  mojo run -I . tests/nn/peft/test_freeze_optimizers_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/peft/test_freeze_optimizers_gpu.mojo

`Sequential[Linear, Linear, Linear]` with the MIDDLE layer frozen: its backward
still runs (the first layer needs its grad_input), its parameters must not
move. The reference does not use the freeze at all: an identical model whose
middle layer stays trainable, with that layer's gradient zeroed after every
backward and its values restored after every step. The two must agree:

  [1] the frozen layer's values are bit-unchanged after K steps, with weight
      decay on (a frozen weight the arena still held would decay)
  [2] the frozen layer still has no gradient and no moments
  [3] the trainable layers are BIT-identical to the reference, or, with the
      global gradient clip, equal to fp32 rounding: the clip's norm is a sum
      reduced in a different order once the frozen layer has no arena slot
  [4] with the clip, it really clipped (pre-clip norm above the limit)

Arms: per-parameter Adam, arena Adam (`adopt`), arena SGD, arena Adam with
the global clip. The arena is where a frozen weight could silently train
again: it re-points every gradient it adopts.
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.sgd import SGD

comptime D = 6
comptime H = 10
comptime H2 = 12
comptime O = 5
comptime B = 7
comptime K = 5
comptime NET = Sequential[Linear[D, H], Linear[H, H2], Linear[H2, O]]
comptime MID = Linear[H, H2]


def _feed(mut x: Tensor, mut go: Tensor, step: Int, c: DeviceContext) raises:
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
    x.upload(c)
    go.upload(c)


def _get(mut t: Tensor, n: Int, c: DeviceContext) raises -> List[Scalar[DT]]:
    t.download(c)
    var out = List[Scalar[DT]](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def _put(mut t: Tensor, ref vals: List[Scalar[DT]], c: DeviceContext) raises:
    for i in range(len(vals)):
        t.data[i] = vals[i]
    t.upload_resident(c)
    t.version += 1


def _cmp(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Tuple[Int, Float64]:
    """(bitwise mismatches, max relative difference)."""
    var bad = 0
    var rel = Float64(0)
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
            var d = abs(Float64(a[i]) - Float64(b[i]))
            var s = max(abs(Float64(b[i])), 1e-6)
            rel = max(rel, d / s)
    return (bad, rel)


def _step_pair[
    ADOPT: Bool, USE_SGD: Bool, CLIP: Bool
](c: DeviceContext, label: String) raises:
    var ctx = Optional(c)
    var a = NET.make["gpu", Deterministic](ctx)      # middle layer frozen
    var r = NET.make["gpu", Deterministic](ctx)      # reference: zero + restore
    a.children[1].freeze()
    var w0 = _get(a.children[1].weight.val, MID.W_SIZE, c)
    var b0 = _get(a.children[1].bias.val, MID.B_SIZE, c)
    var rw0 = _get(r.children[1].weight.val, MID.W_SIZE, c)
    var rb0 = _get(r.children[1].bias.val, MID.B_SIZE, c)

    var adam_a = Adam(lr=1e-2, wd=0.1)
    var adam_r = Adam(lr=1e-2, wd=0.1)
    var sgd_a = SGD(lr=1e-2, wd=0.1)
    var sgd_r = SGD(lr=1e-2, wd=0.1)
    comptime if ADOPT:
        comptime if USE_SGD:
            sgd_a.adopt["gpu"](a, ctx)
            sgd_r.adopt["gpu"](r, ctx)
        else:
            adam_a.adopt["gpu"](a, ctx)
            adam_r.adopt["gpu"](r, ctx)

    var norm_a = Scalar[DT](0)
    for step in range(K):
        var x = Tensor.alloc(B * D)
        var go = Tensor.alloc(B * O)
        _feed(x, go, step, c)
        var ya = Tensor.alloc_gpu(c, B * O)
        var yr = Tensor.alloc_gpu(c, B * O)
        var ga = Tensor.alloc_gpu(c, B * D)
        var gr = Tensor.alloc_gpu(c, B * D)
        comptime if USE_SGD:
            sgd_a.zero_grad["gpu"](a, ctx)
            sgd_r.zero_grad["gpu"](r, ctx)
        else:
            adam_a.zero_grad["gpu"](a, ctx)
            adam_r.zero_grad["gpu"](r, ctx)
        a.forward["gpu", B](TensorRefs[1](x), ya, ctx)
        r.forward["gpu", B](TensorRefs[1](x), yr, ctx)
        a.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](ga), ctx)
        r.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gr), ctx)
        r.children[1].weight.zero_grad["gpu"](ctx)
        r.children[1].bias.zero_grad["gpu"](ctx)
        comptime if CLIP:
            norm_a = adam_a.clip_grads["gpu"](a, Scalar[DT](0.05), ctx)
            _ = adam_r.clip_grads["gpu"](r, Scalar[DT](0.05), ctx)
        comptime if USE_SGD:
            sgd_a.step["gpu"](a, ctx)
            sgd_r.step["gpu"](r, ctx)
        else:
            adam_a.step["gpu"](a, ctx)
            adam_r.step["gpu"](r, ctx)
        _put(r.children[1].weight.val, rw0, c)
        _put(r.children[1].bias.val, rb0, c)
    c.synchronize()

    # [1] + [2]
    var w1 = _get(a.children[1].weight.val, MID.W_SIZE, c)
    var b1 = _get(a.children[1].bias.val, MID.B_SIZE, c)
    var moved = _cmp(w1, w0)[0] + _cmp(b1, b0)[0]
    var empty = (
        a.children[1].weight.grd.n == 0 and a.children[1].bias.grd.n == 0
        and a.children[1].weight.m.n == 0 and a.children[1].weight.v.n == 0
    )
    # [3]
    var bad = 0
    var rel = Float64(0)
    var trained = Float64(0)
    var l0a = _get(a.children[0].weight.val, Linear[D, H].W_SIZE, c)
    var l0r = _get(r.children[0].weight.val, Linear[D, H].W_SIZE, c)
    var l2a = _get(a.children[2].weight.val, Linear[H2, O].W_SIZE, c)
    var l2r = _get(r.children[2].weight.val, Linear[H2, O].W_SIZE, c)
    var c0 = _cmp(l0a, l0r)
    var c2 = _cmp(l2a, l2r)
    bad = c0[0] + c2[0]
    rel = max(c0[1], c2[1])
    var fresh = NET.make["gpu", Deterministic](ctx)
    var l2f = _get(fresh.children[2].weight.val, Linear[H2, O].W_SIZE, c)
    trained = _cmp(l2a, l2f)[1]

    print(
        "  ", label, ":  [1] frozen values moved ", moved,
        "  [2] no grad/moments ", empty,
        "  [3] trainable vs reference: mismatches ", bad, ", max rel ", rel,
        "  (trained by ", trained, " rel)",
        "  [4] pre-clip norm ", norm_a, sep="",
    )
    assert_true(moved == 0, "an optimizer moved a frozen parameter")
    assert_true(empty, "a frozen parameter got a gradient or moments back")
    assert_true(trained > 1e-3, "the trainable layers did not train: [3] is vacuous")
    comptime if CLIP:
        assert_true(norm_a > Scalar[DT](0.05), "the clip never engaged")
        assert_true(rel < 1e-5, "clipped training drifted beyond fp32 rounding")
    else:
        assert_true(bad == 0, "freezing changed the trainable layers' updates")


def main() raises:
    with DeviceContext() as c:
        print("optimizers on a model with its middle layer frozen (GPU)")
        _step_pair[False, False, False](c, "Adam, per parameter")
        _step_pair[True, False, False](c, "Adam, arena")
        _step_pair[True, True, False](c, "SGD, arena")
        _step_pair[True, False, True](c, "Adam, arena, global clip 0.05")
        print("PASSED")
