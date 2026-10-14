"""`freeze_params` / `unfreeze_params` over a model with an RMSNorm (CPU).

    pixi run mojo run -I . tests/nn/peft/test_freeze_params.mojo

`Sequential[Linear, RMSNorm, Linear]`, every parameter frozen with one walk:

  [1] the walk returns the number of values it froze, and every gradient
      and moment buffer is empty afterwards
  [2] the backward still runs (`Linear` and `RMSNorm` skip their parameter
      gradients) and its grad_input is BIT-identical to the trainable model's
  [3] `unfreeze_params` gives every parameter a zeroed gradient back, and the
      same backward then forms parameter gradients identical to the trainable
      model's
"""

from std.math import abs
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.peft.freeze import freeze_params, unfreeze_params

comptime D = 6
comptime H = 8
comptime O = 5
comptime B = 9
comptime NET = Sequential[Linear[D, H], RMSNorm[H], Linear[H, O]]
comptime N_VALUES = D * H + H + H + H * O + O


def _same(ref a: Tensor, ref b: Tensor, n: Int) -> Int:
    var bad = 0
    for i in range(n):
        if a.data[i] != b.data[i]:
            bad += 1
    return bad


def main() raises:
    print("freeze_params / unfreeze_params, Linear + RMSNorm (CPU)")
    var f = NET.make["cpu", Deterministic]()
    var t = NET.make["cpu", Deterministic]()
    var x = Tensor.alloc(B * D)
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i * 7) % 11) - 5) * 0.2
    var go = Tensor.alloc(B * O)
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 5) % 9) - 4) * 0.3

    # [1]
    var n = freeze_params["cpu"](f)
    var empty = (
        f.children[0].weight.grd.n == 0 and f.children[0].bias.grd.n == 0
        and f.children[1].gamma.grd.n == 0
        and f.children[2].weight.grd.n == 0 and f.children[2].bias.grd.n == 0
    )
    print("  [1] froze", n, "values (expect", N_VALUES, "), all gradients empty:", empty)
    assert_true(n == N_VALUES, "freeze_params did not count every value")
    assert_true(empty, "freeze_params left a gradient allocated")

    # [2]
    var yf = Tensor.alloc(B * O)
    var yt = Tensor.alloc(B * O)
    var gf = Tensor.alloc(B * D)
    var gt = Tensor.alloc(B * D)
    f.forward["cpu", B](TensorRefs[1](x), yf)
    t.forward["cpu", B](TensorRefs[1](x), yt)
    f.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gf))
    t.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gt))
    var bad = _same(gf, gt, B * D)
    print("  [2] grad_input, frozen vs trainable: mismatches", bad)
    assert_true(bad == 0, "a frozen backward changed grad_input")
    assert_true(f.children[1].gamma.grd.n == 0, "RMSNorm re-allocated gamma's gradient")

    # [3]
    var m = unfreeze_params["cpu"](f)
    t.zero_grad["cpu"](None)
    f.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gf))
    t.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gt))
    var bad_p = (
        _same(f.children[0].weight.grd, t.children[0].weight.grd, D * H)
        + _same(f.children[1].gamma.grd, t.children[1].gamma.grd, H)
        + _same(f.children[2].weight.grd, t.children[2].weight.grd, H * O)
        + _same(f.children[2].bias.grd, t.children[2].bias.grd, O)
    )
    var mag = Float64(0)
    for i in range(H):
        mag = max(mag, abs(Float64(t.children[1].gamma.grd.data[i])))
    print("  [3] unfroze", m, "values; parameter gradients vs trainable: mismatches",
          bad_p, " (max |dgamma|", mag, ")")
    assert_true(m == N_VALUES, "unfreeze_params did not count every value")
    assert_true(bad_p == 0, "unfreeze is not a round trip")
    assert_true(mag > 0.0, "dgamma is zero: [3] compared nothing for RMSNorm")
    print("PASSED")
