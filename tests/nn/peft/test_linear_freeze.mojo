"""`Linear.freeze()` on the CPU: grad_input unchanged, no weight gradient.

    pixi run mojo run -I . tests/nn/peft/test_linear_freeze.mojo

A frozen `Linear` has EMPTY gradient buffers, and `vjp` then forms grad_input
only. The property is exact, not approximate: grad_input = go @ Wᵀ does not
read dW or db, so a frozen and a trainable copy of the same layer must return
the SAME BITS. Three more things are checked, because each is a way a frozen
layer could go wrong silently:

  [1] grad_input, frozen vs trainable: bit-identical
  [2] the frozen layer's gradients are still empty after its vjp (nothing
      re-allocated them), and `zero_grad` on it does not touch them
  [3] after `unfreeze`, the same vjp forms dW and db again, bit-identical to
      the trainable copy's: freeze/unfreeze is a round trip
  [4] the trainable copy's dW is nonzero, so [3] compares something

The GPU twin (`test_linear_freeze_gpu.mojo`) covers the padded, aligned,
cuBLAS and bf16 paths.
"""

from std.math import abs
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.linear import Linear


def _bits_equal(a: Tensor, b: Tensor, n: Int) -> Int:
    """Number of positions where the two tensors differ (0 = bit-identical)."""
    var bad = 0
    for i in range(n):
        if a.data[i] != b.data[i]:
            bad += 1
    return bad


def check[IN: Int, OUT: Int, B: Int]() raises:
    comptime L = Linear[IN, OUT]
    var la = L.make["cpu", Kaiming]()
    var lf = L.make["cpu", Kaiming]()
    for i in range(L.W_SIZE):
        lf.weight.val.data[i] = la.weight.val.data[i]
    for i in range(L.B_SIZE):
        la.bias.val.data[i] = Scalar[DT](0.01) * Scalar[DT]((i % 7) - 3)
        lf.bias.val.data[i] = la.bias.val.data[i]
    lf.freeze()
    assert_true(lf.is_frozen(), "freeze() did not empty the gradients")

    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT]((i * 7 % 37) - 18)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.003) * Scalar[DT]((i * 5 % 23) - 11)

    var ya = Tensor.alloc(B * OUT)
    var yf = Tensor.alloc(B * OUT)
    la.forward["cpu", B](TensorRefs[1](x), ya)
    lf.forward["cpu", B](TensorRefs[1](x), yf)
    assert_true(_bits_equal(ya, yf, B * OUT) == 0, "forward differs")

    var gia = Tensor.alloc(B * IN)
    var gif = Tensor.alloc(B * IN)
    la.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gia))
    lf.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gif))

    # [1]
    var bad = _bits_equal(gia, gif, B * IN)
    # [2]
    var still_empty = lf.weight.grd.n == 0 and lf.bias.grd.n == 0
    lf.zero_grad["cpu"](None)
    var empty_after_zero = lf.weight.grd.n == 0 and lf.bias.grd.n == 0
    # [4]
    var dw_mag = Float64(0)
    for i in range(L.W_SIZE):
        dw_mag = max(dw_mag, abs(Float64(la.weight.grd.data[i])))

    # [3] unfreeze, run the same vjp on a fresh trainable copy and on lf.
    lf.unfreeze["cpu"]()
    la.zero_grad["cpu"](None)
    la.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gia))
    lf.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gif))
    var bad_dw = _bits_equal(la.weight.grd, lf.weight.grd, L.W_SIZE)
    var bad_db = _bits_equal(la.bias.grd, lf.bias.grd, L.B_SIZE)

    print(
        "  Linear[", IN, ", ", OUT, "] B=", B,
        "  [1] grad_input mismatches ", bad,
        "  [2] empty after vjp ", still_empty, " / after zero_grad ",
        empty_after_zero,
        "  [3] dW/db mismatches after unfreeze ", bad_dw, "/", bad_db,
        "  [4] max|dW| ", dw_mag,
        sep="",
    )
    assert_true(bad == 0, "frozen grad_input differs from the trainable one")
    assert_true(still_empty, "the frozen vjp allocated a gradient")
    assert_true(empty_after_zero, "zero_grad allocated a frozen gradient")
    assert_true(bad_dw == 0 and bad_db == 0, "unfreeze is not a round trip")
    assert_true(dw_mag > 0.0, "dW is zero: [3] compared nothing")


def main() raises:
    print("Linear.freeze(), CPU")
    check[6, 101, 33]()
    check[64, 64, 16]()
    check[960, 320, 140]()   # SmolVLA's k/v projection at the prefill's P
    print("PASSED")
