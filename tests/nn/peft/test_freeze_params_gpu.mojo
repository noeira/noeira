"""`freeze_params` over Linear + RMSNorm on the GPU, and a load after a forward.

    pixi run -e apple  mojo run -I . tests/nn/peft/test_freeze_params_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/peft/test_freeze_params_gpu.mojo

[1]-[3] as in `test_freeze_params.mojo`, on the GPU backward paths.

[4] `fill_param` (the safetensors and torch-name loaders' write) must advance
the weight's `version`. `Linear` caches a padded copy of its weight (`w_pad`,
fp32 with a padded K) and a bf16 copy (`w_bf`, bf16-flow), both rebuilt only
when the version moved. A forward run BEFORE a load built them from the old
weight; without the bump the next forward keeps using it. A trained layer
hides that at its first optimizer step (which bumps); a frozen layer is never
stepped and would serve the old weight forever. The check: forward, load new
weights, forward again, against a layer that was loaded before any forward.
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.safetensors_io import fill_param
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
comptime BF16 = DType.bfloat16


def _same[dt: DType](mut a: TensorImpl[dt], mut b: TensorImpl[dt], n: Int, c: DeviceContext) raises -> Int:
    a.download(c)
    b.download(c)
    c.synchronize()
    var bad = 0
    for i in range(n):
        if a.data[i] != b.data[i]:
            bad += 1
    return bad


def freeze_legs(c: DeviceContext) raises:
    var ctx = Optional(c)
    var f = NET.make["gpu", Deterministic](ctx)
    var t = NET.make["gpu", Deterministic](ctx)
    var x = Tensor.alloc(B * D)
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i * 7) % 11) - 5) * 0.2
    var go = Tensor.alloc(B * O)
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 5) % 9) - 4) * 0.3
    x.upload(c)
    go.upload(c)

    var n = freeze_params["gpu"](f, ctx)
    var empty = (
        f.children[0].weight.grd.n == 0 and f.children[1].gamma.grd.n == 0
        and f.children[2].weight.grd.n == 0 and f.children[2].bias.grd.n == 0
    )
    print("  [1] froze", n, "values (expect", N_VALUES, "), all gradients empty:", empty)
    assert_true(n == N_VALUES and empty, "freeze_params on the GPU")

    var yf = Tensor.alloc_gpu(c, B * O)
    var yt = Tensor.alloc_gpu(c, B * O)
    var gf = Tensor.alloc_gpu(c, B * D)
    var gt = Tensor.alloc_gpu(c, B * D)
    f.forward["gpu", B](TensorRefs[1](x), yf, ctx)
    t.forward["gpu", B](TensorRefs[1](x), yt, ctx)
    f.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gf), ctx)
    t.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gt), ctx)
    var bad = _same(gf, gt, B * D, c)
    print("  [2] grad_input, frozen vs trainable: mismatches", bad)
    assert_true(bad == 0, "a frozen GPU backward changed grad_input")
    assert_true(f.children[1].gamma.grd.n == 0, "RMSNorm re-allocated gamma's gradient")

    var m = unfreeze_params["gpu"](f, ctx)
    t.zero_grad["gpu"](ctx)
    f.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gf), ctx)
    t.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gt), ctx)
    var bad_p = (
        _same(f.children[0].weight.grd, t.children[0].weight.grd, D * H, c)
        + _same(f.children[1].gamma.grd, t.children[1].gamma.grd, H, c)
        + _same(f.children[2].weight.grd, t.children[2].weight.grd, H * O, c)
    )
    print("  [3] unfroze", m, "values; parameter gradients vs trainable: mismatches", bad_p)
    assert_true(m == N_VALUES and bad_p == 0, "unfreeze on the GPU is not a round trip")


def load_after_forward[IN: Int, OUT: Int, ADT: DType](c: DeviceContext, label: String) raises:
    """[4] forward -> fill_param -> forward, against a layer loaded first."""
    comptime L = Linear[IN, OUT, ADT]
    comptime BB = 16
    var ctx = Optional(c)
    var stale = L.make["gpu", Deterministic](ctx)
    var fresh = L.make["gpu", Deterministic](ctx)
    var new_w = List[Float32](capacity=L.W_SIZE)
    for i in range(L.W_SIZE):
        new_w.append(Float32(((i * 13) % 17) - 8) * 0.01)
    var x = TensorImpl[ADT].alloc(BB * IN)
    for i in range(BB * IN):
        x.data[i] = (Scalar[DT](((i * 3) % 7) - 3) * 0.1).cast[ADT]()
    x.upload(c)
    var y_stale = TensorImpl[ADT].alloc(BB * OUT)
    var y_fresh = TensorImpl[ADT].alloc(BB * OUT)
    stale.forward["gpu", BB](TensorRefs[1, ADT=ADT](x), y_stale, ctx)  # caches built
    fill_param(stale.weight.val, new_w, ctx)
    fill_param(fresh.weight.val, new_w, ctx)
    stale.forward["gpu", BB](TensorRefs[1, ADT=ADT](x), y_stale, ctx)
    fresh.forward["gpu", BB](TensorRefs[1, ADT=ADT](x), y_fresh, ctx)
    var bad = _same(y_stale, y_fresh, BB * OUT, c)
    print("  [4]", label, ": forward after a load vs a layer loaded first: mismatches", bad, "of", BB * OUT)
    assert_true(bad == 0, "a load after a forward did not reach the next forward (stale weight cache)")


def main() raises:
    with DeviceContext() as c:
        print("freeze_params over Linear + RMSNorm, and a load after a forward (GPU)")
        freeze_legs(c)
        load_after_forward[200, 256, DT](c, "fp32, K padded (w_pad)")
        load_after_forward[128, 256, BF16](c, "bf16 flow (w_bf)")
        print("PASSED")
