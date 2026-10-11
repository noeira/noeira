"""`Linear.freeze()` on the GPU: every vjp path, grad_input bit-identical.

    pixi run -e apple  mojo run -I . tests/nn/peft/test_linear_freeze_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/peft/test_linear_freeze_gpu.mojo

Same four legs as `test_linear_freeze.mojo`, over the shapes that select each
GPU branch of `Linear.vjp`:

    shape            Apple (PAD_TO 128)          NVIDIA (CUBLAS_BWD)
    [6, 101]         K and N padded              cuBLAS, unpadded
    [128, 128]       aligned (unpadded branch)   cuBLAS
    [200, 256]       K padded only               cuBLAS
    [960, 320]       K and N padded (SmolVLA)    cuBLAS
    [128, 256] bf16  MAX bf16 path               cuBLASLt bf16 (CUB_LP)

On NVIDIA, `NN_GEMM_PATH=max` puts the fp32 shapes back on the MAX branches.
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear

comptime BF16 = DType.bfloat16


def _mismatches[dt: DType](
    a: TensorImpl[dt], b: TensorImpl[dt], n: Int
) -> Int:
    var bad = 0
    for i in range(n):
        if a.data[i] != b.data[i]:
            bad += 1
    return bad


def check[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    comptime L = Linear[IN, OUT]
    var c = Optional(ctx)
    var la = L.make["gpu", Deterministic](c)
    var lf = L.make["gpu", Deterministic](c)
    lf.freeze()
    assert_true(lf.is_frozen(), "freeze() did not empty the gradients")

    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT]((i * 7 % 37) - 18)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.003) * Scalar[DT]((i * 5 % 23) - 11)
    x.upload(ctx)
    go.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * OUT)
    var yf = Tensor.alloc_gpu(ctx, B * OUT)
    la.forward["gpu", B](TensorRefs[1](x), ya, c)
    lf.forward["gpu", B](TensorRefs[1](x), yf, c)

    var gia = Tensor.alloc_gpu(ctx, B * IN)
    var gif = Tensor.alloc_gpu(ctx, B * IN)
    la.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gia), c)
    lf.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gif), c)
    gia.download(ctx)
    gif.download(ctx)
    ctx.synchronize()
    var bad = _mismatches(gia, gif, B * IN)
    var still_empty = lf.weight.grd.n == 0 and lf.bias.grd.n == 0
    lf.zero_grad["gpu"](c)
    var empty_after_zero = lf.weight.grd.n == 0 and lf.bias.grd.n == 0

    la.weight.grd.download(ctx)
    ctx.synchronize()
    var dw_mag = Float64(0)
    for i in range(L.W_SIZE):
        dw_mag = max(dw_mag, abs(Float64(la.weight.grd.data[i])))

    # unfreeze: the full path again, against a freshly zeroed trainable copy
    lf.unfreeze["gpu"](c)
    la.zero_grad["gpu"](c)
    la.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gia), c)
    lf.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gif), c)
    la.weight.grd.download(ctx)
    lf.weight.grd.download(ctx)
    la.bias.grd.download(ctx)
    lf.bias.grd.download(ctx)
    ctx.synchronize()
    var bad_dw = _mismatches(la.weight.grd, lf.weight.grd, L.W_SIZE)
    var bad_db = _mismatches(la.bias.grd, lf.bias.grd, L.B_SIZE)

    print(
        "  Linear[", IN, ", ", OUT, "] B=", B,
        " pad K/N ", L.NEEDS_PAD, "/", L.NEEDS_N_PAD,
        "  [1] grad_input mismatches ", bad,
        "  [2] empty ", still_empty, "/", empty_after_zero,
        "  [3] dW/db after unfreeze ", bad_dw, "/", bad_db,
        "  [4] max|dW| ", dw_mag,
        sep="",
    )
    assert_true(bad == 0, "frozen grad_input differs from the trainable one")
    assert_true(still_empty, "the frozen vjp allocated a gradient")
    assert_true(empty_after_zero, "zero_grad allocated a frozen gradient")
    assert_true(bad_dw == 0 and bad_db == 0, "unfreeze is not a round trip")
    assert_true(dw_mag > 0.0, "dW is zero: [3] compared nothing")


def check_bf16[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    """The bf16-flow Linear: activations at bf16, master grads fp32."""
    comptime L = Linear[IN, OUT, BF16]
    var c = Optional(ctx)
    var la = L.make["gpu", Deterministic](c)
    var lf = L.make["gpu", Deterministic](c)
    lf.freeze()

    var x = TensorImpl[BF16].alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = (Scalar[DT](0.01) * Scalar[DT]((i * 7 % 37) - 18)).cast[BF16]()
    var go = TensorImpl[BF16].alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = (Scalar[DT](0.003) * Scalar[DT]((i * 5 % 23) - 11)).cast[BF16]()
    x.upload(ctx)
    go.upload(ctx)

    var ya = TensorImpl[BF16].alloc(B * OUT)
    var yf = TensorImpl[BF16].alloc(B * OUT)
    la.forward["gpu", B](TensorRefs[1, ADT=BF16](x), ya, c)
    lf.forward["gpu", B](TensorRefs[1, ADT=BF16](x), yf, c)

    var gia = TensorImpl[BF16].alloc(B * IN)
    var gif = TensorImpl[BF16].alloc(B * IN)
    la.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gia), c
    )
    lf.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gif), c
    )
    gia.download(ctx)
    gif.download(ctx)
    la.weight.grd.download(ctx)
    ctx.synchronize()
    var bad = _mismatches(gia, gif, B * IN)
    var still_empty = lf.weight.grd.n == 0 and lf.bias.grd.n == 0
    var dw_mag = Float64(0)
    for i in range(L.W_SIZE):
        dw_mag = max(dw_mag, abs(Float64(la.weight.grd.data[i])))

    lf.unfreeze["gpu"](c)
    la.zero_grad["gpu"](c)
    la.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gia), c
    )
    lf.vjp["gpu", B](
        TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gif), c
    )
    la.weight.grd.download(ctx)
    lf.weight.grd.download(ctx)
    la.bias.grd.download(ctx)
    lf.bias.grd.download(ctx)
    ctx.synchronize()
    var bad_dw = _mismatches(la.weight.grd, lf.weight.grd, L.W_SIZE)
    var bad_db = _mismatches(la.bias.grd, lf.bias.grd, L.B_SIZE)

    print(
        "  Linear[", IN, ", ", OUT, ", bf16] B=", B,
        "  [1] grad_input mismatches ", bad,
        "  [2] empty ", still_empty,
        "  [3] dW/db after unfreeze ", bad_dw, "/", bad_db,
        "  [4] max|dW| ", dw_mag,
        sep="",
    )
    assert_true(bad == 0, "frozen bf16 grad_input differs from the trainable one")
    assert_true(still_empty, "the frozen bf16 vjp allocated a gradient")
    assert_true(bad_dw == 0 and bad_db == 0, "bf16 unfreeze is not a round trip")
    assert_true(dw_mag > 0.0, "bf16 dW is zero: [3] compared nothing")


def main() raises:
    with DeviceContext() as ctx:
        print("Linear.freeze(), GPU")
        check[6, 101, 33](ctx)
        check[128, 128, 64](ctx)
        check[200, 256, 40](ctx)
        check[960, 320, 140](ctx)
        check_bf16[128, 256, 64](ctx)
        print("PASSED")
