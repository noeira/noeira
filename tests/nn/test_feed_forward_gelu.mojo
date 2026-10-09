"""FeedForwardGELU on the GPU against its CPU path (= the Sequential
`Tok[Linear] -> GELUTanh -> Tok[Linear]` it replaces).

On NVIDIA the forward stores z and GELU(z) from one GEMM's epilogue (MAX's
`elementwise_lambda_fn`) and the backward runs through cuBLAS; elsewhere it
runs its three children. Gates, per shape:

  - out, dx, dW1, db1, dW2, db2 in std units of the CPU value, after TWO
    vjps without zero_grad (gradients must accumulate). Tolerance per
    backend: TF32 GEMMs on CUDA (1e-2), float32 on Metal (1e-4);
  - the GELU FLAVOUR, which that band cannot see (tanh vs erf differ by
    ~4e-4): the module's own z and h are read back and h must equal the
    tanh-approximate GELU of z to float32 precision (the erf distance is
    printed for contrast).

  - NVIDIA only, the bf16-flow block (cuBLASLt: h = GELU(x·W1 + b1) and z
    from one GEMM, dz = (go·W2ᵀ) ⊙ GELU'(z) from another) against the fp32
    CPU block on the same weights and bf16-rounded inputs, band `TOL_BF16`
    (bf16 activations: ~0.4 % per rounding). The GELU flavour cannot be
    told apart at bf16 precision (the tanh / erf gap is under one bf16 ulp),
    so this arm does not gate it.

    pixi run -e apple mojo run -I . tests/nn/test_feed_forward_gelu.mojo
"""

from std.math import sqrt, abs, tanh, erf
from std.random import seed, random_float64
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.feed_forward_gelu import FeedForwardGELU


comptime TOL = 1e-2 if has_nvidia_gpu_accelerator() else 1e-4
comptime TOL_BF16 = 5e-2
comptime BF16 = DType.bfloat16


def _err(ref_: List[Scalar[DT]], got: List[Scalar[DT]], n: Int) -> Float64:
    var mean = 0.0
    for i in range(n):
        mean += Float64(ref_[i])
    mean /= Float64(n)
    var v = 0.0
    var w = 0.0
    for i in range(n):
        v += (Float64(ref_[i]) - mean) ** 2
        w = max(w, abs(Float64(got[i]) - Float64(ref_[i])))
    var sd = sqrt(v / Float64(n))
    return w / (sd if sd > 0.0 else 1.0)


def _gelu_tanh(z: Float64) -> Float64:
    return 0.5 * z * (1.0 + tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))


def _gelu_erf(z: Float64) -> Float64:
    return 0.5 * z * (1.0 + erf(z * 0.7071067811865476))


def check[S: Int, D: Int, FF: Int, B: Int](ctx: DeviceContext) raises -> Bool:
    comptime M = FeedForwardGELU[S, D, FF]
    comptime R = B * S
    seed(S * 31 + D)
    var mc = M.make["cpu", Kaiming](None)
    var mg = M.make["gpu", Kaiming](Optional(ctx))
    # Same weights: copy the CPU module's into the GPU one.
    for k in range(D * FF):
        mg.fc1.weight.val.data[k] = mc.fc1.weight.val.data[k]
        mg.fc2.weight.val.data[k] = mc.fc2.weight.val.data[k]
    for k in range(FF):
        var b = Scalar[DT](random_float64(-0.2, 0.2))
        mc.fc1.bias.val.data[k] = b
        mg.fc1.bias.val.data[k] = b
    for k in range(D):
        var b = Scalar[DT](random_float64(-0.2, 0.2))
        mc.fc2.bias.val.data[k] = b
        mg.fc2.bias.val.data[k] = b
    mg.fc1.weight.val.upload_resident(ctx)
    mg.fc2.weight.val.upload_resident(ctx)
    mg.fc1.bias.val.upload_resident(ctx)
    mg.fc2.bias.val.upload_resident(ctx)
    mg.fc1.weight.val.version += 1
    mg.fc2.weight.val.version += 1

    var x = Tensor.alloc(R * D)
    var go = Tensor.alloc(R * D)
    for i in range(R * D):
        x.data[i] = Scalar[DT](random_float64(-1.5, 1.5))
        go.data[i] = Scalar[DT](random_float64(-1, 1))
    var xg = Tensor.alloc(R * D)
    var gog = Tensor.alloc(R * D)
    for i in range(R * D):
        xg.data[i] = x.data[i]
        gog.data[i] = go.data[i]
    xg.upload(ctx)
    gog.upload(ctx)

    var yc = Tensor()
    var gic = Tensor()
    mc.forward["cpu", B](TensorRefs[1](x), yc, None)
    mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)
    mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)

    var yg = Tensor()
    var gig = Tensor()
    mg.zero_grad["gpu"](Optional(ctx))
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), Optional(ctx))
    ctx.synchronize()
    yg.download(ctx)
    gig.download(ctx)
    mg.fc1.weight.grd.download(ctx)
    mg.fc1.bias.grd.download(ctx)
    mg.fc2.weight.grd.download(ctx)
    mg.fc2.bias.grd.download(ctx)
    mg.z.download(ctx)
    mg.h.download(ctx)

    var e = List[Float64]()
    e.append(_err(yc.data, yg.data, R * D))
    e.append(_err(gic.data, gig.data, R * D))
    e.append(_err(mc.fc1.weight.grd.data, mg.fc1.weight.grd.data, D * FF))
    e.append(_err(mc.fc1.bias.grd.data, mg.fc1.bias.grd.data, FF))
    e.append(_err(mc.fc2.weight.grd.data, mg.fc2.weight.grd.data, D * FF))
    e.append(_err(mc.fc2.bias.grd.data, mg.fc2.bias.grd.data, D))
    var worst = 0.0
    for v in e:
        worst = max(worst, v)
    # GELU flavour on the GPU's own z.
    var d_tanh = 0.0
    var d_erf = 0.0
    for i in range(R * FF):
        var zz = Float64(mg.z.data[i])
        var hh = Float64(mg.h.data[i])
        d_tanh = max(d_tanh, abs(hh - _gelu_tanh(zz)))
        d_erf = max(d_erf, abs(hh - _gelu_erf(zz)))
    var ok = worst < TOL and d_tanh < 1e-5
    print(
        "  S=", S, " D=", D, " FF=", FF, " B=", B,
        " | out ", e[0], " dx ", e[1], " dW1 ", e[2], " db1 ", e[3],
        " dW2 ", e[4], " db2 ", e[5],
        " | h-geluTanh(z) ", d_tanh, " h-geluErf(z) ", d_erf,
        "" if ok else " ✗", sep="",
    )
    return ok


def check_bf16[S: Int, D: Int, FF: Int, B: Int](ctx: DeviceContext) raises -> Bool:
    comptime M = FeedForwardGELU[S, D, FF]
    comptime MB = FeedForwardGELU[S, D, FF, BF16]
    comptime R = B * S
    seed(S * 37 + D)
    var mc = M.make["cpu", Kaiming](None)
    var mg = MB.make["gpu", Kaiming](Optional(ctx))
    for k in range(D * FF):
        mg.fc1.weight.val.data[k] = mc.fc1.weight.val.data[k]
        mg.fc2.weight.val.data[k] = mc.fc2.weight.val.data[k]
    for k in range(FF):
        var b = Scalar[DT](random_float64(-0.2, 0.2))
        mc.fc1.bias.val.data[k] = b
        mg.fc1.bias.val.data[k] = b
    for k in range(D):
        var b = Scalar[DT](random_float64(-0.2, 0.2))
        mc.fc2.bias.val.data[k] = b
        mg.fc2.bias.val.data[k] = b
    mg.fc1.weight.val.upload_resident(ctx)
    mg.fc2.weight.val.upload_resident(ctx)
    mg.fc1.bias.val.upload_resident(ctx)
    mg.fc2.bias.val.upload_resident(ctx)
    mg.fc1.weight.val.version += 1
    mg.fc2.weight.val.version += 1

    # bf16 inputs; the CPU reference gets the same (rounded) values.
    var x = Tensor.alloc(R * D)
    var go = Tensor.alloc(R * D)
    var xb = TensorImpl[BF16].alloc(R * D)
    var gob = TensorImpl[BF16].alloc(R * D)
    for i in range(R * D):
        xb.data[i] = Scalar[DT](random_float64(-1.5, 1.5)).cast[BF16]()
        gob.data[i] = Scalar[DT](random_float64(-1, 1)).cast[BF16]()
        x.data[i] = xb.data[i].cast[DT]()
        go.data[i] = gob.data[i].cast[DT]()
    xb.upload(ctx)
    gob.upload(ctx)

    var yc = Tensor()
    var gic = Tensor()
    mc.forward["cpu", B](TensorRefs[1](x), yc, None)
    mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)
    mc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)

    var yg = TensorImpl[BF16]()
    var gig = TensorImpl[BF16]()
    mg.zero_grad["gpu"](Optional(ctx))
    mg.forward["gpu", B](TensorRefs[1, ADT=BF16](xb), yg, Optional(ctx))
    for _ in range(2):
        mg.vjp["gpu", B](
            TensorRefs[1, ADT=BF16](xb), gob, TensorRefs[1, ADT=BF16](gig),
            Optional(ctx),
        )
    ctx.synchronize()
    yg.download(ctx)
    gig.download(ctx)
    mg.fc1.weight.grd.download(ctx)
    mg.fc1.bias.grd.download(ctx)
    mg.fc2.weight.grd.download(ctx)
    mg.fc2.bias.grd.download(ctx)
    var y32 = List[Scalar[DT]]()
    var gi32 = List[Scalar[DT]]()
    for i in range(R * D):
        y32.append(yg.data[i].cast[DT]())
        gi32.append(gig.data[i].cast[DT]())

    var e = List[Float64]()
    e.append(_err(yc.data, y32, R * D))
    e.append(_err(gic.data, gi32, R * D))
    e.append(_err(mc.fc1.weight.grd.data, mg.fc1.weight.grd.data, D * FF))
    e.append(_err(mc.fc1.bias.grd.data, mg.fc1.bias.grd.data, FF))
    e.append(_err(mc.fc2.weight.grd.data, mg.fc2.weight.grd.data, D * FF))
    e.append(_err(mc.fc2.bias.grd.data, mg.fc2.bias.grd.data, D))
    var half = List[Scalar[DT]]()
    for k in range(D * FF):
        half.append(mc.fc1.weight.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(mc.fc1.weight.grd.data, half, D * FF)
    var worst = 0.0
    for v in e:
        worst = max(worst, v)
    var ok = worst < TOL_BF16 and e_half > 10 * TOL_BF16
    print(
        "  bf16 S=", S, " D=", D, " FF=", FF, " B=", B, " fused=", MB.LT_FUSED,
        " | out ", e[0], " dx ", e[1], " dW1 ", e[2], " db1 ", e[3],
        " dW2 ", e[4], " db2 ", e[5], " (half-dW1 control ", e_half, ")",
        "" if ok else " ✗", sep="",
    )
    return ok


def main() raises:
    print("FeedForwardGELU GPU vs CPU (std units, tol", TOL, "), two vjps")
    var ctx = DeviceContext()
    var ok = True
    ok = check[256, 384, 1536, 2](ctx) and ok   # GPT block
    ok = check[64, 192, 768, 4](ctx) and ok     # CIFAR ViT block
    ok = check[10, 64, 128, 3](ctx) and ok      # small / odd rows
    comptime if has_nvidia_gpu_accelerator():
        print("bf16-flow block vs the fp32 CPU block (std units, tol", TOL_BF16, ")")
        ok = check_bf16[256, 384, 1536, 2](ctx) and ok   # GPT block
        ok = check_bf16[64, 192, 768, 4](ctx) and ok     # CIFAR ViT block
    assert_true(ok, "FeedForwardGELU off its CPU path")
    print("PASS")
