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

    pixi run -e apple mojo run -I . tests/nn/test_feed_forward_gelu.mojo
"""

from std.math import sqrt, abs, tanh, erf
from std.random import seed, random_float64
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.feed_forward_gelu import FeedForwardGELU


comptime TOL = 1e-2 if has_nvidia_gpu_accelerator() else 1e-4


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


def main() raises:
    print("FeedForwardGELU GPU vs CPU (std units, tol", TOL, "), two vjps")
    var ctx = DeviceContext()
    var ok = True
    ok = check[256, 384, 1536, 2](ctx) and ok   # GPT block
    ok = check[64, 192, 768, 4](ctx) and ok     # CIFAR ViT block
    ok = check[10, 64, 128, 3](ctx) and ok      # small / odd rows
    assert_true(ok, "FeedForwardGELU off its CPU path")
    print("PASS")
