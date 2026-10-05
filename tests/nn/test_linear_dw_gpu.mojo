"""Linear's GPU backward against its CPU path, at transformer shapes.

On NVIDIA an ALIGNED Linear's weight gradient is one cuBLAS call
(`cublas_dw_accumulate`: dw += xᵀ·go, β = 1 — no transposed copy, no
temporary, no accumulate kernel); a padded one keeps the transpose + GEMM +
strided accumulate; Apple keeps the transpose + GEMM everywhere. Each shape
runs the vjp TWICE without zero_grad: the weight and bias gradients must
accumulate. Compared in std units of the CPU value: dW, dB and dx.

Tolerance per backend: the CUDA GEMMs (MAX's and cuBLAS's) run TF32, ~1e-3
relative — a band written on Metal's float32 would fail them for no defect
(`test_linear_pad_parity` does exactly that); Metal is held to 1e-4.

    pixi run -e apple mojo run -I . tests/nn/test_linear_dw_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.linear import Linear


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


def check[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises -> Bool:
    comptime L = Linear[IN, OUT]
    seed(IN * 7 + OUT)
    var lc = L.make["cpu", Kaiming]()
    var lg = L.make["gpu", Kaiming](ctx)
    lg.weight.val.ensure_host(ctx, L.W_SIZE)
    lg.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lg.weight.val.data[i] = lc.weight.val.data[i]
    for i in range(L.B_SIZE):
        var bv = Scalar[DT](random_float64(-0.1, 0.1))
        lc.bias.val.data[i] = bv
        lg.bias.val.data[i] = bv
    lg.weight.val.upload_resident(ctx)
    lg.bias.val.upload_resident(ctx)
    lg.weight.val.version += 1
    var x = Tensor.alloc(B * IN)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](random_float64(-1, 1))
    var xg = Tensor.alloc(B * IN)
    var gog = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        xg.data[i] = x.data[i]
    for i in range(B * OUT):
        gog.data[i] = go.data[i]
    xg.upload(ctx)
    gog.upload(ctx)

    var yc = Tensor()
    var gic = Tensor()
    lc.forward["cpu", B](TensorRefs[1](x), yc, None)
    lc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)
    lc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)

    var yg = Tensor()
    var gig = Tensor()
    lg.zero_grad["gpu"](Optional(ctx))
    lg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    lg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), Optional(ctx))
    lg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), Optional(ctx))
    ctx.synchronize()
    lg.weight.grd.download(ctx)
    lg.bias.grd.download(ctx)
    gig.download(ctx)

    var ew = _err(lc.weight.grd.data, lg.weight.grd.data, L.W_SIZE)
    var eb = _err(lc.bias.grd.data, lg.bias.grd.data, L.B_SIZE)
    var ex = _err(gic.data, gig.data, B * IN)
    var half = List[Scalar[DT]]()
    for i in range(L.W_SIZE):
        half.append(lc.weight.grd.data[i] * Scalar[DT](0.5))
    var eh = _err(lc.weight.grd.data, half, L.W_SIZE)
    var mag = 0.0
    for i in range(B * IN):
        mag = max(mag, abs(Float64(gig.data[i])))
    var ok = ew < TOL and eb < TOL and ex < TOL and eh > 100 * TOL and mag > 0.0
    print(
        "  [", IN, "->", OUT, "] B=", B, " pad=", L.NEEDS_PAD or L.NEEDS_N_PAD,
        " dW ", ew, " dB ", eb, " dx ", ex, " (dW vs half: ", eh, ", |dx|max ", mag, ")",
        "" if ok else " ✗", sep="",
    )
    return ok


def main() raises:
    print("Linear GPU backward vs CPU (std units, tol", TOL, "), two vjps")
    var ctx = DeviceContext()
    var ok = True
    ok = check[384, 1152, 1024](ctx) and ok   # GPT qkv projection
    ok = check[1536, 384, 512](ctx) and ok    # GPT fc2
    ok = check[192, 768, 640](ctx) and ok     # ViT fc1
    ok = check[100, 64, 96](ctx) and ok       # padded: keeps the transpose path
    assert_true(ok, "GPU backward off the CPU path")
    print("PASS")
