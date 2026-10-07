"""Conv2D's GPU forward and backward against its CPU path, at the conv shapes
noeira's agents and nn examples run.

On NVIDIA the three GEMMs (forward, dW, d_col) run on `cublas_gemm` with no
padding (`Conv2D.CUB`); Apple, and `-D NN_GEMM_PATH=max`, keep MAX's padded
GEMMs. Each shape runs the vjp TWICE without zero_grad: the weight and bias
gradients must accumulate, and the second vjp must not reuse the forward's
im2col after the first one overwrote it with d_colᵀ (it did before the A2
invalidation: dW 0.5-0.9x the CPU's). Compared in std units of the CPU value: y, dx,
dW, dB, with a half-dW control that must fail the band.

Tolerance per backend: TF32 GEMMs on CUDA (1e-2 std units), float32 on Metal
(1e-4).

    pixi run -e apple mojo run -I . tests/nn/test_conv2d_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.conv2d import Conv2D


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


def check[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, B: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](name: String, ctx: DeviceContext) raises -> Bool:
    comptime Cv = Conv2D[IC, OC, K, S, P, H, W, DT, LAYOUT]
    comptime IN_N = B * Cv.IN_FLAT
    comptime OUT_N = B * Cv.OUT_FLAT
    var octx = Optional[DeviceContext](ctx)
    seed(IC * 131 + OC * 7 + K)
    var cc = Cv.make["cpu", Kaiming](None)
    seed(IC * 131 + OC * 7 + K)
    var cg = Cv.make["gpu", Kaiming](octx)
    var x = Tensor.alloc(IN_N)
    var xg = Tensor.alloc(IN_N)
    for i in range(IN_N):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
        xg.data[i] = x.data[i]
    xg.upload(ctx)
    var go = Tensor.alloc(OUT_N)
    var gog = Tensor.alloc(OUT_N)
    for i in range(OUT_N):
        go.data[i] = Scalar[DT](random_float64(-1, 1))
        gog.data[i] = go.data[i]
    gog.upload(ctx)

    var yc = Tensor()
    var gic = Tensor()
    cc.forward["cpu", B](TensorRefs[1](x), yc, None)
    cc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)
    cc.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gic), None)

    var yg = Tensor()
    var gig = Tensor()
    cg.zero_grad["gpu"](octx)
    cg.forward["gpu", B](TensorRefs[1](xg), yg, octx)
    cg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), octx)
    cg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), octx)
    ctx.synchronize()
    yg.download(ctx)
    gig.download(ctx)
    cg.weight.grd.download(ctx)
    cg.bias.grd.download(ctx)

    var ey = _err(yc.data, yg.data, OUT_N)
    var ex = _err(gic.data, gig.data, IN_N)
    var ew = _err(cc.weight.grd.data, cg.weight.grd.data, Cv.W_SIZE)
    var eb = _err(cc.bias.grd.data, cg.bias.grd.data, OC)
    var half = List[Scalar[DT]]()
    for i in range(Cv.W_SIZE):
        half.append(cc.weight.grd.data[i] * Scalar[DT](0.5))
    var eh = _err(cc.weight.grd.data, half, Cv.W_SIZE)
    var ok = ey < TOL and ex < TOL and ew < TOL and eb < TOL and eh > 100 * TOL
    print(
        "  ", name, " [", IC, "->", OC, " k", K, " s", S, " ", H, "x", W, "] B=", B,
        " cublas=", Cv.CUB, " cudnn=", Cv.use_cudnn[B](), " | y ", ey, " dx ", ex, " dW ", ew, " dB ", eb,
        " (dW vs half: ", eh, ")", "" if ok else " ✗", sep="",
    )
    return ok


def check_polyak[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, B: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](name: String, ctx: DeviceContext) raises -> Bool:
    """A target net's sync: `tgt.polyak_from(src, tau=1)` after BOTH ran a
    forward (so any cached weight copy is warm) must make the next `tgt`
    forward equal `src`'s. `polyak_tensor` does not bump `val.version`, so a
    cache gated on it (the padded MAX path's `w_pad`) served the pre-sync
    weight: a target conv net frozen at its init."""
    comptime Cv = Conv2D[IC, OC, K, S, P, H, W, DT, LAYOUT]
    comptime IN_N = B * Cv.IN_FLAT
    comptime OUT_N = B * Cv.OUT_FLAT
    var octx = Optional[DeviceContext](ctx)
    seed(11)
    var src = Cv.make["gpu", Kaiming](octx)
    seed(12)
    var tgt = Cv.make["gpu", Kaiming](octx)
    var x = Tensor.alloc(IN_N)
    for i in range(IN_N):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
    x.upload(ctx)
    var ys = Tensor()
    var yt = Tensor()
    src.forward["gpu", B](TensorRefs[1](x), ys, octx)
    tgt.forward["gpu", B](TensorRefs[1](x), yt, octx)
    tgt.polyak_from["gpu"](src, Scalar[DT](1.0), octx)
    tgt.forward["gpu", B](TensorRefs[1](x), yt, octx)
    ctx.synchronize()
    ys.download(ctx)
    yt.download(ctx)
    var e = _err(ys.data, yt.data, OUT_N)
    var ok = e < 1e-6
    print("  polyak ", name, ": target vs source after tau=1 sync ", e, "" if ok else " ✗ (stale weight cache)", sep="")
    return ok


def main() raises:
    print("Conv2D GPU vs CPU (std units, tol", TOL, "), forward + two vjps")
    var ctx = DeviceContext()
    var ok = True
    ok = check[3, 16, 3, 1, 1, 32, 32, 8]("ResNet-20 stem", ctx) and ok
    ok = check[16, 32, 3, 2, 1, 32, 32, 8]("ResNet-20 downsample", ctx) and ok
    ok = check[64, 64, 3, 1, 1, 8, 8, 8]("ResNet-20 stage 3", ctx) and ok
    ok = check[3, 32, 3, 1, 1, 32, 32, 4]("CNN CIFAR conv1", ctx) and ok
    ok = check[4, 32, 8, 4, 0, 84, 84, 4]("Atari stem", ctx) and ok
    ok = check[32, 64, 4, 2, 0, 20, 20, 4]("Atari mid", ctx) and ok
    ok = check[64, 64, 3, 1, 1, 6, 7, 16]("MuZero C4 res", ctx) and ok
    ok = check[64, 16, 1, 1, 0, 6, 7, 16]("1x1 bottleneck", ctx) and ok
    ok = check[16, 16, 3, 1, 1, 16, 16, 4, LAYOUT_NHWC]("NHWC 16->16", ctx) and ok
    ok = check[32, 1, 5, 1, 2, 16, 16, 4]("OC=1 decoder out", ctx) and ok
    ok = check[4, 32, 8, 4, 0, 84, 84, 4, LAYOUT_NHWC]("Nature c1 NHWC (Rainbow pixel)", ctx) and ok
    ok = check[32, 64, 4, 2, 0, 20, 20, 4, LAYOUT_NHWC]("Nature c2 NHWC", ctx) and ok
    ok = check[64, 64, 3, 1, 0, 9, 9, 4, LAYOUT_NHWC]("Nature c3 NHWC", ctx) and ok
    ok = check[16, 16, 3, 1, 1, 32, 32, 16]("ResNet s1 (cuDNN under auto)", ctx) and ok
    ok = check_polyak[4, 32, 8, 4, 0, 84, 84, 4, LAYOUT_NHWC]("Nature c1 NHWC", ctx) and ok
    ok = check_polyak[64, 64, 3, 1, 1, 6, 7, 16]("C4 res", ctx) and ok
    ok = check_polyak[16, 16, 3, 1, 1, 32, 32, 16]("ResNet s1", ctx) and ok
    assert_true(ok, "Conv2D GPU off its CPU path")
    print("PASS")
