"""LinearReLU's GPU forward and backward against its CPU path, at RL shapes.

On NVIDIA, where `LinearAct.use_lt_relu` holds (printed per shape), the fp32
forward runs bias + ReLU in a cuBLASLt `RELU_AUX_BIAS` epilogue and caches
its BITMASK of z > 0; the backward gate reads the bits
(`_relu_mask_gate_kernel`). Elsewhere, and everywhere with `-D
NN_LT_RELU=0`, the GEMM + bias/activation kernel caching z.

ReLU gate parity is decidable only away from z = 0 (TF32 can flip a sign
there: `_a_discontinuity_makes_per_element_parity_undecidable`), so the
inputs are built to keep z away from 0: x[b, k] = s_b * U(0.5, 1) and
w[k, j] = c_j * U(0.5, 1) * 2 / IN (s_b, c_j = +-1), so
z[b, j] = s_b * c_j * (0.5 .. 2) + bias_j with |bias| = 0.3, every term of
the same size (a single dominant weight row inflates the TF32 error in std
units past the band). The sign pattern varies along BOTH rows and
columns, so a wrong bit order inside a byte (bit j^7), a wrong row stride
(OUT instead of the 128-rounded mask stride) or a transposed mask gates the
wrong elements and dx / dW / dB move by O(1) std units. Shapes cover OUT not
a multiple of 128 / 8 and OUT < 4 (mask larger than z).

Each shape runs the vjp TWICE without zero_grad (gradients accumulate; the
gate is applied in place both sides). Errors in std units of the CPU value;
tolerance TF32 on CUDA (1e-2), float32 on Metal (1e-4).

    pixi run -e default mojo run -I . tests/nn/test_linear_relu_gpu.mojo
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
from noeira.nn.core.cublaslt_gemm import LT_RELU
from noeira.nn.primitives.linear_relu import LinearReLU


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


def _sign() -> Float64:
    return 1.0 if random_float64(0, 1) < 0.5 else -1.0


def check[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises -> Bool:
    comptime L = LinearReLU[IN, OUT]
    seed(IN * 7 + OUT + B)
    var lc = L.make["cpu", Kaiming]()
    var lg = L.make["gpu", Kaiming](ctx)
    var c = List[Float64]()
    for _ in range(OUT):
        c.append(_sign())
    lg.weight.val.ensure_host(ctx, L.W_SIZE)
    lg.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        var wv = Scalar[DT](
            c[i % OUT] * random_float64(0.5, 1.0) * 2.0 / Float64(IN)
        )
        lc.weight.val.data[i] = wv
        lg.weight.val.data[i] = wv
    for i in range(L.B_SIZE):
        var bv = Scalar[DT](0.3 * _sign())
        lc.bias.val.data[i] = bv
        lg.bias.val.data[i] = bv
    lg.weight.val.upload_resident(ctx)
    lg.bias.val.upload_resident(ctx)
    lg.weight.val.version += 1
    var x = Tensor.alloc(B * IN)
    var go = Tensor.alloc(B * OUT)
    var xg = Tensor.alloc(B * IN)
    var gog = Tensor.alloc(B * OUT)
    for b in range(B):
        var sb = _sign()
        for k in range(IN):
            x.data[b * IN + k] = Scalar[DT](sb * random_float64(0.5, 1.0))
    for i in range(B * IN):
        xg.data[i] = x.data[i]
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](random_float64(-1, 1))
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
    yg.download(ctx)
    gig.download(ctx)
    lg.weight.grd.download(ctx)
    lg.bias.grd.download(ctx)

    # The fixture must gate a real fraction both ways, or the mask is untested.
    var on = 0
    for i in range(B * OUT):
        if yc.data[i] > 0:
            on += 1
    var frac = Float64(on) / Float64(B * OUT)
    var ey = _err(yc.data, yg.data, B * OUT)
    var ex = _err(gic.data, gig.data, B * IN)
    var ew = _err(lc.weight.grd.data, lg.weight.grd.data, L.W_SIZE)
    var eb = _err(lc.bias.grd.data, lg.bias.grd.data, L.B_SIZE)
    var ok = (
        ey < TOL and ex < TOL and ew < TOL and eb < TOL
        and (frac > 0.2 or OUT * B < 8) and frac < 0.8
    )
    print(
        "  [", IN, "->", OUT, "] B=", B,
        " cublas_fwd=", L.use_cublas_fwd[B](), " lt_relu=", L.use_lt_relu[B](),
        " on=", frac, " | y ", ey, " dx ", ex, " dW ", ew, " dB ", eb,
        "" if ok else " ✗", sep="",
    )
    return ok


def main() raises:
    print(
        "LinearReLU GPU vs CPU (std units, tol", TOL, "), NN_LT_RELU =",
        LT_RELU, ", forward + two vjps",
    )
    var ctx = DeviceContext()
    var ok = True
    ok = check[17, 256, 256](ctx) and ok    # SAC HC actor, layer 1
    ok = check[23, 256, 256](ctx) and ok    # SAC critic, layer 1 (obs | act)
    ok = check[256, 256, 256](ctx) and ok   # SAC trunk, layer 2
    ok = check[256, 256, 32](ctx) and ok    # acting batch
    ok = check[17, 256, 1](ctx) and ok      # acting batch of one
    ok = check[64, 100, 96](ctx) and ok     # OUT not a multiple of 8
    ok = check[64, 300, 64](ctx) and ok     # OUT past one 128-bit stride
    ok = check[64, 3, 64](ctx) and ok       # OUT < 4: mask larger than z
    ok = check[64, 64, 16384](ctx) and ok   # big batch (TF32, K 64: kernel)
    ok = check[1024, 256, 256](ctx) and ok  # TF32 with K >= 1024: epilogue
    assert_true(ok, "LinearReLU GPU off its CPU path")
    print("PASS")
