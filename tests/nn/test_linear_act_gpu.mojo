"""LinearAct's GPU forward and backward against its CPU path, at RL shapes.

On NVIDIA the fp32 forward GEMM of a shape MAX would need padded runs through
`cublas_gemm` (unpadded) + the fused bias/activation kernel, and the backward
is always two cuBLAS calls (dW += xᵀ·go with β = 1, dx = go·Wᵀ). On Apple
the padded / unpadded MAX paths run as before. Each shape runs the vjp TWICE
without zero_grad: the weight and bias gradients must accumulate (the vjp
gates grad_output in place, on both sides, so the second call sees the gated
grad on both). Compared in std units of the CPU value: y, dx, dW, dB.

Tanh, not ReLU: under TF32 a pre-activation near 0 can flip ReLU's gate, and
the difference is then the size of the gradient, not of the precision
(`_a_discontinuity_makes_per_element_parity_undecidable`). The GEMM paths are
the same for every activation.

Tolerance per backend: TF32 on CUDA (1e-2 std units), float32 on Metal (1e-4).

    pixi run -e apple mojo run -I . tests/nn/test_linear_act_gpu.mojo
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
from noeira.nn.primitives.linear_tanh import LinearTanh


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
    comptime L = LinearTanh[IN, OUT]
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
    var xg = Tensor.alloc(B * IN)
    var gog = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
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

    var ey = _err(yc.data, yg.data, B * OUT)
    var ex = _err(gic.data, gig.data, B * IN)
    var ew = _err(lc.weight.grd.data, lg.weight.grd.data, L.W_SIZE)
    var eb = _err(lc.bias.grd.data, lg.bias.grd.data, L.B_SIZE)
    # The band must be able to fail: half the true dW is far outside it.
    var half = List[Scalar[DT]]()
    for i in range(L.W_SIZE):
        half.append(lc.weight.grd.data[i] * Scalar[DT](0.5))
    var eh = _err(lc.weight.grd.data, half, L.W_SIZE)
    var ok = ey < TOL and ex < TOL and ew < TOL and eb < TOL and eh > 100 * TOL
    print(
        "  [", IN, "->", OUT, "] B=", B, " cublas_fwd=", L.use_cublas_fwd[B](),
        " | y ", ey, " dx ", ex, " dW ", ew, " dB ", eb,
        " (dW vs half: ", eh, ")", "" if ok else " ✗", sep="",
    )
    return ok


def main() raises:
    print("LinearTanh GPU vs CPU (std units, tol", TOL, "), forward + two vjps")
    var ctx = DeviceContext()
    var ok = True
    ok = check[8, 64, 64](ctx) and ok      # PPO LunarLander trunk, layer 1
    ok = check[64, 64, 64](ctx) and ok     # PPO trunk, layer 2
    ok = check[23, 256, 256](ctx) and ok   # SAC critic, layer 1 (obs | act)
    ok = check[256, 256, 256](ctx) and ok  # SAC trunk, layer 2 (MAX's gate passes)
    ok = check[17, 256, 1](ctx) and ok     # acting batch of one
    assert_true(ok, "LinearAct GPU off its CPU path")
    print("PASS")
