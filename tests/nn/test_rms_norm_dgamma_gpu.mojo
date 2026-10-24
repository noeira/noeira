"""RMSNorm's GPU γ gradient against the CPU path, on both of its GPU routes.

At B·DIM >= `TWO_PASS_MIN_ELEMS` dγ = Σ go·n goes through the two-pass
`enqueue_dot_grad` (chunks of 1024 rows, lanes along the features); below it
the one-launch block-per-column kernel stays. 2,500 x 100 takes the two-pass
route and exercises its edges (three chunks, the last partial; 100 features,
not a multiple of the 32-wide column block); 16 x 100 takes the old one. Two
vjps without a zero_grad between them: the gradient must ACCUMULATE, as the
CPU path does. Compared in std units of the CPU value; dx is checked too.

    pixi run -e apple mojo run -I . tests/nn/test_rms_norm_dgamma_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.linear import TWO_PASS_MIN_ELEMS

comptime TOL = 1e-4


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


def run[B: Int, DIM: Int](c: DeviceContext) raises:
    print(
        "  B", B, "DIM", DIM,
        "(two-pass)" if B * DIM >= TWO_PASS_MIN_ELEMS else "(one launch)",
    )
    var cpu = RMSNorm[DIM].make["cpu", Kaiming](None)
    var gpu = RMSNorm[DIM].make["gpu", Kaiming](Optional(c))
    for k in range(DIM):
        var g = Scalar[DT](random_float64(0.5, 1.5))
        cpu.gamma.val.data[k] = g
        gpu.gamma.val.data[k] = g
    gpu.gamma.val.upload(c)
    var x = Tensor.alloc(B * DIM)
    var go = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = Scalar[DT](random_float64(-2, 2) + 0.3)
        go.data[i] = Scalar[DT](random_float64(-1, 1))
    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        gx.data[i] = x.data[i]
        ggo.data[i] = go.data[i]
    gx.upload(c)
    ggo.upload(c)

    var c_out = Tensor()
    var c_gi = Tensor()
    cpu.forward["cpu", B](TensorRefs[1](x), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)

    var g_out = Tensor()
    var g_gi = Tensor()
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    c.synchronize()
    gpu.gamma.grd.download(c)
    g_gi.download(c)

    var e_dg = _err(cpu.gamma.grd.data, gpu.gamma.grd.data, DIM)
    var e_dx = _err(c_gi.data, g_gi.data, B * DIM)
    print("    dgamma", e_dg, " dx", e_dx, "(std units, tol", TOL, ")")
    var half = List[Scalar[DT]]()
    for k in range(DIM):
        half.append(cpu.gamma.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.gamma.grd.data, half, DIM)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_dg < TOL, "GPU dgamma off the CPU path")
    assert_true(e_dx < TOL, "GPU dx off the CPU path")


def main() raises:
    print("RMSNorm γ gradient, GPU vs CPU")
    seed(19)
    var c = DeviceContext()
    comptime assert 2500 * 100 >= TWO_PASS_MIN_ELEMS, "must take the two-pass route"
    comptime assert 16 * 100 < TWO_PASS_MIN_ELEMS, "must take the one-launch route"
    run[2500, 100](c)
    run[16, 100](c)
    print("PASS")
