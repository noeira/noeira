"""ConvRMSNorm's GPU γ gradient (two-pass) against the CPU path.

dγ[c] = Σ_{b, p} go·n. Past `CHANNEL_TWO_PASS_MIN_ROWS` rows per channel
(target-dependent): per-(sample, channel) sums over the contiguous HW plane,
then a column sum of the [B, C] row sums over chunks of 1024 rows; below it
one block per channel (the 16 x 16 case, on a 5090).
Gated at 2,500 samples (three chunks, the last partial), a tiny C = 3 and a
C = 100 (not a multiple of the 32-wide column block), and HW = 144 (> the 128
threads of a row sum). Two vjps without a zero_grad between them: the
gradient must ACCUMULATE, as the CPU path does. Compared in std units of the
CPU value; dx is checked too (unchanged kernel, same inputs).

    pixi run -e apple mojo run -I . tests/nn/test_conv_rms_norm_dgamma_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.conv_rms_norm import ConvRMSNorm
from noeira.nn.primitives.linear import CHANNEL_TWO_PASS_MIN_ROWS

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


def run[B: Int, C: Int, HW: Int](c: DeviceContext) raises:
    comptime N = B * C * HW
    print(
        "  B", B, "C", C, "HW", HW,
        "(two-pass)" if B * HW >= CHANNEL_TWO_PASS_MIN_ROWS else "(one block per channel)",
    )
    var cpu = ConvRMSNorm[C, HW].make["cpu", Deterministic](None)
    var gpu = ConvRMSNorm[C, HW].make["gpu", Deterministic](Optional(c))
    for k in range(C):
        var g = Scalar[DT](random_float64(0.5, 1.5))
        cpu.gamma.val.data[k] = g
        gpu.gamma.val.data[k] = g
    gpu.gamma.val.upload(c)
    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT](random_float64(-2, 2) + 0.3)
        go.data[i] = Scalar[DT](random_float64(-1, 1))
    var gx = Tensor.alloc(N)
    var ggo = Tensor.alloc(N)
    for i in range(N):
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

    var e_dg = _err(cpu.gamma.grd.data, gpu.gamma.grd.data, C)
    var e_dx = _err(c_gi.data, g_gi.data, N)
    print("    dgamma", e_dg, " dx", e_dx, "(std units, tol", TOL, ")")
    var half = List[Scalar[DT]]()
    for k in range(C):
        half.append(cpu.gamma.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.gamma.grd.data, half, C)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_dg < TOL, "GPU dgamma off the CPU path")
    assert_true(e_dx < TOL, "GPU dx off the CPU path")


def main() raises:
    print("ConvRMSNorm γ gradient, GPU (two-pass) vs CPU")
    seed(17)
    var c = DeviceContext()
    run[2500, 3, 144](c)
    run[2500, 100, 16](c)
    run[16, 3, 16](c)
    print("PASS")
