"""Conv2DTranspose's GPU bias gradient (two-pass) against the CPU path.

Past `_db_two_pass` (target-dependent) NCHW sums each (sample, channel) plane,
then column-sums the [B, OC] row sums over chunks of 1024 rows, and NHWC
column-sums [B·OH·OW, OC] directly; below it one block per channel. The
two-pass cases exercise every edge: 2,500 samples (three chunks, the last
partial), a tiny OC = 3 and an OC = 100 (not a multiple of the 32-wide column
block), and a 12x12 output plane (144 > the 128 threads of a row sum); the
16-sample cases take the one-block route (on Metal the NCHW one stays
two-pass). Two vjps without a zero_grad between them: the gradient must
ACCUMULATE, as the CPU path does. Compared in std units of the CPU value; dx
is checked too, within TF32's reach on NVIDIA (its GEMM rounds operands to a
10-bit mantissa: 1.5e-3 std units measured on a 5090).

    pixi run -e apple mojo run -I . tests/nn/test_conv2d_transpose_db_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.sys import default_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.conv2d_transpose import Conv2DTranspose
from noeira.nn.primitives.conv2d import _db_two_pass

comptime TOL = 1e-4
comptime TOL_DX = 5e-3 if default_accelerator().is_nvidia_gpu() else 1e-4


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


def run[
    B: Int, IC: Int, OC: Int, H: Int, LAYOUT: Int
](c: DeviceContext) raises:
    comptime CT = Conv2DTranspose[IC, OC, 4, 2, 1, H, H, 0, LAYOUT]
    comptime IN = CT.IN_FLAT
    comptime OUT = CT.OUT_FLAT
    print(
        "  B", B, "OC", OC, "plane", CT.OHt, "x", CT.OWt,
        "NHWC" if LAYOUT == LAYOUT_NHWC else "NCHW",
        "(two-pass)" if _db_two_pass[LAYOUT, B, OC, CT.SBIG]() else "(one block per channel)",
    )
    var cpu = CT.make["cpu", Deterministic](None)
    var gpu = CT.make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(B * IN)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](random_float64(-1, 1))
    var gx = Tensor.alloc(B * IN)
    var ggo = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        gx.data[i] = x.data[i]
    for i in range(B * OUT):
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
    gpu.bias.grd.download(c)
    g_gi.download(c)

    var e_db = _err(cpu.bias.grd.data, gpu.bias.grd.data, OC)
    var e_dx = _err(c_gi.data, g_gi.data, B * IN)
    print("    dbias", e_db, " dx", e_dx, "(std units, tol", TOL, "/ dx", TOL_DX, ")")
    # Vacuity: the accumulated (2-call) gradient is not the 1-call one.
    var half = List[Scalar[DT]]()
    for k in range(OC):
        half.append(cpu.bias.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.bias.grd.data, half, OC)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_db < TOL, "GPU bias gradient off the CPU path")
    assert_true(e_dx < TOL_DX, "GPU dx off the CPU path")


def main() raises:
    print("Conv2DTranspose bias gradient, GPU (two-pass) vs CPU")
    seed(11)
    var c = DeviceContext()
    run[2500, 2, 3, 6, LAYOUT_NCHW](c)
    run[2500, 2, 100, 2, LAYOUT_NCHW](c)
    run[2500, 2, 3, 6, LAYOUT_NHWC](c)
    run[16, 2, 3, 2, LAYOUT_NCHW](c)
    run[16, 2, 3, 2, LAYOUT_NHWC](c)
    print("PASS")
