"""Conv2D's bf16-flow GPU bias gradient (two-pass) against the fp32 CPU path.

The bf16 path reads a bf16 grad_output and accumulates into the fp32 master
bias grad: past `_db_two_pass` (target-dependent) through `_enqueue_db` (NCHW:
per-(sample, channel) plane sums, then a chunked [B, OC] column sum; NHWC: a
chunked [B·OH·OW, OC] column sum), below it one block per channel. The
CPU reference runs the fp32 module on the SAME bf16-rounded grad_output, so
the two sums see identical inputs and differ by summation order only.

Gated at 2,500 samples (three chunks, the last partial), a tiny OC = 3 and an
OC = 100 (not a multiple of 32), and a 12x12 plane (144 > the 128 threads of a
row sum); the OC = 100 2x2 case is one-block on a 5090, the 16-sample NHWC one
on both. Two vjps without a zero_grad: the gradient must ACCUMULATE.
Compared in std units of the CPU value. Only the bias gradient is gated (the
bf16 dx and dW carry bf16 rounding of their own).

    pixi run -e apple mojo run -I . tests/nn/test_conv2d_bf16_db_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.conv2d import Conv2D, _db_two_pass

comptime BF16 = DType.bfloat16
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


def run[
    B: Int, IC: Int, OC: Int, K: Int, H: Int, LAYOUT: Int
](c: DeviceContext) raises:
    comptime P = K // 2
    comptime C32 = Conv2D[IC, OC, K, 1, P, H, H, DT, LAYOUT]
    comptime C16 = Conv2D[IC, OC, K, 1, P, H, H, BF16, LAYOUT]
    comptime IN = C32.IN_FLAT
    comptime OUT = C32.OUT_FLAT
    print(
        "  B", B, "OC", OC, "plane", H, "x", H,
        "NHWC" if LAYOUT == LAYOUT_NHWC else "NCHW",
        "(two-pass)" if _db_two_pass[LAYOUT, B, OC, H * H]() else "(one block per channel)",
    )
    var cpu = C32.make["cpu", Deterministic](None)
    var gpu = C16.make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(B * IN)
    var go = Tensor.alloc(B * OUT)
    var xb = TensorImpl[BF16].alloc(B * IN)
    var gob = TensorImpl[BF16].alloc(B * OUT)
    for i in range(B * IN):
        var v = Scalar[DT](random_float64(-1, 1)).cast[BF16]()
        xb.data[i] = v
        x.data[i] = v.cast[DT]()
    for i in range(B * OUT):
        var v = Scalar[DT](random_float64(-1, 1)).cast[BF16]()
        gob.data[i] = v
        go.data[i] = v.cast[DT]()
    xb.upload(c)
    gob.upload(c)

    var c_out = Tensor()
    var c_gi = Tensor()
    cpu.forward["cpu", B](TensorRefs[1](x), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)

    var g_out = TensorImpl[BF16]()
    var g_gi = TensorImpl[BF16]()
    gpu.forward["gpu", B](TensorRefs[1, _, BF16](xb), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](
        TensorRefs[1, _, BF16](xb), gob, TensorRefs[1, _, BF16](g_gi), Optional(c)
    )
    gpu.vjp["gpu", B](
        TensorRefs[1, _, BF16](xb), gob, TensorRefs[1, _, BF16](g_gi), Optional(c)
    )
    c.synchronize()
    gpu.bias.grd.download(c)

    var e_db = _err(cpu.bias.grd.data, gpu.bias.grd.data, OC)
    print("    dbias", e_db, "(std units, tol", TOL, ")")
    var half = List[Scalar[DT]]()
    for k in range(OC):
        half.append(cpu.bias.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.bias.grd.data, half, OC)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_db < TOL, "bf16 GPU bias gradient off the CPU path")


def main() raises:
    print("Conv2D bf16-flow bias gradient, GPU (two-pass) vs fp32 CPU")
    seed(13)
    var c = DeviceContext()
    run[2500, 2, 3, 3, 12, LAYOUT_NCHW](c)
    run[2500, 2, 100, 1, 2, LAYOUT_NCHW](c)
    run[2500, 2, 3, 3, 12, LAYOUT_NHWC](c)
    run[16, 2, 3, 3, 4, LAYOUT_NHWC](c)
    print("PASS")
