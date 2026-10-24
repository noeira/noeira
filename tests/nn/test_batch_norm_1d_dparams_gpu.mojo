"""BatchNorm1D's GPU train-mode backward against the CPU path, on both of its
GPU routes.

At B·DIM >= `BN_TWO_PASS_MIN_ELEMS` the column sums Σ dy·x̂ and Σ dy go
through the two-pass `enqueue_dot_bias_grad` and dx is a separate
element-wise pass; below it the fused one-block-per-feature kernel stays.
2,700 x 100 takes the two-pass route and exercises its edges (three chunks,
the last partial; 100 features, not a multiple of the 32-wide column block);
16 x 100 takes the fused one. Two vjps without a zero_grad between them: the
γ / β gradients must ACCUMULATE, as the CPU path does. Compared in std units
of the CPU value; dx (which reads the sums) is checked too.

    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_1d_dparams_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.batch_norm_1d import (
    BatchNorm1D,
    BN_TWO_PASS_MIN_ELEMS,
)

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
        "(two-pass)" if B * DIM >= BN_TWO_PASS_MIN_ELEMS else "(fused)",
    )
    var cpu = BatchNorm1D[DIM].make["cpu", Kaiming](None)
    var gpu = BatchNorm1D[DIM].make["gpu", Kaiming](Optional(c))
    for k in range(DIM):
        var g = Scalar[DT](random_float64(0.5, 1.5))
        var bb = Scalar[DT](random_float64(-0.5, 0.5))
        cpu.gamma.val.data[k] = g
        gpu.gamma.val.data[k] = g
        cpu.beta.val.data[k] = bb
        gpu.beta.val.data[k] = bb
    gpu.gamma.val.upload(c)
    gpu.beta.val.upload(c)
    cpu.set_attr["training"](Scalar[DT](1.0))
    gpu.set_attr["training"](Scalar[DT](1.0))
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
    gpu.beta.grd.download(c)
    g_gi.download(c)

    var e_dg = _err(cpu.gamma.grd.data, gpu.gamma.grd.data, DIM)
    var e_db = _err(cpu.beta.grd.data, gpu.beta.grd.data, DIM)
    var e_dx = _err(c_gi.data, g_gi.data, B * DIM)
    print(
        "    dgamma", e_dg, " dbeta", e_db, " dx", e_dx,
        "(std units, tol", TOL, ")",
    )
    var half = List[Scalar[DT]]()
    for k in range(DIM):
        half.append(cpu.beta.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.beta.grd.data, half, DIM)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_dg < TOL and e_db < TOL, "GPU dgamma / dbeta off the CPU path")
    assert_true(e_dx < TOL, "GPU dx off the CPU path")


def main() raises:
    print("BatchNorm1D train-mode backward, GPU vs CPU")
    seed(29)
    var c = DeviceContext()
    comptime assert 2700 * 100 >= BN_TWO_PASS_MIN_ELEMS, "must take the two-pass route"
    comptime assert 16 * 100 < BN_TWO_PASS_MIN_ELEMS, "must take the fused route"
    run[2700, 100](c)
    run[16, 100](c)
    print("PASS")
