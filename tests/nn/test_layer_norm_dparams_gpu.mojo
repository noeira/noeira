"""LayerNorm's GPU γ / β gradient (two-stage, chunked) against the CPU path.

The GPU reduction sums go·x̂ and go over chunks of 1024 rows (lanes along the
features, coalesced), then adds the chunks in order. Gated at a shape that
exercises every edge: 2,500 rows (three chunks, the last partial) x 100
features (not a multiple of the 32-wide column block). Two vjps without a
zero_grad between them: the gradients must ACCUMULATE (the second call adds
to the first), as the CPU path does. Compared in std units of the CPU value;
dx is checked too (unchanged kernel, same inputs).

    pixi run -e apple mojo run -I . tests/nn/test_layer_norm_dparams_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.layer_norm import LayerNorm


comptime DIM = 100
comptime B = 2500
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


def main() raises:
    print("LayerNorm γ/β gradient, GPU (chunked) vs CPU,", B, "x", DIM)
    seed(3)
    var c = DeviceContext()
    var cpu = LayerNorm[DIM].make["cpu", Kaiming](None)
    var gpu = LayerNorm[DIM].make["gpu", Kaiming](Optional(c))
    for k in range(DIM):
        var g = Scalar[DT](random_float64(0.5, 1.5))
        var bb = Scalar[DT](random_float64(-0.5, 0.5))
        cpu.gamma.val.data[k] = g
        gpu.gamma.val.data[k] = g
        cpu.beta.val.data[k] = bb
        gpu.beta.val.data[k] = bb
    gpu.gamma.val.upload(c)
    gpu.beta.val.upload(c)
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

    var e_dx = _err(c_gi.data, g_gi.data, B * DIM)
    var e_dg = _err(cpu.gamma.grd.data, gpu.gamma.grd.data, DIM)
    var e_db = _err(cpu.beta.grd.data, gpu.beta.grd.data, DIM)
    print("  dx", e_dx, " dgamma", e_dg, " dbeta", e_db, "(std units, tol", TOL, ")")
    # Vacuity: the accumulated (2-call) gradient is not the 1-call one.
    var half = List[Scalar[DT]]()
    for k in range(DIM):
        half.append(cpu.beta.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.beta.grd.data, half, DIM)
    print("  vacuity: dbeta vs half of it", e_half)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_dx < TOL and e_dg < TOL and e_db < TOL, "GPU off the CPU path")
    print("PASS")
