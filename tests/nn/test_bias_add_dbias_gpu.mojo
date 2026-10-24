"""BiasAdd's GPU bias gradient against the CPU path, fp32 and bf16 flow, on
both of its GPU routes.

At B·DIM >= `TWO_PASS_MIN_ELEMS` the column sum goes through the two-pass
`enqueue_bias_grad`; below it the one-launch block-per-column kernel stays.
2,500 x 100 takes the two-pass route and exercises its edges (three chunks,
the last partial; 100 columns, not a multiple of 32); 16 x 100 takes the old
one. The bf16 cases feed the CPU the same bf16-rounded grad_output, so the
sums differ by order only. Two vjps without a zero_grad between them: the
gradient must ACCUMULATE. Compared in std units of the CPU value; dx (a copy)
is checked too.

    pixi run -e apple mojo run -I . tests/nn/test_bias_add_dbias_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.bias_add import BiasAdd
from noeira.nn.primitives.linear import TWO_PASS_MIN_ELEMS

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


def run[B: Int, DIM: Int, ADT: DType](c: DeviceContext) raises:
    print(
        "  B", B, "DIM", DIM, "bf16" if ADT == BF16 else "fp32",
        "(two-pass)" if B * DIM >= TWO_PASS_MIN_ELEMS else "(one launch)",
    )
    var cpu = BiasAdd[DIM].make["cpu", Kaiming](None)
    var gpu = BiasAdd[DIM, ADT].make["gpu", Kaiming](Optional(c))
    var x = Tensor.alloc(B * DIM)
    var go = Tensor.alloc(B * DIM)
    var gx = TensorImpl[ADT].alloc(B * DIM)
    var ggo = TensorImpl[ADT].alloc(B * DIM)
    for i in range(B * DIM):
        var xv = Scalar[DT](random_float64(-1, 1)).cast[ADT]()
        var gv = Scalar[DT](random_float64(-1, 1)).cast[ADT]()
        gx.data[i] = xv
        ggo.data[i] = gv
        x.data[i] = xv.cast[DT]()
        go.data[i] = gv.cast[DT]()
    gx.upload(c)
    ggo.upload(c)

    var c_out = Tensor()
    var c_gi = Tensor()
    cpu.forward["cpu", B](TensorRefs[1](x), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)

    var g_out = TensorImpl[ADT]()
    var g_gi = TensorImpl[ADT]()
    gpu.forward["gpu", B](TensorRefs[1, _, ADT](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](
        TensorRefs[1, _, ADT](gx), ggo, TensorRefs[1, _, ADT](g_gi), Optional(c)
    )
    gpu.vjp["gpu", B](
        TensorRefs[1, _, ADT](gx), ggo, TensorRefs[1, _, ADT](g_gi), Optional(c)
    )
    c.synchronize()
    gpu.bias.grd.download(c)
    g_gi.download(c)
    var gi32 = List[Scalar[DT]]()
    for i in range(B * DIM):
        gi32.append(g_gi.data[i].cast[DT]())

    var e_db = _err(cpu.bias.grd.data, gpu.bias.grd.data, DIM)
    var e_dx = _err(c_gi.data, gi32, B * DIM)
    print("    dbias", e_db, " dx", e_dx, "(std units, tol", TOL, ")")
    var half = List[Scalar[DT]]()
    for k in range(DIM):
        half.append(cpu.bias.grd.data[k] * Scalar[DT](0.5))
    var e_half = _err(cpu.bias.grd.data, half, DIM)
    assert_true(e_half > 100 * TOL, "the gate cannot see a missed accumulation")
    assert_true(e_db < TOL, "GPU bias gradient off the CPU path")
    assert_true(e_dx < TOL, "GPU dx off the CPU path")


def main() raises:
    print("BiasAdd bias gradient, GPU vs CPU")
    seed(23)
    var c = DeviceContext()
    comptime assert 2500 * 100 >= TWO_PASS_MIN_ELEMS, "must take the two-pass route"
    comptime assert 16 * 100 < TWO_PASS_MIN_ELEMS, "must take the one-launch route"
    run[2500, 100, DT](c)
    run[16, 100, DT](c)
    run[2500, 100, BF16](c)
    run[16, 100, BF16](c)
    print("PASS")
