"""BatchNorm1D's GPU train-mode forward statistics against the CPU path, on
both of its GPU routes, and against a Float64 reference where they could
cancel.

At B·DIM >= `BN_TWO_PASS_MIN_ELEMS` (2^18) the batch mean / variance come from
`enqueue_col_mean_var` (a centred two-pass inside each 1024-row chunk, the
chunks merged in order) and a separate normalize pass; below it the fused
one-block-per-feature kernel stays. 2,700 x 200 takes the two-pass route and
exercises its edges (three chunks, the last partial; 200 features, not a
multiple of the 32-wide column block); 16 x 100 takes the fused one. Checked:
the output, the running mean / variance after one forward, and a vjp after
it (dx and dγ / dβ read the cached x̂ and inv_std), in std units of the CPU
value.

Stability: x = 1000 + U(-1, 1) at 2,700 rows, compared with Float64 batch
statistics computed here from the same fp32 inputs. The GPU's batch mean must
be within 3e-4 of a std and its variance within 2e-6 relative (the old fused
kernel: 4.6e-4 and 2.2e-7; the two-pass route without its shift leaked 1.2e-5
into the variance; Σx² - B·mean² per chunk, unshifted, failed the output at 2.1 std
units), and its output within 1e-3 std units of the Float64 output (the CPU
module's own error is printed for scale).

    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_1d_stats_gpu.mojo
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
    BN_DEFAULT_EPS,
    BN_TWO_PASS_MIN_ELEMS,
)

comptime TOL = 1e-4
comptime TOL_OFFSET = 1e-3


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


def run[B: Int, DIM: Int](c: DeviceContext, offset: Float64) raises:
    print(
        "  B", B, "DIM", DIM, "x offset", offset,
        "(two-pass)" if B * DIM >= BN_TWO_PASS_MIN_ELEMS else "(fused)",
    )
    var cpu = BatchNorm1D[DIM].make["cpu", Kaiming](None)
    var gpu = BatchNorm1D[DIM].make["gpu", Kaiming](Optional(c))
    var gam = List[Scalar[DT]]()
    var bet = List[Scalar[DT]]()
    for k in range(DIM):
        var g = Scalar[DT](random_float64(0.5, 1.5))
        var bb = Scalar[DT](random_float64(-0.5, 0.5))
        gam.append(g)
        bet.append(bb)
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
    var spread = 1.0 if offset != 0.0 else 2.0
    for i in range(B * DIM):
        # Per-feature scale so the variances differ across features.
        var f = i % DIM
        x.data[i] = Scalar[DT](
            offset + random_float64(-spread, spread) * (0.5 + Float64(f) / Float64(DIM))
        )
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

    var g_out = Tensor()
    var g_gi = Tensor()
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    c.synchronize()
    g_out.download(c)
    g_gi.download(c)
    gpu.running_mean.t.download(c)
    gpu.running_var.t.download(c)
    gpu.gamma.grd.download(c)
    gpu.beta.grd.download(c)

    var e_y = _err(c_out.data, g_out.data, B * DIM)
    var e_rm = _err(cpu.running_mean.t.data, gpu.running_mean.t.data, DIM)
    var e_rv = _err(cpu.running_var.t.data, gpu.running_var.t.data, DIM)
    var e_dx = _err(c_gi.data, g_gi.data, B * DIM)
    var e_dg = _err(cpu.gamma.grd.data, gpu.gamma.grd.data, DIM)
    var e_db = _err(cpu.beta.grd.data, gpu.beta.grd.data, DIM)
    print(
        "    vs CPU: y", e_y, " run_mean", e_rm, " run_var", e_rv,
        " dx", e_dx, " dgamma", e_dg, " dbeta", e_db,
    )

    if offset != 0.0:
        # Float64 batch statistics from the same fp32 inputs.
        var y64 = List[Scalar[DT]](length=B * DIM, fill=0)
        gpu.cache_mean.download(c)
        gpu.cache_var.download(c)
        var e_mean = 0.0
        var e_var = 0.0
        for f in range(DIM):
            var m = 0.0
            for b in range(B):
                m += Float64(x.data[b * DIM + f])
            m /= Float64(B)
            var v = 0.0
            for b in range(B):
                v += (Float64(x.data[b * DIM + f]) - m) ** 2
            v /= Float64(B)
            e_mean = max(e_mean, abs(Float64(gpu.cache_mean.data[f]) - m) / sqrt(v))
            e_var = max(e_var, abs(Float64(gpu.cache_var.data[f]) - v) / v)
            var istd = 1.0 / sqrt(v + BN_DEFAULT_EPS)
            for b in range(B):
                var xh = (Float64(x.data[b * DIM + f]) - m) * istd
                y64[b * DIM + f] = Scalar[DT](
                    Float64(gam[f]) * xh + Float64(bet[f])
                )
        var e_gpu = _err(y64, g_out.data, B * DIM)
        var e_cpu = _err(y64, c_out.data, B * DIM)
        print(
            "    vs Float64: GPU mean", e_mean, "std  var", e_var, "rel  y", e_gpu,
            " CPU y", e_cpu,
        )
        assert_true(e_mean < 3e-4, "GPU batch mean loses precision")
        assert_true(e_var < 2e-6, "GPU batch variance loses precision")
        assert_true(e_gpu < TOL_OFFSET, "GPU output loses precision")
    else:
        assert_true(e_y < TOL, "GPU output off the CPU path")
        assert_true(e_rm < TOL and e_rv < TOL, "GPU running stats off the CPU path")
        assert_true(e_dx < TOL, "GPU dx off the CPU path")
        assert_true(e_dg < TOL and e_db < TOL, "GPU dgamma / dbeta off the CPU path")


def main() raises:
    print("BatchNorm1D train-mode forward statistics, GPU vs CPU")
    seed(31)
    var c = DeviceContext()
    comptime assert 2700 * 200 >= BN_TWO_PASS_MIN_ELEMS, "must take the two-pass route"
    comptime assert 16 * 100 < BN_TWO_PASS_MIN_ELEMS, "must take the fused route"
    run[2700, 200](c, 0.0)
    run[16, 100](c, 0.0)
    run[2700, 200](c, 1000.0)
    run[16, 100](c, 1000.0)
    print("PASS")
