"""BatchNorm1D: the eval-mode backward, and the stale-cache bug it closed.

In eval mode the forward is affine, y = γ·(x − μ_run)/√(σ²_run + ε) + β, so
dx = dy·γ·inv, dγ = Σ_b dy·x̂, dβ = Σ_b dy. Checked against a float64
reference written here, CPU and GPU.

The bug: `cache_is_training` was set by a training forward and never
cleared, so a TRAINING forward, then an EVAL forward, then a vjp silently
returned the training forward's gradient. Exactly that sequence must now give
the eval gradient. (Test-time adaptation runs this sequence: BN frozen in eval
while the rest adapts.)

    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_1d_eval_vjp.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.batch_norm_1d import BatchNorm1D, BN_DEFAULT_MOM, BN_DEFAULT_EPS


comptime B = 8
comptime D = 37
comptime BN = BatchNorm1D[D, BN_DEFAULT_MOM, BN_DEFAULT_EPS, DT, True]


def _run[target: StaticString](
    x: List[Scalar[DT]], x_train: List[Scalar[DT]], g: List[Scalar[DT]],
    gamma: List[Scalar[DT]], beta: List[Scalar[DT]],
    rm: List[Scalar[DT]], rv: List[Scalar[DT]],
    ctx: Optional[DeviceContext],
) raises -> Int:
    var bn = BN.make[target, Kaiming](ctx)
    for f in range(D):
        bn.gamma.val.data[f] = gamma[f]
        bn.beta.val.data[f] = beta[f]
    var fails = 0
    # a training forward first (fills the training caches), whose running-stat
    # update we then overwrite with the reference's
    var xt = Tensor.alloc(B * D)
    var xe = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * D)
    for i in range(B * D):
        xt.data[i] = x_train[i]
        xe.data[i] = x[i]
        go.data[i] = g[i]
    comptime if target == "gpu":
        var c = ctx.value()
        bn.gamma.val.upload(c)
        bn.beta.val.upload(c)
        xt.upload(c)
        xe.upload(c)
        go.upload(c)
    var y = Tensor()
    bn.set_attr["training"](Scalar[DT](1))
    bn.forward[target, B](TensorRefs[1](xt), y, ctx)
    for f in range(D):
        bn.running_mean.t.data[f] = rm[f]
        bn.running_var.t.data[f] = rv[f]
    comptime if target == "gpu":
        bn.running_mean.t.upload(ctx.value())
        bn.running_var.t.upload(ctx.value())
    bn.zero_grad[target](ctx)
    # then EVAL forward + vjp
    bn.set_attr["training"](Scalar[DT](0))
    bn.forward[target, B](TensorRefs[1](xe), y, ctx)
    var gi = Tensor()
    bn.vjp[target, B](TensorRefs[1](xe), go, TensorRefs[1](gi), ctx)
    comptime if target == "gpu":
        var c = ctx.value()
        c.synchronize()
        y.download(c)
        gi.download(c)
        bn.gamma.grd.download(c)
        bn.beta.grd.download(c)
    var e_y = 0.0
    var e_x = 0.0
    var e_g = 0.0
    var e_b = 0.0
    for f in range(D):
        var inv = 1.0 / sqrt(Float64(rv[f]) + Float64(BN_DEFAULT_EPS))
        var dg = 0.0
        var db = 0.0
        for b in range(B):
            var i = b * D + f
            var xh = (Float64(x[i]) - Float64(rm[f])) * inv
            e_y = max(e_y, abs(Float64(y.data[i]) - (Float64(gamma[f]) * xh + Float64(beta[f]))))
            e_x = max(e_x, abs(Float64(gi.data[i]) - Float64(g[i]) * Float64(gamma[f]) * inv))
            dg += Float64(g[i]) * xh
            db += Float64(g[i])
        e_g = max(e_g, abs(Float64(bn.gamma.grd.data[f]) - dg))
        e_b = max(e_b, abs(Float64(bn.beta.grd.data[f]) - db))
    var bad = e_y > 1e-5 or e_x > 1e-5 or e_g > 1e-4 or e_b > 1e-4
    print("  --", target, ": train fwd, eval fwd, vjp — max |err| y", e_y, " dx", e_x,
          " dgamma", e_g, " dbeta", e_b, " ✗" if bad else "")
    if bad:
        fails += 1
    return fails


def main() raises:
    seed(13)
    var x = List[Scalar[DT]]()
    var xt = List[Scalar[DT]]()
    var g = List[Scalar[DT]]()
    for _ in range(B * D):
        x.append(Scalar[DT](random_float64(-2, 2)))
        xt.append(Scalar[DT](random_float64(-5, 5)))  # a different batch
        g.append(Scalar[DT](random_float64(-1, 1)))
    var gamma = List[Scalar[DT]]()
    var beta = List[Scalar[DT]]()
    var rm = List[Scalar[DT]]()
    var rv = List[Scalar[DT]]()
    for _ in range(D):
        gamma.append(Scalar[DT](random_float64(0.5, 1.5)))
        beta.append(Scalar[DT](random_float64(-0.5, 0.5)))
        rm.append(Scalar[DT](random_float64(-1, 1)))
        rv.append(Scalar[DT](random_float64(0.2, 3.0)))
    print("BatchNorm1D eval-mode vjp (after a training forward)")
    var fails = _run["cpu"](x, xt, g, gamma, beta, rm, rv, None)
    var c = DeviceContext()
    fails += _run["gpu"](x, xt, g, gamma, beta, rm, rv, Optional(c))
    if fails > 0:
        raise Error("FAIL: " + String(fails))
    print("PASS")
