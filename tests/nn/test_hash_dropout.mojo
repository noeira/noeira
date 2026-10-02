"""HashDropout and ScaledDotProductAttention's attention dropout.

HashDropout[DIM, P]:
  * CPU and GPU draw the SAME mask (bit-equal outputs);
  * the kept fraction is 1 - P (±4 sigma); kept values are x / (1 - P);
  * the vjp is grad · mask — the forward's mask, redrawn from its counter;
  * a second instance draws a different mask; a second forward too;
  * `dropout` off is the identity.

ScaledDotProductAttention[..., P_DROP]: against an INDEPENDENT float64
reference written out here (softmax, a·m/(1-p), A·V, and the analytic
backward for Q, K, V) using the module's seed and counter:
  * forward and input gradient, CPU and GPU, within 1e-4 std units;
  * with `dropout` off the module equals plain attention exactly.

    pixi run -e apple mojo run -I . tests/nn/test_hash_dropout.mojo
"""

from std.math import sqrt, abs, exp
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.random.hash_mask import hash_keep
from noeira.nn.primitives.hash_dropout import HashDropout
from noeira.nn.primitives.attention import ScaledDotProductAttention


comptime P = 0.1


def _std_err(ref_: List[Float64], got: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for v in ref_:
        m += v
    m /= Float64(len(ref_))
    var var_ = 0.0
    var w = 0.0
    for i in range(len(ref_)):
        var_ += (ref_[i] - m) ** 2
        w = max(w, abs(Float64(got[i]) - ref_[i]))
    return w / sqrt(var_ / Float64(len(ref_)))


def _hash_dropout(ctx: DeviceContext) raises -> Int:
    comptime B = 64
    comptime D = 512
    comptime N = B * D
    var fails = 0
    seed(3)
    var x = Tensor.alloc(N)
    var g = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT](random_float64(0.5, 2.0))
        g.data[i] = Scalar[DT](random_float64(0.5, 2.0))
    var mc = HashDropout[D, P].make["cpu", Kaiming](None)
    var mg = HashDropout[D, P].make["gpu", Kaiming](Optional(ctx))
    mg.seed = mc.seed  # the same mask stream
    var yc = Tensor()
    var gc = Tensor()
    mc.forward["cpu", B](TensorRefs[1](x), yc, None)
    mc.vjp["cpu", B](TensorRefs[1](x), g, TensorRefs[1](gc), None)
    var xg = Tensor.alloc(N)
    var gg = Tensor.alloc(N)
    for i in range(N):
        xg.data[i] = x.data[i]
        gg.data[i] = g.data[i]
    xg.upload(ctx)
    gg.upload(ctx)
    var yg = Tensor()
    var gig = Tensor()
    mg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gg, TensorRefs[1](gig), Optional(ctx))
    ctx.synchronize()
    yg.download(ctx)
    gig.download(ctx)
    var kept = 0
    var diff = 0
    var bad = 0
    for i in range(N):
        if yc.data[i] != yg.data[i] or gc.data[i] != gig.data[i]:
            diff += 1
        var k = yc.data[i] != Scalar[DT](0)
        if k:
            kept += 1
        var want_y = x.data[i] / Scalar[DT](1.0 - P) if k else Scalar[DT](0)
        var want_g = g.data[i] / Scalar[DT](1.0 - P) if k else Scalar[DT](0)
        if abs(Float64(yc.data[i] - want_y)) > 1e-6 or abs(Float64(gc.data[i] - want_g)) > 1e-6:
            bad += 1
    var frac = Float64(kept) / Float64(N)
    var sig = sqrt(P * (1.0 - P) / Float64(N))
    print("  HashDropout: kept", frac, "(want", 1.0 - P, "±", 4.0 * sig, ");",
          "CPU vs GPU differing", diff, "; wrong value or grad", bad)
    if diff > 0 or bad > 0 or abs(frac - (1.0 - P)) > 4.0 * sig:
        fails += 1
    # a second forward, and a second instance, draw other masks
    var y2 = Tensor()
    mc.forward["cpu", B](TensorRefs[1](x), y2, None)
    var other = HashDropout[D, P].make["cpu", Kaiming](None)
    var y3 = Tensor()
    other.forward["cpu", B](TensorRefs[1](x), y3, None)
    var same2 = 0
    var same3 = 0
    for i in range(N):
        if (y2.data[i] == Scalar[DT](0)) == (yc.data[i] == Scalar[DT](0)):
            same2 += 1
        if (y3.data[i] == Scalar[DT](0)) == (yc.data[i] == Scalar[DT](0)):
            same3 += 1
    # independent masks agree on ~ (1-p)^2 + p^2 = 0.82 of the elements
    print("  mask agreement with the first draw: next forward", Float64(same2) / Float64(N),
          " another instance", Float64(same3) / Float64(N), "(independent: 0.82)")
    if Float64(same2) / Float64(N) > 0.9 or Float64(same3) / Float64(N) > 0.9:
        fails += 1
    # off = identity
    mc.set_attr["dropout"](Scalar[DT](0))
    var y4 = Tensor()
    mc.forward["cpu", B](TensorRefs[1](x), y4, None)
    var n_off = 0
    for i in range(N):
        if y4.data[i] != x.data[i]:
            n_off += 1
    print("  dropout off: elements changed", n_off)
    if n_off > 0:
        fails += 1
    return fails


def _attn_ref(
    x: List[Scalar[DT]], gy: List[Scalar[DT]], B: Int, H: Int, S: Int, HD: Int,
    causal: Bool, seed_: UInt64, ctr: UInt64, drop: Bool,
) -> Tuple[List[Float64], List[Float64]]:
    """float64: out = (softmax(QKᵀ/√d) ⊙ M) V, and d/dx of <out, gy>."""
    var D = H * HD
    var y = List[Float64](length=B * S * D, fill=0.0)
    var gx = List[Float64](length=B * 3 * S * D, fill=0.0)
    var sc = 1.0 / sqrt(Float64(HD))
    for b in range(B):
        for h in range(H):
            var bh = b * H + h
            for i in range(S):
                var a = List[Float64](length=S, fill=0.0)
                var mx = -1e300
                var jend = i + 1 if causal else S
                for j in range(jend):
                    var s = 0.0
                    for d in range(HD):
                        s += Float64(x[b * 3 * S * D + i * D + h * HD + d]) * Float64(x[b * 3 * S * D + S * D + j * D + h * HD + d])
                    a[j] = s * sc
                    mx = max(mx, a[j])
                var z = 0.0
                for j in range(jend):
                    a[j] = exp(a[j] - mx)
                    z += a[j]
                var m = List[Float64](length=S, fill=0.0)
                for j in range(jend):
                    a[j] /= z
                    m[j] = 1.0
                    if drop:
                        m[j] = 1.0 / (1.0 - P) if hash_keep(seed_, ctr, UInt64(bh * S * S + i * S + j), Float32(P)) else 0.0
                # out and its backward for this (b, h, i)
                var ga = List[Float64](length=S, fill=0.0)
                for j in range(jend):
                    for d in range(HD):
                        var v = Float64(x[b * 3 * S * D + 2 * S * D + j * D + h * HD + d])
                        var go = Float64(gy[b * S * D + i * D + h * HD + d])
                        y[b * S * D + i * D + h * HD + d] += a[j] * m[j] * v
                        ga[j] += go * v
                        gx[b * 3 * S * D + 2 * S * D + j * D + h * HD + d] += a[j] * m[j] * go
                var dot = 0.0
                for j in range(jend):
                    ga[j] *= m[j]
                    dot += a[j] * ga[j]
                for j in range(jend):
                    var ds = a[j] * (ga[j] - dot) * sc
                    for d in range(HD):
                        gx[b * 3 * S * D + i * D + h * HD + d] += ds * Float64(x[b * 3 * S * D + S * D + j * D + h * HD + d])
                        gx[b * 3 * S * D + S * D + j * D + h * HD + d] += ds * Float64(x[b * 3 * S * D + i * D + h * HD + d])
    return (y^, gx^)


def _attn[target: StaticString, CAUSAL: Bool, B: Int](
    mut m: ScaledDotProductAttention[64, 2, 9, CAUSAL, True, DT, P],
    x: List[Scalar[DT]], gy: List[Scalar[DT]], ctx: Optional[DeviceContext],
) raises -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    comptime S = 9
    comptime D = 64
    var xt = Tensor.alloc(B * 3 * S * D)
    var gt = Tensor.alloc(B * S * D)
    for i in range(B * 3 * S * D):
        xt.data[i] = x[i]
    for i in range(B * S * D):
        gt.data[i] = gy[i]
    comptime if target == "gpu":
        xt.upload(ctx.value())
        gt.upload(ctx.value())
    var y = Tensor()
    var gi = Tensor()
    m.forward[target, B](TensorRefs[1](xt), y, ctx)
    m.vjp[target, B](TensorRefs[1](xt), gt, TensorRefs[1](gi), ctx)
    comptime if target == "gpu":
        ctx.value().synchronize()
        y.download(ctx.value())
        gi.download(ctx.value())
    var yo = List[Scalar[DT]]()
    for i in range(B * S * D):
        yo.append(y.data[i])
    var go = List[Scalar[DT]]()
    for i in range(B * 3 * S * D):
        go.append(gi.data[i])
    return (yo^, go^)


def _attention_dropout[CAUSAL: Bool](name: String, ctx: DeviceContext) raises -> Int:
    comptime B = 3
    comptime S = 9
    comptime D = 64
    seed(5)
    var x = List[Scalar[DT]]()
    var gy = List[Scalar[DT]]()
    for _ in range(B * 3 * S * D):
        x.append(Scalar[DT](random_float64(-1, 1)))
    for _ in range(B * S * D):
        gy.append(Scalar[DT](random_float64(-1, 1)))
    var fails = 0
    var mc = ScaledDotProductAttention[64, 2, 9, CAUSAL, True, DT, P].make["cpu", Kaiming](None)
    var mg = ScaledDotProductAttention[64, 2, 9, CAUSAL, True, DT, P].make["gpu", Kaiming](Optional(ctx))
    mg.drop_seed = mc.drop_seed
    for rep in range(2):  # the second forward draws a new mask (counter 1)
        var rc = _attn["cpu", CAUSAL, B](mc, x, gy, None)
        var rg = _attn["gpu", CAUSAL, B](mg, x, gy, Optional(ctx))
        var want = _attn_ref(x, gy, B, 2, S, 32, CAUSAL, mc.drop_seed, mc.drop_ctr_fwd, True)
        var e = List[Float64]()
        e.append(_std_err(want[0], rc[0]))
        e.append(_std_err(want[1], rc[1]))
        e.append(_std_err(want[0], rg[0]))
        e.append(_std_err(want[1], rg[1]))
        var bad = False
        for v in e:
            if v > 1e-4:
                bad = True
        print("  ", name, "dropout ON  draw", rep, ": cpu out", e[0], "grad", e[1], "| gpu out", e[2], "grad", e[3], " ✗" if bad else "")
        if bad:
            fails += 1
    # off: equal to the float64 reference without any mask
    mc.set_attr["dropout"](Scalar[DT](0))
    mg.set_attr["dropout"](Scalar[DT](0))
    var rc = _attn["cpu", CAUSAL, B](mc, x, gy, None)
    var rg = _attn["gpu", CAUSAL, B](mg, x, gy, Optional(ctx))
    var want = _attn_ref(x, gy, B, 2, S, 32, CAUSAL, 0, 0, False)
    var e0 = _std_err(want[0], rc[0])
    var e1 = _std_err(want[1], rc[1])
    var e2 = _std_err(want[0], rg[0])
    var e3 = _std_err(want[1], rg[1])
    var bad = e0 > 1e-4 or e1 > 1e-4 or e2 > 1e-4 or e3 > 1e-4
    print("  ", name, "dropout OFF        : cpu out", e0, "grad", e1, "| gpu out", e2, "grad", e3, " ✗" if bad else "")
    if bad:
        fails += 1
    return fails


def main() raises:
    print("HashDropout + attention dropout (p =", P, ")")
    var c = DeviceContext()
    var fails = _hash_dropout(c)
    fails += _attention_dropout[False](String("bidirectional"), c)
    fails += _attention_dropout[True](String("causal       "), c)
    if fails > 0:
        raise Error("FAIL: " + String(fails) + " check(s)")
    print("PASS")
