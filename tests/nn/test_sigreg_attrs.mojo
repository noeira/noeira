"""SIGReg's runtime attributes reach the leaf: `resample` and `fixed_a`.

  [1] `resample`. The storage port (2026-06-23) dropped SIGReg's `set_attr`, so
      `set_node_attr["sig", "resample"](1)` hit the trait's no-op and every LeWM
      run since kept one projection matrix for life. On: two forwards of the
      SAME input draw different matrices, so the statistic differs. Off: the
      two are bit-identical.
  [2] `fixed_a`. An injected matrix replaces the PRNG draw. The statistic is
      checked against an INDEPENDENT float64 Epps–Pulley written here from the
      reference (`stable_worldmodel/wm/loss.py:SIGReg.forward`):
          z      = x @ A                    (per b, t; A [D, P], unit columns)
          err_k  = (mean_b cos(z t_k) - φ_k)² + (mean_b sin(z t_k))²
          stat   = B · Σ_k err_k w_k,  w = trapezoid(2dt; ends dt) · φ,
          φ      = exp(-t²/2),  t = linspace(0, 3, K)
          out    = mean over (t, p)
      and a second, different matrix must change the answer (the injection is
      actually read).

Run:
    pixi run -e apple mojo run -I . tests/nn/test_sigreg_attrs.mojo
"""

from std.math import abs, cos, sin, exp, sqrt
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.sigreg import SIGReg


comptime D = 6
comptime T = 2
comptime P = 8
comptime K = 5
comptime B = 7
comptime SR = SIGReg[D, T, P, K]


def _x(mut x: Tensor):
    for i in range(B * T * D):
        x.data[i] = Scalar[DT](Float64((i * 29) % 17 - 8) * 0.21)


def _a(variant: Int) -> List[Float64]:
    """A [D, P] row-major with unit-norm columns."""
    var a = List[Float64](length=D * P, fill=0.0)
    for i in range(D * P):
        a[i] = Float64((i * (7 + variant * 4)) % 13 - 6) + 0.5
    for p in range(P):
        var n = 0.0
        for d in range(D):
            n += a[d * P + p] * a[d * P + p]
        n = sqrt(n)
        for d in range(D):
            a[d * P + p] /= n
    return a^


def _reference(ref x: Tensor, a: List[Float64]) -> Float64:
    var dt = 3.0 / Float64(K - 1)
    var total = 0.0
    for t in range(T):
        for p in range(P):
            var stat = 0.0
            for k in range(K):
                var tk = Float64(k) * dt
                var phi = exp(-tk * tk / 2.0)
                var wk = (dt if k == 0 or k == K - 1 else 2.0 * dt) * phi
                var mc = 0.0
                var ms = 0.0
                for b in range(B):
                    var z = 0.0
                    for d in range(D):
                        z += Float64(x.data[b * T * D + t * D + d]) * a[d * P + p]
                    mc += cos(z * tk)
                    ms += sin(z * tk)
                mc /= Float64(B)
                ms /= Float64(B)
                stat += ((mc - phi) * (mc - phi) + ms * ms) * wk
            total += stat * Float64(B)
    return total / Float64(T * P)


def _forward[target: StaticString](
    mut sr: SR, mut x: Tensor, c: Optional[DeviceContext]
) raises -> Float64:
    var out = Tensor.alloc(B)
    sr.forward[target, B](TensorRefs[1](x), out, c)
    comptime if target == "gpu":
        out.download(c.value())
    return Float64(out.data[0])


def _resample[target: StaticString](c: Optional[DeviceContext]) raises:
    var sr = SR.make[target, Deterministic](c)
    var x = Tensor.alloc(B * T * D)
    _x(x)
    comptime if target == "gpu":
        x.upload(c.value())
    var off1 = _forward[target](sr, x, c)
    var off2 = _forward[target](sr, x, c)
    sr.set_attr["resample"](Scalar[DT](1.0))
    var on1 = _forward[target](sr, x, c)
    var on2 = _forward[target](sr, x, c)
    print("  [1]", target, " off:", off1, off2, " on:", on1, on2)
    if off1 != off2:
        raise Error("FAIL [1] " + String(target) + ": resample OFF is not deterministic")
    if on1 == on2:
        raise Error(
            "FAIL [1] " + String(target)
            + ": resample ON drew the same matrix twice — the attr is not reaching the leaf"
        )


def _fixed[target: StaticString](c: DeviceContext) raises:
    var sr = SR.make[target, Deterministic](Optional(c) if target == "gpu" else None)
    var oc: Optional[DeviceContext] = Optional(c) if target == "gpu" else None
    var x = Tensor.alloc(B * T * D)
    _x(x)
    var want0 = _reference(x, _a(0))
    var want1 = _reference(x, _a(1))
    comptime if target == "gpu":
        x.upload(c)
    var got = List[Float64]()
    for v in range(2):
        var a = _a(v)
        var buf = c.enqueue_create_buffer[DT](D * P)
        with buf.map_to_host() as h:
            for i in range(D * P):
                h[i] = Scalar[DT](a[i])
        sr.set_attr_buf["fixed_a"](buf)
        got.append(_forward[target](sr, x, oc))
    print("  [2]", target, " A0:", got[0], "vs", want0, "  A1:", got[1], "vs", want1)
    for v in range(2):
        var w = want0 if v == 0 else want1
        if abs(got[v] - w) > 1e-5 * max(1.0, abs(w)):
            raise Error("FAIL [2] " + String(target) + ": injected-A statistic != reference")
    if abs(want0 - want1) < 1e-3:
        raise Error("FAIL [2]: the two matrices give the same statistic — vacuous")


def main() raises:
    print("SIGReg attributes reach the leaf")
    var c = DeviceContext()
    _resample["cpu"](None)
    _resample["gpu"](Optional(c))
    _fixed["cpu"](c)
    _fixed["gpu"](c)
    print("PASS")
