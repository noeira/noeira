"""The world model's pose update under the build's drift weight
(`state_spec.SM_WM_DRIFT`, `-D SM_WM_DRIFT_MILLI=`): poses advance by
dt·((1 − c)·v + c·v'), c = 1 by default (the reproduction's `p' = p + dt v'`),
0.625 in the public recipes.

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_drift.mojo
    pixi run mojo build -I . -D SM_WM_DRIFT_MILLI=625 tests/deep_agents/supermimic/test_supermimic_drift.mojo -o t && ./t

Float64, per-lane functions only (no network, no simulator). The expected
values are written in terms of c, so the same file gates both builds.

WHAT EACH SECTION CAN CATCH
===========================
[1] known answers of `sm_integrate_lane`: a zero delta is constant velocity
    whatever c; a joint-velocity increment moves the joint by dt·c of it, a
    root-velocity increment the root by dt·c of it, a root angular-velocity
    increment turns the root by dt·c of it; the new velocities are the old
    ones plus the full increment.
[2] `sm_integrate_vjp_lane` against central finite differences on every
    entry of the state and of the delta, on random states.
"""

from std.math import abs, asin, sqrt
from std.random import random_float64, seed

from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.robot import G1Robot, SMDims
from noeira.deep_agents.supermimic.state_spec import SM_WM_DRIFT, SM_WM_DRIFT_MILLI, SM_WM_DT, sm_integrate_lane, sm_integrate_vjp_lane


comptime R = G1Robot
comptime n = R.NDOF
comptime NQ = R.NQ
comptime S = SMDims[R].S
comptime DD = SMDims[R].DELTA
comptime F64 = DType.float64


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _p(mut l: List[Float64]) -> FP[F64]:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _near(a: Float64, b: Float64, tol: Float64 = 1e-12) -> Bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def _state() -> List[Float64]:
    """A state with the identity root orientation and fixed velocities."""
    var s = List[Float64](length=S, fill=0.0)
    s[0] = 0.3
    s[1] = -0.2
    s[2] = 0.75
    s[3] = 1.0
    for j in range(n):
        s[7 + j] = 0.01 * Float64(j) - 0.1
        s[NQ + 6 + j] = 0.3 - 0.02 * Float64(j)
    s[NQ] = 0.4
    s[NQ + 1] = -0.1
    s[NQ + 2] = 0.05
    return s^


def _dot_integrate(mut s: List[Float64], mut d: List[Float64], g: List[Float64]) -> Float64:
    var o = List[Float64](length=S, fill=0.0)
    sm_integrate_lane[F64, R](_p(s), _p(d), _p(o))
    var acc = 0.0
    for i in range(S):
        acc += g[i] * o[i]
    return acc


def main() raises:
    var t = Tally()
    comptime c = SM_WM_DRIFT
    comptime dt = SM_WM_DT
    print("drift weight c =", c, "(", SM_WM_DRIFT_MILLI, "/ 1000 )")

    print("\n[1] known answers")
    var s = _state()
    var zero = List[Float64](length=DD, fill=0.0)
    var o = List[Float64](length=S, fill=0.0)
    sm_integrate_lane[F64, R](_p(s), _p(zero), _p(o))
    var ok = True
    for k in range(3):
        ok = ok and _near(o[k], s[k] + dt * s[NQ + k]) and _near(o[NQ + k], s[NQ + k])
    for j in range(n):
        ok = ok and _near(o[7 + j], s[7 + j] + dt * s[NQ + 6 + j]) and _near(o[NQ + 6 + j], s[NQ + 6 + j])
    t.truth(ok, "a zero delta is constant velocity (root and joints), whatever c")

    comptime J0 = 5
    var d1 = List[Float64](length=DD, fill=0.0)
    d1[6 + J0] = 0.7                                 # dqd of joint 5, × out_scale 2
    sm_integrate_lane[F64, R](_p(s), _p(d1), _p(o))
    t.truth(_near(o[NQ + 6 + J0], s[NQ + 6 + J0] + 1.4), "a joint-velocity increment adds in full to the joint's velocity")
    t.truth(
        _near(o[7 + J0], s[7 + J0] + dt * (s[NQ + 6 + J0] + c * 1.4)),
        "and moves the joint by dt x (old velocity + c x increment): " + String(o[7 + J0] - s[7 + J0]),
    )

    var d2 = List[Float64](length=DD, fill=0.0)
    d2[0] = 0.5                                      # dv_h along x, × out_scale 0.3; the heading is the identity
    sm_integrate_lane[F64, R](_p(s), _p(d2), _p(o))
    t.truth(_near(o[NQ], s[NQ] + 0.15), "a root-velocity increment adds in full to the root's velocity")
    t.truth(_near(o[0], s[0] + dt * (s[NQ] + c * 0.15)), "and moves the root by dt x (old velocity + c x increment)")

    var d3 = List[Float64](length=DD, fill=0.0)
    d3[5] = 0.8                                      # dw_b about z, × out_scale 0.5; the root starts at rest
    sm_integrate_lane[F64, R](_p(s), _p(d3), _p(o))
    var angle = 2.0 * asin(o[6])                     # q' = (cos(a/2), 0, 0, sin(a/2))
    t.truth(_near(o[NQ + 5], 0.4), "a root angular-velocity increment adds in full")
    t.truth(_near(angle, dt * c * 0.4, 1e-9), "and turns the root by dt x c x increment: " + String(angle) + " rad")

    print("\n[2] the VJP against central finite differences")
    seed(31)
    comptime EPS = 1e-6
    var worst_s = 0.0
    var worst_d = 0.0
    var scale = 0.0
    for trial in range(4):
        var sv = List[Float64](length=S, fill=0.0)
        var dv = List[Float64](length=DD, fill=0.0)
        var g = List[Float64](length=S, fill=0.0)
        for i in range(S):
            sv[i] = random_float64(-1.0, 1.0)
            g[i] = random_float64(-1.0, 1.0)
        var qn = sqrt(sv[3] * sv[3] + sv[4] * sv[4] + sv[5] * sv[5] + sv[6] * sv[6])
        for k in range(4):
            sv[3 + k] /= qn
        for i in range(DD):
            dv[i] = random_float64(-1.0, 1.0)
        var gs = List[Float64](length=S, fill=0.0)
        var gd = List[Float64](length=DD, fill=0.0)
        sm_integrate_vjp_lane[F64, R](_p(sv), _p(dv), _p(g), _p(gs), _p(gd))
        for i in range(S):
            var keep = sv[i]
            sv[i] = keep + EPS
            var fp = _dot_integrate(sv, dv, g)
            sv[i] = keep - EPS
            var fm = _dot_integrate(sv, dv, g)
            sv[i] = keep
            var fd = (fp - fm) / (2.0 * EPS)
            worst_s = max(worst_s, abs(fd - gs[i]))
            scale = max(scale, abs(fd))
        for i in range(DD):
            var keep = dv[i]
            dv[i] = keep + EPS
            var fp = _dot_integrate(sv, dv, g)
            dv[i] = keep - EPS
            var fm = _dot_integrate(sv, dv, g)
            dv[i] = keep
            var fd = (fp - fm) / (2.0 * EPS)
            worst_d = max(worst_d, abs(fd - gd[i]))
    t.truth(worst_s <= 1e-7 * scale, "every state entry: worst " + String(worst_s) + " of " + String(scale))
    t.truth(worst_d <= 1e-7 * scale, "every delta entry: worst " + String(worst_d))

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_drift: " + String(t.fails) + " failed")
