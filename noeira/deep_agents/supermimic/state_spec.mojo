# +--------------------------------------------------------------------------+ #
# | SuperMimic M4: the world model's state spec — features, integration, error
# +--------------------------------------------------------------------------+ #
"""Per-lane `StateSpec` of the reproduction (`world_model/state_spec.py`), each
with its VJP, so a k-step world-model loss and an H-step imagined rollout can
be differentiated through the dynamics chain.

    s (S)       = [p 3, q 4 (WXYZ), qj n, v 3 (world), w 3 (body), qd n]
    feat (S)    = [p_z, rot6d(canonical(h̄ ⊗ q)), R_h⁻¹ v, w, qj, qd] / in_scale
                  h = the heading (yaw) quaternion of q: heading-invariant input
    delta (D)   = [dv_h 3, dw_b 3, dqd n, dp_h 3, dθ_b 3, dq n] (× out_scale)
    integrate:  v' = v + R_h dv_h      w' = w + dw_b       qd' = qd + dqd
                p' = p + dt v' + R_h dp_h
                q' = normalise(q ⊗ exp(dt w' + dθ_b))      qj' = qj + dt qd' + dq
    error (D)   = [Δp, log(q̄_true ⊗ q), Δv, Δw, Δqj, Δqd] / err_scale

(n DoF, S = nq + nv, D = 12 + 2n: `SMDims[R]`; 29 / 71 / 70 on the G1.)

A zero `delta` integrates at constant velocity; the dynamics MLP's last layer
starts at zero, so training starts from that prior.

Rotation helpers are the reproduction's `kinematics/rotations.py`:
`quat_from_rotvec` (series below |r| = 1e-4), `rotvec_from_quat` (canonical
w ≥ 0, series below |xyz| = 1e-6), `rot6d_from_quat` (mjlab's
`matrix_from_quat` columns 0–1, row-major), `heading_quat` (atan2 of yaw).
"""

from std.math import atan2, cos, sin, sqrt

from noeira.deep_agents.supermimic.kinematics import FP, Q4, cross3, dot3, qapply, qapply_vjp, qconj, qmul
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.heads import qapply_inv, yaw_quat, yaw_quat_vjp


comptime SM_WM_DT: Float64 = 0.02


# StateSpecCfg's scales, expanded to one value per entry (n = the robot's DoF)
@always_inline
def sm_in_scale[n: Int, dt: DType = DType.float64](i: Int) -> Scalar[dt]:
    """[root_z 1, rot6d 6, lin_vel_h 3, ang_vel_b 3, joint_pos n, joint_vel n]
    divided by (1, 1, 1, 2, 1, 5)."""
    if i < 10:
        return 1.0
    if i < 13:
        return 2.0
    if i < 13 + n:
        return 1.0
    return 5.0


@always_inline
def sm_out_scale[n: Int, dt: DType = DType.float64](i: Int) -> Scalar[dt]:
    """[dv_h 3, dw_b 3, dqd n, dp_h 3, dθ_b 3, dq n] × (0.3, 0.5, 2, 0.01, 0.02, 0.02)."""
    if i < 3:
        return 0.3
    if i < 6:
        return 0.5
    if i < 6 + n:
        return 2.0
    if i < 9 + n:
        return 0.01
    return 0.02


@always_inline
def sm_err_scale[n: Int, dt: DType = DType.float64](i: Int) -> Scalar[dt]:
    """[Δp 3, rotvec 3, Δv 3, Δw 3, Δqj n, Δqd n] / (0.05, 0.1, 0.5, 1, 0.1, 2)."""
    if i < 3:
        return 0.05
    if i < 6:
        return 0.1
    if i < 9:
        return 0.5
    if i < 12:
        return 1.0
    if i < 12 + n:
        return 0.1
    return 2.0


@always_inline
def sm_xi_normalize[n: Int, dt: DType = DType.float64](i: Int, x: Scalar[dt]) -> Scalar[dt]:
    """`(xi − offset) / scale`: friction (0.75, 0.45), n encoder biases (0, 0.01),
    CoM offset (0, 0.05)."""
    if i == 0:
        return (x - 0.75) / 0.45
    if i < 1 + n:
        return x / 0.01
    return x / 0.05


# ── small pieces ────────────────────────────────────────────────────────────
@always_inline
def _at[dt: DType](p: FP[dt], i: Int) -> Scalar[dt]:
    return p[unsafe_offset=i]


@always_inline
def _v3[dt: DType](p: FP[dt], i: Int) -> Q4[dt]:
    return Q4[dt](p[unsafe_offset=i], p[unsafe_offset=i + 1], p[unsafe_offset=i + 2], 0)


@always_inline
def _q4[dt: DType](p: FP[dt], i: Int) -> Q4[dt]:
    return Q4[dt](p[unsafe_offset=i], p[unsafe_offset=i + 1], p[unsafe_offset=i + 2], p[unsafe_offset=i + 3])


@always_inline
def _put3[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] = v[0]
    p[unsafe_offset=i + 1] = v[1]
    p[unsafe_offset=i + 2] = v[2]


@always_inline
def _add3[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] += v[0]
    p[unsafe_offset=i + 1] += v[1]
    p[unsafe_offset=i + 2] += v[2]


@always_inline
def _add4[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] += v[0]
    p[unsafe_offset=i + 1] += v[1]
    p[unsafe_offset=i + 2] += v[2]
    p[unsafe_offset=i + 3] += v[3]


@always_inline
def qnormalize[dt: DType](q: Q4[dt]) -> Tuple[Q4[dt], Scalar[dt]] where dt.is_floating_point():
    """`quat_normalize` (|q| clamped at 1e-8): (q / |q|, the clamped norm)."""
    var n = max(sqrt((q * q).reduce_add()), Scalar[dt](1e-8))
    return (q / n, n)


@always_inline
def qnormalize_vjp[dt: DType](qn: Q4[dt], n: Scalar[dt], g: Q4[dt]) -> Q4[dt]:
    return (g - qn * (qn * g).reduce_add()) / n


@always_inline
def qapply_inv_vjp[dt: DType](q: Q4[dt], v: Q4[dt], g: Q4[dt]) -> Tuple[Q4[dt], Q4[dt]]:
    """(∂/∂q, ∂/∂v) of <g, qapply_inv(q, v)>: `qapply_inv(q, v)` is `qapply(q̄, v)`."""
    var r = qapply_vjp(qconj(q), v, g)
    return (qconj(r[0]), r[1])


@always_inline
def rot6d[dt: DType](q: Q4[dt]) -> Array[Scalar[dt], 6]:
    """[m00, m01, m10, m11, m20, m21] of mjlab's `matrix_from_quat`."""
    var w = q[0]
    var x = q[1]
    var y = q[2]
    var z = q[3]
    var t = Scalar[dt](2) / (q * q).reduce_add()
    var r = Array[Scalar[dt], 6](uninitialized=True)
    r[0] = 1 - t * (y * y + z * z)
    r[1] = t * (x * y - z * w)
    r[2] = t * (x * y + z * w)
    r[3] = 1 - t * (x * x + z * z)
    r[4] = t * (x * z - y * w)
    r[5] = t * (y * z + x * w)
    return r^


@always_inline
def rot6d_vjp[dt: DType](q: Q4[dt], g: Array[Scalar[dt], 6]) -> Q4[dt]:
    var w = q[0]
    var x = q[1]
    var y = q[2]
    var z = q[3]
    var n2 = (q * q).reduce_add()
    var t = Scalar[dt](2) / n2
    var A = y * y + z * z
    var Bv = x * y - z * w
    var C = x * y + z * w
    var D = x * x + z * z
    var E = x * z - y * w
    var F = y * z + x * w
    var gt = -g[0] * A + g[1] * Bv + g[2] * C - g[3] * D + g[4] * E + g[5] * F
    var out = q * (gt * Scalar[dt](-4) / (n2 * n2))     # ∂t/∂q = −4 q / n²²
    out += Q4[dt](0, 0, 2 * y, 2 * z) * (-g[0] * t)
    out += Q4[dt](-z, y, x, -w) * (g[1] * t)
    out += Q4[dt](z, y, x, w) * (g[2] * t)
    out += Q4[dt](0, 2 * x, 0, 2 * z) * (-g[3] * t)
    out += Q4[dt](-y, z, -w, x) * (g[4] * t)
    out += Q4[dt](x, w, z, y) * (g[5] * t)
    return out


@always_inline
def quat_from_rotvec[dt: DType](r: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    var a2 = dot3(r, r)
    var ang = sqrt(max(a2, Scalar[dt](1e-30)))
    if ang < Scalar[dt](1e-4):
        var k = Scalar[dt](0.5) - a2 / 48
        return Q4[dt](1 - a2 / 8, r[0] * k, r[1] * k, r[2] * k)
    var k = sin(ang / 2) / ang
    return Q4[dt](cos(ang / 2), r[0] * k, r[1] * k, r[2] * k)


@always_inline
def quat_from_rotvec_vjp[dt: DType](r: Q4[dt], g: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    var a2 = dot3(r, r)
    var ang = sqrt(max(a2, Scalar[dt](1e-30)))
    var gx = Q4[dt](g[1], g[2], g[3], 0)
    var rg = dot3(r, gx)
    if ang < Scalar[dt](1e-4):
        var k = Scalar[dt](0.5) - a2 / 48
        # ∂k/∂r = −r/24, ∂w/∂r = −r/4
        return gx * k + r * (rg * Scalar[dt](-1.0 / 24.0) + g[0] * Scalar[dt](-0.25))
    var h = ang / 2
    var k = sin(h) / ang
    var dk_dang = (cos(h) * ang / 2 - sin(h)) / (ang * ang)
    var dw_dang = -sin(h) / 2
    return gx * k + r * ((rg * dk_dang + g[0] * dw_dang) / ang)


@always_inline
def rotvec_from_quat[dt: DType](q0: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    var q = -q0 if q0[0] < 0 else q0
    var x = Q4[dt](q[1], q[2], q[3], 0)
    var sn = sqrt(dot3(x, x))
    var ang = atan2(sn, q[0]) * 2
    if sn < Scalar[dt](1e-6):
        return x * (Scalar[dt](2) + ang * ang / 12)
    return x * (ang / sn)


@always_inline
def rotvec_from_quat_vjp[dt: DType](q0: Q4[dt], g: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    var s = Scalar[dt](-1) if q0[0] < 0 else Scalar[dt](1)
    var q = q0 * s
    var w = q[0]
    var x = Q4[dt](q[1], q[2], q[3], 0)
    var sn = sqrt(dot3(x, x))
    var n2 = sn * sn + w * w
    var ang = atan2(sn, w) * 2
    var dang_dsn = 2 * w / n2
    var dang_dw = -2 * sn / n2
    var xg = dot3(x, g)
    var k: Scalar[dt]
    var dk_dsn: Scalar[dt]
    var dk_dw: Scalar[dt]
    if sn < Scalar[dt](1e-6):
        k = Scalar[dt](2) + ang * ang / 12
        dk_dsn = ang / 6 * dang_dsn
        dk_dw = ang / 6 * dang_dw
    else:
        k = ang / sn
        dk_dsn = (dang_dsn * sn - ang) / (sn * sn)
        dk_dw = dang_dw / sn
    var gxv = g * k
    if sn > 0:
        gxv += x * (xg * dk_dsn / sn)
    var gw = xg * dk_dw
    return Q4[dt](gw, gxv[0], gxv[1], gxv[2]) * s


# ── canonical features ──────────────────────────────────────────────────────
def sm_canonical_lane[dt: DType, R: TrackingRobot](s: FP[dt], feat: FP[dt]) where dt.is_floating_point():
    """`StateSpec.canonical`: the scaled, heading-invariant input (S)."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    comptime DD = SMDims[R].DELTA
    var qn = qnormalize(_q4(s, 3))
    var q = qn[0]
    var hq = yaw_quat(q)
    var ql = qmul(qconj(hq), q)
    if ql[0] < 0:
        ql = -ql
    var r6 = rot6d(ql)
    var vh = qapply_inv(hq, _v3(s, NQ))
    feat[unsafe_offset=0] = _at(s, 2)
    comptime for k in range(6):
        feat[unsafe_offset=1 + k] = r6[k]
    _put3(feat, 7, vh)
    comptime for k in range(3):
        feat[unsafe_offset=10 + k] = _at(s, NQ + 3 + k) / 2
    for j in range(n):
        feat[unsafe_offset=13 + j] = _at(s, 7 + j)
        feat[unsafe_offset=13 + n + j] = _at(s, NQ + 6 + j) / 5


def sm_canonical_vjp_lane[dt: DType, R: TrackingRobot](s: FP[dt], g_feat: FP[dt], g_s: FP[dt]) where dt.is_floating_point():
    """`g_s += ∂/∂s <g_feat, canonical(s)>` (ADDS)."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    comptime DD = SMDims[R].DELTA
    var qn = qnormalize(_q4(s, 3))
    var q = qn[0]
    var hq = yaw_quat(q)
    var ql = qmul(qconj(hq), q)
    var sg = Scalar[dt](1)
    if ql[0] < 0:
        sg = -1
        ql = -ql
    var g6 = Array[Scalar[dt], 6](fill=Scalar[dt](0))
    comptime for k in range(6):
        g6[k] = _at(g_feat, 1 + k)
    var gql = rot6d_vjp(ql, g6) * sg                   # back through the sign flip
    # ql = h̄ ⊗ q
    var g_hq = qconj(qmul(gql, qconj(q)))
    var g_q = qmul(hq, gql)
    # vh = R_h⁻¹ v
    var gi = qapply_inv_vjp(hq, _v3(s, NQ), _v3(g_feat, 7))
    g_hq += gi[0]
    _add3(g_s, NQ, gi[1])
    g_q += yaw_quat_vjp(q, g_hq)
    _add4(g_s, 3, qnormalize_vjp(q, qn[1], g_q))
    g_s[unsafe_offset=2] += _at(g_feat, 0)
    comptime for k in range(3):
        g_s[unsafe_offset=NQ + 3 + k] += _at(g_feat, 10 + k) / 2
    for j in range(n):
        g_s[unsafe_offset=7 + j] += _at(g_feat, 13 + j)
        g_s[unsafe_offset=NQ + 6 + j] += _at(g_feat, 13 + n + j) / 5


# ── integration ─────────────────────────────────────────────────────────────
def sm_integrate_lane[dt: DType, R: TrackingRobot](s: FP[dt], delta: FP[dt], s_out: FP[dt]) where dt.is_floating_point():
    """`StateSpec.integrate(s, delta)` → `s_out` (S); `delta` unscaled (× out_scale here)."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    comptime DD = SMDims[R].DELTA
    var dtt = Scalar[dt](SM_WM_DT)
    var qn = qnormalize(_q4(s, 3))
    var q = qn[0]
    var hq = yaw_quat(q)
    var dvh = _v3(delta, 0) * Scalar[dt](0.3)
    var dwb = _v3(delta, 3) * Scalar[dt](0.5)
    var dph = _v3(delta, 6 + n) * Scalar[dt](0.01)
    var dth = _v3(delta, 9 + n) * Scalar[dt](0.02)
    var v1 = _v3(s, NQ) + qapply(hq, dvh)
    var w1 = _v3(s, NQ + 3) + dwb
    var p1 = _v3(s, 0) + v1 * dtt + qapply(hq, dph)
    var q1 = qnormalize(qmul(q, quat_from_rotvec(w1 * dtt + dth)))[0]
    _put3(s_out, 0, p1)
    comptime for k in range(4):
        s_out[unsafe_offset=3 + k] = q1[k]
    for j in range(n):
        var qd1 = _at(s, NQ + 6 + j) + _at(delta, 6 + j) * Scalar[dt](2.0)
        s_out[unsafe_offset=NQ + 6 + j] = qd1
        s_out[unsafe_offset=7 + j] = _at(s, 7 + j) + dtt * qd1 + _at(delta, 12 + n + j) * Scalar[dt](0.02)
    _put3(s_out, NQ, v1)
    _put3(s_out, NQ + 3, w1)


def sm_integrate_vjp_lane[dt: DType, R: TrackingRobot](
    s: FP[dt], delta: FP[dt], g_out: FP[dt], g_s: FP[dt], g_delta: FP[dt]
) where dt.is_floating_point():
    """`g_s += ∂/∂s`, `g_delta = ∂/∂delta` of <g_out, integrate(s, delta)>
    (`g_s` ADDS, `g_delta` is WRITTEN)."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    comptime DD = SMDims[R].DELTA
    var dtt = Scalar[dt](SM_WM_DT)
    var qn = qnormalize(_q4(s, 3))
    var q = qn[0]
    var hq = yaw_quat(q)
    var dvh = _v3(delta, 0) * Scalar[dt](0.3)
    var dwb = _v3(delta, 3) * Scalar[dt](0.5)
    var dph = _v3(delta, 6 + n) * Scalar[dt](0.01)
    var dth = _v3(delta, 9 + n) * Scalar[dt](0.02)
    var w1 = _v3(s, NQ + 3) + dwb
    var r = w1 * dtt + dth
    var e = quat_from_rotvec(r)
    var qq = qmul(q, e)
    var qqn = qnormalize(qq)
    # g_delta is written in place, already × out_scale (the d = delta ⊙ out_scale chain)
    # qj' = qj + dt qd' + dq ; qd' = qd + dqd
    for j in range(n):
        var gqj = _at(g_out, 7 + j)
        var gqd1 = _at(g_out, NQ + 6 + j) + dtt * gqj
        g_s[unsafe_offset=7 + j] += gqj
        g_delta[unsafe_offset=12 + n + j] = gqj * Scalar[dt](0.02)
        g_s[unsafe_offset=NQ + 6 + j] += gqd1
        g_delta[unsafe_offset=6 + j] = gqd1 * Scalar[dt](2.0)
    # q' = normalise(q ⊗ exp(r)),  r = dt w' + dθ
    var gqq = qnormalize_vjp(qqn[0], qqn[1], _q4(g_out, 3))
    var g_q = qmul(gqq, qconj(e))
    var gr = quat_from_rotvec_vjp(r, qmul(qconj(q), gqq))
    var gw1 = _v3(g_out, NQ + 3) + gr * dtt
    comptime for k in range(3):
        g_delta[unsafe_offset=9 + n + k] = gr[k] * Scalar[dt](0.02)
        g_delta[unsafe_offset=3 + k] = gw1[k] * Scalar[dt](0.5)
    _add3(g_s, NQ + 3, gw1)
    # p' = p + dt v' + R_h dp_h ; v' = v + R_h dv_h
    var gp1 = _v3(g_out, 0)
    _add3(g_s, 0, gp1)
    var a1 = qapply_vjp(hq, dph, gp1)
    var gv1 = _v3(g_out, NQ) + gp1 * dtt
    _add3(g_s, NQ, gv1)
    var a2 = qapply_vjp(hq, dvh, gv1)
    comptime for k in range(3):
        g_delta[unsafe_offset=6 + n + k] = a1[1][k] * Scalar[dt](0.01)
        g_delta[unsafe_offset=k] = a2[1][k] * Scalar[dt](0.3)
    g_q += yaw_quat_vjp(q, a1[0] + a2[0])
    _add4(g_s, 3, qnormalize_vjp(q, qn[1], g_q))


# ── the training error ──────────────────────────────────────────────────────
def sm_error_lane[dt: DType, R: TrackingRobot](sp: FP[dt], st: FP[dt], err: FP[dt]) where dt.is_floating_point():
    """`StateSpec.error_vector(s_pred, s_true)` (DELTA, normalised)."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    comptime DD = SMDims[R].DELTA
    var q1 = qnormalize(_q4(sp, 3))[0]
    var q2 = qnormalize(_q4(st, 3))[0]
    var rv = rotvec_from_quat(qmul(qconj(q2), q1))
    comptime for k in range(3):
        err[unsafe_offset=k] = (_at(sp, k) - _at(st, k)) / (sm_err_scale[n, dt](k))
        err[unsafe_offset=3 + k] = rv[k] / (sm_err_scale[n, dt](3))
        err[unsafe_offset=6 + k] = (_at(sp, NQ + k) - _at(st, NQ + k)) / (sm_err_scale[n, dt](6))
        err[unsafe_offset=9 + k] = (_at(sp, NQ + 3 + k) - _at(st, NQ + 3 + k)) / (sm_err_scale[n, dt](9))
    for j in range(n):
        err[unsafe_offset=12 + j] = (_at(sp, 7 + j) - _at(st, 7 + j)) / (sm_err_scale[n, dt](12))
        err[unsafe_offset=12 + n + j] = (_at(sp, NQ + 6 + j) - _at(st, NQ + 6 + j)) / (sm_err_scale[n, dt](12 + n))


def sm_error_vjp_lane[dt: DType, R: TrackingRobot](sp: FP[dt], st: FP[dt], g_err: FP[dt], g_sp: FP[dt]) where dt.is_floating_point():
    """`g_sp += ∂/∂s_pred <g_err, error_vector(s_pred, s_true)>` (ADDS)."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    comptime DD = SMDims[R].DELTA
    var n1 = qnormalize(_q4(sp, 3))
    var q2 = qnormalize(_q4(st, 3))[0]
    var dq = qmul(qconj(q2), n1[0])
    var g_rv = Q4[dt](_at(g_err, 3), _at(g_err, 4), _at(g_err, 5), 0) / (sm_err_scale[n, dt](3))
    var g_q1 = qmul(q2, rotvec_from_quat_vjp(dq, g_rv))
    _add4(g_sp, 3, qnormalize_vjp(n1[0], n1[1], g_q1))
    comptime for k in range(3):
        g_sp[unsafe_offset=k] += _at(g_err, k) / (sm_err_scale[n, dt](k))
        g_sp[unsafe_offset=NQ + k] += _at(g_err, 6 + k) / (sm_err_scale[n, dt](6))
        g_sp[unsafe_offset=NQ + 3 + k] += _at(g_err, 9 + k) / (sm_err_scale[n, dt](9))
    for j in range(n):
        g_sp[unsafe_offset=7 + j] += _at(g_err, 12 + j) / (sm_err_scale[n, dt](12))
        g_sp[unsafe_offset=NQ + 6 + j] += _at(g_err, 12 + n + j) / (sm_err_scale[n, dt](12 + n))
