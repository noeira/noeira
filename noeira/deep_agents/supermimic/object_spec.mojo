# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the object's state spec — features, integration, error
# +--------------------------------------------------------------------------+ #
"""A free rigid object beside the robot, for the world model: what
`state_spec.mojo` is for the robot's root, per lane, each function with its
VJP. `docs/OBJECT_INTERACTION_PLAN.md` §4.2, step O1.3. There is no
reference implementation: the oracle is `tools/supermimic/obj_oracle.py`, an
independent torch transcription of the formulas below.

    o (13)      = [p 3, q 4 (WXYZ), v 3 (world), w 3 (the object's frame)]
                  MuJoCo's free-joint layout, as the robot's root in `s`
    feat (15)   = [R_h⁻¹ (p − p_root) 3, rot6d(h̄ ⊗ q) 6, R_h⁻¹ v 3, w / 2 3]
                  h = the heading (yaw) quaternion of the ROBOT's root: the
                  object is seen from the robot, heading-invariant like the
                  robot's own features
    delta (12)  = [dv_h 3, dw_b 3, dp_h 3, dθ_b 3] × (0.3, 0.5, 0.01, 0.02)
    integrate:  v' = v + R_h dv_h      w' = w + dw_b
                p' = p + dt v' + R_h dp_h
                q' = normalise(q ⊗ exp(dt w' + dθ_b))
                the root's rule in `sm_integrate_lane`, with the same scales
                and the robot's heading at the SAME step
    error (12)  = [Δp, log(q̄_true ⊗ q), Δv, Δw] / (0.05, 0.1, 0.5, 1)
    ctx (2)     = [mass (kg), sliding friction], normalised (m − 1.05) / 0.55,
                  (μ − 0.9) / 0.4 (the carry box: 0.1–2 kg, friction 0.9)

A zero `delta` integrates at constant velocity: an object at rest stays at
rest, which is where the object network starts (its last layer is zero).

⚠ THE ERROR's rotation term is the plain rotation vector between the two
poses. A cuboid looks the same after a half turn about an axis; a logged and a
predicted pose of the same body one step apart never differ by that much, so
the world model's loss needs no symmetry handling. A REWARD against a
reference pose would.

Pointer-in / pointer-out, no allocation, no Float64 at runtime: the bodies of
GPU kernels (`lanes.mojo`'s rules).
"""

from noeira.deep_agents.supermimic.heads import qapply_inv, yaw_quat, yaw_quat_vjp
from noeira.deep_agents.supermimic.kinematics import FP, Q4, qapply, qapply_vjp, qconj, qmul
from noeira.deep_agents.supermimic.robot import TrackingRobot
from noeira.deep_agents.supermimic.state_spec import (
    SM_WM_DT, qapply_inv_vjp, qnormalize, qnormalize_vjp, quat_from_rotvec, quat_from_rotvec_vjp, rot6d, rot6d_vjp,
    rotvec_from_quat, rotvec_from_quat_vjp,
)


comptime SM_OBJ: Int = 13            # the object's state
comptime SM_OBJ_FEAT: Int = 15
comptime SM_OBJ_DELTA: Int = 12
comptime SM_OBJ_CTX: Int = 2
comptime SM_OBJ_ERR_POS: Float64 = 0.05
"""The position scale of the training error (m): its first three entries are
the position difference in the world over this."""
# offsets inside o
comptime _OP: Int = 0
comptime _OQ: Int = 3
comptime _OV: Int = 7
comptime _OW: Int = 10


@always_inline
def sm_obj_ctx_normalize[dt: DType = DType.float64](i: Int, x: Scalar[dt]) -> Scalar[dt]:
    """mass: (m − 1.05) / 0.55; friction: (μ − 0.9) / 0.4."""
    if i == 0:
        return (x - 1.05) / 0.55
    return (x - 0.9) / 0.4


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


# ── features ────────────────────────────────────────────────────────────────
def sm_obj_feat_lane[dt: DType, R: TrackingRobot](s: FP[dt], o: FP[dt], feat: FP[dt]) where dt.is_floating_point():
    """The object seen from the robot (15), scaled: `s` is the robot's state
    (only its root pose is read), `o` the object's."""
    var q = qnormalize(_q4(s, 3))[0]
    var hq = yaw_quat(q)
    var qo = qnormalize(_q4(o, _OQ))[0]
    var r6 = rot6d(qmul(qconj(hq), qo))
    _put3(feat, 0, qapply_inv(hq, _v3(o, _OP) - _v3(s, 0)))
    comptime for k in range(6):
        feat[unsafe_offset=3 + k] = r6[k]
    _put3(feat, 9, qapply_inv(hq, _v3(o, _OV)))
    comptime for k in range(3):
        feat[unsafe_offset=12 + k] = o[unsafe_offset=_OW + k] / 2


def sm_obj_feat_vjp_lane[dt: DType, R: TrackingRobot](
    s: FP[dt], o: FP[dt], g_feat: FP[dt], g_s: FP[dt], g_o: FP[dt]
) where dt.is_floating_point():
    """`g_s += ∂/∂s`, `g_o += ∂/∂o` of <g_feat, feat(s, o)> (both ADD). Of `s`
    only the root position and quaternion get a cotangent."""
    var qn = qnormalize(_q4(s, 3))
    var q = qn[0]
    var hq = yaw_quat(q)
    var on = qnormalize(_q4(o, _OQ))
    var qo = on[0]
    # rel = R_h⁻¹ (p_o − p_root)
    var a = qapply_inv_vjp(hq, _v3(o, _OP) - _v3(s, 0), _v3(g_feat, 0))
    var g_hq = a[0]
    _add3(g_o, _OP, a[1])
    _add3(g_s, 0, -a[1])
    # r6 = rot6d(ql), ql = h̄ ⊗ q_o
    var g6 = Array[Scalar[dt], 6](fill=Scalar[dt](0))
    comptime for k in range(6):
        g6[k] = g_feat[unsafe_offset=3 + k]
    var gql = rot6d_vjp(qmul(qconj(hq), qo), g6)
    g_hq += qconj(qmul(gql, qconj(qo)))
    _add4(g_o, _OQ, qnormalize_vjp(qo, on[1], qmul(hq, gql)))
    # vh = R_h⁻¹ v_o
    var c = qapply_inv_vjp(hq, _v3(o, _OV), _v3(g_feat, 9))
    g_hq += c[0]
    _add3(g_o, _OV, c[1])
    comptime for k in range(3):
        g_o[unsafe_offset=_OW + k] += g_feat[unsafe_offset=12 + k] / 2
    _add4(g_s, 3, qnormalize_vjp(q, qn[1], yaw_quat_vjp(q, g_hq)))


# ── integration ─────────────────────────────────────────────────────────────
def sm_obj_integrate_lane[dt: DType, R: TrackingRobot](
    s: FP[dt], o: FP[dt], delta: FP[dt], o_out: FP[dt]
) where dt.is_floating_point():
    """`o_out` (13) = the object one step on, by `delta` (12, unscaled) in the
    heading frame of the robot state `s` of the same step."""
    var dtt = Scalar[dt](SM_WM_DT)
    var hq = yaw_quat(qnormalize(_q4(s, 3))[0])
    var qo = qnormalize(_q4(o, _OQ))[0]
    var dvh = _v3(delta, 0) * Scalar[dt](0.3)
    var dwb = _v3(delta, 3) * Scalar[dt](0.5)
    var dph = _v3(delta, 6) * Scalar[dt](0.01)
    var dth = _v3(delta, 9) * Scalar[dt](0.02)
    var v1 = _v3(o, _OV) + qapply(hq, dvh)
    var w1 = _v3(o, _OW) + dwb
    var p1 = _v3(o, _OP) + v1 * dtt + qapply(hq, dph)
    var q1 = qnormalize(qmul(qo, quat_from_rotvec(w1 * dtt + dth)))[0]
    _put3(o_out, _OP, p1)
    comptime for k in range(4):
        o_out[unsafe_offset=_OQ + k] = q1[k]
    _put3(o_out, _OV, v1)
    _put3(o_out, _OW, w1)


def sm_obj_integrate_vjp_lane[dt: DType, R: TrackingRobot](
    s: FP[dt], o: FP[dt], delta: FP[dt], g_out: FP[dt], g_s: FP[dt], g_o: FP[dt], g_delta: FP[dt]
) where dt.is_floating_point():
    """Of <g_out, integrate(s, o, delta)>: `g_s += ∂/∂s` (the robot's root
    quaternion only, through the heading), `g_o += ∂/∂o`, `g_delta = ∂/∂delta`
    (WRITTEN, 12)."""
    var dtt = Scalar[dt](SM_WM_DT)
    var qn = qnormalize(_q4(s, 3))
    var q = qn[0]
    var hq = yaw_quat(q)
    var on = qnormalize(_q4(o, _OQ))
    var qo = on[0]
    var dvh = _v3(delta, 0) * Scalar[dt](0.3)
    var dwb = _v3(delta, 3) * Scalar[dt](0.5)
    var dph = _v3(delta, 6) * Scalar[dt](0.01)
    var dth = _v3(delta, 9) * Scalar[dt](0.02)
    var w1 = _v3(o, _OW) + dwb
    var r = w1 * dtt + dth
    var e = quat_from_rotvec(r)
    var qqn = qnormalize(qmul(qo, e))
    # q' = normalise(q_o ⊗ exp(r)),  r = dt w' + dθ
    var gqq = qnormalize_vjp(qqn[0], qqn[1], _q4(g_out, _OQ))
    var gr = quat_from_rotvec_vjp(r, qmul(qconj(qo), gqq))
    var gw1 = _v3(g_out, _OW) + gr * dtt
    comptime for k in range(3):
        g_delta[unsafe_offset=9 + k] = gr[k] * Scalar[dt](0.02)
        g_delta[unsafe_offset=3 + k] = gw1[k] * Scalar[dt](0.5)
    _add3(g_o, _OW, gw1)
    _add4(g_o, _OQ, qnormalize_vjp(qo, on[1], qmul(gqq, qconj(e))))
    # p' = p + dt v' + R_h dp_h ; v' = v + R_h dv_h
    var gp1 = _v3(g_out, _OP)
    _add3(g_o, _OP, gp1)
    var a1 = qapply_vjp(hq, dph, gp1)
    var gv1 = _v3(g_out, _OV) + gp1 * dtt
    _add3(g_o, _OV, gv1)
    var a2 = qapply_vjp(hq, dvh, gv1)
    comptime for k in range(3):
        g_delta[unsafe_offset=6 + k] = a1[1][k] * Scalar[dt](0.01)
        g_delta[unsafe_offset=k] = a2[1][k] * Scalar[dt](0.3)
    _add4(g_s, 3, qnormalize_vjp(q, qn[1], yaw_quat_vjp(q, a1[0] + a2[0])))


# ── the training error ──────────────────────────────────────────────────────
def sm_obj_error_lane[dt: DType](op: FP[dt], ot: FP[dt], err: FP[dt]) where dt.is_floating_point():
    """The normalised error (12) of a predicted object state against a logged one."""
    var q1 = qnormalize(_q4(op, _OQ))[0]
    var q2 = qnormalize(_q4(ot, _OQ))[0]
    var rv = rotvec_from_quat(qmul(qconj(q2), q1))
    comptime for k in range(3):
        err[unsafe_offset=k] = (op[unsafe_offset=_OP + k] - ot[unsafe_offset=_OP + k]) / Scalar[dt](SM_OBJ_ERR_POS)
        err[unsafe_offset=3 + k] = rv[k] / Scalar[dt](0.1)
        err[unsafe_offset=6 + k] = (op[unsafe_offset=_OV + k] - ot[unsafe_offset=_OV + k]) / Scalar[dt](0.5)
        err[unsafe_offset=9 + k] = op[unsafe_offset=_OW + k] - ot[unsafe_offset=_OW + k]


def sm_obj_error_vjp_lane[dt: DType](op: FP[dt], ot: FP[dt], g_err: FP[dt], g_op: FP[dt]) where dt.is_floating_point():
    """`g_op += ∂/∂o_pred <g_err, error(o_pred, o_true)>` (ADDS)."""
    var n1 = qnormalize(_q4(op, _OQ))
    var q2 = qnormalize(_q4(ot, _OQ))[0]
    var g_rv = Q4[dt](g_err[unsafe_offset=3], g_err[unsafe_offset=4], g_err[unsafe_offset=5], 0) / Scalar[dt](0.1)
    var g_q1 = qmul(q2, rotvec_from_quat_vjp(qmul(qconj(q2), n1[0]), g_rv))
    _add4(g_op, _OQ, qnormalize_vjp(n1[0], n1[1], g_q1))
    comptime for k in range(3):
        g_op[unsafe_offset=_OP + k] += g_err[unsafe_offset=k] / Scalar[dt](SM_OBJ_ERR_POS)
        g_op[unsafe_offset=_OV + k] += g_err[unsafe_offset=6 + k] / Scalar[dt](0.5)
        g_op[unsafe_offset=_OW + k] += g_err[unsafe_offset=9 + k]
