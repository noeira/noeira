# +--------------------------------------------------------------------------+ #
# | SuperMimic: differentiable per-lane FK of a free-root hinge tree, with its VJP
# +--------------------------------------------------------------------------+ #
"""Every body's world pose and world velocity at its ORIGIN from `(qpos, qvel)`,
and the vector-Jacobian product back to `(qpos, qvel)`. One lane = one state;
nothing here allocates, so the same functions become a GPU kernel's body.

The reproduction's `G1Kinematics.forward` (`kinematics/g1_fk.py`), formula for
formula, on any robot's tree (`SMFKTree`, read from its simulator's model
records by `sm_fk_tree[R]`). `NB` is the body count without the world. MuJoCo conventions: `qpos` = root pos, root quat WXYZ (normalised
here, so the VJP carries the normalisation), 29 hinges; `qvel` = root linear
velocity (world), root angular velocity (BODY frame), 29 hinge rates.

Per body `i` with parent `p` (`R(q)v` is mjlab's `quat_apply`):

    bp = P_p + R(Q_p) o_i              bq = Q_p ⊗ c_i         (fixed offset)
    hinge:  anchor = bp + R(bq) j_i    axis_w = R(bq) a_i
            Q_i = bq ⊗ (cos θ/2, a_i sin θ/2),  θ = q − qpos0
            P_i = anchor − R(Q_i) j_i
            W_i = W_p + axis_w q̇
            V_i = V_p + W_p × (P_i − P_p) + (axis_w q̇) × (P_i − anchor)
    fixed:  Q_i = bq,  P_i = bp,  W_i = W_p,  V_i = V_p + W_p × (P_i − P_p)

THE VJP is the reverse of exactly these formulas, bodies in decreasing index
order (MuJoCo's bodies are topologically sorted, so every child is done
before its parent). It differentiates `quat_apply` as the POLYNOMIAL mjlab
evaluates (`v + w t + u × t`, `t = 2 u × v`), not as a rotation: the two agree
on unit quaternions, and their derivatives agree along every direction that
keeps the quaternion unit, which is every direction the FK ever moves one in.
"""

from std.math import cos, sin, sqrt

from noeira.deep_agents.supermimic.robot import TrackingRobot, TrackingSim
from noeira.physics3d.gpu.constants import (
    BODY_IDX_PARENT, BODY_IDX_POS_X, BODY_IDX_POS_Y, BODY_IDX_POS_Z, BODY_IDX_QUAT_W,
    BODY_IDX_QUAT_X, BODY_IDX_QUAT_Y, BODY_IDX_QUAT_Z, JOINT_IDX_AXIS_X, JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z, JOINT_IDX_BODY_ID, JOINT_IDX_DOF_ADR, JOINT_IDX_POS_X, JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z, JOINT_IDX_QPOS_ADR, JOINT_IDX_TYPE,
)
from noeira.physics3d.joint_types import JNT_FREE, JNT_HINGE


comptime SM_FK_FIXED: Int = -1     # a body with no joint

# One tree record per body (index = MuJoCo body id − 1).
comptime FK_REC: Int = 18
comptime FK_PARENT: Int = 0        # parent's record index, −1 for the root
comptime FK_JTYPE: Int = 1         # JNT_FREE, JNT_HINGE or SM_FK_FIXED
comptime FK_QADR: Int = 2
comptime FK_DADR: Int = 3
comptime FK_POS: Int = 4           # body_pos (3), in the parent frame
comptime FK_QUAT: Int = 7          # body_quat (4, WXYZ), in the parent frame
comptime FK_JPOS: Int = 11         # jnt_pos (3), body frame
comptime FK_JAXIS: Int = 14        # jnt_axis (3), body frame
comptime FK_QPOS0: Int = 17

comptime FP[dt: DType] = Pointer[Scalar[dt], MutAnyOrigin]
comptime Q4[dt: DType] = SIMD[dt, 4]   # quaternion (w, x, y, z) or vector (x, y, z, 0)


struct SMFKTree[dt: DType](Movable):
    """The tree table, `NB × FK_REC` scalars of `dt` (body id − 1 per record)."""
    var data: List[Scalar[Self.dt]]

    def __init__(out self, var data: List[Scalar[Self.dt]]):
        self.data = data^

    def __init__(out self, *, deinit move: Self):
        self.data = move.data^

    def ptr(mut self) -> FP[Self.dt]:
        return self.data.unsafe_ptr().as_unsafe_any_origin()


def sm_fk_tree[dt: DType, R: TrackingRobot](sim: R.SIM) raises -> SMFKTree[dt]:
    """The FK table from the simulator's model records (what MuJoCo calls
    `body_parentid`, `body_pos`, `body_quat`, `jnt_*`). Refuses a body with
    more than one joint, a joint other than free / hinge, or a root that is
    not the free joint."""
    comptime NB = R.NBODY
    var t = List[Scalar[dt]](length=NB * FK_REC, fill=Scalar[dt](0))
    var njnt = List[Int](length=NB, fill=0)
    for b in range(1, NB + 1):
        var r = (b - 1) * FK_REC
        var parent = Int(sim.body_field(b, BODY_IDX_PARENT))
        if parent >= b:
            raise Error("sm_fk_tree: body " + String(b) + " precedes its parent " + String(parent))
        t[r + FK_PARENT] = Scalar[dt](parent - 1)
        t[r + FK_JTYPE] = Scalar[dt](SM_FK_FIXED)
        t[r + FK_POS + 0] = Scalar[dt](sim.body_field(b, BODY_IDX_POS_X))
        t[r + FK_POS + 1] = Scalar[dt](sim.body_field(b, BODY_IDX_POS_Y))
        t[r + FK_POS + 2] = Scalar[dt](sim.body_field(b, BODY_IDX_POS_Z))
        t[r + FK_QUAT + 0] = Scalar[dt](sim.body_field(b, BODY_IDX_QUAT_W))
        t[r + FK_QUAT + 1] = Scalar[dt](sim.body_field(b, BODY_IDX_QUAT_X))
        t[r + FK_QUAT + 2] = Scalar[dt](sim.body_field(b, BODY_IDX_QUAT_Y))
        t[r + FK_QUAT + 3] = Scalar[dt](sim.body_field(b, BODY_IDX_QUAT_Z))
    for j in range(R.NJOINT):
        var b = Int(sim.joint_field(j, JOINT_IDX_BODY_ID))
        var jt = Int(sim.joint_field(j, JOINT_IDX_TYPE))
        if jt != JNT_FREE and jt != JNT_HINGE:
            raise Error("sm_fk_tree: joint " + String(j) + " is neither free nor hinge")
        if b < 1:
            raise Error("sm_fk_tree: joint " + String(j) + " on the world")
        njnt[b - 1] += 1
        var r = (b - 1) * FK_REC
        t[r + FK_JTYPE] = Scalar[dt](jt)
        t[r + FK_QADR] = Scalar[dt](sim.joint_field(j, JOINT_IDX_QPOS_ADR))
        t[r + FK_DADR] = Scalar[dt](sim.joint_field(j, JOINT_IDX_DOF_ADR))
        t[r + FK_JPOS + 0] = Scalar[dt](sim.joint_field(j, JOINT_IDX_POS_X))
        t[r + FK_JPOS + 1] = Scalar[dt](sim.joint_field(j, JOINT_IDX_POS_Y))
        t[r + FK_JPOS + 2] = Scalar[dt](sim.joint_field(j, JOINT_IDX_POS_Z))
        t[r + FK_JAXIS + 0] = Scalar[dt](sim.joint_field(j, JOINT_IDX_AXIS_X))
        t[r + FK_JAXIS + 1] = Scalar[dt](sim.joint_field(j, JOINT_IDX_AXIS_Y))
        t[r + FK_JAXIS + 2] = Scalar[dt](sim.joint_field(j, JOINT_IDX_AXIS_Z))
        # qpos0 is a hinge's `ref`; 0 unless the asset sets one (the FK gate
        # against MuJoCo would show it)
    for i in range(NB):
        if njnt[i] > 1:
            raise Error("sm_fk_tree: body " + String(i + 1) + " has " + String(njnt[i]) + " joints")
    if Int(t[FK_JTYPE]) != JNT_FREE or Int(t[FK_PARENT]) != -1:
        raise Error("sm_fk_tree: body 1 must be the free-joint root")
    return SMFKTree[dt](t^)


# ── small quaternion / vector algebra on SIMD[dt, 4] ────────────────────────
@always_inline
def _at[dt: DType](p: FP[dt], i: Int) -> Scalar[dt]:
    return p[unsafe_offset=i]


@always_inline
def _ix[dt: DType](p: FP[dt], i: Int) -> Int:
    """An integer field of the tree table."""
    return Int(p[unsafe_offset=i])


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
def _put4[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] = v[0]
    p[unsafe_offset=i + 1] = v[1]
    p[unsafe_offset=i + 2] = v[2]
    p[unsafe_offset=i + 3] = v[3]


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
def cross3[dt: DType](a: Q4[dt], b: Q4[dt]) -> Q4[dt]:
    return Q4[dt](a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0)


@always_inline
def dot3[dt: DType](a: Q4[dt], b: Q4[dt]) -> Scalar[dt]:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@always_inline
def qmul[dt: DType](a: Q4[dt], b: Q4[dt]) -> Q4[dt]:
    """Hamilton a ⊗ b, WXYZ."""
    return Q4[dt](
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@always_inline
def qconj[dt: DType](q: Q4[dt]) -> Q4[dt]:
    return Q4[dt](q[0], -q[1], -q[2], -q[3])


@always_inline
def qapply[dt: DType](q: Q4[dt], v: Q4[dt]) -> Q4[dt]:
    """mjlab's `quat_apply`: v + w t + u × t, t = 2 u × v."""
    var u = Q4[dt](q[1], q[2], q[3], 0)
    var t = cross3(u, v) * 2
    return v + t * q[0] + cross3(u, t)


@always_inline
def qapply_vjp[dt: DType](q: Q4[dt], v: Q4[dt], g: Q4[dt]) -> Tuple[Q4[dt], Q4[dt]]:
    """(∂/∂q, ∂/∂v) of <g, qapply(q, v)> — the polynomial's derivative."""
    var u = Q4[dt](q[1], q[2], q[3], 0)
    var t = cross3(u, v) * 2
    var gv = g
    var gw = dot3(g, t)
    var gt = g * q[0] + cross3(g, u)       # from w t and from u × t
    var gu = cross3(t, g)                   # from u × t
    gu += cross3(v, gt) * 2                 # from t = 2 u × v
    gv += cross3(gt, u) * 2
    return (Q4[dt](gw, gu[0], gu[1], gu[2]), gv)


# ── forward ─────────────────────────────────────────────────────────────────
def sm_fk_lane[dt: DType, NB: Int](
    tree: FP[dt], qpos: FP[dt], qvel: FP[dt], pos: FP[dt], quat: FP[dt], lin: FP[dt], ang: FP[dt]
) where dt.is_floating_point():
    """One state's FK: `pos`/`lin`/`ang` `NB × 3`, `quat` `NB × 4`
    (WXYZ), world frame, at each body's origin."""
    var q = _q4(qpos, 3)
    var nrm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q = q / max(nrm, Scalar[dt](1e-8))
    _put3(pos, 0, _v3(qpos, 0))
    _put4(quat, 0, q)
    _put3(lin, 0, _v3(qvel, 0))
    _put3(ang, 0, qapply(q, _v3(qvel, 3)))
    for i in range(1, NB):
        var r = i * FK_REC
        var p = _ix(tree, r + FK_PARENT)
        var pp = _v3(pos, p * 3)
        var pq = _q4(quat, p * 4)
        var pw = _v3(ang, p * 3)
        var pv = _v3(lin, p * 3)
        var bp = pp + qapply(pq, _v3(tree, r + FK_POS))
        var bq = qmul(pq, _q4(tree, r + FK_QUAT))
        if _ix(tree, r + FK_JTYPE) == JNT_HINGE:
            var jp = _v3(tree, r + FK_JPOS)
            var ax = _v3(tree, r + FK_JAXIS)
            var anchor = bp + qapply(bq, jp)
            var axw = qapply(bq, ax)
            var h = (_at(qpos, _ix(tree, r + FK_QADR)) - _at(tree, r + FK_QPOS0)) * Scalar[dt](0.5)
            var sh = sin(h)
            var ql = Q4[dt](cos(h), ax[0] * sh, ax[1] * sh, ax[2] * sh)
            var qi = qmul(bq, ql)
            var pi = anchor - qapply(qi, jp)
            var wj = axw * _at(qvel, _ix(tree, r + FK_DADR))
            _put3(pos, i * 3, pi)
            _put4(quat, i * 4, qi)
            _put3(ang, i * 3, pw + wj)
            _put3(lin, i * 3, pv + cross3(pw, pi - pp) + cross3(wj, pi - anchor))
        else:
            _put3(pos, i * 3, bp)
            _put4(quat, i * 4, bq)
            _put3(ang, i * 3, pw)
            _put3(lin, i * 3, pv + cross3(pw, bp - pp))


# ── VJP ─────────────────────────────────────────────────────────────────────
def sm_fk_vjp_lane[dt: DType, NB: Int, NQ: Int, NV: Int](
    tree: FP[dt], qpos: FP[dt], qvel: FP[dt],
    pos: FP[dt], quat: FP[dt], ang: FP[dt],
    gpos: FP[dt], gquat: FP[dt], glin: FP[dt], gang: FP[dt],
    gqpos: FP[dt], gqvel: FP[dt],
) where dt.is_floating_point():
    """`gqpos` (NQ) and `gqvel` (NV) ← the VJP of `sm_fk_lane` at `(qpos, qvel)`
    against the cotangents `gpos`, `gquat`, `glin`, `gang` (same layouts as the
    outputs). `pos`, `quat`, `ang` are that call's outputs.

    ⚠ THE FOUR COTANGENT BUFFERS ARE ACCUMULATORS AND ARE OVERWRITTEN: on
    return they hold the total gradient of each body's own output (its
    cotangent plus everything its subtree sends up). Pass copies if the
    caller needs them. `gqpos` / `gqvel` are written, not added to.
    """
    for k in range(NQ):
        gqpos[unsafe_offset=k] = 0
    for k in range(NV):
        gqvel[unsafe_offset=k] = 0
    var i = NB - 1
    while i >= 1:
        var r = i * FK_REC
        var p = _ix(tree, r + FK_PARENT)
        var pp = _v3(pos, p * 3)
        var pq = _q4(quat, p * 4)
        var pw = _v3(ang, p * 3)
        var off = _v3(tree, r + FK_POS)
        var cq = _q4(tree, r + FK_QUAT)
        var bq = qmul(pq, cq)
        var pi = _v3(pos, i * 3)
        var g_p = _v3(gpos, i * 3)
        var g_q = _q4(gquat, i * 4)
        var g_v = _v3(glin, i * 3)
        var g_w = _v3(gang, i * 3)
        var gpp = Q4[dt](0)
        var gpq = Q4[dt](0)
        var gpw = Q4[dt](0)
        var gbq: Q4[dt]
        var gbp: Q4[dt]
        # V_i = V_p + W_p × (P_i − P_p) [+ wj × (P_i − anchor)]
        var gpv = g_v
        var dpp = pi - pp
        gpw += cross3(dpp, g_v)
        var gdpp = cross3(g_v, pw)
        gpw += g_w                          # W_i = W_p (+ wj)
        if _ix(tree, r + FK_JTYPE) == JNT_HINGE:
            var jp = _v3(tree, r + FK_JPOS)
            var ax = _v3(tree, r + FK_JAXIS)
            var qi = _q4(quat, i * 4)
            var bp = pp + qapply(pq, off)
            var anchor = bp + qapply(bq, jp)
            var axw = qapply(bq, ax)
            var qd = _at(qvel, _ix(tree, r + FK_DADR))
            var wj = axw * qd
            var dpa = pi - anchor
            var gwj = cross3(dpa, g_v) + g_w
            var gdpa = cross3(g_v, wj)
            var gpi = g_p + gdpp + gdpa     # P_i's total gradient
            gpp -= gdpp
            var ganchor = -gdpa
            gqvel[unsafe_offset=_ix(tree, r + FK_DADR)] = dot3(gwj, axw)
            var gaxw = gwj * qd
            # P_i = anchor − R(Q_i) j
            ganchor += gpi
            var gqi_c = qapply_vjp(qi, jp, -gpi)
            var gqi = g_q + gqi_c[0]
            # Q_i = bq ⊗ ql
            var h = (_at(qpos, _ix(tree, r + FK_QADR)) - _at(tree, r + FK_QPOS0)) * Scalar[dt](0.5)
            var sh = sin(h)
            var ch = cos(h)
            var ql = Q4[dt](ch, ax[0] * sh, ax[1] * sh, ax[2] * sh)
            gbq = qmul(gqi, qconj(ql))
            var gql = qmul(qconj(bq), gqi)
            gqpos[unsafe_offset=_ix(tree, r + FK_QADR)] = Scalar[dt](0.5) * (-sh * gql[0] + ch * (ax[0] * gql[1] + ax[1] * gql[2] + ax[2] * gql[3]))
            # axis_w = R(bq) a ; anchor = bp + R(bq) j
            gbq += qapply_vjp(bq, ax, gaxw)[0]
            gbq += qapply_vjp(bq, jp, ganchor)[0]
            gbp = ganchor
        else:
            gbp = g_p + gdpp
            gpp -= gdpp
            gbq = g_q
        # bq = Q_p ⊗ c ; bp = P_p + R(Q_p) o
        gpq += qmul(gbq, qconj(cq))
        gpp += gbp
        gpq += qapply_vjp(pq, off, gbp)[0]
        _add3(gpos, p * 3, gpp)
        _add4(gquat, p * 4, gpq)
        _add3(glin, p * 3, gpv)
        _add3(gang, p * 3, gpw)
        i -= 1
    # the root: P = qpos[0:3], Q = normalise(qpos[3:7]), V = qvel[0:3], W = R(Q) qvel[3:6]
    var x = _q4(qpos, 3)
    var nrm = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
    var den = max(nrm, Scalar[dt](1e-8))
    var q = x / den
    var gw = qapply_vjp(q, _v3(qvel, 3), _v3(gang, 0))
    var gq = _q4(gquat, 0) + gw[0]
    _put3(gqvel, 3, gw[1])
    _put3(gqvel, 0, _v3(glin, 0))
    _put3(gqpos, 0, _v3(gpos, 0))
    # d(x/|x|): (g − q <q, g>) / |x|  (the clamp is inactive on any real state)
    _put4(gqpos, 3, (gq - q * (q * gq).reduce_add()) / den)
