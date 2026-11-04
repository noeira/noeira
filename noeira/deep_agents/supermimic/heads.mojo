# +--------------------------------------------------------------------------+ #
# | SuperMimic M3b: the analytic heads (reward, termination features, obs)
# +--------------------------------------------------------------------------+ #
"""The world model's analytic heads for one lane, and their VJP back to the
predicted state, the action and the previous action — what the policy's BPTT
differentiates at every imagined step.

The reproduction's `world_model/heads.py` + `envs/mdp/rewards.py` (BeyondMimic's
eight terms, mjlab's `update_relative_body_poses` re-anchoring), with mjlab's
quaternion helpers (`references/mjlab-8bb2bc9`, `utils/lab_api/math.py`):

    r = Σ  dt·w·exp(−e/σ²)  over root pos / root ori / body pos / body ori /
                             body lin vel / body ang vel
        + dt·w_ar·|a − a_prev|²  + dt·w_jl·Σ (relu(lo − q) + relu(q − hi))
    term features = [ref root z − z,  (R_ref⁻¹ g)_z − (R⁻¹ g)_z],  g = (0, 0, −1)
    obs (93) = [q + bias − q0, q̇, ω_body, R⁻¹ g, a]

`R⁻¹ g` uses the state's root quaternion AS IS (not normalised), as the
reproduction does; the FK normalises it for the body poses.

THE QUATERNION ERROR. `quat_box_minus(q1, q2) = axis_angle(q1 ⊗ q̄2)`, and the
reward uses its squared norm. mjlab's `axis_angle_from_quat` flips the sign
when w < 0 and switches to a Taylor form when |angle| ≤ 1e-6. On its main
branch |v|² equals 4·atan2(|xyz|, w)²·|q|² EXACTLY (sin(½θ) = |xyz|/|q|),
which is what is computed here; the Taylor branch is reproduced as written.
Both forms and their derivatives are gated against torch.

Layouts (per lane; G1 widths in brackets, `SMDims[R]` in general):
    s      S [71]    qpos | qvel (root angular velocity in the BODY frame)
    rf     R [189]   n_tracked bodies × [pos 3, quat 4, lin vel 3, ang vel 3]
                     (world), then the reference root pos 3, quat 4
    xi     X [33]    sim context; [1 : 1 + n] is the encoder bias
    terms  8         REWARD_TERM_NAMES' order, weighted and × dt
The robot's tracked rows, anchor, q0 and soft limits are a runtime table
(`SMHeadLayout[R]`, built by `tables.sm_tables`). Everything here is
pointer-in / pointer-out with no allocation: a GPU kernel's per-lane body.
"""

from std.math import atan2, cos, exp, sin, sqrt

from noeira.deep_agents.supermimic.kinematics import (
    FP, Q4, cross3, dot3, qapply, qapply_vjp, qconj, qmul, sm_fk_lane, sm_fk_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot


comptime SM_N_TERMS: Int = 8

# TrackingRewardCfg (rewards.py)
comptime SM_DT: Float64 = 0.02
comptime SM_STD_BODY_POS: Float64 = 0.3
comptime SM_STD_BODY_ORI: Float64 = 0.4
comptime SM_STD_BODY_LIN: Float64 = 1.0
comptime SM_STD_BODY_ANG: Float64 = 3.14
comptime SM_STD_ROOT_POS: Float64 = 0.3
comptime SM_STD_ROOT_ORI: Float64 = 0.4
comptime SM_W_BODY_POS: Float64 = 1.0
comptime SM_W_BODY_ORI: Float64 = 1.0
comptime SM_W_BODY_LIN: Float64 = 1.0
comptime SM_W_BODY_ANG: Float64 = 1.0
comptime SM_W_ROOT_POS: Float64 = 0.5
comptime SM_W_ROOT_ORI: Float64 = 0.5
comptime SM_W_ACTION_RATE: Float64 = -0.1
comptime SM_W_JOINT_LIMIT: Float64 = -2.0

# term order (compiled.py REWARD_TERM_NAMES)
comptime T_ROOT_POS: Int = 0
comptime T_ROOT_ORI: Int = 1
comptime T_BODY_POS: Int = 2
comptime T_BODY_ORI: Int = 3
comptime T_BODY_LIN: Int = 4
comptime T_BODY_ANG: Int = 5
comptime T_ACTION_RATE: Int = 6
comptime T_JOINT_LIMIT: Int = 7


struct SMHeadLayout[R: TrackingRobot]:
    """The heads' per-robot table (built by `tables.sm_tables`):
    [anchor slot | tracked FK rows (n_tracked) | q0 (n) | soft lo (n) | soft hi (n)].
    The anchor slot is the anchor's index in the tracked list."""
    comptime ANCHOR = 0
    comptime TRACKED = 1
    comptime Q0 = 1 + Self.R.N_TRACKED
    comptime LO = Self.Q0 + Self.R.NDOF
    comptime HI = Self.LO + Self.R.NDOF
    comptime SIZE = Self.HI + Self.R.NDOF
    # per-lane scratch: the FK block (pos 3 | quat 4 | lin 3 | ang 3 per body),
    # the same for its cotangents, then ∂/∂qpos, ∂/∂qvel
    comptime SCRATCH = 2 * SMDims[Self.R].FK_BLOCK + Self.R.NQ + Self.R.NV


# ── pieces ──────────────────────────────────────────────────────────────────
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
def qapply_inv[dt: DType](q: Q4[dt], v: Q4[dt]) -> Q4[dt]:
    """mjlab's `quat_apply_inverse`: v − w t + u × t, t = 2 u × v."""
    var u = Q4[dt](q[1], q[2], q[3], 0)
    var t = cross3(u, v) * 2
    return v - t * q[0] + cross3(u, t)


@always_inline
def qapply_inv_vjp_q[dt: DType](q: Q4[dt], v: Q4[dt], g: Q4[dt]) -> Q4[dt]:
    """∂/∂q of <g, qapply_inv(q, v)>."""
    var u = Q4[dt](q[1], q[2], q[3], 0)
    var t = cross3(u, v) * 2
    var gw = -dot3(g, t)
    var gt = g * (-q[0]) + cross3(g, u)
    var gu = cross3(t, g) + cross3(v, gt) * 2
    return Q4[dt](gw, gu[0], gu[1], gu[2])


@always_inline
def sm_term_features[dt: DType](s: FP[dt], rf_root_z: Scalar[dt], rf_root_q: Q4[dt]) -> Tuple[Scalar[dt], Scalar[dt]]:
    """`AnalyticHeads.termination_features`: [ref root z − z, (R_ref⁻¹ g)_z −
    (R⁻¹ g)_z], g = (0, 0, −1), R from the state's RAW root quaternion. The
    paper terminates at |height| > 0.25 m or |tilt| > 0.8."""
    var grav = Q4[dt](0, 0, -1, 0)
    return (rf_root_z - _at(s, 2), qapply_inv(rf_root_q, grav)[2] - qapply_inv(_q4(s, 3), grav)[2])


@always_inline
def sm_term_features_vjp[dt: DType](s: FP[dt], g0: Scalar[dt], g1: Scalar[dt], g_s: FP[dt]):
    """`g_s += ∂/∂s <(g0, g1), sm_term_features(s, ·)>` (ADDS)."""
    g_s[unsafe_offset=2] -= g0
    _add4(g_s, 3, qapply_inv_vjp_q(_q4(s, 3), Q4[dt](0, 0, -1, 0), Q4[dt](0, 0, -g1, 0)))


@always_inline
def sq_quat_err[dt: DType](d: Q4[dt]) -> Scalar[dt] where dt.is_floating_point():
    """|axis_angle_from_quat(d)|² (mjlab), d = q1 ⊗ q̄2."""
    var s = Scalar[dt](-1) if d[0] < 0 else Scalar[dt](1)
    var w = d[0] * s
    var m2 = d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
    var mag = sqrt(m2)
    var a = atan2(mag, w)
    var angle = a * 2
    if abs(angle) > Scalar[dt](1e-6):
        return angle * angle * (m2 + w * w)
    var k = Scalar[dt](0.5) - angle * angle / 48
    return m2 / (k * k)


@always_inline
def sq_quat_err_vjp[dt: DType](d: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    """∂/∂d of `sq_quat_err(d)`."""
    var s = Scalar[dt](-1) if d[0] < 0 else Scalar[dt](1)
    var w = d[0] * s
    var x = Q4[dt](d[1], d[2], d[3], 0) * s
    var m2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    var mag = sqrt(m2)
    var a = atan2(mag, w)
    var angle = a * 2
    var n2 = m2 + w * w
    var gw: Scalar[dt]
    var gx: Q4[dt]
    if abs(angle) > Scalar[dt](1e-6):
        # e = 4 a² n²:  ∂e/∂w = 8a(a w − |x|),  ∂e/∂x = 8a(w/|x| + a) x
        gw = a * 8 * (a * w - mag)
        gx = x * (a * 8 * (w / mag + a))
    else:
        # e = |x|² / k²,  k = ½ − angle²/48,  angle = 2 atan2(|x|, w)
        var k = Scalar[dt](0.5) - angle * angle / 48
        var k3 = k * k * k
        gx = x * (Scalar[dt](2) / (k * k) + mag * angle * w / (k3 * n2 * 6))
        gw = -(m2 * mag * angle) / (k3 * n2 * 6)
    return Q4[dt](gw * s, gx[0] * s, gx[1] * s, gx[2] * s)


@always_inline
def yaw_quat[dt: DType](q: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    """mjlab's `yaw_quat` (normalised; already unit)."""
    var yaw = atan2((q[0] * q[3] + q[1] * q[2]) * 2, Scalar[dt](1) - (q[2] * q[2] + q[3] * q[3]) * 2)
    var o = Q4[dt](cos(yaw / 2), 0, 0, sin(yaw / 2))
    var n = sqrt((o * o).reduce_add())
    return o / max(n, Scalar[dt](1e-9))


@always_inline
def yaw_quat_vjp[dt: DType](q: Q4[dt], g: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    """∂/∂q of <g, yaw_quat(q)> (the final normalise is of a unit vector along
    a unit-preserving curve: its projection changes nothing)."""
    var Y = (q[0] * q[3] + q[1] * q[2]) * 2
    var X = Scalar[dt](1) - (q[2] * q[2] + q[3] * q[3]) * 2
    var yaw = atan2(Y, X)
    var gyaw = (-sin(yaw / 2) * g[0] + cos(yaw / 2) * g[3]) / 2
    var r2 = X * X + Y * Y
    var gY = gyaw * X / r2
    var gX = -gyaw * Y / r2
    return Q4[dt](gY * 2 * q[3], gY * 2 * q[2], gY * 2 * q[1] - gX * 4 * q[2], gY * 2 * q[0] - gX * 4 * q[3])


@always_inline
def _qinv[dt: DType](q: Q4[dt]) -> Q4[dt]:
    """mjlab's `quat_inv`: q̄ / max(|q|², 1e-9)."""
    return qconj(q) / max((q * q).reduce_add(), Scalar[dt](1e-9))


@always_inline
def _exp_term[dt: DType, w: Float64, std: Float64](e: Scalar[dt]) -> Scalar[dt] where dt.is_floating_point():
    """`dt · w · exp(−e / std²)`; the constants are parameters so no double
    reaches a kernel (Metal has none)."""
    comptime k = SM_DT * w
    comptime s2 = std * std
    return Scalar[dt](k) * exp(-e / Scalar[dt](s2))


def sm_observation_lane[dt: DType, R: TrackingRobot](table: FP[dt], s: FP[dt], a: FP[dt], xi: FP[dt], obs: FP[dt]):
    """`analytic_observation(s, a, xi, q0)` (3n + 6): [q + bias − q0, q̇,
    ω_body, R⁻¹ g (RAW root quaternion), a] — the noise-free proprioception
    after taking `a` and landing in `s`."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    for j in range(D.NDOF):
        obs[unsafe_offset=j] = _at(s, 7 + j) + _at(xi, 1 + j) - _at(table, L.Q0 + j)
        obs[unsafe_offset=D.NDOF + j] = _at(s, D.NQ + 6 + j)
        obs[unsafe_offset=2 * D.NDOF + 6 + j] = _at(a, j)
    var pg = qapply_inv(_q4(s, 3), Q4[dt](0, 0, -1, 0))
    comptime for c in range(3):
        obs[unsafe_offset=2 * D.NDOF + c] = _at(s, D.NQ + 3 + c)
        obs[unsafe_offset=2 * D.NDOF + 3 + c] = pg[c]


def sm_observation_vjp_lane[dt: DType, R: TrackingRobot](s: FP[dt], g_obs: FP[dt], g_s: FP[dt], g_a: FP[dt]):
    """`g_s`, `g_a` += the observation's VJP (ADDS)."""
    comptime D = SMDims[R]
    var g_pg = Q4[dt](g_obs[unsafe_offset=2 * D.NDOF + 3], g_obs[unsafe_offset=2 * D.NDOF + 4], g_obs[unsafe_offset=2 * D.NDOF + 5], 0)
    _add4(g_s, 3, qapply_inv_vjp_q(_q4(s, 3), Q4[dt](0, 0, -1, 0), g_pg))
    for j in range(D.NDOF):
        g_s[unsafe_offset=7 + j] += g_obs[unsafe_offset=j]
        g_s[unsafe_offset=D.NQ + 6 + j] += g_obs[unsafe_offset=D.NDOF + j]
        g_a[unsafe_offset=j] += g_obs[unsafe_offset=2 * D.NDOF + 6 + j]
    comptime for c in range(3):
        g_s[unsafe_offset=D.NQ + 3 + c] += g_obs[unsafe_offset=2 * D.NDOF + c]


# ── forward ─────────────────────────────────────────────────────────────────
def sm_heads_lane[dt: DType, R: TrackingRobot](
    tree: FP[dt], table: FP[dt], s: FP[dt], a: FP[dt], a_prev: FP[dt], xi: FP[dt], rf: FP[dt],
    terms: FP[dt], feat: FP[dt], obs: FP[dt], scratch: FP[dt],
) -> Scalar[dt] where dt.is_floating_point():
    """Writes the 8 weighted terms, the 2 termination features and the
    observation of reaching `s` (= s_{t+1}) by `a` after `a_prev`, against
    the reference frame `rf`; returns the total reward. `scratch` holds
    `SMHeadLayout[R].SCRATCH` values; its first `FK_BLOCK` (the FK) are what
    `sm_heads_vjp_lane` reads back."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var pos = scratch
    var quat = scratch.unsafe_offset(D.NB * 3)
    var lin = scratch.unsafe_offset(D.NB * 7)
    var ang = scratch.unsafe_offset(D.NB * 10)
    sm_fk_lane[dt, D.NB](tree, s, s.unsafe_offset(D.NQ), pos, quat, lin, ang)

    var aslot = Int(_at(table, L.ANCHOR))
    var ai = Int(_at(table, L.TRACKED + aslot))
    var ra_p = _v3(pos, ai * 3)
    var ra_q = _q4(quat, ai * 4)
    var mb = aslot * D.REF_BODY
    var ma_p = _v3(rf, mb)
    var ma_q = _q4(rf, mb + 3)
    # re-anchoring: xy and yaw of the robot anchor, z / roll / pitch of the reference
    var dpos = Q4[dt](ra_p[0], ra_p[1], ma_p[2], 0)
    var dori = yaw_quat(qmul(ra_q, _qinv(ma_q)))

    var e_bp = Scalar[dt](0)
    var e_bo = Scalar[dt](0)
    var e_bl = Scalar[dt](0)
    var e_ba = Scalar[dt](0)
    for k in range(D.N_TRACKED):
        var b = Int(_at(table, L.TRACKED + k))
        var rb = k * D.REF_BODY
        var rel_p = dpos + qapply(dori, _v3(rf, rb) - ma_p)
        var rel_q = qmul(dori, _q4(rf, rb + 3))
        var dp = rel_p - _v3(pos, b * 3)
        e_bp += dot3(dp, dp)
        e_bo += sq_quat_err(qmul(rel_q, qconj(_q4(quat, b * 4))))
        var dl = _v3(rf, rb + 7) - _v3(lin, b * 3)
        e_bl += dot3(dl, dl)
        var da = _v3(rf, rb + 10) - _v3(ang, b * 3)
        e_ba += dot3(da, da)
    comptime INV_N = 1.0 / Float64(D.N_TRACKED)
    var inv_n = Scalar[dt](INV_N)
    var drp = ma_p - ra_p
    terms[unsafe_offset=T_ROOT_POS] = _exp_term[dt, SM_W_ROOT_POS, SM_STD_ROOT_POS](dot3(drp, drp))
    terms[unsafe_offset=T_ROOT_ORI] = _exp_term[dt, SM_W_ROOT_ORI, SM_STD_ROOT_ORI](sq_quat_err(qmul(ma_q, qconj(ra_q))))
    terms[unsafe_offset=T_BODY_POS] = _exp_term[dt, SM_W_BODY_POS, SM_STD_BODY_POS](e_bp * inv_n)
    terms[unsafe_offset=T_BODY_ORI] = _exp_term[dt, SM_W_BODY_ORI, SM_STD_BODY_ORI](e_bo * inv_n)
    terms[unsafe_offset=T_BODY_LIN] = _exp_term[dt, SM_W_BODY_LIN, SM_STD_BODY_LIN](e_bl * inv_n)
    terms[unsafe_offset=T_BODY_ANG] = _exp_term[dt, SM_W_BODY_ANG, SM_STD_BODY_ANG](e_ba * inv_n)
    var ar = Scalar[dt](0)
    var jl = Scalar[dt](0)
    for j in range(D.NDOF):
        var d = _at(a, j) - _at(a_prev, j)
        ar += d * d
        var q = _at(s, 7 + j)
        jl += max(_at(table, L.LO + j) - q, Scalar[dt](0)) + max(q - _at(table, L.HI + j), Scalar[dt](0))
    terms[unsafe_offset=T_ACTION_RATE] = Scalar[dt](SM_DT * SM_W_ACTION_RATE) * ar
    terms[unsafe_offset=T_JOINT_LIMIT] = Scalar[dt](SM_DT * SM_W_JOINT_LIMIT) * jl
    var total = terms[unsafe_offset=0]
    for k in range(1, SM_N_TERMS):
        total += terms[unsafe_offset=k]

    # termination features (the state's root pose, not the FK's)
    var grav = Q4[dt](0, 0, -1, 0)
    var sq = _q4(s, 3)
    var rr = D.N_TRACKED * D.REF_BODY
    var tf = sm_term_features(s, _at(rf, rr + 2), _q4(rf, rr + 3))
    feat[unsafe_offset=0] = tf[0]
    feat[unsafe_offset=1] = tf[1]

    sm_observation_lane[dt, R](table, s, a, xi, obs)
    return total


# ── VJP ─────────────────────────────────────────────────────────────────────
def sm_heads_vjp_lane[dt: DType, R: TrackingRobot](
    tree: FP[dt], table: FP[dt], s: FP[dt], a: FP[dt], a_prev: FP[dt], rf: FP[dt],
    g_r: Scalar[dt], g_feat: FP[dt], g_obs: FP[dt], scratch: FP[dt],
    g_s: FP[dt], g_a: FP[dt], g_ap: FP[dt],
) where dt.is_floating_point():
    """`g_s` (S), `g_a` (n), `g_ap` (n) ← the VJP of `sm_heads_lane` for the
    cotangents `g_r` (total reward), `g_feat` (2) and `g_obs` (3n + 6). `scratch`
    must hold that forward call's FK (its first `FK_BLOCK` values); the
    rest of it is overwritten. Outputs are written, not added to."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var pos = scratch
    var quat = scratch.unsafe_offset(D.NB * 3)
    var lin = scratch.unsafe_offset(D.NB * 7)
    var ang = scratch.unsafe_offset(D.NB * 10)
    var gpos = scratch.unsafe_offset(D.FK_BLOCK)
    var gquat = gpos.unsafe_offset(D.NB * 3)
    var glin = gpos.unsafe_offset(D.NB * 7)
    var gang = gpos.unsafe_offset(D.NB * 10)
    var gqpos = gpos.unsafe_offset(D.FK_BLOCK)
    var gqvel = gqpos.unsafe_offset(D.NQ)
    for i in range(D.FK_BLOCK):
        gpos[unsafe_offset=i] = 0
    for i in range(D.S):
        g_s[unsafe_offset=i] = 0
    for j in range(D.NDOF):
        g_a[unsafe_offset=j] = 0
        g_ap[unsafe_offset=j] = 0

    # ── the reward ──
    var aslot = Int(_at(table, L.ANCHOR))
    var ai = Int(_at(table, L.TRACKED + aslot))
    var ra_p = _v3(pos, ai * 3)
    var ra_q = _q4(quat, ai * 4)
    var mb = aslot * D.REF_BODY
    var ma_p = _v3(rf, mb)
    var ma_q = _q4(rf, mb + 3)
    var dpos = Q4[dt](ra_p[0], ra_p[1], ma_p[2], 0)
    var minv = _qinv(ma_q)
    var yin = qmul(ra_q, minv)
    var dori = yaw_quat(yin)
    # the errors again (the exp terms' values)
    var e_bp = Scalar[dt](0)
    var e_bo = Scalar[dt](0)
    var e_bl = Scalar[dt](0)
    var e_ba = Scalar[dt](0)
    for k in range(D.N_TRACKED):
        var b = Int(_at(table, L.TRACKED + k))
        var rb = k * D.REF_BODY
        var dp = dpos + qapply(dori, _v3(rf, rb) - ma_p) - _v3(pos, b * 3)
        e_bp += dot3(dp, dp)
        e_bo += sq_quat_err(qmul(qmul(dori, _q4(rf, rb + 3)), qconj(_q4(quat, b * 4))))
        var dl = _v3(rf, rb + 7) - _v3(lin, b * 3)
        e_bl += dot3(dl, dl)
        var da = _v3(rf, rb + 10) - _v3(ang, b * 3)
        e_ba += dot3(da, da)
    comptime INV_N = 1.0 / Float64(D.N_TRACKED)
    var inv_n = Scalar[dt](INV_N)
    var drp = ma_p - ra_p
    var d_root = qmul(ma_q, qconj(ra_q))
    # ∂r/∂e = −term / σ²
    var ge_rp = -g_r * _exp_term[dt, SM_W_ROOT_POS, SM_STD_ROOT_POS](dot3(drp, drp)) / Scalar[dt](SM_STD_ROOT_POS * SM_STD_ROOT_POS)
    var ge_ro = -g_r * _exp_term[dt, SM_W_ROOT_ORI, SM_STD_ROOT_ORI](sq_quat_err(d_root)) / Scalar[dt](SM_STD_ROOT_ORI * SM_STD_ROOT_ORI)
    var ge_bp = -g_r * _exp_term[dt, SM_W_BODY_POS, SM_STD_BODY_POS](e_bp * inv_n) / Scalar[dt](SM_STD_BODY_POS * SM_STD_BODY_POS) * inv_n
    var ge_bo = -g_r * _exp_term[dt, SM_W_BODY_ORI, SM_STD_BODY_ORI](e_bo * inv_n) / Scalar[dt](SM_STD_BODY_ORI * SM_STD_BODY_ORI) * inv_n
    var ge_bl = -g_r * _exp_term[dt, SM_W_BODY_LIN, SM_STD_BODY_LIN](e_bl * inv_n) / Scalar[dt](SM_STD_BODY_LIN * SM_STD_BODY_LIN) * inv_n
    var ge_ba = -g_r * _exp_term[dt, SM_W_BODY_ANG, SM_STD_BODY_ANG](e_ba * inv_n) / Scalar[dt](SM_STD_BODY_ANG * SM_STD_BODY_ANG) * inv_n

    var g_rap = drp * (ge_rp * -2)                       # e_root_pos = |ma − ra|²
    # e_root_ori = sqerr(ma ⊗ r̄a): ∂/∂ra = conj(conj(ma) ⊗ g_d)
    var g_raq = qconj(qmul(qconj(ma_q), sq_quat_err_vjp(d_root) * ge_ro))
    var g_dori = Q4[dt](0)
    var g_dpos = Q4[dt](0)
    for k in range(D.N_TRACKED):
        var b = Int(_at(table, L.TRACKED + k))
        var rb = k * D.REF_BODY
        var off = _v3(rf, rb) - ma_p
        var rel_p = dpos + qapply(dori, off)
        var rq = _q4(rf, rb + 3)
        var rel_q = qmul(dori, rq)
        var bq = _q4(quat, b * 4)
        # body pos
        var gdp = (rel_p - _v3(pos, b * 3)) * (ge_bp * 2)
        _add3(gpos, b * 3, -gdp)
        g_dpos += gdp
        g_dori += qapply_vjp(dori, off, gdp)[0]
        # body ori: d = rel_q ⊗ b̄q
        var gd = sq_quat_err_vjp(qmul(rel_q, qconj(bq))) * ge_bo
        g_dori += qmul(qmul(gd, bq), qconj(rq))         # ∂/∂rel_q = gd ⊗ bq ; rel_q = dori ⊗ rq
        _add4(gquat, b * 4, qconj(qmul(qconj(rel_q), gd)))
        # body velocities
        _add3(glin, b * 3, (_v3(rf, rb + 7) - _v3(lin, b * 3)) * (ge_bl * -2))
        _add3(gang, b * 3, (_v3(rf, rb + 10) - _v3(ang, b * 3)) * (ge_ba * -2))
    # dpos = (ra.x, ra.y, ma.z) ; dori = yaw(ra_q ⊗ minv)
    g_rap += Q4[dt](g_dpos[0], g_dpos[1], 0, 0)
    g_raq += qmul(yaw_quat_vjp(yin, g_dori), qconj(minv))
    _add3(gpos, ai * 3, g_rap)
    _add4(gquat, ai * 4, g_raq)

    # action rate and joint limits
    for j in range(D.NDOF):
        var d = (_at(a, j) - _at(a_prev, j)) * Scalar[dt](2 * SM_DT * SM_W_ACTION_RATE) * g_r
        g_a[unsafe_offset=j] = d
        g_ap[unsafe_offset=j] = -d
        var q = _at(s, 7 + j)
        var gj = Scalar[dt](0)
        if _at(table, L.LO + j) - q > 0:
            gj -= 1
        if q - _at(table, L.HI + j) > 0:
            gj += 1
        g_s[unsafe_offset=7 + j] = gj * Scalar[dt](SM_DT * SM_W_JOINT_LIMIT) * g_r

    # through the FK
    sm_fk_vjp_lane[dt, D.NB, D.NQ, D.NV](tree, s, s.unsafe_offset(D.NQ), pos, quat, ang, gpos, gquat, glin, gang, gqpos, gqvel)
    for i in range(D.NQ):
        g_s[unsafe_offset=i] += gqpos[unsafe_offset=i]
    for i in range(D.NV):
        g_s[unsafe_offset=D.NQ + i] += gqvel[unsafe_offset=i]

    # ── termination features and observation: the raw root quaternion ──
    sm_term_features_vjp(s, g_feat[unsafe_offset=0], g_feat[unsafe_offset=1], g_s)
    sm_observation_vjp_lane[dt, R](s, g_obs, g_s, g_a)
