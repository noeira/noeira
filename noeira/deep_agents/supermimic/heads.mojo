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

Layouts (per lane):
    s       71   qpos 36 | qvel 35 (root angular velocity in the BODY frame)
    rf    189   14 tracked bodies × [pos 3, quat 4, lin vel 3, ang vel 3]
                 (world), then the reference root pos 3, quat 4
    xi      33   sim context; [1:30] is the encoder bias
    terms    8   REWARD_TERM_NAMES' order, weighted and × dt
"""

from std.math import atan2, cos, exp, sin, sqrt

from noeira.deep_agents.supermimic.g1_fk import (
    FP, G1_FK_NB, G1_FK_NQ, G1_FK_NV, Q4, cross3, dot3, g1_fk_lane, g1_fk_vjp_lane, qapply,
    qapply_vjp, qconj, qmul,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_priv_obs import g1_skeleton_body
from noeira.physics3d.gpu.constants import JOINT_IDX_RANGE_MAX, JOINT_IDX_RANGE_MIN, MODEL_JOINT_SIZE


comptime SM_N_DOF: Int = 29
comptime SM_STATE_DIM: Int = 71
comptime SM_OBS_DIM: Int = 93
comptime SM_XI_DIM: Int = 33
comptime SM_N_TRACKED: Int = 14
comptime SM_REF_BODY: Int = 13
comptime SM_REF_DIM: Int = SM_N_TRACKED * SM_REF_BODY + 7
comptime SM_N_TERMS: Int = 8
comptime SM_ANCHOR_SLOT: Int = 7         # torso_link in the tracked list

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
comptime SM_SOFT_LIMIT_FACTOR: Float64 = 0.9   # mjlab's G1 `soft_joint_pos_limit_factor`

# term order (compiled.py REWARD_TERM_NAMES)
comptime T_ROOT_POS: Int = 0
comptime T_ROOT_ORI: Int = 1
comptime T_BODY_POS: Int = 2
comptime T_BODY_ORI: Int = 3
comptime T_BODY_LIN: Int = 4
comptime T_BODY_ANG: Int = 5
comptime T_ACTION_RATE: Int = 6
comptime T_JOINT_LIMIT: Int = 7

# per-lane FK scratch: pos 3 | quat 4 | lin 3 | ang 3 per body, then the same for its cotangents
comptime SM_FK_BLOCK: Int = G1_FK_NB * 13
comptime SM_SCRATCH: Int = 2 * SM_FK_BLOCK + G1_FK_NQ + G1_FK_NV


def sm_tracked_skeleton_index(k: Int) -> Int:
    """Tracked body k (the reproduction's TRACKED_BODIES order) → its index in
    the reference's 30 `body_names` (`g1_skeleton_body`'s order): pelvis,
    left hip roll / knee / ankle roll, the same on the right, torso, left
    shoulder roll / elbow / wrist yaw, the same on the right."""
    var t: List[Int] = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]
    return t[k]


def sm_tracked_fk_index(k: Int) -> Int:
    """Tracked body k → its row in `g1_fk_lane`'s outputs (body id − 1)."""
    return g1_skeleton_body(sm_tracked_skeleton_index(k)) - 1


struct G1HeadsTable[dt: DType](Movable):
    """Per-model constants: tracked FK rows (14), q0 (29), soft lo (29), soft hi (29)."""
    var data: List[Scalar[Self.dt]]

    def __init__(out self, var data: List[Scalar[Self.dt]]):
        self.data = data^

    def __init__(out self, *, deinit move: Self):
        self.data = move.data^

    def ptr(mut self) -> FP[Self.dt]:
        return self.data.unsafe_ptr().as_unsafe_any_origin()


comptime HT_TRACKED: Int = 0
comptime HT_Q0: Int = SM_N_TRACKED
comptime HT_LO: Int = HT_Q0 + SM_N_DOF
comptime HT_HI: Int = HT_LO + SM_N_DOF
comptime HT_SIZE: Int = HT_HI + SM_N_DOF


def g1_soft_limits(env: UnitreeG1[DType.float64], j: Int, factor: Float64 = SM_SOFT_LIMIT_FACTOR) -> Tuple[Float64, Float64]:
    """mjlab's soft joint limits of hinge `j` (0..28): the range shrunk about
    its midpoint by `factor` (`entity.py`, `soft_joint_pos_limit_factor`)."""
    var o = (j + 1) * MODEL_JOINT_SIZE
    var lo = Float64(env.mf.joints.data[o + JOINT_IDX_RANGE_MIN])
    var hi = Float64(env.mf.joints.data[o + JOINT_IDX_RANGE_MAX])
    var mid = (lo + hi) / 2.0
    var half = 0.5 * (hi - lo) * factor
    return (mid - half, mid + half)


def g1_heads_table[dt: DType](env: UnitreeG1[DType.float64], q0: List[Float64]) raises -> G1HeadsTable[dt]:
    """The table for our G1 with default joint positions `q0` (29)."""
    if len(q0) != SM_N_DOF:
        raise Error("g1_heads_table: q0 has " + String(len(q0)) + " entries, want 29")
    var t = List[Scalar[dt]](length=HT_SIZE, fill=Scalar[dt](0))
    for k in range(SM_N_TRACKED):
        t[HT_TRACKED + k] = Scalar[dt](sm_tracked_fk_index(k))
    for j in range(SM_N_DOF):
        var lim = g1_soft_limits(env, j)
        t[HT_Q0 + j] = Scalar[dt](q0[j])
        t[HT_LO + j] = Scalar[dt](lim[0])
        t[HT_HI + j] = Scalar[dt](lim[1])
    return G1HeadsTable[dt](t^)


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
def sm_term_features[dt: DType](s: FP[dt], rf_root_z: Scalar[dt], rf_root_q: Q4[dt]) -> SIMD[dt, 2]:
    """`AnalyticHeads.termination_features`: [ref root z − z, (R_ref⁻¹ g)_z −
    (R⁻¹ g)_z], g = (0, 0, −1), R from the state's RAW root quaternion. The
    paper terminates at |height| > 0.25 m or |tilt| > 0.8."""
    var grav = Q4[dt](0, 0, -1, 0)
    return SIMD[dt, 2](rf_root_z - _at(s, 2), qapply_inv(rf_root_q, grav)[2] - qapply_inv(_q4(s, 3), grav)[2])


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
def _exp_term[dt: DType](w: Float64, std: Float64, e: Scalar[dt]) -> Scalar[dt] where dt.is_floating_point():
    return Scalar[dt](SM_DT * w) * exp(-e / Scalar[dt](std * std))


def sm_observation_lane[dt: DType](table: FP[dt], s: FP[dt], a: FP[dt], xi: FP[dt], obs: FP[dt]):
    """`analytic_observation(s, a, xi, q0)` (93): [q + bias − q0, q̇, ω_body,
    R⁻¹ g (RAW root quaternion), a] — the noise-free proprioception after
    taking `a` and landing in `s`."""
    for j in range(SM_N_DOF):
        obs[unsafe_offset=j] = _at(s, 7 + j) + _at(xi, 1 + j) - _at(table, HT_Q0 + j)
        obs[unsafe_offset=29 + j] = _at(s, 42 + j)
        obs[unsafe_offset=64 + j] = _at(a, j)
    var pg = qapply_inv(_q4(s, 3), Q4[dt](0, 0, -1, 0))
    for c in range(3):
        obs[unsafe_offset=58 + c] = _at(s, 39 + c)
        obs[unsafe_offset=61 + c] = pg[c]


def sm_observation_vjp_lane[dt: DType](s: FP[dt], g_obs: FP[dt], g_s: FP[dt], g_a: FP[dt]):
    """`g_s`, `g_a` += the observation's VJP (ADDS)."""
    var g_pg = Q4[dt](g_obs[unsafe_offset=61], g_obs[unsafe_offset=62], g_obs[unsafe_offset=63], 0)
    _add4(g_s, 3, qapply_inv_vjp_q(_q4(s, 3), Q4[dt](0, 0, -1, 0), g_pg))
    for j in range(SM_N_DOF):
        g_s[unsafe_offset=7 + j] += g_obs[unsafe_offset=j]
        g_s[unsafe_offset=42 + j] += g_obs[unsafe_offset=29 + j]
        g_a[unsafe_offset=j] += g_obs[unsafe_offset=64 + j]
    for c in range(3):
        g_s[unsafe_offset=39 + c] += g_obs[unsafe_offset=58 + c]


# ── forward ─────────────────────────────────────────────────────────────────
def g1_heads_lane[dt: DType](
    tree: FP[dt], table: FP[dt], s: FP[dt], a: FP[dt], a_prev: FP[dt], xi: FP[dt], rf: FP[dt],
    terms: FP[dt], feat: FP[dt], obs: FP[dt], scratch: FP[dt],
) -> Scalar[dt] where dt.is_floating_point():
    """Writes the 8 weighted terms, the 2 termination features and the 93-d
    observation of reaching `s` (= s_{t+1}) by `a` after `a_prev`, against
    the reference frame `rf`; returns the total reward. `scratch` holds
    `SM_SCRATCH` values; its first `SM_FK_BLOCK` (the FK) are what
    `g1_heads_vjp_lane` reads back."""
    var pos = scratch
    var quat = scratch.unsafe_offset(G1_FK_NB * 3)
    var lin = scratch.unsafe_offset(G1_FK_NB * 7)
    var ang = scratch.unsafe_offset(G1_FK_NB * 10)
    g1_fk_lane[dt](tree, s, s.unsafe_offset(G1_FK_NQ), pos, quat, lin, ang)

    var ai = Int(_at(table, HT_TRACKED + SM_ANCHOR_SLOT))
    var ra_p = _v3(pos, ai * 3)
    var ra_q = _q4(quat, ai * 4)
    var mb = SM_ANCHOR_SLOT * SM_REF_BODY
    var ma_p = _v3(rf, mb)
    var ma_q = _q4(rf, mb + 3)
    # re-anchoring: xy and yaw of the robot anchor, z / roll / pitch of the reference
    var dpos = Q4[dt](ra_p[0], ra_p[1], ma_p[2], 0)
    var dori = yaw_quat(qmul(ra_q, _qinv(ma_q)))

    var e_bp = Scalar[dt](0)
    var e_bo = Scalar[dt](0)
    var e_bl = Scalar[dt](0)
    var e_ba = Scalar[dt](0)
    for k in range(SM_N_TRACKED):
        var b = Int(_at(table, HT_TRACKED + k))
        var rb = k * SM_REF_BODY
        var rel_p = dpos + qapply(dori, _v3(rf, rb) - ma_p)
        var rel_q = qmul(dori, _q4(rf, rb + 3))
        var dp = rel_p - _v3(pos, b * 3)
        e_bp += dot3(dp, dp)
        e_bo += sq_quat_err(qmul(rel_q, qconj(_q4(quat, b * 4))))
        var dl = _v3(rf, rb + 7) - _v3(lin, b * 3)
        e_bl += dot3(dl, dl)
        var da = _v3(rf, rb + 10) - _v3(ang, b * 3)
        e_ba += dot3(da, da)
    var inv_n = Scalar[dt](1.0 / Float64(SM_N_TRACKED))
    var drp = ma_p - ra_p
    terms[unsafe_offset=T_ROOT_POS] = _exp_term(SM_W_ROOT_POS, SM_STD_ROOT_POS, dot3(drp, drp))
    terms[unsafe_offset=T_ROOT_ORI] = _exp_term(SM_W_ROOT_ORI, SM_STD_ROOT_ORI, sq_quat_err(qmul(ma_q, qconj(ra_q))))
    terms[unsafe_offset=T_BODY_POS] = _exp_term(SM_W_BODY_POS, SM_STD_BODY_POS, e_bp * inv_n)
    terms[unsafe_offset=T_BODY_ORI] = _exp_term(SM_W_BODY_ORI, SM_STD_BODY_ORI, e_bo * inv_n)
    terms[unsafe_offset=T_BODY_LIN] = _exp_term(SM_W_BODY_LIN, SM_STD_BODY_LIN, e_bl * inv_n)
    terms[unsafe_offset=T_BODY_ANG] = _exp_term(SM_W_BODY_ANG, SM_STD_BODY_ANG, e_ba * inv_n)
    var ar = Scalar[dt](0)
    var jl = Scalar[dt](0)
    for j in range(SM_N_DOF):
        var d = _at(a, j) - _at(a_prev, j)
        ar += d * d
        var q = _at(s, 7 + j)
        jl += max(_at(table, HT_LO + j) - q, Scalar[dt](0)) + max(q - _at(table, HT_HI + j), Scalar[dt](0))
    terms[unsafe_offset=T_ACTION_RATE] = Scalar[dt](SM_DT * SM_W_ACTION_RATE) * ar
    terms[unsafe_offset=T_JOINT_LIMIT] = Scalar[dt](SM_DT * SM_W_JOINT_LIMIT) * jl
    var total = terms[unsafe_offset=0]
    for k in range(1, SM_N_TERMS):
        total += terms[unsafe_offset=k]

    # termination features (the state's root pose, not the FK's)
    var grav = Q4[dt](0, 0, -1, 0)
    var sq = _q4(s, 3)
    var rr = SM_N_TRACKED * SM_REF_BODY
    var tf = sm_term_features(s, _at(rf, rr + 2), _q4(rf, rr + 3))
    feat[unsafe_offset=0] = tf[0]
    feat[unsafe_offset=1] = tf[1]

    sm_observation_lane(table, s, a, xi, obs)
    return total


# ── VJP ─────────────────────────────────────────────────────────────────────
def g1_heads_vjp_lane[dt: DType](
    tree: FP[dt], table: FP[dt], s: FP[dt], a: FP[dt], a_prev: FP[dt], rf: FP[dt],
    g_r: Scalar[dt], g_feat: FP[dt], g_obs: FP[dt], scratch: FP[dt],
    g_s: FP[dt], g_a: FP[dt], g_ap: FP[dt],
) where dt.is_floating_point():
    """`g_s` (71), `g_a` (29), `g_ap` (29) ← the VJP of `g1_heads_lane` for the
    cotangents `g_r` (total reward), `g_feat` (2) and `g_obs` (93). `scratch`
    must hold that forward call's FK (its first `SM_FK_BLOCK` values); the
    rest of it is overwritten. Outputs are written, not added to."""
    var pos = scratch
    var quat = scratch.unsafe_offset(G1_FK_NB * 3)
    var lin = scratch.unsafe_offset(G1_FK_NB * 7)
    var ang = scratch.unsafe_offset(G1_FK_NB * 10)
    var gpos = scratch.unsafe_offset(SM_FK_BLOCK)
    var gquat = gpos.unsafe_offset(G1_FK_NB * 3)
    var glin = gpos.unsafe_offset(G1_FK_NB * 7)
    var gang = gpos.unsafe_offset(G1_FK_NB * 10)
    var gqpos = gpos.unsafe_offset(SM_FK_BLOCK)
    var gqvel = gqpos.unsafe_offset(G1_FK_NQ)
    for i in range(SM_FK_BLOCK):
        gpos[unsafe_offset=i] = 0
    for i in range(SM_STATE_DIM):
        g_s[unsafe_offset=i] = 0
    for j in range(SM_N_DOF):
        g_a[unsafe_offset=j] = 0
        g_ap[unsafe_offset=j] = 0

    # ── the reward ──
    var ai = Int(_at(table, HT_TRACKED + SM_ANCHOR_SLOT))
    var ra_p = _v3(pos, ai * 3)
    var ra_q = _q4(quat, ai * 4)
    var mb = SM_ANCHOR_SLOT * SM_REF_BODY
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
    for k in range(SM_N_TRACKED):
        var b = Int(_at(table, HT_TRACKED + k))
        var rb = k * SM_REF_BODY
        var dp = dpos + qapply(dori, _v3(rf, rb) - ma_p) - _v3(pos, b * 3)
        e_bp += dot3(dp, dp)
        e_bo += sq_quat_err(qmul(qmul(dori, _q4(rf, rb + 3)), qconj(_q4(quat, b * 4))))
        var dl = _v3(rf, rb + 7) - _v3(lin, b * 3)
        e_bl += dot3(dl, dl)
        var da = _v3(rf, rb + 10) - _v3(ang, b * 3)
        e_ba += dot3(da, da)
    var inv_n = Scalar[dt](1.0 / Float64(SM_N_TRACKED))
    var drp = ma_p - ra_p
    var d_root = qmul(ma_q, qconj(ra_q))
    # ∂r/∂e = −term / σ²
    var ge_rp = -g_r * _exp_term(SM_W_ROOT_POS, SM_STD_ROOT_POS, dot3(drp, drp)) / Scalar[dt](SM_STD_ROOT_POS * SM_STD_ROOT_POS)
    var ge_ro = -g_r * _exp_term(SM_W_ROOT_ORI, SM_STD_ROOT_ORI, sq_quat_err(d_root)) / Scalar[dt](SM_STD_ROOT_ORI * SM_STD_ROOT_ORI)
    var ge_bp = -g_r * _exp_term(SM_W_BODY_POS, SM_STD_BODY_POS, e_bp * inv_n) / Scalar[dt](SM_STD_BODY_POS * SM_STD_BODY_POS) * inv_n
    var ge_bo = -g_r * _exp_term(SM_W_BODY_ORI, SM_STD_BODY_ORI, e_bo * inv_n) / Scalar[dt](SM_STD_BODY_ORI * SM_STD_BODY_ORI) * inv_n
    var ge_bl = -g_r * _exp_term(SM_W_BODY_LIN, SM_STD_BODY_LIN, e_bl * inv_n) / Scalar[dt](SM_STD_BODY_LIN * SM_STD_BODY_LIN) * inv_n
    var ge_ba = -g_r * _exp_term(SM_W_BODY_ANG, SM_STD_BODY_ANG, e_ba * inv_n) / Scalar[dt](SM_STD_BODY_ANG * SM_STD_BODY_ANG) * inv_n

    var g_rap = drp * (ge_rp * -2)                       # e_root_pos = |ma − ra|²
    # e_root_ori = sqerr(ma ⊗ r̄a): ∂/∂ra = conj(conj(ma) ⊗ g_d)
    var g_raq = qconj(qmul(qconj(ma_q), sq_quat_err_vjp(d_root) * ge_ro))
    var g_dori = Q4[dt](0)
    var g_dpos = Q4[dt](0)
    for k in range(SM_N_TRACKED):
        var b = Int(_at(table, HT_TRACKED + k))
        var rb = k * SM_REF_BODY
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
    for j in range(SM_N_DOF):
        var d = (_at(a, j) - _at(a_prev, j)) * Scalar[dt](2 * SM_DT * SM_W_ACTION_RATE) * g_r
        g_a[unsafe_offset=j] = d
        g_ap[unsafe_offset=j] = -d
        var q = _at(s, 7 + j)
        var gj = Scalar[dt](0)
        if _at(table, HT_LO + j) - q > 0:
            gj -= 1
        if q - _at(table, HT_HI + j) > 0:
            gj += 1
        g_s[unsafe_offset=7 + j] = gj * Scalar[dt](SM_DT * SM_W_JOINT_LIMIT) * g_r

    # through the FK
    g1_fk_vjp_lane[dt](tree, s, s.unsafe_offset(G1_FK_NQ), pos, quat, ang, gpos, gquat, glin, gang, gqpos, gqvel)
    for i in range(G1_FK_NQ):
        g_s[unsafe_offset=i] += gqpos[unsafe_offset=i]
    for i in range(G1_FK_NV):
        g_s[unsafe_offset=G1_FK_NQ + i] += gqvel[unsafe_offset=i]

    # ── termination features and observation: the raw root quaternion ──
    sm_term_features_vjp(s, g_feat[unsafe_offset=0], g_feat[unsafe_offset=1], g_s)
    sm_observation_vjp_lane(s, g_obs, g_s, g_a)
