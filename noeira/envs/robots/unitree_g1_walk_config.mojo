"""Unitree G1 walker — velocity-command tracking with a trained stop (L0).

    obs     = [root linear velocity, pelvis frame (3), root angular velocity,
               pelvis frame (3), projected gravity (3), command vx vy wz (3),
               q - q_default (29), qdot (29)]                            (70)
              The driver appends the last action (29) — `meta` has no room
              for it (57 words).
    action  = 29 PD targets, `target = q_default + 0.25 clip(a, +-2)`, then
              BFM-Zero's torque law every substep (kp / kd / effort and the
              ctrlrange clip of `unitree_g1_config.mojo`)
    control = 50 Hz over 200 Hz physics, as BFM-Zero
    command = (vx, vy, wz) per lane in `meta`, redrawn every 10 s; 20 % of
              draws are exactly zero — standing is a trained behaviour
    reward  = Playground's G1 joystick terms + RoboParty's standing terms
              + an alive bonus, each logged separately (`g1_walk_terms`),
              summed x dt
    end     = fallen (pelvis below 0.45 m or tilted past 60 deg), a non-foot
              body on the floor, or the two lower legs touching

Plan and sources: `docs/G1_WALKER_PLAN.md`. Recipe from MuJoCo Playground
`locomotion/g1/joystick.py` (Apache-2.0) and RoboParty `robolab/tasks/
direct/interrupt` (BSD-3-Clause); the weights and deviations are listed in
the plan, §4.

⚠ SAME BODY, SAME CONTROLLER AS BFM-ZERO. The model is `unitree_g1.xml`
and the PD law is `g1_pd_torque`; only the ACTION MAP differs
(`g1_walk_target` vs BFM's `g1_pd_target`). The room hands control between
the two policies on one body (plan L3), which a second G1 model would turn
into a sim-to-sim problem as well.

⚠ ONE IMPLEMENTATION, TWO CALLERS. Every hook here is the GPU hook; the
host evaluates the same functions over `[1, N]` views of a CPU env's
`Data` (`g1_walk_host.mojo`, the pattern of `tasks/host_reward.mojo`).
The CPU hooks the trait requires are the deterministic stand reset and the
torque law; the CPU reward hook reports termination only — reward terms
need the per-lane feet state in `meta`, which the trait's CPU hook cannot
write (`d` is borrowed immutably). Use `g1_walk_host_step` for rewards on
the host.

⚠ PER-LANE STATE IN `meta[TASK_PARAM_0..9]` (`G1W_*` below). `_reset_env_lane`
zeroes only the step counter; `init_qpos_gpu` writes every word this config
reads, so no word survives an episode boundary by accident.

⚠ A NEGATIVE COMMAND TIMER FREEZES THE COMMAND. Training draws; an eval or
the room writes `cmd` and `G1W_CMD_TIMER = -1` and the hook never redraws.
"""

from std.math import exp, sqrt
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, DimsLike
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    META_IDX_STEP_COUNT,
    META_IDX_TASK_PARAM_0,
    MODEL_JOINT_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_CURRICULUM_SIZE,
    METADATA_SIZE,
)
from ..phyics3d_env_config import Phyics3dEnvConfig
from ..dm_control.dtype_math import sin_dt, cos_dt
from ..dm_control.gpu_reset import reset_seed
from .unitree_g1_xml import (
    UNITREE_G1_NMESH_VERTS,
    ROOT_QPOS_SIZE,
    ROOT_QVEL_SIZE,
    TORSO_BODY_IDX,
)
from .unitree_g1_config import g1_pd_torque, _clamp, _projected_gravity
from .unitree_g1_pd import (
    G1_N_DOF,
    G1_INIT_ROOT_Z,
    G1_SIM_TIMESTEP,
    G1_CONTROL_DECIMATION,
    g1_default_pos,
    g1_pos_lower,
    g1_pos_upper,
)


# ── observation ───────────────────────────────────────────────────────────

comptime G1_WALK_OBS_DIM: Int = 70
comptime G1_WALK_OBS_LINVEL: Int = 0
comptime G1_WALK_OBS_ANGVEL: Int = 3
comptime G1_WALK_OBS_GRAVITY: Int = 6
comptime G1_WALK_OBS_CMD: Int = 9
comptime G1_WALK_OBS_Q: Int = 12
comptime G1_WALK_OBS_QD: Int = 12 + G1_N_DOF

# Playground's uniform noise half-widths (`joystick.py` `noise_config`).
comptime G1_WALK_NOISE_LINVEL: Float64 = 0.1
comptime G1_WALK_NOISE_GYRO: Float64 = 0.2
comptime G1_WALK_NOISE_GRAVITY: Float64 = 0.05
comptime G1_WALK_NOISE_Q: Float64 = 0.03
comptime G1_WALK_NOISE_QD: Float64 = 1.5

# ── action ────────────────────────────────────────────────────────────────

comptime G1_WALK_ACTION_SCALE: Float64 = 0.25
"""rad per unit action — RoboParty's `action_scale` (Playground: 0.5)."""
comptime G1_WALK_ACTION_CLIP: Float64 = 2.0
"""A Gaussian policy is unbounded; +-2 caps a target at +-0.5 rad off
default — Playground's range (tanh x 0.5). ⚠ Run s2 had +-4 (+-1 rad) and
learned a bang-bang standing policy that shook the torso at ~3.7 rad/s:
the range is part of what keeps the actions smooth. The driver's
`action_scale` must equal this (it clamps before the obs history)."""

# ── commands ──────────────────────────────────────────────────────────────

comptime G1_WALK_VX_MIN: Float64 = -0.6
comptime G1_WALK_VX_MAX: Float64 = 1.0
comptime G1_WALK_VY_MAX: Float64 = 0.5
comptime G1_WALK_WZ_MAX: Float64 = 1.0
comptime G1_WALK_P_STAND: Float64 = 0.2
comptime G1_WALK_CMD_PERIOD: Int = 500
"""Control steps between command draws (10 s, RoboParty)."""
comptime G1_WALK_CMD_ZERO: Float64 = 0.01
"""`|vx| + |vy| + |wz|` below this IS the standing command (RoboParty)."""

# ── pushes (training only) ────────────────────────────────────────────────

comptime G1_WALK_PUSH_MIN_STEPS: Int = 250
comptime G1_WALK_PUSH_MAX_STEPS: Int = 500
comptime G1_WALK_PUSH_MIN: Float64 = 0.1
comptime G1_WALK_PUSH_MAX: Float64 = 2.0

# ── episode, termination ──────────────────────────────────────────────────

comptime G1_WALK_MAX_STEPS: Int = 1000
comptime G1_WALK_MIN_PELVIS_Z: Float64 = 0.45
comptime G1_WALK_MAX_GRAVITY_Z: Float64 = -0.5
"""Pelvis-frame gravity z above this = tilted more than 60 deg."""

# ── per-lane meta words ───────────────────────────────────────────────────

comptime G1W_CMD_VX: Int = META_IDX_TASK_PARAM_0 + 0
comptime G1W_CMD_VY: Int = META_IDX_TASK_PARAM_0 + 1
comptime G1W_CMD_WZ: Int = META_IDX_TASK_PARAM_0 + 2
comptime G1W_CMD_TIMER: Int = META_IDX_TASK_PARAM_0 + 3
comptime G1W_AIR_L: Int = META_IDX_TASK_PARAM_0 + 4
comptime G1W_AIR_R: Int = META_IDX_TASK_PARAM_0 + 5
comptime G1W_LAST_L: Int = META_IDX_TASK_PARAM_0 + 6
comptime G1W_LAST_R: Int = META_IDX_TASK_PARAM_0 + 7
comptime G1W_PUSH_TIMER: Int = META_IDX_TASK_PARAM_0 + 8
comptime G1W_KEY: Int = META_IDX_TASK_PARAM_0 + 9

# ── bodies (worldbody DFS order, `unitree_g1_xml.mojo`) ──────────────────
# Left lower leg: knee 5, ankle pitch 6, ankle roll 7, contact dummies 8-11.
# Right lower leg: knee 15, ankle pitch 16, ankle roll 17, dummies 18-21.
# The four 5 mm spheres on the dummies are what touches the floor when
# standing; the ankle-pitch mesh collides too, so a tilted foot still counts.
comptime G1_WALK_L_FOOT_LO: Int = 6
comptime G1_WALK_L_FOOT_HI: Int = 11
comptime G1_WALK_R_FOOT_LO: Int = 16
comptime G1_WALK_R_FOOT_HI: Int = 21
comptime G1_WALK_L_KNEE: Int = 5
comptime G1_WALK_R_KNEE: Int = 15

# ── reward ────────────────────────────────────────────────────────────────

comptime G1_WALK_N_TERMS: Int = 17
comptime T_TRACK_LIN: Int = 0
comptime T_TRACK_ANG: Int = 1
comptime T_ANG_VEL_XY: Int = 2
comptime T_ORIENTATION: Int = 3
comptime T_LIN_VEL_Z: Int = 4
comptime T_FEET_AIR: Int = 5
comptime T_FEET_SLIP: Int = 6
comptime T_FEET_STILL: Int = 7
comptime T_STAND_STILL: Int = 8
comptime T_HIP_DEV: Int = 9
comptime T_KNEE_DEV: Int = 10
comptime T_POSE: Int = 11
comptime T_JOINT_LIMITS: Int = 12
comptime T_TORQUES: Int = 13
comptime T_TERMINATION: Int = 14
comptime T_ALIVE: Int = 15
comptime T_STAND_VEL: Int = 16

comptime G1_WALK_TRACKING_SIGMA: Float64 = 0.25
comptime G1_WALK_AIR_MIN: Float64 = 0.2
comptime G1_WALK_AIR_MAX: Float64 = 0.5
comptime G1_WALK_SOFT_LIMIT: Float64 = 0.95


@always_inline
def g1_walk_weight(t: Int) -> Float64:
    """Per-term weight (plan §4.4). The action-rate term is the driver's:
    it needs the previous action, which the env does not keep."""
    if t == T_TRACK_LIN:
        return 1.0
    elif t == T_TRACK_ANG:
        return 0.75
    elif t == T_ANG_VEL_XY:
        return -0.15
    elif t == T_ORIENTATION:
        return -2.0
    elif t == T_LIN_VEL_Z:
        return -0.2
    elif t == T_FEET_AIR:
        return 2.0
    elif t == T_FEET_SLIP:
        return -0.25
    elif t == T_FEET_STILL:
        return 0.5
    elif t == T_STAND_STILL:
        return -0.5
    elif t == T_HIP_DEV:
        return -0.25
    elif t == T_KNEE_DEV:
        return -0.1
    elif t == T_POSE:
        return -0.1
    elif t == T_JOINT_LIMITS:
        return -1.0
    elif t == T_TORQUES:
        return -1.0e-5
    elif t == T_TERMINATION:
        return -100.0
    elif t == T_STAND_VEL:
        # -2 (run s5) made stopping a fall: 8 / 10 random-command episodes
        return -0.5
    return 1.0  # T_ALIVE (see `g1_walk_reward`)


def g1_walk_term_name(t: Int) -> StaticString:
    if t == T_TRACK_LIN:
        return "track_lin_vel"
    elif t == T_TRACK_ANG:
        return "track_ang_vel"
    elif t == T_ANG_VEL_XY:
        return "ang_vel_xy"
    elif t == T_ORIENTATION:
        return "orientation"
    elif t == T_LIN_VEL_Z:
        return "lin_vel_z"
    elif t == T_FEET_AIR:
        return "feet_air_time"
    elif t == T_FEET_SLIP:
        return "feet_slip"
    elif t == T_FEET_STILL:
        return "feet_contact_no_cmd"
    elif t == T_STAND_STILL:
        return "stand_still"
    elif t == T_HIP_DEV:
        return "joint_dev_hip"
    elif t == T_KNEE_DEV:
        return "joint_dev_knee"
    elif t == T_POSE:
        return "pose"
    elif t == T_JOINT_LIMITS:
        return "joint_limits"
    elif t == T_TORQUES:
        return "torques"
    elif t == T_TERMINATION:
        return "termination"
    elif t == T_STAND_VEL:
        return "stand_vel"
    return "alive"


@always_inline
def g1_pose_weight(i: Int) -> Float64:
    """Playground's `_weights`: hip pitch and knee nearly free (0.01), the
    rest 1. DOF order: legs 0-11 (pitch, roll, yaw, knee, ankle p, ankle r),
    waist 12-14, arms 15-28."""
    if i == 0 or i == 3 or i == 6 or i == 9:
        return 0.01
    return 1.0


# ── pure arithmetic ───────────────────────────────────────────────────────


@always_inline
def g1_walk_target(i: Int, a: Float64) -> Float64:
    """Policy output -> joint target: `q_default + 0.25 clip(a, +-2)`."""
    return g1_default_pos(i) + G1_WALK_ACTION_SCALE * _clamp(
        a, -G1_WALK_ACTION_CLIP, G1_WALK_ACTION_CLIP
    )


@always_inline
def g1_rotate_inverse(
    qw: Float64, qx: Float64, qy: Float64, qz: Float64,
    vx: Float64, vy: Float64, vz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """`R(q)^T v` for a unit quaternion (w, x, y, z): world -> body frame."""
    # t = 2 (u x v) with u = -(x, y, z), then v + w t + u x t
    var ux = -qx
    var uy = -qy
    var uz = -qz
    var tx = 2.0 * (uy * vz - uz * vy)
    var ty = 2.0 * (uz * vx - ux * vz)
    var tz = 2.0 * (ux * vy - uy * vx)
    return (
        vx + qw * tx + (uy * tz - uz * ty),
        vy + qw * ty + (uz * tx - ux * tz),
        vz + qw * tz + (ux * ty - uy * tx),
    )


@always_inline
def g1_walk_command(
    u_stand: Float64, u_vx: Float64, u_vy: Float64, u_wz: Float64
) -> Tuple[Float64, Float64, Float64]:
    """Four uniforms in [0, 1) -> a command. `u_stand < P_STAND` is the
    exact zero; otherwise each component is uniform over its range."""
    if u_stand < G1_WALK_P_STAND:
        return (0.0, 0.0, 0.0)
    return (
        G1_WALK_VX_MIN + (G1_WALK_VX_MAX - G1_WALK_VX_MIN) * u_vx,
        G1_WALK_VY_MAX * (2.0 * u_vy - 1.0),
        G1_WALK_WZ_MAX * (2.0 * u_wz - 1.0),
    )


@always_inline
def g1_walk_is_stand(vx: Float64, vy: Float64, wz: Float64) -> Bool:
    var s = (vx if vx > 0.0 else -vx) + (vy if vy > 0.0 else -vy) + (
        wz if wz > 0.0 else -wz
    )
    return s < G1_WALK_CMD_ZERO


@always_inline
def g1_walk_feet_update(
    contact: Bool, last_contact: Bool, air: Float64, dt: Float64
) -> Tuple[Float64, Float64]:
    """Playground's air-time bookkeeping for one foot, one control step.

    Returns (reward contribution, new air time). `first_contact` uses the
    contact OR'ed with last step's (Playground's `contact_filt`), the air
    time keeps counting through the touchdown step and is zeroed while the
    foot is down."""
    var filt = contact or last_contact
    var new_air = air + dt
    var r = 0.0
    if air > 0.0 and filt:
        r = _clamp(new_air - G1_WALK_AIR_MIN, -1.0e9, G1_WALK_AIR_MAX - G1_WALK_AIR_MIN)
    if contact:
        new_air = 0.0
    return (r, new_air)


# ── the step: terms + termination, generic over the tensors' target ──────


@always_inline
def _foot_side(b: Int) -> Int:
    """1 left lower foot set, 2 right, 0 neither."""
    if b >= G1_WALK_L_FOOT_LO and b <= G1_WALK_L_FOOT_HI:
        return 1
    if b >= G1_WALK_R_FOOT_LO and b <= G1_WALK_R_FOOT_HI:
        return 2
    return 0


@always_inline
def _lower_leg_side(b: Int) -> Int:
    if b == G1_WALK_L_KNEE or _foot_side(b) == 1:
        return 1
    if b == G1_WALK_R_KNEE or _foot_side(b) == 2:
        return 2
    return 0


@always_inline
def g1_walk_contacts[
    DTYPE: DType, BATCH_SIZE: Int, MC_F: Int
](
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE), MutAnyOrigin
    ],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
) -> Tuple[Bool, Bool, Bool]:
    """(left foot on the floor, right foot on the floor, a fall contact).

    A fall contact is a non-foot body against the world, or the
    two lower legs against each other (Playground's foot-foot / foot-shin
    termination)."""
    var n = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS]))
    if n > MC_F:
        n = MC_F
    var left = False
    var right = False
    var bad = False
    for c in range(n):
        var a = Int(rebind[Scalar[DTYPE]](
            contacts[env, c * CONTACT_SIZE + CONTACT_IDX_BODY_A]
        ))
        var b = Int(rebind[Scalar[DTYPE]](
            contacts[env, c * CONTACT_SIZE + CONTACT_IDX_BODY_B]
        ))
        # ⚠ THE WORLD IS 0 OR -1: `detect_contacts` writes 0, the SAP path
        # -1 (`collision/contact_order.mojo`). The CPU env's records carry -1.
        if a <= 0 or b <= 0:
            var other = b if a <= 0 else a
            var side = _foot_side(other)
            if side == 1:
                left = True
            elif side == 2:
                right = True
            elif other > 0:
                bad = True
        else:
            var sa = _lower_leg_side(a)
            var sb = _lower_leg_side(b)
            if sa != 0 and sb != 0 and sa != sb:
                bad = True
    return (left, right, bad)


@always_inline
def _rd[DTYPE: DType](x: SIMD[DTYPE, _]) -> Float64:
    return Float64(rebind[Scalar[DTYPE]](x))


def g1_walk_terms[
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    ACTION_DIM: Int,
    MC_F: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE), MutAnyOrigin
    ],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    actions: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
    ],
    env: Int,
    mut terms: Array[Float64, G1_WALK_N_TERMS],
) -> Bool:
    """Every reward term for one lane after one control step, and whether
    the episode ends. ADVANCES the lane's feet state in `meta` — call it
    exactly once per control step (the reward hook does)."""
    comptime DT_CTRL = G1_SIM_TIMESTEP * Float64(G1_CONTROL_DECIMATION)
    for t in range(G1_WALK_N_TERMS):
        terms[t] = 0.0

    var qw = _rd[DTYPE](qpos[env, 3])
    var qx = _rd[DTYPE](qpos[env, 4])
    var qy = _rd[DTYPE](qpos[env, 5])
    var qz = _rd[DTYPE](qpos[env, 6])
    var lv = g1_rotate_inverse(
        qw, qx, qy, qz,
        _rd[DTYPE](qvel[env, 0]), _rd[DTYPE](qvel[env, 1]), _rd[DTYPE](qvel[env, 2]),
    )
    var wz = _rd[DTYPE](qvel[env, 5])
    var g = _projected_gravity(qw, qx, qy, qz)

    var cvx = _rd[DTYPE](meta[env, G1W_CMD_VX])
    var cvy = _rd[DTYPE](meta[env, G1W_CMD_VY])
    var cwz = _rd[DTYPE](meta[env, G1W_CMD_WZ])
    var stand = g1_walk_is_stand(cvx, cvy, cwz)

    # ── tracking, base ──
    var e_lin = (cvx - lv[0]) ** 2 + (cvy - lv[1]) ** 2
    terms[T_TRACK_LIN] = exp(-e_lin / G1_WALK_TRACKING_SIGMA)
    terms[T_TRACK_ANG] = exp(-((cwz - wz) ** 2) / G1_WALK_TRACKING_SIGMA)
    var twx = _rd[DTYPE](xangvel[env, TORSO_BODY_IDX * 3 + 0])
    var twy = _rd[DTYPE](xangvel[env, TORSO_BODY_IDX * 3 + 1])
    terms[T_ANG_VEL_XY] = twx * twx + twy * twy
    terms[T_ORIENTATION] = g[0] * g[0] + g[1] * g[1]
    var vz = _rd[DTYPE](qvel[env, 2])
    terms[T_LIN_VEL_Z] = vz * vz

    # ── feet ──
    var fc = g1_walk_contacts[DTYPE, BATCH_SIZE, MC_F](contacts, meta, env)
    var last_l = _rd[DTYPE](meta[env, G1W_LAST_L]) > 0.5
    var last_r = _rd[DTYPE](meta[env, G1W_LAST_R]) > 0.5
    var ul = g1_walk_feet_update(fc[0], last_l, _rd[DTYPE](meta[env, G1W_AIR_L]), DT_CTRL)
    var ur = g1_walk_feet_update(fc[1], last_r, _rd[DTYPE](meta[env, G1W_AIR_R]), DT_CTRL)
    meta[env, G1W_AIR_L] = Scalar[DTYPE](ul[1])
    meta[env, G1W_AIR_R] = Scalar[DTYPE](ur[1])
    meta[env, G1W_LAST_L] = Scalar[DTYPE](1.0 if fc[0] else 0.0)
    meta[env, G1W_LAST_R] = Scalar[DTYPE](1.0 if fc[1] else 0.0)
    if not stand:
        terms[T_FEET_AIR] = ul[0] + ur[0]
    var vxy = sqrt(
        _rd[DTYPE](qvel[env, 0]) ** 2 + _rd[DTYPE](qvel[env, 1]) ** 2
    )
    terms[T_FEET_SLIP] = vxy * (
        (1.0 if fc[0] else 0.0) + (1.0 if fc[1] else 0.0)
    )
    if stand and fc[0] and fc[1]:
        terms[T_FEET_STILL] = 1.0
    if stand:
        # ⚠ LINEAR, so a slow drift costs. The tracking term's
        # exp(-v^2 / 0.25) charges 0.08 m/s only 3 %, and run s4 drifted
        # 0.85 m in a 10 s hold with it alone.
        var wa = wz if wz > 0.0 else -wz
        terms[T_STAND_VEL] = sqrt(lv[0] * lv[0] + lv[1] * lv[1]) + 0.5 * wa

    # ── joints ──
    var dev_abs = 0.0
    var pose = 0.0
    var lim = 0.0
    var tau_abs = 0.0
    comptime for i in range(G1_N_DOF):
        var q = _rd[DTYPE](qpos[env, ROOT_QPOS_SIZE + i])
        var qd = _rd[DTYPE](qvel[env, ROOT_QVEL_SIZE + i])
        var e = q - g1_default_pos(i)
        var ea = e if e > 0.0 else -e
        dev_abs += ea
        pose += g1_pose_weight(i) * e * e
        comptime c = 0.5 * (g1_pos_lower(i) + g1_pos_upper(i))
        comptime r = 0.5 * (g1_pos_upper(i) - g1_pos_lower(i)) * G1_WALK_SOFT_LIMIT
        if q < c - r:
            lim += (c - r) - q
        elif q > c + r:
            lim += q - (c + r)
        var tau = g1_pd_torque(
            i, g1_walk_target(i, _rd[DTYPE](actions[env, i])), q, qd
        )
        tau_abs += tau if tau > 0.0 else -tau
        # hip roll / yaw: 1, 2, 7, 8; knees: 3, 9
        comptime if i == 1 or i == 2 or i == 7 or i == 8:
            terms[T_HIP_DEV] += ea
        comptime if i == 3 or i == 9:
            terms[T_KNEE_DEV] += ea
    terms[T_POSE] = pose
    terms[T_JOINT_LIMITS] = lim
    terms[T_TORQUES] = tau_abs
    if stand:
        # position only (Playground's `stand_still`); RoboParty's
        # 0.04 x sum|qdot| charged exploration noise (run s1)
        terms[T_STAND_STILL] = dev_abs

    # ── termination ──
    var pz = _rd[DTYPE](qpos[env, 2])
    var fallen = (
        pz < G1_WALK_MIN_PELVIS_Z or g[2] > G1_WALK_MAX_GRAVITY_Z or fc[2]
        or pz != pz
    )
    if fallen:
        terms[T_TERMINATION] = 1.0
    else:
        terms[T_ALIVE] = 1.0
    return fallen


@always_inline
def g1_walk_reward(terms: Array[Float64, G1_WALK_N_TERMS]) -> Float64:
    """`dt x sum(w_k t_k)` — Playground's scale (reward per second).

    ⚠ NO ZERO CLIP, AND AN ALIVE BONUS INSTEAD. Three runs on 2026-10-05:
    s1 (no clip, no alive bonus, a joint-VELOCITY standing term and the
    action-rate penalty at log-std 0) learned to end its episodes — living
    paid ~-0.05 per step against a one-off -2 for a fall. s2 / s3 clipped
    the step sum at 0 before the termination penalty (legged_gym's
    `only_positive_rewards`): no suicide, but the clip zeroes every
    penalty's gradient wherever their sum is negative, and both learned a
    shaking stand that ignored the command. Measured under the initial
    policy (falls in ~49 steps), the unclipped sum is -1.5 per second and
    positive on 11 % of steps: the clip removed most of the early signal.
    Now: no clip, `alive` pays +1 per second (RoboParty's `upward` +0.4
    plays this role), and the standing term is position-only (Playground's).
    A balanced stand is clearly positive; only falling is negative."""
    comptime DT_CTRL = G1_SIM_TIMESTEP * Float64(G1_CONTROL_DECIMATION)
    var r = 0.0
    comptime for t in range(G1_WALK_N_TERMS):
        r += g1_walk_weight(t) * terms[t]
    return r * DT_CTRL


# ── the per-lane draws ────────────────────────────────────────────────────


@always_inline
def g1_walk_pre_step[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NV: Int, PUSHES: Bool
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
):
    """Before physics: count down, redraw the command, kick the root.

    Draws come from Philox keyed by the lane's reset key and offset by the
    step count, so a lane's stream is reproducible from its reset seed."""
    _ = qpos
    var key = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, G1W_KEY])))
    var step = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_STEP_COUNT])))
    var rng = PhiloxRandom(seed=key, offset=step * 4)
    var timer = Int(rebind[Scalar[DTYPE]](meta[env, G1W_CMD_TIMER]))
    if timer >= 0:
        timer -= 1
        if timer <= 0:
            var u = rng.step_uniform()
            var cmd = g1_walk_command(
                Float64(u[0]), Float64(u[1]), Float64(u[2]), Float64(u[3])
            )
            meta[env, G1W_CMD_VX] = Scalar[DTYPE](cmd[0])
            meta[env, G1W_CMD_VY] = Scalar[DTYPE](cmd[1])
            meta[env, G1W_CMD_WZ] = Scalar[DTYPE](cmd[2])
            timer = G1_WALK_CMD_PERIOD
        meta[env, G1W_CMD_TIMER] = Scalar[DTYPE](timer)
    comptime if PUSHES:
        var pt = Int(rebind[Scalar[DTYPE]](meta[env, G1W_PUSH_TIMER])) - 1
        if pt <= 0:
            var p = rng.step_uniform()
            var th = Scalar[DTYPE](6.283185307179586) * Scalar[DTYPE](p[0])
            var mag = Scalar[DTYPE](
                G1_WALK_PUSH_MIN + (G1_WALK_PUSH_MAX - G1_WALK_PUSH_MIN) * Float64(p[1])
            )
            qvel[env, 0] = qvel[env, 0] + mag * cos_dt[DTYPE](th)
            qvel[env, 1] = qvel[env, 1] + mag * sin_dt[DTYPE](th)
            pt = G1_WALK_PUSH_MIN_STEPS + Int(
                Float64(G1_WALK_PUSH_MAX_STEPS - G1_WALK_PUSH_MIN_STEPS) * Float64(p[2])
            )
        meta[env, G1W_PUSH_TIMER] = Scalar[DTYPE](pt)


@always_inline
def g1_walk_init[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NV: Int, RANDOM: Bool
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
    seed: Int,
):
    """The reset. RANDOM: Playground's spread — root xy +-0.5 m, yaw uniform,
    root velocity +-0.5 (linear and angular), legs and waist +-0.1 rad about
    default, arms +-0.2. Otherwise the BFM stand pose at rest.

    ⚠ ADDITIVE JOINT NOISE, NOT PLAYGROUND'S x U(0.5, 1.5). Its keyframe is
    `knees_bent`; ours is the BFM stand (knee 0.3), where a multiplicative
    draw is +-0.15 rad on the knee and a 1-2 cm foot penetration at the
    fixed root height the reset uses.

    Every walker meta word is written here; the command timer is 0, so the
    first `pre_step` draws (or -1 = frozen when not RANDOM, the eval's
    caller writes the command)."""
    for i in range(NQ):
        qpos[env, i] = Scalar[DTYPE](0)
    for i in range(NV):
        qvel[env, i] = Scalar[DTYPE](0)
    qpos[env, 2] = Scalar[DTYPE](G1_INIT_ROOT_Z)
    qpos[env, 3] = Scalar[DTYPE](1)
    comptime for i in range(G1_N_DOF):
        qpos[env, ROOT_QPOS_SIZE + i] = Scalar[DTYPE](g1_default_pos(i))
    var key = reset_seed(env, seed)
    comptime if RANDOM:
        var rng = PhiloxRandom(seed=key, offset=0)
        var p0 = rng.step_uniform()
        qpos[env, 0] = Scalar[DTYPE](p0[0] - 0.5)
        qpos[env, 1] = Scalar[DTYPE](p0[1] - 0.5)
        var half = Scalar[DTYPE](3.141592653589793) * (
            Scalar[DTYPE](2.0) * Scalar[DTYPE](p0[2]) - Scalar[DTYPE](1.0)
        ) * Scalar[DTYPE](0.5)
        qpos[env, 3] = cos_dt[DTYPE](half)
        qpos[env, 6] = sin_dt[DTYPE](half)
        var pv = rng.step_uniform()
        var pw = rng.step_uniform()
        qvel[env, 0] = Scalar[DTYPE](pv[0] - 0.5)
        qvel[env, 1] = Scalar[DTYPE](pv[1] - 0.5)
        qvel[env, 2] = Scalar[DTYPE](pv[2] - 0.5)
        qvel[env, 3] = Scalar[DTYPE](pv[3] - 0.5)
        qvel[env, 4] = Scalar[DTYPE](pw[0] - 0.5)
        qvel[env, 5] = Scalar[DTYPE](pw[1] - 0.5)
        comptime for k in range(8):
            var u = rng.step_uniform()
            comptime for j in range(4):
                comptime i = k * 4 + j
                comptime if i < G1_N_DOF:
                    comptime amp = 0.1 if i < 15 else 0.2
                    qpos[env, ROOT_QPOS_SIZE + i] = Scalar[DTYPE](
                        g1_default_pos(i) + amp * (2.0 * Float64(u[j]) - 1.0)
                    )
        var pp = rng.step_uniform()
        meta[env, G1W_PUSH_TIMER] = Scalar[DTYPE](
            G1_WALK_PUSH_MIN_STEPS + Int(
                Float64(G1_WALK_PUSH_MAX_STEPS - G1_WALK_PUSH_MIN_STEPS) * Float64(pp[0])
            )
        )
        meta[env, G1W_CMD_TIMER] = Scalar[DTYPE](0)
    else:
        meta[env, G1W_PUSH_TIMER] = Scalar[DTYPE](G1_WALK_PUSH_MAX_STEPS)
        meta[env, G1W_CMD_TIMER] = Scalar[DTYPE](-1)
    meta[env, G1W_CMD_VX] = Scalar[DTYPE](0)
    meta[env, G1W_CMD_VY] = Scalar[DTYPE](0)
    meta[env, G1W_CMD_WZ] = Scalar[DTYPE](0)
    meta[env, G1W_AIR_L] = Scalar[DTYPE](0)
    meta[env, G1W_AIR_R] = Scalar[DTYPE](0)
    meta[env, G1W_LAST_L] = Scalar[DTYPE](1)
    meta[env, G1W_LAST_R] = Scalar[DTYPE](1)
    # A float32 word holds integers exactly below 2^24.
    meta[env, G1W_KEY] = Scalar[DTYPE](Int(key % UInt64(1 << 23)))


@always_inline
def g1_walk_obs[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NV: Int, OBS_DIM: Int, NOISE: Bool
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    obs: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    env: Int,
):
    """The 70-D observation (module docstring), with Playground's uniform
    noise when NOISE. Its stream is offset past the pre-step's."""
    comptime assert OBS_DIM == G1_WALK_OBS_DIM, "g1_walk_obs: OBS_DIM must be 70"
    var qw = _rd[DTYPE](qpos[env, 3])
    var qx = _rd[DTYPE](qpos[env, 4])
    var qy = _rd[DTYPE](qpos[env, 5])
    var qz = _rd[DTYPE](qpos[env, 6])
    var lv = g1_rotate_inverse(
        qw, qx, qy, qz,
        _rd[DTYPE](qvel[env, 0]), _rd[DTYPE](qvel[env, 1]), _rd[DTYPE](qvel[env, 2]),
    )
    var g = _projected_gravity(qw, qx, qy, qz)
    obs[env, G1_WALK_OBS_LINVEL + 0] = Scalar[DTYPE](lv[0])
    obs[env, G1_WALK_OBS_LINVEL + 1] = Scalar[DTYPE](lv[1])
    obs[env, G1_WALK_OBS_LINVEL + 2] = Scalar[DTYPE](lv[2])
    for k in range(3):
        obs[env, G1_WALK_OBS_ANGVEL + k] = qvel[env, 3 + k]
    obs[env, G1_WALK_OBS_GRAVITY + 0] = Scalar[DTYPE](g[0])
    obs[env, G1_WALK_OBS_GRAVITY + 1] = Scalar[DTYPE](g[1])
    obs[env, G1_WALK_OBS_GRAVITY + 2] = Scalar[DTYPE](g[2])
    obs[env, G1_WALK_OBS_CMD + 0] = meta[env, G1W_CMD_VX]
    obs[env, G1_WALK_OBS_CMD + 1] = meta[env, G1W_CMD_VY]
    obs[env, G1_WALK_OBS_CMD + 2] = meta[env, G1W_CMD_WZ]
    comptime for i in range(G1_N_DOF):
        obs[env, G1_WALK_OBS_Q + i] = qpos[env, ROOT_QPOS_SIZE + i] - Scalar[
            DTYPE
        ](g1_default_pos(i))
    for i in range(G1_N_DOF):
        obs[env, G1_WALK_OBS_QD + i] = qvel[env, ROOT_QVEL_SIZE + i]
    comptime if NOISE:
        var key = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, G1W_KEY])))
        var step = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_STEP_COUNT])))
        # 18 draws of 4 cover the 70 words; the pre-step uses offsets
        # [4 step, 4 step + 3], so this starts far past any of them.
        var rng = PhiloxRandom(seed=key + UInt64(0x9E3779B9), offset=step * 18)
        comptime for k in range(18):
            var u = rng.step_uniform()
            comptime for j in range(4):
                comptime i = k * 4 + j
                comptime if i < G1_WALK_OBS_DIM:
                    comptime if i < G1_WALK_OBS_ANGVEL:
                        obs[env, i] = obs[env, i] + Scalar[DTYPE](
                            G1_WALK_NOISE_LINVEL * (2.0 * Float64(u[j]) - 1.0)
                        )
                    elif i < G1_WALK_OBS_GRAVITY:
                        obs[env, i] = obs[env, i] + Scalar[DTYPE](
                            G1_WALK_NOISE_GYRO * (2.0 * Float64(u[j]) - 1.0)
                        )
                    elif i < G1_WALK_OBS_CMD:
                        obs[env, i] = obs[env, i] + Scalar[DTYPE](
                            G1_WALK_NOISE_GRAVITY * (2.0 * Float64(u[j]) - 1.0)
                        )
                    elif i < G1_WALK_OBS_Q:
                        pass  # the command is exact
                    elif i < G1_WALK_OBS_QD:
                        obs[env, i] = obs[env, i] + Scalar[DTYPE](
                            G1_WALK_NOISE_Q * (2.0 * Float64(u[j]) - 1.0)
                        )
                    else:
                        obs[env, i] = obs[env, i] + Scalar[DTYPE](
                            G1_WALK_NOISE_QD * (2.0 * Float64(u[j]) - 1.0)
                        )


def _host_copy[DTYPE: DType](src: TensorImpl[DTYPE], n: Int) -> TensorImpl[DTYPE]:
    var t = TensorImpl[DTYPE].alloc(n)
    for i in range(n):
        t.data[i] = src.data[i]
    return t^


# ── the config ────────────────────────────────────────────────────────────


struct UnitreeG1WalkConfig[TRAIN: Bool = True](Phyics3dEnvConfig):
    """TRAIN: random resets, command draws, observation noise and pushes.
    False: the stand reset, a frozen zero command (the caller writes its
    own), no noise, no pushes — the eval and the room."""

    comptime FRAME_SKIP: Int = G1_CONTROL_DECIMATION
    comptime MAX_STEPS: Int = G1_WALK_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    # The reward reads the torso's world angular velocity (`xangvel`).
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = True
    comptime CUSTOM_ACTIONS_EVERY_SUBSTEP: Bool = True
    comptime NORMALIZED_ACTIONS: Bool = False
    comptime NMESH_VERTS: Int = UNITREE_G1_NMESH_VERTS

    # ── CPU hooks (eval: the stand reset and the torque law) ──────────────

    @staticmethod
    def custom_apply_actions_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        m_tendons: List[Scalar[DTYPE]],
        m_actuators: List[Scalar[DTYPE]],
        m_act_tendons: List[Scalar[DTYPE]],
        actions: List[Float64],
    ) -> Bool:
        """BFM-Zero's torque law on the walker's targets; both clips, as
        `UnitreeG1Config._write_pd_torques`."""
        for i in range(D.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        for i in range(G1_N_DOF):
            var a = actions[i] if i < len(actions) else 0.0
            var q = Float64(d.qpos.data[ROOT_QPOS_SIZE + i])
            var qd = Float64(d.qvel.data[ROOT_QVEL_SIZE + i])
            var ao = i * MODEL_ACTUATOR_SIZE
            d.qfrc.data[ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1_pd_torque(i, g1_walk_target(i, a), q, qd),
                    Float64(m_actuators[ao + ACT_IDX_CTRL_MIN]),
                    Float64(m_actuators[ao + ACT_IDX_CTRL_MAX]),
                )
            )
        return True

    @staticmethod
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`g1_walk_obs` without noise, over the CPU env's lane."""
        # `d` is borrowed immutably and `lt` is a mutating view: the hook
        # reads copies (three short rows, CPU eval only).
        var q = _host_copy(d.qpos, D.NQ)
        var v = _host_copy(d.qvel, D.NV)
        var m = _host_copy(d.meta, METADATA_SIZE)
        var o = TensorImpl[DTYPE].alloc(G1_WALK_OBS_DIM)
        g1_walk_obs[DTYPE, 1, D.NQ, D.NV, G1_WALK_OBS_DIM, False](
            q.lt["cpu", Layout.row_major(1, D.NQ)](),
            v.lt["cpu", Layout.row_major(1, D.NV)](),
            m.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
            o.lt["cpu", Layout.row_major(1, G1_WALK_OBS_DIM)](),
            0,
        )
        for i in range(G1_WALK_OBS_DIM):
            obs.append(o.data[i])
        return True

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`g1_walk_init` for lane 0. RANDOM follows TRAIN; the seed is 0, so
        a CPU env's random reset is one fixed draw — the eval resets with
        `g1_walk_host_init` and its own seed instead."""
        g1_walk_init[DTYPE, 1, D.NQ, D.NV, Self.TRAIN](
            d.qpos.lt["cpu", Layout.row_major(1, D.NQ)](),
            d.qvel.lt["cpu", Layout.row_major(1, D.NV)](),
            d.meta.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
            0, 0,
        )

    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """Termination only (module docstring: the reward terms advance
        per-lane feet state this hook cannot write). Rewards on the host:
        `g1_walk_host_step`."""
        var qw = Float64(d.qpos.data[3])
        var qx = Float64(d.qpos.data[4])
        var qy = Float64(d.qpos.data[5])
        var qz = Float64(d.qpos.data[6])
        var g = _projected_gravity(qw, qx, qy, qz)
        var pz = Float64(d.qpos.data[2])
        var c = _host_copy(d.contacts, D.MAX_CONTACTS * CONTACT_SIZE)
        var m = _host_copy(d.meta, METADATA_SIZE)
        var fc = g1_walk_contacts[DTYPE, 1, D.MAX_CONTACTS](
            c.lt["cpu", Layout.row_major(1, D.MAX_CONTACTS * CONTACT_SIZE)](),
            m.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
            0,
        )
        var fallen = (
            pz < G1_WALK_MIN_PELVIS_Z or g[2] > G1_WALK_MAX_GRAVITY_Z
            or fc[2] or pz != pz
        )
        return (Scalar[DTYPE](0), fallen)

    @staticmethod
    def get_timestep() -> Float64:
        return G1_SIM_TIMESTEP

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.0

    # ── GPU hooks ─────────────────────────────────────────────────────────

    @always_inline
    @staticmethod
    def custom_apply_actions_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        NTENDON_F: Int,
        ACTION_DIM: Int,
        NA_F: Int,
        NACT_F: Int,
    ](
        qfrc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        tendons: LayoutTensor[
            DTYPE, Layout.row_major(NTENDON_F, MODEL_TENDON_SIZE), MutAnyOrigin
        ],
        acts: LayoutTensor[
            DTYPE, Layout.row_major(NACT_F * MODEL_ACTUATOR_SIZE), MutAnyOrigin
        ],
        act_tendons: LayoutTensor[
            DTYPE,
            Layout.row_major(NTENDON_F * MODEL_ACT_TENDON_SIZE),
            MutAnyOrigin,
        ],
        env: Int,
    ):
        """The CPU hook's twin, one lane, every substep."""
        for i in range(NV):
            qfrc[env, i] = Scalar[DTYPE](0)
        comptime for i in range(G1_N_DOF):
            var a = Float64(rebind[Scalar[DTYPE]](actions[env, i]))
            var q = Float64(rebind[Scalar[DTYPE]](qpos[env, ROOT_QPOS_SIZE + i]))
            var qd = Float64(rebind[Scalar[DTYPE]](qvel[env, ROOT_QVEL_SIZE + i]))
            comptime ao = i * MODEL_ACTUATOR_SIZE
            qfrc[env, ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1_pd_torque(i, g1_walk_target(i, a), q, qd),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MIN])),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MAX])),
                )
            )

    @always_inline
    @staticmethod
    def pre_step_full_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        g1_walk_pre_step[DTYPE, BATCH_SIZE, NQ, NV, Self.TRAIN](
            qpos, qvel, meta, env
        )

    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        g1_walk_obs[DTYPE, BATCH_SIZE, NQ, NV, OBS_DIM, Self.TRAIN](
            qpos, qvel, meta, obs, env
        )
        return True

    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        cfrc_ext: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        curriculum: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        var terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
        var done = g1_walk_terms[
            DTYPE, BATCH_SIZE, NQ, NV, NBODY, ACTION_DIM, MC_F
        ](qpos, qvel, xangvel, contacts, meta, actions, env, terms)
        return (Scalar[DTYPE](g1_walk_reward(terms)), done)

    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NJOINT: Int,
        NV: Int,
        NBODY: Int,
        NGEOM_F: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        g1_walk_init[DTYPE, BATCH_SIZE, NQ, NV, Self.TRAIN](
            qpos, qvel, meta, env, seed
        )
