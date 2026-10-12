"""Unitree G1 walker, RoboParty's recipe — G1_WALKER_PLAN §10 (L0d).

RoboParty's `robolab/tasks/direct/interrupt` (BSD-3-Clause, local copy
`references/roboto_origin-main/modules/roboparty_train/robolab`) on
BFM-Zero's G1 body and torque PD, with the interrupt (arm override) off:

    obs     = [actor 67 | privileged 15]                                  (82)
      actor      angular velocity, pelvis frame (3), projected gravity (3),
                 command vx vy wz (3), q - q_default (29), qdot (29)
                 — their `current_actor_obs` minus the last action, which
                 the history wrapper appends (it sees the executed action)
      privileged linear velocity, pelvis frame (3), foot contact L R (2),
                 foot contact force L R, world, x0.01 (6), foot air time L R (2),
                 foot height L R (2) — their critic extras that our hooks
                 can produce (joint acc / torques are not: §10)
    action  = 29 PD targets, `q_default + 0.25 clip(a, +-4)`, BFM-Zero's
              torque law with a per-lane, per-joint kp / kd scale U(0.9, 1.1)
    command = vx [-0.6, 1.0], vy [-0.5, 0.5], wz [-1.57, 1.57], redrawn every
              10 s, 20 % standing. Their heading mode is NOT ported: the
              deployed policy reads (vx, vy, wz) either way, and the room's
              planner commands wz
    reward  = their 26 terms (`G1R_*`), weights per second, summed x dt; the
              action rate and action smoothness are the driver's
    end     = the torso, a hip-roll or a hip-yaw link against the floor (their
              `terminate_contacts_body_names`, world contacts only — see
              `g1r_contacts`), or a non-finite state
    reset   = xy +-0.5, yaw uniform, root velocity (x y +-0.5, z +-0.2, roll
              pitch +-0.52, yaw +-0.78), joints default +-0.15
    push    = every 10-15 s, the root velocity += the reset ranges' draw

⚠ G1 MAPPING OF THEIR JOINT GROUPS (RPO is 23 DoF, no wrists):
    `joint_deviation_hip`   hip roll + hip yaw            (thigh roll / yaw)
    `joint_deviation_torso` waist x3 + wrist x6           (torso + elbow yaw;
                            the G1's forearm rotation is the wrist roll, and
                            its two extra wrist joints are held the same way)
    `joint_deviation_legs`  hip pitch, knee, ankle pitch, ankle roll
    `joint_deviation_interrupt` (unmasked with the interrupt off)
                            1.0 x (shoulder roll, shoulder yaw, elbow)
                            + 0.06 x shoulder pitch
    feet = the ankle-roll links and everything below them (the G1's sole
    spheres hang off `dummy_*` children), the ankle-pitch mesh included.

⚠ CONTACT FORCES ARE THE SOLVER'S, PER CONTACT, IN THE CONTACT FRAME. Their
sensor gives a net world force per body. Here: the normal part summed as
`f_n n` (sign set so a floor push is +z), and the tangential magnitude
added to the horizontal component as an upper bound — exact for the norm
threshold of `feet_force`, conservative for `feet_stumble`.

⚠ PER-LANE STATE: `meta[TASK_PARAM_0..10]` (`G1R_*` words below). The DR
scales are not stored: they are a hash of the lane's reset key and the
joint, recomputed in the actuation hook, so they change at every reset and
cost no meta word.
"""

from std.math import exp, sqrt
from std.random.philox import Random as PhiloxRandom
from std.sys import get_defined_int
from layout import Layout, LayoutTensor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, DimsLike
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
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
from .unitree_g1_config import _clamp, _projected_gravity
from .unitree_g1_walk_config import g1_rotate_inverse
from .unitree_g1_pd import (
    G1_N_DOF,
    G1_INIT_ROOT_Z,
    G1_SIM_TIMESTEP,
    G1_CONTROL_DECIMATION,
    g1_kp,
    g1_kd,
    g1_effort,
    g1_default_pos,
    g1_pos_lower,
    g1_pos_upper,
)


# ── observation ───────────────────────────────────────────────────────────

comptime G1R_OBS_ACTOR: Int = 67
comptime G1R_OBS_PRIV: Int = 15
comptime G1R_OBS_DIM: Int = G1R_OBS_ACTOR + G1R_OBS_PRIV
comptime G1R_O_ANGVEL: Int = 0
comptime G1R_O_GRAVITY: Int = 3
comptime G1R_O_CMD: Int = 6
comptime G1R_O_Q: Int = 9
comptime G1R_O_QD: Int = 9 + G1_N_DOF
comptime G1R_O_LINVEL: Int = G1R_OBS_ACTOR
comptime G1R_O_CONTACT: Int = G1R_OBS_ACTOR + 3
comptime G1R_O_FORCE: Int = G1R_OBS_ACTOR + 5
comptime G1R_O_AIR: Int = G1R_OBS_ACTOR + 11
comptime G1R_O_HEIGHT: Int = G1R_OBS_ACTOR + 13

comptime G1R_FORCE_OBS_SCALE: Float64 = 0.01
"""The privileged foot forces are observed in units of 100 N (standing ~1.7).
⚠ In newtons, a landing impact passes 1000 and `run_ppo_vec`'s divergence
guard (`obs_bound` 1e3) ended the lane: run rp1 (2026-10-05) lost 1.4 M
lanes that way in 6 M steps, episodes ~3 steps long."""

# their `NoiseScalesCfg` (uniform half-widths), actor part only
comptime G1R_NOISE_ANGVEL: Float64 = 0.2
comptime G1R_NOISE_GRAVITY: Float64 = 0.05
comptime G1R_NOISE_Q: Float64 = 0.01
comptime G1R_NOISE_QD: Float64 = 1.5

# ── action, DR ────────────────────────────────────────────────────────────

comptime G1R_ACTION_SCALE: Float64 = 0.25
comptime G1R_ACTION_CLIP: Float64 = 4.0
"""Theirs is 100 (none). +-4 = +-1 rad off default; the PD's effort limits
bind long before."""
comptime G1R_GAIN_DR: Float64 = 0.1
"""kp and kd each x U(1 - 0.1, 1 + 0.1), per joint, per episode."""

# ── commands, pushes, episode ─────────────────────────────────────────────

comptime G1R_VX_MIN: Float64 = -0.6
comptime G1R_VX_MAX: Float64 = 1.0
comptime G1R_VY_MAX: Float64 = 0.5
comptime G1R_WZ_MAX: Float64 = 1.57
comptime G1R_P_STAND: Float64 = 0.2
comptime G1R_CMD_PERIOD: Int = 500
comptime G1R_CMD_ZERO: Float64 = 0.01
comptime G1R_PUSH_MIN_STEPS: Int = 500
comptime G1R_PUSH_MAX_STEPS: Int = 750
comptime G1R_MAX_STEPS: Int = 1000

# ── per-lane meta words ───────────────────────────────────────────────────

comptime G1R_CMD_VX: Int = META_IDX_TASK_PARAM_0 + 0
comptime G1R_CMD_VY: Int = META_IDX_TASK_PARAM_0 + 1
comptime G1R_CMD_WZ: Int = META_IDX_TASK_PARAM_0 + 2
comptime G1R_CMD_TIMER: Int = META_IDX_TASK_PARAM_0 + 3
comptime G1R_AIR_L: Int = META_IDX_TASK_PARAM_0 + 4
comptime G1R_AIR_R: Int = META_IDX_TASK_PARAM_0 + 5
comptime G1R_CON_L: Int = META_IDX_TASK_PARAM_0 + 6
comptime G1R_CON_R: Int = META_IDX_TASK_PARAM_0 + 7
comptime G1R_PUSH_TIMER: Int = META_IDX_TASK_PARAM_0 + 8
comptime G1R_KEY: Int = META_IDX_TASK_PARAM_0 + 9

# ── bodies ────────────────────────────────────────────────────────────────

comptime G1R_PELVIS: Int = 1
comptime G1R_L_HIP_ROLL: Int = 3
comptime G1R_L_HIP_YAW: Int = 4
comptime G1R_L_KNEE: Int = 5
comptime G1R_L_ANKLE_ROLL: Int = 7
comptime G1R_R_HIP_ROLL: Int = 13
comptime G1R_R_HIP_YAW: Int = 14
comptime G1R_R_KNEE: Int = 15
comptime G1R_R_ANKLE_ROLL: Int = 17
comptime G1R_L_FOOT_LO: Int = 6
comptime G1R_L_FOOT_HI: Int = 11
comptime G1R_R_FOOT_LO: Int = 16
comptime G1R_R_FOOT_HI: Int = 21
comptime G1R_SOLE_BELOW_ANKLE: Float64 = 0.035
"""The sole spheres sit 0.03 below the ankle-roll origin, radius 0.005."""

# ── reward terms (their weights, per second) ──────────────────────────────

comptime G1R_N_TERMS: Int = 26
comptime R_TRACK_LIN: Int = 0
comptime R_TRACK_ANG: Int = 1
comptime R_LIN_VEL_Z: Int = 2
comptime R_ANG_VEL_XY: Int = 3
comptime R_ENERGY: Int = 4
comptime R_TORQUES: Int = 5
comptime R_JOINT_VEL: Int = 6
comptime R_UNDESIRED: Int = 7
comptime R_FLAT_ORIENT: Int = 8
comptime R_TERMINATION: Int = 9
comptime R_FEET_AIR: Int = 10
comptime R_FEET_SLIDE: Int = 11
comptime R_FEET_FORCE: Int = 12
comptime R_FEET_DIST: Int = 13
comptime R_KNEE_DIST: Int = 14
comptime R_FEET_STUMBLE: Int = 15
comptime R_FEET_ORIENT: Int = 16
comptime R_POS_LIMITS: Int = 17
comptime R_DEV_HIP: Int = 18
comptime R_DEV_TORSO: Int = 19
comptime R_DEV_LEGS: Int = 20
comptime R_DEV_ARMS: Int = 21
comptime R_FEET_STILL: Int = 22
comptime R_UPWARD: Int = 23
comptime R_STAND_STILL: Int = 24
comptime R_FEET_HEIGHT: Int = 25

comptime G1R_FEET_AIR_W: Float64 = Float64(
    get_defined_int["G1R_FEET_AIR_MILLI", 250]()
) / 1000.0
"""`feet_air_time`'s weight, theirs 0.25; `-D G1R_FEET_AIR_MILLI=1000` at
`mojo build` raises it (the lever for a policy stuck standing — run rp5 at
60-100 M never stepped)."""
comptime G1R_TRACK_STD: Float64 = 0.5
comptime G1R_AIR_THRESHOLD: Float64 = 0.4
comptime G1R_SOFT_LIMIT: Float64 = 0.9
comptime G1R_FOOT_CLEAR: Float64 = 0.02
"""`feet_height` threshold: sole clearance (their 0.02 above `ankle_height`)."""


@always_inline
def g1r_weight(t: Int) -> Float64:
    if t == R_TRACK_LIN:
        return 1.0
    elif t == R_TRACK_ANG:
        return 1.0
    elif t == R_LIN_VEL_Z:
        return -0.2
    elif t == R_ANG_VEL_XY:
        return -0.1
    elif t == R_ENERGY:
        return -1.0e-4
    elif t == R_TORQUES:
        return -1.0e-5
    elif t == R_JOINT_VEL:
        return -2.0e-4
    elif t == R_UNDESIRED:
        return -1.0
    elif t == R_FLAT_ORIENT:
        return -1.0
    elif t == R_TERMINATION:
        return -200.0
    elif t == R_FEET_AIR:
        return G1R_FEET_AIR_W
    elif t == R_FEET_SLIDE:
        return -0.3
    elif t == R_FEET_FORCE:
        return -3.0e-3
    elif t == R_FEET_DIST:
        return 0.1
    elif t == R_KNEE_DIST:
        return 0.1
    elif t == R_FEET_STUMBLE:
        return -1.0
    elif t == R_FEET_ORIENT:
        return -0.1
    elif t == R_POS_LIMITS:
        return -1.0
    elif t == R_DEV_HIP:
        return -0.03
    elif t == R_DEV_TORSO:
        return -1.0
    elif t == R_DEV_LEGS:
        return -0.01
    elif t == R_DEV_ARMS:
        return -1.0
    elif t == R_FEET_STILL:
        return 0.1
    elif t == R_UPWARD:
        return 0.4
    elif t == R_STAND_STILL:
        return -0.2
    return 0.2  # R_FEET_HEIGHT


def g1r_term_name(t: Int) -> StaticString:
    if t == R_TRACK_LIN:
        return "track_lin_vel_xy_exp"
    elif t == R_TRACK_ANG:
        return "track_ang_vel_z_exp"
    elif t == R_LIN_VEL_Z:
        return "lin_vel_z_l2"
    elif t == R_ANG_VEL_XY:
        return "ang_vel_xy_l2"
    elif t == R_ENERGY:
        return "energy"
    elif t == R_TORQUES:
        return "joint_torques_l2"
    elif t == R_JOINT_VEL:
        return "joint_vel_l2"
    elif t == R_UNDESIRED:
        return "undesired_contacts"
    elif t == R_FLAT_ORIENT:
        return "flat_orientation_l2"
    elif t == R_TERMINATION:
        return "termination_penalty"
    elif t == R_FEET_AIR:
        return "feet_air_time"
    elif t == R_FEET_SLIDE:
        return "feet_slide"
    elif t == R_FEET_FORCE:
        return "feet_force"
    elif t == R_FEET_DIST:
        return "feet_distance"
    elif t == R_KNEE_DIST:
        return "knee_distance"
    elif t == R_FEET_STUMBLE:
        return "feet_stumble"
    elif t == R_FEET_ORIENT:
        return "feet_orientation_l2"
    elif t == R_POS_LIMITS:
        return "dof_pos_limits"
    elif t == R_DEV_HIP:
        return "joint_deviation_hip"
    elif t == R_DEV_TORSO:
        return "joint_deviation_torso"
    elif t == R_DEV_LEGS:
        return "joint_deviation_legs"
    elif t == R_DEV_ARMS:
        return "joint_deviation_arms"
    elif t == R_FEET_STILL:
        return "feet_contact_without_cmd"
    elif t == R_UPWARD:
        return "upward"
    elif t == R_STAND_STILL:
        return "stand_still"
    return "feet_height"


@always_inline
def g1r_dev_group(i: Int) -> Int:
    """0 none, 1 hip, 2 torso, 3 legs, 4 arms (x1), 5 arms (x0.06).
    DOF order: per leg pitch roll yaw knee ankle-p ankle-r (0-5, 6-11),
    waist 12-14, per arm shoulder p r y, elbow, wrist r p y (15-21, 22-28)."""
    if i < 12:
        var k = i % 6
        if k == 1 or k == 2:
            return 1
        return 3
    if i < 15:
        return 2
    var k = (i - 15) % 7
    if k >= 4:
        return 2
    if k == 0:
        return 5
    return 4


# ── left / right mirror (the sagittal plane, y -> -y) ─────────────────────


@always_inline
def g1_mirror_joint(i: Int) -> Tuple[Int, Float64]:
    """The joint `i` maps to under the left / right mirror, and the sign:
    pitch joints keep their sign, roll and yaw joints flip it (their axes are
    x and z, which a y -> -y reflection reverses as rotations). Legs and
    arms swap sides; the waist maps to itself. Checked against the model's
    joint ranges and default pose in `test_unitree_g1_walk_rp`."""
    if i < 12:
        var k = i % 6
        var j = i + 6 if i < 6 else i - 6
        # pitch roll yaw knee ankle-p ankle-r
        return (j, -1.0 if (k == 1 or k == 2 or k == 5) else 1.0)
    if i < 15:
        # waist yaw roll pitch
        return (i, 1.0 if i == 14 else -1.0)
    var k = (i - 15) % 7
    var j = i + 7 if i < 22 else i - 7
    # shoulder p r y, elbow, wrist r p y
    return (j, -1.0 if (k == 1 or k == 2 or k == 4 or k == 6) else 1.0)


def g1r_mirror_env_obs() -> Tuple[List[Int], List[Float64]]:
    """The 82-word env observation's mirror (index, sign) per word."""
    var idx = List[Int](length=G1R_OBS_DIM, fill=0)
    var sgn = List[Float64](length=G1R_OBS_DIM, fill=1.0)
    for k in range(G1R_OBS_DIM):
        idx[k] = k
    # body angular velocity (wx, wy, wz) -> (-wx, wy, -wz)
    sgn[G1R_O_ANGVEL + 0] = -1.0
    sgn[G1R_O_ANGVEL + 2] = -1.0
    # projected gravity (gx, gy, gz) -> (gx, -gy, gz)
    sgn[G1R_O_GRAVITY + 1] = -1.0
    # command (vx, vy, wz) -> (vx, -vy, -wz)
    sgn[G1R_O_CMD + 1] = -1.0
    sgn[G1R_O_CMD + 2] = -1.0
    for i in range(G1_N_DOF):
        var m = g1_mirror_joint(i)
        idx[G1R_O_Q + i] = G1R_O_Q + m[0]
        sgn[G1R_O_Q + i] = m[1]
        idx[G1R_O_QD + i] = G1R_O_QD + m[0]
        sgn[G1R_O_QD + i] = m[1]
    # privileged: linvel (vx, -vy, vz); contacts / forces / air / height swap
    # L <-> R, the forces' y negated
    sgn[G1R_O_LINVEL + 1] = -1.0
    idx[G1R_O_CONTACT + 0] = G1R_O_CONTACT + 1
    idx[G1R_O_CONTACT + 1] = G1R_O_CONTACT + 0
    for k in range(3):
        idx[G1R_O_FORCE + k] = G1R_O_FORCE + 3 + k
        idx[G1R_O_FORCE + 3 + k] = G1R_O_FORCE + k
    sgn[G1R_O_FORCE + 1] = -1.0
    sgn[G1R_O_FORCE + 4] = -1.0
    idx[G1R_O_AIR + 0] = G1R_O_AIR + 1
    idx[G1R_O_AIR + 1] = G1R_O_AIR + 0
    idx[G1R_O_HEIGHT + 0] = G1R_O_HEIGHT + 1
    idx[G1R_O_HEIGHT + 1] = G1R_O_HEIGHT + 0
    return (idx^, sgn^)


# ── per-lane hashes (DR) ──────────────────────────────────────────────────


@always_inline
def _hash01(key: Int, salt: Int) -> Float64:
    """A uniform in [0, 1) from (key, salt) — integer mix (splitmix-style),
    cheap enough to run per joint per substep."""
    var x = UInt64(key) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt) * UInt64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * UInt64(0x94D049BB133111EB)
    x = x ^ (x >> 31)
    return Float64(x >> 11) * (1.0 / 9007199254740992.0)


@always_inline
def g1r_gain_scales(key: Int, i: Int) -> Tuple[Float64, Float64]:
    """(kp scale, kd scale) for joint `i` of the lane whose reset key is
    `key`; key 0 (the eval's stand reset) is exactly (1, 1)."""
    if key == 0:
        return (1.0, 1.0)
    return (
        1.0 + G1R_GAIN_DR * (2.0 * _hash01(key, 2 * i) - 1.0),
        1.0 + G1R_GAIN_DR * (2.0 * _hash01(key, 2 * i + 1) - 1.0),
    )


@always_inline
def g1r_target(i: Int, a: Float64) -> Float64:
    return g1_default_pos(i) + G1R_ACTION_SCALE * _clamp(
        a, -G1R_ACTION_CLIP, G1R_ACTION_CLIP
    )


@always_inline
def g1r_torque(
    i: Int, a: Float64, q: Float64, qd: Float64, kps: Float64, kds: Float64
) -> Float64:
    var tau = kps * g1_kp(i) * (g1r_target(i, a) - q) - kds * g1_kd(i) * qd
    var lim = g1_effort(i)
    return _clamp(tau, -lim, lim)


@always_inline
def g1r_command(
    u_stand: Float64, u_vx: Float64, u_vy: Float64, u_wz: Float64
) -> Tuple[Float64, Float64, Float64]:
    if u_stand < G1R_P_STAND:
        return (0.0, 0.0, 0.0)
    return (
        G1R_VX_MIN + (G1R_VX_MAX - G1R_VX_MIN) * u_vx,
        G1R_VY_MAX * (2.0 * u_vy - 1.0),
        G1R_WZ_MAX * (2.0 * u_wz - 1.0),
    )


@always_inline
def g1r_cmd_norm(vx: Float64, vy: Float64, wz: Float64) -> Float64:
    """Their `|cmd_xy| + |wz|`."""
    return sqrt(vx * vx + vy * vy) + (wz if wz > 0.0 else -wz)


@always_inline
def g1r_feet_times(
    contact: Bool, air: Float64, con: Float64, dt: Float64
) -> Tuple[Float64, Float64]:
    """IsaacLab's contact-sensor clocks for one foot, one control step:
    (air time, contact time), each reset to 0 on the opposite mode."""
    if contact:
        return (0.0, con + dt)
    return (air + dt, 0.0)


@always_inline
def g1r_distance_band(d: Float64, lo: Float64, hi: Float64) -> Float64:
    """Their `body_distance_y` shaping: 1 inside [lo, hi], decaying at 100/m
    outside, the two sides averaged."""
    var dmin = _clamp(d - lo, -0.5, 0.0)
    var dmax = _clamp(d - hi, 0.0, 0.5)
    return (exp(-(dmin if dmin > 0.0 else -dmin) * 100.0)
            + exp(-(dmax if dmax > 0.0 else -dmax) * 100.0)) * 0.5


@always_inline
def _rd[DTYPE: DType](x: SIMD[DTYPE, _]) -> Float64:
    return Float64(rebind[Scalar[DTYPE]](x))


@always_inline
def _side(b: Int) -> Int:
    if b >= G1R_L_FOOT_LO and b <= G1R_L_FOOT_HI:
        return 1
    if b >= G1R_R_FOOT_LO and b <= G1R_R_FOOT_HI:
        return 2
    return 0


@always_inline
def _terminating(b: Int) -> Bool:
    return (
        b == TORSO_BODY_IDX or b == G1R_L_HIP_ROLL or b == G1R_L_HIP_YAW
        or b == G1R_R_HIP_ROLL or b == G1R_R_HIP_YAW
    )


@always_inline
def _arm_torso(a: Int, b: Int) -> Bool:
    """An upper-body self-contact that is the G1's resting geometry, not a
    fault: the waist / torso (22-24) against an arm link (25-40), or two
    links of the SAME arm (left 25-32, right 33-40). Left arm against right
    arm, and anything against the floor, still count."""
    var ta = a >= 22 and a <= 24
    var tb = b >= 22 and b <= 24
    var aa = a >= 25 and a <= 40
    var ab = b >= 25 and b <= 40
    return (ta and ab) or (tb and aa) or (aa and ab and ((a <= 32) == (b <= 32)))


@fieldwise_init
struct G1RContacts(Copyable, ImplicitlyCopyable, Movable):
    """One scan of a lane's contact records."""

    var left: Bool
    var right: Bool
    var flx: Float64          # left foot normal force, world (+z = floor push)
    var fly: Float64
    var flz: Float64
    var frx: Float64
    var fry: Float64
    var frz: Float64
    var fl_tan: Float64               # summed tangential magnitudes
    var fr_tan: Float64
    var undesired: Int                # distinct non-foot bodies in contact
    var terminate: Bool


@always_inline
def g1r_contacts[
    DTYPE: DType, BATCH_SIZE: Int, MC_F: Int
](
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE), MutAnyOrigin
    ],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
) -> G1RContacts:
    var n = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS]))
    if n > MC_F:
        n = MC_F
    var out = G1RContacts(
        False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False,
    )
    var seen = UInt64(0)
    for c in range(n):
        var o = c * CONTACT_SIZE
        var a = Int(_rd[DTYPE](contacts[env, o + CONTACT_IDX_BODY_A]))
        var b = Int(_rd[DTYPE](contacts[env, o + CONTACT_IDX_BODY_B]))
        var f_n = _rd[DTYPE](contacts[env, o + CONTACT_IDX_FORCE_N])
        var ft = sqrt(
            _rd[DTYPE](contacts[env, o + CONTACT_IDX_FORCE_T1]) ** 2
            + _rd[DTYPE](contacts[env, o + CONTACT_IDX_FORCE_T2]) ** 2
        )
        var nx = _rd[DTYPE](contacts[env, o + CONTACT_IDX_NX])
        var ny = _rd[DTYPE](contacts[env, o + CONTACT_IDX_NY])
        var nz = _rd[DTYPE](contacts[env, o + CONTACT_IDX_NZ])
        var sgn = 1.0 if nz >= 0.0 else -1.0
        # ⚠ the world is 0 or -1 (`collision/contact_order.mojo`)
        for k in range(2):
            var body = a if k == 0 else b
            var other = b if k == 0 else a
            if body <= 0:
                continue
            var side = _side(body)
            if side == 1:
                out.left = True
                out.flx += sgn * f_n * nx
                out.fly += sgn * f_n * ny
                out.flz += sgn * f_n * nz
                out.fl_tan += ft
            elif side == 2:
                out.right = True
                out.frx += sgn * f_n * nx
                out.fry += sgn * f_n * ny
                out.frz += sgn * f_n * nz
                out.fr_tan += ft
            else:
                # ⚠ ARM AGAINST TORSO IS NOT "UNDESIRED" ON THIS BODY. On BFM's
                # G1 the upper arms hang against the torso mesh and a walking
                # gait brushes them every step: under RoboParty's reward, s6's
                # walk paid -2.10 / s here (torso-upper-arm pairs, 33 k records
                # in 8 k steps) against +0.32 / s of extra tracking — standing
                # still won (runs rp5-rp7). On their RPO the arms clear the
                # torso, so the term never charged a gait. Floor contacts and
                # leg-leg contacts still count.
                if not _arm_torso(body, other) and body < 64 and (seen >> UInt64(body)) & 1 == 0:
                    seen = seen | (UInt64(1) << UInt64(body))
                    out.undesired += 1
                # ⚠ AGAINST THE WORLD ONLY. RoboParty terminates on any
                # contact of these links (self-collision on), but the RPO's
                # arms clear its torso; on BFM's G1 the upper arm hangs
                # against the torso mesh, and the reset's +-0.15 rad offsets
                # touched it on half the lanes — run rp2 (2026-10-05) ended
                # every episode in ~1.3 steps. Self-contacts stay in
                # `undesired_contacts`.
                if _terminating(body) and other <= 0:
                    out.terminate = True
    return out


@always_inline
def g1r_stumble(fc: G1RContacts) -> Bool:
    """Their `feet_stumble`: a foot whose horizontal force exceeds 3x its
    vertical one. ⚠ DEAD ON A FLAT FLOOR BY PHYSICS: the normal is +z and
    friction bounds the tangential part at ~mu N (mu ~1 here), so the ratio
    never reaches 3. It fires on edges and slopes (their gravel terrain);
    the test checks it on a synthetic record."""
    var hl = sqrt(fc.flx ** 2 + fc.fly ** 2) + fc.fl_tan
    var hr = sqrt(fc.frx ** 2 + fc.fry ** 2) + fc.fr_tan
    var vl = fc.flz if fc.flz > 0.0 else -fc.flz
    var vr = fc.frz if fc.frz > 0.0 else -fc.frz
    return (fc.left and hl > 3.0 * vl) or (fc.right and hr > 3.0 * vr)


def g1r_terms[
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
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin],
    xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
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
    mut terms: Array[Float64, G1R_N_TERMS],
) -> Bool:
    """Every term for one lane after one control step; ADVANCES the lane's
    feet clocks. Returns the termination."""
    comptime DT_CTRL = G1_SIM_TIMESTEP * Float64(G1_CONTROL_DECIMATION)
    for t in range(G1R_N_TERMS):
        terms[t] = 0.0
    var qw = _rd[DTYPE](qpos[env, 3])
    var qx = _rd[DTYPE](qpos[env, 4])
    var qy = _rd[DTYPE](qpos[env, 5])
    var qz = _rd[DTYPE](qpos[env, 6])
    var vwx = _rd[DTYPE](qvel[env, 0])
    var vwy = _rd[DTYPE](qvel[env, 1])
    var vwz = _rd[DTYPE](qvel[env, 2])
    var vb = g1_rotate_inverse(qw, qx, qy, qz, vwx, vwy, vwz)
    var wbx = _rd[DTYPE](qvel[env, 3])
    var wby = _rd[DTYPE](qvel[env, 4])
    var wbz = _rd[DTYPE](qvel[env, 5])
    var g = _projected_gravity(qw, qx, qy, qz)
    var up = _clamp(-g[2], 0.0, 0.7) / 0.7

    var cvx = _rd[DTYPE](meta[env, G1R_CMD_VX])
    var cvy = _rd[DTYPE](meta[env, G1R_CMD_VY])
    var cwz = _rd[DTYPE](meta[env, G1R_CMD_WZ])
    var cn = g1r_cmd_norm(cvx, cvy, cwz)
    var moving_cmd = cn > G1R_CMD_ZERO

    # ── tracking: linear in the YAW frame, angular in the world ──
    var yaw_c = qw * qw + qx * qx - qy * qy - qz * qz   # cos(yaw)-ish, normalised below
    var yaw_s = 2.0 * (qw * qz + qx * qy)
    var yn = sqrt(yaw_c * yaw_c + yaw_s * yaw_s)
    if yn > 1e-9:
        yaw_c /= yn
        yaw_s /= yn
    var vyx = yaw_c * vwx + yaw_s * vwy
    var vyy = -yaw_s * vwx + yaw_c * vwy
    var std2 = G1R_TRACK_STD * G1R_TRACK_STD
    terms[R_TRACK_LIN] = exp(-((cvx - vyx) ** 2 + (cvy - vyy) ** 2) / std2) * up
    # world z angular velocity = third row of R(q) . w_body
    var wwz = (2.0 * (qx * qz - qw * qy)) * wbx + (2.0 * (qy * qz + qw * qx)) * wby + (
        1.0 - 2.0 * (qx * qx + qy * qy)
    ) * wbz
    terms[R_TRACK_ANG] = exp(-((cwz - wwz) ** 2) / std2) * up
    terms[R_LIN_VEL_Z] = vb[2] * vb[2] * up
    terms[R_ANG_VEL_XY] = (wbx * wbx + wby * wby) * up
    terms[R_FLAT_ORIENT] = g[0] * g[0] + g[1] * g[1]
    terms[R_UPWARD] = -g[2]

    # ── contacts, feet ──
    var fc = g1r_contacts[DTYPE, BATCH_SIZE, MC_F](contacts, meta, env)
    var tl = g1r_feet_times(
        fc.left, _rd[DTYPE](meta[env, G1R_AIR_L]), _rd[DTYPE](meta[env, G1R_CON_L]), DT_CTRL
    )
    var tr = g1r_feet_times(
        fc.right, _rd[DTYPE](meta[env, G1R_AIR_R]), _rd[DTYPE](meta[env, G1R_CON_R]), DT_CTRL
    )
    meta[env, G1R_AIR_L] = Scalar[DTYPE](tl[0])
    meta[env, G1R_CON_L] = Scalar[DTYPE](tl[1])
    meta[env, G1R_AIR_R] = Scalar[DTYPE](tr[0])
    meta[env, G1R_CON_R] = Scalar[DTYPE](tr[1])
    terms[R_UNDESIRED] = Float64(fc.undesired)
    var single = fc.left != fc.right
    if single and moving_cmd:
        var ml = tl[1] if fc.left else tl[0]
        var mr = tr[1] if fc.right else tr[0]
        terms[R_FEET_AIR] = _clamp(ml if ml < mr else mr, 0.0, G1R_AIR_THRESHOLD) * up
    # slide: the foot body's own world velocity while in contact
    var la = G1R_L_ANKLE_ROLL * 3
    var ra = G1R_R_ANKLE_ROLL * 3
    if fc.left:
        terms[R_FEET_SLIDE] += sqrt(_rd[DTYPE](xvel[env, la]) ** 2 + _rd[DTYPE](xvel[env, la + 1]) ** 2)
    if fc.right:
        terms[R_FEET_SLIDE] += sqrt(_rd[DTYPE](xvel[env, ra]) ** 2 + _rd[DTYPE](xvel[env, ra + 1]) ** 2)
    var fnl = sqrt(fc.flx ** 2 + fc.fly ** 2 + fc.flz ** 2 + fc.fl_tan ** 2)
    var fnr = sqrt(fc.frx ** 2 + fc.fry ** 2 + fc.frz ** 2 + fc.fr_tan ** 2)
    terms[R_FEET_FORCE] = _clamp(fnl + fnr - 500.0, 0.0, 400.0)
    if g1r_stumble(fc):
        terms[R_FEET_STUMBLE] = 1.0
    # feet / knee spacing along the pelvis y axis
    var px = _rd[DTYPE](qpos[env, 0])
    var py = _rd[DTYPE](qpos[env, 1])
    var pz = _rd[DTYPE](qpos[env, 2])
    var pl = g1_rotate_inverse(
        qw, qx, qy, qz, _rd[DTYPE](xpos[env, la]) - px,
        _rd[DTYPE](xpos[env, la + 1]) - py, _rd[DTYPE](xpos[env, la + 2]) - pz,
    )
    var pr = g1_rotate_inverse(
        qw, qx, qy, qz, _rd[DTYPE](xpos[env, ra]) - px,
        _rd[DTYPE](xpos[env, ra + 1]) - py, _rd[DTYPE](xpos[env, ra + 2]) - pz,
    )
    var fd = pl[1] - pr[1]
    terms[R_FEET_DIST] = g1r_distance_band(fd if fd > 0.0 else -fd, 0.16, 0.50)
    var lk = G1R_L_KNEE * 3
    var rk = G1R_R_KNEE * 3
    var kl = g1_rotate_inverse(
        qw, qx, qy, qz, _rd[DTYPE](xpos[env, lk]) - px,
        _rd[DTYPE](xpos[env, lk + 1]) - py, _rd[DTYPE](xpos[env, lk + 2]) - pz,
    )
    var kr = g1_rotate_inverse(
        qw, qx, qy, qz, _rd[DTYPE](xpos[env, rk]) - px,
        _rd[DTYPE](xpos[env, rk + 1]) - py, _rd[DTYPE](xpos[env, rk + 2]) - pz,
    )
    var kd_ = kl[1] - kr[1]
    terms[R_KNEE_DIST] = g1r_distance_band(kd_ if kd_ > 0.0 else -kd_, 0.18, 0.35)
    # foot orientation: gravity in each foot frame, xy^2.
    # ⚠ BODY `xquat` IS (x, y, z, w) — the root's `qpos[3:7]` is (w, x, y, z).
    # The first draft read xquat as (w, x, y, z): exact at the stand (an
    # identity read that way is a 180 deg yaw, gravity unchanged), wrong at
    # any tilt — `test_unitree_g1_walk_rp` pitches the robot to catch it.
    for f in range(2):
        var b4 = (G1R_L_ANKLE_ROLL if f == 0 else G1R_R_ANKLE_ROLL) * 4
        var gf = _projected_gravity(
            _rd[DTYPE](xquat[env, b4 + 3]), _rd[DTYPE](xquat[env, b4]),
            _rd[DTYPE](xquat[env, b4 + 1]), _rd[DTYPE](xquat[env, b4 + 2]),
        )
        terms[R_FEET_ORIENT] += gf[0] * gf[0] + gf[1] * gf[1]
    # swing-foot clearance in single stance
    if single and moving_cmd:
        var swing_z = _rd[DTYPE](xpos[env, la + 2]) if not fc.left else _rd[DTYPE](xpos[env, ra + 2])
        if swing_z - G1R_SOLE_BELOW_ANKLE > G1R_FOOT_CLEAR:
            terms[R_FEET_HEIGHT] = up
    if not moving_cmd and fc.left and fc.right:
        terms[R_FEET_STILL] = up

    # ── joints ──
    var key = Int(_rd[DTYPE](meta[env, G1R_KEY]))
    var pos_abs = 0.0
    var vel_abs = 0.0
    comptime for i in range(G1_N_DOF):
        var q = _rd[DTYPE](qpos[env, ROOT_QPOS_SIZE + i])
        var qd = _rd[DTYPE](qvel[env, ROOT_QVEL_SIZE + i])
        var e = q - g1_default_pos(i)
        var ea = e if e > 0.0 else -e
        pos_abs += ea
        vel_abs += qd if qd > 0.0 else -qd
        terms[R_JOINT_VEL] += qd * qd
        comptime c = 0.5 * (g1_pos_lower(i) + g1_pos_upper(i))
        comptime r = 0.5 * (g1_pos_upper(i) - g1_pos_lower(i)) * G1R_SOFT_LIMIT
        if q < c - r:
            terms[R_POS_LIMITS] += (c - r) - q
        elif q > c + r:
            terms[R_POS_LIMITS] += q - (c + r)
        var s = g1r_gain_scales(key, i)
        var tau = g1r_torque(i, _rd[DTYPE](actions[env, i]), q, qd, s[0], s[1])
        terms[R_TORQUES] += tau * tau
        var p = tau * qd
        terms[R_ENERGY] += p if p > 0.0 else -p
        comptime grp = g1r_dev_group(i)
        comptime if grp == 1:
            terms[R_DEV_HIP] += ea
        elif grp == 2:
            terms[R_DEV_TORSO] += ea
        elif grp == 3:
            terms[R_DEV_LEGS] += ea
        elif grp == 4:
            terms[R_DEV_ARMS] += ea
        elif grp == 5:
            terms[R_DEV_ARMS] += 0.06 * ea
    var body_vel = sqrt(vb[0] * vb[0] + vb[1] * vb[1]) + (wbz if wbz > 0.0 else -wbz)
    if not moving_cmd and body_vel <= 0.5:
        terms[R_STAND_STILL] = (pos_abs + 0.04 * vel_abs) * up

    var done = fc.terminate or pz != pz
    if done:
        terms[R_TERMINATION] = 1.0
    return done


@always_inline
def g1r_reward(terms: Array[Float64, G1R_N_TERMS]) -> Float64:
    """`dt x sum(w_k t_k)` — IsaacLab's reward manager scales every term by
    the step dt; no clip."""
    comptime DT_CTRL = G1_SIM_TIMESTEP * Float64(G1_CONTROL_DECIMATION)
    var r = 0.0
    comptime for t in range(G1R_N_TERMS):
        r += g1r_weight(t) * terms[t]
    return r * DT_CTRL


# ── draws: pre-step, reset ────────────────────────────────────────────────


@always_inline
def _push_velocity[DTYPE: DType, BATCH_SIZE: Int, NV: Int](
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    env: Int,
    p0: SIMD[DType.float32, 4],
    p1: SIMD[DType.float32, 4],
    add: Bool,
):
    """Their velocity ranges: x y +-0.5, z +-0.2, roll pitch +-0.52, yaw
    +-0.78 — added (push) or set (reset)."""
    var d = Array[Float64, 6](fill=0.0)
    d[0] = 0.5 * (2.0 * Float64(p0[0]) - 1.0)
    d[1] = 0.5 * (2.0 * Float64(p0[1]) - 1.0)
    d[2] = 0.2 * (2.0 * Float64(p0[2]) - 1.0)
    d[3] = 0.52 * (2.0 * Float64(p0[3]) - 1.0)
    d[4] = 0.52 * (2.0 * Float64(p1[0]) - 1.0)
    d[5] = 0.78 * (2.0 * Float64(p1[1]) - 1.0)
    for k in range(6):
        if add:
            qvel[env, k] = qvel[env, k] + Scalar[DTYPE](d[k])
        else:
            qvel[env, k] = Scalar[DTYPE](d[k])


@always_inline
def g1r_pre_step[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NV: Int, PUSHES: Bool
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
):
    _ = qpos
    var key = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, G1R_KEY])))
    var step = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_STEP_COUNT])))
    var rng = PhiloxRandom(seed=key, offset=step * 4)
    var timer = Int(rebind[Scalar[DTYPE]](meta[env, G1R_CMD_TIMER]))
    if timer >= 0:
        timer -= 1
        if timer <= 0:
            var u = rng.step_uniform()
            var cmd = g1r_command(
                Float64(u[0]), Float64(u[1]), Float64(u[2]), Float64(u[3])
            )
            meta[env, G1R_CMD_VX] = Scalar[DTYPE](cmd[0])
            meta[env, G1R_CMD_VY] = Scalar[DTYPE](cmd[1])
            meta[env, G1R_CMD_WZ] = Scalar[DTYPE](cmd[2])
            timer = G1R_CMD_PERIOD
        meta[env, G1R_CMD_TIMER] = Scalar[DTYPE](timer)
    comptime if PUSHES:
        var pt = Int(rebind[Scalar[DTYPE]](meta[env, G1R_PUSH_TIMER])) - 1
        if pt <= 0:
            var p0 = rng.step_uniform()
            var p1 = rng.step_uniform()
            _push_velocity[DTYPE, BATCH_SIZE, NV](qvel, env, p0, p1, True)
            pt = G1R_PUSH_MIN_STEPS + Int(
                Float64(G1R_PUSH_MAX_STEPS - G1R_PUSH_MIN_STEPS) * Float64(p1[2])
            )
        meta[env, G1R_PUSH_TIMER] = Scalar[DTYPE](pt)


@always_inline
def g1r_init[
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
    """RANDOM: their reset (module docstring). Otherwise the BFM stand at
    rest, a frozen zero command, key 0 (nominal gains)."""
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
        var pa = rng.step_uniform()
        var pb = rng.step_uniform()
        _push_velocity[DTYPE, BATCH_SIZE, NV](qvel, env, pa, pb, False)
        comptime for k in range(8):
            var u = rng.step_uniform()
            comptime for j in range(4):
                comptime i = k * 4 + j
                comptime if i < G1_N_DOF:
                    qpos[env, ROOT_QPOS_SIZE + i] = Scalar[DTYPE](
                        g1_default_pos(i) + 0.15 * (2.0 * Float64(u[j]) - 1.0)
                    )
        var pp = rng.step_uniform()
        meta[env, G1R_PUSH_TIMER] = Scalar[DTYPE](
            G1R_PUSH_MIN_STEPS + Int(
                Float64(G1R_PUSH_MAX_STEPS - G1R_PUSH_MIN_STEPS) * Float64(pp[0])
            )
        )
        meta[env, G1R_CMD_TIMER] = Scalar[DTYPE](0)
        meta[env, G1R_KEY] = Scalar[DTYPE](1 + Int(key % UInt64((1 << 23) - 1)))
    else:
        meta[env, G1R_PUSH_TIMER] = Scalar[DTYPE](G1R_PUSH_MAX_STEPS)
        meta[env, G1R_CMD_TIMER] = Scalar[DTYPE](-1)
        meta[env, G1R_KEY] = Scalar[DTYPE](0)
    meta[env, G1R_CMD_VX] = Scalar[DTYPE](0)
    meta[env, G1R_CMD_VY] = Scalar[DTYPE](0)
    meta[env, G1R_CMD_WZ] = Scalar[DTYPE](0)
    meta[env, G1R_AIR_L] = Scalar[DTYPE](0)
    meta[env, G1R_AIR_R] = Scalar[DTYPE](0)
    meta[env, G1R_CON_L] = Scalar[DTYPE](0)
    meta[env, G1R_CON_R] = Scalar[DTYPE](0)


@always_inline
def g1r_obs[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NV: Int, NBODY: Int, MC_F: Int,
    OBS_DIM: Int, NOISE: Bool,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE), MutAnyOrigin
    ],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    obs: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin],
    env: Int,
):
    """The 82-word observation. ⚠ The privileged feet clocks are the ones
    the PREVIOUS reward step left (the obs hook runs first in the env's
    extract kernel): one control step stale, like a sensor's history."""
    comptime assert OBS_DIM == G1R_OBS_DIM, "g1r_obs: OBS_DIM must be 82"
    var qw = _rd[DTYPE](qpos[env, 3])
    var qx = _rd[DTYPE](qpos[env, 4])
    var qy = _rd[DTYPE](qpos[env, 5])
    var qz = _rd[DTYPE](qpos[env, 6])
    var g = _projected_gravity(qw, qx, qy, qz)
    for k in range(3):
        obs[env, G1R_O_ANGVEL + k] = qvel[env, 3 + k]
    obs[env, G1R_O_GRAVITY + 0] = Scalar[DTYPE](g[0])
    obs[env, G1R_O_GRAVITY + 1] = Scalar[DTYPE](g[1])
    obs[env, G1R_O_GRAVITY + 2] = Scalar[DTYPE](g[2])
    obs[env, G1R_O_CMD + 0] = meta[env, G1R_CMD_VX]
    obs[env, G1R_O_CMD + 1] = meta[env, G1R_CMD_VY]
    obs[env, G1R_O_CMD + 2] = meta[env, G1R_CMD_WZ]
    comptime for i in range(G1_N_DOF):
        obs[env, G1R_O_Q + i] = qpos[env, ROOT_QPOS_SIZE + i] - Scalar[DTYPE](
            g1_default_pos(i)
        )
    for i in range(G1_N_DOF):
        obs[env, G1R_O_QD + i] = qvel[env, ROOT_QVEL_SIZE + i]
    # ── privileged ──
    var vb = g1_rotate_inverse(
        qw, qx, qy, qz, _rd[DTYPE](qvel[env, 0]), _rd[DTYPE](qvel[env, 1]),
        _rd[DTYPE](qvel[env, 2]),
    )
    obs[env, G1R_O_LINVEL + 0] = Scalar[DTYPE](vb[0])
    obs[env, G1R_O_LINVEL + 1] = Scalar[DTYPE](vb[1])
    obs[env, G1R_O_LINVEL + 2] = Scalar[DTYPE](vb[2])
    var fc = g1r_contacts[DTYPE, BATCH_SIZE, MC_F](contacts, meta, env)
    obs[env, G1R_O_CONTACT + 0] = Scalar[DTYPE](1.0 if fc.left else 0.0)
    obs[env, G1R_O_CONTACT + 1] = Scalar[DTYPE](1.0 if fc.right else 0.0)
    obs[env, G1R_O_FORCE + 0] = Scalar[DTYPE](fc.flx * G1R_FORCE_OBS_SCALE)
    obs[env, G1R_O_FORCE + 1] = Scalar[DTYPE](fc.fly * G1R_FORCE_OBS_SCALE)
    obs[env, G1R_O_FORCE + 2] = Scalar[DTYPE](fc.flz * G1R_FORCE_OBS_SCALE)
    obs[env, G1R_O_FORCE + 3] = Scalar[DTYPE](fc.frx * G1R_FORCE_OBS_SCALE)
    obs[env, G1R_O_FORCE + 4] = Scalar[DTYPE](fc.fry * G1R_FORCE_OBS_SCALE)
    obs[env, G1R_O_FORCE + 5] = Scalar[DTYPE](fc.frz * G1R_FORCE_OBS_SCALE)
    obs[env, G1R_O_AIR + 0] = meta[env, G1R_AIR_L]
    obs[env, G1R_O_AIR + 1] = meta[env, G1R_AIR_R]
    obs[env, G1R_O_HEIGHT + 0] = Scalar[DTYPE](
        _rd[DTYPE](xpos[env, G1R_L_ANKLE_ROLL * 3 + 2]) - G1R_SOLE_BELOW_ANKLE
    )
    obs[env, G1R_O_HEIGHT + 1] = Scalar[DTYPE](
        _rd[DTYPE](xpos[env, G1R_R_ANKLE_ROLL * 3 + 2]) - G1R_SOLE_BELOW_ANKLE
    )
    comptime if NOISE:
        var key = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, G1R_KEY])))
        var step = UInt64(Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_STEP_COUNT])))
        var rng = PhiloxRandom(seed=key + UInt64(0x9E3779B9), offset=step * 17)
        comptime for k in range(17):
            var u = rng.step_uniform()
            comptime for j in range(4):
                comptime i = k * 4 + j
                comptime if i < G1R_OBS_ACTOR:
                    comptime scale = (
                        G1R_NOISE_ANGVEL if i < G1R_O_GRAVITY
                        else (G1R_NOISE_GRAVITY if i < G1R_O_CMD
                        else (0.0 if i < G1R_O_Q
                        else (G1R_NOISE_Q if i < G1R_O_QD else G1R_NOISE_QD)))
                    )
                    comptime if scale > 0.0:
                        obs[env, i] = obs[env, i] + Scalar[DTYPE](
                            scale * (2.0 * Float64(u[j]) - 1.0)
                        )


def _host_copy[DTYPE: DType](src: TensorImpl[DTYPE], n: Int) -> TensorImpl[DTYPE]:
    var t = TensorImpl[DTYPE].alloc(n)
    for i in range(n):
        t.data[i] = src.data[i]
    return t^


# ── the config ────────────────────────────────────────────────────────────


struct UnitreeG1WalkRPConfig[TRAIN: Bool = True](Phyics3dEnvConfig):
    """TRAIN: their resets, command draws, observation noise, pushes and
    gain DR. False: stand reset, frozen command, nominal gains, no noise."""

    comptime FRAME_SKIP: Int = G1_CONTROL_DECIMATION
    comptime MAX_STEPS: Int = G1R_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = True
    comptime CUSTOM_ACTIONS_EVERY_SUBSTEP: Bool = True
    comptime NORMALIZED_ACTIONS: Bool = False
    comptime NMESH_VERTS: Int = UNITREE_G1_NMESH_VERTS

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
        var key = Int(d.meta.data[G1R_KEY])
        for i in range(D.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        for i in range(G1_N_DOF):
            var a = actions[i] if i < len(actions) else 0.0
            var q = Float64(d.qpos.data[ROOT_QPOS_SIZE + i])
            var qd = Float64(d.qvel.data[ROOT_QVEL_SIZE + i])
            var s = g1r_gain_scales(key, i)
            var ao = i * MODEL_ACTUATOR_SIZE
            d.qfrc.data[ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1r_torque(i, a, q, qd, s[0], s[1]),
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
        """`g1r_obs` without noise, over copies of the CPU lane (`d` is
        borrowed immutably; `lt` is a mutating view)."""
        var q = _host_copy(d.qpos, D.NQ)
        var v = _host_copy(d.qvel, D.NV)
        var x = _host_copy(d.xpos, D.NBODY * 3)
        var c = _host_copy(d.contacts, D.MAX_CONTACTS * CONTACT_SIZE)
        var m = _host_copy(d.meta, METADATA_SIZE)
        var o = TensorImpl[DTYPE].alloc(G1R_OBS_DIM)
        g1r_obs[DTYPE, 1, D.NQ, D.NV, D.NBODY, D.MAX_CONTACTS, G1R_OBS_DIM, False](
            q.lt["cpu", Layout.row_major(1, D.NQ)](),
            v.lt["cpu", Layout.row_major(1, D.NV)](),
            x.lt["cpu", Layout.row_major(1, D.NBODY * 3)](),
            c.lt["cpu", Layout.row_major(1, D.MAX_CONTACTS * CONTACT_SIZE)](),
            m.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
            o.lt["cpu", Layout.row_major(1, G1R_OBS_DIM)](),
            0,
        )
        for i in range(G1R_OBS_DIM):
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
        g1r_init[DTYPE, 1, D.NQ, D.NV, Self.TRAIN](
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
        """Termination only; rewards on the host: `g1r_host_terms`."""
        var c = _host_copy(d.contacts, D.MAX_CONTACTS * CONTACT_SIZE)
        var m = _host_copy(d.meta, METADATA_SIZE)
        var fc = g1r_contacts[DTYPE, 1, D.MAX_CONTACTS](
            c.lt["cpu", Layout.row_major(1, D.MAX_CONTACTS * CONTACT_SIZE)](),
            m.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
            0,
        )
        var pz = Float64(d.qpos.data[2])
        return (Scalar[DTYPE](0), fc.terminate or pz != pz)

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
        qfrc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
        actions: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin],
        qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
        qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
        act: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin],
        meta: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin],
        joints: LayoutTensor[DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin],
        tendons: LayoutTensor[DTYPE, Layout.row_major(NTENDON_F, MODEL_TENDON_SIZE), MutAnyOrigin],
        acts: LayoutTensor[DTYPE, Layout.row_major(NACT_F * MODEL_ACTUATOR_SIZE), MutAnyOrigin],
        act_tendons: LayoutTensor[
            DTYPE, Layout.row_major(NTENDON_F * MODEL_ACT_TENDON_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        var key = Int(rebind[Scalar[DTYPE]](meta[env, G1R_KEY]))
        for i in range(NV):
            qfrc[env, i] = Scalar[DTYPE](0)
        comptime for i in range(G1_N_DOF):
            var a = Float64(rebind[Scalar[DTYPE]](actions[env, i]))
            var q = Float64(rebind[Scalar[DTYPE]](qpos[env, ROOT_QPOS_SIZE + i]))
            var qd = Float64(rebind[Scalar[DTYPE]](qvel[env, ROOT_QVEL_SIZE + i]))
            var s = g1r_gain_scales(key, i)
            comptime ao = i * MODEL_ACTUATOR_SIZE
            qfrc[env, ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1r_torque(i, a, q, qd, s[0], s[1]),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MIN])),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MAX])),
                )
            )

    @always_inline
    @staticmethod
    def pre_step_full_gpu[
        DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NV: Int,
    ](
        qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
        qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
        meta: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin],
        env: Int,
    ):
        g1r_pre_step[DTYPE, BATCH_SIZE, NQ, NV, Self.TRAIN](qpos, qvel, meta, env)

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
        qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
        qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
        xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        xquat: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin],
        xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        bodies: LayoutTensor[DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin],
        site_xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin],
        contacts: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE), MutAnyOrigin
        ],
        sites: LayoutTensor[DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin],
        geoms: LayoutTensor[DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin],
        meta: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin],
        obs: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin],
        xipos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        xangvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        cvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        cacc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        cfrc_int: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        subtree_com: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        site_xpos_acc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin],
        xquat_acc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin],
        act: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin],
        env: Int,
    ) -> Bool:
        g1r_obs[DTYPE, BATCH_SIZE, NQ, NV, NBODY, MC_F, OBS_DIM, Self.TRAIN](
            qpos, qvel, xpos, contacts, meta, obs, env
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
        qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
        qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
        xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        xipos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        xquat: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin],
        xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        bodies: LayoutTensor[DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin],
        site_xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin],
        contacts: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE), MutAnyOrigin
        ],
        sites: LayoutTensor[DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin],
        geoms: LayoutTensor[DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin],
        cfrc_ext: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        cvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        meta: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin],
        curriculum: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin],
        actions: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin],
        xangvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        cacc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        cfrc_int: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 6), MutAnyOrigin],
        subtree_com: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        site_xpos_acc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin],
        xquat_acc: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin],
        act: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
        var done = g1r_terms[DTYPE, BATCH_SIZE, NQ, NV, NBODY, ACTION_DIM, MC_F](
            qpos, qvel, xpos, xquat, xvel, contacts, meta, actions, env, terms
        )
        return (Scalar[DTYPE](g1r_reward(terms)), done)

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
        qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin],
        qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin],
        joints: LayoutTensor[DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin],
        mocap_pos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 3), MutAnyOrigin],
        mocap_quat: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NBODY * 4), MutAnyOrigin],
        bodies: LayoutTensor[DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin],
        geoms: LayoutTensor[DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin],
        meta: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin],
        env: Int,
        seed: Int,
    ):
        g1r_init[DTYPE, BATCH_SIZE, NQ, NV, Self.TRAIN](qpos, qvel, meta, env, seed)
