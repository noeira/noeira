"""Berkeley Humanoid Lite — the env: BHL's PD law, a 50-D proprio state.

    action  = 22 values: target = q_default + 0.25 · a, NO clip (BHL's
              `JointPositionActionCfg(scale=0.25, use_default_offset=True)`)
    torque  = kp (target − q) − kd q̇, clipped to ± effort, EVERY substep, and
              set to 0 when it would push a joint that already moves faster
              than its speed limit. Gains and limits: BHL's `HUMANOID_LITE_CFG`
              (kp 20 / kd 2 / 6 N·m on the legs, kp 10 / kd 2 / 4 N·m on the
              arms, 10 rad/s everywhere), the generated
              `berkeley_humanoid_lite_pd.mojo`
    control = 50 Hz = 10 substeps of 0.002 s
    obs     = [q − q_default (22), q̇ (22), projected gravity (3), root
              angular velocity (3)] (50) — a diagnostic; SuperMimic builds
              its own observation from qpos / qvel
    reward  = 0, no termination (the task layer owns both)
    reset   = the `stand` keyframe: BHL's `init_state`, soles on the ground

⚠ THE SPEED LIMIT IS OURS. Isaac Lab's `velocity_limit` caps the joint speed
inside PhysX; physics3d has no such cap, so the env removes any torque that
would accelerate a joint beyond the limit (a motor at its no-load speed gives
none). Braking torque is untouched.

TWO BUILD-TIME KNOBS, for the question "which limit binds": `-D
BHL_EFFORT_PCT=150` scales every effort limit (100 = BHL's), `-D
BHL_SPEED_PCT=200` scales every speed limit (100 = BHL's, 0 = no limit).
The model's ctrlrange stays BHL's and is NOT applied here: the env's clip is
the only one.

`test_berkeley_humanoid_lite_vs_mujoco` drives MuJoCo with a separate Python
transcription of the same law per substep and compares `qpos`.
"""

from std.sys import get_defined_int

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
)
from ..phyics3d_env_config import Phyics3dEnvConfig
from .berkeley_humanoid_lite_xml import ROOT_QPOS_SIZE, ROOT_QVEL_SIZE
from .berkeley_humanoid_lite_pd import (
    BHL_N_DOF,
    BHL_INIT_ROOT_Z,
    BHL_SIM_TIMESTEP,
    BHL_CONTROL_DECIMATION,
    BHL_ACTION_SCALE,
    bhl_kp,
    bhl_kd,
    bhl_effort,
    bhl_speed,
    bhl_default_pos,
)
from noeira.core.tiles import Tile


comptime BHL_EFFORT_PCT: Int = get_defined_int["BHL_EFFORT_PCT", 100]()
comptime BHL_SPEED_PCT: Int = get_defined_int["BHL_SPEED_PCT", 100]()


@always_inline
def bhl_effort_limit(i: Int) -> Float64:
    """The effort limit in force: BHL's, scaled by `-D BHL_EFFORT_PCT`."""
    return bhl_effort(i) * Float64(BHL_EFFORT_PCT) / 100.0


@always_inline
def bhl_speed_limit(i: Int) -> Float64:
    """The speed limit in force: BHL's, scaled by `-D BHL_SPEED_PCT`; 0 = none."""
    return bhl_speed(i) * Float64(BHL_SPEED_PCT) / 100.0


@always_inline
def bhl_pd_target(i: Int, a: Float64) -> Float64:
    """BHL's joint-position action: offset + 0.25 · a, unclipped."""
    return bhl_default_pos(i) + BHL_ACTION_SCALE * a


@always_inline
def bhl_pd_torque(i: Int, target: Float64, q: Float64, qd: Float64) -> Float64:
    """The servo's torque: clipped to ± effort, and 0 when it would speed up a
    joint already beyond its speed limit."""
    var tau = bhl_kp(i) * (target - q) - bhl_kd(i) * qd
    var lim = bhl_effort_limit(i)
    if tau > lim:
        tau = lim
    if tau < -lim:
        tau = -lim
    var vmax = bhl_speed_limit(i)
    if vmax > 0.0:
        if qd > vmax and tau > 0.0:
            return 0.0
        if qd < -vmax and tau < 0.0:
            return 0.0
    return tau


@always_inline
def _projected_gravity(qw: Float64, qx: Float64, qy: Float64, qz: Float64) -> Tuple[Float64, Float64, Float64]:
    """`R(q)^T (0, 0, -1)`."""
    return (
        2.0 * (qw * qy - qx * qz),
        -2.0 * (qy * qz + qw * qx),
        2.0 * (qx * qx + qy * qy) - 1.0,
    )


struct BerkeleyHumanoidLiteConfig(Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = BHL_CONTROL_DECIMATION
    comptime MAX_STEPS: Int = 500
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = True
    comptime CUSTOM_ACTIONS_EVERY_SUBSTEP: Bool = True
    comptime NORMALIZED_ACTIONS: Bool = False

    # ── CPU hooks ─────────────────────────────────────────────────────────

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
        """The servo torque on every hinge, straight into `qfrc` (a gear-1
        motor). Called every substep."""
        for i in range(D.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        for i in range(BHL_N_DOF):
            var a = actions[i] if i < len(actions) else 0.0
            var q = Float64(d.qpos.data[ROOT_QPOS_SIZE + i])
            var qd = Float64(d.qvel.data[ROOT_QVEL_SIZE + i])
            d.qfrc.data[ROOT_QVEL_SIZE + i] = Scalar[DTYPE](bhl_pd_torque(i, bhl_pd_target(i, a), q, qd))
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
        for i in range(BHL_N_DOF):
            obs.append(d.qpos.data[ROOT_QPOS_SIZE + i] - Scalar[DTYPE](bhl_default_pos(i)))
        for i in range(BHL_N_DOF):
            obs.append(d.qvel.data[ROOT_QVEL_SIZE + i])
        var g = _projected_gravity(
            Float64(d.qpos.data[3]), Float64(d.qpos.data[4]), Float64(d.qpos.data[5]), Float64(d.qpos.data[6]),
        )
        obs.append(Scalar[DTYPE](g[0]))
        obs.append(Scalar[DTYPE](g[1]))
        obs.append(Scalar[DTYPE](g[2]))
        for k in range(3):
            obs.append(d.qvel.data[3 + k])
        return True

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """The `stand` keyframe: the base frame at floor level (z ≈ −0.04),
        identity, BHL's default angles, at rest."""
        for i in range(D.NQ):
            d.qpos.data[i] = Scalar[DTYPE](0)
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)
        d.qpos.data[2] = Scalar[DTYPE](BHL_INIT_ROOT_Z)
        d.qpos.data[3] = Scalar[DTYPE](1)
        for i in range(BHL_N_DOF):
            d.qpos.data[ROOT_QPOS_SIZE + i] = Scalar[DTYPE](bhl_default_pos(i))

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
        return (Scalar[DTYPE](0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return BHL_SIM_TIMESTEP

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
        qfrc: Tile[DTYPE, BATCH_SIZE, NV],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
        tendons: Tile[DTYPE, NTENDON_F, MODEL_TENDON_SIZE],
        acts: Tile[DTYPE, NACT_F * MODEL_ACTUATOR_SIZE],
        act_tendons: Tile[DTYPE, NTENDON_F * MODEL_ACT_TENDON_SIZE],
        env: Int,
    ):
        """The CPU hook's twin, one lane, every substep."""
        for i in range(NV):
            qfrc[env, i] = Scalar[DTYPE](0)
        for i in range(BHL_N_DOF):
            var a = Float64(rebind[Scalar[DTYPE]](actions[env, i]))
            var q = Float64(rebind[Scalar[DTYPE]](qpos[env, ROOT_QPOS_SIZE + i]))
            var qd = Float64(rebind[Scalar[DTYPE]](qvel[env, ROOT_QVEL_SIZE + i]))
            qfrc[env, ROOT_QVEL_SIZE + i] = Scalar[DTYPE](bhl_pd_torque(i, bhl_pd_target(i, a), q, qd))

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
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
    ) -> Bool:
        """The CPU observation's order. The default angles are unrolled so
        they fold into immediates (a Metal kernel cannot carry a double)."""
        comptime for i in range(BHL_N_DOF):
            obs[env, i] = qpos[env, ROOT_QPOS_SIZE + i] - Scalar[DTYPE](bhl_default_pos(i))
        for i in range(BHL_N_DOF):
            obs[env, BHL_N_DOF + i] = qvel[env, ROOT_QVEL_SIZE + i]
        var g = _projected_gravity(
            Float64(rebind[Scalar[DTYPE]](qpos[env, 3])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 4])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 5])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 6])),
        )
        var b = 2 * BHL_N_DOF
        obs[env, b + 0] = Scalar[DTYPE](g[0])
        obs[env, b + 1] = Scalar[DTYPE](g[1])
        obs[env, b + 2] = Scalar[DTYPE](g[2])
        for k in range(3):
            obs[env, b + 3 + k] = qvel[env, 3 + k]
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
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return (Scalar[DTYPE](0), False)

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
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
        mocap_pos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        mocap_quat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
        seed: Int,
    ):
        """`custom_reset_cpu`'s twin on every lane."""
        for i in range(NQ):
            qpos[env, i] = Scalar[DTYPE](0)
        for i in range(NV):
            qvel[env, i] = Scalar[DTYPE](0)
        qpos[env, 2] = Scalar[DTYPE](BHL_INIT_ROOT_Z)
        qpos[env, 3] = Scalar[DTYPE](1)
        comptime for i in range(BHL_N_DOF):
            qpos[env, ROOT_QPOS_SIZE + i] = Scalar[DTYPE](bhl_default_pos(i))
