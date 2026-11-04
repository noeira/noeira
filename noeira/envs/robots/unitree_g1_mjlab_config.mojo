"""Unitree G1 as mjlab models it — the env: mjlab's PD law, a 64-D proprio state.

    action  = 29 values in mjlab's convention (`JointPositionAction`,
              `use_default_offset`): target = q_default + scale · a, NO clip,
              scale = 0.25 · effort / kp (`G1_ACTION_SCALE`)
    torque  = kp (target − q) − kd q̇, clipped to ± effort, EVERY substep —
              mjlab's `BuiltinPositionActuator` (a `<position kp kv
              forcerange>` servo), computed here because the motors are
              torque motors (see `unitree_g1_mjlab_xml.mojo`)
    control = 50 Hz = 4 substeps of 0.005 s (the reproduction's decimation)
    obs     = [q − q_default (29), q̇ (29), projected gravity (3), root
              angular velocity (3)] (64) — a diagnostic; SuperMimic builds
              its own observation from qpos / qvel
    reward  = 0, no termination (the task layer owns both)
    reset   = the `stand` keyframe: mjlab's `KNEES_BENT_KEYFRAME`

Gains and defaults: the generated `unitree_g1_mjlab_pd.mojo`.
`test_unitree_g1_mjlab_vs_mujoco` drives MuJoCo with a separate Python
transcription of the same law per substep and compares `qpos`.
"""

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
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
from .unitree_g1_mjlab_xml import ROOT_QPOS_SIZE, ROOT_QVEL_SIZE
from .unitree_g1_mjlab_pd import (
    G1M_N_DOF,
    G1M_INIT_ROOT_Z,
    G1M_SIM_TIMESTEP,
    G1M_CONTROL_DECIMATION,
    g1m_kp,
    g1m_kd,
    g1m_effort,
    g1m_default_pos,
    g1m_action_scale,
)
from noeira.core.tiles import Tile


@always_inline
def g1m_pd_target(i: Int, a: Float64) -> Float64:
    """mjlab's `JointPositionAction`: offset + scale · a, unclipped."""
    return g1m_default_pos(i) + g1m_action_scale(i) * a


@always_inline
def g1m_pd_torque(i: Int, target: Float64, q: Float64, qd: Float64) -> Float64:
    """The position servo's force, clipped to its forcerange (± effort)."""
    var tau = g1m_kp(i) * (target - q) - g1m_kd(i) * qd
    var lim = g1m_effort(i)
    if tau > lim:
        return lim
    if tau < -lim:
        return -lim
    return tau


@always_inline
def _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    if x > hi:
        return hi
    if x < lo:
        return lo
    return x


@always_inline
def _projected_gravity(qw: Float64, qx: Float64, qy: Float64, qz: Float64) -> Tuple[Float64, Float64, Float64]:
    """`R(q)^T (0, 0, -1)`."""
    return (
        2.0 * (qw * qy - qx * qz),
        -2.0 * (qy * qz + qw * qx),
        2.0 * (qx * qx + qy * qy) - 1.0,
    )


struct UnitreeG1MjlabConfig(Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = G1M_CONTROL_DECIMATION
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
        motor), clamped by the motor's ctrlrange (= ± effort, the same
        number: one clip in effect). Called every substep."""
        for i in range(D.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        for i in range(G1M_N_DOF):
            var a = actions[i] if i < len(actions) else 0.0
            var q = Float64(d.qpos.data[ROOT_QPOS_SIZE + i])
            var qd = Float64(d.qvel.data[ROOT_QVEL_SIZE + i])
            var ao = i * MODEL_ACTUATOR_SIZE
            d.qfrc.data[ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1m_pd_torque(i, g1m_pd_target(i, a), q, qd),
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
        for i in range(G1M_N_DOF):
            obs.append(d.qpos.data[ROOT_QPOS_SIZE + i] - Scalar[DTYPE](g1m_default_pos(i)))
        for i in range(G1M_N_DOF):
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
        """The `stand` keyframe: pelvis at (0, 0, 0.76), identity, the
        knees-bent angles, at rest."""
        for i in range(D.NQ):
            d.qpos.data[i] = Scalar[DTYPE](0)
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)
        d.qpos.data[2] = Scalar[DTYPE](G1M_INIT_ROOT_Z)
        d.qpos.data[3] = Scalar[DTYPE](1)
        for i in range(G1M_N_DOF):
            d.qpos.data[ROOT_QPOS_SIZE + i] = Scalar[DTYPE](g1m_default_pos(i))

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
        return G1M_SIM_TIMESTEP

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
        for i in range(G1M_N_DOF):
            var a = Float64(rebind[Scalar[DTYPE]](actions[env, i]))
            var q = Float64(rebind[Scalar[DTYPE]](qpos[env, ROOT_QPOS_SIZE + i]))
            var qd = Float64(rebind[Scalar[DTYPE]](qvel[env, ROOT_QVEL_SIZE + i]))
            var ao = i * MODEL_ACTUATOR_SIZE
            qfrc[env, ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1m_pd_torque(i, g1m_pd_target(i, a), q, qd),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MIN])),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MAX])),
                )
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
        comptime for i in range(G1M_N_DOF):
            obs[env, i] = qpos[env, ROOT_QPOS_SIZE + i] - Scalar[DTYPE](g1m_default_pos(i))
        for i in range(G1M_N_DOF):
            obs[env, G1M_N_DOF + i] = qvel[env, ROOT_QVEL_SIZE + i]
        var g = _projected_gravity(
            Float64(rebind[Scalar[DTYPE]](qpos[env, 3])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 4])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 5])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 6])),
        )
        var b = 2 * G1M_N_DOF
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
        qpos[env, 2] = Scalar[DTYPE](G1M_INIT_ROOT_Z)
        qpos[env, 3] = Scalar[DTYPE](1)
        comptime for i in range(G1M_N_DOF):
            qpos[env, ROOT_QPOS_SIZE + i] = Scalar[DTYPE](g1m_default_pos(i))
