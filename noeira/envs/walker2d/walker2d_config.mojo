"""Walker2d environment configuration for generic Phyics3dEnv."""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .walker2d_xml import Walker2dModel

from ..phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


struct Walker2dConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 4
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        Walker2dModel.NQ, Walker2dModel.NV
    ]()

    # Reward weights
    comptime FORWARD_REWARD_WEIGHT = 1.0
    comptime CTRL_COST_WEIGHT = 0.001
    comptime HEALTHY_REWARD = 1.0

    # Health bounds in world-z coordinates (qpos[1] = rootz, reset to ref=1.25).
    # Gymnasium: healthy_z_range = (0.8, 2.0) — exclusive on both ends.
    comptime MIN_Z = 0.8  # qpos[1] lower bound (world z)
    comptime MAX_Z = 2.0  # qpos[1] upper bound (world z)
    comptime MAX_ANGLE = 1.0  # |qpos[2]| (rooty) upper bound

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        prev_x = d.qpos.data[0]  # Save rootx position

    # === CPU: Reward + termination ===
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
        # x velocity from position change
        var x_after = d.qpos.data[0]
        var dt = Scalar[DTYPE](Self.get_timestep()) * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / dt

        # Health check: qpos[1]=rootz, qpos[2]=rooty
        var z = d.qpos.data[1]
        var angle = d.qpos.data[2]
        var abs_angle = angle if angle >= Scalar[DTYPE](0.0) else -angle
        var is_healthy = (
            z > Scalar[DTYPE](Self.MIN_Z)
            and z < Scalar[DTYPE](Self.MAX_Z)
            and abs_angle < Scalar[DTYPE](Self.MAX_ANGLE)
        )

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0.0)
        for i in range(len(actions)):
            ctrl_cost += Scalar[DTYPE](actions[i] * actions[i])
        ctrl_cost = Scalar[DTYPE](Self.CTRL_COST_WEIGHT) * ctrl_cost

        # Healthy reward
        var healthy_reward = Scalar[DTYPE](0.0)
        if is_healthy:
            healthy_reward = Scalar[DTYPE](Self.HEALTHY_REWARD)

        var forward_reward = (
            Scalar[DTYPE](Self.FORWARD_REWARD_WEIGHT) * x_velocity
        )
        var reward = forward_reward + healthy_reward - ctrl_cost

        return (reward, not is_healthy)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(Walker2dModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.005

    # === GPU inline: Pre-step hook ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
    ):
        # Save rootx position into META_IDX_PREV_X
        meta[env, META_IDX_PREV_X] = qpos[env, 0]

    # === GPU inline: Reward + termination ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        bodies: Tile[DTYPE, NBODY_F, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # x velocity from position change
        var x_after = rebind[Scalar[DTYPE]](qpos[env, 0])
        var prev_x = rebind[Scalar[DTYPE]](
            meta[env, META_IDX_PREV_X]
        )
        var effective_dt = timestep * Scalar[DTYPE](frame_skip)
        var x_velocity = (x_after - prev_x) / effective_dt

        # Health check: qpos[1]=rootz, qpos[2]=rooty
        var z = rebind[Scalar[DTYPE]](qpos[env, 1])
        var angle = rebind[Scalar[DTYPE]](qpos[env, 2])
        var abs_angle = angle
        if abs_angle < Scalar[DTYPE](0.0):
            abs_angle = -abs_angle

        var is_healthy = (
            z > Scalar[DTYPE](0.8)
            and z < Scalar[DTYPE](2.0)
            and abs_angle < Scalar[DTYPE](1.0)
        )

        # Control cost (clamp actions)
        var ctrl_cost_sum = Scalar[DTYPE](0.0)
        for a_idx in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, a_idx])
            if a > Scalar[DTYPE](1.0):
                a = Scalar[DTYPE](1.0)
            elif a < Scalar[DTYPE](-1.0):
                a = Scalar[DTYPE](-1.0)
            ctrl_cost_sum += a * a
        var ctrl_cost = Scalar[DTYPE](0.001) * ctrl_cost_sum

        var healthy_reward = Scalar[DTYPE](1.0)
        if not is_healthy:
            healthy_reward = Scalar[DTYPE](0.0)

        var reward = x_velocity + healthy_reward - ctrl_cost

        return (reward, not is_healthy)

    # === GPU inline: Non-zero qpos init (no-op for Walker2d) ===
    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NJOINT_F: Int,
        NV_F: Int,
        NBODY_M: Int,
        NGEOM_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        joints: Tile[DTYPE, NJOINT_F, MODEL_JOINT_SIZE],
        mocap_pos: Tile[DTYPE, BATCH_SIZE, NBODY_M * 3],
        mocap_quat: Tile[DTYPE, BATCH_SIZE, NBODY_M * 4],
        bodies: Tile[DTYPE, NBODY_M, MODEL_BODY_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
        seed: Int,
    ):
        # rootz joint ref=1.25 in MJCF → qpos[1] resets to 1.25 (world z).
        # Health bounds use world-z coordinates directly.
        pass

