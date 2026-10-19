"""Reacher environment configuration for generic Phyics3dEnv.

Gymnasium Reacher-v5 equivalent.
Observation: [cos(q0), cos(q1), sin(q0), sin(q1), qpos[2:4], qvel[0:2], delta_xy]
Reward: -||fingertip - target|| - sum(action^2)
No early termination; truncated after 50 steps.
"""

from std.math import sin, cos, sqrt
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

from .reacher_xml import ReacherModel

from ..phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# Body indices (depth-first traversal of XML body tree)
comptime FINGERTIP_BODY_IDX: Int = 3
comptime TARGET_BODY_IDX: Int = 4


struct ReacherConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 2
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 50
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        ReacherModel.NQ, ReacherModel.NV
    ]()

    # Reward weights (Gymnasium v5 defaults)
    comptime REWARD_DIST_WEIGHT = 1.0
    comptime REWARD_CTRL_WEIGHT = 1.0

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        pass  # No pre-step state needed for Reacher

    # === CPU: Custom observation extraction ===
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
        """Gymnasium Reacher-v5 observation: cos/sin encoding + target pos + vel + delta.
        """
        var q0 = Float64(d.qpos.data[0])
        var q1 = Float64(d.qpos.data[1])

        # cos(theta) [2]
        obs.append(Scalar[DTYPE](cos(q0)))
        obs.append(Scalar[DTYPE](cos(q1)))
        # sin(theta) [2]
        obs.append(Scalar[DTYPE](sin(q0)))
        obs.append(Scalar[DTYPE](sin(q1)))
        # target joint positions (qpos[2:4]) [2]
        obs.append(d.qpos.data[2])
        obs.append(d.qpos.data[3])
        # joint velocities (qvel[0:2]) [2]
        obs.append(d.qvel.data[0])
        obs.append(d.qvel.data[1])
        # fingertip - target world position delta (x, y only) [2]
        var ftip_x = d.xpos.data[FINGERTIP_BODY_IDX * 3 + 0]
        var ftip_y = d.xpos.data[FINGERTIP_BODY_IDX * 3 + 1]
        var tgt_x = d.xpos.data[TARGET_BODY_IDX * 3 + 0]
        var tgt_y = d.xpos.data[TARGET_BODY_IDX * 3 + 1]
        obs.append(ftip_x - tgt_x)
        obs.append(ftip_y - tgt_y)
        return True

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
        # Distance: fingertip to target (3D Euclidean norm)
        var dx = Float64(d.xpos.data[FINGERTIP_BODY_IDX * 3 + 0]) - Float64(
            d.xpos.data[TARGET_BODY_IDX * 3 + 0]
        )
        var dy = Float64(d.xpos.data[FINGERTIP_BODY_IDX * 3 + 1]) - Float64(
            d.xpos.data[TARGET_BODY_IDX * 3 + 1]
        )
        var dz = Float64(d.xpos.data[FINGERTIP_BODY_IDX * 3 + 2]) - Float64(
            d.xpos.data[TARGET_BODY_IDX * 3 + 2]
        )
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var reward_dist = -dist * Self.REWARD_DIST_WEIGHT

        # Control cost: sum of squared actions
        var ctrl_cost = Float64(0)
        for i in range(len(actions)):
            ctrl_cost += actions[i] * actions[i]
        var reward_ctrl = -ctrl_cost * Self.REWARD_CTRL_WEIGHT

        var reward = Scalar[DTYPE](reward_dist + reward_ctrl)

        # Reacher never terminates early
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(ReacherModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.1

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
        pass  # No pre-step state needed

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
        # Fingertip - target distance (3D)
        var ftip_x = rebind[Scalar[DTYPE]](
            xpos[env, FINGERTIP_BODY_IDX * 3 + 0]
        )
        var ftip_y = rebind[Scalar[DTYPE]](
            xpos[env, FINGERTIP_BODY_IDX * 3 + 1]
        )
        var ftip_z = rebind[Scalar[DTYPE]](
            xpos[env, FINGERTIP_BODY_IDX * 3 + 2]
        )
        var tgt_x = rebind[Scalar[DTYPE]](
            xpos[env, TARGET_BODY_IDX * 3 + 0]
        )
        var tgt_y = rebind[Scalar[DTYPE]](
            xpos[env, TARGET_BODY_IDX * 3 + 1]
        )
        var tgt_z = rebind[Scalar[DTYPE]](
            xpos[env, TARGET_BODY_IDX * 3 + 2]
        )
        var dx = ftip_x - tgt_x
        var dy = ftip_y - tgt_y
        var dz = ftip_z - tgt_z
        var dist = sqrt(dx * dx + dy * dy + dz * dz)

        var reward_dist = -dist * Scalar[DTYPE](Self.REWARD_DIST_WEIGHT)

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0)
        comptime for i in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, i])
            ctrl_cost += a * a
        var reward_ctrl = -ctrl_cost * Scalar[DTYPE](Self.REWARD_CTRL_WEIGHT)

        var reward = reward_dist + reward_ctrl

        # Reacher never terminates
        return (reward, False)

    # === GPU inline: Non-zero qpos init (no-op) ===
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
        pass

    # === GPU inline: Custom obs extraction (10D with cos/sin + delta) ===
    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        bodies: Tile[DTYPE, NBODY_F, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
    ) -> Bool:
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        var q0 = rebind[Scalar[DTYPE]](qpos[env, 0])
        var q1 = rebind[Scalar[DTYPE]](qpos[env, 1])

        # cos(theta) [2]
        obs[env, 0] = cos(q0)
        obs[env, 1] = cos(q1)
        # sin(theta) [2]
        obs[env, 2] = sin(q0)
        obs[env, 3] = sin(q1)
        # target joint positions (qpos[2:4]) [2]
        obs[env, 4] = rebind[Scalar[DTYPE]](qpos[env, 2])
        obs[env, 5] = rebind[Scalar[DTYPE]](qpos[env, 3])
        # joint velocities (qvel[0:2]) [2]
        obs[env, 6] = rebind[Scalar[DTYPE]](qvel[env, 0])
        obs[env, 7] = rebind[Scalar[DTYPE]](qvel[env, 1])
        # fingertip - target delta (x, y) [2]
        var ftip_x = rebind[Scalar[DTYPE]](
            xpos[env, FINGERTIP_BODY_IDX * 3 + 0]
        )
        var ftip_y = rebind[Scalar[DTYPE]](
            xpos[env, FINGERTIP_BODY_IDX * 3 + 1]
        )
        var tgt_x = rebind[Scalar[DTYPE]](
            xpos[env, TARGET_BODY_IDX * 3 + 0]
        )
        var tgt_y = rebind[Scalar[DTYPE]](
            xpos[env, TARGET_BODY_IDX * 3 + 1]
        )
        obs[env, 8] = ftip_x - tgt_x
        obs[env, 9] = ftip_y - tgt_y

        return True
