"""Pusher environment configuration for generic Phyics3dEnv.

Gymnasium Pusher-v5 equivalent.
Observation: [qpos[:7], qvel[:7], tips_arm_xpos(3), object_xpos(3), goal_xpos(3)]
Reward: -||obj - goal|| - 0.1 * sum(action^2) - 0.5 * ||fingertip - obj||
No early termination; truncated after 100 steps.
Zero gravity table-top manipulation.
"""

from std.math import sqrt
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
)

from .pusher_xml import PusherModel

from ..phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# Body indices (depth-first traversal of XML body tree)
comptime TIPS_ARM_BODY_IDX: Int = 10  # End effector (fingertip)
comptime OBJECT_BODY_IDX: Int = 11  # Pushable cylinder
comptime GOAL_BODY_IDX: Int = 12  # Target position

# Number of arm joints (first 7 joints are the arm)
comptime NUM_ARM_JOINTS: Int = 7


struct PusherConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 100
    comptime INTEGRATOR_WS_EXTRA: Int = 0  # EulerIntegrator needs no extra workspace
    comptime INTEGRATOR: StaticString = "euler"  # matches physics_substep (Euler+Newton)

    # Reward weights (Gymnasium v5 defaults)
    comptime REWARD_DIST_WEIGHT = 1.0  # ||object - goal||
    comptime REWARD_CTRL_WEIGHT = 0.1  # sum(action^2)
    comptime REWARD_NEAR_WEIGHT = 0.5  # ||fingertip - object||

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        pass  # No pre-step state needed for Pusher

    # === CPU: Custom reset — set goal joints to 0 (fixed position) ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](mut d: Data[DTYPE, D, 1]):
        # Fix goal position: set goal slide joints to 0 (stays at XML body pos)
        # Goal joints are at indices 9 and 10 (goal_slidey, goal_slidex)
        d.qpos.data[9] = Scalar[DTYPE](0)
        d.qpos.data[10] = Scalar[DTYPE](0)
        # Zero goal velocities
        d.qvel.data[9] = Scalar[DTYPE](0)
        d.qvel.data[10] = Scalar[DTYPE](0)
        # Zero object velocities (object position gets noise from standard reset)
        d.qvel.data[7] = Scalar[DTYPE](0)
        d.qvel.data[8] = Scalar[DTYPE](0)

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
        """Gymnasium Pusher-v5 observation: qpos[:7] + qvel[:7] + 3 body positions.
        """
        # Arm joint positions [7]
        for i in range(NUM_ARM_JOINTS):
            obs.append(d.qpos.data[i])
        # Arm joint velocities [7]
        for i in range(NUM_ARM_JOINTS):
            obs.append(d.qvel.data[i])
        # Fingertip (tips_arm) world position [3]
        obs.append(d.xpos.data[TIPS_ARM_BODY_IDX * 3 + 0])
        obs.append(d.xpos.data[TIPS_ARM_BODY_IDX * 3 + 1])
        obs.append(d.xpos.data[TIPS_ARM_BODY_IDX * 3 + 2])
        # Object world position [3]
        obs.append(d.xpos.data[OBJECT_BODY_IDX * 3 + 0])
        obs.append(d.xpos.data[OBJECT_BODY_IDX * 3 + 1])
        obs.append(d.xpos.data[OBJECT_BODY_IDX * 3 + 2])
        # Goal world position [3]
        obs.append(d.xpos.data[GOAL_BODY_IDX * 3 + 0])
        obs.append(d.xpos.data[GOAL_BODY_IDX * 3 + 1])
        obs.append(d.xpos.data[GOAL_BODY_IDX * 3 + 2])
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
        # Object - Goal distance
        var obj_x = Float64(d.xpos.data[OBJECT_BODY_IDX * 3 + 0])
        var obj_y = Float64(d.xpos.data[OBJECT_BODY_IDX * 3 + 1])
        var obj_z = Float64(d.xpos.data[OBJECT_BODY_IDX * 3 + 2])
        var goal_x = Float64(d.xpos.data[GOAL_BODY_IDX * 3 + 0])
        var goal_y = Float64(d.xpos.data[GOAL_BODY_IDX * 3 + 1])
        var goal_z = Float64(d.xpos.data[GOAL_BODY_IDX * 3 + 2])
        var d2_x = obj_x - goal_x
        var d2_y = obj_y - goal_y
        var d2_z = obj_z - goal_z
        var dist_obj_goal = sqrt(d2_x * d2_x + d2_y * d2_y + d2_z * d2_z)

        # Fingertip - Object distance
        var tip_x = Float64(d.xpos.data[TIPS_ARM_BODY_IDX * 3 + 0])
        var tip_y = Float64(d.xpos.data[TIPS_ARM_BODY_IDX * 3 + 1])
        var tip_z = Float64(d.xpos.data[TIPS_ARM_BODY_IDX * 3 + 2])
        var d1_x = obj_x - tip_x
        var d1_y = obj_y - tip_y
        var d1_z = obj_z - tip_z
        var dist_tip_obj = sqrt(d1_x * d1_x + d1_y * d1_y + d1_z * d1_z)

        var reward_dist = -dist_obj_goal * Self.REWARD_DIST_WEIGHT
        var reward_near = -dist_tip_obj * Self.REWARD_NEAR_WEIGHT

        # Control cost
        var ctrl_cost = Float64(0)
        for i in range(len(actions)):
            ctrl_cost += actions[i] * actions[i]
        var reward_ctrl = -ctrl_cost * Self.REWARD_CTRL_WEIGHT

        var reward = Scalar[DTYPE](reward_dist + reward_ctrl + reward_near)

        # Pusher never terminates early
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(PusherModel.TIMESTEP)

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
        # Object - Goal distance
        var obj_x = rebind[Scalar[DTYPE]](
            xpos[env, OBJECT_BODY_IDX * 3 + 0]
        )
        var obj_y = rebind[Scalar[DTYPE]](
            xpos[env, OBJECT_BODY_IDX * 3 + 1]
        )
        var obj_z = rebind[Scalar[DTYPE]](
            xpos[env, OBJECT_BODY_IDX * 3 + 2]
        )
        var goal_x = rebind[Scalar[DTYPE]](
            xpos[env, GOAL_BODY_IDX * 3 + 0]
        )
        var goal_y = rebind[Scalar[DTYPE]](
            xpos[env, GOAL_BODY_IDX * 3 + 1]
        )
        var goal_z = rebind[Scalar[DTYPE]](
            xpos[env, GOAL_BODY_IDX * 3 + 2]
        )
        var d2x = obj_x - goal_x
        var d2y = obj_y - goal_y
        var d2z = obj_z - goal_z
        var dist_obj_goal = sqrt(d2x * d2x + d2y * d2y + d2z * d2z)

        # Fingertip - Object distance
        var tip_x = rebind[Scalar[DTYPE]](
            xpos[env, TIPS_ARM_BODY_IDX * 3 + 0]
        )
        var tip_y = rebind[Scalar[DTYPE]](
            xpos[env, TIPS_ARM_BODY_IDX * 3 + 1]
        )
        var tip_z = rebind[Scalar[DTYPE]](
            xpos[env, TIPS_ARM_BODY_IDX * 3 + 2]
        )
        var d1x = obj_x - tip_x
        var d1y = obj_y - tip_y
        var d1z = obj_z - tip_z
        var dist_tip_obj = sqrt(d1x * d1x + d1y * d1y + d1z * d1z)

        var reward_dist = -dist_obj_goal * Scalar[DTYPE](
            Self.REWARD_DIST_WEIGHT
        )
        var reward_near = -dist_tip_obj * Scalar[DTYPE](Self.REWARD_NEAR_WEIGHT)

        # Control cost
        var ctrl_cost = Scalar[DTYPE](0)
        comptime for i in range(ACTION_DIM):
            var a = rebind[Scalar[DTYPE]](actions[env, i])
            ctrl_cost += a * a
        var reward_ctrl = -ctrl_cost * Scalar[DTYPE](Self.REWARD_CTRL_WEIGHT)

        var reward = reward_dist + reward_ctrl + reward_near

        # Pusher never terminates
        return (reward, False)

    # === GPU inline: Init qpos — fix goal position ===
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
        # Fix goal joints to 0 (goal stays at XML body position)
        qpos[env, 9] = Scalar[DTYPE](0)
        qpos[env, 10] = Scalar[DTYPE](0)

    # === GPU inline: Custom obs extraction (23D) ===
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
        # Arm joint positions [7]
        comptime for i in range(NUM_ARM_JOINTS):
            obs[env, i] = rebind[Scalar[DTYPE]](qpos[env, i])

        # Arm joint velocities [7]
        comptime for i in range(NUM_ARM_JOINTS):
            obs[env, 7 + i] = rebind[Scalar[DTYPE]](qvel[env, i])

        # Tips arm world position [3]
        obs[env, 14] = rebind[Scalar[DTYPE]](
            xpos[env, TIPS_ARM_BODY_IDX * 3 + 0]
        )
        obs[env, 15] = rebind[Scalar[DTYPE]](
            xpos[env, TIPS_ARM_BODY_IDX * 3 + 1]
        )
        obs[env, 16] = rebind[Scalar[DTYPE]](
            xpos[env, TIPS_ARM_BODY_IDX * 3 + 2]
        )

        # Object world position [3]
        obs[env, 17] = rebind[Scalar[DTYPE]](
            xpos[env, OBJECT_BODY_IDX * 3 + 0]
        )
        obs[env, 18] = rebind[Scalar[DTYPE]](
            xpos[env, OBJECT_BODY_IDX * 3 + 1]
        )
        obs[env, 19] = rebind[Scalar[DTYPE]](
            xpos[env, OBJECT_BODY_IDX * 3 + 2]
        )

        # Goal world position [3]
        obs[env, 20] = rebind[Scalar[DTYPE]](
            xpos[env, GOAL_BODY_IDX * 3 + 0]
        )
        obs[env, 21] = rebind[Scalar[DTYPE]](
            xpos[env, GOAL_BODY_IDX * 3 + 1]
        )
        obs[env, 22] = rebind[Scalar[DTYPE]](
            xpos[env, GOAL_BODY_IDX * 3 + 2]
        )

        return True
