"""Sawyer Reach-v3 environment configuration for Phyics3dEnv.

Implements mocap position control (XYZ delta + gripper) instead of torque motors.
Actions: [delta_x, delta_y, delta_z, gripper_effort] (4D, all in [-1, 1])
Observations: hand_xyz(3) + gripper_dist(1) + obj_xyz(3) + goal_xyz(3) = 10D

Reference: Metaworld-master/metaworld/envs/sawyer_reach_v3.py
"""

from std.math import sqrt
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike

from .sawyer_reach_xml import SawyerReachModel

from ..phyics3d_env_config import Phyics3dEnvConfig
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)
from noeira.core.tiles import Tile

# Body indices in the parsed model (from test output)
comptime MOCAP_BODY_IDX: Int = 32
comptime HAND_BODY_IDX: Int = 24  # "hand" body
comptime OBJ_BODY_IDX: Int = 33  # "obj" body

# Action scaling (MetaWorld default: 1cm per unit)
comptime ACTION_SCALE: Float64 = 0.01

# Mocap workspace bounds (MetaWorld SawyerXYZEnv)
comptime MOCAP_LOW_X: Float64 = -0.2
comptime MOCAP_LOW_Y: Float64 = 0.5
comptime MOCAP_LOW_Z: Float64 = 0.06
comptime MOCAP_HIGH_X: Float64 = 0.2
comptime MOCAP_HIGH_Y: Float64 = 0.7
comptime MOCAP_HIGH_Z: Float64 = 0.6

# Goal position (fixed for now, will be randomized later)
comptime GOAL_X: Float64 = -0.1
comptime GOAL_Y: Float64 = 0.8
comptime GOAL_Z: Float64 = 0.2

# Reach success threshold
comptime TARGET_RADIUS: Float64 = 0.05

# Observation: hand_xyz(3) + gripper_dist(1) + obj_xyz(3) + goal_xyz(3) = 10
comptime SAWYER_REACH_OBS_DIM: Int = 10
comptime SAWYER_REACH_ACTION_DIM: Int = 4


def _clamp(val: Float64, lo: Float64, hi: Float64) -> Float64:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


struct SawyerReachConfig(Phyics3dEnvConfig):
    # === Physics ===
    # Sawyer is the ONLY ported model with collidable mesh geoms: `l6` (the
    # forearm, contype=4 conaffinity=2) and `eGripperBase` (the gripper shell,
    # contype=1 conaffinity=1). Their exact convex hulls are 345 and 883
    # vertices — 1228 total, 3684 scalars. 2048 leaves ~65% headroom, and
    # `fields_build` raises with the exact requirement if a mesh is added.
    #
    # ⚠ NOT 5597. That figure is the total for ALL TWELVE mesh geoms and comes
    # from the capacity ERROR printed BEFORE non-collidable meshes were
    # skipped; ten of those twelve are visual and are no longer loaded. Sizing
    # off a pre-filter measurement over-allocates by ~4.5x. The raw STL counts
    # (l6 3021, eGripperBase 6693) are a different number again — the hull is a
    # small subset of the mesh, so neither raw verts nor the old total is the
    # quantity to size against.
    #
    # ⚠ WITHOUT THIS THE GRIPPER DOES NOT COLLIDE. `Phyics3dEnv` used to
    # hardcode 0, so every mesh pair was skipped and the arm passed through
    # objects — which is precisely what a manipulation task cannot tolerate.
    # 5597 * 3 * 8 B = 134 KiB, one copy in `Model` (not batched), so the cost
    # is compile time on the mesh branch, not memory.
    comptime NMESH_VERTS: Int = 2048

    comptime FRAME_SKIP: Int = 5  # MetaWorld frame_skip
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 500  # MetaWorld max_path_length

    # Dimensions
    comptime OBS_DIM: Int = SAWYER_REACH_OBS_DIM
    comptime ACTION_DIM: Int = SAWYER_REACH_ACTION_DIM

    comptime INTEGRATOR_WS_EXTRA: Int = 0  # Euler doesn't need extra
    comptime INTEGRATOR: StaticString = "euler"  # matches physics_substep (Euler+Newton) workspace

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
        """Extract MetaWorld-style observation: hand + gripper + obj + goal."""
        # Hand position (3)
        obs.append(d.xpos.data[HAND_BODY_IDX * 3 + 0])
        obs.append(d.xpos.data[HAND_BODY_IDX * 3 + 1])
        obs.append(d.xpos.data[HAND_BODY_IDX * 3 + 2])
        # Gripper distance (1) — placeholder (needs finger pad site positions)
        obs.append(Scalar[DTYPE](0.0))
        # Object position (3)
        obs.append(d.xpos.data[OBJ_BODY_IDX * 3 + 0])
        obs.append(d.xpos.data[OBJ_BODY_IDX * 3 + 1])
        obs.append(d.xpos.data[OBJ_BODY_IDX * 3 + 2])
        # Goal position (3)
        obs.append(Scalar[DTYPE](GOAL_X))
        obs.append(Scalar[DTYPE](GOAL_Y))
        obs.append(Scalar[DTYPE](GOAL_Z))
        return True

    # === CPU: Custom reset — set mocap position + warmup arm ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        # Set initial mocap position (MetaWorld hand_init_pos = [0, 0.6, 0.2])
        d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 0] = Scalar[DTYPE](0.0)
        d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 1] = Scalar[DTYPE](0.6)
        d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 2] = Scalar[DTYPE](0.2)
        # Fixed orientation (MetaWorld: quat=[1,0,1,0] wxyz → [0,1,0,1] xyzw)
        d.mocap_quat.data[MOCAP_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[MOCAP_BODY_IDX * 4 + 1] = Scalar[DTYPE](1)
        d.mocap_quat.data[MOCAP_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[MOCAP_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)

        # Set initial arm qpos from MuJoCo reference (after _reset_hand warmup).
        # These values place the hand at approximately (0, 0.6, 0.2).
        # Obtained by running MetaWorld SawyerReachEnvV3.reset() in MuJoCo.
        d.qpos.data[0] = Scalar[DTYPE](1.889288)  # j0
        d.qpos.data[1] = Scalar[DTYPE](-0.575769)  # j1
        d.qpos.data[2] = Scalar[DTYPE](-0.976659)  # j2
        d.qpos.data[3] = Scalar[DTYPE](1.641991)  # j3
        d.qpos.data[4] = Scalar[DTYPE](0.942860)  # j4
        d.qpos.data[5] = Scalar[DTYPE](1.043696)  # j5
        d.qpos.data[6] = Scalar[DTYPE](2.292833)  # j6
        d.qpos.data[7] = Scalar[DTYPE](0.0)  # r_close
        d.qpos.data[8] = Scalar[DTYPE](0.0)  # l_close

        # Object free joint (qpos 9-15): on table at z=0.02
        # (MuJoCo reference position from sawyer_reach_task_xml)
        d.qpos.data[9] = Scalar[DTYPE](0.0)  # obj x
        d.qpos.data[10] = Scalar[DTYPE](0.6)  # obj y
        d.qpos.data[11] = Scalar[DTYPE](0.02)  # obj z (on table)
        d.qpos.data[12] = Scalar[DTYPE](1.0)  # obj quat w
        d.qpos.data[13] = Scalar[DTYPE](0.0)  # obj quat x
        d.qpos.data[14] = Scalar[DTYPE](0.0)  # obj quat y
        d.qpos.data[15] = Scalar[DTYPE](0.0)  # obj quat z
        # (No FK here — the facade runs the fields FK right after this hook.)

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        pass

    # === CPU: Custom action application (mocap position control) ===
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
        """Apply 4D action as mocap position delta + gripper control."""
        # Extract action components
        var dx = Float64(0)
        var dy = Float64(0)
        var dz = Float64(0)
        var gripper = Float64(0)
        if len(actions) >= 1:
            dx = actions[0] * ACTION_SCALE
        if len(actions) >= 2:
            dy = actions[1] * ACTION_SCALE
        if len(actions) >= 3:
            dz = actions[2] * ACTION_SCALE
        if len(actions) >= 4:
            gripper = actions[3]

        # Update mocap position (current + delta, clamped to workspace)
        var cur_x = Float64(d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 0])
        var cur_y = Float64(d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 1])
        var cur_z = Float64(d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 2])

        var new_x = _clamp(cur_x + dx, MOCAP_LOW_X, MOCAP_HIGH_X)
        var new_y = _clamp(cur_y + dy, MOCAP_LOW_Y, MOCAP_HIGH_Y)
        var new_z = _clamp(cur_z + dz, MOCAP_LOW_Z, MOCAP_HIGH_Z)

        d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 0] = Scalar[DTYPE](new_x)
        d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 1] = Scalar[DTYPE](new_y)
        d.mocap_pos.data[MOCAP_BODY_IDX * 3 + 2] = Scalar[DTYPE](new_z)

        # Gripper: apply as qfrc to the gripper slide joints
        # r_close is DOF 7, l_close is DOF 8 (NOT NV-2/NV-1 which would be
        # the object free joint when an object is present in the model)
        d.qfrc.data[7] = Scalar[DTYPE](gripper * 400.0)  # r_close
        d.qfrc.data[8] = Scalar[DTYPE](-gripper * 400.0)  # l_close (mirrored)

        return True  # Handled — skip MODEL_DEF.apply_actions

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
        """MetaWorld Reach-v2 reward: 10 * tolerance(reach_dist)."""
        # Hand position (TCP approximation via hand body xpos)
        var hand_x = Float64(d.xpos.data[HAND_BODY_IDX * 3 + 0])
        var hand_y = Float64(d.xpos.data[HAND_BODY_IDX * 3 + 1])
        var hand_z = Float64(d.xpos.data[HAND_BODY_IDX * 3 + 2])

        # Distance to goal
        var gdx = hand_x - GOAL_X
        var gdy = hand_y - GOAL_Y
        var gdz = hand_z - GOAL_Z
        var tcp_to_target = sqrt(gdx * gdx + gdy * gdy + gdz * gdz)

        # MetaWorld "long_tail" sigmoid tolerance
        # tolerance(x, bounds=(0, 0.05), margin=margin, sigmoid="long_tail")
        # long_tail: 1 / ((x * scale)^2 + 1)  where scale = sqrt(1/value_at_margin - 1)
        # value_at_margin = 0.1, so scale = sqrt(1/0.1 - 1) = sqrt(9) = 3
        var in_place_margin = Float64(0.3)  # typical ||hand_init - goal||
        var in_place: Float64 = 1.0
        if tcp_to_target > TARGET_RADIUS:
            var d = (tcp_to_target - TARGET_RADIUS) / in_place_margin
            var scale: Float64 = 3.0  # sqrt(1/0.1 - 1)
            in_place = 1.0 / ((d * scale) * (d * scale) + 1.0)

        var reward = Scalar[DTYPE](10.0 * in_place)

        # No early termination in MetaWorld Reach
        return (reward, False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(SawyerReachModel.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.0  # MetaWorld uses specific reset, not noise

    # === GPU stubs (CPU-only for now) ===
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
        pass

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
        return (Scalar[DTYPE](0), False)

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
        return False
