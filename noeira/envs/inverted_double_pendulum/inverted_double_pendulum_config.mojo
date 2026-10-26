"""InvertedDoublePendulum environment configuration for generic Phyics3dEnv."""

from std.math import sin, cos
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .inverted_double_pendulum_xml import InvertedDoublePendulumModel

from ..phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# Pole segment length (each segment 0.6 m)
comptime _POLE_LEN = 0.6


struct InvertedDoublePendulumConfig(Phyics3dEnvConfig):
    # === Physics ===
    comptime FRAME_SKIP: Int = 5
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = rk4_extra_workspace_size[
        InvertedDoublePendulumModel.NQ, InvertedDoublePendulumModel.NV
    ]()

    # Termination threshold: height of tip must be > 1.0 m
    comptime MIN_TIP_HEIGHT = 1.0

    # === CPU: Custom obs extraction (9D with sin/cos encoding) ===
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
        # OBS_DIM=9: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2),
        #              clip(qvel[0:3], -10, 10), 0.0]
        obs.append(d.qpos.data[0])
        obs.append(Scalar[DTYPE](sin(Float64(d.qpos.data[1]))))
        obs.append(Scalar[DTYPE](sin(Float64(d.qpos.data[2]))))
        obs.append(Scalar[DTYPE](cos(Float64(d.qpos.data[1]))))
        obs.append(Scalar[DTYPE](cos(Float64(d.qpos.data[2]))))
        for i in range(3):
            var v = d.qvel.data[i]
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs.append(v)
        obs.append(Scalar[DTYPE](0.0))  # qfrc_constraint placeholder
        return True

    # === CPU: Pre-step hook ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        # Save cart x position (qpos[0]) — unused for reward but required by trait
        prev_x = d.qpos.data[0]

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
        var q0 = d.qpos.data[0]  # cart x
        var q1 = d.qpos.data[1]  # pole1 angle
        var q2 = d.qpos.data[2]  # pole2 angle

        # Tip position (analytical from joint angles)
        var pole_len = Scalar[DTYPE](_POLE_LEN)
        var x_tip = (
            q0
            + pole_len * Scalar[DTYPE](sin(Float64(q1)))
            + pole_len * Scalar[DTYPE](sin(Float64(q1) + Float64(q2)))
        )
        var z_tip = pole_len * Scalar[DTYPE](
            cos(Float64(q1))
        ) + pole_len * Scalar[DTYPE](cos(Float64(q1) + Float64(q2)))

        var terminated = z_tip <= Scalar[DTYPE](Self.MIN_TIP_HEIGHT)

        var dist_penalty = Scalar[DTYPE](0.01) * x_tip * x_tip + (
            z_tip - Scalar[DTYPE](2.0)
        ) * (z_tip - Scalar[DTYPE](2.0))
        var v1 = d.qvel.data[1]
        var v2 = d.qvel.data[2]
        var vel_penalty = (
            Scalar[DTYPE](1e-3) * v1 * v1 + Scalar[DTYPE](5e-3) * v2 * v2
        )

        var alive_bonus = Scalar[DTYPE](0.0) if terminated else Scalar[DTYPE](
            10.0
        )
        var reward = alive_bonus - dist_penalty - vel_penalty

        return (reward, terminated)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(InvertedDoublePendulumModel.TIMESTEP)

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
        # Save cart x position (qpos[0]) into META_IDX_PREV_X
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
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"

        var q0 = rebind[Scalar[DTYPE]](qpos[env, 0])  # cart x
        var q1 = rebind[Scalar[DTYPE]](qpos[env, 1])  # pole1 angle
        var q2 = rebind[Scalar[DTYPE]](qpos[env, 2])  # pole2 angle

        # Tip position: analytical from joint angles
        var pole_len = Scalar[DTYPE](_POLE_LEN)
        var s1 = Scalar[DTYPE](sin(q1))
        var s12 = Scalar[DTYPE](sin(q1 + q2))
        var c1 = Scalar[DTYPE](cos(q1))
        var c12 = Scalar[DTYPE](cos(q1 + q2))

        var x_tip = q0 + pole_len * s1 + pole_len * s12
        var z_tip = pole_len * c1 + pole_len * c12

        var terminated = z_tip <= Scalar[DTYPE](1.0)

        var dist_penalty = Scalar[DTYPE](0.01) * x_tip * x_tip + (
            z_tip - Scalar[DTYPE](2.0)
        ) * (z_tip - Scalar[DTYPE](2.0))

        var v1 = rebind[Scalar[DTYPE]](qvel[env, 1])
        var v2 = rebind[Scalar[DTYPE]](qvel[env, 2])
        var vel_penalty = (
            Scalar[DTYPE](1e-3) * v1 * v1 + Scalar[DTYPE](5e-3) * v2 * v2
        )

        var alive_bonus = Scalar[DTYPE](0.0) if terminated else Scalar[DTYPE](
            10.0
        )
        var reward = alive_bonus - dist_penalty - vel_penalty

        return (reward, terminated)

    # === GPU inline: Custom obs extraction (9D with sin/cos encoding) ===
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
        # OBS_DIM=9: [cart_x, sin(q1), sin(q2), cos(q1), cos(q2),
        #              clip(qvel[0],-10,10), clip(qvel[1],-10,10), clip(qvel[2],-10,10),
        #              0.0]  # qfrc_constraint[0] not in state buffer → 0
        var q0 = rebind[Scalar[DTYPE]](qpos[env, 0])
        var q1 = rebind[Scalar[DTYPE]](qpos[env, 1])
        var q2 = rebind[Scalar[DTYPE]](qpos[env, 2])

        obs[env, 0] = q0
        obs[env, 1] = Scalar[DTYPE](sin(q1))
        obs[env, 2] = Scalar[DTYPE](sin(q2))
        obs[env, 3] = Scalar[DTYPE](cos(q1))
        obs[env, 4] = Scalar[DTYPE](cos(q2))

        comptime for i in range(3):
            var v = rebind[Scalar[DTYPE]](qvel[env, i])
            if v > Scalar[DTYPE](10.0):
                v = Scalar[DTYPE](10.0)
            elif v < Scalar[DTYPE](-10.0):
                v = Scalar[DTYPE](-10.0)
            obs[env, 5 + i] = v

        # qfrc_constraint[0] is not stored in the state buffer; use 0.0
        obs[env, 8] = Scalar[DTYPE](0.0)

        return True
