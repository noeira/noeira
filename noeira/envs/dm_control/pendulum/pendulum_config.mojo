"""`dm_control` `pendulum-swingup` task config.

Port of `dm_control/suite/pendulum.py` (class `SwingUp`).

    observation = [xmat['pole','zz'], xmat['pole','xz'], qvel['hinge']]   (3)
    reward      = tolerance(xmat['pole','zz'], bounds=(cos(8deg), 1))
    reset       = qpos['hinge'] ~ U[-pi, pi), qvel = 0
    episode     = 1000 control steps, no early termination

The reward has NO margin, so it is a hard indicator: 1 while the pole is
within 8 degrees of vertical, 0 otherwise. Max return is therefore 1000, and
a policy that never reaches vertical scores exactly 0 — do not mistake an
all-zero learning curve for a broken env early in training.
"""

from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import pi
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.kinematics.xmat import (
    xmat_elem,
    xmat_elem_gpu,
    XMAT_ZZ,
    XMAT_XZ,
)
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
)

from .pendulum_xml import DMPendulumModel, POLE_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN
from noeira.core.tiles import Tile


# Reference: `_ANGLE_BOUND = 8` degrees, `_COSINE_BOUND = cos(deg2rad(8))`.
comptime ANGLE_BOUND_DEG: Float64 = 8.0
comptime COSINE_BOUND: Float64 = 0.99026806874157036  # cos(8 * pi / 180)


struct DMPendulumConfig(Phyics3dEnvConfig):
    # === Physics ===
    # pendulum.xml has timestep=0.02 and pendulum.py sets no _CONTROL_TIMESTEP,
    # so control_timestep == physics timestep => 1 substep per env step.
    comptime FRAME_SKIP: Int = 1
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    # Every suite task is time_limit / control_timestep = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # dm_control syncs mjData to the integrated qpos before the task
    # reads obs/reward; without this the xmat terms lag one step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # pendulum.xml's <option> carries no `integrator`, so MuJoCo's default
    # (Euler) applies. cartpole/acrobot DO request RK4 — check per domain.
    comptime INTEGRATOR: StaticString = "euler"

    # === CPU: Observation ===
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
        """OrderedDict order from `SwingUp.get_observation`: orientation
        (xmat columns zz, xz) then velocity (the hinge's qvel)."""
        obs.append(
            Scalar[DTYPE](xmat_elem(d, POLE_BODY_IDX, XMAT_ZZ))
        )
        obs.append(
            Scalar[DTYPE](xmat_elem(d, POLE_BODY_IDX, XMAT_XZ))
        )
        obs.append(d.qvel.data[0])
        return True

    # === CPU: Reset — pole at a uniformly random angle ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        # `physics.named.data.qpos['hinge'] = self.random.uniform(-pi, pi)`.
        # qvel is left at the zeros `reset_data` wrote (the reference relies
        # on `physics.reset()` having zeroed it).
        var angle = (random_float64() * 2.0 - 1.0) * pi
        d.qpos.data[0] = Scalar[DTYPE](angle)

    # === CPU: Reward — sparse "within 8 degrees of vertical" ===
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
        var pole_vertical = xmat_elem(d, POLE_BODY_IDX, XMAT_ZZ)
        var r = tolerance(pole_vertical, COSINE_BOUND, 1.0, 0.0)
        # dm_control tasks never terminate early — only the time limit ends
        # an episode.
        return (Scalar[DTYPE](r), False)

    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # First dm_control task off the CPU-only list (G10; see
    # docs/DM_CONTROL_GPU_TRAINING_G10.md). Three things make it possible:
    #   * `xquat` is now a hook operand, so `xmat_elem_gpu` gives the two
    #     rotation-matrix entries the task reads without an NBODY*9 tensor;
    #   * `tolerance` is DTYPE-generic, so the reward can be computed in
    #     float32 — Metal has no `double`;
    #   * `init_qpos_gpu` now receives a seed, so the pole can start at a
    #     random angle as the reference does.
    #
    # ⚠ These MUST stay numerically identical to the CPU hooks above. The
    # CPU hooks are what `tests/dm_control/test_pendulum_vs_dm_control.mojo`
    # gates against MuJoCo; nothing gates these except
    # `tests/dm_control/test_pendulum_gpu_vs_cpu.mojo`, which diffs the two
    # paths step for step.
    # =====================================================================

    # === GPU inline: Observation ===
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
        """`SwingUp.get_observation` — mirrors `custom_extract_obs_cpu`."""
        obs[env, 0] = xmat_elem_gpu[DTYPE](
            xquat, env, POLE_BODY_IDX, XMAT_ZZ
        )
        obs[env, 1] = xmat_elem_gpu[DTYPE](
            xquat, env, POLE_BODY_IDX, XMAT_XZ
        )
        obs[env, 2] = qvel[env, 0]
        return True

    # === GPU inline: Reward ===
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
        """`SwingUp.get_reward` — mirrors `compute_reward_and_done_cpu`.

        `margin` is 0, so this is a hard indicator and the float32 arithmetic
        can only differ from the CPU path within ~1e-7 of the bound itself.
        """
        var pole_vertical = xmat_elem_gpu[DTYPE](
            xquat, env, POLE_BODY_IDX, XMAT_ZZ
        )
        var r = tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE](
            pole_vertical,
            Scalar[DTYPE](COSINE_BOUND),
            Scalar[DTYPE](1.0),
            Scalar[DTYPE](0.0),
        )
        # dm_control tasks never terminate early.
        return (r, False)

    # === GPU inline: Reset — pole at a uniformly random angle ===
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
        """`physics.named.data.qpos['hinge'] = random.uniform(-pi, pi)`.

        Written directly rather than through
        `dm_control/gpu_reset.randomize_limited_and_rotational_joints_gpu`
        because the reference does the same: `SwingUp.initialize_episode` sets
        the one hinge explicitly instead of calling the randomizer. qvel stays
        at the zeros `reset_env_gpu` wrote.
        """
        var rng = PhiloxRandom(
            seed=UInt64(seed * 1103515245 + env * 98765431), offset=0
        )
        var u = Scalar[DTYPE](rng.step_uniform()[0])
        qpos[env, 0] = (
            (u * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)) * Scalar[DTYPE](pi)
        )

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMPendulumModel.TIMESTEP)
