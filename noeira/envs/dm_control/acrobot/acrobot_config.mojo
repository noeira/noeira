"""`dm_control` `acrobot` task config — port of `suite/acrobot.py` (`Balance`).

One parameterized config covers both registered tasks:

    swingup        = DMAcrobotConfig[SPARSE=False]
    swingup_sparse = DMAcrobotConfig[SPARSE=True]

    observation = [xz(upper), xz(lower), zz(upper), zz(lower), qvel(2)]   (6)
    reward      = tolerance(||target - tip||, bounds=(0, .2),
                            margin = 0 if sparse else 1)
    reset       = qpos['shoulder'], qpos['elbow'] ~ U[-pi, pi)
    episode     = 1000 control steps (10 s / 0.01 s), no early termination

`swingup_sparse` is a hard indicator: the tip has to be within 0.2 m of the
target before ANY reward appears, so an all-zero learning curve early in
training is expected rather than a broken env.
"""

from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import pi, sqrt
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
    MODEL_JOINT_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
)

from .acrobot_xml import (
    DMAcrobotModel,
    UPPER_ARM_BODY_IDX,
    LOWER_ARM_BODY_IDX,
    TARGET_SITE_IDX,
    TIP_SITE_IDX,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN
from ..gpu_reset import reset_seed
from noeira.core.tiles import Tile


# `physics.named.model.site_size['target', 0]` — the target sphere's radius.
# Site sizes are not carried in our model records (they are render-only for
# every other domain), so the value is lifted from acrobot.xml here. The
# parity test asserts it still matches `model.site_size` in MuJoCo.
comptime TARGET_RADIUS: Float64 = 0.2


struct DMAcrobotConfig[SPARSE: Bool](Phyics3dEnvConfig):
    # === Physics ===
    # acrobot.py sets no _CONTROL_TIMESTEP, so control_timestep == the model's
    # 0.01 s timestep => 1 substep per env step.
    comptime FRAME_SKIP: Int = 1
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    # _DEFAULT_TIME_LIMIT = 10 s / 0.01 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # dm_control syncs mjData to the integrated qpos before the task reads
    # obs/reward; without this the xmat and site terms lag one step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # `<option ... integrator="RK4">`.
    comptime INTEGRATOR: StaticString = "rk4"

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
        """`Balance.get_observation`: orientations then velocity.

        `orientations()` is `concatenate((horizontal(), vertical()))`, and each
        of those is a two-body slice — so the layout is BODY-MINOR:
        (xz upper, xz lower, zz upper, zz lower), NOT (xz, zz) per body.
        """
        obs.append(Scalar[DTYPE](xmat_elem(d, UPPER_ARM_BODY_IDX, XMAT_XZ)))
        obs.append(Scalar[DTYPE](xmat_elem(d, LOWER_ARM_BODY_IDX, XMAT_XZ)))
        obs.append(Scalar[DTYPE](xmat_elem(d, UPPER_ARM_BODY_IDX, XMAT_ZZ)))
        obs.append(Scalar[DTYPE](xmat_elem(d, LOWER_ARM_BODY_IDX, XMAT_ZZ)))
        obs.append(d.qvel.data[0])
        obs.append(d.qvel.data[1])
        return True

    # === CPU: Reset — both joints at a uniformly random angle ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        # `physics.named.data.qpos[['shoulder', 'elbow']] =
        #      self.random.uniform(-pi, pi, 2)`; qvel stays at the zeros
        # `reset_data` wrote.
        d.qpos.data[0] = Scalar[DTYPE]((random_float64() * 2.0 - 1.0) * pi)
        d.qpos.data[1] = Scalar[DTYPE]((random_float64() * 2.0 - 1.0) * pi)

    # === CPU: Reward — tip-to-target distance ===
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
        # `Physics.to_target`: norm(site_xpos['target'] - site_xpos['tip']).
        var dx = Float64(
            d.site_xpos.data[TARGET_SITE_IDX * 3 + 0]
            - d.site_xpos.data[TIP_SITE_IDX * 3 + 0]
        )
        var dy = Float64(
            d.site_xpos.data[TARGET_SITE_IDX * 3 + 1]
            - d.site_xpos.data[TIP_SITE_IDX * 3 + 1]
        )
        var dz = Float64(
            d.site_xpos.data[TARGET_SITE_IDX * 3 + 2]
            - d.site_xpos.data[TIP_SITE_IDX * 3 + 2]
        )
        var to_target = sqrt(dx * dx + dy * dy + dz * dz)

        # `margin=0 if sparse else 1` — margin 0 makes tolerance a hard
        # indicator, so the sparse task pays only inside the target sphere.
        comptime margin = 0.0 if Self.SPARSE else 1.0
        var r = tolerance(to_target, 0.0, TARGET_RADIUS, margin)
        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](r), False)


    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # First tranche-2 domain: the first whose reward reads `site_xpos`, which
    # became a hook operand for exactly this. ⚠ Must stay numerically identical
    # to the CPU hooks above (gated vs MuJoCo by
    # `test_acrobot_vs_dm_control.mojo`); `test_tranche2_gpu_vs_cpu.mojo` diffs
    # the two paths step for step, for BOTH the sparse and dense margins.
    # =====================================================================

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
        """`Balance.get_observation` — mirrors `custom_extract_obs_cpu`.

        ⚠ BODY-MINOR: `orientations()` is `concatenate((horizontal(),
        vertical()))` and each half is a two-body slice, so the order is
        (xz upper, xz lower, zz upper, zz lower) — NOT (xz, zz) per body.
        """
        obs[env, 0] = xmat_elem_gpu[DTYPE](
            xquat, env, UPPER_ARM_BODY_IDX, XMAT_XZ
        )
        obs[env, 1] = xmat_elem_gpu[DTYPE](
            xquat, env, LOWER_ARM_BODY_IDX, XMAT_XZ
        )
        obs[env, 2] = xmat_elem_gpu[DTYPE](
            xquat, env, UPPER_ARM_BODY_IDX, XMAT_ZZ
        )
        obs[env, 3] = xmat_elem_gpu[DTYPE](
            xquat, env, LOWER_ARM_BODY_IDX, XMAT_ZZ
        )
        obs[env, 4] = qvel[env, 0]
        obs[env, 5] = qvel[env, 1]
        return True

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
        """`Balance.get_reward` — mirrors `compute_reward_and_done_cpu`."""
        var dx = (
            rebind[Scalar[DTYPE]](site_xpos[env, TARGET_SITE_IDX * 3 + 0])
            - rebind[Scalar[DTYPE]](site_xpos[env, TIP_SITE_IDX * 3 + 0])
        )
        var dy = (
            rebind[Scalar[DTYPE]](site_xpos[env, TARGET_SITE_IDX * 3 + 1])
            - rebind[Scalar[DTYPE]](site_xpos[env, TIP_SITE_IDX * 3 + 1])
        )
        var dz = (
            rebind[Scalar[DTYPE]](site_xpos[env, TARGET_SITE_IDX * 3 + 2])
            - rebind[Scalar[DTYPE]](site_xpos[env, TIP_SITE_IDX * 3 + 2])
        )
        var to_target = sqrt(dx * dx + dy * dy + dz * dz)
        # margin 0 => hard indicator (the sparse task); 1 => shaped.
        comptime margin = 0.0 if Self.SPARSE else 1.0
        var r = tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE](
            to_target,
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](TARGET_RADIUS),
            Scalar[DTYPE](margin),
        )
        return (r, False)

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
        """`qpos[['shoulder','elbow']] = uniform(-pi, pi, 2)`.

        Written directly rather than through the shared randomizer: the
        reference sets these two joints explicitly, and acrobot's hinges are
        UNLIMITED, so the shared helper would draw them from the same
        distribution but by a different route. qvel stays at the zeros
        `reset_env_gpu` wrote.
        """
        var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)
        var u = rng.step_uniform()
        qpos[env, 0] = (
            Scalar[DTYPE](u[0]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
        ) * Scalar[DTYPE](pi)
        qpos[env, 1] = (
            Scalar[DTYPE](u[1]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
        ) * Scalar[DTYPE](pi)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMAcrobotModel.TIMESTEP)
