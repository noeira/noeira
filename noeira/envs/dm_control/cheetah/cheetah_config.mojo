"""`dm_control` `cheetah` task config — port of `suite/cheetah.py` (`Cheetah`).

The domain registers a single task:

    run = DMCheetahConfig

    observation = [position (qpos[1:], 8), velocity (qvel, 9)]        (17)
    reward      = tolerance(speed, bounds=(10, inf), margin=10,
                            value_at_margin=0, sigmoid='linear')
    episode     = 1000 control steps (10 s / 0.01 s), no early termination

`speed` is the x component of `sensordata['torso_subtreelinvel']`, i.e. the
CoM velocity of the whole body — not qvel[0], which is only the root slider.
"""

from std.random import random_float64
from std.math import inf

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.sensors.subtree import (
    subtree_linvel,
    subtree_linvel_gpu,
)
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE

from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .cheetah_xml import TORSO_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_LINEAR
from ..gpu_reset import randomize_limited_and_rotational_joints_gpu
from noeira.core.tiles import Tile


# `cheetah._RUN_SPEED`.
comptime RUN_SPEED: Float64 = 10.0


struct DMCheetahConfig(Phyics3dEnvConfig):
    # === Physics ===
    # cheetah.xml timestep = 0.01 and cheetah.py passes no control_timestep,
    # so control_timestep == physics timestep => 1 substep per env step.
    comptime FRAME_SKIP: Int = 1
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    # time_limit 10 s / 0.01 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # The reward reads xvel (via subtree_linvel) of the INTEGRATED state.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # cheetah.xml states no integrator => MuJoCo's Euler default.
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
        """`Cheetah.get_observation`: qpos[1:] then qvel.

        qpos[0] is the root x slider — dropped so the policy cannot see its
        absolute horizontal position.
        """
        for i in range(1, D.NQ):
            obs.append(d.qpos.data[i])
        for i in range(D.NV):
            obs.append(d.qvel.data[i])
        return True

    # === CPU: Reset ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Cheetah.initialize_episode`: limited joints drawn uniformly in range.

        The reference then runs `physics.step(nstep=200)` to settle the model
        before zeroing the clock. That settle is NOT done here — this hook only
        writes qpos, and the driver does not offer a post-reset warm-up. It
        matters for the initial state distribution but not for correctness of
        the dynamics; noted in docs/DM_CONTROL_PORT.md as an open item.

        "Limited" is read the way the engine reads it (`constraints/limits.mojo`):
        a range beyond +-1e9 means unlimited.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var lo = Float64(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            var hi = Float64(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
            if lo <= -1e9 or hi >= 1e9:
                continue  # unlimited — the reference leaves these alone
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            d.qpos.data[adr] = Scalar[DTYPE](lo + random_float64() * (hi - lo))

    # === CPU: Reward ===
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
        """`Cheetah.get_reward`."""
        var vx = Float64(0)
        var vy = Float64(0)
        var vz = Float64(0)
        subtree_linvel(
            d.xvel.data, m_bodies, D.NBODY, TORSO_BODY_IDX, vx, vy, vz
        )
        var r = tolerance[SIGMOID_LINEAR, 0.0](
            vx, RUN_SPEED, inf[DType.float64](), RUN_SPEED
        )
        return (Scalar[DTYPE](r), False)


    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # ⚠ Must stay numerically identical to the CPU hooks above, which are what
    # `tests/dm_control/test_cheetah_vs_dm_control.mojo` gates against MuJoCo.
    # `tests/dm_control/test_locomotion_gpu_vs_cpu.mojo` diffs the two paths
    # step for step.
    #
    # The observation is `qpos[1:] + qvel`, which IS the model default
    # (`extract_obs_gpu` with obs_qpos_skip=1), so no `custom_extract_obs_gpu`
    # override is needed — returning False from the default is correct here.
    # ⚠ Do not "add one for symmetry": a second implementation of an identical
    # observation is a second thing to keep in sync.
    # =====================================================================

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
        """`Cheetah.get_reward` — mirrors `compute_reward_and_done_cpu`.

        `speed` is the x component of `torso_subtreelinvel`: the CoM velocity
        of the WHOLE body, not qvel[0] (the root slider alone).
        """
        var vx = Scalar[DTYPE](0)
        var vy = Scalar[DTYPE](0)
        var vz = Scalar[DTYPE](0)
        subtree_linvel_gpu[DTYPE](
            Dims[nq=NQ_F, nv=NV_F, nbody=NBODY_F, nsite=NSITE_F, ngeom=NGEOM_F](),
            xvel, bodies, env, TORSO_BODY_IDX, vx, vy, vz
        )
        # bounds=(RUN_SPEED, inf): `inf[DTYPE]` NOT `inf[float64]` — the
        # float64 spelling would not type-check against Scalar[DTYPE], and
        # silently casting one would collapse the upper bound.
        var r = tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
            vx,
            Scalar[DTYPE](RUN_SPEED),
            inf[DTYPE](),
            Scalar[DTYPE](RUN_SPEED),
        )
        return (r, False)

    # === GPU inline: Reset ===
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
        """`Cheetah.initialize_episode` — mirrors `custom_reset_cpu`.

        ⚠ `RANDOMIZE_UNLIMITED_HINGES=False`: cheetah walks `jnt_range` and
        touches ONLY the limited joints. Leaving it True would also randomize
        the three unlimited root dofs (rootx/rootz slides + rooty hinge) and
        start the cheetah at a random body angle — a different initial state
        distribution, and not the reference's.

        The reference's `physics.step(nstep=200)` settle is NOT done here, as
        on the CPU path. Noted in docs/DM_CONTROL_PORT.md as an open item; it
        affects the initial state distribution, not the dynamics.
        """
        randomize_limited_and_rotational_joints_gpu[
            DTYPE, BATCH_SIZE, NQ_F, NJOINT_F, RANDOMIZE_UNLIMITED_HINGES=False
        ](qpos, joints, env, seed)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.01
