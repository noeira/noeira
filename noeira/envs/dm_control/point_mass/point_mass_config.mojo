"""`dm_control` `point_mass-easy` task config — port of `suite/point_mass.py`.

    observation = [qpos(2), qvel(2)]                                     (4)
    reward      = near_target * small_control
                  near_target   = tolerance(||target - mass||,
                                            bounds=(0, .015), margin=.015)
                  small_control = (mean_i tolerance(ctrl_i, margin=1,
                                       value_at_margin=0, quadratic) + 4) / 5
    reset       = randomize_limited_and_rotational_joints
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

Only `easy` is ported. `hard` randomizes the tendon mixing matrix per episode
(`model.wrap_prm`), which our engine cannot express — see `point_mass_xml` for
why the tendons are written as joint motors here.

The reward is far sharper than "smooth" suggests: `margin` is the 1.5 cm target
radius, so a mass 10 cm away scores ~1e-245, not "a bit less than 1". An
untrained policy therefore returns a flat zero for a long time — that is the
reference's behaviour, gated to 2.7e-11 against dm_control's own
`rewards.tolerance`, not a broken env.
"""

from std.random import random_float64
from std.math import pi, sqrt
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.kinematics.geom_xpos import (
    geom_xpos,
    geom_xpos_gpu,
)
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE

from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .point_mass_xml import (
    DMPointMassModel,
    POINTMASS_GEOM_IDX,
    TARGET_GEOM_IDX,
    TARGET_SIZE,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import (
    tolerance,
    SIGMOID_QUADRATIC,
    SIGMOID_GAUSSIAN,
    DEFAULT_VALUE_AT_MARGIN,
)
from ..gpu_reset import randomize_limited_and_rotational_joints_gpu
from noeira.core.tiles import Tile


struct DMPointMassConfig(Phyics3dEnvConfig):
    # === Physics ===
    # point_mass.py passes no control_timestep, so one env step is one
    # physics step of 0.02 s.
    comptime FRAME_SKIP: Int = 1
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    # _DEFAULT_TIME_LIMIT = 20 s / 0.02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # `<option timestep="0.02">` carries no `integrator`, so MuJoCo's Euler
    # default applies.
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
        """`PointMass.get_observation`: position then velocity, both whole."""
        for i in range(D.NQ):
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
        """`randomizers.randomize_limited_and_rotational_joints` — identical to
        the walker config's; both of point_mass's joints are limited slides, so
        only the first branch is ever taken here."""
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            var lo = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
            )
            var hi = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
            )
            var limited = lo > -1e9 and hi < 1e9
            if limited:
                d.qpos.data[adr] = Scalar[DTYPE](
                    lo + random_float64() * (hi - lo)
                )
            elif jtype == JNT_HINGE:
                d.qpos.data[adr] = Scalar[DTYPE](
                    -pi + random_float64() * 2.0 * pi
                )

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
        # `Physics.mass_to_target_dist`.
        var tp = geom_xpos(d, m_geoms, TARGET_GEOM_IDX)
        var mp = geom_xpos(d, m_geoms, POINTMASS_GEOM_IDX)
        var dx = tp[0] - mp[0]
        var dy = tp[1] - mp[1]
        var dz = tp[2] - mp[2]
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var near_target = tolerance(dist, 0.0, TARGET_SIZE, TARGET_SIZE)

        # `tolerance(physics.control(), margin=1, value_at_margin=0,
        #            sigmoid='quadratic').mean()`. `physics.control()` is
        # mjData.ctrl, which MuJoCo has already clamped to ctrlrange.
        var acc = 0.0
        comptime nact = DMPointMassModel.nact
        for a in range(nact):
            var c = actions[a] if a < len(actions) else 0.0
            if c > 1.0:
                c = 1.0
            elif c < -1.0:
                c = -1.0
            acc += tolerance[SIGMOID_QUADRATIC, 0.0](c, 0.0, 0.0, 1.0)
        var small_control = (acc / Float64(nact) + 4.0) / 5.0

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](near_target * small_control), False)


    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # FIRST CONSUMER OF `geom_xpos_gpu`, which is DERIVED from xpos+xquat+geom
    # records rather than stored — see that function on why blocker F did not
    # need a new `Data` field after all.
    #
    # The observation is the whole qpos+qvel, which IS the model default, so
    # there is no `custom_extract_obs_gpu` here on purpose.
    #
    # ⚠ EASY ONLY. `point_mass-hard` mutates `Model.tendons` per episode
    # (`custom_reset_model_cpu` writes the actuator->joint mixing coefs), and
    # `fields.Model` is SHARED and UNBATCHED — every lane would get the last
    # lane's draw. That is gap G4, not an oversight.
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
        """`PointMass.get_observation`: the WHOLE qpos, then the whole qvel.

        ⚠ THIS IS NOT THE MODEL DEFAULT, and assuming it was is what the
        tranche-2/3 gate caught at step 0: the default is
        `qpos[obs_qpos_skip:] + qvel` and point_mass's skip is 1, so the whole
        observation came out SHIFTED BY ONE with a zero in the last slot
        (gpu[i] == cpu[i+1]). The reference keeps qpos[0] — the mass's x
        position is the task, not a nuisance coordinate to be made invariant,
        unlike cheetah's rootx.

        Returning False here to "use the default" is therefore wrong for this
        domain even though the shapes match. Shapes matching is not the test.
        """
        for i in range(NQ_F):
            obs[env, i] = qpos[env, i]
        for i in range(NV_F):
            obs[env, NQ_F + i] = qvel[env, i]
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
        """`PointMass.get_reward` — mirrors `compute_reward_and_done_cpu`."""
        comptime ONE = Scalar[DTYPE](1.0)
        comptime ZERO = Scalar[DTYPE](0.0)
        var tp = geom_xpos_gpu[DTYPE](
            xpos, xquat, geoms, env, TARGET_GEOM_IDX
        )
        var mp = geom_xpos_gpu[DTYPE](
            xpos, xquat, geoms, env, POINTMASS_GEOM_IDX
        )
        var dx = tp[0] - mp[0]
        var dy = tp[1] - mp[1]
        var dz = tp[2] - mp[2]
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var near_target = tolerance[
            SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
        ](
            dist, ZERO, Scalar[DTYPE](TARGET_SIZE), Scalar[DTYPE](TARGET_SIZE)
        )

        var acc = ZERO
        comptime nact = DMPointMassModel.nact
        for a in range(nact):
            var c = (
                rebind[Scalar[DTYPE]](actions[env, a])
                if a < ACTION_DIM
                else ZERO
            )
            if c > ONE:
                c = ONE
            elif c < -ONE:
                c = -ONE
            acc += tolerance[SIGMOID_QUADRATIC, 0.0, DTYPE](c, ZERO, ZERO, ONE)
        var small_control = (
            acc / Scalar[DTYPE](nact) + Scalar[DTYPE](4.0)
        ) / Scalar[DTYPE](5.0)
        return (near_target * small_control, False)

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
        """`randomize_limited_and_rotational_joints`. Both of point_mass's
        joints are LIMITED slides, so only the first branch is ever taken —
        the unlimited-hinge flag is irrelevant here and left at the default."""
        randomize_limited_and_rotational_joints_gpu[
            DTYPE, BATCH_SIZE, NQ_F, NJOINT_F, RANDOMIZE_UNLIMITED_HINGES=True
        ](qpos, joints, env, seed)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMPointMassModel.TIMESTEP)
