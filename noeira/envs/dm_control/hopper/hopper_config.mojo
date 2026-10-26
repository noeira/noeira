"""`dm_control` `hopper` task configs — port of `suite/hopper.py` (`Hopper`).

One parameterized config covers both registered tasks:

    stand = DMHopperConfig[HOPPING=False]
    hop   = DMHopperConfig[HOPPING=True]

    observation = [qpos[1:] (6), qvel (7), log1p(touch) (2)]           (15)
    reward      = standing * small_control          (stand)
                  standing * hopping                (hop)
                  standing = tolerance(torso_z - foot_z, (0.6, 2))  [margin 0]
    reset       = randomize_limited_and_rotational_joints
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

`standing` has NO margin, so it is a hard indicator: the reward is exactly 0
until the torso is 0.6 m above the foot. Both tasks therefore start at a flat
zero return, as with pendulum and cartpole-sparse.

The touch terms are the first use of `physics3d.sensors.touch`; see that module
for the zone semantics and for why contacts must be read post-solve.
"""

from std.random import random_float64
from std.math import pi, log, inf
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from noeira.physics3d.sensors.subtree import (
    subtree_linvel,
    subtree_linvel_gpu,
)
from noeira.physics3d.sensors.touch import (
    touch_sphere_site,
    touch_sphere_site_gpu,
)

from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .hopper_xml import (
    DMHopperModel,
    TORSO_BODY_IDX,
    FOOT_BODY_IDX,
    TOUCH_TOE_SITE_IDX,
    TOUCH_HEEL_SITE_IDX,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import (
    tolerance,
    SIGMOID_QUADRATIC,
    SIGMOID_LINEAR,
    SIGMOID_GAUSSIAN,
    DEFAULT_VALUE_AT_MARGIN,
)
from ..gpu_reset import randomize_limited_and_rotational_joints_gpu
from ..dtype_math import log1p_dt
from noeira.core.tiles import Tile


# `hopper.py`: minimal torso-over-foot height scoring 1, and the hopping speed
# above which the hop reward saturates.
comptime STAND_HEIGHT: Float64 = 0.6
comptime HOP_SPEED: Float64 = 2.0

# `CONTACT_IDX_FORCE_N` is already a FORCE, in the same units as
# `mj_contactForce`'s normal component — measured, not assumed: on a settling
# drop our raw values track MuJoCo's within the solver's own disagreement
# (274 vs 336, 174 vs 208), whereas a 1/timestep scale would be 200x out.
# No conversion, then.
comptime TOUCH_FORCE_SCALE: Float64 = 1.0


struct DMHopperConfig[HOPPING: Bool](Phyics3dEnvConfig):
    # === Physics ===
    # `_CONTROL_TIMESTEP = .02` over the model's 0.005 s step => 4 substeps.
    comptime FRAME_SKIP: Int = 4
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    # _DEFAULT_TIME_LIMIT = 20 s / 0.02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # `<option timestep="0.005"/>` names no integrator => MuJoCo's Euler.
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
        """`Hopper.get_observation`: position, velocity, touch.

        Position drops qpos[0] (rootx) for translational invariance, exactly as
        the reference comments say.
        """
        for i in range(1, D.NQ):
            obs.append(d.qpos.data[i])
        for i in range(D.NV):
            obs.append(d.qvel.data[i])

        # `np.log1p(sensordata[['touch_toe', 'touch_heel']])`.
        try:
            var toe = touch_sphere_site(
                d, m_sites, TOUCH_TOE_SITE_IDX, TOUCH_FORCE_SCALE
            )
            var heel = touch_sphere_site(
                d, m_sites, TOUCH_HEEL_SITE_IDX, TOUCH_FORCE_SCALE
            )
            obs.append(log1p_dt[DTYPE](Scalar[DTYPE](toe)))
            obs.append(log1p_dt[DTYPE](Scalar[DTYPE](heel)))
        except:
            # touch_sphere_site only raises on a non-sphere site, which is a
            # model-authoring error rather than a runtime condition. Keep the
            # observation the right LENGTH so a mis-edited model fails as an
            # obviously-dead sensor rather than a shape mismatch.
            obs.append(Scalar[DTYPE](0))
            obs.append(Scalar[DTYPE](0))
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
        the walker and point_mass configs'."""
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
        # `Physics.height`: xipos['torso','z'] - xipos['foot','z'].
        var height = Float64(d.xipos.data[TORSO_BODY_IDX * 3 + 2]) - Float64(
            d.xipos.data[FOOT_BODY_IDX * 3 + 2]
        )
        # margin defaults to 0 => hard indicator.
        var standing = tolerance(height, STAND_HEIGHT, 2.0, 0.0)

        comptime if Self.HOPPING:
            # `physics.speed()` = sensordata['torso_subtreelinvel'][0].
            var vx = Float64(0)
            var vy = Float64(0)
            var vz = Float64(0)
            subtree_linvel(
                d.xvel.data, m_bodies, D.NBODY, TORSO_BODY_IDX, vx, vy, vz
            )
            var hopping = tolerance[SIGMOID_LINEAR, 0.5](
                Float64(vx), HOP_SPEED, inf[DType.float64](), HOP_SPEED / 2.0
            )
            return (Scalar[DTYPE](standing * hopping), False)
        else:
            # small_control = (mean_i tolerance(ctrl_i, margin=1,
            #                    value_at_margin=0, quadratic) + 4) / 5
            var acc = 0.0
            comptime nact = DMHopperModel.nact
            for a in range(nact):
                var c = actions[a] if a < len(actions) else 0.0
                if c > 1.0:
                    c = 1.0
                elif c < -1.0:
                    c = -1.0
                acc += tolerance[SIGMOID_QUADRATIC, 0.0](c, 0.0, 0.0, 1.0)
            var small_control = (acc / Float64(nact) + 4.0) / 5.0
            return (Scalar[DTYPE](standing * small_control), False)


    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # FIRST CONSUMER OF `touch_sphere_site_gpu`. That helper is shared with
    # finger, manipulator, stacker and dog, so a defect here is a defect in
    # four more domains later — which is why the gate drives the foot into the
    # ground rather than testing a hovering hopper whose touch terms are 0.
    #
    # ⚠ Must stay numerically identical to the CPU hooks above (gated vs
    # MuJoCo by `test_hopper_vs_dm_control.mojo`).
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
        """`Hopper.get_observation` — position, velocity, touch.

        qpos[0] (rootx) is dropped for translational invariance, as the
        reference comments say.
        """
        var k = 0
        for i in range(1, NQ_F):
            obs[env, k] = qpos[env, i]
            k += 1
        for i in range(NV_F):
            obs[env, k] = qvel[env, i]
            k += 1
        # `np.log1p(sensordata[['touch_toe','touch_heel']])`.
        #
        # ⚠ No `try` here, unlike the CPU hook: a kernel cannot raise. A zone
        # type the GPU sensor does not implement comes back as
        # TOUCH_UNSUPPORTED_ZONE (negative), and log1p of a negative is NaN —
        # which propagates into the observation and is impossible to miss.
        # That is deliberate: the CPU hook's `except` writes 0.0, which reads
        # as "nothing is touching" and would be silent.
        var toe = touch_sphere_site_gpu[
            DTYPE](
            Dims[nq=NQ_F, nv=NV_F, nbody=NBODY_F, nsite=NSITE_F, ngeom=NGEOM_F](),
            contacts, site_xpos, sites, meta, xquat, env, TOUCH_TOE_SITE_IDX,
            Scalar[DTYPE](TOUCH_FORCE_SCALE),
        )
        var heel = touch_sphere_site_gpu[
            DTYPE](
            Dims[nq=NQ_F, nv=NV_F, nbody=NBODY_F, nsite=NSITE_F, ngeom=NGEOM_F](),
            contacts, site_xpos, sites, meta, xquat, env, TOUCH_HEEL_SITE_IDX,
            Scalar[DTYPE](TOUCH_FORCE_SCALE),
        )
        obs[env, k] = log1p_dt[DTYPE](toe)
        obs[env, k + 1] = log1p_dt[DTYPE](heel)
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
        """`Hopper.get_reward` — mirrors `compute_reward_and_done_cpu`."""
        comptime ONE = Scalar[DTYPE](1.0)
        comptime ZERO = Scalar[DTYPE](0.0)
        # `Physics.height`: xipos['torso','z'] - xipos['foot','z'].
        var height = (
            rebind[Scalar[DTYPE]](xipos[env, TORSO_BODY_IDX * 3 + 2])
            - rebind[Scalar[DTYPE]](xipos[env, FOOT_BODY_IDX * 3 + 2])
        )
        # margin 0 => hard indicator.
        var standing = tolerance[
            SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
        ](height, Scalar[DTYPE](STAND_HEIGHT), Scalar[DTYPE](2.0), ZERO)

        comptime if Self.HOPPING:
            var vx = ZERO
            var vy = ZERO
            var vz = ZERO
            subtree_linvel_gpu[DTYPE](
                Dims[nq=NQ_F, nv=NV_F, nbody=NBODY_F, nsite=NSITE_F, ngeom=NGEOM_F](),
                xvel, bodies, env, TORSO_BODY_IDX, vx, vy, vz
            )
            var hopping = tolerance[SIGMOID_LINEAR, 0.5, DTYPE](
                vx,
                Scalar[DTYPE](HOP_SPEED),
                inf[DTYPE](),
                Scalar[DTYPE](HOP_SPEED / 2.0),
            )
            return (standing * hopping, False)
        else:
            var acc = ZERO
            comptime nact = DMHopperModel.nact
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
                acc += tolerance[SIGMOID_QUADRATIC, 0.0, DTYPE](
                    c, ZERO, ZERO, ONE
                )
            var small_control = (
                acc / Scalar[DTYPE](nact) + Scalar[DTYPE](4.0)
            ) / Scalar[DTYPE](5.0)
            return (standing * small_control, False)

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
        """`randomizers.randomize_limited_and_rotational_joints` — as walker's."""
        randomize_limited_and_rotational_joints_gpu[
            DTYPE, BATCH_SIZE, NQ_F, NJOINT_F, RANDOMIZE_UNLIMITED_HINGES=True
        ](qpos, joints, env, seed)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMHopperModel.TIMESTEP)
