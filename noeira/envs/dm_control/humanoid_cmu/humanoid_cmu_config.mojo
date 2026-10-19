"""`dm_control` `humanoid_CMU` task config — port of `suite/humanoid_CMU.py`.

One parameterized config covers all three registered tasks, which differ only
in the target speed:

    stand = DMHumanoidCMUConfig[MOVE_SPEED=0.0]
    walk  = DMHumanoidCMUConfig[MOVE_SPEED=1.0]
    run   = DMHumanoidCMUConfig[MOVE_SPEED=10.0]

    observation = [joint_angles(56), head_height(1), extremities(12),
                   torso_vertical(3), com_velocity(3), velocity(62)]    (137)
    reward      = small_control * stand_reward * dont_move  (MOVE_SPEED == 0)
                  small_control * stand_reward * move       (otherwise)
    reset       = randomize_limited_and_rotational_joints, rejected until the
                  configuration is collision-free
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

This is `humanoid`'s task class with a different skeleton, and the reward and
observation code below is deliberately the same shape. THREE THINGS DIFFER, and
each of them produces plausible numbers when wrong:

  1. THE REFERENCE BODY IS `thorax` (id 14), not `torso`. humanoid_CMU has no
     body called torso; `humanoid`'s TORSO_BODY_IDX is 1, which here is the
     free-jointed `root`.
  2. `thorax_upright()` READS `xmat['thorax', 'zy']`, NOT `'zz'`.
     `humanoid.py`'s `torso_upright()` reads `zz`. The CMU skeleton is authored
     y-up inside its own frame — the root body carries `euler="90 0 0"` to
     stand it up in the world — so the thorax's local +y is what points at the
     world z. Using `zz` here would score a STANDING humanoid as fallen, and
     the reward would still be a smooth number in [0, 1] the whole time.
     ⚠ THE OBSERVATION DOES NOT FOLLOW THE REWARD HERE.
     `torso_vertical_orientation()` is `xmat['thorax', ['zx','zy','zz']]` — the
     whole z ROW, same as `humanoid`. So the obs reads a row and the reward
     reads one element of it, and they are not the same element. Both are
     transcribed from the reference; the asymmetry is upstream's, not ours.
  3. `nact` is 56, against `humanoid`'s 21, and it divides `small_control`.

NOTHING here reads a `<sensor>`. The one sensor quantity in play is
`thorax_subtreelinvel`, which `sensors.subtree_linvel` computes from
`Data.xvel`. See `humanoid_cmu_xml` for why the block is dropped outright.
"""

from std.random import random_float64
from std.math import pi, inf, sqrt

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.kinematics.xmat import (
    xmat_elem,
    xmat_elem_gpu,
    XMAT_XX,
    XMAT_XY,
    XMAT_XZ,
    XMAT_YX,
    XMAT_YY,
    XMAT_YZ,
    XMAT_ZX,
    XMAT_ZY,
    XMAT_ZZ,
)
from noeira.physics3d.sensors.subtree import (
    subtree_linvel,
    subtree_linvel_gpu,
)
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_FREE

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

from .humanoid_cmu_xml import (
    pmhc,
    THORAX_BODY_IDX,
    HEAD_BODY_IDX,
    extremity_body_indices,
    LEFT_HAND_BODY_IDX,
    LEFT_FOOT_BODY_IDX,
    RIGHT_HAND_BODY_IDX,
    RIGHT_FOOT_BODY_IDX,
    ROOT_QPOS_SIZE,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import (
    tolerance,
    SIGMOID_LINEAR,
    SIGMOID_QUADRATIC,
    SIGMOID_GAUSSIAN,
    DEFAULT_VALUE_AT_MARGIN,
)
from ..gpu_reset import randomize_limited_and_rotational_joints_gpu
from noeira.core.tiles import Tile


# `humanoid_CMU._STAND_HEIGHT`. Same value as `humanoid`'s, and the margin is
# again a QUARTER of it, so the standing term is smooth rather than an
# indicator.
comptime STAND_HEIGHT: Float64 = 1.4

# `_move_speed` for the three tasks: stand 0, walk 1, run 10.
comptime WALK_SPEED: Float64 = 1.0
comptime RUN_SPEED: Float64 = 10.0

# `mjModel.nu`. Divides `small_control`, so a wrong value rescales the reward
# rather than breaking it. Measured, not counted from the XML text — see the
# grep warning in `humanoid_cmu_xml`.
comptime N_ACTUATORS: Int = 56


struct DMHumanoidCMUConfig[MOVE_SPEED: Float64](Phyics3dEnvConfig):
    # === Physics ===
    # humanoid_CMU.xml declares NO <option>, so the timestep is MuJoCo's
    # default .002, and humanoid_CMU.py's _CONTROL_TIMESTEP is .02
    # => 10 physics substeps per control step. (`humanoid` is 5, from an
    # explicit `<option timestep=".005"/>` and a .025 control step; neither
    # number carries over.)
    comptime FRAME_SKIP: Int = 10
    # GPU hooks implemented below — see Phyics3dEnvConfig.HAS_GPU_HOOKS.
    comptime HAS_GPU_HOOKS: Bool = True
    # _DEFAULT_TIME_LIMIT 20 s / .02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # The obs and reward read xpos/xmat/xvel of the INTEGRATED state, so FK and
    # the body velocities `subtree_linvel` consumes must be refreshed after
    # the step; otherwise every derived term lags one control step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # No <option> element at all => MuJoCo's default Euler integrator.
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
        """`HumanoidCMU.get_observation`."""
        # joint_angles(): qpos[7:], dropping the free root joint.
        for i in range(ROOT_QPOS_SIZE, D.NQ):
            obs.append(d.qpos.data[i])

        # head_height(): xpos['head', 'z'].
        obs.append(d.xpos.data[HEAD_BODY_IDX * 3 + 2])

        # extremities(): each limb offset from the THORAX, expressed in the
        # thorax frame. numpy's `torso_to_limb.dot(torso_frame)` is a row
        # vector times the matrix, i.e. R^T v — the transpose matters, and
        # picking the wrong one still produces plausible-looking numbers.
        var tx = Float64(d.xpos.data[THORAX_BODY_IDX * 3 + 0])
        var ty = Float64(d.xpos.data[THORAX_BODY_IDX * 3 + 1])
        var tz = Float64(d.xpos.data[THORAX_BODY_IDX * 3 + 2])
        var r00 = xmat_elem(d, THORAX_BODY_IDX, XMAT_XX)
        var r01 = xmat_elem(d, THORAX_BODY_IDX, XMAT_XY)
        var r02 = xmat_elem(d, THORAX_BODY_IDX, XMAT_XZ)
        var r10 = xmat_elem(d, THORAX_BODY_IDX, XMAT_YX)
        var r11 = xmat_elem(d, THORAX_BODY_IDX, XMAT_YY)
        var r12 = xmat_elem(d, THORAX_BODY_IDX, XMAT_YZ)
        var r20 = xmat_elem(d, THORAX_BODY_IDX, XMAT_ZX)
        var r21 = xmat_elem(d, THORAX_BODY_IDX, XMAT_ZY)
        var r22 = xmat_elem(d, THORAX_BODY_IDX, XMAT_ZZ)
        var limbs = extremity_body_indices()
        for li in range(len(limbs)):
            var b = limbs[li]
            var vx = Float64(d.xpos.data[b * 3 + 0]) - tx
            var vy = Float64(d.xpos.data[b * 3 + 1]) - ty
            var vz = Float64(d.xpos.data[b * 3 + 2]) - tz
            obs.append(Scalar[DTYPE](vx * r00 + vy * r10 + vz * r20))
            obs.append(Scalar[DTYPE](vx * r01 + vy * r11 + vz * r21))
            obs.append(Scalar[DTYPE](vx * r02 + vy * r12 + vz * r22))

        # torso_vertical_orientation(): xmat['thorax', ['zx', 'zy', 'zz']].
        # The whole z row — NOT the single element the reward reads. See the
        # module docstring, point 2.
        obs.append(Scalar[DTYPE](r20))
        obs.append(Scalar[DTYPE](r21))
        obs.append(Scalar[DTYPE](r22))

        # center_of_mass_velocity(): sensordata['thorax_subtreelinvel'].
        var cx = Float64(0)
        var cy = Float64(0)
        var cz = Float64(0)
        subtree_linvel(
            d.xvel.data, m_bodies, D.NBODY, THORAX_BODY_IDX, cx, cy, cz
        )
        obs.append(Scalar[DTYPE](cx))
        obs.append(Scalar[DTYPE](cy))
        obs.append(Scalar[DTYPE](cz))

        # velocity(): the whole qvel.
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
        """`randomizers.randomize_limited_and_rotational_joints`.

        Identical to `humanoid`'s, including the two things about it that look
        like mistakes and are not:

        ONE REFERENCE QUIRK IS REPRODUCED DELIBERATELY. randomizers.py says the
        free joint's quaternion is "sampled uniformly on the unit 3-sphere",
        but the free-joint branch calls `random.rand(4)` (uniform on [0,1)^4),
        not `randn(4)` — the ball branch two lines up does use `randn`. So
        every component is non-negative and the orientations are confined to
        one orthant, nowhere near uniform on SO(3). We match the CODE, not the
        docstring: an agent trained against dm_control sees this distribution.

        DEVIATION, stated plainly: the reference wraps this in
        `while penetrating: ... physics.after_reset(); ncon > 0`, rejecting
        self-colliding draws. We do not. This hook runs BEFORE the facade's
        FK/contact pass, so `d.meta[NUM_CONTACTS]` still describes the previous
        configuration — rejecting on it would filter the wrong thing, which is
        worse than not filtering. Consequence: a fraction of our episodes start
        with limbs interpenetrating and the solver pushes them apart over the
        first few steps. This affects the initial-state DISTRIBUTION, not the
        dynamics, and the parity test drives explicit states rather than
        resets. Closing it needs a post-FK resample hook on
        `Phyics3dEnvConfig` — see docs/DM_CONTROL_PORT.md.

        ⚠ humanoid_CMU is a HARSHER case for that deviation than humanoid: it
        has fingers, thumbs and toes packed close together, so a uniform draw
        over 56 joint ranges self-collides far more often. Whoever adds the
        resample hook should re-measure the rejection rate here.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])

            if jtype == JNT_FREE:
                # qpos[adr+0..2] (position) deliberately untouched.
                var q0 = random_float64()
                var q1 = random_float64()
                var q2 = random_float64()
                var q3 = random_float64()
                var n = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
                if n < 1e-12:
                    q0 = 1.0
                    q1 = 0.0
                    q2 = 0.0
                    q3 = 0.0
                    n = 1.0
                d.qpos.data[adr + 3] = Scalar[DTYPE](q0 / n)
                d.qpos.data[adr + 4] = Scalar[DTYPE](q1 / n)
                d.qpos.data[adr + 5] = Scalar[DTYPE](q2 / n)
                d.qpos.data[adr + 6] = Scalar[DTYPE](q3 / n)
                continue

            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
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
        # standing = tolerance(head_height, (1.4, inf), margin=1.4/4)
        var head_z = Float64(d.xpos.data[HEAD_BODY_IDX * 3 + 2])
        var standing = tolerance(
            head_z, STAND_HEIGHT, inf[DType.float64](), STAND_HEIGHT / 4.0
        )
        # upright = tolerance(thorax_upright(), (0.9, inf), sigmoid='linear',
        #                     margin=1.9, value_at_margin=0)
        #
        # ⚠ `thorax_upright()` is `xmat['thorax', 'zy']` — the ZY element, not
        # ZZ. `humanoid.py`'s `torso_upright()` reads ZZ and this is the one
        # line where the two domains genuinely disagree. See the module
        # docstring, point 2.
        var upright_zy = xmat_elem(d, THORAX_BODY_IDX, XMAT_ZY)
        var upright = tolerance[SIGMOID_LINEAR, 0.0](
            upright_zy, 0.9, inf[DType.float64](), 1.9
        )
        var stand_reward = standing * upright

        # small_control = (4 + mean_i tolerance(ctrl_i, margin=1,
        #                     value_at_margin=0, quadratic)) / 5
        var acc = 0.0
        for a in range(N_ACTUATORS):
            var c = actions[a] if a < len(actions) else 0.0
            if c > 1.0:
                c = 1.0
            elif c < -1.0:
                c = -1.0
            acc += tolerance[SIGMOID_QUADRATIC, 0.0](c, 0.0, 0.0, 1.0)
        var small_control = (4.0 + acc / Float64(N_ACTUATORS)) / 5.0

        var cx = Float64(0)
        var cy = Float64(0)
        var cz = Float64(0)
        subtree_linvel(
            d.xvel.data, m_bodies, D.NBODY, THORAX_BODY_IDX, cx, cy, cz
        )

        comptime if Self.MOVE_SPEED == 0.0:
            # dont_move = tolerance(com_velocity[[0,1]], margin=2).mean()
            # — a MEAN over the two horizontal components, each scored
            # separately, NOT the norm. The move branch below uses the norm.
            var dm0 = tolerance(cx, 0.0, 0.0, 2.0)
            var dm1 = tolerance(cy, 0.0, 0.0, 2.0)
            var dont_move = (dm0 + dm1) / 2.0
            return (
                Scalar[DTYPE](small_control * stand_reward * dont_move),
                False,
            )
        else:
            var speed = sqrt(cx * cx + cy * cy)
            var move = tolerance[SIGMOID_LINEAR, 0.0](
                speed,
                Self.MOVE_SPEED,
                inf[DType.float64](),
                Self.MOVE_SPEED,
            )
            move = (5.0 * move + 1.0) / 6.0
            return (
                Scalar[DTYPE](small_control * stand_reward * move),
                False,
            )


    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # ⚠ Must stay numerically identical to the CPU hooks above, gated against
    # MuJoCo by `test_humanoid_cmu_vs_dm_control.mojo`;
    # `test_humanoid_gpu_vs_cpu.mojo` diffs the two paths step for step.
    #
    # ⚠⚠ THE THORAX/ZY TRAP. This domain differs from `humanoid` in exactly two
    # places and BOTH are easy to "fix" wrongly when copying that config's GPU
    # hooks: the subtree root and extremity frame are the THORAX (not the
    # torso), and `thorax_upright()` reads xmat ZY (not ZZ). Copying humanoid's
    # ZZ here yields a plausible reward that is simply a different task.
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
        """`HumanoidCMU.get_observation` — mirrors the CPU hook."""
        var k = 0
        for i in range(ROOT_QPOS_SIZE, NQ_F):
            obs[env, k] = qpos[env, i]
            k += 1

        obs[env, k] = xpos[env, HEAD_BODY_IDX * 3 + 2]
        k += 1

        # extremities(), in the THORAX frame. R^T v — see the CPU hook.
        var tx = rebind[Scalar[DTYPE]](xpos[env, THORAX_BODY_IDX * 3 + 0])
        var ty = rebind[Scalar[DTYPE]](xpos[env, THORAX_BODY_IDX * 3 + 1])
        var tz = rebind[Scalar[DTYPE]](xpos[env, THORAX_BODY_IDX * 3 + 2])
        var r00 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_XX
        )
        var r01 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_XY
        )
        var r02 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_XZ
        )
        var r10 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_YX
        )
        var r11 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_YY
        )
        var r12 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_YZ
        )
        var r20 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_ZX
        )
        var r21 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_ZY
        )
        var r22 = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_ZZ
        )
        # Order: left_hand, left_foot, right_hand, right_foot. Must stay in
        # step with `extremity_body_indices()`, which a kernel cannot call.
        comptime LIMBS = [
            LEFT_HAND_BODY_IDX,
            LEFT_FOOT_BODY_IDX,
            RIGHT_HAND_BODY_IDX,
            RIGHT_FOOT_BODY_IDX,
        ]
        comptime for li in range(4):
            comptime b = LIMBS[li]
            var vx = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 0]) - tx
            var vy = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1]) - ty
            var vz = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]) - tz
            obs[env, k] = vx * r00 + vy * r10 + vz * r20
            obs[env, k + 1] = vx * r01 + vy * r11 + vz * r21
            obs[env, k + 2] = vx * r02 + vy * r12 + vz * r22
            k += 3

        # torso_vertical_orientation(): the whole z ROW — not the single ZY
        # element the reward reads.
        obs[env, k] = r20
        obs[env, k + 1] = r21
        obs[env, k + 2] = r22
        k += 3

        var cx = Scalar[DTYPE](0)
        var cy = Scalar[DTYPE](0)
        var cz = Scalar[DTYPE](0)
        subtree_linvel_gpu[DTYPE](
            Dims[nq=NQ_F, nv=NV_F, nbody=NBODY_F, nsite=NSITE_F, ngeom=NGEOM_F](),
            xvel, bodies, env, THORAX_BODY_IDX, cx, cy, cz
        )
        obs[env, k] = cx
        obs[env, k + 1] = cy
        obs[env, k + 2] = cz
        k += 3

        for i in range(NV_F):
            obs[env, k + i] = qvel[env, i]
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
        """`HumanoidCMU.get_reward` — mirrors `compute_reward_and_done_cpu`."""
        comptime ONE = Scalar[DTYPE](1.0)
        comptime ZERO = Scalar[DTYPE](0.0)

        var head_z = rebind[Scalar[DTYPE]](xpos[env, HEAD_BODY_IDX * 3 + 2])
        var standing = tolerance[
            SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
        ](
            head_z,
            Scalar[DTYPE](STAND_HEIGHT),
            inf[DTYPE](),
            Scalar[DTYPE](STAND_HEIGHT / 4.0),
        )
        # ⚠⚠ ZY, not ZZ. `thorax_upright()` differs from humanoid's
        # `torso_upright()` here and nowhere else in the reward.
        var upright_zy = xmat_elem_gpu[DTYPE](
            xquat, env, THORAX_BODY_IDX, XMAT_ZY
        )
        var upright = tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
            upright_zy, Scalar[DTYPE](0.9), inf[DTYPE](), Scalar[DTYPE](1.9)
        )
        var stand_reward = standing * upright

        var acc = ZERO
        for a in range(N_ACTUATORS):
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
            Scalar[DTYPE](4.0) + acc / Scalar[DTYPE](N_ACTUATORS)
        ) / Scalar[DTYPE](5.0)

        var cx = ZERO
        var cy = ZERO
        var cz = ZERO
        subtree_linvel_gpu[DTYPE](
            Dims[nq=NQ_F, nv=NV_F, nbody=NBODY_F, nsite=NSITE_F, ngeom=NGEOM_F](),
            xvel, bodies, env, THORAX_BODY_IDX, cx, cy, cz
        )

        comptime if Self.MOVE_SPEED == 0.0:
            var dm0 = tolerance[
                SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
            ](cx, ZERO, ZERO, Scalar[DTYPE](2.0))
            var dm1 = tolerance[
                SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
            ](cy, ZERO, ZERO, Scalar[DTYPE](2.0))
            var dont_move = (dm0 + dm1) / Scalar[DTYPE](2.0)
            return (small_control * stand_reward * dont_move, False)
        else:
            var speed = sqrt(cx * cx + cy * cy)
            var move = tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
                speed,
                Scalar[DTYPE](Self.MOVE_SPEED),
                inf[DTYPE](),
                Scalar[DTYPE](Self.MOVE_SPEED),
            )
            move = (Scalar[DTYPE](5.0) * move + ONE) / Scalar[DTYPE](6.0)
            return (small_control * stand_reward * move, False)

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
        """As `humanoid`'s — mirrors `custom_reset_cpu`, including the
        no-rejection DEVIATION documented there. ⚠ That deviation bites HARDER
        here: humanoid_CMU has fingers, thumbs and toes packed close together,
        so a uniform draw self-collides more often."""
        randomize_limited_and_rotational_joints_gpu[
            DTYPE,
            BATCH_SIZE,
            NQ_F,
            NJOINT_F,
            RANDOMIZE_UNLIMITED_HINGES=True,
            RANDOMIZE_FREE_QUAT=True,
        ](qpos, joints, env, seed)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(pmhc.TIMESTEP)
