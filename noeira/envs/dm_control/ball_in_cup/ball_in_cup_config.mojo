"""`dm_control` `ball_in_cup-catch` task config — port of `suite/ball_in_cup.py`.

    observation = [qpos(4), qvel(4)]                                     (8)
    reward      = in_target()  -- SPARSE, exactly 0 or 1
    reset       = ball_x ~ U(-.2, .2), ball_z ~ U(.2, .5), rejected while
                  the ball starts in contact
    episode     = 1000 control steps (20 s / .02 s), no early termination

`in_target` is a BOX test in the x-z plane, not a radial one:

    |site_xpos[target] - xpos[ball]|  <  site_size[target] - geom_size[ball]

componentwise over (x, z), i.e. |dx| < .025 and |dz| < .025. Note the two
operands come from different tables — the TARGET is a SITE, the BALL is a
BODY (`named.data.xpos['ball']`, not the geom) — and this port keeps that
split rather than using the ball geom, which sits at the body origin here but
need not in general.

CONTROL TIMESTEP. `_CONTROL_TIMESTEP = .02` against a `.002` physics step
gives FRAME_SKIP = 10, unlike point_mass (which passes no control timestep and
so runs 1:1).
"""

from std.random import random_float64
from std.math import sqrt, abs
from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_JOINT_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
)

from .ball_in_cup_xml import (
    DMBallInCupModel,
    BALL_BODY_IDX,
    TARGET_SITE_IDX,
    BALL_GEOM_IDX,
    CUP_GEOM_FIRST,
    CUP_GEOM_LAST,
    TARGET_HALF_X,
    TARGET_HALF_Z,
    BALL_RADIUS,
)

from ..dtype_math import sqrt_dt
from ..gpu_reset import reset_seed
from ...phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# Ball spawn box, `BallInCup.initialize_episode`.
comptime SPAWN_X_LO: Float64 = -0.2
comptime SPAWN_X_HI: Float64 = 0.2
comptime SPAWN_Z_LO: Float64 = 0.2
comptime SPAWN_Z_HI: Float64 = 0.5

# qpos layout: [cup_x, cup_z, ball_x, ball_z]
comptime QADR_BALL_X: Int = 2
comptime QADR_BALL_Z: Int = 3


struct DMBallInCupConfig(Phyics3dEnvConfig):
    # === Physics ===
    # _CONTROL_TIMESTEP .02 / timestep .002.
    comptime FRAME_SKIP: Int = 10
    # _DEFAULT_TIME_LIMIT 20 s / .02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # No `<option integrator>`, so MuJoCo's Euler default applies.
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
        """`BallInCup.get_observation`: position then velocity, both whole."""
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
        """`BallInCup.initialize_episode` — resample the ball until it starts
        out of contact.

        The reference calls `physics.after_reset()` and tests `data.ncon > 0`.
        Running collision detection from inside a reset hook is not available
        here, so the rejection test is done in closed form: with the cup at
        its own qpos0 the only geoms the ball can reach are the five cup
        capsules, and a sphere-capsule contact exists exactly when the
        centre-to-segment distance is below the radius sum. The ground is
        unreachable (ball z >= .2 - .025) and the cup cannot touch it either.

        Consequence of the closed form: it is the SAME acceptance region, but
        the accepted sample is not the same draw as dm_control's for a given
        seed. Episode-for-episode reproduction of a reference rollout is not
        a goal of this port; the parity test sets qpos in both engines.
        """
        var tries = 0
        while tries < 100:
            var bx = SPAWN_X_LO + random_float64() * (SPAWN_X_HI - SPAWN_X_LO)
            var bz = SPAWN_Z_LO + random_float64() * (SPAWN_Z_HI - SPAWN_Z_LO)
            tries += 1

            # The cup body is at its qpos0 here (cup_x = cup_z = 0) and is a
            # direct child of worldbody, so its world origin IS the model's
            # `body_pos` — read from the record, not from `d.xpos`.
            #
            # ⚠ THIS USED TO READ `d.xpos`, AND THAT WAS WRONG ON EVERY RESET
            # BUT THE FIRST. `Phyics3dEnv._reset_state` calls `reset_data`
            # (which zeroes qpos) and then this hook, and only runs
            # `_fields_fk()` AFTERWARDS — so `d.xpos` still held wherever the
            # PREVIOUS episode left the cup. The cup is on two damped springs
            # and swings a good fraction of the .2-wide spawn box, so the
            # acceptance region drifted with the last episode's ending pose.
            # dm_control has no such problem: it calls `physics.after_reset()`,
            # which recomputes FK from the reset qpos before testing `ncon`.
            var cb = BALL_BODY_IDX - 1  # the cup, body 1
            var cup_x = Float64(m_bodies[cb * MODEL_BODY_SIZE + BODY_IDX_POS_X])
            var cup_y = Float64(m_bodies[cb * MODEL_BODY_SIZE + BODY_IDX_POS_Y])
            var cup_z = Float64(m_bodies[cb * MODEL_BODY_SIZE + BODY_IDX_POS_Z])

            var hit = False
            for g in range(CUP_GEOM_FIRST, CUP_GEOM_LAST + 1):
                var o = g * MODEL_GEOM_SIZE
                var r = Float64(m_geoms[o + GEOM_IDX_RADIUS])
                var hl = Float64(m_geoms[o + GEOM_IDX_HALF_LENGTH])
                # Capsule axis in the body frame: local z rotated by the geom
                # quaternion, scaled by the half-length.
                var qx = Float64(m_geoms[o + GEOM_IDX_QUAT_X])
                var qy = Float64(m_geoms[o + GEOM_IDX_QUAT_Y])
                var qz = Float64(m_geoms[o + GEOM_IDX_QUAT_Z])
                var qw = Float64(m_geoms[o + GEOM_IDX_QUAT_W])
                var ax = 2.0 * (qx * qz + qw * qy)
                var ay = 2.0 * (qy * qz - qw * qx)
                var az = 1.0 - 2.0 * (qx * qx + qy * qy)

                var cx = cup_x + Float64(m_geoms[o + GEOM_IDX_POS_X])
                var cy = cup_y + Float64(m_geoms[o + GEOM_IDX_POS_Y])
                var cz = cup_z + Float64(m_geoms[o + GEOM_IDX_POS_Z])

                # Closest point on the segment [c - hl*a, c + hl*a] to (bx,0,bz).
                var px = bx - cx
                var py = 0.0 - cy
                var pz = bz - cz
                var t = px * ax + py * ay + pz * az
                if t > hl:
                    t = hl
                elif t < -hl:
                    t = -hl
                var dx = px - t * ax
                var dy = py - t * ay
                var dz = pz - t * az
                var dist = sqrt(dx * dx + dy * dy + dz * dz)
                if dist < r + BALL_RADIUS:
                    hit = True
                    break

            if not hit:
                d.qpos.data[QADR_BALL_X] = Scalar[DTYPE](bx)
                d.qpos.data[QADR_BALL_Z] = Scalar[DTYPE](bz)
                return

        # Fell through 100 rejections (the spawn box is mostly free, so this
        # is unreachable in practice). Park the ball at the box centre, which
        # is well clear of every capsule.
        d.qpos.data[QADR_BALL_X] = Scalar[DTYPE](0.0)
        d.qpos.data[QADR_BALL_Z] = Scalar[DTYPE](0.35)

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
        """`Physics.in_target()` — 1.0 inside the target box, else 0.0."""
        # `named.data.site_xpos['target', ['x', 'z']]`
        var tx = Float64(d.site_xpos.data[TARGET_SITE_IDX * 3 + 0])
        var tz = Float64(d.site_xpos.data[TARGET_SITE_IDX * 3 + 2])
        # `named.data.xpos['ball', ['x', 'z']]` — the BODY, not the geom.
        var bx = Float64(d.xpos.data[BALL_BODY_IDX * 3 + 0])
        var bz = Float64(d.xpos.data[BALL_BODY_IDX * 3 + 2])

        var in_target = (
            abs(tx - bx) < TARGET_HALF_X - BALL_RADIUS
            and abs(tz - bz) < TARGET_HALF_Z - BALL_RADIUS
        )

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](1.0) if in_target else Scalar[DTYPE](0.0), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMBallInCupModel.TIMESTEP)

    # ── GPU hooks ────────────────────────────────────────────────────────
    comptime HAS_GPU_HOOKS: Bool = True

    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NJOINT: Int,
        NV: Int,
        NBODY: Int,
        NGEOM_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
        mocap_pos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        mocap_quat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
        seed: Int,
    ):
        """`BallInCup.initialize_episode` — the rejection sampler, per lane.

        The SAME closed-form acceptance test the CPU twin uses, and for the
        same reason: collision detection is not available from inside a reset
        hook. With the cup at qpos0 the only geoms the ball can reach are the
        five cup capsules, and a sphere-capsule contact exists exactly when
        the centre-to-segment distance is below the radius sum.

        ⚠ THE CUP ORIGIN COMES FROM THE BODY RECORD, NOT FROM `xpos`. This
        hook runs BEFORE the reset FK, so `Data.xpos` holds the PREVIOUS
        episode's pose — and the cup rides two damped springs, so it swings a
        good fraction of the .2-wide spawn box. Reading `xpos` here made the
        acceptance region drift with wherever the last episode ended; that was
        a real defect on the CPU path too, fixed in the same commit. At qpos0
        the cup is a direct child of worldbody, so `body_pos` IS its world
        origin.

        ⚠ THE LOOP IS FIXED-TRIP, not `while`. Every lane of the batch runs
        the same kernel, so an early `break` buys nothing and a data-dependent
        trip count would diverge the warp; instead all lanes run the bound and
        keep the FIRST accepted draw. Same acceptance region, same
        distribution.
        """
        var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)

        var cb = BALL_BODY_IDX - 1  # the cup, body 1
        var cup_x = rebind[Scalar[DTYPE]](bodies[cb, BODY_IDX_POS_X])
        var cup_y = rebind[Scalar[DTYPE]](bodies[cb, BODY_IDX_POS_Y])
        var cup_z = rebind[Scalar[DTYPE]](bodies[cb, BODY_IDX_POS_Z])

        # Fallback if all 100 draws are rejected: the box centre, which is
        # well clear of every capsule. Mirrors the CPU twin.
        var acc_x = Scalar[DTYPE](0.0)
        var acc_z = Scalar[DTYPE](0.35)
        var found = False

        for _try in range(100):
            var b = rng.step_uniform()
            var bx = Scalar[DTYPE](SPAWN_X_LO) + Scalar[DTYPE](
                b[0]
            ) * Scalar[DTYPE](SPAWN_X_HI - SPAWN_X_LO)
            var bz = Scalar[DTYPE](SPAWN_Z_LO) + Scalar[DTYPE](
                b[1]
            ) * Scalar[DTYPE](SPAWN_Z_HI - SPAWN_Z_LO)

            var hit = False
            for g in range(CUP_GEOM_FIRST, CUP_GEOM_LAST + 1):
                var r = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
                var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])
                # Capsule axis in the body frame: local z rotated by the geom
                # quaternion, scaled by the half-length.
                var qx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_X])
                var qy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Y])
                var qz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Z])
                var qw = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_W])
                var ax = Scalar[DTYPE](2.0) * (qx * qz + qw * qy)
                var ay = Scalar[DTYPE](2.0) * (qy * qz - qw * qx)
                var az = Scalar[DTYPE](1.0) - Scalar[DTYPE](2.0) * (
                    qx * qx + qy * qy
                )

                var cx = cup_x + rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_X])
                var cy = cup_y + rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Y])
                var cz = cup_z + rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Z])

                # Closest point on [c - hl*a, c + hl*a] to (bx, 0, bz).
                var px = bx - cx
                var py = Scalar[DTYPE](0.0) - cy
                var pz = bz - cz
                var t = px * ax + py * ay + pz * az
                if t > hl:
                    t = hl
                elif t < -hl:
                    t = -hl
                var ddx = px - t * ax
                var ddy = py - t * ay
                var ddz = pz - t * az
                var dist = sqrt_dt[DTYPE](
                    ddx * ddx + ddy * ddy + ddz * ddz
                )
                if dist < r + Scalar[DTYPE](BALL_RADIUS):
                    hit = True

            if not hit and not found:
                acc_x = bx
                acc_z = bz
                found = True

        qpos[env, QADR_BALL_X] = acc_x
        qpos[env, QADR_BALL_Z] = acc_z

    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
    ) -> Bool:
        """`BallInCup.get_observation`: position then velocity, both whole."""
        comptime assert NQ + NV == OBS_DIM, (
            "ball_in_cup: qpos(NQ) + qvel(NV) must equal OBS_DIM exactly."
        )
        var k = 0
        for i in range(NQ):
            obs[env, k] = qpos[env, i]
            k += 1
        for i in range(NV):
            obs[env, k] = qvel[env, i]
            k += 1
        return True

    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """`Physics.in_target()` — 1.0 inside the target box, else 0.0."""
        # `named.data.site_xpos['target', ['x', 'z']]`
        var tx = rebind[Scalar[DTYPE]](site_xpos[env, TARGET_SITE_IDX * 3 + 0])
        var tz = rebind[Scalar[DTYPE]](site_xpos[env, TARGET_SITE_IDX * 3 + 2])
        # `named.data.xpos['ball', ['x', 'z']]` — the BODY, not the geom.
        var bx = rebind[Scalar[DTYPE]](xpos[env, BALL_BODY_IDX * 3 + 0])
        var bz = rebind[Scalar[DTYPE]](xpos[env, BALL_BODY_IDX * 3 + 2])

        var dx = tx - bx
        if dx < Scalar[DTYPE](0):
            dx = -dx
        var dz = tz - bz
        if dz < Scalar[DTYPE](0):
            dz = -dz

        var in_target = (
            dx < Scalar[DTYPE](TARGET_HALF_X - BALL_RADIUS)
            and dz < Scalar[DTYPE](TARGET_HALF_Z - BALL_RADIUS)
        )
        return (
            Scalar[DTYPE](1.0) if in_target else Scalar[DTYPE](0.0),
            False,
        )
