"""`dm_control` `swimmer` task config — port of `suite/swimmer.py`.

ONE task shape (`Swimmer`) over two model sizes, so one config serves both:

    swimmer6  = DMSwimmerConfig  over DMSwimmer6Model    (obs 25)
    swimmer15 = DMSwimmerConfig  over DMSwimmer15Model   (obs 61)

    obs = joints          (n-1)  = qpos[3:]
        + to_target       (2)    = nose->target, in the HEAD frame, x/y
        + body_velocities (3n)   = per link [vx, vy, wz], in the LINK frame

    reward = tolerance(||to_target||, bounds=(0, .1), margin=.5,
                       sigmoid='long_tail')
    episode = 1000 control steps (30 s / .03 s), no early termination

Nothing here is parameterised by the link count: the head is body 1, the
segments run 2..NBODY-2, the mocap target is always last, and `joints()` is
everything past the three root DOFs. `DMSwimmerConfig` is therefore generic in
`NQ`/`NV`/`NBODY` and both registered tasks share it.

SENSORS. The reference routes every observation through `<sensor>`; each one is
a direct read here (gap G1 — there is still no sensor framework):

    framepos  of a geom  ->  body xpos + R_body * geom local pos
    velocimeter / gyro   ->  `sensors/frame_vel.site_frame_velocity`

`body_velocities()` slices `sensordata[12:].reshape(-1, 6)[:, [0, 1, 5]]`. The
12 skipped floats are `nose_pos`, `target_pos`, `head_xaxis`, `head_yaxis`; the
rest is one `(velocimeter, gyro)` pair per link, head first, so the rows are
exactly [head, segment_0, ...] and the columns are linear x, linear y, angular
z — all in the link's own frame. That ordering is asserted against the
reference's `sensor_adr` table in the parity test, because getting it wrong
would silently transpose the second half of the observation.

⚠ THE FLUID PATH IS LOAD-BEARING HERE, AND ONLY HERE. `<option density="3000">`
with `<flag contact="disable"/>` means `dynamics/fluid_forces.mojo` (MuJoCo's
`mj_inertiaBoxFluidModel`) is the ONLY thing that turns joint torque into
locomotion: no contacts, and gravity does nothing to a body sliding in the
x-y plane on three planar root DOFs. Every other model in this repo runs with
density = viscosity = 0, so before swimmer that code path was never gated.

⚠ RESET DOES NOT REJECTION-SAMPLE. `randomize_limited_and_rotational_joints`
is reproduced exactly, but the reference's `Swimmer` task does not wrap it in a
contact-free retry loop (finger and ball_in_cup do), so there is nothing to
skip — this one is faithful.
"""

from std.random import random_float64
from std.math import pi, sqrt
from std.collections import Array

from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.sensors.frame_vel import (
    site_frame_velocity,
    site_frame_velocity_gpu,
)
from noeira.physics3d.kinematics.quat_math import quat_rotate, quat_rotate_inverse
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_GEOM_SIZE,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
)

from .swimmer_xml import (
    DMSwimmer6Model,
    DMSwimmer15Model,
    HEAD_BODY_IDX,
    FIRST_SEGMENT_BODY_IDX,
    NOSE_GEOM_IDX,
    N_ROOT_DOF,
    TARGET_SIZE,
    TARGET_Z,
)

from ..dtype_math import sqrt_dt
from ..gpu_reset import (
    reset_seed,
    randomize_limited_and_rotational_joints_gpu,
)
from ..rewards import (
    tolerance,
    SIGMOID_LONG_TAIL,
    DEFAULT_VALUE_AT_MARGIN,
)
from ...phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# `close_target = self.random.rand() < .2`, then `.3` or `2`.
comptime CLOSE_TARGET_PROB: Float64 = 0.2
comptime CLOSE_TARGET_BOX: Float64 = 0.3
comptime FAR_TARGET_BOX: Float64 = 2.0


def _nose_to_target[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
) raises -> Tuple[Float64, Float64]:
    """`Physics.nose_to_target` — target minus nose, rotated into the head
    frame, x and y only.

    The reference is

        (geom_xpos['target'] - geom_xpos['nose']).dot(xmat['head'])[:2]

    and `v.dot(M)` for a 1-D `v` is `M^T v`, i.e. the vector expressed in the
    head's local frame — which is `quat_rotate_inverse` by the head's `xquat`.

    Both geoms sit at their body's origin except the nose, whose local offset
    is read from the geom record rather than hardcoded so a model edit shows
    up as a parity failure instead of a silent bias.
    """
    var go = NOSE_GEOM_IDX * MODEL_GEOM_SIZE
    var lx = m_geoms[go + GEOM_IDX_POS_X]
    var ly = m_geoms[go + GEOM_IDX_POS_Y]
    var lz = m_geoms[go + GEOM_IDX_POS_Z]

    var hqx = d.xquat.data[HEAD_BODY_IDX * 4 + 0]
    var hqy = d.xquat.data[HEAD_BODY_IDX * 4 + 1]
    var hqz = d.xquat.data[HEAD_BODY_IDX * 4 + 2]
    var hqw = d.xquat.data[HEAD_BODY_IDX * 4 + 3]

    var off = quat_rotate[DTYPE](hqx, hqy, hqz, hqw, lx, ly, lz)
    var nose_x = d.xpos.data[HEAD_BODY_IDX * 3 + 0] + off[0]
    var nose_y = d.xpos.data[HEAD_BODY_IDX * 3 + 1] + off[1]
    var nose_z = d.xpos.data[HEAD_BODY_IDX * 3 + 2] + off[2]

    # The target geom sits at its mocap body's origin, so `geom_xpos` is the
    # body's world position.
    comptime TGT = D.NBODY - 1
    var dx = d.xpos.data[TGT * 3 + 0] - nose_x
    var dy = d.xpos.data[TGT * 3 + 1] - nose_y
    var dz = d.xpos.data[TGT * 3 + 2] - nose_z

    var loc = quat_rotate_inverse[DTYPE](hqx, hqy, hqz, hqw, dx, dy, dz)
    return (Float64(loc[0]), Float64(loc[1]))


# XORed into the reset key for the TARGET draw, so it is a different stream
# from the joint randomizer's. Continuing that generator would make the target
# distribution depend on how many joints preceded it — and swimmer6 and
# swimmer15 differ in exactly that.
comptime _TARGET_STREAM: UInt64 = 0x9E3779B97F4A7C15


@always_inline
def _nose_to_target_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NBODY: Int, NGEOM_F: Int
](
    xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
    xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
    geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
    env: Int,
) -> Array[Scalar[DTYPE], 2]:
    """`_nose_to_target` against the batched field tensors.

    ⚠ THE ARITHMETIC IS THE CPU FUNCTION'S, transcribed — same rotation, same
    order. The two are diffed element-wise by the GPU-vs-CPU gate.

    The nose's local offset is read from the geom record rather than
    hardcoded, exactly as the CPU form does, so a model edit shows up as a
    parity failure instead of a silent bias.
    """
    var lx = rebind[Scalar[DTYPE]](geoms[NOSE_GEOM_IDX, GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](geoms[NOSE_GEOM_IDX, GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](geoms[NOSE_GEOM_IDX, GEOM_IDX_POS_Z])

    var hqx = rebind[Scalar[DTYPE]](xquat[env, HEAD_BODY_IDX * 4 + 0])
    var hqy = rebind[Scalar[DTYPE]](xquat[env, HEAD_BODY_IDX * 4 + 1])
    var hqz = rebind[Scalar[DTYPE]](xquat[env, HEAD_BODY_IDX * 4 + 2])
    var hqw = rebind[Scalar[DTYPE]](xquat[env, HEAD_BODY_IDX * 4 + 3])

    var off = quat_rotate[DTYPE](hqx, hqy, hqz, hqw, lx, ly, lz)
    var nose_x = rebind[Scalar[DTYPE]](xpos[env, HEAD_BODY_IDX * 3 + 0]) + off[0]
    var nose_y = rebind[Scalar[DTYPE]](xpos[env, HEAD_BODY_IDX * 3 + 1]) + off[1]
    var nose_z = rebind[Scalar[DTYPE]](xpos[env, HEAD_BODY_IDX * 3 + 2]) + off[2]

    # The target geom sits at its mocap body's origin, so `geom_xpos` is the
    # body's world position.
    comptime TGT = NBODY - 1
    var dx = rebind[Scalar[DTYPE]](xpos[env, TGT * 3 + 0]) - nose_x
    var dy = rebind[Scalar[DTYPE]](xpos[env, TGT * 3 + 1]) - nose_y
    var dz = rebind[Scalar[DTYPE]](xpos[env, TGT * 3 + 2]) - nose_z

    var loc = quat_rotate_inverse[DTYPE](hqx, hqy, hqz, hqw, dx, dy, dz)
    var out = Array[Scalar[DTYPE], 2](fill=Scalar[DTYPE](0))
    out[0] = loc[0]
    out[1] = loc[1]
    return out^


struct DMSwimmerConfig(Phyics3dEnvConfig):
    """`Swimmer`: reach the target with the nose, in a viscous fluid."""

    # === Physics ===
    # `_CONTROL_TIMESTEP = .03` over a `.002` physics step => 15 substeps,
    # and `_DEFAULT_TIME_LIMIT = 30` s => 1000 control steps.
    comptime FRAME_SKIP: Int = 15
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # `body_velocities` reads xvel/xangvel, which only the mid-step velocity
    # pass writes — without this the observation lags the state by a step.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime INTEGRATOR: StaticString = "euler"

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
        """`Swimmer.get_observation`: joints, to_target, body_velocities."""
        try:
            # `physics.joints()` = `qpos[3:]` — the internal hinges only.
            for q in range(N_ROOT_DOF, D.NQ):
                obs.append(d.qpos.data[q])

            var tt = _nose_to_target(d, m_geoms)
            obs.append(Scalar[DTYPE](tt[0]))
            obs.append(Scalar[DTYPE](tt[1]))

            # `body_velocities()`: one row per link, head first, each the
            # [linear x, linear y, angular z] of that link's own site. Site i
            # is mounted on body i + HEAD_BODY_IDX (head -> site 0, and
            # segment_k -> site k+1), which the parity test pins.
            comptime N_LINKS = D.NBODY - 2
            for k in range(N_LINKS):
                var body = HEAD_BODY_IDX + k
                var v = site_frame_velocity[DTYPE](
                    d.xvel.data,
                    d.xangvel.data,
                    d.xipos.data,
                    d.xquat.data,
                    d.site_xpos.data,
                    m_sites,
                    body,
                    k,
                )
                obs.append(Scalar[DTYPE](v[0]))
                obs.append(Scalar[DTYPE](v[1]))
                obs.append(Scalar[DTYPE](v[5]))
        except:
            return False
        return True

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Swimmer.initialize_episode` — joints, then the target box.

        `randomize_limited_and_rotational_joints`: limited hinges and slides
        get a uniform draw inside their range, unlimited HINGES get [-pi, pi),
        and unlimited SLIDES are left alone. That last case is why the swimmer
        always starts at the world origin — `rootx`/`rooty` are unlimited
        slides, so only `rootz` (the heading) is randomised.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jbase = j * MODEL_JOINT_SIZE
            var jtype = Int(m_joints[jbase + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var adr = Int(m_joints[jbase + JOINT_IDX_QPOS_ADR])
            var lo = Float64(m_joints[jbase + JOINT_IDX_RANGE_MIN])
            var hi = Float64(m_joints[jbase + JOINT_IDX_RANGE_MAX])
            if lo > -1e9 and hi < 1e9:
                d.qpos.data[adr] = Scalar[DTYPE](
                    lo + random_float64() * (hi - lo)
                )
            elif jtype == JNT_HINGE:
                d.qpos.data[adr] = Scalar[DTYPE](
                    -pi + random_float64() * 2.0 * pi
                )

        # `close_target = random.rand() < .2` -> a .3 or 2.0 half-width box.
        var box = FAR_TARGET_BOX
        if random_float64() < CLOSE_TARGET_PROB:
            box = CLOSE_TARGET_BOX
        var tx = -box + random_float64() * 2.0 * box
        var ty = -box + random_float64() * 2.0 * box

        comptime TGT = D.NBODY - 1
        d.mocap_pos.data[TGT * 3 + 0] = Scalar[DTYPE](tx)
        d.mocap_pos.data[TGT * 3 + 1] = Scalar[DTYPE](ty)
        d.mocap_pos.data[TGT * 3 + 2] = Scalar[DTYPE](TARGET_Z)
        d.mocap_quat.data[TGT * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[TGT * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[TGT * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[TGT * 4 + 3] = Scalar[DTYPE](1)

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
        """`tolerance(nose_to_target_dist, (0, size), 5*size, 'long_tail')`."""
        try:
            var tt = _nose_to_target(d, m_geoms)
            var dist = sqrt(tt[0] * tt[0] + tt[1] * tt[1])
            var r = tolerance[SIGMOID_LONG_TAIL](
                dist, 0.0, TARGET_SIZE, 5.0 * TARGET_SIZE
            )
            return (Scalar[DTYPE](r), False)
        except:
            return (Scalar[DTYPE](0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        # Both generated models come from the same `<option timestep="0.002">`
        # line in `_swimmer_body_xml`, which is why one config can serve both.
        comptime assert (
            DMSwimmer6Model.TIMESTEP == DMSwimmer15Model.TIMESTEP
        ), "the two swimmer models no longer share a timestep"
        return Float64(DMSwimmer6Model.TIMESTEP)

    # ── GPU hooks ────────────────────────────────────────────────────────
    comptime HAS_GPU_HOOKS: Bool = True
    # The target rides a mocap body (gap G4); without this the batched env
    # would never push `mocap_pos` into the body pose and every episode would
    # aim at the XML target — blocker H.
    comptime USES_MOCAP: Bool = True

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
        """`Swimmer.initialize_episode` — joints, then the target box.

        The joint draw is the shared randomizer with its DEFAULTS: unlimited
        hinges get [-pi, pi) and unlimited slides are skipped, which is why the
        swimmer always starts at the world origin (`rootx`/`rooty` are
        unlimited slides; only `rootz`, the heading, moves).
        """
        randomize_limited_and_rotational_joints_gpu[
            DTYPE, BATCH_SIZE, NQ, NJOINT
        ](qpos, joints, env, seed)

        # `close_target = random.rand() < .2` -> a .3 or 2.0 half-width box.
        # ⚠ A SEPARATE Philox stream from the joint draw above, which consumed
        # an unknown number of blocks (it depends on the joint count, so
        # swimmer6 and swimmer15 differ). Continuing that generator would make
        # the target distribution depend on the link count.
        var rng = PhiloxRandom(
            seed=reset_seed(env, seed) ^ _TARGET_STREAM, offset=0
        )
        var b = rng.step_uniform()
        var box = Scalar[DTYPE](FAR_TARGET_BOX)
        if Scalar[DTYPE](b[0]) < Scalar[DTYPE](CLOSE_TARGET_PROB):
            box = Scalar[DTYPE](CLOSE_TARGET_BOX)
        var tx = -box + Scalar[DTYPE](b[1]) * Scalar[DTYPE](2.0) * box
        var ty = -box + Scalar[DTYPE](b[2]) * Scalar[DTYPE](2.0) * box

        comptime TGT = NBODY - 1
        mocap_pos[env, TGT * 3 + 0] = tx
        mocap_pos[env, TGT * 3 + 1] = ty
        mocap_pos[env, TGT * 3 + 2] = Scalar[DTYPE](TARGET_Z)
        mocap_quat[env, TGT * 4 + 0] = Scalar[DTYPE](0)
        mocap_quat[env, TGT * 4 + 1] = Scalar[DTYPE](0)
        mocap_quat[env, TGT * 4 + 2] = Scalar[DTYPE](0)
        mocap_quat[env, TGT * 4 + 3] = Scalar[DTYPE](1)

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
        """`Swimmer.get_observation`: joints, to_target, body_velocities."""
        comptime N_LINKS = NBODY - 2
        comptime assert (
            (NQ - N_ROOT_DOF) + 2 + 3 * N_LINKS == OBS_DIM
        ), (
            "swimmer.custom_extract_obs_gpu: joints(NQ-3) + to_target(2) +"
            " body_velocities(3*(NBODY-2)) must equal OBS_DIM exactly. This"
            " writes by running index, so a short block leaves the tail"
            " holding the previous step's values."
        )

        var k = 0
        # `physics.joints()` = `qpos[3:]` — the internal hinges only.
        for q in range(N_ROOT_DOF, NQ):
            obs[env, k] = qpos[env, q]
            k += 1

        var tt = _nose_to_target_gpu[
            DTYPE, BATCH_SIZE, NBODY, NGEOM_F
        ](xpos, xquat, geoms, env)
        obs[env, k] = tt[0]
        k += 1
        obs[env, k] = tt[1]
        k += 1

        # One row per link, head first: [linear x, linear y, angular z] of
        # that link's own site. Site i is on body i + HEAD_BODY_IDX.
        for i in range(N_LINKS):
            var v = site_frame_velocity_gpu[
                DTYPE](
                xvel, xangvel, xipos, xquat, site_xpos, sites,
                env, HEAD_BODY_IDX + i, i,
            )
            obs[env, k] = v[0]
            obs[env, k + 1] = v[1]
            obs[env, k + 2] = v[5]
            k += 3
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
        """`tolerance(nose_to_target_dist, (0, size), 5*size, 'long_tail')`."""
        var tt = _nose_to_target_gpu[
            DTYPE, BATCH_SIZE, NBODY, NGEOM_F
        ](xpos, xquat, geoms, env)
        var dist = sqrt_dt[DTYPE](tt[0] * tt[0] + tt[1] * tt[1])
        var r = tolerance[SIGMOID_LONG_TAIL, DEFAULT_VALUE_AT_MARGIN, DTYPE](
            dist,
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](TARGET_SIZE),
            Scalar[DTYPE](5.0 * TARGET_SIZE),
        )
        return (r, False)
