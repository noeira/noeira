"""`dm_control` `fish` task configs — port of `suite/fish.py`.

Two tasks over one model:

    upright = DMFishUprightConfig   (obs 21)
    swim    = DMFishSwimConfig      (obs 24)

    joint_angles = qpos[7:14]                    (the 7 named _JOINTS)
    upright      = xmat['torso', 'zz']
    velocity     = qvel                          (all 13, free root included)
    target       = mouth_to_target               (swim only, 3)

    reward (upright) = tolerance(upright, bounds=(1, 1), margin=1)
    reward (swim)    = (7*in_target + is_upright) / 8
                       in_target  = tolerance(||mouth_to_target||,
                                              (0, radii), margin=2*radii)
                       is_upright = 0.5 * (upright + 1)
    episode          = 1000 control steps (40 s / .04 s), no early termination

`mouth_to_target` is the one observation that needs geom kinematics on BOTH
sides: it is `geom_xpos['target'] - geom_xpos['mouth']` expressed in the MOUTH
GEOM's frame, and the mouth's frame is not its body's — it is a `fromto`
capsule, so the compiler derived a quaternion for it. Hence
`kinematics/geom_xmat.geom_xquat` rather than a body `xquat`.

⚠ THE ACTUATORS ARE POSITION SERVOS. Their force reads `qpos`, so it is
recomputed every physics substep (see `ModelDefFromXML.apply_actions` and the
loop in `Phyics3dEnv.step`). Nothing in this file has to know that, but a
config that reimplements actuation via `custom_apply_actions_cpu` would — that
hook still runs once per control step, by design.

⚠ RESET DRAWS A UNIFORM RANDOM ORIENTATION. `qpos['root'][3:7] = randn(4)`
normalized is a uniform point on the unit 3-sphere; `random_float64` is
uniform, so the four normals come from Box-Muller here. The DISTRIBUTION
matches the reference, the stream does not — as everywhere else in this port,
the parity test seeds the state explicitly instead of comparing resets.
"""

from std.random import random_float64
from std.math import sqrt, log, cos, sin, pi

from std.collections import Array
from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.kinematics.xmat import xmat_elem, xmat_elem_gpu, XMAT_ZZ
from noeira.physics3d.kinematics.geom_xpos import geom_xpos, geom_xpos_gpu
from noeira.physics3d.kinematics.geom_xmat import geom_xquat, geom_xquat_gpu
from noeira.physics3d.kinematics.quat_math import quat_rotate_inverse
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_JOINT_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
)

from .fish_xml import (
    DMFishUprightModel,
    DMFishSwimModel,
    TORSO_BODY_IDX,
    TARGET_BODY_IDX,
    MOUTH_GEOM_IDX,
    TARGET_GEOM_IDX,
    N_ROOT_QPOS,
    FREE_QUAT_ADR,
    MOUTH_RADIUS,
    TARGET_RADIUS,
    JOINT_INIT_SPREAD,
    TARGET_BOX_XY,
    TARGET_Z_MIN,
    TARGET_Z_MAX,
)

from ..dtype_math import sqrt_dt
from ..gpu_reset import reset_seed, standard_normal
from ..rewards import tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN
from ...phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# `radii = physics.named.model.geom_size[['mouth', 'target'], 0].sum()`.
comptime SWIM_RADII: Float64 = MOUTH_RADIUS + TARGET_RADIUS


def _standard_normal() -> Float64:
    """One N(0, 1) draw (Box-Muller), for the random-orientation reset."""
    var u1 = random_float64()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _mouth_to_target[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
) raises -> Tuple[Float64, Float64, Float64]:
    """`Physics.mouth_to_target` — target minus mouth, in the MOUTH's frame.

        (geom_xpos['target'] - geom_xpos['mouth']).dot(geom_xmat['mouth'])

    and `v.dot(M)` for a 1-D `v` is `M^T v`, i.e. the vector expressed in the
    mouth geom's local frame — `quat_rotate_inverse` by its world quaternion.
    """
    var mouth = geom_xpos(d, m_geoms, MOUTH_GEOM_IDX)
    var target = geom_xpos(d, m_geoms, TARGET_GEOM_IDX)
    var q = geom_xquat(d, m_geoms, MOUTH_GEOM_IDX)
    var loc = quat_rotate_inverse[DType.float64](
        q[0], q[1], q[2], q[3],
        target[0] - mouth[0],
        target[1] - mouth[1],
        target[2] - mouth[2],
    )
    return (loc[0], loc[1], loc[2])


def _upright[DTYPE: DType, D: DimsLike](d: Data[DTYPE, D, 1]) raises -> Float64:
    """`Physics.upright` — `xmat['torso', 'zz']`, the torso z-axis projected
    onto the world z-axis. +1 upright, -1 upside down."""
    return xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)


def _append_shared_obs[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    mut obs: List[Scalar[DTYPE]],
) raises:
    """`joint_angles` then `upright` — the head of both observations."""
    for q in range(N_ROOT_QPOS, D.NQ):
        obs.append(d.qpos.data[q])
    obs.append(Scalar[DTYPE](_upright(d)))


def _append_velocity[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    mut obs: List[Scalar[DTYPE]],
):
    """`physics.velocity()` — the WHOLE `qvel`, free root included.

    Note this is `mujoco.Physics.velocity()`, not the `torso_velocity()`
    sensor pair defined next to it in `fish.py`, which no task reads.
    """
    for v in range(D.NV):
        obs.append(d.qvel.data[v])


def _reset_pose[DTYPE: DType, D: DimsLike](mut d: Data[DTYPE, D, 1]):
    """The half of `initialize_episode` both tasks share.

    A uniform random root orientation, then every internal joint uniform in
    +-.2. The free joint's TRANSLATION is untouched, exactly as in the
    reference — the fish always starts at the model's `pos="0 0 .1"`.
    """
    var qw = _standard_normal()
    var qx = _standard_normal()
    var qy = _standard_normal()
    var qz = _standard_normal()
    var n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        qw = 1.0
        qx = 0.0
        qy = 0.0
        qz = 0.0
        n = 1.0
    # qpos[3:7] is (w, x, y, z) — MuJoCo's free-joint layout, which our FK
    # reads in that order too.
    d.qpos.data[FREE_QUAT_ADR + 0] = Scalar[DTYPE](qw / n)
    d.qpos.data[FREE_QUAT_ADR + 1] = Scalar[DTYPE](qx / n)
    d.qpos.data[FREE_QUAT_ADR + 2] = Scalar[DTYPE](qy / n)
    d.qpos.data[FREE_QUAT_ADR + 3] = Scalar[DTYPE](qz / n)

    for q in range(N_ROOT_QPOS, D.NQ):
        d.qpos.data[q] = Scalar[DTYPE](
            -JOINT_INIT_SPREAD + random_float64() * 2.0 * JOINT_INIT_SPREAD
        )


# XORed into the reset key for the TARGET draw so it is a different Philox
# stream from the pose draw — see swimmer's `_TARGET_STREAM` for why sharing
# one generator across two independent quantities is a trap.
comptime _TARGET_STREAM: UInt64 = 0xD1B54A32D192ED03


@always_inline
def _mouth_to_target_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NBODY: Int, NGEOM_F: Int
](
    xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
    xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
    geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
    env: Int,
) -> Array[Scalar[DTYPE], 3]:
    """`_mouth_to_target` against the batched field tensors.

    ⚠ THE MOUTH'S FRAME IS NOT ITS BODY'S. It is a `fromto` capsule, so the
    compiler derived a quaternion for it — hence `geom_xquat_gpu` rather than
    a body `xquat`. Substituting the body quaternion here would be wrong by
    the fromto rotation (90 degrees for this geom) and would still produce a
    plausible-looking 3-vector.
    """
    var mouth = geom_xpos_gpu[DTYPE](
        xpos, xquat, geoms, env, MOUTH_GEOM_IDX
    )
    var target = geom_xpos_gpu[DTYPE](
        xpos, xquat, geoms, env, TARGET_GEOM_IDX
    )
    var q = geom_xquat_gpu[DTYPE](
        xquat, geoms, env, MOUTH_GEOM_IDX
    )
    var loc = quat_rotate_inverse[DTYPE](
        q[0], q[1], q[2], q[3],
        target[0] - mouth[0],
        target[1] - mouth[1],
        target[2] - mouth[2],
    )
    var out = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    out[0] = loc[0]
    out[1] = loc[1]
    out[2] = loc[2]
    return out^


@always_inline
def _reset_pose_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int
](
    qpos: Tile[DTYPE, BATCH_SIZE, NQ],
    env: Int,
    seed: Int,
):
    """`_reset_pose` — uniform random root orientation, joints in +-.2.

    The free joint's TRANSLATION is untouched, exactly as in the reference:
    the fish always starts at the model's `pos="0 0 .1"`.

    ⚠ A normalized 4-vector of NORMALS is uniform on the 3-sphere; a
    normalized vector of UNIFORMS is not. `standard_normal` is load-bearing.
    (This is the opposite of the reference's FREE-JOINT randomizer quirk that
    `gpu_reset` reproduces — fish's own reset really does call `randn`.)
    """
    var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)
    var b0 = rng.step_uniform()
    var b1 = rng.step_uniform()

    var q = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    q[0] = standard_normal[DTYPE](Scalar[DTYPE](b0[0]), Scalar[DTYPE](b0[1]))
    q[1] = standard_normal[DTYPE](Scalar[DTYPE](b0[2]), Scalar[DTYPE](b0[3]))
    q[2] = standard_normal[DTYPE](Scalar[DTYPE](b1[0]), Scalar[DTYPE](b1[1]))
    q[3] = standard_normal[DTYPE](Scalar[DTYPE](b1[2]), Scalar[DTYPE](b1[3]))

    var n = sqrt_dt[DTYPE](
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    )
    if n < Scalar[DTYPE](1e-12):
        q[0] = Scalar[DTYPE](1)
        q[1] = Scalar[DTYPE](0)
        q[2] = Scalar[DTYPE](0)
        q[3] = Scalar[DTYPE](0)
        n = Scalar[DTYPE](1)

    # qpos[3:7] is (w, x, y, z) — MuJoCo's free-joint layout.
    for i in range(4):
        qpos[env, FREE_QUAT_ADR + i] = q[i] / n

    # Every internal joint uniform in +-JOINT_INIT_SPREAD. Philox yields four
    # uniforms per step, so draw in blocks of four and index into the block.
    var blk = rng.step_uniform()
    var slot = 0
    for j in range(N_ROOT_QPOS, NQ):
        if slot == 4:
            blk = rng.step_uniform()
            slot = 0
        var u = Scalar[DTYPE](blk[slot])
        slot += 1
        qpos[env, j] = Scalar[DTYPE](-JOINT_INIT_SPREAD) + u * Scalar[DTYPE](
            2.0 * JOINT_INIT_SPREAD
        )


@always_inline
def _upright_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NBODY: Int
](
    xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
    env: Int,
) -> Scalar[DTYPE]:
    """`Physics.upright` — `xmat['torso', 'zz']`."""
    return xmat_elem_gpu[DTYPE](
        xquat, env, TORSO_BODY_IDX, XMAT_ZZ
    )


@always_inline
def _append_shared_obs_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NQ: Int, NBODY: Int, OBS_DIM: Int
](
    qpos: Tile[DTYPE, BATCH_SIZE, NQ],
    xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
    obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
    env: Int,
    mut k: Int,
):
    """`joint_angles` then `upright` — the head of both observations."""
    for q in range(N_ROOT_QPOS, NQ):
        obs[env, k] = qpos[env, q]
        k += 1
    obs[env, k] = _upright_gpu[DTYPE, BATCH_SIZE, NBODY](xquat, env)
    k += 1


@always_inline
def _append_velocity_gpu[
    DTYPE: DType, BATCH_SIZE: Int, NV: Int, OBS_DIM: Int
](
    qvel: Tile[DTYPE, BATCH_SIZE, NV],
    obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
    env: Int,
    mut k: Int,
):
    """`physics.velocity()` — the WHOLE `qvel`, free root included."""
    for v in range(NV):
        obs[env, k] = qvel[env, v]
        k += 1


struct DMFishUprightConfig(Phyics3dEnvConfig):
    """`Upright`: get the torso's z-axis pointing at the world's."""

    # === Physics ===
    # `_CONTROL_TIMESTEP = .04` over a `.004` physics step => 10 substeps,
    # and `_DEFAULT_TIME_LIMIT = 40` s => 1000 control steps.
    comptime FRAME_SKIP: Int = 10
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
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
        """`Upright.get_observation`: joint_angles, upright, velocity."""
        try:
            _append_shared_obs(d, obs)
        except:
            return False
        _append_velocity(d, obs)
        return True

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Upright.initialize_episode` — pose only.

        The `geom_rgba['target', 3] = 0` write is the task hiding an object it
        never reads; purely visual, dropped.
        """
        _reset_pose(d)

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
        """`tolerance(upright, bounds=(1, 1), margin=1)` — a degenerate
        interval, so the reward is the gaussian sigmoid of `1 - upright`."""
        try:
            var u = _upright(d)
            return (Scalar[DTYPE](tolerance(u, 1.0, 1.0, 1.0)), False)
        except:
            return (Scalar[DTYPE](0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMFishUprightModel.TIMESTEP)


    # ── GPU hooks ────────────────────────────────────────────────────────
    comptime HAS_GPU_HOOKS: Bool = True
    # ⚠ TRUE EVEN THOUGH `Upright` NEVER MOVES THE TARGET. The model still
    # carries the mocap `target` body (both tasks share one XML), and
    # `Phyics3dBatchedEnv.__init__` RAISES if a body is mocap-flagged while
    # this is False. Declaring it costs one no-op sync kernel per step; the
    # alternative is a refused instantiation.
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
        """`Upright.initialize_episode` — pose only.

        The `geom_rgba['target', 3] = 0` write is the task hiding an object it
        never reads; purely visual, dropped.
        """
        _reset_pose_gpu[DTYPE, BATCH_SIZE, NQ](qpos, env, seed)

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
        """`Upright.get_observation`: joint_angles, upright, velocity."""
        comptime assert (
            (NQ - N_ROOT_QPOS) + 1 + NV == OBS_DIM
        ), (
            "fish-upright: joint_angles(NQ-7) + upright(1) + velocity(NV)"
            " must equal OBS_DIM exactly."
        )
        var k = 0
        _append_shared_obs_gpu[DTYPE, BATCH_SIZE, NQ, NBODY, OBS_DIM](
            qpos, xquat, obs, env, k
        )
        _append_velocity_gpu[DTYPE, BATCH_SIZE, NV, OBS_DIM](
            qvel, obs, env, k
        )
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
        """`tolerance(upright, bounds=(1, 1), margin=1)` — a degenerate
        interval, so the reward is the gaussian sigmoid of `1 - upright`."""
        var u = _upright_gpu[DTYPE, BATCH_SIZE, NBODY](xquat, env)
        var r = tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE](
            u, Scalar[DTYPE](1.0), Scalar[DTYPE](1.0), Scalar[DTYPE](1.0)
        )
        return (r, False)


struct DMFishSwimConfig(Phyics3dEnvConfig):
    """`Swim`: bring the mouth to the target, staying upright."""

    comptime FRAME_SKIP: Int = 10
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
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
        """`Swim.get_observation`: + target, before velocity."""
        try:
            _append_shared_obs(d, obs)
            var t = _mouth_to_target(d, m_geoms)
            obs.append(Scalar[DTYPE](t[0]))
            obs.append(Scalar[DTYPE](t[1]))
            obs.append(Scalar[DTYPE](t[2]))
        except:
            return False
        _append_velocity(d, obs)
        return True

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`Swim.initialize_episode` — pose, then the target box.

        The reference writes `model.geom_pos['target', 'xyz']`; ours is the
        per-env mocap pose the geom rides on (gap G4).
        """
        _reset_pose(d)

        var tx = -TARGET_BOX_XY + random_float64() * 2.0 * TARGET_BOX_XY
        var ty = -TARGET_BOX_XY + random_float64() * 2.0 * TARGET_BOX_XY
        var tz = TARGET_Z_MIN + random_float64() * (
            TARGET_Z_MAX - TARGET_Z_MIN
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = Scalar[DTYPE](tx)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = Scalar[DTYPE](ty)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = Scalar[DTYPE](tz)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)

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
        """`(7*in_target + is_upright) / 8`."""
        try:
            var t = _mouth_to_target(d, m_geoms)
            var dist = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
            var in_target = tolerance(
                dist, 0.0, SWIM_RADII, 2.0 * SWIM_RADII
            )
            var is_upright = 0.5 * (_upright(d) + 1.0)
            return (
                Scalar[DTYPE]((7.0 * in_target + is_upright) / 8.0),
                False,
            )
        except:
            return (Scalar[DTYPE](0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMFishSwimModel.TIMESTEP)

    # ── GPU hooks ────────────────────────────────────────────────────────
    comptime HAS_GPU_HOOKS: Bool = True
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
        """`Swim.initialize_episode` — pose, then the target box.

        The reference writes `model.geom_pos['target', 'xyz']`; ours is the
        per-env mocap pose the geom rides on (gap G4).
        """
        _reset_pose_gpu[DTYPE, BATCH_SIZE, NQ](qpos, env, seed)

        # A SEPARATE stream from the pose draw, which consumed a
        # joint-count-dependent number of Philox blocks.
        var rng = PhiloxRandom(
            seed=reset_seed(env, seed) ^ _TARGET_STREAM, offset=0
        )
        var b = rng.step_uniform()
        var tx = Scalar[DTYPE](-TARGET_BOX_XY) + Scalar[DTYPE](
            b[0]
        ) * Scalar[DTYPE](2.0 * TARGET_BOX_XY)
        var ty = Scalar[DTYPE](-TARGET_BOX_XY) + Scalar[DTYPE](
            b[1]
        ) * Scalar[DTYPE](2.0 * TARGET_BOX_XY)
        var tz = Scalar[DTYPE](TARGET_Z_MIN) + Scalar[DTYPE](
            b[2]
        ) * Scalar[DTYPE](TARGET_Z_MAX - TARGET_Z_MIN)

        mocap_pos[env, TARGET_BODY_IDX * 3 + 0] = tx
        mocap_pos[env, TARGET_BODY_IDX * 3 + 1] = ty
        mocap_pos[env, TARGET_BODY_IDX * 3 + 2] = tz
        mocap_quat[env, TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        mocap_quat[env, TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](0)
        mocap_quat[env, TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        mocap_quat[env, TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)

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
        """`Swim.get_observation`: + target, before velocity."""
        comptime assert (
            (NQ - N_ROOT_QPOS) + 1 + 3 + NV == OBS_DIM
        ), (
            "fish-swim: joint_angles(NQ-7) + upright(1) + target(3) +"
            " velocity(NV) must equal OBS_DIM exactly."
        )
        var k = 0
        _append_shared_obs_gpu[DTYPE, BATCH_SIZE, NQ, NBODY, OBS_DIM](
            qpos, xquat, obs, env, k
        )
        var t = _mouth_to_target_gpu[DTYPE, BATCH_SIZE, NBODY, NGEOM_F](
            xpos, xquat, geoms, env
        )
        for i in range(3):
            obs[env, k] = t[i]
            k += 1
        _append_velocity_gpu[DTYPE, BATCH_SIZE, NV, OBS_DIM](
            qvel, obs, env, k
        )
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
        """`(7*in_target + is_upright) / 8`."""
        var t = _mouth_to_target_gpu[DTYPE, BATCH_SIZE, NBODY, NGEOM_F](
            xpos, xquat, geoms, env
        )
        var dist = sqrt_dt[DTYPE](
            t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
        )
        var in_target = tolerance[
            SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
        ](
            dist,
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](SWIM_RADII),
            Scalar[DTYPE](2.0 * SWIM_RADII),
        )
        var is_upright = Scalar[DTYPE](0.5) * (
            _upright_gpu[DTYPE, BATCH_SIZE, NBODY](xquat, env)
            + Scalar[DTYPE](1.0)
        )
        return (
            (Scalar[DTYPE](7.0) * in_target + is_upright)
            / Scalar[DTYPE](8.0),
            False,
        )
