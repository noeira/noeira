"""`dm_control` `quadruped` task config — port of `suite/quadruped.py`'s `Move`.

One parameterized config covers both in-scope tasks, which differ only in the
target speed (and, in the XML, the floor's half-extent):

    walk = DMQuadrupedConfig[DESIRED_SPEED=0.5]
    run  = DMQuadrupedConfig[DESIRED_SPEED=5.0]

    observation = [egocentric_state(44), torso_velocity(3), torso_upright(1),
                   imu(6), force_torque(24)]                            (78)
    reward      = _upright_reward * tolerance(velocimeter_x, ...)
    reset       = random orientation, then raised in 1 cm steps until nothing
                  touches the floor
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

FIRST DOMAIN HERE WHOSE OBSERVATION IS MOSTLY SENSORS. 34 of the 78 numbers
come from `<sensor>` elements — the velocimeter, the IMU pair, and the eight
force/torque sensors at the toes. The last three need
`mj_rnePostConstraint`, which is why `RNE_POST` is set; without it `d.cacc`
and `d.cfrc_int` stay zero and 30 of the 78 dims are silently zero too.

ORDERING IS BY SENSOR ID, NOT BY THE `_TOES` LIST. `physics.imu()` and
`physics.force_torque()` both build their name lists from
`np.where(np.isin(model.sensor_type, ...))`, which yields sensor IDs in
ascending order — i.e. XML declaration order. So:

    imu          = [accelerometer(3), gyro(3)]        (accel is declared first)
    force_torque = [force x4, then torque x4], each in FL, FR, BR, BL order

`_TOES` in the reference is a DIFFERENT order (FL, BL, BR, FR) and is used
only by `toe_positions()`, which `Move` never calls. Following it here would
transpose twelve of the observation's dims with nothing to complain.

`data.act` SITS INSIDE `egocentric_state`, not at the end of the observation,
which is why `custom_extract_obs_cpu` takes an `act` argument at all — the
env cannot append the activation after the fact.
"""

from std.math import log, sqrt, cos, sin, pi, inf, abs
from std.random import random_float64
from std.collections import Array

from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.kinematics.xmat import xmat_elem, xmat_elem_gpu, XMAT_ZZ
from noeira.physics3d.sensors.frame_vel import (
    site_frame_velocity,
    site_frame_velocity_gpu,
)
from noeira.physics3d.sensors.site_acc import (
    site_accelerometer,
    site_accelerometer_gpu,
    site_force_torque,
    site_force_torque_gpu,
)
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    MODEL_JOINT_SIZE,
)

from ..dtype_math import asinh_dt, sqrt_dt, inf_dt
from ..gpu_reset import reset_seed, standard_normal
from ..rewards import tolerance, SIGMOID_LINEAR
from .quadruped_xml import (
    qwp,
    QUADRUPED_OBS_DIM,
    TORSO_BODY_IDX,
    TORSO_SITE_IDX,
    TOE_BODY_0,
    TOE_BODY_STRIDE,
    TOE_SITE_0,
    N_HINGE,
    HINGE_QPOS_0,
    HINGE_DOF_0,
)
from ...phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# `_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP` = 20 / .02.
comptime QUADRUPED_MAX_STEPS: Int = 1000
# `_CONTROL_TIMESTEP / <option timestep>` = .02 / .005.
comptime QUADRUPED_FRAME_SKIP: Int = 4


@always_inline
def _asinh(x: Float64) -> Float64:
    """`np.arcsinh`. The identity, folded onto |x| — see the note below.

    ⚠ THE FOLD IS NOT COSMETIC. `x + sqrt(x*x + 1)` cancels catastrophically
    for large negative x, and a loaded toe here reaches |force| ~ 1.4e3 — far
    enough that the float32 GPU twin collapsed all four toes onto a single
    value (see `dm_control/dtype_math.asinh_dt`, which carries the
    measurement). float64 degrades later rather than escaping, and the two
    paths are diffed element-wise, so both use the same stable form.

    ⚠⚠ `std.math.asinh` EXISTS AND IS BETTER AND WE STILL CANNOT USE IT — see
    `dtype_math.asinh_dt` for the measurements and the reason. It does not
    lower to the GPU target, and this observable has a GPU twin that is diffed
    against this one element-wise, so switching only here would trade a known
    accuracy loss for an unknown CPU/GPU divergence. The fold stays until the
    GPU side can follow.
    """
    var a = abs(x)
    var r = log(a + sqrt(a * a + 1.0))
    return -r if x < 0.0 else r


@always_inline
def _random_root_orientation[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1]
):
    """`orientation = randn(4); orientation /= norm(orientation)`, into qpos.

    ⚠ A NORMALISED GAUSSIAN, NOT A NORMALISED UNIFORM. A normalised vector of
    uniforms clusters toward the cube's corners, so the Box-Muller pair is
    load-bearing rather than decoration.

    The height is left at 0 for `_find_non_contacting_height` to raise — the
    env does that after the reset hooks, because it needs FK and broadphase.

    ⚠ SHARED BY `Move` AND `Escape`. Both draw the same orientation; keeping
    one copy is what stops the two drifting when one is touched.
    """
    var q = Array[Float64, 4](fill=0.0)
    for pair in range(2):
        var u1 = random_float64()
        if u1 < 1e-300:
            u1 = 1e-300
        var u2 = random_float64()
        var r = sqrt(-2.0 * log(u1))
        q[2 * pair + 0] = r * cos(2.0 * pi * u2)
        q[2 * pair + 1] = r * sin(2.0 * pi * u2)

    var n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        q[0] = 1.0
        q[1] = 0.0
        q[2] = 0.0
        q[3] = 0.0
        n = 1.0

    # Free-joint qpos is [x, y, z, qw, qx, qy, qz] — w FIRST.
    d.qpos.data[0] = Scalar[DTYPE](0)
    d.qpos.data[1] = Scalar[DTYPE](0)
    d.qpos.data[2] = Scalar[DTYPE](0)
    for k in range(4):
        d.qpos.data[3 + k] = Scalar[DTYPE](q[k] / n)


@always_inline
def _upright_reward_at(zz: Float64, deviation: Float64) -> Float64:
    """`_upright_reward(physics, deviation_angle)` for any deviation.

    `bounds=(deviation, inf)`, `margin=1 + deviation`, linear, 0 at margin —
    so the term is 1 while the torso z-axis is within `acos(deviation)` of
    world up and falls linearly to 0 at exactly upside-down.

    ⚠ `Move` passes `deviation_angle=0` (deviation 1.0) and `Escape` passes
    **20 degrees** (deviation 0.9397). Sharing one hardcoded 1.0 between them
    would make escape strictly harder in a way no gate on the walk model can
    see, because walk never exercises the other value.
    """
    return tolerance[SIGMOID_LINEAR, 0.0](
        zz, deviation, inf[DType.float64](), 1.0 + deviation
    )


@always_inline
def _upright_reward(zz: Float64) -> Float64:
    """`_upright_reward(physics, deviation_angle=0)`.

    `deviation = cos(0) = 1`, so the bound is [1, inf) with `margin = 1 + 1`
    and `value_at_margin = 0` on a linear sigmoid: 1 when the torso z-axis is
    exactly up, 0 when it is exactly upside-down, linear in between.
    """
    return tolerance[SIGMOID_LINEAR, 0.0](zz, 1.0, inf[DType.float64](), 2.0)


@always_inline
def _common_obs_cpu[DTYPE: DType, TORSO_SITE: Int, TOE_SITE_0_P: Int, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_joints: List[Scalar[DTYPE]],
    m_geoms: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    act: List[Scalar[DTYPE]],
    mut obs: List[Scalar[DTYPE]],
) -> Bool:
    """`_common_observations` — the five blocks, in order, 78 numbers.

    ⚠ THE SITE INDICES ARE PARAMETERS, not the module constants, because
    `fetch` declares a `target` site ahead of the torso body and every site id
    shifts by one. Hardcoding walk/run's ids here would leave fetch reading the
    velocimeter off the wrong site — finite, plausible, and wrong.
    """
    try:
        # --- egocentric_state: hinge qpos, hinge qvel, act -----------------
        for k in range(N_HINGE):
            obs.append(d.qpos.data[HINGE_QPOS_0 + k])
        for k in range(N_HINGE):
            obs.append(d.qvel.data[HINGE_DOF_0 + k])
        for k in range(len(act)):
            obs.append(act[k])

        # --- torso_velocity: the velocimeter at the torso site -------------
        var fv = site_frame_velocity[DTYPE](
            d.xvel.data, d.xangvel.data, d.xipos.data, d.xquat.data,
            d.site_xpos.data, m_sites, TORSO_BODY_IDX, TORSO_SITE,
        )
        obs.append(Scalar[DTYPE](fv[0]))
        obs.append(Scalar[DTYPE](fv[1]))
        obs.append(Scalar[DTYPE](fv[2]))

        # --- torso_upright: xmat['torso', 'zz'] ----------------------------
        obs.append(Scalar[DTYPE](xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)))

        # --- imu: accelerometer THEN gyro (sensor-id order) ----------------
        # ⚠ The acceleration-stage SNAPSHOT (`*_acc`), not the live FK
        # products — defect 19. quadruped's own gate runs at frame_skip=1,
        # where the two happen to differ less, which is why this went
        # unnoticed here and surfaced on dog.
        var acc = site_accelerometer[DTYPE](
            d.cvel.data, d.cacc.data, d.subtree_com.data,
            d.site_xpos_acc.data, d.xquat_acc.data, m_bodies, m_sites,
            TORSO_BODY_IDX, TORSO_SITE,
        )
        obs.append(Scalar[DTYPE](acc[0]))
        obs.append(Scalar[DTYPE](acc[1]))
        obs.append(Scalar[DTYPE](acc[2]))
        obs.append(Scalar[DTYPE](fv[3]))
        obs.append(Scalar[DTYPE](fv[4]))
        obs.append(Scalar[DTYPE](fv[5]))

        # --- force_torque: arcsinh(all four forces, then all four torques)
        #     — two passes over the toes, not one interleaved.
        var fx = Array[Float64, 12](fill=0.0)
        var tx = Array[Float64, 12](fill=0.0)
        for t in range(4):
            var ftt = site_force_torque[DTYPE](
                d.cfrc_int.data, d.subtree_com.data, d.site_xpos_acc.data,
                d.xquat_acc.data, m_bodies, m_sites,
                TOE_BODY_0 + t * TOE_BODY_STRIDE, TOE_SITE_0_P + t,
            )
            # Tuple subscripts need a comptime index; unpack once.
            fx[t * 3 + 0] = ftt[0]
            fx[t * 3 + 1] = ftt[1]
            fx[t * 3 + 2] = ftt[2]
            tx[t * 3 + 0] = ftt[3]
            tx[t * 3 + 1] = ftt[4]
            tx[t * 3 + 2] = ftt[5]
        for k in range(12):
            obs.append(Scalar[DTYPE](_asinh(fx[k])))
        for k in range(12):
            obs.append(Scalar[DTYPE](_asinh(tx[k])))
    except:
        return False
    return True


@always_inline
def _common_obs_gpu[
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    OBS_DIM: Int,
    SITE_DIM: Int,
    NSITE_F: Int,
    NA_F: Int,
    TORSO_SITE: Int,
    TOE_SITE_0_P: Int,
    COMMON_DIM: Int = OBS_DIM,
](
    qpos: Tile[DTYPE, BATCH_SIZE, NQ],
    qvel: Tile[DTYPE, BATCH_SIZE, NV],
    xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
    xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
    xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
    xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
    sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
    cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
    cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
    cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
    subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
    site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
    xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
    act: Tile[DTYPE, BATCH_SIZE, NA_F],
    obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
    env: Int,
) -> Bool:
    """`_common_obs_cpu` against the batched field tensors, block for block.

    ⚠ THE SITE INDICES ARE PARAMETERS for the same reason they are on the CPU
    twin: `fetch` declares a `target` site ahead of the torso body, so every
    site id shifts by one.

    ⚠ THE SIZE ASSERT IS LOAD-BEARING. This writes `obs[env, k]` by running
    index rather than appending to a `List`, so a block that is one dim short
    silently leaves the tail at whatever the buffer last held — a stale
    observation, not a crash. The assert below is what makes a mis-sized
    block a compile error.
    """
    # ⚠ AGAINST `COMMON_DIM`, WHICH DEFAULTS TO `OBS_DIM`. walk/run/fetch
    # observe nothing beyond these five blocks, so for them the two are the
    # same number and the check is what it always was. `escape` appends an
    # origin and twenty rangefinders AFTER them, so it passes the common
    # width explicitly and the exact check survives for both.
    comptime assert (
        2 * N_HINGE + NA_F + 3 + 1 + 6 + 24 == COMMON_DIM
    ), (
        "quadruped._common_obs_gpu: the five blocks do not sum to COMMON_DIM."
        " egocentric_state(2*N_HINGE + NA_F) + torso_velocity(3) +"
        " torso_upright(1) + imu(6) + force_torque(24) must equal it exactly."
    )
    comptime assert COMMON_DIM <= OBS_DIM, (
        "quadruped._common_obs_gpu: COMMON_DIM exceeds the observation buffer."
    )

    var k = 0

    # --- egocentric_state: hinge qpos, hinge qvel, act ---------------------
    for i in range(N_HINGE):
        obs[env, k] = qpos[env, HINGE_QPOS_0 + i]
        k += 1
    for i in range(N_HINGE):
        obs[env, k] = qvel[env, HINGE_DOF_0 + i]
        k += 1
    # `act` is the ACTUATOR ACTIVATION, and it sits INSIDE this block rather
    # than at the end of the observation. Zero here means the twelve
    # `dyntype="filter"` servos are dead — blocker E3's failure mode, visible
    # as twelve permanently-zero dims.
    for i in range(NA_F):
        obs[env, k] = act[env, i]
        k += 1

    # --- torso_velocity: the velocimeter at the torso site -----------------
    var fv = site_frame_velocity_gpu[
        DTYPE](
        xvel, xangvel, xipos, xquat, site_xpos, sites,
        env, TORSO_BODY_IDX, TORSO_SITE,
    )
    for i in range(3):
        obs[env, k] = fv[i]
        k += 1

    # --- torso_upright: xmat['torso', 'zz'] --------------------------------
    obs[env, k] = xmat_elem_gpu[DTYPE](
        xquat, env, TORSO_BODY_IDX, XMAT_ZZ
    )
    k += 1

    # --- imu: accelerometer THEN gyro (sensor-id order) --------------------
    # ⚠ `site_xpos_acc` / `xquat_acc`, NOT the live products — defect 19.
    var acc = site_accelerometer_gpu[
        DTYPE](
        cvel, cacc, subtree_com, site_xpos_acc, xquat_acc, bodies, sites,
        env, TORSO_BODY_IDX, TORSO_SITE,
    )
    for i in range(3):
        obs[env, k] = acc[i]
        k += 1
    for i in range(3):
        obs[env, k] = fv[3 + i]
        k += 1

    # --- force_torque: arcsinh(all four forces, then all four torques) -----
    #     Two passes over the toes, not one interleaved.
    var fx = Array[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))
    var tx = Array[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))
    for t in range(4):
        var ftt = site_force_torque_gpu[
            DTYPE](
            cfrc_int, subtree_com, site_xpos_acc, xquat_acc, bodies, sites,
            env, TOE_BODY_0 + t * TOE_BODY_STRIDE, TOE_SITE_0_P + t,
        )
        for i in range(3):
            fx[t * 3 + i] = ftt[i]
            tx[t * 3 + i] = ftt[3 + i]
    for i in range(12):
        obs[env, k] = asinh_dt[DTYPE](fx[i])
        k += 1
    for i in range(12):
        obs[env, k] = asinh_dt[DTYPE](tx[i])
        k += 1

    return True


struct DMQuadrupedConfig[DESIRED_SPEED: Float64](Phyics3dEnvConfig):
    """`Move`, for both walk (0.5) and run (5.0)."""

    comptime FRAME_SKIP: Int = QUADRUPED_FRAME_SKIP
    comptime MAX_STEPS: Int = QUADRUPED_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # MuJoCo's default integrator, which quadruped.xml does not override.
    comptime INTEGRATOR: StaticString = "euler"
    # dm_control runs mj_step1 after the last substep, so the task reads
    # position- and velocity-dependent fields at the NEW state.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # accelerometer + force/torque need `mj_rnePostConstraint`.
    comptime RNE_POST: Bool = True
    # `_find_non_contacting_height` after the orientation draw below.
    comptime RESET_FIND_HEIGHT: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(qwp.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        # `Move.initialize_episode` randomizes the ORIENTATION only — no joint
        # or velocity jitter. The hinges stay at qpos0.
        return 0.0

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`orientation = randn(4); orientation /= norm(orientation)`.

        A normalized 4-vector of independent normals is uniform on SO(3); a
        normalized vector of UNIFORMS is not (it clusters toward the cube's
        corners), so the Box-Muller pair below is load-bearing, not decoration.

        The height is left at 0 for `_find_non_contacting_height` to raise —
        the env does that after this hook, because it needs FK and broadphase.
        """
        _random_root_orientation[DTYPE, D](d)

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
        """`_common_observations` — the five blocks, in order."""
        return _common_obs_cpu[DTYPE, TORSO_SITE_IDX, TOE_SITE_0](d, m_bodies, m_joints, m_geoms, m_sites, act, obs)

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
        """`Move.get_reward` — upright x move, never terminates early."""
        var zz = xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)
        var upright = _upright_reward(zz)

        var vx: Float64
        try:
            var fv = site_frame_velocity[DTYPE](
                d.xvel.data, d.xangvel.data, d.xipos.data, d.xquat.data,
                d.site_xpos.data, m_sites, TORSO_BODY_IDX, TORSO_SITE_IDX,
            )
            vx = fv[0]
        except:
            return (Scalar[DTYPE](0), False)

        # `margin = desired_speed` and `value_at_margin = 0.5`: at zero speed
        # the move term is exactly 0.5, not 0 — so a quadruped that only
        # stands still scores half of the upright reward rather than nothing.
        var move = tolerance[SIGMOID_LINEAR, 0.5](
            vx, Self.DESIRED_SPEED, inf[DType.float64](), Self.DESIRED_SPEED
        )
        return (Scalar[DTYPE](upright * move), False)


    # ── GPU hooks ────────────────────────────────────────────────────────

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
        """`Move.initialize_episode` — a uniform random torso orientation.

        The GPU twin of `custom_reset_cpu`. A normalized 4-vector of
        INDEPENDENT NORMALS is uniform on SO(3); a normalized vector of
        UNIFORMS is not (it clusters toward the cube's corners), so the
        Box-Muller draws are load-bearing rather than decoration. This is why
        the hook cannot reuse `randomize_limited_and_rotational_joints_gpu`,
        whose free-joint branch deliberately reproduces the reference's
        `random.rand(4)` quirk — quadruped's own reset does `randn(4)`.

        The height stays 0 for `_find_non_contacting_height_batch` to raise;
        the hinges stay at qpos0 (`get_reset_noise` is 0 for the same reason).

        ⚠ THE STREAM DOES NOT MATCH THE CPU PATH, by construction: the CPU
        draws from host `random_float64` and keeps BOTH Box-Muller halves,
        this draws from per-lane Philox and keeps the cosine half only (see
        `standard_normal`). Same distribution, different sequence — so the
        GPU-vs-CPU gate injects a shared qpos/qvel rather than comparing
        resets, exactly as every other suite gate does.
        """
        var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)
        var b0 = rng.step_uniform()
        var b1 = rng.step_uniform()

        var q = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        q[0] = standard_normal[DTYPE](
            Scalar[DTYPE](b0[0]), Scalar[DTYPE](b0[1])
        )
        q[1] = standard_normal[DTYPE](
            Scalar[DTYPE](b0[2]), Scalar[DTYPE](b0[3])
        )
        q[2] = standard_normal[DTYPE](
            Scalar[DTYPE](b1[0]), Scalar[DTYPE](b1[1])
        )
        q[3] = standard_normal[DTYPE](
            Scalar[DTYPE](b1[2]), Scalar[DTYPE](b1[3])
        )

        # `sqrt` needs `is_floating_point()` evidence an unconstrained trait
        # DTYPE cannot supply — hence the shim
        # (`feedback_where_clause_cannot_cross_trait_boundary`).
        var nn = sqrt_dt[DTYPE](
            q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
        )
        if nn < Scalar[DTYPE](1e-12):
            q[0] = Scalar[DTYPE](1)
            q[1] = Scalar[DTYPE](0)
            q[2] = Scalar[DTYPE](0)
            q[3] = Scalar[DTYPE](0)
            nn = Scalar[DTYPE](1)

        # Free-joint qpos is [x, y, z, qw, qx, qy, qz] — W FIRST. (Only
        # `Data.xquat` is xyzw.) The three linear dofs stay at 0; the height
        # is the search's job.
        for i in range(3):
            qpos[env, i] = Scalar[DTYPE](0)
        for i in range(4):
            qpos[env, 3 + i] = q[i] / nn

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
        """`_common_observations` — the five blocks, in order."""
        return _common_obs_gpu[
            DTYPE, BATCH_SIZE, NQ, NV, NBODY, OBS_DIM, SITE_DIM, NSITE_F,
            NA_F, TORSO_SITE_IDX, TOE_SITE_0,
        ](
            qpos, qvel, xipos, xquat, xvel, xangvel, bodies, site_xpos,
            sites, cvel, cacc, cfrc_int, subtree_com, site_xpos_acc,
            xquat_acc, act, obs, env,
        )

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
        """`Move.get_reward` — upright x move, never terminates early."""
        var zz = xmat_elem_gpu[DTYPE](
            xquat, env, TORSO_BODY_IDX, XMAT_ZZ
        )
        # `_upright_reward` in DTYPE: bound [1, inf), margin 2,
        # value_at_margin 0 on a linear sigmoid.
        var upright = tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
            zz,
            Scalar[DTYPE](1.0),
            inf_dt[DTYPE](),
            Scalar[DTYPE](2.0),
        )

        # The velocimeter's x component — the LIVE site pose, not the `*_acc`
        # snapshot: this is a velocity-stage sensor, and dm_control reads it
        # after `mj_step1` has refreshed position and velocity at the new
        # state. Only the acceleration stage is frozen (defect 19).
        var fv = site_frame_velocity_gpu[
            DTYPE](
            xvel, xangvel, xipos, xquat, site_xpos, sites,
            env, TORSO_BODY_IDX, TORSO_SITE_IDX,
        )

        # `margin = desired_speed` and `value_at_margin = 0.5`: at zero speed
        # the move term is exactly 0.5, not 0 — so a quadruped that only
        # stands still scores half the upright reward rather than nothing.
        var move = tolerance[SIGMOID_LINEAR, 0.5, DTYPE](
            fv[0],
            Scalar[DTYPE](Self.DESIRED_SPEED),
            inf_dt[DTYPE](),
            Scalar[DTYPE](Self.DESIRED_SPEED),
        )
        return (upright * move, False)

comptime DMQuadrupedWalkConfig = DMQuadrupedConfig[0.5]
comptime DMQuadrupedRunConfig = DMQuadrupedConfig[5.0]
