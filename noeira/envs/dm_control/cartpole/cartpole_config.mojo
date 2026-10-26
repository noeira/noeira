"""`dm_control` `cartpole` task configs — port of `suite/cartpole.py` (`Balance`).

One parameterized config covers all six registered tasks:

    balance        = DMCartpoleConfig[1, SWING_UP=False, SPARSE=False]
    balance_sparse = DMCartpoleConfig[1, SWING_UP=False, SPARSE=True]
    swingup        = DMCartpoleConfig[1, SWING_UP=True,  SPARSE=False]
    swingup_sparse = DMCartpoleConfig[1, SWING_UP=True,  SPARSE=True]
    two_poles      = DMCartpoleConfig[2, SWING_UP=True,  SPARSE=False]
    three_poles    = DMCartpoleConfig[3, SWING_UP=True,  SPARSE=False]

    observation = [cart_pos, (zz,xz) per pole..., qvel...]   (2 + 3*N_POLES)
    reward      = sparse: cart_in_bounds * prod(angle_in_bounds)
                  dense : upright.mean * small_control * small_velocity * centered
    episode     = 1000 control steps (10 s / 0.01 s), no early termination
"""

from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import pi, log, sqrt, cos
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

from .cartpole_xml import CART_BODY_IDX, FIRST_POLE_BODY_IDX

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import (
    tolerance,
    SIGMOID_QUADRATIC,
    SIGMOID_GAUSSIAN,
    DEFAULT_VALUE_AT_MARGIN,
)
from ..gpu_reset import reset_seed, standard_normal
from noeira.core.tiles import Tile


# `Balance._CART_RANGE` and `Balance._ANGLE_COSINE_RANGE`.
comptime CART_RANGE_LO: Float64 = -0.25
comptime CART_RANGE_HI: Float64 = 0.25
comptime ANGLE_COSINE_LO: Float64 = 0.995
comptime ANGLE_COSINE_HI: Float64 = 1.0


def _randn() -> Float64:
    """Standard normal via Box-Muller.

    The reference draws its episode init from `numpy.random.RandomState.randn`.
    We cannot reproduce that stream, and do not try: reset randomness is
    explicitly outside the parity test, which injects a fixed state with
    `set_state` instead. Only the DISTRIBUTION needs to match.
    """
    var u1 = random_float64()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


struct DMCartpoleConfig[
    N_POLES: Int,
    SWING_UP: Bool,
    SPARSE: Bool,
](Phyics3dEnvConfig):
    # === Physics ===
    # cartpole.xml: timestep=0.01, and cartpole.py sets no _CONTROL_TIMESTEP,
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
    # <option ... integrator="RK4"> — unlike pendulum, which omits it and so
    # gets MuJoCo's Euler default.
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
        """`Balance.get_observation`: bounded_position() then velocity().

        bounded_position = hstack(cart_position, xmat[2:, ['zz','xz']].ravel())
        so the pole columns interleave as zz_1, xz_1, zz_2, xz_2, ...
        """
        obs.append(d.qpos.data[0])  # cart_position (the slider)
        for p in range(Self.N_POLES):
            var b = FIRST_POLE_BODY_IDX + p
            obs.append(Scalar[DTYPE](xmat_elem(d, b, XMAT_ZZ)))
            obs.append(Scalar[DTYPE](xmat_elem(d, b, XMAT_XZ)))
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
        """`Balance.initialize_episode`."""
        comptime if Self.SWING_UP:
            # Cart centred, first pole pointing down, deeper poles jittered.
            d.qpos.data[0] = Scalar[DTYPE](0.01 * _randn())
            d.qpos.data[1] = Scalar[DTYPE](pi + 0.01 * _randn())
            for i in range(2, D.NQ):
                d.qpos.data[i] = Scalar[DTYPE](0.1 * _randn())
        else:
            # Cart anywhere on the slider, poles near vertical.
            d.qpos.data[0] = Scalar[DTYPE](
                (random_float64() * 2.0 - 1.0) * 0.1
            )
            for i in range(1, D.NQ):
                d.qpos.data[i] = Scalar[DTYPE](
                    (random_float64() * 2.0 - 1.0) * 0.034
                )
        # Small random velocity in both modes, to break symmetry.
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0.01 * _randn())

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
        var cart_pos = Float64(d.qpos.data[0])

        comptime if Self.SPARSE:
            # cart_in_bounds * angle_in_bounds.prod(), both zero-margin.
            var cart_in_bounds = tolerance(
                cart_pos, CART_RANGE_LO, CART_RANGE_HI, 0.0
            )
            var angle_in_bounds = 1.0
            for p in range(Self.N_POLES):
                angle_in_bounds *= tolerance(
                    xmat_elem(d, FIRST_POLE_BODY_IDX + p, XMAT_ZZ),
                    ANGLE_COSINE_LO,
                    ANGLE_COSINE_HI,
                    0.0,
                )
            return (Scalar[DTYPE](cart_in_bounds * angle_in_bounds), False)
        else:
            # upright = ((cos + 1) / 2).mean()
            var upright_sum = 0.0
            for p in range(Self.N_POLES):
                upright_sum += (
                    xmat_elem(d, FIRST_POLE_BODY_IDX + p, XMAT_ZZ) + 1.0
                ) / 2.0
            var upright = upright_sum / Float64(Self.N_POLES)

            # centered = (1 + tolerance(cart_pos, margin=2)) / 2
            var centered = (1.0 + tolerance(cart_pos, 0.0, 0.0, 2.0)) / 2.0

            # small_control = (4 + tolerance(ctrl, margin=1, v@m=0,
            #                                sigmoid='quadratic')[0]) / 5
            var ctrl = actions[0] if len(actions) > 0 else 0.0
            if ctrl > 1.0:
                ctrl = 1.0
            elif ctrl < -1.0:
                ctrl = -1.0
            var small_control = (
                4.0
                + tolerance[SIGMOID_QUADRATIC, 0.0](ctrl, 0.0, 0.0, 1.0)
            ) / 5.0

            # small_velocity = (1 + tolerance(angular_vel, margin=5).min()) / 2
            # angular_vel is qvel[1:] — the hinges, excluding the slider.
            var min_sv = 1.0
            for i in range(1, D.NV):
                var sv = tolerance(Float64(d.qvel.data[i]), 0.0, 0.0, 5.0)
                if sv < min_sv:
                    min_sv = sv
            var small_velocity = (1.0 + min_sv) / 2.0

            var r = upright * small_control * small_velocity * centered
            return (Scalar[DTYPE](r), False)

    # =====================================================================
    # GPU hooks — the batched (`Phyics3dBatchedEnv`) path.
    #
    # ⚠ These MUST stay numerically identical to the CPU hooks above, which
    # are what `tests/dm_control/test_cartpole_vs_dm_control.mojo` gates
    # against MuJoCo. `tests/dm_control/test_cartpole_gpu_vs_cpu.mojo` diffs
    # the two paths step for step, over all three N_POLES and both reward
    # modes — the comptime branches below are separate code paths and a gate
    # on one says nothing about the others.
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
        """`Balance.get_observation` — mirrors `custom_extract_obs_cpu`."""
        obs[env, 0] = qpos[env, 0]  # cart_position (the slider)
        for p in range(Self.N_POLES):
            var b = FIRST_POLE_BODY_IDX + p
            obs[env, 1 + 2 * p] = xmat_elem_gpu[DTYPE](
                xquat, env, b, XMAT_ZZ
            )
            obs[env, 2 + 2 * p] = xmat_elem_gpu[DTYPE](
                xquat, env, b, XMAT_XZ
            )
        for i in range(NV_F):
            obs[env, 1 + 2 * Self.N_POLES + i] = qvel[env, i]
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
        """`Balance.get_reward` — mirrors `compute_reward_and_done_cpu`."""
        comptime ONE = Scalar[DTYPE](1.0)
        var cart_pos = rebind[Scalar[DTYPE]](qpos[env, 0])

        comptime if Self.SPARSE:
            var cart_in_bounds = tolerance[
                SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
            ](
                cart_pos,
                Scalar[DTYPE](CART_RANGE_LO),
                Scalar[DTYPE](CART_RANGE_HI),
                Scalar[DTYPE](0.0),
            )
            var angle_in_bounds = ONE
            for p in range(Self.N_POLES):
                angle_in_bounds *= tolerance[
                    SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
                ](
                    xmat_elem_gpu[DTYPE](
                        xquat, env, FIRST_POLE_BODY_IDX + p, XMAT_ZZ
                    ),
                    Scalar[DTYPE](ANGLE_COSINE_LO),
                    Scalar[DTYPE](ANGLE_COSINE_HI),
                    Scalar[DTYPE](0.0),
                )
            return (cart_in_bounds * angle_in_bounds, False)
        else:
            # upright = ((cos + 1) / 2).mean()
            var upright_sum = Scalar[DTYPE](0.0)
            for p in range(Self.N_POLES):
                upright_sum += (
                    xmat_elem_gpu[DTYPE](
                        xquat, env, FIRST_POLE_BODY_IDX + p, XMAT_ZZ
                    )
                    + ONE
                ) / Scalar[DTYPE](2.0)
            var upright = upright_sum / Scalar[DTYPE](Self.N_POLES)

            # centered = (1 + tolerance(cart_pos, margin=2)) / 2
            var centered = (
                ONE
                + tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE](
                    cart_pos,
                    Scalar[DTYPE](0.0),
                    Scalar[DTYPE](0.0),
                    Scalar[DTYPE](2.0),
                )
            ) / Scalar[DTYPE](2.0)

            # small_control from the CLAMPED control, as on the CPU side.
            var ctrl = rebind[Scalar[DTYPE]](actions[env, 0])
            if ctrl > ONE:
                ctrl = ONE
            elif ctrl < -ONE:
                ctrl = -ONE
            var small_control = (
                Scalar[DTYPE](4.0)
                + tolerance[SIGMOID_QUADRATIC, 0.0, DTYPE](
                    ctrl,
                    Scalar[DTYPE](0.0),
                    Scalar[DTYPE](0.0),
                    Scalar[DTYPE](1.0),
                )
            ) / Scalar[DTYPE](5.0)

            # small_velocity over qvel[1:] — the hinges, excluding the slider.
            var min_sv = ONE
            for i in range(1, NV_F):
                var sv = tolerance[
                    SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
                ](
                    rebind[Scalar[DTYPE]](qvel[env, i]),
                    Scalar[DTYPE](0.0),
                    Scalar[DTYPE](0.0),
                    Scalar[DTYPE](5.0),
                )
                if sv < min_sv:
                    min_sv = sv
            var small_velocity = (ONE + min_sv) / Scalar[DTYPE](2.0)

            return (upright * small_control * small_velocity * centered, False)

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
        """`Balance.initialize_episode` — mirrors `custom_reset_cpu`.

        ⚠ The DRAW ORDER matters for nothing here (the two paths use different
        RNGs and the parity gate injects a fixed state), but the DISTRIBUTION
        must match term for term — that is what the reference specifies and all
        the port can honour. Same split ball_in_cup and point_mass-hard used.
        """
        var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)

        # Local wrapper: `standard_normal` takes the two uniforms rather than
        # the generator (no PhiloxRandom in its signature — see gpu_reset).
        @always_inline
        def _normal[D: DType](mut g: PhiloxRandom) {imm} -> Scalar[D]:
            var p = g.step_uniform()
            return standard_normal[D](Scalar[D](p[0]), Scalar[D](p[1]))

        comptime if Self.SWING_UP:
            qpos[env, 0] = Scalar[DTYPE](0.01) * _normal[DTYPE](rng)
            qpos[env, 1] = Scalar[DTYPE](pi) + Scalar[DTYPE](
                0.01
            ) * _normal[DTYPE](rng)
            for i in range(2, NQ_F):
                qpos[env, i] = Scalar[DTYPE](0.1) * _normal[DTYPE](rng)
        else:
            var u = rng.step_uniform()
            qpos[env, 0] = (
                Scalar[DTYPE](u[0]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
            ) * Scalar[DTYPE](0.1)
            for i in range(1, NQ_F):
                # A fresh block per joint: `step_uniform` yields 4 lanes and
                # reusing one block across joints would tie a 3-pole model's
                # qpos[3] to its qpos[1] through the same Philox counter.
                var uu = rng.step_uniform()
                qpos[env, i] = (
                    Scalar[DTYPE](uu[0]) * Scalar[DTYPE](2.0)
                    - Scalar[DTYPE](1.0)
                ) * Scalar[DTYPE](0.034)

        for i in range(NV_F):
            qvel[env, i] = Scalar[DTYPE](0.01) * _normal[DTYPE](rng)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.01
