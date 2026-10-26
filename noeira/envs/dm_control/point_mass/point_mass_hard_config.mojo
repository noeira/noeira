"""`dm_control` `point_mass-hard` task config — port of `suite/point_mass.py`.

Identical to `easy` in observation, reward, episode length and model. The ONE
difference is `initialize_episode`, which additionally randomizes the mapping
from controls to joints (`PointMass.__init__(randomize_gains=True)`):

    dir1 = random.randn(2); dir1 /= norm(dir1)
    parallel = True
    while parallel:                       # reject a near-parallel second axis
      dir2 = random.randn(2); dir2 /= norm(dir2)
      parallel = abs(dot(dir1, dir2)) > 0.9
    physics.model.wrap_prm[[0, 1]] = dir1
    physics.model.wrap_prm[[2, 3]] = dir2

so each control drives a random linear combination of `root_x`/`root_y`, and
the policy has to infer the mixing from experience. `easy`'s identity mixing is
the special case.

WHERE THE COEFS LIVE. `wrap_prm` for a fixed tendon is the per-joint `coef` of
each `<joint>` wrap, in declaration order: entries 0,1 are `t1`'s coefs on
(root_x, root_y) and 2,3 are `t2`'s. Here that is
`Model.tendons[t, TENDON_IDX_COEF_0/1]` — the RUNTIME records, written by
`custom_reset_model_cpu` below.

WHY THIS CONFIG ALSO OWNS ACTUATION. `MODEL_DEF.apply_actions` reads its
transmission from the COMPTIME tables (`_acd.motor_trn_coef`), baked from the
XML at build time and therefore blind to per-episode writes. Inheriting it
would silently keep the identity mixing and turn `hard` back into `easy` — a
task that trains perfectly well and is simply the wrong one. So the config
returns True from `custom_apply_actions_cpu` and redoes the (short) motor
transmission against the runtime records instead.

Everything else is `easy`'s, reused rather than restated: the joint randomizer,
the sparse-ish `tolerance` reward and the timestep all come from
`DMPointMassConfig`.
"""

from std.random import random_float64
from std.math import sqrt, log, cos, pi, abs

from std.collections import Array
from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    ACT_IDX_GEAR,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    MODEL_JOINT_SIZE,
    JOINT_IDX_DOF_ADR,
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    META_IDX_TASK_PARAM_0,
)

from ..dtype_math import sqrt_dt
from ..gpu_reset import (
    reset_seed,
    standard_normal,
    randomize_limited_and_rotational_joints_gpu,
)

from .point_mass_xml import DMPointMassModel
from .point_mass_config import DMPointMassConfig

from ...phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# `abs(np.dot(dir1, dir2)) > 0.9` — the rejection threshold on the cosine
# between the two actuation directions (both are unit vectors).
comptime PARALLEL_COS: Float64 = 0.9

# Bail-out for the rejection loop. The accepted set is |cos| <= .9, i.e. ~2/pi
# * arccos(.9) ~= 71% of the circle, so 64 draws miss it with probability
# ~1e-35 — the guard is against a wedged RNG, not against bad luck.
comptime MAX_REJECT_TRIES: Int = 64


def _randn() -> Float64:
    """One standard normal, Box-Muller. `np.random.randn` in the reference.

    Only the cosine branch is kept: the sine partner would have to be cached
    across calls to be used, and this is called four times per episode.
    """
    var u1 = random_float64()
    # log(0) is -inf; nudge off the open end rather than resampling, which
    # would bias nothing but reads as if it might.
    if u1 <= 0.0:
        u1 = 1e-12
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


# XORed into the reset key for the MIXING draw, so it is a different stream
# from the joint randomizer's — which consumes a joint-count-dependent number
# of Philox blocks.
comptime _MIXING_STREAM: UInt64 = 0xBF58476D1CE4E5B9

struct DMPointMassHardConfig(Phyics3dEnvConfig):
    # === Physics === (identical to easy — one XML, one timestep, one horizon)
    comptime FRAME_SKIP: Int = DMPointMassConfig.FRAME_SKIP
    comptime MAX_STEPS: Int = DMPointMassConfig.MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = DMPointMassConfig.INTEGRATOR_WS_EXTRA
    comptime SYNC_FK_AFTER_STEP: Bool = DMPointMassConfig.SYNC_FK_AFTER_STEP
    comptime INTEGRATOR: StaticString = DMPointMassConfig.INTEGRATOR

    # === CPU: Observation === (`easy`'s: qpos then qvel, both whole)
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
        return DMPointMassConfig.custom_extract_obs_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites, act, obs
        )

    # === CPU: Reset (state) === (`easy`'s joint randomizer)
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        DMPointMassConfig.custom_reset_cpu(
            d, m_bodies, m_joints, m_geoms, m_sites
        )

    # === CPU: Reset (model) — THE task difference ===
    @staticmethod
    def custom_reset_model_cpu[
        DTYPE: DType,
    ](
        mut m_bodies: List[Scalar[DTYPE]],
        mut m_joints: List[Scalar[DTYPE]],
        mut m_geoms: List[Scalar[DTYPE]],
        mut m_sites: List[Scalar[DTYPE]],
        mut m_tendons: List[Scalar[DTYPE]],
    ):
        """Randomize the control-to-joint mixing: two unit directions that are
        not too parallel, one per tendon.

        The reference's own draw sequence is NOT reproducible here (different
        RNG), so a rollout cannot be matched episode-for-episode against
        dm_control. The parity test writes the same coefs into both engines and
        gates the physics instead — the same split the ball_in_cup test uses
        for its rejection-sampled qpos.
        """
        var d1x = _randn()
        var d1y = _randn()
        var n1 = sqrt(d1x * d1x + d1y * d1y)
        d1x /= n1
        d1y /= n1

        var d2x = 0.0
        var d2y = 1.0
        for _ in range(MAX_REJECT_TRIES):
            var cx = _randn()
            var cy = _randn()
            var n2 = sqrt(cx * cx + cy * cy)
            cx /= n2
            cy /= n2
            if abs(d1x * cx + d1y * cy) <= PARALLEL_COS:
                d2x = cx
                d2y = cy
                break

        # `wrap_prm[[0, 1]] = dir1` / `wrap_prm[[2, 3]] = dir2`: the two joint
        # wraps of tendon 0, then those of tendon 1, in declaration order.
        var t0 = 0 * MODEL_TENDON_SIZE
        m_tendons[t0 + TENDON_IDX_COEF_0 + 0] = Scalar[DTYPE](d1x)
        m_tendons[t0 + TENDON_IDX_COEF_0 + 1] = Scalar[DTYPE](d1y)
        var t1 = 1 * MODEL_TENDON_SIZE
        m_tendons[t1 + TENDON_IDX_COEF_0 + 0] = Scalar[DTYPE](d2x)
        m_tendons[t1 + TENDON_IDX_COEF_0 + 1] = Scalar[DTYPE](d2y)

    # === CPU: Actuation against the RUNTIME tendon records ===
    @staticmethod
    def custom_apply_actions_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        m_tendons: List[Scalar[DTYPE]],
        m_actuators: List[Scalar[DTYPE]],
        m_act_tendons: List[Scalar[DTYPE]],
        actions: List[Float64],
    ) -> Bool:
        """`qfrc[dof] += gear * coef * ctrl`, with `coef` read per episode.

        The same arithmetic as `ModelDefFromXML.apply_actions`' motor branch
        (both are `moment^T force` over the transmission, and a `<motor>`'s
        force is just its clamped ctrl), differing only in reading `coef` and
        the DOF address from the runtime records rather than the comptime
        tables. Gear and ctrlrange still come from the comptime tables: `hard`
        randomizes the mixing only, so those are as constant as the XML.

        ACTUATOR a DRIVES TENDON a. The model declares one `<motor tendon=>`
        per `<fixed>`, in the same order, and nothing at runtime records that
        pairing — the parity test pins it by checking each actuator's comptime
        transmission against its tendon's XML coefs.
        """
        # ⚠ GEAR AND CTRLRANGE COME FROM `m_actuators` NOW, not from three
        # materialized comptime arrays. Same values, same clamp, one source —
        # and the hoisting the old code needed (a comptime `Array` cannot be
        # indexed by a runtime value, so each had to be copied once per call)
        # is gone with them.
        for i in range(D.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)

        comptime nact = DMPointMassModel.nact
        for a in range(nact):
            if a >= len(actions):
                break
            var ao = a * MODEL_ACTUATOR_SIZE
            var ctrl = actions[a]
            # ⚠ UNCONDITIONAL, matching what this hook has always done.
            # `ModelDefFromXML.apply_actions` gates its clamp on
            # `ctrllimited`; point_mass declares `ctrlrange="-1 1"` on both
            # actuators, so the two agree here — but if this model ever grew an
            # unlimited actuator the gate would have to come with it.
            var c_max = Float64(m_actuators[ao + ACT_IDX_CTRL_MAX])
            var c_min = Float64(m_actuators[ao + ACT_IDX_CTRL_MIN])
            if ctrl > c_max:
                ctrl = c_max
            elif ctrl < c_min:
                ctrl = c_min
            var gear = Float64(m_actuators[ao + ACT_IDX_GEAR])

            var to = a * MODEL_TENDON_SIZE
            var njnt = Int(m_tendons[to + TENDON_IDX_NUM_JOINTS])
            for k in range(njnt):
                var jid = Int(m_tendons[to + TENDON_IDX_JOINT_0 + k])
                var coef = Float64(m_tendons[to + TENDON_IDX_COEF_0 + k])
                var dadr = Int(
                    m_joints[jid * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR]
                )
                if dadr < 0 or dadr >= D.NV:
                    continue
                d.qfrc.data[dadr] += Scalar[DTYPE](gear * coef * ctrl)
        return True

    # === CPU: Reward === (`easy`'s, unchanged)
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
        return DMPointMassConfig.compute_reward_and_done_cpu(
            d,
            m_bodies,
            m_joints,
            m_geoms,
            m_sites,
            prev_x,
            actions,
            step_count,
            frame_skip,
        )

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMPointMassModel.TIMESTEP)

    # ── GPU hooks ────────────────────────────────────────────────────────
    comptime HAS_GPU_HOOKS: Bool = True
    # The transmission is redrawn every episode, so the comptime actuator
    # tables are wrong by construction — see `custom_apply_actions_gpu`.
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = True

    # ⚠ THE `meta` TASK_PARAM SLOTS ARE ONLY ENOUGH BECAUSE NOTHING ELSE
    # READS THESE TENDONS. A `limited` tendon emits a solver limit row, a
    # spring-loaded one a passive force, and an `<equality><tendon>` a
    # constraint — all built from `Model.tendons`, which is SHARED across the
    # batch and which this config no longer keeps in sync. Randomizing a
    # tendon with any of those needs real per-env model storage (G4), not
    # this. point_mass's two tendons carry none; the assert lives inside
    # `custom_apply_actions_gpu` because a `comptime for` must be contained
    # in a function, not at struct scope.

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
        """`easy`'s joint randomizer, then the control-to-joint mixing.

        The mixing is TWO UNIT DIRECTIONS that are not too parallel, stored in
        the four `META_IDX_TASK_PARAM_*` slots as (d1x, d1y, d2x, d2y) — the
        per-env stand-in for `model.wrap_prm`, which cannot be per-lane
        because `Model` is shared. `custom_apply_actions_gpu` reads them back.

        ⚠ THE REJECTION LOOP IS FIXED-TRIP, unlike the CPU's `while`. Every
        lane runs the same kernel, so an early break buys nothing and a
        data-dependent trip count would diverge the warp; all lanes run the
        bound and keep the FIRST accepted draw. Same acceptance region.

        ⚠ FALLING BACK TO THE IDENTITY MIXING WOULD BE A SILENT `easy`. The
        fallback here is (0, 1) against a d1 that is a unit vector — always
        accepted unless d1 is within 26 degrees of the y-axis, and even then
        it is a legitimate non-identity mixing rather than the XML's. With 64
        tries the fallback is unreachable at ~1e-35.
        """
        randomize_limited_and_rotational_joints_gpu[
            DTYPE, BATCH_SIZE, NQ, NJOINT, RANDOMIZE_UNLIMITED_HINGES=True
        ](qpos, joints, env, seed)

        var rng = PhiloxRandom(
            seed=reset_seed(env, seed) ^ _MIXING_STREAM, offset=0
        )
        var b = rng.step_uniform()
        var d1x = standard_normal[DTYPE](
            Scalar[DTYPE](b[0]), Scalar[DTYPE](b[1])
        )
        var d1y = standard_normal[DTYPE](
            Scalar[DTYPE](b[2]), Scalar[DTYPE](b[3])
        )
        var n1 = sqrt_dt[DTYPE](d1x * d1x + d1y * d1y)
        if n1 < Scalar[DTYPE](1e-12):
            d1x = Scalar[DTYPE](1)
            d1y = Scalar[DTYPE](0)
            n1 = Scalar[DTYPE](1)
        d1x /= n1
        d1y /= n1

        var d2x = Scalar[DTYPE](0)
        var d2y = Scalar[DTYPE](1)
        var found = False
        for _t in range(MAX_REJECT_TRIES):
            var c = rng.step_uniform()
            var cx = standard_normal[DTYPE](
                Scalar[DTYPE](c[0]), Scalar[DTYPE](c[1])
            )
            var cy = standard_normal[DTYPE](
                Scalar[DTYPE](c[2]), Scalar[DTYPE](c[3])
            )
            var n2 = sqrt_dt[DTYPE](cx * cx + cy * cy)
            if n2 < Scalar[DTYPE](1e-12):
                continue
            cx /= n2
            cy /= n2
            var dot = d1x * cx + d1y * cy
            var adot = -dot if dot < Scalar[DTYPE](0) else dot
            if adot <= Scalar[DTYPE](PARALLEL_COS) and not found:
                d2x = cx
                d2y = cy
                found = True

        meta[env, META_IDX_TASK_PARAM_0] = d1x
        meta[env, META_IDX_TASK_PARAM_0 + 1] = d1y
        meta[env, META_IDX_TASK_PARAM_0 + 2] = d2x
        meta[env, META_IDX_TASK_PARAM_0 + 3] = d2y

    @always_inline
    @staticmethod
    def custom_apply_actions_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        NTENDON_F: Int,
        ACTION_DIM: Int,
        NA_F: Int,
        NACT_F: Int,
    ](
        qfrc: Tile[DTYPE, BATCH_SIZE, NV],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
        tendons: Tile[DTYPE, NTENDON_F, MODEL_TENDON_SIZE],
        acts: Tile[DTYPE, NACT_F * MODEL_ACTUATOR_SIZE],
        act_tendons: Tile[DTYPE, NTENDON_F * MODEL_ACT_TENDON_SIZE],
        env: Int,
    ):
        """`qfrc[dof] += gear * coef * ctrl`, with `coef` read PER LANE.

        The same arithmetic as `custom_apply_actions_cpu`, and for the same
        reason: `MODEL_DEF.apply_actions_kernel_gpu` reads its transmission
        from the COMPTIME tables, baked from the XML and therefore blind to
        the per-episode mixing. Inheriting it would silently keep the identity
        coefficients and turn `hard` back into `easy` — a task that trains
        perfectly well and is simply the wrong one.

        Only the COEFS come from `meta`; the joint ids and the wrap count
        still come from the RUNTIME tendon records, and gear/ctrlrange from
        the comptime tables (`hard` randomizes the mixing only, so those are
        as constant as the XML).

        ACTUATOR a DRIVES TENDON a — the model declares one `<motor tendon=>`
        per `<fixed>`, in the same order, and nothing at runtime records that
        pairing. The parity test pins it.
        """
        # ⚠⚠ THE "A TENDON GREW A SPRING" GUARD MOVED TO
        # `test_point_mass_hard_vs_dm_control`, AND THAT IS A WEAKENING.
        # It was a `comptime assert` over `_acd.tendon_stiffness`; the data is
        # a runtime record now and no `comptime assert` can read one. This
        # hook cannot raise and runs per lane per SUBSTEP, so it is the wrong
        # place for a model-invariant check, and no once-per-build CONFIG hook
        # receives `SpecFields`. What it guards is still real: the per-episode
        # coefs live in `d.meta`, NOT in the tendon records, so a spring would
        # be built from the XML's coefs while actuation used the drawn ones —
        # and with `HAS_CUSTOM_ACTUATION_GPU` the spring is not applied at all,
        # because this hook REPLACES `apply_actions_kernel_gpu` entirely.
        # It is now a gate assertion rather than a build error.

        for i in range(NV):
            qfrc[env, i] = Scalar[DTYPE](0)

        comptime nact = DMPointMassModel.nact
        comptime for a in range(nact):
            comptime if a < ACTION_DIM:
                # ⚠ RUNTIME READS OFF `acts`, the same operand
                # `apply_actions_kernel_gpu` reads. The `comptime` bindings
                # these replace were baked literals off `_acd`.
                var ao = a * MODEL_ACTUATOR_SIZE
                var c_max = rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MAX])
                var c_min = rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MIN])
                var gear = Float64(
                    rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_GEAR])
                )

                var ctrl = rebind[Scalar[DTYPE]](actions[env, a])
                if ctrl > c_max:
                    ctrl = c_max
                elif ctrl < c_min:
                    ctrl = c_min

                comptime if a < NTENDON_F:
                    var njnt = Int(
                        rebind[Scalar[DTYPE]](
                            tendons[a, TENDON_IDX_NUM_JOINTS]
                        )
                    )
                    for k in range(njnt):
                        var jid = Int(
                            rebind[Scalar[DTYPE]](
                                tendons[a, TENDON_IDX_JOINT_0 + k]
                            )
                        )
                        # ⚠ THE COEF, AND ONLY THE COEF, COMES FROM `meta`.
                        # Slot layout is (t0.c0, t0.c1, t1.c0, t1.c1) —
                        # `wrap_prm[[0,1]]` then `[[2,3]]`, the reference's own
                        # order.
                        var coef = rebind[Scalar[DTYPE]](
                            meta[env, META_IDX_TASK_PARAM_0 + a * 2 + k]
                        )
                        var dadr = Int(
                            rebind[Scalar[DTYPE]](
                                joints[jid, JOINT_IDX_DOF_ADR]
                            )
                        )
                        if dadr < 0 or dadr >= NV:
                            continue
                        qfrc[env, dadr] = rebind[Scalar[DTYPE]](
                            qfrc[env, dadr]
                        ) + Scalar[DTYPE](gear) * coef * ctrl

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
        """`easy`'s: qpos then qvel, both whole."""
        return DMPointMassConfig.custom_extract_obs_gpu[
            DTYPE, BATCH_SIZE, NQ, NV, NBODY, OBS_DIM, SITE_DIM, MC_F,
            NSITE_F, NGEOM_F, NA_F,
        ](
            qpos, qvel, xpos, xquat, xvel, bodies, site_xpos, contacts,
            sites, geoms, meta, obs, xipos, xangvel, cvel, cacc, cfrc_int,
            subtree_com, site_xpos_acc, xquat_acc, act, env,
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
        """`easy`'s, unchanged."""
        return DMPointMassConfig.compute_reward_and_done_gpu[
            DTYPE, BATCH_SIZE, NQ, NV, NBODY, ACTION_DIM, SITE_DIM, MC_F,
            NSITE_F, NGEOM_F, NA_F,
        ](
            qpos, qvel, xpos, xipos, xquat, xvel, bodies, site_xpos,
            contacts, sites, geoms, cfrc_ext, cvel, meta, curriculum,
            actions, xangvel, cacc, cfrc_int, subtree_com, site_xpos_acc,
            xquat_acc, act, env, step_count, frame_skip, timestep,
        )
