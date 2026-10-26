"""GPU-side reset randomizers for the dm_control suite.

⚠ COMPILED AND GATED as of 2026-08-06, via cartpole (`standard_normal`) and
cheetah/walker (`randomize_limited_and_rotational_joints_gpu`). It did NOT
compile the first time it was instantiated — a generic body is not type-checked
until something binds its parameters, and `import` is not a compile gate
(`feedback_ungated_generic_is_uncompiled_code`). Two errors surfaced then:
naming `PhiloxRandom` as a RETURN TYPE, and the `log`/`cos`
`is_floating_point()` evidence. Both shaped the API below — keep generator
types OUT of signatures. Anything added here is a DRAFT until a config
instantiates it.

Port of `dm_control/suite/utils/randomizers.py` for the batched GPU path.
The CPU counterpart is open-coded in each config's `custom_reset_cpu` (it can
reach `std.random.random_float64()` and iterate `m_joints` as a `List`); a
kernel can do neither, so the shared form lives here.

Called from `Phyics3dEnvConfig.init_qpos_gpu`, which since 2026-08-06 receives
the `joints` model tensor and a `seed` for exactly this (G10 step 4 — see
docs/DM_CONTROL_GPU_TRAINING_G10.md §6).

⚠ WHAT "LIMITED" MEANS HERE. The reference asks MuJoCo for `jnt_limited`; we
have no such column, so — exactly as every `custom_reset_cpu` in the suite
does, and as `constraints/limits.mojo` does when it decides whether to emit a
limit row — a range beyond +-1e9 is read as UNLIMITED. Keeping the two paths on
the same rule is what makes the CPU gate meaningful for the GPU path.

FREE JOINTS: covered as of 2026-08-06, when humanoid needed it (`RANDOMIZE_FREE_QUAT`).
The three LINEAR dofs are left alone and qpos[adr+3 .. adr+6] gets a random
quaternion. ⚠ ONE REFERENCE QUIRK IS REPRODUCED DELIBERATELY, exactly as the
CPU hook does: `randomizers.py` says "sampled uniformly on the unit 3-sphere",
but its free-joint branch calls `random.rand(4)` — uniform on [0,1)^4, NOT
`randn(4)` as the ball branch two lines up does. So every component is
non-negative and the orientations sit in one orthant, nowhere near uniform on
SO(3). We match the CODE, not the docstring: an agent trained against
dm_control sees this distribution.

⚠ STILL NOT COVERED: `mjJNT_BALL`. Nothing ported has one. Add it with the
first domain that does, and gate it — an ungated branch here is uncompiled
code (`feedback_ungated_generic_is_uncompiled_code`), which this module has
already been bitten by once.
"""

from std.math import pi, sqrt, log, cos
from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_FREE
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from noeira.core.tiles import Tile


# A range wider than this counts as unlimited. Same threshold as
# `constraints/limits.mojo` and as every `custom_reset_cpu` in the suite.
comptime UNLIMITED: Float64 = 1e9

# Philox key mixing. `seed` is shared with `MODEL_DEF.reset_env_gpu`, which
# derives its own key as `seed * 2654435761 + env * 12345`. A DIFFERENT mix is
# used here so the joint angles are not correlated with the qpos/qvel reset
# noise drawn from the same seed — they are independent draws in the reference.
comptime _KEY_A: Int = 1103515245
comptime _KEY_B: Int = 98765431


@always_inline
def reset_seed(env: Int, seed: Int) -> UInt64:
    """The Philox key every reset hook here draws from: `PhiloxRandom(
    seed=reset_seed(env, seed), offset=0)`.

    ⚠ Deliberately a DIFFERENT key mix from `MODEL_DEF.reset_env_gpu`
    (`seed * 2654435761 + env * 12345`), which runs first and writes the qpos0
    noise. Sharing the mix would correlate a task's episode-init draw with that
    noise; the reference treats them as independent.

    ⚠ Returns the KEY, not a `PhiloxRandom`: `std.random.philox.Random` is
    parameterized, so naming it as a return type needs its parameters bound
    ("'Random[_]' is not concrete"). Constructing at the call site infers them.
    """
    return UInt64(seed * _KEY_A + env * _KEY_B)


@always_inline
def standard_normal[
    DTYPE: DType
](u1_raw: Scalar[DTYPE], u2: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """One N(0,1) draw by Box-Muller from two uniforms in [0,1).

    Takes the uniforms rather than the generator for the same reason
    `reset_seed` returns a key — no `PhiloxRandom` in the signature. Callers do:

        var p = rng.step_uniform()
        var z = standard_normal[DTYPE](Scalar[DTYPE](p[0]), Scalar[DTYPE](p[1]))

    Matches `cartpole_config._randn` (the CPU side) formula for formula. The
    sin half of the pair is DISCARDED rather than cached: the reference calls
    `numpy.random.randn()` per element, and keeping it would make a lane's k-th
    draw depend on the parity of k — a different distribution to reason about,
    for no measurable gain at reset.

    ⚠ Float32 `log` underflows far earlier than Float64's, hence the clamp; the
    CPU version's 1e-300 guard is meaningless at float32.
    """
    comptime if DTYPE == DType.float32:
        return rebind[Scalar[DTYPE]](
            _standard_normal_impl[DType.float32](
                rebind[Float32](u1_raw), rebind[Float32](u2)
            )
        )
    elif DTYPE == DType.float64:
        return rebind[Scalar[DTYPE]](
            _standard_normal_impl[DType.float64](
                rebind[Float64](u1_raw), rebind[Float64](u2)
            )
        )
    else:
        comptime assert False, (
            "gpu_reset.standard_normal: only float32 / float64 are supported."
        )


@always_inline
def _standard_normal_impl[
    DTYPE: DType
](u1_raw: Scalar[DTYPE], u2: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """The Box-Muller body. Reached only through `standard_normal`, which binds
    `DTYPE` to a concrete float type first — `std.math`'s `log`/`cos` require
    `is_floating_point()` evidence that an unconstrained trait-method `DTYPE`
    cannot supply. Same pattern as `dm_control/rewards.sigmoids`; see
    `feedback_where_clause_cannot_cross_trait_boundary`."""
    comptime U1_MIN = Scalar[DTYPE](1e-7)
    var u1 = u1_raw
    if u1 < U1_MIN:
        u1 = U1_MIN
    return sqrt(Scalar[DTYPE](-2.0) * log(u1)) * cos(
        Scalar[DTYPE](2.0 * pi) * u2
    )


@always_inline
def randomize_limited_and_rotational_joints_gpu[
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ: Int,
    NJOINT: Int,
    RANDOMIZE_UNLIMITED_HINGES: Bool = True,
    RANDOMIZE_FREE_QUAT: Bool = False,
](
    qpos: Tile[DTYPE, BATCH_SIZE, NQ],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    env: Int,
    seed: Int,
):
    """`randomizers.randomize_limited_and_rotational_joints` for one env lane.

    Limited hinges/slides are drawn uniformly inside their range. UNLIMITED
    hinges are drawn uniformly in [-pi, pi); unlimited slides are left alone
    (the reference has no sensible range for them and skips them).

    `RANDOMIZE_FREE_QUAT=True` additionally randomizes a free root's
    orientation (humanoid, humanoid_cmu); it is off by default because a
    domain without a free joint would pay a dead branch, and because getting a
    free root's pose randomized when the reference does not would change the
    initial-state distribution silently.

    Set `RANDOMIZE_UNLIMITED_HINGES=False` for the domains whose reference only
    randomizes the limited joints — cheetah does exactly that
    (`suite/cheetah.py` walks `physics.model.jnt_range` and touches nothing
    else), so passing True there would randomize its three unlimited root dofs
    and change the initial state distribution.

    All arithmetic is `DTYPE`: this runs inside a Metal kernel, where there is
    no `double`.
    """
    # One Philox stream per (seed, env). Philox yields 4 uniforms per step, so
    # draw in blocks of 4 and index into the block — the same shape
    # `MODEL_DEF.reset_env_gpu` uses.
    var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)
    var block = rng.step_uniform()
    var slot = 0

    for j in range(NJOINT):
        var jtype = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])
        )

        comptime if RANDOMIZE_FREE_QUAT:
            if jtype == JNT_FREE:
                var adr_f = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
                )
                # `random.rand(4)` — see the module docstring on why this is
                # NOT `randn(4)`. Free-joint qpos is [x, y, z, w, x, y, z]:
                # the three LINEAR dofs first, then a W-FIRST quaternion.
                # (Only `Data.xquat` is xyzw. Getting this backwards yields a
                # valid-looking unit quaternion and a wrong initial pose.)
                var qb = rng.step_uniform()
                var q0 = Scalar[DTYPE](qb[0])
                var q1 = Scalar[DTYPE](qb[1])
                var q2 = Scalar[DTYPE](qb[2])
                var q3 = Scalar[DTYPE](qb[3])
                var nn = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
                if nn < Scalar[DTYPE](1e-12):
                    q0 = Scalar[DTYPE](1)
                    q1 = Scalar[DTYPE](0)
                    q2 = Scalar[DTYPE](0)
                    q3 = Scalar[DTYPE](0)
                    nn = Scalar[DTYPE](1)
                qpos[env, adr_f + 3] = q0 / nn
                qpos[env, adr_f + 4] = q1 / nn
                qpos[env, adr_f + 5] = q2 / nn
                qpos[env, adr_f + 6] = q3 / nn
                continue

        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue

        var lo = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
        var hi = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
        var limited = (
            lo > Scalar[DTYPE](-UNLIMITED) and hi < Scalar[DTYPE](UNLIMITED)
        )

        # ⚠ The draw is consumed for every hinge/slide the reference would
        # visit, INCLUDING the unlimited slides it then skips. Doing it the
        # other way would make a model's random stream depend on how many
        # unlimited slides precede a given joint, so two models that differ
        # only in an unlimited root would disagree on every later joint.
        if slot == 4:
            block = rng.step_uniform()
            slot = 0
        var u = Scalar[DTYPE](block[slot])
        slot += 1

        var adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        if limited:
            qpos[env, adr] = lo + u * (hi - lo)
        elif jtype == JNT_HINGE:
            comptime if RANDOMIZE_UNLIMITED_HINGES:
                qpos[env, adr] = (
                    Scalar[DTYPE](-pi) + u * Scalar[DTYPE](2.0 * pi)
                )
