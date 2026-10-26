"""One reach-task config, shared by SO-ARM100 and SO-ARM101.

`docs/SO_ARM101_PORT_ASSESSMENT.md` §1 concludes "one env, two model files".
This file is what makes that structural rather than aspirational: both arms
have the same topology (nbody 8 / njnt 6 / nq 6 / nu 6) and — measured — the
same body indices for the end-effector (7) and the appended target (8), so a
single `Phyics3dEnvConfig` parameterised by four numbers serves both.

    observation = [qpos(6), qvel(6), ee_xyz(3), target_xyz(3), ee_to_target(3)]
    action      = 6 joint POSITION TARGETS, in RADIANS
    reward      = tolerance(||ee - target||, (0, TARGET_RADIUS), margin)
    episode     = MAX_STEPS control steps, no early termination

⚠⚠ THE ACTION SPACE IS RADIANS, NOT [-1, 1]. Both models drive `<position>`
servos whose `ctrlrange` is the joint's own range, so `action_low_at(i)` /
`action_high_at(i)` differ per actuator (`Pitch` is [-3.32, 0.174]). A policy
emitting tanh-bounded actions MUST be rescaled by the caller. This is the exact
shape of the defect `c1ebf61e` fixed — an env that advertised one scalar pair
for six differently-ranged actuators — so it is stated here rather than left to
be discovered. `apply_actions` clamps to the per-actuator range, so a policy
that ignores this is silently squashed onto the first radian of travel.

⚠ THE TARGET IS SAMPLED IN THE ARM'S OWN AZIMUTH CONE. `AZ_CENTER` differs
between the two models because their base frames do — SO-100's arm extends
along -y at qpos 0, SO-101's along +x — and both `Rotation`/`shoulder_pan`
ranges are ±1.92 rad, not a full circle. Sampling a full circle would make a
third of every episode physically unreachable and read as "the policy plateaus".

GPU HOOKS LANDED 2026-08-25 — `HAS_GPU_HOOKS = True`, so the batched path
compiles and trains. They were written against
`dm_control/reacher/reacher_config.mojo`, the only other config in the tree
with GPU hooks AND a mocap target, and the note this paragraph replaces was
right that writing them blind is how `ungated_generic_is_uncompiled_code`
happens — so they are gated CPU-vs-GPU rather than merely compiled.

⚠ THE OBSERVATION ORDER IS THE CONTRACT. The batched trainer writes a
checkpoint the single-env eval loads, so `custom_extract_obs_cpu` and
`custom_extract_obs_gpu` must emit the same 21 values in the same order. A
permutation is a policy that works on one device and is nonsense on the other,
with no error anywhere — which is why the two are compared directly rather
than each being checked against a description.

⚠ RESET DOES NOT USE `<keyframe>`. Our parser ignores the section
(`docs/TODDLERBOT_PORT_PLAN.md` §4.6), so SO-100's `home` pose is baked into
`SO_ARM100_HOME` in `so_arm100_config.mojo` and written here by index. When
`<keyframe>` lands, this should read the parsed key and the baked copy should
be DELETED rather than kept alongside it.
"""

from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.math import pi, sqrt, sin, cos


from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0,
    META_IDX_TASK_PARAM_6,
    MODEL_JOINT_SIZE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
)

from ..phyics3d_env_config import Phyics3dEnvConfig
from ..dm_control.rewards import (
    tolerance,
    SIGMOID_GAUSSIAN,
    DEFAULT_VALUE_AT_MARGIN,
    SIGMOID_QUADRATIC,
)
from ..dm_control.gpu_reset import reset_seed
from ..dm_control.dtype_math import sin_dt, cos_dt
from noeira.core.tiles import Tile


struct SoArmReachConfig[
    # ⚠ NOT `NMESHV`. `Phyics3dEnvConfig.custom_reset_full_cpu` is a default
    # trait method carrying its own `NMESHV` parameter, and a struct parameter
    # of the same name shadows it — which the compiler rejects outright
    # ("name conflict between parameter 'NMESHV' in the default trait method
    # and a parameter in the struct"), taking both SO-ARM envs down with it.
    # Any parameter named after one of that trait's is a build break waiting
    # for the next hook to be added.
    NMESHVERT: Int,
    EE_BODY: Int,
    TARGET_BODY: Int,
    TIMESTEP: Float64,
    # Centre of the reachable azimuth cone, radians, measured from +x in the
    # world XY plane. See the module docstring.
    AZ_CENTER: Float64,
    # Half-width of that cone. Both arms' base-rotation range is 1.92 rad.
    AZ_HALF: Float64 = 1.92,
    # Target shell, metres from the base origin.
    #
    # ⚠ THE NEAR END IS WHERE THE POLICY FAILS, and it is a WORKSPACE fact,
    # not a training one. Measured 2026-08-26 on SO-101 with a trained
    # checkpoint, 24 episodes (`examples/so101/sac_so_arm101_reach_diag.mojo`):
    #
    #     r <  0.17 m   reached <=20 mm in  1 of  6 episodes   (17%)
    #     r >= 0.17 m   reached <=20 mm in 14 of 18 episodes   (78%)
    #
    # An FK sweep of 40 000 random poses inside the joint limits says why: the
    # jaw's reachable set thins out sharply toward the base. Poses landing in
    # the (radius, elevation) cell at r = 0.15 fall from 396 at low elevation
    # to 1 at el = 1.3, against 528..455 for the same elevations at r = 0.275.
    # A near target is reachable only through a narrow, tightly-folded band of
    # configurations — so it is not "hard to learn", it is nearly singular.
    #
    # R_MIN IS NOW 0.18, which drops that tail. ⚠ EVERY NUMBER MEASURED
    # AGAINST 0.15 IS NOW INCOMPARABLE, including the reach rates quoted
    # above and the ~46 untrained baseline in the eval script's bands: this
    # is a strictly easier target distribution, so a rate measured here must
    # not be compared with one from a checkpoint trained before it. Shared
    # with SO-100 deliberately — the two arms differ in their ROBOT, not in
    # what is asked of them, and that is what makes them comparable.
    #
    # ⚠ ELEVATION IS NOT THE SIGNAL, though a first 24-episode draw suggested
    # it was (2 of 9 above el 0.9 against 9 of 15 below). The second draw put
    # failures at el 0.18, 0.29 and 0.34 as readily as at 1.18 and 1.20. One
    # draw of 24 is not enough to split a second axis; the radius effect
    # survived both.
    R_MIN: Float64 = 0.18,
    R_MAX: Float64 = 0.30,
    # Elevation band, radians above the XY plane through the base.
    EL_MIN: Float64 = 0.17,
    EL_MAX: Float64 = 1.22,
    # Height of the base origin, added to every sampled target so nothing is
    # ever generated below the floor.
    BASE_Z: Float64 = 0.05,
    # `tolerance`'s upper bound: inside this the reward is exactly 1.
    TARGET_RADIUS: Float64 = 0.02,
    # ...and the width of the shaped falloff outside it. Unlike dm_control's
    # `reacher` (margin 0, a hard indicator) this one is SHAPED, because a
    # 6-DOF arm reaching a 2 cm ball has far too little chance of stumbling
    # onto a sparse reward.
    #
    # ⚠⚠ WAS 0.25 — TWELVE TIMES THE SUCCESS RADIUS — AND THAT WAS THE ROOT
    # DEFECT BEHIND EVERY SYMPTOM THIS TASK HAS SHOWN. At 0.25 the falloff is
    # nearly flat across the whole neighbourhood of the target, so:
    #
    #   * hovering 40 mm away scored 0.985/step, i.e. 492.7/500 for an arm
    #     that never touches the target;
    #   * closing the last 24 mm was worth 0.0006/step, against 0.20 for the
    #     stillness term — a 333x inversion of the task's own priorities;
    #   * ORBITING between 4 mm and 40 mm cost 0.015/step, so a policy that
    #     swings through the target and back out lost essentially nothing.
    #
    # That last one is not hypothetical: measured on hardware 2026-08-26, the
    # trained policy reached 3.9 mm and then orbited 8..43 mm for the rest of
    # the run, at 40% of steps inside the radius. Sim shows the same shape
    # (per-episode closest 4..9 mm, final 20..41 mm), so it transferred
    # faithfully — the behaviour is the REWARD's, not the hardware's.
    #
    # At 0.05 that same orbit costs 0.308/step, twenty times more and now the
    # dominant term. Resting on target finally beats passing through it.
    #
    #     dist    margin .25   margin .05
    #     20 mm       1.000        1.000
    #     30 mm       0.996        0.912
    #     40 mm       0.985        0.692
    #    100 mm       0.790        0.003
    #
    # ⚠ THE ORIGINAL REASON FOR 0.25 WAS EXPLORATION and it was a good one —
    # at 0.05 a target 100 mm away is worth 0.003/step, so a policy that
    # cannot already find the target learns nothing. FINE-TUNE FROM AN
    # EXISTING CHECKPOINT rather than training from scratch: exploration is
    # demonstrably solved (the arm reaches 4 mm), and the tighter margin only
    # sharpens the endgame. From scratch, keep 0.25 until the arm reaches,
    # then tighten and continue.
    REWARD_MARGIN: Float64 = 0.05,
    # ── the stillness term ────────────────────────────────────────────────
    #
    # ⚠⚠ WITHOUT THIS THE POLICY SHAKES, AND THE REWARD CANNOT SEE IT.
    # `tolerance`'s margin is 12x its success radius, so hovering 40 mm away
    # scores 0.985 and there is nothing left to distinguish a still arm from a
    # vibrating one. Measured on a trained checkpoint, over the 300 control
    # steps AFTER it has arrived: mean per-joint |qvel| **1.21 rad/s** (69
    # deg/s), peak **5.8 rad/s**, and the velocity REVERSES SIGN on 59% of
    # control steps. The commanded position moves ~500 servo ticks per joint
    # per step, reversing on 83%. On hardware that is continuous current
    # reversal through the gear train for the whole episode.
    #
    # ⚠ THE PENALTY IS ON `qvel`, NOT ON THE ACTION RATE, and the reason is
    # MARKOV rather than aesthetic: the previous action is NOT in the
    # observation, so an agent penalised for `a_t - a_{t-1}` is being charged
    # for a quantity it cannot observe, and can only learn a blurred average
    # of it. `qvel` IS in the observation (indices 6..11). Penalising the
    # action rate properly means widening the observation to carry the
    # previous action — a different and larger change, and the measurement
    # above says the joints really are moving, so this term has something
    # real to bite on. `deploy_reach_real.mojo`'s `SMOOTH` covers whatever
    # command chatter survives.
    #
    # ⚠ NOT ON THE ACTION MAGNITUDE EITHER — the gym `ctrl_cost` idiom is
    # actively WRONG here. These are `<position>` actuators, so an action IS a
    # joint angle: penalising its magnitude pulls the arm toward zero, which
    # is a POSE, not an effort.
    #
    # Shape is dm_control's, from `suite/humanoid.py`:
    #     small_control = (4 + small_control) / 5
    #     return small_control * stand_reward * move
    # i.e. a MULTIPLICATIVE term squashed into [FLOOR, 1]. Multiplicative
    # matters: the cost of moving scales with how well the arm is doing, so
    # travelling fast while still far away is nearly free and vibrating on
    # target is not — which is the distinction the task wants and a subtractive
    # penalty would flatten.
    #
    # Speed is the L2 norm over all DOF. `VEL_FREE` is the speed that costs
    # nothing; past it the quadratic sigmoid falls to zero at `VEL_FREE +
    # VEL_MARGIN`. Defaults put the measured shake (L2 ~3 rad/s) at the floor
    # and a still arm at 1.0.
    VEL_FREE: Float64 = 0.3,
    # ⚠⚠ 5.0, NOT 2.0, AND THIS IS WHY FOUR CHECKPOINTS SAT AT THE FLOOR.
    # `_still` used `SIGMOID_QUADRATIC` with value-at-margin 0, which has
    # COMPACT SUPPORT: exactly zero past `VEL_FREE + VEL_MARGIN` = 2.3 rad/s,
    # and exactly zero means ZERO DERIVATIVE. Measured on the trained policy:
    # L2 joint speed after arrival averages **2.483 rad/s** over a 0.96..4.42
    # range, so **69.5% of control steps sat in the flat region** and the term
    # said nothing at all about slowing down. Mean `calm` was 0.080.
    #
    # That is the same defect as the action clamp — a dead gradient band —
    # built by taking dm_control's `small_control` margin (1.0 on a [-1, 1]
    # control) without checking it against THIS arm's speed range.
    #
    #     speed   quad m=2   gauss m=5
    #      1.0      0.878       0.956
    #      2.0      0.278       0.766
    #      2.5      0.000       0.640     <- where the policy lives
    #      4.4      0.000       0.213
    VEL_MARGIN: Float64 = 5.0,
    # The floor of the multiplicative term — THE STRENGTH KNOB.
    #
    # ⚠⚠ 0.5, NOT dm_control's 0.8 ((4 + s) / 5), BECAUSE 0.8 WAS MEASURABLY
    # NOT WORTH TAKING. Across THREE trained checkpoints the policy has left
    # this term pinned at its floor — it pays the velocity penalty in full,
    # every step, every episode:
    #
    #     checkpoint            return/500   reach    still
    #     margin .25            400.6/500    ~1.00     0.81   <- floor
    #     margin .05            321.6/500     0.80     0.80   <- floor
    #     normalized actions    389.4/500     0.97     0.80   <- floor
    #
    # At 0.8 the whole term is worth 0.2/step against a reach term that swings
    # by more than that, so it never became the thing to optimise. At 0.5 an
    # arm that arrives and FREEZES scores 1.0/step where one that arrives and
    # shakes scores 0.5 — the difference is now the largest single term in the
    # reward.
    #
    # ⚠ AND IT DOES NOT REOPEN THE "STOP SHORT TO BE CALM" EXPLOIT that the
    # 0.25 margin created, because the margin is 0.05 now. Parking still at
    # 30 mm scores reach(30mm) = 0.912; arriving and freezing scores 1.000;
    # arriving and shaking scores 0.500. The ordering puts arrive-and-hold
    # first and, tellingly, prefers a still arm parked short over an orbiting
    # one on target — which is also the better hardware behaviour of the two.
    #
    # ⚠ MEASURED CONTEXT: the shake is REAL, not a command artefact — 1.06
    # rad/s mean joint speed with the velocity reversing on 85% of control
    # steps, while a CONSTANT command settles to exactly zero in under 50
    # steps and holds its pose to 0.03 deg. The behaviour is available; it has
    # simply never been worth more than the noise.
    VEL_FLOOR: Float64 = 0.5,
    # ── the action-rate term ──────────────────────────────────────────────
    # L2 change in the NORMALIZED action per control step. The action box is
    # [-1, 1] per joint, so an L2 rate of 0.05 is a gentle command and 1.0 is
    # a quarter of the full range on every joint at once, every step.
    #
    # ⚠ MEASURED: the raw greedy policy commands ~4438 servo ticks/step summed
    # over six joints, which is ~1.1 in normalized L2 — so `RATE_FREE = 0.05`
    # with a margin of 1.0 puts today's behaviour near the floor and a smooth
    # command near 1.0, with gradient across the whole span.
    RATE_FREE: Float64 = 0.05,
    RATE_MARGIN: Float64 = 1.0,
    # Same strength knob as `VEL_FLOOR`, and deliberately the same value: the
    # two terms multiply, so an arm that both drifts and chatters pays twice.
    RATE_FLOOR: Float64 = 0.5,
    # Uniform noise on each joint at reset, radians, clipped to joint range.
    RESET_NOISE: Float64 = 0.05,
    MAX_STEPS_P: Int = 500,
    # The reset pose, joint by joint. ⚠ SIX SCALARS RATHER THAN AN ARRAY
    # because a comptime array parameter is not expressible here; the arm
    # modules pass them by name so the call site still reads as a pose.
    #
    # ⚠⚠ THIS EXISTS BECAUSE `<keyframe>` IS UNPARSED. SO-100 ships a `home`
    # key of [0, -1.57, 1.57, 1.57, -1.57, 0] which is what the reference
    # resets to; `qpos0` is all zeros, i.e. the arm fully extended — a
    # DIFFERENT robot in a different posture, and nothing raises. SO-101 has
    # no keyframe at all (measured, nkey 0), so it legitimately passes zeros.
    HOME_0: Float64 = 0.0,
    HOME_1: Float64 = 0.0,
    HOME_2: Float64 = 0.0,
    HOME_3: Float64 = 0.0,
    HOME_4: Float64 = 0.0,
    HOME_5: Float64 = 0.0,
](Phyics3dEnvConfig):
    # === Physics ===
    # Both models declare `timestep="0.002"`. FRAME_SKIP 10 gives a 50 Hz
    # control rate against 500 Hz physics — the rate the real STS3215 bus
    # runs at, and the same 1:10 ratio ToddlerBot's walk env uses.
    comptime FRAME_SKIP: Int = 10
    comptime MAX_STEPS: Int = Self.MAX_STEPS_P
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # Neither model names an integrator, so MuJoCo's Euler default applies.
    # ⚠ `opt.integrator` is one of the fields the layer-1 gate compares, so
    # this is pinned by measurement rather than chosen.
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # The per-episode target is a mocap body — `Model` is not batched,
    # `Data.mocap_pos` is. Same mechanism `reacher` uses.
    comptime USES_MOCAP: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True
    # ⚠⚠ THE ACTION IS [-1, 1] PER JOINT, mapped affinely onto each joint's
    # own `ctrlrange` — see `Phyics3dEnvConfig.NORMALIZED_ACTIONS` for the
    # measurement that forced this. With `ACTION_SCALE = 2.0` against ranges
    # of 1.66..2.84 the trained policy commanded out-of-range poses on 24% to
    # 100% of steps and sat at the tanh rail up to 49% of the time; past the
    # clamp the gradient is zero, which is why three reward shapes in a row
    # produced the same shaking arm.
    #
    # ⚠ `action_scale` MUST NOW BE 1.0 in every script that builds an agent
    # for this env — trainer, eval, viewer and the hardware deploy. A scale of
    # 2.0 here would map [-2, 2] onto the range and put the useful band back
    # inside the rails, i.e. undo the fix while still looking configured.
    comptime NORMALIZED_ACTIONS: Bool = True
    # ⚠⚠ THE PREVIOUS ACTION IS IN THE OBSERVATION (indices 21..26) AND THE
    # REWARD PENALISES THE ACTION RATE. Five checkpoints reduced joint SPEED
    # and none reduced OSCILLATION, because a velocity penalty cannot see one:
    #
    #     ckpt                 mean |qvel|   reversals
    #     margin .25              1.214         59%
    #     margin .05              0.737         85%
    #     normalized actions      1.061         85%
    #     gaussian, margin 5      0.923         92%
    #
    # Amplitude fell and FREQUENCY ROSE every time — a small fast oscillation
    # has a lower mean speed than a large slow one, so the policy kept
    # satisfying the term the cheapest way available. `|a_t - a_{t-1}|` is the
    # quantity that actually distinguishes them.
    #
    # ⚠ AND IT ONLY WORKS BECAUSE THE PREVIOUS ACTION IS OBSERVABLE. Penalise
    # a quantity the agent cannot see and it can only learn a blurred average
    # of it; that is why this waited for the observation to widen rather than
    # being added to the reward alone.
    #
    # ⚠ The orbit is the POLICY's, not the deploy wrapper's — measured: with
    # NO smoothing and NO rate clamp the sim arm still swings 12.0 mm after
    # settling, against 14.5 mm through the full deploy loop.
    # ⚠ MEASURED OUTCOME (2026-08-27), the sixth attempt at the shake and the
    # first that moved it. On the REAL ARM, one central target, 10 s:
    #
    #                       before          after
    #     steps inside 20mm   32%            82%
    #     final error         31.0 mm         9.7 mm
    #     the trace          9..44 mm orbit  9.1/10.6/10.8/10.7/10.9/9.5/11.6
    #
    # It arrives and HOLDS to about +-1 mm where it used to swing +-20.
    #
    # ⚠ AND IT COST COVERAGE AT THE NEAR EDGE. Over the full target shell the
    # successes got much tighter (finals of 9..20 mm against a 14.3 mm mean
    # before) while two of 24 became outright failures — r=0.180 stopping at
    # 43 mm and r=0.193 never approaching at all, 176 mm. The aggregate
    # 15/24 is those two dragging a distribution that is otherwise better.
    # A near-edge coverage problem is NOT the oscillation problem and wants
    # its own diagnosis rather than another reward term.
    comptime RECORD_PREV_ACTION: Bool = True

    # ⚠⚠ NOT A SIZE HINT. Zero silently disables every mesh contact — both
    # narrow phases guard their mesh branch on `NMESH_VERTS > 0`. The value is
    # the measured convex-hull vertex total, NOT the raw STL count.
    comptime NMESH_VERTS: Int = Self.NMESHVERT

    @staticmethod
    def _home(i: Int) -> Float64:
        """The reset pose, by qpos address.

        ⚠ A CHAIN, NOT AN `Array`. `Array[Float64, 6](a, b, ...)`
        does not construct under Mojo 1.0 (the variadic ctor wants
        `__list_literal__`), and the `fill=` + subscript-assign form runs into
        `feedback_mojo_inlinearray_subscript_is_a_copy`. Six branches read
        worse and are simply correct.
        """
        if i == 0:
            return Self.HOME_0
        if i == 1:
            return Self.HOME_1
        if i == 2:
            return Self.HOME_2
        if i == 3:
            return Self.HOME_3
        if i == 4:
            return Self.HOME_4
        return Self.HOME_5

    # === the reward, ONCE ===
    #
    # ⚠⚠ THE CPU AND GPU HOOKS BOTH CALL THESE. They used to spell the reward
    # out twice — once over `Data`, once over `LayoutTensor` — with a comment
    # on each asking the reader to keep them identical. That is the setup for
    # a policy trained against one reward and evaluated against another, with
    # nothing raising, and this repo has already paid for a rule written twice
    # (`body_geom_visible`, three copies, two thresholds). The containers
    # differ, so the SUM over DOF still happens on each side; everything after
    # it happens here, once.

    @always_inline
    @staticmethod
    def _reach[DTYPE: DType](dist: Scalar[DTYPE]) -> Scalar[DTYPE]:
        """Shaped distance term: 1 inside `TARGET_RADIUS`, decaying over
        `REWARD_MARGIN`.

        ⚠ The sigmoid and its value-at-margin are NAMED rather than left to
        the default, because the GPU call site cannot use defaults and a
        silent mismatch would be a different reward curve per device.
        """
        return tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE](
            dist,
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](Self.TARGET_RADIUS),
            Scalar[DTYPE](Self.REWARD_MARGIN),
        )

    @always_inline
    @staticmethod
    def _smooth[DTYPE: DType](rate_sq: Scalar[DTYPE]) -> Scalar[DTYPE]:
        """Multiplicative action-rate term in `[RATE_FLOOR, 1]`, from the
        SQUARED L2 change in the normalized action since the last control step.

        ⚠ THE QUANTITY IS THE RATE, NOT THE SPEED, and that distinction is the
        entire reason this exists — see `RECORD_PREV_ACTION`. An action that
        alternates +-0.5 every step has a large rate and a small net motion;
        joint speed cannot tell it from a slow drift.

        ⚠ GAUSSIAN, so the term never reaches exactly zero. `_still` was
        quadratic-with-value-0 and spent 69.5% of its steps in a region with
        no derivative at all; that mistake is not repeated here.
        """
        var calm = tolerance[SIGMOID_GAUSSIAN, 0.1, DTYPE](
            sqrt(rate_sq),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](Self.RATE_FREE),
            Scalar[DTYPE](Self.RATE_MARGIN),
        )
        return Scalar[DTYPE](Self.RATE_FLOOR) + Scalar[DTYPE](
            1.0 - Self.RATE_FLOOR
        ) * calm

    @always_inline
    @staticmethod
    def _still[DTYPE: DType](speed_sq: Scalar[DTYPE]) -> Scalar[DTYPE]:
        """Multiplicative stillness term in `[VEL_FLOOR, 1]`, from the SQUARED
        L2 joint speed. See `VEL_FREE` for the measurement behind it."""
        # ⚠⚠ GAUSSIAN, NOT QUADRATIC, AND value_at_margin 0.1 NOT 0. A
        # quadratic with value-at-margin 0 reaches exactly zero at the margin
        # and stays there, so any policy beyond it gets no gradient — see
        # `VEL_MARGIN` for the 69.5% of steps that was true for. A gaussian
        # never reaches zero, so the term keeps a derivative however fast the
        # arm happens to be moving. The margin is wide enough that this is
        # belt-and-braces rather than the load-bearing part, and that is the
        # point: the shape should not be able to create a dead zone again if
        # the operating range moves.
        var calm = tolerance[SIGMOID_GAUSSIAN, 0.1, DTYPE](
            sqrt(speed_sq),
            Scalar[DTYPE](0.0),
            Scalar[DTYPE](Self.VEL_FREE),
            Scalar[DTYPE](Self.VEL_MARGIN),
        )
        return Scalar[DTYPE](Self.VEL_FLOOR) + Scalar[DTYPE](
            1.0 - Self.VEL_FLOOR
        ) * calm

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
        """`qpos`, `qvel`, the end-effector, the target, and the vector between.

        ⚠ `ee_to_target` is redundant with the two absolute positions and is
        included anyway: it is the only term whose scale matches the reward,
        and an arm policy that has to learn the subtraction wastes capacity.
        """
        for i in range(D.NQ):
            obs.append(d.qpos.data[i])
        for i in range(D.NV):
            obs.append(d.qvel.data[i])

        var ex = d.xpos.data[Self.EE_BODY * 3 + 0]
        var ey = d.xpos.data[Self.EE_BODY * 3 + 1]
        var ez = d.xpos.data[Self.EE_BODY * 3 + 2]
        var tx = d.xpos.data[Self.TARGET_BODY * 3 + 0]
        var ty = d.xpos.data[Self.TARGET_BODY * 3 + 1]
        var tz = d.xpos.data[Self.TARGET_BODY * 3 + 2]

        obs.append(ex)
        obs.append(ey)
        obs.append(ez)
        obs.append(tx)
        obs.append(ty)
        obs.append(tz)
        obs.append(tx - ex)
        obs.append(ty - ey)
        obs.append(tz - ez)
        # ⚠ THE PREVIOUS ACTION, indices 21..26 — the env wrote it at
        # action-application time (`RECORD_PREV_ACTION`), which is the one
        # point the CPU and GPU paths share; the reward hook is NOT, because
        # the two devices run obs and reward in opposite order.
        for i in range(6):
            obs.append(d.meta.data[META_IDX_TASK_PARAM_6 + i])
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
        """The HOME pose plus small noise, then a fresh target.

        ⚠ The noise is CLIPPED TO EACH JOINT'S RANGE, read from the joint
        table rather than assumed. Every joint here is `limited`, and a reset
        that starts outside the limit hands the solver a violated constraint on
        step 0 — which looks like an unstable model, not a bad reset.
        """
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var base = j * MODEL_JOINT_SIZE
            var adr = Int(m_joints[base + JOINT_IDX_QPOS_ADR])
            if adr < 0 or adr >= D.NQ:
                continue
            var lo = Float64(m_joints[base + JOINT_IDX_RANGE_MIN])
            var hi = Float64(m_joints[base + JOINT_IDX_RANGE_MAX])
            # ⚠ The mocap target's body carries no joint, so `njoint` is 6 for
            # both arms and every `adr` lands inside `home`. The guard is here
            # anyway: a future task fragment with a free-jointed prop would
            # otherwise read past the end of a six-element pose.
            var q0 = Self._home(adr) if adr < 6 else Float64(d.qpos.data[adr])
            var q = q0 + (
                random_float64() * 2.0 - 1.0
            ) * Self.RESET_NOISE
            if lo > -1e9 and q < lo:
                q = lo
            if hi < 1e9 and q > hi:
                q = hi
            d.qpos.data[adr] = Scalar[DTYPE](q)
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)

        # Target: uniform in the arm's azimuth cone, in an elevation band, in
        # a radial shell. Not uniform in VOLUME — deliberately, since uniform
        # volume concentrates targets at the outer radius where the arm is
        # least dexterous.
        var az = Self.AZ_CENTER + (random_float64() * 2.0 - 1.0) * Self.AZ_HALF
        var el = Self.EL_MIN + random_float64() * (Self.EL_MAX - Self.EL_MIN)
        var r = Self.R_MIN + random_float64() * (Self.R_MAX - Self.R_MIN)
        d.mocap_pos.data[Self.TARGET_BODY * 3 + 0] = Scalar[DTYPE](
            r * cos(el) * cos(az)
        )
        d.mocap_pos.data[Self.TARGET_BODY * 3 + 1] = Scalar[DTYPE](
            r * cos(el) * sin(az)
        )
        d.mocap_pos.data[Self.TARGET_BODY * 3 + 2] = Scalar[DTYPE](
            Self.BASE_Z + r * sin(el)
        )
        # ⚠ CLEAR THE ACTION SLOTS. `_reset_env_lane` does not touch the
        # TASK_PARAM range, so without this the first step of an episode reads
        # the LAST action of the previous one — an enormous fake action rate on
        # step 0, and a previous-action observation belonging to another
        # episode's target.
        for i in range(6):
            d.meta.data[META_IDX_TASK_PARAM_0 + i] = Scalar[DTYPE](0)
            d.meta.data[META_IDX_TASK_PARAM_6 + i] = Scalar[DTYPE](0)
        # Identity orientation, stored [x, y, z, w].
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[Self.TARGET_BODY * 4 + 3] = Scalar[DTYPE](1)

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
        """Shaped `tolerance` on the end-effector-to-target distance.

        Never terminates early — like every dm_control task, and unlike the
        MuJoCo-gym arms. An arm has no "unhealthy" state to fall into.
        """
        var dx = Float64(
            d.xpos.data[Self.TARGET_BODY * 3 + 0]
            - d.xpos.data[Self.EE_BODY * 3 + 0]
        )
        var dy = Float64(
            d.xpos.data[Self.TARGET_BODY * 3 + 1]
            - d.xpos.data[Self.EE_BODY * 3 + 1]
        )
        var dz = Float64(
            d.xpos.data[Self.TARGET_BODY * 3 + 2]
            - d.xpos.data[Self.EE_BODY * 3 + 2]
        )
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        # Only the SUM is per-container; the terms themselves are shared with
        # the GPU hook — see `_reach` / `_still`.
        var speed2 = Scalar[DTYPE](0)
        for i in range(D.NV):
            var v = d.qvel.data[i]
            speed2 += v * v
        var rate2 = Scalar[DTYPE](0)
        for i in range(6):
            var dA = (
                d.meta.data[META_IDX_TASK_PARAM_6 + i]
                - d.meta.data[META_IDX_TASK_PARAM_0 + i]
            )
            rate2 += dA * dA
        return (
            Self._reach(Scalar[DTYPE](dist))
            * Self._still(speed2)
            * Self._smooth(rate2),
            False,
        )


    # =====================================================================
    # GPU hooks — the batched twin of the three CPU hooks above.
    #
    # ⚠⚠ `Phyics3dBatchedEnv` carries a `comptime assert HAS_GPU_HOOKS`
    # precisely because a CPU-only config wired to it COMPILES AND RUNS,
    # training against a flat-zero reward. Flip the flag in the same
    # commit as the hooks, never ahead of them.
    #
    # Modelled on `dm_control/reacher/reacher_config.mojo`, the only other
    # config with GPU hooks AND a mocap target — the same task shape, at
    # 2 DoF and a 2D target instead of 6 and 3D.
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
        """`qpos`, `qvel`, the end-effector, the target, and the vector between.

        ⚠⚠ MUST STAY BYTE-FOR-BYTE IN THE SAME ORDER AS
        `custom_extract_obs_cpu`. The batched trainer writes a checkpoint the
        single-env eval loads, so a permuted observation here is a policy that
        works on the GPU and is nonsense on the CPU, with no error anywhere.
        `tests/robots/test_so_arm101_obs_cpu_vs_gpu.mojo` is what keeps the two
        honest.
        """
        for i in range(NQ_F):
            obs[env, i] = qpos[env, i]
        for i in range(NV_F):
            obs[env, NQ_F + i] = qvel[env, i]

        var ex = xpos[env, Self.EE_BODY * 3 + 0]
        var ey = xpos[env, Self.EE_BODY * 3 + 1]
        var ez = xpos[env, Self.EE_BODY * 3 + 2]
        var tx = xpos[env, Self.TARGET_BODY * 3 + 0]
        var ty = xpos[env, Self.TARGET_BODY * 3 + 1]
        var tz = xpos[env, Self.TARGET_BODY * 3 + 2]

        var b = NQ_F + NV_F
        obs[env, b + 0] = ex
        obs[env, b + 1] = ey
        obs[env, b + 2] = ez
        obs[env, b + 3] = tx
        obs[env, b + 4] = ty
        obs[env, b + 5] = tz
        obs[env, b + 6] = tx - ex
        obs[env, b + 7] = ty - ey
        obs[env, b + 8] = tz - ez
        # The previous action, 21..26 — byte-for-byte the CPU hook's tail.
        for i in range(6):
            obs[env, b + 9 + i] = meta[env, META_IDX_TASK_PARAM_6 + i]
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
        """Shaped `tolerance` on end-effector-to-target distance.

        The GPU twin of `compute_reward_and_done_cpu`, same constants, same
        never-terminates rule. ⚠ `tolerance`'s SIGMOID and value-at-margin are
        spelled explicitly here because the GPU form takes them as parameters
        while the CPU call relies on their defaults — a silent mismatch would
        be a different reward curve on each device.
        """
        # ⚠ `rebind[Scalar[DTYPE]]`, not `Scalar[DTYPE](...)`. A LayoutTensor
        # subscript yields a layout-parameterised `element_type`; the Scalar
        # constructor has no conversion from it, `rebind` reinterprets it.
        # Same idiom as `quadruped_escape_config.mojo:497`.
        var dx = rebind[Scalar[DTYPE]](
            xpos[env, Self.TARGET_BODY * 3 + 0]
        ) - rebind[Scalar[DTYPE]](xpos[env, Self.EE_BODY * 3 + 0])
        var dy = rebind[Scalar[DTYPE]](
            xpos[env, Self.TARGET_BODY * 3 + 1]
        ) - rebind[Scalar[DTYPE]](xpos[env, Self.EE_BODY * 3 + 1])
        var dz = rebind[Scalar[DTYPE]](
            xpos[env, Self.TARGET_BODY * 3 + 2]
        ) - rebind[Scalar[DTYPE]](xpos[env, Self.EE_BODY * 3 + 2])
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        # ⚠ ONLY THE SUM IS WRITTEN TWICE — `qvel` is a `LayoutTensor` here
        # and a `Data` field there. Both terms come from `_reach` / `_still`,
        # so the reward the trainer optimises and the reward the eval, the
        # viewer and the hardware deploy score are the same expression.
        var speed2 = Scalar[DTYPE](0)
        for i in range(NV_F):
            var v = rebind[Scalar[DTYPE]](qvel[env, i])
            speed2 += v * v
        var rate2 = Scalar[DTYPE](0)
        for i in range(6):
            var dA = rebind[Scalar[DTYPE]](
                meta[env, META_IDX_TASK_PARAM_6 + i]
            ) - rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + i])
            rate2 += dA * dA
        return (
            Self._reach(dist) * Self._still(speed2) * Self._smooth(rate2),
            False,
        )

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
        """HOME pose plus noise, then a fresh target — the GPU twin of
        `custom_reset_cpu`.

        ⚠ The noise is CLIPPED TO EACH JOINT'S RANGE, read from the joint
        table rather than assumed, for the same reason the CPU hook does it: a
        reset that starts outside a limit hands the solver a violated
        constraint on step 0, which reads as an unstable model rather than a
        bad reset.

        ⚠ ONE PHILOX DRAW PER VALUE, `[0]` only. Wasteful of lanes and
        deliberately so — the alternative is an index into a 4-lane block that
        has to stay in step with a future edit, and this runs once per episode.
        """
        # Joints: home + uniform noise, clipped to the model's own limits.
        # ⚠⚠ `joints[j, FIELD]`, NOT `joints[j * MODEL_JOINT_SIZE + FIELD]`.
        # The CPU hook reads a FLAT `List`; here `joints` is a 2-D
        # `LayoutTensor(NJOINT_F, MODEL_JOINT_SIZE)` and a single-integer
        # subscript selects a ROW, not an element. Same trap as the flat-index
        # TileTensor note; `gpu_reset.mojo:226` is the correct idiom.
        var rng = PhiloxRandom(seed=reset_seed(env, seed), offset=0)
        for j in range(NJOINT_F):
            var adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            if adr < 0 or adr >= NQ_F:
                continue
            var q0 = Scalar[DTYPE](Self._home(adr)) if adr < 6 else rebind[
                Scalar[DTYPE]
            ](qpos[env, adr])
            var u = rng.step_uniform()
            var q = q0 + (
                Scalar[DTYPE](u[0]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
            ) * Scalar[DTYPE](Self.RESET_NOISE)
            var lo = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
            var hi = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
            if lo > Scalar[DTYPE](-1e9) and q < lo:
                q = lo
            if hi < Scalar[DTYPE](1e9) and q > hi:
                q = hi
            qpos[env, adr] = q
        for i in range(NV_F):
            qvel[env, i] = Scalar[DTYPE](0)

        # Target: the same azimuth cone / elevation band / radial shell the CPU
        # hook draws from. NOT uniform in volume, deliberately — uniform volume
        # concentrates targets at the outer radius where the arm is least
        # dexterous.
        var ua = rng.step_uniform()
        var ue = rng.step_uniform()
        var ur = rng.step_uniform()
        var az = Scalar[DTYPE](Self.AZ_CENTER) + (
            Scalar[DTYPE](ua[0]) * Scalar[DTYPE](2.0) - Scalar[DTYPE](1.0)
        ) * Scalar[DTYPE](Self.AZ_HALF)
        var el = Scalar[DTYPE](Self.EL_MIN) + Scalar[DTYPE](ue[0]) * Scalar[
            DTYPE
        ](Self.EL_MAX - Self.EL_MIN)
        var r = Scalar[DTYPE](Self.R_MIN) + Scalar[DTYPE](ur[0]) * Scalar[
            DTYPE
        ](Self.R_MAX - Self.R_MIN)

        # ⚠ `sin_dt`/`cos_dt`, not `sin`/`cos`: the plain ones do not resolve
        # in a GPU kernel ("lacking evidence to prove correctness").
        var ce = cos_dt[DTYPE](el)
        mocap_pos[env, Self.TARGET_BODY * 3 + 0] = r * ce * cos_dt[DTYPE](az)
        mocap_pos[env, Self.TARGET_BODY * 3 + 1] = r * ce * sin_dt[DTYPE](az)
        mocap_pos[env, Self.TARGET_BODY * 3 + 2] = Scalar[DTYPE](
            Self.BASE_Z
        ) + r * sin_dt[DTYPE](el)
        # Identity orientation, [x, y, z, w].
        # ⚠ The GPU twin of the CPU reset's clear — same reason, same slots.
        for i in range(6):
            meta[env, META_IDX_TASK_PARAM_0 + i] = Scalar[DTYPE](0)
            meta[env, META_IDX_TASK_PARAM_6 + i] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 0] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 1] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 2] = Scalar[DTYPE](0)
        mocap_quat[env, Self.TARGET_BODY * 4 + 3] = Scalar[DTYPE](1)

    @staticmethod
    def get_timestep() -> Float64:
        return Self.TIMESTEP
