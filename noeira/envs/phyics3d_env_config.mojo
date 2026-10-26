"""Phyics3dEnvConfig trait — captures what varies between Phyics3d environments.

Phyics3dEnv[MODEL_DEF: ModelDefLike, CONFIG: Phyics3dEnvConfig] delegates everything to CONFIG:
  - Model setup, integrator choice, reward, termination, GPU model init
  - Obs extraction, reset, enforce limits (delegates to Joints internally)
  - Action application (delegates to Actuators internally)

The config has full access to physics state (qpos, qvel, etc.) for reward
and termination — no hardcoded assumptions about which joints matter.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Model, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_MESH_META_SIZE,
    MAX_GPU_HFIELDS,
    MODEL_HFIELD_META_SIZE,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_TENDON_SIZE,
)
from noeira.core.tiles import Tile


trait Phyics3dEnvConfig:
    # === Physics ===
    comptime FRAME_SKIP: Int
    comptime MAX_STEPS: Int
    comptime INTEGRATOR_WS_EXTRA: Int  # 0 for RK4/Euler, >0 for ImplicitFast
    # Which fields integrator the facades dispatch on ("rk4" | "euler"), with
    # Newton as the solver. Default "rk4" (9/12 envs); HalfCheetah/Pusher/
    # MetaWorld override to "euler".
    comptime INTEGRATOR: StaticString = "rk4"

    # Convex-hull vertex capacity for COLLIDABLE mesh geoms.
    #
    # ⚠⚠ ZERO MEANS MESH GEOMS DO NOT COLLIDE — it is not a "size hint". Both
    # narrow phases guard their mesh branch with `comptime if NMESH_VERTS > 0`
    # and otherwise emit no contact for the pair. This was hardcoded to 0 in
    # `Phyics3dEnv` for every environment, so the entire mesh collider was
    # unreachable from the env layer while the fixtures that build `Model`
    # directly kept passing.
    #
    # Default 0 is CORRECT for almost everything and must stay: of the ported
    # models only sawyer has collidable meshes (2 geoms — `l6` and
    # `eGripperBase`). Dog has 162 mesh geoms and every one is visual
    # (`contype=0 conaffinity=0`); the rest of the suite has no meshes at all.
    # A nonzero value compiles the whole GJK/EPA mesh branch into that model's
    # collision kernel, which costs compile time and registers and risks the
    # Metal wide-Array cliff — so pay it only where a mesh can collide.
    #
    # It cannot be derived from the XML at comptime (mesh assets need file I/O
    # and the comptime parser cannot read an STL), so it is hand-supplied like
    # MAX_CONTACTS. `fields_build` RAISES if it is too small, quoting the
    # required count — an undersized value can no longer truncate silently.
    comptime NMESH_VERTS: Int = 0

    comptime NMESH_TRI: Int = 0
    """Total ORIGINAL triangles across every collidable mesh — the RAY soup.

    ⚠⚠ A DIFFERENT SURFACE FROM `NMESH_VERTS`, WHICH IS THE CONVEX HULL. The
    hull is what collision wants and what MuJoCo collides too; it cannot answer
    a RAY, because a ray aimed into a bracket's cutout hits hull where the real
    part has a hole. `ray/mesh.mojo` walks these triangles.

    ⚠ ZERO MEANS EVERY MESH IS INVISIBLE TO A RAY — a rangefinder, studio
    picking, or a camera looks straight through the robot's own arm. It is not
    a size hint and it is not derived: `ray_mesh` returns NO HIT when `ntri` is
    0, which is a capacity being read as an absence of geometry.

    ⚠ EXACT, and `fields_build` RAISES quoting the required count if it is too
    small — so the way to find the number is to set it to 1 and read the error,
    not to count triangles by hand.

    Costs `NMESH_TRI * 27` floats in `Model` (triangles plus their BVH — see
    `MESH_ARENA_FLOATS_PER_TRI`) and nothing per step. Leave it 0 on
    a model nothing rays at.
    """

    comptime NHFIELD_DATA: Int = 0
    """Total heightfield grid samples, `sum(nrow * ncol)` over `<hfield>`.

    ⚠ EXACT, NOT A BUDGET, unlike `NMESH_VERTS` — the grid's size is known
    from the XML's `nrow`/`ncol` and nothing derives it later. 0 is right for
    every model without a heightfield, which until `quadruped escape` was all
    of them.

    ⚠⚠ LEAVING IT AT 0 ON A MODEL THAT HAS ONE IS SILENT: the grid allocates
    a single element, every elevation reads as whatever that one slot holds,
    and the terrain collides as a flat plane at z=0. It is a capacity being
    read as geometry — the shape
    `feedback_a_capacity_answer_read_as_a_feature_gate` warns about.
    """

    # Refresh forward kinematics AFTER the frame-skip loop, so that reward and
    # observation hooks see xpos/xquat/xipos/site_xpos consistent with the
    # INTEGRATED qpos.
    #
    # Off by default because that matches raw `mj_step`, and therefore
    # Gymnasium: MuJoCo computes FK in mj_step1 and integrates in mj_step2, so
    # after `mj_step(n)` the derived fields still describe the state BEFORE the
    # last substep. Every Gym-derived env here (Ant/Humanoid/Sawyer read xpos,
    # xipos, cfrc_ext) is calibrated against that convention — flipping this
    # globally would shift their rewards by one step.
    #
    # dm_control does the opposite: `Physics._step_with_up_to_date_position_
    # velocity` runs mj_step2 then mj_step1 specifically so "(most of) mjData
    # is in sync with qpos and qvel" when the task reads it. Suite ports must
    # therefore set this True, or every xmat/site-based observation and reward
    # is silently one control step stale.
    comptime SYNC_FK_AFTER_STEP: Bool = False

    # Run `mj_rnePostConstraint` inside every substep, filling `Data.cacc`
    # and `Data.cfrc_int` (dynamics/rne_post.mojo). Needed ONLY by the
    # acceleration-stage sensors — `accelerometer`, `force`, `torque` — and
    # off by default because it is pure overhead for every other model.
    #
    # The values land at MuJoCo's `mj_sensorAcc` point: the state BEFORE the
    # substep's integration, with that substep's constrained qacc. After the
    # frame-skip loop they therefore describe the second-to-last state, which
    # is exactly what dm_control observes — `mj_step1` refreshes position and
    # velocity sensors at the new state but leaves the acceleration stage
    # alone. Do NOT "fix" this to agree with SYNC_FK_AFTER_STEP.
    #
    # Euler only: the RK4 integrator would need the hook inside its base
    # stage, and no in-scope model wants both.
    comptime RNE_POST: Bool = False

    # Does this config implement the GPU hooks (`compute_reward_and_done_gpu`
    # and, where the model default is not enough, `custom_extract_obs_gpu`)?
    #
    # ⚠ THIS EXISTS TO MAKE A SILENT FAILURE LOUD. Those hooks carry INERT
    # DEFAULTS — zero reward, "use the model default obs" — rather than being
    # abstract, so that CPU-only configs need not restate ~90 lines of stub
    # each. The cost is that wiring a CPU-only config to `Phyics3dBatchedEnv`
    # COMPILES AND RUNS, and trains against a flat-zero reward curve. That cost
    # was paid for real by the dm_control suite, whose ~36 task configs were
    # CPU-only by construction for months (gap G10).
    #
    # Mojo cannot ask "did you override this method?", so the config declares
    # it and `Phyics3dBatchedEnv` asserts it at compile time. Flipping this to
    # True without actually implementing the hooks reinstates exactly the trap
    # it closes — so flip it in the same commit as the hooks, never ahead.
    comptime HAS_GPU_HOOKS: Bool = False

    # Does this config drive a MOCAP body (a per-episode target parked on a
    # `<body mocap="true">`, the G4 workaround)? reacher, finger, fish, swimmer,
    # manipulator, stacker and SawyerReach all do.
    #
    # ⚠ WHY A DECLARATION AND NOT A MODEL QUERY. `is_mocap` is parsed by the
    # RUNTIME parser into `Model.bodies[.., BODY_IDX_MOCAP]`; the COMPTIME
    # parser does not carry it (`physics3d has TWO MJCF parsers`). Making the
    # batched env comptime-gate on it therefore needs either a declaration or
    # new comptime-parser work, and the declaration is the smaller change.
    #
    # ⚠ FORGETTING IT IS NOT SILENT. `Phyics3dBatchedEnv.__init__` reads the
    # built model and RAISES if any body is mocap-flagged while this is False —
    # the failure mode it prevents is a target frozen at its XML pose, i.e. a
    # silently easier task, which no gate would flag as an error.
    comptime USES_MOCAP: Bool = False

    # Raise the free root in 1 cm steps at reset until nothing is touching,
    # after `custom_reset_cpu` has set the orientation
    # (`Phyics3dEnv._find_non_contacting_height`). dm_control's quadruped is
    # the only user: it draws a random orientation per episode, so a fixed
    # spawn height would sometimes start the robot inside the floor.
    comptime RESET_FIND_HEIGHT: Bool = False

    # Does this config drive the actuators itself on the GPU path, via
    # `custom_apply_actions_gpu` below?
    #
    # ⚠ THE GPU TWIN CANNOT BE A `return False` DEFAULT LIKE THE CPU ONE.
    # `Phyics3dEnv` calls `custom_apply_actions_cpu` and branches on its return
    # value at runtime; the batched path has to choose between two KERNEL
    # LAUNCHES, so the choice must be comptime. Hence a declaration, checked
    # the same way `HAS_GPU_HOOKS` is.
    #
    # Setting this True and not implementing the hook gives a permanently zero
    # `qfrc` — every actuator dead, which trains to a flat curve rather than
    # crashing. Flip it in the same commit as the hook.
    comptime RECORD_PREV_ACTION: Bool = False
    """The env records each control step's action into `Data.meta`, so a
    config can put the PREVIOUS ACTION in its observation and penalise the
    ACTION RATE.

    Layout, when set: `META_IDX_TASK_PARAM_0..5` hold the action of the
    PREVIOUS control step and `..._6..11` the CURRENT one. The env shifts
    current -> previous and writes the new action, once per control step,
    AT ACTION-APPLICATION TIME.

    ⚠⚠ THE WRITE SITE IS THE WHOLE DESIGN. The obvious place is the reward
    hook, which already receives both `actions` and a mutable `meta` — and it
    is WRONG, because the two devices run the hooks in OPPOSITE ORDER:

        CPU (`Phyics3dEnv.step`)          physics -> REWARD -> obs
        GPU (`Phyics3dBatchedEnv`)        physics -> OBS -> reward

    so an action stashed by the reward hook is visible to the CPU's obs on the
    SAME step and to the GPU's obs only on the NEXT one. That is a silent
    one-step skew between a policy's training and its evaluation, with nothing
    raising. Action application happens before physics on both paths, so a
    write there is the same value at every later hook on both devices.

    ⚠ Costs 12 of the 12 `TASK_PARAM` slots for a 6-DOF model, so a config
    cannot combine this with per-episode randomised model parameters without
    widening `METADATA_SIZE`.

    ⚠ `init_qpos` must clear both groups, or the first step of an episode sees
    the last step of the previous one — `_reset_env_lane` does not zero these.
    """
    comptime NORMALIZED_ACTIONS: Bool = False
    """Actions arrive in [-1, 1] and are mapped AFFINELY onto each actuator's
    own `ctrlrange`, instead of being taken as raw control values and clamped.

    ⚠⚠ THIS IS AN OPTIMISATION FIX, NOT A CONVENIENCE. A policy emitting
    `tanh(mu) * action_scale` with ONE scalar scale cannot fit a set of joints
    with different ranges: pick a scale large enough for the widest joint and
    the narrow ones saturate. Measured on SO-ARM101 (`ACTION_SCALE = 2.0`
    against ranges of 1.66 to 2.84): the trained policy commanded an
    out-of-range pose on **24% to 100% of control steps**, `elbow_flex` sat at
    the tanh rail 49% of the time, and the gripper — range -0.17..1.75 against
    a symmetric +-2.0 — was out of range on EVERY step.

    Everything past the clamp is then a DEAD GRADIENT BAND: a whole interval
    of actor outputs maps to one clamped pose, so the actor can drift across
    it at no cost and flip to the far rail for free. That is what made three
    successive reward shapes produce the same shaking behaviour — the clamp
    was destroying the gradient the reward was trying to supply.

    ⚠ Only `ctrllimited` actuators are mapped; an unlimited one has no range
    to map onto and passes through raw. A model mixing the two gets a mixed
    action space, which is why this is per-CONFIG rather than per-actuator.

    ⚠ CHANGING THIS INVALIDATES CHECKPOINTS. The action means something
    different on each side of the flag."""
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = False

    comptime HAS_OSC_CONTROLLER: Bool = False
    """Drive the model through OSC_POSE instead of feeding actions to the
    actuators — robosuite's `OSC_POSE`, which is what LIBERO's demonstrations
    and every LIBERO policy speak.

    ⚠⚠ THIS IS A CONTROLLER, NOT AN ACTUATION HOOK, AND THE DIFFERENCE IS THE
    WHOLE REASON IT IS NOT `HAS_CUSTOM_ACTUATION_GPU`. That hook REPLACES
    actuation, `qfrc` zeroing included, so using it here would mean
    re-implementing every `<motor>` and `<position>` in the model. OSC maps a
    SEVEN-word policy action onto the model's `nact` actuator commands, and the
    ordinary actuation path then turns those into forces exactly as it does for
    any other env. The batched env therefore runs the controller BEFORE
    `apply_actions_kernel_gpu` and feeds it the `ctrl` vector.

    ⚠ SO `ACTION_DIM` IS 7 AND NOT `nact`. A model def for an OSC family must
    say `action_dim_override=7`; the actuator count is read from the model.

    ⚠ AND THE ENV NEEDS ITS `refs` BEFORE IT CAN STEP. Which velocity index,
    qpos address, joint and actuator each of the seven arm joints is — plus the
    grip site and the two finger actuators — are properties of the COMPOSED
    scene, which a comptime config cannot read. `Phyics3dBatchedEnv.set_osc_refs`
    takes the record `build_osc_refs` makes from the parse, and `step_batch`
    RAISES if it was never called. A controller silently driving zeros is an
    arm that falls over slowly.

    ⚠ IT COSTS A SECOND DYNAMICS PASS PER SUBSTEP. `osc_control_step` refreshes
    FK, `subtree_com`, `cdof`, CRBA and RNE at the current state and the
    integrator then recomputes all of it — robosuite pays the same
    (`sim.forward()` then `mj_step`). See `dynamics/osc_control.mojo`."""

    comptime CUSTOM_ACTIONS_EVERY_SUBSTEP: Bool = False
    """Call `custom_apply_actions_cpu` at the top of EVERY substep, not once
    per control step.

    The CPU hook's documented cadence is once per control step, which is
    right for action SEMANTICS (a mocap delta that must not compound) and
    wrong for a state-dependent FORCE LAW: a PD controller evaluated once and
    frozen for `FRAME_SKIP` substeps is a different controller from one that
    reads `qpos`/`qvel` every substep, which is what the GPU hook already does
    and what IsaacLab / MuJoCo do. Unitree G1 (BFM-Zero's torque-level PD at
    200 Hz under 50 Hz control) is the first model where the two differ
    measurably. With this True the CPU and GPU cadences match.

    ⚠ The hook is still called once BEFORE the loop (that call is the first
    substep's); the extra calls are substeps 2..FRAME_SKIP. A hook that is
    not idempotent in its side effects must not set this."""

    # === CPU: Pre-step hook — save any per-env state before physics ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        """Save per-env state before physics step.

        The prev_x parameter is a single scalar stored per-env (in the
        metadata region on GPU). Configs use it to store whatever they need
        (e.g., rootx position for velocity computation).
        """
        # Default: memoryless reward — nothing to stash.
        pass

    # ── Model record lists on the CPU hooks ──────────────────────────────
    # `compute_reward_and_done_cpu`, `custom_extract_obs_cpu`,
    # `custom_reset_cpu` and `custom_apply_actions_cpu` receive the packed
    # model record tensors alongside the state.
    #
    # They are the host `List`s behind `Model.bodies/joints/geoms/sites`
    # (plus `Model.tendons` on the two action-side hooks), borrowed (never
    # copied), indexed with the usual column constants from
    # `physics3d.gpu.constants`:
    #
    #     m_bodies [b * MODEL_BODY_SIZE   + BODY_IDX_MASS]
    #     m_geoms  [g * MODEL_GEOM_SIZE   + GEOM_IDX_SIZE_0]
    #     m_tendons[t * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0]
    #
    # Added 2026-07-29: the hooks previously saw `Data` only, so a reward or
    # observation could not read a single model constant. That blocked every
    # mass-weighted sensor (subtreelinvel) and the dm_control tasks that size
    # their reward from the model (reacher/acrobot/ball_in_cup/point_mass read
    # geom_size / site_size; the joint randomizer needs joint ranges).
    #
    # Passing the record lists rather than `Model` itself keeps the hook
    # signatures free of Model's six extra compile-time parameters.
    #
    # `m_tendons` (2026-07-31) is on `custom_apply_actions_cpu` and on the new
    # `custom_reset_model_cpu` only, not on the reward/obs hooks, because the
    # only thing that reads it is transmission: a fixed tendon's per-joint
    # `coef` IS its actuator moment arm (MuJoCo's `wrap_prm`), so a config that
    # drives the DOFs itself needs the coefs, and a config that randomizes the
    # control-to-joint mixing per episode needs to WRITE them.

    # === CPU: Unified reward + termination ===
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
        """Compute reward and early termination from full physics state.

        Args:
            d: Fields physics state with qpos, qvel, xpos, etc.
            m_bodies: Flat body records of the model.
            m_joints: Flat joint records of the model.
            m_geoms: Flat geom records of the model.
            m_sites: Flat site records of the model.
            prev_x: Value saved by pre_step_cpu (e.g., previous x position).
            actions: Clamped action values.
            step_count: Current step count (for truncation checking outside).
            frame_skip: Number of physics substeps per env step.

        Returns:
            (reward, terminated) — terminated is True for early termination
            (NOT truncation — truncation is handled by the generic env).
        """
        ...

    # === CPU: Custom reset hook (called after _reset_state) ===
    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """Custom reset logic (e.g., set initial mocap position, pin goal
        joints). The facade runs the fields FK after this hook, so writes to
        qpos/mocap take effect before the first observation. Default: no-op."""
        pass

    # === CPU: Reset hook with the FULL model (kinematics + collision) ===
    @staticmethod
    def custom_reset_full_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
    ) raises:
        """Reset logic that needs the whole `Model`. Default: no-op.

        ⚠ WHY THIS EXISTS ALONGSIDE `custom_reset_cpu`. That hook is handed the
        record LISTS (`bodies`/`joints`/`geoms`/`sites`), which is enough to
        read a joint range or write a mocap pose and NOT enough to run forward
        kinematics, build a Jacobian or detect contacts — all of which take
        `Model` itself. dm_control's manipulation reset is exactly that shape:
        `ToolCenterPointInitializer` solves site IK and re-runs the narrow
        phase to reject colliding poses. Before this hook it could not be
        reached from a config at all, so `reach_site_features` reset to qpos0
        — a 55-contact pose.

        ⚠ RUNS AFTER `custom_reset_model_cpu` AND `custom_reset_cpu`, so it
        sees whatever they wrote. That ordering is load-bearing for
        manipulation: `set_grasp` closes the fingers in the state hook, and the
        IK here restores ONLY the arm joints on a rejected sample precisely so
        that the grasp survives.

        ⚠ IT MAY LEAVE `d` MID-KINEMATICS. The facade re-runs `_fields_fk` and
        `_fields_vel` after this returns, so a hook that moved `qpos` need not
        refresh them itself — but a hook that CACHED an FK product locally
        across the call is reading a stale one.

        `raises` because the routines it exists to call do (mesh collision,
        `detect_contacts`). The default body cannot raise; overriders may.

        Cf. `RESET_FIND_HEIGHT`, which solved the same problem for one env by
        putting a comptime flag on the config and the code on `Phyics3dEnv`.
        That does not generalise — the next env needs different code, not the
        same code behind a second flag.
        """
        pass

    # === CPU: Per-episode MODEL randomization (called before custom_reset_cpu) ===
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
        """Randomize MODEL constants per episode. Default: no-op.

        Distinct from `custom_reset_cpu`, which randomizes STATE, and it runs
        first so the state hook reads whatever this wrote. dm_control does the
        two together in `initialize_episode` (point_mass `hard` randomizes the
        joints, then the tendon mixing matrix); the split here is only so that
        the ~16 configs which touch state alone need not restate a model
        signature they never use.

        ⚠ Three things this hook does NOT do, all of which it would have to if
        the randomized quantity were anything but a transmission coefficient:

        1. It does not re-upload to the device. These are the HOST lists; a
           GPU-batched env would keep stepping the stale copy. Nothing but
           `Phyics3dEnv._reset_state` calls this hook today, so that is a
           real limit rather than a latent bug — but it is why the name says
           `_cpu`, and it is also why a batched driver cannot simply start
           calling it: `Model` is SHARED and unbatched, so per-episode model
           randomization there would apply one env's draw to all of them.
        2. It does not recompute anything DERIVED from the model. Masses feed
           the CRBA mass matrix; tendon coefs feed `tendon_invweight0`
           (`J M^-1 J^T` at qpos0, the limit row's diagApprox). Writing a coef
           here leaves that constant describing the OLD tendon. Harmless for
           point_mass, whose tendons carry no limit and no equality — check
           before randomizing a tendon that has either.
        3. It does not touch the COMPTIME actuator tables (`_acd.motor_trn_*`),
           which is where `MODEL_DEF.apply_actions` reads transmission from.
           A config that randomizes coefs must therefore also drive the DOFs
           itself via `custom_apply_actions_cpu` — the comptime path cannot
           see runtime writes, and would silently keep the XML coefficients.
        """
        pass

    # === CPU: Custom observation extraction, WITH the ray tables ===
    @staticmethod
    def custom_extract_obs_ray_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) raises -> Bool:
        """`custom_extract_obs_cpu` plus `Data` and `Model` themselves.

        ⚠⚠ A SECOND HOOK RATHER THAN A WIDER FIRST ONE, and the reason is
        arithmetic: fifty types implement `Phyics3dEnvConfig`, and widening
        the existing signature would edit all fifty to give one of them four
        arguments. This defaults to FORWARDING to `custom_extract_obs_cpu`, so
        every config that does not care is untouched and unaware.

        Override THIS one when an observation needs `mj_ray` — a
        `<rangefinder>`, a lidar, a line-of-sight test. `dm_control`'s
        `quadruped escape` is the first: twenty rangefinders, which need the
        mesh and heightfield tables the four record lists above do not carry.

        ⚠ IT TAKES THE CONTAINERS, NOT MORE RECORD LISTS. `ray_model` reads
        eight tensors and composes geom world poses itself; handing over eight
        more `List`s would have put a second copy of that composition in every
        config that casts a ray. `mut` is what `TensorImpl.lt_dyn` needs to
        produce a view, and it is why `Env.get_state` carries the marker —
        see `core/env.mojo`.
        """
        # ⚠ THE RECORD LISTS COME OFF `mf` HERE, not from four more
        # parameters. Passing `mf` mutably AND `mf.bodies.data` immutably in
        # ONE call is an alias Mojo rejects; reading them inside is a separate
        # call with no mutable argument, so it is fine. They were redundant
        # once the container was in hand anyway.
        return Self.custom_extract_obs_cpu[DTYPE, D](
            d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
            act, obs,
        )

    # === CPU: Custom observation extraction (default: use MODEL_DEF.extract_obs) ===
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
        """Extract observations from data. Return True if handled, False for default.

        Override for envs that need non-standard observations
        (e.g., hand position + object position instead of qpos/qvel).

        `act` is MuJoCo's `d->act` — the actuator ACTIVATION state, one scalar
        per activation variable, empty-but-length-1 on the models that have
        none (`MODEL_DEF.NA == 0`, which is all of them but quadruped). It is
        here because dm_control's quadruped puts `data.act` inside its
        `egocentric_state` block, i.e. in the MIDDLE of the observation, so
        the env cannot append it after the fact. See `Phyics3dEnv.act` for why
        the activation lives on the env rather than in `Data`.
        """
        return False

    # === CPU: Custom action application (default: use MODEL_DEF.apply_actions) ===
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
        """Apply actions to data. Return True if handled, False for default.

        Override for envs that need non-standard action application
        (e.g., mocap position control instead of torque motors, or a
        transmission whose coefficients are randomized per episode and so
        cannot live in the shared actuator records).

        ⚠ `m_actuators` / `m_act_tendons` ARE `SpecFields`' PACKED RECORDS,
        added 2026-08-15 with phase 1a.4. Index them
        `i * MODEL_ACTUATOR_SIZE + ACT_IDX_*` and
        `t * MODEL_ACT_TENDON_SIZE + ACTTEN_IDX_*`. They are here for the same
        reason `m_joints` and `m_tendons` are: an override that reimplements
        actuation needs the gains and ranges, and the alternative was reading
        them off the comptime `_acd`, which a hook cannot see once the data is
        runtime.
        Default returns False, which causes Phyics3dEnv.step() to call
        MODEL_DEF.apply_actions() as usual.

        Returning True suppresses `MODEL_DEF.apply_actions` ENTIRELY, including
        the `qfrc` zeroing it opens with — an override that sums several
        transmissions onto one DOF must zero `d.qfrc` itself, or each control
        step adds to the previous one instead of replacing it.

        Note the call is ONCE PER CONTROL STEP, outside the frame-skip loop
        (see `Phyics3dEnv.step`), whereas `MODEL_DEF.apply_actions` runs every
        substep. Identical for a `<motor>`, whose force `gear*ctrl` is constant
        across the step; NOT identical for a state-dependent force law.
        """
        return False

    # === GPU inline: Custom action application (per-lane; G4 workaround) ===
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
        """The batched twin of `custom_apply_actions_cpu`, one lane.

        Reached only when `HAS_CUSTOM_ACTUATION_GPU` is True, in which case it
        REPLACES `MODEL_DEF.apply_actions_kernel_gpu` entirely — including the
        `qfrc` zeroing that opens it. Zero `qfrc[env, :]` yourself.

        ⚠ UNLIKE THE CPU TWIN, THIS RUNS ONCE PER SUBSTEP. The batched env
        calls it at the top of every frame-skip iteration, exactly where it
        calls the model default, so a state-dependent force law is correct
        here where the CPU hook's once-per-control-step cadence would not be.
        `Phyics3dEnv` is the one that needs fixing if that ever matters.

        `meta` is here for the TASK_PARAM slots — a per-episode model
        parameter that cannot live in the shared `Model` (see
        `physics3d/gpu/constants.META_IDX_TASK_PARAM_0`). `tendons` and
        `joints` are the RUNTIME records, so a transmission read from them
        follows the model rather than the comptime tables.

        ⚠ `acts` / `act_tendons` ARE `SpecFields`' PACKED RECORDS, added
        2026-08-15 with phase 1a.4 — the same operands
        `apply_actions_kernel_gpu` reads, so an override and the default read
        one source. FLAT, addressed `i * MODEL_ACTUATOR_SIZE + ACT_IDX_*`.
        """
        pass

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===
    @staticmethod
    def get_timestep() -> Float64:
        ...

    @staticmethod
    def get_reset_noise() -> Float64:
        # Default: no symmetric qpos/qvel jitter; envs that want it
        # override, envs that randomize in custom_reset_cpu do not.
        return 0.0

    # =====================================================================
    # GPU hooks.
    #
    # `pre_step_gpu` and `compute_reward_and_done_gpu` carry INERT DEFAULTS
    # (no-op / zero reward) rather than being abstract, so CPU-only envs need
    # not restate ~90 lines of stub each.
    #
    # ⚠ `xquat` and `xvel` were added 2026-08-06 (G10 step 2, see
    # docs/DM_CONTROL_GPU_TRAINING_G10.md) — they are what the dm_control suite
    # needs and what its facades cite as the reason for being CPU-only.
    # `xquat` gives `xmat` on demand via `kinematics/xmat.xmat_elem_gpu` (no
    # NBODY*9 tensor); `xvel` is what `subtree_linvel` consumes. The Gym-derived
    # configs take both and ignore them.
    #
    # Operand budget: binding these brought the batched env's `extract_kernel`
    # to 15 of the measured Metal cliff of 28 (29 = JIT abort). The remaining
    # suite quantities (site_xpos, subtree_com, cacc, cfrc_int, contacts) fit
    # under it too — add plain operands, do not pack.
    #
    # ANY env wired to a GPU-batched driver MUST override
    # `compute_reward_and_done_gpu`. Inheriting the default there gives a
    # flat-zero reward curve — that is the symptom to look for. All 11
    # Gym-derived configs override it today.
    # =====================================================================

    # === GPU inline: Pre-step hook (per-field tensors; G5) ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
    ):
        """Save per-env state before physics (GPU inline version).

        Write to meta[env, META_IDX_PREV_X] to persist a value for use in
        compute_reward_and_done_gpu.

        ⚠ NO `qvel`. Override `pre_step_full_gpu` below if you need it — this
        signature is implemented by fourteen configs and is not worth widening
        for the one that does.
        """
        # Default: nothing to stash between steps.
        pass

    # === GPU inline: the pre-step hook, WITH qvel ===
    @always_inline
    @staticmethod
    def pre_step_full_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
    ):
        """`pre_step_gpu` plus `qvel` — the hook a state REWRITE needs.

        A SECOND HOOK for the same arithmetic reason `custom_extract_obs_ray_
        gpu` is a second hook: fourteen configs implement `pre_step_gpu`, and
        widening that signature would edit all fourteen to give one of them
        one more argument. This DEFAULTS TO FORWARDING, so a config that only
        stashes a scalar is untouched and unaware.

        ⚠⚠ THE ENV CALLS **ONLY THIS ONE**. That is what makes the forward
        safe: a config overriding the narrow hook is still reached, exactly
        once, through the default below. Calling both would give a config that
        overrode `pre_step_gpu` two invocations per step — and for a hook
        whose whole job is to stash the PREVIOUS value, being called twice is
        silently wrong rather than loud.

        Override THIS one to pin, clamp or zero part of the state before
        physics. `tasks/family_config.So101TabletopConfig` is the first: it
        re-parks the free slots a task did not activate, which needs `qpos`
        for the pose AND `qvel` to stop the body accumulating the velocity the
        integrator keeps giving it.
        """
        Self.pre_step_gpu[DTYPE, BATCH_SIZE, NQ](qpos, meta, env)
        _ = qvel

    # === GPU inline: Unified reward + termination (per-field tensors; G5) ===
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
        """Compute reward and early termination from the per-field GPU state.

        The hook reads exactly the field tensors it needs (joint state, FK
        products, contact forces, CoM velocities, hook metadata, curriculum
        params) — the legacy `[BATCH, STATE_SIZE]` slab + offset ABI died at
        the G5 fields sunset.

        Returns:
            (reward, terminated).
        """
        # Default: NOT IMPLEMENTED — see the section note above.
        return (Scalar[DTYPE](0), False)

    # === GPU: Observation-based termination (for model-based rollouts) ===
    @always_inline
    @staticmethod
    def is_terminal_from_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        env: Int,
    ) -> Bool:
        """Check termination from observations only (no full state access).

        Used by model-based agents (MBPO) during synthetic rollouts where
        only predicted observations are available. Default: no termination.

        Override for envs with observation-based termination conditions.
        Observation layout matches the env's observation vector.
        """
        return False

    # === GPU inline: Non-zero qpos init after reset (per-field; G5) ===
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
        """Apply non-zero initial qpos offsets after noise (default: no-op).

        Override for envs whose initial qpos is non-zero (e.g., Humanoid
        z=1.4 / quat_w=1.0, HumanoidStandup z=0.105). Called by
        _reset_env_gpu after noise has been applied around zero.

        ⚠ MISNOMER, kept for continuity: it writes **qvel too**. `joints`,
        `seed` and `qvel` were all added 2026-08-06 (G10 step 4). Without them
        the hook could express a fixed qpos offset and nothing else — but the
        suite resets are real distributions over the FULL state: walker uses
        `randomizers.randomize_limited_and_rotational_joints` (needs per-joint
        RANGES + a draw) and cartpole draws Gaussian qpos AND qvel. The CPU
        counterpart `custom_reset_cpu` gets all of `Data`; this is the GPU
        hook's equivalent reach. The Gym configs ignore the three new args.

        `seed` is the same value `MODEL_DEF.reset_env_gpu` receives; derive an
        independent stream from it rather than reusing its exact Philox key, or
        the joint angles correlate with the reset noise. The shared helper
        `dm_control/gpu_reset.randomize_limited_and_rotational_joints_gpu`
        does this; prefer it over open-coding a draw.
        """
        pass

    # === GPU inline: initial ACTUATOR ACTIVATION (2026-08-12) ===
    @always_inline
    @staticmethod
    def init_hfield_gpu[
        DTYPE: DType, BATCH_SIZE: Int, NHFIELD_DATA: Int
    ](
        hfield_meta: Tile[DTYPE, MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE],
        hfield_data: Tile[DTYPE, BATCH_SIZE * NHFIELD_DATA],
        env: Int,
        seed: Int,
    ):
        """Rewrite THIS LANE's heightfield grid. Default: leave it alone.

        ⚠⚠ PER LANE, AND THAT IS THE WHOLE REASON THE GRID MOVED INTO `Data`.
        A batched env resets lanes INDEPENDENTLY (`_reset_mask`), so a task
        whose terrain is per-episode — `quadruped escape` — must regenerate
        only the lanes that reset. A shared grid would hand every environment
        whichever terrain reset last, which reads as correlated policy
        behaviour rather than as a bug.

        ⚠ RUNS BEFORE THE NON-CONTACTING-HEIGHT SEARCH, and the order is
        load-bearing: that search raises the body in 1 cm steps until nothing
        touches, so it has to see the terrain it will stand on. Generating
        afterwards spawns the robot inside a hill.

        ⚠ THE DEFAULT IS A NO-OP, NOT A ZERO-FILL. A model whose terrain came
        from a FILE has it in every lane already (`init_hfield_data` at
        construction); zeroing here would flatten it on the first reset.
        """
        pass

    @staticmethod
    def init_act_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NA_F: Int,
    ](
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        seed: Int,
    ):
        """Draw the episode's initial `d->act` (default: leave it at zero).

        ⚠ THE CALLER HAS ALREADY ZEROED `act[env, :]` — that is `mj_resetData`
        and it is NOT this hook's job, so a config that does not override this
        still gets correct MuJoCo reset semantics. Override only when the
        reference's `initialize_episode` draws a NON-ZERO activation.

        ⚠⚠ WHY THIS IS A SEPARATE HOOK RATHER THAN AN ARGUMENT TO
        `init_qpos_gpu`: that signature is implemented by ~20 configs, and
        widening it would touch every one of them to express something only
        dog needs. Cost of the split: two hooks fire per reset instead of one.

        Only relevant to actuators with a `dyntype` — a plain `<motor>` never
        reads `act`. dm_control's dog is the in-scope model that draws one
        (`act[i] = uniform(*ctrlrange[i])` for all 38 `dyntype="filter"`
        actuators, whose force IS `gainprm[0] * act`, so a zero activation
        makes the task materially easier).

        `seed` is the reset seed; derive an independent Philox stream from it
        rather than reusing a key another draw already consumed.
        """
        pass

    # === GPU inline: Custom observation extraction (per-field; G5) ===
    # === GPU: Custom observation extraction, WITH the ray tables ===
    @always_inline
    @staticmethod
    def custom_extract_obs_ray_gpu[
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
        NMESH_TRI_F: Int,
        NHFIELD_DATA: Int,
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
        mesh_meta: Tile[DTYPE, MAX_GPU_MESHES * MODEL_MESH_META_SIZE],
        mesh_tris: Tile[DTYPE, NMESH_TRI_F * MESH_ARENA_FLOATS_PER_TRI],
        hfield_meta: Tile[DTYPE, MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE],
        hfield_data: Tile[DTYPE, BATCH_SIZE * NHFIELD_DATA],
        env: Int,
    ) -> Bool:
        """`custom_extract_obs_gpu` plus the four tables `mj_ray` reads.

        The GPU twin of `custom_extract_obs_ray_cpu`, and a SECOND hook for the
        same arithmetic reason: fifty types implement this trait, and widening
        the existing signature would edit all fifty to give one of them four
        more arguments. This DEFAULTS to forwarding, so a config that does not
        cast rays is untouched and unaware.

        Override THIS one for a `<rangefinder>`, a lidar, or a line-of-sight
        test on the batched path. `dm_control`'s `quadruped escape` is the
        first: twenty rangefinders per lane.

        ⚠ THE FOUR EXTRA TABLES ARE EXACTLY `ray_model`'s TAIL, in its order —
        `mesh_meta`, `mesh_tris`, `hfield_meta`, `hfield_data`. `hfield_data`
        is FLAT with an `env * NHFIELD_DATA` base, which is the offset form
        `ray_model` takes; it is the same buffer `init_hfield_gpu` writes.

        ⚠ COST, BECAUSE IT IS NOT SMALL. `ray_model` is a linear scan over
        every geom and one lane owns one ray, so N rangefinders on G geoms is
        N*G geom queries PER LANE PER STEP. Escape's 20 x 18 is fine. The
        MESH part of that is no longer linear — every mesh now carries a BVH
        (`MESH_META_IDX_BVHNUM`) and `ray_model` takes it — but the GEOM loop
        still is: there is no broad phase over geoms here, and that is the
        next one to notice on a scene with hundreds.

        Returns:
            True if custom extraction was performed (skip model default).
            False to fall back to `custom_extract_obs_gpu`, then to the model
            default.
        """
        return Self.custom_extract_obs_gpu[
            DTYPE, BATCH_SIZE, NQ, NV, NBODY, OBS_DIM, SITE_DIM, MC_F,
            NSITE_F, NGEOM_F, NA_F,
        ](
            qpos, qvel, xpos, xquat, xvel, bodies, site_xpos, contacts,
            sites, geoms, meta, obs, xipos, xangvel, cvel, cacc, cfrc_int,
            subtree_com, site_xpos_acc, xquat_acc, act, env,
        )

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
        """Custom observation extraction (default: False = use model default).

        Override for envs that need non-standard observations (sin/cos
        transforms, body COM positions, etc.).  Return True and write the
        full observation into obs[env, :] to bypass the default
        qpos[obs_qpos_skip:] + qvel[:] extraction.

        Args:
            qpos: Generalized positions, [BATCH_SIZE, NQ].
            qvel: Generalized velocities, [BATCH_SIZE, NV].
            xpos: Body world positions, [BATCH_SIZE, NBODY * 3].
            xquat: Body world orientations [x,y,z,w], [BATCH_SIZE, NBODY * 4].
                Use `kinematics/xmat.xmat_elem_gpu` for MuJoCo `xmat` entries.
            xvel: Body world linear velocities, [BATCH_SIZE, NBODY * 3].
            bodies: Model body records, [NBODY, MODEL_BODY_SIZE].
            site_xpos: Site world positions, [BATCH_SIZE, SITE_DIM].
            contacts: Contact records, [BATCH_SIZE, MC_F * CONTACT_SIZE].
            sites: Model site records, [NSITE_F, MODEL_SITE_SIZE].
            geoms: Model geom records, [NGEOM_F, MODEL_GEOM_SIZE].
            meta: Per-env metadata slots, [BATCH_SIZE, METADATA_SIZE].
            obs: Output observation buffer to write into.
            xipos: Body COM world positions, [BATCH_SIZE, NBODY * 3].
            xangvel: Body world angular velocities, [BATCH_SIZE, NBODY * 3].
            cvel: Body COM-frame spatial velocities, [BATCH_SIZE, NBODY * 6].
            cacc: Body COM-frame spatial accelerations, [BATCH_SIZE, NBODY * 6].
            cfrc_int: Body COM-frame interaction forces, [BATCH_SIZE, NBODY * 6].
            subtree_com: Subtree COM positions, [BATCH_SIZE, NBODY * 3].
            site_xpos_acc: `site_xpos` as it stood when `cacc`/`cfrc_int` were
                written. Acceleration-stage sensors must read this, not the
                post-integration `site_xpos` (see defect 19 in `fields/data`).
            xquat_acc: `xquat` at the same instant as `site_xpos_acc`.
            act: Actuator activations, [BATCH_SIZE, NA_F].
            env: Environment index.

        Returns:
            True if custom extraction was performed (skip model default).
            False to fall back to model's default extraction.
        """
        return False
