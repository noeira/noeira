"""Physics3D GPU constants - Flat buffer layout for GPU kernels.

Primary state is qpos/qvel (joint space). Body positions (xpos, xquat, xipos)
are computed via forward kinematics and stored for collision detection.

State buffer layout per environment:
  [qpos: NQ | qvel: NV | qacc: NV | qfrc: NV |
   xpos: NBODY*3 | xquat: NBODY*4 | xipos: NBODY*3 |
   xvel: NBODY*3 | xangvel: NBODY*3 |
   contacts: MAX_CONTACTS*CONTACT_SIZE | metadata: METADATA_SIZE |
   site_xpos: NSITE*3 |
   cfrc_ext: NBODY*6 | cvel: NBODY*6 | cinert: NBODY*10 | qfrc_actuator: NV]

Model buffer (static, same for all environments):
  Per body (MODEL_BODY_SIZE=28): [mass, inv_mass, inertia(3), inv_inertia(3),
    pos(3), quat(4), parent, ipos(3), iquat(4), rootid, weldid, mocap, gravcomp, dofnum]
  Per joint (MODEL_JOINT_SIZE=27): [type, body_id, qpos_adr, dof_adr,
    pos(3), axis(3), tau_limit, range_min/max, armature, damping, stiffness, springref, frictionloss,
    solref_limit(2), solimp_limit(5), qpos0, margin]
  Metadata (MODEL_META_SIZE): [NBODY, NJOINT, gravity(3), timestep, _reserved(2),
    solref_contact(2), solimp_contact(5), solref_limit(2), solimp_limit(5), impratio, nequality,
    ntendon, nexclude, meaninertia, npair, noslip_tolerance, ccd_tolerance, ccd_iterations,
    ctrl_min, ctrl_max, multiccd_disabled, noslip_iterations,
    solver_iterations, solver_tolerance, ls_iterations, ls_tolerance]
  Curriculum (MODEL_CURRICULUM_SIZE=128): [curriculum parameters / the task
    layer's region table, 8 words a region — tasks/gpu_eval.mojo]
  Per geom (MODEL_GEOM_SIZE=29): [type, body, pos(3), quat(4), radius, half_length,
    half_x/y/z, friction, contype, conaffinity, condim, friction_spin, friction_roll,
    rbound, solref(2), solimp(5), margin]
"""

# =============================================================================
# GPU Configuration
# =============================================================================

comptime TPB: Int = 256  # Threads per block (optimal for most GPUs)
comptime TILE: Int = 8  # Tile size for 2D operations


# =============================================================================
# Physics Defaults
# =============================================================================

comptime DEFAULT_GRAVITY_Z: Float32 = -9.81
comptime DEFAULT_TIMESTEP: Float32 = 0.01
comptime DEFAULT_RESTITUTION: Float32 = 0.0
comptime MAX_POS_CORRECTION_VEL: Float32 = 10.0  # Legacy, unused after accel-level migration


# =============================================================================
# State Buffer Layout - Joint Space (qpos, qvel, qacc, qfrc)
# =============================================================================

# These are computed as offsets based on NQ and NV parameters
# For a system with NQ total qpos and NV total qvel:
#
#   qpos: [0, NQ)
#   qvel: [NQ, NQ + NV)
#   qacc: [NQ + NV, NQ + 2*NV)
#   qfrc: [NQ + 2*NV, NQ + 3*NV)


# =============================================================================
# State Buffer Layout - World Space (xpos, xquat, xvel, xangvel)
# =============================================================================


# =============================================================================
# State Buffer Layout - Contacts
# =============================================================================

# Contact layout (same as Cartesian engine: 12 floats per contact)
comptime CONTACT_SIZE: Int = 30

comptime CONTACT_IDX_BODY_A: Int = 0
comptime CONTACT_IDX_BODY_B: Int = 1
comptime CONTACT_IDX_POS_X: Int = 2
comptime CONTACT_IDX_POS_Y: Int = 3
comptime CONTACT_IDX_POS_Z: Int = 4
comptime CONTACT_IDX_NX: Int = 5
comptime CONTACT_IDX_NY: Int = 6
comptime CONTACT_IDX_NZ: Int = 7
comptime CONTACT_IDX_DIST: Int = 8
comptime CONTACT_IDX_INCLUDEMARGIN: Int = 9
comptime CONTACT_IDX_FORCE_N: Int = 10
comptime CONTACT_IDX_FORCE_T1: Int = 11
comptime CONTACT_IDX_FORCE_T2: Int = 12
comptime CONTACT_IDX_FRICTION: Int = 13
comptime CONTACT_IDX_FRICTION_SPIN: Int = 14
comptime CONTACT_IDX_FRICTION_ROLL: Int = 15
comptime CONTACT_IDX_CONDIM: Int = 16
comptime CONTACT_IDX_FORCE_TORSION: Int = 17
comptime CONTACT_IDX_FORCE_ROLL1: Int = 18
comptime CONTACT_IDX_FORCE_ROLL2: Int = 19
comptime CONTACT_IDX_FRAME_T1_X: Int = 20  # T1 hint for tangent frame (capsule axis)
comptime CONTACT_IDX_FRAME_T1_Y: Int = 21
comptime CONTACT_IDX_FRAME_T1_Z: Int = 22
# ── Per-contact solver parameters, appended 2026-08-03 ──────────────────────
#
# Until now every contact row read ONE MODEL-LEVEL solref/solimp
# (`MODEL_META_IDX_SOLREF_CONTACT_*`), while the record already carried
# per-contact friction, condim and margin. So `<geom solref=... solimp=.../>`
# was parsed, written into the GEOM record (`GEOM_IDX_SOLREF_0`..`SOLIMP_4`)
# and then read by NOTHING — dead data on every build.
#
# These slots hold the pair's MIXED values, computed in the narrow phase by
# MuJoCo's rule (`engine_collision_driver.c:1426-1480`), so the solver reads
# them the same way it already reads friction.
#
# ⚠ APPENDED, NOT INSERTED. Every index 0..22 keeps its value, which is what
# lets a layout change like this be verified as inert: nothing that does not
# read 23..29 can behave differently.
comptime CONTACT_IDX_SOLREF_0: Int = 23  # mixed solref timeconst (or -stiffness)
comptime CONTACT_IDX_SOLREF_1: Int = 24  # mixed solref dampratio (or -damping)
comptime CONTACT_IDX_SOLIMP_0: Int = 25  # mixed solimp dmin
comptime CONTACT_IDX_SOLIMP_1: Int = 26  # mixed solimp dmax
comptime CONTACT_IDX_SOLIMP_2: Int = 27  # mixed solimp width
comptime CONTACT_IDX_SOLIMP_3: Int = 28  # mixed solimp midpoint
comptime CONTACT_IDX_SOLIMP_4: Int = 29  # mixed solimp power


# =============================================================================
# State Buffer Layout - Metadata
# =============================================================================

comptime METADATA_SIZE: Int = 57
"""Per-env metadata words: 4 fixed slots, `META_IDX_TASK_PARAM_0..11`,
`META_IDX_ACTDAMP_LIVE`, `META_IDX_SIM_TIME`, `META_IDX_TASK_ACTIVE`, three
RETIRED words (19..21), `META_IDX_GOAL_HELD`, the four shaping words,
`META_IDX_EQ_FORCE_LIVE`, `META_IDX_LS_EVAL`, the `META_INIT_SLOTS`-word
init block `META_IDX_INIT_REGION_0..12` (29..41), the joint-init block
`META_IDX_JINIT_0` (42..47, `META_JINIT_WORDS` per draw), and the
`META_SOLVER_WORDS` solver counters `META_IDX_NEWTON_ITER..` (48..52), and
the `META_REWARD_WORDS` reward-mode block `META_IDX_REWARD_MODE..` (53..56).

⚠ RAISED FROM 53 TO 57 FOR THE POTENTIAL-BASED REWARD (2026-09-26), by
APPENDING, like every widening before it.

⚠ RAISED FROM 29 TO 42 FOR THE LIBERO DEVICE RESET, by APPENDING. The init
block was three words at 19..21, one per `so101_tabletop` free slot, and
`libero_object` has eleven. Growing it in place would have shifted every word
after it; moving it to the end shifts none, and the three words it left are
retired rather than reused (see the block below).

⚠ RAISED FROM 8 FOR `reassemble_5_bricks_random_order`, which stores TWO
five-entry orders — `desired_order` and `initial_order`, the second because its
relabeling is built from the first entry of it. Four slots were not enough for
even one of them, and the encodings that would have made four enough (a Lehmer
code, base-5 packing, deriving the last entry from the other four) all buy the
space by hiding the layout.

⚠ EVERY INDEX INTO `meta` IS `env * METADATA_SIZE + META_IDX_*`, never a
literal, so widening is a buffer-size change and nothing else. It costs 8
floats per env — `Data.meta` is `alloc(B * METADATA_SIZE)` — which is noise
next to a single body's state."""

comptime META_IDX_NUM_CONTACTS: Int = 0
comptime META_IDX_STEP_COUNT: Int = 1  # Episode step counter for truncation
comptime META_IDX_PREV_X: Int = 2  # Previous x position for velocity computation
comptime META_IDX_PREV_COM_X: Int = 3  # Reserved for prev CoM x (unused with cvel approach)

# ── Per-episode TASK-RANDOMIZED MODEL PARAMETERS (the G4 workaround) ──────
#
# ⚠ WHY THESE LIVE IN PER-ENV STATE AND NOT IN `Model`. A few dm_control
# tasks randomize a MODEL field per episode, not just the state:
# `point_mass-hard` redraws the two fixed-tendon coefficient vectors
# (`model.wrap_prm`) so each control drives a random linear combination of
# root_x/root_y. `Model` is deliberately SHARED across the batch — the design
# batches STATE, not MODEL — so a lane cannot own a different `Model.tendons`
# row without batching the whole record set.
#
# These four slots are the narrow escape: the randomized quantity is a handful
# of floats, it is per-episode (written by `init_qpos_gpu`, which already runs
# per lane at reset), and the ONLY consumer is the config's own actuation
# hook. That last part is a property to CHECK, not to assume — a tendon that
# is also `limited`, spring-loaded, or named in an `<equality>` would be read
# by the SOLVER out of `Model.tendons`, where these writes are invisible. The
# configs that use these slots carry a comptime assert to that effect.
#
# NOT ZEROED between episodes beyond what the writer does: `_reset_env_lane`
# sets `META_IDX_STEP_COUNT` and leaves the rest, exactly as it always did for
# `META_IDX_PREV_X`. A hook that reads a slot it never wrote gets the previous
# episode's value.
# ⚠ ALSO THE HOME OF PER-EPISODE TASK STATE THAT IS NOT A MODEL FIELD. The
# brick tasks' `desired_order` and relabeling live here for the same reason:
# they are per-episode, they are a handful of floats, and `prev_x` — the only
# other per-env scalar — is rewritten every step and would lose them.
comptime META_IDX_TASK_PARAM_0: Int = 4
comptime META_IDX_TASK_PARAM_1: Int = 5
comptime META_IDX_TASK_PARAM_2: Int = 6
comptime META_IDX_TASK_PARAM_3: Int = 7
comptime META_IDX_TASK_PARAM_4: Int = 8
comptime META_IDX_TASK_PARAM_5: Int = 9
comptime META_IDX_TASK_PARAM_6: Int = 10
comptime META_IDX_TASK_PARAM_7: Int = 11
comptime META_IDX_TASK_PARAM_8: Int = 12
comptime META_IDX_TASK_PARAM_9: Int = 13
comptime META_IDX_TASK_PARAM_10: Int = 14
comptime META_IDX_TASK_PARAM_11: Int = 15

# ── `d.dof_actdamp` holds THIS STEP's actuator damping diagonal ───────────
#
# ⚠⚠ THE ACTUATOR VELOCITY DERIVATIVE IS STATE-DEPENDENT, WHICH `Model` CANNOT
# EXPRESS. `mjd_actuator_vel` opens with
#
#     // skip if force is clamped by forcerange
#     if (m->actuator_forcelimited[i]) {
#       mjtNum force = d->actuator_force[i];
#       if (force <= range[0] || force >= range[1]) continue;
#     }                              // engine_derivative.c
#
# — a SATURATED actuator contributes NOTHING to `qDeriv`, because its force is
# pinned at the bound and no longer depends on velocity. Whether it is
# saturated changes every step, so `Model.dof_actdamp` (baked from `kv` at
# build time) is only correct while nothing is clamped.
#
# This flag says "`d.dof_actdamp` was filled this step". `apply_actions_fields`
# sets it; an integrator run without any actuation call leaves it 0 and falls
# back to `Model.dof_actdamp`, which is the right answer there — an unactuated
# model saturates nothing.
comptime META_IDX_ACTDAMP_LIVE: Int = 16

# ⚠⚠ MuJoCo HAS `d->time` AND WE DID NOT. Nothing in the smooth dynamics reads
# it, which is why it was never missed — but `mujoco.pid`'s slew-rate limiter
# does: `Pid::GetState` sets `previous_ctrl_exists = d->time > 0`, so on the
# FIRST step the stored previous control is not a control at all and must not
# clamp anything. A zero in the slot cannot say that; zero is a legal control.
#
# Advanced by `apply_actions_fields` at the END of its body, which runs once
# per control substep — the same cadence `mj_advance` advances `d->time` at.
# ⚠ A CALLER THAT RESETS MUST ZERO IT. `Phyics3dEnv._reset_state` does, beside
# the `act` vector it already clears; a caller that zeroes `act` and not this
# would start the next episode with a previous control of 0 treated as real.
comptime META_IDX_SIM_TIME: Int = 17


# ── WHICH SLOTS OF A TASK FAMILY THIS LANE IS ACTUALLY RUNNING ────────────
#
# A BITMASK over family slot indices, LSB = slot 0, float-encoded. Written by
# `tasks/active.active_mask` (host), decoded by `tasks/obs.slot_active`
# (device).
#
# ⚠⚠ WHY IT IS NOT ONE OF THE TWELVE `TASK_PARAM` WORDS. All twelve are the
# goal tape and `tape.encode_goal` writes EVERY one of them — `OP_NONE` into
# the slots a shorter goal does not use. There is no spare word there; a mask
# parked in one would be erased by the next episode's `encode_goal`.
#
# ⚠ WHY A BITMASK AND NOT ONE WORD PER SLOT. `METADATA_SIZE` is paid by every
# env in the tree, not just by a task family, and the compile ceiling for a
# family is k=13 slots (`docs/TASK_LAYER_IMPLEMENTATION.md` §1.0). Thirteen
# words against one, for a quantity a kernel reads with a shift and an AND.
# Integers stay EXACT in the float that carries them well past that ceiling —
# float32's mantissa is 24 bits, so 2^24 slots — and `active_mask` refuses a
# family wide enough to test it.
#
# ⚠⚠ IT MUST BE REWRITTEN EVERY EPISODE, and the reason is the same property
# that makes it work: `_reset_env_lane` preserves it, so a lane that ran a
# 3-slot task and then a 2-slot one KEEPS the stale third bit. Exactly the trap
# `OP_NONE` exists for in the tape beside it. Zero here means "no slot is
# active", which is a legible state, not a safe default — a lane whose mask was
# never written observes every prop as absent while the physics moves them.
comptime META_IDX_TASK_ACTIVE: Int = 18

# ── WHICH REGION EACH FREE SLOT STARTS IN — one word per free slot ─────────
#
# ⚠⚠ THE RESET NEEDED THIS AND HAD NO WAY TO SAY IT. `init_qpos_gpu` runs per
# lane at reset and is the only place a prop can be PLACED on the GPU path,
# but it was handed no way to learn WHERE: the twelve `TASK_PARAM` words are
# the goal tape and `TASK_ACTIVE` is the mask. So no active free slot was ever
# placed, `_reset_env_lane` restored the composed scene's `qpos0` — the PARK
# pose, 50 m up — and only INACTIVE slots were pinned there afterwards by
# `pre_step_full_gpu`. An ACTIVE prop therefore began every training episode
# in the sky and fell through the whole horizon with its qpos and qvel in the
# observation. The HOST placed props all along (`tasks/sampler` +
# `tasks/reset`), which is why every eval and the viewer looked right and
# nothing saw it until something tried to TRAIN.
#
# ⚠ A REGION INDEX, NOT A POSE. A pose per slot is 7 words each and freezes
# the episode's layout on the host; an index is one word and lets the DEVICE
# draw, which is what makes 1024 lanes cheap and what `tasks/sampler.mojo` was
# written to be callable from.
#
# ⚠⚠ THE WORD IS `region_index + 1`, SO **ZERO MEANS NOT PLACED**. That looks
# like an off-by-one waiting to happen and it is the opposite: `Data.__init__`
# uploads a ZERO-FILLED `meta` to device (`fields/data.mojo`), so zero is what
# every lane reads until a driver writes something. With `-1` as the sentinel
# and 0 meaning `table_top`, a driver that forgot these words would silently
# PLACE every free slot on the table instead of parking it — a real, wrong
# scene, in the tasks that park a prop deliberately. Biasing by one makes the
# untouched value mean the safe thing.
#
# ⚠⚠ ONE WORD PER FREE SLOT, UP TO `META_INIT_SLOTS`, AND THE BLOCK IS
# CONTIGUOUS — three drivers index it as `META_IDX_INIT_REGION_0 + j`. It was
# three words here (19..21), matching `So101TabletopConfig.N_FREE_SLOTS`, and
# `libero_object` has eleven free slots; it now lives at the END of `meta`
# (29..41) so that widening it shifted no other word. 19..21 are RETIRED:
# nothing writes them and nothing may read them, and reusing them for a new
# word would hand an old driver's init words to that word's reader.
#
# ⚠ THE WORD IS NO LONGER ONLY A REGION. `tasks/placement/table.mojo` owns the
# encoding — a region index, the same with the `In` bias, or a STACK onto
# another slot — because the device reset now walks `spec.order_inits` and
# needs all three to agree with the host sampler. `tasks/active.
# init_region_words` is its one writer.
comptime META_INIT_SLOTS: Int = 13
"""Init words per lane: the family compile ceiling of 13 slots
(`docs/TASK_LAYER_IMPLEMENTATION.md` §1.0). A family with more free slots is
refused by `tasks/active.init_region_words`."""

comptime META_IDX_INIT_REGION_0: Int = 29
comptime META_IDX_INIT_REGION_1: Int = 30
comptime META_IDX_INIT_REGION_2: Int = 31

# ── WHICH JOINTS THIS LANE DRAWS AT RESET — `jinit=`, `META_JINIT_WORDS` each ──
#
# ⚠⚠ A DRAWER THAT A TASK OPENS STARTED SHUT ON THE DEVICE. `jinit=` is a
# uniform draw per reset (LIBERO's `OpenCloseSampler`), the host has always made
# it, and no word carried it to `init_qpos_gpu` — so 27 tasks began with their
# drawer or knob at `qpos0` on the device, and a prop drawn INTO a drawer region
# had a frame that only the host knew. `tasks/placement/table.draw_joint_inits`
# reads these.
#
#     [k*3 + 0]  table joint index + 1   (0 = no draw — what a zero meta holds)
#     [k*3 + 1]  lo
#     [k*3 + 2]  hi
#
# ⚠ `lo`/`hi` IN `meta`, NOT IN THE TABLE, because the RANGE is the task's: the
# same microwave joint is drawn in [-2.094, -1.3] by one task and [-0.005, 0.0]
# by another. The JOINT is the family's, so its address is in the table.
#
# ⚠ APPENDED, like the init block, so no word moved. Two draws: the corpus uses
# at most one per task, and `placement/check.joint_init_words` refuses a third.
comptime META_JINIT_SLOTS: Int = 2
comptime META_JINIT_WORDS: Int = 3
comptime META_IDX_JINIT_0: Int = 42

comptime INIT_REGION_NONE: Float64 = 0.0
"""What `META_IDX_INIT_REGION_*` holds for a slot with no `init=` — and what
an untouched `meta` already holds. See the note above."""

# ── DID THIS LANE'S GOAL HOLD ON THE LAST STEP — 1.0 or 0.0 ───────────────
#
# ⚠⚠ THE SUCCESS SIGNAL AND THE REWARD ARE DIFFERENT THINGS AND WERE ONE.
# The reward started sparse — +1 exactly when the goal held — so `reward > 0.5`
# WAS "solved", and three files came to read success that way. Then shaping
# was subtracted from the same scalar, and the whole design had to be bent
# around keeping the penalty under 0.5 so that test kept working: a comptime
# `SHAPE_CLIP`, a bound checked at three call sites, a gate asserting the
# product, and a ceiling on how large any shaping weight could ever be.
#
# That ceiling is what blocked the reward from taking the shape that works on
# this robot — `tolerance` in [0, 1] EVERY step, which alone exceeds 0.5.
#
# So the goal bit gets its own word. The reward is then free to be any shape,
# and success is read from the thing that means success.
#
# ⚠ WRITTEN BY THE REWARD HOOK, which is the one place that already evaluates
# the goal — a second evaluation somewhere else would be a second answer.
comptime META_IDX_GOAL_HELD: Int = 22

# ── THE SHAPED REWARD'S PARAMETERS, PER LANE ──────────────────────────────
#
# ⚠⚠ PER LANE BECAUSE A BATCH RUNS MORE THAN ONE TASK. They lived in
# `curriculum`, which is `[1, MODEL_CURRICULUM_SIZE]` — ONE row for the whole
# batch — and that is right for the region table (a region belongs to the
# FAMILY) and wrong for shaping, because what a weight is worth depends on the
# task's own distance scale. Measured on `so101_tabletop` at identical weights
# and margins:
#
#     task     op      goal dist   goal term   total reward
#     gather   Near        0.139       0.011          0.292
#     lift     Above       0.030       0.811          1.171
#     settle   On          0.000       1.000          1.360
#
# A 4.7x spread in the reward and 91x in the goal term, from the SAME numbers.
# A two-task batch would hand its lanes a bimodal reward, which on this family
# is how a critic is destabilised.
#
# ⚠ ZERO IS "NO SHAPING", AND THAT IS WHAT AN UNTOUCHED `meta` HOLDS. `Data`
# uploads a zero-filled `meta`, so a driver that never writes these gets the
# SPARSE reward — the goal bit and nothing else — rather than a shaped one
# with meaningless parameters. The same bias-toward-safe the init-region words
# use, for the same reason.
#
# ⚠ A MARGIN OF ZERO IS A HARD INDICATOR, not an error: `tolerance` with
# `margin == 0` returns 1 inside the bounds and 0 outside. So a nonzero weight
# with a zero margin is a SPARSE term, which is legible but almost certainly
# not what the caller meant — `tasks.shaping.shaping_words` refuses it.
comptime META_IDX_SHAPE_W_GOAL: Int = 23
comptime META_IDX_SHAPE_W_REACH: Int = 24
comptime META_IDX_GOAL_MARGIN: Int = 25
comptime META_IDX_REACH_MARGIN: Int = 26

# ⚠⚠ HOW MANY EQUALITY ROWS THIS STEP'S SOLVER RETAINED in `Data.efc_eq_force`.
# 0 means none — either the model has none, or this solver does not keep them —
# and 0 is what a fresh `Data` holds, so a step that never solved reads as
# not-live rather than as zero forces.
#
# ⚠ IT IS A COUNT AND NOT A BOOLEAN BECAUSE THE READER CHECKS IT. `cfrc_ext`'s
# equality walk recomputes the row layout from `m.equality` — 1 row per joint
# equality, 3 per connect, 6 per weld, which is MuJoCo's own cursor
# (engine_core_smooth.c:2522) — and a builder that skipped a malformed row
# would leave the two out of step with no symptom but a wrong force. They must
# agree exactly or the walk refuses to run.
#
# `mj_rnePostConstraint` adds each connect/weld row's force into `cfrc_ext`
# (engine_core_smooth.c:2464), which is what the `force`/`torque` sensors read
# through `cfrc_int`. Only the Newton path keeps those forces; PGS, CG and the
# island solver discard them, and a GPU step does not run this stage at all.
# Without a flag the difference is SILENT — a force sensor below a closed loop
# would report the fraction of the load that happens to travel through the
# joint chain — so `_cfrc_ext_env` writes NaN on the two bodies of every
# connect/weld when this is 0. Loud on the sensors that would be wrong, inert
# on every other body.
comptime META_IDX_EQ_FORCE_LIVE: Int = 27

comptime META_IDX_LS_EVAL: Int = 28
"""Line-search evaluations the LAST `mj_forward` spent, summed over the
Newton iterations of this env's solve — MuJoCo's `d->solver[i].neval`
added up.

⚠⚠ IT IS A COUNTER, NOT A RESULT, AND IT EXISTS TO BE GATED. The primal
line search is the solver's hottest inner loop and its ANSWER is nearly
invariant to how it is found: bisection and MuJoCo's three-candidate
bracket land on the same alpha to ~1e-16 on every model in this tree, so
a gate over `qacc` cannot tell them apart while one does 72% more work
(AUD-40: 1207 evaluations against 701 over the same 168 searches). This
word is what makes the difference observable — see
`test_elliptic_linesearch_evals_vs_mujoco`.

⚠ WRITTEN BY ALL THREE NEWTON LEGS since 2026-09-14 (the elliptic per-env
leg, the pyramidal per-env leg, the blocked kernel), through ONE helper —
`newton_solve._publish_solver_counters` — together with the five counter
words below. A reader that finds 0 on a model with contacts is looking at a
solve that did no work (no rows), or at a solver that is not Newton.
"""

# ── THE SOLVER'S COUNTERS — `METADATA_SIZE` 48 -> 53, APPENDED (2026-09-14) ──
#
# What §13.53 of `physics3d/PERFORMANCE.md` asked for: the iteration count
# and the line-search work of every solve, per lane, readable from a driver
# without a print per solve (`NEWTON_ITER_REPORT` serialises the block and
# reports env 0 only). One "last solve" word and four RUNNING SUMS, so a
# driver that reads `meta` once per CONTROL step still sees every substep:
# it differences the sums against its previous read.
#
# ⚠ THE SUMS ARE NEVER RESET BY THE ENGINE. `Data.__init__` uploads a
# zero-filled `meta` and nothing on the device zeroes them afterwards, so
# they count from the allocation. They are `DTYPE` words: exact below 2^24
# in float32 (a 60-step, 25-substep replay at the 100-iteration cap is
# 1.5e5), and a driver that runs longer than that must difference often or
# re-zero them itself (download, zero, upload — between steps).
#
# ⚠ `META_IDX_SOLVER_ACC_NCON` IS THE COUNT THE SOLVE WAS HANDED — `nc`
# clamped to `MAX_CONTACTS` — not the collision kernel's raw count, which
# `META_IDX_NUM_CONTACTS` still holds. The two differ only when the budget
# truncates.
comptime META_IDX_NEWTON_ITER: Int = 48
"""Newton iterations the LAST solve of this env ran (MuJoCo `solver_niter`)."""
comptime META_IDX_SOLVER_ACC_ITER: Int = 49
"""Running sum of `META_IDX_NEWTON_ITER` over every solve since allocation."""
comptime META_IDX_SOLVER_ACC_LSEV: Int = 50
"""Running sum of `META_IDX_LS_EVAL` over every solve since allocation."""
comptime META_IDX_SOLVER_ACC_NCON: Int = 51
"""Running sum of the contact count each solve was handed."""
comptime META_IDX_SOLVER_ACC_CAPPED: Int = 52
"""Running count of solves that ran to the model's `iterations` cap."""
comptime META_SOLVER_WORDS: Int = 5
"""The five words above, contiguous from `META_IDX_NEWTON_ITER`, followed by
the reward block below — `test_device_placement` pins the layout."""

# ── THE REWARD'S MODE AND ITS PER-EPISODE STATE — four words ──────────────
#
# `tasks/family_config.So101FamilyConfig.compute_reward_and_done_gpu` pays the
# LEGACY reward (every shaped term a raw per-step value) when
# `META_IDX_REWARD_MODE` is 0 — what an untouched, zero-filled `meta` holds,
# so every driver and recorder written before this block is unchanged — and
# the POTENTIAL-BASED one when it is 1 (`tasks/shaping.reward_mode_words`,
# `noeira-docs/SO101_PIXEL_RL_PLAN.md`).
#
# ⚠⚠ THE LAST TWO ARE EPISODE STATE AND THE RESET ZEROES THEM
# (`Phyics3dBatchedEnv._reset_env_lane`, `Phyics3dEnv.reset`). A potential
# delta needs the previous step's potential; carried across an episode
# boundary it would pay the jump from the last episode's final state to this
# one's first. ZERO FLAGS MEAN A FRESH EPISODE: the first step pays no shaping
# (it only records its potential) and the bonus is unpaid.
#
# ⚠ `Phi` IS STORED AS ITSELF, WITH A SEPARATE "SET" BIT — not as `Phi + 1`
# with zero meaning unset, which was the first version: `(Phi + 1) - 1` does
# not round-trip (one ulp at Phi ~1.1 in float64), so a policy standing still
# was paid a nonzero dwell, and `test_tower_reward_potential` caught it.
comptime META_IDX_REWARD_MODE: Int = 53
"""0 = legacy raw terms, 1 = potential-based deltas + success lift."""
comptime META_IDX_SUCCESS_BONUS: Int = 54
"""A one-time reward paid on the first step of an episode the goal holds
(mode 1 only); 0 = none."""
comptime META_IDX_PHI_PREV: Int = 55
"""The previous step's potential (valid when `EPISODE_FLAG_PHI_SET` is set)."""
comptime META_IDX_EPISODE_FLAGS: Int = 56
"""Bit field, stored as a float: `EPISODE_FLAG_PHI_SET` | `EPISODE_FLAG_BONUS_PAID`."""
comptime EPISODE_FLAG_PHI_SET: Int = 1
comptime EPISODE_FLAG_BONUS_PAID: Int = 2
comptime META_REWARD_WORDS: Int = 4
"""The four words above, contiguous from `META_IDX_REWARD_MODE`; they END
`meta`."""


# =============================================================================
# Total State Size Computation
# =============================================================================


# =============================================================================
# Model Buffer Layout - Per Body
# =============================================================================

comptime MODEL_BODY_SIZE: Int = 28

comptime BODY_IDX_MASS: Int = 0
comptime BODY_IDX_INV_MASS: Int = 1
comptime BODY_IDX_IXX: Int = 2
comptime BODY_IDX_IYY: Int = 3
comptime BODY_IDX_IZZ: Int = 4
comptime BODY_IDX_INV_IXX: Int = 5
comptime BODY_IDX_INV_IYY: Int = 6
comptime BODY_IDX_INV_IZZ: Int = 7
comptime BODY_IDX_POS_X: Int = 8  # Local position in parent frame
comptime BODY_IDX_POS_Y: Int = 9
comptime BODY_IDX_POS_Z: Int = 10
comptime BODY_IDX_QUAT_X: Int = 11  # Local orientation in parent frame
comptime BODY_IDX_QUAT_Y: Int = 12
comptime BODY_IDX_QUAT_Z: Int = 13
comptime BODY_IDX_QUAT_W: Int = 14
comptime BODY_IDX_PARENT: Int = 15  # Parent body index (-1 for world)
comptime BODY_IDX_IPOS_X: Int = 16  # CoM offset from body origin (body frame)
comptime BODY_IDX_IPOS_Y: Int = 17
comptime BODY_IDX_IPOS_Z: Int = 18
comptime BODY_IDX_IQUAT_X: Int = 19  # Inertia frame quaternion (body frame)
comptime BODY_IDX_IQUAT_Y: Int = 20
comptime BODY_IDX_IQUAT_Z: Int = 21
comptime BODY_IDX_IQUAT_W: Int = 22
comptime BODY_IDX_ROOTID: Int = 23  # Root body index (child of worldbody)
comptime BODY_IDX_WELDID: Int = 24  # Weld body index (MuJoCo body_weldid)
comptime BODY_IDX_MOCAP: Int = 25  # 1.0 if body pose is externally set (mocap)
# ⚠⚠ `<body gravcomp="F">` — the FRACTION OF ITS OWN WEIGHT a body is held up
# by, and the whole of `d->qfrc_passive` on every model in this tree that sets
# it. `mj_gravcomp` (engine_passive.c:817) applies
# `force = -gravity * body_mass[i] * body_gravcomp[i]` at `xipos[i]` through
# `mj_applyFT`, accumulates into `d->qfrc_gravcomp`, and `mj_passive` then adds
# that into `qfrc_passive` for every dof whose joint does NOT set
# `actuatorgravcomp`.
#
# ⚠ IT IS NOT A SMALL TERM. Eight Menagerie models declare it and on
# `hello_robot_stretch_3` — the top of the fidelity board — the peak is
# **46.9 N·m**, against a `qfrc_passive` that is otherwise exactly zero. Read
# as "an arm that sags": without it every gravcomp link falls under its own
# weight and the whole chain diverges from the first step.
#
# Appended (26 -> 27), so every index 0..25 keeps its value.
comptime BODY_IDX_GRAVCOMP: Int = 26
# MuJoCo `body_dofnum`: the body's OWN dof count. Read by the body-pair
# filter's 3.12 rule — two dof-less weld roots exchange no force and are
# skipped (engine_collision_driver.c:296-300; commit ed13bf56, AUD-35).
comptime BODY_IDX_DOFNUM: Int = 27


# =============================================================================
# Model Buffer Layout - Per Joint
# =============================================================================

comptime MODEL_JOINT_SIZE: Int = 27  # +7 for per-joint solref/solimp limits (5 params) + qpos0, +1 margin

comptime JOINT_IDX_TYPE: Int = 0  # JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
comptime JOINT_IDX_BODY_ID: Int = 1
comptime JOINT_IDX_QPOS_ADR: Int = 2
comptime JOINT_IDX_DOF_ADR: Int = 3
comptime JOINT_IDX_POS_X: Int = 4
comptime JOINT_IDX_POS_Y: Int = 5
comptime JOINT_IDX_POS_Z: Int = 6
comptime JOINT_IDX_AXIS_X: Int = 7
comptime JOINT_IDX_AXIS_Y: Int = 8
comptime JOINT_IDX_AXIS_Z: Int = 9
comptime JOINT_IDX_TAU_LIMIT: Int = 10
comptime JOINT_IDX_RANGE_MIN: Int = 11  # Minimum position (radians for hinge, meters for slide)
comptime JOINT_IDX_RANGE_MAX: Int = 12  # Maximum position
comptime JOINT_IDX_ARMATURE: Int = 13  # Rotor inertia (added to M diagonal)
comptime JOINT_IDX_DAMPING: Int = 14  # Passive joint damping
comptime JOINT_IDX_STIFFNESS: Int = 15  # Passive joint stiffness (spring)
comptime JOINT_IDX_SPRINGREF: Int = 16  # Spring reference position (rest position)
comptime JOINT_IDX_FRICTIONLOSS: Int = 17  # Dry friction loss (Coulomb friction)
comptime JOINT_IDX_SOLREF_LIMIT_0: Int = 18  # Per-joint limit solref timeconst
comptime JOINT_IDX_SOLREF_LIMIT_1: Int = 19  # Per-joint limit solref dampratio
comptime JOINT_IDX_SOLIMP_LIMIT_0: Int = 20  # Per-joint limit solimp dmin
comptime JOINT_IDX_SOLIMP_LIMIT_1: Int = 21  # Per-joint limit solimp dmax
comptime JOINT_IDX_SOLIMP_LIMIT_2: Int = 22  # Per-joint limit solimp width
comptime JOINT_IDX_SOLIMP_LIMIT_3: Int = 23  # Per-joint limit solimp midpoint
comptime JOINT_IDX_SOLIMP_LIMIT_4: Int = 24  # Per-joint limit solimp power
comptime JOINT_IDX_QPOS0: Int = 25  # Joint reference position (MuJoCo qpos0 / ref)
# `<joint margin>` — MuJoCo's `jnt_margin`: a limit row is built when
# `dist < margin` and its impedance/aref use `dist - margin` (AUD-03,
# engine_core_constraint.c:1394-1425). NOT angle-converted by the compiler
# (measured: `margin="30"` under angle="degree" stores 30).
comptime JOINT_IDX_MARGIN: Int = 26

# ⚠⚠ HOW AN UNLIMITED JOINT IS ENCODED, AND IT IS NOT MuJoCo'S ENCODING.
# The record has NO `limited` flag: `FlatModelDef`'s `JointData.is_limited` is
# known by the parser and dropped by `fields_build`. An unlimited joint instead
# carries a range of `[-JOINT_RANGE_UNLIMITED, +JOINT_RANGE_UNLIMITED]`, wide
# enough that the limit row can never activate.
#
# MuJoCo stores the opposite: `jnt_range = [0, 0]` with `jnt_limited = 0`. So
# **`range_min < range_max` DOES NOT MEAN "limited" HERE** — it is true for
# every joint in every model. Code ported from a routine that tests MuJoCo's
# `[0, 0]` (or resolves `limited="auto"` that way) reads every unlimited joint
# as limited to +-1e10, which is not a compile error and not a wrong number
# anywhere the range is only a clamp — it goes wrong where the range is used as
# a SAMPLING BOUND.
#
# That is exactly how it was found: `manipulation_reach_config` draws IK retry
# poses uniformly over each arm joint's range, and four of Jaco's six arm
# joints are unlimited. dm_control gives an unlimited HINGE `[0, 2*pi]`
# (`entities/manipulators/base._get_joint_pos_sampling_bounds`); we drew from
# +-1e10, so every retry started from a meaningless pose and the TCP
# initializer exhausted on 7 of 24 resets with 10/10 IK failures. dm_control's
# own IK reaches 30/30 of the same targets in 2.4 attempts.
#
# Test against THIS constant, never against `min < max`.
comptime JOINT_RANGE_UNLIMITED: Float64 = 1e10

# `mjMAXVAL` (`mjmodel.h:25`) — "maximum value in qpos, qvel, qacc". It is the
# ONLY velocity bound MuJoCo has, and it is not a clamp: `mj_checkVel`
# (`engine_forward.c`) tests every dof with `mju_isBad` (NaN, inf, or |v| >
# this) and, on a hit, WARNS and RESETS the whole state — it never quietly
# rescales a velocity into range.
#
# ⚠⚠ THE INTEGRATORS USED TO SATURATE AT 100 rad/s. That is a physics change
# no reference performs, and it is silent: one step out of kinova_gen3's own
# `home` keyframe — where its base and shoulder collision hulls start 12 mm
# interpenetrated — MuJoCo answers |qvel|max 165.583 and ours stopped dead on
# 100.0000. The fingerprint is a joint that lands EXACTLY `100*dt` from where
# it began.
#
# ⚠ IT HAPPENS TO EQUAL `JOINT_RANGE_UNLIMITED` ABOVE, and they are NOT the
# same quantity — that one is this parser's spelling of "no limit", this one
# is MuJoCo's numeric ceiling. Do not fold them.
comptime MJ_MAXVAL: Float64 = 1e10

# MuJoCo 3.11's Newton termination (AUD-39, engine_solver.c:2401-2418,
# 2473-2481): the Newton DECREMENT `0.5*scale*grad'H^-1 grad` is a third exit
# criterion, and on the first pass, gated on the gradient criterion, it is
# the zero-iteration certificate (a warm start already inside tolerance
# takes no line search). It also moves the line-search exit from
# `alpha < 1e-10` to `alpha == 0` exactly, and the improvement exit to
# `improvement > 0 && improvement < tol` — tested after EVERY search and
# never reverting.
#
# ⚠ THE KNOB IS THE COMMON SPELLING OF THREE LEGS, and it has to stay that
# way: the per-env pyramidal and elliptic solvers and the blocked GPU kernel
# each carry their own copy of these exits, because the kernel's rows live in
# SHARED-memory LayoutTensors and cannot call the per-env helpers. A rule
# written inline three times drifts — flipping this to False must return ALL
# THREE to the 3.10 exits together, which is what makes CPU/GPU bit-identity
# a meaningful A/B rather than a comparison of two different solvers.
comptime NEWTON_312_CRITERIA: Bool = True


# =============================================================================
# Model Buffer Layout - Global Metadata
# =============================================================================

comptime MODEL_META_SIZE: Int = 51

comptime MODEL_META_IDX_NBODY: Int = 0
comptime MODEL_META_IDX_NJOINT: Int = 1
comptime MODEL_META_IDX_GRAVITY_X: Int = 2
comptime MODEL_META_IDX_GRAVITY_Y: Int = 3
comptime MODEL_META_IDX_GRAVITY_Z: Int = 4
comptime MODEL_META_IDX_TIMESTEP: Int = 5
# Fluid dynamics parameters (MuJoCo option.density / option.viscosity)
# These occupy the previously-reserved slots 6 and 7.
comptime MODEL_META_IDX_DENSITY: Int = 6  # Fluid mass density (kg/m³), 0 = disabled
comptime MODEL_META_IDX_VISCOSITY: Int = 7  # Fluid dynamic viscosity (Pa·s), 0 = disabled
# solref/solimp contact parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_CONTACT_0: Int = 8  # timeconst
comptime MODEL_META_IDX_SOLREF_CONTACT_1: Int = 9  # dampratio
comptime MODEL_META_IDX_SOLIMP_CONTACT_0: Int = 10  # dmin
comptime MODEL_META_IDX_SOLIMP_CONTACT_1: Int = 11  # dmax
comptime MODEL_META_IDX_SOLIMP_CONTACT_2: Int = 12  # width
comptime MODEL_META_IDX_SOLIMP_CONTACT_3: Int = 13  # midpoint
comptime MODEL_META_IDX_SOLIMP_CONTACT_4: Int = 14  # power
# solref/solimp limit parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_LIMIT_0: Int = 15  # timeconst
comptime MODEL_META_IDX_SOLREF_LIMIT_1: Int = 16  # dampratio
comptime MODEL_META_IDX_SOLIMP_LIMIT_0: Int = 17  # dmin
comptime MODEL_META_IDX_SOLIMP_LIMIT_1: Int = 18  # dmax
comptime MODEL_META_IDX_SOLIMP_LIMIT_2: Int = 19  # width
comptime MODEL_META_IDX_SOLIMP_LIMIT_3: Int = 20  # midpoint
comptime MODEL_META_IDX_SOLIMP_LIMIT_4: Int = 21  # power
# Friction cone model
comptime MODEL_META_IDX_IMPRATIO: Int = 22  # MuJoCo impratio
# ⚠ WHICH SOLVER THE **MODEL** ASKS FOR, as opposed to the one the caller
# happened to build. `<option cone/solver/integrator>` went unparsed until
# 2026-08-19, so on the runtime path there was nothing to compare a built
# integrator against and the studio ran every model on ELLIPTIC + PGS + Euler.
# Appended (34 -> 37), so every index 0..33 keeps its value.
comptime MODEL_META_IDX_CONE: Int = 34  # ConeType, MuJoCo default PYRAMIDAL
comptime MODEL_META_IDX_SOLVER: Int = 35  # SolverType, MuJoCo default NEWTON
comptime MODEL_META_IDX_INTEGRATOR: Int = 36  # IntegratorType, default EULER
# ⚠⚠ THE CONDIM THE **MODEL** NEEDS, versus the `MAX_CONDIM` a caller built.
# `contact_solve` clamps `condim > MAX_CONDIM` down to it SILENTLY, in both
# cone branches — so spot's `condim="6"` feet, which want torsional and rolling
# friction, were solved as plain condim 3 with no indication. The comptime path
# has `ParsedModel.MAX_CONDIM` and every def passes it; the runtime path had no
# equivalent at all, so the studio's hardcoded 3 could not even be compared
# against what the file asked for.
comptime MODEL_META_IDX_MAX_CONDIM: Int = 37  # max geom condim, >= 3
# ⚠⚠ `<option><flag eulerdamp="disable"/></option>` — `mjDSBL_EULERDAMP`.
# `mj_EulerSkip` (engine_forward.c) opens by deciding whether ANY dof has
# damping, and that whole scan is behind
# `if (!mjDISABLED(mjDSBL_EULERDAMP) && !mjDISABLED(mjDSBL_DAMPER))`. With the
# flag set MuJoCo integrates velocity EXPLICITLY — `qvel += h * qacc` — where
# the default path solves `(M + h*diag(B)) qacc' = M qacc` first.
#
# ⚠ IT IS NOT A SMALL CORRECTION. On tetheria (h = 0.01, dof_damping 0.1 and
# M's diagonal ~1.6e-03) `h*B/M` is 0.625, so the implicit solve returns 61.5%
# of the explicit velocity. Running the damping on a model that disables it
# was the whole of that scene's 2.590e-01 residual once its actuators worked.
# `_option_flag_disabled` had existed since the `multiccd` work and simply was
# never asked about this name — the same shape as `<option integrator>` being
# parsed and dispatched on by nobody.
comptime MODEL_META_IDX_EULERDAMP_DISABLED: Int = 38
# `mjModel.ngravcomp` — the number of bodies with `body_gravcomp > 0`
# (`engine_setconst.c:99-104`). Its only job is the early-out at the top of
# `mj_gravcomp`, and it is stored rather than recounted so the pass costs one
# load on the ~77 of 85 scenes that have none.
comptime MODEL_META_IDX_NGRAVCOMP: Int = 39
# `mjModel.opt.noslip_iterations` — how many friction-only Gauss-Seidel
# sweeps `mj_solNoSlip` runs after the primal solve. 0 (MuJoCo's default)
# means the pass does not run at all.
#
# ⚠⚠ THIS SLOT IS WHY THE PASS RUNS AT ALL ON THE RUNTIME PATH. The count
# used to reach the solver ONLY as `solve_newton`'s comptime `NOSLIP_ITER`,
# threaded from `ModelDefFromXML.NOSLIP_ITER` — so every caller that builds a
# `Model` from a `FlatModelDef` at runtime (the studio, and the fidelity
# harnesses that mirror it) stepped with the pass silently off no matter what
# the file said. `FlatModelDef` had no such field to carry, either.
#
# The comptime parameter still exists and still gates whether the code is
# EMITTED — a model that will never want the pass reserves nothing for it —
# but it is now an ENABLE, not the count. The count is this slot, read next to
# MEANINERTIA and NOSLIP_TOLERANCE, the two other numbers the same convergence
# test needs. That keeps the studio's integrator instantiations at five rather
# than ten; `stepping.mojo` already rejected a third comptime dispatch axis
# for `MAX_CONDIM` on exactly these grounds.
#
# ⚠ A 0 HERE REALLY MEANS "NO PASS", and a builder that leaves the slot
# unwritten gets it. That is MuJoCo's default and the right one for a
# hand-made fixture, but it also means a model whose `<option>` sets the
# attribute MUST reach the solver through `build_model_fields_from_flat` —
# `model_def_from_xml` raises when its comptime `NOSLIP_ITER` and this slot
# disagree, so the two parsers cannot drift apart in silence.
comptime MODEL_META_IDX_NOSLIP_ITERATIONS: Int = 40
# Equality constraints
comptime MODEL_META_IDX_NEQUALITY: Int = 23  # Number of equality constraints
# Fixed tendons
comptime MODEL_META_IDX_NTENDON: Int = 24  # Number of fixed tendons
comptime MODEL_META_IDX_NEXCLUDE: Int = 25  # Number of contact exclude pairs
# `mjModel.stat.meaninertia` — the MEAN OF THE MASS-MATRIX DIAGONAL at qpos0,
# armature included (`engine_setconst.c:1139-1146`):
#
#     meaninertia = (1/nv) * sum_i qM[dof_Madr[i]]
#
# Its only consumer is `mj_solNoSlip`'s convergence test, which scales the
# per-iteration improvement by `1 / (meaninertia * max(1, nv))` before
# comparing against `opt.noslip_tolerance`. Getting it wrong does not corrupt
# the sweep, it changes WHEN the sweep stops — measured on dm_control's dog,
# suppressing the early exit entirely moves a 120-step rollout by 2.2e-6 of
# qvel, so the stopping rule is not something to approximate.
comptime MODEL_META_IDX_MEANINERTIA: Int = 26
# Number of `<contact><pair>` records — MuJoCo's `npair`.
comptime MODEL_META_IDX_NPAIR: Int = 27
# `mjModel.opt.noslip_tolerance` — the improvement threshold `mj_solNoSlip`
# breaks on. Carried in META rather than as a comptime parameter because it is
# a plain runtime number the solver reads next to MEANINERTIA, and threading a
# Float64 through env -> integrator -> solver -> kernel would touch every
# caller for something that never needs to be known at compile time.
#
# ⚠ 0 IS A REAL SETTING, NOT "UNSET". dm_control's manipulation models use it
# to mean "run all `noslip_iterations`". Any consumer that treats a 0 here as
# "fall back to the default" reintroduces the truncation this slot fixes.
comptime MODEL_META_IDX_NOSLIP_TOLERANCE: Int = 28
# `mjModel.opt.ccd_tolerance` / `.ccd_iterations` — EPA's stopping rule.
# Defaults 1e-6 and 35 (`mjcPhysics/schema.usda`, and confirmed against the
# 3.10.0 runtime: `m.opt.ccd_tolerance == 1e-06`, `m.opt.ccd_iterations == 35`).
#
# ⚠⚠ THESE WERE HARDCODED AT 1e-8 AND 64, i.e. TIGHTER THAN MUJOCO'S, and a
# model that sets them was ignored outright. That is a parity gap in the
# direction that is easy to mistake for safety: EPA's stopping rule decides
# WHICH boundary face it settles on, and the contact NORMAL is that face's,
# so running longer than the reference does not converge toward it — it
# converges away from it. Measured on Jaco `reach_site_features` pose 38,
# ours at 1e-8 sits 9.4e-3 from MuJoCo's normal and at 1e-6 sits 7.6e-3.
#
# ⚠ MATCHING THE TOLERANCE IS NECESSARY, NOT SUFFICIENT. The polytope
# expansion, face ordering and horizon construction all differ from
# `engine_collision_gjk.c`'s, so identical stopping rules still stop on
# different faces. See `test_epa_optimality_cylinder_mesh`, which gates the
# quantity that IS well-posed.
#
# In META rather than as comptime parameters for the reason
# `MODEL_META_IDX_NOSLIP_TOLERANCE` gives above: plain runtime numbers, read
# next to the geoms they apply to, with no compile-time consumer.
comptime MODEL_META_IDX_CCD_TOLERANCE: Int = 29
comptime MODEL_META_IDX_CCD_ITERATIONS: Int = 30

# The ROOT `<default>`'s motor ctrlrange — the scalar action bounds an env
# advertises through `BoxContinuousActionEnv.action_low/action_high`.
#
# ⚠⚠ A SUMMARY, NOT THE CLAMP. `apply_actions` clamps each actuator to its own
# range; this pair only sizes the box a policy samples from, and it is
# knowingly wrong on models that set ranges per actuator or per default class
# (reach_site_features, quadruped). Kept bit-for-bit as it was — see
# `FlatModelDef.default_motor_ctrl_min`.
#
# Here rather than as comptime members of `ModelDefLike` because phase 1b
# removed the last comptime readers of the MJCF: these were
# `_xml_default_motor_ctrlrange[Self.xml]()`, and a comptime reader of the XML
# is exactly what pins a model to a `String` in Mojo source.
comptime MODEL_META_IDX_CTRL_MIN: Int = 31
comptime MODEL_META_IDX_CTRL_MAX: Int = 32

# `mjDSBL_MULTICCD` (1<<19) — non-zero when the model carries
# `<option><flag multiccd="disable"/></option>`.
#
# ⚠⚠ THE SENSE IS "DISABLED", NOT "ENABLED", AND THAT IS DELIBERATE. MuJoCo's
# multi-point convex manifold is ON by default on the 3.10.0 runtime, so 0 —
# what an unseeded slot and every pre-existing builder gives — is the correct
# default and leaves every model that does not set the flag untouched. Storing
# "enabled" here would have made a zeroed slot silently switch the feature off
# for every hand-made fixture and GPU env spec.
#
# ⚠ WHY IT NEEDS A SLOT AT ALL. `collision/multi_ccd.mojo` implemented the
# default-on behaviour UNCONDITIONALLY, so a model asking for single-point
# convex contacts got a 4-point manifold anyway. Measured on
# `manipulation/reassemble5`: ours 437 contacts against MuJoCo's 111, and 3701
# ms per control step against 13-49 ms. Every dm_control manipulation model
# sets this flag; 9 of the 11 baked ones carry it.
#
# ⚠ `nativeccd` IS PARSED TOO BUT HAS NO SLOT, because it has no consumer.
# `mjDSBL_NATIVECCD` only decides whether `mjc_Convex` takes its early return
# before the perturbation loop, and the native `multicontact()` polygon-clipping
# path that early return protects is NOT PORTED (see `multi_ccd.mojo`'s header).
# With multiccd disabled the loop does not run either way, so on every model in
# this tree the two flags agree. A model setting `nativeccd` ALONE — the baked
# `reach_site_features` and `lift_large_box` do — would route BOX/MESH and
# MESH/MESH pairs into the perturbation loop in MuJoCo while we still exclude
# them, a SMALLER divergence in the opposite direction. Recorded, not fixed.
comptime MODEL_META_IDX_MULTICCD_DISABLED: Int = 33

# MuJoCo's defaults, used wherever a Model is built without a parser (hand-made
# fixtures, the GPU env specs) so that those paths behave like the reference
# rather than like whatever the old constants happened to be.
comptime MJ_CCD_TOLERANCE: Float64 = 1e-6
comptime MJ_CCD_ITERATIONS: Int = 35

# `mjModel.opt.iterations` / `.tolerance` / `.ls_iterations` / `.ls_tolerance`
# — the CONSTRAINT SOLVER's budget and stopping rule. Confirmed against the
# 3.10.0 runtime: 100, 1e-8, 50, 0.01.
#
# ⚠⚠ THESE WERE HARDCODED IN `newton_solve.mojo` (200 / 1e-8 / 50 / 0.01) AND
# A MODEL THAT SET THEM WAS IGNORED. That is not conservatism: MuJoCo's answer
# for a model shipping `<option iterations="4">` IS the 4-iteration answer, and
# running to convergence is a DIFFERENT answer, not a better one.
# `apptronik_apollo` ships exactly that and sits at 1.247e-04 on the board,
# where `convsweep.py` measured MuJoCo's own 4-iteration result 1.1667e-04 away
# from where it lands at 1000 while ours matched the CONVERGED one to 2.2e-16.
# Same shape as `noslip_iterations` (`b6be5c48`): a per-model count that only
# ever reached the solver as a compile-time constant.
#
# ⚠ `iterations` IS A LOOP BOUND AND `Model.__init__` MUST SEED IT.
# `TensorImpl.alloc` does not zero, and a 0 here is "never solve" — a
# hand-built `Data` would step with no constraint forces at all.
comptime MODEL_META_IDX_SOLVER_ITERATIONS: Int = 41
comptime MODEL_META_IDX_SOLVER_TOLERANCE: Int = 42
comptime MODEL_META_IDX_LS_ITERATIONS: Int = 43
comptime MODEL_META_IDX_LS_TOLERANCE: Int = 44

# ⚠⚠ `<option><flag warmstart="disable"/></option>` — `mjDSBL_WARMSTART`.
# DISABLE-SENSE like `MODEL_META_IDX_EULERDAMP_DISABLED`: 0 means MuJoCo's
# default, which is warm start ENABLED.
#
# `warmstart()` (engine_forward.c:786) runs before every constraint solve and
# starts the iterate at the CHEAPER of `qacc_warmstart` (the previous
# `mj_forward`'s answer) and `qacc_smooth`, by primal cost. With the flag set
# it cold-starts at `qacc_smooth` and zeroes `efc_force` — which is exactly
# and only what this engine used to do at all three Newton init sites.
#
# ⚠ IT IS A FIDELITY FEATURE BEFORE IT IS A SPEED ONE. Both engines share the
# FIXED POINT, so a converged solve does not care; a model that TRUNCATES its
# solve does. Measured on `apptronik_apollo` (which ships
# `<option iterations="4">`), `|d(qpos)|` after one step against MuJoCo:
# turning MuJoCo's own warm start OFF moved N=1 from 8.06e-03 to 3.29e-05, a
# factor of 245. The board row was mostly this.
comptime MODEL_META_IDX_WARMSTART_DISABLED: Int = 45

# ⚠ THE NUMBER OF KINEMATIC TREES — the row count of the `trees` record below.
# `trees` is sized `[NV, 3]` because a model cannot have more trees than dofs
# (see `Model.L_TREE`), so the ALLOCATION is an exact comptime bound and this
# is the only thing that says how many of those rows are real. A consumer that
# walks `NV` rows instead of this one reads zeroed padding as a tree at dof 0
# of length 0 — legal-looking and wrong.
comptime MODEL_META_IDX_NTREE: Int = 46

# ⚠ `<option wind>` — A WORLD-FRAME FLUID VELOCITY, NOT A FORCE (AUD-27).
# `mj_inertiaBoxFluidModel` and `mj_ellipsoidFluidModel` both rotate it into
# the body's inertial frame and SUBTRACT it from the body's linear velocity
# (`engine_passive.c:1167-1174, :1239-1248`), so every drag term downstream
# sees the velocity RELATIVE TO THE FLUID. With no wind the two are the same,
# which is why the whole fluid model could be ported without it. Three slots,
# because wind is a 3-vector and `opt.wind` has no angular part — MuJoCo
# builds a 6-vector with a zero rotation half purely to reuse
# `mju_transformSpatial`.
comptime MODEL_META_IDX_WIND_X: Int = 47
comptime MODEL_META_IDX_WIND_Y: Int = 48
comptime MODEL_META_IDX_WIND_Z: Int = 49

# ⚠ `<flag filterparent="disable">` (AUD-34). DISABLE-SENSE like
# `MODEL_META_IDX_EULERDAMP_DISABLED`: 0 is MuJoCo's default, which is that
# the filter is ON. It gates exactly one clause of `pair_body_filtered` — the
# weld-parent one — and nothing else: same-weldbody, both-dof-less and
# `<exclude>` all stay unconditional (`filterBodyPair`,
# engine_collision_driver.c:288-317). A model that disables it WANTS a body to
# collide with its own parent, which is how a gripper pad is made to touch the
# hand it hangs from.
comptime MODEL_META_IDX_FILTERPARENT_DISABLED: Int = 50

comptime MJ_SOLVER_ITERATIONS: Int = 100
comptime MJ_SOLVER_TOLERANCE: Float64 = 1e-8
comptime MJ_LS_ITERATIONS: Int = 50
comptime MJ_LS_TOLERANCE: Float64 = 0.01


# =============================================================================
# Model Buffer Layout - Kinematic TREES (M's diagonal blocks)
# =============================================================================
#
# ⚠⚠ A TREE IS A BLOCK OF THE MASS MATRIX, AND THAT IS WHY IT IS A RECORD.
# M couples a dof only with its tree ANCESTORS (mujoco_warp `io.py:165`), so
# M's diagonal blocks are exactly the trees, and every entry outside a block is
# STRUCTURALLY zero — both CRBA paths guarantee it (`mass_matrix.mojo:592-600`
# walks only ancestor dofs after zeroing M; the dense path at `:219-223` needs
# a body in BOTH subtrees, which no cross-tree pair has). Restricting a
# factorisation to these ranges is therefore not an approximation: subtracting
# exact zeros leaves a sequential accumulation bit-identical.
#
# Dof ranges are contiguous per body and bodies are in tree order, so a tree is
# a single `(dof_adr, dof_num)` interval.
#
# ⚠ `KIND` IS NOT `jnt_type`. A free body is USUALLY a diagonal (COMPACT)
# block and NOT ALWAYS: MuJoCo demands `body_ipos == 0` AND
# `body_iquat == identity` (`body_simple`, `user_model.cc:2814`), because the
# free joint's rotational dofs are about the BODY frame axes
# (`cdof.mojo:330-378`, matching `engine_core_smooth.c:318-337`) and only then
# does the inertia come out diagonal in them. Deciding "free joint => compact"
# is a silent wrong answer on a prop whose COM is offset from its frame.
comptime MODEL_TREE_SIZE: Int = 3

comptime TREE_IDX_DOF_ADR: Int = 0  # First dof of the tree
comptime TREE_IDX_DOF_NUM: Int = 1  # Dof count (the block is [adr, adr+num))
comptime TREE_IDX_KIND: Int = 2  # TREE_KIND_* below

# M restricted to this block has off-diagonal STRUCTURE — factor it.
comptime TREE_KIND_DENSE: Int = 0
# M restricted to this block is DIAGONAL — reciprocals, no factorisation.
#
# ⚠ CLASSIFIED, NEVER ASSUMED FROM THE JOINT TYPE. See the KIND note above.
comptime TREE_KIND_COMPACT: Int = 1


# =============================================================================
# Model Buffer Layout - Unified Geoms (body-attached + static)
# =============================================================================

comptime MODEL_GEOM_SIZE: Int = 35  # Per unified geom (+7 solref/solimp, +1 margin, +1 gap, +1 mesh_id, +1 priority, +1 hfield_id)

comptime GEOM_IDX_TYPE: Int = 0
comptime GEOM_IDX_BODY: Int = 1  # Body index (-1 for static)
comptime GEOM_IDX_POS_X: Int = 2
comptime GEOM_IDX_POS_Y: Int = 3
comptime GEOM_IDX_POS_Z: Int = 4
comptime GEOM_IDX_QUAT_X: Int = 5
comptime GEOM_IDX_QUAT_Y: Int = 6
comptime GEOM_IDX_QUAT_Z: Int = 7
comptime GEOM_IDX_QUAT_W: Int = 8
comptime GEOM_IDX_RADIUS: Int = 9
comptime GEOM_IDX_HALF_LENGTH: Int = 10
comptime GEOM_IDX_HALF_X: Int = 11
comptime GEOM_IDX_HALF_Y: Int = 12
comptime GEOM_IDX_HALF_Z: Int = 13
comptime GEOM_IDX_FRICTION: Int = 14
comptime GEOM_IDX_CONTYPE: Int = 15
comptime GEOM_IDX_CONAFFINITY: Int = 16
comptime GEOM_IDX_CONDIM: Int = 17
comptime GEOM_IDX_FRICTION_SPIN: Int = 18
comptime GEOM_IDX_FRICTION_ROLL: Int = 19
comptime GEOM_IDX_RBOUND: Int = 20
comptime GEOM_IDX_SOLREF_0: Int = 21  # Per-geom solref timeconst
comptime GEOM_IDX_SOLREF_1: Int = 22  # Per-geom solref dampratio
comptime GEOM_IDX_SOLIMP_0: Int = 23  # Per-geom solimp dmin
comptime GEOM_IDX_SOLIMP_1: Int = 24  # Per-geom solimp dmax
comptime GEOM_IDX_SOLIMP_2: Int = 25  # Per-geom solimp width
comptime GEOM_IDX_SOLIMP_3: Int = 26  # Per-geom solimp midpoint
comptime GEOM_IDX_SOLIMP_4: Int = 27  # Per-geom solimp power
comptime GEOM_IDX_MARGIN: Int = 28  # Per-geom contact margin
# ⚠⚠ `<geom gap>` — DETECTED OUT TO `margin + gap`, SOLVED ONLY INSIDE
# `margin`. MuJoCo 3.10.0 hands the narrowphase `margin + gap` as its cutoff
# (`collisionTask`, engine_collision_driver.c:1871) and then passes `margin`
# alone to `mj_setContact` as the contact's `includemargin`, which sets
# `con->exclude = (con->dist >= includemargin)`. So a contact in the band
# [margin, margin + gap) EXISTS — it is reported, it can be read — and
# generates no constraint rows.
#
# ⚠ THAT BAND IS NOT COSMETIC: `<adhesion>` acts on those contacts. flybody's
# eight claw/labrum pads all carry `margin="0.0005" gap="0.0005"`, and at its
# keyframe one pad's ONLY contact sits in the band at dist 9.88e-04. Without
# gap that pad has no contact at all and pulls on nothing.
#
# ⚠ BOTH ARE SUMS OVER THE PAIR, not maxima: `getMargin` returns
# `geom_margin[g1] + geom_margin[g2]` and `getGap` returns
# `geom_gap[g1] + geom_gap[g2]` (engine_collision_driver.c:161/170). An
# explicit `<pair>` overrides BOTH with its own single value.
#
# ⚠⚠ THE REFERENCE TREES DISAGREE AND 3.10.0 IS THE RUNTIME. `includemargin`
# is `margin - gap` in the 3.3.6/3.6.0/main trees and `margin` in 3.10.0 —
# measured on flybody, whose labrum pair (both geoms margin 5e-04, gap 5e-04)
# reports `includemargin` 1e-03, i.e. the SUM of the margins with the gaps
# playing no part. Transcribing from the wrong tree here moves which contacts
# the solver sees on every model with a margin.
comptime GEOM_IDX_GAP: Int = 32  # Per-geom contact gap (see above)
comptime GEOM_IDX_MESH_ID: Int = 29  # Mesh hull index (-1 if not mesh)
# `<geom priority="...">`, default 0. When two geoms differ, the HIGHER
# priority one dictates condim, solref, solimp AND friction wholesale — no
# mixing at all (`engine_collision_driver.c:1427-1438`). dm_control's quadruped
# and dog are the only suite models that set it; quadruped's ball uses it to
# force its own `condim="6"` and `solref="-10000 -30"` onto every contact it
# takes part in, including against the floor.
comptime GEOM_IDX_PRIORITY: Int = 30
# `mjModel.geom_dataid` for a HEIGHTFIELD geom, -1 otherwise. MuJoCo reuses one
# `geom_dataid` slot for meshes and heightfields; this keeps them apart so a
# `mesh_id >= 0` test — which nine collision branches make — cannot pick up an
# hfield.
comptime GEOM_IDX_HFIELD_ID: Int = 31
# ---- what `ray_eliminate` needs (`physics3d/ray/model.mojo`) ---------------
#
# `mj_ray` skips a geom on four rules. Two are runtime arguments (a body to
# exclude, a group mask); the other two are static properties of the model and
# are precomputed into these slots at build time.
#
# `RAY_VISIBLE` is 1 unless the geom is INVISIBLE, which `ray_eliminate`
# spells as two cases that collapse to one boolean because both are fixed at
# compile time: no material and `rgba[3] == 0`, or a material whose
# `rgba[3] == 0`. ⚠ IT IS NOT A RENDER FLAG. A rangefinder passes
# `geomgroup = NULL` and `flg_static = 1`, so this is the ONLY exclusion left
# besides the sensor's own body — an alpha-0 geom is one a ray passes straight
# through, and getting it wrong makes a sensor read a decoration.
comptime GEOM_IDX_RAY_VISIBLE: Int = 33
# `<geom group>`, 0-5. Carried so the group mask can be honoured; the physics
# never reads it.
comptime GEOM_IDX_GROUP: Int = 34


# =============================================================================
# Model Buffer Layout - Equality Constraints
# =============================================================================

comptime MODEL_EQ_SIZE: Int = 22  # Per equality constraint

# ⚠ THE SLOTS ARE REUSED PER TYPE, exactly as MuJoCo reuses `eq_obj1id` /
# `eq_obj2id` / `eq_data` for every `mjtEq`. Read the type FIRST:
#
#   EQ_CONNECT / EQ_WELD : BODY_A/BODY_B are BODY indices, ANCHOR_A* and
#                          ANCHOR_B* are `eq_data[0:3]` / `eq_data[3:6]`.
#   EQ_JOINT             : BODY_A/BODY_B are JOINT indices (BODY_B = -1 for
#                          the single-joint form), and ANCHOR_AX..ANCHOR_BY
#                          are `polycoef[0..4]` — MuJoCo's `eq_data[0:5]`,
#                          the same five floats in the same order.
#
# Naming them BODY_*/ANCHOR_* is a wart inherited from when connect and weld
# were the only types; the alternative was renaming them across nine files
# mid-arc. The rule that matters is: NEVER read these without branching on
# EQ_IDX_TYPE.
comptime EQ_IDX_TYPE: Int = 0  # EQ_CONNECT=0, EQ_WELD=1, EQ_JOINT=2 (mjtEq)
comptime EQ_IDX_BODY_A: Int = 1  # body index, or JOINT index when EQ_JOINT
comptime EQ_IDX_BODY_B: Int = 2  # -1 for world / for a single-joint EQ_JOINT
comptime EQ_IDX_ANCHOR_AX: Int = 3  # EQ_JOINT: polycoef[0]
comptime EQ_IDX_ANCHOR_AY: Int = 4  # EQ_JOINT: polycoef[1]
comptime EQ_IDX_ANCHOR_AZ: Int = 5  # EQ_JOINT: polycoef[2]
comptime EQ_IDX_ANCHOR_BX: Int = 6  # EQ_JOINT: polycoef[3]
comptime EQ_IDX_ANCHOR_BY: Int = 7  # EQ_JOINT: polycoef[4]
comptime EQ_IDX_ANCHOR_BZ: Int = 8
comptime EQ_IDX_RELPOSE_X: Int = 9
comptime EQ_IDX_RELPOSE_Y: Int = 10
comptime EQ_IDX_RELPOSE_Z: Int = 11
comptime EQ_IDX_RELPOSE_W: Int = 12
comptime EQ_IDX_SOLREF_0: Int = 13
comptime EQ_IDX_SOLREF_1: Int = 14
comptime EQ_IDX_SOLIMP_0: Int = 15
comptime EQ_IDX_SOLIMP_1: Int = 16
comptime EQ_IDX_SOLIMP_2: Int = 17
comptime EQ_IDX_SOLIMP_3: Int = 18  # solimp midpoint
comptime EQ_IDX_SOLIMP_4: Int = 19  # solimp power

# Weld only — MuJoCo's `eq_data[10]`, scaling the three ORIENTATION rows.
#
# ⚠ NOT COSMETIC AND NOT ALWAYS 1. MuJoCo applies it twice: to the orientation
# residual (`mju_scl3(cpos+3, quat2+1, torquescale)`) and to the rotational
# Jacobian (`mju_scl(jac+3*NV, ..., torquescale, 3*NV)`), so it scales the
# whole rotational half of the constraint. MJCF defaults it to 1, which is why
# ignoring it went unnoticed — but MetaWorld's `reset_mocap_welds` sets **5.0**,
# so sawyer's weld orientation was 5x too soft against the environment we port.
# Unimplemented until 2026-08-12.
comptime EQ_IDX_TORQUESCALE: Int = 20

# MuJoCo's `eq_objtype` — BODY or SITE semantics. MJCF lets both `connect` and
# `weld` name either two bodies (+ an `anchor` in body1's frame) or two SITES,
# and `mj_instantiateEquality` branches on it: the body form builds the anchor
# as `xmat[b]*eq_data + xpos[b]`, the site form reads `site_xpos` directly and
# ignores `eq_data` entirely (engine_core_constraint.c:448).
#
# WE STORE THE SITE FORM REDUCED TO THE BODY FORM: at parse time a site
# reference becomes `(body = site_bodyid, anchor = site local pos)`, which is
# exactly what `site_xpos` expands to in FK. That keeps the row builder and
# every solver path unchanged. The flag still has to be carried because the
# qpos0 derivation below must NOT run on the site form — MuJoCo zeroes
# `eq_data` there, and re-deriving would overwrite the site offsets with the
# anchor MuJoCo never computed.
comptime EQ_IDX_OBJTYPE: Int = 21  # EQ_OBJ_BODY=0 or EQ_OBJ_SITE=1


# =============================================================================
# Model Buffer Layout - Tendons
# =============================================================================
#
# Indices 0..16 are the ORIGINAL fixed-tendon record and keep their offsets;
# 17..35 were appended 2026-07-31 for dm_control's `ball_in_cup`, the first
# model with a SPATIAL (site-routed) tendon and the first with a tendon LIMIT.
# Same append-don't-renumber discipline the site record used for type+size.
#
# A fixed tendon uses NUM_JOINTS/JOINT_*/COEF_*; a spatial one uses
# NUM_SITES/SITE_*. KIND says which. The two halves are mutually exclusive.
#
# ⚠ IS_EQUALITY exists because `_tendon_env` treats every populated record as a
# BILATERAL EQUALITY (ten_length == LENGTH_REF). That was harmless only while
# `fields_build` hardcoded `ntendon = 0`. humanoid and humanoid_standup both
# declare <fixed> tendons that MuJoCo constrains in NO way, so honestly
# populating the count would have silently welded their hips together. Only
# <equality><tendon> sets this flag; `_tendon_env` skips rows without it.

# ⚠ THE WRAP CAP IS ONE CONSTANT AND EVERY OFFSET BELOW IS DERIVED FROM IT.
# It was 4, hardcoded in five places that had to agree: this layout, the loop
# bounds in `full_parser._fill_tendons`, `TendonData`'s three `Array`s,
# the explicit `JOINT_0..3` writes in `fields_build`, and `TENDON_MAX_JOINTS`
# in `constraints/tendon_limit.mojo`. dog's `caudal_extend` wraps ELEVEN
# joints, so it was silently truncated to four on this path exactly as it was
# on the comptime path before defect 17.
#
# 16 matches `MAX_COMPTIME_TENDON_WRAPS`, deliberately: the two parsers must
# not disagree about how wide a tendon may be, and having them differ is its
# own class of bug (see `feedback_physics3d_two_parser_paths`).
#
# ⚠ THE WRAP SLOTS MUST STAY CONTIGUOUS. Consumers read
# `TENDON_IDX_JOINT_0 + k` / `TENDON_IDX_SITE_0 + k` in a loop
# (`tendon_limit.mojo`, `dynamics/tendon.mojo`), so appending new slots at the
# END of the record instead of widening in place would make k=4 silently read
# COEF_0. That is why this is a renumber rather than an append.
comptime TENDON_MAX_WRAPS: Int = 16

# ⚠⚠ A SEPARATE CAP FOR SPATIAL ROUTING, AND THE SPLIT IS THE POINT.
# `TENDON_MAX_WRAPS` was doing three jobs at once: how many joints a FIXED
# tendon may combine, how many waypoints a SPATIAL one may route through, and
# the stride of the ACTUATOR transmission arrays (`motor_trn_qadr` is sized
# `na * TENDON_MAX_WRAPS`). They shared a number, not a meaning.
#
# iit_softfoot routes a tendon through 39 waypoints — 21 sites and 18 wrap
# geoms — so the spatial cap has to be ~3x what it was. Raising the shared
# constant would have tripled ms_human_700's 700 actuator transmission rows
# for a quantity that has nothing to do with tendon routing. Measured across
# Menagerie's 881 spatial tendons: 726 use under 8 waypoints, 150 use 8-15,
# and 5 (softfoot's) use 32-39.
# ── what a spatial waypoint IS ───────────────────────────────────────────
# MuJoCo's `mjtWrap`, restricted to the kinds we route. Defined HERE rather
# than beside `mju_wrap` because the parser writes these values and the
# dynamics reads them: two spellings of one enum is how a wrap geom ends up
# read as a site.
comptime WRAP_NONE: Int = 0
comptime WRAP_SITE: Int = 1
comptime WRAP_SPHERE: Int = 2
comptime WRAP_CYLINDER: Int = 3
comptime WRAP_PULLEY: Int = 4

comptime TENDON_MAX_SPATIAL_WRAPS: Int = 48

# 24 scalar fields, two FIXED runs (joint, coef) and three SPATIAL runs
# (wrap object, wrap type, wrap parameter).
comptime MODEL_TENDON_SIZE: Int = (
    24 + 2 * TENDON_MAX_WRAPS + 3 * TENDON_MAX_SPATIAL_WRAPS
)

comptime TENDON_IDX_NUM_JOINTS: Int = 0
comptime TENDON_IDX_JOINT_0: Int = 1
comptime TENDON_IDX_JOINT_1: Int = TENDON_IDX_JOINT_0 + 1
comptime TENDON_IDX_JOINT_2: Int = TENDON_IDX_JOINT_0 + 2
comptime TENDON_IDX_JOINT_3: Int = TENDON_IDX_JOINT_0 + 3
comptime TENDON_IDX_COEF_0: Int = TENDON_IDX_JOINT_0 + TENDON_MAX_WRAPS
comptime TENDON_IDX_COEF_1: Int = TENDON_IDX_COEF_0 + 1
comptime TENDON_IDX_COEF_2: Int = TENDON_IDX_COEF_0 + 2
comptime TENDON_IDX_COEF_3: Int = TENDON_IDX_COEF_0 + 3
comptime TENDON_IDX_LENGTH_REF: Int = TENDON_IDX_COEF_0 + TENDON_MAX_WRAPS
comptime TENDON_IDX_SOLREF_0: Int = TENDON_IDX_LENGTH_REF + 1
comptime TENDON_IDX_SOLREF_1: Int = TENDON_IDX_LENGTH_REF + 2
comptime TENDON_IDX_SOLIMP_0: Int = TENDON_IDX_LENGTH_REF + 3
comptime TENDON_IDX_SOLIMP_1: Int = TENDON_IDX_LENGTH_REF + 4
comptime TENDON_IDX_SOLIMP_2: Int = TENDON_IDX_LENGTH_REF + 5
comptime TENDON_IDX_SOLIMP_3: Int = TENDON_IDX_LENGTH_REF + 6  # solimp midpoint
comptime TENDON_IDX_SOLIMP_4: Int = TENDON_IDX_LENGTH_REF + 7  # solimp power

# --- appended 2026-07-31 (spatial routing + limits) --------------------------

comptime TENDON_KIND_FIXED: Int = 0
comptime TENDON_KIND_SPATIAL: Int = 1

comptime TENDON_IDX_KIND: Int = TENDON_IDX_SOLIMP_4 + 1  # TENDON_KIND_*
comptime TENDON_IDX_IS_EQUALITY: Int = TENDON_IDX_KIND + 1  # 1 => `_tendon_env` owns this row
comptime TENDON_IDX_NUM_WRAPS: Int = TENDON_IDX_KIND + 2  # spatial only

# ── the spatial routing sequence, three parallel runs ─────────────────────
# MuJoCo's `wrap_type` / `wrap_objid` / `wrap_prm`, flattened into the tendon
# record. Entry `k` is a SITE (`WRAP_SITE`, obj = site id, prm unused) or a
# WRAP GEOM (`WRAP_SPHERE`/`WRAP_CYLINDER`, obj = geom id, prm = the sidesite
# id or -1).
#
# ⚠ THE TYPE RUN IS NOT REDUNDANT WITH THE OBJECT RUN. A site id and a geom
# id are both non-negative integers indexing different tables; without the
# type, entry `k` reads as a site whose position happens to be a geom's, and
# the tendon quietly routes through the wrong point rather than around the
# object. The previous layout had no type run because everything was a site.
comptime TENDON_IDX_WOBJ_0: Int = TENDON_IDX_KIND + 3
comptime TENDON_IDX_WTYPE_0: Int = (
    TENDON_IDX_WOBJ_0 + TENDON_MAX_SPATIAL_WRAPS
)
comptime TENDON_IDX_WPRM_0: Int = (
    TENDON_IDX_WTYPE_0 + TENDON_MAX_SPATIAL_WRAPS
)
comptime TENDON_IDX_LIMITED: Int = (
    TENDON_IDX_WPRM_0 + TENDON_MAX_SPATIAL_WRAPS
)
comptime TENDON_IDX_RANGE_MIN: Int = TENDON_IDX_LIMITED + 1
comptime TENDON_IDX_RANGE_MAX: Int = TENDON_IDX_LIMITED + 2
comptime TENDON_IDX_MARGIN: Int = TENDON_IDX_LIMITED + 3
# J M^-1 J^T at qpos0 — the limit row's diagApprox (engine_setconst.c:256).
comptime TENDON_IDX_INVWEIGHT0: Int = TENDON_IDX_LIMITED + 4
# The LIMIT solref/solimp pair, distinct from the equality pair above
# (MuJoCo keeps tendon_solref_lim separate from tendon_solref_fri).
comptime TENDON_IDX_SOLREF_LIM_0: Int = TENDON_IDX_LIMITED + 5
comptime TENDON_IDX_SOLREF_LIM_1: Int = TENDON_IDX_LIMITED + 6
comptime TENDON_IDX_SOLIMP_LIM_0: Int = TENDON_IDX_LIMITED + 7
comptime TENDON_IDX_SOLIMP_LIM_1: Int = TENDON_IDX_LIMITED + 8
comptime TENDON_IDX_SOLIMP_LIM_2: Int = TENDON_IDX_LIMITED + 9
comptime TENDON_IDX_SOLIMP_LIM_3: Int = TENDON_IDX_LIMITED + 10
comptime TENDON_IDX_SOLIMP_LIM_4: Int = TENDON_IDX_LIMITED + 11

# ⚠ `TENDON_MAX_SITES` IS GONE, not renamed: it was an alias of
# `TENDON_MAX_WRAPS` from when a waypoint could only be a site. The spatial
# cap is `TENDON_MAX_SPATIAL_WRAPS` above and it is a DIFFERENT NUMBER now.


# =============================================================================
# Model Buffer Layout - Curriculum Parameters
# =============================================================================

# Fixed-size curriculum section (environments use what they need)
# ⚠ WIDENED FROM 8 TO 128 FOR THE TASK LAYER'S REGION TABLE (L3). A region
# costs `tasks/gpu_eval.REGION_WORDS` = 8 words (site, rect, half-height,
# box flag, contact body) and LIBERO's `libero_goal` family declares twelve;
# eight words held exactly one. The buffer is ONE row for the whole batch
# (`fields/model.mojo` allocates MODEL_CURRICULUM_SIZE floats), so this
# costs 480 bytes once, not per lane. Every reader indexes by name and the
# first eight words keep their meaning, so the dm_control/Gym configs that
# read `curriculum[0, 0..7]` are unchanged.
comptime MODEL_CURRICULUM_SIZE: Int = 128

# Generic curriculum parameter indices (environments define their own semantics)
comptime CURRICULUM_IDX_PARAM_0: Int = 0
comptime CURRICULUM_IDX_PARAM_1: Int = 1
comptime CURRICULUM_IDX_PARAM_2: Int = 2
comptime CURRICULUM_IDX_PARAM_3: Int = 3
comptime CURRICULUM_IDX_PARAM_4: Int = 4
comptime CURRICULUM_IDX_PARAM_5: Int = 5
comptime CURRICULUM_IDX_PARAM_6: Int = 6
comptime CURRICULUM_IDX_PARAM_7: Int = 7


# =============================================================================
# Model Buffer Layout - Sites
# =============================================================================

# Site layout: [body_idx, pos(3), type, size(3), quat(4)]
#
# type + size were appended 2026-07-29 for the `touch` sensor, which needs the
# site's ZONE (MuJoCo casts a ray from each contact point along the contact
# normal and asks whether it hits the site volume). `SiteData` carried both all
# along; only the serialized record was truncated. Appending keeps every
# existing `SITE_IDX_*` offset put.
#
# quat was appended 2026-08-01 for manipulator, whose `thumb_touch` /
# `finger_touch` zones are BOXES carrying `euler="0 15 0"`. A box zone is
# orientation-dependent, so the sphere-only scope that let the record ship
# without an orientation ended there. Three files had been substituting the
# site's BODY quaternion in the meantime and saying so in their docstrings —
# `sensors/touch.mojo`, `sensors/frame_vel.mojo`, `sensors/site_acc.mojo`.
# Stored (x, y, z, w), the order `BODY_IDX_QUAT_*` and `GEOM_IDX_QUAT_*`
# already use — MuJoCo's own `site_quat` is (w, x, y, z), so a parity test
# reading both has to reorder.
comptime MODEL_SITE_SIZE: Int = 12  # body + pos(3) + type + size(3) + quat(4)

comptime SITE_IDX_BODY: Int = 0  # Body index the site is attached to
comptime SITE_IDX_POS_X: Int = 1  # Local position in body frame
comptime SITE_IDX_POS_Y: Int = 2
comptime SITE_IDX_POS_Z: Int = 3
comptime SITE_IDX_TYPE: Int = 4  # GEOM_* code (sphere/capsule/box/...)
comptime SITE_IDX_SIZE_0: Int = 5  # radius, or half-x for a box
comptime SITE_IDX_SIZE_1: Int = 6  # half-length, or half-y
comptime SITE_IDX_SIZE_2: Int = 7  # half-z (box only)
comptime SITE_IDX_QUAT_X: Int = 8  # Local orientation in body frame
comptime SITE_IDX_QUAT_Y: Int = 9
comptime SITE_IDX_QUAT_Z: Int = 10
comptime SITE_IDX_QUAT_W: Int = 11


# =============================================================================
# Sensor records — `[NSENSOR, MODEL_SENSOR_SIZE]`
# =============================================================================
#
# One row per `<sensor>` element, in DECLARATION ORDER, which is also the order
# `sensordata` is laid out in. The columns are MuJoCo's `m->sensor_*` arrays
# and carry its numbering exactly (`physics3d/constants.mojo` names the enums);
# `test_sensor_table_vs_mujoco` compares every one of them against a live
# `MjModel`.
#
# ⚠ A ROW EXISTS FOR A SENSOR THIS ENGINE DOES NOT COMPUTE. `SENSOR_IDX_SERVED`
# is 0 for those. They are here so that `SENSOR_IDX_ADR` stays a true prefix
# sum — drop one and every sensor after it reports an offset pointing at some
# other sensor's values, which is a plausible wrong number rather than an
# error. See `_fill_sensors` for why neither skipping nor refusing is right.
comptime MODEL_SENSOR_SIZE: Int = 15  # ⚠ COUNT THE IDX LINES BELOW: 0..14

comptime SENSOR_IDX_TYPE: Int = 0  # `mjtSensor`, e.g. SENS_TOUCH
comptime SENSOR_IDX_OBJTYPE: Int = 1  # `mjtObj` — SENSOBJ_SITE / BODY / GEOM …
comptime SENSOR_IDX_OBJID: Int = 2  # index into THAT objtype's table; -1 unserved
comptime SENSOR_IDX_DIM: Int = 3  # values written to sensordata
comptime SENSOR_IDX_ADR: Int = 4  # offset into sensordata
comptime SENSOR_IDX_DATATYPE: Int = 5  # `mjtDataType` — how cutoff clamps
comptime SENSOR_IDX_NEEDSTAGE: Int = 6  # `mjtStage` — which eval pass
comptime SENSOR_IDX_CUTOFF: Int = 7
comptime SENSOR_IDX_BODY: Int = 8  # the site's body, or the subtree root
comptime SENSOR_IDX_SERVED: Int = 9  # 1 if a kernel computes it, else 0
# `<sensor reftype= refname=>` — the frame sensors' optional RELATIVE form
# (`m->sensor_reftype` / `sensor_refid`). `SENSOBJ_UNKNOWN` / `-1` is MuJoCo's
# own "absent", and `engine_sensor.c:692` branches on `refid == -1` exactly.
#
# ⚠ IGNORING THESE IS NOT A SMALL ERROR. A `framepos` with a reference frame
# reports a position IN THAT FRAME; served without it, the sensor would hand
# back the global position — right units, right magnitude, wrong frame.
comptime SENSOR_IDX_REFTYPE: Int = 10  # `mjtObj` of the reference, or UNKNOWN
comptime SENSOR_IDX_REFID: Int = 11  # index of the reference object, or -1

# `<option magnetic>` — the world-frame magnetic field a `<magnetometer>`
# reports, rotated into its site's frame (`engine_sensor.c:538`). MuJoCo's
# default is (0, -0.5, 0), which is NOT zero: a model that declares the
# sensor and no `<option>` still reads a real field.
#
# ⚠⚠ A GLOBAL, STORED PER ROW, AND THAT IS DELIBERATE. `m->opt.magnetic` has
# exactly ONE consumer in the whole reference (grep: `engine_sensor.c:538`),
# so there is nothing else to share it with, and the natural home —
# `Model.meta`, where `<option wind>` lives — is NOT bound by the sensor
# stage kernel. Binding it would take that kernel from 25 buffers to 26,
# against a Metal argument table that fails with NO DIAGNOSTIC at 29 (see
# `sensors/eval.mojo`). Three columns on a row that already exists cost
# nothing and risk nothing. The parser fills all three from one source, so
# the copies cannot disagree.
comptime SENSOR_IDX_MAG_X: Int = 12
comptime SENSOR_IDX_MAG_Y: Int = 13
comptime SENSOR_IDX_MAG_Z: Int = 14


comptime MODEL_EXCLUDE_PAIR_SIZE: Int = 2  # body1, body2


# =============================================================================
# Model Buffer Layout - Predefined contact pairs (`<contact><pair>`)
# =============================================================================
#
# One record per `<contact><pair>`, mirroring MuJoCo's `m->pair_*` arrays.
#
# ⚠ A PREDEFINED PAIR IS NOT A FILTERED PAIR. It collides UNCONDITIONALLY:
# `mj_collideGeoms` skips the contype/conaffinity test whenever `ipair >= 0`
# (`engine_collision_driver.c:1583`), and the whole merge loop runs BEFORE the
# `canCollide2` / `exclude_signature` tests at `:398-412`. Confirmed against the
# 3.10.0 runtime, which emits the contact for every one of: masks cleared to
# `contype=0 conaffinity=0`, an `<exclude>` naming both bodies, two geoms on the
# SAME body, and a welded parent/child. So the pair path must bypass
# `pair_body_filtered` AND the mask AND the plane-vs-world skips — not just one
# of them.
#
# ⚠ THE PARAMETERS ARE PLAIN DEFAULTS, NOT DERIVED FROM THE TWO GEOMS.
# `mjCPair::Compile` (`user/user_objects.cc`) reads as though an omitted
# attribute is filled in from the geoms — max margin, max gap, max condim, max
# friction, solmix-weighted solref/solimp. That code is DEAD on the XML path:
# `mjs_defaultPair` (`user/user_init.c`) memsets the spec and writes concrete
# defaults (condim 3, friction 1/1/0.005/1e-4/1e-4, `mj_defaultSolRefImp`), so
# `mjuu_defined()` is true for every field and no derivation branch is ever
# taken. Measured on 3.10.0 with two geoms deliberately given DIFFERENT solref,
# friction, margin and condim:
#
#     dynamic geom pair -> condim 6, friction 1.5,  solref 0.0125  (mixed)
#     <pair> no attrs   -> condim 3, friction 1.0,  solref 0.02    (defaults)
#
# Transcribing `mjCPair::Compile` would therefore have silently given every
# attribute-less pair the WRONG friction and condim. ToddlerBot's `scene*.xml`
# pairs carry `geom1`/`geom2` and nothing else, so this is the path that matters
# and the error would have been invisible in the geometry.
#
# ⚠ `gap` IS STORED NOW, AND 3.10.0'S RULE IS THE ONE IMPLEMENTED.
# `mj_setContact` is called with `margin-gap` in the 3.3.6, 3.6.0 and main
# trees; the 3.10.0 RUNTIME passes `margin` alone and hands the narrowphase
# `margin + gap` as its cutoff, and 3.11.0 differs again — three behaviours
# across the versions, which is why this was rejected outright until the rule
# was measured against the runtime rather than transcribed. See `GEOM_IDX_GAP`
# for that measurement.
comptime MODEL_PAIR_SIZE: Int = 15

comptime PAIR_IDX_GEOM1: Int = 0  # Geom index (compiler-sorted, g1 < g2)
comptime PAIR_IDX_GEOM2: Int = 1
comptime PAIR_IDX_CONDIM: Int = 2
comptime PAIR_IDX_FRICTION: Int = 3  # Sliding (MuJoCo pair_friction[0..1])
comptime PAIR_IDX_FRICTION_SPIN: Int = 4  # Torsional (pair_friction[2])
comptime PAIR_IDX_FRICTION_ROLL: Int = 5  # Rolling (pair_friction[3..4])
comptime PAIR_IDX_SOLREF_0: Int = 6
comptime PAIR_IDX_SOLREF_1: Int = 7
comptime PAIR_IDX_SOLIMP_0: Int = 8
comptime PAIR_IDX_SOLIMP_1: Int = 9
comptime PAIR_IDX_SOLIMP_2: Int = 10
comptime PAIR_IDX_SOLIMP_3: Int = 11
comptime PAIR_IDX_SOLIMP_4: Int = 12
comptime PAIR_IDX_MARGIN: Int = 13
# `<pair gap>` — the pair's own, overriding the geom sum. See `GEOM_IDX_GAP`.
comptime PAIR_IDX_GAP: Int = 14


# =============================================================================
# Model Buffer Layout - Actuation (phase 1a.2)
# =============================================================================

# The runtime replacement for `ComptimeActData`'s actuator arrays — the record
# layout behind `fields/spec_fields.mojo::SpecFields`. Everything BOTH
# `ModelDefFromXML.apply_actions` (CPU) and `apply_actions_kernel_gpu` read.
#
# ⚠ THESE ARE DELIBERATELY NOT PART OF `fields.Model`. `Model` is the operand
# bundle the integrator / solver / collision kernels bind; actuation is read by
# exactly one function per target and by nothing else. Widening `Model` would
# also have added a fifteenth type parameter to a struct named in 48 files,
# every one of which would have had to thread `NACT` through to keep compiling.
#
# ⚠ THE WRAP STRIDE IS `TENDON_MAX_WRAPS`, SHARED WITH THE TENDON RECORD AND
# WITH `FlatModelDef.motor_trn_*` (which is indexed `ai * TENDON_MAX_WRAPS + k`
# already). The comptime twin uses `_WRAPS`, which collapses to 1 on a model
# with no tendons — so the two strides AGREE ONLY WHEN THE MODEL HAS TENDONS.
# Anything diffing the two must convert; the equivalence gate does.
comptime MODEL_ACTUATOR_SIZE: Int = 33 + 3 * TENDON_MAX_WRAPS

comptime ACT_IDX_KIND: Int = 0  # ACT_KIND_*
comptime ACT_IDX_GEAR: Int = 1
comptime ACT_IDX_CTRL_MIN: Int = 2
comptime ACT_IDX_CTRL_MAX: Int = 3
# ⚠ READ THIS BEFORE READING THE RANGE. MuJoCo's `ctrllimited` defaults to
# "auto", so an actuator declaring no range is UNLIMITED and the stored range
# is a (-1, 1) fallback nobody should clamp to.
comptime ACT_IDX_CTRL_LIMITED: Int = 4
comptime ACT_IDX_FORCE_MIN: Int = 5
comptime ACT_IDX_FORCE_MAX: Int = 6
comptime ACT_IDX_FORCE_LIMITED: Int = 7
# MuJoCo `gainprm[0]` and `-biasprm[2]`. INDEPENDENT — `<velocity>` happens to
# set both to K, but `gainprm="5 0 0" biasprm="0 0 -3"` is legal.
# ⚠ `kp` DEFAULTS TO 1, NOT 0: a plain `<motor>` never writes it and its force
# is `kp * ctrl`, so a zero here silently disables every bare motor.
comptime ACT_IDX_KP: Int = 8
comptime ACT_IDX_KV: Int = 9
# mjDYN_FILTER: `act_dot = (ctrl - act) / dyn_tau`. `act_adr >= 0` means this
# actuator owns one activation variable at that index; -1 means none.
comptime ACT_IDX_DYN_TAU: Int = 10
comptime ACT_IDX_ACT_ADR: Int = 11
comptime ACT_IDX_TRN_N: Int = 12  # 0 => no resolvable transmission, skip
comptime ACT_IDX_DOF_ADR: Int = 13  # the single dof the actuator reports on
comptime ACT_IDX_TENDON_ID: Int = 14  # -1 unless a `tendon=` transmission
comptime ACT_IDX_JOINT_ID: Int = 15  # -1 unless a `joint=` transmission
# The transmission triples, `+ k` for k in [0, TRN_N). A `joint=` actuator is
# ONE triple with coef 1; a `tendon=` one copies the tendon's whole wrap list.
comptime ACT_IDX_TRN_QADR_0: Int = 16
comptime ACT_IDX_TRN_DADR_0: Int = ACT_IDX_TRN_QADR_0 + TENDON_MAX_WRAPS
comptime ACT_IDX_TRN_COEF_0: Int = ACT_IDX_TRN_QADR_0 + 2 * TENDON_MAX_WRAPS
# ⚠⚠ APPENDED AFTER THE TRIPLE BLOCK, so every index 0..15 and the whole
# `[16, 16 + 3*TENDON_MAX_WRAPS)` span keep their meaning. `site=`
# (`mjTRN_SITE`) has no triple — its moment is the site Jacobian at the
# CURRENT pose — so it stores the site and the full six-component gear
# instead, and `dynamics/pose_transmission.mojo` reads them.
#
# ⚠ `GEAR_1..5` ARE ONLY MEANINGFUL ON A SITE TRANSMISSION. For `joint=` and
# `tendon=` MuJoCo uses `gear[0]` alone; here the six are a WRENCH in the site
# frame (force, then torque), and the whole of it is baked into the moment —
# so a site actuator's force must NOT be multiplied by `ACT_IDX_GEAR` again
# the way a joint or fixed-tendon one is.
comptime ACT_IDX_SITE_ID: Int = ACT_IDX_TRN_QADR_0 + 3 * TENDON_MAX_WRAPS
comptime ACT_IDX_GEAR_1: Int = ACT_IDX_SITE_ID + 1
comptime ACT_IDX_GEAR_2: Int = ACT_IDX_SITE_ID + 2
comptime ACT_IDX_GEAR_3: Int = ACT_IDX_SITE_ID + 3
comptime ACT_IDX_GEAR_4: Int = ACT_IDX_SITE_ID + 4
comptime ACT_IDX_GEAR_5: Int = ACT_IDX_SITE_ID + 5

# ⚠⚠ `biasprm[0]` AND `biasprm[1]`, WHICH THIS RECORD DID NOT CARRY. The force
# law MuJoCo evaluates (`mj_fwdActuation`, engine_forward.c:571-628) is
#
#     force = gain * u  +  (biasprm[0] + biasprm[1]*length + biasprm[2]*vel)
#
# and `ACT_IDX_KP` / `ACT_IDX_KV` are `gainprm[0]` and `-biasprm[2]`. With
# nowhere to put the other two, `apply_actions` reconstructed the bias from
# the KIND instead — POSITION assumed `biasprm[1] == -gainprm[0]`, VELOCITY
# assumed `biasprm[1] == 0` — which is true of every `<position>` and
# `<velocity>` ELEMENT by construction and NOT true of a `<general>` that
# writes the two independently.
#
# ⚠ FIVE Menagerie scenes write exactly that: franka_emika_panda,
# robotiq_2f85, robotiq_2f85_v4, stanford_tidybot and ufactory_xarm7, each
# one gripper actuator whose `ctrlrange` is remapped to travel units —
# panda's is `gainprm="0.01568627451 0 0" biasprm="0 -100 -10"`, i.e. a gain
# of 1/64 against a position feedback of 100. We answered its two finger dofs
# -0.00031373 where MuJoCo answers **-2.0**, a factor of 6375.
#
# ⚠ THE PARSER ALREADY KNEW. `_fill_actuators` records
# `bad_actuator_code = 3` for "biasprm[1] not in {-gain, 0}" — and the
# COMPTIME path raises on it while the RUNTIME path reads the field nowhere,
# so every Menagerie model with one loaded silently. A detected-and-unread
# diagnostic is not a diagnostic.
#
# Appended after the site block, so indices 0..15, the triple span and
# `SITE_ID + 0..5` all keep their meaning.
comptime ACT_IDX_BIAS0: Int = ACT_IDX_SITE_ID + 6
comptime ACT_IDX_BIAS1: Int = ACT_IDX_SITE_ID + 7

# ── `<plugin plugin="mujoco.pid">` — a force law with no gainprm at all ───
#
# ⚠⚠ THE PLUGIN'S GAINS ARE NOT IN `gainprm`/`biasprm`. `mjModel` stores an
# actuator plugin's parameters as NUL-separated TEXT in `plugin_attr`, keyed by
# `plugin_attradr[instance]`, and `gaintype`/`biastype` stay at their defaults
# (FIXED / NONE) — so an engine reading only the gain/bias record computes
# `force = 1 * ctrl` for a PID servo. shadow_dexee's twelve actuators are all
# of this shape; we parsed none of them and its `nu` read 0 against MuJoCo's
# 12, which is a MISALIGNED CONTROL VECTOR, not a small force error.
#
# `plugin/actuator/pid.cc` (3.10.0), for `dyntype="none"`:
#
#     ctrl  = clip(d->ctrl[i], ctrlrange)                   [ctrllimited]
#     ctrl  = clip(ctrl, prev +- slewmax*dt)                [if slew AND t>0]
#     err   = ctrl - actuator_length[i]
#     integ = clip(act[adr] + err*dt, +- imax/ki)           [if ki != 0]
#     force = kp*err + kd*(0 - actuator_velocity[i]) + ki*integ
#
# ⚠ `imax` IS A FORCE BOUND IN THE XML AND AN INTEGRAL BOUND HERE.
# `PidConfig::FromModel` divides it by `i_gain` on the way in ("Clamps in the
# XML are specified in terms of maximum forces"), so what is STORED in this
# slot is the already-divided value and a reader must not divide again.
#
# ⚠ `-1` IS "ABSENT", NOT "ZERO", ON BOTH OPTIONAL SLOTS. `imax` and `slewmax`
# are `std::optional` in the reference: absent means NO clamp, and an explicit
# 0 means clamp to zero. Both are validated non-negative by `Pid::Create`, so a
# negative is free to mean absent — see `feedback_a_sentinel_beats_a_default`.
comptime ACT_IDX_PID_KI: Int = ACT_IDX_SITE_ID + 8
comptime ACT_IDX_PID_KD: Int = ACT_IDX_SITE_ID + 9
comptime ACT_IDX_PID_IMAX: Int = ACT_IDX_SITE_ID + 10
comptime ACT_IDX_PID_SLEW: Int = ACT_IDX_SITE_ID + 11

# ── `<adhesion body=...>` (`mjTRN_BODY`) ─────────────────────────────────
#
# ⚠⚠ THE TRANSMISSION IS THE CONTACT SET, WHICH IS WHY THERE IS NO TRIPLE.
# `mj_transmission`'s `mjTRN_BODY` arm sets `length = 0` and builds the moment
# as MINUS THE AVERAGE of the contact NORMAL Jacobians over every contact
# involving this body (engine_core_smooth.c:1623). So an adhesion actuator's
# moment changes with the contact set and cannot be baked into
# `(qadr, dadr, coef)` — it belongs with the site transmission in
# `dynamics/pose_transmission.mojo`, and `ACT_IDX_TRN_N` stays 0.
#
# ⚠ THE REFERENCE GETS THE ACTIVE CONTACTS' JACOBIANS OUT OF `efc_J` and the
# in-gap ones directly, and the two routes give the SAME vector. For a
# pyramidal cone it weights `2*(dim-1)` rows by `0.5/(dim-1)`, and each
# opposing pair is `n +- mu*t`, so the tangents cancel and the sum is exactly
# `n`; for condim 1 and for elliptic cones row 0 IS `n`. Computing the normal
# Jacobian directly is therefore not an approximation of that path, it is the
# same number without an `efc` round trip.
#
# The force law is ordinary: `mjs_setToAdhesion` sets `gainprm[0] = gain`,
# gaintype FIXED, biastype NONE and `ctrllimited = 1`, so `force = gain*ctrl`
# with `gain` in `ACT_IDX_KP`. Everything special is in the moment.
comptime ACT_IDX_BODY_ID: Int = ACT_IDX_SITE_ID + 12

# `mjtDyn` (mjtype.h:242), the values this engine models. ⚠ THE NUMBERS ARE
# MuJoCo'S, not a local encoding: a record that carries a foreign enum's value
# under a local name is a translation table waiting to drift.
comptime ACT_DYN_NONE: Int = 0
comptime ACT_DYN_INTEGRATOR: Int = 1
comptime ACT_DYN_FILTER: Int = 2
comptime ACT_DYN_FILTEREXACT: Int = 3

comptime ACT_IDX_DYN_TYPE: Int = ACT_IDX_SITE_ID + 13
"""`mjModel.actuator_dyntype` — which activation ODE this actuator integrates.

⚠⚠ IT USED TO BE INFERRED FROM `ACT_IDX_DYN_ADR >= 0` AND THAT WAS ONLY
ENOUGH WHILE `filter` WAS THE ONLY KIND MODELLED (AUD-02/AUD-21). An
`integrator` sets `act_dot = ctrl` with no `tau` at all, and a `filterexact`
integrates the SAME `act_dot` as `filter` by a different rule
(`act += act_dot * tau * (1 - exp(-h/tau))`, `mj_nextActivation`), so the tau
column cannot distinguish them. `<position timeconst>` is `filterexact` under
a second spelling.
"""

comptime ACT_IDX_ACT_LIMITED: Int = ACT_IDX_SITE_ID + 14
"""`mjModel.actuator_actlimited` — 1.0 when the activation is clamped.

⚠ THE CLAMP IS PART OF `mj_nextActivation`, NOT OF THE FORCE LAW. It applies
to the integrated activation for EVERY dyntype, after the ODE step, so an
actuator whose `act` saturates keeps producing the clamped activation's force
rather than a growing one.
"""
comptime ACT_IDX_ACT_MIN: Int = ACT_IDX_SITE_ID + 15
comptime ACT_IDX_ACT_MAX: Int = ACT_IDX_SITE_ID + 16

# The tendon SPRING half of actuation (`engine_passive.c`), kept in its own
# record rather than folded into `MODEL_TENDON_SIZE`.
#
# ⚠ SEPARATE FROM `Model.tendons` ON PURPOSE. That record stores wraps as
# JOINT IDS (`TENDON_IDX_JOINT_0 + k`) because its consumers — `tendon_limit`,
# `dynamics/tendon` — want joints. The spring path wants qpos/dof ADDRESSES,
# and resolving one to the other inside the actuation kernel would mean binding
# `Model.joints` as a second operand purely to do a lookup the parser already
# did. Storing the addresses is what the comptime twin does
# (`tendon_trn_qadr` / `_dadr`) and this mirrors it.
# ⚠⚠ THE OFF-DIAGONAL OF `J^T diag(kv) J`, WHICH `dof_actdamp` DOES NOT CARRY.
# `mjd_actuator_vel` (engine_derivative.c:1213) adds `moment^T * biasprm[2] *
# moment` — the FULL outer product over the actuator's transmission — into
# `d->qDeriv`, and the implicit integrators then solve against
# `M_hat = M - h*qDeriv`. `Model.dof_actdamp` is only its DIAGONAL.
#
# ⚠ FOR A JOINT TRANSMISSION THE TWO ARE THE SAME THING. One dof means one
# entry, the outer product IS the diagonal, and nothing is lost — which is why
# this was invisible on every legged model in the tree.
#
# ⚠⚠ SEVEN SCENES IN `mujoco_menagerie-main` HAVE A MULTI-DOF `kv`
# TRANSMISSION, and all seven are a tendon: `hello_robot_stretch` (`arm_extend`,
# 4 dofs, kv 10), `hello_robot_stretch_3` (`arm`, 4 dofs), and five grippers at
# 2 dofs each (franka_emika_panda, robotiq_2f85 + v4, stanford_tidybot,
# ufactory_xarm7). Five of them run `implicitfast`. Measured: forcing BOTH
# engines to Euler — which does NOT use `qDeriv` at all — takes
# `hello_robot_stretch` from **4.406e-05 to 1.823e-10**, while 49 of the other
# 50 `implicitfast` scenes do not move. That scene's whole residual is this
# term.
#
# THE RECORD, one per actuator, holding what the integrator needs to rebuild
# the outer product without binding `SpecFields` (which is deliberately not
# part of `Model` — see `MODEL_ACTUATOR_SIZE`):
#
#     [n, dof_0..dof_{W-1}, pair_00..pair_{W-1,W-1}]   W = TENDON_MAX_WRAPS
#
# where `pair[p*W + q] = kv * mom_p * mom_q` and `mom_k = gear * coef_k`. The
# integrator subtracts `pair[p*W+q]` at `(dof_p, dof_q)`. `n == 0` means "no
# velocity feedback", which is every `<motor>`.
#
# ⚠⚠ THE PRODUCTS ARE PRE-MULTIPLIED AND PRE-FILTERED AT BUILD TIME, and the
# FILTER IS THE POINT. MuJoCo does not write the whole outer product: it
# accumulates through `mju_addToSclSparseInc` into `d->qDeriv`'s SPARSE `D`
# pattern (`engine_derivative.c:768`), and **any column that pattern does not
# have is silently dropped**. `D` is the mass matrix's pattern, which for a
# tree holds `(i, j)` only when the two dofs are ANCESTOR-RELATED.
#
# ⚠⚠⚠ SO A TENDON ACROSS PARALLEL BRANCHES GETS ONLY ITS DIAGONAL. Every
# 2-dof gripper here — franka_emika_panda, robotiq_2f85 + v4,
# stanford_tidybot, ufactory_xarm7 — spans two SIBLING fingers, and MuJoCo's
# `qDeriv` has no off-diagonal for them at all (checked against `D_colind`:
# xarm7's gripper contributes -2.5 at (7,7) and (10,10) and nothing at
# (7,10)). `hello_robot_stretch`'s four TELESCOPING links are a serial chain,
# every pair is ancestor-related, and it gets the full 4x4 block.
#
# ⚠ Writing the whole outer product instead took the 1-step sweep from 74/85
# to **72/85** — stretch was fixed and those grippers each acquired a
# 2.6e-06..3.0e-05 error that had not been there. The rule was then checked
# exhaustively: over **74,671** dof pairs in all 85 models,
# "bodies ancestor-related" and "(i,j) in D" disagree **zero** times.
#
# `p == q` is stored as 0 as well: the diagonal is `dof_actdamp`'s job and
# adding it here would double every servo's damping.
comptime ACTDAMP_TRN_SIZE: Int = 1 + TENDON_MAX_WRAPS + (
    TENDON_MAX_WRAPS * TENDON_MAX_WRAPS
)
comptime ACTDAMP_IDX_N: Int = 0
comptime ACTDAMP_IDX_DOF_0: Int = 1
comptime ACTDAMP_IDX_PAIR_0: Int = 1 + TENDON_MAX_WRAPS

comptime MODEL_ACT_TENDON_SIZE: Int = 5 + 3 * TENDON_MAX_WRAPS

comptime ACTTEN_IDX_STIFFNESS: Int = 0  # 0 => no spring, skip the row
# The deadband bounds. ⚠ WHEN `springlength` IS ABSENT BOTH DEFAULT TO the
# tendon's rest length `sum(coef * joint.ref)`, NOT to zero.
comptime ACTTEN_IDX_SPRING_LO: Int = 1
comptime ACTTEN_IDX_SPRING_HI: Int = 2
comptime ACTTEN_IDX_TRN_N: Int = 3
comptime ACTTEN_IDX_TRN_QADR_0: Int = 4
comptime ACTTEN_IDX_TRN_DADR_0: Int = ACTTEN_IDX_TRN_QADR_0 + TENDON_MAX_WRAPS
# `<tendon damping>` (AUD-08): `-damping * ten_velocity` in `qfrc_passive`.
comptime ACTTEN_IDX_DAMPING: Int = 4 + 3 * TENDON_MAX_WRAPS
comptime ACTTEN_IDX_TRN_COEF_0: Int = (
    ACTTEN_IDX_TRN_QADR_0 + 2 * TENDON_MAX_WRAPS
)

# --- reference pose + keyframes (phase 1a.4) ---------------------------------
#
# `qpos0` itself is a bare `[NQ]` tensor; these are the two scalars that go
# with it. ⚠ `FREE_JOINT_QPOS_ADR` IS -1 WHEN ABSENT AND ZERO IS A VALID
# ADDRESS, so the record is seeded rather than left as `alloc` wrote it.
comptime POSE_META_SIZE: Int = 2
comptime POSE_IDX_QPOS0_NQ: Int = 0
comptime POSE_IDX_FREE_JOINT_QPOS_ADR: Int = 1

# One row per `<keyframe><key>`. ⚠ `NQPOS`/`NQVEL`/`NCTRL` ARE PRESENCE FLAGS
# AS MUCH AS LENGTHS: MuJoCo fills an absent `qpos=` from qpos0 and an absent
# `qvel=`/`ctrl=` with zero, so `key_qpos_at` must know the attribute was
# missing rather than read a row of zeros as a real pose. `init_fields`
# already refuses any length other than the full one.
comptime KEY_META_SIZE: Int = 4
comptime KEY_IDX_TIME: Int = 0
comptime KEY_IDX_NQPOS: Int = 1
comptime KEY_IDX_NQVEL: Int = 2
comptime KEY_IDX_NCTRL: Int = 3

# --- joint limits, for the `enforce_limits` clamp (phase 1a.4) ---------------
#
# ⚠ NOT THE SAME THING AS THE SOLVER'S LIMIT ROWS. `Model.joints` carries
# `JOINT_IDX_RANGE_MIN/MAX` and the per-joint solref/solimp that the
# CONSTRAINT path uses; this record is the hard clamp `enforce_limits` applies
# to `qpos` directly, and it needs one thing the joint record does not have: a
# LIMITED flag. `range_min < range_max` is not that test — MuJoCo spells an
# unlimited joint BOTH as `[0, 0]` and as `[-1e10, 1e10]`, and the second
# satisfies it.
comptime JLIM_SIZE: Int = 8
comptime JLIM_IDX_LIMITED: Int = 0
comptime JLIM_IDX_QPOS_ADR: Int = 1
comptime JLIM_IDX_RANGE_MIN: Int = 2
comptime JLIM_IDX_RANGE_MAX: Int = 3
# --- `<joint actuatorfrcrange>` — MuJoCo's `jnt_actfrcrange` ----------------
#
# ⚠⚠ A SECOND, INDEPENDENT FORCE LIMIT, AND IT IS NOT THE ACTUATOR'S.
# `mj_fwdActuation` clamps TWICE: `actuator_forcerange` on each actuator's own
# scalar force (engine_forward.c:417), and then
#
#     clampVec(d->qfrc_actuator, m->jnt_actfrcrange, m->jnt_actfrclimited,
#              m->njnt, m->jnt_dofadr);                            // :477
#
# on the ACCUMULATED `qfrc_actuator`, per JOINT, at that joint's dof address.
# The two are unrelated: on unitree_g1 `actuator_forcelimited` is FALSE on all
# 29 actuators while `jnt_actfrclimited` is TRUE on 29 of 30 joints, so the
# joint-level clamp is the ONLY force limit the model has. 481 of this tree's
# 2519 joints declare one, across 20 robots — and the tightest are tiny: g1's
# wrists are +-5 N.m against a `kp=500` servo, sharpa_wave's are +-0.19.
#
# ⚠ IT LIVES HERE, NOT ON `Model.joints`, because `apply_actions_fields` is
# handed `sf` and not the model — and because this record already carries the
# LIMITED flag idiom that the joint record deliberately does not have.
# `JLIM_IDX_DOF_ADR` is what `jnt_dofadr` is in the clampVec call above; the
# existing `QPOS_ADR` column is the wrong address for it.
comptime JLIM_IDX_DOF_ADR: Int = 4
comptime JLIM_IDX_ACTFRC_LIMITED: Int = 5
comptime JLIM_IDX_ACTFRC_MIN: Int = 6
comptime JLIM_IDX_ACTFRC_MAX: Int = 7


# =============================================================================
# Model Buffer Layout - Mesh Collision Hull Data
# =============================================================================

# Mesh hull vertices stored AFTER exclude pairs in the model buffer.
# Layout: [mesh_meta(NMESH*2)] [mesh_verts(total_verts*3)]
# mesh_meta: [vertadr, vertnum] per mesh
# mesh_verts: flattened [x0,y0,z0, x1,y1,z1, ...] in local frame
comptime MAX_HULL_VERTS_PER_MESH: Int = 256
"""⚠⚠ DEAD, AND MISLEADING IF READ AS LIVE. Nothing references this. Hulls are
NOT capped per mesh — so_arm100 loads one of 2094 vertices — and the total is
bounded by `dims.get_nmesh_verts()`, a runtime budget. Kept only because
deleting a public constant is a separate change; do not reintroduce a use."""

comptime MAX_GPU_MESHES: Int = 256
"""How many COLLIDABLE meshes one model may have. Was 16.

⚠ IT SIZES ONE TABLE AND NOTHING ELSE. `mesh_meta` is `[MAX_GPU_MESHES, 4]` —
8 KB at 256 and float64, against 512 bytes at 16 — and every other use is a
`Layout` over that table. No `Array` is keyed on it, so raising it costs
memory and nothing else. 16 was never a hardware limit; it was a guess that
predates mesh-heavy models.

⚠⚠ AND EXCEEDING IT NOW RAISES. It used to print `ERROR:` and continue, which
leaves meshes past the cap with a hull built and an id assigned but NO
`mesh_meta` row — so every consumer reads vertadr/vertnum 0 and collides
against an EMPTY mesh. That is wrong physics that runs, and this tree has paid
for the same shape twice (the 16-asset `<mesh>` cap cost SO-ARM100 two
collision surfaces, found only by diffing `rbound` against MuJoCo). A model
that will not open is a better outcome than one that opens and is wrong.

Headroom, measured 2026-08-19: ToddlerBot 2, so_arm100 8, sawyer 10 — the
mesh-heaviest models in or near this tree. See
`tests/physics3d/test_mesh_cap_is_loud.mojo`."""
# vertadr, vertnum, polyadr, polynum per mesh — the last two added with the
# native multi-contact path, mirroring `mesh_polyadr` / `mesh_polynum`.
comptime MODEL_MESH_META_SIZE: Int = 8
comptime MESH_META_IDX_VERTADR: Int = 0
comptime MESH_META_IDX_VERTNUM: Int = 1
comptime MESH_META_IDX_POLYADR: Int = 2
comptime MESH_META_IDX_POLYNUM: Int = 3
# The mesh's window into `Model.mesh_tris` — the ORIGINAL triangle soup that
# `ray/mesh.mojo` walks, which is NOT the hull `VERTADR`/`VERTNUM` describe.
# ⚠ `TRINUM` is 0 on every mesh when the model was built without a triangle
# budget (`nmesh_tri`), which is the default. A reader must treat 0 as "not
# carried" and not as "this mesh has no geometry".
comptime MESH_META_IDX_TRIADR: Int = 4
comptime MESH_META_IDX_TRINUM: Int = 5
# The mesh's BVH over that soup, in RECORDS of the same arena. `BVHNUM` is 0
# exactly when no tree was built, and `ray/mesh.mojo` then falls back to the
# linear sweep — the two must return the SAME distance and normal, which is
# what `tests/physics3d/test_ray_bvh_matches_linear.mojo` exists to hold.
comptime MESH_META_IDX_BVHADR: Int = 6
comptime MESH_META_IDX_BVHNUM: Int = 7

# ---- The mesh geometry ARENA (`Model.mesh_tris`) -----------------------------
#
# ⚠⚠ `mesh_tris` HOLDS TWO KINDS OF RECORD, AND THAT IS THE WHOLE POINT. Nine
# floats per record, triangles first and BVH nodes after them:
#
#   records [0, ntri)                   a triangle: v0 v1 v2, mesh frame
#   records [BVHADR, BVHADR + BVHNUM)   a BVH node, fields below
#
# The alternative was a second `Model` tensor, which would have added a
# `Layout` parameter and an argument to `ray_model`, `render_pixel`,
# `directional_light_term`, both camera kernels and EVERY env config's
# observation hook — a public signature, in this tree, changed for a table
# whose size is a fixed multiple of one that is already there. A node indexes
# the triangles it covers by the same record number the triangle already has,
# so keeping them in one arena also removes the chance of the two tables
# disagreeing about which triangle is which.
#
# ⚠ THE BUDGET IS STILL SPELT IN TRIANGLES. `nmesh_tri` is the caller's
# TRIANGLE count and its meaning has not changed; the arena is
# `MESH_ARENA_FLOATS_PER_TRI` per triangle because a one-triangle-per-leaf BVH
# over n triangles has exactly 2n-1 nodes. 1 + 2 records of 9 floats = 27.
comptime MESH_ARENA_RECORD: Int = 9
comptime MESH_ARENA_FLOATS_PER_TRI: Int = 27

# A BVH node's nine floats. The AABB is CENTRE + HALF-EXTENT, in the mesh's
# principal frame, which is `mjModel.bvh_aabb`'s layout exactly.
comptime MESH_BVH_IDX_CX: Int = 0
comptime MESH_BVH_IDX_CY: Int = 1
comptime MESH_BVH_IDX_CZ: Int = 2
comptime MESH_BVH_IDX_HX: Int = 3
comptime MESH_BVH_IDX_HY: Int = 4
comptime MESH_BVH_IDX_HZ: Int = 5
# ⚠⚠ AN ESCAPE INDEX, NOT A CHILD POINTER, AND THAT IS WHY THERE IS NO STACK.
# The reference walks its two-child tree with `int stack[mjMAXTREEDEPTH]`, a
# per-thread array indexed at RUNTIME — the exact read that is silently wrong
# on Metal and has been four times in this engine. Nodes are stored in
# PRE-ORDER, so the left child is always the next record and the only thing a
# miss needs is where the subtree ends. Traversal is then `node += 1` on a hit
# and `node = escape` on a miss: no stack, no per-thread storage, no bound to
# exceed.
comptime MESH_BVH_IDX_ESCAPE: Int = 6
# The triangle's record number for a leaf, -1 for an internal node.
comptime MESH_BVH_IDX_TRI: Int = 7
# Slot 8 is unused. It buys the node the same stride as a triangle, which is
# what lets both live in one tensor.

# ---- HEIGHTFIELDS -----------------------------------------------------------
#
# `mjModel.hfield_*`: an `nrow x ncol` grid of elevations already normalised to
# [0, 1], scaled at collision time by `size[2]`, sitting on a base that extends
# to `-size[3]`. One row per hfield, the grid itself in `hfield_data`.
#
# ⚠ A FIXED CAP IS FINE HERE AND IS NOT FINE FOR THE DATA. `hfield_meta` is
# `[MAX_GPU_HFIELDS, 7]` — 448 bytes at float64 — so capping the COUNT costs
# nothing. The grid does not get a cap: barkour's single field is 256 x 256 =
# 65 536 samples, which as a fixed allocation every model paid for would be
# half a megabyte of zeros. It is sized by `dims.get_nhfield_data()`.
comptime MAX_GPU_HFIELDS: Int = 8
comptime MODEL_HFIELD_META_SIZE: Int = 7
comptime HFIELD_META_IDX_ADR: Int = 0  # offset into `hfield_data`
comptime HFIELD_META_IDX_NROW: Int = 1
comptime HFIELD_META_IDX_NCOL: Int = 2
comptime HFIELD_META_IDX_SIZE_X: Int = 3  # radius in x
comptime HFIELD_META_IDX_SIZE_Y: Int = 4  # radius in y
comptime HFIELD_META_IDX_SIZE_Z: Int = 5  # elevation scale
comptime HFIELD_META_IDX_SIZE_BASE: Int = 6  # depth of the base below z = 0

# ---- Mesh POLYGON topology (native multi-contact) ---------------------------
#
# `multicontact` (`engine_collision_gjk.c:2111`) recovers the face a contact
# came from and clips it against the opposing face. For a mesh that needs the
# hull's polygons, which `collision/mesh_polygons.mojo` builds at model load.
#
# ⚠ THE CAPS ARE EULER'S FORMULA, NOT A GUESS, which is why none of these need
# to become `Model` type parameters. For a convex polyhedron with V vertices a
# TRIANGULATED hull has at most F = 2V - 4 faces and E = 3V - 6 edges; merging
# coplanar triangles into polygons only ever REDUCES F, and the total number of
# polygon-vertex incidences is exactly 2E <= 6V - 12. The vertex -> polygon map
# holds the same 2E entries. So sizing off NMESH_VERTS is exact, not generous,
# and a mesh cannot overflow these without violating convexity.
comptime MODEL_MESH_POLY_SIZE: Int = 5  # vertadr, vertnum, nx, ny, nz
comptime MESH_POLY_IDX_VERTADR: Int = 0
comptime MESH_POLY_IDX_VERTNUM: Int = 1
comptime MESH_POLY_IDX_NX: Int = 2
comptime MESH_POLY_IDX_NY: Int = 3
comptime MESH_POLY_IDX_NZ: Int = 4


def mesh_max_poly(nmesh_verts: Int) -> Int:
    """Polygon capacity for a hull budget of `nmesh_verts` vertices."""
    return 2 * nmesh_verts if nmesh_verts > 0 else 1


def mesh_max_polyvert(nmesh_verts: Int) -> Int:
    """Polygon-vertex (and vertex->polygon map) capacity. See above."""
    return 6 * nmesh_verts if nmesh_verts > 0 else 1


# The hill climb's direction SEED TABLE (`collision/mesh_seed.mojo`): for a
# hull of at least `MESH_SEED_MIN_VERTS` vertices, `MESH_SEED_N` local vertex
# indices — the extreme vertex for each point of a `MESH_SEED_Q`^3 direction
# grid — stored in `mesh_edges` directly before the mesh's first neighbour
# list, behind one `MESH_SEED_MAGIC` slot. Reader and builder are in
# `mesh_seed.mojo`; the layout is stated there.
comptime MESH_SEED_Q: Int = 5
comptime MESH_SEED_N: Int = MESH_SEED_Q * MESH_SEED_Q * MESH_SEED_Q
comptime MESH_SEED_MIN_VERTS: Int = 64
# Not a vertex id (>= 0) and not the -1 terminator.
comptime MESH_SEED_MAGIC: Int = -7


def mesh_max_edge(nmesh_verts: Int) -> Int:
    """Hull edge-graph capacity for a budget of `nmesh_verts` vertices.

    MuJoCo sizes the same block at `numvert + 3*numface`; a triangulated
    polytope has `F = 2V - 4`, so that is `7V - 12`. 8V leaves headroom rather
    than trusting the identity, and `fields_build` still raises on overflow.
    On top of it, each mesh of at least `MESH_SEED_MIN_VERTS` vertices carries
    a seed table of `MESH_SEED_N + 1` slots, so there are at most
    `V // MESH_SEED_MIN_VERTS` of them.
    """
    if nmesh_verts <= 0:
        return 1
    return 8 * nmesh_verts + (nmesh_verts // MESH_SEED_MIN_VERTS) * (
        MESH_SEED_N + 1
    )


# =============================================================================
# Workspace Buffer Layout (per-environment scratch space for GPU kernels)
# =============================================================================
# Moves all integrator temporaries and solver arrays from InlineArrays
# (register pressure) to device memory.
#
# Layout per environment:
#   [integrator_temps | M_inv: NV*NV | solver workspace: SOLVER.solver_workspace_size()]
#
# Integrator temps section:
#   [cdof: NV*6 | crb: NBODY*10 | M: NV*NV | L: NV*NV | D: NV |
#    bias: NV | f_net: NV | qacc_ws: NV | qacc_constrained: NV]


# =============================================================================
# Implicit Integrator Extra Workspace
# =============================================================================
# Additional workspace for the full implicit integrator's RNE velocity
# derivative computation. Placed AFTER solver workspace so existing offsets
# are unchanged.
#
# Layout within implicit extra section:
#   [qDeriv: NV*NV | cdof_origin: NV*6 | cvel_origin: NBODY*6 |
#    cinert: NBODY*10 | cdof_dot: NV*6 |
#    Dcvel: NBODY*6*NV | Dcdofdot: NV*6*NV |
#    Dcacc: NBODY*6*NV | Dcfrcbody: NBODY*6*NV]


def rk4_extra_workspace_size[NQ: Int, NV: Int]() -> Int:
    """Total RK4-extra workspace size per environment."""
    return NQ + 7 * NV





# =============================================================================
# CAMERAS — the model-side table the batched ray tracer reads
# =============================================================================
#
# `mjModel.cam_*`, minus everything only a rasteriser wants. Cameras existed in
# this tree ONLY as `RenderFields` lists (host-side, `Float64`, `List`), which
# is the right shape for the SDL viewer and the wrong one for a kernel: a
# thread rendering pixel (env, px, py) needs the camera's parent body, local
# pose and fovy as tensor reads, and cannot hold a `List`.
#
# ⚠ A FIXED CAP, LIKE `hfield_meta` AND UNLIKE `hfield_data`. The table is
# `[MAX_GPU_CAMERAS, MODEL_CAM_SIZE]` — 1.5 KB at float64 — so capping the
# COUNT costs nothing and needs no `Dims` member, no config parameter and no
# positional risk in `ModelDims[...]`. Exceeding it RAISES in `fields_build`;
# it does not truncate. A model whose last camera silently vanished would
# render a black image with nothing to point at.
comptime MAX_GPU_CAMERAS: Int = 32
"""Headroom, measured 2026-08-26 over `assets/`, `dm_control-main`,
`mujoco_menagerie-main`, `Metaworld-master` and `so101-nexus-main`: the
camera-heaviest model in or near this tree is `flybody/fruitfly.xml` at **9**.
(MuJoCo's own `100_humanoids_chol.xml` benchmark has 300, which is what a cap
looks like when the model is a hundred copies of one robot — nothing here loads
it.) The table is 6 KB at float64, so the cap is set well clear of the measured
maximum rather than snugly above it."""

comptime MODEL_CAM_SIZE: Int = 24
comptime CAM_IDX_BODY: Int = 0
"""`mjModel.cam_bodyid` — the body `pos`/`quat` below are expressed in.

⚠⚠ THE POSE IS LOCAL, NOT WORLD. `mj_camlight` composes
`xpos[b] + xmat[b]*cam_pos` before it dispatches on the mode, and a camera in
`<worldbody>` has the identity there — which is why dropping this field was
invisible across the whole dm_control suite and broke only the first
body-attached camera. See `kinematics/camera_frame.mojo`."""
comptime CAM_IDX_POS_X: Int = 1
comptime CAM_IDX_POS_Y: Int = 2
comptime CAM_IDX_POS_Z: Int = 3
# ⚠ (x, y, z, w) — the order `GEOM_IDX_QUAT_*` and `BODY_IDX_QUAT_*` use, NOT
# the (w, x, y, z) a `math3d.Quat` constructor takes. The records agree with
# each other; the constructor is where the swap happens.
comptime CAM_IDX_QUAT_X: Int = 4
comptime CAM_IDX_QUAT_Y: Int = 5
comptime CAM_IDX_QUAT_Z: Int = 6
comptime CAM_IDX_QUAT_W: Int = 7
comptime CAM_IDX_FOVY: Int = 8  # VERTICAL field of view, in DEGREES
comptime CAM_IDX_MODE: Int = 9  # one of the `CAM_MODE_*` below
comptime CAM_IDX_TARGET_BODY: Int = 10  # `mode="targetbody"` aim; -1 if none
comptime CAM_IDX_ACTIVE: Int = 11
"""1 on a row a `<camera>` actually filled, 0 on padding.

⚠ NOT REDUNDANT WITH THE COUNT. The count is a host-side `Int` passed to the
kernel, and a kernel that reads past it would find a zero row whose `fovy` is
0 — a degenerate frustum, not an obvious error. This flag is what a reader
tests so the failure names itself."""


# ---- The REFERENCE pose, for `mode="track"` and `mode="trackcom"` ----------
#
# `mjModel.cam_pos0` / `cam_poscom0` / `cam_mat0`. MuJoCo's compiler computes
# these once, at `qpos0`, and `mj_camlight` reads them for the two tracking
# modes:
#
#     TRACK:    cam_xpos = xpos[body]        + cam_pos0
#     TRACKCOM: cam_xpos = subtree_com[body] + cam_poscom0
#     both:     cam_xmat = cam_mat0          (a FIXED world orientation)
#
# ⚠⚠ SO A TRACKING CAMERA'S ORIENTATION IS NOT ITS BODY'S. It translates with
# the body and never turns — which is exactly what makes a `trackcom` view of
# a walker readable, and what makes "compose the body transform" the WRONG
# answer for two of the five modes.
#
# ⚠ WE HAVE NO MODEL COMPILER, so these are not filled by the parser: they
# need forward kinematics at `qpos0`, which happens in `Data`, not in
# `fields_build`. `raytrace.init_camera_reference` fills them from a lane whose
# FK has run, and sets `CAM_IDX_REF_SET`. A tracking camera whose reference was
# never taken is the silent-zero failure this tree keeps paying for, so the
# renderer's HOST entry point refuses to launch on one.
comptime CAM_IDX_POS0_X: Int = 12
comptime CAM_IDX_POS0_Y: Int = 13
comptime CAM_IDX_POS0_Z: Int = 14
comptime CAM_IDX_POSCOM0_X: Int = 15
comptime CAM_IDX_POSCOM0_Y: Int = 16
comptime CAM_IDX_POSCOM0_Z: Int = 17
comptime CAM_IDX_QUAT0_X: Int = 18  # `cam_mat0`, as a quaternion (x, y, z, w)
comptime CAM_IDX_QUAT0_Y: Int = 19
comptime CAM_IDX_QUAT0_Z: Int = 20
comptime CAM_IDX_QUAT0_W: Int = 21
# `mjtCamLight`. ⚠⚠ THESE ARE A SECOND SPELLING OF `parser/flat_model.mojo`'s
# `CAM_MODE_*`, and the two MUST agree — `fields_build` writes the parser's
# value into `CAM_IDX_MODE` and `raytrace/camera.mojo` dispatches on this one.
# A silent disagreement would turn every `trackcom` camera into a `track`
# camera, which still renders. `fields_build` carries a comptime assert
# pairing them, so the agreement is CHECKED and not merely asserted in a
# comment. They live here rather than only in the parser because `raytrace`
# does not (and should not) import the parser.
comptime CAM_MODE_FIXED: Int = 0
comptime CAM_MODE_TRACK: Int = 1
comptime CAM_MODE_TRACKCOM: Int = 2
comptime CAM_MODE_TARGETBODY: Int = 3
comptime CAM_MODE_TARGETBODYCOM: Int = 4

comptime CAM_IDX_TAN_HALF_FOVY: Int = 23
"""`tan(fovy/2)` in RADIANS, precomputed at model build.

⚠⚠ NOT AN OPTIMISATION — `std.math.tan` IS CPU-ONLY. It resolves to libm, and
a kernel calling it fails to compile with *"libm operations are only available
on CPU targets"*. The frustum's half-height depends on nothing but `fovy`, a
per-camera constant, so a per-pixel transcendental was never wanted anyway;
this is the shape `dm_control/dtype_math.mojo` exists to avoid needing.

⚠ `fovy` IS KEPT ALONGSIDE IT rather than replaced. A gate comparing against
`mjModel.cam_fovy` should read the quantity MuJoCo stores, not its tangent.
"""


comptime CAM_IDX_REF_SET: Int = 22
"""1 once `init_camera_reference` has taken this camera's `qpos0` pose.

0 on a FIXED or TARGETBODY camera is harmless — neither mode reads the
reference. On a TRACK/TRACKCOM camera it means the pose above is ZERO, i.e.
the camera sits exactly on its body's origin looking along world -Z. That is a
picture, not an error, which is why it is checked on the host."""


# ---- Per-geom visual colour, for the ray tracer only ------------------------
#
# `Model.geom_rgba` is `[NGEOM, 4]` and is NOT part of `MODEL_GEOM_SIZE`.
#
# ⚠ DELIBERATELY A SEPARATE TENSOR. The geom record is read by the broadphase,
# the narrow phase and the solver on every step; colour is read by nothing but
# a renderer. Widening the hot record by four floats per geom to carry a field
# no physics path touches trades cache for tidiness in the wrong direction.
comptime MODEL_GEOM_RGBA_SIZE: Int = 4
