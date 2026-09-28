"""The family's `Phyics3dEnvConfig` — the reward IS the goal. P3c.

    Phyics3dBatchedEnv[So101TabletopModel, So101TabletopConfig, N_ENVS]
    Phyics3dBatchedEnv[So101TowerModel, So101TowerConfig, N_ENVS]

`So101FamilyConfig[P, ...]` is the one type; the two names above are it at
the tabletop's hand-written table and the tower's GENERATED one.

One config per FAMILY, not per task. That is the fixed scene budget cashing in:
every task in the family shares this type, this model and this monomorphisation,
and what varies between lanes is DATA — the twelve-word tape in
`meta[env, META_IDX_TASK_PARAM_*]`.

## ⚠ WHAT THE HOST WRITES, AND WHEN

    once   : curriculum[0, 0..4]              the region table
    per ep : meta[env, TASK_PARAM_0.._11]     this lane's goal
    per ep : meta[env, TASK_ACTIVE]           this lane's active slots
    per ep : qpos / qvel                      placements + parked slots

`tasks/reset.reset_slots`, `tasks/tape.encode_goal` and
`tasks/active.active_mask` are those writes. None of them is a kernel today,
and none needs to be: a reset is a host operation in the driver, and all three
`meta` writes survive `_reset_env_lane` because that only writes
`META_IDX_STEP_COUNT`.

⚠⚠ THAT SAME PROPERTY IS WHY EVERY ONE OF THEM MUST BE REWRITTEN EVERY
EPISODE. `meta` is not zeroed between episodes, so a lane keeps the previous
episode's goal and the previous episode's mask unless the driver writes over
them. `encode_goal` handles its half by writing `OP_NONE` into the terms it
does not use; the mask is one word and is always written whole.

## THE OBSERVATION, AND THE ACTIVE MASK IN IT

§3.4 asked for `(pose, active)` per slot, and `META_IDX_TASK_ACTIVE` is that
channel — one word, widened out of `METADATA_SIZE`, because all twelve
`TASK_PARAM` words are the tape and `encode_goal` writes every one of them.
The two observation hooks below are its only consumers.

⚠ THE OBSERVATION IS NOT THE MODEL DEFAULT ANY MORE, and it is a word wider
than the mask alone accounts for. The default is `qpos[obs_qpos_skip:] +
qvel`, and `obs_qpos_skip` defaults to 1 — which on a FLOATING-BASE model
drops the root's redundant word and on a DESK ARM drops `shoulder_pan`. This
family has no floating base: the arm is bolted to the world and `qpos[0]` is a
hinge angle the policy needs. So the hook writes the FULL `qpos`.

    OBS_DIM = NQ + NV + N_FREE_SLOTS = 27 + 24 + 3 = 54

⚠ THERE ARE TWO OBSERVATION HOOKS AND THEY ARE PINNED TO EACH OTHER. `_gpu`
takes `LayoutTensor`s and `_cpu` takes a `List`; there is no type that is
both, so the loop is written twice. `tests/tasks/test_active_mask.mojo` runs
BOTH on one state and demands identical vectors — a permutation between them
is a policy that works on the GPU and is nonsense on the CPU, with no error
anywhere.

## ⚠⚠ WHAT THIS STILL DOES NOT DO

* **No per-step repark.** Gap D's fix — pinning a parked slot's pose every
  step — now has the mask it was missing, but `pre_step_gpu` is handed only
  `qpos` and `meta`: it has no way to learn WHERE a slot parks. That is a
  family constant, so its home is `curriculum` (shared, host-written once) and
  reaching it means widening the `pre_step_gpu` signature across the fourteen
  configs that override it. Parked slots therefore still FALL.

  ⚠ THAT IS NOW COSMETIC, WHICH IT WAS NOT BEFORE. The fall is invisible to
  the REWARD (a goal names only active slots), and it is invisible to the
  OBSERVATION (an inactive slot's pose words are zeroed below). What remains
  is the VIEWER, which draws props sinking through the sky, and the invariant
  itself. A parked slot never lands inside a horizon — 7.06 m of free fall
  against 43 m of headroom, `TASK_LAYER_IMPLEMENTATION.md` — so nothing
  downstream reads a wrong number today.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.physics3d.fields import Data, Dims, DimsLike
from std.math import sqrt

from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    META_IDX_TASK_PARAM_0,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    META_IDX_PREV_X,
    META_IDX_TASK_ACTIVE,
    META_IDX_GOAL_HELD,
    META_IDX_REWARD_MODE,
    META_IDX_SUCCESS_BONUS,
    META_IDX_PHI_PREV,
    META_IDX_EPISODE_FLAGS,
    EPISODE_FLAG_PHI_SET,
    EPISODE_FLAG_BONUS_PAID,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    META_IDX_SHAPE_W_GOAL,
    META_IDX_SHAPE_W_REACH,
    META_IDX_GOAL_MARGIN,
    META_IDX_REACH_MARGIN,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .gpu_eval import eval_tape_gpu, tape_distance_gpu
from .placement.table import PlacementTable, reset_task_slots
from .task_hooks import (
    repark_inactive_slots, write_task_obs, write_task_obs_host,
)
from .predicates import OP_NEAR, OP_ABOVE, OP_ON, OP_IN
from .tape import TERM_WORDS
from .shaping import near_transport_shortfall
from noeira.envs.dm_control.rewards import (
    tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN,
)
from .so101_tabletop_xml import (
    So101TabletopModel, SO101_TABLETOP_N_FREE_SLOTS,
)
from noeira.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from .so101_tower_xml import SO101_TOWER_NMESH_VERTS
from .placement.so101_tower import So101TowerPlacement
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig


struct So101TabletopPlacement(PlacementTable):
    """`so101_tabletop`'s placement table, over `So101TabletopConfig`'s own
    restated constants.

    ⚠ HAND-WRITTEN, UNLIKE THE LIBERO TABLES, because this family's free slots
    carry no `slot_geom=`: the host sampler uses the CALLER's radius for them,
    which no generator can read out of a `.family`. `SLOT_RADIUS` is that radius
    on both paths. `check.placement_table_drift` diffs every method against the
    loaded family and FK, as the config's constants always were.
    """

    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = SO101_TABLETOP_N_FREE_SLOTS
    comptime N_REGIONS: Int = 3
    comptime NQ: Int = 27
    comptime NV: Int = 24
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = So101TabletopModel.NBODY
    comptime NSITE: Int = So101TabletopModel.NSITE
    # `so101_tabletop.family` declares no `base_qpos=`: its rest is `qpos0`.
    comptime N_BASE_QPOS: Int = 0

    # ── THE FAMILY CONSTANTS — restated here, ON THE TABLE, not on the config.
    #
    # ⚠ THEY USED TO LIVE ON `So101TabletopConfig`. The config is now
    # `So101FamilyConfig[P]`, ONE type over every SO-101 family, and reads
    # everything family-specific through its `PlacementTable` — so the
    # numbers a family restates belong to its table, where the generated
    # tables (`placement/so101_tower.mojo`) already keep them.
    # ── THE FREE-SLOT TABLE — the one thing this type restates ────────────
    #
    # A config is a comptime TYPE and the `.family` is a runtime file, so this
    # cannot read it — the same constraint `MAX_STEPS` above lives under. The
    # difference is that this restatement is CHECKED: `tests/tasks/
    # test_active_mask.mojo` loads the family, runs `free_slot_addresses`
    # against the composed scene, and asserts every number below. A drift is a
    # failing gate, not a silently permuted observation.
    #
    # Measured on `scenes/so101_tabletop.xml` through MuJoCo 3.10.0:
    #
    #   family slot   joint          qposadr   dofadr
    #   1  brick      brick_free       6         6
    #   2  cube_a     cube_a_free     13        12
    #   3  cube_b     cube_b_free     20        18
    #
    # Slot 0 is `table`, a STATIC fixture: no joint, no state, and therefore
    # nothing in the observation varies with it. It still owns bit 0 of the
    # mask — the mask is indexed by FAMILY slot, so there is no second
    # numbering to keep in step (`tasks/active.mojo`).
    #
    # ⚠ `qposadr` AND `dofadr` DIVERGE AFTER THE FIRST FREE JOINT, because a
    # free joint is 7 `qpos` against 6 `qvel`. Reusing one for the other is
    # right for slot 0 of the three and wrong for the rest — which is exactly
    # the shape that reads as "the last prop's velocity is somebody else's".
    comptime FREE_SLOT_IDX_0: Int = 1
    comptime FREE_SLOT_IDX_1: Int = 2
    comptime FREE_SLOT_IDX_2: Int = 3
    comptime FREE_QADR_0: Int = 6
    comptime FREE_QADR_1: Int = 13
    comptime FREE_QADR_2: Int = 20
    comptime FREE_DADR_0: Int = 6
    comptime FREE_DADR_1: Int = 12
    comptime FREE_DADR_2: Int = 18

    # ── THE PARK POSE, the second thing this type restates ────────────────
    #
    # `tasks/family.park_pos` is `(park_x + slot*PARK_SPACING, park_y,
    # park_z)`, read from the `.family`'s `park=` line. All family constants,
    # so a comptime type can hold them — and `test_active_mask` asserts each
    # against `park_pos(f, si)` on the loaded family, the same way it asserts
    # the address table.
    #
    # ⚠⚠ I SAID THIS NEEDED A NEW OPERAND AND IT DID NOT. The P3d note claimed
    # the repark was blocked because `pre_step_gpu` "has no way to learn WHERE
    # a slot parks" and that reaching it meant putting the pose in
    # `curriculum` and widening a signature across fourteen configs. The pose
    # is a FAMILY CONSTANT, exactly like `FREE_QADR_*` above, and restating it
    # here costs one gate assertion. Only `qvel` actually needed a wider hook.
    comptime PARK_X: Float64 = 10.0
    comptime PARK_Y: Float64 = 0.0
    comptime PARK_Z: Float64 = 50.0
    comptime PARK_SPACING: Float64 = 0.5

    # ── THE REGION TABLE, the third thing this type restates ──────────────
    #
    # ⚠⚠ RESTATED BECAUSE `init_qpos_gpu` IS NOT HANDED `curriculum` OR
    # `site_xpos`. It gets `qpos`, `qvel`, the MODEL records and `meta`, and
    # it runs BEFORE forward kinematics — so the site a region hangs off has
    # no world position it could read. Every region in this family hangs off
    # `table_surface`, which belongs to a STATIC fixture: its world pose is a
    # family constant, and a constant is what a comptime type can hold.
    #
    # ⚠ THE SAME STATUS AS `FREE_QADR_*` AND `PARK_*` ABOVE — restated, and
    # CHECKED. `tests/tasks/test_device_placement.mojo` loads the `.family`,
    # runs FK on the composed scene, and asserts every number below against
    # `region_sites` + `region_rects`. A drift is a failing gate.
    #
    # ⚠ ONE SITE FOR ALL THREE REGIONS, which is true of this family and not
    # of families in general — a family whose regions sit on different
    # fixtures needs one triple each.
    #
    # Region order is FAMILY ORDER, which is what `META_IDX_INIT_REGION_*`
    # holds and what `region_rects` returns:
    #
    #   0  table_top     -0.10,-0.10, 0.10, 0.10
    #   1  table_left    -0.10, 0.04, 0.10, 0.12
    #   2  table_right   -0.10,-0.12, 0.10,-0.04
    comptime REGION_SITE_X: Float64 = 0.25
    comptime REGION_SITE_Y: Float64 = 0.0
    comptime REGION_SITE_Z: Float64 = 0.02
    comptime REGION_X0_0: Float64 = -0.10
    comptime REGION_Y0_0: Float64 = -0.10
    comptime REGION_X1_0: Float64 = 0.10
    comptime REGION_Y1_0: Float64 = 0.10
    comptime REGION_X0_1: Float64 = -0.10
    comptime REGION_Y0_1: Float64 = 0.04
    comptime REGION_X1_1: Float64 = 0.10
    comptime REGION_Y1_1: Float64 = 0.12
    comptime REGION_X0_2: Float64 = -0.10
    comptime REGION_Y0_2: Float64 = -0.12
    comptime REGION_X1_2: Float64 = 0.10
    comptime REGION_Y1_2: Float64 = -0.04

    # ⚠ THE SLOT RADIUS THE SAMPLER REJECTS ON, and the height it rests at.
    # Every free slot in this family is `assets/props/cube.xml`, a 1.2 cm
    # half-size box, so one constant serves all three. `sampler.
    # sample_placements` takes it as `radii[si]` and uses it for BOTH the
    # pairwise clash test and the resting height, so a per-asset table would
    # have to feed both.
    #
    # ⚠⚠ AND A FAMILY WHOSE SLOTS CARRY `slot_geom=` DOES NOT COME THROUGH HERE
    # AT ALL. `SlotSpec.has_geom` — set by `resolve_family` from the asset's own
    # robosuite `bottom_site` / `top_site` / `horizontal_radius_site` — makes
    # the HOST sampler ignore `radii[si]` and use those two separate numbers
    # instead. Every LIBERO family has them (93 assets, 10 distinct triples,
    # radius spanning 0.005 to 0.3).
    #
    # ⚠ AND THE DEVICE RESET NOW READS THEM TOO. This note used to list what a
    # LIBERO twin owed — read `slot_geom=`, walk `spec.order_inits`, add
    # `TABLE_Z_OFFSET` on a region naming no contact slot — and the host had
    # since grown a FOURTH (the fixture's `top_site` for `On`). All four live in
    # `placement/table.place_free_slots`, which this family reaches through
    # `So101TabletopPlacement` with `has_geom` false, so this constant is its
    # fallback radius exactly as it is the host's `radii[si]`.
    #
    # ⚠⚠ IT TRACKS `cube.xml`'s `size` AND THERE IS NOTHING TO ENFORCE THAT.
    # A radius larger than the prop spawns it FLOATING — it drops at reset,
    # and every reset distance the shaping was calibrated against moves. The
    # prop shrank from 0.02 to 0.012 because the SO-101 jaw cannot close on a
    # 4 cm cube (see the header of `cube.xml`); this moved with it.
    comptime SLOT_RADIUS: Float64 = 0.012

    comptime REGION_SITE_ID: Int = 2
    """`table_surface`'s site id — the site EVERY region in this family hangs
    off.

    ⚠⚠ USED BY BOTH OBSERVATION HOOKS AND BY NEITHER EVALUATOR. The device
    evaluator reads the same id out of `curriculum[0, CUR_IDX_REGION_SITE]`,
    but `custom_extract_obs_cpu` is handed no `curriculum` — so having the GPU
    hook read the table and the CPU hook read a constant would put a
    divergence between the two vectors a checkpoint is shaped by.
    `tests/tasks/test_device_placement.mojo` asserts it equals
    `region_sites(f, fmd.site_names)[0]`."""

    comptime GRIPPER_SITE: Int = 1
    """`robot_gripperframe`'s site id in the composed scene.

    ⚠ RESTATED LIKE THE REGION TABLE, and checked the same way — the reward
    hook gets `site_xpos` but no name table. Measured through MuJoCo 3.10.0 on
    `scenes/so101_tabletop.xml`: 0 `robot_baseframe`, 1 `robot_gripperframe`,
    2 `table_surface`."""

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return Self.FREE_SLOT_IDX_0
        if j == 1:
            return Self.FREE_SLOT_IDX_1
        return Self.FREE_SLOT_IDX_2

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return Self.FREE_QADR_0
        if j == 1:
            return Self.FREE_QADR_1
        return Self.FREE_QADR_2

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return Self.FREE_DADR_0
        if j == 1:
            return Self.FREE_DADR_1
        return Self.FREE_DADR_2

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return False

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.SLOT_RADIUS)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.SLOT_RADIUS)

    @staticmethod
    def free_park_x[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        # ⚠ IN `DTYPE`, NOT `Float64`: `j` is a runtime index, so a `Float64`
        # product here would be a `double` in a Metal kernel. Same op order
        # as `family.park_pos`, so the float64 value is identical.
        return Scalar[DTYPE](Self.PARK_X) + Scalar[DTYPE](
            Self.free_slot(j)
        ) * Scalar[DTYPE](Self.PARK_SPACING)

    @staticmethod
    def free_park_y[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.PARK_Y)

    @staticmethod
    def free_park_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.PARK_Z)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    # ⚠ ONE SITE FOR EVERY REGION — true of this family, and the reason the
    # config restates a single triple.
    @staticmethod
    def region_site(r: Int) -> Int:
        return Self.REGION_SITE_ID

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.REGION_SITE_X)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.REGION_SITE_Y)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](Self.REGION_SITE_Z)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        return True

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 1:
            return Scalar[DTYPE](Self.REGION_X0_1)
        if r == 2:
            return Scalar[DTYPE](Self.REGION_X0_2)
        return Scalar[DTYPE](Self.REGION_X0_0)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 1:
            return Scalar[DTYPE](Self.REGION_Y0_1)
        if r == 2:
            return Scalar[DTYPE](Self.REGION_Y0_2)
        return Scalar[DTYPE](Self.REGION_Y0_0)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 1:
            return Scalar[DTYPE](Self.REGION_X1_1)
        if r == 2:
            return Scalar[DTYPE](Self.REGION_X1_2)
        return Scalar[DTYPE](Self.REGION_X1_0)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 1:
            return Scalar[DTYPE](Self.REGION_Y1_1)
        if r == 2:
            return Scalar[DTYPE](Self.REGION_Y1_2)
        return Scalar[DTYPE](Self.REGION_Y1_0)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def joint_name(k: Int) -> String:
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0


struct So101FamilyConfig[
    P: PlacementTable, HORIZON: Int, SKIP: Int, NMESH: Int,
    FALLBACK_RADIUS: Float64, GRASP_W: Float64, CLOSE_W: Float64,
](Phyics3dEnvConfig):
    """ONE config for every SO-101 task family, over its placement table `P`.

    `So101TabletopConfig` and `So101TowerConfig` below are this type at two
    tables. Everything family-specific — slot addresses, park pose, regions,
    the gripper site, nq/nv — is read from `P`; what the parameters carry is
    what no table holds: the horizon, the substep count, the hull-vertex
    budget and the host sampler's fallback radius. Every hook body is the
    one implementation the tabletop family trained with, unchanged.
    """
    comptime FRAME_SKIP: Int = Self.SKIP
    comptime HAS_GPU_HOOKS: Bool = True
    # ⚠ EULER, AS MuJoCo RUNS THIS MODEL. Neither the Menagerie SO-101 nor the
    # generated scenes set `<option integrator>`, so MuJoCo steps them under
    # Euler; RK4 was this config's inherited default, not the model's. RK4
    # at frame skip 2 ran the whole pipeline 8 times per env step — 8
    # collision launches, 8 Newton solves — for 4x the cost of Euler's 2
    # (PERFORMANCE.md §13.38, the parked-slot probe). Fidelity: the studio
    # path under Euler agrees with MuJoCo to 4.2e-17 over 50 steps on the
    # k=0 and k=13 park scenes (2026-09-07). Position actuators at dt=0.002
    # are what MuJoCo's own default runs them with.
    comptime INTEGRATOR: StaticString = "euler"
    comptime MAX_STEPS: Int = Self.HORIZON
    """The family's `horizon=`. ⚠ RESTATED, NOT READ — a config is a comptime
    TYPE and the `.family` is a runtime file, so this cannot import it. Keep
    them in step by hand; a mismatch changes episode length, not correctness."""

    # ⚠⚠ SO-ARM101 SHIPS A MOCAP BODY (`target`). `Phyics3dBatchedEnv.__init__`
    # RAISES if a mocap-flagged body exists while this is False. Frozen at its
    # XML pose here on purpose: no goal in this family reads it, and a target
    # that moved per episode would make the contact set vary run to run.
    comptime USES_MOCAP: Bool = True

    # ⚠⚠ THE ACTION IS [-1, 1] PER JOINT, mapped affinely onto each
    # actuator's own `ctrlrange`. This defaulted to False — raw control values,
    # clamped — for every commit up to the first training run, and the six
    # ranges it would have been clamping against are
    #
    #     shoulder_pan  +-1.9199    wrist_flex  +-1.6581
    #     shoulder_lift +-1.7453    wrist_roll  -2.7438 .. 2.8412
    #     elbow_flex    +-1.6900    gripper     -0.1745 .. 1.7453
    #
    # `Phyics3dEnvConfig.NORMALIZED_ACTIONS` carries the measurement that
    # settled this on the SAME ROBOT: with one scalar `ACTION_SCALE = 2.0`
    # against that spread, the trained policy commanded an out-of-range pose
    # on 24% to 100% of control steps, `elbow_flex` sat at the tanh rail 49%
    # of the time, and the gripper — asymmetric against a symmetric +-2.0 —
    # was out of range on EVERY step. Past the clamp the gradient is zero, and
    # three successive reward shapes produced the same shaking arm before
    # anyone looked at the clamp.
    #
    # ⚠ SO `action_scale` MUST BE 1.0 in every script that builds an agent for
    # this family. A scale of 2.0 maps [-2, 2] onto the range and puts the
    # useful band back inside the rails — undoing the fix while still looking
    # configured.
    #
    # ⚠ AND "DO NOTHING" IS NO LONGER A ZERO ACTION. Zero maps to the CENTRE
    # of each ctrlrange, which for the gripper is 0.785 rad — half open. The
    # zero-action drivers (`examples/tasks/task_eval_frozen.mojo`,
    # `task_batched_gpu.mojo`) therefore command a pose rather than no torque;
    # they are gating determinism and per-lane goal routing, both of which
    # hold under any fixed action, but their printed numbers move.
    comptime NORMALIZED_ACTIONS: Bool = True

    # ⚠⚠ NONZERO OR THE ARM'S 30 COLLISION MESHES SILENTLY STOP COLLIDING.
    # 0 is not a size hint — both narrow phases gate their mesh branch on
    # `NMESH_VERTS > 0` and emit no contact otherwise.
    #
    # ⚠⚠ THE ARM'S OWN CONSTANT, NOT A NUMBER READ OFF AN ERROR. This said
    # 26198 for one commit — the figure `parse_model_runtime` quoted for this
    # exact scene — and the BATCHED path then demanded 26199. One vertex, two
    # code paths, same model: `dims_from_flat` and the batched env's
    # `ModelDims` do not agree to the last hull vertex.
    #
    # Chasing that one vertex is the wrong response. `so101_park_xml` already
    # records the right rule and I should have followed it: reuse the arm's
    # declared budget, which is correct-by-construction for this robot and
    # comfortably above what either path asks. The props are BOXES — a
    # primitive, not a mesh — so they add no hull vertices at all.
    #
    # ⚠ A drift here is LOUD: `fields_build` raises rather than truncating.
    comptime NMESH_VERTS: Int = Self.NMESH

    comptime INTEGRATOR_WS_EXTRA: Int = 0  # Euler needs no extra workspace

    # ── READ FROM THE TABLE — see `So101TabletopPlacement` for the numbers ──
    comptime N_FREE_SLOTS: Int = Self.P.N_FREE
    comptime N_REGIONS: Int = Self.P.N_REGIONS
    comptime GRIPPER_SITE: Int = Self.P.GRIPPER_SITE
    """The end-effector site id the reward measures reach from — `P` carries
    it because the reward hook gets `site_xpos` but no name table."""
    comptime SLOT_RADIUS: Float64 = Self.FALLBACK_RADIUS
    """The host sampler's fallback radius for a free slot WITHOUT
    `slot_geom=` — every tabletop slot; no tower slot. Passed as a parameter
    because the trait has no word for it: `P.free_radius(j)` is per slot and
    already resolved."""

    comptime SHAPE_W_GRASP: Float64 = Self.GRASP_W
    """Paid EVERY STEP the goal's subject body touches BOTH the gripper body
    and the moving jaw — a contact-defined "grasped", the ManiSkill /
    so101-nexus recipe, over the lane's contact records the tape evaluator
    already reads (`gpu_eval`, L3).

    ⚠⚠ WHY IT EXISTS. `so101_tower_lift_brick`, 100k steps, 2026-09-19: the
    policy reached the brick and PARKED on it — shaped return flat at 363 from
    50k, greedy eval flat from the first one, entropy coefficient 0.2 -> 0.002
    by 16k. Reach saturates at 2 cm and `Above`'s tolerance already pays 0.74
    for a brick RESTING on the desk (3.6 cm shortfall against a 10 cm margin),
    so closing the jaw earned nothing until a lift happened, and a
    deterministic policy never tried. This term is the missing rung: it pays
    for the event that has to precede every lift, one step after it happens.

    ⚠ "BOTH SIDES" IS THE PROXY, not a force test: the fixed jaw is part of the
    gripper body (the wrist-roll print) and the moving jaw is its own body, so
    a brick touching both is pinched or straddled. so101-nexus measured the
    straddle case firing on wide YCB objects and added an opposing-normal test;
    on a 25 mm cube between a 3 cm opening it does not arise, and the lift term
    is what pays for a real pinch anyway.

    0.0 ON THE TABLETOP FAMILY (every number it recorded predates this term);
    0.5 on the tower — the reach term's weight, a rung between reach (0.5) and
    the goal (1.0)."""

    comptime SHAPE_W_CLOSE: Float64 = Self.CLOSE_W
    """Paid for a CLOSED jaw while the pinch centre is within `CLOSE_RADIUS`
    of the goal's subject: `CLOSE_W * (open - q) / (open - closed)`, in [0, 1].

    ⚠⚠ WHY IT EXISTS. Watched in the policy viewer, 2026-09-19: the arm at the
    brick, the moving jaw swung FULLY OPEN the whole time, the wrist nudging
    the cube around like a ball. Nothing paid for closing until a pinch had
    already happened (the grasp rung), and a pinch needs the jaw to close
    first — a one-dimensional decision the policy never took because its
    reward was flat across it. This term makes "close when you are there"
    dense; the rung then pays the pinch, the goal term the lift.

    ⚠ GATED ON DISTANCE, or the policy would drive around with the jaw shut
    and never get the brick between the fingers. 3 cm is the pinch centre's
    reach radius plus a brick half-width. 0.25 on the tower — half the rung,
    a nudge and not the objective; 0.0 on the tabletop."""
    comptime CLOSE_RADIUS: Float64 = 0.03

    comptime GRIPPER_BODY: Int = 6
    comptime JAW_BODY: Int = 7
    """`robot_gripper` and `robot_moving_jaw_so101_v1` in the composed scene:
    the SO-101 is the first attached model in every family, so its bodies come
    first. `tests/tasks/test_so101_tower_config.mojo` pins both by name."""

    comptime GRIPPER_QADR: Int = 5
    comptime GRIPPER_OPEN: Float64 = 1.7453291995659765
    comptime GRIPPER_CLOSED: Float64 = -0.17453297762778586
    """The `robot_gripper` hinge: `qpos[5]` (the sixth arm joint), and its
    range from `so_arm101.xml` — open at the upper limit, closed at the lower.
    The tower gate pins the joint by name and the range against the scene."""



    # ── REWARD SHAPING — see `custom_reward_gpu` for the whole argument ────
    #
    # ⚠⚠ SET EITHER WEIGHT TO 0.0 AND THE REWARD IS SPARSE AGAIN, exactly as
    # it was. That is not a courtesy: every baseline this family has recorded
    # was measured at 0.0, and a shaped run is not comparable with them.
    comptime SHAPE_W_GOAL: Float64 = 1.00
    """Weight on `tolerance(goal_distance)` — generic over the goal language.

    ⚠⚠ THESE ARE NOW WEIGHTS ON A `tolerance` IN [0, 1], NOT ON A CLIPPED
    LINEAR PENALTY, and the reward is POSITIVE. The old form was
    `-w * min(distance, CLIP)`: linear everywhere, hard-clipped, and capped
    below 0.5 in total so that `reward > 0.5` could keep meaning "solved".
    That cap is gone — the goal bit lives in `META_IDX_GOAL_HELD` now — and
    with it the reason the reward could not take the shape that demonstrably
    trains this robot.

    ⚠ WHAT THE OLD FORM COST, measured over ten runs on `so101_gather_bricks`:
    a healthy critic at 0.50/0.25 plateaued at 13% better than random and did
    not move again in 290k steps, and reweighting toward the reach term to
    break that plateau DIVERGED the critic at an identical tracking rate. A
    linear penalty pulls uniformly from any distance and never saturates, so
    its variance is set by how fast the subject moves; `tolerance` saturates
    at both ends, which bounds the per-step signal by construction.

    ⚠ `SoArm101ReachConfig` pays exactly this shape and reaches 3.9 mm on real
    hardware."""

    comptime SHAPE_W_REACH: Float64 = 0.50
    """Weight on `tolerance(gripper-to-subject distance)`.

    ⚠ HALF THE GOAL TERM, NOT SEVEN TIMES IT. The 0.10/0.70 pair that
    destabilised the critic weighted the FAST-moving term heaviest; the goal
    term leads here and the reach term is the assist that gets the arm to the
    object at all. See `SHAPE_W_GOAL` for what the reweighting cost."""

    comptime GOAL_RADIUS: Float64 = 0.0
    """`tolerance`'s upper bound for the goal term — inside it the value is 1.

    ⚠ ZERO, because the goal distance is ALREADY a shortfall:
    `tape_distance_gpu` returns 0 exactly when the predicate holds, so the
    band to be inside is `[0, 0]` and the margin does the rest. A nonzero
    radius here would pay full reward for a goal that is not met."""

    comptime GOAL_MARGIN: Float64 = 0.10
    """Where the goal term has decayed to `value_at_margin`.

    ⚠ 0.10 m IS THE MEASURED SCALE OF THE PROBLEM, not a guess:
    `task_shaping_probe.mojo` measures the goal distance at 0.115-0.139 m
    under a random policy, so a margin of 0.10 puts the random state right in
    the band where the sigmoid has gradient. A margin far below the state
    distribution is the `tolerance` version of a clip in the wrong place — the
    term saturates near zero and says nothing."""

    comptime REACH_RADIUS: Float64 = 0.02
    """Inside 2 cm of the subject the reach term is satisfied — the prop's own
    half-size, so "the gripper is at the block" rather than at a point."""

    comptime REACH_MARGIN: Float64 = 0.20
    """Measured reach distance is 0.120-0.191 m under a random policy, so 0.20
    keeps the whole random distribution on the sigmoid's slope."""


    comptime OBS_MASK_BASE: Int = Self.P.NQ + Self.P.NV
    """Where the `N_FREE_SLOTS` active words start in `obs`.

    ⚠ READ FROM THE TABLE, NOT RESTATED. The model def's `OBS_DIM` is
    `SO101_TABLETOP_OBS_DIM`, defined beside the model def as
    `NQ + NV + N_FREE_SLOTS` — so the number the ENV allocates and the number
    this hook lays out are the same expression, not two copies of a total that
    happen to match today."""

    comptime OBS_GOAL_BASE: Int = Self.OBS_MASK_BASE + Self.N_FREE_SLOTS
    """Where the nine goal words start — gripper(3), subject-gripper(3),
    target-subject(3).

    ⚠ AFTER the mask, so every index the mask gates already test is
    unchanged. Inserting them would have renumbered `OBS_MASK_BASE` and made
    `test_active_mask` pass against a shifted layout."""

    # === CPU hooks — present for the trait; this config is GPU-only ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        pass

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
        # ⚠ THE CPU LEG EVALUATES THROUGH `tasks/eval.eval_goal`, which needs
        # the family and the bound goal — neither of which a static hook can
        # hold. The parity gate therefore drives the CPU side itself rather
        # than through this hook, and this returns zero so that a CPU env
        # wired to this config is obviously inert instead of subtly wrong.
        return (Scalar[DTYPE](0), False)

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
        """The single-env twin of `custom_extract_obs_gpu`.

        ⚠⚠ THE ORDER IS THE CONTRACT, AND THE TWO HOOKS MUST AGREE WORD FOR
        WORD. A batched run writes a checkpoint a single-env eval loads; a
        permutation here is a policy that works on the GPU and is nonsense on
        the CPU, with no error anywhere. `test_active_mask` runs both on one
        state and demands identical vectors — it does not check either against
        a description.

        ⚠ THIS CONFIG IS OTHERWISE GPU-ONLY — `compute_reward_and_done_cpu`
        returns a constant zero, deliberately. The observation is the one hook
        that must work on both, because it is what a checkpoint is shaped by.
        Leaving it to the model default would NOT have been inert: the default
        writes `NQ - 1 + NV` words into a vector this family sizes at
        `NQ + NV + N_FREE_SLOTS`, which is a silently truncated observation,
        not a missing one.
        """
        # ⚠⚠ `d.dims.get_nq()`, NOT `D.NQ`. The comptime members are POISON
        # on the DYNAMIC provider — `DynDims.NQ` is `DIM_POISON`, a negative
        # sentinel — so `range(D.NQ)` copies NOTHING there and the hook
        # returns a three-word observation with no error until something
        # indexes past it. The runtime accessors are correct on BOTH
        # providers, which is why `fields/dims.mojo` has all three families.
        write_task_obs_host[Self.P, DTYPE, D](d, obs)
        _ = m_bodies
        _ = m_joints
        _ = m_geoms
        _ = m_sites
        _ = act
        return True

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

    @staticmethod
    def get_reset_noise() -> Float64:
        # ⚠⚠ THIS WAS 0.0, AND 0.0 MADE EVERY LANE THE SAME PROBLEM. The
        # reasoning was sound as far as it went — "every lane's variation comes
        # from the SAMPLER, seeded by (seed, lane); joint noise on top would
        # add a second source" — and it is wrong for any task whose GOAL does
        # not depend on a placement. `so101_reach_brick` asks the gripper to
        # reach a FIXED region: the brick's sampled pose enters the
        # observation and nothing else, so with zero joint noise all N lanes
        # start in the identical arm pose, every episode, and "the task" is one
        # open-loop trajectory rather than a distribution.
        #
        # ⚠ THE DETERMINISM ARGUMENT SURVIVES. Both streams are seeded — the
        # sampler from `(seed, lane)` and this from `reset_batch`'s seed — so
        # two runs at one seed still agree bit for bit, which is what P4's
        # frozen-init-table gate actually asserts. What zero bought was not
        # reproducibility but the absence of a second source, and the cost of
        # that was a degenerate start distribution.
        #
        # ⚠ 0.05 rad is the value `SoArm101ReachConfig` uses on the SAME
        # robot, so the two reach tasks perturb their starts comparably.
        return 0.05

    # === GPU: pre-step ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        # ⚠ THE REPARK IS IN `pre_step_full_gpu` BELOW, which is the same hook
        # plus `qvel`. Nothing here, and in particular NOTHING THAT TOUCHES
        # `meta` — the tape and the active mask live there.
        pass

    # === GPU: pre-step, with qvel — Gap D's repark ===
    @always_inline
    @staticmethod
    def pre_step_full_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
    ](
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Pin every INACTIVE free slot at its park pose, every step.

        `TASK_LAYER_IMPLEMENTATION.md` Gap D. Gravity is a `Model` field
        shared by the batch, so a parked body FALLS — `reset.reset_slots`
        zeroing its velocity at reset stops the fall compounding across
        episodes and does not stop the fall.

        ⚠⚠ THE POSE **AND** THE VELOCITY, AND POSE-ONLY IS HALF A FIX. Writing
        `qpos` back each step pins where the body IS while the integrator
        keeps adding `g*dt` to where it is GOING: the position looks parked
        and `qvel` grows without bound — 11.8 m/s by the end of a 300-step
        horizon. It never becomes a NaN and it never moves the arm (a parked
        slot is its own kinematic tree), so nothing would have caught it; it
        is simply not what "parked" should mean. Zeroing both makes a parked
        slot's state CONSTANT, which is checkable.

        ⚠ AN ACTIVE SLOT IS NOT TOUCHED. This runs before physics on every
        step, so a stray write here would pin the props the task is about —
        and the reward would read a scene that never moves while the arm
        pushed at it. `test_active_mask` asserts the active slots' words are
        BIT-IDENTICAL across the call.

        ⚠ AND IT MUST NOT TOUCH `meta`. The tape and the active mask live
        there, and this hook also runs at the END of `_reset_env_lane` — a
        write here would land after `init_qpos_gpu` and before the first step.
        """
        # ⚠ THE RULE IS `task_hooks.repark_inactive_slots`, shared with every
        # LIBERO config; this family's park poses reach it through
        # `So101TabletopPlacement`, drift-checked against `family.park_pos`.
        repark_inactive_slots[Self.P, DTYPE, BATCH_SIZE, NQ, NV](
            qpos, qvel, meta, env
        )

    # === GPU: the observation — full state, plus §3.4's active mask ===
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
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_F, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
    ) -> Bool:
        """`qpos` in full, `qvel`, then one active word per free slot.

        ⚠ THE FULL `qpos`, NOT `qpos[1:]`. The model default skips a leading
        word that is a floating base's redundant coordinate on a Gym model and
        is `shoulder_pan` here. See `so101_tabletop_xml.SO101_TABLETOP_OBS_DIM`.

        ⚠⚠ AN INACTIVE SLOT IS ZEROED **AND** FLAGGED. Either alone is a bug:
        zeroing alone reinstates the convention the mask exists to remove, and
        flagging alone leaves a +50 in the vector — a parked slot sits 50 m up
        and falls, because nothing reparks it — which dominates the first
        layer whatever the flag says. `tasks/obs.write_free_slot_obs` does
        both, in one place, so a future reader cannot do one of them.

        ⚠ THE MASK WORD IS READ, NEVER WRITTEN, HERE. The host writes it once
        per episode beside the tape; an observation hook that computed it
        would be deciding what the task is while reporting what the state is.
        """
        # ⚠ THE LAYOUT AND THE GOAL WORDS ARE `task_hooks.write_task_obs`,
        # shared with every LIBERO config. The nine goal words are the reward's
        # own geometry — without them the policy cannot see half its reward
        # (`SHAPE_W_REACH` pays on the gripper-to-subject distance, which is
        # forward kinematics over six joint angles): measured over 190k steps
        # on `gather`, a converged critic and a return that never moved.
        write_task_obs[
            Self.P, DTYPE, BATCH_SIZE, NQ_F, NV_F, NBODY_F,
            SITE_DIM, OBS_DIM,
        ](qpos, qvel, xpos, site_xpos, meta, obs, env)
        _ = xquat
        _ = xvel
        _ = bodies
        _ = contacts
        _ = sites
        _ = geoms
        _ = xipos
        _ = xangvel
        _ = cvel
        _ = cacc
        _ = cfrc_int
        _ = subtree_com
        _ = site_xpos_acc
        _ = xquat_acc
        _ = act
        return True

    # === GPU: reward + termination — constant zero, never done ===
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
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xipos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        xquat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        xvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_F, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        site_xpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            DTYPE,
            Layout.row_major(BATCH_SIZE, MC_F * CONTACT_SIZE),
            MutAnyOrigin,
        ],
        sites: LayoutTensor[
            DTYPE, Layout.row_major(NSITE_F, MODEL_SITE_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        cfrc_ext: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        curriculum: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        xangvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        cacc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        cfrc_int: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 6), MutAnyOrigin
        ],
        subtree_com: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 3), MutAnyOrigin
        ],
        site_xpos_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, SITE_DIM), MutAnyOrigin
        ],
        xquat_acc: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_F * 4), MutAnyOrigin
        ],
        act: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NA_F), MutAnyOrigin
        ],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # ⚠⚠ THE WHOLE REWARD IS THE GOAL. §5.3: sparse, +1 on success,
        # matching LIBERO. Shaping is a per-FAMILY concern expressed in a
        # config, not in a task — a shaped reward is a research choice about
        # one experiment, and putting it in the task file would make two runs
        # incomparable while looking identical.
        # ⚠ THE WIDE OVERLOAD (L3): the hook's own qpos / sites / bodies /
        # contacts operands, so a Joint, Touching, On(obj, obj) or box
        # region evaluates here exactly as on the host. The narrow overload
        # compiles those branches OUT and would read such a term as False.
        var holds = eval_tape_gpu[
            DTYPE, BATCH_SIZE, NBODY_F, SITE_DIM, NQ_F, NSITE_F, MC_F
        ](meta, curriculum, xpos, xquat, site_xpos, qpos, sites, bodies, contacts, env)
        # ⚠ ASKS TO TERMINATE ON SUCCESS. A sparse task that keeps running
        # after the goal is met pays for steps that teach nothing and lets a
        # policy bank the reward repeatedly; the driver's truncation still
        # ends the unsolved ones at MAX_STEPS.
        #
        # ⚠⚠ **AND THE ASK IS IGNORED BY DEFAULT.** `Phyics3dBatchedEnv` takes
        # `TERMINATE_ON_UNHEALTHY` as a comptime parameter DEFAULTING TO
        # FALSE, and then does
        #
        #     comptime if not Self.TERMINATE_ON_UNHEALTHY:
        #         is_terminated = False        # phyics3d_batched_env.mojo:1161
        #
        # so this `Bool` is DISCARDED unless the env was instantiated with the
        # flag, and `_done` then carries only truncation. A driver that wants
        # success-termination must spell it:
        #
        #     Phyics3dBatchedEnv[So101TabletopModel, So101TabletopConfig,
        #                        N_ENVS, TERMINATE_ON_UNHEALTHY=True]
        #
        # ⚠ A DRIVER READING SUCCESS OUT OF `_done` WITHOUT IT READS ZERO —
        # not an error, a constant. `examples/tasks/task_eval_frozen.mojo` did
        # exactly that and reported 0/128 on a task that holds at reset; the
        # eval reads `_reward` instead, which is this hook's other return and
        # needs no flag.
        # ── the shaped reward, and the goal bit that is no longer in it ──
        #
        # ⚠⚠ `tolerance` IN [0, 1] PER TERM, POSITIVE, AND THE SUCCESS SIGNAL
        # IS A SEPARATE `meta` WORD. The reward used to be `+1 if holds` minus
        # a clipped linear penalty, so `reward > 0.5` meant "solved" and every
        # shaping weight had to stay small enough to preserve that. Ten runs
        # on `so101_gather_bricks` say what the linear form cost: a healthy
        # critic plateaued at 13% over random and would not move in 290k
        # steps, and reweighting to break the plateau diverged the critic at
        # an identical tracking rate. A linear penalty pulls uniformly from
        # any distance and never saturates; `tolerance` saturates at both
        # ends, so the per-step signal is bounded by construction and the
        # gradient concentrates where the margin puts it.
        #
        # This is the shape `SoArm101ReachConfig` uses, which reaches 3.9 mm
        # on real hardware on this arm.
        #
        # ⚠ THE GOAL BIT GOES TO `META_IDX_GOAL_HELD` AND NOT INTO `r`. Three
        # files read success out of the reward; they read that word now. A
        # success BONUS in the reward would also be fine, but it is a separate
        # decision from how success is REPORTED, and conflating the two is
        # what capped the shaping in the first place.
        meta[env, META_IDX_GOAL_HELD] = (
            Scalar[DTYPE](1) if holds else Scalar[DTYPE](0)
        )

        var dist = tape_distance_gpu[
            DTYPE, BATCH_SIZE, NBODY_F, SITE_DIM, NQ_F, NSITE_F, MC_F
        ](meta, curriculum, xpos, xquat, site_xpos, qpos, sites, bodies, contacts, env)
        # ⚠⚠ PER LANE, OUT OF `meta` — `curriculum` is ONE row for the whole
        # batch and what a weight is worth depends on the TASK's distance
        # scale. At identical weights and margins the three shipped tasks get
        # a 4.7x spread in reward and 91x in the goal term; see
        # `tasks/shaping.mojo` for the table.
        #
        # ⚠ ZERO IS "NO SHAPING" and is what an untouched `meta` holds, so a
        # driver that never writes these gets the SPARSE reward rather than a
        # shaped one with meaningless parameters.
        var w_goal = rebind[Scalar[DTYPE]](meta[env, META_IDX_SHAPE_W_GOAL])
        var w_reach = rebind[Scalar[DTYPE]](meta[env, META_IDX_SHAPE_W_REACH])
        var m_goal = rebind[Scalar[DTYPE]](meta[env, META_IDX_GOAL_MARGIN])
        var m_reach = rebind[Scalar[DTYPE]](meta[env, META_IDX_REACH_MARGIN])

        # ⚠ `tape_distance_gpu` IS ALREADY A SHORTFALL — zero exactly when the
        # predicate holds — so the band is [0, GOAL_RADIUS] and the margin
        # does the shaping. `SIGMOID_GAUSSIAN` and the default
        # `value_at_margin` match `SoArm101ReachConfig`.
        var goal_t = tolerance[
            SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
        ](
            dist,
            Scalar[DTYPE](0),
            Scalar[DTYPE](Self.GOAL_RADIUS),
            m_goal,
        )
        var r = w_goal * goal_t
        # Each term's own value, for the potential-based mode below (the
        # legacy sum `r` is accumulated exactly as it always was).
        var reach_t = Scalar[DTYPE](0)
        var grasp_b = Scalar[DTYPE](0)
        var close_v = Scalar[DTYPE](0)
        var has_hand = False

        # ⚠ THE REACH TERM READS THE FIRST TERM'S SUBJECT OUT OF THE TAPE.
        # `meta[TASK_PARAM_1]` is term 0's `a`, which for `Near`, `Above`,
        # `On` and `In` is a BODY id — and for `AtRegion` is a SITE id, which
        # is why the op is checked before the distance is taken. A site id
        # read as a body id lands on a real, wrong body.
        var op0 = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0]))
        if op0 == OP_NEAR or op0 == OP_ABOVE or op0 == OP_ON or op0 == OP_IN:
            var sb = Int(
                rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + 1])
            )
            comptime GS = Self.GRIPPER_SITE
            var ex = rebind[Scalar[DTYPE]](site_xpos[env, GS * 3]) - rebind[
                Scalar[DTYPE]
            ](xpos[env, sb * 3])
            var ey = rebind[Scalar[DTYPE]](
                site_xpos[env, GS * 3 + 1]
            ) - rebind[Scalar[DTYPE]](xpos[env, sb * 3 + 1])
            var ez = rebind[Scalar[DTYPE]](
                site_xpos[env, GS * 3 + 2]
            ) - rebind[Scalar[DTYPE]](xpos[env, sb * 3 + 2])
            var reach = sqrt(ex * ex + ey * ey + ez * ez)
            has_hand = True
            reach_t = tolerance[
                SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
            ](
                reach,
                Scalar[DTYPE](0),
                Scalar[DTYPE](Self.REACH_RADIUS),
                m_reach,
            )
            r = r + w_reach * reach_t
            # ── the grasp rung — see `SHAPE_W_GRASP` ─────────────────────
            comptime if Self.GRASP_W > 0.0:
                var ncon = Int(
                    rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS])
                )
                var on_grip = False
                var on_jaw = False
                for k in range(ncon):
                    var base = k * CONTACT_SIZE
                    var ba = Int(
                        rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_BODY_A])
                    )
                    var bb = Int(
                        rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_BODY_B])
                    )
                    if ba == sb or bb == sb:
                        var other = bb if ba == sb else ba
                        if other == Self.GRIPPER_BODY:
                            on_grip = True
                        if other == Self.JAW_BODY:
                            on_jaw = True
                if on_grip and on_jaw:
                    grasp_b = Scalar[DTYPE](1)
                    r = r + Scalar[DTYPE](Self.GRASP_W)
            # ── the closing bonus — see `SHAPE_W_CLOSE` ──────────────────
            comptime if Self.CLOSE_W > 0.0:
                if reach < Scalar[DTYPE](Self.CLOSE_RADIUS):
                    var q = rebind[Scalar[DTYPE]](qpos[env, Self.GRIPPER_QADR])
                    var closed = (Scalar[DTYPE](Self.GRIPPER_OPEN) - q) / Scalar[
                        DTYPE
                    ](Self.GRIPPER_OPEN - Self.GRIPPER_CLOSED)
                    if closed < Scalar[DTYPE](0):
                        closed = Scalar[DTYPE](0)
                    if closed > Scalar[DTYPE](1):
                        closed = Scalar[DTYPE](1)
                    close_v = closed
                    r = r + Scalar[DTYPE](Self.CLOSE_W) * closed

        # ── the POTENTIAL-BASED mode (`META_IDX_REWARD_MODE == 1`) ─────────
        #
        # ⚠⚠ WHY. Every term above is a RAW PER-STEP value, so a state that
        # saturates some of them before the goal holds pays their sum every
        # step for the rest of the episode — the grasped brick held OVER the
        # bowl collects reach + grasp + close + most of the goal term, forever,
        # which can beat finishing. so101-nexus hit exactly this on its
        # pick-and-place ("hover a grasped object above the goal") and fixed it
        # with potential-based shaping (Ng, Harada & Russell, ICML 1999):
        # pay the CHANGE of a potential Phi, which telescopes over an episode
        # to Phi(end) - Phi(start), so dwelling anywhere short of the goal pays
        # ~0 per step. `noeira-docs/SO101_PIXEL_RL_PLAN.md`.
        #
        #   Phi   = w_goal goal + w_reach max(reach, H) + GRASP_W max(grasp, H)
        #           + CLOSE_W max(close, H)                 H = the goal holds
        #   r     = Phi - Phi_prev            (0 on an episode's first step)
        #   r     = W = the weights' sum       while the goal holds
        #         + SUCCESS_BONUS              once, on the first such step
        #
        # ⚠ THE HAND TERMS ARE HELD UP BY H, so releasing the brick in the bowl
        # and backing off — mandatory forward progress — pays no negative
        # delta (nexus's `place_grasp_potential` / `place_reach_potential`).
        # Leaving the goal pays `Phi - W`, negative, as a real regression must.
        # ⚠ WHILE IT HOLDS THE STEP PAYS THE FULL BUDGET, the global maximum of
        # a step (ManiSkill's `reward[success] = max`, nexus's
        # `RewardConfig.compute`), so staying solved beats every other state —
        # which is what a fixed-horizon episode needs to learn to HOLD.
        # ⚠ THE LEGACY MODE IS THE ZERO WORD: nothing below runs and `meta` is
        # not written, so every run and recorder before this block is
        # bit-identical.
        var mode = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_REWARD_MODE]))
        if mode == 1:
            var hh = Scalar[DTYPE](1) if holds else Scalar[DTYPE](0)
            # ⚠ A SINGLE `Near` GOAL IS MEASURED AS A TRANSPORT DISTANCE here
            # (`shaping.near_transport_shortfall`): the 3D shortfall the
            # legacy term uses pays negative for lifting the brick off the
            # desk, which `cube_in_bowl` needs to clear the bowl's rim.
            var goal_pot = goal_t
            var op_near = Int(
                rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0])
            )
            var op_next = Int(
                rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + TERM_WORDS])
            )
            if op_near == OP_NEAR and op_next < 0:
                var na = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + 1]))
                var nb = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + 2]))
                var nr = rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + 3])
                var tsf = near_transport_shortfall[DTYPE](
                    rebind[Scalar[DTYPE]](xpos[env, na * 3]) - rebind[Scalar[DTYPE]](xpos[env, nb * 3]),
                    rebind[Scalar[DTYPE]](xpos[env, na * 3 + 1]) - rebind[Scalar[DTYPE]](xpos[env, nb * 3 + 1]),
                    rebind[Scalar[DTYPE]](xpos[env, na * 3 + 2]) - rebind[Scalar[DTYPE]](xpos[env, nb * 3 + 2]),
                    nr,
                )
                goal_pot = tolerance[
                    SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
                ](
                    tsf,
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](Self.GOAL_RADIUS),
                    m_goal,
                )
                # ⚠⚠ AND IT COUNTS ONLY WHILE THE BRICK IS GRASPED (or the
                # goal holds) — ManiSkill's `place x grasped`. Ungated, a
                # PUSH moves the brick toward the bowl as well as a carry
                # does and is far easier: both PPO arms (26 Sep, 5M) lifted
                # the brick in 9 and 7 of 1024 greedy episodes and reached
                # over the bowl in NONE, their ~2 % "success" being pushes
                # that grazed the radius (3 and 6 still held at the end).
                comptime if Self.GRASP_W > 0.0:
                    if has_hand:
                        var gate = grasp_b if grasp_b > hh else hh
                        goal_pot = goal_pot * gate
            var phi = w_goal * goal_pot
            var wsum = w_goal
            if has_hand:
                var rt = reach_t if reach_t > hh else hh
                var gb = grasp_b if grasp_b > hh else hh
                var cv = close_v if close_v > hh else hh
                phi = phi + w_reach * rt + Scalar[DTYPE](Self.GRASP_W) * gb
                phi = phi + Scalar[DTYPE](Self.CLOSE_W) * cv
                wsum = wsum + w_reach + Scalar[DTYPE](Self.GRASP_W)
                wsum = wsum + Scalar[DTYPE](Self.CLOSE_W)
            var flags = Int(
                rebind[Scalar[DTYPE]](meta[env, META_IDX_EPISODE_FLAGS])
            )
            var rp = Scalar[DTYPE](0)
            if (flags & EPISODE_FLAG_PHI_SET) != 0:
                rp = phi - rebind[Scalar[DTYPE]](meta[env, META_IDX_PHI_PREV])
            if holds:
                rp = wsum
                if (flags & EPISODE_FLAG_BONUS_PAID) == 0:
                    rp = rp + rebind[Scalar[DTYPE]](
                        meta[env, META_IDX_SUCCESS_BONUS]
                    )
                    flags = flags | EPISODE_FLAG_BONUS_PAID
            meta[env, META_IDX_PHI_PREV] = phi
            meta[env, META_IDX_EPISODE_FLAGS] = Scalar[DTYPE](
                flags | EPISODE_FLAG_PHI_SET
            )
            r = rp
        _ = qpos
        _ = qvel
        _ = xipos
        _ = xvel
        _ = bodies
        _ = contacts
        _ = sites
        _ = geoms
        _ = cfrc_ext
        _ = cvel
        _ = actions
        _ = xangvel
        _ = cacc
        _ = cfrc_int
        _ = subtree_com
        _ = site_xpos_acc
        _ = xquat_acc
        _ = act
        _ = step_count
        _ = frame_skip
        _ = timestep
        return (r, holds)
    # === GPU: qpos init — the scene's own pose is already right ===
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
        qpos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DTYPE, Layout.row_major(NJOINT_F, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, NBODY_M * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DTYPE, Layout.row_major(NBODY_M, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[
            DTYPE, Layout.row_major(NGEOM_F, MODEL_GEOM_SIZE), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        # ⚠⚠ THIS USED TO BE `pass`, AND THAT IS WHY NOTHING COULD TRAIN.
        # The note here said the host writes the poses before the step loop —
        # true of the eval and viewer paths, and false of every RESET after
        # the first. `_reset_env_lane` restores the composed scene's `qpos0`,
        # which for a free slot is its PARK pose 50 m up, and only INACTIVE
        # slots are pinned there afterwards by `pre_step_full_gpu`. So an
        # ACTIVE prop began every episode after the first in the sky and fell
        # through the whole horizon with its qpos and qvel in the observation.
        # Nothing raised; the curve just looked like a hard task.
        #
        # ⚠⚠ AND IT WAS THEN A HAND-WRITTEN COPY OF `sample_placements` FOR THIS
        # FAMILY ONLY — one site, one radius, slot-order walk — which is correct
        # here and wrong for every LIBERO family on four counts. The rule now
        # lives once in `placement/table.place_free_slots`, and this family
        # supplies its restated constants through `So101TabletopPlacement`
        # below. `tests/tasks/test_device_placement.mojo` still demands the
        # device and host poses agree on every coordinate, for this family and
        # for all twenty-three LIBERO ones.
        #
        # ⚠⚠ AND THE TAPE MUST SURVIVE THIS. `_reset_env_lane` writes
        # META_IDX_STEP_COUNT and leaves the rest (`gpu/constants.mojo`), and
        # this hook writes only `qpos`/`qvel` — never `meta`. A hook that
        # zeroed `meta` here would blank every lane's goal at the first reset
        # and every reward would read 0: a flat curve, not a crash.
        reset_task_slots[Self.P, DTYPE, BATCH_SIZE, NQ_F, NV_F](
            qpos, qvel, meta, env, seed
        )
        _ = joints
        _ = mocap_pos
        _ = mocap_quat
        _ = bodies
        _ = geoms


# ── THE TWO SO-101 FAMILIES, as this one config at their tables ──────────
#
# ⚠ THE PARAMETERS ARE RESTATED FROM THE `.family` (`horizon=`, `control_freq=`)
# and from the scene's hull count; a config is a comptime type and cannot
# read either. `tests/tasks/test_active_mask.mojo` and
# `tests/tasks/test_so101_tower_config.mojo` assert them.

comptime So101TabletopConfig = So101FamilyConfig[
    So101TabletopPlacement, 300, 2, SO_ARM101_NMESH_VERTS, 0.012, 0.0, 0.0
]
"""`so101_tabletop`: horizon 300, frame skip 2 (a 250 Hz policy on a 2 ms
timestep — what every run on this family has used), the bare arm's hull
budget (the props are boxes), and `cube.xml`'s half-size as the sampler's
radius — see `So101TabletopPlacement.SLOT_RADIUS`; NO grasp term (0.0)."""

comptime So101TowerConfig = So101FamilyConfig[
    So101TowerPlacement, 300, 16, SO101_TOWER_NMESH_VERTS, 0.0226, 0.5, 0.25
]
"""`so101_tower`: horizon 300, frame skip 16 — `control_freq=30` in the
family, 1/30 s / 2 ms = 16.7 substeps, rounded to the integer below (31.25
Hz): the rig records and deploys at 30 fps and a policy stepped here keeps
that cadence. The hull budget is measured on the composed scene
(`tools/tasks/mesh_vertex_budget.mojo` — the wrist camera mount replaces a
stock part and the stand adds four meshes, visual only). The fallback
radius is never consulted: both free slots carry `slot_geom=`; the brick's
half-diagonal is written so a wrong path would still place something sane.

⚠⚠ THE GRASP HOLDS BECAUSE OF `impratio="10"` ON THE ELLIPTIC CONE, and it
did not before. Measured 2026-09-19 with `task_grasp_feasibility.mojo
so101_tower_lift_brick`, holding for 6 s of simulated time at this cadence:
under MuJoCo's defaults (pyramidal, impratio 1) the printed 25 mm cube
crept out of the jaw (held for ~1.5 s at frame skip 2, on the desk after
the longer hold); with the elliptic cone at impratio 1 (the control) it is
still dropped; with impratio 10 — the base asset's `<option>`, restated by
the family's `inherit_option=1` — it is held with 0.5 mm of slip over the
6 s, and the 20 and 30 mm cubes hold too. `impratio` is inert on the
pyramidal cone, which is why the two settings travel together (Menagerie's
SO-100, the Robotiq 2F-85, robosuite). The tabletop family still runs the
defaults; its FEASIBLE verdict was the 1.5 s one."""

comptime So101TowerTeleopConfig = So101FamilyConfig[
    So101TowerPlacement, 1200, 16, SO101_TOWER_NMESH_VERTS, 0.0226, 0.5, 0.25
]
"""`So101TowerConfig` with a 1200-step horizon (about 38 s) instead of 300:
what `examples/so101/tower_teleop_record.mojo` runs. A human on the leader
arm needs ~10 s just to reach and grasp in the sim (Denis, 2026-09-20), and
300 steps is 9.6 s. Every other parameter is the tower's, so the reward the
recorder pays through this config is the trainer's reward exactly; only the
truncation differs, and the replay never sees truncation (`done` stays 0)."""
