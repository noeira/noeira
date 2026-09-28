"""The `.family` and `.task` documents — P1b of `docs/TASK_LAYER_PLAN.md`.

A FAMILY is the compile unit: one base scene, a constant slot table, hence
constant `nq`/`nv`/`ngeom` for every task in it. A TASK is DATA: which slots
are active, where each starts, what the goal is, what the instruction says.

    families/so101_tabletop.family   -> one GPU monomorphisation
    tasks/so101_pick_brick.task      -> no rebuild
    tasks/so101_stack_cubes.task     -> no rebuild

## The format, and why it is not JSON

`key=value` lines with repeating keys, following `data/manifest.mojo` and
checkpoint v2 — the two text formats this tree already reads AND writes.
`io/json.mojo` reads but nothing writes, and the studio must WRITE these.
MJCF is not an option either: `PHYSICS3D_STUDIO_PLAN.md` §3's rule is
"composition is MJCF because MuJoCo is the oracle", and a goal predicate is
precisely what MuJoCo cannot express, so there is no oracle to forfeit.

## ⚠⚠ AN UNKNOWN KEY RAISES. THIS IS A DELIBERATE DIVERGENCE FROM THE MANIFEST.

`data/manifest.mojo` IGNORES unknown keys so that a store written by a newer
build stays readable — right for a format embedded in data files that outlive
the code. It is wrong here. These files are hand-authored, and every key is
load-bearing:

  * a dropped `goal=`   -> a task with no success condition;
  * a dropped `active=` -> a slot silently parked, i.e. a different task;
  * a dropped `init=`   -> an object at its XML pose in every episode.

None of those fail loudly. A typo'd key must therefore be an error, not a
shrug. ⚠ The cost is that a NEWER `.task` cannot be read by an OLDER build —
which is correct, because it means something the older build cannot honour.

## ⚠ WHAT THIS FILE DOES NOT DO

* **It does not parse the goal predicate.** `goal=` is carried as TEXT.
  Predicates are P2 (`predicates.mojo`), and a half-parser here would be a
  second one to keep in step.
* **It does not touch MJCF or the scene.** Composition is P1c
  (`family.mojo`), which CALLS `physics3d/studio`'s composer. §7's dependency
  rule: `tasks/` calls the studio, never reimplements it, and `physics3d`
  never imports `tasks`.
* **It does not validate the park pose against geometry** — see `PARK` below.
"""

from noeira.core.kv import kv_lines, split_on, split_once


comptime SCHEMA_VERSION: Int = 1

# A slot's kind. `free` costs 6 dofs and 7 qpos; `static` costs NEITHER.
#
# ⚠⚠ THE DISTINCTION IS THE WHOLE COST MODEL, and it is measured, not
# stylistic. `docs/TASK_LAYER_IMPLEMENTATION.md` §1.0: on an RTX 5090 at 1024
# lanes, six FREE slots cost 2.74x the bare arm and thirteen is the compile
# ceiling. A STATIC slot adds a body and a geom and no dofs at all, so it pays
# none of that. `docs/TASK_LAYER_PLAN.md` §12 asks whether static slots are
# worth having and answers "probably yes; not needed before P1" — the budget
# measurement upgraded that to "this is the lever", so it is here from day one.
comptime SLOT_FREE: Int = 0
comptime SLOT_STATIC: Int = 1


def slot_kind_from_name(s: String) raises -> Int:
    if s == "free":
        return SLOT_FREE
    if s == "static":
        return SLOT_STATIC
    raise Error(
        "tasks: unknown slot kind '" + s + "' — expected 'free' (a movable"
        " prop, 6 dofs) or 'static' (a fixture, no dofs)"
    )


def slot_kind_name(k: Int) -> String:
    return String("static") if k == SLOT_STATIC else String("free")


struct SlotSpec(Copyable, ImplicitlyCopyable, Movable):
    """`slot=<name>:<kind>:<asset>[:x,y,z[,yaw]]` — one object in the family.

    ⚠⚠ THE POSE IS FOR STATIC SLOTS AND IS REQUIRED ON THEM. A `static` slot
    has NO JOINT, so it cannot be moved after composition — parking is by
    pose, and a body with no dofs has no pose to rewrite. Composing a fixture
    at the park pose therefore welds it 50 m in the air FOREVER, with the
    region attached to its site up there and the sampler dutifully placing
    props into the sky.

    That is not hypothetical: it is what the first `so101_tabletop` family
    did, and nothing caught it because every gate up to P2c CONSTRUCTED
    geometry rather than simulating it. `table_fixture xpos=[10 0 50]`,
    `jntnum=0`.

    ⚠ AND A `free` SLOT MUST NOT CARRY ONE. Its compose-time pose is always
    the park pose and its episode pose comes from the sampler, so a pose here
    would be silently ignored — refused instead.
    """

    var name: String
    var kind: Int
    var asset: String
    var has_pose: Bool
    var px: Float64
    var py: Float64
    var pz: Float64
    var yaw: Float64
    """Rotation about +z, RADIANS, optional fourth number of the pose.

    ⚠ RADIANS REGARDLESS OF THE BASE'S `<compiler angle>`, because the
    composer writes it as a `quat` on the `<frame>`, which MuJoCo never
    interprets through the angle setting. LIBERO places 78 of its fixtures at
    yaw pi (`docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md` G3), which is why a
    static slot needs one at all; the SO-101 families leave it 0 and their
    `.family` text is unchanged."""

    var has_geom: Bool
    """Whether `bottom_z` / `top_z` / `h_radius` were read off the asset.

    ⚠⚠ THE PLACEMENT GEOMETRY IS THE ASSET'S, NOT A CONSTANT, AND IT USED TO BE
    A CONSTANT TWELVE TIMES OVER. `sample_placements` took a `radii` list and
    every caller invented one — `0.02` in five drivers, `CFG.SLOT_RADIUS` in
    four — and the sampler used that ONE number for two different physical
    quantities: the rejection distance between objects AND the height at which
    an object rests (`z = site_z + radius`).

    robosuite objects declare all three, and every one of the 93 LIBERO assets
    has them: `bottom_site` (how far the object's resting bottom is below its
    origin), `top_site` (how far its supporting top is above), and
    `horizontal_radius_site` (whose `sqrt(x^2 + y^2)` is robosuite's
    `horizontal_radius`). `SiteRegionRandomSampler` uses `-bottom_offset[-1]`
    for the height and `horizontal_radius` for the rejection — two numbers, not
    one.

    MEASURED, on `akita_black_bowl`: `bottom_site` is at z = -0.06 while the
    collision geoms reach only -0.012, so LIBERO starts the bowl 4.8 cm higher
    than its own geometry needs and lets it fall. With the hard-coded 0.02 the
    bowl was placed 4 cm LOWER — which on a table is a small interpenetration
    the solver absorbs, and on the stove's `cook_region` (whose site sits at the
    vertical CENTRE of a 4 cm base box) is 136 contacts and a wedged bowl.

    ⚠ FALSE FOR A SLOT WHOSE ASSET DECLARES NONE, and then the caller's `radii`
    entry is used exactly as before — which is what keeps `so101_tabletop` and
    the hand-built test families working unchanged."""
    var bottom_z: Float64
    var top_z: Float64
    var h_radius: Float64

    def __init__(out self, name: String, kind: Int, asset: String):
        self.name = name
        self.kind = kind
        self.asset = asset
        self.has_pose = False
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.yaw = 0.0
        self.has_geom = False
        self.bottom_z = 0.0
        self.top_z = 0.0
        self.h_radius = 0.0

    def set_geom(mut self, bottom_z: Float64, top_z: Float64, h_radius: Float64):
        """Record the asset's own `bottom_site` / `top_site` /
        `horizontal_radius_site`. See `has_geom`."""
        self.has_geom = True
        self.bottom_z = bottom_z
        self.top_z = top_z
        self.h_radius = h_radius

    def geom_describe(self) -> String:
        return (
            self.name + ":" + String(self.bottom_z) + "," + String(self.top_z)
            + "," + String(self.h_radius)
        )

    def __init__(
        out self, name: String, kind: Int, asset: String,
        px: Float64, py: Float64, pz: Float64, yaw: Float64 = 0.0,
    ):
        self.name = name
        self.kind = kind
        self.asset = asset
        self.has_pose = True
        self.px = px
        self.py = py
        self.pz = pz
        self.yaw = yaw
        self.has_geom = False
        self.bottom_z = 0.0
        self.top_z = 0.0
        self.h_radius = 0.0

    def describe(self) -> String:
        var s = self.name + ":" + slot_kind_name(self.kind) + ":" + self.asset
        if self.has_pose:
            s += (
                ":" + String(self.px) + "," + String(self.py)
                + "," + String(self.pz)
            )
            # ⚠ WRITTEN ONLY WHEN NON-ZERO so every existing .family
            # round-trips byte-for-byte; a fourth number is an opt-in.
            if self.yaw != 0.0:
                s += "," + String(self.yaw)
        return s^


# ⚠⚠ THE DEFAULT HALF-HEIGHT OF A REGION, AND IT IS A GUESS BY ADMISSION.
# `eval.IN_HALF_HEIGHT` carried this alone until a region could state its own:
# 12 cm, chosen as "a prop sitting in a bin", applied by `In` and `AtRegion` to
# every region in the tree. It is restated here rather than imported because
# `eval` imports THIS module and the cycle would not close; `tests/tasks/
# test_goal_language.mojo` asserts the two agree.
comptime INIT_TARGET_REGION: Int = 0
comptime INIT_TARGET_SLOT: Int = 1
"""What an `init=`'s target names — see `InitSpec` and `init_target_kind`.

`INIT_TARGET_SLOT` is a STACK: the object goes on top of another free slot,
whose own placement is drawn first. LIBERO's `ObjectBasedSampler`."""

comptime STACK_Z_OFFSET: Float64 = 0.01
"""The robosuite `ObjectBasedSampler(z_offset=0.01)` — the gap it leaves between
a stacked object's bottom and the surface it stands on.

⚠ QUOTED, NOT CHOSEN, and it is not zero for a reason: the reference's
`top_site` is a declared margin rather than its true top, so the centimetre is
what keeps a stack from starting interpenetrated when the two margins disagree.
`bddl_base_domain` passes no `z_offset` for a FIXTURE-SITE region and
`SiteRegionRandomSampler` defaults it to 0.0; it passes none for an object one
either, and `ObjectBasedSampler` defaults it to this. See `TABLE_Z_OFFSET` for
the third case — which the first reading of this got wrong."""

comptime TABLE_Z_OFFSET: Float64 = 0.01
"""LIBERO's `TableRegionSampler(z_offset=0.01)` — the gap a TABLE or FLOOR
region leaves under an object it places.

⚠⚠ THE THIRD SAMPLER, AND IT WAS MISSED. `envs/regions/workspace_region_sampler
.py` declares `class TableRegionSampler(MultiRegionRandomSampler)` with
`z_offset=0.01` in its own signature, and `bddl_base_domain` passes none — so
every prop LIBERO starts on a table or on the floor sits a centimetre above
`workspace_site_z - bottom_offset[-1]`, which is the rule this tree had.
`REGION_SAMPLERS` in `envs/regions/__init__.py` routes `libero_floor
_manipulation`'s `floor` and `libero_tabletop_manipulation`'s `table` to it.

⚠ MEASURED AGAINST LIBERO's OWN FROZEN STATES, not against the code alone.
`libero_object`'s `.pruned_init` gives, for all seven props of
`pick_up_the_alphabet_soup`, z = 0.0150 / 0.0350 / 0.0350 / 0.0000 / 0.0350 /
0.0350 / -0.0050, and `workspace_site_z (-0.035) + 0.01 - bottom_z` reproduces
every one of them exactly while the rule without it misses every one by this
centimetre. On the FLOOR family the cost was not cosmetic: the props started
2.5 cm inside the ground plane, 14 to 96 contacts per reset
(`tests/libero/test_libero_object.mojo`).

⚠ NUMERICALLY EQUAL TO `STACK_Z_OFFSET` AND SEPARATELY SOURCED. Two different
sampler classes in LIBERO declare 0.01 in two different signatures; folding
them into one constant would make a change to either silently follow the
other."""

comptime DEFAULT_REGION_HALF_HEIGHT: Float64 = 0.12


struct RegionSpec(Copyable, ImplicitlyCopyable, Movable):
    """`region=<name>:site:<site>[:xmin,ymin,xmax,ymax[:half_height]]`.

    ⚠ RELATIVE TO A SITE, WHICH IS WHY IT TRAVELS. A region attached to a
    movable slot's site moves with that slot, so "in the box" stays true after
    the box is picked up. This is LIBERO's `:regions` mechanism and it is the
    piece that makes a symbolic goal land on real geometry.

    With no rectangle the region IS the site's own extent.

    ## ⚠⚠ THE HALF-HEIGHT, AND WHY IT IS A FIELD AND NOT A CONSTANT

    A region used to carry an XY rectangle and nothing vertical, so `In` and
    `AtRegion` supplied a z band from `eval.IN_HALF_HEIGHT` — +-0.12 m for
    every region in every family. `eval.mojo`'s own header said what to do
    about it: *"when a task needs a real containment volume, the fix is a
    `height=` on the region, not a tuned constant here."*

    It cost a task to make that concrete. `so101_reach_brick` asks the gripper
    to be `AtRegion(table_top)`, and +-0.12 against a 0.20 x 0.20 rect is a
    0.0096 m^3 box sitting in the middle of the arm's workspace: an UNTRAINED
    greedy actor solved it 64 times out of 64, because a zero action under
    `NORMALIZED_ACTIONS` is the centre of every joint's ctrlrange and that pose
    is inside the box. The task was not hard, it was not even a task.

    ⚠ THE HALF-HEIGHT IS A HALF-HEIGHT, NOT A HEIGHT. The band is
    `[site_z - h, site_z + h]`, centred on the site, because a region's site
    sits ON the surface it describes and containment is wanted on both sides
    of it — `On` is the one-sided predicate and it has its own band.

    ## THE `:box:` KIND — LIBERO's SiteObject, verbatim (L3)

    `region=<name>:box:<site>:xmin,ymin,xmax,ymax:half_height[:<contact slot>]`

    A box region is the site's own `<site type="box" size=>`: the rectangle
    is `(-hx, -hy, hx, hy)` and the half-height is `hz`, read off the asset
    by the importer. It changes what `In` and `On` MEAN for that region —
    `eval.pred_box_in` / `pred_box_under` transcribe LIBERO's `in_box` and
    `under` (site-frame rotation, its 0.01 z slack, its (hz-0.005, hz+0.10)
    band, strict inequalities) instead of the rect test above — and an
    optional CONTACT SLOT names the fixture the object must also touch for
    `On`, which is `SiteObjectState.check_ontop`'s
    `env.check_contact(parent_object, other_object)`. A table target zone
    has no parent and names no slot. `is_box` reaches the device as one
    word of the region table; the site frame is computed there from
    `xquat[site body] * site quat`, the same product `sensors/touch.mojo`
    forms, so nothing new is bound per lane.
    """

    var name: String
    var site: String
    var is_box: Bool
    """`:box:` — LIBERO semantics for In/On (see the header)."""
    var contact: String
    """For a box region: the slot whose bodies `On` must also touch; empty
    when none (a table zone). Refused by `parse_family` if not a slot."""
    var has_rect: Bool
    var x_min: Float64
    var y_min: Float64
    var x_max: Float64
    var y_max: Float64
    var half_height: Float64
    """The z band `In`/`AtRegion` accept, either side of the site."""
    var has_height: Bool
    """Whether the `.family` SAID so. ⚠ NOT DERIVABLE FROM THE VALUE — a
    region that explicitly asks for 0.12 and one that defaulted to it are the
    same number and different statements, and `describe()` must round-trip
    which one it was or `test_spec_roundtrip` silently rewrites the file."""

    def __init__(out self, name: String, site: String):
        self.name = name
        self.site = site
        self.is_box = False
        self.contact = String("")
        self.has_rect = False
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.half_height = DEFAULT_REGION_HALF_HEIGHT
        self.has_height = False

    def __init__(
        out self, name: String, site: String,
        x_min: Float64, y_min: Float64, x_max: Float64, y_max: Float64,
    ):
        self.name = name
        self.site = site
        self.is_box = False
        self.contact = String("")
        self.has_rect = True
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.half_height = DEFAULT_REGION_HALF_HEIGHT
        self.has_height = False

    def __init__(
        out self, name: String, site: String,
        x_min: Float64, y_min: Float64, x_max: Float64, y_max: Float64,
        half_height: Float64,
    ):
        self.name = name
        self.site = site
        self.is_box = False
        self.contact = String("")
        self.has_rect = True
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.half_height = half_height
        self.has_height = True

    def describe(self) -> String:
        var s = self.name + (":box:" if self.is_box else ":site:") + self.site
        if self.has_rect:
            s += (
                ":" + String(self.x_min) + "," + String(self.y_min)
                + "," + String(self.x_max) + "," + String(self.y_max)
            )
            if self.has_height:
                s += ":" + String(self.half_height)
        if self.is_box and self.contact.byte_length() > 0:
            s += ":" + self.contact
        return s^

    def half_x(self) -> Float64:
        return 0.5 * (self.x_max - self.x_min)

    def half_y(self) -> Float64:
        return 0.5 * (self.y_max - self.y_min)


struct InitSpec(Copyable, ImplicitlyCopyable, Movable):
    """`init=<slot>@<target>` — a DISTRIBUTION, not a pose.

    The sampler (P2) draws from the target with rejection. Writing a pose here
    instead would make every episode identical, which is the bug that reads as
    a policy that memorised one placement.

    ## ⚠ THE TARGET IS A REGION *OR* ANOTHER SLOT

    `init=akita_black_bowl_1@main_table_plate_region` draws inside a region.
    `init=akita_black_bowl_1@cookies_1` STACKS the bowl on the cookie box —
    LIBERO's `(On akita_black_bowl_1 cookies_1)`, which it samples with an
    `ObjectBasedSampler` whose x and y ranges are both `[0, 0]`: directly on
    top, at the reference's own x/y.

    ⚠ THE FIELD IS STILL CALLED `region` and both spellings are the bare name,
    with no marker. That is deliberate: `goal=On(akita_black_bowl_1, plate_1)`
    already names a slot where a region could go, and the goal language
    resolves it by lookup. A second convention here would mean two ways to say
    the same thing. `FamilySpec.init_target_kind` is the ONE resolver, and it
    refuses a name that is both.
    """

    var slot: String
    var region: String
    """The target: a region name, or another slot's name. See the header."""
    var inside: Bool
    """`:in` — the `.bddl` said `In`, not `On`.

    ## ⚠⚠ A 4.5 cm DIFFERENCE, AND IT IS ONE COMMENTED-OUT LINE IN LIBERO

    `bddl_base_domain` routes `(On obj <fixture>_<region>)` to
    `SiteRegionRandomSampler` and `(In obj <fixture>_<region>)` to
    `InSiteRegionRandomSampler`. The two `sample()` bodies are the same
    arithmetic with one difference, and it is literally commented out in the
    `In` one:

        #  if on_top:
        #      base_offset += np.array((0, 0, ref_obj.top_offset[-1]))

    So `On` adds the FIXTURE's own `top_site` and `In` does not. Every LIBERO
    fixture declares that site at 0.045, and this tree emitted the same
    placement for both — putting a bowl 4.5 cm into the stove and into the
    cabinet's roof, while the bowl INSIDE the top drawer (an `In`) was right.
    MEASURED against LIBERO's frozen states, which is what found it:
    `akita_black_bowl_1` on `flat_stove_1_cook_region` is 1.0100 and in
    `wooden_cabinet_1_top_region` is 1.15063.

    ⚠ IT IS A PROPERTY OF THE INIT, NOT OF THE REGION, because that is how
    LIBERO decides — on the predicate in `:init`. In the corpus the two never
    name the same region (a drawer's interior is only ever `In`), so keying on
    the region would give the same answer today; keying on the predicate cannot
    drift from the benchmark if that ever changes.

    ⚠ FALSE IS THE DEFAULT AND WRITES NOTHING, so every `.task` that predates
    this round-trips byte for byte."""
    var yaw: Bool
    """`:yaw` — the slot's yaw about +z is drawn uniformly on (-pi, pi] each
    episode (`sampler.sample_placements`, `placement/table.YAW_AXIS_BASE`),
    instead of the identity every other placement starts at.

    ⚠ OPT-IN, PER INIT, BECAUSE IT CHANGES WHAT A POLICY SEES AT RESET. The
    real rig's Duplo lies at any yaw and the operator closes the jaw on its
    faces; an axis-aligned sim cube teaches a grasp the real cube does not
    allow. A region draw only: the clash test compares circles, so the yaw
    never changes WHERE a slot lands, and a STACK (no draw at all) refuses it.
    False writes nothing, so every older `.task` round-trips byte for byte."""
    var sep_mm: Int
    """`:sep=<metres>` — this slot's centre keeps at least this far from
    every other placed slot's centre: the clash test's distance becomes
    `max(h_radius_i + h_radius_j, sep_i, sep_j)` (the exemptions — another
    fixture's region, a stack — still apply). Whole MILLIMETRES, 1..1023: it
    travels to the device in the init word (`placement/table`,
    `INIT_WORD_SEP_UNIT`), and host and device must compare the same number.

    ⚠ OPT-IN, PER INIT: `cube_in_bowl`'s bowl, because the recorded layouts
    never have the brick closer than 135 mm while the radii alone allow 79 —
    and the expert grasps 1/14 of the draws under 100 mm. 0 writes nothing, so
    every older `.task` round-trips byte for byte."""

    def __init__(out self, slot: String, region: String):
        self.slot = slot
        self.region = region
        self.inside = False
        self.yaw = False
        self.sep_mm = 0

    def __init__(
        out self, slot: String, region: String, inside: Bool, yaw: Bool = False,
        sep_mm: Int = 0,
    ):
        self.slot = slot
        self.region = region
        self.inside = inside
        self.yaw = yaw
        self.sep_mm = sep_mm

    def sep(self) -> Float64:
        """The separation in metres (0: none)."""
        return Float64(self.sep_mm) / 1000.0

    def describe(self) -> String:
        return (
            self.slot + "@" + self.region + (":in" if self.inside else "")
            + (":yaw" if self.yaw else "")
            + ((":sep=" + _sep_text(self.sep_mm)) if self.sep_mm > 0 else "")
        )


struct JointInitSpec(Copyable, ImplicitlyCopyable, Movable):
    """`jinit=<joint>@<lo>,<hi>` — a joint's starting value, UNIFORMLY DRAWN.

    ⚠⚠ A RANGE, NOT A VALUE, AND THAT IS LIBERO'S OWN SHAPE. A `(Open X)` in a
    `.bddl`'s `:init` builds an `OpenCloseSampler` over the class's
    `default_open_ranges` and `bddl_base_domain._reset_internal` calls
    `np.random.uniform` on it every reset. Writing one number here would make
    every episode start the drawer at exactly the same opening — the same
    degeneracy `init=` exists to avoid for a placement, and the reason that
    line is a region rather than a pose.

    ⚠ IT IS A JOINT, NOT A SLOT. The `.bddl` says `(Open
    wooden_cabinet_1_top_region)` — a REGION — and the importer resolves that
    to the drawer's joint through `categories.kv` and the asset's own `<site>`.
    Doing that resolution here would need the asset XML at load time, which a
    `.task` reader has no business opening.
    """

    var joint: String
    var lo: Float64
    var hi: Float64

    def __init__(out self, joint: String, lo: Float64, hi: Float64):
        self.joint = joint
        self.lo = lo
        self.hi = hi

    def describe(self) -> String:
        return self.joint + "@" + String(self.lo) + "," + String(self.hi)


comptime DEFAULT_TASK_ROOT = "noeira/tasks"
"""The task layer's own root (SO-101 families). Env packages keep theirs
beside their code, e.g. `noeira/envs/libero/`."""


struct FamilySpec(Movable & Deinitable):
    """The compile unit. Every task in the family instantiates EVERY slot."""

    var schema_version: Int
    var name: String
    var base: String
    var horizon: Int
    var control_freq: Int
    var slots: List[SlotSpec]
    var regions: List[RegionSpec]
    var park_x: Float64
    var park_y: Float64
    var park_z: Float64
    var base_x: Float64
    var base_y: Float64
    var base_z: Float64
    """`base_pos=x,y,z` — where the base asset's root frame is composed.
    Default the origin, which is every SO-101 family. LIBERO stands its
    Panda at `(-0.66, 0, 0.912)` (`libero/categories.kv`), and a robot
    cannot be moved after composition any more than a fixture can."""
    var floor: Bool
    """`floor=0|1` — whether the composer adds its own floor plane and
    light. Default 1. A base whose ARENA slot brings the floor sets 0, or
    the scene has two ground planes."""
    var base_qpos: List[Float64]
    """`base_qpos=q1,q2,...` — the base asset's joint positions at REST, in
    its joint order. Optional; empty means MuJoCo's qpos0 (zeros). The
    composer does not write a keyframe from it (an attached model carries
    none); it is what the MuJoCo oracle and the env's default init read.
    A Panda at all-zeros folds link 5 onto link 7 and reports 18 self
    contacts "at rest" — LIBERO never runs it there, and neither must a
    gate."""
    var base_qpos_jitter: List[Float64]
    """`base_qpos_jitter=h1,h2,...` — a per-episode draw around the rest:
    joint i starts at `base_qpos[i] + h_i * (2u - 1)`, `u` uniform on its own
    Philox axis (`placement/table.BASE_JITTER_AXIS_BASE + i`). Optional; the
    same length as `base_qpos` when given; a half-width of 0 draws nothing.
    The so101-tower follower's rest is folded on three hard stops with pan,
    wrist roll and the jaw left wherever the last episode put them — one
    fixed pose would show the student a single start it never meets."""
    var inherit_option: Bool
    """`inherit_option=0|1` — copy the base asset's `<option>` tag, and its
    `inertiagrouprange` / `autolimits` compiler attributes, into the
    composed scene. Default 0, which leaves every existing family
    byte-identical. LIBERO's physics is `impratio=20 cone=elliptic
    density=1.2 viscosity=2e-5 timestep=0.002`, authored once in the vendored
    Panda; MuJoCo's `<attach>` ignores a child's `<option>`, so without this
    the composed scene would run robosuite's robot under our defaults."""
    var headlight: List[Float64]
    """`headlight=ambient,diffuse,specular` — the composed scene's
    `<visual><headlight>` as three GREY levels. Empty (the default) leaves
    MuJoCo's .1/.4/.5 and every existing family byte-identical.

    ⚠ WHY A FAMILY SETS IT: the headlight's AMBIENT is the only light that
    reaches a surface regardless of its angle. Under MuJoCo's defaults (one
    light straight down + a .1-ambient headlight) the tower's WHITE jaws
    render nearly black from the wrist camera — they are seen edge-on to both
    lights — where the real room, lit from everywhere, shows them white
    (MuJoCo's own renderer agrees with the tracer; checked 2026-09-22)."""
    var sunlight: Float64
    """`sunlight=g` — the composed floor light's (the directional light
    straight down) diffuse GREY level. Negative (the default) leaves MuJoCo's
    .7 and every existing family byte-identical.

    ⚠ AN APPEARANCE CALIBRATION, NOT A LOOK: with the .7 sun plus a lit
    headlight the tower desk rendered CLIPPED WHITE on both cameras (255
    against the real 183 overhead / 155 wrist, 2026-09-24), so no randomized
    look around it could reach the real exposure. The so101_tower values are
    fitted against the rig's recorded frames, see its `.family`."""
    var root: String
    """The TASK ROOT this family was loaded from: the directory holding its
    `families/`, `tasks/` and `scenes/`. `load_family(path)` sets it to the
    grandparent of `path`; default `DEFAULT_TASK_ROOT` (the task layer's
    own). NOT part of the spec — never encoded, never compared — it only
    tells `scene_path` / `task_path` where this family's siblings live, so an
    env package (`noeira/envs/libero/`) can own its families without the
    generic layer naming it."""

    def __init__(out self):
        self.schema_version = SCHEMA_VERSION
        self.name = String("")
        self.base = String("")
        self.horizon = 0
        self.control_freq = 0
        self.slots = List[SlotSpec]()
        self.regions = List[RegionSpec]()
        # ⚠⚠ THE DEFAULT IS HIGH AND LATERAL, NOT `(0, 0, -2)`.
        #
        # `docs/TASK_LAYER_PLAN.md` §4.2 writes the park pose as
        # `park=0.0,0.0,-2.0`. Measured against MuJoCo 3.10.0 on the real
        # SO-ARM101 asset, that pose is a FOUR-CONTACT PENETRATION that ejects
        # the body to z=+36.7 within 1.2 s, because the floor is
        # `size="0 0 0.05"` — an INFINITE plane, so there is no "below" it.
        # At four contacts per slot it also overflows a 16-contact budget by
        # the fourth slot, and an overflowed budget DROPS contacts silently.
        #
        # ⚠ THIS FILE CANNOT CHECK THAT — it has no scene. `family.mojo` (P1c)
        # gates it by composing the scene and asserting the parked slots add
        # no contacts at rest, the way `tools/tasks/gen_park_scenes.py`
        # already does for the P0 probe. The default here is only a default.
        self.park_x = 10.0
        self.park_y = 0.0
        self.park_z = 50.0
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_z = 0.0
        self.floor = True
        self.base_qpos = List[Float64]()
        self.base_qpos_jitter = List[Float64]()
        self.inherit_option = False
        self.headlight = List[Float64]()
        self.sunlight = -1.0
        self.root = String(DEFAULT_TASK_ROOT)

    def __init__(out self, *, deinit move: Self):
        self.schema_version = move.schema_version
        self.name = move.name^
        self.base = move.base^
        self.horizon = move.horizon
        self.control_freq = move.control_freq
        self.slots = move.slots^
        self.regions = move.regions^
        self.park_x = move.park_x
        self.park_y = move.park_y
        self.park_z = move.park_z
        self.base_x = move.base_x
        self.base_y = move.base_y
        self.base_z = move.base_z
        self.floor = move.floor
        self.base_qpos = move.base_qpos^
        self.base_qpos_jitter = move.base_qpos_jitter^
        self.inherit_option = move.inherit_option
        self.headlight = move.headlight^
        self.sunlight = move.sunlight
        self.root = move.root^

    def init_target_kind(self, name: String) raises -> Int:
        """`INIT_TARGET_REGION`, `INIT_TARGET_SLOT`, or raises.

        ⚠ THE ONE RESOLVER. `validate_task_against_family` and
        `sampler.sample_placements` both have to answer "is this target a region
        or a slot", and the device twin will make three — so it is answered
        here. A name that is BOTH is refused rather than resolved by
        precedence: whichever order this file picked, the other reading would be
        someone's intention and nothing would report the mismatch.
        """
        var ri = self.region_index(name)
        var si = self.slot_index(name)
        if ri >= 0 and si >= 0:
            raise Error(
                "family '" + self.name + "': '" + name + "' is BOTH a region"
                " and a slot, so an init naming it is ambiguous. Rename one."
            )
        if ri >= 0:
            return INIT_TARGET_REGION
        if si >= 0:
            return INIT_TARGET_SLOT
        raise Error(
            "family '" + self.name + "': '" + name + "' is neither a region nor"
            " a slot"
        )

    def slot_index(self, name: String) -> Int:
        """Index of the named slot, or -1. Slot ORDER is the observation
        layout, so this is an identity lookup and not a convenience."""
        for i in range(len(self.slots)):
            if self.slots[i].name == name:
                return i
        return -1

    def region_index(self, name: String) -> Int:
        for i in range(len(self.regions)):
            if self.regions[i].name == name:
                return i
        return -1

    def n_free_slots(self) -> Int:
        """How many slots carry a free joint — the number that costs.

        ⚠ THE ONE TO WATCH. §1.0 of the implementation doc prices this
        directly: 6 free slots is 2.74x the bare arm, 13 is the compile
        ceiling. Static slots do not appear here because they cost no dofs.
        """
        var n = 0
        for i in range(len(self.slots)):
            if self.slots[i].kind == SLOT_FREE:
                n += 1
        return n

    def encode(self) -> String:
        var s = String()
        s += "schema_version=" + String(self.schema_version) + "\n"
        s += "family=" + self.name + "\n"
        s += "base=" + self.base + "\n"
        s += "horizon=" + String(self.horizon) + "\n"
        s += "control_freq=" + String(self.control_freq) + "\n"
        s += (
            "park=" + String(self.park_x) + "," + String(self.park_y)
            + "," + String(self.park_z) + "\n"
        )
        # ⚠ THE THREE L2 KEYS ARE WRITTEN ONLY WHEN NON-DEFAULT, so every
        # family that predates them round-trips byte-for-byte.
        if self.base_x != 0.0 or self.base_y != 0.0 or self.base_z != 0.0:
            s += (
                "base_pos=" + String(self.base_x) + "," + String(self.base_y)
                + "," + String(self.base_z) + "\n"
            )
        if not self.floor:
            s += "floor=0\n"
        if len(self.base_qpos) > 0:
            s += "base_qpos="
            for i in range(len(self.base_qpos)):
                if i > 0:
                    s += ","
                s += String(self.base_qpos[i])
            s += "\n"
        if len(self.base_qpos_jitter) > 0:
            s += "base_qpos_jitter="
            for i in range(len(self.base_qpos_jitter)):
                if i > 0:
                    s += ","
                s += String(self.base_qpos_jitter[i])
            s += "\n"
        if self.inherit_option:
            s += "inherit_option=1\n"
        if len(self.headlight) == 3:
            s += (
                "headlight=" + String(self.headlight[0]) + ","
                + String(self.headlight[1]) + "," + String(self.headlight[2])
                + "\n"
            )
        if self.sunlight >= 0.0:
            s += "sunlight=" + String(self.sunlight) + "\n"
        for i in range(len(self.slots)):
            s += "slot=" + self.slots[i].describe() + "\n"
        # ⚠ A SEPARATE LINE, NOT A FIFTH FIELD ON `slot=`. The pose field is
        # the fourth and a FREE slot is refused if it carries one — so the
        # placement geometry cannot ride there without making the pose
        # optional-in-the-middle. Written only for slots that have it, so every
        # `.family` that predates this round-trips byte for byte.
        for i in range(len(self.slots)):
            if self.slots[i].has_geom:
                s += "slot_geom=" + self.slots[i].geom_describe() + "\n"
        for i in range(len(self.regions)):
            s += "region=" + self.regions[i].describe() + "\n"
        return s^


struct TaskSpec(Movable & Deinitable):
    """A binding of values into a family. Costs no rebuild."""

    var schema_version: Int
    var name: String
    var family: String
    var suite: String
    """Which BENCHMARK this task belongs to, when it belongs to one.

    ⚠⚠ NOT DERIVABLE FROM `family` ANY MORE, WHICH IS WHY IT IS A FIELD.
    `libero_goal` and `libero_object` are a suite and a family at once, so the
    two names coincided and nothing needed this. LIBERO-10 and LIBERO-90 are
    two published benchmarks sharing TWENTY SCENES: `KITCHEN_SCENE3`'s five
    tasks are one compile unit and one family, and one of them is scored as
    LIBERO-10 while the other four are LIBERO-90. Reporting a success rate for
    either without this would mean re-reading the corpus, or guessing from a
    name.

    Empty for a task that is not part of a published suite (every SO-101 one),
    and `encode` writes nothing then — so every existing `.task` round-trips
    byte for byte."""
    var language: String
    var active: List[String]
    var inits: List[InitSpec]
    var joint_inits: List[JointInitSpec]
    """`jinit=` lines: a joint's starting value, drawn per episode.

    ⚠ EMPTY FOR EVERY TASK THAT PREDATES THEM, and `encode` writes nothing when
    the list is empty — so every existing `.task` round-trips byte for byte."""
    var goal: String
    """The success predicate, as TEXT. Parsed in P2, not here — see the module
    header. Empty is refused by `parse_task`: a task with no goal always
    succeeds, and a policy trained against it learns nothing while every curve
    looks healthy."""

    def __init__(out self):
        self.schema_version = SCHEMA_VERSION
        self.name = String("")
        self.family = String("")
        self.suite = String("")
        self.language = String("")
        self.active = List[String]()
        self.inits = List[InitSpec]()
        self.joint_inits = List[JointInitSpec]()
        self.goal = String("")

    def __init__(out self, *, deinit move: Self):
        self.schema_version = move.schema_version
        self.name = move.name^
        self.family = move.family^
        self.suite = move.suite^
        self.language = move.language^
        self.active = move.active^
        self.inits = move.inits^
        self.joint_inits = move.joint_inits^
        self.goal = move.goal^

    def is_active(self, slot: String) -> Bool:
        for i in range(len(self.active)):
            if self.active[i] == slot:
                return True
        return False

    def encode(self) -> String:
        var s = String()
        s += "schema_version=" + String(self.schema_version) + "\n"
        s += "task=" + self.name + "\n"
        s += "family=" + self.family + "\n"
        if self.suite.byte_length() > 0:
            s += "suite=" + self.suite + "\n"
        s += "language=" + self.language + "\n"
        s += "goal=" + self.goal + "\n"
        for i in range(len(self.active)):
            s += "active=" + self.active[i] + "\n"
        for i in range(len(self.inits)):
            s += "init=" + self.inits[i].describe() + "\n"
        for i in range(len(self.joint_inits)):
            s += "jinit=" + self.joint_inits[i].describe() + "\n"
        return s^


# ═══════════════════════════════════════════════════════════════════════════
# parsing
# ═══════════════════════════════════════════════════════════════════════════


def _check_version(v: Int, what: String) raises:
    if v > SCHEMA_VERSION:
        raise Error(
            what + ": schema_version " + String(v) + " is newer than this"
            " build supports (" + String(SCHEMA_VERSION) + "). Unlike the"
            " data manifest, a task spec REFUSES what it cannot honour — see"
            " the module header."
        )


def _unknown_key(key: String, lineno: Int, what: String, known: String) raises:
    raise Error(
        what + ": unknown key '" + key + "' on line " + String(lineno)
        + ". Known keys are: " + known + ". A task spec refuses unknown keys"
        " rather than ignoring them — a typo'd `goal` is a task that always"
        " succeeds, and nothing downstream would say so."
    )


def parse_slot(spec: String) raises -> SlotSpec:
    """`<name>:<kind>:<asset>[:x,y,z]`."""
    var parts = split_on(spec, String(":"))
    if len(parts) != 3 and len(parts) != 4:
        raise Error(
            "tasks: malformed slot '" + spec + "' — expected"
            " '<name>:<kind>:<asset>' or '<name>:<kind>:<asset>:x,y,z',"
            " e.g. 'brick:free:props/brick.xml' or"
            " 'table:static:props/table.xml:0.25,0,0.3'"
        )
    var name = String(String(parts[0]).strip())
    var kind = slot_kind_from_name(String(String(parts[1]).strip()))
    var asset = String(String(parts[2]).strip())
    if name.byte_length() == 0 or asset.byte_length() == 0:
        raise Error("tasks: slot has an empty name or asset: '" + spec + "'")

    if len(parts) == 3:
        # ⚠ A STATIC SLOT WITHOUT A POSE IS REFUSED, not defaulted to the
        # origin. The origin is inside the robot's base, so a defaulted
        # fixture would intersect the arm — and the failure would be a scene
        # that loads, simulates, and is wrong. See SlotSpec's header for the
        # 50-m-in-the-air version of the same mistake.
        if kind == SLOT_STATIC:
            raise Error(
                "tasks: static slot '" + name + "' has no pose. A static slot"
                " has NO JOINT, so it cannot be moved after composition —"
                " where it is composed is where it stays. Write"
                " 'slot=" + name + ":static:" + asset + ":x,y,z'."
            )
        return SlotSpec(name^, kind, asset^)

    if kind == SLOT_FREE:
        raise Error(
            "tasks: free slot '" + name + "' carries a pose. A free slot is"
            " composed at the family's park pose and placed by the SAMPLER at"
            " reset, so a pose here would be silently ignored. Drop it, or"
            " make the slot static."
        )
    var n = split_on(String(String(parts[3]).strip()), String(","))
    if len(n) != 3 and len(n) != 4:
        raise Error(
            "tasks: slot '" + name + "' pose needs 'x,y,z' or 'x,y,z,yaw'"
            " (yaw in radians about +z), got '" + String(parts[3]) + "'"
        )
    var yaw = 0.0
    if len(n) == 4:
        yaw = Float64(String(String(n[3]).strip()))
    return SlotSpec(
        name^, kind, asset^,
        Float64(String(String(n[0]).strip())),
        Float64(String(String(n[1]).strip())),
        Float64(String(String(n[2]).strip())),
        yaw,
    )


def parse_region(spec: String) raises -> RegionSpec:
    """`<name>:site:<site>[:xmin,ymin,xmax,ymax[:half_height]]`.

    ⚠ `site` IS SPELLED OUT rather than assumed, so that a later target kind
    (a body, a geom) is an added token and not a format change. Anything else
    raises today instead of being silently read as a site.
    """
    var parts = split_on(spec, String(":"))
    if len(parts) < 3 or len(parts) > 6:
        raise Error(
            "tasks: malformed region '" + spec + "' — expected"
            " '<name>:site:<site>', '<name>:site:<site>:xmin,ymin,xmax,ymax'"
            " or that with a fifth field, the half-height:"
            " '<name>:site:<site>:xmin,ymin,xmax,ymax:0.03'; or a box:"
            " '<name>:box:<site>:xmin,ymin,xmax,ymax:hz[:contact slot]'"
        )
    var name = String(String(parts[0]).strip())
    var kind = String(String(parts[1]).strip())
    var site = String(String(parts[2]).strip())
    var is_box = kind == "box"
    if kind != "site" and not is_box:
        raise Error(
            "tasks: region '" + name + "' targets '" + kind + "'; only"
            " 'site' and 'box' are supported. A region is site-relative so"
            " that it TRAVELS with a movable slot — see RegionSpec."
        )
    if name.byte_length() == 0 or site.byte_length() == 0:
        raise Error("tasks: region has an empty name or site: '" + spec + "'")
    if is_box and len(parts) < 5:
        raise Error(
            "tasks: box region '" + name + "' needs its rectangle AND its"
            " half-height — both are the site's `size` and LIBERO's `in_box`"
            " / `under` read all three: '" + spec + "'"
        )
    if not is_box and len(parts) == 6:
        raise Error(
            "tasks: region '" + name + "' has a sixth field; only a `:box:`"
            " region carries a contact slot: '" + spec + "'"
        )
    if len(parts) == 3:
        return RegionSpec(name^, site^)

    var nums = split_on(String(String(parts[3]).strip()), String(","))
    if len(nums) != 4:
        raise Error(
            "tasks: region '" + name + "' rectangle needs exactly four"
            " numbers (xmin,ymin,xmax,ymax), got " + String(len(nums))
        )
    var x0 = Float64(String(String(nums[0]).strip()))
    var y0 = Float64(String(String(nums[1]).strip()))
    var x1 = Float64(String(String(nums[2]).strip()))
    var y1 = Float64(String(String(nums[3]).strip()))
    # ⚠ ORDER IS CHECKED. A reversed rectangle is not an error the sampler can
    # see — it just never accepts a draw, and bounded retries then RAISE with
    # "exhausted", which points at the sampler rather than at this line.
    if x1 <= x0 or y1 <= y0:
        raise Error(
            "tasks: region '" + name + "' has an empty or reversed rectangle"
            " (xmin,ymin must be < xmax,ymax): '" + spec + "'"
        )
    if len(parts) == 4:
        return RegionSpec(name^, site^, x0, y0, x1, y1)

    # ⚠ A FIFTH FIELD WITHOUT A RECTANGLE IS UNREACHABLE — `len(parts) == 5`
    # implies parts[3] was the rect, which the block above already parsed. A
    # height on a rect-less region would have nothing to be the height OF: the
    # region is then the site's own extent and `eval` renders that as a token
    # radius, not a volume this could size.
    var hh = Float64(String(String(parts[4]).strip()))
    # ⚠⚠ A ZERO OR NEGATIVE BAND ACCEPTS NOTHING, and nothing about that reads
    # as an error downstream: `pred_in_rect` simply returns False for every
    # state, the goal is never met, and the run looks like a task the policy
    # cannot solve. Refused here, where the number is.
    if hh <= 0.0:
        raise Error(
            "tasks: region '" + name + "' has half-height " + String(hh)
            + ", which accepts no point at all. `pred_in_rect` would return"
            " False for every state and the task would read as unlearnable"
            " rather than as malformed."
        )
    var out = RegionSpec(name^, site^, x0, y0, x1, y1, hh)
    out.is_box = is_box
    if len(parts) == 6:
        out.contact = String(String(parts[5]).strip())
        if out.contact.byte_length() == 0:
            raise Error(
                "tasks: box region '" + out.name + "' has an empty contact"
                " slot field; drop the trailing ':' instead: '" + spec + "'"
            )
    return out^


def _sep_text(mm: Int) -> String:
    """`135` -> `0.135`: the metres a `.task` writes, exactly what it read."""
    var frac = String(mm % 1000)
    while frac.byte_length() < 3:
        frac = "0" + frac
    return String(mm // 1000) + "." + frac


def parse_init(spec: String) raises -> InitSpec:
    """`<slot>@<region>[:in|:on][:yaw][:sep=<metres>]` — see `InitSpec.inside`,
    `InitSpec.yaw` and `InitSpec.sep_mm` for the suffixes."""
    var parts = split_once(spec, String("@"))
    if len(parts) != 2:
        raise Error(
            "tasks: malformed init '" + spec + "' — expected"
            " '<slot>@<region>[:in]', e.g. 'brick@table'"
        )
    var slot = String(String(parts[0]).strip())
    var rest = String(String(parts[1]).strip())
    var inside = False
    var yaw = False
    var sep_mm = 0
    var sc = rest.rfind(":sep=")
    if sc >= 0:
        var txt = String(rest[byte = sc + 5 : rest.byte_length()])
        var metres = Float64(txt)
        var mm = Int(metres * 1000.0 + 0.5)
        if mm < 1 or mm > 1023 or abs(metres * 1000.0 - Float64(mm)) > 1e-6:
            raise Error(
                "tasks: init '" + spec + "' — ':sep=' takes metres in whole"
                " millimetres, 0.001..1.023 (it travels to the device as mm)"
            )
        sep_mm = mm
        var head_s = String(rest[byte=0:sc])
        rest = head_s^
    if rest.endswith(":yaw"):
        yaw = True
        var head0 = String(rest[byte=0 : rest.byte_length() - 4])
        rest = head0^
    var colon = rest.rfind(":")
    if colon >= 0:
        var tail = String(rest[byte = colon + 1 : rest.byte_length()])
        # ⚠ ONLY `in` AND `on` ARE ACCEPTED. A region name cannot contain a
        # colon, so anything else here is a typo and must not be read as part
        # of the name — a silently mangled target resolves to nothing and the
        # error would name the region, not the line.
        if tail == "in" or tail == "on":
            inside = tail == "in"
            # ⚠ A TEMPORARY: assigning `String(rest[...])` back into `rest`
            # aliases the argument the initialiser is reading.
            var head = String(rest[byte=0:colon])
            rest = head^
        else:
            raise Error(
                "tasks: init '" + spec + "' ends in ':" + tail + "'; the only"
                " suffixes are ':in' / ':on', then ':yaw', then ':sep=M'"
                " (see InitSpec)"
            )
    if slot.byte_length() == 0 or rest.byte_length() == 0:
        raise Error("tasks: init has an empty slot or region: '" + spec + "'")
    if rest.find(":") >= 0:
        raise Error(
            "tasks: init '" + spec + "' — the suffixes go ':in' / ':on'"
            " first, then ':yaw'; a region name has no colon"
        )
    return InitSpec(slot^, rest^, inside, yaw, sep_mm)


def parse_joint_init(spec: String) raises -> JointInitSpec:
    """`<joint>@<lo>,<hi>`."""
    var parts = split_once(spec, String("@"))
    if len(parts) != 2:
        raise Error(
            "tasks: malformed jinit '" + spec + "' — expected"
            " '<joint>@<lo>,<hi>', e.g."
            " 'wooden_cabinet_1_top_level@-0.16,-0.14'"
        )
    var joint = String(String(parts[0]).strip())
    var r = split_on(String(parts[1]), String(","))
    if joint.byte_length() == 0 or len(r) != 2:
        raise Error(
            "tasks: jinit '" + spec + "' needs a joint and a 'lo,hi' range"
        )
    var lo = Float64(String(String(r[0]).strip()))
    var hi = Float64(String(String(r[1]).strip()))
    # ⚠ `OpenCloseSampler.__init__` asserts the same thing. Reversed, every
    # `np.random.uniform(low=hi, high=lo)` draws OUTSIDE the interval.
    if hi < lo:
        raise Error(
            "tasks: jinit '" + spec + "' has hi < lo — a range that samples"
            " outside itself"
        )
    return JointInitSpec(joint^, lo, hi)


def _parse_flag(val: String, what: String) raises -> Bool:
    """`0` or `1`, nothing else — a `yes` that read as False would be silent."""
    if val == "1":
        return True
    if val == "0":
        return False
    raise Error("family spec: " + what + " must be 0 or 1, got '" + val + "'")


def parse_family(text: String) raises -> FamilySpec:
    var f = FamilySpec()
    var saw_version = False
    var lines = kv_lines(text, String("family spec"))

    for i in range(len(lines)):
        var key = lines[i].key
        var val = lines[i].value
        if key == "schema_version":
            f.schema_version = Int(val)
            _check_version(f.schema_version, String("family spec"))
            saw_version = True
        elif key == "family":
            f.name = val
        elif key == "base":
            f.base = val
        elif key == "horizon":
            f.horizon = Int(val)
        elif key == "control_freq":
            f.control_freq = Int(val)
        elif key == "park":
            var p = split_on(val, String(","))
            if len(p) != 3:
                raise Error(
                    "family spec: park needs three numbers 'x,y,z', got '"
                    + val + "'"
                )
            f.park_x = Float64(String(String(p[0]).strip()))
            f.park_y = Float64(String(String(p[1]).strip()))
            f.park_z = Float64(String(String(p[2]).strip()))
        elif key == "base_pos":
            var b = split_on(val, String(","))
            if len(b) != 3:
                raise Error(
                    "family spec: base_pos needs three numbers 'x,y,z', got '"
                    + val + "'"
                )
            f.base_x = Float64(String(String(b[0]).strip()))
            f.base_y = Float64(String(String(b[1]).strip()))
            f.base_z = Float64(String(String(b[2]).strip()))
        elif key == "floor":
            f.floor = _parse_flag(val, String("floor"))
        elif key == "base_qpos":
            var q = split_on(val, String(","))
            f.base_qpos = List[Float64]()
            for k in range(len(q)):
                f.base_qpos.append(Float64(String(String(q[k]).strip())))
            if len(f.base_qpos) == 0:
                raise Error("family spec: base_qpos is empty")
        elif key == "base_qpos_jitter":
            var q = split_on(val, String(","))
            f.base_qpos_jitter = List[Float64]()
            for k in range(len(q)):
                var h = Float64(String(String(q[k]).strip()))
                if not (h >= 0.0):
                    raise Error(
                        "family spec: base_qpos_jitter is a half-width, >= 0;"
                        " got " + String(h)
                    )
                f.base_qpos_jitter.append(h)
        elif key == "inherit_option":
            f.inherit_option = _parse_flag(val, String("inherit_option"))
        elif key == "headlight":
            var h = split_on(val, String(","))
            if len(h) != 3:
                raise Error(
                    "family spec: headlight needs three grey levels"
                    " 'ambient,diffuse,specular', got '" + val + "'"
                )
            f.headlight = List[Float64]()
            for k in range(3):
                var v = Float64(String(String(h[k]).strip()))
                if v < 0.0 or v > 1.0:
                    raise Error("family spec: headlight level outside [0, 1]: " + val)
                f.headlight.append(v)
        elif key == "sunlight":
            var g = Float64(String(val.strip()))
            if g < 0.0 or g > 1.0:
                raise Error("family spec: sunlight grey level outside [0, 1]: " + val)
            f.sunlight = g
        elif key == "slot":
            f.slots.append(parse_slot(val))
        elif key == "slot_geom":
            # ⚠ RESOLVED AGAINST THE SLOTS SEEN SO FAR, so a `slot_geom=` for a
            # slot declared LATER raises rather than being silently dropped —
            # which would restore the hard-coded radius for that one object and
            # nothing would report it.
            var g = split_on(val, String(":"))
            if len(g) != 2:
                raise Error(
                    "tasks: malformed slot_geom '" + val + "' — expected"
                    " '<slot>:<bottom_z>,<top_z>,<h_radius>'"
                )
            var gname = String(String(g[0]).strip())
            var gn = split_on(String(String(g[1]).strip()), String(","))
            if len(gn) != 3:
                raise Error(
                    "tasks: slot_geom '" + val + "' needs three numbers:"
                    " bottom_z, top_z, h_radius"
                )
            var gi = -1
            for si in range(len(f.slots)):
                if f.slots[si].name == gname:
                    gi = si
            if gi < 0:
                raise Error(
                    "tasks: slot_geom names slot '" + gname + "', which no"
                    " earlier slot= line declares"
                )
            f.slots[gi].set_geom(
                Float64(String(String(gn[0]).strip())),
                Float64(String(String(gn[1]).strip())),
                Float64(String(String(gn[2]).strip())),
            )
        elif key == "region":
            f.regions.append(parse_region(val))
        else:
            _unknown_key(
                key, lines[i].lineno, String("family spec"),
                String("schema_version, family, base, horizon, control_freq,"
                       " park, base_pos, floor, base_qpos, base_qpos_jitter, sunlight,"
                       " inherit_option, slot,"
                       " slot_geom, region"),
            )

    if not saw_version:
        raise Error("family spec: no schema_version line")
    if f.name.byte_length() == 0:
        raise Error("family spec: no family= name")
    if f.base.byte_length() == 0:
        raise Error("family spec: no base= scene")
    if f.horizon <= 0:
        raise Error("family spec: horizon must be > 0, got " + String(f.horizon))
    if len(f.base_qpos_jitter) > 0 and len(f.base_qpos_jitter) != len(f.base_qpos):
        raise Error(
            "family spec: base_qpos_jitter has " + String(len(f.base_qpos_jitter))
            + " words but base_qpos has " + String(len(f.base_qpos))
        )

    # ⚠ DUPLICATE NAMES ARE REFUSED. Slot ORDER is the observation layout and
    # the instance prefix is the identity, so two slots sharing a name is two
    # different objects addressed by one key — a silent aliasing bug in the
    # scene, in the obs and in every goal that names it.
    for i in range(len(f.slots)):
        for j in range(i + 1, len(f.slots)):
            if f.slots[i].name == f.slots[j].name:
                raise Error(
                    "family spec: duplicate slot name '" + f.slots[i].name + "'"
                )
    for i in range(len(f.regions)):
        for j in range(i + 1, len(f.regions)):
            if f.regions[i].name == f.regions[j].name:
                raise Error(
                    "family spec: duplicate region name '"
                    + f.regions[i].name + "'"
                )
    # ⚠ A BOX REGION'S CONTACT SLOT MUST BE A SLOT. It binds to that slot's
    # root body at load time; a typo would otherwise surface as "no body
    # named X_" from `slot_body_id`, pointing at the goal instead of here.
    for i in range(len(f.regions)):
        ref r = f.regions[i]
        if r.is_box and r.contact.byte_length() > 0:
            if f.slot_index(r.contact) < 0:
                raise Error(
                    "family spec: box region '" + r.name + "' names contact"
                    " slot '" + r.contact + "', which is not a slot of"
                    " family '" + f.name + "'"
                )
    return f^


def parse_task(text: String) raises -> TaskSpec:
    var t = TaskSpec()
    var saw_version = False
    var lines = kv_lines(text, String("task spec"))

    for i in range(len(lines)):
        var key = lines[i].key
        var val = lines[i].value
        if key == "schema_version":
            t.schema_version = Int(val)
            _check_version(t.schema_version, String("task spec"))
            saw_version = True
        elif key == "task":
            t.name = val
        elif key == "family":
            t.family = val
        elif key == "suite":
            t.suite = val
        elif key == "language":
            t.language = val
        elif key == "goal":
            t.goal = val
        elif key == "active":
            t.active.append(val)
        elif key == "init":
            t.inits.append(parse_init(val))
        elif key == "jinit":
            t.joint_inits.append(parse_joint_init(val))
        else:
            _unknown_key(
                key, lines[i].lineno, String("task spec"),
                String("schema_version, task, family, suite, language, goal,"
                       " active, init, jinit"),
            )

    if not saw_version:
        raise Error("task spec: no schema_version line")
    if t.name.byte_length() == 0:
        raise Error("task spec: no task= name")
    if t.family.byte_length() == 0:
        raise Error("task spec: no family= name")
    # ⚠ AN EMPTY GOAL IS REFUSED, not defaulted. A task with no success
    # condition trains against a flat-zero reward and every curve looks
    # healthy while nothing is learned — the same shape as a config wired to
    # the batched env without GPU hooks (`phyics3d_env_config.HAS_GPU_HOOKS`).
    if t.goal.byte_length() == 0:
        raise Error(
            "task spec '" + t.name + "': no goal= predicate. A task with no"
            " goal always succeeds and trains against a flat-zero reward."
        )
    for i in range(len(t.active)):
        for j in range(i + 1, len(t.active)):
            if t.active[i] == t.active[j]:
                raise Error(
                    "task spec: slot '" + t.active[i] + "' listed active twice"
                )
    return t^


def order_inits(t: TaskSpec, f: FamilySpec) raises -> List[InitSpec]:
    """The ONE canonical order for a task's `init=` lines.

    Family slot order, adjusted so that a STACK follows the slot it stands on.
    Stable: it repeatedly emits the first slot-ordered init whose reference is
    already out, so the result is a pure function of `(t, f)`.

    ## ⚠⚠ WHY THE ORDER IS A RULE AT ALL

    Two samplers draw these placements — the host's `sample_placements`, which
    walks `t.inits` and rejects each draw against the ones already placed, and
    the device's, which walks the FREE SLOT TABLE because a per-lane region
    index is all `meta` can carry. Rejection is order-dependent by
    construction, so if the two orders differ the two samplers produce
    DIFFERENT scenes from one `(seed, lane)` and the eval path and the training
    path silently disagree about where the props are.

    Requiring the file to be sorted made the two orders identical by
    construction. A stack is the one thing that cannot obey it:
    `libero_spatial` declares `akita_black_bowl_1` at slot 3 and `cookies_1` at
    slot 5, and `(On akita_black_bowl_1 cookies_1)` needs the cookie box drawn
    FIRST — a stack takes the reference's own x/y/z.

    ⚠ THE DEVICE NOW WALKS THIS FUNCTION'S RULE. `placement/table.
    place_free_slots` repeats the same loop over the per-lane init words —
    first slot-ordered init whose stack reference has been walked, restart —
    so a task with a stack is device-samplable, and `tests/tasks/
    test_device_placement.mojo` gates it on the three corpus tasks that stack.
    (It replaced `gpu_eval.require_gpu_placement`, which refused every one.)
    For a task with NO stack this returns exactly slot order.

    ⚠ A CYCLE RAISES. `(On a b)` with `(On b a)` has no first draw; emitting
    them in an arbitrary order would place one on the other's PREVIOUS episode
    pose, which is a scene that looks sampled and is not.
    """
    var ordered = List[InitSpec]()
    var taken = List[Bool](length=len(t.inits), fill=False)
    for _pass in range(len(t.inits)):
        var progressed = False
        for si in range(len(f.slots)):
            for k in range(len(t.inits)):
                if taken[k] or t.inits[k].slot != f.slots[si].name:
                    continue
                var ready = True
                if f.region_index(t.inits[k].region) < 0:
                    var needs = False
                    for q in range(len(t.inits)):
                        if t.inits[q].slot == t.inits[k].region:
                            needs = True
                    if needs:
                        ready = False
                        for q in range(len(ordered)):
                            if ordered[q].slot == t.inits[k].region:
                                ready = True
                if not ready:
                    continue
                ordered.append(t.inits[k])
                taken[k] = True
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            break
    if len(ordered) != len(t.inits):
        var stuck = String("")
        for k in range(len(t.inits)):
            if not taken[k]:
                stuck += " " + t.inits[k].describe()
        raise Error(
            "task '" + t.name + "': a cycle in its init= stacking —" + stuck
            + ". Each of these waits on another to be placed first."
        )
    return ordered^


def has_stacked_init(t: TaskSpec, f: FamilySpec) raises -> Bool:
    """Does any `init=` stand on another SLOT rather than a region?"""
    for i in range(len(t.inits)):
        if f.init_target_kind(t.inits[i].region) == INIT_TARGET_SLOT:
            return True
    return False


def validate_task_against_family(t: TaskSpec, f: FamilySpec) raises:
    """⚠ THIS IS WHAT MAKES THE BUDGET REAL — `TASK_LAYER_PLAN.md` §4.4.

    "A task cannot introduce an object the family did not declare." Without
    this check that rule is a comment: a `.task` naming an unknown slot would
    compose a scene missing it, and the failure would surface as a policy that
    cannot reach something the instruction talks about.

    Checked here, at the SPEC level, because it needs no scene and no MuJoCo —
    so it is available to the studio while a human is typing.
    """
    if t.family != f.name:
        raise Error(
            "task '" + t.name + "' declares family '" + t.family
            + "' but was validated against '" + f.name + "'"
        )
    for i in range(len(t.active)):
        if f.slot_index(t.active[i]) < 0:
            raise Error(
                "task '" + t.name + "': active slot '" + t.active[i]
                + "' is not declared by family '" + f.name + "'. A task binds"
                " values into the family's slot table; it cannot add to it."
                " If it needs a new object, that is a NEW FAMILY and a rebuild."
            )
    for i in range(len(t.inits)):
        var slot = t.inits[i].slot
        var region = t.inits[i].region
        if f.slot_index(slot) < 0:
            raise Error(
                "task '" + t.name + "': init names slot '" + slot
                + "', which family '" + f.name + "' does not declare"
            )
        # ⚠ AN INIT FOR A PARKED SLOT IS AN ERROR, NOT A NO-OP. It reads as an
        # object that should be on the table, and it would be silently parked.
        if not t.is_active(slot):
            raise Error(
                "task '" + t.name + "': init places slot '" + slot + "' but it"
                " is not listed active, so it would be PARKED and the init"
                " ignored. Add 'active=" + slot + "' or drop the init."
            )
        # ⚠ A REGION OR ANOTHER SLOT — `init_target_kind` is the one resolver.
        var kind: Int
        try:
            kind = f.init_target_kind(region)
        except e:
            raise Error(
                "task '" + t.name + "': init places '" + slot + "' on '"
                + region + "', which family '" + f.name + "' declares neither"
                " as a region nor as a slot"
            )
        if kind == INIT_TARGET_SLOT:
            # ⚠⚠ A STACK'S REFERENCE MUST BE PLACED, AND PLACED FIRST. Its x/y/z
            # ARE the reference's, so a reference that is parked puts the stack
            # 50 m in the air, and one placed later puts it on wherever the
            # reference happened to be LAST EPISODE. The importer's topological
            # order guarantees the ordering; this refuses a hand-written `.task`
            # that gets it wrong, and a self-reference, which would otherwise
            # read its own uninitialised placement.
            if region == slot:
                raise Error(
                    "task '" + t.name + "': init stacks '" + slot + "' on"
                    " ITSELF"
                )
            if not t.is_active(region):
                raise Error(
                    "task '" + t.name + "': init stacks '" + slot + "' on '"
                    + region + "', which is not active — a parked reference"
                    " puts the stack at the park pose, 50 m up"
                )
            var ref_first = False
            for k in range(len(t.inits)):
                if t.inits[k].slot == region:
                    ref_first = k < i
            var rsi = f.slot_index(region)
            if rsi >= 0 and f.slots[rsi].kind == SLOT_FREE and not ref_first:
                raise Error(
                    "task '" + t.name + "': init stacks '" + slot + "' on free"
                    " slot '" + region + "', whose own init comes LATER (or is"
                    " missing). A stack takes the reference's placement, so the"
                    " reference must be drawn first."
                )
    # ⚠ AN ACTIVE SLOT WITH NO INIT is allowed and deliberate: a fixture
    # (`static`) has a pose from the scene and nothing to sample. A FREE slot
    # without an init would start at its XML pose in every episode, which is a
    # real authoring mistake, so that one is refused.
    for i in range(len(t.active)):
        var si = f.slot_index(t.active[i])
        if si < 0 or f.slots[si].kind != SLOT_FREE:
            continue
        var found = False
        for j in range(len(t.inits)):
            if t.inits[j].slot == t.active[i]:
                found = True
        if not found:
            raise Error(
                "task '" + t.name + "': free slot '" + t.active[i] + "' is"
                " active but has no init=, so it would start at its XML pose"
                " in EVERY episode — an identical placement every time, which"
                " reads as a policy that memorised one layout."
            )

    # ⚠⚠ THE ORDER IS `order_inits`, AND IT IS CHECKED RATHER THAN RE-DERIVED
    # HERE. That function's header carries the whole argument: two samplers draw
    # these placements, rejection is order-dependent, and a task with a STACK is
    # the one case where slot order is not the right answer. Comparing against
    # it keeps the rule in one place.
    var want_order = order_inits(t, f)
    for i in range(len(t.inits)):
        if t.inits[i].slot != want_order[i].slot:
            raise Error(
                "task '" + t.name + "': init= lines are out of order — '"
                + t.inits[i].slot + "' where '" + want_order[i].slot + "' is"
                " expected. The order is family slot order, except that a"
                " stack (`init=x@y` where y is a slot) must follow the slot it"
                " stands on. The host sampler walks this list and the device"
                " sampler re-derives it from the slot table; rejection sampling is"
                " order-dependent, so a different order gives the two DIFFERENT"
                " scenes from one (seed, lane) — the eval and the training run"
                " would then disagree about where the props are, and nothing"
                " else would look wrong."
            )


def task_root_of(path: String) -> String:
    """`<root>/families/x.family` -> `<root>`. A path with fewer than two
    directory levels falls back to `DEFAULT_TASK_ROOT`."""
    var parts = path.split("/")
    if len(parts) < 3:
        return String(DEFAULT_TASK_ROOT)
    var out = String()
    for i in range(len(parts) - 2):
        if i > 0:
            out += "/"
        out += parts[i]
    return out^


def load_family(path: String) raises -> FamilySpec:
    var f: FamilySpec
    with open(path, "r") as fh:
        f = parse_family(fh.read())
    f.root = task_root_of(path)
    return f^


def load_task(path: String) raises -> TaskSpec:
    with open(path, "r") as fh:
        return parse_task(fh.read())
