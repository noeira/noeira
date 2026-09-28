"""Placement sampling — `init=brick@table_top` is a DISTRIBUTION — P2b.

    var r = SampleReport()
    var placed = sample_placements(task, family, frames, radii, seed, lane, r)

`TASK_LAYER_PLAN.md` §6.1: rejection sampling over the declared regions,
seeded, deterministic given `(task, seed, lane)`, bounded retries, and **RAISE
on exhaustion rather than silently returning an overlapping scene** — the
silent version is a bug that surfaces as a policy that cannot learn.

## ⚠⚠ PURE GEOMETRY, ON PURPOSE — IT HAS TO RUN ON DEVICE

This file takes region FRAMES and slot RADII as plain numbers and returns
poses. It never touches `Data`, `Model` or MuJoCo. That is not minimalism: P3
resets 1024 lanes on device, so the sampler must be callable from a kernel,
and a version that reached into `Data` here would be rewritten there — two
implementations of one distribution, drifting.

The caller supplies the frames because IT knows where the sites are: on the
CPU authoring path from `Data.site_xpos` after FK, on the GPU path from the
same tensor, already per-lane.

## ⚠⚠ THE SEED IS SALTED AWAY FROM THE ENV'S RESET NOISE, DELIBERATELY

`ModelDefFromXML.reset_env_gpu` seeds its joint-noise Philox with
`seed * 2654435761 + env * 12345` (`model_def_from_xml.mojo:1742`). Reusing
that spelling here would draw placements from THE SAME STREAM as the arm's
joint noise, so a lane's object layout and its starting pose would be
perfectly correlated across the whole batch — a hidden confound that no test
would report and that reads, much later, as a policy that has learned the
correlation instead of the task.

`PLACEMENT_SALT` is what makes them independent streams. ⚠ It is a magic
number and it is SUPPOSED to be: the requirement is only that it differ, and
recording why is worth more than choosing it cleverly.

## ⚠ COUNTER-BASED, NOT A STATEFUL STREAM

`std.random.philox` is seeded per draw from `(seed, lane, slot, attempt)`
rather than advanced. A stateful stream gives lane 7 a different draw
depending on how many attempts lanes 0-6 needed — so a scene would depend on
its neighbours, reruns would not reproduce, and the CPU and GPU legs could not
be compared at all. Counter-based, every draw is a pure function of its
coordinates.
"""

from std.math import pi
from std.random.philox import Random as PhiloxRandom

from .spec import (
    FamilySpec, TaskSpec, SLOT_FREE, INIT_TARGET_SLOT, STACK_Z_OFFSET,
    TABLE_Z_OFFSET,
)


# ⚠ BOUNDED, AND EXHAUSTION RAISES. An unbounded retry loop on an
# over-constrained region hangs a training run at reset with no diagnostic;
# a loop that gives up and returns the last draw produces an overlapping
# scene, which MuJoCo resolves by launching the objects apart on step 1.
#
# ⚠ `MAX_PLACE_ATTEMPTS` AND `PLACEMENT_SALT` ARE DEFINED IN
# `placement/table.mojo` and re-exported here — the device kernel reads the same
# two names, where they used to be restated on the config and asserted equal.
from .placement.table import (
    MAX_PLACE_ATTEMPTS, PLACEMENT_SALT, JOINT_AXIS_BASE, BASE_JITTER_AXIS_BASE,
    YAW_AXIS_BASE,
)


struct Placement(Copyable, ImplicitlyCopyable, Movable):
    """Where one slot starts this episode, in world coordinates."""

    var slot: Int
    var x: Float64
    var y: Float64
    var z: Float64
    var yaw: Float64
    """About +z, radians; 0 (the identity) unless the init says `:yaw`."""

    def __init__(
        out self, slot: Int, x: Float64, y: Float64, z: Float64,
        yaw: Float64 = 0.0,
    ):
        self.slot = slot
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw


struct RegionFrame(Copyable, ImplicitlyCopyable, Movable):
    """A region's site, in WORLD coordinates, this episode.

    ⚠ RESOLVED PER EPISODE BY THE CALLER, not stored on the family. A region
    attached to a movable slot's site TRAVELS with it — that is the whole
    reason regions are site-relative — so its world frame is only known after
    forward kinematics.
    """

    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z


struct SampleReport(Copyable, ImplicitlyCopyable, Movable):
    """Draws attempted vs accepted.

    ⚠⚠ THE ATTEMPT COUNT IS NOT DIAGNOSTICS, IT IS THE ANTI-VACUITY CHECK. A
    rejection sampler whose rejection branch never runs is indistinguishable
    from one with no rejection at all, and both pass "it returned poses". The
    gate asserts this counter MOVES on a crowded region.
    """

    var attempts: Int
    var accepted: Int
    var exempt: Int
    """Horizontal overlaps the clash test SKIPPED — two different regions with
    either anchored, or an object already placed as a stack. Counted so a gate
    can show its corpus reaches that branch; a skip that never fires is
    indistinguishable from one that is not there."""
    var clamped: Int
    """How many axes collapsed to a point because the object does not fit the
    region it was asked to start in.

    ⚠ COUNTED, NOT SILENT. LIBERO lets the inset range INVERT and draws the
    reversed interval — wider than the region and partly outside it. We clamp to
    the centre instead, which keeps the object inside its region but removes the
    randomisation on that axis; a caller that never looks would report a
    per-episode distribution that is a single point. `libero_spatial`'s top
    drawer is the case: a 3.5 cm bowl in a 3.0 x 7.6 cm interior."""

    var sep_rejected: Int
    """Clashes that ONLY a `:sep=` made (the radii alone would have accepted
    the draw) — the separation's own anti-vacuity counter."""

    def __init__(out self):
        self.clamped = 0
        self.attempts = 0
        self.accepted = 0
        self.exempt = 0
        self.sep_rejected = 0

    def rejected(self) -> Int:
        return self.attempts - self.accepted


@always_inline
def _uniform01(seed: UInt64, lane: Int, axis: Int, attempt: Int) -> Float64:
    """One uniform draw, a PURE FUNCTION of its coordinates.

    ⚠ USES PHILOX'S OWN COUNTERS — `subsequence` and `offset` — rather than
    hashing everything into `seed`. That is what the generator is FOR: the
    three are independent counter axes with guaranteed-decorrelated streams,
    whereas XOR-folding coordinates into a seed can collide two different
    (lane, axis) pairs onto one stream and nothing would report it.

    ⚠ One value per call, not four. `step_uniform` returns four, but reusing
    them would make a draw depend on which SLOT of a batch it came from, and
    the coordinates would have to encode that. One value per
    (lane, axis, attempt) keeps every draw reproducible even when attempt
    counts differ between runs — which is exactly what rejection sampling
    makes happen.
    """
    var rng = PhiloxRandom(
        seed=seed ^ PLACEMENT_SALT,
        subsequence=(UInt64(lane) << 16) | UInt64(axis),
        offset=UInt64(attempt),
    )
    var v = rng.step_uniform()
    return Float64(v[0])


# ⚠ `JOINT_AXIS_BASE` — where a `jinit=` draw's Philox axis starts — is defined
# in `placement/table.mojo` beside the kernel that draws the same numbers.


def sample_joint_inits(
    t: TaskSpec, seed: UInt64, lane: Int
) raises -> List[Float64]:
    """One uniform draw per `jinit=`, in task order. Deterministic in
    `(seed, lane)`.

    ⚠⚠ A DRAW, BECAUSE LIBERO DRAWS. `bddl_base_domain._reset_internal` builds
    an `OpenCloseSampler` over the class's `default_open_ranges` and calls
    `np.random.uniform(low, high)` on it every reset — so a drawer does NOT
    start at its threshold, it starts somewhere in [-0.16, -0.14]. A constant
    would make every episode of every seed open it identically, which is the
    degeneracy `init=` exists to avoid one axis over.

    ⚠ NO REJECTION, AND NONE IS WANTED. A placement is rejected against the
    objects already placed; a joint value has nothing to overlap. So this takes
    `attempt = 0` always, and the draw is a pure function of `(seed, lane, k)`.
    """
    var out = List[Float64]()
    for k in range(len(t.joint_inits)):
        ref j = t.joint_inits[k]
        var u = _uniform01(seed, lane, JOINT_AXIS_BASE + k, 0)
        out.append(j.lo + u * (j.hi - j.lo))
    return out^


def sample_base_qpos(f: FamilySpec, seed: UInt64, lane: Int) -> List[Float64]:
    """The family's rest pose for one episode: `base_qpos[i] + h_i * (2u - 1)`
    with `h` = `base_qpos_jitter=` and `u` on Philox axis
    `BASE_JITTER_AXIS_BASE + i`, attempt 0 — `placement/table.reset_task_slots`
    draws the same numbers on the device. A word without jitter (or a family
    without the key) is `base_qpos[i]` exactly, and draws nothing."""
    var out = List[Float64]()
    for i in range(len(f.base_qpos)):
        var q = f.base_qpos[i]
        if i < len(f.base_qpos_jitter) and f.base_qpos_jitter[i] != 0.0:
            var u = _uniform01(seed, lane, BASE_JITTER_AXIS_BASE + i, 0)
            q += f.base_qpos_jitter[i] * (2.0 * u - 1.0)
        out.append(q)
    return out^


def sample_placements(
    t: TaskSpec,
    f: FamilySpec,
    frames: List[RegionFrame],
    radii: List[Float64],
    seed: UInt64,
    lane: Int,
    mut report: SampleReport,
) raises -> List[Placement]:
    """Every `init=` in the task, placed. Deterministic in `(seed, lane)`.

    `frames` is indexed by FAMILY REGION INDEX; `radii` by FAMILY SLOT INDEX.

    ⚠ INITS ARE PLACED IN TASK ORDER and each is rejected against the ones
    ALREADY placed. That makes the result order-dependent — which is correct
    and must stay stable, because changing the order changes every episode of
    every seed. `spec.mojo` preserves file order for exactly this reason.
    """
    if len(frames) != len(f.regions):
        raise Error(
            "tasks: sampler got " + String(len(frames)) + " region frames for"
            " a family with " + String(len(f.regions)) + " regions. The"
            " caller must resolve EVERY region, in family order."
        )
    if len(radii) != len(f.slots):
        raise Error(
            "tasks: sampler got " + String(len(radii)) + " slot radii for a"
            " family with " + String(len(f.slots)) + " slots."
        )

    var out = List[Placement]()
    # ⚠ WHICH REGION EACH ACCEPTED PLACEMENT CAME FROM, parallel to `out`.
    # `Placement` is the device twin's contract too (`family_config`'s reset
    # hook builds the same four numbers), so the region index is kept beside it
    # rather than added to it.
    var of_region = List[Int]()
    # `:sep=` of each accepted placement, parallel to `out` (metres; 0 = none)
    var sep_of = List[Float64]()
    for i in range(len(t.inits)):
        var si = f.slot_index(t.inits[i].slot)

        # ── a STACK: the target is another slot, not a region ─────────────
        #
        # ⚠⚠ NO DRAW AND NO REJECTION, AND BOTH ARE robosuite's.
        # `bddl_base_domain` samples `(On obj other_obj)` with an
        # `ObjectBasedSampler` whose `x_ranges` and `y_ranges` are both
        # `[[0.0, 0.0]]` and whose `ensure_valid_placement` is False: the object
        # goes at the reference's own x/y, on top of it, full stop. The
        # randomisation that matters already happened when the REFERENCE was
        # drawn, and a stack that wandered would slide off the box it is meant
        # to be standing on.
        #
        # ⚠ THE REFERENCE MUST ALREADY BE IN `out`. `validate_task_against_family`
        # refuses a task whose stack precedes its reference and the importer
        # orders them topologically; this re-checks because the index is used.
        if f.init_target_kind(t.inits[i].region) == INIT_TARGET_SLOT:
            if t.inits[i].yaw:
                raise Error(
                    "tasks: init '" + t.inits[i].describe() + "' stacks, and"
                    " a stack is not drawn: ':yaw' applies to a region draw"
                )
            var rsi = f.slot_index(t.inits[i].region)
            var found = -1
            for j in range(len(out)):
                if out[j].slot == rsi:
                    found = j
            if si < 0 or found < 0:
                raise Error(
                    "tasks: init '" + t.inits[i].describe() + "' stacks on '"
                    + t.inits[i].region + "', which has not been placed yet."
                    " Run validate_task_against_family first."
                )
            ref sls = f.slots[si]
            ref slr = f.slots[rsi]
            if not (sls.has_geom and slr.has_geom):
                raise Error(
                    "tasks: init '" + t.inits[i].describe() + "' stacks '"
                    + f.slots[si].name + "' on '" + f.slots[rsi].name
                    + "', but one of them has no slot_geom=. A stack needs the"
                    " reference's `top_site` and the object's `bottom_site`;"
                    " there is no constant that stands in for either."
                )
            # robosuite: base = ref_pos + (0, 0, top_offset[-1]);
            #            z = z_offset + base.z - bottom_offset[-1]
            var sz = (
                out[found].z + slr.top_z + STACK_Z_OFFSET - sls.bottom_z
            )
            out.append(Placement(si, out[found].x, out[found].y, sz))
            of_region.append(-1)
            sep_of.append(0.0)
            report.attempts += 1
            report.accepted += 1
            continue

        var ri = f.region_index(t.inits[i].region)
        if si < 0 or ri < 0:
            # `validate_task_against_family` refuses this long before here;
            # re-checked because the sampler indexes with the results.
            raise Error(
                "tasks: init '" + t.inits[i].describe() + "' does not resolve"
                " against family '" + f.name + "'. Run"
                " validate_task_against_family first."
            )

        ref reg = f.regions[ri]
        ref fr = frames[ri]
        var placed = False
        for attempt in range(MAX_PLACE_ATTEMPTS):
            report.attempts += 1
            # ⚠ BEFORE THE DRAW: the inset above needs the radius.
            ref sl = f.slots[si]
            var rest = -sl.bottom_z if sl.has_geom else radii[si]
            var rad_i = sl.h_radius if sl.has_geom else radii[si]
            var x = fr.x
            var y = fr.y
            if reg.has_rect:
                var u = _uniform01(seed, lane, si * 2, attempt)
                var v = _uniform01(seed, lane, si * 2 + 1, attempt)
                # ⚠⚠ A FIXTURE-ANCHORED REGION SAMPLES HALF ITS RECT, AND THEN
                # SHRINKS BY THE OBJECT'S RADIUS. LIBERO builds a DIFFERENT
                # sampler for `(On obj <fixture>_<region>)` than for a table
                # region — `bddl_base_domain._add_placement_initializer` routes
                # it to `conditioned_initial_place_state_on_sites` with
                #
                #     x_ranges=[[-size[0] / 2, size[0] / 2]]
                #     ensure_object_boundary_in_range=True
                #
                # where `size` is MuJoCo's HALF-size. So the draw spans half the
                # site's half-extent, further inset by the object's
                # `horizontal_radius`. A table region is the bddl's own
                # `:ranges` and is used whole.
                #
                # ⚠ IT IS WHAT MADE THE DRAWER TASK SEED-DEPENDENT. On the full
                # rect the bowl could land against the top drawer's inner wall:
                # 0 contacts at the viewer's seed and 19 at the gate's, which is
                # the worst kind of bug to find later.
                #
                # ⚠ AND WE CLAMP WHERE LIBERO INVERTS. With a 3.5 cm bowl in a
                # 3.0 x 7.6 cm drawer the inset range crosses over, and
                # `np.random.uniform(low=hi, high=lo)` happily draws the
                # reversed interval — WIDER than the region it names, and partly
                # outside it. Collapsing to the centre keeps the object inside
                # the region it was asked to start in; `SampleReport.clamped`
                # counts it so a family whose slots do not fit is visible
                # instead of silently un-randomised.
                var x0 = reg.x_min
                var x1 = reg.x_max
                var y0 = reg.y_min
                var y1 = reg.y_max
                if reg.contact.byte_length() > 0:
                    x0 *= 0.5
                    x1 *= 0.5
                    y0 *= 0.5
                    y1 *= 0.5
                    x0 += rad_i
                    x1 -= rad_i
                    y0 += rad_i
                    y1 -= rad_i
                    if x1 < x0:
                        var xc = 0.5 * (x0 + x1)
                        x0 = xc
                        x1 = xc
                        report.clamped += 1
                    if y1 < y0:
                        var yc = 0.5 * (y0 + y1)
                        y0 = yc
                        y1 = yc
                        report.clamped += 1
                x = fr.x + x0 + u * (x1 - x0)
                y = fr.y + y0 + v * (y1 - y0)
            # ⚠ RESTING ON THE SURFACE, not centred in it. The site is on the
            # face a region describes, so an object's CENTRE sits one radius
            # above it. Placing it AT the site starts every episode with the
            # prop half inside the table, which the solver resolves by
            # ejecting it — a scene that looks sampled and is not.
            #
            # ⚠⚠ THE HEIGHT AND THE REJECTION RADIUS ARE TWO DIFFERENT NUMBERS,
            # AND THEY USED TO BE ONE. `z = fr.z + radii[si]` treated the
            # caller's radius as a resting half-height, and twelve call sites
            # supplied `0.02` or a config constant. robosuite reads them
            # separately: `SiteRegionRandomSampler.sample` puts the ORIGIN at
            # `site_z - bottom_offset[-1]` (the asset's `bottom_site`) and
            # rejects within `other.horizontal_radius + horizontal_radius` (its
            # `horizontal_radius_site`). MEASURED across the pack: 93 assets,
            # 10 distinct triples, radius spanning 0.005 to 0.3 — so one
            # constant was wrong by up to 15x. And `akita_black_bowl`'s
            # `bottom_site` is -0.06 against the hard-coded 0.02, i.e. 4 cm too
            # low: on a table an overlap the solver absorbs, on the stove's
            # `cook_region` (whose site is at the vertical CENTRE of a 4 cm base
            # box) 136 contacts and a wedged bowl.
            #
            # ⚠ `has_geom` FALSE FALLS BACK TO THE CALLER'S RADIUS, unchanged —
            # `so101_tabletop` and the hand-built test families declare no
            # robosuite sites and their numbers must not move.
            #
            # ⚠⚠ AND A TABLE OR FLOOR REGION ADDS A CENTIMETRE. LIBERO routes
            # `(On obj <table>_<region>)` to `TableRegionSampler`, whose own
            # signature carries `z_offset=0.01`, while a FIXTURE-site region
            # goes to `SiteRegionRandomSampler` at 0.0 — see
            # `spec.TABLE_Z_OFFSET`, which quotes both and records the
            # `.pruned_init` measurement that settled it. The discriminator is
            # the same one the half-rect rule above uses: a region that names a
            # CONTACT slot is anchored to a fixture or an object, and one that
            # does not is the arena's own workspace.
            #
            # ⚠ IT IS ADDED ONLY WHEN THE SLOT CARRIES THE ASSET's SITES. A
            # family with no `slot_geom=` is using the caller's radius as a
            # resting height and is not a LIBERO family; moving it would change
            # `so101_tabletop`'s scenes for no reason.
            var z_off = 0.0
            if sl.has_geom:
                if reg.contact.byte_length() == 0:
                    z_off = TABLE_Z_OFFSET
                elif not t.inits[i].inside:
                    # ⚠⚠ AND `On` A FIXTURE ADDS THE FIXTURE'S OWN `top_site`.
                    # `SiteRegionRandomSampler.sample` builds `base_offset` as
                    # the reference's pose PLUS `ref_obj.top_offset[-1]`, and
                    # `InSiteRegionRandomSampler` has that exact line COMMENTED
                    # OUT — see `spec.InitSpec.inside`. Every LIBERO fixture
                    # declares `top_site` at 0.045, so a bowl `On` the stove or
                    # the cabinet roof sat 4.5 cm inside it while the bowl `In`
                    # the drawer was right.
                    var ci = f.slot_index(reg.contact)
                    if ci < 0:
                        raise Error(
                            "tasks: region '" + reg.name + "' names contact"
                            " slot '" + reg.contact + "', which the family does"
                            " not declare."
                        )
                    if not f.slots[ci].has_geom:
                        raise Error(
                            "tasks: init '" + t.inits[i].describe() + "' places"
                            " a prop ON region '" + reg.name + "', whose"
                            " fixture '" + reg.contact + "' has no slot_geom=."
                            " robosuite adds that fixture's `top_site` to the"
                            " height and there is no constant standing in for"
                            " it — regenerate the family."
                        )
                    z_off = f.slots[ci].top_z
            var z = fr.z + rest + z_off

            var clash = False
            for j in range(len(out)):
                var dx = out[j].x - x
                var dy = out[j].y - y
                ref sj = f.slots[out[j].slot]
                var rad_j = sj.h_radius if sj.has_geom else radii[out[j].slot]
                var rr = rad_i + rad_j
                # `:sep=` — the larger of the radii's sum and either slot's
                # separation (`InitSpec.sep_mm`); the device reads the same
                # millimetres out of the init words
                var r_geom = rr
                rr = max(rr, max(t.inits[i].sep(), sep_of[j]))
                if dx * dx + dy * dy < rr * rr:
                    # ⚠⚠ TWO OBJECTS IN DIFFERENT FIXTURE REGIONS DO NOT CLASH,
                    # AND THE HORIZONTAL TEST ALONE SAYS THEY DO. The drawer
                    # task puts one bowl INSIDE the cabinet's top drawer and
                    # another ON its roof: horizontally almost coincident, and
                    # separated by a shelf. The 2-D test refused the scene after
                    # 64 attempts — correct arithmetic, wrong question.
                    #
                    # ⚠ AND A VERTICAL EXTENT TEST DOES NOT FIX IT, which is
                    # worth recording because it is the obvious next move.
                    # `bottom_site`/`top_site` are generous MARGINS (the bowl
                    # claims 10 cm against 6.5 cm of real geometry) and the
                    # drawer's interior box is 20 cm tall, so a bowl placed at
                    # that region's site floats in the middle of it and falls to
                    # the drawer floor during the settle. Comparing pre-settle
                    # extents has the two bowls overlapping by 6 cm when the
                    # settled scene has them 7 cm apart with a shelf between.
                    #
                    # So the question is WHICH REGION each object is in: two
                    # objects in the SAME region can collide and must reject;
                    # two in different regions where either is anchored to a
                    # fixture are separated by that fixture's own geometry, and
                    # whether they fit is the benchmark's business. Two
                    # unanchored (table) regions still reject, which is what
                    # keeps several props on one workspace from overlapping —
                    # the behaviour every existing family has.
                    # ⚠⚠ A STACK CARRIES `of_region = -1`, AND THIS USED TO
                    # INDEX `f.regions[-1]` WITH IT. The comment here said a
                    # stack is exempt; the code reached `f.regions[ri_j]` for a
                    # TABLE region's draw overlapping a stack, and Mojo's `List`
                    # asserts on a negative index — `mojo run` CRASHED. No
                    # corpus seed had reached it because a stack sits on its
                    # reference, which rejects the draw first unless the stack
                    # is the wider of the two. It is now the exemption the
                    # comment described: a stack sits ON a placed object by
                    # construction, and what it overlaps is that object's
                    # business. `tests/tasks/test_device_placement.mojo`
                    # builds the case.
                    var ri_j = of_region[j]
                    if ri_j < 0:
                        report.exempt += 1
                        continue
                    if ri_j != ri:
                        if (
                            reg.contact.byte_length() > 0
                            or f.regions[ri_j].contact.byte_length() > 0
                        ):
                            report.exempt += 1
                            continue
                    if dx * dx + dy * dy >= r_geom * r_geom:
                        report.sep_rejected += 1
                    clash = True
            if not clash:
                # `:yaw` — its own axis, attempt 0: the clash test is by
                # circles, so the yaw never moves the placement above
                var yaw = 0.0
                if t.inits[i].yaw:
                    var uy = _uniform01(seed, lane, YAW_AXIS_BASE + si, 0)
                    yaw = (2.0 * uy - 1.0) * pi
                out.append(Placement(si, x, y, z, yaw))
                of_region.append(ri)
                sep_of.append(t.inits[i].sep())
                report.accepted += 1
                placed = True
                break

        if not placed:
            raise Error(
                "tasks: could not place slot '" + t.inits[i].slot + "' in"
                " region '" + t.inits[i].region + "' after "
                + String(MAX_PLACE_ATTEMPTS) + " attempts — every draw"
                " overlapped an object already placed. The region is too"
                " small for the objects the task puts in it. ⚠ This RAISES"
                " rather than returning an overlapping scene: an overlap is"
                " resolved by the solver ejecting the props on step 1, which"
                " reads as a policy that cannot learn, far from the cause."
            )
    return out^


def parked_pose(f: FamilySpec, slot_index: Int) -> List[Float64]:
    """Where an INACTIVE slot goes. Re-exported from `family` so that a caller
    resetting a lane has one import and one spelling of "parked"."""
    from .family import park_pos

    return park_pos(f, slot_index)
