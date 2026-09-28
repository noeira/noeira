"""`sampler.sample_placements`, on the device, for any family with a table.

    Self.CONFIG.init_qpos_gpu  ->  place_free_slots[LiberoGoalPlacement, ...](
        qpos, qvel, meta, env, seed)

## ⚠⚠ ONE KERNEL, BECAUSE THE FIRST ONE WAS WRITTEN FOR ONE FAMILY

`So101TabletopConfig.init_qpos_gpu` placed props correctly for `so101_tabletop`
and for nothing else. It restated one site for every region, one radius for
every slot as both the clash radius and the resting height, and walked the free
slots in family order. Every LIBERO family breaks all three, and its own header
listed what a LIBERO twin owed: read `slot_geom=`, walk `spec.order_inits`, add
`TABLE_Z_OFFSET` on a region naming no contact slot — and, once the host grew
it, add the fixture's `top_site` for `On` a fixture region. A second hand-copied
hook per family would have been the drift that header was warning about, so the
rule lives here once and a family supplies DATA through `PlacementTable`.

## WHAT THE KERNEL MIRRORS, LINE FOR LINE

`sample_placements` is the spec. Agreement needs every one of these, and
`tests/tasks/test_device_placement.mojo` counts that its corpus reaches each:

1. the Philox coordinates — `seed ^ PLACEMENT_SALT`, subsequence
   `(lane << 16) | axis`, offset `attempt`, axis `si * 2 (+ 1)` with `si` the
   FAMILY slot index;
2. the WALK ORDER — `spec.order_inits`: family slot order, a stack deferred
   until its reference has been walked;
3. the resting height — `-bottom_z` for a slot with `slot_geom=`, the fallback
   radius otherwise;
4. the z offset — `TABLE_Z_OFFSET` on an unanchored region, the contact
   fixture's `top_z` for `On` an anchored one, nothing for `In`;
5. the draw rectangle — whole for a table region; HALVED then inset by the
   object's `h_radius` for an anchored one, collapsed to its centre when the
   inset inverts;
6. the clash test — `h_radius` sums, skipped between two different regions when
   either is anchored, and skipped against a stack;
7. a STACK — the reference's x/y, `ref_z + ref.top_z + STACK_Z_OFFSET -
   bottom_z`, no draw and no rejection.

## ⚠ WHERE THE TWO CANNOT AGREE, AND WHAT THE KERNEL DOES INSTEAD

The host RAISES on exhaustion and on a stack whose reference was not placed; a
kernel cannot raise. The kernel leaves that slot at whatever `qpos` holds —
the park pose after `_reset_env_lane` — which is the VISIBLE failure: every goal
naming the prop is false. Everything the host raises on at the SPEC level (an
anchored `On` whose fixture has no `slot_geom=`, a stack onto a static slot, a
region the kernel cannot follow) is refused before a word is written, by
`check.require_device_placement`.

## `jinit=` — THE JOINTS, DRAWN FIRST

`draw_joint_inits` is `sampler.sample_joint_inits` on the device, and
`reset_task_slots` runs it before the placements, because a prop drawn into a
drawer region stands in the drawer the draw just opened.
"""

from layout import Layout, LayoutTensor
from std.math import cos, sin, pi
from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_INIT_REGION_0, META_INIT_SLOTS,
    META_IDX_JINIT_0, META_JINIT_SLOTS, META_JINIT_WORDS,
)
from noeira.tasks.obs import FREE_JOINT_NV
from noeira.tasks.spec import TABLE_Z_OFFSET, STACK_Z_OFFSET


comptime MAX_PLACE_ATTEMPTS: Int = 64
"""The rejection budget, host and device alike. See `sampler.mojo`, which
re-exports this — ONE definition, where it used to be restated on the config
and asserted equal by the gate."""

comptime PLACEMENT_SALT: UInt64 = 0x9E3779B97F4A7C15
"""Keeps placement draws off the env's reset-noise stream. See `sampler.mojo`'s
header; any value that is not the env's own works."""


comptime BASE_JITTER_AXIS_BASE: Int = 0xC000
"""Where the rest-pose jitter's Philox axes start: word `i` of `base_qpos`
draws on axis `BASE_JITTER_AXIS_BASE + i`, attempt 0 — clear of the
placement axes and of `JOINT_AXIS_BASE`'s `jinit=` block, and below the
16-bit axis field (`_uniform01`)."""

comptime JOINT_AXIS_BASE: Int = 0x8000
"""Where a `jinit=` draw's Philox axis starts, clear of every placement axis.

⚠ `_uniform01` packs `subsequence = (lane << 16) | axis`, and a placement uses
axis `si * 2` / `si * 2 + 1` — so the placement axes are bounded by twice the
family's slot count. Starting the joint draws at 0x8000 cannot collide with any
of them for any family a scene could hold, and a collision would not be an
error: it would silently correlate a drawer's opening with an object's x.

⚠ It must stay BELOW 0x10000 or it would carry into the lane bits and give two
lanes one stream — which is the same failure a shared seed would cause, and the
reason `_uniform01` uses the counter axes at all."""


# ── THE INIT WORD — one per free slot, `meta[META_IDX_INIT_REGION_0 + j]` ──
#
# ⚠⚠ ZERO STILL MEANS "NO init=", and that is still the load-bearing choice:
# `Data` uploads a zero-filled `meta`, so a driver that forgot these words
# PARKS every free slot rather than placing it somewhere plausible.
#
#     0                         no init — the slot keeps its qpos
#     r + 1                     a region draw in family region r
#     INIT_WORD_IN_BIAS + r + 1 the same, for `In` (no fixture `top_site`)
#     -(s + 1)                  a STACK on family slot s
#
# ⚠ `In` IS PART OF THE WORD, NOT OF THE REGION, because the host keys the
# height on the PREDICATE (`spec.InitSpec.inside`) — `bddl_base_domain` routes
# `In` and `On` to two sampler classes, and one of them has the `top_offset`
# line commented out. A region answers the same either way in today's corpus;
# a word that dropped the predicate would still be wrong the first time it did
# not.
comptime INIT_WORD_IN_BIAS: Int = 4096
"""Added to `r + 1` for an `In`. Regions per family stay far below it, and
`active.init_region_words` refuses one that does not. Exact in float32 (2^24)."""

comptime INIT_WORD_YAW_BIAS: Int = 8192
"""Added on top of a REGION word (`r + 1`, with or without the `In` bias) for
an init with `:yaw` — the slot's yaw is drawn on axis `YAW_AXIS_BASE + si`.
Above every region word, exact in float32; a stack never carries it."""

comptime INIT_WORD_SEP_UNIT: Int = 16384
"""`:sep=` — the separation in WHOLE MILLIMETRES times this unit, added on top
of a region word (with or without the `In` and `:yaw` biases, both below it).
1..1023 mm keeps the word under 2^24, exact in float32. Decode: `sep_mm = w //
INIT_WORD_SEP_UNIT` first, then the yaw and `In` biases from the remainder
(`_init_word_split`)."""


@always_inline
def _init_word_split(w: Int) -> Tuple[Int, Int]:
    """A REGION init word (w > 0) -> (the word without its separation, the
    separation in mm)."""
    var sep_mm = w // INIT_WORD_SEP_UNIT
    return (w - sep_mm * INIT_WORD_SEP_UNIT, sep_mm)


comptime YAW_AXIS_BASE: Int = 0xA000
"""Where a `:yaw` draw's Philox axis starts: family slot `si` draws on
`YAW_AXIS_BASE + si`, attempt 0 — clear of the placement axes (`si * 2`,
`si * 2 + 1`), of `JOINT_AXIS_BASE` and of `BASE_JITTER_AXIS_BASE`."""


trait PlacementTable:
    """One family's task table — reset geometry, and what the per-step task
    hooks (`tasks/task_hooks.mojo`) need — as comptime data a kernel can read.

    ⚠⚠ EVERY FLOAT METHOD RETURNS `Scalar[DTYPE]`, NEVER `Float64`. The kernel
    selects among the entries with a RUNTIME index, and a runtime-selected
    `Float64` survives into the IR as a `double` — which Metal does not have:
    the first version returned `Float64` and `task_batched_gpu` died with
    "select i1 ..., double -4.000000e-02, double 1.000000e-01 returns
    unsupported type 'double'". A literal converted inside each branch folds to
    a `DTYPE` constant instead. (The config's comptime `Float64`s never hit
    this: a comptime value folds before codegen.)

    ⚠ RESTATED, AND CHECKED. A config is a comptime type and the `.family` a
    runtime file, so this cannot read it — the constraint `FREE_QADR_*` has
    always lived under. `check.placement_table_drift` diffs every method below
    against the loaded family and the composed scene's forward kinematics, and
    the LIBERO tables are GENERATED with a `--check`.

    Free slots are indexed by ORDINAL `j` (the `j`-th free slot in family
    order), which is how `meta`'s init block is laid out; regions by family
    region index `r`.
    """

    comptime N_SLOTS: Int
    comptime N_FREE: Int
    comptime N_REGIONS: Int
    comptime NQ: Int
    comptime NV: Int
    comptime NBODY: Int
    comptime NSITE: Int
    comptime GRIPPER_SITE: Int
    """The end-effector site the observation's goal words measure from —
    `robot_grip_site` on the Panda, `robot_gripperframe` on the SO-101."""
    comptime N_BASE_QPOS: Int
    """How many leading `qpos` words `base_qpos[DTYPE](i)` restates."""

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        """The `.family`'s `base_qpos=` — the base asset's joint positions at
        REST, which the reset writes into `qpos[0 .. N_BASE_QPOS)`."""
        ...

    @staticmethod
    def base_qpos_jitter[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        """The `.family`'s `base_qpos_jitter=` half-width for word `i`; the
        reset draws `base_qpos(i) + h * (2u - 1)`. 0 = no draw.

        ⚠ A DEFAULT OF 0, so the families without the key (every LIBERO one,
        the tabletop) need not restate it — and so a table that FORGETS it is
        silent here. `placement/check.check_table` compares it word for word
        with the family's, which is where a missing override fails."""
        return Scalar[DTYPE](0)

    # ── per free slot ──
    @staticmethod
    def free_slot(j: Int) -> Int:
        """The family slot index — the Philox axis base."""
        ...

    @staticmethod
    def free_qadr(j: Int) -> Int:
        ...

    @staticmethod
    def free_dadr(j: Int) -> Int:
        ...

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        """`slot_geom=` present. Without it the three numbers below are the
        caller's fallback radius, as `sample_placements`' `radii[si]`."""
        ...

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        """Origin above the region's site: `-bottom_z`, or the fallback."""
        ...

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        """The clash radius and the anchored inset: `h_radius`, or the
        fallback."""
        ...

    @staticmethod
    def free_park_x[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        """`family.park_pos` — where an INACTIVE slot is pinned every step."""
        ...

    @staticmethod
    def free_park_y[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def free_park_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        """For a stack standing ON this slot's reference."""
        ...

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        """For a stack standing on THIS slot."""
        ...

    # ── per region ──
    @staticmethod
    def region_site(r: Int) -> Int:
        """The region's site id — the goal words' TARGET for `In`/`On`/
        `AtRegion`. Per region: LIBERO's regions hang off many sites, where
        `so101_tabletop`'s all share `table_surface`."""
        ...

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        ...

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        """The region names a CONTACT slot — `RegionSpec.contact` non-empty."""
        ...

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        ...

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        """The contact slot's `top_z`, added for `On`. 0 when it has none —
        and such a word is refused on the host, as the host sampler raises."""
        ...

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        """What carries the region's site.

        * `-1` — nothing: the site is on a body with no joint above it, and
          `region_site_*` is its frame in every episode.
        * `k >= 0` — exactly ONE SLIDE joint, table joint `k`: the frame is
          `region_site_* + region_move_axis_* * qpos[joint_qadr(k)]`, the
          site being measured with that joint at 0. That is a drawer interior.
        * `-2` — anything else (a hinge, two joints, a free body): the frame is
          not affine in one `qpos` word, the kernel runs before FK, and
          `check.require_device_placement` refuses a word naming it.
        """
        ...

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        """World displacement of the site per unit of the carrying slide."""
        ...

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        ...

    # ── per drawable joint — every hinge/slide of a STATIC slot ──
    comptime N_JOINTS: Int

    @staticmethod
    def joint_name(k: Int) -> String:
        """HOST-ONLY — `check.joint_init_words` resolves a `jinit=` by it."""
        ...

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        ...

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        ...


comptime _KIND_NONE: Int = 0
comptime _KIND_REGION: Int = 1
comptime _KIND_STACK: Int = 2


@always_inline
def place_free_slots[
    T: PlacementTable,
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ_F: Int,
    NV_F: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
    seed: Int,
):
    """Place lane `env`'s free slots from its init words. Writes `qpos`/`qvel`
    of the slots it places and nothing else — never `meta`, whose tape must
    survive the reset."""
    # the evidence `sin`/`cos` of a generic `Scalar[DTYPE]` need (`:yaw`),
    # given in the body so the reset hook's trait signature stays unconstrained
    comptime assert DTYPE.is_floating_point(), "DTYPE must be floating point"
    comptime NF = T.N_FREE
    var kind = Array[Int, META_INIT_SLOTS](fill=_KIND_NONE)
    var target = Array[Int, META_INIT_SLOTS](fill=-1)
    var inside = Array[Bool, META_INIT_SLOTS](fill=False)
    var walked = Array[Bool, META_INIT_SLOTS](fill=False)

    # ── decode ──
    for j in range(NF):
        var w = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_INIT_REGION_0 + j]))
        if w > 0:
            kind[j] = _KIND_REGION
            # `:sep=` first (the highest field); it is re-read from the word
            # in the clash test, like `:yaw` where the pose is written — a
            # per-thread array here is the Metal miscompute shape
            w = _init_word_split(w)[0]
            # `:yaw` is re-read from the word where the pose is written — a
            # per-thread flag array here is the Metal miscompute shape
            if w > INIT_WORD_YAW_BIAS:
                w -= INIT_WORD_YAW_BIAS
            if w > INIT_WORD_IN_BIAS:
                inside[j] = True
                w -= INIT_WORD_IN_BIAS
            target[j] = w - 1
        elif w < 0:
            # a stack names a FAMILY slot; the kernel works in ordinals
            var s = -w - 1
            for q in range(NF):
                if T.free_slot(q) == s:
                    target[j] = q
            kind[j] = _KIND_STACK if target[j] >= 0 else _KIND_NONE

    # ── what has been placed so far, in walk order ──
    var px = Array[Scalar[DTYPE], META_INIT_SLOTS](fill=Scalar[DTYPE](0))
    var py = Array[Scalar[DTYPE], META_INIT_SLOTS](fill=Scalar[DTYPE](0))
    var pz = Array[Scalar[DTYPE], META_INIT_SLOTS](fill=Scalar[DTYPE](0))
    var pord = Array[Int, META_INIT_SLOTS](fill=-1)
    # ⚠ -1 FOR A STACK, as the host's `of_region`, and a stack is skipped by the
    # clash test outright — see step 6 in the header.
    var preg = Array[Int, META_INIT_SLOTS](fill=-1)
    var n_placed = 0

    # ⚠⚠ THE WALK IS `spec.order_inits`, RESTARTED FROM THE FIRST SLOT AFTER
    # EVERY EMISSION. A stack is ready once its reference has been WALKED —
    # placed or not — or when the reference has no init at all, exactly the
    # host's `ready` test; the host then raises where this parks.
    for _pass in range(NF):
        var j = -1
        for q in range(NF):
            if kind[q] == _KIND_NONE or walked[q]:
                continue
            if kind[q] == _KIND_STACK:
                var rq = target[q]
                if kind[rq] != _KIND_NONE and not walked[rq]:
                    continue
            j = q
            break
        if j < 0:
            break
        walked[j] = True
        var qa = T.free_qadr(j)
        var da = T.free_dadr(j)

        if kind[j] == _KIND_STACK:
            var rj = target[j]
            var found = -1
            for k in range(n_placed):
                if pord[k] == rj:
                    found = k
            if found < 0:
                continue
            var sx = px[found]
            var sy = py[found]
            var sz = (
                pz[found] + T.free_top_z[DTYPE](rj)
                + Scalar[DTYPE](STACK_Z_OFFSET)
                - T.free_bottom_z[DTYPE](j)
            )
            _write_pose[DTYPE, BATCH_SIZE, NQ_F, NV_F](
                qpos, qvel, env, qa, da, sx, sy, sz
            )
            px[n_placed] = sx
            py[n_placed] = sy
            pz[n_placed] = sz
            pord[n_placed] = j
            preg[n_placed] = -1
            n_placed += 1
            continue

        var r = target[j]
        var si = T.free_slot(j)
        var has_geom = T.free_has_geom(j)
        var rest = T.free_rest[DTYPE](j)
        var rad_i = T.free_radius[DTYPE](j)
        var anchored = T.region_anchored(r)
        var fx = T.region_site_x[DTYPE](r)
        var fy = T.region_site_y[DTYPE](r)
        var fz = T.region_site_z[DTYPE](r)
        # ⚠⚠ A DRAWER'S REGION FOLLOWS THE DRAWER. The host resolves frames by
        # FK AFTER this reset's `jinit=` draws; the kernel has no FK, so a
        # region carried by one slide reads that slide's `qpos` — written by
        # `draw_joint_inits` just before this, or `qpos0` if the task draws
        # none — and shifts the site along the table's world axis. Affine and
        # exact up to the ULP a different association costs.
        var mj = T.region_move_joint(r)
        if mj >= 0:
            var q = rebind[Scalar[DTYPE]](qpos[env, T.joint_qadr(mj)])
            fx = fx + T.region_move_axis_x[DTYPE](r) * q
            fy = fy + T.region_move_axis_y[DTYPE](r) * q
            fz = fz + T.region_move_axis_z[DTYPE](r) * q
        var z_off = Scalar[DTYPE](0)
        if has_geom:
            if not anchored:
                z_off = Scalar[DTYPE](TABLE_Z_OFFSET)
            elif not inside[j]:
                z_off = T.region_contact_top_z[DTYPE](r)
        var z = fz + rest + z_off

        for attempt in range(MAX_PLACE_ATTEMPTS):
            var x = fx
            var y = fy
            if T.region_has_rect(r):
                var ru = PhiloxRandom(
                    seed=UInt64(seed) ^ PLACEMENT_SALT,
                    subsequence=(UInt64(env) << 16) | UInt64(si * 2),
                    offset=UInt64(attempt),
                )
                var rv = PhiloxRandom(
                    seed=UInt64(seed) ^ PLACEMENT_SALT,
                    subsequence=(UInt64(env) << 16) | UInt64(si * 2 + 1),
                    offset=UInt64(attempt),
                )
                var u = Scalar[DTYPE](Float64(ru.step_uniform()[0]))
                var v = Scalar[DTYPE](Float64(rv.step_uniform()[0]))
                var x0 = T.region_x0[DTYPE](r)
                var x1 = T.region_x1[DTYPE](r)
                var y0 = T.region_y0[DTYPE](r)
                var y1 = T.region_y1[DTYPE](r)
                if anchored:
                    comptime HALF = Scalar[DTYPE](0.5)
                    x0 = x0 * HALF
                    x1 = x1 * HALF
                    y0 = y0 * HALF
                    y1 = y1 * HALF
                    x0 = x0 + rad_i
                    x1 = x1 - rad_i
                    y0 = y0 + rad_i
                    y1 = y1 - rad_i
                    if x1 < x0:
                        var xc = HALF * (x0 + x1)
                        x0 = xc
                        x1 = xc
                    if y1 < y0:
                        var yc = HALF * (y0 + y1)
                        y0 = yc
                        y1 = yc
                x = fx + x0 + u * (x1 - x0)
                y = fy + y0 + v * (y1 - y0)

            var clash = False
            var sep_i = _init_word_split(Int(
                rebind[Scalar[DTYPE]](meta[env, META_IDX_INIT_REGION_0 + j])
            ))[1]
            for k in range(n_placed):
                var dx = px[k] - x
                var dy = py[k] - y
                var rr = rad_i + T.free_radius[DTYPE](pord[k])
                # `:sep=` — `sampler.sample_placements`' rule: the larger of
                # the radii's sum and either slot's separation. A stack's
                # word is negative and carries none.
                var wk = Int(
                    rebind[Scalar[DTYPE]](meta[env, META_IDX_INIT_REGION_0 + pord[k]])
                )
                var sep_mm = sep_i
                if wk > 0:
                    sep_mm = max(sep_mm, _init_word_split(wk)[1])
                # ⚠ IN DTYPE, NOT Float64: Metal has no double. A division
                # rounds once, so this is the host's metres rounded to DTYPE.
                var sep = Scalar[DTYPE](sep_mm) / Scalar[DTYPE](1000)
                if sep > rr:
                    rr = sep
                if dx * dx + dy * dy < rr * rr:
                    var rk = preg[k]
                    if rk < 0:
                        continue
                    if rk != r and (anchored or T.region_anchored(rk)):
                        continue
                    clash = True
            if not clash:
                # `:yaw` — `sampler.sample_placements`' draw, its own axis
                var cz = Scalar[DTYPE](1)
                var sz_ = Scalar[DTYPE](0)
                var wj = _init_word_split(Int(
                    rebind[Scalar[DTYPE]](meta[env, META_IDX_INIT_REGION_0 + j])
                ))[0]
                if wj > INIT_WORD_YAW_BIAS:
                    var ry = PhiloxRandom(
                        seed=UInt64(seed) ^ PLACEMENT_SALT,
                        subsequence=(UInt64(env) << 16) | UInt64(YAW_AXIS_BASE + si),
                        offset=UInt64(0),
                    )
                    var uy = Scalar[DTYPE](Float64(ry.step_uniform()[0]))
                    var half = (Scalar[DTYPE](2) * uy - Scalar[DTYPE](1)) * Scalar[
                        DTYPE
                    ](pi) * Scalar[DTYPE](0.5)
                    cz = cos(half)
                    sz_ = sin(half)
                _write_pose[DTYPE, BATCH_SIZE, NQ_F, NV_F](
                    qpos, qvel, env, qa, da, x, y, z, cz, sz_
                )
                px[n_placed] = x
                py[n_placed] = y
                pz[n_placed] = z
                pord[n_placed] = j
                preg[n_placed] = r
                n_placed += 1
                break


@always_inline
def draw_joint_inits[
    T: PlacementTable,
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ_F: Int,
    NV_F: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
    seed: Int,
):
    """`sampler.sample_joint_inits` + `reset.apply_joint_inits`, on one lane.

    Draw `k` is `lo + u * (hi - lo)` with `u` on Philox axis
    `JOINT_AXIS_BASE + k`, attempt 0 — the host's coordinates, where `k` is the
    `jinit=` line's index in the TASK. Writes the joint's `qpos` and zeroes its
    `qvel`; a word of 0 draws nothing and leaves both alone."""
    for k in range(META_JINIT_SLOTS):
        comptime W = META_JINIT_WORDS
        var jw = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_JINIT_0 + k * W]))
        if jw <= 0 or jw > T.N_JOINTS:
            continue
        var jk = jw - 1
        var lo = rebind[Scalar[DTYPE]](meta[env, META_IDX_JINIT_0 + k * W + 1])
        var hi = rebind[Scalar[DTYPE]](meta[env, META_IDX_JINIT_0 + k * W + 2])
        var ru = PhiloxRandom(
            seed=UInt64(seed) ^ PLACEMENT_SALT,
            subsequence=(UInt64(env) << 16) | UInt64(JOINT_AXIS_BASE + k),
            offset=UInt64(0),
        )
        var u = Scalar[DTYPE](Float64(ru.step_uniform()[0]))
        qpos[env, T.joint_qadr(jk)] = lo + u * (hi - lo)
        qvel[env, T.joint_dadr(jk)] = Scalar[DTYPE](0)


@always_inline
def reset_task_slots[
    T: PlacementTable,
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ_F: Int,
    NV_F: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH_SIZE, METADATA_SIZE), MutAnyOrigin
    ],
    env: Int,
    seed: Int,
):
    """The task layer's whole reset for one lane: the base asset's rest pose
    (with its `base_qpos_jitter=` draw), the joint draws, THEN the placements.

    ⚠⚠ THE REST POSE, BECAUSE `qpos0` IS NOT IT. `_reset_env_lane` restores
    the composed scene's `qpos0`, which for the Panda is every joint at ZERO —
    the arm straight up, a SINGULAR configuration: the site Jacobian's
    angular-x row is identically zero there, so OSC_POSE's operational-space
    inertia cannot be inverted and the controller writes no torque. Every
    host path (`libero_eval`, the gates, the generator's FK) writes
    `base_qpos` first; the device reset did not, and the batched LIBERO env
    reported 16 of 16 lanes singular with the arm falling (M1 Pro and 5090
    alike). `qvel` is zeroed over the same words.

    ⚠ THE ORDER IS LOAD-BEARING. A region carried by a drawer reads the
    drawer's drawn `qpos`, so the draw must be written first — the host's order
    too (draw, FK, frames, sample)."""
    for i in range(T.N_BASE_QPOS):
        var q = T.base_qpos[DTYPE](i)
        var h = T.base_qpos_jitter[DTYPE](i)
        if h != Scalar[DTYPE](0):
            # `sampler.sample_base_qpos`'s draw: axis BASE_JITTER_AXIS_BASE + i
            var ru = PhiloxRandom(
                seed=UInt64(seed) ^ PLACEMENT_SALT,
                subsequence=(UInt64(env) << 16) | UInt64(BASE_JITTER_AXIS_BASE + i),
                offset=UInt64(0),
            )
            var u = Scalar[DTYPE](Float64(ru.step_uniform()[0]))
            q = q + h * (Scalar[DTYPE](2) * u - Scalar[DTYPE](1))
        qpos[env, i] = q
        if i < NV_F:
            qvel[env, i] = Scalar[DTYPE](0)
    draw_joint_inits[T, DTYPE, BATCH_SIZE, NQ_F, NV_F](
        qpos, qvel, meta, env, seed
    )
    place_free_slots[T, DTYPE, BATCH_SIZE, NQ_F, NV_F](
        qpos, qvel, meta, env, seed
    )


@always_inline
def _write_pose[DTYPE: DType, BATCH_SIZE: Int, NQ_F: Int, NV_F: Int](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NQ_F), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH_SIZE, NV_F), MutAnyOrigin],
    env: Int,
    qa: Int,
    da: Int,
    x: Scalar[DTYPE],
    y: Scalar[DTYPE],
    z: Scalar[DTYPE],
    qw: Scalar[DTYPE] = Scalar[DTYPE](1),
    qz: Scalar[DTYPE] = Scalar[DTYPE](0),
):
    """`reset.write_free_pose` + `write_free_vel_zero` on one lane; `(qw, qz)`
    is a yaw about +z, the identity by default.

    ⚠ W-FIRST IN `qpos`: a free joint's seven words are (x, y, z, w, x, y, z),
    and the identity is (1, 0, 0, 0) — zeros are a degenerate rotation."""
    qpos[env, qa + 0] = x
    qpos[env, qa + 1] = y
    qpos[env, qa + 2] = z
    qpos[env, qa + 3] = qw
    qpos[env, qa + 4] = Scalar[DTYPE](0)
    qpos[env, qa + 5] = Scalar[DTYPE](0)
    qpos[env, qa + 6] = qz
    for k in range(FREE_JOINT_NV):
        qvel[env, da + k] = Scalar[DTYPE](0)
