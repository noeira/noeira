"""Evaluating the tape INSIDE a kernel — P3b's half of the parity claim.

`tape.eval_tape` reads `List[Float64]`; a reward kernel has `LayoutTensor`s of
`Scalar[DT]`. This is that loop, over those containers.

## ⚠ THE LOOP IS WRITTEN TWICE. THE ARITHMETIC IS NOT.

Both call `pred_in_rect` / `pred_near` / `pred_above` / `pred_upright` from
`eval.mojo`, which are generic over dtype precisely so this is possible — the
host instantiates them at float64 and this at `DT`. A version that reimplemented
a comparison here would be the third spelling of the same rule, and
`_a_rule_written_inline_twice_drifts` is this tree's most recurring defect.

What is genuinely different is the CONTAINER and the INDEXING, and that is
exactly what `tests/tasks/test_task_tape.mojo` (host tape vs host goal) and
P3c's GPU-vs-CPU gate exist to pin.

## ⚠⚠ NO `Float64` ANYWHERE BELOW

Metal has no double; an f64 multiply-add in a kernel is an LLVM-IR verification
failure, not a slow path. Every constant is `Scalar[DTYPE](...)` of a comptime
`Float64`, which folds at compile time.

## WHERE THE INPUTS LIVE, AND WHY THEY NEED NO NEW OPERANDS

* **the tape** — `meta[env, META_IDX_TASK_PARAM_0 .. _11]`, twelve per-lane
  words. Already an operand; reset preserves it (`constants.mojo:164`).
* **the region table** — `curriculum[0, b*REGION_WORDS ..]`, shared across
  lanes because a region belongs to the FAMILY, not to a task. Already an
  operand. ⚠ INDEXED BY THE TERM'S `b` SINCE L3: it used to hold exactly
  one region (`MODEL_CURRICULUM_SIZE` was 8) and every term read region 0;
  the table is 128 words now, 16 regions of 8, and `require_gpu_regions`
  refuses a goal past that.
* **the L3 inputs** — `qpos` (Joint), the model's `sites` table (a box
  region's frame is `xquat[site body] * site quat`, the product
  `sensors/touch.mojo` forms), the model's `bodies` table (the parent
  column, for "is this contact body inside slot X"), and the lane's
  `contacts` + `meta[META_IDX_NUM_CONTACTS]`. Every one of them is ALREADY
  an operand of `compute_reward_and_done_gpu`; nothing new is bound.

⚠ THE NARROW OVERLOADS (poses only) ARE THE PRE-L3 SIGNATURES, kept for the
SO-101 drivers that compute a goal distance on the host from `Data` views.
They compile the L3 branches OUT (`HAS_L3=False`) and evaluate an L3 op as
False — which is why `tape.tape_needs_l3` exists on the host and why a
LIBERO driver must call the wide ones. `family_config` does.
"""

from std.math import sqrt

from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_NUM_CONTACTS,
    MODEL_CURRICULUM_SIZE, MODEL_SITE_SIZE, MODEL_BODY_SIZE,
    SITE_IDX_BODY, SITE_IDX_QUAT_X, SITE_IDX_QUAT_Y, SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W, BODY_IDX_PARENT,
    CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
)
from noeira.physics3d.kinematics.quat_math import gpu_quat_mul, gpu_quat_rotate
from .spec import FamilySpec
from .predicates import (
    BoundGoal,
    OP_IN, OP_ON, OP_NEAR, OP_ABOVE, OP_UPRIGHT, OP_AT_REGION,
    OP_AND, OP_OR, OP_NOT, OP_JOINT, OP_ON_BODY, OP_TOUCHING,
)
from .eval import (
    pred_in_rect, pred_near, pred_above, pred_upright,
    pred_joint, pred_ontop, pred_box_in, pred_box_under,
    ON_MIN_DZ, ON_MAX_DZ, BOX_IN_Z_SLACK, BOX_UNDER_Z_SLACK, BOX_UNDER_HEIGHT,
    ONTOP_XY,
)
from .tape import MAX_TAPE_TERMS, TERM_WORDS


# ── the region table's layout inside `curriculum` ──────────────────────────
# One record of REGION_WORDS per region, at `b * REGION_WORDS`; the CUR_IDX_*
# names are OFFSETS WITHIN A RECORD (region 0's absolute indices are the
# same numbers, which is what every pre-L3 reader assumed).
comptime CUR_IDX_REGION_SITE: Int = 0
comptime CUR_IDX_REGION_X0: Int = 1
comptime CUR_IDX_REGION_Y0: Int = 2
comptime CUR_IDX_REGION_X1: Int = 3
comptime CUR_IDX_REGION_Y1: Int = 4
# ⚠ THE Z HALF-BAND, WORD 5. `RegionSpec.half_height`, carried per region so
# `In`/`AtRegion` on device use the region's own volume rather than
# `eval.IN_HALF_HEIGHT`.
comptime CUR_IDX_REGION_H: Int = 5
# L3: 1 for a `:box:` region (LIBERO SiteObject semantics), else 0.
comptime CUR_IDX_REGION_BOX: Int = 6
# L3: the root body of the region's contact slot, or -1.
comptime CUR_IDX_REGION_CONTACT: Int = 7
comptime REGION_WORDS: Int = 8

# ⚠ AN EMPTY RECORD HAS SITE -1. `region_table_words` writes every unused
# record that way so a stale `curriculum` from a previous family cannot be
# read as a plausible region.
comptime REGION_EMPTY: Float64 = -1.0

comptime MAX_CURRICULUM_REGIONS: Int = MODEL_CURRICULUM_SIZE // REGION_WORDS


def _empty_table() -> List[Float64]:
    var out = List[Float64]()
    for _ in range(MODEL_CURRICULUM_SIZE):
        out.append(0.0)
    for r in range(MAX_CURRICULUM_REGIONS):
        out[r * REGION_WORDS + CUR_IDX_REGION_SITE] = REGION_EMPTY
        out[r * REGION_WORDS + CUR_IDX_REGION_CONTACT] = -1.0
    return out^


def _write_region(
    mut out: List[Float64], r: Int,
    site: Int, x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    half_height: Float64, is_box: Bool, contact_body: Int,
) raises:
    if r >= MAX_CURRICULUM_REGIONS:
        raise Error(
            "tasks: region " + String(r) + " does not fit the device region"
            " table (" + String(MAX_CURRICULUM_REGIONS) + " regions of "
            + String(REGION_WORDS) + " words in MODEL_CURRICULUM_SIZE="
            + String(MODEL_CURRICULUM_SIZE) + ")"
        )
    # ⚠⚠ REQUIRED, NOT DEFAULTED. A defaulted half-height would let a caller
    # that never heard of it keep compiling and silently ship the 0.12
    # fallback to device while the HOST evaluator used the region's real
    # band — the CPU and GPU rewards would then disagree on exactly the
    # regions the field was added for.
    if half_height <= 0.0:
        raise Error(
            "tasks: region half-height " + String(half_height) + " accepts no"
            " point. `pred_in_rect` would return False on device for every"
            " state, which reads as an unlearnable task rather than a bad"
            " number. `spec.parse_region` refuses this too."
        )
    var o = r * REGION_WORDS
    out[o + CUR_IDX_REGION_SITE] = Float64(site)
    out[o + CUR_IDX_REGION_X0] = x0
    out[o + CUR_IDX_REGION_Y0] = y0
    out[o + CUR_IDX_REGION_X1] = x1
    out[o + CUR_IDX_REGION_Y1] = y1
    out[o + CUR_IDX_REGION_H] = half_height
    out[o + CUR_IDX_REGION_BOX] = 1.0 if is_box else 0.0
    out[o + CUR_IDX_REGION_CONTACT] = Float64(contact_body)


def region_table_words(
    site: Int, x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    half_height: Float64,
) raises -> List[Float64]:
    """The `curriculum` words for a ONE-region family (region 0, plain).
    Host-side. The pre-L3 writer, kept for the SO-101 drivers; a family
    whose goals name more than region 0, or a box region, uses the overload
    that takes the family."""
    var out = _empty_table()
    _write_region(out, 0, site, x0, y0, x1, y1, half_height, False, -1)
    return out^


def region_table_words(
    f: FamilySpec, region_site: List[Int], region_contact: List[Int],
) raises -> List[Float64]:
    """Every region of the family, in family order — `eval.region_sites`,
    `region_rects`, `region_half_heights`, `region_box_flags` and
    `region_contact_bodies` folded into the 8-word records the device reads.
    RAISES past `MAX_CURRICULUM_REGIONS`."""
    from .eval import region_rects, region_half_heights
    var out = _empty_table()
    var rects = region_rects(f)
    var hh = region_half_heights(f)
    for r in range(len(f.regions)):
        var cb = -1
        if r < len(region_contact):
            cb = region_contact[r]
        _write_region(
            out, r, region_site[r], rects[r][0], rects[r][1], rects[r][2],
            rects[r][3], hh[r], f.regions[r].is_box, cb,
        )
    return out^


# ── the kernel loops ───────────────────────────────────────────────────────


@always_inline
def _slots_touching_gpu[
    DTYPE: DType, BATCH: Int, L_BODIES: TensorLayout, L_CON: TensorLayout,
](
    root_a: Int, root_b: Int, ncon: Int,
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CON, MutAnyOrigin],
    env: Int,
) -> Bool:
    """`eval.slots_touching` over the lane's contact records: any contact
    with one body under `root_a` and the other under `root_b`, walking
    `BODY_IDX_PARENT` while the id exceeds the root (tree order)."""
    for k in range(ncon):
        var base = k * CONTACT_SIZE
        var ba = Int(rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_BODY_A]))
        var bb = Int(rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_BODY_B]))
        var xa = ba
        while xa > root_a:
            xa = Int(rebind[Scalar[DTYPE]](bodies[xa, BODY_IDX_PARENT]))
        var xb = bb
        while xb > root_b:
            xb = Int(rebind[Scalar[DTYPE]](bodies[xb, BODY_IDX_PARENT]))
        if xa == root_a and xb == root_b:
            return True
        var ya = bb
        while ya > root_a:
            ya = Int(rebind[Scalar[DTYPE]](bodies[ya, BODY_IDX_PARENT]))
        var yb = ba
        while yb > root_b:
            yb = Int(rebind[Scalar[DTYPE]](bodies[yb, BODY_IDX_PARENT]))
        if ya == root_a and yb == root_b:
            return True
    return False


@always_inline
def _eval_tape_impl[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
    HAS_L3: Bool,
    L_Q: TensorLayout, L_SITES: TensorLayout, L_BODIES: TensorLayout, L_CON: TensorLayout,
](
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY_F * 3],
    xquat: Tile[DTYPE, BATCH, NBODY_F * 4],
    site_xpos: Tile[DTYPE, BATCH, SITE_DIM],
    qpos: TileTensor[DTYPE, L_Q, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CON, MutAnyOrigin],
    env: Int,
) -> Bool:
    """This lane's goal, from this lane's tape. One `Bool`, no allocation.

    ⚠ `HAS_L3=False` COMPILES THE L3 BRANCHES OUT and the four extra
    tensors are then never read (the narrow overload passes `xpos` for all
    of them). An L3 op under `HAS_L3=False` evaluates FALSE — the host
    refuses such a tape before it reaches a narrow caller
    (`tape.tape_needs_l3`); the kernel cannot raise.
    """
    var v0 = False
    var v1 = False
    var v2 = False
    var last = False

    var ncon = 0
    comptime if HAS_L3:
        ncon = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS]))

    # ⚠ `comptime for`: MAX_TAPE_TERMS is 3 and the body branches on an op
    # code, so unrolling keeps every index a constant. A runtime loop here
    # would also index `meta` with a computed offset, which is fine, but the
    # `@parameter`-free form was a Metal f64 trap once already.
    comptime for i in range(MAX_TAPE_TERMS):
        comptime w = META_IDX_TASK_PARAM_0 + i * TERM_WORDS
        var op = Int(rebind[Scalar[DTYPE]](meta[env, w]))
        # ⚠ THE EMPTY MARKER STOPS THE SWEEP. `meta` is not zeroed between
        # episodes, so an unused slot holds the PREVIOUS episode's term.
        if op >= 0:
            var a = Int(rebind[Scalar[DTYPE]](meta[env, w + 1]))
            var b = Int(rebind[Scalar[DTYPE]](meta[env, w + 2]))
            var param = rebind[Scalar[DTYPE]](meta[env, w + 3])
            var r = False

            if op == OP_AND:
                var pa = v0 if a == 0 else (v1 if a == 1 else v2)
                var pb = v0 if b == 0 else (v1 if b == 1 else v2)
                r = pa and pb
            elif op == OP_OR:
                var pa = v0 if a == 0 else (v1 if a == 1 else v2)
                var pb = v0 if b == 0 else (v1 if b == 1 else v2)
                r = pa or pb
            elif op == OP_NOT:
                r = not (v0 if a == 0 else (v1 if a == 1 else v2))
            elif op == OP_NEAR:
                r = pred_near[DTYPE](
                    rebind[Scalar[DTYPE]](xpos[env, a * 3]),
                    rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1]),
                    rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]),
                    param,
                )
            elif op == OP_ABOVE:
                r = pred_above[DTYPE](
                    rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]),
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]),
                    param,
                )
            elif op == OP_UPRIGHT:
                # ⚠ `xquat` IS (x, y, z, W) — W LAST. Third reader of that
                # layout in this package; `eval.mojo` records what it cost.
                r = pred_upright[DTYPE](
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 3]),
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 0]),
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 1]),
                    rebind[Scalar[DTYPE]](xquat[env, a * 4 + 2]),
                    param,
                )
            elif op == OP_JOINT:
                comptime if HAS_L3:
                    r = pred_joint[DTYPE](
                        rebind[Scalar[DTYPE]](qpos[env, a]), b, param
                    )
            elif op == OP_TOUCHING or op == OP_ON_BODY:
                comptime if HAS_L3:
                    var touching = _slots_touching_gpu[
                        DTYPE, BATCH, L_BODIES, L_CON
                    ](a, b, ncon, bodies, contacts, env)
                    if op == OP_TOUCHING:
                        r = touching
                    else:
                        r = pred_ontop[DTYPE](
                            rebind[Scalar[DTYPE]](xpos[env, a * 3]),
                            rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1]),
                            rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]),
                            rebind[Scalar[DTYPE]](xpos[env, b * 3]),
                            rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1]),
                            rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]),
                            touching,
                        )
            else:
                # IN / ON / AT_REGION — region record `b`
                var ro = b * REGION_WORDS
                var rs = Int(rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_SITE]))
                var rx0 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_X0])
                var ry0 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_Y0])
                var rx1 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_X1])
                var ry1 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_Y1])
                var rh = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_H])
                var px: Scalar[DTYPE]
                var py: Scalar[DTYPE]
                var pz: Scalar[DTYPE]
                if op == OP_AT_REGION:
                    px = rebind[Scalar[DTYPE]](site_xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 2])
                else:
                    px = rebind[Scalar[DTYPE]](xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                var sx = rebind[Scalar[DTYPE]](site_xpos[env, rs * 3])
                var sy = rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 1])
                var sz = rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 2])
                var is_box = False
                comptime if HAS_L3:
                    is_box = (
                        Int(rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_BOX])) != 0
                        and op != OP_AT_REGION
                    )
                if is_box:
                    comptime if HAS_L3:
                        # ── LIBERO's SiteObject, as `eval.eval_goal` ──────
                        var sb = Int(rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_BODY]))
                        var wq = gpu_quat_mul[DTYPE](
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 0]),
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 1]),
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 2]),
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 3]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_X]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_Y]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_Z]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_W]),
                        )
                        var half = Scalar[DTYPE](0.5)
                        var cx = half * (rx0 + rx1)
                        var cy = half * (ry0 + ry1)
                        var hx = half * (rx1 - rx0)
                        var hy = half * (ry1 - ry0)
                        if op == OP_IN:
                            r = pred_box_in[DTYPE](
                                px, py, pz, sx, sy, sz,
                                wq[0], wq[1], wq[2], wq[3], cx, cy, hx, hy, rh,
                            )
                        else:
                            var cb = Int(rebind[Scalar[DTYPE]](
                                curriculum[0, ro + CUR_IDX_REGION_CONTACT]
                            ))
                            var touching = True
                            if cb >= 0:
                                touching = _slots_touching_gpu[
                                    DTYPE, BATCH, L_BODIES, L_CON
                                ](a, cb, ncon, bodies, contacts, env)
                            r = pred_box_under[DTYPE](
                                px, py, pz, sx, sy, sz,
                                wq[0], wq[1], wq[2], wq[3], cx, cy, hx, hy, rh,
                                touching,
                            )
                else:
                    # ⚠ THE REGION'S OWN BAND, from `curriculum`, matching
                    # `eval.eval_goal`'s `reg.half_height`.
                    var dz_min = -rh
                    var dz_max = rh
                    if op == OP_ON:
                        dz_min = Scalar[DTYPE](ON_MIN_DZ)
                        dz_max = Scalar[DTYPE](ON_MAX_DZ)
                    r = pred_in_rect[DTYPE](
                        px, py, pz, sx, sy, sz,
                        rx0, ry0, rx1, ry1, dz_min, dz_max,
                    )

            if i == 0:
                v0 = r
            elif i == 1:
                v1 = r
            else:
                v2 = r
            last = r
    return last


@always_inline
def eval_tape_gpu[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
](
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY_F * 3],
    xquat: Tile[DTYPE, BATCH, NBODY_F * 4],
    site_xpos: Tile[DTYPE, BATCH, SITE_DIM],
    env: Int,
) -> Bool:
    """The pre-L3 signature: poses only, L3 branches compiled out. See
    `_eval_tape_impl` for what an L3 op evaluates to here."""
    comptime L_XP = RowMajor[BATCH, NBODY_F * 3]
    return _eval_tape_impl[
        DTYPE, BATCH, NBODY_F, SITE_DIM, False, L_XP, L_XP, L_XP, L_XP
    ](meta, curriculum, xpos, xquat, site_xpos, xpos, xpos, xpos, xpos, env)


@always_inline
def eval_tape_gpu[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
    NQ_F: Int, NSITE_F: Int, MC_F: Int,
](
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY_F * 3],
    xquat: Tile[DTYPE, BATCH, NBODY_F * 4],
    site_xpos: Tile[DTYPE, BATCH, SITE_DIM],
    qpos: Tile[DTYPE, BATCH, NQ_F],
    sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
    bodies: Tile[DTYPE, NBODY_F, MODEL_BODY_SIZE],
    contacts: Tile[DTYPE, BATCH, MC_F * CONTACT_SIZE],
    env: Int,
) -> Bool:
    """The whole language (L3): the reward hook's own operands, one for one."""
    return _eval_tape_impl[
        DTYPE, BATCH, NBODY_F, SITE_DIM, True,
        RowMajor[BATCH, NQ_F],
        RowMajor[NSITE_F, MODEL_SITE_SIZE],
        RowMajor[NBODY_F, MODEL_BODY_SIZE],
        RowMajor[BATCH, MC_F * CONTACT_SIZE],
    ](meta, curriculum, xpos, xquat, site_xpos, qpos, sites, bodies, contacts, env)


@always_inline
def goal_frame_ids(
    op: Int, a: Int, b: Int, region_site: Int
) -> Tuple[Int, Int, Int, Int]:
    """`(subject_is_site, subject_id, target_is_site, target_id)` for a term.

    ⚠⚠ THE RULE IS WRITTEN ONCE HERE AND THE MEMORY ACCESS TWICE. The two
    observation hooks read `xpos`/`site_xpos` through different types — a
    `LayoutTensor` on device and a `Data` on the host — so the READS cannot be
    shared, but which body and which site to read is a rule, and a rule
    written inline twice drifts. `test_active_mask` compares the two hooks'
    output vectors and would catch a divergence; this makes one impossible.

    ⚠ `In`/`On`/`AtRegion` TARGET THE REGION'S SITE, and only `AtRegion`'s
    SUBJECT is a site. Mixing those up reads a site id out of the body array
    and lands on a real, wrong position — the same trap `eval_goal` records.

    ⚠ `Upright` HAS NO TARGET, so it points at its own subject and the
    relative vector comes out zero. That is the honest answer: there is no
    second frame in the predicate, and inventing one would put a number in the
    observation that means nothing. `Joint` (L3) has no FRAME at all — its
    subject is a qpos address — so both ids are the world body and the
    vector is zero for the same reason.

    ⚠ TERM 0 IS ALWAYS A LEAF, so `And`/`Or`/`Not` never reach this. The tape
    is POST-ORDER — every child index is lower than its parent's, asserted in
    `test_goal_language` — so the first term cannot be a composition.
    """
    if op == OP_AT_REGION:
        return (1, a, 1, region_site)
    if op == OP_IN or op == OP_ON:
        return (0, a, 1, region_site)
    if op == OP_UPRIGHT:
        return (0, a, 0, a)
    if op == OP_JOINT:
        return (0, 0, 0, 0)
    # NEAR / ABOVE / TOUCHING / ON_BODY — both arguments are bodies.
    return (0, a, 0, b)


# ⚠ THE FLOOR UNDER A NON-ZERO DISTANCE. `tape_distance_gpu`'s contract is
# ZERO IFF THE GOAL HOLDS. For the L3 ops the predicate is the authority
# (strict inequalities, a contact bit) and the geometry only supplies the
# gradient, so a state that fails the predicate with a zero geometric
# shortfall — a Joint exactly at a strict threshold, an object resting on
# another with no contact recorded — still reports THIS, never 0.
comptime DIST_EPS: Float64 = 1e-6


@always_inline
def _tape_distance_impl[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
    HAS_L3: Bool,
    L_Q: TensorLayout, L_SITES: TensorLayout, L_BODIES: TensorLayout, L_CON: TensorLayout,
](
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY_F * 3],
    xquat: Tile[DTYPE, BATCH, NBODY_F * 4],
    site_xpos: Tile[DTYPE, BATCH, SITE_DIM],
    qpos: TileTensor[DTYPE, L_Q, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CON, MutAnyOrigin],
    env: Int,
) -> Scalar[DTYPE]:
    """HOW FAR this lane is from its goal, in metres. ZERO iff the goal holds.

    ## ⚠⚠ WHY A DISTANCE EXISTS AT ALL: A SPARSE REWARD HAS NOTHING TO LEARN

    `so101_gather_bricks` pays +1 on success and nothing otherwise. Measured on
    a 5090: random actions meet it on about 1.5% of episodes, so 125k env-steps
    produced roughly SIX rewarding transitions out of 125,000 — 5e-05 of the
    replay buffer, which a batch of 256 contains 1.4% of the time. The critic
    almost never sees a success and `mean_q` has nothing to move toward. That
    is not a hyperparameter problem and no amount of tuning reaches it.

    This is the quantity a shaped term needs, and it is derived from the TAPE
    so it works for every goal the language can express rather than for one
    task — a `Near` special case inside a family config is exactly the
    "rule written inline twice" defect this tree keeps paying for.

    ## THE CONTRACT, AND THE GATE THAT HOLDS IT

    **Zero iff the goal holds.** `tests/tasks/test_goal_distance.mojo` sweeps
    states and asserts `tape_distance_gpu(s) == 0` exactly when
    `eval_tape_gpu(s)` is True — two separate switches over one tape, so an
    op this one forgot shows up as a disagreement rather than as a term with
    no gradient.

    ⚠ FOR THE L3 OPS THE PREDICATE DECIDES AND THE GEOMETRY GRADES. `Joint`,
    `Touching`, `On(obj, obj)` and the box regions call the SAME predicate
    the boolean loop calls; when it holds the distance is 0, when it does
    not the distance is the geometric shortfall floored at `DIST_EPS` (plus
    the origin distance when a required contact is missing, so "touching"
    has a gradient too). Two switches still, one predicate.

    ⚠ AND ZERO IS ALSO WHAT AN UNSHAPEABLE TERM RETURNS. `Not` has no
    monotone distance and Tier B has no geometry here, so both contribute 0
    — no gradient, never a WRONG gradient. The zero-iff contract still holds
    because those ops are refused as goals long before this
    (`require_tier_a`).

    ## ⚠ THE COMPOSITION IS max FOR `And` AND min FOR `Or`

    Both must hold, so the distance to satisfying a conjunction is the WORST
    of its parts; either will do for a disjunction, so it is the best. Summing
    an `And` would let a policy trade one term off against another and sit
    between two half-satisfied goals.
    """
    var d0 = Scalar[DTYPE](0)
    var d1 = Scalar[DTYPE](0)
    var d2 = Scalar[DTYPE](0)
    var last = Scalar[DTYPE](0)
    var zero = Scalar[DTYPE](0)
    var eps = Scalar[DTYPE](DIST_EPS)

    var ncon = 0
    comptime if HAS_L3:
        ncon = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS]))

    comptime for i in range(MAX_TAPE_TERMS):
        comptime w = META_IDX_TASK_PARAM_0 + i * TERM_WORDS
        var op = Int(rebind[Scalar[DTYPE]](meta[env, w]))
        if op >= 0:
            var a = Int(rebind[Scalar[DTYPE]](meta[env, w + 1]))
            var b = Int(rebind[Scalar[DTYPE]](meta[env, w + 2]))
            var param = rebind[Scalar[DTYPE]](meta[env, w + 3])
            var d = Scalar[DTYPE](0)

            if op == OP_AND:
                var pa = d0 if a == 0 else (d1 if a == 1 else d2)
                var pb = d0 if b == 0 else (d1 if b == 1 else d2)
                d = pa if pa > pb else pb
            elif op == OP_OR:
                var pa = d0 if a == 0 else (d1 if a == 1 else d2)
                var pb = d0 if b == 0 else (d1 if b == 1 else d2)
                d = pa if pa < pb else pb
            elif op == OP_NEAR:
                var ex = rebind[Scalar[DTYPE]](xpos[env, a * 3]) - rebind[
                    Scalar[DTYPE]
                ](xpos[env, b * 3])
                var ey = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1]) - rebind[
                    Scalar[DTYPE]
                ](xpos[env, b * 3 + 1])
                var ez = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2]) - rebind[
                    Scalar[DTYPE]
                ](xpos[env, b * 3 + 2])
                var r = sqrt(ex * ex + ey * ey + ez * ez) - param
                d = r if r > zero else zero
            elif op == OP_ABOVE:
                # ⚠ `pred_above` is `za > zb + margin`, so the shortfall is
                # how much higher `a` still has to be — and it is ZERO the
                # instant the predicate flips, which is the contract.
                var r2 = (
                    rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2]) + param
                ) - rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                d = r2 if r2 > zero else zero
            elif op == OP_UPRIGHT:
                # ⚠ NOT IN METRES, AND THAT IS STATED RATHER THAN SCALED. It
                # is the shortfall in the cosine `pred_upright` compares, so
                # it is in [0, 2] and mixes with a metre term only through the
                # weight the config gives it.
                var qx = rebind[Scalar[DTYPE]](xquat[env, a * 4 + 0])
                var qy = rebind[Scalar[DTYPE]](xquat[env, a * 4 + 1])
                var cosang = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (
                    qx * qx + qy * qy
                )
                var need = Scalar[DTYPE](1) - param
                var r3 = need - cosang
                d = r3 if r3 > zero else zero
            elif op == OP_JOINT:
                comptime if HAS_L3:
                    var q = rebind[Scalar[DTYPE]](qpos[env, a])
                    if not pred_joint[DTYPE](q, b, param):
                        var gap = q - param
                        if gap < zero:
                            gap = -gap
                        d = gap if gap > eps else eps
            elif op == OP_TOUCHING or op == OP_ON_BODY:
                comptime if HAS_L3:
                    var ax = rebind[Scalar[DTYPE]](xpos[env, a * 3])
                    var ay = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1])
                    var az = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                    var bx = rebind[Scalar[DTYPE]](xpos[env, b * 3])
                    var by = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 1])
                    var bz = rebind[Scalar[DTYPE]](xpos[env, b * 3 + 2])
                    var touching = _slots_touching_gpu[
                        DTYPE, BATCH, L_BODIES, L_CON
                    ](a, b, ncon, bodies, contacts, env)
                    var ex = ax - bx
                    var ey = ay - by
                    var ez = az - bz
                    var dist = sqrt(ex * ex + ey * ey + ez * ez)
                    if op == OP_TOUCHING:
                        if not touching:
                            d = dist if dist > eps else eps
                    else:
                        if not pred_ontop[DTYPE](ax, ay, az, bx, by, bz, touching):
                            var xy = sqrt(ex * ex + ey * ey) - Scalar[DTYPE](ONTOP_XY)
                            if xy < zero:
                                xy = zero
                            var zs = bz - az
                            if zs < zero:
                                zs = zero
                            var g = xy + zs
                            if not touching:
                                g += dist
                            d = g if g > eps else eps
            else:
                # IN / ON / AT_REGION — region record `b`
                var ro = b * REGION_WORDS
                var rs = Int(rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_SITE]))
                var rx0 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_X0])
                var ry0 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_Y0])
                var rx1 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_X1])
                var ry1 = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_Y1])
                var rh = rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_H])
                var px: Scalar[DTYPE]
                var py: Scalar[DTYPE]
                var pz: Scalar[DTYPE]
                if op == OP_AT_REGION:
                    px = rebind[Scalar[DTYPE]](site_xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](site_xpos[env, a * 3 + 2])
                else:
                    px = rebind[Scalar[DTYPE]](xpos[env, a * 3])
                    py = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 1])
                    pz = rebind[Scalar[DTYPE]](xpos[env, a * 3 + 2])
                var sx = rebind[Scalar[DTYPE]](site_xpos[env, rs * 3])
                var sy = rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 1])
                var sz = rebind[Scalar[DTYPE]](site_xpos[env, rs * 3 + 2])
                var is_box = False
                comptime if HAS_L3:
                    is_box = (
                        Int(rebind[Scalar[DTYPE]](curriculum[0, ro + CUR_IDX_REGION_BOX])) != 0
                        and op != OP_AT_REGION
                    )
                if is_box:
                    comptime if HAS_L3:
                        var sb = Int(rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_BODY]))
                        var wq = gpu_quat_mul[DTYPE](
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 0]),
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 1]),
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 2]),
                            rebind[Scalar[DTYPE]](xquat[env, sb * 4 + 3]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_X]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_Y]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_Z]),
                            rebind[Scalar[DTYPE]](sites[rs, SITE_IDX_QUAT_W]),
                        )
                        var half = Scalar[DTYPE](0.5)
                        var cx = half * (rx0 + rx1)
                        var cy = half * (ry0 + ry1)
                        var hx = half * (rx1 - rx0)
                        var hy = half * (ry1 - ry0)
                        var c = gpu_quat_rotate[DTYPE](wq[0], wq[1], wq[2], wq[3], cx, cy, zero)
                        var ox = sx + c[0]
                        var oy = sy + c[1]
                        var oz = sz + c[2]
                        if op == OP_IN:
                            if not pred_box_in[DTYPE](
                                px, py, pz, sx, sy, sz,
                                wq[0], wq[1], wq[2], wq[3], cx, cy, hx, hy, rh,
                            ):
                                var ts = gpu_quat_rotate[DTYPE](wq[0], wq[1], wq[2], wq[3], hx, hy, rh)
                                var tx = ts[0] if ts[0] >= zero else -ts[0]
                                var ty = ts[1] if ts[1] >= zero else -ts[1]
                                var tz = ts[2] if ts[2] >= zero else -ts[2]
                                var gx = zero
                                if px < ox - tx:
                                    gx = (ox - tx) - px
                                elif px > ox + tx:
                                    gx = px - (ox + tx)
                                var gy = zero
                                if py < oy - ty:
                                    gy = (oy - ty) - py
                                elif py > oy + ty:
                                    gy = py - (oy + ty)
                                var zlo = oz - tz - Scalar[DTYPE](BOX_IN_Z_SLACK)
                                var gz = zero
                                if pz < zlo:
                                    gz = zlo - pz
                                elif pz > oz + tz:
                                    gz = pz - (oz + tz)
                                var g = sqrt(gx * gx + gy * gy + gz * gz)
                                d = g if g > eps else eps
                        else:
                            var cb = Int(rebind[Scalar[DTYPE]](
                                curriculum[0, ro + CUR_IDX_REGION_CONTACT]
                            ))
                            var touching = True
                            if cb >= 0:
                                touching = _slots_touching_gpu[
                                    DTYPE, BATCH, L_BODIES, L_CON
                                ](a, cb, ncon, bodies, contacts, env)
                            if not pred_box_under[DTYPE](
                                px, py, pz, sx, sy, sz,
                                wq[0], wq[1], wq[2], wq[3], cx, cy, hx, hy, rh,
                                touching,
                            ):
                                var dl = gpu_quat_rotate[DTYPE](
                                    wq[0], wq[1], wq[2], wq[3], px - ox, py - oy, pz - oz
                                )
                                var adx = dl[0] if dl[0] >= zero else -dl[0]
                                var ady = dl[1] if dl[1] >= zero else -dl[1]
                                var gx = adx - hx
                                if gx < zero:
                                    gx = zero
                                var gy = ady - hy
                                if gy < zero:
                                    gy = zero
                                var zlo = rh - Scalar[DTYPE](BOX_UNDER_Z_SLACK)
                                var zhi = rh + Scalar[DTYPE](BOX_UNDER_HEIGHT)
                                var gz = zero
                                if dl[2] < zlo:
                                    gz = zlo - dl[2]
                                elif dl[2] > zhi:
                                    gz = dl[2] - zhi
                                var g = sqrt(gx * gx + gy * gy + gz * gz)
                                if not touching:
                                    var ex = px - ox
                                    var ey = py - oy
                                    var ez = pz - oz
                                    g += sqrt(ex * ex + ey * ey + ez * ez)
                                d = g if g > eps else eps
                else:
                    # the distance to the box, in site coordinates. ⚠ AXIS-
                    # ALIGNED AND CLAMPED PER AXIS: an overshoot on one axis
                    # must not cancel a shortfall on another, which is what a
                    # signed sum would do.
                    var zlo = -rh
                    var zhi = rh
                    if op == OP_ON:
                        zlo = Scalar[DTYPE](ON_MIN_DZ)
                        zhi = Scalar[DTYPE](ON_MAX_DZ)
                    var ux = px - sx
                    var uy = py - sy
                    var uz = pz - sz
                    var gx = zero
                    if ux < rx0:
                        gx = rx0 - ux
                    elif ux > rx1:
                        gx = ux - rx1
                    var gy = zero
                    if uy < ry0:
                        gy = ry0 - uy
                    elif uy > ry1:
                        gy = uy - ry1
                    var gz = zero
                    if uz < zlo:
                        gz = zlo - uz
                    elif uz > zhi:
                        gz = uz - zhi
                    d = sqrt(gx * gx + gy * gy + gz * gz)

            if i == 0:
                d0 = d
            elif i == 1:
                d1 = d
            else:
                d2 = d
            last = d
    return last


@always_inline
def tape_distance_gpu[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
](
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY_F * 3],
    xquat: Tile[DTYPE, BATCH, NBODY_F * 4],
    site_xpos: Tile[DTYPE, BATCH, SITE_DIM],
    env: Int,
) -> Scalar[DTYPE]:
    """The pre-L3 signature: poses only, L3 branches compiled out."""
    comptime L_XP = RowMajor[BATCH, NBODY_F * 3]
    return _tape_distance_impl[
        DTYPE, BATCH, NBODY_F, SITE_DIM, False, L_XP, L_XP, L_XP, L_XP
    ](meta, curriculum, xpos, xquat, site_xpos, xpos, xpos, xpos, xpos, env)


@always_inline
def tape_distance_gpu[
    DTYPE: DType, BATCH: Int, NBODY_F: Int, SITE_DIM: Int,
    NQ_F: Int, NSITE_F: Int, MC_F: Int,
](
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY_F * 3],
    xquat: Tile[DTYPE, BATCH, NBODY_F * 4],
    site_xpos: Tile[DTYPE, BATCH, SITE_DIM],
    qpos: Tile[DTYPE, BATCH, NQ_F],
    sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
    bodies: Tile[DTYPE, NBODY_F, MODEL_BODY_SIZE],
    contacts: Tile[DTYPE, BATCH, MC_F * CONTACT_SIZE],
    env: Int,
) -> Scalar[DTYPE]:
    """The whole language (L3): the reward hook's own operands, one for one."""
    return _tape_distance_impl[
        DTYPE, BATCH, NBODY_F, SITE_DIM, True,
        RowMajor[BATCH, NQ_F],
        RowMajor[NSITE_F, MODEL_SITE_SIZE],
        RowMajor[NBODY_F, MODEL_BODY_SIZE],
        RowMajor[BATCH, MC_F * CONTACT_SIZE],
    ](meta, curriculum, xpos, xquat, site_xpos, qpos, sites, bodies, contacts, env)


def require_gpu_regions(g: BoundGoal, task_name: String) raises:
    """⚠⚠ THE REGION-TABLE RULE, MADE REAL. Call it beside `require_tier_a`.

    `eval_tape_gpu` reads region record `b` out of `curriculum`; the table
    holds `MAX_CURRICULUM_REGIONS` records (16 since L3, when
    `MODEL_CURRICULUM_SIZE` went from 8 to 128 and the term's `b` started
    indexing it — before that it held ONE and every term read region 0
    whatever its `b` said, which is the disagreement this check was born
    to refuse). A family may DECLARE more regions than fit — `init=` is
    sampled on the HOST — but a GOAL naming one past the table would read
    garbage on device and the right region on the host, in the REWARD.
    """
    for i in range(len(g.terms)):
        ref t = g.terms[i]
        if t.op != OP_IN and t.op != OP_ON and t.op != OP_AT_REGION:
            continue
        if t.b >= MAX_CURRICULUM_REGIONS:
            raise Error(
                "task '" + task_name + "': goal names region index "
                + String(t.b) + ", but the device region table holds "
                + String(MAX_CURRICULUM_REGIONS) + " (curriculum is "
                + String(MODEL_CURRICULUM_SIZE) + " words and a region costs "
                + String(REGION_WORDS) + "). The device evaluator would read"
                " past the table for it and the host evaluator would read"
                " the right one, so the GPU and CPU rewards would disagree"
                " silently. Widen MODEL_CURRICULUM_SIZE or reorder the"
                " family's regions so the goal regions come first."
            )
