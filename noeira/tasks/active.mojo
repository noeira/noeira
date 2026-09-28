"""WHICH SLOTS THIS LANE IS RUNNING — the per-lane active mask. P3d.

    var mask = active_mask(t, f)                    # host, once per episode
    meta[e, META_IDX_TASK_ACTIVE] = Scalar[DT](mask)
    ...
    var on = slot_active[DTYPE](meta[env, META_IDX_TASK_ACTIVE], si)  # device

⚠ THIS FILE IS THE HOST HALF. `slot_active` and the observation writer live in
`tasks/obs.mojo`, which imports no spec parser — the same split as
`tape.mojo` / `gpu_eval.mojo`, so a kernel does not drag `spec.mojo` into its
compile unit. They are re-exported here because the mask is one story.

`TASK_LAYER_PLAN.md` §3.4, and the gap `tasks/family_config.mojo` stated
rather than implied: an inactive slot still EXISTS — the fixed scene budget
means every task in a family instantiates every slot — so its pose is in the
state whether or not the task uses it. Without a mask the policy reads a
parked pose and has to learn that one particular constant means "absent". That
is a convention, and a convention nothing checks is a representation bug
waiting for the day the constant changes.

## ⚠ THE MASK IS OVER FAMILY SLOT INDICES, NOT OVER FREE SLOTS

`f.slots` order is the family's declaration order and is already the
observation layout and the `qpos` layout (`family.compose_family`). Bit `i` is
family slot `i`, static slots included — which costs nothing and means
`slot_active(mask, f.slot_index(name))` is the whole decode, with no second
numbering to keep in step.

⚠ A STATIC SLOT'S BIT IS MEANINGFUL BUT NOT OBSERVABLE. It has no joint and
therefore no state, so nothing in the observation varies with it; the bit
records what the TASK said. Do not read that as licence to make a fixture
disappear — `spec.validate_task_against_family` is where that argument
belongs.

## ⚠⚠ IT LIVES IN `meta`, AND `meta` IS NOT ZEROED BETWEEN EPISODES

`META_IDX_TASK_ACTIVE` survives `_reset_env_lane`, which is what lets it be
written once per episode and read every step — the same property the tape
relies on. The other half of that property is that a lane KEEPS the previous
episode's mask unless the writer rewrites it. The caller therefore writes the
whole word every episode, never a bit, and there is no partial-update API here
on purpose.

## ⚠ WHY THIS IS NOT A CHANNEL OF ITS OWN

Gap E's instinct was a new per-lane kernel operand, measured against Metal's
cliff of 28. `meta` is already per-lane and already an operand of
`pre_step_gpu`, `init_qpos_gpu`, `custom_extract_obs_gpu` and
`compute_reward_and_done_gpu` — every hook that could want this. It cost one
word of `METADATA_SIZE` and no signature anywhere.
"""

from .spec import FamilySpec, TaskSpec, SLOT_FREE, INIT_TARGET_SLOT
from .placement.table import (
    INIT_WORD_IN_BIAS, INIT_WORD_SEP_UNIT, INIT_WORD_YAW_BIAS,
)
from noeira.physics3d.gpu.constants import META_INIT_SLOTS
from .obs import slot_active, write_free_slot_obs, FREE_JOINT_NQ, FREE_JOINT_NV


# ⚠ THE CEILING IS THE FLOAT'S, NOT THE FAMILY'S. `meta` is `Scalar[DT]` and
# DT may be float32, whose mantissa is 24 bits — integers are exact to 2^24.
# The compile ceiling for a family is k=13 slots
# (`docs/TASK_LAYER_IMPLEMENTATION.md` §1.0), so this is never the binding
# limit; it is here so that the day it WOULD be, the answer is an error and
# not a bit that rounds away.
comptime MASK_SLOT_LIMIT: Int = 24


def active_mask(t: TaskSpec, f: FamilySpec) raises -> Float64:
    """This task's active set as a bitmask over `f.slots` indices.

    ⚠ DERIVED FROM `t.active`, NOT FROM THE PLACEMENTS. `reset.reset_slots`
    also knows which slots are active, but only for FREE ones and only for the
    ones the sampler managed to place. A slot the sampler failed on is still
    active — the episode is degenerate, not re-specified — and deriving the
    mask from placements would quietly relabel it as absent, which is the one
    reading under which the failure looks fine.

    ⚠ ASSUMES `validate_task_against_family` HAS RUN. It is what rejects an
    `active=` naming a slot the family does not declare; here such a name
    raises rather than being skipped, because a skipped name is a slot the
    author asked for and did not get.
    """
    if len(f.slots) > MASK_SLOT_LIMIT:
        raise Error(
            "tasks: family '" + f.name + "' declares " + String(len(f.slots))
            + " slots; the active mask is one float and holds "
            + String(MASK_SLOT_LIMIT) + " bits exactly. Widen"
            " `MASK_SLOT_LIMIT` only after checking `DT`'s mantissa — past it"
            " the high bits do not overflow, they ROUND."
        )
    var mask = 0
    for i in range(len(t.active)):
        var si = f.slot_index(t.active[i])
        if si < 0:
            raise Error(
                "tasks: task '" + t.name + "' lists active slot '"
                + t.active[i] + "', which family '" + f.name + "' does not"
                " declare. `validate_task_against_family` should have caught"
                " this first."
            )
        mask |= 1 << si
    return Float64(mask)


def mask_slots(mask: Float64, nslots: Int) -> List[Bool]:
    """`mask` expanded to one `Bool` per family slot. Host-side, for gates and
    diagnostics — the device decodes one bit at a time with `slot_active`."""
    var out = List[Bool]()
    var m = Int(mask)
    for i in range(nslots):
        out.append(((m >> i) & 1) == 1)
    return out^


# ── the per-free-slot init region words ────────────────────────────────────

def init_region_words(t: TaskSpec, f: FamilySpec) raises -> List[Float64]:
    """One word per FREE slot: where its `init=` starts it, or 0.

    ⚠ THE ENCODING IS `placement/table.mojo`'s — a region `r + 1`, the same
    plus `INIT_WORD_IN_BIAS` for an `In` and `INIT_WORD_YAW_BIAS` for `:yaw`,
    plus `sep_mm * INIT_WORD_SEP_UNIT` for `:sep=`, `-(s + 1)` for a STACK on
    family slot `s` — and this is its one writer. The paragraphs below predate stacks and
    `In`, and still hold for the region case.

    ⚠⚠ INDEXED BY FREE-SLOT ORDINAL, NOT BY FAMILY SLOT INDEX. The `meta`
    words are `META_IDX_INIT_REGION_0..` and there are `N_FREE_SLOTS` of
    them, while the family's slot 0 here is a static fixture — so the returned
    list is `[brick, cube_a, cube_b]`, not `[table, brick, cube_a, cube_b]`.
    The active MASK is the other convention (bit per FAMILY slot, so the
    fixture owns bit 0), and the two disagree on purpose: the mask describes
    the scene and this describes the reset, which only free slots take part in.

    ⚠⚠ THE VALUE IS `region_index + 1`, AND ZERO MEANS "NO init=". `Data`
    uploads a ZERO-FILLED `meta` at construction, so zero is what a lane reads
    until a driver writes these words — and with 0 meaning `table_top` a
    driver that forgot them would silently PLACE every free slot instead of
    parking it. Biasing by one makes the untouched value mean the safe thing.
    `constants.INIT_REGION_NONE` is that zero.

    ⚠ THE UNBIASED VALUE IS A FAMILY REGION INDEX, so it indexes the same
    table `region_rects` and `region_half_heights` produce — and it is what
    the DEVICE resolves against its restated copy. A task naming a region the
    family does not declare is refused by `validate_task_against_family` long
    before here.
    """
    var out = List[Float64]()
    for i in range(len(f.slots)):
        if f.slots[i].kind != SLOT_FREE:
            continue
        var w = 0
        for k in range(len(t.inits)):
            if t.inits[k].slot != f.slots[i].name:
                continue
            ref it = t.inits[k]
            if f.init_target_kind(it.region) == INIT_TARGET_SLOT:
                # ⚠ A STACK NAMES THE FAMILY SLOT, NOT A REGION. This used to
                # fall through `region_index` to -1 and write 0 — so a stacked
                # prop was silently PARKED on the device, which is why
                # `require_gpu_placement` had to refuse every task with one.
                if it.yaw or it.sep_mm > 0:
                    raise Error(
                        "task '" + t.name + "': init '" + it.describe()
                        + "' stacks, and ':yaw' / ':sep=' apply to a region draw"
                    )
                w = -(f.slot_index(it.region) + 1)
            else:
                var ri = f.region_index(it.region)
                if ri + 1 >= INIT_WORD_IN_BIAS:
                    raise Error(
                        "task '" + t.name + "': init region '" + it.region
                        + "' is family region " + String(ri) + ", and an init"
                        " word holds regions below "
                        + String(INIT_WORD_IN_BIAS - 1) + " (the `In` bias)."
                    )
                w = ri + 1
                if it.inside:
                    w += INIT_WORD_IN_BIAS
                if it.yaw:
                    w += INIT_WORD_YAW_BIAS
                # `:sep=` in whole mm, above both biases (`InitSpec.sep_mm`
                # refuses anything outside 1..1023)
                w += it.sep_mm * INIT_WORD_SEP_UNIT
        out.append(Float64(w))
    if len(out) > META_INIT_SLOTS:
        raise Error(
            "task '" + t.name + "': family '" + f.name + "' has "
            + String(len(out)) + " free slots and `meta` holds "
            + String(META_INIT_SLOTS) + " init words. Widen META_INIT_SLOTS"
            " (it is appended at the end of `meta`) or split the family."
        )
    return out^
