"""The per-lane ACTIVE MASK, and the observation that carries it — P3d.

`TASK_LAYER_PLAN.md` §3.4: an inactive slot still exists, so its pose is in
the state whether or not the task uses it. Give each slot `(pose, active)`
explicitly rather than letting one particular constant mean "absent".

⚠ RUN ON THE CPU, AT float64. `So101TabletopConfig.custom_extract_obs_gpu` is
generic over dtype and over the tensor's origin, so the REAL HOOK — not a
re-spelling of it — is instantiated here without a GPU. Same trick, same
reason as `test_tape_gpu_parity`.

## WHAT IT ASSERTS, AND THE ORDER MATTERS

1. **THE RESTATEMENT IS TRUE.** `So101TabletopConfig` restates the free
   slots' `qposadr` / `dofadr` and their family indices, because a config is a
   comptime type and a `.family` is a runtime file. This loads the family,
   resolves the addresses out of the COMPOSED SCENE through
   `reset.free_slot_addresses`, and asserts every one. Without this check the
   restatement is a second copy of the layout with nothing holding it to the
   first — `_a_rule_written_inline_twice_drifts`.

2. **THE MASK ROUND-TRIPS**, host writer to device decoder, on the three real
   tasks.

3. **THE OBSERVATION IS WHAT THE LAYOUT SAYS**: full `qpos`, then `qvel`, then
   one active word per free slot, with an inactive slot's pose AND velocity
   words zeroed.

4. ⚠⚠ **THE NEGATIVE LEG — TWO LANES, DIFFERENT ACTIVE SETS.** Every check
   above passes on a hook that reads lane 0's mask for every lane, because at
   one lane there is nothing to be wrong about. `gather` runs three slots and
   `reach` runs two, so the two lanes' observations must DIFFER in exactly
   `cube_a`'s words. That is the same defect shape P3's 1024-lane gate exists
   for, at the smallest size that can show it.

5. **THE CPU HOOK AND THE GPU HOOK AGREE WORD FOR WORD.** They are written
   twice because one is handed a `List` and the other a `LayoutTensor`; a
   batched run writes a checkpoint a single-env eval loads, so a permutation
   between them is a policy that works on one device and is nonsense on the
   other, with no error anywhere. They are compared to each other, never each
   to a description.

6. **THE REPARK PINS EXACTLY THE INACTIVE SLOTS** (Gap D). Pose AND velocity,
   every step, and an ACTIVE slot's words bit-identical across the call — a
   stray write there would freeze the props the task is about while the arm
   pushed at them, and the reward would read a scene that never moves.

7. ⚠⚠ **THE WIDE PRE-STEP HOOK STILL REACHES A NARROW OVERRIDE.** Gap D added
   `pre_step_full_gpu` (a second hook, `qvel` included) and rewired the env to
   call ONLY it — so all fourteen configs that override `pre_step_gpu` now
   depend on the trait's default forward. The gate calls the WIDE hook on
   `HalfCheetahConfig`, which overrides only the narrow one, and asserts its
   write lands. A broken forward is otherwise silent: the hook stashes a
   value nothing reads until the reward, and the reward just gets stale.

8. ⚠ **ANTI-VACUITY.** `qpos` and `qvel` are seeded with values that are
   nowhere zero, and the gate prints how many words it copied against how many
   it zeroed. "Every inactive word is zero" is also true of an all-zero
   observation, and an all-zero observation is what a hook that returned
   `False` would leave behind.

Run: pixi run mojo run -I . tests/tasks/test_active_mask.mojo
"""

from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_ACTIVE,
    MODEL_BODY_SIZE, MODEL_SITE_SIZE, MODEL_GEOM_SIZE, CONTACT_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from noeira.tasks.family import scene_path, park_pos
from noeira.tasks.active import active_mask, mask_slots, MASK_SLOT_LIMIT
from noeira.tasks.obs import slot_active, FREE_JOINT_NQ, FREE_JOINT_NV
from noeira.physics3d.fields import Data, DynDims
from noeira.tasks.reset import free_slot_addresses
from noeira.tasks.family_config import So101TabletopConfig, So101TabletopPlacement
from noeira.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from noeira.physics3d.gpu.constants import (
    META_IDX_PREV_X, META_IDX_TASK_PARAM_0,
)
from noeira.tasks.predicates import parse_goal, bind_goal
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.so101_tabletop_xml import (
    SO101_TABLETOP_N_GOAL_WORDS,
    So101TabletopModel, SO101_TABLETOP_N_FREE_SLOTS, SO101_TABLETOP_OBS_DIM,
)


comptime DTYPE = DType.float64
comptime BATCH = 2

comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV
comptime NB = So101TabletopModel.NBODY
comptime NS = So101TabletopModel.NSITE
comptime NG = So101TabletopModel.NGEOM
comptime NA = So101TabletopModel.ACTION_DIM
comptime MC = So101TabletopModel.MAX_CONTACTS
comptime OD = So101TabletopModel.OBS_DIM
comptime SD = NS * 3
comptime NFREE = So101TabletopConfig.N_FREE_SLOTS
comptime NGOAL = SO101_TABLETOP_N_GOAL_WORDS

comptime CFG = So101TabletopConfig


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _cfg_slot_idx(j: Int) -> Int:
    """The config's restated family slot index for free slot `j`."""
    if j == 0:
        return So101TabletopPlacement.FREE_SLOT_IDX_0
    if j == 1:
        return So101TabletopPlacement.FREE_SLOT_IDX_1
    return So101TabletopPlacement.FREE_SLOT_IDX_2


def _cfg_qadr(j: Int) -> Int:
    if j == 0:
        return So101TabletopPlacement.FREE_QADR_0
    if j == 1:
        return So101TabletopPlacement.FREE_QADR_1
    return So101TabletopPlacement.FREE_QADR_2


def _cfg_dadr(j: Int) -> Int:
    if j == 0:
        return So101TabletopPlacement.FREE_DADR_0
    if j == 1:
        return So101TabletopPlacement.FREE_DADR_1
    return So101TabletopPlacement.FREE_DADR_2


def main() raises:
    print("=== the per-lane active mask, and the obs that carries it ===")
    var ta = Tally()

    var f = load_family("noeira/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))

    # ── 1. the config's restatement, against the composed scene ───────────
    print()
    print("--- 1. the free-slot table the config restates ---")
    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    # The family's free slots, in declaration order — the ordering the config
    # numbers its `FREE_*_j` by.
    var free_idx = List[Int]()
    for si in range(len(f.slots)):
        if f.slots[si].kind == SLOT_FREE:
            free_idx.append(si)

    ta.check(
        len(free_idx) == NFREE and NFREE == SO101_TABLETOP_N_FREE_SLOTS,
        "the family declares " + String(len(free_idx)) + " free slots and the"
        " config says " + String(NFREE),
    )
    var table_ok = len(free_idx) == NFREE
    for j in range(len(free_idx)):
        if j >= NFREE:
            break
        var si = free_idx[j]
        var got_si = _cfg_slot_idx(j)
        var got_q = _cfg_qadr(j)
        var got_d = _cfg_dadr(j)
        var want_q = addrs[si].qadr
        var want_d = addrs[si].dadr
        var row_ok = (got_si == si) and (got_q == want_q) and (got_d == want_d)
        if not row_ok:
            table_ok = False
        print(
            "    free", j, "=", f.slots[si].name,
            "| scene: slot", si, "qadr", want_q, "dadr", want_d,
            "| config: slot", got_si, "qadr", got_q, "dadr", got_d,
        )
    ta.check(
        table_ok,
        "every restated (slot, qposadr, dofadr) matches the composed scene",
    )
    # ⚠ THE TWO ADDRESS SPACES DIVERGE — 7 qpos against 6 qvel per free joint.
    # A table that reused one for the other is right for the first free slot
    # and wrong for every later one, which reads as "the last prop's velocity
    # belongs to somebody else".
    var diverges = False
    for j in range(1, len(free_idx)):
        if j < NFREE and _cfg_qadr(j) != _cfg_dadr(j):
            diverges = True
    ta.check(
        diverges,
        "qposadr and dofadr DIVERGE after the first free slot (7 vs 6)",
    )

    # ⚠ NINE GOAL WORDS AFTER THE MASK — gripper(3), subject-gripper(3),
    # target-subject(3). This gate caught the widening, which is what it is
    # for: it asserts the EXPRESSION, so a layout change has to be stated here
    # before it can pass. See `so101_tabletop_xml.SO101_TABLETOP_OBS_DIM` for
    # the measurement that forced them in — half the shaped reward was
    # computed from the gripper's Cartesian position, which nothing in
    # qpos/qvel gives.
    ta.check(
        OD == NQ + NV + NFREE + NGOAL and OD == SO101_TABLETOP_OBS_DIM,
        "OBS_DIM " + String(OD) + " == NQ " + String(NQ) + " + NV "
        + String(NV) + " + " + String(NFREE) + " active + "
        + String(NGOAL) + " goal words",
    )
    ta.check(
        CFG.OBS_MASK_BASE == NQ + NV,
        "the mask words start at obs[" + String(CFG.OBS_MASK_BASE) + "]",
    )
    # ⚠ THE `obs_qpos_skip` CORRECTION. The model default would have made
    # OBS_DIM `NQ - 1 + NV`; on a desk arm that dropped `shoulder_pan`.
    ta.check(
        OD != NQ - 1 + NV + NFREE,
        "the observation carries the FULL qpos, not qpos[1:]",
    )

    # ── 2. the mask, host writer to device decoder ────────────────────────
    print()
    print("--- 2. the mask round-trips ---")
    var tg = load_task("noeira/tasks/tasks/so101_gather_bricks.task")
    var tr = load_task("noeira/tasks/tasks/so101_reach_brick.task")
    # ⚠ REAL TAPES, for the nine goal words below — see the note at their
    # write. `gather`'s term 0 is `Near` (two BODIES) and `reach`'s is
    # `AtRegion` (a SITE subject), so the two lanes exercise both arms of
    # `goal_frame_ids`.
    var fmd_t = parse_model_runtime(scene_path(f))
    var tape_g = encode_goal(
        bind_goal(parse_goal(tg.goal), f, fmd_t.body_names, fmd_t.site_names)
    )
    var tape_r = encode_goal(
        bind_goal(parse_goal(tr.goal), f, fmd_t.body_names, fmd_t.site_names)
    )
    validate_task_against_family(tg, f)
    validate_task_against_family(tr, f)
    var mg = active_mask(tg, f)
    var mr = active_mask(tr, f)
    print("    ", tg.name, "-> mask", mg)
    print("    ", tr.name, "-> mask", mr)

    var round_ok = True
    var lit = 0
    var bits_g = mask_slots(mg, len(f.slots))
    for si in range(len(f.slots)):
        var want = tg.is_active(f.slots[si].name)
        if bits_g[si] != want:
            round_ok = False
        if slot_active[DTYPE](Scalar[DTYPE](mg), si) != want:
            round_ok = False
        if want:
            lit += 1
    ta.check(round_ok, "every slot's bit == `t.is_active` (host and device)")
    # ⚠ ANTI-VACUITY: an all-zero mask also round-trips against an
    # `is_active` that answered False to everything.
    ta.check(
        lit > 0 and lit < len(f.slots),
        String(lit) + " of " + String(len(f.slots))
        + " slots lit — the mask is neither empty nor full",
    )
    ta.check(
        mg != mr,
        "the two tasks have DIFFERENT masks (" + String(mg) + " vs "
        + String(mr) + ")",
    )
    ta.check(
        len(f.slots) <= MASK_SLOT_LIMIT,
        "the family fits the mask's " + String(MASK_SLOT_LIMIT) + " bits",
    )

    # ── 3 & 4. the observation hook, two lanes, different active sets ─────
    print()
    print("--- 3. the observation, lane 0 =", tg.name, "lane 1 =", tr.name, "---")
    comptime L_QPOS = RowMajor[BATCH, NQ]
    comptime L_QVEL = RowMajor[BATCH, NV]
    comptime L_META = RowMajor[BATCH, METADATA_SIZE]
    comptime L_OBS = RowMajor[BATCH, OD]
    comptime L_B3 = RowMajor[BATCH, NB * 3]
    comptime L_B4 = RowMajor[BATCH, NB * 4]
    comptime L_B6 = RowMajor[BATCH, NB * 6]
    comptime L_SXP = RowMajor[BATCH, SD]
    comptime L_BODY = RowMajor[NB, MODEL_BODY_SIZE]
    comptime L_SITE = RowMajor[NS, MODEL_SITE_SIZE]
    comptime L_GEOM = RowMajor[NG, MODEL_GEOM_SIZE]
    comptime L_CON = RowMajor[BATCH, MC * CONTACT_SIZE]
    comptime L_ACT = RowMajor[BATCH, NA]

    var qpos = TensorImpl[DTYPE].alloc(BATCH * NQ)
    var qvel = TensorImpl[DTYPE].alloc(BATCH * NV)
    var meta = TensorImpl[DTYPE].alloc(BATCH * METADATA_SIZE)
    var obs = TensorImpl[DTYPE].alloc(BATCH * OD)
    var b3 = TensorImpl[DTYPE].alloc(BATCH * NB * 3)
    var b4 = TensorImpl[DTYPE].alloc(BATCH * NB * 4)
    var b6 = TensorImpl[DTYPE].alloc(BATCH * NB * 6)
    var sxp = TensorImpl[DTYPE].alloc(BATCH * SD)
    var mbody = TensorImpl[DTYPE].alloc(NB * MODEL_BODY_SIZE)
    var msite = TensorImpl[DTYPE].alloc(NS * MODEL_SITE_SIZE)
    var mgeom = TensorImpl[DTYPE].alloc(NG * MODEL_GEOM_SIZE)
    var con = TensorImpl[DTYPE].alloc(BATCH * MC * CONTACT_SIZE)
    var act = TensorImpl[DTYPE].alloc(BATCH * NA)

    # ⚠ NOWHERE ZERO. "The inactive words came back zero" says nothing about a
    # hook that wrote nothing at all, unless the source was nonzero first.
    for e in range(BATCH):
        for i in range(NQ):
            qpos.data[e * NQ + i] = Scalar[DTYPE](100.0 + Float64(i))
        for i in range(NV):
            qvel.data[e * NV + i] = Scalar[DTYPE](200.0 + Float64(i))
        for i in range(METADATA_SIZE):
            meta.data[e * METADATA_SIZE + i] = Scalar[DTYPE](0)
        for i in range(OD):
            obs.data[e * OD + i] = Scalar[DTYPE](-999.0)
    meta.data[0 * METADATA_SIZE + META_IDX_TASK_ACTIVE] = Scalar[DTYPE](mg)
    meta.data[1 * METADATA_SIZE + META_IDX_TASK_ACTIVE] = Scalar[DTYPE](mr)

    # ⚠⚠ A REAL TAPE AND REAL SITE POSITIONS, OR THE NINE GOAL WORDS ARE
    # VACUOUS. They are derived from term 0 of the tape and from `site_xpos`,
    # and with `meta` zeroed the op reads as 0 — which is `OP_IN`, not "empty"
    # — with body 0 and a site at the origin, so every one of the nine comes
    # out ZERO. "The two hooks agree word for word" is then true of nine zeros
    # and says nothing about the half of the observation that was just added.
    for k in range(TAPE_WORDS):
        meta.data[0 * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = (
            Scalar[DTYPE](tape_g[k])
        )
        meta.data[1 * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = (
            Scalar[DTYPE](tape_r[k])
        )
    # distinct, nonzero, and different per lane so a hook that ignored `env`
    # is visible too
    for e in range(BATCH):
        for i in range(SD):
            sxp.data[e * SD + i] = Scalar[DTYPE](
                0.5 + Float64(i) * 0.01 + Float64(e) * 0.1
            )
        for i in range(NB * 3):
            b3.data[e * NB * 3 + i] = Scalar[DTYPE](
                1.5 + Float64(i) * 0.02 + Float64(e) * 0.1
            )

    var wrote = False
    for e in range(BATCH):
        wrote = CFG.custom_extract_obs_gpu[
            DTYPE, BATCH, NQ, NV, NB, OD, SD, MC, NS, NG, NA
        ](
            qpos.tt["cpu", BATCH, NQ](), qvel.tt["cpu", BATCH, NV](),
            b3.tt["cpu", BATCH, NB * 3](), b4.tt["cpu", BATCH, NB * 4](), b3.tt["cpu", BATCH, NB * 3](),
            mbody.tt["cpu", NB, MODEL_BODY_SIZE](), sxp.tt["cpu", BATCH, SD](),
            con.tt["cpu", BATCH, MC * CONTACT_SIZE](), msite.tt["cpu", NS, MODEL_SITE_SIZE](),
            mgeom.tt["cpu", NG, MODEL_GEOM_SIZE](), meta.tt["cpu", BATCH, METADATA_SIZE](),
            obs.tt["cpu", BATCH, OD](),
            b3.tt["cpu", BATCH, NB * 3](), b3.tt["cpu", BATCH, NB * 3](), b6.tt["cpu", BATCH, NB * 6](),
            b6.tt["cpu", BATCH, NB * 6](), b6.tt["cpu", BATCH, NB * 6](), b3.tt["cpu", BATCH, NB * 3](),
            sxp.tt["cpu", BATCH, SD](), b4.tt["cpu", BATCH, NB * 4](),
            act.tt["cpu", BATCH, NA](),
            e,
        )
    ta.check(wrote, "the hook claims the observation (returns True)")

    # Expected, from the family + the two tasks — NOT from the config's table.
    var copied = 0
    var zeroed = 0
    var lane_ok = True
    for e in range(BATCH):
        var t_active = List[Bool]()
        for j in range(NFREE):
            var nm = f.slots[free_idx[j]].name
            t_active.append(
                tg.is_active(nm) if e == 0 else tr.is_active(nm)
            )
        # which qpos / qvel words belong to an INACTIVE free slot
        var q_dead = List[Bool]()
        for _ in range(NQ):
            q_dead.append(False)
        var v_dead = List[Bool]()
        for _ in range(NV):
            v_dead.append(False)
        for j in range(NFREE):
            if not t_active[j]:
                for k in range(FREE_JOINT_NQ):
                    q_dead[addrs[free_idx[j]].qadr + k] = True
                for k in range(FREE_JOINT_NV):
                    v_dead[addrs[free_idx[j]].dadr + k] = True

        for i in range(NQ):
            var got = Float64(obs.data[e * OD + i])
            if q_dead[i]:
                zeroed += 1
                if got != 0.0:
                    lane_ok = False
            else:
                copied += 1
                if got != Float64(qpos.data[e * NQ + i]):
                    lane_ok = False
        for i in range(NV):
            var got = Float64(obs.data[e * OD + NQ + i])
            if v_dead[i]:
                zeroed += 1
                if got != 0.0:
                    lane_ok = False
            else:
                copied += 1
                if got != Float64(qvel.data[e * NV + i]):
                    lane_ok = False
        for j in range(NFREE):
            var got = Float64(obs.data[e * OD + NQ + NV + j])
            var want = 1.0 if t_active[j] else 0.0
            if got != want:
                lane_ok = False

    print("    ", copied, "state words copied,", zeroed, "zeroed as inactive")
    ta.check(lane_ok, "every obs word is the state, a zero, or the right bit")
    # ⚠ ANTI-VACUITY, THE OTHER HALF: with nothing inactive there is no
    # zeroing to get wrong, and the check above would pass on a hook that
    # never zeroes.
    ta.check(zeroed > 0, "the sweep actually exercised an INACTIVE slot")
    ta.check(copied > 0, "and an active one")

    # ── 4. the negative leg ───────────────────────────────────────────────
    print()
    print("--- 4. the two lanes differ (each read ITS OWN mask) ---")
    var bit_diff = 0
    for j in range(NFREE):
        if obs.data[0 * OD + NQ + NV + j] != obs.data[1 * OD + NQ + NV + j]:
            bit_diff += 1
    var word_diff = 0
    for i in range(OD):
        if obs.data[0 * OD + i] != obs.data[1 * OD + i]:
            word_diff += 1
    print("    ", bit_diff, "active words differ,", word_diff,
          "observation words differ in total")
    # ⚠⚠ THE WHOLE POINT. The two lanes were handed IDENTICAL qpos and qvel;
    # the only thing that differs is the mask. A hook that read lane 0's mask
    # for every lane produces two identical observations and passes every
    # check above this line.
    ta.check(
        bit_diff > 0,
        "the lanes' active words DIFFER — the mask is read per lane",
    )
    # ⚠ THE GOAL WORDS DIFFER TOO, AND THEY SHOULD. Lane 0 runs `gather`
    # (`Near`, two BODIES) and lane 1 runs `reach` (`AtRegion`, a SITE
    # subject), so `goal_frame_ids` resolves different frames and both the
    # subject and target vectors differ. Counting them is the assertion that
    # the goal words are READ PER LANE from that lane's own tape — a hook that
    # used lane 0's tape for everyone would leave these nine identical.
    var goal_diff = 0
    for j in range(NGOAL):
        if obs.data[0 * OD + CFG.OBS_GOAL_BASE + j] != obs.data[
            1 * OD + CFG.OBS_GOAL_BASE + j
        ]:
            goal_diff += 1
    print("     ", goal_diff, "of", NGOAL, "goal words differ between lanes")
    ta.check(
        goal_diff > 0,
        "the lanes' GOAL words differ — term 0 is read from each lane's own"
        " tape",
    )
    ta.check(
        word_diff == bit_diff + FREE_JOINT_NQ + FREE_JOINT_NV + goal_diff,
        "and they differ in exactly cube_a's "
        + String(FREE_JOINT_NQ + FREE_JOINT_NV) + " state words plus "
        + String(bit_diff) + " active and " + String(goal_diff)
        + " goal word(s)",
    )

    # ── 5. the CPU hook == the GPU hook, word for word ────────────────────
    print()
    print("--- 5. the single-env hook agrees with the batched one ---")
    var dyn = DynDims(
        nq=NQ, nv=NV, nbody=NB, njoint=len(fmd.joints), ngeom=NG,
        nsite=NS, nact=NA, max_contacts=MC,
    )
    var cpu_ok = True
    var cpu_words = 0
    for e in range(BATCH):
        var dd = Data[DTYPE, DynDims, 1](dyn)
        for i in range(NQ):
            dd.qpos.data[i] = qpos.data[e * NQ + i]
        for i in range(NV):
            dd.qvel.data[i] = qvel.data[e * NV + i]
        for i in range(METADATA_SIZE):
            dd.meta.data[i] = meta.data[e * METADATA_SIZE + i]
        # ⚠⚠ AND THE FRAMES THE GOAL WORDS READ. This loop copied qpos, qvel
        # and meta only, so the CPU hook saw ZEROS where the GPU hook saw the
        # positions set above — and the word-for-word comparison then failed
        # on nine words for a reason that was in the TEST. It is also what
        # made those nine words vacuous before they were populated: two hooks
        # agreeing on nine zeros is not agreement about anything.
        for i in range(NB * 3):
            dd.xpos.data[i] = b3.data[e * NB * 3 + i]
        for i in range(SD):
            dd.site_xpos.data[i] = sxp.data[e * SD + i]
        var hobs = List[Scalar[DTYPE]]()
        var empty = List[Scalar[DTYPE]]()
        var handled = CFG.custom_extract_obs_cpu[DTYPE, DynDims](
            dd, empty, empty, empty, empty, empty, hobs
        )
        if not handled or len(hobs) != OD:
            cpu_ok = False
        else:
            for i in range(OD):
                cpu_words += 1
                if hobs[i] != obs.data[e * OD + i]:
                    cpu_ok = False
    print("    ", cpu_words, "words compared across", BATCH, "lanes")
    ta.check(
        cpu_ok,
        "custom_extract_obs_cpu == custom_extract_obs_gpu, word for word",
    )
    ta.check(
        cpu_words == BATCH * OD,
        "both hooks produced " + String(OD) + "-wide observations",
    )

    # ── 6. Gap D: the repark pins the inactive slots, every step ──────────
    print()
    print("--- 6. the repark (Gap D) ---")
    # ⚠ THE PARK CONSTANTS THE CONFIG RESTATES, against the loaded family.
    # Same drift risk and same closure as the address table above.
    var park_ok = True
    for j in range(NFREE):
        var si = free_idx[j]
        var want = park_pos(f, si)
        var gx = So101TabletopPlacement.PARK_X + Float64(si) * So101TabletopPlacement.PARK_SPACING
        if gx != want[0] or So101TabletopPlacement.PARK_Y != want[1] or So101TabletopPlacement.PARK_Z != want[2]:
            park_ok = False
        print("    slot", si, f.slots[si].name, "parks at (", want[0], ",",
              want[1], ",", want[2], ") | config (", gx, ",", So101TabletopPlacement.PARK_Y,
              ",", So101TabletopPlacement.PARK_Z, ")")
    ta.check(park_ok, "the restated park pose matches `park_pos` on the family")

    # Lane 0 = gather (cube_b inactive), lane 1 = reach (cube_a and cube_b
    # inactive). Poison every free slot's state so a pin is visible and a
    # NON-pin is too.
    for e in range(BATCH):
        for i in range(NQ):
            qpos.data[e * NQ + i] = Scalar[DTYPE](-3333.0)
        for i in range(NV):
            qvel.data[e * NV + i] = Scalar[DTYPE](-4444.0)

    for e in range(BATCH):
        CFG.pre_step_full_gpu[DTYPE, BATCH, NQ, NV](
            qpos.tt["cpu", BATCH, NQ](), qvel.tt["cpu", BATCH, NV](),
            meta.tt["cpu", BATCH, METADATA_SIZE](), e,
        )

    var pinned = 0
    var untouched = 0
    var repark_ok = True
    for e in range(BATCH):
        for j in range(NFREE):
            var si = free_idx[j]
            var qa = addrs[si].qadr
            var da = addrs[si].dadr
            var nm = f.slots[si].name
            var on = tg.is_active(nm) if e == 0 else tr.is_active(nm)
            if on:
                # ⚠⚠ AN ACTIVE SLOT MUST BE BIT-IDENTICAL — still the poison.
                untouched += 1
                for k in range(FREE_JOINT_NQ):
                    if Float64(qpos.data[e * NQ + qa + k]) != -3333.0:
                        repark_ok = False
                for k in range(FREE_JOINT_NV):
                    if Float64(qvel.data[e * NV + da + k]) != -4444.0:
                        repark_ok = False
            else:
                pinned += 1
                var pk = park_pos(f, si)
                if (
                    Float64(qpos.data[e * NQ + qa + 0]) != pk[0]
                    or Float64(qpos.data[e * NQ + qa + 1]) != pk[1]
                    or Float64(qpos.data[e * NQ + qa + 2]) != pk[2]
                ):
                    repark_ok = False
                # ⚠ THE QUATERNION, W FIRST. `(0,0,0,0)` is degenerate and FK
                # turns it into a NaN pose — the trap `write_free_pose`
                # records, here on every step instead of once.
                if Float64(qpos.data[e * NQ + qa + 3]) != 1.0:
                    repark_ok = False
                for k in range(4, FREE_JOINT_NQ):
                    if Float64(qpos.data[e * NQ + qa + k]) != 0.0:
                        repark_ok = False
                # ⚠ AND THE VELOCITY. Pose-only leaves `qvel` growing at g*dt
                # forever while the position looks parked.
                for k in range(FREE_JOINT_NV):
                    if Float64(qvel.data[e * NV + da + k]) != 0.0:
                        repark_ok = False
    print("    ", pinned, "slot-lanes pinned,", untouched,
          "left bit-identical")
    ta.check(repark_ok,
             "every INACTIVE slot is at its park pose with zero velocity, and"
             " every ACTIVE one is untouched")
    # ⚠ ANTI-VACUITY BOTH WAYS: with nothing inactive there is no pin to get
    # wrong, and with nothing active a hook that pinned EVERYTHING passes.
    ta.check(pinned > 0 and untouched > 0,
             "the sweep exercised both an inactive slot and an active one")

    # ⚠ THE ARM IS NOT A SLOT AND MUST NOT BE TOUCHED. `qpos[0..5]` are the
    # six hinges; a repark that walked the wrong address space would land in
    # them and freeze the robot, which reads as a policy that cannot move.
    var arm_ok = True
    for e in range(BATCH):
        for k in range(6):
            if Float64(qpos.data[e * NQ + k]) != -3333.0:
                arm_ok = False
    ta.check(arm_ok, "the arm's own six qpos words are untouched")

    # ── 7. the default forward, on a config that never heard of it ────────
    print()
    print("--- 7. a NARROW override is still reached through the wide hook ---")
    # ⚠ A REAL CONFIG, NOT A FIXTURE. `HalfCheetahConfig.pre_step_gpu` stashes
    # `qpos[env, 0]` into `META_IDX_PREV_X` and overrides nothing else; it is
    # one of the fourteen that would go silently uncalled if the forward
    # broke. Its own gates need a GPU, so this is the cheapest place that can
    # notice.
    comptime L_HQ = RowMajor[BATCH, 4]
    comptime L_HV = RowMajor[BATCH, 3]
    var hq = TensorImpl[DTYPE].alloc(BATCH * 4)
    var hv = TensorImpl[DTYPE].alloc(BATCH * 3)
    var hm = TensorImpl[DTYPE].alloc(BATCH * METADATA_SIZE)
    for e in range(BATCH):
        for i in range(4):
            hq.data[e * 4 + i] = Scalar[DTYPE](70.0 + Float64(e))
        for i in range(3):
            hv.data[e * 3 + i] = Scalar[DTYPE](0)
        for i in range(METADATA_SIZE):
            hm.data[e * METADATA_SIZE + i] = Scalar[DTYPE](-1234.0)
    for e in range(BATCH):
        HalfCheetahConfig.pre_step_full_gpu[DTYPE, BATCH, 4, 3](
            hq.tt["cpu", BATCH, 4](), hv.tt["cpu", BATCH, 3](),
            hm.tt["cpu", BATCH, METADATA_SIZE](), e,
        )
    var fwd_ok = True
    for e in range(BATCH):
        var got = Float64(hm.data[e * METADATA_SIZE + META_IDX_PREV_X])
        print("     lane", e, "META_IDX_PREV_X =", got, "(want",
              70.0 + Float64(e), ")")
        if got != 70.0 + Float64(e):
            fwd_ok = False
    ta.check(
        fwd_ok,
        "HalfCheetahConfig's narrow pre_step_gpu ran through"
        " pre_step_full_gpu's default forward",
    )
    # ⚠ ANTI-VACUITY: the two lanes get DIFFERENT values, so a forward that
    # ran once and wrote lane 0's answer everywhere is visible.
    ta.check(
        hm.data[0 * METADATA_SIZE + META_IDX_PREV_X]
        != hm.data[1 * METADATA_SIZE + META_IDX_PREV_X],
        "and per lane, not once for the batch",
    )

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "active mask: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")
