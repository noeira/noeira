"""1024 lanes, TWO DIFFERENT TASKS, one binary — P3's gate.

    pixi run -e nvidia mojo run -I . examples/tasks/task_batched_gpu.mojo

⚠⚠ RUN THIS ON NVIDIA. The family is `nv = 24`, and that is exactly where the
P0 park probe died on Metal — `k=3` is nv 24 and fails at pipeline creation
with "Compute function exceeds available stack space", because the physics
kernels stack-allocate per-thread arrays sized by `nv`. It COMPILES on Apple
and cannot launch, so an Apple run proves only that it builds.

## WHAT IT ASSERTS, AND THE SECOND ONE IS THE POINT

1. **PARITY** — every lane's GPU reward equals the CPU evaluation of that
   lane's downloaded state, through `tasks/eval.eval_goal`. Two evaluators,
   two containers, one set of predicates.
2. ⚠⚠ **THE NEGATIVE LEG** — lanes running task A and lanes running task B do
   NOT all agree. A tape misindexed by lane hands every lane the SAME goal,
   and check 1 still passes: the CPU leg reads the same misindexed tape and
   agrees with itself. Only a difference BETWEEN the task groups catches it.
   `tests/tasks/test_tape_gpu_parity.mojo` caught exactly that at BATCH=2 by
   ablation (50 agreements -> 35); this is the same shape at scale.

## HOW A LANE GETS ITS TASK

Even lanes run `settle`, odd lanes run `lift`. The tape is written into
`meta[env, META_IDX_TASK_PARAM_0 .. _11]` once, before the loop — reset
preserves it (`gpu/constants.mojo:164`) — and the region table into
`curriculum[0, 0..4]`, which is shared because a region belongs to the FAMILY.

⚠ NEITHER NEEDED A NEW KERNEL OPERAND. Gap E scoped this as two; both channels
already existed and were unused.

## AND GAP D — A PARKED SLOT THAT STAYS PARKED

`So101TabletopConfig.pre_step_full_gpu` pins every inactive free slot at its
park pose, pose and velocity, before every step. `tests/tasks/
test_active_mask.mojo` gates the arithmetic of one call; what only a stepped
batch can say is whether it holds after real physics, and that is the check at
the end of this file. Gravity is a `Model` field shared by the batch, so
before Gap D a parked body fell 7.06 m over a full horizon.

## AND THE ACTIVE MASK, IN THE OBSERVATION

`meta[env, META_IDX_TASK_ACTIVE]` carries which slots this lane is running
(§3.4), and `So101TabletopConfig.custom_extract_obs_gpu` turns it into the
observation's last `N_FREE_SLOTS` words. The two tasks here have the SAME
active set — `settle` runs `cube_a`, `lift` does too, so a third task would
be needed to separate them on that axis; what separates them here is the mask
WORD, which is read per lane from `meta`.

⚠ `tests/tasks/test_active_mask.mojo` gates the hook's arithmetic on the CPU
at float64. What only THIS can say is that the word survives the round trip
through a real device buffer, a real reset and eight real steps — the mask is
written once, before the loop, and `_reset_env_lane` is what has to preserve
it.
"""

from std.random import seed as seed_rng
from noeira.core.tiles import RowMajor
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.envs.dm_control.rewards import (
    tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN,
)
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_GOAL_HELD, META_IDX_SHAPE_W_GOAL,
    MODEL_CURRICULUM_SIZE,
)
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.physics3d.parser.runtime_load import parse_model_runtime

from noeira.tasks.spec import load_family, load_task, validate_task_against_family
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TabletopConfig, So101TabletopPlacement
from noeira.tasks.so101_tabletop_xml import So101TabletopModel
from noeira.tasks.predicates import parse_goal, bind_goal, require_tier_a
from noeira.tasks.eval import (
    eval_goal, region_sites, region_rects, region_half_heights,
)
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import (
    region_table_words, require_gpu_regions, tape_distance_gpu,
)
from noeira.tasks.sampler import sample_placements, RegionFrame, SampleReport
from noeira.tasks.reset import free_slot_addresses, reset_slots
from noeira.tasks.active import active_mask
from noeira.tasks.shaping import shaping_words, SHAPING_WORDS


comptime N_ENVS = 1024
comptime STEPS = 8
comptime SEED = UInt64(11)

comptime EnvT = Phyics3dBatchedEnv[
    So101TabletopModel, So101TabletopConfig, N_ENVS
]
comptime NB = So101TabletopModel.NBODY
comptime NS = So101TabletopModel.NSITE
comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV


def parked_drift(
    mut env: EnvT, ctx: DeviceContext, pk_z: Float64
) raises -> Tuple[Float64, Float64]:
    """`(max |dz|, max |v|)` of the PARKED slot across the batch.

    ⚠ A FUNCTION BECAUSE IT IS CALLED TWICE, at two step counts — see the
    caller. A nested def cannot capture `env` mutably.
    """
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    ctx.synchronize()
    var mdz = 0.0
    var mv = 0.0
    for e in range(N_ENVS):
        var dz = Float64(
            env.d.qpos.data[e * NQ + So101TabletopPlacement.FREE_QADR_2 + 2]
        ) - pk_z
        if dz < 0.0:
            dz = -dz
        if dz > mdz:
            mdz = dz
        for k in range(6):
            var v = Float64(
                env.d.qvel.data[
                    e * NV + So101TabletopPlacement.FREE_DADR_2 + k
                ]
            )
            if v < 0.0:
                v = -v
            if v > mv:
                mv = v
    return (mdz, mv)


def main() raises:
    seed_rng(0)
    print("=" * 68)
    print("P3 — 1024 lanes, two tasks, one monomorphisation")
    print("=" * 68)

    var f = load_family("noeira/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)

    # ⚠⚠ `settle`, NOT `gather`, AND THE NEGATIVE LEG BELOW IS WHY. That leg
    # demands the two task groups score DIFFERENTLY, so one of them has to be
    # TRUE under the untrained (zero-torque) policy this file runs. `gather`
    # used to be that lane by accident — its goal held at reset — and fixing it
    # to be a real task would have left both groups at 0 and made the leg
    # vacuous. `so101_settle_brick.task` is that lane on purpose and says so in
    # its own header.
    var ta = load_task("noeira/tasks/tasks/so101_settle_brick.task")
    var tb = load_task("noeira/tasks/tasks/so101_lift_brick.task")
    validate_task_against_family(ta, f)
    validate_task_against_family(tb, f)
    var ga = bind_goal(parse_goal(ta.goal), f, fmd.body_names, fmd.site_names)
    var gb = bind_goal(parse_goal(tb.goal), f, fmd.body_names, fmd.site_names)
    require_tier_a(ga, ta.name)
    require_tier_a(gb, tb.name)
    # ⚠ THE REGION COUNTERPART OF `require_tier_a`. The device reads ONE
    # region table and ignores a term's region index; this family declares
    # three regions (two of them for `init=` only), so a goal that named one
    # of the other two would disagree with the CPU leg below — silently, and
    # in the reward. See `gpu_eval.require_gpu_regions`.
    require_gpu_regions(ga, ta.name)
    require_gpu_regions(gb, tb.name)
    var tpa = encode_goal(ga)
    var tpb = encode_goal(gb)
    var mka = active_mask(ta, f)
    var mkb = active_mask(tb, f)
    # ⚠ DELIBERATELY UNEQUAL. If both groups shared one pair, the per-lane
    # read would be indistinguishable from a global one.
    var swa = shaping_words(1.0, 0.25, 0.10, 0.20)
    var swb = shaping_words(0.40, 0.60, 0.05, 0.30)
    print("  even lanes:", ta.name, "|", ta.goal)
    print("  odd  lanes:", tb.name, "|", tb.goal)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    with DeviceContext() as ctx:
        var env = EnvT(ctx)

        # ── the region table, once ────────────────────────────────────────
        # ⚠ THE HALF-HEIGHT IS THE FIFTH NUMBER AND IT IS REQUIRED. Without
        # it the device would use `IN_HALF_HEIGHT` while `eval.eval_goal` used
        # the region's own band — a CPU/GPU disagreement inside the reward.
        var cw = region_table_words(
            rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
            rheights[0],
        )
        for i in range(MODEL_CURRICULUM_SIZE):
            env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        env.mf.curriculum.upload(ctx)

        env.reset_batch[N_ENVS](ctx, SEED)
        env.d.qpos.download(ctx)
        env.d.site_xpos.download(ctx)
        # ⚠ SYNC BEFORE READING `.data`. `download` ENQUEUES a copy; the host
        # array is not valid until the stream drains. Reading it straight away
        # is a stale read, not a crash — the region frames would come from
        # whatever the buffer held before the reset, and every lane would be
        # sampled around the wrong site.
        ctx.synchronize()

        # ── per lane: its task's tape, and its sampled placements ─────────
        var frames = List[RegionFrame]()
        for i in range(len(f.regions)):
            var s = rsites[i]
            frames.append(RegionFrame(
                Float64(env.d.site_xpos.data[s * 3]),
                Float64(env.d.site_xpos.data[s * 3 + 1]),
                Float64(env.d.site_xpos.data[s * 3 + 2]),
            ))
        var radii = List[Float64]()
        for _ in range(len(f.slots)):
            radii.append(0.02)

        var rep = SampleReport()
        for e in range(N_ENVS):
            var is_a = (e % 2) == 0
            for k in range(TAPE_WORDS):
                env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = (
                    Scalar[DT](tpa[k]) if is_a else Scalar[DT](tpb[k])
                )
            # ⚠ THE WHOLE WORD, EVERY EPISODE. `meta` is not zeroed between
            # episodes — that is what lets this be written once and read every
            # step — so a lane keeps the PREVIOUS episode's mask unless it is
            # rewritten. Same trap `OP_NONE` exists for in the tape above.
            env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = (
                Scalar[DT](mka) if is_a else Scalar[DT](mkb)
            )
            # ⚠⚠ THE TWO TASK GROUPS GET DIFFERENT SHAPING ON PURPOSE. These
            # words are per-lane precisely so a batch can run tasks whose
            # distance scales differ — `settle`'s goal distance is 0 at reset
            # and `lift`'s is a 0.03 m z-shortfall, so one weight pair cannot
            # serve both. Giving them DIFFERENT weights here is what makes the
            # shaped-value check below a test of per-lane shaping rather than
            # of a constant.
            for j in range(SHAPING_WORDS):
                env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_SHAPE_W_GOAL + j
                ] = Scalar[DT](swa[j]) if is_a else Scalar[DT](swb[j])
            var qpos = List[Float64]()
            for i in range(NQ):
                qpos.append(Float64(env.d.qpos.data[e * NQ + i]))
            var qvel = List[Float64]()
            for _ in range(NV):
                qvel.append(0.0)
            # ⚠ BRANCHED, NOT `ta if is_a else tb`. `TaskSpec` owns Lists and
            # is Movable but NOT ImplicitlyCopyable, so a ternary would try to
            # copy one per lane — 1024 copies of two Lists, and a compile
            # error rather than a slow loop, which is the better outcome.
            if is_a:
                var pl = sample_placements(ta, f, frames, radii, SEED, e, rep)
                reset_slots(ta, f, pl, addrs, qpos, qvel)
            else:
                var pl = sample_placements(tb, f, frames, radii, SEED, e, rep)
                reset_slots(tb, f, pl, addrs, qpos, qvel)
            for i in range(NQ):
                env.d.qpos.data[e * NQ + i] = Scalar[DT](qpos[i])
            for i in range(NV):
                env.d.qvel.data[e * NV + i] = Scalar[DT](qvel[i])
        env.d.meta.upload(ctx)
        env.d.qpos.upload(ctx)
        env.d.qvel.upload(ctx)
        print("  sampler:", rep.accepted, "placements in", rep.attempts,
              "draws across", N_ENVS, "lanes")

        # ── step, then compare ────────────────────────────────────────────
        for _ in range(STEPS):
            env.step_batch[N_ENVS](ctx, UInt64(0))
        ctx.synchronize()

        env.d.xpos.download(ctx)
        env.d.xquat.download(ctx)
        env.d.site_xpos.download(ctx)
        # ⚠ `meta` TOO, because the goal bit is written INTO it by the reward
        # hook on device now. Reading the host copy without this compares
        # against whatever was uploaded before the loop — which is the tape
        # and the mask, and a goal-held word of zero for every lane.
        env.d.meta.download(ctx)
        ctx.synchronize()

        # ⚠⚠ `reward_ptr()` IS A **DEVICE** POINTER ON THIS ENV.
        # `Phyics3dBatchedEnv._reward` is a `DeviceBuffer[DT]`
        # (`phyics3d_batched_env.mojo:266`) and `reward_ptr` hands back
        # `mptr(self._reward.unsafe_ptr())`. Dereferencing it on the host is a
        # segfault, not a wrong number — which is exactly what the first 5090
        # run did, crashing after the sampler with a bare libc backtrace.
        #
        # ⚠ THE ABI IS THE TRAP. `BatchedEnv.reward_ptr` is documented as
        # something a driver reads "in place", and the CPU-backed
        # implementations in `training/batched_env.mojo` return a pointer into
        # a host `List`. Same signature, two residencies; only this one needs
        # a copy. A host caller must never assume which it has.
        var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        ctx.enqueue_copy(rew_h, env._reward)
        ctx.synchronize()
        var rew = rew_h.unsafe_ptr()

        var mismatch = 0
        var shaped_bad = 0
        var shaped_worst = 0.0
        comptime L_META_H = RowMajor[N_ENVS, METADATA_SIZE]
        comptime L_CUR_H = RowMajor[1, MODEL_CURRICULUM_SIZE]
        comptime L_XP_H = RowMajor[N_ENVS, NB * 3]
        comptime L_XQ_H = RowMajor[N_ENVS, NB * 4]
        comptime L_SP_H = RowMajor[N_ENVS, NS * 3]
        var a_true = 0
        var b_true = 0
        var first_bad = -1
        for e in range(N_ENVS):
            # ⚠ THE CPU LEG READS THE STATE THE GPU PRODUCED, at the same
            # float32 values — widened, not recomputed. Recomputing FK on the
            # host would compare two physics runs, not two evaluators.
            var xb = List[Float64]()
            for i in range(NB * 3):
                xb.append(Float64(env.d.xpos.data[e * NB * 3 + i]))
            var xq = List[Float64]()
            for i in range(NB * 4):
                xq.append(Float64(env.d.xquat.data[e * NB * 4 + i]))
            var sp = List[Float64]()
            for i in range(NS * 3):
                sp.append(Float64(env.d.site_xpos.data[e * NS * 3 + i]))
            var is_a = (e % 2) == 0
            var host = eval_goal(ga if is_a else gb, f, xb, xq, sp, rsites)
            # ⚠⚠ THE GOAL BIT COMES FROM `META_IDX_GOAL_HELD`, NOT FROM THE
            # REWARD. The reward is two `tolerance` terms now and carries no
            # success bonus, so `reward > 0.5` is a statement about DISTANCE
            # and not about the goal — it would read a lane hovering near the
            # blocks as solved.
            var gpu = Float64(
                env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]
            ) > 0.5
            if host != gpu:
                mismatch += 1

            # ⚠⚠ AND THE SHAPED VALUE, NOT JUST THE GOAL BIT. This compared
            # only `reward > 0.5` against `eval_goal`, so the two SHAPED terms
            # the reward subtracts had never been checked against a host
            # computation at all — and a run whose per-step cost was 3.4x
            # larger than any legal weight pair can produce went unexplained
            # for a round because nothing could say what the device actually
            # charged.
            #
            # The host recomputes it from the SAME downloaded state, through
            # the same `tape_distance_gpu` the kernel uses, plus the reach
            # term spelled out here. A disagreement is a defect in the reward,
            # which is the one quantity every curve in every run is made of.
            var d_h = Float64(
                tape_distance_gpu[DT, N_ENVS, NB, NS * 3](
                    env.d.meta.tt["cpu", N_ENVS, METADATA_SIZE](),
                    env.mf.curriculum.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
                    env.d.xpos.tt["cpu", N_ENVS, NB * 3](),
                    env.d.xquat.tt["cpu", N_ENVS, NB * 4](),
                    env.d.site_xpos.tt["cpu", N_ENVS, NS * 3](),
                    e,
                )
            )
            comptime GS = So101TabletopConfig.GRIPPER_SITE
            var sb = Int(tpa[1]) if is_a else Int(tpb[1])
            var rx = sp[GS * 3] - xb[sb * 3]
            var ry = sp[GS * 3 + 1] - xb[sb * 3 + 1]
            var rz = sp[GS * 3 + 2] - xb[sb * 3 + 2]
            var reach_h = (rx * rx + ry * ry + rz * rz) ** 0.5
            # ⚠ EACH LANE'S OWN FOUR WORDS. Using the config's constants here
            # would agree with the kernel only while every lane shares them,
            # which is the case this gate exists to break.
            var wg = swa[0] if is_a else swb[0]
            var wr = swa[1] if is_a else swb[1]
            var mg_ = swa[2] if is_a else swb[2]
            var mr_ = swa[3] if is_a else swb[3]
            var want = wg * Float64(
                tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
                    Scalar[DT](d_h), Scalar[DT](0),
                    Scalar[DT](So101TabletopConfig.GOAL_RADIUS),
                    Scalar[DT](mg_),
                )
            ) + wr * Float64(
                tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
                    Scalar[DT](reach_h), Scalar[DT](0),
                    Scalar[DT](So101TabletopConfig.REACH_RADIUS),
                    Scalar[DT](mr_),
                )
            )
            var err = Float64(rew[e]) - want
            if err < 0.0:
                err = -err
            if err > shaped_worst:
                shaped_worst = err
            if err > 1e-6:
                shaped_bad += 1
                if shaped_bad <= 3:
                    print("      lane", e, "reward", Float64(rew[e]),
                          "but the host computes", want,
                          " (goal_d", d_h, " reach_d", reach_h, ")")
                if first_bad < 0:
                    first_bad = e
            if is_a and gpu:
                a_true += 1
            if (not is_a) and gpu:
                b_true += 1

        print()
        print("  lanes on", ta.name, "meeting their goal:", a_true, "/", N_ENVS // 2)
        print("  lanes on", tb.name, "meeting their goal:", b_true, "/", N_ENVS // 2)
        print("  GPU-vs-CPU mismatches:", mismatch, "of", N_ENVS)

        if mismatch != 0:
            raise Error(
                "P3: " + String(mismatch) + " lanes disagree, first lane "
                + String(first_bad) + ". The device evaluator and the host one"
                + " read the same state and must agree."
            )
        print("  ok: every lane's GPU reward equals the CPU evaluation")

        print("  shaped-value mismatches:", shaped_bad, "of", N_ENVS,
              " worst |error|:", shaped_worst)
        if shaped_bad != 0:
            raise Error(
                "P3: " + String(shaped_bad) + " lanes' GPU reward differs from"
                " the host's recomputation of the SHAPED value (worst "
                + String(shaped_worst) + "). The goal BIT agrees, so this is"
                " the two shaped terms — `SHAPE_W_GOAL * goal_distance` and"
                " `SHAPE_W_REACH * gripper_to_subject` — and the reward is the"
                " one quantity every curve in every run is made of."
            )
        print("  ok: and the SHAPED value agrees to", shaped_worst)

        # ⚠⚠ THE NEGATIVE LEG. Without it a tape misindexed by lane passes
        # everything above — the CPU leg reads the same misindexed tape and
        # agrees with itself. Only a DIFFERENCE between the task groups shows
        # that each lane ran ITS OWN goal.
        if a_true == b_true:
            raise Error(
                "P3: both task groups scored identically (" + String(a_true)
                + "). Either the two goals happen to coincide on this state —"
                + " change the seed — or EVERY LANE IS RUNNING THE SAME TAPE,"
                + " which the parity check above cannot see."
            )
        print("  ok: the two task groups scored DIFFERENTLY —",
              a_true, "vs", b_true, "— so each lane ran its own goal")

        # ── the active mask, through the real device path ─────────────────
        #
        # ⚠ WHAT THIS ADDS OVER `tests/tasks/test_active_mask.mojo`, WHICH
        # ALREADY GATES THE HOOK'S ARITHMETIC ON THE CPU: the word was written
        # into a DEVICE buffer, survived `reset_batch`'s `_reset_env_lane` and
        # eight steps, and came back out of the observation kernel. That chain
        # is what the CPU gate cannot exercise, and it is the half that would
        # break if the reset ever started zeroing `meta`.
        #
        # ⚠⚠ THESE TWO TASKS SHARE AN ACTIVE SET — `settle` and `lift` both
        # run table/brick/cube_a — so the mask does NOT vary by lane here and
        # this cannot catch a hook that reads lane 0's word for everyone.
        # `test_active_mask` runs `reach` beside `gather` for exactly that, at
        # BATCH=2. Stated rather than left for a reader to discover, because a
        # per-lane check that cannot vary is the shape of a vacuous gate.
        var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * EnvT.OBS_DIM)
        ctx.enqueue_copy(obs_h, env._obs)
        ctx.synchronize()
        var ob = obs_h.unsafe_ptr()

        comptime MB = So101TabletopConfig.OBS_MASK_BASE
        comptime NF = So101TabletopConfig.N_FREE_SLOTS
        var mask_bad = 0
        var lit = 0
        var off = 0
        for e in range(N_ENVS):
            var want = mka if (e % 2) == 0 else mkb
            for j in range(NF):
                var si = (
                    So101TabletopPlacement.FREE_SLOT_IDX_0 if j == 0
                    else (So101TabletopPlacement.FREE_SLOT_IDX_1 if j == 1
                          else So101TabletopPlacement.FREE_SLOT_IDX_2)
                )
                var on = ((Int(want) >> si) & 1) == 1
                var got = Float64(ob[e * EnvT.OBS_DIM + MB + j])
                if got != (1.0 if on else 0.0):
                    mask_bad += 1
                if on:
                    lit += 1
                else:
                    off += 1

        print()
        print("  active words:", lit, "on,", off, "off, across",
              N_ENVS * NF, "lane-slots")
        if mask_bad != 0:
            raise Error(
                "P3: " + String(mask_bad) + " active words in the observation"
                " disagree with the mask the host wrote. The word is at"
                " `meta[env, META_IDX_TASK_ACTIVE]`; if it reads 0 everywhere,"
                " something in the reset path now ZEROES `meta` and the tape"
                " beside it is next."
            )
        # ⚠ ANTI-VACUITY. "Every active word matches" is also true when every
        # slot is active and every word is 1 — and `cube_b` is parked in all
        # three shipped tasks, so a 0 must appear or the comparison above was
        # against a constant.
        if lit == 0 or off == 0:
            raise Error(
                "P3: the observation's active words are all "
                + ("0" if lit == 0 else "1") + ". A mask that never varies"
                " cannot be told from a hook that writes a constant."
            )
        print("  ok: every lane's active words are the mask the host wrote,"
              " and BOTH values occur")

        # ── Gap D: a parked slot did not move ─────────────────────────────
        #
        # ⚠ THE ONE THING ONLY A STEPPED BATCH CAN SAY. `test_active_mask`
        # gates the repark's arithmetic on one call; this is whether it holds
        # after real physics — gravity is a `Model` field shared by the batch,
        # so before Gap D a parked body FELL, 7.06 m over a full horizon.
        # ⚠⚠ THE CLAIM IS NON-ACCUMULATION, NOT EXACTNESS, AND THIS CHECK
        # USED TO DEMAND EXACTNESS. It asserted `dz != 0.0` for every lane and
        # failed on the 5090 with **max |dz| = 7.62939453125e-05 m** — which
        # is not drift, it is arithmetic:
        #
        #   one control step of free fall = 0.5 * 9.81 * (0.002 * 2)^2
        #                                 = 7.848e-05 m
        #   stored in float32 at z = 50    ulp = 2^(5-23) = 3.815e-06
        #   7.848e-05 / 3.815e-06          = 20.6 ulps -> 20 ulps
        #   20 * 3.815e-06                 = 7.62939453125e-05    <- measured
        #
        # and the velocity says the same thing with no rounding in the way:
        # one control step's dv is 9.81 * 0.002 * 2 = 0.03924 m/s, and the
        # measurement is 0.03924000263214111. Two independent numbers, both
        # exactly one step.
        #
        # `pre_step_full_gpu` pins the slot BEFORE the step, so any state read
        # AFTER a step necessarily shows one step of fall. An exact check can
        # therefore never pass, and the version that did pass had simply never
        # run on a device with gravity — this is the first GPU run since Gap D
        # landed.
        #
        # ⚠ SO THE GATE IS A RATE TEST, AND IT NEEDS NO PHYSICAL CONSTANT.
        # Pinned, the drift is one step's fall at ANY step count. Unpinned it
        # is quadratic in the step count. ABLATED by skipping the pin: 8
        # steps gives 5.02e-03 m and 32 steps 8.04e-02 m, a ratio of exactly
        # 16.0, with |v| going 0.314 -> 1.256 m/s. Measuring at both and
        # demanding the
        # SECOND is not materially larger separates the two by 8x with nothing
        # tuned. The original symptom Gap D fixed — 7.06 m over a horizon — is
        # the unpinned branch and fails this by five orders of magnitude.
        var pk_z = f.park_z
        var d1 = parked_drift(env, ctx, pk_z)
        for _ in range(STEPS * 3):
            env.step_batch[N_ENVS](ctx, UInt64(0))
        var d2 = parked_drift(env, ctx, pk_z)

        print()
        print("  parked cube_b  |dz| after", STEPS, "steps:", d1[0],
              "m   after", STEPS * 4, "steps:", d2[0], "m")
        print("  parked cube_b  |v|  after", STEPS, "steps:", d1[1],
              "m/s after", STEPS * 4, "steps:", d2[1], "m/s")

        # ⚠ THE ABSOLUTE BOUND IS A SANITY RAIL, NOT THE CHECK. It catches
        # "pinned, but to the wrong place" — a millimetre is already 13x the
        # one-step fall and 5e4 times below the unpinned value at 32 steps.
        if d1[0] > 1e-3 or d2[0] > 1e-3:
            raise Error(
                "P3/Gap D: the parked slot is " + String(d2[0]) + " m from"
                " z=" + String(pk_z) + ", past the 1 mm rail. One control"
                " step of free fall is 7.8e-05 m, so this is not the"
                " pin-before-step lag — `pre_step_full_gpu` is pinning to the"
                " wrong place, or not at all."
            )
        # ⚠⚠ AND THIS IS THE CHECK. Quadratic growth is what an unpinned body
        # does; a factor of 2 is generous against the 16 it would show.
        if d2[0] > d1[0] * 2.0 + 1e-9:
            raise Error(
                "P3/Gap D: the parked slot's offset GREW from "
                + String(d1[0]) + " m at " + String(STEPS) + " steps to "
                + String(d2[0]) + " m at " + String(STEPS * 4) + ". Pinned,"
                " it is one step's fall at any step count; growing with the"
                " step count is what a body nobody pins does — the env stopped"
                " calling the wide pre-step hook, or the active mask says"
                " cube_b is ACTIVE."
            )
        if d2[1] > d1[1] * 2.0 + 1e-9:
            raise Error(
                "P3/Gap D: the parked slot's speed GREW from " + String(d1[1])
                + " to " + String(d2[1]) + " m/s. `pre_step_full_gpu` zeroes"
                " qvel every step, so one step's acceleration is the ceiling."
            )
        print("  ok: the parked slot's offset and speed do NOT grow with the"
              " step count — it is pinned, and what is left is the"
              " pin-before-step lag")

        # ⚠⚠ ANTI-VACUITY, AND IT MATTERS MORE NOW THAT THE CHECK ABOVE IS A
        # RATIO. "The offset did not grow" is also true of a batch that never
        # stepped, of a scene with no gravity, and of a repark that pinned
        # EVERY slot — all three make BOTH measurements equal and pass. An
        # ACTIVE prop must be somewhere else entirely.
        comptime BR_QADR = So101TabletopPlacement.FREE_QADR_0
        comptime CB_QADR = So101TabletopPlacement.FREE_QADR_2
        var active_moved = 0
        for e in range(N_ENVS):
            if Float64(env.d.qpos.data[e * NQ + BR_QADR + 2]) != Float64(
                env.d.qpos.data[e * NQ + CB_QADR + 2]
            ):
                active_moved += 1
        if active_moved == 0:
            raise Error(
                "P3/Gap D: the ACTIVE brick sits at the parked slot's height"
                " in every lane, so 'the parked one did not move' says"
                " nothing — either the repark pinned everything, or the batch"
                " never stepped."
            )
        print("  ok:", active_moved, "lanes' ACTIVE brick is nowhere near the"
              " parked slot — the batch really stepped and gravity is on")
        print()
        print("=== PASS ===")
