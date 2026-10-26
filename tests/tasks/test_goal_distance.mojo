"""THE SHAPED TERM'S DISTANCE — zero exactly when the goal holds.

    pixi run mojo run -I . tests/tasks/test_goal_distance.mojo

## ⚠⚠ WHY A DISTANCE EXISTS

`so101_gather_bricks` paid +1 on success and nothing otherwise. Measured on a
5090: 125k env-steps, ~384 episodes, a rate indistinguishable from random. At
~1.5% success that is about SIX rewarding transitions in 125,000 — 5e-05 of
the replay buffer, which a batch of 256 contains 1.4% of the time. The critic
almost never saw a success. `gpu_eval.tape_distance_gpu` is the quantity a
shaped term needs, derived from the TAPE so it covers the whole goal language
rather than one task.

## WHAT THIS ASSERTS

1. **ZERO IFF THE GOAL HOLDS**, over a sweep of states and every shipped task.
   Two separate switches walk one tape — `eval_tape_gpu` returns a Bool and
   this returns a metre — so an op the distance forgot shows up as a
   disagreement rather than as a term with no gradient. A forgotten op is the
   likely failure and it is SILENT: the reward keeps working, the shaping just
   quietly does nothing for that predicate.

2. **THE 0.5 BOUND**, which three files depend on without saying so.
   `task_batched_gpu.mojo`, `task_eval_frozen.mojo` and this family's own
   success counting all read "solved" as `reward > 0.5`, and shaping is
   SUBTRACTED from that same scalar. If the penalty could reach 0.5 a solved
   lane would report 0.4 and every success counter in the tree would read it
   as a miss — with no error anywhere.

3. **MONOTONE TOWARD THE GOAL** — moving a prop closer must not increase the
   distance. A sign error passes checks 1 and 2 perfectly and trains the
   policy to run away.
"""

from noeira.core.tiles import RowMajor

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.spec import load_family, load_task
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TabletopConfig
from noeira.tasks.so101_tabletop_xml import So101TabletopModel
from noeira.tasks.predicates import parse_goal, bind_goal
from noeira.tasks.eval import (
    region_sites, region_rects, region_half_heights,
)
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import (
    region_table_words, eval_tape_gpu, tape_distance_gpu,
)


comptime DTYPE = DType.float64
comptime BATCH = 1
comptime CFG = So101TabletopConfig
# ⚠ COMPTIME, from the model def — a `Layout` parameter cannot take a runtime
# length, and `len(fmd.body_names)` is one. `test_family_compose_vs_mujoco`
# is what keeps these equal to the composed scene's.
comptime NB = So101TabletopModel.NBODY
comptime NS = So101TabletopModel.NSITE
comptime L_XP = RowMajor[BATCH, NB * 3]
comptime L_XQ = RowMajor[BATCH, NB * 4]
comptime L_SP = RowMajor[BATCH, NS * 3]


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


def main() raises:
    print("=== the shaped term's distance vs the goal it shapes ===")
    var ta = Tally()

    # ── 2. the bound, first — it is a comptime fact and costs nothing ─────
    # ⚠⚠ THE 0.5 PRODUCT CHECK THAT WAS HERE IS GONE WITH THE CONSTRAINT IT
    # ENFORCED. Success used to be read out of the reward (`reward > 0.5`), so
    # the shaping had to stay under it; the bit is `META_IDX_GOAL_HELD` now
    # and a weight is just a weight. What is still worth asserting is that the
    # weights are NON-NEGATIVE — they multiply a `tolerance` that is LARGER
    # nearer the goal, so a negative one pays the policy to stay away and
    # would train toward exactly that.
    print("  shaping weights:", CFG.SHAPE_W_GOAL, "/", CFG.SHAPE_W_REACH,
          " margins:", CFG.GOAL_MARGIN, "/", CFG.REACH_MARGIN)
    ta.check(
        CFG.SHAPE_W_GOAL >= 0.0 and CFG.SHAPE_W_REACH >= 0.0,
        "the shaping weights are non-negative (they weight a tolerance that"
        " REWARDS proximity)",
    )
    # ⚠ AND THAT THE MARGINS STRADDLE THE MEASURED STATE DISTRIBUTION.
    # `task_shaping_probe.mojo` measures goal 0.115-0.139 m and reach
    # 0.120-0.191 m under a random policy. A margin far below that is the
    # `tolerance` version of a clip in the wrong place: the term saturates
    # near zero over the states the policy actually occupies and says nothing.
    ta.check(
        CFG.GOAL_MARGIN >= 0.05 and CFG.GOAL_MARGIN <= 0.40,
        "the goal margin brackets the measured goal distance (0.115-0.139 m)",
    )
    ta.check(
        CFG.REACH_MARGIN >= 0.08 and CFG.REACH_MARGIN <= 0.60,
        "the reach margin brackets the measured reach distance"
        " (0.120-0.191 m)",
    )

    var f = load_family("noeira/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var nb = NB
    var ns = NS

    comptime L_META = RowMajor[BATCH, METADATA_SIZE]
    comptime L_CUR = RowMajor[1, MODEL_CURRICULUM_SIZE]
    var meta = TensorImpl[DTYPE].alloc(BATCH * METADATA_SIZE)
    var cur = TensorImpl[DTYPE].alloc(MODEL_CURRICULUM_SIZE)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for i in range(MODEL_CURRICULUM_SIZE):
        cur.data[i] = Scalar[DTYPE](cw[i])

    var xpos = TensorImpl[DTYPE].alloc(BATCH * nb * 3)
    var xquat = TensorImpl[DTYPE].alloc(BATCH * nb * 4)
    var sxp = TensorImpl[DTYPE].alloc(BATCH * ns * 3)

    var names = List[String]()
    names.append(String("so101_reach_brick"))
    names.append(String("so101_lift_brick"))
    names.append(String("so101_gather_bricks"))
    names.append(String("so101_settle_brick"))

    var brick = -1
    var cube_a = -1
    var table = -1
    for b in range(nb):
        if String(fmd.body_names[b]).startswith("brick_"):
            brick = b
        if String(fmd.body_names[b]).startswith("cube_a_"):
            cube_a = b
        if String(fmd.body_names[b]).startswith("table_"):
            table = b
    var grip = -1
    for si in range(ns):
        if String(fmd.site_names[si]) == "robot_gripperframe":
            grip = si
    ta.check(grip == CFG.GRIPPER_SITE,
             "GRIPPER_SITE is `robot_gripperframe`'s id in the composed scene")

    # ── 1. zero iff it holds, over a sweep ────────────────────────────────
    print()
    print("--- 1. distance == 0 exactly when the goal holds ---")
    var total = 0
    var bad = 0
    var n_hold = 0
    var n_miss = 0

    for n in range(len(names)):
        var t = load_task("noeira/tasks/tasks/" + names[n] + ".task")
        var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
        var tp = encode_goal(g)
        for k in range(TAPE_WORDS):
            meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[DTYPE](tp[k])

        var t_hold = 0
        for ix in range(6):
            for iz in range(6):
                for iy in range(3):
                    for i in range(nb * 3):
                        xpos.data[i] = Scalar[DTYPE](0)
                    for b in range(nb):
                        xquat.data[b * 4] = Scalar[DTYPE](0)
                        xquat.data[b * 4 + 1] = Scalar[DTYPE](0)
                        xquat.data[b * 4 + 2] = Scalar[DTYPE](0)
                        xquat.data[b * 4 + 3] = Scalar[DTYPE](1)
                    for i in range(ns * 3):
                        sxp.data[i] = Scalar[DTYPE](0)

                    var ts = rsites[0]
                    sxp.data[ts * 3] = Scalar[DTYPE](0.25)
                    sxp.data[ts * 3 + 2] = Scalar[DTYPE](0.02)
                    sxp.data[grip * 3] = Scalar[DTYPE](0.10 + Float64(ix) * 0.06)
                    sxp.data[grip * 3 + 2] = Scalar[DTYPE](Float64(iz) * 0.04)

                    # the table where the family composes it
                    xpos.data[table * 3] = Scalar[DTYPE](0.25)
                    xpos.data[table * 3 + 2] = Scalar[DTYPE](0.01)
                    # the brick sweeps; cube_a is pinned so `Near` varies
                    xpos.data[brick * 3] = Scalar[DTYPE](
                        0.10 + Float64(ix) * 0.06
                    )
                    xpos.data[brick * 3 + 1] = Scalar[DTYPE](
                        Float64(iy) * 0.05
                    )
                    xpos.data[brick * 3 + 2] = Scalar[DTYPE](
                        Float64(iz) * 0.04
                    )
                    xpos.data[cube_a * 3] = Scalar[DTYPE](0.25)
                    xpos.data[cube_a * 3 + 2] = Scalar[DTYPE](0.04)

                    var holds = eval_tape_gpu[DTYPE, BATCH, NB, NS * 3](
                        meta.tt["cpu", BATCH, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
                        xpos.tt["cpu", BATCH, NB * 3](), xquat.tt["cpu", BATCH, NB * 4](),
                        sxp.tt["cpu", BATCH, NS * 3](),
                        0,
                    )
                    var d = Float64(
                        tape_distance_gpu[DTYPE, BATCH, NB, NS * 3](
                            meta.tt["cpu", BATCH, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
                            xpos.tt["cpu", BATCH, NB * 3](), xquat.tt["cpu", BATCH, NB * 4](),
                            sxp.tt["cpu", BATCH, NS * 3](),
                            0,
                        )
                    )
                    total += 1
                    if holds:
                        n_hold += 1
                        t_hold += 1
                    else:
                        n_miss += 1
                    # ⚠ `d == 0.0` EXACTLY. The distance is built to be
                    # clamped at zero, not to approach it, so a tolerance
                    # here would hide a predicate whose boundary the distance
                    # puts in a slightly different place.
                    if holds != (d == 0.0):
                        bad += 1
                        if bad <= 5:
                            print("      ", names[n], "holds =", holds,
                                  " distance =", d)
        print("     ", names[n], ":", t_hold, "of", total // (n + 1),
              "swept states hold")

    print("  states compared:", total, " disagreeing:", bad)
    # ⚠⚠ ANTI-VACUITY. "0 disagreeing" is what a sweep in which the goal never
    # holds reports, and also what one where it always holds reports — in both
    # cases the interesting half was never exercised.
    if n_hold == 0 or n_miss == 0:
        raise Error(
            "goal distance: the sweep produced only "
            + ("HOLDS" if n_miss == 0 else "MISSES")
            + ", so `zero iff it holds` was only ever checked on one side."
        )
    ta.check(bad == 0,
             "the distance is zero EXACTLY when the goal holds, on "
             + String(total) + " states with both outcomes present")

    # ── 3. monotone: closer is not farther ────────────────────────────────
    print()
    print("--- 3. moving toward the goal does not increase the distance ---")
    var t_g = load_task("noeira/tasks/tasks/so101_gather_bricks.task")
    var g_g = bind_goal(
        parse_goal(t_g.goal), f, fmd.body_names, fmd.site_names
    )
    var tp_g = encode_goal(g_g)
    for k in range(TAPE_WORDS):
        meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[DTYPE](tp_g[k])
    xpos.data[cube_a * 3] = Scalar[DTYPE](0.25)
    xpos.data[cube_a * 3 + 1] = Scalar[DTYPE](0)
    xpos.data[cube_a * 3 + 2] = Scalar[DTYPE](0.04)
    var prev = 1e9
    var mono = True
    for k in range(12):
        # walk the brick in from 0.24 m away to touching
        xpos.data[brick * 3] = Scalar[DTYPE](0.25 + 0.24 - Float64(k) * 0.02)
        xpos.data[brick * 3 + 1] = Scalar[DTYPE](0)
        xpos.data[brick * 3 + 2] = Scalar[DTYPE](0.04)
        var d = Float64(
            tape_distance_gpu[DTYPE, BATCH, NB, NS * 3](
                meta.tt["cpu", BATCH, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
                xpos.tt["cpu", BATCH, NB * 3](), xquat.tt["cpu", BATCH, NB * 4](),
                sxp.tt["cpu", BATCH, NS * 3](),
                0,
            )
        )
        if d > prev + 1e-12:
            mono = False
        prev = d
    print("      distance at the closest step:", prev)
    ta.check(mono,
             "the distance never INCREASES as the brick approaches cube_a")
    # ⚠ AND IT REACHES ZERO, or "never increases" is satisfied by a constant.
    ta.check(prev == 0.0,
             "and it reaches exactly zero once the goal is met")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "goal distance: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")
