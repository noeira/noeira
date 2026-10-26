"""IS THE GOAL SOLVED BY A CONSTANT ACTION? — CPU, and it is a GATE.

    pixi run mojo run -I . examples/tasks/task_null_action.mojo
    pixi run mojo run -I . examples/tasks/task_null_action.mojo <task> [action]

## ⚠⚠ THE DEGENERACY `test_task_reset_steps` CANNOT SEE

That gate asserts every shipped task's goal is FALSE AT RESET, and every one
of them passes. `so101_reach_brick` passed it — 0 of 8 lanes — and an
UNTRAINED greedy SAC actor then met it on 64 of 64 episodes. "Solved at reset"
and "solved by a constant action" are DIFFERENT degeneracies: the first is
evaluated before any step, the second after the policy's default output has
been applied for a while.

⚠ AND A CONSTANT ACTION IS NOT AN ODD CASE, IT IS WHAT AN UNTRAINED POLICY
EMITS. `tanh(mu)` for a freshly initialised actor is a near-constant vector,
so whatever a constant action achieves is the floor every training curve
starts from — and a task whose floor is 1.0 has nothing above it.

## ⚠⚠ WHAT IT MEASURES, AND WHY THE PER-STEP TRACE IS THE POINT

A goal met on ONE step of a 300-step episode and a goal met on the LAST 200
are the same "success" to a reward that fires per step and an episode that
terminates on the first hit. They are not the same task:

    passed THROUGH     the arm swept the gripper across the region on its way
                       somewhere else. Any large joint motion does this, and
                       no policy had to aim.
    arrived and HELD   the commanded pose is inside the region.

So this prints the first hit, the total hits and the LONGEST RUN of
consecutive hits. A long run at the end of the episode is "arrived"; a short
run in the middle is "swept through", and a task that a sweep satisfies is not
a reaching task — it is a moving test.

⚠ CPU AND SINGLE-ENV ON PURPOSE. The question is about one deterministic
rollout, `Phyics3dEnv` gives exactly that, and it runs in the normal suite
where a GPU example cannot.
"""

from std.sys import argv, has_accelerator


from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TabletopConfig
from noeira.tasks.so101_tabletop_xml import So101TabletopModel
from noeira.tasks.predicates import parse_goal, bind_goal
from noeira.tasks.eval import eval_goal, region_sites
from noeira.tasks.active import active_mask, init_region_words
from noeira.tasks.sampler import (
    sample_placements, RegionFrame, SampleReport,
)
from noeira.tasks.reset import free_slot_addresses, reset_slots
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.core.cont_action import ContAction


comptime DT = DType.float64
comptime FAMILY = "noeira/tasks/families/so101_tabletop.family"
comptime DEFAULT_TASK = "so101_reach_clear"
comptime ACT_DIM = 6
comptime SEED = 11

comptime EnvT = Phyics3dEnv[So101TabletopModel, So101TabletopConfig, DT]


def probe(task_name: String, act_value: Float64, verbose: Bool) raises -> Int:
    """Longest run of consecutive steps meeting the goal, or -1 if never.

    ⚠ THE RETURN IS THE LONGEST RUN AND NOT THE HIT COUNT, because the count
    cannot tell a sweep from an arrival — see the module header.
    """
    if verbose:
        print("=" * 72)
        print("a CONSTANT action of", act_value, "against", task_name)
        print("=" * 72)

    var f = load_family(FAMILY)
    var t = load_task("noeira/tasks/tasks/" + task_name + ".task")
    validate_task_against_family(t, f)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    var tape = encode_goal(g)
    var mask = active_mask(t, f)

    if verbose:
        print("  goal :", t.goal)

    var env = EnvT()
    _ = env.reset()

    # ⚠⚠ THE PROPS ARE PLACED BY THE HOST SAMPLER HERE, NOT BY THE CONFIG.
    # `So101TabletopConfig.init_qpos_gpu` places them on the DEVICE path;
    # the CPU env's reset hook is `custom_reset_cpu`, which this family does
    # not override. Rather than write the placement a SECOND time — the drift
    # `tests/tasks/test_device_placement.mojo` exists to catch — this file
    # does what the eval and the viewer do and calls the host sampler, whose
    # answer that gate proves identical to the device's for the same
    # (seed, lane).
    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    var nq = env.d.dims.get_nq()
    var nv = env.d.dims.get_nv()
    var sp0 = List[Float64]()
    for i in range(len(fmd.site_names) * 3):
        sp0.append(Float64(env.d.site_xpos.data[i]))
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var rs = rsites[i]
        frames.append(RegionFrame(
            sp0[rs * 3], sp0[rs * 3 + 1], sp0[rs * 3 + 2]
        ))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(So101TabletopConfig.SLOT_RADIUS)
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, UInt64(SEED), 0, rep)
    var q0 = List[Float64]()
    for i in range(nq):
        q0.append(Float64(env.d.qpos.data[i]))
    var v0 = List[Float64]()
    for _ in range(nv):
        v0.append(0.0)
    reset_slots(t, f, placed, addrs, q0, v0)
    for i in range(nq):
        env.d.qpos.data[i] = Scalar[DT](q0[i])
    for i in range(nv):
        env.d.qvel.data[i] = Scalar[DT](v0[i])
    if verbose:
        print("  placed :", len(placed), "free slot(s) from the host sampler")

    # ⚠ THE TAPE AND MASK GO INTO `meta` HERE TOO. The CPU env's reward hook
    # is the SAME `custom_reward_gpu`-shaped evaluator reading the same words;
    # without them it would read the previous contents and score a different
    # goal. Written after `reset` because `_reset_state` writes STEP_COUNT and
    # leaves the rest alone — the same property the GPU driver relies on.
    for w in range(TAPE_WORDS):
        env.d.meta.data[META_IDX_TASK_PARAM_0 + w] = Scalar[DT](tape[w])
    env.d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[DT](mask)

    var nb = len(fmd.body_names)
    var ns = len(fmd.site_names)
    var horizon = So101TabletopConfig.MAX_STEPS

    var first_hit = -1
    var hits = 0
    var run = 0
    var best_run = 0
    var best_run_end = -1
    var last_hit = -1

    for step in range(horizon):
        var av = List[Float64]()
        for _ in range(ACT_DIM):
            av.append(act_value)
        _ = env.step(ContAction[ACT_DIM].from_list(av))

        # ⚠ EVALUATED THROUGH `eval.eval_goal`, the HOST evaluator, on the
        # env's own `Data`. Reading the env's returned reward instead would
        # tie this file to whether `TERMINATE_ON_UNHEALTHY` was spelled — and
        # the whole question is about the GOAL, not about the driver.
        var xb = List[Float64]()
        for i in range(nb * 3):
            xb.append(Float64(env.d.xpos.data[i]))
        var xq = List[Float64]()
        for i in range(nb * 4):
            xq.append(Float64(env.d.xquat.data[i]))
        var sp = List[Float64]()
        for i in range(ns * 3):
            sp.append(Float64(env.d.site_xpos.data[i]))

        if eval_goal(g, f, xb, xq, sp, rsites):
            hits += 1
            last_hit = step
            if first_hit < 0:
                first_hit = step
            run += 1
            if run > best_run:
                best_run = run
                best_run_end = step
        else:
            run = 0

    if verbose:
        print()
        print("  horizon        :", horizon, "control steps")
        print("  steps meeting  :", hits)
        print("  first hit      :", first_hit)
        print("  last hit       :", last_hit)
        print("  longest run    :", best_run, "ending at step", best_run_end)
        print()
        if hits == 0:
            print("  a constant action of", act_value, "NEVER meets this goal.")
        elif best_run_end == horizon - 1 and best_run > horizon // 4:
            print("  ARRIVED AND HELD —", best_run, "consecutive steps ending")
            print("  at the horizon: the commanded pose is INSIDE the region.")
        else:
            print("  SWEPT THROUGH — longest run", best_run, "of", horizon,
                  "ending at", best_run_end)
    if hits == 0:
        return -1
    if best_run_end == horizon - 1 and best_run > horizon // 4:
        return horizon        # arrived and held
    return best_run           # swept through


# ── the recorded behaviour of every shipped task ──────────────────────────
#
# ⚠⚠ THESE ARE MEASUREMENTS, AND THE GATE IS THAT THEY HAVE NOT CHANGED. A
# task moving between these categories is a real event in either direction —
# a real task going degenerate, or a degenerate one being fixed — and both
# should fail here rather than pass quietly.
comptime KIND_NEVER: Int = 0
"""No constant action meets the goal. What a real task looks like."""
comptime KIND_ALWAYS: Int = 1
"""Every constant action meets it, at every step. `so101_settle_brick` is
this ON PURPOSE — it is the probe two GPU gates need a true lane for."""
comptime KIND_SWEPT: Int = 2
"""Some constant action meets it TRANSIENTLY — the arm passes across the
region on its way somewhere else. Degenerate, and recorded as such."""


def kind_name(k: Int) -> String:
    if k == KIND_NEVER:
        return String("NEVER (a real task)")
    if k == KIND_ALWAYS:
        return String("ALWAYS (the probe, on purpose)")
    return String("SWEPT THROUGH (degenerate)")


def _run() raises:
    var a = argv()
    if len(a) > 1:
        # single-task diagnosis
        var av = 0.0
        if len(a) > 2:
            av = Float64(String(a[2]))
        _ = probe(String(a[1]), av, True)
        print()
        print("=== MEASURED ===")
        return

    print("=" * 72)
    print("every shipped task against a sweep of CONSTANT actions")
    print("=" * 72)

    var names = List[String]()
    var expect = List[Int]()
    names.append(String("so101_reach_brick")); expect.append(KIND_SWEPT)
    names.append(String("so101_reach_clear")); expect.append(KIND_SWEPT)
    names.append(String("so101_lift_brick")); expect.append(KIND_NEVER)
    names.append(String("so101_gather_bricks")); expect.append(KIND_NEVER)
    names.append(String("so101_settle_brick")); expect.append(KIND_ALWAYS)

    var acts = List[Float64]()
    acts.append(-0.6)
    acts.append(0.0)
    acts.append(0.3)
    acts.append(0.6)

    var failures = 0
    var saw_never = 0
    var saw_other = 0

    for n in range(len(names)):
        var n_never = 0
        var n_always = 0
        var n_swept = 0
        var worst = 0
        for k in range(len(acts)):
            var r = probe(names[n], acts[k], False)
            if r < 0:
                n_never += 1
            elif r >= 300:
                n_always += 1
            else:
                n_swept += 1
                if r > worst:
                    worst = r
        var got = KIND_NEVER
        if n_always == len(acts):
            got = KIND_ALWAYS
        elif n_swept > 0:
            got = KIND_SWEPT
        elif n_always > 0:
            got = KIND_SWEPT      # held for some actions, not all
        if got == KIND_NEVER:
            saw_never += 1
        else:
            saw_other += 1
        var mark = "ok  " if got == expect[n] else "FAIL"
        print("  " + mark, names[n], "->", kind_name(got),
              " (never", n_never, "/ swept", n_swept, "/ held", n_always,
              "; longest transient", worst, ")")
        if got != expect[n]:
            failures += 1
            print("       expected", kind_name(expect[n]))

    print()
    # ⚠⚠ ANTI-VACUITY, BOTH WAYS. A probe that always returned -1 reports
    # every task as NEVER and passes three of five; one that always returned
    # 300 reports every task as ALWAYS. The corpus has both, so both must
    # appear or the sweep is not measuring anything.
    if saw_never == 0 or saw_other == 0:
        raise Error(
            "null action: every task came back the SAME kind (" 
            + String(saw_never) + " never, " + String(saw_other) + " other)."
            " The corpus contains real tasks AND a probe that is true at every"
            " step, so one of each must appear — a uniform answer means the"
            " rollout or the evaluator is not running."
        )
    if failures != 0:
        raise Error(
            "null action: " + String(failures) + " task(s) changed category."
            " A real task becoming solvable by a constant action, or a"
            " degenerate one becoming a task, are both real events — update"
            " the expectation ON PURPOSE, with the measurement."
        )
    print("  ok: every shipped task behaves as recorded")
    print()
    print("  ⚠ `so101_reach_brick` and `so101_reach_clear` are recorded as")
    print("  SWEPT ON PURPOSE. `AtRegion` over a CONTROLLED end effector, with")
    print("  a per-step reward and first-hit termination, asks 'did the")
    print("  gripper ever pass through here' — which most large joint motions")
    print("  satisfy without aiming. A predicate over an OBJECT's pose")
    print("  (`Above`, `Near`, `On`) is not that shape: a sweep does not lift")
    print("  a brick. That is why the SAC example trains `lift`.")
    print()
    print("=== PASS ===")


def main() raises:
    # `Phyics3dEnv()` opens a DeviceContext for its model upload, so there is
    # nothing to run on a host with no accelerator (the CI runner).
    comptime if has_accelerator():
        _run()
    else:
        print(
            "task_null_action: SKIPPED (no accelerator;"
            " run with -e apple / -e nvidia)"
        )
