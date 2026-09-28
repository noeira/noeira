"""THE VISION STUDENT, CLOSED LOOP — cube-in-bowl rate with the cameras in the loop.

    pixi run -e nvidia mojo run -I . examples/so101/tower_act_eval.mojo \\
        --ckpt <run_id> --episodes 128
    pixi run -e nvidia mojo run -I . examples/so101/tower_act_eval.mojo \\
        --ckpt runs/<id>/checkpoints --dr full --dr-seed 1000   # held-out looks
    pixi run -e nvidia mojo run -I . examples/so101/tower_act_eval.mojo \\
        --ckpt runs/<id>/checkpoints --record-demo round1.demo   # DAgger states
    pixi run -e apple  mojo run -I . examples/so101/tower_act_eval.mojo --policy hold --steps 60

`LANES` episodes per round on the batched GPU env, lane per episode:

    reset_batch, then each lane's qpos = `posed_qpos(task, seed0 + episode)`
        — the expert recorder's placements (its host sampler), so the student
        and the teacher can be scored on the SAME scenes
    5 settle steps holding the arm where it is (the recorder's settle)
    up to `--steps` policy steps:
        host FK of every lane's qpos -> the rig's tracer, both cameras
        -> uint8 CHW -> `normalize_camera_chw`; the six joints in LeRobot
        units -> the checkpoint's `norm.json`; one ACT forward at batch LANES;
        the chunk (ensembled, or executed open-loop) denormalised -> a joint
        target in LeRobot units -> radians / gripper fraction -> the env's
        action word, clamped to [-1, 1]
    success = the task's goal bit (`META_IDX_GOAL_HELD`) held `HOLD_STEPS`
        consecutive steps — the recorder's rule, so a demo's "success" and
        this one are the same event

## ⚠⚠ SAME PIXELS, SAME UNITS AS THE STORE — BY CONSTRUCTION

The renderer, its visual set and background, the camera slots (overhead 0,
wrist 1), the byte packing and the LeRobot unit map are
`tasks/so101_tower_rig.mojo`, the module `tower_demo_rerender.mojo` renders
the training store with. The frame is host FK of the lane's CURRENT `qpos`
(the env's device poses lag one substep: the tower config does not sync FK
after a step), exactly as the store's frame r is FK of recorded `qpos[r]`.

## Seeds

`--seed0` (default 30000) + episode index. The expert files were recorded at
11000+ and 14000+ (`scripts/so101_tower_vision_box.sh`), 300 episodes each,
so the default block is disjoint from both. A run is a frozen set: the same
flags score the same scenes.

## Where the failures happen

Per failed lane, from the brick's and bowl's body positions after the settle
(`rest` = the brick's height then):

    no grasp        the brick never rose LIFT_DZ above rest
    dropped         lifted, then back to the desk (rest + LAND_DZ) more than
                    NEAR_BOWL from the bowl's centre
    missed bowl     lifted, landed within NEAR_BOWL of the bowl but the goal
                    never held
    goal not held   the goal bit was set at some step but not HOLD_STEPS in a row
    held to end     lifted and still in the air when the steps ran out

## `--joint-zero none|follower|follower-v1` — THE TRAINING STORE'S UNIT MAP

The joint zero the store was rendered with (`tower_demo_rerender.mojo
--joint-zero`, recorded in the store manifest's provenance line; absent =
`none`). The student's degrees mean nothing without it: evaluated under the
other map, every commanded pan is ~10 degrees off and the rate collapses with
nothing raising (`tasks/so101_tower_rig.mojo`).
`follower-v1` is `follower` before 2026-09-25 (roll zero 0, not +5.0): a
`follower` store from before then says so in its provenance degrees, whose
fifth number reads 0.00.

## `--look calibrated|legacy` — THE TRAINING STORE'S LOOK

The rig's lights and colours the store was rendered with
(`tower_demo_rerender.mojo --look`, in its provenance line; a store without
it predates the calibration and is `legacy`). A student evaluated under the
other look sees a different scene than it trained on.

## `--dr off|light|full` — held-out appearance

`physics3d/raytrace/randomize.mojo` on the rig's tables, ONE draw per ROUND
(draw index = round + `--dr-draw0`): every lane of a round shares the look,
successive rounds get new looks. A HELD-OUT appearance set is a seed the
store was not rendered with (`--dr-seed`), from the same ranges; the plan's
Phase 3 gate compares that rate with the rate under the store's own looks.

## `--record-demo FILE` — DAgger's data collector

Every lane's visited transitions (the env's observation, the action the
student EXECUTED, the env's reward, the next observation), one episode per
lane, success-stamped, written as a `.demo` — the file
`tower_expert_record.mojo`'s labelling pass reads. Rows after a lane's
success are not recorded.

## ⚠ THE LOOP LIVES IN THE LIBRARY: `tasks/so101_tower_act_eval.mojo`

This file parses the command line and calls `TowerActEval.run`. It was one
470-line `main` holding the env, the renderer and the ACT trainer as locals
across ~60 raising calls, and did not compile in 62 GB of RAM (its parts
compile in 1-6 GB each). The cause was the ACT trainer held in an
`Optional` (the library module's `act` field says why); the loop moved to the
library at the same time. Do not grow `main` back: add a method there.

## `--policy hold` — the null control

Hold the arm where it is, no network. The rate must be 0: a goal that the
null action meets is a goal defect, not a policy result. It also runs the
whole pipeline (reset, settle, render, report) without a checkpoint.
"""

from std.os.path import exists, isdir
from std.sys import argv, has_accelerator
from max.gpu.host import DeviceContext

from noeira.deep_agents.act.config import ACT_TEMPORAL_ENSEMBLE_M
from noeira.core.run import resolve_checkpoint
from noeira.tasks.so101_tower_rig import RIG_JOINT_ZERO_NONE, RIG_LOOK_CALIBRATED
from noeira.tasks.so101_tower_act_eval import (
    TowerActEval, TowerEvalConfig, TOWER_EVAL_DEFAULT_SEED0,
    TOWER_EVAL_DEFAULT_STEPS,
)


comptime LANES = 32
"""Episodes per round. Comptime: the env, the tracer and ACT are instantiated
at it. 32 x 4 rounds = the 128 the other evals report over."""
comptime DEFAULT_TASK = "so101_tower_cube_in_bowl"
comptime DEFAULT_SEED0 = TOWER_EVAL_DEFAULT_SEED0
comptime DEFAULT_STEPS = TOWER_EVAL_DEFAULT_STEPS


def _usage() -> String:
    return String(
        "usage: tower_act_eval.mojo [--ckpt RUN_ID|DIR|FILE] [--ckpt-name best|last]"
        " [--norm FILE] [--policy act|hold] [--episodes N] [--seed0 S]"
        " [--steps N] [--exec N] [--m M] [--task NAME]"
        " [--dr off|light|full] [--dr-seed N] [--dr-draw0 N]"
        " [--record-demo FILE] [--snap DIR] [--joint-zero none|follower|follower-v1]"
        " [--look calibrated|legacy]"
    )


def main() raises:
    comptime if not has_accelerator():
        print("  SKIPPED: no accelerator — the env, tracer and ACT are device code")
        print("=== SKIPPED (this is not a pass) ===")
        return

    # ── args ──────────────────────────────────────────────────────────────
    var args = argv()
    var ckpt_arg = String("")
    var ckpt_name = String("best")
    var norm_path = String("")
    var policy = String("act")
    var n_episodes = 128
    var seed0 = DEFAULT_SEED0
    var steps = DEFAULT_STEPS
    var exec_n = 0
    var ens_m = ACT_TEMPORAL_ENSEMBLE_M
    var task = String(DEFAULT_TASK)
    var dr_name = String("off")
    var dr_seed = 0
    var dr_draw0 = 0
    var demo_out = String("")
    var snap_dir = String("")
    var joint_zero = String(RIG_JOINT_ZERO_NONE)
    var look = String(RIG_LOOK_CALIBRATED)
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if not a.startswith("--") or i + 1 >= len(args):
            raise Error("bad argument " + a + "\n" + _usage())
        var v = String(args[i + 1])
        if a == "--ckpt":
            ckpt_arg = v
        elif a == "--ckpt-name":
            ckpt_name = v
        elif a == "--norm":
            norm_path = v
        elif a == "--policy":
            policy = v
        elif a == "--episodes":
            n_episodes = Int(v)
        elif a == "--seed0":
            seed0 = Int(v)
        elif a == "--steps":
            steps = Int(v)
        elif a == "--exec":
            exec_n = Int(v)
        elif a == "--m":
            ens_m = Float64(v)
        elif a == "--task":
            task = v
        elif a == "--dr":
            dr_name = v
        elif a == "--dr-seed":
            dr_seed = Int(v)
        elif a == "--dr-draw0":
            dr_draw0 = Int(v)
        elif a == "--record-demo":
            demo_out = v
        elif a == "--snap":
            snap_dir = v
        elif a == "--joint-zero":
            joint_zero = v
        elif a == "--look":
            look = v
        else:
            raise Error("unknown option " + a + "\n" + _usage())
        i += 2
    if policy != "act" and policy != "hold":
        raise Error("--policy is act or hold, not " + policy)
    var use_act = policy == "act"
    var ckpt_path = String("")
    if use_act:
        if ckpt_arg.byte_length() == 0:
            raise Error("--ckpt is required with --policy act\n" + _usage())
        if isdir(ckpt_arg):
            ckpt_path = ckpt_arg + "/" + ckpt_name + ".ckpt"
            if norm_path.byte_length() == 0:
                norm_path = ckpt_arg + "/norm.json"
        elif exists(ckpt_arg):
            ckpt_path = ckpt_arg
        else:
            # a RUN ID: its `checkpoints/<ckpt-name>.ckpt`, with the
            # trainer's `norm.json` beside it
            ckpt_path = resolve_checkpoint(ckpt_arg, ckpt_name)
            if norm_path.byte_length() == 0:
                norm_path = (
                    String(ckpt_path[byte=0 : ckpt_path.rfind("/")]) + "/norm.json"
                )
        if norm_path.byte_length() == 0:
            raise Error("--norm is required when --ckpt names a file")
        for pth in [ckpt_path, norm_path]:
            if not exists(pth):
                raise Error("no such file: " + pth)
    var cfg = TowerEvalConfig()
    cfg.use_act = use_act
    cfg.ckpt_path = ckpt_path
    cfg.norm_path = norm_path
    cfg.n_episodes = n_episodes
    cfg.seed0 = seed0
    cfg.steps = steps
    cfg.exec_n = exec_n
    cfg.ens_m = ens_m
    cfg.task = task
    cfg.dr_name = dr_name
    cfg.dr_seed = dr_seed
    cfg.dr_draw0 = dr_draw0
    cfg.demo_out = demo_out
    cfg.snap_dir = snap_dir
    cfg.joint_zero = joint_zero
    cfg.look = look
    print("=" * 78)
    print("so101_tower — vision student, CLOSED LOOP —", task)
    print("=" * 78)
    print("  policy :", policy, (" " + ckpt_path if use_act else String("")),
          "| exec", ("ensemble m=" + String(ens_m)) if exec_n == 0 else String(exec_n))
    var ev = TowerActEval[LANES](cfg^, DeviceContext())
    print(ev.describe())
    var tally = ev.run()
    tally.report(use_act)
    if demo_out.byte_length() > 0:
        print("  wrote", demo_out)
    tally.check()
    print("=== DONE ===")
