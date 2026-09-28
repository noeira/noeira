"""TD-MPC2 on dm_control `walker`, N_ENVS BATCHED (GPU env + GPU nets).

The batched counterpart of `tdmpc2_dm_walker_gpu.mojo`. Same agent, same task,
same reward — the difference is that the env is `Phyics3dBatchedEnv` (N walkers
stepped in one kernel grid, the same env SAC uses here) and the driver is
`agent.train_batched`, which did not exist until now.

Why it matters: the single-env script measured ~2.9 env-steps/s on Apple with
MPC, because one CPU-stepped walker at a time cannot fill a GPU. Batching moves
three things onto the batch axis at once:

  * the physics — one `step_batch` for all N envs;
  * acting — ONE `encoder → policy → rsample` pass over [N, ·], or ONE
    `plan_gpu` whose grid is N × (MPC_SAMPLES + MPC_PI_TRAJS) MPPI candidates;
  * the host sync — one D2H per ITERATION (N env-steps) instead of per step.

`train_step` is untouched: it samples B windows from replay, which never cared
how the data was collected.

## The replay subtlety this driver handles for you

N envs write into one sequence ring round-robin, so slot p and p+1 are
DIFFERENT envs while p and p+N are consecutive frames of the SAME env. A
contiguous window would be a world model trained on transitions that never
happened — and it would still train, and the loss would still fall. The driver
calls `replay.set_env_stride(N_ENVS)` so the sampler walks lanes of stride N.
`tests/deep_agents/test_tdmpc2_batched_smoke.mojo` asserts that windows are
single-env directly, because the loss curve cannot.

## What has actually been measured (2026-08-11)

Apple/Metal, 4 envs, MPC on, B=64: the loop runs end to end and the WM loss
falls (0.27 → 0.15 over 400 iterations). That is a FUNCTIONAL result, not a
throughput one — there is no clean apples-to-apples speedup number yet, because
the single-env baseline runs one gradient step per env-step while the batched
run does one per ITERATION. Benchmark this on NVIDIA (the standing rule for
this repo: bench on NVIDIA, Apple is for parity), holding
`updates_per_step / N_ENVS` fixed between the two, before quoting any factor.

Expect the win to come from collection and acting, not from `train_step` —
that one does identical work either way, and on Metal it dominates.

## Sizing

`N_ENVS` multiplies the MPPI grid: BATCH_TOTAL = N_ENVS × (MPC_SAMPLES +
MPC_PI_TRAJS) rows through dynamics/reward/Q per horizon step per iteration.
At the defaults below that is 8 × 268 = 2144 — comfortable. Raising N_ENVS to
32 with the reference 512/24 budget puts 17 152 rows through a 512-wide MLP
twelve times per env-step; feasible on an NVIDIA card, not on Apple. Scale
N_ENVS up and the MPPI budget down together, or run MPC-off.

`updates_per_step` is per ITERATION. `updates_per_step=N_ENVS` reproduces the
single-env ratio of one gradient step per env-step (the reference ratio);
lower it to trade sample-efficiency for wall-clock.

## UTD — what this file is currently configured to test (2026-08-12)

Every walker run before today used `updates_per_step=1`, i.e. UTD=0.125 —
**one eighth of the reference ratio**. The diagnostics say that is the binding
constraint: over a clean 220k-step run, `consistency_loss` fell all the way
(0.045 → 0.011) while `value_loss` and `reward_loss` went FLAT at ~0.028 from
50k onward. The dynamics model kept learning; the two heads that decide control
stopped. `q_mean` tracks `td_target_mean` to within 0.5% and sits 7-18% under
realized returns, so the critic is calibrated — under-trained, not broken.

⚠ Price a UTD change in GRADIENT STEPS, not env-steps. Holding TOTAL fixed and
raising the ratio 8x reads as "7x slower" and is the wrong comparison: at
UTD=1, 150k env-steps costs ~2.2h and buys 150k updates, where the 220k-step
control run took ~20 min and bought 27.5k. Fewer steps, far more learning.

Measured (single-task walk, N_ENVS=8, post-`1cc6f779`):

| config          | steps to eval ~800 | wall-clock | steps/s |
|-----------------|--------------------|------------|---------|
| UTD=1,     off  | 62k                | 38.9 min   | 26.3    |
| UTD=0.125, on   | 200k               | 26.5 min   | 125.8   |
| UTD=0.125, off  | never (377 @ 220k) | —          | 185.4   |

UTD=1 is **3.2x more sample-efficient and 1.5x slower in wall-clock**. It beat
the entire 220k-step MPC-off control by 2.1x on under a third of the data.
Training returns confirm it is not eval luck: 62 → 111 → 214 → 309 across the
bins where the control managed 36 → 56 → 91 → 119.

The critic was the thing that was starved. Implied value from realized returns
vs `q_mean`: at UTD=0.125 / 100k, implied ~37.7 against 12.05 — under by 3x;
at UTD=1 / 75-100k, implied ~80 against 69.9 — under by 13%. No instability at
8x either: `q_mean` tracks `td_target_mean` within 1%, `pi_scale`/`q_mean`
holds at 0.47 against the control's 0.41.

⚠ Cost decomposition, from those three throughputs: ~37 ms per update, ~6 ms
env+acting per iteration, ~21 ms for MPC. So the planner is 47% at UTD=0.125
and **~7% at UTD=1** — the update budget dwarfs it. Hence MPC ON here.

⚠ An earlier version of this file ran the UTD test MPC-OFF, reasoning that the
controllers are indistinguishable below ~120k. That crossover was measured at
UTD=0.125; at UTD=1 the model is far better far earlier, so it almost certainly
moves down — plausibly inside the 40-60k range where that run did its work. Do
not reuse a crossover measured at one update ratio to justify a choice at
another.

⚠ UTD=1 sat at 798/797 across two evals while the UTD=0.125 MPC-on control was
still climbing (872, 891) when it stopped. Two points is not a plateau, but if
UTD=1 does cap near 800 while the slower config keeps going, that matters
before this becomes the recipe for the multi-task rerun.

⚠⚠ Both control curves above are only valid post-`1cc6f779`. Runs built in the
`517084c2`..`baeaa9bc` window had FROZEN target Q nets (a version-gated weight
cache the polyak write never invalidated) and scored 74.7 at 150k where the
fixed build scores 230.5. Do not compare against a number from that window.

## The run baseline — what this file is set to measure (2026-08-13)

Multi-task TD-MPC2 (`tdmpc2_dm_walker_multitask_gpu.mojo`, post-`8d7f07d8`,
UTD=1 + MPC) solved two of its three tasks and left one behind:

    stand  988      walk  976      run  ~160        @ 288k total env-steps
                                                     (~96k per task)

Walk did not merely survive multi-task — 976 at ~85k walk-steps beats the
dedicated single-task run's 845 at 99k. So the multi-task machinery is sound
and the open question is run alone, which has never been trained by itself.

TASK="run" answers it, and the two outcomes point opposite ways:

  * run alone also lands ~160 → multi-task costs run NOTHING. It is simply the
    hard task (speed 8; SAC needed a long run to reach 730). Reallocating
    collection away from the solved tasks would buy nothing, and the answer is
    more steps or more capacity.
  * run alone reaches 500+ → multi-task is genuinely starving run, and a
    learning-progress curriculum — cut stand/walk collection once they hit
    ceiling, spend it on run — is well motivated with a number behind it.

⚠ Read the MATCHED point (~96k env-steps) for the comparison, and the TAIL
(96k → 150k) for whether run is still climbing. They answer different
questions and only the first one is a like-for-like comparison against the
multi-task result.

Run:
    pixi run -e nvidia mojo run -I . examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo
    pixi run -e apple  mojo run -I . examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.tdmpc2.config import TDMPC2
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig


# ── pick ONE ─────────────────────────────────────────────────────────────
# ⚠ "run" — this is the RUN BASELINE (see "The run baseline" in the docstring).
# Every run gets its own directory (the slug carries TASK), so flipping this
# back to "walk" recovers the earlier baseline without clobbering anything.
comptime TASK: StaticString = "run"  # "stand" | "walk" | "run"

comptime MOVE_SPEED: Float64 = 0.0 if TASK == "stand" else (
    1.0 if TASK == "walk" else 8.0
)

comptime TARGET = "gpu"        # batched driver requires env target == this
# MPC ON. At UTD=1 the update budget dwarfs everything else, so the planner
# costs ~7% here against 47% at UTD=0.125 — see the UTD block below. There is
# no longer a reason to run this configuration without it.
comptime USE_MPC = True
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime N_ENVS = 8            # training envs stepped in lockstep
comptime EVAL_ENVS = 8         # isolated eval env batch (see `eval_env`)

comptime OBS = DMWalkerModel.OBS_DIM       # 24
comptime ACT = DMWalkerModel.ACTION_DIM    #  6
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000       # MUST be a multiple of N_ENVS (driver asserts)

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
# All step counts below are TOTAL env-steps ACROSS ALL ENVS (SAC's convention):
# the driver runs `TOTAL // N_ENVS` iterations.
comptime LEARN_START = 5_000
# ⚠ TOTAL is deliberately SHORT here. At UTD=1 the run is priced in GRADIENT
# STEPS, not env-steps: 150k env-steps buys 150k updates, against the 27.5k
# that a 220k-step run at UPDATES_PER_STEP=1 delivered. Do not "restore" this
# to 1M without also dropping UPDATES_PER_STEP — that is a ~20h run.
# 150k, not 100k: the multi-task run gave each task ~96k of its own env-steps,
# so 96k is the MATCHED-BUDGET point the comparison needs — and the extra 54k
# says whether run is still climbing past it, which is the difference between
# "starved by multi-task" and "just needs more steps".
comptime TOTAL = 150_000
# Per ITERATION, and an iteration is N_ENVS env-steps — so this value IS the
# UTD numerator: N_ENVS gives the reference ratio of 1 update per env-step,
# 1 gives 0.125. Every walker run before 2026-08-12 used 1, i.e. 1/8 of the
# published recipe, which is why the reward and value heads flatlined at 50k
# while consistency_loss kept falling.
comptime UPDATES_PER_STEP = N_ENVS
# Halved vs the control run's 25k: eval spread on this task is ~±65, so the
# curve needs points. Every SECOND point still lands on a 25k multiple and so
# lines up exactly with the UPDATES_PER_STEP=1 control.
comptime EVAL_EVERY = 12_500
comptime EP_LEN = 1_000        # dm_control's own limit
comptime DIAG_EVERY = 1_000
comptime PRINT_EVERY = 10_000
comptime CHECKPOINT_EVERY = 50_000

comptime MAX_RETURN = 1000.0

# TERMINATE_ON_UNHEALTHY=False: dm_control never terminates early, so the
# driver records terminated=0 throughout and the value bootstrap survives the
# 1000-step truncation.
comptime Env = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[MOVE_SPEED], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
comptime EvalEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[MOVE_SPEED], EVAL_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]


def main() raises:
    comptime assert (
        TASK == "stand" or TASK == "walk" or TASK == "run"
    ), "TASK must be 'stand', 'walk' or 'run'"

    var mode = "MPC" if USE_MPC else "MPC-off"
    print("=" * 70)
    print("TD-MPC2 — dm_control walker", TASK, "— BATCHED (", mode, ")")
    print("=" * 70)
    print("  N_ENVS =", N_ENVS, " OBS =", OBS, " ACT =", ACT)
    print("  latent =", LATENT, " B =", B, " H =", H)
    print("  total env-steps =", TOTAL, " (", TOTAL // N_ENVS, "iterations )")
    print("  updates/iteration =", UPDATES_PER_STEP)
    comptime if USE_MPC:
        print(
            "  MPPI =", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs x", MPC_ITERS,
            "iters  → grid", N_ENVS * (MPC_SAMPLES + MPC_PI_TRAJS), "rows",
        )
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()

    var env = Env(ctx)
    var eval_env = EvalEnv(ctx)
    var eval_env_ptr = Pointer(to=eval_env).as_unsafe_any_origin()

    # ⚠ The UTD tag is part of the slug on purpose: the UPDATES_PER_STEP=1
    # run is the CONTROL for the comparison this one exists to make, and the
    # two must be told apart by name as well as by directory.
    var run = RunContext(
        project=String("dm-control"),
        driver=String("examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo"),
        slug=(
            String("tdmpc2-dm-walker-batched-") + String(TASK)
            + ("-mpc" if USE_MPC else "-mpcoff")
            + "-utd" + String(UPDATES_PER_STEP)
        ),
        env=String("builtin:dm_control/walker-") + String(TASK),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run:", run.dir)

    var ag = TDMPC2[
        TARGET, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
    ](
        ctx=ctx, lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
    )

    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "TD-MPC2")
    logger.set_config("env", String("dm_control/walker-") + String(TASK))
    logger.set_config("target", TARGET)
    logger.set_config("mpc", String("1") if USE_MPC else String("0"))
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("latent", String(LATENT))
    logger.set_config("batch", String(B))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    print("Starting training...")
    print("-" * 70)
    var t_start = perf_counter_ns()
    var best = ag.train_batched[
        Env, N_ENVS, RunLogger, USE_MPC, EvalEnv, EVAL_ENVS
    ](
        env,
        TOTAL,
        rng_seed=UInt64(42),
        updates_per_step=UPDATES_PER_STEP,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=checkpoint_path,
        checkpoint_every=CHECKPOINT_EVERY,
        artifacts=artifacts,
        run_dir=run.dir,
        eval_env=eval_env_ptr,
        eval_every=EVAL_EVERY,
        eval_max_steps=EP_LEN,
    )
    _ = eval_env  # lifetime extender for eval_env_ptr
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9

    finish_run(
        run, logger, artifacts, String("best_eval_return=") + String(best)
    )
    _ = logger  # lifetime extender for logger_ptr

    print("-" * 70)
    print("=" * 70)
    print("Training complete —", TASK, "(", mode, ", ", N_ENVS, "envs )")
    print("  total env_steps  =", TOTAL)
    print("  elapsed          =", elapsed, "s")
    print("  env-steps/s      =", Float64(TOTAL) / elapsed)
    print("  best eval return =", best)
    print("  run              =", run.dir)
    print("=" * 70)

    var frac = Float64(best) / MAX_RETURN
    if frac > 0.8:
        print("EXCELLENT — near the dm_control ceiling (>0.8 x 1000).")
    elif frac > 0.5:
        print("STRONG — solved the task (>0.5 x 1000).")
    elif frac > 0.2:
        print("PROGRESS — partial competence (>0.2 x 1000).")
    else:
        print("WEAK — check `wm=` in the progress lines first.")
    print("")
    print("Single-env baseline (same task, same reward):")
    print("  examples/dm_control/tdmpc2_dm_walker_gpu.mojo")
    print("SAC on the same task:")
    print("  examples/dm_control/sac_dm_walker_training_gpu.mojo")
    print("=" * 70)
