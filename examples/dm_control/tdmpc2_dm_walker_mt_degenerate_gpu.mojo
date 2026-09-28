"""DEGENERATE multi-task: the MT agent driven on ONE task. Where is the defect?

The multi-task run at UTD=1 + MPC did not learn anything — stand flat at ~145
over 184k steps, walk 48 where single-task reaches 583 at the same walk-steps,
run flat at ~30. The diagnostics say why:

    consistency_loss   MT: 0.157 → 0.191 → 0.183 → 0.172   (rising)
                       ST: 0.033 → 0.015 → 0.011 → 0.008   (falling)

~20x worse and never improving, with `q_mean` 36 against an implied ~7.5 from
realized returns (single-task sat within 13%).

That cannot be a property of the task mix. All three walker tasks share ONE
body, ONE observation and ONE action space — `DMWalkerConfig[MOVE_SPEED]`
changes the REWARD and nothing else. The latent dynamics model is therefore
learning a single dynamics function from 3x the data, so its consistency loss
should be equal or better than single-task, not twenty times worse.

So the defect is in the multi-task PATH, and this file splits the search space.

## The two configurations

`ROTATE_IDS = False` — one continuous `train_batched_mt` call, task id 1, walk
env, for TOTAL steps. Same data and same reward as the single-task run, the
only difference being that it flows through `TDMPC2MultiTaskAgent`: the `*MT`
nets, the `[obs|tem]` / `[z|a|tem]` concats, the embedding gather and the MT
world-model graph.

    consistency ~0.01 and walk climbs → the MT path is sound; the defect is in
      how tasks are MIXED (window sampling across segments, the replay's task
      column, the env stride). Go to ROTATE_IDS=True.
    consistency ~0.17 → the defect is in the MT path ITSELF and has nothing to
      do with multiple tasks. Nothing about task mixing needs investigating.

`ROTATE_IDS = True` — segments alternate task ids 0/1/2 while every segment
runs the SAME walk env. Identical data, identical reward, only the id label
rotates, so the three ids are semantically the same task. Any degradation here
is purely the conditioning/mixing machinery, measured against ROTATE_IDS=False
on data that is byte-for-byte the same kind.

⚠ Run False FIRST. True is only meaningful once the MT path is cleared on one
task; if False already fails, True cannot attribute anything.

## Comparison target

Single-task walk, N_ENVS=8, UTD=1, MPC on (`tdmpc2_dm_walker_batched_gpu.mojo`
at commit e02127ff), `avg_reward` by 25k bin: 63.6 / 140.3 / 280.2 / 435.4,
eval 845 at 99k, consistency_loss 0.0326 → 0.0079. Every dim below is set to
match that run — do not "tidy" any of them.

Run:
    pixi run -e nvidia mojo run -I . \\
        examples/dm_control/tdmpc2_dm_walker_mt_degenerate_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime TARGET = "gpu"

# ── the experiment switch ────────────────────────────────────────────────
comptime ROTATE_IDS = False
comptime N_SEGMENTS = 12       # only read when ROTATE_IDS

comptime USE_MPC = True
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime N_ENVS = 8
comptime EVAL_ENVS = 8

# ── identical to the single-task comparison run ──────────────────────────
comptime MAX_OBS = DMWalkerModel.OBS_DIM      # 24
comptime MAX_ACT = DMWalkerModel.ACTION_DIM   #  6
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000
comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
comptime LEARN_START = 5_000
comptime UPDATES_PER_STEP = N_ENVS            # UTD = 1, the reference ratio
comptime TOTAL = 100_000
comptime EPISODE_LEN = 1_000
comptime EVAL_EVERY = 12_500
comptime DIAG_EVERY = 1_000
comptime PRINT_EVERY = 10_000

# ── multi-task surface, kept even though only one task is driven ─────────
# ⚠ NUM_TASKS stays 3. Dropping it to 1 would change the net shapes and stop
# this from testing the configuration that failed.
comptime NUM_TASKS = 3
comptime TASK_EMB = 32
comptime T_WALK = 1

comptime SEG_STEPS = TOTAL // N_SEGMENTS if ROTATE_IDS else TOTAL

# ⚠ ONE env config for every segment: walk. The task ID varies under
# ROTATE_IDS, the ENVIRONMENT never does — that is the whole point.
comptime WalkEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime WalkEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]


def main() raises:
    print("=" * 74)
    print("DEGENERATE multi-task — MT agent, ONE task (walk)")
    print("=" * 74)
    var mode = "ROTATE_IDS (ids 0/1/2, same walk env)" if ROTATE_IDS else (
        "SINGLE id 1, one continuous call"
    )
    print("  mode     =", mode)
    print("  acting   =", "MPC" if USE_MPC else "MPC-off")
    print("  UTD      =", Float64(UPDATES_PER_STEP) / Float64(N_ENVS),
          " (updates/iteration =", UPDATES_PER_STEP, ")")
    print("  total    =", TOTAL, "env-steps  segment =", SEG_STEPS)
    print("  compare  → single-task walk UTD=1 MPC: avg_reward 63.6 / 140.3 /")
    print("             280.2 / 435.4 by 25k bin, consistency 0.033 → 0.008")
    print("=" * 74)

    seed(0)
    var ctx = DeviceContext()

    var env = WalkEnv(ctx)
    var ev = WalkEval(ctx)
    var ev_p = Pointer(to=ev).as_unsafe_any_origin()

    var ag = TDMPC2MultiTask[
        TARGET, MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES=MPC_SAMPLES, NUM_PI_TRAJS=MPC_PI_TRAJS,
        NUM_ELITES=MPC_ELITES, NUM_ITERS=MPC_ITERS,
    ](
        ctx=ctx, lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
    )

    var run = RunContext(
        project=String("dm-control"),
        driver=String(
            "examples/dm_control/tdmpc2_dm_walker_mt_degenerate_gpu.mojo"
        ),
        slug=(
            String("tdmpc2-mt-degenerate-dm-walker")
            + ("-rotate" if ROTATE_IDS else "-single")
            + ("-mpc" if USE_MPC else "-mpcoff")
        ),
        env=String("builtin:dm_control/walker-walk"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run:", run.dir)

    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "TD-MPC2-MT-degenerate")
    logger.set_config("env", "dm_control/walker-walk")
    logger.set_config("rotate_ids", String("1") if ROTATE_IDS else String("0"))
    logger.set_config("updates_per_step", String(UPDATES_PER_STEP))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var lg = Pointer(to=logger).as_unsafe_any_origin()

    print("Starting ...")
    print("-" * 74)
    var t0 = perf_counter_ns()
    var best: Scalar[DT] = -1.0e30

    comptime if ROTATE_IDS:
        var at = 0
        for s in range(N_SEGMENTS):
            # ⚠ id rotates, env does NOT. The three ids therefore label the
            # SAME task, so any per-id difference in the eval column is the
            # conditioning machinery and nothing else.
            var tid = s % NUM_TASKS
            var lab = String("id") + String(tid)
            var r = ag.train_batched_mt[
                WalkEnv, N_ENVS, RunLogger, USE_MPC, WalkEval, EVAL_ENVS
            ](
                env, tid, SEG_STEPS,
                rng_seed=UInt64(100 + s), updates_per_step=UPDATES_PER_STEP,
                print_every=PRINT_EVERY, verbose=True, logger=lg,
                diag_every=DIAG_EVERY, base_step=at,
                checkpoint_path=checkpoint_path, checkpoint_every=0,
                artifacts=artifacts, run_dir=run.dir,
                eval_env=ev_p, eval_every=EVAL_EVERY,
                eval_max_steps=EPISODE_LEN, task_label=lab,
            )
            at += SEG_STEPS
            if r > best:
                best = r
            ag.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run.dir)
    else:
        best = ag.train_batched_mt[
            WalkEnv, N_ENVS, RunLogger, USE_MPC, WalkEval, EVAL_ENVS
        ](
            env, T_WALK, TOTAL,
            rng_seed=UInt64(42), updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY, verbose=True, logger=lg,
            diag_every=DIAG_EVERY, base_step=0,
            checkpoint_path=checkpoint_path, checkpoint_every=25_000,
            artifacts=artifacts, run_dir=run.dir,
            eval_env=ev_p, eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN, task_label=String("walk"),
        )

    _ = ev
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    finish_run(
        run, logger, artifacts, String("best_eval_return=") + String(best)
    )
    _ = logger

    print("-" * 74)
    print("=" * 74)
    print("  best eval  =", best)
    print("  elapsed    =", elapsed, "s")
    print("  run        =", run.dir)
    print("=" * 74)
    print("Read consistency_loss FIRST, not the return:")
    print("  ~0.01 and falling → the MT path is sound on one task. The defect")
    print("    is in task MIXING; re-run with ROTATE_IDS=True.")
    print("  ~0.17 and flat    → the defect is in the MT path ITSELF (the *MT")
    print("    nets, the embedding concat, the MT world-model graph). Task")
    print("    mixing is exonerated and does not need investigating.")
    print("=" * 74)
