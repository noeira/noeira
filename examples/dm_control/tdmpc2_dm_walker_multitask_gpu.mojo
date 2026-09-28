"""TD-MPC2 MULTI-TASK on dm_control `walker` — stand + walk + run, one agent.

The point of the whole TD-MPC2 arc: ONE task-conditioned world model, one
Q-ensemble and one policy trained on three rewards at once, against the
single-task runs (`tdmpc2_dm_walker_batched_gpu.mojo`) as the baseline. Multi-
task TD-MPC2 has never been validated here — the pieces existed
(`TDMPC2MultiTaskAgent`, the `*MT` nets, the task-embedding table) but nothing
trained them, because there was no multi-task env to train on until dm_control
landed.

## Why walker is the right first multi-task target

All three tasks share ONE model, ONE observation (24) and ONE action space (6);
only the reward differs (MOVE_SPEED 0 / 1 / 8). So the obs/action PADDING that
multi-task TD-MPC2 normally needs is a no-op here, and what is left under test
is exactly the interesting part: does the task embedding let one world model
serve three reward functions? A negative result here is about conditioning, not
about plumbing dims — which is what makes it a clean first experiment.

That is also a hard requirement today: `train_batched_mt` asserts
`E.OBS_DIM == MAX_OBS` and `E.ACT_DIM == MAX_ACT`. Nothing pads an env's slabs
yet, so heterogeneous bodies need a padding `BatchedEnv` wrapper first.

## How the three tasks share the agent

Each dm_control task is a distinct `Phyics3dEnvConfig`, hence a distinct env
TYPE, so they cannot be iterated at runtime — the three `train_batched_mt`
calls below are written out, one per task, and each is its own instantiation.

Training alternates SEGMENTS: task 0 collects for `SEGMENT_STEPS`, then task 1,
then task 2, and round again. This is still joint multi-task learning, not
sequential fine-tuning, because the replay keeps every task and `train_step`
samples windows uniformly across all of them — during run's segment, most of
each batch is stand and walk data. What alternates is only which task the
ACTING policy is driving.

⚠ Keep `SEGMENT_STEPS` well above `EPISODE_LEN * N_ENVS` (8000 here = one
episode per env) or a segment cannot finish an episode and its `mean_ret` is
stale from the previous round.

## MPC

`USE_MPC` picks the controller, and it decides what this run is comparable to:

  * False (default) — `a = π(encode([obs|tem]))`. Fast, and it tests exactly
    the conditioning question: can one world model + one prior serve three
    rewards? Compare against an MPC-OFF single-task run.
  * True — MPPI planning through the task-conditioned world model
    (`TDMPC2RolloutCallbackGPUMT`). This is TD-MPC2 as published, and the
    only setting comparable to the single-task MPC numbers. It costs a full
    MPPI search per env-step: at N_ENVS=8 the planner's grid is
    8 × (MPC_SAMPLES + MPC_PI_TRAJS) rows through the world model, every
    horizon step, every iteration.

⚠ Do not compare an MPC-off multi-task run against a single-task MPC number.
On walker the planner is worth a large fraction of the score, so that
comparison charges the planner's absence to multi-task conditioning.

## Which regime — and why the first multi-task run does not count (2026-08-12)

The first multi-task run (130k steps, MPC-off, UTD=0.125) produced a striking
result: stand climbed, walk and run stalled, the embedding table collapsed
(rows grew 22x while their pairwise cosines converged to 0.9995), and a
cross-task eval matrix showed the STAND task id beating the correct id in every
environment. It was read as multi-task conditioning failing. That reading does
not survive the single-task controls measured afterwards:

  * At UTD=0.125 the planner buys nothing before ~120k env-steps in SINGLE-task
    either (eval 148 with vs 159 without at 50k). So "MPC adds nothing on
    walk/run" was the normal result at that data scale, not evidence of a
    task-blind model.
  * UTD=0.125 starves the critic outright: implied value from realized returns
    ~37.7 against `q_mean` 12.05, under by 3x, with `value_loss` and
    `reward_loss` flat from 50k while `consistency_loss` kept falling. At
    UTD=1 the same comparison is 80 vs 69.9, and `value_loss` finally drops.
  * UTD=1 is ~3.2x more sample-efficient: eval 750+ at 62-99k env-steps where
    UTD=0.125 needed 200k.

So the entire first multi-task run happened in a regime where the critic never
converged, and nothing measured in it — the embedding collapse included —
should be carried forward. This configuration re-runs it at UTD=1 with the
planner on, which is the first setting where the single-task critic actually
fits.

⚠ Wall-clock across runs is NOT comparable right now: Linear and MPPI were
optimised mid-sequence, and the UTD=1 MPC run clocked 33.2 steps/s against 26.3
for the UTD=1 MPC-OFF run that preceded it. Sample efficiency (score vs
env-steps) is unaffected; any cost claim needs a fresh A/B on one build.

Run:
    pixi run -e nvidia mojo run -I . examples/dm_control/tdmpc2_dm_walker_multitask_gpu.mojo
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

# ── acting mode ──────────────────────────────────────────────────────────
comptime USE_MPC = True
# MPPI budget (only read when USE_MPC). Reference TD-MPC2 is 512/24/64/6;
# these are the lighter numbers the single-task walker scripts use.
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

# ── PER-TASK POLICY-LOSS SCALE — a DEVIATION from TD-MPC2, default OFF ───
# The reference normalizes the policy loss by ONE running scale across every
# task (`tdmpc2/tdmpc2.py:34` — a single `RunningScale`, even for MT80). False
# reproduces that and every result measured before 2026-08-13.
#
# Turn it on to test the run hypothesis: at 312k steps the shared scale was set
# by the two SOLVED tasks (Q ~98) while run sat at ~16 and collapsed to the
# standing floor of 164 — with a MATCHED run-weighted gradient budget (104k vs
# 99k), so it was not a data problem. See `docs/TDMPC2_MULTITASK_VALIDATION.md`.
#
# ⚠ PASS/FAIL: run climbs off 164 while stand and walk HOLD ~980. Run improving
# at the cost of the other two is not a win — it is the same interference
# pointed the other way.
comptime PER_TASK_PI_SCALE = False
comptime PI_SCALE_MAX_REWEIGHT = 10.0

comptime N_ENVS = 8
comptime EVAL_ENVS = 8

comptime MAX_OBS = DMWalkerModel.OBS_DIM  # 24 — shared by all three
comptime MAX_ACT = DMWalkerModel.ACTION_DIM  #  6 — shared by all three
comptime NUM_TASKS = 3
# ⚠ RAISED 32 -> 96 (2026-08-13). The reference never uses 32: `task_dim` is
# 64 for MT30 and 96 for MT80 (`common/parser.py:75`), and `walker-run` is in
# BOTH sets alongside walker-stand and walker-walk — i.e. the reference's
# multi-task configuration contains exactly our three tasks plus 27 or 77
# others, and the paper reports walker-run around 600-700. Our 163 is a real
# gap against a setting that solves it.
#
# 96 rather than 64: this is a capacity EXPERIMENT, so err toward the upper
# reference value. The extra fan-in is negligible at 3 tasks.
#
# ⚠⚠ CHANGES EVERY NET'S FIRST-LAYER SHAPE. Checkpoints do not transfer across
# TASK_EMB, which is why it is in the filename. The viewer and probe hardcode
# 32 to stay compatible with the validated 312k checkpoint — edit theirs to
# match before pointing them at a 96 run.
comptime TASK_EMB = 96

comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000  # MUST be a multiple of N_ENVS

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
comptime LEARN_START = 5_000  # replay frames before the policy takes over
# Per ITERATION (= N_ENVS env-steps), so this value IS the UTD numerator:
# N_ENVS = the reference ratio of one update per env-step. The first
# multi-task run used 1 (UTD=0.125) and its conclusions do not carry over —
# see "Which regime" below.
comptime UPDATES_PER_STEP = N_ENVS
comptime EPISODE_LEN = 1_000
comptime SEGMENT_STEPS = EPISODE_LEN * N_ENVS  # 8 000 — one episode per env
# 3 x 13 x 8 000 = 312 k env-steps, i.e. ~100 k PER TASK. Sized off the
# single-task UTD=1 runs, which reached eval 750+ at 62-99 k env-steps; 960 k
# was sized for UTD=0.125, where the same score took 200 k.
comptime N_ROUNDS = 13
comptime EVAL_EVERY = SEGMENT_STEPS  # once per segment, on that task
comptime DIAG_EVERY = 1_000
comptime PRINT_EVERY = 4_000
# The checkpoint lives in the run's own directory (`RunContext` in main), so no
# run overwrites another. The slug is still tagged with the acting mode and
# UTD: `tdmpc2_dm_walker_multitask.ckpt` was the 130k MPC-off / UTD=0.125 run
# analysed on 2026-08-12, the control for every comparison here, and the name
# is what tells the runs apart on the dashboard.

comptime MAX_RETURN = 1000.0

# One env TYPE per task — the reason the three calls below are written out.
comptime StandEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[0.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime WalkEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime RunEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[8.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime StandEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[0.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime WalkEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime RunEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[8.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]

# Task ids — the index into the embedding table. ⚠ These must stay stable
# across a resume: a checkpoint's embedding row 1 means "walk" only because
# this table says so.
comptime T_STAND = 0
comptime T_WALK = 1
comptime T_RUN = 2


def main() raises:
    print("=" * 70)
    print("TD-MPC2 MULTI-TASK — dm_control walker stand + walk + run")
    print("=" * 70)
    print("  tasks    =", NUM_TASKS, " task_emb =", TASK_EMB)
    print("  N_ENVS   =", N_ENVS, " OBS =", MAX_OBS, " ACT =", MAX_ACT)
    print("  latent   =", LATENT, " B =", B, " H =", H)
    print("  segment  =", SEGMENT_STEPS, "env-steps x", N_ROUNDS, "rounds")
    print("  total    =", SEGMENT_STEPS * N_ROUNDS * NUM_TASKS, "env-steps")
    var mode = "MPC" if USE_MPC else "MPC-off (policy prior)"
    print("  acting   =", mode)
    comptime if USE_MPC:
        print(
            "  MPPI     =",
            MPC_SAMPLES,
            "+",
            MPC_PI_TRAJS,
            "trajs x",
            MPC_ITERS,
            "iters → grid",
            N_ENVS * (MPC_SAMPLES + MPC_PI_TRAJS),
            "rows",
        )
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()

    var run = RunContext(
        project=String("dm-control"),
        driver=String("examples/dm_control/tdmpc2_dm_walker_multitask_gpu.mojo"),
        slug=(
            String("tdmpc2-mt-dm-walker")
            + ("-mpc" if USE_MPC else "-mpcoff")
            + "-utd"
            + String(UPDATES_PER_STEP)
            # ⚠ The deviation is in the name. A per-task-scale run and a
            # reference run must never be confused — the whole experiment is
            # the comparison between them.
            + ("-ptscale" if PER_TASK_PI_SCALE else "")
            # ⚠ TASK_EMB changes every net's first-layer shape, so a
            # checkpoint from another value cannot load. Keep it in the name.
            + "-emb" + String(TASK_EMB)
        ),
        env=String("builtin:dm_control/walker-stand+walk+run"),
    )
    var CKPT = run.checkpoint_path(String("last"))
    print("  Run:", run.dir)

    var stand = StandEnv(ctx)
    var walk = WalkEnv(ctx)
    var run_e = RunEnv(ctx)
    var stand_ev = StandEval(ctx)
    var walk_ev = WalkEval(ctx)
    var run_ev = RunEval(ctx)
    var stand_ev_p = Pointer(to=stand_ev).as_unsafe_any_origin()
    var walk_ev_p = Pointer(to=walk_ev).as_unsafe_any_origin()
    var run_ev_p = Pointer(to=run_ev).as_unsafe_any_origin()

    var ag = TDMPC2MultiTask[
        TARGET,
        MAX_OBS,
        MAX_ACT,
        NUM_TASKS,
        TASK_EMB,
        B,
        CAP,
        ENC,
        LATENT,
        MLP,
        BINS,
        SN,
        VMIN,
        VMAX,
        H,
        # ⚠ KEYWORDS, not positional. `TDMPC2MultiTask` takes QP BEFORE the
        # MPPI budget while the single-task `TDMPC2` takes it LAST, so the
        # positional spelling that works there silently shifts every value by
        # one here (MPC_SAMPLES lands in QP). Keywords also survive any future
        # param insertion.
        NUM_SAMPLES=MPC_SAMPLES,
        NUM_PI_TRAJS=MPC_PI_TRAJS,
        NUM_ELITES=MPC_ELITES,
        NUM_ITERS=MPC_ITERS,
    ](
        ctx=ctx,
        lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE),
        learning_starts=LEARN_START,
    )
    # ⚠ Runtime, not a construction parameter — so the DEVIATION is one visible
    # call rather than something buried in a preset's defaults.
    ag.set_per_task_pi_scale(
        PER_TASK_PI_SCALE, Scalar[DT](PI_SCALE_MAX_REWEIGHT)
    )

    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "TD-MPC2-MT")
    logger.set_config("env", "dm_control/walker-stand+walk+run")
    logger.set_config("target", TARGET)
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("num_tasks", String(NUM_TASKS))
    logger.set_config("task_emb", String(TASK_EMB))
    logger.set_config("segment_steps", String(SEGMENT_STEPS))
    logger.set_config(
        "per_task_pi_scale", String("1") if PER_TASK_PI_SCALE else String("0")
    )
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var lg = Pointer(to=logger).as_unsafe_any_origin()

    print("Starting multi-task training ...")
    print("-" * 70)
    var t_start = perf_counter_ns()

    var best_stand: Scalar[DT] = -1.0e30
    var best_walk: Scalar[DT] = -1.0e30
    var best_run: Scalar[DT] = -1.0e30
    var at = 0

    for rnd in range(N_ROUNDS):
        # ⚠ `base_step` must keep counting ACROSS tasks and rounds, or the
        # dashboard shows N_ROUNDS x 3 overlapping runs. The warmup gate reads
        # the replay count, not this, so it is a logging axis only.
        var b0 = ag.train_batched_mt[
            StandEnv, N_ENVS, RunLogger, USE_MPC, StandEval, EVAL_ENVS
        ](
            stand,
            T_STAND,
            SEGMENT_STEPS,
            rng_seed=UInt64(100 + rnd),
            updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=lg,
            diag_every=DIAG_EVERY,
            base_step=at,
            checkpoint_path=CKPT,
            checkpoint_every=0,
            artifacts=artifacts,
            run_dir=run.dir,
            eval_env=stand_ev_p,
            eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN,
            task_label=String("stand"),
        )
        at += SEGMENT_STEPS
        if b0 > best_stand:
            best_stand = b0

        var b1 = ag.train_batched_mt[
            WalkEnv, N_ENVS, RunLogger, USE_MPC, WalkEval, EVAL_ENVS
        ](
            walk,
            T_WALK,
            SEGMENT_STEPS,
            rng_seed=UInt64(200 + rnd),
            updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=lg,
            diag_every=DIAG_EVERY,
            base_step=at,
            checkpoint_path=CKPT,
            checkpoint_every=0,
            artifacts=artifacts,
            run_dir=run.dir,
            eval_env=walk_ev_p,
            eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN,
            task_label=String("walk"),
        )
        at += SEGMENT_STEPS
        if b1 > best_walk:
            best_walk = b1

        var b2 = ag.train_batched_mt[
            RunEnv, N_ENVS, RunLogger, USE_MPC, RunEval, EVAL_ENVS
        ](
            run_e,
            T_RUN,
            SEGMENT_STEPS,
            rng_seed=UInt64(300 + rnd),
            updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=lg,
            diag_every=DIAG_EVERY,
            base_step=at,
            checkpoint_path=CKPT,
            checkpoint_every=0,
            artifacts=artifacts,
            run_dir=run.dir,
            eval_env=run_ev_p,
            eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN,
            task_label=String("run"),
        )
        at += SEGMENT_STEPS
        if b2 > best_run:
            best_run = b2

        # One checkpoint per ROUND — after all three tasks have collected, so
        # the file is never a mid-round snapshot biased to the last task.
        ag.save_state(CKPT)
        announce_checkpoint(CKPT, artifacts, run.dir)
        # ⚠ Log the per-task scales, not just the shared one: whether the three
        # spreads actually SEPARATE is what makes this experiment readable. If
        # they stay near-equal, the reweight is ~1 and a null result says
        # nothing about the hypothesis — only that the mechanism never engaged.
        comptime if PER_TASK_PI_SCALE:
            logger.log_scalar(
                "pi_scale_stand", Float64(ag.task_pi_scale(T_STAND)), at
            )
            logger.log_scalar(
                "pi_scale_walk", Float64(ag.task_pi_scale(T_WALK)), at
            )
            logger.log_scalar(
                "pi_scale_run", Float64(ag.task_pi_scale(T_RUN)), at
            )
            print(
                "     pi_scale — shared",
                ag.pi_scale(),
                " stand",
                ag.task_pi_scale(T_STAND),
                " walk",
                ag.task_pi_scale(T_WALK),
                " run",
                ag.task_pi_scale(T_RUN),
            )
        print(
            "  ── round",
            rnd + 1,
            "/",
            N_ROUNDS,
            " @",
            at,
            "env-steps",
            "  best: stand",
            best_stand,
            " walk",
            best_walk,
            " run",
            best_run,
        )

    _ = stand_ev
    _ = walk_ev
    _ = run_ev
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
    finish_run(
        run, logger, artifacts,
        String("best_eval_stand=") + String(best_stand)
        + " best_eval_walk=" + String(best_walk)
        + " best_eval_run=" + String(best_run),
    )
    _ = logger

    print("-" * 70)
    print("=" * 70)
    print("Multi-task training complete")
    print("  total env_steps =", at)
    print("  elapsed         =", elapsed, "s")
    print(
        "  best eval — stand:",
        best_stand,
        " walk:",
        best_walk,
        " run:",
        best_run,
    )
    print("  run             =", run.dir)
    print("=" * 70)
    print("Read it against the SINGLE-TASK, MPC-OFF baseline per task —")
    print(
        "  examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo"
        " (USE_MPC=False)"
    )
    print("  ⚠ match the ACTING MODE across the comparison: an MPC-off")
    print("    multi-task run vs a single-task MPC number charges the")
    print("    planner's absence to multi-task conditioning.")
    print("")
    print("  3 tasks at parity with single-task  → conditioning works.")
    print("  stand fine, run collapsed           → the shared model is being")
    print("    dominated by the easy task; raise TASK_EMB or rebalance the")
    print("    segment lengths before concluding anything about TD-MPC2.")
    print("=" * 70)
