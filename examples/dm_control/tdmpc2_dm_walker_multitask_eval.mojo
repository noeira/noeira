"""Evaluate a MULTI-TASK TD-MPC2 walker checkpoint — planner ON vs planner OFF.

This is a DISCRIMINATOR, not a benchmark. During the multi-task run stand
climbs while walk and run stall, and two explanations fit that log equally
well:

  H1  conditioning interference — the shared model is dominated by the easy
      task and genuinely has not learned walk/run.
  H2  the run is simply MPC-OFF — standing is reachable by the policy prior
      alone, walking is not, and the planner (worth a large fraction of the
      single-task score) is absent.

The training log cannot separate them because it only ever reports one
controller. This script reports BOTH, from the SAME weights and the SAME
episodes, so the difference is attributable to the controller and nothing
else:

  * walk/run jump sharply with MPC on  → H2. The world model is fine, the
    prior is the bottleneck, the run is healthy. Re-run with `USE_MPC=True`
    (or read the final result against an MPC-OFF single-task baseline).
  * walk/run barely move with MPC on   → H1. The shared model has not learned
    the locomotion tasks; TASK_EMB and the segment balance are the levers,
    not the controller.

The MPC-OFF column doubles as the load check. It should land in the same
ballpark as the `eval=` numbers in the training log for the round the
checkpoint came from — NOT exactly on them, since the reset seeds differ. A
near-zero MPC-off column against a healthy log means the checkpoint did not
actually load.

⚠ Copy the checkpoint aside before running this. The training script rewrites
`tdmpc2_dm_walker_multitask.ckpt` once per round, and reading it mid-write
gives a truncated file.

⚠ The comptime net config below MUST match the run that wrote the checkpoint
(`tdmpc2_dm_walker_multitask_gpu.mojo`). `B` and `CAP` do not affect any
parameter shape — only the replay, which eval never touches — so they are
shrunk here. Everything else is copied verbatim.

Run:
    pixi run -e nvidia mojo run -I . \\
        examples/dm_control/tdmpc2_dm_walker_multitask_eval.mojo [--ckpt <run_id>]
"""

from std.random import seed
from std.time import perf_counter_ns
from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime TARGET = "gpu"
comptime CKPT = "tdmpc2_dm_walker_multitask.ckpt"

comptime EVAL_ENVS = 8
comptime EPISODE_LEN = 1_000
# Eval variance over 8 episodes is large on a half-trained walker; averaging a
# couple of seeds keeps a 30-point wobble from reading as a trend.
comptime N_SEEDS = 2
comptime SEED_BASE = 90_000

# ── MPPI budget — the numbers a `USE_MPC=True` training run would use, so the
# MPC column here predicts what that run would score. Raising these makes the
# planner stronger and the eval slower; they do NOT need to match anything in
# the checkpoint (the planner is stateless — it holds no learned weights).
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

# ── must match the training run ──────────────────────────────────────────
comptime MAX_OBS = DMWalkerModel.OBS_DIM      # 24
comptime MAX_ACT = DMWalkerModel.ACTION_DIM   #  6
comptime NUM_TASKS = 3
comptime TASK_EMB = 32
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime H = 3
comptime ACTION_SCALE = 1.0
# ── eval-only: no replay is sampled, so these are free to shrink ─────────
comptime B = 256
comptime CAP = 1_024

comptime StandEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[0.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime WalkEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime RunEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[8.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]

# ⚠ Task ids must match the training script's table — a checkpoint's embedding
# row 1 means "walk" only because that table said so.
comptime T_STAND = 0
comptime T_WALK = 1
comptime T_RUN = 2


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), String(CKPT)), String("last"))
    print("=" * 70)
    print("TD-MPC2 MULTI-TASK checkpoint eval — MPC ON vs OFF")
    print("=" * 70)
    print("  checkpoint =", ckpt)
    print("  eval_envs  =", EVAL_ENVS, " seeds =", N_SEEDS,
          " episode_len =", EPISODE_LEN)
    print("  MPPI       =", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs x",
          MPC_ITERS, "iters → grid",
          EVAL_ENVS * (MPC_SAMPLES + MPC_PI_TRAJS), "rows")
    print("=" * 70)

    seed(0)
    var ctx = DeviceContext()

    var stand_ev = StandEval(ctx)
    var walk_ev = WalkEval(ctx)
    var run_ev = RunEval(ctx)

    var ag = TDMPC2MultiTask[
        TARGET, MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        # ⚠ KEYWORDS — `TDMPC2MultiTask` takes QP BEFORE the MPPI budget.
        NUM_SAMPLES=MPC_SAMPLES, NUM_PI_TRAJS=MPC_PI_TRAJS,
        NUM_ELITES=MPC_ELITES, NUM_ITERS=MPC_ITERS,
    ](
        ctx=ctx, action_scale=Scalar[DT](ACTION_SCALE), learning_starts=0,
    )

    print("Loading", ckpt, "...")
    ag.load_state(ckpt)
    print("  loaded.")
    print("-" * 70)

    var t0 = perf_counter_ns()

    # Two modes x three tasks. `evaluate_batched_mt` takes USE_MPC as a
    # comptime param, so each cell is its own instantiation — written out
    # rather than looped, same reason the training script writes out its three
    # segments.
    var off_stand: Scalar[DT] = 0.0
    var off_walk: Scalar[DT] = 0.0
    var off_run: Scalar[DT] = 0.0
    var on_stand: Scalar[DT] = 0.0
    var on_walk: Scalar[DT] = 0.0
    var on_run: Scalar[DT] = 0.0

    for k in range(N_SEEDS):
        var sd = UInt64(SEED_BASE + k * 1_000)
        print("  seed", sd, "...")

        var a = ag.evaluate_batched_mt[StandEval, EVAL_ENVS, False](
            stand_ev, T_STAND, max_steps=EPISODE_LEN, rng_seed=sd
        )
        var b = ag.evaluate_batched_mt[WalkEval, EVAL_ENVS, False](
            walk_ev, T_WALK, max_steps=EPISODE_LEN, rng_seed=sd
        )
        var c = ag.evaluate_batched_mt[RunEval, EVAL_ENVS, False](
            run_ev, T_RUN, max_steps=EPISODE_LEN, rng_seed=sd
        )
        print("    MPC off — stand", a, " walk", b, " run", c)
        off_stand += a
        off_walk += b
        off_run += c

        # SAME seed → the planner faces the SAME initial states and the same
        # env noise as the prior did above. That is what makes the two columns
        # a controlled comparison rather than two unrelated samples.
        var d = ag.evaluate_batched_mt[StandEval, EVAL_ENVS, True](
            stand_ev, T_STAND, max_steps=EPISODE_LEN, rng_seed=sd
        )
        var e = ag.evaluate_batched_mt[WalkEval, EVAL_ENVS, True](
            walk_ev, T_WALK, max_steps=EPISODE_LEN, rng_seed=sd
        )
        var f = ag.evaluate_batched_mt[RunEval, EVAL_ENVS, True](
            run_ev, T_RUN, max_steps=EPISODE_LEN, rng_seed=sd
        )
        print("    MPC on  — stand", d, " walk", e, " run", f)
        on_stand += d
        on_walk += e
        on_run += f

    var n = Scalar[DT](N_SEEDS)
    off_stand /= n
    off_walk /= n
    off_run /= n
    on_stand /= n
    on_walk /= n
    on_run /= n

    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    _ = stand_ev
    _ = walk_ev
    _ = run_ev

    print("-" * 70)
    print("=" * 70)
    print("  task     MPC off      MPC on      gain")
    print("  stand   ", off_stand, "  ", on_stand, "  ",
          on_stand - off_stand)
    print("  walk    ", off_walk, "  ", on_walk, "  ", on_walk - off_walk)
    print("  run     ", off_run, "  ", on_run, "  ", on_run - off_run)
    print("  elapsed =", elapsed, "s")
    print("=" * 70)
    print("Reading this:")
    print("  MPC-off column ~ the training log's eval= for that round → the")
    print("    checkpoint loaded. Near zero against a healthy log → it did not.")
    print("  walk/run gain LARGE  → H2: the prior is the bottleneck, not the")
    print("    conditioning. The multi-task run is healthy; switch USE_MPC on,")
    print("    or judge it against an MPC-OFF single-task baseline.")
    print("  walk/run gain SMALL  → H1: the shared model has not learned the")
    print("    locomotion tasks. Raise TASK_EMB or rebalance the segments.")
    print("=" * 70)
