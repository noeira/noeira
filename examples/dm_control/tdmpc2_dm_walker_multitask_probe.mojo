"""Is the multi-task walker model TASK-BLIND? A cross-task eval matrix.

The planner-on/off eval (`tdmpc2_dm_walker_multitask_eval.mojo`) found MPPI
worth +87 on stand and nothing on walk or run. Planning can only help where the
model tells good action sequences from bad ones FOR THAT TASK, so that pattern
has a specific candidate explanation:

    the model has collapsed to a TASK-AVERAGED reward, and stand happens to
    sit closest to the blend.

A task-blind model would look exactly like what we measured — search improves
the task nearest the average and does nothing for the two that need a velocity
term — while a model that simply hasn't trained long enough would look the same
from the outside. This probe separates them.

## The measurement

`task_id` is a RUNTIME argument to `evaluate_batched_mt`, so each env can be
driven under every task id. That gives a 3x3 matrix: rows = which env (which
reward function is actually paying), columns = which task id the agent is
conditioned on.

           id=stand  id=walk  id=run
  stand       .        .        .
  walk        .        .        .
  run         .        .        .

Read the ROWS:

  * DIAGONAL WINS — conditioning carries real task information. Telling the
    agent "you are walking" measurably helps it in the walk env. Then the
    flat walk/run scores are an under-training / capacity story, and more
    steps, higher UTD or MPC-collected data are the levers.

  * ROW IS FLAT — the task id changes nothing about behaviour that the reward
    can see. The model is task-blind, and no amount of UTD or MPC fixes that;
    it is a conditioning problem (where the embedding enters, TASK_EMB, the
    shared policy-loss scale).

The off-diagonal is the control that a plain per-task eval cannot give you: a
diagonal number alone is consistent with both stories, and only the contrast
against the SAME env under a WRONG task id separates them.

⚠ Read the walk and run rows, not stand. If the blended reward really does
resemble stand, then every task id scores well in the stand env — a flat stand
row is EXPECTED under both hypotheses and discriminates nothing.

## The embedding table

Printed first, and nearly free: row norms, pairwise distances and cosine
similarities, for the FRESH init and again after loading. If the three rows
barely moved from their random init, or collapsed onto each other, that is the
mechanism visible directly in the parameters. It is corroboration, not the
verdict — rows can be far apart and still feed nothing useful downstream, which
is why the matrix above is the real test.

⚠ Copy the checkpoint aside first; the training run rewrites it once per round.

Run:
    pixi run -e nvidia mojo run -I . \\
        examples/dm_control/tdmpc2_dm_walker_multitask_probe.mojo [--ckpt <run_id>]
"""

from std.math import sqrt
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
comptime N_SEEDS = 2
comptime SEED_BASE = 90_000
# The MPC matrix is the informative one (planning is what exposes whether the
# model can score task-specific trajectories); the prior matrix is ~free, so
# both are run.
comptime RUN_MPC_MATRIX = True
comptime RUN_PRIOR_MATRIX = True

comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

# ── must match the training run ──────────────────────────────────────────
comptime MAX_OBS = DMWalkerModel.OBS_DIM
comptime MAX_ACT = DMWalkerModel.ACTION_DIM
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

def _row_norm(ref d: List[Scalar[DT]], t: Int) -> Float64:
    var s = 0.0
    for e in range(TASK_EMB):
        var x = Float64(d[t * TASK_EMB + e])
        s += x * x
    return sqrt(s)


def _dist(ref d: List[Scalar[DT]], a: Int, b: Int) -> Float64:
    var s = 0.0
    for e in range(TASK_EMB):
        var x = Float64(d[a * TASK_EMB + e]) - Float64(d[b * TASK_EMB + e])
        s += x * x
    return sqrt(s)


def _cos(ref d: List[Scalar[DT]], a: Int, b: Int) -> Float64:
    var dot = 0.0
    for e in range(TASK_EMB):
        dot += Float64(d[a * TASK_EMB + e]) * Float64(d[b * TASK_EMB + e])
    var na = _row_norm(d, a)
    var nb = _row_norm(d, b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _report_embedding(ref d: List[Scalar[DT]], tag: String) raises:
    print("  ", tag)
    print(
        "     |row| :  stand", _row_norm(d, 0),
        "  walk", _row_norm(d, 1), "  run", _row_norm(d, 2),
    )
    print(
        "     dist  :  stand-walk", _dist(d, 0, 1),
        "  stand-run", _dist(d, 0, 2), "  walk-run", _dist(d, 1, 2),
    )
    print(
        "     cos   :  stand-walk", _cos(d, 0, 1),
        "  stand-run", _cos(d, 0, 2), "  walk-run", _cos(d, 1, 2),
    )


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
    print("=" * 74)
    print("TD-MPC2 MULTI-TASK probe — is the model task-blind?")
    print("=" * 74)
    print("  checkpoint =", ckpt)
    print("  eval_envs  =", EVAL_ENVS, " seeds =", N_SEEDS)
    print("=" * 74)

    seed(0)
    var ctx = DeviceContext()

    var stand_ev = StandEval(ctx)
    var walk_ev = WalkEval(ctx)
    var run_ev = RunEval(ctx)

    var ag = TDMPC2MultiTask[
        TARGET, MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        NUM_SAMPLES=MPC_SAMPLES, NUM_PI_TRAJS=MPC_PI_TRAJS,
        NUM_ELITES=MPC_ELITES, NUM_ITERS=MPC_ITERS,
    ](
        ctx=ctx, action_scale=Scalar[DT](ACTION_SCALE), learning_starts=0,
    )

    # ── 1. the embedding table, before and after the load ────────────────
    print("TASK EMBEDDING")
    ag.task_emb.sync_to_host()
    _report_embedding(ag.task_emb.param.data, "FRESH init (control)")
    ag.load_state(ckpt)
    ag.task_emb.sync_to_host()
    _report_embedding(ag.task_emb.param.data, "AFTER load (trained)")
    print(
        "  → rows barely off the fresh control, or cos ~ 1.0, means the table"
    )
    print("    itself never separated the tasks.")
    print("-" * 74)

    var t0 = perf_counter_ns()

    # ── 2. the cross-task matrix ─────────────────────────────────────────
    # The env is a comptime TYPE so the three rows are written out; the task id
    # is a runtime argument, so the columns are a plain loop.
    comptime if RUN_PRIOR_MATRIX:
        print("MATRIX — MPC OFF (policy prior)")
        print("   env \\ id        stand         walk          run")
        for r in range(3):
            var acc = List[Scalar[DT]](length=3, fill=Scalar[DT](0))
            for t in range(3):
                for k in range(N_SEEDS):
                    var sd = UInt64(SEED_BASE + k * 1_000)
                    if r == 0:
                        acc[t] += ag.evaluate_batched_mt[
                            StandEval, EVAL_ENVS, False
                        ](stand_ev, t, max_steps=EPISODE_LEN, rng_seed=sd)
                    elif r == 1:
                        acc[t] += ag.evaluate_batched_mt[
                            WalkEval, EVAL_ENVS, False
                        ](walk_ev, t, max_steps=EPISODE_LEN, rng_seed=sd)
                    else:
                        acc[t] += ag.evaluate_batched_mt[
                            RunEval, EVAL_ENVS, False
                        ](run_ev, t, max_steps=EPISODE_LEN, rng_seed=sd)
                acc[t] /= Scalar[DT](N_SEEDS)
            var name = String("stand") if r == 0 else (
                String("walk") if r == 1 else String("run")
            )
            print("   ", name, "     ", acc[0], "  ", acc[1], "  ", acc[2])
        print("-" * 74)

    comptime if RUN_MPC_MATRIX:
        print("MATRIX — MPC ON (planner) ← the informative one")
        print("   env \\ id        stand         walk          run")
        for r in range(3):
            var acc = List[Scalar[DT]](length=3, fill=Scalar[DT](0))
            for t in range(3):
                for k in range(N_SEEDS):
                    var sd = UInt64(SEED_BASE + k * 1_000)
                    if r == 0:
                        acc[t] += ag.evaluate_batched_mt[
                            StandEval, EVAL_ENVS, True
                        ](stand_ev, t, max_steps=EPISODE_LEN, rng_seed=sd)
                    elif r == 1:
                        acc[t] += ag.evaluate_batched_mt[
                            WalkEval, EVAL_ENVS, True
                        ](walk_ev, t, max_steps=EPISODE_LEN, rng_seed=sd)
                    else:
                        acc[t] += ag.evaluate_batched_mt[
                            RunEval, EVAL_ENVS, True
                        ](run_ev, t, max_steps=EPISODE_LEN, rng_seed=sd)
                acc[t] /= Scalar[DT](N_SEEDS)
            var name = String("stand") if r == 0 else (
                String("walk") if r == 1 else String("run")
            )
            print("   ", name, "     ", acc[0], "  ", acc[1], "  ", acc[2])

    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    _ = stand_ev
    _ = walk_ev
    _ = run_ev

    print("-" * 74)
    print("=" * 74)
    print("  elapsed =", elapsed, "s")
    print("Reading the WALK and RUN rows (the stand row discriminates nothing")
    print("if the blended reward already resembles stand):")
    print("  diagonal cell clearly best in its row → the task id carries real")
    print("    information. Under-training / capacity, not conditioning:")
    print("    more steps, higher UTD, MPC-collected data.")
    print("  row flat within noise → TASK-BLIND. UTD and MPC will not fix it;")
    print("    the levers are TASK_EMB, where the embedding enters the reward")
    print("    and value heads, and the shared policy-loss scale.")
    print("=" * 74)
