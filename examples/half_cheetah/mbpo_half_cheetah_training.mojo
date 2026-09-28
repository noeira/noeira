"""MBPO training on HalfCheetah (CPU) via the new `MBPOAgent` facade.

Sibling of `sac_half_cheetah_training.mojo`. MBPO is model-based:
on top of a SAC backbone (`StochasticActor` + twin critics +
auto-alpha) it trains a probabilistic dynamics ensemble and
augments the replay distribution with synthetic rollouts.

Uses the new `deep_agents/` surface:

  * `MBPOAgent[...]` — facade over `MBPOTrainer` + the single-env
    off-policy driver. Comptime params: ActorNet, CriticNet, DynNet,
    OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, SYNTH_CAPACITY,
    N_ENSEMBLE, NUM_ELITES, REAL_RATIO_PCT, LOGVAR_MIN, LOGVAR_MAX.
    Settings here mirror the `pendulum_mbpo_training.mojo` tuning
    (REAL_RATIO_PCT=50, LOGVAR_MAX=-5) — reference defaults (5/-2)
    diverge catastrophically on this surface.
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor, driver `print_every`
    cadence + chunk cadence.
  * One-file `.ckpt` envelope (`run.checkpoint_path("last")`) holding actor +
    critics + alpha + Adam states. NOTE: dynamics ensemble is NOT
    checkpointed; on `load` it restarts from scratch and the model_train
    cadence rebuilds it.

DynNet output layout: `2 * (1 + OBS_DIM)` = `[r_mean, r_logvar,
Δobs_mean[OBS_DIM], Δobs_logvar[OBS_DIM]]`. Logvar is clamped inside
the trainer to `[LOGVAR_MIN, LOGVAR_MAX]`.

Metric names (driver cadence): `avg_reward`, `episodes`.
Chunk cadence (`MBPOMetrics` fields): `actor_loss`, `critic_loss`,
  `alpha`, `train_steps`, `n_updates`.

HalfCheetah (Physics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

NOTE: MBPO is compute-heavy on CPU (ensemble training every
`model_train_freq` env-steps + synthetic rollouts every train_step).
`num_rollouts_per_step` and `dyn_epochs_per_round` are dialed down
vs. the GPU reference to keep CPU wall-time manageable.

Run:
    pixi run mojo run -I . examples/half_cheetah/mbpo_half_cheetah_training.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.mbpo import MBPOAgent
from noeira.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime DYN_HIDDEN = 200
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000
comptime SYNTH_CAPACITY = 400_000
comptime N_ENSEMBLE = 7
comptime NUM_ELITES = 5
# Kept at 50% (NOT the legacy GPU's 5%): the legacy's aggressive synthetic
# reliance only works with 100k FRESH rollouts/round, which is GPU-scale.
# At our CPU rollout budget, leaning harder on a thin/stale synth pool
# measurably HURT (a real_ratio=35 / UTD=15 sweep degraded faster). So the
# conservative CPU regime + the LayerNorm critic is the sweet spot here.
comptime REAL_RATIO_PCT = 50
comptime LOGVAR_MIN_F = -10.0
comptime LOGVAR_MAX_F = -5.0  # tuned

comptime NUM_STEPS = 100_000
comptime PRINT_EVERY = 5_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
# Critic with pre-activation LayerNorm (REDQ/SR-SAC stability fix; mirrors
# the legacy MBPO critic). Pattern: Linear → LayerNorm → ReLU, repeated.
# Bounds the critic's feature magnitudes so Q can't drift to ±∞ under the
# high-UTD / stale-synthetic-batch pressure of this CPU regime. Without it
# the critic loss explodes (~1e8) and Q diverges (→ −13k), dragging the
# actor + entropy temperature with it. Not paper-faithful (vanilla MBPO
# uses plain Linear+ReLU and avoids the blow-up via a huge fresh synthetic
# buffer at real_ratio=0.05), but it fixes the mechanism directly.
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
# Dynamics output = 2 * (1 + OBS_DIM) = 2 * 18 = 36
# Layout: [r_mean, r_logvar, Δobs_mean[OBS_DIM], Δobs_logvar[OBS_DIM]]
comptime DynNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS_DIM)],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO (deep_agents) — HalfCheetah CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN (SAC)       =", HIDDEN)
    print("  DYN_HIDDEN         =", DYN_HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  SYNTH_CAPACITY     =", SYNTH_CAPACITY)
    print("  N_ENSEMBLE/ELITES  =", N_ENSEMBLE, "/", NUM_ELITES)
    print("  REAL_RATIO_PCT     =", REAL_RATIO_PCT)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)

    # ─── Run + logger ───────────────────────────────────────────────────────
    var run = RunContext(
        project=String("mujoco"),
        driver=String("examples/half_cheetah/mbpo_half_cheetah_training.mojo"),
        slug=String("mbpo-half-cheetah"),
        env=String("builtin:mujoco/half_cheetah"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run                =", run.dir)
    print("=" * 70)
    var logger = run_logger(run, buffer_size=200)
    logger.set_config("algorithm", "MBPO")
    logger.set_config("env", "HalfCheetah")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("dyn_hidden", String(DYN_HIDDEN))
    logger.set_config("batch", String(BATCH))
    logger.set_config("ensemble", String(N_ENSEMBLE))
    logger.set_config("real_ratio_pct", String(REAL_RATIO_PCT))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = MBPOAgent[
        "cpu",
        ActorNet,
        CriticNet,
        DynNet,
        OBS_DIM,
        ACT_DIM,
        BATCH,
        REPLAY_CAPACITY,
        SYNTH_CAPACITY,
        N_ENSEMBLE,
        NUM_ELITES,
        REAL_RATIO_PCT,
        LOGVAR_MIN_F,
        LOGVAR_MAX_F,
        USE_TRAIN_CUDA_GRAPH=True,
    ](
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        model_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        init_alpha=0.2,
        target_entropy=-Scalar[DT](ACT_DIM),  # SAC default heuristic
        learning_starts=1_000,
        window_size=100,
        initial_episode_fill=0.0,
        # CPU-friendly cadences (vs. GPU defaults of 400 rollouts / 4 epochs).
        # A sweep toward the legacy GPU recipe (num_rollouts 1000 / UTD 15 /
        # target_entropy -3 / real_ratio 35) degraded FASTER on CPU: those
        # values assume 100k fresh synthetic transitions/round to stay
        # in-distribution. With the LayerNorm critic stabilizing things, the
        # conservative regime below is the best CPU operating point.
        model_train_freq=250,
        dyn_epochs_per_round=2,
        rollout_length=1,
        num_rollouts_per_step=100,
        sac_updates_per_step=5,
        dyn_batch_size=256,
    )
    var env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()

    # ─── Single train() call — auto-flush + auto-checkpoint ──────────────
    var t_start = perf_counter_ns()
    _ = agent.train_single[
        HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False],
        L=RunLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=checkpoint_path,
        checkpoint_every=CHECKPOINT_EVERY,
        artifacts=artifacts,
        run_dir=run.dir,
    )
    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    var total = NUM_STEPS
    var sent = logger.b.total_logged()
    finish_run(
        run, logger, artifacts,
        String("mean_return_100=") + String(agent.mean_return()),
    )
    _ = logger  # lifetime extender for logger_ptr

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete")
    print("  total env_steps        =", total)
    print("  elapsed                =", elapsed_s, "s")
    print("  mean ep return (last 100) =", agent.mean_return())
    print("  episodes completed     =", agent.ep_count())
    print("  remote points sent     =", sent)
    print("  run record             =", run.kv_path())
    print("=" * 70)

    var final_avg = Float64(agent.mean_return())
    if final_avg > 4000.0:
        print("EXCELLENT — running fast (mean > 4000).")
    elif final_avg > 1000.0:
        print("STRONG — learned locomotion (mean > 1000).")
    elif final_avg > 100.0:
        print("PROGRESS — early locomotion (mean > 100).")
    elif final_avg > 0.0:
        print("LEARNING — positive return (mean > 0).")
    else:
        print("EARLY — still exploring (mean < 0).")
    print("=" * 70)

    # ─── Save/load round-trip smoke test ─────────────────────────────────
    # NOTE: MBPO checkpoint does NOT include the dynamics ensemble — on
    # load, dynamics restarts from scratch but actor/critic/alpha are
    # exactly restored, so greedy action is still bit-identical.
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    for d in range(OBS_DIM):
        probe_obs[d] = Scalar[DT](0.1 * Float64(d - OBS_DIM // 2))
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    agent.load(checkpoint_path)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_after)

    print("Save/load round-trip on probe obs:")
    var ok = True
    for j in range(ACT_DIM):
        var diff = Float64(act_after[j] - act_before[j])
        if diff < 0:
            diff = -diff
        print(
            "  dim",
            j,
            " before =",
            act_before[j],
            " after =",
            act_after[j],
            " |diff| =",
            diff,
        )
        if diff > 1e-5:
            ok = False
    if ok:
        print("Round-trip OK (max |diff| < 1e-5 on every action dim).")
    else:
        print("Round-trip MISMATCH — investigate save/load semantics.")
    print("=" * 70)
