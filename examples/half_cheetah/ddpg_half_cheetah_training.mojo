"""DDPG training on HalfCheetah (CPU) via the new `DDPGAgent` facade.

Sibling of `sac_half_cheetah_training.mojo` for the deterministic
DDPG actor + single-critic flavour. Uses the new `deep_agents/`
surface:

  * `DDPGAgent[...]` — facade over `DDPGTrainer` + the single-env
    off-policy driver. Same comptime parameterisation as SAC minus
    `target` / `SAMPLE` (DDPG is CPU-only here and uses the default
    uniform replay sampler internally).
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor, at every chunk boundary
    AND at the driver's `print_every` cadence.
  * Single-file checkpointing — `run.checkpoint_path("last")`, overwritten
    each chunk and uploaded through the run's artifact sink: actor + critic +
    their Adam states.

Metric names (driver cadence): `avg_reward`, `episodes`.
Chunk cadence (`DDPGMetrics` fields): `actor_loss`, `critic_loss`,
  `train_steps`, `n_updates`.

After training, the final checkpoint is reloaded into the same agent
and a greedy probe confirms the action vector reproduces dimension-by-
dimension to `|diff| < 1e-5`.

HalfCheetah (Physics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run mojo run -I . examples/half_cheetah/ddpg_half_cheetah_training.mojo
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
from noeira.nn.primitives.activations import Tanh
from noeira.deep_agents.ddpg import DDPGAgent
from noeira.deep_agents.training.blocks import UniformSampleCpuStep
from noeira.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 64
comptime REPLAY_CAPACITY = 100_000

comptime NUM_STEPS = 100_000
comptime PRINT_EVERY = 5_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000


# DDPG actor is deterministic: outputs a Tanh-bounded action vector
# scaled by `action_scale` (set to 1.0 here — HalfCheetah accepts
# torques in [-1, 1]).
comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, ACT_DIM],
    Tanh[ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("DDPG (deep_agents) — HalfCheetah CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)

    # ─── Run + logger ───────────────────────────────────────────────────────
    var run = RunContext(
        project=String("mujoco"),
        driver=String("examples/half_cheetah/ddpg_half_cheetah_training.mojo"),
        slug=String("ddpg-half-cheetah"),
        env=String("builtin:mujoco/half_cheetah"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run                =", run.dir)
    print("=" * 70)
    var logger = run_logger(run, buffer_size=200)
    logger.set_config("algorithm", "DDPG")
    logger.set_config("env", "HalfCheetah")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("batch", String(BATCH))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = DDPGAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ](
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        noise_scale=0.1,
        learning_starts=1_000,
        window_size=100,
        initial_episode_fill=0.0,
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
