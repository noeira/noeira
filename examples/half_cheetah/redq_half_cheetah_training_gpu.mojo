"""REDQ training on HalfCheetah (GPU) via the new deep_agents `REDQAgent`.

deep_agents counterpart of the legacy `redq_half_cheetah_training_gpu.mojo`
(which uses `deep_agents.redq.REDQAgent.train_gpu`). Uses the new
`deep_agents/` surface end-to-end:

  * `REDQ["gpu", ...]` — paper-faithful preset (N=10 critics, M=2 subset-MIN
    target, UTD=20 critic updates per env step, policy_delay=20). Builds the
    primitive `REDQAgent` over the GPU `REDQTrainer`; all optimizers, the
    replay buffer, and the REDQ train-step pipeline run on-device.
  * `train_single` — the single-env off-policy driver (`run_offpolicy_train`)
    on the cross-target `(env=cpu, train=gpu)` path. This matches REDQ's paper
    setup (one parallel env; each transition triggers UTD inner gradient
    updates). The HalfCheetah physics env steps on CPU; every REDQ gradient
    update runs on the GPU.
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor: `avg_reward` +
    `episodes` at the driver's `print_every` cadence and (via `diag_every`)
    the full REDQ metric bundle (critic_loss, actor_loss, alpha, mean_q, ...).

deep_agents REDQ ships single-env training only (R.5); the batched multi-env
`train()` entry point that SAC has is a follow-up. REDQ's paper uses a single
env anyway, so this mirrors the legacy GPU script's setup directly.

HalfCheetah (Phyics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run -e apple  mojo run -I . examples/half_cheetah/redq_half_cheetah_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/redq_half_cheetah_training_gpu.mojo  # NVIDIA GPU
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.deep_agents.redq import REDQ
from noeira.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture / hyperparameters (paper-faithful REDQ, mirrors legacy script)
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256

# CUDA-graph capture of the UTD inner loop (NVIDIA only; no-op on Apple, where
# the device-resident path runs eagerly — the parity/verification path). The
# REDQ inner loop fires ~hundreds of tiny kernels/env-step (UTD=20 ×
# target_y/critic/polyak/actor over N=10 critics); capture collapses each
# env-step into one cuGraphLaunch.
comptime USE_TRAIN_CUDA_GRAPH = True

# Off-policy GPU training parameters.
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000

# Training duration (off-policy uses env steps, not episodes). Drop NUM_STEPS
# to ~30_000 for a smoke run.
comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 5_000
comptime PRINT_EVERY = 10_000  # driver-cadence verbose + env/mean_ret emit
comptime DIAG_EVERY = 1_000  # full metric-bundle flush cadence (mean_q, ...)

comptime CHECKPOINT_EVERY = 50_000

comptime EnvT = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ (deep_agents) — HalfCheetah GPU + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  N_ENSEMBLE         = 10  (paper-faithful)")
    print("  N_MIN (subset-min) = 2")
    print("  UTD ratio          = 20")
    print("  POLICY_DELAY       = 20")
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  WARMUP_STEPS       =", WARMUP_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Run + logger ───────────────────────────────────────────────────
        var run = RunContext(
            project=String("mujoco"),
            driver=String("examples/half_cheetah/redq_half_cheetah_training_gpu.mojo"),
            slug=String("redq-half-cheetah-gpu"),
            env=String("builtin:mujoco/half_cheetah"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run                =", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "REDQ")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))
        logger.set_config("num_ensemble", "10")
        logger.set_config("num_min", "2")
        logger.set_config("utd_ratio", "20")
        logger.set_config("policy_delay", "20")
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent (GPU) + single CPU HalfCheetah env ────────────────────
        # GPU training: the DeviceContext MUST be threaded through the agent
        # (the trainer keeps it for H2D/D2H staging + all on-device kernels).
        # The `REDQ` preset bakes in N=10 / M=2 / UTD=20 / POLICY_DELAY=20;
        # only the scalar knobs below are overridden.
        var agent = REDQ[
            "gpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN,
            USE_TRAIN_CUDA_GRAPH,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
            max_grad_norm=0.0,  # paper does not clip
        )
        var env = EnvT()

        # To resume from a previous run, uncomment:
        # agent.load(checkpoint_path)

        # ─── Single train_single() call — off-policy GPU driver ──────────
        # Each env transition triggers UTD=20 gradient updates on-device.
        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train_single[
            EnvT,
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
        var sent = logger.b.total_logged()
        finish_run(
            run, logger, artifacts,
            String("mean_return_100=") + String(agent.mean_return()),
        )
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("-" * 70)
        print("=" * 70)
        print("Training complete")
        print("  total env_steps           =", NUM_STEPS)
        print("  elapsed                   =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed        =", agent.ep_count())
        print("  remote points sent        =", sent)
        print("  run record                =", run.kv_path())
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
