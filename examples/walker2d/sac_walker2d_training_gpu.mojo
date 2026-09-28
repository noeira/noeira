"""SAC training on Walker2d (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_walker2d_training.mojo` and counterpart of the legacy
`sac_walker2d_training_gpu.mojo`. Mirrors
`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`:

  * `SACAgent["gpu", ...]` — facade over the GPU `SACTrainer` + the batched
    off-policy driver. All optimizers, the replay buffer, and the SAC
    train-step pipeline run on-device.
  * `Phyics3dBatchedEnv[Walker2dModel, Walker2dConfig, N_ENVS]` — the env
    (`GPUContinuousEnv`) into a `BatchedEnv`.
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor: `avg_reward` + `episodes`
    at `print_every`, AND (via `diag_every`) the full SAC metric bundle
    (`actor_loss`, `critic_loss`, `alpha`, `mean_q`, `mean_reward`,
    `train_steps`, …).

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected transition.

NOTE on checkpointing: the batched `train` entry point now supports an inline
checkpoint cadence (`checkpoint_every` + `checkpoint_path`) — it auto-saves the
trainer's one-file `nn-ckpt v2` envelope (actor + twin critics + optimizers +
alpha optimizer) every `CHECKPOINT_EVERY` env-steps and one final time at the
end. The save runs between iterations (a D2H of the live GPU params) so it is
safe to combine with the CUDA-graph capture below. The replay buffer / episode
tracker are NOT persisted, so a resumed run starts with a fresh replay. Load a
saved checkpoint back into a fresh agent with `agent.load(path)`; the path
is `run.checkpoint_path("last")`, in the run directory.

Walker2d (Phyics3dEnv, MuJoCo-style):
  * 17D observation (qpos[1:9] + qvel[0:9])
  * 6D continuous action (thigh/leg/foot torques × 2 legs)
  * Reward ≈ forward velocity + healthy bonus − control cost; episode ends
    when the torso leaves a healthy height/angle range
    (`TERMINATE_ON_UNHEALTHY=True`).

Run:
    pixi run -e apple  mojo run -I . examples/walker2d/sac_walker2d_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/walker2d/sac_walker2d_training_gpu.mojo  # NVIDIA GPU
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.envs.walker2d.walker2d_config import Walker2dConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = Walker2dModel.OBS_DIM  # 17
comptime ACT_DIM = Walker2dModel.ACTION_DIM  # 6
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Walker2d physics (NV=9, articulated chain) allocates a sizeable per-env RK4
# workspace (mass matrix ∝ NV² + contacts), replicated across all N_ENVS. The
# legacy `sac_walker2d_training_gpu.mojo` used 4; bump up if you have headroom.
comptime N_ENVS = 4

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 1_000_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 50_000
comptime DIAG_EVERY = 1_000  # full metric-bundle flush cadence (mean_q, …)
comptime CHECKPOINT_EVERY = 50_000  # auto-save cadence (env steps)


comptime BatchedEnvT = Phyics3dBatchedEnv[
    Walker2dModel, Walker2dConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=True
]

# Actor + twin critics come from the `SAC[...]` preset (deep_agents.sac),
# which bundles the canonical fused-`LinearReLU` `SACActorNet` /
# `SACCriticNet` (matmul+bias+ReLU in one kernel — halves the per-hidden-
# layer launch count on the eager GPU path) plus SAC's tuned defaults.


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Walker2d GPU (multi-env) + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  N_ENVS             =", N_ENVS)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  WARMUP_STEPS       =", WARMUP_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)

    with DeviceContext() as ctx:
        # ─── Run + logger ───────────────────────────────────────────────────
        var run = RunContext(
            project=String("mujoco"),
            driver=String("examples/walker2d/sac_walker2d_training_gpu.mojo"),
            slug=String("sac-walker2d-gpu"),
            env=String("builtin:mujoco/walker2d"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run                =", run.dir)
        print("=" * 70)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "Walker2d")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent + batched GPU env ─────────────────────────────────────
        # `SAC[target, OBS, ACT, BATCH, CAP, HIDDEN]` reads like a
        # constructor: it builds the SACAgent with the fused default nets
        # and SAC's tuned scalar defaults (lr=3e-4, gamma=0.99, tau=0.005,
        # init_alpha=0.2, target_entropy=-ACT, …). We override only the
        # example-specific knobs below; everything else comes from the preset.
        var agent = SAC[
            "gpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
        ](
            ctx=ctx,
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        # ─── Single train() call — batched GPU off-policy driver ─────────
        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            L=RunLogger,
            # CUDA-graph capture of the train step. The earlier capture
            # divergence was a replay-buffer bug — the uniform sample kernel
            # took the buffer fill count as a HOST scalar, which capture baked
            # at capture time, freezing sampling to the warmup-era transitions.
            # Fixed in gpu_replay.mojo (device-resident `size`); the sample
            # range now tracks the live count on every replay. Uniform replay
            # only — do NOT combine with ERE (still host-scalar / not capture
            # safe). NVIDIA only; no-op on Apple/Metal.
            USE_TRAIN_CUDA_GRAPH=True,
            # Capture the deterministic physics step too — collapses the env's
            # per-step eager kernel launches (newton/integrators/collision) into
            # one graph replay/iteration. The decisive lever at N_ENVS=4 / 250k
            # iters, where per-iteration launch+dispatch (not GPU compute)
            # dominates wall-clock. Safe: physics3d's GPU step is RNG-free
            # (RNG only in reset, which stays eager). NVIDIA only.
            USE_ENV_CUDA_GRAPH=True,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            # Defer the per-iteration episode-tracking D2H+synchronize: batch
            # the reward/done readback over this many iterations so the host
            # only stalls the GPU pipeline ~1/32 as often (returns are drained
            # exactly at every print/diag boundary, so logged values are fresh).
            episode_sync_every=32,
            # Auto-save the SAC weights (no optimizer state) every CHECKPOINT_EVERY
            # env-steps (and once more at the end). Safe alongside the
            # CUDA-graph capture above — the save is host-side D2H between
            # iterations. Resume/eval later via `agent.load(checkpoint_path)`.
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=checkpoint_path,
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
        print("  checkpoint saved to       =", checkpoint_path)
        print("=" * 70)

        var final_avg = Float64(agent.mean_return())
        if final_avg > 4000.0:
            print("EXCELLENT — walking fast (mean > 4000).")
        elif final_avg > 2000.0:
            print("STRONG — sustained walking (mean > 2000).")
        elif final_avg > 1000.0:
            print("PROGRESS — staying upright + moving (mean > 1000).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
