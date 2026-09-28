"""SAC training on Humanoid (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_humanoid_training.mojo` and counterpart of the legacy
`sac_humanoid_training_gpu.mojo`. Mirrors
`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`:

  * `SACAgent["gpu", ...]` — facade over the GPU `SACTrainer` + the batched
    off-policy driver. All optimizers, the replay buffer, and the SAC
    train-step pipeline run on-device.
  * `Phyics3dBatchedEnv[HumanoidModel, HumanoidConfig, N_ENVS]` — the physics3d env
    (`GPUContinuousEnv`) into a `BatchedEnv`.
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor: `env/mean_ret` and
    `env/ep_count`.

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected transition.

CRITIC: this run uses a pre-activation **LayerNorm** critic (`Linear →
LayerNorm → ReLU`, repeated) — the REDQ/SR-SAC stability fix. It is the
VALIDATED config (HIDDEN=256, greedy ~5715): the critic loss stays bounded
where the plain critic explodes (~3800 @1.45M) and ERE blew the Q-values up
entirely. Everything else (fused actor, tuned scalars, uniform replay) matches
the `SAC[...]` preset, so the critic is the only change vs the 6006 baseline.

EVAL: a periodic GPU-parallel DETERMINISTIC eval logs `eval/mean_return` (the
deployable-policy signal) — the always-on stochastic `avg_reward` under-reports
SAC by the entropy term.

NOTE on checkpointing: the batched `train` entry point auto-saves the SAC
weights (one-file v3 checkpoint: actor + online critics, no optimizer state) every `CHECKPOINT_EVERY` env-steps
and once at the end (a host-side D2H between iterations, safe with the CUDA-
graph capture). The LayerNorm critic changes `PARAM_SIZE`, so this checkpoint
(`run.checkpoint_path("last")`) is NOT loadable by the preset-based eval
script — render it with `sac_humanoid_nn_ln_eval_cpu.mojo` (same LayerNorm
critic).

Humanoid (Phyics3dEnv, MuJoCo-style):
  * 45D observation (qpos[2:24] + qvel[0:23])
  * 17D continuous action (joint torques); `action_scale=0.4` to match the
    legacy Humanoid SAC runs.
  * Reward ≈ forward velocity + healthy bonus − control/contact costs;
    episode ends when the torso leaves a healthy z-range
    (`TERMINATE_ON_UNHEALTHY=True`).

Run:
    pixi run -e apple  mojo run -I . examples/humanoid/sac_humanoid_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/humanoid/sac_humanoid_training_gpu.mojo  # NVIDIA GPU
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
from noeira.envs.humanoid.humanoid_xml import HumanoidModel
from noeira.envs.humanoid.humanoid_config import HumanoidConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = HumanoidModel.OBS_DIM  # 45
comptime ACT_DIM = HumanoidModel.ACTION_DIM  # 17
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
# 256 matches the validated LayerNorm+H256 run (greedy ~5715). Bump to 512 only
# if the critic looks undersampled — your earlier test showed 512 added critic
# noise, so keep 256 here.
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Humanoid physics (NV=23) is the heaviest model here — its per-env RK4
# workspace (mass matrix ∝ NV² + contacts) is replicated across all N_ENVS.
# Kept small to avoid OOM (Ant at NV=14 already OOMs at 32). Bump up if you
# have GPU headroom.
comptime N_ENVS = 32

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 3_000_000
comptime WARMUP_STEPS = 25_000
comptime PRINT_EVERY = 50_000
comptime DIAG_EVERY = 1_000  # full metric-bundle flush cadence (mean_q, …)
comptime CHECKPOINT_EVERY = 50_000  # auto-save cadence (env steps)
# render with `sac_humanoid_nn_eval_cpu.mojo` (HIDDEN=256).

# Periodic DETERMINISTIC eval (greedy, no exploration noise) on an isolated set
# of `EVAL_ENVS` parallel envs — the deployable-policy signal. The always-on
# `avg_reward` is a STOCHASTIC rollout that under-reports SAC by the entropy
# term (training showed ~2655 stochastic vs ~5700 greedy), so `eval/mean_return`
# is the curve to trust. Eval runs GPU-parallel and touches no replay/optimizer
# state. VRAM: a 2nd BatchedGpuEnv adds EVAL_ENVS more per-env RK4 workspaces
# (NV=23) — kept at 16 (< N_ENVS) to stay within headroom; drop if OOM.
comptime EVAL_ENVS = 16
comptime EVAL_EVERY = 250_000  # env-steps between eval passes (~40 over 10M)
comptime EVAL_EPISODES = 16  # <= EVAL_ENVS → completes in one eval window


# Per-field tensor physics path (migration P5+): the batched fields facade is
# a `BatchedEnv` running the LEGACY PRODUCTION physics bundle by default
# (RK4 + Newton, parallel _mt schedules, treewalk CRBA, auto broadphase).
comptime BatchedEnvT = Phyics3dBatchedEnv[
    HumanoidModel, HumanoidConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=True
]
comptime EvalEnvT = Phyics3dBatchedEnv[
    HumanoidModel, HumanoidConfig, EVAL_ENVS, TERMINATE_ON_UNHEALTHY=True
]

# Actor + twin critics come from the `SAC[...]` preset (deep_agents.sac),
# which bundles the canonical fused-`LinearReLU` `SACActorNet` /
# `SACCriticNet` plus SAC's tuned defaults.


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Humanoid GPU (multi-env) + logger")
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
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Run + logger ───────────────────────────────────────────────────
        var run = RunContext(
            project=String("mujoco"),
            driver=String("examples/humanoid/sac_humanoid_training_gpu.mojo"),
            slug=String("sac-humanoid-gpu"),
            env=String("builtin:mujoco/humanoid"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run                =", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "Humanoid")
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
        # Humanoid keeps action_scale=0.4 + the example-specific warmup/window.
        var agent = SAC[
            "gpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
        ](
            ctx=ctx,
            action_scale=0.4,  # match legacy Humanoid SAC runs
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)
        # Isolated eval env (greedy deterministic rollouts; never touches the
        # training env's state or the replay buffer).
        var eval_env = EvalEnvT(ctx)
        # `.as_unsafe_any_origin()` — the facade takes
        # Optional[Pointer[EE, MutAnyOrigin]]; a tracked-origin
        # pointer doesn't convert (same idiom as logger_ptr above).
        var eval_env_ptr = Pointer(to=eval_env).as_unsafe_any_origin()

        # ─── Single train() call — batched GPU off-policy driver ─────────
        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            L=RunLogger,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=False,
            EE=EvalEnvT,
            EVAL_ENVS=EVAL_ENVS,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            episode_sync_every=32,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run.dir,
            eval_env=eval_env_ptr,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            eval_max_steps=1000,
        )
        _ = eval_env  # lifetime extender for eval_env_ptr
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
        if final_avg > 5000.0:
            print("EXCELLENT — walking (mean > 5000).")
        elif final_avg > 2000.0:
            print("STRONG — sustained upright locomotion (mean > 2000).")
        elif final_avg > 1000.0:
            print("PROGRESS — staying upright (mean > 1000).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
