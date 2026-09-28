"""SAC on dm_control `walker-walk` (GPU, multi-env).

THE FIRST TRAINING SCRIPT FOR A dm_control SUITE TASK. Every ported task was
CPU-only until 2026-08-06 — the batched hook ABI could not carry the quantities
their rewards read (gap G10). See docs/DM_CONTROL_GPU_TRAINING_G10.md.

Mirrors `examples/walker2d/sac_walker2d_training_gpu.mojo`, which trains the
GYM walker. ⚠ They are DIFFERENT TASKS and the numbers do not compare:

  | | Gym Walker2d | dm_control walker-walk |
  |---|---|---|
  | obs | 17 (qpos[1:] + qvel) | 24 (14 xmat entries + height + 9 qvel) |
  | reward | forward vel + healthy bonus - ctrl cost, UNBOUNDED | in [0, 1] per step |
  | episode end | terminates when unhealthy | NEVER — 1000 steps, always |
  | max return | unbounded (~4000+ is good) | exactly 1000 |

**Read the return against 1000, not against the Gym script's thresholds.**
dm_control's convention (every suite task: 1000 steps, per-step reward in
[0,1], no early termination) is what makes cross-task comparison meaningful,
and it is why `TERMINATE_ON_UNHEALTHY=False` is baked into the `*Batched`
aliases rather than being a knob here.

⚠ walker runs FRAME_SKIP=10 physics substeps per control step (control 0.025 s
over a 0.0025 s timestep), so an env step here is ~10x a Gym-Walker2d step of
physics. Expect it to be correspondingly slower per env-step, and do not read
that as the GPU path being slow.

⚠ NOT A CONVERGENCE CLAIM. This script is gated only by
`tests/dm_control/test_locomotion_gpu_vs_cpu.mojo`, which shows the GPU
obs/reward match the MuJoCo-gated CPU path step for step. Whether SAC learns
walker-walk from them is a separate question, belongs on NVIDIA, and has not
been run. See the standing note in docs/DM_CONTROL_PORT.md.

Run:
    pixi run -e apple  mojo run -I . examples/dm_control/sac_dm_walker_walk_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/dm_control/sac_dm_walker_walk_gpu.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.sac import SAC
from noeira.envs.dm_control.walker import (
    DMWalkerWalkBatched,
    DMWalkerModel,
)


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = DMWalkerModel.OBS_DIM  # 24
comptime ACT_DIM = DMWalkerModel.ACTION_DIM  # 6
comptime HIDDEN = 256

comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Walker's per-env RK4/Euler workspace (mass matrix ∝ NV², plus contacts) is
# replicated across lanes, and FRAME_SKIP=10 multiplies the per-step work.
# Raise if you have headroom; 8 is comfortable on an Apple laptop.
comptime N_ENVS = 8

comptime NUM_STEPS = 1_000_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 50_000
comptime DIAG_EVERY = 1_000
comptime CHECKPOINT_EVERY = 50_000

comptime BatchedEnvT = DMWalkerWalkBatched[N_ENVS]


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC — dm_control walker-walk (GPU, multi-env)")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  N_ENVS             =", N_ENVS)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  WARMUP_STEPS       =", WARMUP_STEPS)
    print("  max return         = 1000 (dm_control convention)")
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Run + logger ───────────────────────────────────────────────────
        var run = RunContext(
            project=String("dm-control"),
            driver=String("examples/dm_control/sac_dm_walker_walk_gpu.mojo"),
            slug=String("sac-dm-walker-walk"),
            env=String("builtin:dm_control/walker-walk"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run                =", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "dm_control/walker-walk")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = SAC[
            "gpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
        ](
            ctx=ctx,
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            L=RunLogger,
            USE_TRAIN_CUDA_GRAPH=True,
            # ⚠ Capturing the env step is safe here for the same reason as in
            # the Gym scripts — physics3d's GPU step is RNG-free (RNG lives in
            # reset, which stays eager) — AND it now also covers the
            # `SYNC_FK_AFTER_STEP` FK/velocity refresh that every suite config
            # turns on. Both of those calls are deterministic too. NVIDIA only.
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
            episode_sync_every=32,
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

        print("-" * 70)
        print("=" * 70)
        print("Training complete")
        print("  total env_steps           =", NUM_STEPS)
        print("  elapsed                   =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed        =", agent.ep_count())
        print("  checkpoint saved to       =", checkpoint_path)
        print("=" * 70)

        # dm_control thresholds, NOT the Gym script's. Every episode is exactly
        # 1000 steps with a per-step reward in [0, 1], so the return IS the
        # mean per-step reward x 1000.
        var final_avg = Float64(agent.mean_return())
        if final_avg > 800.0:
            print("EXCELLENT — walking at target speed (return > 800/1000).")
        elif final_avg > 500.0:
            print("STRONG — sustained walking (return > 500/1000).")
        elif final_avg > 200.0:
            print("PROGRESS — standing, some forward motion (> 200/1000).")
        elif final_avg > 50.0:
            print("EARLY — mostly standing (> 50/1000).")
        else:
            print("EXPLORING — not yet standing (< 50/1000).")
        print("=" * 70)
