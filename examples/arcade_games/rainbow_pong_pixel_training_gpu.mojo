"""Rainbow DQN CNN GPU Training on Pong — Pixel Observations (deep_agents).

Trains a Rainbow agent on the native Pong environment using pixel
observations (4×84×84 stacked grayscale frames), stepping `N_ENVS`
environments in parallel on the GPU via `BatchedGpuDiscreteEnv` while the
CNN Q-network trains on the same device.

Rainbow components: C51 + Double + PER + Dueling + Noisy + N-step.

The whole agent comes from the `RainbowCNN` preset in
`noeira/deep_agents/c51/config.mojo` — Nature-CNN backbone + noisy
dueling distributional heads + N-step-over-PER replay with the uint8 obs
ring, tuned pixel defaults baked in (lr 6.25e-5, warmup 20k, ε=0).
Training runs through `agent.train_gpu_batched`, the facade over the
GPU-batched discrete driver. Only the Pong-specific value support
(V_MIN/V_MAX ±2) is overridden below.

Memory note: deep_agents's prioritized replay is GPU-resident, so unlike
the legacy host-memory buffer the capacity here is bounded by device memory.
Obs/next_obs are stored as `uint8` (`OBS_STORE_DT = DType.uint8`): the store
kernel quantizes `round(x·255)` and the gather dequantizes `k/255` — lossless
for the pixel pipeline (it emits exact `k/255` grayscale values) and 4× the
capacity of the float ring (2·28224 bytes per transition instead of floats).
Raise BUFFER_CAPACITY further on large-VRAM cards.

Run with:
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo   # compile/smoke
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo   # training
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import Pointer

from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC

from noeira.deep_agents.c51.config import RainbowCNN
from noeira.deep_agents.training import BatchedGpuDiscreteEnv
from noeira.envs.arcade_games.pong import PongPixelEnv
from noeira.core.fmt import fit


# =============================================================================
# Constants
# =============================================================================

# Pong pixel: 4×84×84 = 28224 observation, 3 discrete actions.
comptime OBS_DIM = PongPixelEnv[DType.float64].OBS_DIM  # 28224
comptime NUM_ACTIONS = PongPixelEnv[DType.float64].NUM_ACTIONS  # 3
comptime FRAMES = 4

# ── Channels-last A/B toggle: True = NHWC conv tower + NHWC pixel obs ──────────
# The Nature-CNN's 84²→20²→9² convs coalesce channels-last (the EZv2/conv NHWC
# win on large maps). Flip and run both — eval return should match (NHWC learns
# identically; convergence-validated generically on ResNet-20 CIFAR). The pixel
# env's frame-stack obs layout is COUPLED to this so they never mismatch.
comptime USE_NHWC = True
comptime LAYOUT = LAYOUT_NHWC if USE_NHWC else LAYOUT_NCHW

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
# N_STEP=1 mirrors the converged clean-obs run (the value-config fix below was
# validated at N_STEP=1). Bump to 3 for full Rainbow once a pixel run confirms
# convergence.
comptime N_STEP = 3

# GPU-resident replay → capacity is VRAM-bound (obs + next_obs per slot).
# uint8 obs storage (OBS_STORE_DT below) shrinks each slot's pixel payload
# 4× vs the float ring, so 48k slots ≈ the old 12k float footprint.
comptime BUFFER_CAPACITY = 48_000
# Obs ring storage dtype: pixels are exact k/255 → uint8 quantize/dequant
# is bit-lossless. Pixel-only — keep DT for state-vector envs.
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32
comptime N_ENVS = 64  # fewer envs — each owns a pixel render/frame-stack workspace

# Distributional support — must bracket the DISCOUNTED return (≈ ±0.3..±6 with
# γ=0.99 + sparse ±1 rewards), NOT the raw ±21 episode score. [-2, 2] → atom
# spacing 0.08 (vs 0.84 at [-21, 21], too coarse to separate the 3 actions).
# This is the lever that made the clean-obs run converge (-19 → +21 perfect
# game); legacy Rainbow's [-21, 21] never got off the floor.
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

# Dense ball-return shaping (env `HIT_REWARD`): 0.0 = clean sparse ±1 rewards;
# 0.1 = original shaping (distorts the value scale, worse here since FRAME_SKIP
# accumulates it). Disabled to match the converged clean-obs config.
comptime HIT_REWARD = 0.0

# Replay ratio = GRAD_STEPS / N_ENVS = 16/64 = 0.25 (CleanRL train_freq=4).
comptime GRAD_STEPS = 16
comptime WARMUP = 20_000
comptime NUM_STEPS = 5_000_000
comptime LR = Scalar[DT](6.25e-5)

# Checkpointing. The CNN q-net + optimizer + epsilon are written to the run's
# `checkpoints/last.ckpt` every CKPT_EVERY env-steps (and once at the end); the
# replay buffer is NOT saved. `rainbow_pong_pixel_eval_render.mojo` reconstructs
# the same trainer config and `load_state`s that file to play a live game.
comptime CKPT_EVERY = 250_000


comptime PongPixelBatched = BatchedGpuDiscreteEnv[
    PongPixelEnv[DT, HIT_REWARD, LAYOUT=LAYOUT], N_ENVS, OBS_DIM, 1
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on Pong — Pixel (deep_agents)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Whole agent from the preset — Nature-CNN backbone + noisy dueling
        # distributional heads + N-step-over-PER replay (uint8 obs ring, the
        # preset default). Config-tuned scalars (lr 6.25e-5, ε=0 noisy,
        # warmup 20k, PER α=0.5/β=0.4, nstep=N_STEP) apply; only the Pong
        # value support deviates.
        var agent = RainbowCNN[
            "gpu",
            NUM_ACTIONS,
            BATCH_SIZE,
            BUFFER_CAPACITY,
            FRAMES,
            NUM_ATOMS,
            HIDDEN,
            N_STEP,
            OBS_STORE_DT,
            LAYOUT,
        ](
            ctx=ctx,
            lr=LR,
            learning_starts=WARMUP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = PongPixelBatched(ctx)
        # Separate env instance for deterministic (noise-off) greedy eval.
        var eval_env = PongPixelBatched(ctx)

        print("Environment: Pong (GPU-batched Pixel,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN CNN (deep_agents C51, GPU)")
        print(
            "  Components: C51 + Double + PER + Dueling + Noisy +",
            N_STEP,
            "-step",
        )
        print("  Observation: 4 × 84 × 84 =", OBS_DIM)
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Network: Nature CNN + Noisy Dueling Distributional heads")
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  Hit-reward shaping:", HIT_REWARD)
        print("  N-step:", N_STEP)
        print("  N envs (parallel):", N_ENVS)
        print(
            "  Buffer capacity:",
            BUFFER_CAPACITY,
            "(GPU-resident, uint8 obs ring)",
        )
        print("  Batch size:", BATCH_SIZE)
        print("  Grad steps / iter:", GRAD_STEPS, "(replay ratio 0.25)")
        print("  Learning rate:", LR)
        print("  Warmup:", WARMUP)
        print("  Total transitions:", NUM_STEPS)
        print()

        # =====================================================================
        # Logger
        # =====================================================================

        var run = RunContext(
            project=String("arcade"),
            driver=String("examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo"),
            slug=String("rainbow-pong-pixel"),
            env=String("builtin:arcade/pong"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents)")
        logger.set_config("env", "Pong (Pixel)")
        logger.set_config("obs", "4x84x84")
        logger.set_config("lr", String(LR))
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("obs_store_dtype", "uint8")
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))
        logger.set_config("v_min", String(V_MIN))
        logger.set_config("v_max", String(V_MAX))
        logger.set_config("hit_reward", String(HIT_REWARD))
        logger.set_config("grad_steps", String(GRAD_STEPS))

        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_gpu_batched[
                PongPixelBatched, N_ENVS, N_STEP, RunLogger
            ](
                env,
                NUM_STEPS,
                rng_seed=UInt64(42),
                updates_per_step=GRAD_STEPS,
                print_every=20_000,
                verbose=True,
                nstep_gamma=Scalar[DT](0.99),
                logger=Pointer(to=logger).as_unsafe_any_origin(),
                diag_every=5_000,
                checkpoint_every=CKPT_EVERY,
                checkpoint_path=checkpoint_path,
                artifacts=artifacts,
                run_dir=run.dir,
                eval_env=Pointer(to=eval_env).as_unsafe_any_origin(),
                eval_every=100_000,
                eval_episodes=10,
                episode_sync_every=32,
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            finish_run(
                run, logger, artifacts,
                String("mean_return_10=") + String(agent.mean_return()),
            )

            print("-" * 70)
            print()
            print("=" * 70)
            print("Rainbow CNN GPU Training Complete")
            print("=" * 70)
            print("Total transitions:", NUM_STEPS)
            print("Training time:", fit(String(elapsed_s), 6), "seconds")
            print(
                "Transitions/second:",
                fit(String(Float64(NUM_STEPS) / elapsed_s), 9),
            )
            print("Final mean return (last 10):", agent.mean_return())
            print("Episodes completed:", agent.ep_count())
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
