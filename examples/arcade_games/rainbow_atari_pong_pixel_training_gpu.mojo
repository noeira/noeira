"""Rainbow DQN CNN GPU Training on Atari 2600 Pong — Pixel Observations.

Trains a Rainbow agent on the real Atari 2600 Pong ROM (6502/TIA/RIOT
emulation) using pixel observations (4×84×84 stacked grayscale frames).

This is the Atari counterpart to `rainbow_pong_pixel_training_gpu.mojo`,
which trains on the *native* GPU Pong physics engine. The crucial
difference: the Atari emulator is **CPU-only** (the 6502 opcode dispatch
diverges on the GPU), so there is no `BatchedGpuDiscreteEnv` path here.
Instead `N_ENVS` CPU-emulated envs step in parallel across CPU cores
(`BatchedCpuDiscreteEnv`, the Stage-1 lever from `docs/ATARI_AUDIT.md`)
while the CNN Q-network selects actions and trains on the GPU, via
`run_offpolicy_discrete_train_cpu_env_gpu_agent` (driver row: cpu env /
gpu train / N_ENVS). Each env applies ALE-style random no-op starts
(`noop_max=30`) on reset — without them N deterministic emulators driven
by a near-deterministic policy would step in lockstep.

The 4×84×84 pixel pipeline — render → max-pool (sprite-flicker) →
grayscale → box-filter resize 160×210→84×84 → 4-frame ring stack — is
already built into `AtariEnv[1]`; no wrapper needed. Frame skip
is fixed at 4 internally for pixel mode.

Rainbow components: C51 + Double + PER + Dueling + Noisy + N-step.

The whole agent comes from the `RainbowCNN` preset in
`noeira/deep_agents/c51/config.mojo` — Nature-CNN backbone + noisy
dueling distributional heads + N-step-over-PER replay with the uint8 obs
ring (lossless here too: AtariEnv emits exact `k/255` pixel obs), tuned
pixel defaults baked in (lr 6.25e-5, warmup 20k, ε=0). Only the
Pong-specific value support (V_MIN/V_MAX ±2) is overridden.

Note: Atari Pong exposes 6 ALE actions (NOOP, FIRE, RIGHT, LEFT,
RIGHTFIRE, LEFTFIRE) — FIRE serves the ball — vs the native engine's 3.

Requires the Pong ROM at `roms/pong.bin` (symlink to ale_py/roms/).

Run with:
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_training_gpu.mojo   # compile/smoke
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_training_gpu.mojo   # training
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import Pointer

from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT

from noeira.deep_agents.c51.config import RainbowCNN
from noeira.deep_agents.training.batched_env import BatchedCpuDiscreteEnv

from noeira.envs.atari import AtariEnv, load_rom
from noeira.envs.atari.games.registry import AtariGame
from noeira.envs.atari.flags import OBS_WIDTH, OBS_HEIGHT
from noeira.core.fmt import fit


# =============================================================================
# Constants
# =============================================================================

# Atari Pong pixel: 4×84×84 = 28224 observation, 6 ALE actions.
comptime FRAMES = 4
comptime OBS_DIM = FRAMES * OBS_WIDTH * OBS_HEIGHT  # 4 * 84 * 84 = 28224
comptime NUM_ACTIONS = 6  # Pong minimal set (NOOP/FIRE/RIGHT/LEFT/R+F/L+F)

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

# Emulator envs stepped in parallel across CPU cores. Size to the host's
# core count (each env is an independent 6502/TIA emulation, ~near-linear
# scaling); env memory is trivial (~0.3 MB/env).
comptime N_ENVS = 8
# Gradient updates per driver iteration (= per N_ENVS env transitions).
# Replay ratio = UPDATES_PER_STEP / N_ENVS = 0.25, matching the converged
# native Rainbow-pixel run (64 envs / 16 grad steps).
comptime UPDATES_PER_STEP = 2
# ALE-standard random no-op starts: k ~ U[0, 30] NOOPs after every reset.
comptime NOOP_MAX = 30

# GPU-resident replay → capacity is VRAM-bound (obs + next_obs per slot).
# uint8 obs storage (OBS_STORE_DT below) shrinks each slot's pixel payload
# 4× vs the float ring, so 48k slots ≈ the old 12k float footprint.
comptime BUFFER_CAPACITY = 48_000
# Obs ring storage dtype: AtariEnv pixel obs are exact k/255 → uint8
# quantize/dequant is bit-lossless. Pixel-only.
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32

# Distributional support — must bracket the DISCOUNTED return (≈ ±0.3..±6
# with γ=0.99 + sparse ±1 rewards), NOT the raw ±21 episode score. [-2, 2]
# → atom spacing 0.08. This is the lever that made the native pixel run
# converge; legacy Rainbow's [-21, 21] support never got off the floor.
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

comptime WARMUP = 20_000
comptime NUM_STEPS = 2_000_000
comptime LR = Scalar[DT](6.25e-5)

# Checkpointing. The CNN q-net + optimizer + epsilon are written to the run's
# `checkpoints/last.ckpt` every CKPT_EVERY env-steps (and once at the end);
# the replay buffer is NOT saved.
comptime CKPT_EVERY = 250_000

comptime ROM_PATH = "roms/pong.bin"


comptime AtariPongPixel = AtariEnv[1, DT]
comptime BatchedAtariPong = BatchedCpuDiscreteEnv[
    AtariPongPixel, N_ENVS, OBS_DIM
]


def _make_envs(
    # `MutUntrackedOrigin` on both sides of this helper: it takes what
    # `load_rom` owns and hands it straight to `AtariEnv`, which stores it
    # in a field. Nothing here needs an Any origin.
    rom: Pointer[UInt8, MutUntrackedOrigin],
    rom_size: Int,
) -> List[AtariPongPixel]:
    """N_ENVS independent emulator instances sharing the read-only ROM."""
    var envs = List[AtariPongPixel]()
    for _ in range(N_ENVS):
        envs.append(AtariPongPixel(AtariGame.PONG, rom, rom_size))
    return envs^


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on Atari 2600 Pong — Pixel")
    print("=" * 70)
    print()

    # Load ROM once; all env instances share the read-only buffer.
    print("Loading ROM:", ROM_PATH)
    var rom_data = load_rom(ROM_PATH)
    print("ROM loaded:", rom_data.size, "bytes")
    print()

    with DeviceContext() as ctx:
        # Whole agent from the preset — Nature-CNN backbone + noisy dueling
        # distributional heads + N-step-over-PER replay (uint8 obs ring, the
        # preset default). Config-tuned scalars (lr 6.25e-5, ε=0 noisy,
        # warmup 20k, PER α=0.5/β=0.4, nstep=N_STEP) apply; only the Pong
        # value support deviates.
        var agent = RainbowCNN[
            "gpu", NUM_ACTIONS, BATCH_SIZE, BUFFER_CAPACITY,
            FRAMES, NUM_ATOMS, HIDDEN, N_STEP, OBS_STORE_DT,
        ](
            ctx=ctx,
            lr=LR,
            learning_starts=WARMUP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = BatchedAtariPong(
            _make_envs(rom_data.data.value(), rom_data.size),
            noop_max=NOOP_MAX,
        )
        # Separate batched env for deterministic (noise-off) greedy eval.
        var eval_env = BatchedAtariPong(
            _make_envs(rom_data.data.value(), rom_data.size),
            noop_max=NOOP_MAX,
        )

        print(
            "Environment: Atari 2600 Pong (CPU emulation,",
            N_ENVS,
            "parallel envs, Pixel)",
        )
        print("Agent: Rainbow DQN CNN (deep_agents C51, GPU train)")
        print(
            "  Components: C51 + Double + PER + Dueling + Noisy +",
            N_STEP,
            "-step",
        )
        print("  Observation: 4 × 84 × 84 =", OBS_DIM)
        print(
            "  Actions:",
            NUM_ACTIONS,
            "(NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)",
        )
        print("  Network: Nature CNN + Noisy Dueling Distributional heads")
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP)
        print("  N envs (parallel CPU):", N_ENVS)
        print(
            "  Updates/iter:",
            UPDATES_PER_STEP,
            "(replay ratio",
            Float64(UPDATES_PER_STEP) / Float64(N_ENVS),
            ")",
        )
        print("  No-op starts: U[0,", NOOP_MAX, "]")
        print(
            "  Buffer capacity:",
            BUFFER_CAPACITY,
            "(GPU-resident, uint8 obs ring)",
        )
        print("  Batch size:", BATCH_SIZE)
        print("  Learning rate:", LR)
        print("  Warmup:", WARMUP)
        print("  Total transitions:", NUM_STEPS)
        print()

        # =====================================================================
        # Logger
        # =====================================================================

        var run = RunContext(
            project=String("atari"),
            driver=String("examples/arcade_games/rainbow_atari_pong_pixel_training_gpu.mojo"),
            slug=String("rainbow-atari-pong-pixel"),
            env=String("builtin:atari/pong"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents)")
        logger.set_config("env", "Atari Pong (Pixel)")
        logger.set_config("obs", "4x84x84")
        logger.set_config("lr", String(LR))
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("obs_store_dtype", "uint8")
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))
        logger.set_config("v_min", String(V_MIN))
        logger.set_config("v_max", String(V_MAX))
        logger.set_config("num_actions", String(NUM_ACTIONS))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("updates_per_step", String(UPDATES_PER_STEP))
        logger.set_config("noop_max", String(NOOP_MAX))

        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_cpu_batched[
                BatchedAtariPong, N_ENVS, N_STEP, RunLogger
            ](
                env,
                NUM_STEPS,
                rng_seed=UInt64(42),
                updates_per_step=UPDATES_PER_STEP,
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
                eval_episodes=N_ENVS,  # one parallel wave of full episodes
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            finish_run(
                run, logger, artifacts,
                String("mean_return_10=") + String(agent.mean_return()),
            )

            print("-" * 70)
            print()
            print("=" * 70)
            print("Rainbow Atari CNN GPU Training Complete")
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
