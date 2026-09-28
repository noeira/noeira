"""Rainbow DQN GPU training on Atari Pong — RAM mode, GPU-emulated env.

The Stage-2a payoff (docs/ATARI_AUDIT.md §3): trains Rainbow on the REAL Atari
2600 Pong ROM, with `N_ENVS` emulators stepping on the GPU (one env per thread,
~184K steps/s on NVIDIA) and the Q-network training on the same device — through
the identical `train_gpu_batched` path that converged on the native Pong
clean-obs engine.

RAM mode: obs = the 128-byte console RAM / 255 (a small MLP, not a CNN — which is
what lets the GPU emulator's throughput translate into training throughput;
pixel obs would re-introduce the 28K-float obs-movement bottleneck). Reward =
score delta (±1 per point); 6 actions (ALE minimal set); is_terminal at 21.

Config mirrors the converged clean-obs Pong recipe: V_MIN/V_MAX ±2 (the support
lever), lr 6.25e-5, warmup 20k, N-step 3, replay ratio 0.25. Differences for RAM:
OBS_DIM 128 (vs 6), NUM_ACTIONS 6 (vs 3), HIDDEN 256 (vs 128).

    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_atari_pong_ram_training_gpu.mojo  # compile/smoke
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_atari_pong_ram_training_gpu.mojo  # training

Requires roms/pong.bin.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import Pointer

from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT

from noeira.deep_agents.c51.config import Rainbow
from noeira.deep_agents.training.atari_gpu_env import AtariGpuBatchedEnv
from noeira.envs.atari.environment import load_rom
from noeira.envs.atari.games import PongDef
from noeira.core.fmt import fit


# RAM-mode Atari Pong: 128-byte RAM obs, 6 actions (ALE minimal set).
comptime OBS_DIM = 128
comptime NUM_ACTIONS = PongDef.NUM_ACTIONS  # 6

# Rainbow architecture / replay.
comptime HIDDEN_DIM = 256  # RAM is 128-D opaque (vs 6-D clean obs)
comptime NUM_ATOMS = 51
comptime N_STEP = 3
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 64
comptime N_ENVS = 256  # parallel GPU emulators (raise once converged)

# C51 support must bracket the *discounted* return, not raw score: with γ=0.99 +
# sparse ±1 point rewards the discounted Q lives in roughly ±0.3..±6 — [-2,2] is
# the proven Pong support (the lever that converged the clean-obs run).
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

# Replay ratio = GRAD_STEPS / N_ENVS = 64/256 = 0.25 (CleanRL Atari freq 4).
comptime GRAD_STEPS = 64
comptime WARMUP = 20_000
comptime NUM_STEPS = 10_000_000
comptime LR = Scalar[DT](6.25e-5)

# Env: real frame_skip 4, ALE random no-op starts (0..30), 108k-frame cap.
comptime FRAME_SKIP = 4
comptime NOOP_MAX = 30
comptime MAX_FRAMES = 108_000

comptime CKPT_EVERY = 500_000

comptime AtariBatched = AtariGpuBatchedEnv[
    PongDef, N_ENVS, FRAME_SKIP, NOOP_MAX, MAX_FRAMES
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN GPU Training — Atari Pong (RAM mode, GPU emulator)")
    print("=" * 70)

    var rom = load_rom("roms/pong.bin")
    var rom_ptr = rom.data.value()
    var rom_size = rom.size

    with DeviceContext() as ctx:
        var agent = Rainbow[
            "gpu", OBS_DIM, NUM_ACTIONS, BATCH_SIZE, BUFFER_CAPACITY,
            NUM_ATOMS, HIDDEN_DIM, N_STEP,
        ](
            ctx=ctx,
            lr=LR,
            learning_starts=WARMUP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        var env = AtariBatched(ctx, rom_ptr, rom_size)
        var eval_env = AtariBatched(ctx, rom_ptr, rom_size)

        print("Environment: Atari Pong RAM (GPU-emulated,", N_ENVS, "envs)")
        print("  Obs dim:", OBS_DIM, "(128-byte RAM / 255)")
        print("  Actions:", NUM_ACTIONS, "(ALE minimal set)")
        print("  Hidden:", HIDDEN_DIM, "| Atoms:", NUM_ATOMS,
              "support [", V_MIN, ",", V_MAX, "] | N-step:", N_STEP)
        print("  Frame skip:", FRAME_SKIP, "| no-op starts 0..", NOOP_MAX)
        print("  Grad steps/iter:", GRAD_STEPS, "(replay ratio 0.25)")
        print("Expected: random ~-21 → good policy > 0 (beating CPU)")
        print("-" * 70)

        var run = RunContext(
            project=String("atari"),
            driver=String("examples/arcade_games/rainbow_atari_pong_ram_training_gpu.mojo"),
            slug=String("rainbow-atari-pong-ram"),
            env=String("builtin:atari/pong"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("agent", "Rainbow DQN (deep_agents)")
        logger.set_config("env", "Atari Pong RAM (GPU emulator)")
        logger.set_config("obs_dim", String(OBS_DIM))
        logger.set_config("num_actions", String(NUM_ACTIONS))
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("lr", String(LR))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("n_step", String(N_STEP))

        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var start_time = perf_counter_ns()
        try:
            var _ep_returns = agent.train_gpu_batched[
                AtariBatched, N_ENVS, N_STEP, RunLogger
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
                eval_episodes=20,
            )
            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            finish_run(
                run, logger, artifacts,
                String("mean_return_10=") + String(agent.mean_return()),
            )
            print("-" * 70)
            print("Training complete in", fit(String(elapsed_s), 8), "s")
            print("Final mean return (last 10):", agent.mean_return())
            print("Episodes completed:", agent.ep_count())
        except e:
            print("!!! EXCEPTION:", e)

    print(">>> done <<<")
