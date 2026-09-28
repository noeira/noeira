"""Rainbow DQN CNN GPU training on Procgen Maze — pixel obs, CPU-batched envs.

Scales the CPU proof to a real run: `N_ENVS` maze envs step in parallel across CPU
cores (`BatchedCpuDiscreteEnv`) while the Nature-CNN Rainbow agent selects actions
and trains on the GPU (`RainbowCNN["gpu"]` + `agent.train_cpu_batched`), with a
GPU-resident uint8 obs replay. Same architecture as the Atari-Pong pixel trainer
(`rainbow_atari_pong_pixel_training_gpu.mojo`) — used because maze generation +
sprite rendering are CPU-side (no GPU-env kernels), exactly like the Atari
emulator. A full GPU-resident maze env (`GPUDiscreteEnv`) is a separate,
much larger effort.

Procgen generalization setup: each parallel env gets a distinct `rand_seed`, so
they sample different levels from the shared train set (`num_levels`) — this both
covers the level distribution and de-correlates the parallel rollouts.

**Start with a `num_levels` sweep (this file defaults to 1).** Maze is a hard
sparse-reward generalization task: 200 distinct pixel mazes need ~10-200M steps
(the paper uses 200M). Confirm the GPU pipeline learns on ONE level first
(`num_levels=1` → `eval/mean_return` should climb toward ~10, matching the CPU
proof), then step up 1 → 10 → 50 → 200. Each level-count increase needs
proportionally more steps; at a fixed budget, `mean_return` falls off as levels
grow (that's the generalization curve, not a bug).

Run:
    pixi run -e apple  mojo run -I . examples/procgen/maze_rainbow_training_gpu.mojo   # compile/smoke
    pixi run -e nvidia mojo run -I . examples/procgen/maze_rainbow_training_gpu.mojo   # training

Heavy training should run on NVIDIA (Apple Metal is flaky under sustained load);
the CNN q-net + optimizer are checkpointed to the run's `checkpoints/last.ckpt`
(a `RunContext`, project `procgen`; `run_logger` writes `metrics.csv` + the monitor).
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import Pointer, ArcPointer
from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT

from noeira.deep_agents.c51.config import RainbowCNN
from noeira.deep_agents.training.batched_env import BatchedCpuDiscreteEnv

from noeira.envs.procgen.games import MazeGymEnv, MazeAssets
from noeira.envs.procgen.games.maze import DIST_EASY
from noeira.core.fmt import fit

comptime ASSET_ROOT = String("assets/procgen/")

comptime MazeCNNEnv = MazeGymEnv[DT]
comptime OBS_DIM = MazeCNNEnv.OBS_DIM  # 3 × 84 × 84 = 21168
comptime NUM_ACTIONS = MazeCNNEnv.NUM_ACTIONS  # 15

comptime FRAMES = 3  # RGB channels (single frame — maze is fully observable)
comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

# Maze envs stepped in parallel across CPU cores; GPU trains the CNN.
comptime N_ENVS = 16
comptime UPDATES_PER_STEP = 4  # replay ratio = 4 / 16 = 0.25
comptime SEED_BASE = 1000  # per-env rand_seed = SEED_BASE + i (distinct levels)

# Procgen generalization: train on a finite level set, Easy difficulty.
# SWEEP KNOB — 1 (trivial: seed-0 Easy is a tiny 3×3 maze) confirmed learning;
# 10 spans seeds 0..9 (maze_dim up to 15), a real multi-maze test; then 50, 200.
comptime DIST_MODE = DIST_EASY
comptime NUM_LEVELS = 10

# GPU-resident uint8 obs ring (pixel obs are exact k/255 → lossless).
comptime BUFFER_CAPACITY = 50_000
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32

# Value support must bracket the discounted return: 0 (timeout) .. ~10 (goal).
comptime V_MIN = Scalar[DT](-1.0)
comptime V_MAX = Scalar[DT](11.0)

comptime WARMUP = 10_000
comptime NUM_STEPS = 1_000_000  # single level converges well inside this
comptime LR = Scalar[DT](6.25e-5)

comptime CKPT_EVERY = 250_000

comptime BatchedMaze = BatchedCpuDiscreteEnv[MazeCNNEnv, N_ENVS, OBS_DIM]


def _make_envs(assets: ArcPointer[MazeAssets]) -> List[MazeCNNEnv]:
    """N_ENVS independent maze envs sharing one read-only asset bundle, each
    with a distinct rand_seed so they sample different levels from the train set.
    """
    var envs = List[MazeCNNEnv]()
    for i in range(N_ENVS):
        envs.append(
            MazeCNNEnv(
                assets,
                rand_seed=SEED_BASE + i,
                num_levels=NUM_LEVELS,
                start_level=0,
                dist_mode=DIST_MODE,
            )
        )
    return envs^


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on Procgen Maze — Pixel (CPU-batched)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
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
        ](
            ctx=ctx,
            lr=LR,
            learning_starts=WARMUP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        # Load the sprite set ONCE; all train + eval envs share it.
        var assets = ArcPointer(MazeAssets(ASSET_ROOT))
        var env = BatchedMaze(_make_envs(assets))
        var eval_env = BatchedMaze(_make_envs(assets))

        print(
            "Environment: Procgen Maze (CPU-batched,",
            N_ENVS,
            "parallel envs, Easy,",
            NUM_LEVELS,
            "levels)",
        )
        print("Agent: Rainbow DQN CNN (deep_agents C51, GPU train)")
        print("  Observation:", FRAMES, "× 84 × 84 =", OBS_DIM, "(NCHW)")
        print("  Actions:", NUM_ACTIONS)
        print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
        print("  N envs (parallel CPU):", N_ENVS)
        print(
            "  Updates/iter:",
            UPDATES_PER_STEP,
            "(replay ratio",
            Float64(UPDATES_PER_STEP) / Float64(N_ENVS),
            ")",
        )
        print("  Buffer:", BUFFER_CAPACITY, "(GPU-resident, uint8 obs ring)")
        print("  Batch:", BATCH_SIZE, " N-step:", N_STEP, " lr:", LR)
        print("  Warmup:", WARMUP, " Total steps:", NUM_STEPS)
        var run = RunContext(
            project=String("procgen"),
            driver=String("examples/procgen/maze_rainbow_training_gpu.mojo"),
            slug=String("rainbow-maze"),
            env=String("builtin:procgen/maze"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir, "(checkpoint every", CKPT_EVERY, "steps)")
        print()

        var logger = run_logger(run, buffer_size=64)
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents, GPU)")
        logger.set_config("env", "procgen-maze-easy")
        logger.set_config("obs", "3x84x84")
        logger.set_config("num_levels", String(NUM_LEVELS))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("lr", String(LR))
        logger.set_config("n_step", String(N_STEP))

        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        print("Starting GPU training...")
        print("-" * 70)
        var start = perf_counter_ns()
        try:
            var _ret = agent.train_cpu_batched[
                BatchedMaze, N_ENVS, N_STEP, RunLogger
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
                eval_episodes=N_ENVS,
            )
            var elapsed = Float64(perf_counter_ns() - start) / 1e9
            finish_run(
                run, logger, artifacts,
                String("mean_return_10=") + String(agent.mean_return()),
            )
            print("-" * 70)
            print("Done in", fit(String(elapsed), 8), "s")
            print("Final mean return (last 10):", agent.mean_return())
            print("Episodes:", agent.ep_count())
        except e:
            print("!!! EXCEPTION:", e)

    print(">>> main() completed <<<")
