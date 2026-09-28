"""Rainbow DQN CNN GPU training on Procgen Bossfight — pixel obs, CPU-batched envs.

Scales the CPU proof to a real run: `N_ENVS` bossfight envs step in parallel across
CPU cores (`BatchedCpuDiscreteEnv`) while the Nature-CNN Rainbow agent selects
actions and trains on the GPU (`RainbowCNN["gpu"]` + `agent.train_cpu_batched`),
with a GPU-resident uint8 obs replay. Same path as the maze/plunder/Atari-Pong
pixel trainers — used because bossfight generation + rendering are CPU-side (no
GPU-env kernels). A full GPU-resident bossfight env is a separate, larger effort.

Bossfight reward is +1 per boss health chunk cleared and +10 for the kill, with a
hard death on any collision → harder than the dense-reward shooters; start with
`num_levels=1` and a longer horizon.

Procgen generalization setup: each parallel env gets a distinct `rand_seed`, so
they sample different levels from the shared train set (`num_levels`) — covering
the level distribution and de-correlating the parallel rollouts.

**Start with a `num_levels` sweep (this file defaults to 1).** Confirm the GPU
pipeline learns on ONE level first (`num_levels=1` → `eval/mean_return` should
climb as it clears boss rounds), then step up 1 → 10 → 50 → 200. Each level-count
increase needs proportionally more steps.

Run:
    pixi run mojo run -I . examples/procgen/bossfight_rainbow_training_gpu.mojo   # (apple) compile/smoke
    pixi run -e nvidia mojo run -I . examples/procgen/bossfight_rainbow_training_gpu.mojo   # training

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

from noeira.envs.procgen.games import BossfightGymEnv, BossfightAssets
from noeira.envs.procgen.games.bossfight import DIST_EASY
from noeira.core.fmt import fit

comptime ASSET_ROOT = String("assets/procgen/")

comptime BossfightCNNEnv = BossfightGymEnv[DT]
comptime OBS_DIM = BossfightCNNEnv.OBS_DIM  # 3 × 84 × 84 = 21168
comptime NUM_ACTIONS = BossfightCNNEnv.NUM_ACTIONS  # 15

comptime FRAMES = 3  # RGB channels (single frame — bossfight is fully observable)
comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3

# Bossfight envs stepped in parallel across CPU cores; GPU trains the CNN.
comptime N_ENVS = 16
comptime UPDATES_PER_STEP = 4  # replay ratio = 4 / 16 = 0.25
comptime SEED_BASE = 1000  # per-env rand_seed = SEED_BASE + i (distinct levels)

# Procgen generalization: train on a finite level set, Easy difficulty.
# SWEEP KNOB — start at 1 (confirm the GPU pipeline learns), then 10, 50, 200.
comptime DIST_MODE = DIST_EASY
comptime NUM_LEVELS = 1

# GPU-resident uint8 obs ring (pixel obs are exact k/255 → lossless).
comptime BUFFER_CAPACITY = 50_000
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32

# Value support: -1 .. ~16 (up to 5 rounds cleared at +1 plus the +10 kill bonus).
comptime V_MIN = Scalar[DT](-1.0)
comptime V_MAX = Scalar[DT](16.0)

comptime WARMUP = 10_000
comptime NUM_STEPS = 1_000_000  # single level converges well inside this
comptime LR = Scalar[DT](6.25e-5)

comptime CKPT_EVERY = 250_000

comptime BatchedBossfight = BatchedCpuDiscreteEnv[BossfightCNNEnv, N_ENVS, OBS_DIM]


def _make_envs(assets: ArcPointer[BossfightAssets]) -> List[BossfightCNNEnv]:
    """N_ENVS independent bossfight envs sharing one read-only asset bundle, each
    with a distinct rand_seed so they sample different levels from the train set.
    """
    var envs = List[BossfightCNNEnv]()
    for i in range(N_ENVS):
        envs.append(
            BossfightCNNEnv(
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
    print("Rainbow DQN CNN GPU Training on Procgen Bossfight — Pixel (CPU-batched)")
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
        var assets = ArcPointer(BossfightAssets(ASSET_ROOT))
        var env = BatchedBossfight(_make_envs(assets))
        var eval_env = BatchedBossfight(_make_envs(assets))

        print(
            "Environment: Procgen Bossfight (CPU-batched,",
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
            driver=String("examples/procgen/bossfight_rainbow_training_gpu.mojo"),
            slug=String("rainbow-bossfight"),
            env=String("builtin:procgen/bossfight"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir, "(checkpoint every", CKPT_EVERY, "steps)")
        print()

        var logger = run_logger(run, buffer_size=64)
        logger.set_config("agent", "Rainbow DQN CNN (deep_agents, GPU)")
        logger.set_config("env", "procgen-bossfight-easy")
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
                BatchedBossfight, N_ENVS, N_STEP, RunLogger
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
