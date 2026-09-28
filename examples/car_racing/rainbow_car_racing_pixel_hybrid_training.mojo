"""Rainbow DQN CNN on CarRacing — HYBRID (CPU env stepped on host + GPU train).

The Atari-Pong "Stage-1" pattern: step N parallel CarRacingMB instances on CPU
cores via `BatchedCpuDiscreteEnv`, while the Rainbow CNN Q-network trains on the
GPU. Because the env IS the CPU `CarRacingMB` (faithful Gymnasium track + the
exact multi-body physics + the CPU pixel rasterizer), this:
  - transfers to CPU eval by construction (you train on the eval env), and
  - cannot be cheated — the faithful track is a proper closed loop (unlike the
    GPU embedded `_gen_track`, which is half-width and self-intersecting, which
    let the agent "cut to the end").

Observation: 4x84x84 grayscale (CarRacingMB[DT, PIXEL_OBS=True]). Slower than the
pure-GPU env (the render runs on CPU), so N_ENVS is smaller.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_training.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_training.mojo
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
from noeira.deep_agents.training import BatchedCpuDiscreteEnv
from noeira.envs.car_racing import CarRacingMB
from noeira.core.fmt import fit


# =============================================================================
# Constants
# =============================================================================

comptime CarRacingPx = CarRacingMB[DT, True]  # PIXEL_OBS=True
comptime OBS_DIM = CarRacingPx.EFF_OBS_DIM  # 4*84*84 = 28224
comptime NUM_ACTIONS = CarRacingPx.NUM_ACTIONS  # 5
comptime FRAMES = 4

comptime NUM_ATOMS = 101
comptime HIDDEN = 512
comptime N_STEP = 3

# uint8 obs storage → ~56.5 KB/transition (full 4-frame obs + next_obs, no
# dedup). 250k ≈ 14 GB VRAM, fits a 32 GB card with room to spare. At 48k the
# buffer fully recycled every ~6k grad-steps, evicting rare off-track/recovery
# transitions before the agent could learn from them; 250k retains them.
comptime BUFFER_CAPACITY = 250_000
comptime OBS_STORE_DT = DType.uint8
comptime BATCH_SIZE = 32
comptime N_ENVS = 8  # CPU-stepped (render on host) → fewer envs than the GPU env

# Value support must BRACKET the discounted return, not the per-step reward.
# Gymnasium reward (+1000/N ≈ +3.4/tile, -0.1/frame, -100 OOB) gives episode
# returns ~400 and discounted Q well above 30 — the old ±30 support saturated
# (mean_q pinned at ~28 = ceiling) and the policy collapsed periodically.
# [-100, 200] / 101 atoms → spacing 3 (≈ one tile reward; two-hot handles
# sub-spacing rewards). -100 covers the OOB penalty + accumulated time cost.
comptime V_MIN = Scalar[DT](-100.0)
comptime V_MAX = Scalar[DT](200.0)

# Replay ratio = UPDATES_PER_STEP / N_ENVS.
comptime UPDATES_PER_STEP = 2
comptime WARMUP = 20_000
comptime NUM_STEPS = 10_000_000
comptime LR = Scalar[DT](6.25e-5)

comptime CKPT_EVERY = 250_000

comptime BatchedCarRacing = BatchedCpuDiscreteEnv[CarRacingPx, N_ENVS, OBS_DIM]


def _make_envs() -> List[CarRacingPx]:
    var envs = List[CarRacingPx]()
    for _ in range(N_ENVS):
        envs.append(CarRacingMB[DT, True]())
    return envs^


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow CNN on CarRacing — HYBRID (CPU env + GPU train)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
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

        var env = BatchedCarRacing(_make_envs())
        var eval_env = BatchedCarRacing(_make_envs())

        print("Environment: CarRacingMB pixel (CPU-stepped,", N_ENVS, "envs)")
        print("Agent: Rainbow DQN CNN (GPU)")
        print("  Observation:", OBS_DIM, "= 4x84x84  Actions:", NUM_ACTIONS)
        print("  Hidden:", HIDDEN, " Atoms:", NUM_ATOMS, " support [", V_MIN, ",", V_MAX, "]")
        print("  N-step:", N_STEP, " Buffer:", BUFFER_CAPACITY, "(uint8)")
        print("  Updates/step:", UPDATES_PER_STEP, " LR:", LR, " Warmup:", WARMUP)
        print()

        var run = RunContext(
            project=String("box2d"),
            driver=String("examples/car_racing/rainbow_car_racing_pixel_hybrid_training.mojo"),
            slug=String("rainbow-car-racing-pixel"),
            env=String("builtin:box2d/car_racing"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("agent", "Rainbow DQN CNN (hybrid)")
        logger.set_config("env", "CarRacingMB pixel (CPU)")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("n_step", String(N_STEP))

        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        print("Starting hybrid training...")
        print("-" * 70)
        var start_time = perf_counter_ns()

        try:
            var _ep_returns = agent.train_cpu_batched[
                BatchedCarRacing, N_ENVS, N_STEP, RunLogger
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
                # 8 episodes is too few — a couple of -100 run-offs drag the
                # mean by ~25 each (e.g. eval reported -47 at 600k while the
                # train policy was at 452). 40 eps (~5 waves over N_ENVS) cuts
                # the eval-mean noise >2x without touching training.
                eval_episodes=40,
            )

            var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
            finish_run(
                run, logger, artifacts,
                String("mean_return_10=") + String(agent.mean_return()),
            )
            print("-" * 70)
            print("Hybrid Training Complete")
            print("Training time:", fit(String(elapsed_s), 6), "seconds")
            print("Final mean return (last 10):", agent.mean_return())

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)

    print(">>> main() completed normally <<<")
