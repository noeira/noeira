"""SAC HalfCheetah — wall clock for a fixed env-step budget, noeira vs CleanRL.

Community-meeting benchmark: HalfCheetah has no "solved" threshold, so every
backend trains for the same number of env steps and reports the wall clock,
the env steps / s and the greedy return at the end. The PyTorch reference
with the same settings is `benchmarks/cleanrl_sac_half_cheetah.py`.

Hyper-parameters = CleanRL `sac_continuous_action.py` defaults: 1M replay,
batch 256, 5k uniform-random warm-up steps, gamma 0.99, tau 0.005, actor lr
3e-4, critic + alpha lr 1e-3, automatic alpha (init 1.0, target entropy
-|A| = -6), 256x256 ReLU actor and twin critics, one update per env step.
CleanRL runs with `--policy-frequency 1` (noeira has no delayed actor).

The clock covers the training loop only (not compilation, device init, the
final eval or the checkpoint write).

Targets (comptime):
  default         — CPU trainer + one CPU env (`train_single`)
  -D SAC_GPU      — GPU trainer + `Phyics3dBatchedEnv` on the device
  -D SAC_N_ENVS=N — batched GPU envs (default 1); one update per transition
                    (`updates_per_step = N`)
  -D SAC_ENV_GRAPH — replay the env's physics step from a CUDA graph
                    (`USE_ENV_CUDA_GRAPH`; NVIDIA only, run through `pixi run`
                    so the CUDA interceptor is preloaded)

Args (positional): [seed=1] [env_steps=200000] [checkpoint_path=""]

    pixi run mojo run -I . examples/half_cheetah/sac_half_cheetah_benchmark.mojo 1
    pixi run -e nvidia mojo build -I . -D SAC_GPU -D SAC_N_ENVS=32 \\
        examples/half_cheetah/sac_half_cheetah_benchmark.mojo -o sac_hc_gpu32
    ./sac_hc_gpu32 1
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv
from std.sys.defines import get_defined_int, is_defined
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.sac import SACAgent
from noeira.deep_agents.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig


comptime GPU = is_defined["SAC_GPU"]()
comptime N_ENVS = get_defined_int["SAC_N_ENVS", 1]()
comptime ENV_GRAPH = is_defined["SAC_ENV_GRAPH"]()

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  # 6
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
comptime LEARNING_STARTS = 5_000
comptime PRINT_EVERY = 10_000
comptime EVAL_EPISODES = 10

comptime EnvT = Phyics3dEnv[
    HalfCheetahModel, HalfCheetahConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime BatchedEnvT = Phyics3dBatchedEnv[
    HalfCheetahModel, HalfCheetahConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    var args = argv()
    var run_seed = Int(String(args[1])) if len(args) > 1 else 1
    var env_steps = Int(String(args[2])) if len(args) > 2 else 200_000
    var ckpt_path = String(args[3]) if len(args) > 3 else String("")
    seed(run_seed)

    var backend = String("CPU")
    comptime if GPU:
        backend = String("GPU")
    print("=" * 70)
    print("SAC HalfCheetah — wall clock for", env_steps, "env steps")
    print(
        "  backend:", backend, "| n_envs:", N_ENVS, "| env graph:", ENV_GRAPH,
        "| seed:", run_seed,
    )
    print("=" * 70)

    var wall_s: Float64 = 0.0
    var train_mean: Scalar[DT] = 0.0
    var eval_mean: Scalar[DT] = 0.0
    var ctx = DeviceContext()
    print("  device:", ctx.name())

    comptime if GPU:
        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=1e-3,
            alpha_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=1.0,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=LEARNING_STARTS,
            window_size=10,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        # The env graph is off by default, as in sac_half_cheetah_training_gpu.mojo.
        # On HalfCheetah the captured step replays bit-identically to eager
        # (benchmarks/physics3d_gpu/bench_half_cheetah_batch.mojo checks it).
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=ENV_GRAPH,
        ](
            env,
            env_steps,
            rng_seed=UInt64(run_seed),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            episode_sync_every=32,
        )
        ctx.synchronize()
        wall_s = Float64(perf_counter_ns() - t0) / 1e9
        train_mean = agent.mean_return()
        if ckpt_path.byte_length() > 0:
            agent.save(ckpt_path)
        var eval_env = EnvT(ctx)
        eval_mean = agent.eval(eval_env, EVAL_EPISODES)
    else:
        var agent = SACAgent[
            "cpu",
            UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            actor_lr=3e-4,
            critic_lr=1e-3,
            alpha_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=1.0,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=LEARNING_STARTS,
            window_size=10,
            initial_episode_fill=0.0,
        )
        var env = EnvT(ctx)
        var t0 = perf_counter_ns()
        _ = agent.train_single[EnvT](
            env, env_steps, print_every=PRINT_EVERY, verbose=True
        )
        wall_s = Float64(perf_counter_ns() - t0) / 1e9
        train_mean = agent.mean_return()
        if ckpt_path.byte_length() > 0:
            agent.save(ckpt_path)
        var eval_env = EnvT(ctx)
        eval_mean = agent.eval(eval_env, EVAL_EPISODES)

    print("=" * 70)
    print(
        "RESULT backend=noeira-" + backend,
        "n_envs=" + String(N_ENVS),
        "env_graph=" + String(ENV_GRAPH),
        "seed=" + String(run_seed),
        "env_steps=" + String(env_steps),
        "wall_s=" + String(wall_s),
        "sps=" + String(Int(Float64(env_steps) / wall_s)),
        "train_return_last10=" + String(train_mean),
        "greedy_eval_" + String(EVAL_EPISODES) + "ep=" + String(eval_mean),
    )
    if ckpt_path.byte_length() > 0:
        print("checkpoint:", ckpt_path)
    print("=" * 70)
