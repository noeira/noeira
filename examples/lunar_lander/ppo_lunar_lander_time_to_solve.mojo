"""PPO LunarLander (discrete) — wall clock to a mean return >= 200.

Community-meeting benchmark: the same PPO hyper-parameters on CPU, Metal
and CUDA, stopped the first time the mean return over the last 100
episodes reaches 200. The PyTorch reference with the same settings is
`benchmarks/cleanrl_ppo_lunar_lander.py`.

Hyper-parameters are RL Baselines3 Zoo's tuned PPO LunarLander set:
16 envs x 1024 steps per update, minibatch 64, 4 epochs, gamma 0.999,
GAE lambda 0.98, entropy 0.01, lr 3e-4, grad-norm clip 0.5, 64x64 tanh
actor and critic.

The clock covers `train_batched` only: not compilation, not the device
context, not the evaluation or the checkpoint write after it.

Args (positional): [seed=1] [checkpoint_path=""] [max_env_steps=3000000]

    # CPU (envs stepped across cores, nets on the CPU)
    pixi run mojo run -I . examples/lunar_lander/ppo_lunar_lander_time_to_solve.mojo 1 ppo_ll_cpu_s1.ckpt
    # Metal
    pixi run -e apple mojo run -I . -D PPO_GPU examples/lunar_lander/ppo_lunar_lander_time_to_solve.mojo 1
    # CUDA
    pixi run -e nvidia mojo build -I . -D PPO_GPU examples/lunar_lander/ppo_lunar_lander_time_to_solve.mojo -o ppo_ll_gpu
    ./ppo_ll_gpu 1 ppo_ll_cuda_s1.ckpt

Render a checkpoint with `ppo_lunar_lander_render_gif.mojo`.
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv
from std.sys.defines import is_defined
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.deep_agents.ppo_discrete import PPODiscreteAgent
from noeira.deep_agents.ppo_discrete.config import (
    PPODiscreteActorNet,
    PPODiscreteCriticNet,
)
from noeira.deep_agents.training.batched_env import (
    BatchedCpuDiscreteEnv,
    BatchedGpuDiscreteEnv,
)
from noeira.envs.lunar_lander import LunarLander


comptime TARGET: StaticString = "gpu" if is_defined["PPO_GPU"]() else "cpu"

comptime OBS_DIM = 8
comptime N_ACTIONS = 4
comptime HIDDEN = 64
comptime N_ENVS = 16
comptime ROLLOUT_LEN = 1024
comptime MINIBATCH = 64
comptime N_EPOCHS = 4

comptime TARGET_RETURN: Scalar[DT] = 200.0
comptime WINDOW = 100  # episodes in the solved criterion
comptime EVAL_EPISODES = 20

comptime Agent = PPODiscreteAgent[
    TARGET,
    PPODiscreteActorNet[OBS_DIM, N_ACTIONS, HIDDEN],
    PPODiscreteCriticNet[OBS_DIM, HIDDEN],
    OBS_DIM, N_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS, N_ENVS,
]


def make_agent(ctx: Optional[DeviceContext]) raises -> Agent:
    return Agent(
        ctx=ctx,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.999,
        gae_lambda=0.98,
        clip_eps=0.2,
        entropy_coef=0.01,
        window_size=WINDOW,
        initial_episode_fill=0.0,
        max_grad_norm=0.5,
    )


def main() raises:
    var args = argv()
    var run_seed = Int(String(args[1])) if len(args) > 1 else 1
    var ckpt_path = String(args[2]) if len(args) > 2 else String("")
    var max_env_steps = Int(String(args[3])) if len(args) > 3 else 3_000_000
    seed(run_seed)

    var backend = String("CPU")
    comptime if TARGET == "gpu":
        backend = String("GPU")
    print("=" * 70)
    print("PPO LunarLander (discrete) — time to mean return >= 200")
    print("  backend:", backend, "| seed:", run_seed)
    print(
        "  envs", N_ENVS, "x rollout", ROLLOUT_LEN, "| minibatch", MINIBATCH,
        "| epochs", N_EPOCHS, "| budget", max_env_steps, "env steps",
    )
    print("=" * 70)

    var solved_mean: Scalar[DT] = 0.0
    var episodes = 0
    var wall_s: Float64 = 0.0
    var eval_mean: Scalar[DT] = 0.0

    comptime if TARGET == "gpu":
        var ctx = DeviceContext()
        print("  device:", ctx.name())
        var agent = make_agent(ctx)
        var env = BatchedGpuDiscreteEnv[
            LunarLander[DT], N_ENVS, OBS_DIM, 1
        ](ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        _ = agent.train_batched(
            ctx, env, max_env_steps,
            rng_seed=UInt64(run_seed),
            print_every=50_000,
            stop_at_mean_return=TARGET_RETURN,
            stop_min_episodes=WINDOW,
        )
        ctx.synchronize()
        wall_s = Float64(perf_counter_ns() - t0) / 1e9
        solved_mean = agent.mean_return()
        episodes = agent.ep_count()
        if ckpt_path.byte_length() > 0:
            agent.save(ckpt_path)
        var eval_env = LunarLander[DT](seed=UInt64(10_000 + run_seed))
        eval_mean = agent.eval(eval_env, EVAL_EPISODES)
    else:
        var agent = make_agent(None)
        var envs = List[LunarLander[DT]]()
        for i in range(N_ENVS):
            envs.append(LunarLander[DT](seed=UInt64(run_seed * 1000 + i)))
        var env = BatchedCpuDiscreteEnv[LunarLander[DT], N_ENVS, OBS_DIM](
            envs^
        )
        var t0 = perf_counter_ns()
        _ = agent.train_batched(
            None, env, max_env_steps,
            rng_seed=UInt64(run_seed),
            print_every=50_000,
            stop_at_mean_return=TARGET_RETURN,
            stop_min_episodes=WINDOW,
        )
        wall_s = Float64(perf_counter_ns() - t0) / 1e9
        solved_mean = agent.mean_return()
        episodes = agent.ep_count()
        if ckpt_path.byte_length() > 0:
            agent.save(ckpt_path)
        var eval_env = LunarLander[DT](seed=UInt64(10_000 + run_seed))
        eval_mean = agent.eval(eval_env, EVAL_EPISODES)

    var solved = episodes >= WINDOW and solved_mean >= TARGET_RETURN
    print("=" * 70)
    print(
        "RESULT backend=" + backend,
        "seed=" + String(run_seed),
        "solved=" + String(solved),
        "wall_s=" + String(wall_s),
        "episodes=" + String(episodes),
        "mean_return_last100=" + String(solved_mean),
        "greedy_eval_mean_" + String(EVAL_EPISODES) + "ep=" + String(eval_mean),
    )
    if ckpt_path.byte_length() > 0:
        print("checkpoint:", ckpt_path)
    print("=" * 70)
