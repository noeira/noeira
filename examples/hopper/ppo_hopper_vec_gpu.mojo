"""PPO on Hopper through the generic vectorised driver (`run_ppo_vec`) — its
learning check: CleanRL's MuJoCo recipe (batch 2048, 32 minibatches,
10 epochs, lr 3e-4 annealed, obs + reward normalisation), the rollout on the
GPU.

    pixi run -e nvidia mojo build -I . -D PPO_VEC_ENV_GRAPH -D PPO_VEC_TRAIN_GRAPH \\
        examples/hopper/ppo_hopper_vec_gpu.mojo -o ppo_hopper_vec
    pixi run -e nvidia ./ppo_hopper_vec --steps 2000000 --seed 1

`-D PPO_VEC_HIST=K` appends the last K actions to the observation;
`--act-rate W` charges W |a_t - a_{t-1}|^2 per step. CleanRL's Hopper-v4
reaches ~2000-2500 by 1M steps (one env).
"""

from std.random import seed as seed_rng
from std.sys import argv, get_defined_int, is_defined

from max.gpu.host import DeviceContext

from noeira.core.logger import NoOpLogger
from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.deep_agents.training.obs_norm import RunningMeanStd
from noeira.deep_agents.training.ppo_vec_driver import (
    PPOVecConfig, run_ppo_vec,
)
from noeira.envs.hopper import HopperModel, HopperConfig
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.primitives.activations import Tanh
from noeira.nn.primitives.linear import Linear


comptime N_ENVS = 128
comptime ROLLOUT = 16
comptime N_MINIBATCHES = 32
comptime MINIBATCH = N_ENVS * ROLLOUT // N_MINIBATCHES
comptime EPOCHS = 10
comptime HID = 64
comptime HIST = get_defined_int["PPO_VEC_HIST", 0]()
comptime ENV_GRAPH = is_defined["PPO_VEC_ENV_GRAPH"]()
comptime TRAIN_GRAPH = is_defined["PPO_VEC_TRAIN_GRAPH"]()
comptime EnvT = Phyics3dBatchedEnv[
    HopperModel, HopperConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=True
]
comptime ACT = EnvT.ACT_DIM
comptime OBS = EnvT.OBS_DIM + HIST * ACT
comptime Actor = Sequential[
    Linear[OBS, HID], Tanh[HID], Linear[HID, HID], Tanh[HID],
    GaussianHead[HID, ACT],
]
comptime Critic = Sequential[
    Linear[OBS, HID], Tanh[HID], Linear[HID, HID], Tanh[HID], Linear[HID, 1],
]


def _arg(name: String, default: String) raises -> String:
    var a = argv()
    for i in range(len(a) - 1):
        if String(a[i]) == name:
            return String(a[i + 1])
    return default


def main() raises:
    var steps = Int(_arg("--steps", "2000000"))
    var seed = Int(_arg("--seed", "1"))
    var act_rate = Float64(_arg("--act-rate", "0"))
    seed_rng(seed)
    print("PPO (run_ppo_vec) on Hopper | lanes", N_ENVS, "| rollout", ROLLOUT,
          "| minibatch", MINIBATCH, "x", N_MINIBATCHES, "| epochs", EPOCHS,
          "| hist", HIST, "| env graph", ENV_GRAPH, "| train graph", TRAIN_GRAPH,
          "| steps", steps, "| seed", seed)
    with DeviceContext() as ctx:
        var agent = PPOAgent[
            "gpu", Actor, Critic, OBS, ACT, ROLLOUT, MINIBATCH, EPOCHS, N_ENVS
        ](
            ctx=ctx, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99,
            gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.0,
            action_scale=1.0, log_std_init=0.0, max_grad_norm=0.5,
        )
        var env = EnvT(ctx)
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed))
        ctx.synchronize()
        var rms = RunningMeanStd(OBS)
        var logger = NoOpLogger()
        var cfg = PPOVecConfig(
            total_steps=steps, lr=3e-4, anneal=True, act_rate_w=act_rate,
            seed=seed, print_every=20,
        )
        var res = run_ppo_vec[HIST=HIST, ENV_GRAPH=ENV_GRAPH, TRAIN_GRAPH=TRAIN_GRAPH](
            agent, env, ctx, cfg, rms, logger,
            String("/tmp/ppo_hopper_vec_s") + String(seed) + ".ckpt",
            String("/tmp/ppo_hopper_vec_s") + String(seed) + "_obs_norm.txt",
        )
        print("RESULT steps", res.steps, "| seconds", res.seconds,
              "| episodes", res.episodes, "| return (last 100)", res.mean_return,
              "| length", res.mean_length, "| diverged", res.diverged,
              "|", Int(Float64(res.steps) / res.seconds), "steps/s")
