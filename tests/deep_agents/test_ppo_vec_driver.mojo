"""`run_ppo_vec`: the env / train CUDA graphs replay the eager run bit for bit,
with every option on (action history, both normalisations, annealing — the
update graph re-captured after each schedule step — and the action-rate
penalty), on Hopper, whose falls are true terminations.

Gates, each against an independent fact:
  - the eager run and the graph run write byte-identical checkpoints and
    observation statistics (same seeds; the graphs replay the same kernels);
  - the run is not vacuous: episodes ended (Hopper falls well inside the
    budget), the statistics moved off their initial count, and the weights
    moved off their initialisation.

    pixi run -e nvidia mojo run -I . tests/deep_agents/test_ppo_vec_driver.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_true

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


comptime N_ENVS = 64
comptime ROLLOUT = 16
comptime MINIBATCH = N_ENVS * ROLLOUT // 4
comptime EPOCHS = 2
comptime HIST = 2
comptime HID = 64
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
comptime AgentT = PPOAgent[
    "gpu", Actor, Critic, OBS, ACT, ROLLOUT, MINIBATCH, EPOCHS, N_ENVS
]
comptime STEPS = N_ENVS * ROLLOUT * 4


def _read(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def _cfg() -> PPOVecConfig:
    return PPOVecConfig(
        total_steps=STEPS, lr=3e-4, ent0=0.01, ent1=0.001, anneal=True,
        act_rate_w=0.05, seed=7, print_every=1,
    )


def _run[ENV_GRAPH: Bool, TRAIN_GRAPH: Bool](
    ctx: DeviceContext, tag: String
) raises -> Tuple[Int, Float64]:
    seed(11)
    var agent = AgentT(
        ctx=ctx, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2, entropy_coef=0.01, action_scale=1.0, log_std_init=-0.5,
        max_grad_norm=0.5,
    )
    var init_path = String("/tmp/ppo_vec_init_") + tag + ".ckpt"
    agent.trainer.save_state(init_path)
    var env = EnvT(ctx)
    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(3))
    ctx.synchronize()
    var rms = RunningMeanStd(OBS)
    var logger = NoOpLogger()
    var res = run_ppo_vec[HIST=HIST, ENV_GRAPH=ENV_GRAPH, TRAIN_GRAPH=TRAIN_GRAPH](
        agent, env, ctx, _cfg(), rms, logger,
        String("/tmp/ppo_vec_") + tag + ".ckpt",
        String("/tmp/ppo_vec_") + tag + "_obs_norm.txt",
    )
    print(" ", tag, "| episodes", res.episodes, "| mean return", res.mean_return,
          "| mean length", res.mean_length, "| diverged", res.diverged)
    assert_true(res.episodes > N_ENVS, tag + ": too few episodes ended")
    assert_true(rms.count > 1.0, tag + ": the statistics never updated")
    var a = _read(init_path)
    var b = _read(String("/tmp/ppo_vec_") + tag + ".ckpt")
    var moved = len(a) != len(b)
    if not moved:
        for i in range(len(a)):
            if a[i] != b[i]:
                moved = True
                break
    assert_true(moved, tag + ": the weights never moved")
    return (res.episodes, res.mean_return)


def main() raises:
    print("--- run_ppo_vec: eager vs env + train graphs, Hopper ---")
    with DeviceContext() as ctx:
        var e = _run[False, False](ctx, "eager")
        var g = _run[True, True](ctx, "graph")
        assert_true(e[0] == g[0], "episode counts differ")
        for suffix in [".ckpt", "_obs_norm.txt"]:
            var x = _read(String("/tmp/ppo_vec_eager") + suffix)
            var y = _read(String("/tmp/ppo_vec_graph") + suffix)
            var same = len(x) == len(y)
            if same:
                for i in range(len(x)):
                    if x[i] != y[i]:
                        same = False
                        break
            assert_true(same, String("eager and graph runs differ: ") + suffix)
            print("  identical:", suffix, len(x), "bytes")
    print("ALL PASSED")
