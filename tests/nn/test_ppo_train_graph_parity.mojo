"""PPO's CUDA-graph update (`USE_TRAIN_CUDA_GRAPH`) trains bit-identically to
the eager update — discrete (LunarLander) and continuous (HalfCheetah).

The graph path moves the K-epoch minibatch work onto the device (pool and
indices uploaded once per rollout, a gather kernel, device losses, the device
grad clip) and captures one minibatch step. None of that may change the
numbers: the same seed, trained once through `train_step` and once through the
capture surface, must leave byte-identical checkpoints.

Each pair trains in this process from the same host seed, with its own agent
and env, `max_grad_norm` on (so the device clip is exercised). Asserted:
  - the two checkpoints are byte-identical;
  - both completed the same episodes;
  - both ran `updates * N_EPOCHS * minibatches` updates — so a path that
    silently skipped its updates cannot pass by leaving two untrained nets
    equal.

The DEVICE-RESIDENT rollout (`DEVICE_ROLLOUT`) samples with a device RNG,
so it cannot match the host rollout; its own claim is checked instead: the
device rollout with nothing captured against the same with the train, env
step and env reset graphs all captured — byte-identical checkpoints, the same
episodes, the expected update count, and episodes > 0 on LunarLander (the
deferred episode readback actually delivered).

On NVIDIA run it through `pixi run` (the CUDA interceptor), so the graph is
really captured and replayed — it prints `captured N nodes`. Elsewhere the
device path runs eagerly; the parity claim is the same.

    pixi run -e nvidia mojo run -I . tests/nn/test_ppo_train_graph_parity.mojo
    pixi run -e apple  mojo run -I . tests/nn/test_ppo_train_graph_parity.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.testing import assert_equal, assert_true

from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import _read_file_bytes
from noeira.deep_agents.ppo import PPOAgent, PPOActorNet, PPOCriticNet
from noeira.deep_agents.ppo_discrete import PPODiscreteAgent
from noeira.deep_agents.ppo_discrete.config import (
    PPODiscreteActorNet,
    PPODiscreteCriticNet,
)
from noeira.deep_agents.training.batched_env import BatchedGpuDiscreteEnv
from noeira.envs.lunar_lander import LunarLander
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig


comptime N_ENVS = 8
comptime ROLLOUT_LEN = 32
comptime MINIBATCH = 32
comptime N_EPOCHS = 2
comptime UPDATES = 3
comptime STEPS = ROLLOUT_LEN * N_ENVS * UPDATES
comptime TRAIN_STEPS = UPDATES * N_EPOCHS * (ROLLOUT_LEN * N_ENVS // MINIBATCH)

comptime LL_OBS = 8
comptime LL_ACTIONS = 4
comptime DiscAgent = PPODiscreteAgent[
    "gpu",
    PPODiscreteActorNet[LL_OBS, LL_ACTIONS, 64],
    PPODiscreteCriticNet[LL_OBS, 64],
    LL_OBS, LL_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS, N_ENVS,
]
comptime DiscEnv = BatchedGpuDiscreteEnv[LunarLander[DT], N_ENVS, LL_OBS, 1]

comptime HC_OBS = HalfCheetahConfig.OBS_DIM
comptime HC_ACT = HalfCheetahConfig.ACTION_DIM
comptime ContAgent = PPOAgent[
    "gpu",
    PPOActorNet[HC_OBS, HC_ACT, 64],
    PPOCriticNet[HC_OBS, 64],
    HC_OBS, HC_ACT, ROLLOUT_LEN, MINIBATCH, N_EPOCHS, N_ENVS,
]
comptime ContEnv = Phyics3dBatchedEnv[
    HalfCheetahModel, HalfCheetahConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]


def _train_discrete[GRAPH: Bool, DEVICE: Bool = False](
    ctx: DeviceContext, path: String
) raises -> Tuple[Int, Int]:
    seed(11)
    var agent = DiscAgent(
        ctx=ctx, clip_eps=0.2, entropy_coef=0.01, max_grad_norm=0.5
    )
    var env = DiscEnv(ctx)
    _ = agent.train_batched[
        USE_TRAIN_CUDA_GRAPH=GRAPH,
        DEVICE_ROLLOUT=DEVICE,
        USE_ENV_CUDA_GRAPH=DEVICE and GRAPH,
    ](ctx, env, STEPS, rng_seed=UInt64(5), verbose=False, episode_sync_every=4)
    ctx.synchronize()
    agent.save(path)
    return (agent.ep_count(), agent.trainer.total_train_steps())


def _train_continuous[GRAPH: Bool, DEVICE: Bool = False](
    ctx: DeviceContext, path: String
) raises -> Tuple[Int, Int]:
    seed(11)
    var agent = ContAgent(ctx=ctx, max_grad_norm=0.5)
    var env = ContEnv(ctx)
    _ = agent.train[
        USE_TRAIN_CUDA_GRAPH=GRAPH,
        DEVICE_ROLLOUT=DEVICE,
        USE_ENV_CUDA_GRAPH=DEVICE and GRAPH,
    ](env, STEPS, rng_seed=UInt64(5), verbose=False, episode_sync_every=4)
    ctx.synchronize()
    agent.save(path)
    return (agent.ep_count(), agent.trainer.total_train_steps())


def _assert_same(
    name: String,
    eager: Tuple[Int, Int],
    graph: Tuple[Int, Int],
    eager_path: String,
    graph_path: String,
) raises:
    print(
        " ", name, "| episodes", eager[0], "/", graph[0],
        "| updates", eager[1], "/", graph[1],
    )
    assert_equal(eager[1], TRAIN_STEPS, name + ": eager update count")
    assert_equal(graph[1], TRAIN_STEPS, name + ": graph update count")
    assert_equal(eager[0], graph[0], name + ": episodes differ")
    var a = _read_file_bytes(eager_path)
    var b = _read_file_bytes(graph_path)
    assert_equal(len(a), len(b), name + ": checkpoint sizes differ")
    var first_diff = -1
    for i in range(len(a)):
        if a[i] != b[i]:
            first_diff = i
            break
    assert_true(
        first_diff < 0,
        name + ": checkpoints differ from byte " + String(first_diff),
    )
    print("  PASS", name, "— checkpoints byte-identical (", len(a), "bytes )")


def main() raises:
    print("--- PPO CUDA-graph update vs eager: bit parity ---")
    var ctx = DeviceContext()
    print("  device:", ctx.name())

    var de = _train_discrete[False](ctx, "/tmp/ppo_graph_parity_disc_eager.ckpt")
    var dg = _train_discrete[True](ctx, "/tmp/ppo_graph_parity_disc_graph.ckpt")
    _assert_same(
        "discrete (LunarLander)", de, dg,
        "/tmp/ppo_graph_parity_disc_eager.ckpt",
        "/tmp/ppo_graph_parity_disc_graph.ckpt",
    )

    var ce = _train_continuous[False](ctx, "/tmp/ppo_graph_parity_cont_eager.ckpt")
    var cg = _train_continuous[True](ctx, "/tmp/ppo_graph_parity_cont_graph.ckpt")
    _assert_same(
        "continuous (HalfCheetah)", ce, cg,
        "/tmp/ppo_graph_parity_cont_eager.ckpt",
        "/tmp/ppo_graph_parity_cont_graph.ckpt",
    )
    var dde = _train_discrete[False, True](
        ctx, "/tmp/ppo_graph_parity_disc_dev_eager.ckpt"
    )
    var ddg = _train_discrete[True, True](
        ctx, "/tmp/ppo_graph_parity_disc_dev_graph.ckpt"
    )
    assert_true(dde[0] > 0, "device rollout delivered no LunarLander episode")
    _assert_same(
        "discrete device rollout", dde, ddg,
        "/tmp/ppo_graph_parity_disc_dev_eager.ckpt",
        "/tmp/ppo_graph_parity_disc_dev_graph.ckpt",
    )

    var cde = _train_continuous[False, True](
        ctx, "/tmp/ppo_graph_parity_cont_dev_eager.ckpt"
    )
    var cdg = _train_continuous[True, True](
        ctx, "/tmp/ppo_graph_parity_cont_dev_graph.ckpt"
    )
    _assert_same(
        "continuous device rollout", cde, cdg,
        "/tmp/ppo_graph_parity_cont_dev_eager.ckpt",
        "/tmp/ppo_graph_parity_cont_dev_graph.ckpt",
    )
    print("ALL PASSED")
