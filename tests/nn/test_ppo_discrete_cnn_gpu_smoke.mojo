"""Discrete PPO CNN — GPU image-obs plumbing smoke (5.2, Apple).

GPU sibling of `test_ppo_discrete_cnn_smoke.mojo`: confirms a Nature-
style CNN actor/critic trains finitely through `PPODiscreteTrainer`'s
`train_target="gpu"` path (device Conv2D forward/backward inside the
act-step + on-device clipped-surrogate / MSE train steps). Real numeric
parity vs CPU is NVIDIA-gated; this asserts finiteness on Apple Metal.

Run:
    pixi run -e apple mojo run -I . \
        tests/nn/test_ppo_discrete_cnn_gpu_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed, random_float64
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.flatten import Flatten
from noeira.deep_agents.ppo_discrete.trainer import PPODiscreteTrainer


comptime C = 1
comptime IMG = 12
comptime OBS_DIM = C * IMG * IMG     # 144
comptime N_ACTIONS = 4
comptime ROLLOUT_LEN = 32
comptime MINIBATCH = 16
comptime N_EPOCHS = 2
comptime HIDDEN = 64

comptime CnnTrunk = Sequential[
    Conv2D[C, 8, 3, 2, 1, IMG, IMG], ReLU[8 * 6 * 6],
    Conv2D[8, 16, 3, 2, 1, 6, 6], ReLU[16 * 3 * 3],
    Flatten[16 * 3 * 3],
    Linear[16 * 3 * 3, HIDDEN], ReLU[HIDDEN],
]
comptime ActorNet = Sequential[CnnTrunk, Linear[HIDDEN, N_ACTIONS]]
comptime CriticNet = Sequential[CnnTrunk, Linear[HIDDEN, 1]]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def main() raises:
    seed(42)
    print("--- discrete PPO CNN GPU smoke (Apple) ---")
    var ctx = DeviceContext()

    var trainer = PPODiscreteTrainer[
        "gpu", ActorNet, CriticNet,
        OBS_DIM, N_ACTIONS, ROLLOUT_LEN, MINIBATCH, N_EPOCHS,
    ].make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        entropy_coef=Scalar[DT](0.01),
        ctx=ctx,
    )

    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))

    var n_updates = 0
    var ep_len = 0
    for step in range(160):
        for d in range(OBS_DIM):
            obs[d] = Scalar[DT](random_float64())
        var a = trainer.select_action(obs, step)
        assert_true(a >= 0 and a < N_ACTIONS, "action out of range")
        for d in range(OBS_DIM):
            next_obs[d] = Scalar[DT](random_float64())
        var reward = Scalar[DT](random_float64()) - Scalar[DT](0.5)
        ep_len += 1
        var done = ep_len >= 20
        trainer.record_transition(
            obs, a, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.mark_terminal()
            trainer.end_episode()
            ep_len = 0
        if trainer.train_step(step):
            n_updates += 1

    assert_true(n_updates > 0, "train_step never fired")
    var m = trainer.flush_metrics()
    _finite(Float64(m.policy_loss.to_f64()), "policy_loss")
    _finite(Float64(m.critic_loss.to_f64()), "critic_loss")
    print("rollout updates fired:", n_updates)
    print("actor_loss :", m.policy_loss.to_f64())
    print("critic_loss:", m.critic_loss.to_f64())
    print("PASS: CNN GPU path runs finite through discrete on-policy.")
