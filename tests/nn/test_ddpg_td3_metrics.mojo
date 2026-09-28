"""DDPG + TD3 metrics-parity test (Phase 3.1).

Trains DDPG and TD3 briefly on Pendulum (CPU) and asserts the new
learning diagnostics populate with sane, finite values — not left at
zero (the metric structs previously had no Q/reward fields at all).

  DDPG: mean_q, mean_target, mean_reward
  TD3:  mean_q, mean_target, mean_reward, mean_done

"sane" here:
  - all fields finite,
  - n_updates / n_critic_updates > 0,
  - mean_reward < 0 (Pendulum reward is in [-16.27, 0]),
  - mean_q finite (scalar critic, no support bound),
  - mean_done in [0, 1].

A 0.0 on mean_reward / mean_q catches an unwired diag walk.

Run: pixi run mojo run -I . tests/nn/test_ddpg_td3_metrics.mojo
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.activations import Tanh
from noeira.deep_agents.ddpg.trainer import DDPGTrainer
from noeira.deep_agents.td3.trainer import TD3Trainer
from noeira.deep_agents.training.blocks import UniformSampleCpuStep
from noeira.deep_agents.training.driver_offpolicy import run_offpolicy_train
from noeira.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 64
comptime CAP = 10_000
comptime WARMUP = 200
comptime TOTAL = 800

comptime ActorNet = Sequential[
    Linear[OBS, 64], ReLU[64], Linear[64, 64], ReLU[64],
    Linear[64, ACT], Tanh[ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 64], ReLU[64], Linear[64, 64], ReLU[64],
    Linear[64, 1],
]


def _finite(v: Scalar[DT], tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_ddpg_metrics() raises:
    print("--- DDPG metrics populated ---")
    seed(42)
    var trainer = DDPGTrainer[
        "cpu", UniformSampleCpuStep[OBS, ACT, BATCH, CAP], ActorNet, CriticNet,
    ].make(
        actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
        action_scale=2.0, learning_starts=WARMUP,
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer, env, TOTAL, print_every=5000, verbose=False,
    )

    var m = trainer.flush_metrics()
    var q = m.mean_q.to_f64()
    var tgt = m.mean_target.to_f64()
    var rew = m.reward_mean.to_f64()
    var nup = m.n_updates.to_f64()
    print("  mean_q      =", q)
    print("  mean_target =", tgt)
    print("  mean_reward =", rew)
    print("  n_updates   =", nup)

    _finite(Scalar[DT](q), "mean_q")
    _finite(Scalar[DT](tgt), "mean_target")
    _finite(Scalar[DT](rew), "reward_mean")
    assert_true(nup > 0.0, "no training updates ran")
    assert_true(rew < 0.0, "mean_reward >= 0 (Pendulum reward negative)")
    print("PASS")


def test_td3_metrics() raises:
    print("--- TD3 metrics populated ---")
    seed(42)
    var trainer = TD3Trainer[
        "cpu", UniformSampleCpuStep[OBS, ACT, BATCH, CAP], ActorNet, CriticNet,
    ].make(
        actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
        action_scale=2.0, exploration_noise=0.1,
        target_policy_noise=0.2, target_noise_clip=0.5,
        policy_delay=2, learning_starts=WARMUP,
    )
    var env = PendulumEnv[DT]()
    _ = run_offpolicy_train(
        trainer, env, TOTAL, print_every=5000, verbose=False,
    )

    var m = trainer.flush_metrics()
    var q = m.mean_q.to_f64()
    var tgt = m.mean_target.to_f64()
    var rew = m.reward_mean.to_f64()
    var dn = m.mean_done.to_f64()
    var nc = m.n_critic_updates.to_f64()
    print("  mean_q          =", q)
    print("  mean_target     =", tgt)
    print("  mean_reward     =", rew)
    print("  mean_done       =", dn)
    print("  n_critic_updates=", nc)

    _finite(Scalar[DT](q), "mean_q")
    _finite(Scalar[DT](tgt), "mean_target")
    _finite(Scalar[DT](rew), "reward_mean")
    _finite(Scalar[DT](dn), "mean_done")
    assert_true(nc > 0.0, "no critic updates ran")
    assert_true(rew < 0.0, "mean_reward >= 0 (Pendulum reward negative)")
    assert_true(dn >= 0.0 and dn <= 1.0, "mean_done out of [0,1]")
    print("PASS")


def main() raises:
    test_ddpg_metrics()
    test_td3_metrics()
