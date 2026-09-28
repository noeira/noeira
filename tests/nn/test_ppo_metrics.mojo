"""PPO metrics-parity test (Phase 3).

Trains PPO continuous briefly on Pendulum (CPU) and asserts the new
per-minibatch diagnostics (`entropy`, `approx_kl`, `clip_fraction`,
`explained_variance`) populate with sane, finite values — not left at
zero. Mirrors test_dqn_metrics / test_c51_metrics.

"sane" here:
  - all fields finite,
  - entropy > 0 (Gaussian entropy is positive for the init log_std),
  - approx_kl >= 0 and > 0 (Schulman (r-1)-log r is 0 only if the
    policy never moved — after 16 updates it has),
  - clip_fraction in [0, 1],
  - explained_variance <= 1 (it may be negative early),
  - n_updates > 0.

A 0.0 on entropy / approx_kl catches an unwired accumulator.

Run: pixi run mojo run -I . tests/nn/test_ppo_metrics.mojo
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.deep_agents.ppo.trainer import PPOTrainer
from noeira.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 16
comptime ROLLOUT = 64
comptime MB = 16
comptime EPOCHS = 4
comptime TOTAL_STEPS = 4 * ROLLOUT + 5

comptime ActorNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Trainer = PPOTrainer[
    "cpu", ActorNet, CriticNet, OBS, ACT, ROLLOUT, MB, EPOCHS,
]


def _finite(v: Scalar[DT], tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_ppo_metrics_populated() raises:
    print("--- PPO metrics populated ---")
    seed(42)
    var t = Trainer.make(action_scale=Scalar[DT](2.0))
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs = env.get_obs_list()
    var action = List[Scalar[DT]](capacity=ACT)
    action.append(Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    for step_idx in range(TOTAL_STEPS):
        t.select_action(obs, action, step_idx)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = Scalar[DT](nxt[d])
        t.record_transition(
            obs, action, Scalar[DT](reward), next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            t.end_episode()
            _ = env.reset()
            obs = env.get_obs_list()
        else:
            obs = nxt.copy()
        _ = t.train_step(step_idx)

    var m = t.flush_metrics()
    var aL = m.policy_loss.to_f64()
    var cL = m.critic_loss.to_f64()
    var ent = m.entropy.to_f64()
    var kl = m.approx_kl.to_f64()
    var clipf = m.clip_fraction.to_f64()
    var ev = m.explained_variance.to_f64()
    var nup = m.n_updates.to_f64()
    print("  actor_loss         =", aL)
    print("  critic_loss        =", cL)
    print("  entropy            =", ent)
    print("  approx_kl          =", kl)
    print("  clip_fraction      =", clipf)
    print("  explained_variance =", ev)
    print("  n_updates          =", nup)

    _finite(Scalar[DT](aL), "policy_loss")
    _finite(Scalar[DT](cL), "critic_loss")
    _finite(Scalar[DT](ent), "entropy")
    _finite(Scalar[DT](kl), "approx_kl")
    _finite(Scalar[DT](clipf), "clip_fraction")
    _finite(Scalar[DT](ev), "explained_variance")

    assert_true(nup > 0.0, "no training updates ran")
    assert_true(ent > 0.0, "entropy is 0 (accumulator unwired?)")
    assert_true(kl >= 0.0, "approx_kl negative (impossible for Schulman est.)")
    assert_true(kl > 0.0, "approx_kl is 0 (accumulator unwired?)")
    assert_true(
        clipf >= 0.0 and clipf <= 1.0, "clip_fraction out of [0, 1]"
    )
    assert_true(ev <= 1.0 + 1e-4, "explained_variance above 1.0")
    print("PASS")


def main() raises:
    test_ppo_metrics_populated()
