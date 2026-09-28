"""DQN metrics-parity test (Phase 3).

Trains DQN briefly on CartPole (CPU) and asserts the new per-batch
diagnostics (`mean_q`, `mean_target`, `mean_td_error`, `mean_reward`,
`mean_done`) are populated with sane, finite values — not left at zero.

What "sane" means here:
  - all fields finite (no NaN/Inf),
  - mean_td_error > 0 (there IS a Bellman residual during learning),
  - mean_q is non-zero (the Q-net produces real values),
  - mean_reward is non-zero (CartPole gives +1 per step),
  - n_updates > 0 (training actually ran).

This guards the diagnostic-accumulator wiring: a regression that leaves
an accumulator unpopulated would show up as an exact 0.0 here.

Run: pixi run mojo run -I . tests/nn/test_dqn_metrics.mojo
"""

from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.deep_agents.dqn.trainer import DQNTrainer
from noeira.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
)
from noeira.deep_agents.training.blocks import UniformSampleCpuStep
from noeira.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]


def _finite(v: Scalar[DT], tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_dqn_metrics_populated() raises:
    print("--- DQN metrics populated ---")
    seed(42)
    var trainer = DQNTrainer[
        "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS, print_every=5000, verbose=False,
    )

    var m = trainer.flush_metrics()
    var q = m.mean_q.to_f64()
    var tgt = m.mean_target.to_f64()
    var te = m.td_error_abs_mean.to_f64()
    var rew = m.reward_mean.to_f64()
    var dn = m.mean_done.to_f64()
    var nup = m.n_updates.to_f64()
    print("  mean_q       =", q)
    print("  mean_target  =", tgt)
    print("  mean_td_error=", te)
    print("  mean_reward  =", rew)
    print("  mean_done    =", dn)
    print("  n_updates    =", nup)

    _finite(Scalar[DT](q), "mean_q")
    _finite(Scalar[DT](tgt), "mean_target")
    _finite(Scalar[DT](te), "td_error_abs_mean")
    _finite(Scalar[DT](rew), "reward_mean")
    _finite(Scalar[DT](dn), "mean_done")

    assert_true(nup > 0.0, "no training updates ran")
    # Bellman residual is strictly positive while learning.
    assert_true(te > 0.0, "mean_td_error is exactly 0 (accumulator unwired?)")
    # CartPole reward is +1/step → minibatch mean reward must be > 0.
    assert_true(rew > 0.0, "mean_reward is exactly 0 (accumulator unwired?)")
    # Q-net outputs are non-trivial.
    assert_true(q != 0.0, "mean_q is exactly 0 (accumulator unwired?)")
    # done fraction is a probability in [0, 1].
    assert_true(dn >= 0.0 and dn <= 1.0, "mean_done out of [0,1]")
    print("PASS")


def main() raises:
    test_dqn_metrics_populated()
