"""C51 metrics-parity test (Phase 3).

Trains C51 briefly on CartPole (CPU) and asserts the new per-batch
distributional diagnostics (`mean_q`, `mean_target`, `dist_entropy`,
`mean_reward`, `mean_done`) populate with sane, finite values — not left
at zero.

"sane" here:
  - all fields finite,
  - mean_q / mean_target in the atom support [V_MIN, V_MAX],
  - dist_entropy in (0, log(N_ATOMS)],
  - mean_reward > 0 (CartPole gives +1/step),
  - n_updates > 0.

A 0.0 on any diagnostic catches an unwired accumulator.

Run: pixi run mojo run -I . tests/nn/test_c51_metrics.mojo
"""

from std.math import isnan, isinf, log as flog
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.deep_agents.c51.trainer import C51Trainer
from noeira.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
)
from noeira.deep_agents.training.blocks import UniformSampleCpuStep
from noeira.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime NA = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500
comptime V_MIN = -10.0
comptime V_MAX = 10.0

comptime C51Net = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS * NA],
]


def _finite(v: Scalar[DT], tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_c51_metrics_populated() raises:
    print("--- C51 metrics populated ---")
    seed(42)
    var trainer = C51Trainer[
        "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], C51Net,
        NA, NUM_ACTIONS,
    ].make(
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        learning_starts=WARMUP,
        target_update_freq=500,
        v_min=Scalar[DT](V_MIN),
        v_max=Scalar[DT](V_MAX),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS, print_every=5000, verbose=False,
    )

    var m = trainer.flush_metrics()
    var q = m.mean_q.to_f64()
    var tgt = m.mean_target.to_f64()
    var ent = m.dist_entropy_mean.to_f64()
    var rew = m.reward_mean.to_f64()
    var dn = m.mean_done.to_f64()
    var nup = m.n_updates.to_f64()
    print("  mean_q       =", q)
    print("  mean_target  =", tgt)
    print("  dist_entropy =", ent)
    print("  mean_reward  =", rew)
    print("  mean_done    =", dn)
    print("  n_updates    =", nup)

    _finite(Scalar[DT](q), "mean_q")
    _finite(Scalar[DT](tgt), "mean_target")
    _finite(Scalar[DT](ent), "dist_entropy_mean")
    _finite(Scalar[DT](rew), "reward_mean")
    _finite(Scalar[DT](dn), "mean_done")

    assert_true(nup > 0.0, "no training updates ran")
    assert_true(q >= V_MIN and q <= V_MAX, "mean_q outside [V_MIN, V_MAX]")
    assert_true(
        tgt >= V_MIN and tgt <= V_MAX, "mean_target outside [V_MIN, V_MAX]"
    )
    var max_ent = Float64(flog(Float64(NA)))
    assert_true(ent > 0.0, "dist_entropy is 0 (accumulator unwired?)")
    assert_true(ent <= max_ent + 1e-4, "dist_entropy above log(N_ATOMS)")
    assert_true(rew > 0.0, "mean_reward is 0 (accumulator unwired?)")
    assert_true(dn >= 0.0 and dn <= 1.0, "mean_done out of [0,1]")
    print("PASS")


def main() raises:
    test_c51_metrics_populated()
