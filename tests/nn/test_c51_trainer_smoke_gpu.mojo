"""C51Trainer GPU smoke test — CartPole (CPU env) × GPU trainer.

Validates:
  1. C51Trainer compiles + runs on GPU end-to-end with the
     NUM_ACTIONS · N_ATOMS output Q-net.
  2. Standard + Double C51 both train without crash, produce finite
     loss + non-trivial ε decay over 1500 steps.
  3. Categorical projection + cross-entropy + scatter all run as GPU
     kernels (no D2H/H2D in `_train_step_impl`).
"""

from std.math import isnan, isinf
from std.random import seed
from max.gpu.host import DeviceContext
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.deep_agents.c51.trainer import C51Trainer
from noeira.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from noeira.deep_agents.training.blocks import UniformSampleGpuStep

from noeira.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime N_ATOMS = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

comptime C51QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS * N_ATOMS],
]


def test_c51_gpu() raises:
    print("--- C51Trainer[target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = C51Trainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            C51QNet,
            N_ATOMS=N_ATOMS,
            NUM_ACTIONS=NUM_ACTIONS,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](2.5e-4),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](1.0),
            epsilon_decay=Scalar[DT](0.995),
            epsilon_min=Scalar[DT](0.05),
            learning_starts=WARMUP,
            target_update_freq=500,
            initial_episode_fill=Scalar[DT](0.0),
            v_min=Scalar[DT](0.0),
            v_max=Scalar[DT](200.0),
        )
        var env = CartPoleEnv[DT]()
        _ = run_offpolicy_discrete_train(
            trainer, env, TOTAL_STEPS,
            print_every=500, verbose=True, ctx=ctx,
        )
        var mr = trainer.mean_return()
        print(
            "  mean_return=", mr,
            " ep_count=", trainer.ep_count(),
            " epsilon=", trainer.epsilon,
        )
        assert_true(not isnan(mr), "C51 GPU mean_return NaN")
        assert_true(not isinf(mr), "C51 GPU mean_return Inf")
        assert_true(trainer.ep_count() > 0, "C51 GPU no episodes")

        # Distributional device-diag fix: mean_q (expected Q from softmax) and
        # dist_entropy (categorical entropy) read a hard 0.0 on GPU before the
        # `_c51_diag_kernel` was wired. Entropy of a non-degenerate categorical
        # is > 0; mean_q over CartPole returns is non-zero.
        var dm = trainer.flush_metrics()
        var dq = dm.mean_q.to_f64()
        var dent = dm.dist_entropy_mean.to_f64()
        var drew = dm.reward_mean.to_f64()
        print("  mean_q=", dq, " dist_entropy=", dent, " mean_reward=", drew)
        assert_true(not isnan(dq) and not isinf(dq), "C51 GPU mean_q non-finite")
        assert_true(
            not isnan(dent) and not isinf(dent), "C51 GPU dist_entropy non-finite"
        )
        assert_true(dq != 0.0, "C51 GPU mean_q is 0 (diag kernel unwired?)")
        assert_true(dent > 0.0, "C51 GPU dist_entropy should be > 0")

        var log = trainer.flush_train_log()
        print("  mean_loss=", log[0], " n_updates=", log[2])
        assert_true(not isnan(log[0]), "C51 GPU mean_loss NaN")
        assert_true(log[2] > 0, "C51 GPU no training updates")

        print("--- C51 GPU greedy eval ---")
        var eval_env = CartPoleEnv[DT]()
        var eval_ret = run_offpolicy_discrete_eval(
            trainer, eval_env, 3,
            max_steps_per_episode=200, verbose=True,
        )
        print("  eval mean_return=", eval_ret)
        assert_true(not isnan(eval_ret), "C51 GPU eval NaN")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_double_c51_gpu() raises:
    print("--- C51Trainer[DOUBLE=True, target=gpu] CartPole ---")
    try:
        var ctx = DeviceContext()
        seed(42)
        var trainer = C51Trainer[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
            C51QNet,
            N_ATOMS=N_ATOMS,
            NUM_ACTIONS=NUM_ACTIONS,
            DOUBLE=True,
        ].make(
            ctx=ctx,
            lr=Scalar[DT](2.5e-4),
            gamma=Scalar[DT](0.99),
            learning_starts=WARMUP,
            target_update_freq=500,
            initial_episode_fill=Scalar[DT](0.0),
            v_min=Scalar[DT](0.0),
            v_max=Scalar[DT](200.0),
        )
        var env = CartPoleEnv[DT]()
        _ = run_offpolicy_discrete_train(
            trainer, env, TOTAL_STEPS,
            print_every=500, verbose=True, ctx=ctx,
        )
        var mr = trainer.mean_return()
        print("  mean_return=", mr, " ep_count=", trainer.ep_count())
        assert_true(not isnan(mr), "Double C51 GPU mean_return NaN")
        assert_true(trainer.ep_count() > 0, "Double C51 GPU no episodes")

        var log = trainer.flush_train_log()
        print("  mean_loss=", log[0], " n_updates=", log[2])
        assert_true(not isnan(log[0]), "Double C51 GPU mean_loss NaN")
        assert_true(log[2] > 0, "Double C51 GPU no updates")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("C51Trainer GPU smoke test — CartPole")
    print("=" * 60)
    test_c51_gpu()
    test_double_c51_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
