"""REDQ AGENT convergence on Pendulum-v1 — storage REDQTrainer driver gate.

Builds the REDQ agent via the `SmallREDQ["cpu", 3, 1, BATCH, CAP]()` facade
(SAC-shape REDQ: N=2, N_MIN=2, UTD=1, POLICY_DELAY=1 — the cheapest REDQ regime;
storage trainer + drivers), trains single-env on the real Pendulum env through
`run_offpolicy_train` (via `agent.train_single`), then greedy-evals. Asserts the
trained policy clearly beats random and improves over its own untrained baseline.
Also prints + checks the DIAG metrics.

REDQ specifics: stochastic squashed-Gaussian actor + auto-α (like SAC), N-critic
ENSEMBLE with a randomized MIN-subset TD target and an actor loss that MEANs over
the N online critics (the algorithmic difference vs SAC).

Run: pixi run mojo run -I . tests/deep_agents/test_storage_redq_cpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.deep_agents.redq.config import SmallREDQ
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 128
comptime BATCH = 128
comptime CAP = 100_000
comptime ASCALE = Scalar[DT](2.0)   # Pendulum torque in [-2, 2]


def main() raises:
    seed(42)
    print("=" * 60)
    print("REDQ AGENT Pendulum-v1 convergence (storage trainer, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()

    # SAC-shape REDQ: N=2/N_MIN=2/UTD=1/POLICY_DELAY=1. action_scale must match
    # the env torque range (±2); critic_lr=1e-3 per the SmallREDQ preset.
    var agent = SmallREDQ["cpu", OBS, ACT, BATCH, CAP, H](
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=ASCALE,
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-Float64(ACT)),
        learning_starts=1_000,
    )

    var rand_eval = agent.eval(env, num_episodes=5, max_steps_per_episode=200)
    print("eval @0 (untrained baseline):", rand_eval)

    _ = agent.train_single(
        env, total_timesteps=18_000, print_every=3_000, diag_every=3_000
    )

    var metrics = agent.flush_metrics()
    print("-" * 60)
    print("DIAG  actor_loss :", metrics.policy_loss.v)
    print("DIAG  critic_loss:", metrics.critic_loss.v)
    print("DIAG  alpha      :", metrics.alpha.v)
    print("DIAG  mean_q     :", metrics.mean_q.v)
    print("DIAG  mean_target:", metrics.mean_target.v)
    print("DIAG  mean_reward:", metrics.reward_mean.v)
    print("DIAG  mean_done  :", metrics.mean_done.v)
    print("DIAG  train_steps:", metrics.train_steps.v)
    print("-" * 60)

    var final_eval = agent.eval(
        env, num_episodes=10, max_steps_per_episode=200
    )
    print("FINAL greedy eval(10):", final_eval)

    assert_true(
        final_eval > Scalar[DT](-700.0),
        "REDQ agent learns Pendulum (eval return > -700)",
    )
    assert_true(
        final_eval > rand_eval + Scalar[DT](300.0),
        "REDQ agent clearly beats its untrained baseline (+300)",
    )
    assert_true(
        metrics.critic_loss.v > Scalar[DT](0.0),
        "critic_loss diagnostic populated (> 0)",
    )
    assert_true(
        metrics.reward_mean.v < Scalar[DT](0.0),
        "mean_reward diagnostic populated (Pendulum reward < 0)",
    )
    assert_true(
        metrics.train_steps.v > Scalar[DT](0.0),
        "train_steps advanced",
    )
    print("REDQ AGENT PENDULUM CONVERGENCE OK")
