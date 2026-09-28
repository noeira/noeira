"""DDPG AGENT convergence on Pendulum-v1 — storage DDPGTrainer driver gate.

Builds the DDPG agent via the `DDPG["cpu", 3, 1, BATCH, CAP]()` facade (storage
trainer + drivers), trains single-env on the real Pendulum env through
`run_offpolicy_train` (via `agent.train_single`), then greedy-evals. Asserts the
trained policy clearly beats random and improves over its own untrained
baseline. Also prints + checks the DIAG metrics (actor_loss / critic_loss /
mean_q / mean_reward) are populated.

DDPG specifics: deterministic Tanh-bounded actor + Gaussian exploration noise,
single critic, no entropy temperature. The Tanh actor output is fed RAW to the
env (clamped to ±action_scale) — legacy-DDPG parity (the policy uses [-1, 1]
torque even though action_scale=2 bounds the warmup uniform).

Run: pixi run mojo run -I . tests/deep_agents/test_storage_ddpg_cpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.deep_agents.ddpg.config import DDPG
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
    print("DDPG AGENT Pendulum-v1 convergence (storage trainer, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()

    var agent = DDPG["cpu", OBS, ACT, BATCH, CAP, H](
        actor_lr=Scalar[DT](1e-3),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=ASCALE,
        noise_scale=Scalar[DT](0.1),
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
    print("DIAG  mean_q     :", metrics.mean_q.v)
    print("DIAG  mean_target:", metrics.mean_target.v)
    print("DIAG  mean_reward:", metrics.reward_mean.v)
    print("DIAG  train_steps:", metrics.train_steps.v)
    print("-" * 60)

    var final_eval = agent.eval(
        env, num_episodes=10, max_steps_per_episode=200
    )
    print("FINAL greedy eval(10):", final_eval)

    assert_true(
        final_eval > Scalar[DT](-700.0),
        "DDPG agent learns Pendulum (eval return > -700)",
    )
    assert_true(
        final_eval > rand_eval + Scalar[DT](300.0),
        "DDPG agent clearly beats its untrained baseline (+300)",
    )
    # Diagnostics must be populated (critic loss > 0; mean_reward negative on
    # Pendulum; train_steps advanced).
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
    print("DDPG AGENT PENDULUM CONVERGENCE OK")
