"""TD3 AGENT convergence on Pendulum-v1 — storage TD3Trainer driver gate.

Builds the TD3 agent via the `TD3["cpu", 3, 1, BATCH, CAP]()` facade (storage
trainer + drivers), trains single-env on the real Pendulum env through
`run_offpolicy_train` (via `agent.train_single`), then greedy-evals. Asserts the
trained policy clearly beats random and improves over its own untrained
baseline. Also prints + checks the DIAG metrics.

TD3 specifics: deterministic Tanh-bounded actor + Gaussian exploration noise,
TWIN critics (min-of-2 target), target-policy smoothing, delayed actor update.
The Tanh actor output is fed RAW to the env (clamped to ±action_scale).

Run: pixi run mojo run -I . tests/deep_agents/test_storage_td3_cpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.deep_agents.td3.config import TD3
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
    print("TD3 AGENT Pendulum-v1 convergence (storage trainer, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()

    var agent = TD3["cpu", OBS, ACT, BATCH, CAP, H](
        actor_lr=Scalar[DT](1e-3),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=ASCALE,
        exploration_noise=Scalar[DT](0.1),
        target_policy_noise=Scalar[DT](0.2),
        target_noise_clip=Scalar[DT](0.5),
        policy_delay=2,
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
    print("DIAG  mean_done  :", metrics.mean_done.v)
    print("DIAG  train_steps:", metrics.train_steps.v)
    print("DIAG  n_actor_upd:", metrics.n_actor_updates.v)
    print("DIAG  n_crit_upd :", metrics.n_critic_updates.v)
    print("-" * 60)

    var final_eval = agent.eval(
        env, num_episodes=10, max_steps_per_episode=200
    )
    print("FINAL greedy eval(10):", final_eval)

    assert_true(
        final_eval > Scalar[DT](-700.0),
        "TD3 agent learns Pendulum (eval return > -700)",
    )
    assert_true(
        final_eval > rand_eval + Scalar[DT](300.0),
        "TD3 agent clearly beats its untrained baseline (+300)",
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
    print("TD3 AGENT PENDULUM CONVERGENCE OK")
