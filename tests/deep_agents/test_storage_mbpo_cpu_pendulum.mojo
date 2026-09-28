"""MBPO AGENT convergence on Pendulum-v1 — storage MBPOTrainer driver gate.

Builds the MBPO agent via the `MBPO["cpu", OBS, ACT, BATCH, REPLAY_CAP,
SYNTH_CAP]()` preset (the tuned converging regime: REAL_RATIO_PCT=50 +
LOGVAR_MAX=-5), trains single-env on the real Pendulum env through
`run_offpolicy_train` (via `agent.train_single`), then greedy-evals. Asserts the
trained policy clearly beats random and improves over its own untrained
baseline. MBPO = SAC trained on a mix of real + model-generated transitions
from a probabilistic dynamics ENSEMBLE.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_mbpo_cpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.deep_agents.mbpo import MBPO
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 128
comptime REPLAY_CAP = 100_000
comptime SYNTH_CAP = 400_000
comptime ASCALE = Scalar[DT](2.0)   # Pendulum torque in [-2, 2]


def main() raises:
    seed(42)
    print("=" * 60)
    print("MBPO AGENT Pendulum-v1 convergence (storage trainer, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()

    # Tuned converging regime: REAL_RATIO=50, LOGVAR_MAX=-5 (preset defaults).
    # action_scale must match the env torque range (±2).
    # Smaller ensemble/nets for a CPU-affordable smoke (keeps the converging
    # REAL_RATIO=50 + LOGVAR_MAX=-5 regime; N_ENS=5/N_ELITES=2, HIDDEN/DYN=128).
    var agent = MBPO[
        "cpu", OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP,
        N_ENS=5, N_ELITES=2, REAL_RATIO=50, HIDDEN=128, DYN_HIDDEN=128,
    ](
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        model_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=ASCALE,
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-Float64(ACT)),
        learning_starts=1_000,
        model_train_freq=250,
        num_rollouts_per_step=400,
        sac_updates_per_step=10,
        dyn_batch_size=128,
        dyn_max_epochs=20,
    )

    var rand_eval = agent.eval(env, num_episodes=5, max_steps_per_episode=200)
    print("eval @0 (untrained baseline):", rand_eval)

    _ = agent.train_single(
        env, total_timesteps=6_000, print_every=2_000, diag_every=2_000,
        verbose=False,
    )

    var metrics = agent.flush_metrics()
    print("-" * 60)
    print("DIAG  actor_loss :", metrics.policy_loss.v)
    print("DIAG  critic_loss:", metrics.critic_loss.v)
    print("DIAG  alpha      :", metrics.alpha.v)
    print("DIAG  mean_q     :", metrics.mean_q.v)
    print("DIAG  mean_reward:", metrics.reward_mean.v)
    print("DIAG  dyn_loss   :", metrics.dyn_loss.v)
    print("DIAG  train_steps:", metrics.train_steps.v)
    print("-" * 60)

    var final_eval = agent.eval(
        env, num_episodes=10, max_steps_per_episode=200
    )
    print("FINAL greedy eval(10):", final_eval)

    assert_true(
        final_eval > Scalar[DT](-700.0),
        "MBPO agent learns Pendulum (eval return > -700)",
    )
    assert_true(
        final_eval > rand_eval + Scalar[DT](300.0),
        "MBPO agent clearly beats its untrained baseline (+300)",
    )
    assert_true(
        metrics.train_steps.v > Scalar[DT](0.0),
        "train_steps advanced",
    )
    print("MBPO AGENT PENDULUM CONVERGENCE OK")
