"""TD3 GPU path gate — train on a batched GPU env, greedy-eval on CPU.

Mirrors `test_storage_sac_gpu_pendulum.mojo` for the migrated TD3Trainer GPU
surface: builds the gpu agent via `TD3["gpu", ...]`, a
`BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS, ACT]`, trains batched on GPU
(N_ENVS=8, ~8k steps, capture DEFERRED → USE_TRAIN_CUDA_GRAPH=False), then
flushes + prints the DIAG bundle. The GATE is that the GPU training path runs
to completion AND the DIAG metrics are populated (mean_q / critic_loss /
mean_target / mean_reward != 0).

A GREEDY EVAL on a CPU Pendulum env is PRINTED but NOT hard-asserted: on Apple
Metal the single-env (B=1) `select_greedy_action` eval does not reflect
convergence (a known Apple B=1 issue, NOT a training bug — see the SAC GPU
test docstring). DIAG health is the real signal here.

TD3 specifics: deterministic Tanh-bounded actor + Gaussian exploration noise,
TWIN critics (min-of-2 target), target-policy smoothing, delayed actor update.
NO entropy temperature (no alpha). The bundle has mean_done + separate
n_actor_updates / n_critic_updates counts (no mean_abs_action / mean_next_q).

Run (Apple Metal or NVIDIA):
  pixi run mojo run -I . tests/deep_agents/test_storage_td3_gpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.td3.config import TD3
from noeira.deep_agents.training.batched_env import BatchedGpuEnv
from noeira.envs.pendulum.pendulum_v2 import PendulumV2
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 128
comptime BATCH = 256
comptime CAP = 100_000
comptime N_ENVS = 8
comptime NUM_STEPS = 8_000

comptime BatchedEnvT = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS, ACT]


def main() raises:
    seed(42)
    print("=" * 64)
    print("TD3 GPU path gate — Pendulum (batched GPU train, CPU greedy eval)")
    print("=" * 64)

    with DeviceContext() as ctx:
        var agent = TD3["gpu", OBS, ACT, BATCH, CAP, H](
            ctx=ctx,
            actor_lr=Scalar[DT](1e-3),
            critic_lr=Scalar[DT](1e-3),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            action_scale=Scalar[DT](2.0),
            exploration_noise=Scalar[DT](0.1),
            target_policy_noise=Scalar[DT](0.2),
            target_noise_clip=Scalar[DT](0.5),
            policy_delay=2,
            learning_starts=1_000,
            window_size=20,
        )
        var env = BatchedEnvT(ctx)

        # Baseline greedy eval on a CPU env (untrained net).
        var cpu_env = PendulumEnv[DT]()
        var rand_eval = agent.eval(
            cpu_env, num_episodes=5, max_steps_per_episode=200
        )
        print("greedy eval @0 (untrained):", rand_eval)

        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=False,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=2_500,
            verbose=True,
        )

        var m = agent.flush_metrics()
        print("  DIAG actor_loss =", m.policy_loss.to_f64())
        print("  DIAG critic_loss=", m.critic_loss.to_f64())
        print("  DIAG mean_q     =", m.mean_q.to_f64())
        print("  DIAG mean_target=", m.mean_target.to_f64())
        print("  DIAG mean_reward=", m.reward_mean.to_f64())
        print("  DIAG mean_done  =", m.mean_done.to_f64())
        print("  DIAG train_steps=", m.train_steps.to_f64())
        print("  DIAG n_actor_upd=", m.n_actor_updates.to_f64())
        print("  DIAG n_crit_upd =", m.n_critic_updates.to_f64())

        # The GPU path must run to completion AND populate the diagnostics.
        # mean_done MAY legitimately be 0 (Pendulum is timeout-only), so it is
        # NOT asserted.
        assert_true(
            m.train_steps.to_f64() > 0.0, "GPU train_steps advanced (> 0)"
        )
        assert_true(m.mean_q.to_f64() != 0.0, "GPU mean_q populated (not 0.0)")
        assert_true(
            m.critic_loss.to_f64() != 0.0, "GPU critic_loss populated (not 0.0)"
        )
        assert_true(
            m.mean_target.to_f64() != 0.0, "GPU mean_target populated (not 0.0)"
        )
        assert_true(
            m.reward_mean.to_f64() != 0.0, "GPU mean_reward populated (not 0.0)"
        )

        # Greedy eval is informational only (known Apple B=1 issue).
        var final_eval = agent.eval(
            cpu_env, num_episodes=10, max_steps_per_episode=200
        )
        print("greedy eval (trained):", final_eval)
        print("  (eval is informational — Apple B=1 greedy eval may not improve)")

        print("TD3 GPU PATH OK")
