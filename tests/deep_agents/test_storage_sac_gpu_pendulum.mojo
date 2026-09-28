"""SAC GPU path gate — train on a batched GPU env, greedy-eval on CPU.

Exercises the migrated SACTrainer GPU surface end-to-end:
  - select_action_batched[N_ENVS] GPU (Philox warmup kernel + device obs
    bridge + actor/rsample forward + clamp kernel),
  - record_batch_gpu[N_ENVS] (device replay store),
  - train_step on GPU (target_y / twin-critic / actor-loss / α / polyak blocks),
  - select_greedy_action GPU (used by the eval).

CUDA-graph capture is ENABLED (USE_TRAIN_CUDA_GRAPH=True) — exercises the
trainer's `train_device_kernels` / `note_train_update` / `learning_starts_count`
capture surface. On Apple Metal `CUDAGraph` is a no-op (the closure runs each
iteration, identical to the non-captured path); on NVIDIA it captures + replays.

Convergence is measured by a GREEDY EVAL on a CPU Pendulum env (the GPU agent's
`select_greedy_action` runs host→device→host per step), which gives a real
return independent of the batched driver's episode tracker. A random Pendulum
policy returns ≈ -1200; a learned one clears -500.

Run (Apple Metal or NVIDIA):
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_sac_gpu_pendulum.mojo
  pixi run mojo run -I . tests/deep_agents/test_storage_sac_gpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.sac.config import SAC
from noeira.deep_agents.training.batched_env import BatchedGpuEnv
from noeira.envs.pendulum.pendulum_v2 import PendulumV2
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 128
comptime BATCH = 256
comptime CAP = 100_000
comptime N_ENVS = 8
# NOTE (Apple Metal): the no-capture per-step D2H path is slow + memory-growing
# on long runs (capture is the fix — separate stretch). On NVIDIA this is much
# faster, so NUM_STEPS is set for a real convergence window. On Apple, drop it.
#
# STATUS: on Apple Metal the GPU training signal looks healthy (critic_loss
# ~0.6, α adapts) but the greedy eval shows no improvement — under
# investigation (suspect the B=1 select_greedy_action path, not the training).
# This gate prints DIAG metrics + both evals so the NVIDIA behaviour is visible.
comptime NUM_STEPS = 8_000

comptime BatchedEnvT = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS, ACT]


def main() raises:
    seed(42)
    print("=" * 64)
    print("SAC GPU path gate — Pendulum (batched GPU train, CPU greedy eval)")
    print("=" * 64)

    with DeviceContext() as ctx:
        var agent = SAC["gpu", OBS, ACT, BATCH, CAP, H](
            ctx=ctx,
            actor_lr=Scalar[DT](3e-4),
            critic_lr=Scalar[DT](1e-3),
            alpha_lr=Scalar[DT](3e-4),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            action_scale=Scalar[DT](2.0),
            init_alpha=Scalar[DT](0.2),
            target_entropy=Scalar[DT](-Float64(ACT)),
            learning_starts=1_000,
            window_size=20,
        )
        var env = BatchedEnvT(ctx)

        # Baseline on the CPU env (greedy on the untrained net).
        var cpu_env = PendulumEnv[DT]()
        var rand_eval = agent.eval(
            cpu_env, num_episodes=5, max_steps_per_episode=200
        )
        print("greedy eval @0 (untrained):", rand_eval)

        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=True,
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
        print("  DIAG alpha      =", m.alpha.to_f64())
        print("  DIAG train_steps=", m.train_steps.to_f64())
        # Per-batch means must be real on the GPU path too (they used to flush
        # 0.0 — the GPU branch now D2H-stages the batch tensors via the
        # persistent host-staging buffer). mean_done MAY legitimately be 0.
        print("  DIAG mean_q         =", m.mean_q.to_f64())
        print("  DIAG mean_target    =", m.mean_target.to_f64())
        print("  DIAG mean_next_q    =", m.mean_next_q.to_f64())
        print("  DIAG mean_reward    =", m.reward_mean.to_f64())
        print("  DIAG mean_done      =", m.mean_done.to_f64())
        print("  DIAG mean_abs_action=", m.action_abs_mean.to_f64())
        assert_true(m.mean_q.to_f64() != 0.0, "GPU mean_q populated (not 0.0)")
        assert_true(
            m.mean_target.to_f64() != 0.0, "GPU mean_target populated (not 0.0)"
        )
        assert_true(
            m.mean_next_q.to_f64() != 0.0, "GPU mean_next_q populated (not 0.0)"
        )
        assert_true(
            m.reward_mean.to_f64() != 0.0, "GPU mean_reward populated (not 0.0)"
        )
        assert_true(
            m.action_abs_mean.to_f64() > 0.0, "GPU mean_abs_action populated"
        )

        var final_eval = agent.eval(
            cpu_env, num_episodes=10, max_steps_per_episode=200
        )
        print("greedy eval (trained):", final_eval)
        assert_true(
            final_eval > Scalar[DT](-700.0),
            "SAC GPU learns Pendulum (greedy eval > -700; random ≈ -1700)",
        )
        assert_true(
            final_eval > rand_eval + Scalar[DT](400.0),
            "trained policy clearly beats the untrained baseline",
        )
        print("SAC GPU PENDULUM OK")
