"""REDQ GPU path gate — train on a batched GPU env, greedy-eval on CPU.

Mirrors `test_storage_ddpg_gpu_pendulum.mojo` / `test_storage_td3_gpu_pendulum.mojo`
for the migrated `REDQTrainer` GPU surface: builds the gpu agent via
`SmallREDQ["gpu", ...]` (SAC-shape REDQ: N=2, N_MIN=2, UTD=1, POLICY_DELAY=1 —
the cheapest ensemble regime, for speed), a
`BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS, ACT]`, trains batched on GPU
(N_ENVS=8, ~8k steps, capture DEFERRED → USE_TRAIN_CUDA_GRAPH=False), then
flushes + prints the DIAG bundle. The GATE is that the GPU training path runs
to completion AND the DIAG metrics are populated (mean_q / critic_loss /
mean_reward / alpha != 0).

The `REDQAgent` facade does NOT (yet) expose a batched `train()` over a
`BatchedEnv` (its docstring leaves that as a follow-up; only `train_single`
ships). Since `REDQTrainer` conforms to `OffPolicyAgentGpu`, this test drives
the GPU batched path the same way the facade would — by calling the shared
`run_offpolicy_train_batched` driver directly on `agent.trainer`, exactly as
`SACAgent.train` does internally. This exercises the same GPU assembly
(ensemble combined-Q kernel + N-critic loop + Adam.adopt + DeviceMeanAccum).

A GREEDY EVAL on a CPU Pendulum env is PRINTED but NOT hard-asserted: on Apple
Metal the single-env (B=1) `select_greedy_action` eval does not reflect
convergence (a known Apple B=1 issue, NOT a training bug — identical to the
SAC/DDPG/TD3 GPU smokes). DIAG health is the real signal here.

REDQ specifics: stochastic squashed-Gaussian actor + auto-α (like SAC), N-critic
ENSEMBLE with a randomized MIN-subset TD target and an actor loss that MEANs over
the N online critics (the algorithmic difference vs SAC).

Run (Apple Metal or NVIDIA):
  pixi run mojo run -I . tests/deep_agents/test_storage_redq_gpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.redq.config import SmallREDQ
from noeira.deep_agents.redq.trainer import REDQTrainer
from noeira.deep_agents.redq.kernels import REDQ_TARGET_MIN
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.deep_agents.training.batched_env import BatchedGpuEnv
from noeira.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.redq.config import REDQActor, REDQCritic
from noeira.envs.pendulum.pendulum_v2 import PendulumV2
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 64
comptime BATCH = 256
comptime CAP = 100_000
comptime N_ENVS = 8
comptime NUM_STEPS = 8_000

comptime ASCALE = Scalar[DT](2.0)  # Pendulum torque in [-2, 2]

comptime BatchedEnvT = BatchedGpuEnv[PendulumV2[DT], N_ENVS, OBS, ACT]

# Compile-time identity of the SmallREDQ["gpu", ...] trainer so we can drive its
# GPU batched path directly via the shared off-policy driver (the facade itself
# doesn't expose a batched `train()` yet).
comptime REDQTrainerT = REDQTrainer[
    "gpu",
    ReplaySampleStep[AnyReplay["gpu", OBS, ACT, CAP], BATCH],
    REDQActor[OBS, ACT, H],
    REDQCritic[OBS, ACT, H],
    2, 2, 1, 1, REDQ_TARGET_MIN,
]


def main() raises:
    seed(42)
    print("=" * 64)
    print("REDQ GPU path gate — Pendulum (batched GPU train, CPU greedy eval)")
    print("=" * 64)

    with DeviceContext() as ctx:
        # SAC-shape REDQ on the GPU target. action_scale must match the env
        # torque range (±2); critic_lr=1e-3 per the SmallREDQ preset.
        var agent = SmallREDQ["gpu", OBS, ACT, BATCH, CAP, H](
            ctx=ctx,
            actor_lr=Scalar[DT](3e-4),
            critic_lr=Scalar[DT](1e-3),
            alpha_lr=Scalar[DT](3e-4),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            action_scale=ASCALE,
            init_alpha=Scalar[DT](0.2),
            target_entropy=Scalar[DT](-Float64(ACT)),
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

        # The facade lacks a batched train(); drive the shared GPU off-policy
        # driver directly on the trainer (same call SACAgent.train makes).
        # USE_TRAIN_CUDA_GRAPH=False — REDQ defers capture.
        _ = run_offpolicy_train_batched[
            REDQTrainerT,
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=False,
            USE_ENV_CUDA_GRAPH=False,
        ](
            ctx,
            agent.trainer,
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=2_500,
            verbose=True,
        )

        var m = agent.flush_metrics()
        print("-" * 64)
        print("  DIAG actor_loss =", m.policy_loss.v)
        print("  DIAG critic_loss=", m.critic_loss.v)
        print("  DIAG alpha      =", m.alpha.v)
        print("  DIAG mean_q     =", m.mean_q.v)
        print("  DIAG mean_target=", m.mean_target.v)
        print("  DIAG mean_reward=", m.reward_mean.v)
        print("  DIAG mean_next_q=", m.mean_next_q.v)
        print("  DIAG mean_done  =", m.mean_done.v)
        print("  DIAG mean_abs_a =", m.action_abs_mean.v)
        print("  DIAG train_steps=", m.train_steps.v)
        print("  DIAG n_updates  =", m.n_updates.v)
        print("-" * 64)

        # The GPU path must run to completion AND populate the diagnostics.
        # mean_done MAY legitimately be 0 (Pendulum is timeout-only), so it is
        # NOT asserted.
        assert_true(
            m.train_steps.v > Scalar[DT](0.0), "GPU train_steps advanced (> 0)"
        )
        assert_true(
            m.mean_q.v != Scalar[DT](0.0), "GPU mean_q populated (not 0.0)"
        )
        assert_true(
            m.critic_loss.v != Scalar[DT](0.0),
            "GPU critic_loss populated (not 0.0)",
        )
        assert_true(
            m.reward_mean.v != Scalar[DT](0.0),
            "GPU mean_reward populated (not 0.0)",
        )
        assert_true(
            m.alpha.v != Scalar[DT](0.0), "GPU alpha populated (not 0.0)"
        )

        # Greedy eval is informational only (known Apple B=1 issue).
        var final_eval = agent.eval(
            cpu_env, num_episodes=10, max_steps_per_episode=200
        )
        print("greedy eval (trained):", final_eval)
        print("  (eval is informational — Apple B=1 greedy eval may not improve)")

        print("REDQ GPU PATH OK")
