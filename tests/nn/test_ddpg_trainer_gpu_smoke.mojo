"""DDPGTrainer GPU smoke (Phase 4.1).

Runs the GPU DDPG trainer end-to-end against the CPU Pendulum env via the
manual loop (the `run_offpolicy_train` single-env path: cpu env + gpu
train) for a short horizon, then asserts:

  * training updates ran (n_updates > 0),
  * the on-device actor + critic loss accumulators drain to finite values
    (read at flush, never per-step D2H),
  * the per-batch diag means are 0 on GPU (CPU-only diag walk — same
    convention as SAC), and
  * mean_return is finite.

This locks in the DDPG GPU path unblocked by Phase 4.1 (SingleCriticStep
GPU, DDPGActorLoss GPU, DDPGTargetYStep GPU). Numeric convergence
validation on NVIDIA is a separate HW-gated step (Apple Metal FD/numeric
checks are unreliable — TF32 / launch-overhead artifacts).

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn/test_ddpg_trainer_gpu_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.activations import Tanh
from noeira.deep_agents.ddpg.trainer import DDPGTrainer
from noeira.deep_agents.training.blocks import UniformSampleGpuStep
from noeira.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 128
comptime CAP = 20_000
comptime WARMUP = 256
comptime TOTAL = 3_000

comptime ActorNet = Sequential[
    Linear[OBS, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, ACT], Tanh[ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN], Linear[HIDDEN, 1],
]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_ddpg_gpu_smoke() raises:
    print("--- DDPG GPU smoke ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = DDPGTrainer[
        "gpu",
        UniformSampleGpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](1e-4),
        critic_lr=Scalar[DT](1e-3),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        noise_scale=Scalar[DT](0.1),
        learning_starts=WARMUP,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs_self = env.get_obs_list()

    var step: Int = 0
    while step < TOTAL:
        for d in range(OBS):
            obs[d] = obs_self[d]
        trainer.select_action(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = nxt[d]
        trainer.record(
            obs, action, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
        step += 1
        _ = trainer.train_step(step)

    var m = trainer.flush_metrics()
    var al = m.policy_loss.to_f64()
    var cl = m.critic_loss.to_f64()
    var nup = m.n_updates.to_f64()
    var tsteps = m.train_steps.to_f64()
    var mq = m.mean_q.to_f64()
    var mr = m.reward_mean.to_f64()
    var mtgt = m.mean_target.to_f64()
    var mret = Float64(trainer.mean_return())
    print("  actor_loss  =", al)
    print("  critic_loss =", cl)
    print("  n_updates   =", nup)
    print("  train_steps =", tsteps)
    print("  mean_q      =", mq)
    print("  mean_target =", mtgt)
    print("  mean_reward =", mr)
    print("  mean_ret(10)=", mret)

    _finite(al, "policy_loss")
    _finite(cl, "critic_loss")
    _finite(mq, "mean_q")
    _finite(mr, "reward_mean")
    _finite(mtgt, "mean_target")
    _finite(mret, "mean_return")
    assert_true(nup > 0.0, "no training updates ran")
    assert_true(tsteps > 0.0, "no cumulative train steps recorded")
    assert_true(cl >= 0.0, "critic MSE loss should be >= 0")
    # Device-diag fix: these read a hard 0.0 on GPU before the device
    # reductions were wired. Pendulum reward is strictly negative.
    assert_true(mr < 0.0, "mean_reward should be < 0 on Pendulum (device diag)")
    assert_true(mq != 0.0, "mean_q is 0 on GPU (device accumulator unwired?)")
    print("PASS")


def main() raises:
    print("=" * 70)
    print("DDPG GPU smoke (Phase 4.1)")
    print("=" * 70)
    test_ddpg_gpu_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
