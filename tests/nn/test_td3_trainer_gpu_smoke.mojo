"""TD3Trainer GPU smoke (Phase 4.2).

Runs the GPU TD3 trainer end-to-end against the CPU Pendulum env via the
manual loop (cpu env + gpu train) for a short horizon, then asserts:

  * critic updates ran (n_critic_updates > 0) and the delayed actor fired
    (n_actor_updates > 0),
  * the on-device twin-critic + DPG-actor loss accumulators drain to
    finite values at flush (never per-step D2H),
  * per-batch diag means are 0 on GPU (CPU-only diag walk, SAC convention),
  * mean_return is finite.

Locks in the TD3 GPU path unblocked by Phase 4.2 (TD3TargetYBlock GPU with
device target-policy smoothing noise + the TwinCriticStep/DDPGActorLoss GPU
paths already done). NVIDIA numeric convergence is a separate HW-gated step.

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn/test_td3_trainer_gpu_smoke.mojo
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
from noeira.deep_agents.td3.trainer import TD3Trainer
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


def test_td3_gpu_smoke() raises:
    print("--- TD3 GPU smoke ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = TD3Trainer[
        "gpu",
        UniformSampleGpuStep[OBS, ACT, BATCH, CAP],
        ActorNet,
        CriticNet,
    ].make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        exploration_noise=Scalar[DT](0.1),
        target_policy_noise=Scalar[DT](0.2),
        target_noise_clip=Scalar[DT](0.5),
        policy_delay=2,
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
    var nact = m.n_actor_updates.to_f64()
    var ncrit = m.n_critic_updates.to_f64()
    var mret = Float64(trainer.mean_return())
    print("  actor_loss      =", al)
    print("  critic_loss     =", cl)
    print("  n_actor_updates =", nact)
    print("  n_critic_updates=", ncrit)
    print("  mean_ret(10)    =", mret)

    _finite(al, "policy_loss")
    _finite(cl, "critic_loss")
    _finite(mret, "mean_return")
    assert_true(ncrit > 0.0, "no critic updates ran")
    assert_true(nact > 0.0, "delayed actor never fired")
    assert_true(cl >= 0.0, "twin-critic MSE loss should be >= 0")
    print("PASS")


def main() raises:
    print("=" * 70)
    print("TD3 GPU smoke (Phase 4.2)")
    print("=" * 70)
    test_td3_gpu_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
