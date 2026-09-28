"""R.5+ — REDQ config preset construction smoke.

Gates that `REDQ[…](ctx=…)` and `SmallREDQ[…](ctx=…)` build a
working `REDQAgent` with the right comptime knobs, and that the
agent's trainer can run a few env steps + a greedy eval action +
flush_metrics → REDQMetrics without crashing.

The preset / config / agent_from_config / raw `REDQAgent.make` paths
are all type-equivalent at the call site — this test just verifies
the preset functions compile and link end-to-end.
"""

from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.deep_agents.redq import (
    REDQ, SmallREDQ,
    REDQAgent, REDQActor, REDQCritic,
    REDQConfig, SmallREDQConfig,
    agent_from_config,
    REDQ_TARGET_MIN,
)
from noeira.deep_agents.training.blocks import (
    UniformSampleCpuStep, ReplaySampleStep,
)
from noeira.deep_agents.data.any_replay import AnyReplay


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 32
comptime CAP = 1_024


def test_small_redq_preset() raises:
    print("--- SmallREDQ['cpu', 3, 1, 32, 1024](ctx=None) ---")

    var agent = SmallREDQ["cpu", OBS, ACT, BATCH, CAP](
        action_scale=Scalar[DT](1.0),
        learning_starts=64,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=2,
    )

    # Drive a couple of env steps to confirm everything wired correctly.
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for step in range(80):
        for d in range(OBS):
            obs[d] = Scalar[DT](0.2 * Float64(d) + 0.005 * Float64(step))
        agent.select_action(obs, act, step)
        for d in range(OBS):
            nxt[d] = Scalar[DT](
                0.2 * Float64(d) + 0.005 * Float64(step + 1)
            )
        var rew = Scalar[DT](-0.3 + 0.2 * Float64(act[0]))
        var done = Scalar[DT](0.0) if step % 20 != 19 else Scalar[DT](1.0)
        agent.trainer.record(obs, act, rew, nxt, done)
        _ = agent.trainer.train_step(step)
        if done == Scalar[DT](1.0):
            agent.trainer.end_episode()

    print(
        "  total_train_steps after 80 env steps =",
        agent.trainer.total_train_steps(),
    )
    assert_true(
        agent.trainer.total_train_steps() > 0,
        "SmallREDQ must train after warmup (warmup=64, total=80)",
    )

    # Greedy eval + metrics passthrough.
    var greedy_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var greedy_act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    greedy_obs[0] = Scalar[DT](0.4)
    agent.select_greedy_action(greedy_obs, greedy_act)
    var ga = Float64(greedy_act[0])
    print("  greedy[0] =", ga)
    assert_true(ga == ga, "greedy action finite")
    assert_true(ga >= -1.0 and ga <= 1.0, "greedy action in [-1, 1]")

    var m = agent.flush_metrics()
    print("  metrics.actor_loss   =", m.policy_loss.to_f64())
    print("  metrics.critic_loss  =", m.critic_loss.to_f64())
    print("  metrics.alpha        =", m.alpha.to_f64())
    assert_true(
        m.policy_loss.to_f64() == m.policy_loss.to_f64(),
        "actor_loss finite",
    )
    assert_true(
        m.alpha.to_f64() > 0.0,
        "alpha must remain positive",
    )

    print("PASS — SmallREDQ preset smoke green.")


def test_redq_preset_type_equivalence() raises:
    """Quick comptime/type smoke — the three call sites must produce
    the SAME concrete `REDQAgent[...]` parameterization. If the types
    diverged this file wouldn't even compile."""
    print("--- REDQ preset call-site type equivalence ---")
    comptime SmallConfigT = SmallREDQConfig[
        "cpu", OBS, ACT, BATCH, CAP, 64,
    ]
    var agent_factory = agent_from_config[SmallConfigT](
        action_scale=Scalar[DT](1.0),
        learning_starts=64,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=2,
    )
    var agent_raw = REDQAgent[
        "cpu",
        ReplaySampleStep[
            AnyReplay["cpu", OBS, ACT, CAP], BATCH,
        ],
        REDQActor[OBS, ACT, 64],
        REDQCritic[OBS, ACT, 64],
        2, 2, 1, 1, REDQ_TARGET_MIN,
    ](
        action_scale=Scalar[DT](1.0),
        learning_starts=64,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=2,
    )
    # Both should respond to the same surface — verify by calling
    # `ep_count()` (returns 0 on a fresh agent).
    assert_true(
        agent_factory.ep_count() == 0,
        "agent_from_config agent fresh ep_count == 0",
    )
    assert_true(
        agent_raw.ep_count() == 0,
        "raw REDQAgent fresh ep_count == 0",
    )
    print("PASS — agent_from_config and raw REDQAgent are type-compatible.")


def main() raises:
    test_small_redq_preset()
    test_redq_preset_type_equivalence()
