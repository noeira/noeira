"""SAC training on Pendulum V1 via the storage `SAC[...]` facade + a run logger.

Demonstrates the Track-1 monitoring path: pass a `logger=` kwarg to
`agent.train_single()` and the off-policy driver emits `avg_reward` +
`episodes` at the `print_every` cadence automatically. After training, the
agent is round-tripped through `save()` / `load()`.

The run lives in a `RunContext` (project `classic-control`): the checkpoint,
`metrics.csv` and `run.kv` all land in `runs/<id>/`. `run_logger(run)` writes
the CSV and streams to the monitor named in `.env` (no URL there => the remote
half is inert, so this example runs end-to-end without a server).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_training_remote.mojo
"""

from std.random import seed

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.deep_agents.data.any_replay import AnyReplay

from noeira.envs.pendulum import PendulumEnv


comptime EnvT = PendulumEnv[DT]
comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime NUM_STEPS = 3_000
comptime PRINT_EVERY = 500


comptime SAC_T = SACAgent[
    "cpu",
    ReplaySampleStep[
        AnyReplay["cpu", OBS_DIM, ACT_DIM, REPLAY_CAPACITY], BATCH
    ],
    SACActorNet[OBS_DIM, ACT_DIM, HIDDEN],
    SACCriticNet[OBS_DIM, ACT_DIM, HIDDEN],
]


def _make_agent() raises -> SAC_T:
    return SAC[
        "cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
    ](
        window_size=10,
        initial_episode_fill=-1250.0,
    )


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC + run logger demo — Pendulum V1 (CPU)")
    print("=" * 70)

    # 1. Open the run and its logger (metrics.csv + the monitor from `.env`).
    # If the dashboard server isn't running, the remote half silently no-ops.
    var run = RunContext(
        project=String("classic-control"),
        driver=String("examples/pendulum/pendulum_sac_training_remote.mojo"),
        slug=String("sac-pendulum"),
        env=String("builtin:classic-control/pendulum"),
        seed=42,
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("Run:", run.dir)
    var logger = run_logger(run, buffer_size=50)
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "Pendulum-v1")
    logger.set_config("seed", "42")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    # 2. Build the agent + env.
    var agent = _make_agent()
    var env = EnvT()

    # 3. Single train() call — the driver flushes `avg_reward` and `episodes`
    # through the logger at `print_every` cadence automatically.
    _ = agent.train_single[
        EnvT,
        L=RunLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
    )

    # 4. Save the agent (single-file `.ckpt`) and hand it to the artifact
    # sink BEFORE the run is finished (finish_run closes the sink).
    agent.save(checkpoint_path)
    announce_checkpoint(checkpoint_path, artifacts, run.dir)
    var sent = logger.b.total_logged()
    finish_run(
        run, logger, artifacts,
        String("mean_return_10=") + String(agent.mean_return()),
    )
    _ = logger  # lifetime extender for logger_ptr

    print("=" * 70)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Total logged points:            ", sent)
    print("Saved agent state to:           ", checkpoint_path)

    # 5. Probe greedy action, reload into a fresh agent, confirm it matches.
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    probe_obs[0] = Scalar[DT](0.5)
    probe_obs[1] = Scalar[DT](0.8)
    probe_obs[2] = Scalar[DT](-1.2)
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    var fresh = _make_agent()
    fresh.load(checkpoint_path)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    fresh.select_greedy_action(probe_obs, act_after)

    print(
        "Greedy action before save:      ", act_before[0],
        " after load:", act_after[0],
    )
    print("=" * 70)
