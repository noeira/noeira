# +--------------------------------------------------------------------------+ #
# | Opening and closing a run's loggers and uplink in the right order
# +--------------------------------------------------------------------------+ #
"""The two halves every training driver repeats around its loop.

    var run = RunContext(project="so101", driver="examples/...", seed=seed)
    var logger = run_logger(run)                 # metrics.csv + the monitor
    logger.set_config("algorithm", "SAC")        # ... then register
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    ...                                          # run.checkpoint_path("last")
    finish_run(run, logger, artifacts, "eval_return=812.4")

⚠⚠ `finish_run` EXISTS BECAUSE THE ORDER WAS GOT WRONG AT THE SITES THAT
WROTE IT INLINE. The SAC family driver closed its logger and THEN set the
outcome, so the monitor's `/finish` had already gone out with an empty one
(`close()` sends `done, ""` for any run that did not `finish` first); FB and
BFM never set one. The dashboard then cannot tell a good run from a bad one —
which was the point of the outcome. One function, one order:

    1. `logger.finish(status, outcome)`  — the monitor gets the verdict
    2. `logger.close()`                  — metrics.csv whole, queue drained
    3. `close_sink(artifacts)`           — last checkpoint uploaded, reported
    4. `run.set_outcome` + `run.close`   — run.kv: outcome, status, artifacts

⚠ `run_logger` RETURNS AN UNREGISTERED LOGGER, on purpose. `/runs` carries
the config, and the caller's `set_config` calls come after construction; the
caller registers (`register_run`) once the config is complete.
"""

from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger, Logger
from noeira.core.run import RunContext
from noeira.io.artifact_sink import ArtifactSink, close_sink


comptime RunLogger = CompositeLogger[CsvLogger, RemoteLogger]
"""`metrics.csv` in the run directory, plus the monitor when `.env` names one
(an empty URL makes the remote half inert — a box with no credentials still
trains, and still has its CSV)."""


def run_logger(
    ref run: RunContext,
    buffer_size: Int = 64,
    env_path: String = String(".env"),
) raises -> RunLogger:
    """The run's logger: CSV at `run.metrics_path()`, remote under `run.id`.

    ⚠ THE REMOTE RUN ID IS THE RUN'S ID, never one the logger minted. That is
    what makes a dashboard row, a `run.kv` and a checkpoint directory the same
    object (`core/run.mojo`, pain 2).

    What the monitor never took is spooled to `<run dir>/remote.spool`, beside
    the `metrics.csv` it duplicates — `pixi run logger-replay` sends it.
    """
    var url = String("")
    var key = String("")
    try:
        var env = load_dotenv(env_path)
        url = env.get("NOEIRA_CLOUD_URL", "")
        key = env.get("NOEIRA_CLOUD_API_KEY", "")
    except:
        pass
    return CompositeLogger(
        CsvLogger(run.metrics_path(), buffer_size=buffer_size),
        RemoteLogger(
            server_url=url,
            run_name=run.name(),
            run_id=run.id,
            buffer_size=buffer_size,
            api_key=key,
            spool_path=run.dir + "/remote.spool",
        ),
    )


def finish_run[L: Logger](
    mut run: RunContext,
    mut logger: L,
    mut artifacts: Optional[ArtifactSink],
    outcome: String = String(""),
    status: String = String("done"),
) raises:
    """End a run: verdict to the monitor, drain, upload, then `run.kv`.

    `outcome` is the run's own best numbers (`success_rate=0.82`,
    `eval_return=812.4`); `status` is `done` unless the driver knows better
    (`killed`, `crashed`). See the module header for why the order is fixed.
    """
    logger.finish(status, outcome)
    logger.close()
    close_sink(artifacts)
    if outcome.byte_length() > 0:
        run.set_outcome(outcome)
    run.close(status)
