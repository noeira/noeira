"""Trait-based training metrics logger with pluggable backends.

Logger trait defines the interface. Concrete implementations:
  - NoOpLogger: does nothing (zero overhead, default)
  - CsvLogger: appends CSV rows to a local file
  - RemoteLogger: POSTs JSON batches to an HTTP server
  - CompositeLogger[A, B]: fans out to two loggers

Collection is pure Mojo with near-zero overhead, and so is the remote
backend: it serialises with `noeira/io/json.mojo` and hands the POST to
`noeira/io/http_sink.mojo`, which queues it for a background thread.

⚠ THE NETWORK IS NO LONGER ON THE TRAINING THREAD. `flush` used to POST
synchronously; against a dashboard answering in 100 ms that cost **629.7 ms**
for eleven batches, versus **0.003 ms** to queue the same eleven. A run whose
dashboard is DOWN now pays 4.6 ms for 2000 `log_scalar` calls and closes in
under 2 ms, where before it paid a connection attempt per flush.

⚠ THIS CALL IS WHY A TRAINING BINARY USED TO NEED A CPython AT ALL. `flush`
went through Python `urllib`, so every run — GPU training included — had to
find `libpython3.13`, which is what `pixi.toml`'s activation block pins and
why it names `RemoteLogger.flush` when it explains the pin. Nothing in the
training path imports Python now.

⚠ REQUIRES THE HTTP SHIM: `pixi run build-http`. A missing one is reported
once and every payload goes to the spool file instead — a dashboard that
cannot be reached must never take the training run with it.

⚠⚠ METRICS ARE HELD, RETRIED AND SPOOLED — NOT DROPPED. This used to drop:
the first failed POST latched the transport dead and "metrics will be dropped
for the rest of this run" followed, so a blip at minute one of a twelve-hour
run cost eleven hours and fifty-nine minutes of dashboard. Now an outage holds
the batches in memory and retries on a backoff capped at a minute; what cannot
be delivered live (a server rejection, an outage past `hold_bytes`, an outage
still going at `close()`) lands in the spool file, replayable with
`pixi run logger-replay <spool>`. `close()` prints the tally (`sink_report()`)
and the only loss it can show is `lost` — a spool that could not be written.
See `noeira/io/http_sink.mojo` for the policy.

Usage:
    # CSV only
    var logger = CsvLogger("logs/run_001.csv")

    # Remote only
    var logger = RemoteLogger(
        server_url="http://host:3000/api",
        run_name="ppo_halfcheetah_v3",
    )

    # Both (fan-out)
    var logger = CompositeLogger(
        CsvLogger("logs/run_001.csv"),
        RemoteLogger(server_url="http://host:3000/api"),
    )

    logger.set_config("algorithm", "PPO")
    logger.register()                      # announce the run BEFORE step 0
    logger.log_scalar("reward", avg_reward, step)
    logger.close()                         # sends status=done on the way out

⚠ THE RUN'S LIFE IS THREE CALLS: `register`, the metrics, `close`. `register`
is what makes a run that dies at step 0 visible at all — the remote backend
used to announce itself on the first flush, so a run that never got that far
never existed as far as the dashboard was concerned. `close` reports how it
ended; a driver that ended some other way says so with `finish("killed", ...)`
first, and the first call wins.
"""

from std.time import perf_counter_ns
from std.math import isnan, isinf


from noeira.io.http_sink import HttpPostSink, append_spool, spool_record
from noeira.io.json import JsonWriter
from noeira.io.fileio import write_text_atomic

# =============================================================================
# MetricEntry — single buffered data point
# =============================================================================


struct MetricEntry(Copyable, Movable):
    """A single scalar metric data point."""

    var step: Int
    var wall_time_ms: Float64
    var name: String
    var value: Float64

    def __init__(
        out self,
        step: Int,
        wall_time_ms: Float64,
        name: String,
        value: Float64,
    ):
        self.step = step
        self.wall_time_ms = wall_time_ms
        self.name = name
        self.value = value

    def __init__(out self, *, copy: Self):
        self.step = copy.step
        self.wall_time_ms = copy.wall_time_ms
        self.name = copy.name
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.step = move.step
        self.wall_time_ms = move.wall_time_ms
        self.name = move.name^
        self.value = move.value


# =============================================================================
# Logger Trait
# =============================================================================


trait Logger(Copyable, Deinitable, Movable):
    """Interface for training metrics loggers.

    All deep RL training loops and agent structs are parameterized on
    `L: Logger = NoOpLogger`.  When L = NoOpLogger every method is a no-op
    and `is_active()` returns False, giving zero overhead identical to the
    old null-pointer pattern.
    """

    comptime ENABLED: Bool = True

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        ...

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        ...

    def flush(mut self) raises:
        ...

    def register(mut self) raises:
        """Announce the run to the backend NOW, before the first metric.

        ⚠⚠ THE REMOTE BACKEND USED TO REGISTER ON THE FIRST FLUSH, AND THAT IS
        A HOLE. A run that dies before it logs anything — a bad config, an OOM
        while building the model, a compile that never reaches step 0 — never
        appeared on the dashboard at all, so the failure a liveness signal most
        needs to show is the one case with no row to mark. This is the call
        that closes it.

        ⚠ CALL IT AFTER `set_config`, NOT BEFORE. The registration payload
        carries the config, and every driver fills that in after constructing
        the logger; registering from a constructor would ship an empty config
        on every run and trade this hole for a different one.

        Idempotent. A backend with nothing to register does nothing.
        """
        ...

    def finish(mut self, status: String, outcome: String) raises:
        """Record how the run ENDED. `status` is the terminal state; `outcome`
        is the run's own summary of whether it was any good.

        ⚠ `close()` CALLS THIS WITH `done` IF THE DRIVER DID NOT, because
        reaching `close()` at all is a clean end. A run the kernel killed never
        arrives here, which is exactly the case only the server can conclude.

        Idempotent: the first call wins, so a driver that reports `killed` does
        not have it overwritten by the `done` from `close()`.
        """
        ...

    def close(mut self) raises:
        ...

    def set_config(mut self, key: String, value: String):
        ...

    def is_active(self) -> Bool:
        ...


# =============================================================================
# NoOpLogger — zero-overhead default
# =============================================================================


struct NoOpLogger(Logger):
    """Logger that does nothing. Default for all training loops and agents."""

    comptime ENABLED: Bool = False

    def __init__(out self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        pass

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        pass

    def flush(mut self) raises:
        pass

    def register(mut self) raises:
        pass

    def finish(mut self, status: String, outcome: String) raises:
        pass

    def close(mut self) raises:
        pass

    def set_config(mut self, key: String, value: String):
        pass

    def is_active(self) -> Bool:
        return False


# =============================================================================
# CsvLogger — local CSV file backend
# =============================================================================


struct CsvLogger(Logger):
    """Buffered CSV file logger.

    Accumulates MetricEntry objects and appends them to a CSV file when
    the buffer reaches `buffer_size` or on flush()/close().

    CSV format: step,wall_time_ms,name,value
    """

    var file_path: String
    var entries: List[MetricEntry]
    var buffer_size: Int
    var _start_ns: Int
    var _file_header_written: Bool
    var _total_logged: Int
    var _config_keys: List[String]
    var _config_vals: List[String]

    def __init__(
        out self,
        file_path: String,
        buffer_size: Int = 200,
    ):
        self.file_path = file_path
        self.entries = List[MetricEntry]()
        self.buffer_size = buffer_size
        self._start_ns = perf_counter_ns()
        self._file_header_written = False
        self._total_logged = 0
        self._config_keys = List[String]()
        self._config_vals = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.file_path = move.file_path^
        self.entries = move.entries^
        self.buffer_size = move.buffer_size
        self._start_ns = move._start_ns
        self._file_header_written = move._file_header_written
        self._total_logged = move._total_logged
        self._config_keys = move._config_keys^
        self._config_vals = move._config_vals^

    def config_path(self) -> String:
        """`<csv minus .csv>.config.kv` — `runs/<id>/metrics.config.kv` for a
        run's CSV. Beside the file it describes, so the two travel together."""
        var p = self.file_path
        if p.endswith(".csv"):
            return String(p[byte = 0 : p.byte_length() - 4]) + ".config.kv"
        return p + ".config.kv"

    def _write_config(mut self):
        """⚠ NEVER RAISES: a config file that cannot be written must not stop
        a run. It prints once instead, because a silently missing config is
        the failure this file exists to end."""
        if len(self._config_keys) == 0:
            return
        var out = String("")
        for i in range(len(self._config_keys)):
            # One line per pair; a newline inside a value would split it.
            out += (
                self._config_keys[i].replace("\n", " ") + "="
                + self._config_vals[i].replace("\n", " ") + "\n"
            )
        try:
            write_text_atomic(self.config_path(), out)
        except e:
            print("  [csv] could not write", self.config_path(), ":", e)

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        if isnan(value) or isinf(value):
            return
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        self.entries.append(MetricEntry(step, wall_time_ms, name, value))
        self._total_logged += 1
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        var n = min(len(names), len(values))
        for i in range(n):
            if isnan(values[i]) or isinf(values[i]):
                continue
            self.entries.append(
                MetricEntry(step, wall_time_ms, names[i], values[i])
            )
        self._total_logged += n
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def flush(mut self) raises:
        if len(self.entries) == 0:
            return
        var content = String("")
        if not self._file_header_written:
            content += "step,wall_time_ms,name,value\n"
            self._file_header_written = True
        for i in range(len(self.entries)):
            var e = self.entries[i].copy()
            content += (
                String(e.step)
                + ","
                + String(e.wall_time_ms)
                + ","
                + e.name
                + ","
                + String(e.value)
                + "\n"
            )
        with open(self.file_path, "a") as f:
            f.write(content)
        self.entries.clear()

    def register(mut self) raises:
        """Nothing to announce — the file IS the registration, and it is created
        by the first `flush`. The config is written here, because this is the
        point every driver has finished calling `set_config`."""
        self._write_config()

    def finish(mut self, status: String, outcome: String) raises:
        """A CSV has no room for a terminal state.

        ⚠ THIS IS NOT THE PLACE TO RECORD IT. `run.kv`'s `status=` is, and it
        is written by the run directory rather than smuggled into a metrics
        column that every reader of this file would then have to skip."""
        pass

    def close(mut self) raises:
        self.flush()
        self._write_config()

    def set_config(mut self, key: String, value: String):
        """Recorded in `config_path()`, written at `register` and `close`.

        ⚠⚠ THIS USED TO BE `pass`, AND THE DRIVERS WORKED AROUND IT BY LOGGING
        THEIR CONFIG AS METRICS. `cfg/lanes`, `cfg/tau`, `cfg/seed` … went
        into the CSV at step 0 as scalars (the SAC family driver, BFM,
        HIL-SERL), because only the remote half kept a config and "a CSV that
        cannot say what produced it" was worse. On the dashboard each became
        a one-point chart. The config now has a file of its own.
        """
        for i in range(len(self._config_keys)):
            if self._config_keys[i] == key:
                self._config_vals[i] = value
                return
        self._config_keys.append(key)
        self._config_vals.append(value)

    def is_active(self) -> Bool:
        return True

    def total_logged(self) -> Int:
        return self._total_logged

    def pending(self) -> Int:
        return len(self.entries)


# =============================================================================
# RemoteLogger — HTTP POST backend
# =============================================================================


struct RemoteLogger(Logger):
    """Buffered HTTP logger that POSTs JSON to a dashboard server.

    Sends metrics as JSON batches to `server_url/ingest` and registers
    the run at `server_url/runs` on first flush.

    ⚠ THE POST HAPPENS ON ANOTHER THREAD. `flush` serialises the batch and
    queues it; `noeira/io/http_sink.mojo` owns the client and the connection.
    One client for the run either way — a client per flush would pay a full
    TLS handshake every `buffer_size` metrics — but now none of it, handshake
    included, is on the training thread.

    ⚠ ORDER IS PRESERVED, WHICH THE `/runs` REGISTRATION DEPENDS ON. One ring
    and one worker means the registration queued by the first `flush` is sent
    before the `/ingest` batch behind it.

    ⚠ `close()` IS NOT OPTIONAL. It drains the queue, spools what did not
    drain, and joins the worker; without it, whatever is still held at process
    exit is lost — the spool is written by `close()`, not by the kernel.
    """

    var run_id: String
    var run_name: String
    var server_url: String
    var api_key: String
    var entries: List[MetricEntry]
    var buffer_size: Int
    var _start_ns: Int
    var _config_keys: List[String]
    var _config_vals: List[String]
    var _run_registered: Bool
    var _total_logged: Int
    var _sink: Optional[HttpPostSink]
    """The POST queue and its background thread. Built lazily ON THE FIRST
    PAYLOAD: constructing it spawns a thread, and a `RemoteLogger` with no
    `server_url` — which is the default in several drivers — must stay inert.

    ⚠ COPIES SHARE ONE SINK, hence one thread, one queue and one connection.
    That is the right meaning (two copies of a logger are one run) and it is
    also forced: `Logger` is `Copyable`, `CompositeLogger` copies its halves,
    and a libcurl easy handle may not be shared across threads."""
    var spool_path: String
    """Where payloads the dashboard never took are written, for
    `pixi run logger-replay`. Defaults to `logs/remote_spool/<run_id>.spool`;
    `run_logger` puts it in the run directory."""
    var _reported: Bool
    """Whether a PERMANENT problem (no shim, no sink) has been printed. Once
    per run, not once per flush."""
    var _was_down: Bool
    """The outage state last printed, so each transition prints once — down,
    then back — rather than once per flush for the whole outage."""
    var _reported_spool: Bool
    var _finished: Bool
    """Whether the terminal state has been sent.

    ⚠ THE FIRST `finish` WINS. `close()` sends `done` for any run that reaches
    it, so without this latch a driver that reported `killed` on its way out
    would have that overwritten by the `done` behind it — and a killed run
    filed as clean is worse than no record at all."""

    def __init__(
        out self,
        server_url: String,
        run_name: String = "",
        run_id: String = "",
        buffer_size: Int = 200,
        api_key: String = "",
        spool_path: String = "",
    ):
        self._start_ns = perf_counter_ns()
        if run_id.byte_length() > 0:
            self.run_id = run_id
        else:
            self.run_id = "run_" + String(self._start_ns)
        self.run_name = run_name if run_name.byte_length() > 0 else self.run_id
        self.server_url = server_url
        self.api_key = api_key
        self.entries = List[MetricEntry]()
        self.buffer_size = buffer_size
        self._config_keys = List[String]()
        self._config_vals = List[String]()
        self._run_registered = False
        self._total_logged = 0
        self._sink = None
        if spool_path.byte_length() > 0:
            self.spool_path = spool_path
        else:
            self.spool_path = "logs/remote_spool/" + self.run_id + ".spool"
        self._reported = False
        self._was_down = False
        self._reported_spool = False
        self._finished = False

    def __init__(out self, *, deinit move: Self):
        self.run_id = move.run_id^
        self.run_name = move.run_name^
        self.server_url = move.server_url^
        self.api_key = move.api_key^
        self.entries = move.entries^
        self.buffer_size = move.buffer_size
        self._start_ns = move._start_ns
        self._config_keys = move._config_keys^
        self._config_vals = move._config_vals^
        self._run_registered = move._run_registered
        self._total_logged = move._total_logged
        self._sink = move._sink^
        self.spool_path = move.spool_path^
        self._reported = move._reported
        self._was_down = move._was_down
        self._reported_spool = move._reported_spool
        self._finished = move._finished

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        if self.server_url.byte_length() == 0:
            return
        if isnan(value) or isinf(value):
            return
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        self.entries.append(MetricEntry(step, wall_time_ms, name, value))
        self._total_logged += 1
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        if self.server_url.byte_length() == 0:
            return
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        var n = min(len(names), len(values))
        for i in range(n):
            if isnan(values[i]) or isinf(values[i]):
                continue
            self.entries.append(
                MetricEntry(step, wall_time_ms, names[i], values[i])
            )
        self._total_logged += n
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def flush(mut self) raises:
        if self.server_url.byte_length() == 0:
            return
        if len(self.entries) == 0:
            # Nothing new, but a batch parked while the ring was full still
            # needs the owner thread to move it — the worker cannot.
            if self._sink:
                self._sink.value().pump()
            return

        if not self._run_registered:
            self._register_run()
            self._run_registered = True

        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), self.run_id)
        w.key(String("metrics"))
        w.begin_array()
        for i in range(len(self.entries)):
            var e = self.entries[i].copy()
            w.begin_object()
            w.member(String("step"), e.step)
            w.member(String("wall_time_ms"), e.wall_time_ms)
            w.member(String("name"), e.name)
            w.member(String("value"), e.value)
            w.end_object()
        w.end_array()
        w.end_object()

        self._post(self._ingest_url(), w.done())
        self.entries.clear()

    def register(mut self) raises:
        """POST `/runs` now, before the first metric. See the trait.

        ⚠ THIS IS THE ONLY WAY THE DASHBOARD LEARNS ABOUT A RUN THAT NEVER
        LOGS. `flush` keeps registering lazily for the drivers that never call
        this, so nothing regresses — but a lazily registered run is invisible
        until its first batch, and a run that dies before then stays invisible
        forever.
        """
        if self.server_url.byte_length() == 0 or self._run_registered:
            return
        self._register_run()
        self._run_registered = True

    def finish(mut self, status: String, outcome: String) raises:
        """POST the terminal state to `/runs/<id>/finish`. See the trait.

        ⚠ INERT FOR A RUN THAT WAS NEVER REGISTERED. There is no server row to
        finish, and inventing one at the end would advertise a run whose whole
        history is the fact that it stopped.

        ⚠ THE EMPTY-URL CLAUSE BELOW IS UNREACHABLE AND KEPT ANYWAY. A run with
        no `server_url` can never register, so the registration guard already
        covers it — `tests/core/test_run_lifecycle.mojo` confirms deleting the
        clause changes nothing observable. It stays because every public method
        here opens with the same inert check, and the one method that did not
        would be the one a later refactor trips over.
        """
        if (
            self.server_url.byte_length() == 0
            or self._finished
            or not self._run_registered
        ):
            return
        self.flush()
        self._finished = True
        self._post(self._finish_url(), self._finish_payload(status, outcome))

    def close(mut self) raises:
        """Flush, then drain the sink and join its thread.

        ⚠ THE DRAIN IS BOUNDED. `drain_ms` is a budget, not a promise — see
        `HttpPostSink`. A hung dashboard is bounded by the worker's close-time
        `dead` latch instead, at one client timeout rather than one per
        payload, and what did not drain is SPOOLED, not discarded.

        ⚠⚠ REACHING HERE IS ITSELF THE END SIGNAL. A run that arrives at
        `close()` finished cleanly, so it reports `done` unless the driver
        already said otherwise. The runs that never arrive — SIGKILL, OOM, a
        released instance — are the ones only the server can conclude anything
        about, and it does so from the heartbeat rather than from silence here.

        ⚠ THE FINISH IS QUEUED BEFORE THE DRAIN, NOT AFTER. One ring and one
        worker means it lands behind the last metric batch and ahead of the
        join; sending it after the drain would race the join it was meant to
        precede.
        """
        self.flush()
        self.finish(String("done"), String(""))
        if self._sink:
            self._report_transport()
            self._sink.value().close(drain_ms=3000)
            var final = self.sink_report()
            if final.byte_length() > 0:
                print(final)
            if self._sink.value().spooled() > 0:
                print(
                    "  [logger] replay what the dashboard missed with:"
                    " pixi run logger-replay " + self.spool_path
                )

    def set_config(mut self, key: String, value: String):
        for i in range(len(self._config_keys)):
            if self._config_keys[i] == key:
                self._config_vals[i] = value
                return
        self._config_keys.append(key)
        self._config_vals.append(value)

    def is_active(self) -> Bool:
        return self.server_url.byte_length() > 0

    def _base(self) -> String:
        return String(self.server_url.removesuffix("/"))

    def _ingest_url(self) -> String:
        return self._base() + "/ingest"

    def _runs_url(self) -> String:
        return self._base() + "/runs"

    def _finish_url(self) -> String:
        """`/runs/<id>/finish`.

        ⚠ THE ID IS INTERPOLATED, NOT ESCAPED, AND THAT IS A CONSTRAINT ON THE
        ID RATHER THAN A BUG HERE. Both shapes this tree mints are URL-safe by
        construction — `run_<ns>` from the fallback below, and the project
        layer's `<date>_<slug>_<hash8>`. A future id that is not must be
        rejected where it is minted, because a path segment repaired at the
        last moment stops matching the one written into `run.kv`.
        """
        return self._runs_url() + "/" + self.run_id + "/finish"

    def _ping_url(self) -> String:
        """`/runs/<id>/ping` — see `_finish_url` on why the id is interpolated.

        ⚠⚠ THE HEARTBEAT IS WHAT MAKES A SIGKILLED RUN KNOWABLE. `finish` covers
        a clean end and `status=killed` covers a Ctrl-C that reaches `close()`;
        neither runs for an OOM, a released instance, or a power cut. Those the
        server can only conclude from silence, and it can only do that if
        silence means something — which is what this route establishes.
        """
        return self._runs_url() + "/" + self.run_id + "/ping"

    def _ping_payload(self) raises -> String:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), self.run_id)
        w.end_object()
        return w.done()

    def _finish_payload(
        self, status: String, outcome: String
    ) raises -> String:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), self.run_id)
        w.member(String("status"), status)
        w.member(String("outcome"), outcome)
        w.end_object()
        return w.done()

    def _register_payload(self) raises -> String:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), self.run_id)
        w.member(String("run_name"), self.run_name)
        w.key(String("config"))
        w.begin_object()
        for i in range(len(self._config_keys)):
            w.member(self._config_keys[i], self._config_vals[i])
        w.end_object()
        w.end_object()
        return w.done()

    def _register_run(mut self) raises:
        self._post(self._runs_url(), self._register_payload())

    def _post(mut self, url: String, payload: String):
        """Queue a JSON POST. Returns immediately; the network happens on the
        sink's thread.

        ⚠ EVERY FAILURE IS SWALLOWED. A dead or slow dashboard must not be able
        to kill a training run or flood its stdout, so this reports at most
        once and returns. That is a deliberate asymmetry with the rest of
        `io/`, where a failed transfer raises.

        Measured: twenty flushes against a dashboard answering in 100ms cost
        **2090 ms** of training time synchronously and **0.7 ms** through the
        sink, arriving byte-identical and in order
        (`docs/design_spikes/spike_async_post_spsc_ring.mojo`).
        """
        try:
            if not self._sink:
                # ⚠ THE HEARTBEAT IS CONFIGURED HERE, at the one place the sink
                # is built, because the thread reads its ping URL once at start
                # and never again. A run with no `server_url` never reaches
                # this line, so it never acquires a heartbeat either.
                self._sink = Optional(
                    HttpPostSink(
                        api_key=self.api_key,
                        timeout_ms=5000,
                        ping_url=self._ping_url(),
                        ping_body=self._ping_payload(),
                        spool_path=self.spool_path,
                    )
                )
            if not self._sink.value().post(url, payload):
                if not self._reported:
                    self._reported = True
                    print(
                        "  [logger] a payload was LOST (oversize, or the spool "
                        + self.spool_path + " is not writable)"
                    )
        except e:
            # ⚠ NO SINK MEANS NO WORKER TO SPOOL FOR US, so this path spools
            # itself. Rare (a thread that would not start), but it is the one
            # place a payload could otherwise vanish without a count.
            if not append_spool(self.spool_path, spool_record(url, payload)):
                print("  [logger] a payload was LOST: " + String(e))
            if not self._reported:
                self._reported = True
                print(
                    "  [logger] could not start the POST sink: " + String(e)
                    + " — payloads go to " + self.spool_path
                )
        self._report_transport()

    def _report_transport(mut self):
        """Print each change in the transport's state, once.

        The worker never prints: it runs on another thread and would interleave
        with training output. It records into atomic cells and the owning
        thread reports here — on a flush, so a run that logs nothing for an
        hour reports its outage when it next does.
        """
        if not self._sink:
            return
        var s = self._sink.value()
        if s.shim_missing():
            if not self._reported:
                self._reported = True
                print(
                    "  [logger] the HTTP shim is missing — build it with"
                    " `pixi run build-http`. Metrics go to the spool "
                    + self.spool_path
                )
            return
        var down = s.down()
        if down and not self._was_down:
            self._was_down = True
            print(
                "  [logger] the dashboard is unreachable (last status "
                + String(s.last_status())
                + "); holding metrics and retrying with backoff — nothing is"
                " dropped."
            )
        elif not down and self._was_down:
            self._was_down = False
            print(
                "  [logger] the dashboard is back; "
                + String(s.held())
                + " held batches are being delivered."
            )
        if s.spooled() > 0 and not self._reported_spool:
            self._reported_spool = True
            print(
                "  [logger] some payloads could not go live and were written to "
                + self.spool_path
                + " — `pixi run logger-replay` sends them once the dashboard"
                " takes them."
            )

    def sink_report(self) -> String:
        """One line of delivery accounting, or empty if nothing was sent.

        ⚠ `lost` IS THE NUMBER THAT MUST BE ZERO, AND MUST BE VISIBLE. Every
        other outcome — delivered, spooled — still has the payload somewhere.
        `close()` prints this.
        """
        if not self._sink:
            return String("")
        var s = self._sink.value()
        var lost = s.dropped() + s.abandoned()
        var total = s.sent() + s.failed() + s.spooled() + lost + s.pings()
        if total == 0:
            return String("")
        var line = "  [logger] " + String(s.sent()) + " batches delivered"
        if s.held() > 0:
            line += ", " + String(s.held()) + " still held"
        if s.spooled() > 0:
            line += (
                ", " + String(s.spooled()) + " spooled to " + self.spool_path
            )
            if s.rejected() > 0:
                line += " (" + String(s.rejected()) + " rejected by the server)"
        line += ", " + String(lost) + " lost"
        if s.outages() > 0:
            line += (
                "; " + String(s.outages()) + " outage(s), "
                + String(s.failed()) + " failed attempts"
            )
        if s.pings() > 0:
            line += ", " + String(s.pings()) + " heartbeats"
        return line

    def total_logged(self) -> Int:
        return self._total_logged

    def pending(self) -> Int:
        return len(self.entries)

    def posts_attempted(self) -> Int:
        """Payloads accounted for — delivered, spooled, or lost. Each payload
        once, however many attempts it took (`failed()` counts attempts).

        ⚠ IT IS ONLY FINAL AFTER `close()`. Before the drain it is a snapshot of
        a live counter and says nothing about what is still in flight."""
        if not self._sink:
            return 0
        var s = self._sink.value()
        return s.sent() + s.spooled() + s.dropped() + s.abandoned()

    def registered(self) -> Bool:
        """Whether `/runs` has been queued. Diagnostics and gates."""
        return self._run_registered

    def finished(self) -> Bool:
        """Whether a terminal state has been queued. Diagnostics and gates."""
        return self._finished


# =============================================================================
# CompositeLogger — fan-out to two loggers
# =============================================================================


struct CompositeLogger[A: Logger, B: Logger](Logger):
    """Fans out log calls to two underlying loggers.

    Usage:
        var logger = CompositeLogger(
            CsvLogger("logs/run.csv"),
            RemoteLogger(server_url="http://host:3000/api"),
        )
    """

    var a: Self.A
    var b: Self.B

    def __init__(out self, a: Self.A, b: Self.B):
        self.a = a.copy()
        self.b = b.copy()

    def __init__(out self, *, deinit move: Self):
        self.a = move.a^
        self.b = move.b^

    def __init__(out self, *, copy: Self):
        self.a = copy.a.copy()
        self.b = copy.b.copy()

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        self.a.log_scalar(name, value, step)
        self.b.log_scalar(name, value, step)

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        self.a.log_scalars(names, values, step)
        self.b.log_scalars(names, values, step)

    def flush(mut self) raises:
        self.a.flush()
        self.b.flush()

    def register(mut self) raises:
        self.a.register()
        self.b.register()

    def finish(mut self, status: String, outcome: String) raises:
        self.a.finish(status, outcome)
        self.b.finish(status, outcome)

    def close(mut self) raises:
        self.a.close()
        self.b.close()

    def set_config(mut self, key: String, value: String):
        self.a.set_config(key, value)
        self.b.set_config(key, value)

    def is_active(self) -> Bool:
        return True

    def __deinit__(deinit self):
        _ = self.a^
        _ = self.b^


