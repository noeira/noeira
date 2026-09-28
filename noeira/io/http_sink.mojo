# +--------------------------------------------------------------------------+ #
# | HTTP POSTs that do not block the thread that asked for them
# +--------------------------------------------------------------------------+ #
"""Fire-and-forget HTTP POSTs, drained by one background thread.

    var sink = HttpPostSink(api_key=key)
    _ = sink.post(url, json_body)      # ~microseconds, never touches the network
    ...
    sink.close(drain_ms=2000)          # flush what is queued, then join

## Why

`RemoteLogger.flush` used to POST synchronously from the training thread.
Against a dashboard that answers in 100 ms, twenty flushes cost **2090 ms of
training time**; through this sink the same twenty cost **0.7 ms** and arrive
byte-identical and in order
(`docs/design_spikes/spike_async_post_spsc_ring.mojo`).

## The policy is HOLD, RETRY, SPOOL — never stall, never drop

⚠⚠ THIS USED TO BE "DROP", AND THE FIRST FAILED POST ENDED THE RUN'S METRICS.
One refused connection latched the worker dead and every payload after it was
discarded — so a dashboard blip in the first minute of a twelve-hour run left
the dashboard with one minute of it. Now nothing a run logs is thrown away:

  * The worker drains the ring into a FIFO of its own as fast as payloads
    arrive, so the ring stays empty and `post()` stays a `memcpy` — during an
    outage too. If the ring is momentarily full (the worker is blocked in one
    POST), `post()` parks the payload in an owner-side overflow and re-offers it
    on the next call. Neither side ever blocks the training thread.
  * A transport error or a retryable status (408/425/429/5xx) keeps the payload
    at the HEAD of the FIFO and retries it on a capped exponential backoff
    (`retry_min_ms` doubling to `retry_max_ms`). The heartbeat is suppressed
    while down; the retry itself is the probe. The first 2xx clears the outage.
  * What cannot go live ends in the SPOOL FILE (`spool_path`), one record per
    POST, replayable with `replay_spool` (`pixi run logger-replay <file>`):
    a payload the server rejected (non-retryable 4xx, or a 5xx past
    `max_status_retries` — a poison payload must not block the queue), the
    newest payloads once the held FIFO passes `hold_bytes`, and whatever is
    still held when `close()` runs out of drain budget.

`abandoned()` counts the only true losses left: a payload with no spool to go
to (no `spool_path`, or the write failed). `dropped()` counts payloads larger
than a ring slot — a caller bug. Both are printed by `RemoteLogger.close`.

⚠ A RETRY CAN DUPLICATE. A POST whose response was lost after the server
committed it is sent again; `/ingest` stores rows without a uniqueness key, so
that batch appears twice. A duplicated batch is recoverable; a dropped one is
not.

## Ordering

One ring, one worker, FIFO, and a failed payload is retried IN PLACE — nothing
overtakes it. So a `/runs` registration posted before an `/ingest` batch is
still sent first, which is what the dashboard requires. Two sinks would not
give you that. What goes to the spool leaves the live order: a replay lands
after the payloads that followed it, which `/ingest` does not mind (every row
carries its step).

## What crosses the thread boundary

Bytes only, in one frame per POST:

    [ Int32 url_len ][ url_len bytes of URL ][ the rest is the body ]

The worker owns its own `HttpClient`. That is not a nicety: `io/http.mojo`
states the rule outright — a libcurl easy handle must not be shared across
threads — and `native/nra_http.c` is already prepared for this, with
`pthread_once` around `curl_global_init` (`:123`) and `CURLOPT_NOSIGNAL`
(`:581`).

⚠ NOTHING IS REPORTED FROM THE WORKER THREAD. Failures land in atomic cells and
the OWNING thread prints them, so a dead dashboard cannot interleave garbage
into training output from a second thread.

⚠ A HUNG DASHBOARD AT CLOSE IS BOUNDED BY THE `dead` LATCH, NOT BY `drain_ms`.
There is no bounded `pthread_join`, so the worker itself must give up: once
`close()` has been called, the first transport failure latches it dead and the
rest of the FIFO goes to the spool. Without that, a drain pays the client
timeout once per queued payload — measured at 15.0 s for three payloads,
versus 5.0 s for eight with the latch
(`docs/design_spikes/spike_bounded_close_hung_dashboard.mojo`). DURING the run
the latch is not used: that is the backoff's job, and a latch there is what
used to throw away every metric after the first blip.
"""

from std.memory import ArcPointer, Pointer, unsafe_memcpy
from std.os import makedirs
from std.time import perf_counter_ns

from ..core.concurrent.block import SharedBlock
from ..core.concurrent.ring import SharedRing
from ..core.concurrent.thread import sleep_us
from ..core.concurrent.worker import (
    POLL_DID_WORK,
    POLL_IDLE,
    BackgroundThread,
    BackgroundWorker,
    WorkerCtl,
)
from .fileio import parent_dir, read_file_bytes, remove_file, write_text_atomic
from .http import HttpClient, http_shim_available


# ── stat cells, written by the worker, read by the owner ──────────────────

comptime STAT_SENT: Int = 0
"""Payloads delivered (a 2xx). Heartbeats are counted in `STAT_PINGS`."""
comptime STAT_FAILED: Int = 1
"""ATTEMPTS that raised or came back non-2xx, heartbeats included. A payload
retried five times through an outage counts five here and once elsewhere."""
comptime STAT_ABANDONED: Int = 2
"""Payloads LOST: they could not be delivered and there was no spool to write
them to (no `spool_path`, or the write failed). ⚠ THE ONE NUMBER THAT MUST BE
ZERO."""
comptime STAT_LAST_STATUS: Int = 3
"""HTTP status of the last completed POST. -1 for a transport error."""
comptime STAT_DEAD: Int = 4
"""1 once the worker has given up on the network for good: no shim, no client,
or a transport failure after `close()` was called. Everything from then on
goes to the spool."""
comptime STAT_NO_SHIM: Int = 5
"""1 if `libnra_http` was missing when the worker started."""
comptime STAT_PINGS: Int = 6
"""Heartbeats this worker sent of its own accord. ⚠ REPORT THIS BESIDE `sent()`:
a run whose only traffic is pings is silent for a reason worth knowing."""
comptime STAT_REJECTED: Int = 7
"""Payloads the server refused for good — a non-retryable status, or a
retryable one past `max_status_retries`. Spooled, so also in `STAT_SPOOLED`."""
comptime STAT_SPOOLED: Int = 8
"""Payloads written to the spool file by the worker."""
comptime STAT_DOWN: Int = 9
"""1 while the transport is failing and payloads are being HELD for retry."""
comptime STAT_OUTAGES: Int = 10
"""Times the transport went from up to down."""
comptime STAT_HELD: Int = 11
"""Payloads in the worker's FIFO right now, waiting for the dashboard."""
comptime STAT_HELD_BYTES: Int = 12
"""Their size, for the report and the `hold_bytes` cap."""
comptime STAT_CELLS: Int = 16


comptime DEFAULT_CAPACITY: Int = 16
comptime DEFAULT_PING_INTERVAL_MS: Int = 60_000
"""How long the worker stays silent before saying "still here".

⚠ THIS IS THE CLOCK'S RESOLUTION, NOT ITS THRESHOLD. The server calls a run
`stale` after ~3 min and `lost` after 30 — three missed pings and ten times
that. One minute is chosen so that a legitimate silence (an eval pass, a ~15
minute physics3d kernel build on the 5090) never approaches even the first
boundary."""

comptime DEFAULT_SLOT_BYTES: Int = 256 * 1024
"""256 KB per slot. A `RemoteLogger` flush of 200 metrics is ~20 KB, so this
has generous headroom; an over-long payload is refused and counted in
`oversize()` rather than truncated."""

comptime DEFAULT_RETRY_MIN_MS: Int = 1_000
comptime DEFAULT_RETRY_MAX_MS: Int = 60_000
"""The outage backoff: the first retry after 1 s, doubling to one a minute.

⚠ THE CAP IS WHAT A LONG RUN FEELS. A dashboard back after an hour is found
within a minute of its return, and a black-holed host — where each attempt
costs a full client timeout — costs the worker at most one timeout a minute."""

comptime DEFAULT_MAX_STATUS_RETRIES: Int = 10
"""Attempts one payload gets against a server that ANSWERS with a retryable
status (5xx, 429, …) before it is spooled and the queue moves on. ~5 min on
the default backoff. A transport error is retried without limit — the server
is not there to have an opinion about the payload."""

comptime DEFAULT_HOLD_BYTES: Int = 256 * 1024 * 1024
"""How much the worker holds in memory through an outage before newer payloads
go straight to the spool. 256 MB is ~13k `/ingest` batches of 200 metrics —
many hours of a run flushing every few seconds."""


@always_inline
def _frame_len(head: String, tail: String) -> Int:
    return 4 + head.byte_length() + tail.byte_length()


# ── the worker ────────────────────────────────────────────────────────────


struct HttpPostWorker(BackgroundWorker):
    """Drains the ring into a FIFO it owns, and POSTs the FIFO's head.

    ⚠ THE CLIENT IS BUILT IN `on_start`, ON THIS THREAD. Building it in the
    constructor would create the libcurl handle on the owning thread and use it
    here, which is exactly what `io/http.mojo` forbids.

    ⚠ THE RING IS EMPTIED EVERY LAP, DOWN OR NOT. The FIFO — not the ring — is
    where an outage waits, so the producer keeps finding free slots while the
    dashboard is gone. A worker that popped only what it could send would fill
    the 16-slot ring in 16 flushes and hand every later payload to the owner's
    overflow for the rest of the outage.
    """

    var ring: SharedRing
    var stats: SharedBlock
    var api_key: String
    var timeout_ms: Int
    var client: Optional[HttpClient]
    var dead: Bool
    """Given up on the network for good (no shim, no client, or a transport
    failure after `close()`). Thread-local, mirrored into `STAT_DEAD`."""
    var ping_url: String
    """Empty disables the heartbeat entirely. A sink with no run to speak for
    must stay a pure queue."""
    var ping_body: String
    var ping_interval_ns: Int64
    var last_send_ns: Int64
    """When this thread last put bytes on the wire, successfully or not.
    Thread-local, like `dead` — nobody else reads it."""
    var spool_path: String
    """Where undeliverable payloads go. Empty means they are LOST — counted in
    `STAT_ABANDONED`, which is how a sink with no spool admits it."""
    var retry_min_ns: Int64
    var retry_max_ns: Int64
    var max_status_retries: Int
    var hold_bytes: Int
    var urls: List[String]
    """The FIFO, as two parallel lists read from `head`. Thread-local."""
    var bodies: List[String]
    var head: Int
    var held_bytes: Int
    var down: Bool
    """The transport is failing and the FIFO is being held for retry."""
    var backoff_ns: Int64
    var retry_at_ns: Int64
    var head_status_failures: Int
    """Retryable-status answers the CURRENT head has had. Reset on every pop."""

    def __init__(
        out self,
        ring: SharedRing,
        stats: SharedBlock,
        api_key: String,
        timeout_ms: Int,
        ping_url: String = String(""),
        ping_body: String = String("{}"),
        ping_interval_ms: Int = DEFAULT_PING_INTERVAL_MS,
        spool_path: String = String(""),
        retry_min_ms: Int = DEFAULT_RETRY_MIN_MS,
        retry_max_ms: Int = DEFAULT_RETRY_MAX_MS,
        max_status_retries: Int = DEFAULT_MAX_STATUS_RETRIES,
        hold_bytes: Int = DEFAULT_HOLD_BYTES,
    ):
        self.ring = ring
        self.stats = stats
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self.client = None
        self.dead = False
        self.ping_url = ping_url
        self.ping_body = ping_body
        self.ping_interval_ns = Int64(ping_interval_ms) * 1_000_000
        self.last_send_ns = 0
        self.spool_path = spool_path
        self.retry_min_ns = Int64(max(retry_min_ms, 1)) * 1_000_000
        self.retry_max_ns = Int64(max(retry_max_ms, retry_min_ms, 1)) * 1_000_000
        self.max_status_retries = max(max_status_retries, 1)
        self.hold_bytes = hold_bytes
        self.urls = List[String]()
        self.bodies = List[String]()
        self.head = 0
        self.held_bytes = 0
        self.down = False
        self.backoff_ns = self.retry_min_ns
        self.retry_at_ns = 0
        self.head_status_failures = 0

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring
        self.stats = move.stats
        self.api_key = move.api_key^
        self.timeout_ms = move.timeout_ms
        self.client = move.client^
        self.dead = move.dead
        self.ping_url = move.ping_url^
        self.ping_body = move.ping_body^
        self.ping_interval_ns = move.ping_interval_ns
        self.last_send_ns = move.last_send_ns
        self.spool_path = move.spool_path^
        self.retry_min_ns = move.retry_min_ns
        self.retry_max_ns = move.retry_max_ns
        self.max_status_retries = move.max_status_retries
        self.hold_bytes = move.hold_bytes
        self.urls = move.urls^
        self.bodies = move.bodies^
        self.head = move.head
        self.held_bytes = move.held_bytes
        self.down = move.down
        self.backoff_ns = move.backoff_ns
        self.retry_at_ns = move.retry_at_ns
        self.head_status_failures = move.head_status_failures

    def on_start(mut self, ctl: WorkerCtl):
        # The heartbeat is measured from the START of the run, not from zero:
        # otherwise the first `poll` would find itself infinitely overdue and
        # ping before the registration it is supposed to follow.
        self.last_send_ns = Int64(perf_counter_ns())
        if not http_shim_available():
            self.dead = True
            self.stats.release_store(STAT_NO_SHIM, Int64(1))
            self.stats.release_store(STAT_DEAD, Int64(1))
            return
        try:
            var c = HttpClient(self.timeout_ms, self.timeout_ms)
            if self.api_key.byte_length() > 0:
                c.bearer(self.api_key)
            self.client = Optional(c^)
        except:
            self.dead = True
            self.stats.release_store(STAT_DEAD, Int64(1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        var moved = self._absorb()
        if self._held() == 0:
            if moved:
                return POLL_DID_WORK
            return self._maybe_ping(ctl)

        # Nowhere to send it, or no time left to: the spool, not the floor.
        if self.dead or not self.client or ctl.drain_deadline_passed():
            self._spool_all()
            return POLL_DID_WORK

        # ⚠ IDLE ONLY WHILE NOT STOPPING. The driver reads idle-while-stopping
        # as "drained" and exits; a stopping worker skips the backoff instead
        # and makes its one last attempt now.
        if (
            self.down
            and not ctl.should_stop()
            and Int64(perf_counter_ns()) < self.retry_at_ns
        ):
            return POLL_IDLE

        self._send_head(ctl)
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        # Normally empty by now; the poll loop only exits once the FIFO is.
        _ = self._absorb()
        if self._held() > 0:
            self._spool_all()

    # ── the FIFO ──────────────────────────────────────────────────────────

    @always_inline
    def _held(self) -> Int:
        return len(self.urls) - self.head

    def _publish_held(mut self):
        self.stats.release_store(STAT_HELD, Int64(self._held()))
        self.stats.release_store(STAT_HELD_BYTES, Int64(self.held_bytes))

    def _absorb(mut self) -> Bool:
        """Move every frame in the ring onto the FIFO. True if any moved.

        ⚠ PAST `hold_bytes` THE NEWEST PAYLOAD IS SPOOLED, NOT THE OLDEST. The
        oldest is the `/runs` registration every later `/ingest` depends on;
        evicting it would leave the dashboard, once back, receiving metrics
        for a run it has never heard of.
        """
        var moved = False
        while True:
            var claim = self.ring.begin_pop()
            if not claim.ok():
                break
            moved = True
            var url: String
            var body: String
            try:
                url, body = unframe(claim.data(), claim.len)
            except:
                # A malformed frame has no URL to deliver it to. Only a bug in
                # `frame_into` makes one, and it is counted as the loss it is.
                _ = self.stats.fetch_add(STAT_ABANDONED, Int64(1))
                self.ring.end_pop()
                continue
            self.ring.end_pop()
            var n = url.byte_length() + body.byte_length()
            if self._held() > 0 and self.held_bytes + n > self.hold_bytes:
                self._spool_text(spool_record(url, body), 1)
            else:
                self.urls.append(url^)
                self.bodies.append(body^)
                self.held_bytes += n
        if moved:
            self._publish_held()
        return moved

    def _pop_head(mut self):
        self.held_bytes -= (
            self.urls[self.head].byte_length()
            + self.bodies[self.head].byte_length()
        )
        self.head += 1
        self.head_status_failures = 0
        if self.head == len(self.urls):
            self.urls.clear()
            self.bodies.clear()
            self.head = 0
        elif self.head >= 4096:
            # Compact a FIFO that has been draining a long backlog, so the
            # delivered prefix does not stay resident for the rest of the run.
            var u = List[String]()
            var b = List[String]()
            for i in range(self.head, len(self.urls)):
                u.append(self.urls[i])
                b.append(self.bodies[i])
            self.urls = u^
            self.bodies = b^
            self.head = 0
        self._publish_held()

    # ── the spool ─────────────────────────────────────────────────────────

    def _spool_text(mut self, text: String, count: Int):
        if append_spool(self.spool_path, text):
            _ = self.stats.fetch_add(STAT_SPOOLED, Int64(count))
        else:
            _ = self.stats.fetch_add(STAT_ABANDONED, Int64(count))

    def _spool_head(mut self):
        self._spool_text(
            spool_record(self.urls[self.head], self.bodies[self.head]), 1
        )
        self._pop_head()

    def _spool_all(mut self):
        var n = self._held()
        if n == 0:
            return
        var text = String("")
        for i in range(self.head, len(self.urls)):
            text += spool_record(self.urls[i], self.bodies[i])
        self._spool_text(text, n)
        self.urls.clear()
        self.bodies.clear()
        self.head = 0
        self.held_bytes = 0
        self.head_status_failures = 0
        self._publish_held()

    # ── the transport state ───────────────────────────────────────────────

    def _went_down(mut self):
        """Schedule the next attempt. The first failure of an outage waits
        `retry_min`; every further one doubles it, up to `retry_max`."""
        if not self.down:
            self.down = True
            self.backoff_ns = self.retry_min_ns
            _ = self.stats.fetch_add(STAT_OUTAGES, Int64(1))
            self.stats.release_store(STAT_DOWN, Int64(1))
        else:
            self.backoff_ns = min(self.backoff_ns * 2, self.retry_max_ns)
        self.retry_at_ns = Int64(perf_counter_ns()) + self.backoff_ns

    def _came_up(mut self):
        if self.down:
            self.down = False
            self.backoff_ns = self.retry_min_ns
            self.stats.release_store(STAT_DOWN, Int64(0))

    def _send_head(mut self, ctl: WorkerCtl):
        # ⚠ THE HEARTBEAT'S STAMP MOVES ON EVERY REAL SEND. A run that is
        # logging has already proved it is alive; a ping on top of that is a
        # POST that says nothing new. This one line is the difference between
        # a stamp and a free-running timer.
        self.last_send_ns = Int64(perf_counter_ns())
        var status: Int
        try:
            var r = self.client.value().post_json(
                self.urls[self.head], self.bodies[self.head]
            )
            status = r.status
        except:
            status = -1
        self.stats.release_store(STAT_LAST_STATUS, Int64(status))

        if status >= 200 and status < 300:
            _ = self.stats.fetch_add(STAT_SENT, Int64(1))
            self._pop_head()
            self._came_up()
            return

        _ = self.stats.fetch_add(STAT_FAILED, Int64(1))
        var stopping = ctl.should_stop()
        if status == -1:
            if stopping:
                # ⚠ THE CLOSE-TIME LATCH. Past here every attempt costs a
                # client timeout, and nothing bounds `pthread_join` but us.
                self.dead = True
                self.stats.release_store(STAT_DEAD, Int64(1))
                self._spool_all()
            else:
                self._went_down()
            return

        # The server answered. Retry what it may accept later; spool the rest
        # so one poison payload cannot hold every payload behind it.
        if retryable_status(status) and not stopping:
            self.head_status_failures += 1
            if self.head_status_failures < self.max_status_retries:
                self._went_down()
                return
        _ = self.stats.fetch_add(STAT_REJECTED, Int64(1))
        self._spool_head()
        if not retryable_status(status):
            self._came_up()  # a 4xx is a server that is there

    def _maybe_ping(mut self, ctl: WorkerCtl) -> Int:
        """The §7b heartbeat, sent from the idle branch of the poll loop.

        ⚠⚠ IT IS POSTED DIRECTLY, NOT PUSHED ONTO THE RING. `SharedRing` is
        SPSC and this thread is its CONSUMER; a producer here would be a second
        writer against a queue whose whole correctness argument is that there
        is one. So the ping goes straight out through the client this thread
        already owns — which is also why it can only happen while the ring and
        the FIFO are empty, and therefore can never reorder ahead of a metric
        batch.

        ⚠ IT IS A STAMP COMPARISON, NOT A TIMEOUT. `worker.mojo`'s loop is a
        1 ms poll with nothing to parameterise, so there is no blocking wait to
        shorten. This costs one `perf_counter_ns` per idle lap and adds no
        argument to `BackgroundThread`, which four other things depend on.

        ⚠ WHILE DOWN, THE BACKOFF — NOT THE INTERVAL — SETS THE PACE. An idle
        run in an outage probes at `retry_at`, so it notices the dashboard's
        return even with nothing to send, and a black-holed host costs one
        client timeout per backoff step rather than one per ping interval.

        Returns `POLL_IDLE` when it sends nothing, which is the ring's true
        state and what lets a stopping worker conclude it has drained.
        """
        if self.ping_url.byte_length() == 0:
            return POLL_IDLE
        if self.dead or not self.client:
            return POLL_IDLE
        # A run that is shutting down has nothing to prove about being alive,
        # and `close()` has already queued the finish that says so properly.
        if ctl.should_stop():
            return POLL_IDLE
        var now = Int64(perf_counter_ns())
        if self.down:
            if now < self.retry_at_ns:
                return POLL_IDLE
        elif now - self.last_send_ns < self.ping_interval_ns:
            return POLL_IDLE

        self.last_send_ns = now
        var status: Int
        try:
            var r = self.client.value().post_json(self.ping_url, self.ping_body)
            status = r.status
        except:
            status = -1
        self.stats.release_store(STAT_LAST_STATUS, Int64(status))
        if status >= 200 and status < 300:
            _ = self.stats.fetch_add(STAT_PINGS, Int64(1))
            self._came_up()
        else:
            _ = self.stats.fetch_add(STAT_FAILED, Int64(1))
            if status == -1 or retryable_status(status):
                self._went_down()
        # ⚠ POLL_DID_WORK, NOT POLL_IDLE: the loop must take another lap to
        # find the ring genuinely empty. Reporting idle here would be true of
        # the ring but would also skip the 1 ms sleep, spinning a core.
        return POLL_DID_WORK


def retryable_status(status: Int) -> Bool:
    """A status the same payload may get a 2xx for later: the server is
    overloaded, restarting, or rate-limiting. Everything else non-2xx is about
    the payload or the credentials, and repeating it cannot help."""
    return status == 408 or status == 425 or status == 429 or status >= 500


# ── the spool file ────────────────────────────────────────────────────────


def spool_record(url: String, body: String) -> String:
    """One POST as a spool record: `<url>\\t<body byte length>\\t<body>\\n`.

    ⚠ LENGTH-PREFIXED, NOT LINE-DELIMITED. `RemoteLogger`'s JSON never holds a
    raw newline, but this sink carries any body, and a record format that
    breaks on one would lose the payload it was written to keep."""
    return url + "\t" + String(body.byte_length()) + "\t" + body + "\n"


def append_spool(path: String, text: String) -> Bool:
    """Append records to the spool. False if there is no spool or the write
    failed — the caller counts that as a loss. Never raises."""
    if path.byte_length() == 0:
        return False
    try:
        makedirs(parent_dir(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(text)
        return True
    except:
        return False


def _bytes_to_string(ref data: List[UInt8], start: Int, end: Int) -> String:
    var b = List[UInt8]()
    b.reserve(end - start + 1)
    for i in range(start, end):
        b.append(data[i])
    b.append(0)
    return String(unsafe_from_utf8_ptr=b.unsafe_ptr())


def read_spool(
    path: String, mut urls: List[String], mut bodies: List[String]
) raises:
    """Append every record in a spool file to `urls`/`bodies`, in the order it
    was written.

    Raises:
        Error: the file is unreadable or a record is malformed.
    """
    var data = read_file_bytes(path)
    var i = 0
    var n = len(data)
    while i < n:
        var t1 = i
        while t1 < n and Int(data[t1]) != 0x09:
            t1 += 1
        var t2 = t1 + 1
        while t2 < n and Int(data[t2]) != 0x09:
            t2 += 1
        if t2 >= n:
            raise Error(
                "spool " + path + ": truncated record at byte " + String(i)
            )
        var blen = 0
        for k in range(t1 + 1, t2):
            var c = Int(data[k])
            if c < 0x30 or c > 0x39:
                raise Error(
                    "spool " + path + ": bad length at byte " + String(t1 + 1)
                )
            blen = blen * 10 + (c - 0x30)
        var b0 = t2 + 1
        if b0 + blen >= n or Int(data[b0 + blen]) != 0x0A:
            raise Error(
                "spool " + path + ": record at byte " + String(i)
                + " does not end where its length says"
            )
        urls.append(_bytes_to_string(data, i, t1))
        bodies.append(_bytes_to_string(data, b0, b0 + blen))
        i = b0 + blen + 1


def replay_spool(
    path: String, api_key: String = String(""), timeout_ms: Int = 10000
) raises -> Tuple[Int, Int]:
    """POST a spool's records in order. Returns `(delivered, kept)`.

    A record that is not delivered is KEPT: the file is rewritten with exactly
    those, or removed when none are left, so a replay can be run again until
    it reports zero kept. The first transport failure stops the attempts — a
    dashboard that is still down does not need to refuse every record to say
    so — and everything from there on is kept too.

    Raises:
        Error: the spool is unreadable or malformed, or the HTTP shim is
            missing. The file is untouched in every one of those cases.
    """
    var urls = List[String]()
    var bodies = List[String]()
    read_spool(path, urls, bodies)
    var c = HttpClient(timeout_ms, timeout_ms)
    if api_key.byte_length() > 0:
        c.bearer(api_key)
    var kept = String("")
    var delivered = 0
    var nkept = 0
    var reachable = True
    for i in range(len(urls)):
        if reachable:
            try:
                var r = c.post_json(urls[i], bodies[i])
                if r.ok():
                    delivered += 1
                    continue
            except:
                reachable = False
        kept += spool_record(urls[i], bodies[i])
        nkept += 1
    if nkept == 0:
        remove_file(path)
    else:
        write_text_atomic(path, kept)
    return (delivered, nkept)


def frame_into(
    ring: SharedRing, head: String, tail: String, count_full: Bool = True
) -> Bool:
    """Write `[Int32 head_len][head][tail]` into a free slot. False if dropped.

    `count_full=False` is for a caller that keeps the payload itself when the
    ring is full (`HttpPostSink.post`): not a drop, so not counted as one.

    ⚠ THE FRAMING IS GENERIC, THE NAMES WERE NOT. This POSTs a `(url, body)`
    pair and `artifact_sink.mojo` sends a `(kind, path)` pair through the same
    two functions — so the parameters say `head`/`tail` rather than pretending
    there is only one caller. Two sinks framing bytes two ways would be the
    same rule written twice.

    Module-level so the gate can exercise the real framing rather than a
    re-implementation of it — `unframe` is its inverse and the two are tested
    as a pair in `tests/io/test_http_sink.mojo`.
    """
    var n = _frame_len(head, tail)
    if n > ring.slot_bytes():
        ring.drop_oversize()
        return False
    var claim = ring.begin_push()
    if not claim.ok():
        if count_full:
            ring.drop_full()
        return False
    var dst = claim.data()
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(dst))[] = Int32(
        head.byte_length()
    )
    if head.byte_length() > 0:
        unsafe_memcpy(
            dest=dst.unsafe_offset(4),
            src=head.as_bytes().unsafe_ptr(),
            count=head.byte_length(),
        )
    if tail.byte_length() > 0:
        unsafe_memcpy(
            dest=dst.unsafe_offset(4 + head.byte_length()),
            src=tail.as_bytes().unsafe_ptr(),
            count=tail.byte_length(),
        )
    ring.end_push(n)
    return True


def unframe(
    p: Pointer[UInt8, MutUntrackedOrigin], n: Int
) raises -> Tuple[String, String]:
    """`[Int32 head_len][head][tail]` back into two strings."""
    if n < 4:
        raise Error("http_sink: frame shorter than its header")
    var head_len = Int(
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(p))[]
    )
    if head_len < 0 or 4 + head_len > n:
        raise Error("http_sink: frame head_len out of range")
    var head_b = List[UInt8]()
    for i in range(head_len):
        head_b.append(p[unsafe_offset = 4 + i])
    head_b.append(0)
    var tail_b = List[UInt8]()
    for i in range(4 + head_len, n):
        tail_b.append(p[unsafe_offset=i])
    tail_b.append(0)
    return (
        String(unsafe_from_utf8_ptr=head_b.unsafe_ptr()),
        String(unsafe_from_utf8_ptr=tail_b.unsafe_ptr()),
    )


# ── the sink ──────────────────────────────────────────────────────────────


struct _Overflow(Movable):
    """Payloads `post()` could not put in the ring, oldest first.

    Only non-empty while the worker is blocked inside one POST (up to a client
    timeout) and the producer outruns 16 slots; the worker empties the ring
    every lap otherwise. Owner-thread only."""

    var urls: List[String]
    var bodies: List[String]
    var head: Int

    def __init__(out self):
        self.urls = List[String]()
        self.bodies = List[String]()
        self.head = 0

    def __init__(out self, *, deinit move: Self):
        self.urls = move.urls^
        self.bodies = move.bodies^
        self.head = move.head

    @always_inline
    def count(self) -> Int:
        return len(self.urls) - self.head

    def pop(mut self):
        self.head += 1
        if self.head == len(self.urls):
            self.urls.clear()
            self.bodies.clear()
            self.head = 0


struct HttpPostSink(ImplicitlyCopyable, Movable):
    """A queue of POSTs and the one thread that drains it.

    ⚠ COPIES SHARE ONE THREAD AND ONE QUEUE, which is the right meaning: two
    copies of a logger are one run and should be one connection. It also means
    `close()` on either copy stops both — the same asymmetry the synchronous
    version had with its shared client.
    """

    var _ring: SharedRing
    var _stats: SharedBlock
    var _bg: ArcPointer[BackgroundThread[HttpPostWorker]]
    var _closed: ArcPointer[Bool]
    """Refcounted so `close()` through one copy is visible to the others."""
    var _overflow: ArcPointer[_Overflow]
    var _spool_path: String

    def __init__(
        out self,
        api_key: String = String(""),
        timeout_ms: Int = 5000,
        capacity: Int = DEFAULT_CAPACITY,
        slot_bytes: Int = DEFAULT_SLOT_BYTES,
        ping_url: String = String(""),
        ping_body: String = String("{}"),
        ping_interval_ms: Int = DEFAULT_PING_INTERVAL_MS,
        spool_path: String = String(""),
        retry_min_ms: Int = DEFAULT_RETRY_MIN_MS,
        retry_max_ms: Int = DEFAULT_RETRY_MAX_MS,
        max_status_retries: Int = DEFAULT_MAX_STATUS_RETRIES,
        hold_bytes: Int = DEFAULT_HOLD_BYTES,
    ) raises:
        """Allocate the queue and START THE THREAD.

        ⚠ CONSTRUCTING THIS SPAWNS A THREAD. Build it lazily, on the first
        payload — a logger with no server configured must stay inert.

        ⚠ WITHOUT `spool_path` A PAYLOAD THE DASHBOARD NEVER TAKES IS LOST, and
        counted in `abandoned()`. Retries still happen; only the last resort
        is missing. `RemoteLogger` always passes one.

        Raises:
            Error: the ring or the thread could not be created.
        """
        self._ring = SharedRing(capacity, slot_bytes)
        self._stats = SharedBlock(STAT_CELLS)
        self._stats.release_store(STAT_LAST_STATUS, Int64(0))
        self._bg = ArcPointer(
            BackgroundThread(
                HttpPostWorker(
                    self._ring,
                    self._stats,
                    api_key,
                    timeout_ms,
                    ping_url,
                    ping_body,
                    ping_interval_ms,
                    spool_path,
                    retry_min_ms,
                    retry_max_ms,
                    max_status_retries,
                    hold_bytes,
                )
            )
        )
        self._closed = ArcPointer(False)
        self._overflow = ArcPointer(_Overflow())
        self._spool_path = spool_path

    def post(mut self, url: String, body: String) -> Bool:
        """Queue a POST. False only if the payload was LOST.

        Never blocks and never raises: a dead dashboard must not be able to
        stop a training run. The cost is a `memcpy` and a release-store —
        measured at 0.003 ms for eleven POSTs whose synchronous equivalent
        cost 629.7 ms.

        ⚠ A FULL RING IS NOT A DROP ANY MORE. The payload waits in the
        owner-side overflow and is re-offered, in order, on the next `post` —
        and by `close()`. Only an oversize payload (a caller bug, counted in
        `oversize()`) or a `post` after `close()` with no spool loses one.
        """
        if self._closed[]:
            # No worker to hand it to. Straight to the spool, which is safe
            # from this thread now: the worker has joined.
            if append_spool(self._spool_path, spool_record(url, body)):
                _ = self._stats.fetch_add(STAT_SPOOLED, Int64(1))
                return True
            _ = self._stats.fetch_add(STAT_ABANDONED, Int64(1))
            return False
        if _frame_len(url, body) > self._ring.slot_bytes():
            self._ring.drop_oversize()
            return False
        # ⚠ THE OVERFLOW GOES FIRST. A payload that found the ring free while
        # older ones were still waiting outside it would overtake them.
        self.pump()
        if self._overflow[].count() == 0 and frame_into(
            self._ring, url, body, count_full=False
        ):
            return True
        self._overflow[].urls.append(url)
        self._overflow[].bodies.append(body)
        return True

    def pump(mut self):
        """Move the overflow into the ring, oldest first, until it is full.

        `post()` and `close()` call this. ⚠ A CALLER THAT GOES QUIET SHOULD
        TOO — the overflow is owner-thread state, so the worker cannot fetch
        it, and whatever sits there waits for the owner's next call.
        `RemoteLogger.flush` pumps even when it has nothing new to send.
        Costs one load when the overflow is empty."""
        while self._overflow[].count() > 0:
            var h = self._overflow[].head
            if not frame_into(
                self._ring,
                self._overflow[].urls[h],
                self._overflow[].bodies[h],
                count_full=False,
            ):
                return
            self._overflow[].pop()

    def close(mut self, drain_ms: Int = 2000) raises:
        """Stop accepting, drain what is queued, join. Idempotent.

        ⚠ WHAT DOES NOT DRAIN IS SPOOLED, IN ORDER. The worker spools its FIFO
        before it exits; the owner's overflow — every payload of which is newer
        — is appended after the join, when there is no second writer left.

        Raises:
            Error: the join failed.
        """
        if self._closed[]:
            return
        self._closed[] = True
        var deadline = perf_counter_ns() + max(drain_ms, 1) * 1_000_000
        # The worker empties the ring every lap, so this waits only while it is
        # blocked inside one POST.
        while self._overflow[].count() > 0:
            self.pump()
            if self._overflow[].count() == 0:
                break
            if self.dead() or perf_counter_ns() > deadline:
                break
            _ = sleep_us(1000)
        var left_ms = max((deadline - perf_counter_ns()) // 1_000_000, 1)
        self._bg[].stop(left_ms)

        var n = self._overflow[].count()
        if n > 0:
            var text = String("")
            for i in range(self._overflow[].head, len(self._overflow[].urls)):
                text += spool_record(
                    self._overflow[].urls[i], self._overflow[].bodies[i]
                )
            if append_spool(self._spool_path, text):
                _ = self._stats.fetch_add(STAT_SPOOLED, Int64(n))
            else:
                _ = self._stats.fetch_add(STAT_ABANDONED, Int64(n))
            self._overflow[] = _Overflow()

    # ── observation, all snapshots of live counters ───────────────────────

    @always_inline
    def sent(self) -> Int:
        """Payloads delivered."""
        return Int(self._stats.acquire_load(STAT_SENT))

    @always_inline
    def failed(self) -> Int:
        """Failed ATTEMPTS, heartbeats included — not failed payloads. A
        payload retried through an outage counts once per try."""
        return Int(self._stats.acquire_load(STAT_FAILED))

    @always_inline
    def abandoned(self) -> Int:
        """Payloads LOST: undeliverable with no spool to go to. ⚠ REPORT THIS.
        With a working `spool_path` it stays 0."""
        return Int(self._stats.acquire_load(STAT_ABANDONED))

    @always_inline
    def dropped(self) -> Int:
        """Refused at `post()` — now only a payload larger than a slot, since a
        full ring parks the payload instead. Lost; a caller bug."""
        return self._ring.dropped()

    @always_inline
    def oversize(self) -> Int:
        """Subset of `dropped()` larger than `slot_bytes` — a caller bug, not
        back-pressure."""
        return self._ring.oversize()

    @always_inline
    def queued(self) -> Int:
        """Payloads waiting in the ring right now."""
        return self._ring.depth()

    @always_inline
    def held(self) -> Int:
        """Payloads accepted and not yet delivered or spooled: the worker's
        FIFO, the ring, and the owner-side overflow."""
        return (
            Int(self._stats.acquire_load(STAT_HELD))
            + self._ring.depth()
            + self._overflow[].count()
        )

    @always_inline
    def held_bytes(self) -> Int:
        """Size of the worker's FIFO — what an outage is costing in memory."""
        return Int(self._stats.acquire_load(STAT_HELD_BYTES))

    @always_inline
    def rejected(self) -> Int:
        """Payloads the server refused for good. Spooled, not lost."""
        return Int(self._stats.acquire_load(STAT_REJECTED))

    @always_inline
    def spooled(self) -> Int:
        """Payloads written to `spool_path()` — replay them with
        `replay_spool`."""
        return Int(self._stats.acquire_load(STAT_SPOOLED))

    @always_inline
    def spool_path(self) -> String:
        return self._spool_path

    @always_inline
    def down(self) -> Bool:
        """Whether the transport is failing right now and payloads are being
        held for retry."""
        return self._stats.acquire_load(STAT_DOWN) != 0

    @always_inline
    def outages(self) -> Int:
        """Times the transport went down during the run."""
        return Int(self._stats.acquire_load(STAT_OUTAGES))

    @always_inline
    def last_status(self) -> Int:
        """Status of the last completed POST; -1 for a transport error."""
        return Int(self._stats.acquire_load(STAT_LAST_STATUS))

    @always_inline
    def pings(self) -> Int:
        """Heartbeats the worker sent because nothing else was queued."""
        return Int(self._stats.acquire_load(STAT_PINGS))

    @always_inline
    def dead(self) -> Bool:
        """Whether the worker has given up on the network for good — no shim,
        no client, or a transport failure after `close()`. NOT set by an
        outage during the run; that is `down()`."""
        return self._stats.acquire_load(STAT_DEAD) != 0

    @always_inline
    def shim_missing(self) -> Bool:
        """Whether `libnra_http` was absent when the worker started. Callers
        report this once — `pixi run build-http` is the fix."""
        return self._stats.acquire_load(STAT_NO_SHIM) != 0

    @always_inline
    def closed(self) -> Bool:
        return self._closed[]
