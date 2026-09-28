"""HttpPostSink — framing, accounting, and non-blocking under failure.

Run: pixi run mojo run -I . tests/io/test_http_sink.mojo

⚠ THIS GATE IS DELIBERATELY OFFLINE. It never talks to a live dashboard: a
self-contained test cannot depend on a server, and starting one would drag a
Python process back into a path we just made Python-free. What it CAN gate is
everything that is not the network — the wire framing, the drop accounting, and
the property that actually matters operationally: **a dashboard that is down
must cost the training thread nothing and must not hold the process open.**

Delivery against a real dashboard is validated by running a real training job,
which is what step 2 of `docs/CONCURRENCY_BACKBONE.md` is for. Measured there:
500 `log_scalar` calls across 10 flushes cost the caller **1.7 ms**, and the
same 11 POSTs cost **629.7 ms** synchronously versus **0.003 ms** queued.

`127.0.0.1:9` is the discard port. Nothing listens, so the connection is
refused promptly and deterministically — no timeout, no waiting.
"""

from std.memory import Pointer
from std.time import perf_counter_ns

from noeira.core.concurrent.ring import SharedRing
from noeira.core.logger import RemoteLogger
from noeira.io.fileio import remove_file
from noeira.io.http_sink import (
    DEFAULT_SLOT_BYTES,
    HttpPostSink,
    frame_into,
    read_spool,
    unframe,
)


comptime DEAD_URL = "http://127.0.0.1:9/ingest"


def test_frame_roundtrip() raises:
    """`frame_into` and `unframe` are inverses, including at the edges."""
    var ring = SharedRing(capacity=8, slot_bytes=4096)
    var urls = List[String]()
    var bodies = List[String]()
    urls.append(String("http://h/a"))
    bodies.append(String('{"k":1}'))
    urls.append(String(""))  # empty url
    bodies.append(String("body-only"))
    urls.append(String("http://h/b"))
    bodies.append(String(""))  # empty body
    urls.append(String("http://h/") + String("x") * 200)
    bodies.append(String("{") + String("y") * 1000 + String("}"))

    for i in range(len(urls)):
        if not frame_into(ring, urls[i], bodies[i]):
            raise Error("frame_into refused case " + String(i))

    var compared = 0
    var differing = 0
    for i in range(len(urls)):
        var c = ring.begin_pop()
        if not c.ok():
            raise Error("ring emptied early at case " + String(i))
        var pair = unframe(c.data(), c.len)
        compared += 1
        if pair[0] != urls[i] or pair[1] != bodies[i]:
            differing += 1
            print(
                "    case", i, "url", pair[0] == urls[i], "body",
                pair[1] == bodies[i],
            )
        ring.end_pop()
    if differing != 0 or compared != len(urls):
        raise Error(
            "framing: " + String(differing) + " of " + String(compared)
            + " roundtrips differed"
        )
    print(
        "  framing:", compared, "of", len(urls),
        "url/body pairs round-tripped,", differing, "differing",
    )


def test_oversize_is_refused_not_truncated() raises:
    """A payload past `slot_bytes` is dropped and counted, never cut short —
    half a JSON body would be worse than none."""
    var ring = SharedRing(capacity=2, slot_bytes=256)
    var body = String("z") * 400
    if frame_into(ring, String("http://h/x"), body):
        raise Error("an oversize frame was accepted")
    if ring.oversize() != 1 or ring.dropped() != 1:
        raise Error(
            "oversize accounting: oversize=" + String(ring.oversize())
            + " dropped=" + String(ring.dropped())
        )
    if ring.depth() != 0:
        raise Error("an oversize frame consumed a slot")
    print(
        "  oversize:", 4 + 10 + 400, "byte frame into a 256 byte slot ->"
        " refused, oversize=1 of dropped=1, depth 0",
    )


def test_full_queue_drops_and_counts() raises:
    """Deterministic: no sink, so nothing drains. Fill it, then overflow."""
    var ring = SharedRing(capacity=4, slot_bytes=1024)
    var accepted = 0
    for i in range(9):
        if frame_into(ring, String("http://h/x"), String("b") + String(i)):
            accepted += 1
    if accepted != 4 or ring.dropped() != 5:
        raise Error(
            "queue-full accounting: " + String(accepted) + " accepted, "
            + String(ring.dropped()) + " dropped, expected 4 and 5"
        )
    if accepted + ring.dropped() != 9:
        raise Error("accepted + dropped did not account for every post")
    print(
        "  queue full:", accepted, "accepted +", ring.dropped(), "dropped =",
        accepted + ring.dropped(), "posts",
    )


def _spool_count(path: String) raises -> Int:
    var u = List[String]()
    var b = List[String]()
    read_spool(path, u, b)
    return len(u)


def _rm(path: String):
    try:
        remove_file(path)
    except:
        pass


def test_dead_dashboard_costs_the_caller_nothing() raises:
    """The operational property. Nothing is listening on the discard port.

    Asserts what a synchronous POST could not give: the caller is not blocked,
    `close()` returns promptly instead of paying a timeout per queued item —
    the close-time `dead` latch — and every payload that did not go out is in
    the spool rather than on the floor.
    """
    comptime N = 64
    var spool = String("/tmp/noeira_http_sink_gate_dead.spool")
    _rm(spool)
    var sink = HttpPostSink(
        timeout_ms=2000, capacity=N, slot_bytes=4096, spool_path=spool
    )
    var t0 = perf_counter_ns()
    var queued = 0
    for i in range(N):
        if sink.post(String(DEAD_URL), String('{"i":') + String(i) + "}"):
            queued += 1
    var post_ms = Float64(perf_counter_ns() - t0) / 1e6

    var t1 = perf_counter_ns()
    sink.close(drain_ms=3000)
    var close_ms = Float64(perf_counter_ns() - t1) / 1e6

    var handled = sink.sent() + sink.spooled() + sink.abandoned()
    if queued != N:
        raise Error(
            "queued " + String(queued) + " of " + String(N)
            + " into a ring that had room for all of them"
        )
    if post_ms > 50.0:
        raise Error(
            "queueing " + String(N) + " POSTs blocked the caller for "
            + String(post_ms) + " ms — it is not going through the ring"
        )
    if sink.sent() != 0:
        raise Error(
            String(sink.sent()) + " POSTs reported success against a port"
            " where nothing is listening"
        )
    if not sink.dead():
        raise Error(
            "the worker did not latch dead at close after a refused"
            " connection; a drain will pay one client timeout PER payload"
        )
    # ⚠ ONE OR TWO, NOT ONE: the worker may try the head once while the run
    # is live (and back off), then once more when `close()` asks it to stop.
    # What the latch forbids is one attempt PER payload — 64 here.
    if sink.failed() < 1 or sink.failed() > 2:
        raise Error(
            "the worker tried " + String(sink.failed()) + " POSTs; the latch"
            " should stop it after the first failure at close"
        )
    if handled != N or sink.abandoned() != 0:
        raise Error(
            "accounting: " + String(sink.sent()) + " sent + "
            + String(sink.spooled()) + " spooled + " + String(sink.abandoned())
            + " lost = " + String(handled) + ", expected " + String(N)
            + " with 0 lost"
        )
    if _spool_count(spool) != N:
        raise Error(
            "the spool holds " + String(_spool_count(spool)) + " of "
            + String(N) + " payloads"
        )
    if close_ms > 2500.0:
        raise Error(
            "close() took " + String(close_ms) + " ms against a dead port"
        )
    _rm(spool)
    print(
        "  dead dashboard:", queued, "queued in", post_ms, "ms; close() in",
        close_ms, "ms;", sink.failed(), "tried,", sink.spooled(),
        "spooled, 0 lost ( dead latched at close )",
    )


def test_full_ring_parks_rather_than_drops() raises:
    """⚠⚠ A FULL RING USED TO BE A DROP. Two slots, 50 posts faster than any
    worker drains them: every one must be accepted, none counted dropped, and
    all 50 must come out the other end — here, in order, in the spool."""
    comptime N = 50
    var spool = String("/tmp/noeira_http_sink_gate_full.spool")
    _rm(spool)
    var sink = HttpPostSink(
        timeout_ms=500, capacity=2, slot_bytes=1024, spool_path=spool
    )
    var accepted = 0
    for i in range(N):
        if sink.post(String(DEAD_URL), String('{"i":') + String(i) + "}"):
            accepted += 1
    sink.close(drain_ms=1000)
    if accepted != N or sink.dropped() != 0:
        raise Error(
            "full ring: " + String(accepted) + " accepted, "
            + String(sink.dropped()) + " dropped; want " + String(N) + ", 0"
        )
    var u = List[String]()
    var b = List[String]()
    read_spool(spool, u, b)
    if len(u) != N or sink.spooled() != N or sink.abandoned() != 0:
        raise Error(
            "full ring: spool holds " + String(len(u)) + ", spooled="
            + String(sink.spooled()) + " lost=" + String(sink.abandoned())
        )
    for i in range(N):
        if b[i] != String('{"i":') + String(i) + "}":
            raise Error("full ring: spool record " + String(i) + " is " + b[i])
    _rm(spool)
    print("  full ring: 50 posts through 2 slots, 0 dropped, 50 spooled in order")


def test_close_is_idempotent() raises:
    var sink = HttpPostSink(timeout_ms=1000, capacity=4, slot_bytes=1024)
    sink.close(drain_ms=500)
    if not sink.closed():
        raise Error("close() did not record itself")
    sink.close(drain_ms=500)
    sink.close()
    print("  close: idempotent across 3 calls")


def test_inert_logger_starts_no_sink() raises:
    """A `RemoteLogger` with no server must not spawn a thread or need the
    shim — several drivers construct one unconditionally."""
    var lg = RemoteLogger(server_url=String(""))
    for step in range(500):
        lg.log_scalar(String("loss"), Float64(step), step)
    lg.flush()
    lg.close()
    if lg.is_active():
        raise Error("a logger with no server_url reported itself active")
    if lg.total_logged() != 0:
        raise Error(
            "an inert logger buffered " + String(lg.total_logged())
            + " metrics"
        )
    if lg.sink_report().byte_length() != 0:
        raise Error("an inert logger built a sink: " + lg.sink_report())
    print("  inert logger: 500 log_scalar calls, no sink, no thread")


def test_logger_does_not_block_on_a_dead_dashboard() raises:
    """The end-to-end shape, minus the network: a run whose dashboard is down
    must pay effectively nothing and must still close."""
    comptime STEPS = 2000
    comptime BUFFER = 100
    var spool = String("/tmp/noeira_http_sink_gate_logger.spool")
    _rm(spool)
    var lg = RemoteLogger(
        server_url=String("http://127.0.0.1:9"),
        run_name=String("offline-gate"),
        buffer_size=BUFFER,
        spool_path=spool,
    )
    var t0 = perf_counter_ns()
    for step in range(STEPS):
        lg.log_scalar(String("loss"), 1.0 / Float64(step + 1), step)
    var log_ms = Float64(perf_counter_ns() - t0) / 1e6

    var t1 = perf_counter_ns()
    lg.close()
    var close_ms = Float64(perf_counter_ns() - t1) / 1e6

    var flushes = STEPS // BUFFER
    if lg.total_logged() != STEPS:
        raise Error(
            "logged " + String(lg.total_logged()) + " of " + String(STEPS)
        )
    if log_ms > 500.0:
        raise Error(
            String(STEPS) + " log_scalar calls (" + String(flushes)
            + " flushes) blocked the caller for " + String(log_ms)
            + " ms against a dead dashboard"
        )
    if close_ms > 3500.0:
        raise Error("close() took " + String(close_ms) + " ms")
    if lg.sink_report().byte_length() == 0:
        raise Error(
            "no delivery accounting was recorded — the sink was never used,"
            " so this gate is VACUOUS"
        )
    # /runs + one /ingest per flush + /finish, every one of them in the spool.
    if lg.posts_attempted() != flushes + 2 or _spool_count(spool) != flushes + 2:
        raise Error(
            "offline run: " + String(lg.posts_attempted()) + " accounted, "
            + String(_spool_count(spool)) + " spooled; want "
            + String(flushes + 2) + " of each"
        )
    _rm(spool)
    print(
        "  offline run:", STEPS, "log_scalar calls /", flushes,
        "flushes in", log_ms, "ms; close() in", close_ms, "ms",
    )
    print("               ", lg.sink_report().strip())


def main() raises:
    print("=" * 62)
    print("HttpPostSink — framing, drops, and a dashboard that is down")
    print("=" * 62)
    test_frame_roundtrip()
    test_oversize_is_refused_not_truncated()
    test_full_queue_drops_and_counts()
    test_dead_dashboard_costs_the_caller_nothing()
    test_full_ring_parks_rather_than_drops()
    test_close_is_idempotent()
    test_inert_logger_starts_no_sink()
    test_logger_does_not_block_on_a_dead_dashboard()
    print("[PASS] http_sink")
