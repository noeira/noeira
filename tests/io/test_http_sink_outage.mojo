# +--------------------------------------------------------------------------+ #
# | An outage costs the dashboard time, never metrics
# +--------------------------------------------------------------------------+ #
"""Gate the hold / retry / spool policy in `noeira/io/http_sink.mojo`.

    pixi run build-http                                    # ONCE
    pixi run mojo run -I . tests/io/test_http_sink_outage.mojo

⚠⚠ THE DEFECT THIS EXISTS FOR: the first failed POST used to latch the
transport dead, and every payload after it was discarded — "metrics will be
dropped for the rest of this run", printed in the first minute of a run that
went on for hours. So the decisive check is the first one: a monitor that is
DOWN when the run starts and comes back later receives EVERYTHING, in order.

Hermetic: `tools/io/mock_monitor_server.py` on loopback, started on a port
chosen in advance so the sink can be pointed at it while nothing listens there
yet — a refused connection, which is the transport error that used to latch.

## What each check is for

1. **Down at start, back later.** Everything is delivered, in order, nothing
   spooled, nothing lost — and the ring (4 slots) was far smaller than the
   backlog, so the FIFO and the owner-side overflow both had to hold.
2. **An overloaded server (503).** Retried in place: delivered once each, in
   order, and nothing overtakes the payload being retried.
3. **A poison payload.** A 500-forever payload is spooled after
   `max_status_retries` and the queue moves on; a 400 is spooled at once.
4. **Still down at close.** `close()` is bounded, everything is spooled, and
   `replay_spool` delivers the spool once a monitor is there — then removes it.
"""

from std.os.path import exists
from std.time import perf_counter_ns, sleep

from noeira.io.fileio import remove_file
from noeira.io.http import http_shim_available
from noeira.io.http_sink import HttpPostSink, read_spool, replay_spool
from noeira.io.proc import run_capture


comptime TMP = "/tmp/noeira_outage_gate"


def _free_port() raises -> String:
    return String(
        run_capture(
            "python3 -c \"import socket;s=socket.socket();"
            "s.bind(('127.0.0.1',0));print(s.getsockname()[1])\""
        ).strip()
    )


def _rm(p: String):
    try:
        remove_file(p)
    except:
        pass


def _start_server(tag: String, port: String) raises -> String:
    var port_file = String(TMP) + "_" + tag + "_port"
    var log_file = String(TMP) + "_" + tag + "_log"
    _rm(port_file)
    _rm(log_file)
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py " + port_file + " "
        + log_file + " 120 " + port + " > " + String(TMP) + "_" + tag
        + "_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(port_file):
            var f = open(port_file, "r")
            var p = String(f.read().strip())
            f.close()
            if p.byte_length() > 0:
                return "http://127.0.0.1:" + p
        sleep(0.1)
    raise Error("the mock monitor '" + tag + "' never wrote its port")


def _stop_server(base: String) raises:
    _ = run_capture(
        "curl -s -X POST " + base + "/__shutdown > /dev/null 2>&1 || true"
    )


def _log(tag: String) raises -> List[String]:
    """`<path> <body>` for every request the fixture saw, in arrival order."""
    var out = List[String]()
    var log_file = String(TMP) + "_" + tag + "_log"
    if not exists(log_file):
        return out^
    var f = open(log_file, "r")
    var text = String(f.read())
    f.close()
    for line in text.split("\n"):
        var parts = String(line).strip().split(" ", maxsplit=3)
        if len(parts) >= 4:
            out.append(String(parts[2]) + " " + String(parts[3]))
    return out^


def _count(tag: String, path: String) raises -> Int:
    var n = 0
    for l in _log(tag):
        if String(l).startswith(path + " "):
            n += 1
    return n


def _wait_sent(mut sink: HttpPostSink, want: Int, seconds: Float64) -> Bool:
    """Wait for `want` deliveries, pumping like a run that keeps flushing.

    ⚠ THE PUMP IS PART OF THE CONTRACT, NOT A TEST CONVENIENCE: the owner-side
    overflow only moves on an owner-thread call. `RemoteLogger.flush` makes
    it; a bare sink's caller has to."""
    var t0 = perf_counter_ns()
    while Float64(perf_counter_ns() - t0) / 1e9 < seconds:
        sink.pump()
        if sink.sent() >= want:
            return True
        sleep(0.02)
    return sink.sent() >= want


def _body(i: Int) -> String:
    return String('{"i":') + String(i) + "}"


def _assert_in_order(tag: String, path: String, n: Int) raises:
    """The DISTINCT bodies seen on `path`, in arrival order, are 0..n-1.
    Repeats of the same body (a retried 503) collapse; an overtake does not."""
    var seen = List[String]()
    for l in _log(tag):
        var s = String(l)
        if not s.startswith(path + " "):
            continue
        var body = String(s[byte = path.byte_length() + 1 :])
        if len(seen) == 0 or seen[len(seen) - 1] != body:
            seen.append(body)
    if len(seen) != n:
        raise Error(
            tag + ": " + String(len(seen)) + " distinct " + path
            + " bodies in a row, want " + String(n)
        )
    for i in range(n):
        if seen[i] != _body(i):
            raise Error(
                tag + ": body #" + String(i) + " on " + path + " was "
                + seen[i] + ", want " + _body(i) + " — out of order"
            )


def check_down_at_start_then_back() raises -> Int:
    comptime N = 30
    var port = _free_port()
    var base = "http://127.0.0.1:" + port
    var spool = String(TMP) + "_a.spool"
    _rm(spool)
    var sink = HttpPostSink(
        timeout_ms=500,
        capacity=4,
        slot_bytes=1024,
        spool_path=spool,
        retry_min_ms=100,
        retry_max_ms=400,
    )
    _ = sink.post(base + "/runs", String('{"run_id":"outage"}'))
    for i in range(N):
        _ = sink.post(base + "/ingest", _body(i))
    sleep(0.9)
    # ⚠ THE PRECONDITION, OR THE REST IS VACUOUS: the monitor really was down,
    # the worker really did fail, and it is holding rather than discarding.
    if sink.sent() != 0 or not sink.down() or sink.failed() < 2:
        raise Error(
            "before the monitor starts: sent=" + String(sink.sent())
            + " down=" + String(sink.down()) + " failed="
            + String(sink.failed()) + " — the outage was not observed"
        )
    if sink.dropped() != 0 or sink.held() != N + 1:
        raise Error(
            "during the outage: dropped=" + String(sink.dropped())
            + " held=" + String(sink.held()) + ", want 0 and "
            + String(N + 1)
        )
    var failed_while_down = sink.failed()

    _ = _start_server(String("a"), port)
    var ok = _wait_sent(sink, N + 1, 10.0)
    sink.close(drain_ms=2000)
    _stop_server(base)
    if not ok:
        raise Error(
            "the monitor came back and only " + String(sink.sent()) + " of "
            + String(N + 1) + " payloads were delivered"
        )
    var paths = _log(String("a"))
    if len(paths) == 0 or not String(paths[0]).startswith("/runs "):
        raise Error("the registration was not the first request delivered")
    _assert_in_order(String("a"), String("/ingest"), N)
    if _count(String("a"), String("/ingest")) != N:
        raise Error(
            "the monitor saw " + String(_count(String("a"), String("/ingest")))
            + " /ingest requests, want exactly " + String(N)
        )
    if sink.spooled() != 0 or sink.abandoned() != 0 or sink.down():
        raise Error(
            "after recovery: spooled=" + String(sink.spooled()) + " lost="
            + String(sink.abandoned()) + " down=" + String(sink.down())
        )
    if sink.outages() != 1:
        raise Error("outages=" + String(sink.outages()) + ", want 1")
    print(
        "  down at start, back 0.9 s later: " + String(N + 1)
        + " of " + String(N + 1) + " delivered in order through a 4-slot"
        " ring, after " + String(failed_while_down)
        + " failed attempts; 0 spooled, 0 lost"
    )
    return 5


def check_overloaded_server_is_retried_in_place(base: String) raises -> Int:
    comptime N = 5
    _ = run_capture(
        "curl -s -X POST -d 3 " + base + "/__fail_next > /dev/null 2>&1"
    )
    var sink = HttpPostSink(
        timeout_ms=2000, spool_path=String(TMP) + "_b.spool", retry_min_ms=30
    )
    for i in range(N):
        _ = sink.post(base + "/ingest", _body(i))
    var ok = _wait_sent(sink, N, 5.0)
    sink.close(drain_ms=1000)
    if not ok:
        raise Error("503 burst: " + String(sink.sent()) + " of 5 delivered")
    var hits = _count(String("live"), String("/ingest"))
    if hits != N + 3:
        raise Error(
            "503 burst: the monitor saw " + String(hits)
            + " /ingest requests, want 8 (3 refused + 5 taken)"
        )
    _assert_in_order(String("live"), String("/ingest"), N)
    if sink.failed() != 3 or sink.rejected() != 0 or sink.spooled() != 0:
        raise Error(
            "503 burst: failed=" + String(sink.failed()) + " rejected="
            + String(sink.rejected()) + " spooled=" + String(sink.spooled())
        )
    print(
        "  3 x 503: retried in place, 5 of 5 delivered in order, nothing"
        " overtook the retried payload"
    )
    return 3


def check_poison_is_spooled_and_the_queue_moves(base: String) raises -> Int:
    var spool = String(TMP) + "_c.spool"
    _rm(spool)
    var sink = HttpPostSink(
        timeout_ms=2000,
        spool_path=spool,
        retry_min_ms=20,
        retry_max_ms=40,
        max_status_retries=3,
    )
    _ = sink.post(base + "/fail", String('{"poison":1}'))
    _ = sink.post(base + "/reject", String('{"bad":1}'))
    _ = sink.post(base + "/ingest", _body(0))
    var ok = _wait_sent(sink, 1, 5.0)
    sink.close(drain_ms=1000)
    if not ok:
        raise Error("poison: the payload behind it never went out")
    var fails = _count(String("live"), String("/fail"))
    var rejects = _count(String("live"), String("/reject"))
    if fails != 3 or rejects != 1:
        raise Error(
            "poison: /fail tried " + String(fails) + " times (want 3), /reject "
            + String(rejects) + " (want 1)"
        )
    var u = List[String]()
    var b = List[String]()
    read_spool(spool, u, b)
    if (
        len(u) != 2
        or u[0] != base + "/fail"
        or b[0] != '{"poison":1}'
        or u[1] != base + "/reject"
        or sink.rejected() != 2
        or sink.spooled() != 2
        or sink.abandoned() != 0
    ):
        raise Error(
            "poison: spool held " + String(len(u)) + " records, rejected="
            + String(sink.rejected()) + " spooled=" + String(sink.spooled())
        )
    _rm(spool)
    print(
        "  poison: a 500-forever payload spooled after 3 tries, a 400 at once;"
        " the payload behind them delivered"
    )
    return 3


def check_down_at_close_spools_and_replays() raises -> Int:
    comptime N = 10
    var port = _free_port()
    var base = "http://127.0.0.1:" + port
    var spool = String(TMP) + "_d.spool"
    _rm(spool)
    var sink = HttpPostSink(timeout_ms=500, capacity=4, spool_path=spool)
    for i in range(N):
        _ = sink.post(base + "/ingest", _body(i))
    var t0 = perf_counter_ns()
    sink.close(drain_ms=500)
    var close_ms = Float64(perf_counter_ns() - t0) / 1e6
    if close_ms > 2500.0:
        raise Error("close() took " + String(close_ms) + " ms while down")
    if sink.spooled() != N or sink.abandoned() != 0 or sink.sent() != 0:
        raise Error(
            "down at close: spooled=" + String(sink.spooled()) + " lost="
            + String(sink.abandoned()) + " sent=" + String(sink.sent())
        )
    var u = List[String]()
    var b = List[String]()
    read_spool(spool, u, b)
    if len(u) != N:
        raise Error("the spool holds " + String(len(u)) + " of " + String(N))
    for i in range(N):
        if b[i] != _body(i):
            raise Error("spool record " + String(i) + " is " + b[i])

    # A replay while the monitor is STILL down keeps everything.
    var r0 = replay_spool(spool, String(""), 500)
    if r0[0] != 0 or r0[1] != N or not exists(spool):
        raise Error("a replay against a dead port did not keep the spool")

    _ = _start_server(String("d"), port)
    var r = replay_spool(spool, String(""), 2000)
    _stop_server(base)
    if r[0] != N or r[1] != 0:
        raise Error(
            "replay: " + String(r[0]) + " delivered, " + String(r[1]) + " kept"
        )
    if exists(spool):
        raise Error("a fully replayed spool was not removed")
    _assert_in_order(String("d"), String("/ingest"), N)
    print(
        "  down at close: close() in " + String(Int(close_ms)) + " ms, "
        + String(N) + " spooled, 0 lost; replay kept all while down, then "
        + "delivered " + String(N) + " in order and removed the spool"
    )
    return 5


def main() raises:
    print("=== http_sink outage: hold, retry, spool ===")
    if not http_shim_available():
        raise Error(
            "the HTTP shim is not built — run `pixi run build-http` first"
        )
    var checks = 0
    checks += check_down_at_start_then_back()
    var live = _start_server(String("live"), String("0"))
    checks += check_overloaded_server_is_retried_in_place(live)
    checks += check_poison_is_spooled_and_the_queue_moves(live)
    _stop_server(live)
    checks += check_down_at_close_spools_and_replays()
    print("[PASS] http_sink outage (" + String(checks) + " checks)")


# MUTANTS THIS FILE WAS CHECKED AGAINST (each turned it red, 28 Sep 2026):
#   M1  a transport error during the run latches dead (the old policy) -> 1
#   M3  a retryable-status failure pops the head instead of holding it -> 2
#   M4  a transport failure at close discards the FIFO, not spools it  -> 4
# and `tests/io/test_http_sink.mojo` against
#   M2  `post()` drops on a full ring instead of parking the payload
