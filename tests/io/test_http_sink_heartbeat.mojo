# +--------------------------------------------------------------------------+ #
# | The heartbeat: when a silent run says "still here", and when it must not
# +--------------------------------------------------------------------------+ #
"""Gate the §7b ping in `noeira/io/http_sink.mojo`.

    pixi run build-http                                    # ONCE
    pixi run mojo run -I . tests/io/test_http_sink_heartbeat.mojo

Hermetic: `tools/io/mock_monitor_server.py` is a recording stand-in on a
loopback port. It answers 200 to everything and writes one timestamped line per
request, because the questions here are about WHEN and HOW OFTEN the client
speaks — which no status code can express.

## What could be wrong, and what each check is for

* **It never fires.** A heartbeat that does not beat leaves every SIGKILLed
  run `running` forever, which is the state P2 exists to end.
* **⚠⚠ It fires while payloads are flowing.** This is the defect the STAMP
  design exists to prevent, and a free-running timer would have it. A run
  logging every second would then send a pointless POST a minute — and worse,
  the ping would be arriving as evidence of liveness that the metrics were
  already providing.
* **It fires with no run to speak for.** A sink used for anything else must
  stay a pure queue.
* **It jumps the queue.** `SharedRing` is SPSC and the worker is its consumer,
  so the ping is posted DIRECTLY rather than pushed. That is only safe while
  the ring is empty; if a ping could precede the registration, the server would
  be pinged about a run it has never heard of.

## What is deliberately NOT gated here

⚠ The `ctl.should_stop()` guard in `_maybe_ping` suppresses at most ONE stray
ping between `close()` and the drain finishing. Distinguishing that from the
ping stream costs a blocking fixture route and ~4 s of wall clock for a
behaviour whose failure mode is one extra POST. It is asserted by reading, not
by this file. Saying so is cheaper than a check that would pass either way.
"""

from std.os.path import exists
from std.time import sleep

from noeira.io.fileio import remove_file
from noeira.io.http import http_shim_available
from noeira.io.http_sink import HttpPostSink
from noeira.io.proc import run_capture


comptime PORT_FILE = "/tmp/noeira_hb_gate_port"
comptime LOG_FILE = "/tmp/noeira_hb_gate_log"
comptime RUN_ID = "2026-09-10_hb-gate_deadbeef"


def _start_server() raises -> String:
    for p in [String(PORT_FILE), String(LOG_FILE)]:
        try:
            remove_file(p)
        except:
            pass
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py "
        + String(PORT_FILE)
        + " "
        + String(LOG_FILE)
        + " 120 > /tmp/noeira_hb_gate_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(PORT_FILE):
            var f = open(String(PORT_FILE), "r")
            var port = String(f.read().strip())
            f.close()
            if port.byte_length() > 0:
                return "http://127.0.0.1:" + port
        sleep(0.1)
    raise Error("the mock monitor never wrote " + String(PORT_FILE))


def _log() raises -> List[String]:
    """Every request the fixture saw, in arrival order."""
    var out = List[String]()
    if not exists(LOG_FILE):
        return out^
    var f = open(String(LOG_FILE), "r")
    var text = String(f.read())
    f.close()
    for line in text.split("\n"):
        var s = String(line).strip()
        if String(s).byte_length() > 0:
            out.append(String(s))
    return out^


def _paths() raises -> List[String]:
    var out = List[String]()
    for line in _log():
        var parts = String(line).split(" ")
        if len(parts) >= 3:
            out.append(String(parts[2]))
    return out^


def _count(needle: String) raises -> Int:
    var n = 0
    for p in _paths():
        if String(p) == needle:
            n += 1
    return n


def _clear() raises:
    var f = open(String(LOG_FILE), "w")
    f.write(String(""))
    f.close()


def main() raises:
    print("=== http_sink heartbeat (§7b) ===")
    if not http_shim_available():
        raise Error(
            "the HTTP shim is not built — run `pixi run build-http` first"
        )

    var base = _start_server()
    var ping = "/runs/" + String(RUN_ID) + "/ping"
    var ping_url = base + ping
    print("  fixture at " + base)
    var checks = 0

    # ── 1. a silent run beats ────────────────────────────────────────
    #
    # 150 ms interval, ~750 ms of silence. A run that never speaks is the
    # SIGKILL case the whole state machine hangs on.
    _clear()
    var a = HttpPostSink(
        timeout_ms=2000, ping_url=ping_url, ping_interval_ms=150
    )
    sleep(0.75)
    a.close(drain_ms=500)
    var beats = _count(String(ping))
    if beats < 3:
        raise Error(
            "a silent run beat only "
            + String(beats)
            + " times in 750 ms at a 150 ms interval — expected >= 3"
        )
    # ⚠ AND AN UPPER BOUND, which is the half that gates the stamp. Without
    # one, a worker that pings on EVERY idle lap — a 1 ms beat — passes a
    # "did it fire?" check with flying colours. 750/150 is five, plus slack.
    if beats > 8:
        raise Error(
            "a silent run beat "
            + String(beats)
            + " times in 750 ms at a 150 ms interval — the interval is not"
            " being honoured"
        )
    if a.pings() != beats:
        raise Error(
            "the sink counted "
            + String(a.pings())
            + " pings, the server saw "
            + String(beats)
        )
    print(
        "  silent run: "
        + String(beats)
        + " heartbeats in 750 ms at 150 ms, counter agrees"
    )
    checks += 2

    # ── 2. ⚠⚠ a BUSY run does not beat at all ───────────────────────
    #
    # THE DECISIVE CHECK. Interval 400 ms; a payload every ~250 ms for ~1.1 s.
    # A free-running timer fires twice in that window. A stamp that every send
    # resets fires ZERO times — and zero is the correct answer, because the
    # payloads themselves already proved the run alive.
    _clear()
    var b = HttpPostSink(
        timeout_ms=2000, ping_url=ping_url, ping_interval_ms=400
    )
    for _ in range(5):
        _ = b.post(base + "/ingest", String('{"run_id":"x","metrics":[]}'))
        sleep(0.25)
    b.close(drain_ms=1000)
    var busy_pings = _count(String(ping))
    var busy_posts = _count(String("/ingest"))
    if busy_posts != 5:
        raise Error(
            "the fixture saw "
            + String(busy_posts)
            + " of 5 payloads — the gate below would be vacuous"
        )
    if busy_pings != 0:
        raise Error(
            "a busy run sent "
            + String(busy_pings)
            + " heartbeats over 1.25 s at a 400 ms interval; a stamp reset by"
            " every send must send NONE"
        )
    print(
        "  busy run: 5 of 5 payloads delivered, "
        + String(busy_pings)
        + " heartbeats (a free-running timer would have sent 2)"
    )
    checks += 2

    # ── 3. no run to speak for, no heartbeat ────────────────────────
    _clear()
    var c = HttpPostSink(timeout_ms=2000, ping_interval_ms=50)
    sleep(0.4)
    c.close(drain_ms=200)
    var stray = len(_paths())
    if stray != 0:
        raise Error(
            "a sink with no ping_url sent "
            + String(stray)
            + " requests in 400 ms at a 50 ms interval"
        )
    if c.pings() != 0:
        raise Error("pings() nonzero with no ping_url")
    # ⚠ AND IT MUST NOT HAVE TRIED. A worker that pings an EMPTY url sends no
    # request the fixture can see, but it does latch the transport dead — so
    # "the server saw nothing" is true of both the correct code and the bug.
    # This is the check that separates them.
    if c.dead() or c.failed() != 0:
        raise Error(
            "a sink with no ping_url attempted "
            + String(c.failed())
            + " sends (dead="
            + String(c.dead())
            + ") — it must stay a pure queue, not POST to an empty URL"
        )
    print("  no ping_url: 0 requests in 400 ms at a 50 ms interval")
    checks += 2

    # ── 4. the heartbeat never jumps the queue ──────────────────────
    #
    # The stamp starts at `on_start`, so the first beat is one interval away —
    # long after the registration that must precede it. Interval 10 ms makes
    # the ping as eager as it can possibly be, and it still arrives second.
    _clear()
    var d = HttpPostSink(
        timeout_ms=2000, ping_url=ping_url, ping_interval_ms=10
    )
    _ = d.post(base + "/runs", String('{"run_id":"' + String(RUN_ID) + '"}'))
    sleep(0.25)
    d.close(drain_ms=500)
    var seen = _paths()
    if len(seen) < 2:
        raise Error("expected a registration and at least one beat")
    if String(seen[0]) != "/runs":
        raise Error(
            "the first request was "
            + String(seen[0])
            + ", not the registration — a ping preceded the run it names"
        )
    if _count(String(ping)) < 2:
        raise Error("no heartbeats followed the registration")
    print(
        "  ordering: /runs first, then "
        + String(_count(String(ping)))
        + " beats — the ping never precedes the run it names"
    )
    checks += 2

    # ── 5. the body carries the run id ──────────────────────────────
    var body_ok = False
    for line in _log():
        if String(line).find(ping) >= 0 and String(line).find(RUN_ID) >= 0:
            body_ok = True
    if not body_ok:
        raise Error("no ping line mentioned " + String(RUN_ID))
    print("  path and body both name the run")
    checks += 1

    # ── 6. the first beat is one interval away, not immediate ───────
    #
    # ⚠ THE STAMP STARTS AT `on_start`, NOT AT ZERO. Left at zero the very
    # first poll finds itself infinitely overdue and beats at t=0 — ahead of
    # the registration, if the ring happens to be empty for a moment while the
    # client is being built. Check 4 does NOT catch this: there the payload is
    # already queued by the time `poll` first runs, so the ordering holds by
    # luck rather than by design. This is the check that does not depend on
    # that race.
    _clear()
    var e = HttpPostSink(
        timeout_ms=2000, ping_url=ping_url, ping_interval_ms=5000
    )
    sleep(0.4)
    e.close(drain_ms=200)
    var early = _count(String(ping))
    if early != 0:
        raise Error(
            "a sink 400 ms old with a 5 s interval already beat "
            + String(early)
            + " times — the stamp does not start at on_start"
        )
    print("  first beat: 0 in 400 ms at a 5 s interval, not one at t=0")
    checks += 1

    # ── 7. a down transport is probed on the backoff, not the interval ──
    #
    # ⚠ THE BACKOFF, NOT THE PING INTERVAL, SETS THE PACE WHILE DOWN.
    # `127.0.0.1:9` is the discard port. This used to be the dead latch —
    # one failure, then silence for the rest of the run, which is also what
    # threw away every later metric. Now the transport is retried, but at
    # `retry_min` (100 ms here) doubling (200, 400, 800 …): about four probes
    # in 1.5 s, where a heartbeat that ignored the outage would make ~14.
    var f = HttpPostSink(
        timeout_ms=300,
        ping_url=String("http://127.0.0.1:9/runs/x/ping"),
        ping_interval_ms=100,
        retry_min_ms=100,
        retry_max_ms=5000,
    )
    sleep(1.5)
    f.close(drain_ms=200)
    if not f.down() or f.outages() != 1:
        raise Error(
            "the discard port did not register as an outage (down="
            + String(f.down()) + " outages=" + String(f.outages())
            + ") — the gate below would be vacuous"
        )
    if f.failed() < 2 or f.failed() > 6:
        raise Error(
            "a down transport was probed "
            + String(f.failed())
            + " times in 1.5 s; a 100 ms backoff doubling gives ~4, a"
            " heartbeat on its 100 ms interval ~14, a dead latch 1"
        )
    print(
        "  down transport: " + String(f.failed()) + " probes in 1.5 s on a"
        " doubling backoff (a 100 ms interval would have cost ~14)"
    )
    checks += 2

    _ = run_capture(
        "curl -s -X POST " + base + "/__shutdown > /dev/null 2>&1 || true"
    )
    print("[PASS] http_sink heartbeat (" + String(checks) + " checks)")


# MUTANTS THIS FILE WAS CHECKED AGAINST (each must turn it red):
#   N1  the ping does not reset the stamp        -> check 1's UPPER bound
#   N2  a real send does not reset the stamp     -> check 2 (the busy run)
#   N3  the empty-ping_url guard is removed      -> check 3 (dead/failed != 0)
#   N4  on_start leaves the stamp at 0           -> check 6
#   N5  the interval is in ms, not ns            -> check 2
#   N6  the ping ignores the outage backoff       -> check 7 (upper bound)
#
# ⚠ N4 AND N6 SURVIVED THE FIRST SWEEP and checks 6 and 7 exist because of it.
# Check 4 appeared to cover N4 and did not: there the payload is already queued
# by the time `poll` first runs, so the ordering held by a race rather than by
# the stamp. A check that passes for the right reason ONLY SOMETIMES is not a
# gate — and reading it would never have told you that.
