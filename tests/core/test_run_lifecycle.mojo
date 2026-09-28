"""A run's END is a signal it sends, and its START is not the first metric.

Run: pixi run mojo run -I . tests/core/test_run_lifecycle.mojo

Two holes, closed together, both of them about a run whose identity the
dashboard never learned:

  * `RemoteLogger` used to POST `/runs` from `flush`, which returns early on an
    empty buffer. A run that died before its first batch — a bad config, an OOM
    while building the model, a compile that never reached step 0 — **never
    appeared at all**, so the failure a liveness signal most needs to show was
    the one case with no row to mark. `register()` is the fix.
  * Nothing ever told the server a run was over, so every run on the dashboard
    is `isActive` forever. `finish()` is the fix, and `close()` calls it.

⚠ THIS GATE IS OFFLINE, like `tests/io/test_http_sink.mojo` beside it. It never
talks to a live dashboard. What it can gate is everything that is not the
network: the URLs and payloads by VALUE, the ordering rules, the idempotence
latches, and the property that keeps all of it out of the way of a run with no
dashboard configured at all.

`127.0.0.1:9` is the discard port — nothing listens, so a connection is refused
promptly and deterministically.
"""

from noeira.core.logger import (
    CompositeLogger,
    CsvLogger,
    NoOpLogger,
    RemoteLogger,
)
from noeira.io.fileio import remove_file


comptime DEAD = "http://127.0.0.1:9"


def _logger(run_id: String) raises -> RemoteLogger:
    # ⚠ THE SPOOL GOES TO /tmp: against the discard port every payload ends
    # there, and the default (`logs/remote_spool/`) is inside the tree. It is
    # appended to, so a stale one from the last run is removed first.
    var spool = "/tmp/noeira_lifecycle_gate_" + run_id + ".spool"
    try:
        remove_file(spool)
    except:
        pass
    return RemoteLogger(
        server_url=String(DEAD),
        run_name=String("gate"),
        run_id=run_id,
        spool_path=spool,
    )


# =============================================================================


def test_urls_are_built_from_the_base() raises:
    """The four routes, by value.

    ⚠ A TRAILING SLASH ON THE BASE MUST NOT DOUBLE. `.env` is hand-edited and
    `NOEIRA_CLOUD_URL` has arrived both ways.
    """
    var checked = 0
    var wrong = 0
    for base in [String(DEAD), String(DEAD) + "/"]:
        var lg = RemoteLogger(server_url=base, run_id=String("r1"))
        var cases = [
            (lg._ingest_url(), String(DEAD) + "/ingest"),
            (lg._runs_url(), String(DEAD) + "/runs"),
            (lg._finish_url(), String(DEAD) + "/runs/r1/finish"),
            (lg._ping_url(), String(DEAD) + "/runs/r1/ping"),
        ]
        for c in cases:
            checked += 1
            if c[0] != c[1]:
                wrong += 1
                print("    got", c[0], "want", c[1])
    print("  urls:", checked, "checked,", wrong, "differing")
    if wrong != 0 or checked != 8:
        raise Error("url construction: " + String(wrong) + " of 8 wrong")


def test_register_payload_carries_the_config() raises:
    """⚠ WHICH IS WHY `register()` IS NOT CALLED FROM THE CONSTRUCTOR.

    Every driver calls `set_config` after building the logger. Registering any
    earlier would ship an empty config on every run — trading the hole this
    closes for a quieter one.
    """
    var lg = _logger(String("2026-09-09_act-reach_a3f21c8b"))
    lg.set_config(String("algorithm"), String("ACT"))
    lg.set_config(String("lr"), String("3e-4"))
    var p = lg._register_payload()
    var want = [
        String('"run_id":"2026-09-09_act-reach_a3f21c8b"'),
        String('"run_name":"gate"'),
        String('"algorithm":"ACT"'),
        String('"lr":"3e-4"'),
    ]
    var missing = 0
    for w in want:
        if p.find(w) < 0:
            missing += 1
            print("    absent:", w)
    print("  register payload:", len(want) - missing, "of", len(want), "fields")
    if missing != 0:
        raise Error("register payload missing " + String(missing) + " fields")


def test_ping_payload_names_the_run() raises:
    """⚠ THE PING'S BODY IS NOT EMPTY, and the reason is not the server.

    The route already carries the id, so the body is redundant to it — but the
    sink's transport POSTs JSON, and a body that names the run is what makes a
    captured ping readable in a log without cross-referencing the URL.
    """
    var lg = _logger(String("2026-09-10_hb_deadbeef"))
    var p = lg._ping_payload()
    if p.find('"run_id":"2026-09-10_hb_deadbeef"') < 0:
        raise Error("ping payload does not name the run: " + p)
    print("  ping payload names the run")


def test_finish_payload_carries_status_and_outcome() raises:
    var lg = _logger(String("r2"))
    var p = lg._finish_payload(String("killed"), String("success_rate=0.82"))
    var want = [
        String('"run_id":"r2"'),
        String('"status":"killed"'),
        String('"outcome":"success_rate=0.82"'),
    ]
    var missing = 0
    for w in want:
        if p.find(w) < 0:
            missing += 1
            print("    absent:", w)
    print("  finish payload:", len(want) - missing, "of", len(want), "fields")
    if missing != 0:
        raise Error("finish payload missing " + String(missing) + " fields")


def test_registration_precedes_the_first_metric() raises:
    """The hole this whole gate exists for.

    ⚠ THE ASSERTION THAT MATTERS IS `total_logged() == 0` BESIDE `registered()`.
    "It registered eventually" was always true; the claim is that it registered
    with **no metric having been logged**, which is the state a run that dies at
    step 0 is in.
    """
    var lg = _logger(String("r3"))
    if lg.registered():
        raise Error("registered before register() was called")
    lg.register()
    if not lg.registered() or lg.total_logged() != 0 or lg.pending() != 0:
        raise Error(
            "after register(): registered=" + String(lg.registered())
            + " logged=" + String(lg.total_logged())
            + " pending=" + String(lg.pending())
        )
    lg.register()  # idempotent
    lg.close()
    # /runs + /finish, and nothing else — no metric was ever logged.
    var n = lg.posts_attempted()
    print("  register-then-die:", n, "payloads accounted for (want 2)")
    if n != 2:
        raise Error("expected 2 payloads (runs, finish), saw " + String(n))


def test_close_reports_done_and_does_not_overwrite_a_stated_end() raises:
    """⚠ THE FIRST `finish` WINS.

    A driver that reports `killed` on its way out must not have that overwritten
    by the `done` from the `close()` behind it — a killed run filed as clean is
    worse than no record at all.
    """
    var clean = _logger(String("r4"))
    clean.register()
    clean.close()
    if not clean.finished():
        raise Error("close() did not report a terminal state")

    var killed = _logger(String("r5"))
    killed.register()
    killed.finish(String("killed"), String("diverged at 40k"))
    if not killed.finished():
        raise Error("finish() did not latch")
    killed.close()
    # ⚠ THE COUNT IS THE ASSERTION. /runs + the killed /finish is two, and a
    # `close()` that re-finished would make it three. Offline there is no way
    # to read back which status was sent, so the latch is gated by the payload
    # that was NOT sent rather than by the one that was.
    var n = killed.posts_attempted()
    print("  stated end:", n, "payloads accounted for (want 2, a third = re-finish)")
    if n != 2:
        raise Error("close() re-finished a stated end: " + String(n))


def test_an_unregistered_run_has_nothing_to_finish() raises:
    """⚠ INVENTING A ROW AT THE END WOULD ADVERTISE A RUN WHOSE WHOLE HISTORY IS
    THE FACT THAT IT STOPPED."""
    var lg = _logger(String("r6"))
    lg.close()
    if lg.finished() or lg.posts_attempted() != 0:
        raise Error(
            "unregistered run posted: finished=" + String(lg.finished())
            + " posts=" + String(lg.posts_attempted())
        )
    print("  unregistered run: 0 payloads, as intended")


def test_inert_without_a_server_url() raises:
    """The property that lets one driver serve both worlds (§10 of the plan).

    ⚠ `close()` NOW CALLS `finish()`, SO THIS IS A NEW WAY TO BREAK INERTNESS.
    A logger with no `server_url` must still start no sink and no thread.
    """
    var lg = RemoteLogger(server_url=String(""), run_id=String("r7"))
    lg.register()
    for i in range(200):
        lg.log_scalar(String("reward"), Float64(i), i)
    lg.finish(String("killed"), String(""))
    lg.close()
    # ⚠ WHAT THIS DOES *NOT* GATE, SAID OUT LOUD. `finish`'s own empty-url
    # clause is unreachable: an inert logger never registers, so the
    # registration guard fires first. Deleting that clause leaves this test
    # green — measured, not assumed. It is kept in the source for consistency
    # with every other method there, not because this asserts it.
    if lg.registered() or lg.finished() or lg.posts_attempted() != 0:
        raise Error(
            "inert logger was not inert: registered="
            + String(lg.registered())
            + " finished=" + String(lg.finished())
            + " posts=" + String(lg.posts_attempted())
        )
    print("  inert logger: 200 log_scalar calls, no sink, no registration")


def test_composite_fans_out_both_calls() raises:
    """⚠ THE SHAPE EVERY DRIVER ACTUALLY HOLDS.

    `CompositeLogger` is what a driver with a CSV and a dashboard passes around,
    so a `register`/`finish` that stopped at the wrapper would reach nothing.
    The `Logger` trait declares both with `...` and no default body precisely so
    that a missing override is a compile error rather than a silent no-op
    (`_a_trait_default_of_pass_makes_a_missing_override_silent`).

    ⚠⚠ ASSERT BEFORE `close()`, NEVER AFTER. `close()` fans out on its own, and
    the remote half's own `close()` finishes the run by itself — so a check made
    after it passes whether or not `register`/`finish` fanned out at all. That is
    not hypothetical: deleting `self.b.finish(...)` left the assert-after-close
    version of this test GREEN. It is the vacuity shape, in the gate written to
    prevent it.
    """
    var remote = _logger(String("r8"))
    var lg = CompositeLogger(CsvLogger(String("/dev/null")), remote)

    # ⚠ THE LATCHES, NOT THE COUNTERS, ARE WHAT CAN BE READ HERE.
    # `posts_attempted` counts what the worker has PROCESSED, so before the
    # drain it is 0 no matter what was queued. `registered`/`finished` are set
    # on the calling thread and are true the instant the fan-out reaches the
    # remote half — which is precisely the claim under test.
    lg.register()
    if not lg.b.registered():
        raise Error("register() did not reach the remote half")

    lg.finish(String("crashed"), String("oom"))
    if not lg.b.finished():
        raise Error("finish() did not reach the remote half")

    lg.close()
    var n = lg.b.posts_attempted()
    print("  composite: both latches set before close,", n, "payloads (want 2)")
    if n != 2:
        raise Error("composite fan-out posted " + String(n) + ", want 2")



def test_noop_stays_free() raises:
    var lg = NoOpLogger()
    lg.register()
    lg.finish(String("done"), String(""))
    lg.close()
    if lg.is_active():
        raise Error("NoOpLogger reported active")
    print("  noop: register/finish/close are free")


def main() raises:
    print("=" * 62)
    print("Run lifecycle — registration before step 0, and an end signal")
    print("=" * 62)
    test_urls_are_built_from_the_base()
    test_register_payload_carries_the_config()
    test_ping_payload_names_the_run()
    test_finish_payload_carries_status_and_outcome()
    test_registration_precedes_the_first_metric()
    test_close_reports_done_and_does_not_overwrite_a_stated_end()
    test_an_unregistered_run_has_nothing_to_finish()
    test_inert_without_a_server_url()
    test_composite_fans_out_both_calls()
    test_noop_stays_free()
    print("[PASS] run lifecycle")
