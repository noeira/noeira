"""`noeira.ai.local_service`: a model on its own thread, polled like an API call.

    pixi run mojo run -I . tests/ai/test_local_service.mojo

Toy models only (no weights, no GPU): the machinery the in-process speech
services stand on — the load state, a call whose `poll` never blocks while the
model works, `cancel` dropping the stale reply, a failed load and a failing
call reported through `take`, a reply too big for its slot.
"""

from std.time import perf_counter_ns

from noeira.ai.local_service import (
    LocalCall, LocalModel, LocalWorker, SVC_LOADING, SVC_READY, SVC_FAILED,
)
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread


struct Upper(LocalModel):
    """Uppercases its request after `delay_us`; "boom" raises; "big" answers
    more than any slot holds."""

    var delay_us: Int
    var fail_load: Bool
    var loaded: Bool

    def __init__(out self, delay_us: Int, fail_load: Bool = False):
        self.delay_us = delay_us
        self.fail_load = fail_load
        self.loaded = False

    def load(mut self) raises:
        _ = sleep_us(20_000)
        if self.fail_load:
            raise Error("no weights here")
        self.loaded = True

    def warm(mut self) raises:
        pass

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        if not self.loaded:
            raise Error("served before load")
        _ = sleep_us(self.delay_us)
        var s = String(unsafe_from_utf8=Span(request))
        if s == "boom":
            raise Error("the model raised")
        if s == "big":
            return List[UInt8](length=1 << 20, fill=65)
        var out = List[UInt8](capacity=len(request))
        for b in request:
            out.append(b - 32 if b >= 97 and b <= 122 else b)
        return out^


def bytes_of(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for x in s.as_bytes():
        b.append(x)
    return b^


def text_of(ref b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=Span(b))


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok:", what)
    else:
        print("  FAIL:", what)
        fails += 1


def wait_done(mut call: LocalCall, timeout_ms: Int) -> Tuple[Bool, Int]:
    """Poll like a frame loop; returns (finished, polls)."""
    var t0 = perf_counter_ns()
    var polls = 0
    while (perf_counter_ns() - t0) // 1_000_000 < timeout_ms:
        polls += 1
        if call.poll():
            return (True, polls)
        _ = sleep_us(500)
    return (False, polls)


def main() raises:
    print("local_service")
    var fails = 0
    # ── a working model ──
    var call = LocalCall(4096, 64 * 1024)
    var bg = BackgroundThread(LocalWorker(Upper(100_000), call))
    check(fails, call.state() == SVC_LOADING, "LOADING right after the thread starts (load takes 20 ms)")
    var b = bytes_of("hello, robot")
    call.begin(b, "upper")          # queued while it loads: served after
    var t0 = perf_counter_ns()
    var r = wait_done(call, 5000)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    check(fails, r[0], "the call finishes")
    check(fails, r[1] > 20, String("poll never blocks: ") + String(r[1]) + " polls over " + String(ms) + " ms of a 100 ms model")
    check(fails, call.state() == SVC_READY, "READY once loaded")
    var out = call.take("upper")
    check(fails, text_of(out) == "HELLO, ROBOT", String("the reply: '") + text_of(out) + "'")
    check(fails, call.latency_ms() >= 99.0 and call.latency_ms() < 300.0,
          String("latency is the model's own: ") + String(call.latency_ms()) + " ms")

    # ── cancel: the abandoned reply is dropped, the next one is ours ──
    var b1 = bytes_of("first")
    call.begin(b1, "upper")
    call.cancel()
    var b2 = bytes_of("second")
    call.begin(b2, "upper")
    r = wait_done(call, 5000)
    out = call.take("upper")
    check(fails, text_of(out) == "SECOND", String("after a cancel, the reply is the new call's: '") + text_of(out) + "'")

    # ── a model error comes back through take ──
    var bb = bytes_of("boom")
    call.begin(bb, "upper")
    _ = wait_done(call, 5000)
    var raised = String("")
    try:
        _ = call.take("upper")
    except e:
        raised = String(e)
    check(fails, raised.find("the model raised") >= 0, String("a serve error raises from take: ") + raised)

    # ── a reply bigger than its slot ──
    var bg_ = bytes_of("big")
    call.begin(bg_, "upper")
    _ = wait_done(call, 5000)
    raised = String("")
    try:
        _ = call.take("upper")
    except e:
        raised = String(e)
    check(fails, raised.find("does not fit") >= 0, String("an oversize reply raises: ") + raised)

    # ── no call: poll is True, take raises ──
    check(fails, call.poll(), "poll with no call in flight returns True (a loop cannot hang on it)")
    bg.stop()

    # ── a failed load ──
    var call2 = LocalCall(4096, 4096)
    var bg2 = BackgroundThread(LocalWorker(Upper(0, fail_load=True), call2))
    var t1 = perf_counter_ns()
    while call2.state() == SVC_LOADING and (perf_counter_ns() - t1) < 2_000_000_000:
        _ = sleep_us(1000)
    check(fails, call2.state() == SVC_FAILED, "FAILED after a load that raised")
    var bh = bytes_of("hello")
    call2.begin(bh, "upper")
    _ = wait_done(call2, 2000)
    raised = String("")
    try:
        _ = call2.take("upper")
    except e:
        raised = String(e)
    check(fails, raised.find("no weights here") >= 0, String("a request after a failed load says why: ") + raised)
    bg2.stop()

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("=== local_service: ALL PASS ===")
