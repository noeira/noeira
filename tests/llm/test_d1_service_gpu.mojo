"""`D1Service`: d1-2B on its own thread behind `DecisionService`, asked the way
the voice loop asks Jev (`JevQuestions` in, `JevAnswers` out).

    build on the Orin: build-jetson tests/llm/test_d1_service_gpu.mojo [-D LFM3_BF16=1]

The reference's `room_en` case asked through the service, with a 50 Hz caller:
the answers equal d1's own (`answers.json`: same picks, probabilities within the
build's bound), read through `JevAnswers`' accessors, and no `poll` holds the
caller more than 2 ms while the model works. Then a missing checkpoint ends in
FAILED with its reason. Only ONE copy of the model is loaded (the service's):
two copies plus the load's staging would not fit next to each other on the Orin.
"""

from std.os import getenv
from std.time import perf_counter_ns

from noeira.ai.jev import JevQuestions
from noeira.ai.local_service import SVC_LOADING, SVC_READY, SVC_FAILED
from noeira.core.concurrent.thread import sleep_us
from noeira.io.json import load_json
from noeira.llm.d1.service import D1Service
from noeira.llm.lfm3.model import LF_BF16


comptime PROB_BOUND: Float64 = 3e-3 if LF_BF16 else 1e-4
comptime STATE = """{"robot": {"at": "door", "facing": "kitchen", "doing": "standing"}, "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right", "sofa": "4.5 m behind"}, "heard": "Can you go to the fridge and wait there?", "recently": ["walked to the door", "heard: hello robot"]}"""


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def main() raises:
    var fails = 0
    var home = getenv("HOME")
    print("D1Service (d1 in-process behind DecisionService)", "(LFM3_BF16)" if LF_BF16 else "(fp32)")
    var q = JevQuestions()
    q.noul("addressed", "Is the speaker asking the robot to do something?")
    q.choice("destination", "Where should the robot walk?", ["fridge", "door", "table", "sofa", "nowhere"],
             ["walk to the fridge", "walk to the door", "walk to the table", "walk to the sofa", "do not walk anywhere"])
    q.score("urgency", "How urgent is the request?", ["not urgent", "somewhat urgent", "very urgent"])

    var t0 = perf_counter_ns()
    var dec = D1Service.from_spec(String("native"))
    while dec.state() == SVC_LOADING and perf_counter_ns() - t0 < 300_000_000_000:
        _ = sleep_us(10_000)
    check(fails, dec.state() == SVC_READY, String("READY after ") + String((perf_counter_ns() - t0) // 1_000_000)
          + " ms (load + one warm call: " + String(dec.load_ms()) + " ms)")

    var worst_us = 0
    var frames = 0
    var t1 = perf_counter_ns()
    var f0 = perf_counter_ns()
    dec.start(STATE, q)
    worst_us = max(worst_us, (perf_counter_ns() - f0) // 1000)
    while True:
        frames += 1
        f0 = perf_counter_ns()
        var done = dec.poll()
        worst_us = max(worst_us, (perf_counter_ns() - f0) // 1000)
        if done:
            break
        _ = sleep_us(20_000)
    var a = dec.result()
    var wall = (perf_counter_ns() - t1) // 1_000_000

    var want = load_json(home + "/.cache/noeira/d1/ref/room_en/answers.json")
    var wa = want.field(want.root(), "answers")
    var wd = want.field(wa, "destination")
    var worst = abs(a.noul("addressed") - want.number(want.field(want.field(wa, "addressed"), "noul")))
    var opts: List[String] = ["fridge", "door", "table", "sofa", "nowhere"]
    for o in opts:
        worst = max(worst, abs(a.probability("destination", o) - want.number(want.field(want.field(wd, "probabilities"), o))))
    var wu = want.field(wa, "urgency")
    for lv in range(3):
        worst = max(worst, abs(a.probability("urgency", String(lv)) - want.number(want.field(want.field(wu, "probabilities"), String(lv)))))
    check(fails, a.choice("destination") == want.string(want.field(wd, "choice"))
          and a.score("urgency") == want.number(want.field(wu, "score")) and worst < PROB_BOUND,
          String("answers through JevAnswers equal d1's own: ") + a.choice("destination") + " ("
          + String(a.confidence("destination")) + "), worst probability |d| " + String(worst)
          + ", " + String(a.input_tokens) + " tokens")
    check(fails, worst_us < 2000, String("the caller's worst call ") + String(worst_us) + " us over " + String(frames)
          + " frames (" + String(wall) + " ms; the model's own " + String(a.latency_ms) + " ms)")

    var bad = D1Service.from_spec(String("native:/nonexistent/d1"))
    var t2 = perf_counter_ns()
    while bad.state() == SVC_LOADING and perf_counter_ns() - t2 < 60_000_000_000:
        _ = sleep_us(10_000)
    check(fails, bad.state() == SVC_FAILED, "a missing checkpoint ends in FAILED")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
