"""`D1Service`: d1-2B on its own thread behind `DecisionService`, asked the way
the voice loop asks Jev (`JevQuestions` in, `JevAnswers` out).

    build on the Orin: build-jetson tests/llm/test_d1_service_gpu.mojo [-D LFM3_BF16=1]

The reference's `room_en` case asked through the service, with a 50 Hz caller:
the answers equal d1's own (`answers.json`: same picks, probabilities within the
build's bound), read through `JevAnswers`' accessors, and no `poll` holds the
caller more than 2 ms while the model works. Then WITH A PICTURE
(`start_seeing`, the VL reference's `vl_one`: a 640x480 NYU room frame, 0.9 MB
through the request ring): the answers equal d1's own with that picture, and
neither `start_seeing` nor any `poll` holds the caller more than 2 ms. A
picture the tower does not take (100x100) fails THAT call with the reason, and
the next call works. Then a missing checkpoint ends in FAILED with its reason.
Only ONE copy of the model is loaded (the service's): two copies plus the
load's staging would not fit next to each other on the Orin.
"""

from std.os import getenv
from std.time import perf_counter_ns

from noeira.ai.jev import JevAnswers, JevQuestions
from noeira.ai.picture import Picture
from noeira.ai.local_service import SVC_LOADING, SVC_READY, SVC_FAILED
from noeira.core.concurrent.thread import sleep_us
from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.llm.d1.service import D1Service
from noeira.llm.lfm3.model import LF_BF16


comptime PROB_BOUND: Float64 = 3e-3 if LF_BF16 else 1e-4
comptime PIC_PROB_BOUND: Float64 = 5e-3 if LF_BF16 else 1e-4
"""With a picture: the VL gate's bound (bf16 measured <= 2.6e-3: the tower drifts as torch's own bf16 does)."""
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

    # with a picture
    var pq = JevQuestions()
    pq.noul("door_open", "Is there an open door in the picture?")
    pq.choice("furniture", "Which of these is in the picture?", ["sofa", "bed", "fridge", "car"],
              ["a sofa", "a bed", "a fridge", "a car"])
    pq.score("light", "How bright is the room?", ["dark", "dim", "bright"])
    var vdir = home + "/.cache/noeira/d1/ref_vl/vl_one"
    var px = load_npy_f64(vdir + "/image0.npy")
    var rgb = List[UInt8](capacity=len(px))
    for v in px:
        rgb.append(UInt8(Int(v)))
    var pics = List[Picture]()
    pics.append(Picture(rgb^, 480, 640))
    var pworst = 0
    var pframes = 0
    var t3 = perf_counter_ns()
    var g0 = perf_counter_ns()
    dec.start_seeing(String("The robot's head camera sees this."), False, pq, pics)
    var start_us = (perf_counter_ns() - g0) // 1000
    pworst = start_us
    while True:
        pframes += 1
        g0 = perf_counter_ns()
        var done = dec.poll()
        pworst = max(pworst, (perf_counter_ns() - g0) // 1000)
        if done:
            break
        _ = sleep_us(20_000)
    var pa = dec.result()
    var pwall = (perf_counter_ns() - t3) // 1_000_000
    var pw = load_json(vdir + "/answers.json")
    var pwa = pw.field(pw.root(), "answers")
    var pwf = pw.field(pwa, "furniture")
    var pwl = pw.field(pwa, "light")
    var perr = abs(pa.noul("door_open") - pw.number(pw.field(pw.field(pwa, "door_open"), "noul")))
    var fo: List[String] = ["sofa", "bed", "fridge", "car"]
    for o in fo:
        perr = max(perr, abs(pa.probability("furniture", o) - pw.number(pw.field(pw.field(pwf, "probabilities"), o))))
    for lv in range(3):
        perr = max(perr, abs(pa.probability("light", String(lv)) - pw.number(pw.field(pw.field(pwl, "probabilities"), String(lv)))))
    check(fails, pa.choice("furniture") == pw.string(pw.field(pwf, "choice"))
          and pa.score("light") == pw.number(pw.field(pwl, "score")) and perr < PIC_PROB_BOUND,
          String("WITH A PICTURE (640x480): answers equal d1's own: open door ") + String(pa.noul("door_open"))
          + ", " + pa.choice("furniture") + " (" + String(pa.confidence("furniture")) + "), worst probability |d| "
          + String(perr) + ", " + String(pa.input_tokens) + " tokens")
    check(fails, pworst < 2000, String("the caller's worst call with the picture ") + String(pworst) + " us (start_seeing "
          + String(start_us) + " us) over " + String(pframes) + " frames (" + String(pwall) + " ms; the model's own "
          + String(pa.latency_ms) + " ms)")
    # a picture the tower does not take fails that call, and the service goes on
    var small = List[Picture]()
    small.append(Picture(List[UInt8](length=100 * 100 * 3, fill=0), 100, 100))
    dec.start_seeing(String("The robot's head camera sees this."), False, pq, small)
    while not dec.poll():
        _ = sleep_us(10_000)
    var why = String("")
    try:
        _ = dec.result()
    except e:
        why = String(e)
    dec.start(STATE, q)
    while not dec.poll():
        _ = sleep_us(10_000)
    var again = dec.result()
    check(fails, why.find("resampled") >= 0 and again.choice("destination") == want.string(want.field(wd, "choice")),
          "a 100x100 picture fails its call (" + why + "), and the next call answers (text, "
          + String(again.latency_ms) + " ms)")

    var bad = D1Service.from_spec(String("native:/nonexistent/d1"))
    var t2 = perf_counter_ns()
    while bad.state() == SVC_LOADING and perf_counter_ns() - t2 < 60_000_000_000:
        _ = sleep_us(10_000)
    check(fails, bad.state() == SVC_FAILED, "a missing checkpoint ends in FAILED")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
