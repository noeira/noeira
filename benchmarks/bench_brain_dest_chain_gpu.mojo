# +--------------------------------------------------------------------------+ #
# | "Go to the table, then to the door": two places in order, from d1
# +--------------------------------------------------------------------------+ #
"""The two-place request's bench (docs/ROBOT_BRAIN_PLAN.md, step B2).

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_brain_dest_chain_gpu.mojo -o build/bench_chain
    ./build/bench_chain [out.tsv] [tests/brain/dest_chain.tsv] [d1 spec, default native-text]

Each labelled sentence (`first`, `then`: the places the robot should be sent
to, `none` for nothing) goes through what the voice loop does with
`request_wording` and `dest_chain`:

1. the destination request (the command state, the destination question and
   its gate, RB-2's triage question, d1's `name_desc` options), plus one yes/no
   question: are two places asked for (`g1_add_two_places_gate`);
2. only when a place was routed and that gate says yes: a second request,
   the destination request again with which place second and which first
   (`g1_two_places_questions`), against a state with nothing of the room in
   it (`g1_two_places_state`).

Step 1 is sent under TWO states:

    bare   the robot standing, at no place, nothing else in the state
    room   the room demo's state (`--room-state`): the places around the robot,
           its trail, and the robot standing AT a place the sentence does not
           name, which it has just arrived at

⚠ The second exists because the first hid a fault: asked against the room's
state, the which-place questions answered the place the robot stood at. A
bench must send what the demo sends.

Printed per split and state: first places right without and with the gate
question on the request (asking it must not move them); both places right;
two-place sentences right; one-place sentences given a second place; how many
sentences paid the second request; the requests' latency.
"""

from std.sys import argv
from std.time import perf_counter_ns

from noeira.ai.jev import JevQuestions, JevAnswers
from noeira.brain.geometry import PlaceXY, place_lines
from noeira.core.bytes import string_from_bytes
from noeira.core.concurrent.thread import sleep_us
from noeira.envs.robots.g1_command_language import (
    G1Context, g1_command_state, g1_destination_questions, g1_route_destination, g1_add_triage,
    g1_add_two_places_gate, g1_two_places_questions, g1_two_places_state, g1_route_two_places,
    G1_Q_IS_TWO_PLACES, G1_Q_DESTINATION_FIRST, G1_Q_DESTINATION_THEN,
)
from noeira.io.fileio import read_file_bytes
from noeira.llm.d1.service import D1Service


struct Row(Copyable, Movable):
    var split: String
    var state: String
    var first: String
    var then: String
    var lang: String
    var text: String
    var got_plain: String
    var got_first: String
    var got_then: String
    var two_p: Float64
    var asked_pair: Bool
    var pair: String
    var ms_plain: Float64
    var ms_gate: Float64
    var ms_pair: Float64

    def __init__(out self, split: String, state: String, first: String, then: String, lang: String, text: String):
        self.split = split.copy()
        self.state = state.copy()
        self.first = first.copy()
        self.then = then.copy()
        self.lang = lang.copy()
        self.text = text.copy()
        self.got_plain = String("")
        self.got_first = String("")
        self.got_then = String("none")
        self.two_p = 0.0
        self.asked_pair = False
        self.pair = String("")
        self.ms_plain = 0.0
        self.ms_gate = 0.0
        self.ms_pair = 0.0


def wait(mut d1: D1Service) raises -> JevAnswers:
    while not d1.poll():
        _ = sleep_us(1000)
    return d1.result()


def or_none(s: String) -> String:
    return String("none") if s == "" else s


def main() raises:
    var args = argv()
    var out_path = String(args[1]) if len(args) > 1 else String("")
    var set_path = String(args[2]) if len(args) > 2 else String("tests/brain/dest_chain.tsv")
    var spec = String(args[3]) if len(args) > 3 else String("native-text")
    # `nokind`: the second request without the triage question (`g1_two_places_questions(with_kind=False)`)
    var with_kind = not (len(args) > 4 and String(args[4]) == "nokind")
    var rows = List[Row]()
    for l in string_from_bytes(read_file_bytes(set_path)).split("\n"):
        var line = String(l)
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var c = List[String]()
        for x in line.split("\t"):
            c.append(String(x))
        if len(c) != 5:
            raise Error("dest_chain: a row with " + String(len(c)) + " columns")
        rows.append(Row(c[0], String("bare"), c[1], c[2], c[3], c[4]))
        rows.append(Row(c[0], String("room"), c[1], c[2], c[3], c[4]))
    print("two-place requests:", len(rows) // 2, "sentences under 2 states; d1", spec)

    var places: List[String] = ["fridge", "door", "table", "sofa"]
    var q_plain = g1_destination_questions(places)
    g1_add_triage(q_plain, True)
    var q_gate = g1_destination_questions(places)
    g1_add_triage(q_gate, True)
    g1_add_two_places_gate(q_gate)
    var q_pair = g1_two_places_questions(places, True, with_kind)
    print("the second request:", "as measured first" if with_kind else "WITHOUT the triage question")
    var layout: List[PlaceXY] = [
        PlaceXY("fridge", 3.0, 1.5), PlaceXY("door", -3.2, 2.0), PlaceXY("table", 1.0, -2.5), PlaceXY("sofa", -2.0, -2.8),
    ]
    var warm = G1Context()
    warm.places = True
    var d1 = D1Service.from_spec(spec)
    d1.start(g1_command_state(String("Hello."), warm), q_gate)
    _ = wait(d1)
    d1.start(g1_command_state(String("Hello."), warm), q_plain)
    _ = wait(d1)
    d1.start(g1_two_places_state(String("Hello.")), q_pair)
    _ = wait(d1)
    for i in range(len(rows)):
        var ctx = G1Context()
        ctx.places = True
        if rows[i].state == "room":
            # the robot stands at a place the sentence does not name, just arrived there
            var here = 0
            for k in range(len(layout)):
                if layout[k].name != rows[i].first and layout[k].name != rows[i].then:
                    here = k
                    break
            ctx.geometry = True
            ctx.memory = True
            ctx.began(String("stand"))
            ctx.at = layout[here].name.copy()
            ctx.arrived = layout[here].name.copy()
            ctx.arrived_at(layout[here].name, 1.0)
            ctx.around = place_lines(layout, layout[here].x * 0.8, layout[here].y * 0.8, 30.0)
        var state = g1_command_state(rows[i].text, ctx)
        var t0 = perf_counter_ns()
        d1.start(state, q_gate)
        var a = wait(d1)
        rows[i].ms_gate = Float64(perf_counter_ns() - t0) / 1e6
        rows[i].got_first = or_none(g1_route_destination(a))
        rows[i].two_p = a.noul(String(G1_Q_IS_TWO_PLACES))
        if rows[i].got_first != "none" and rows[i].two_p >= 0.5:
            var t2 = perf_counter_ns()
            d1.start(g1_two_places_state(rows[i].text), q_pair)
            var p = wait(d1)
            rows[i].ms_pair = Float64(perf_counter_ns() - t2) / 1e6
            rows[i].asked_pair = True
            rows[i].pair = p.choice(String(G1_Q_DESTINATION_FIRST)) + ">" + p.choice(String(G1_Q_DESTINATION_THEN))
            var two = g1_route_two_places(p)
            if two[0] != "":
                rows[i].got_first = two[0]
                rows[i].got_then = two[1]
        var t1 = perf_counter_ns()
        d1.start(state, q_plain)
        var b = wait(d1)
        rows[i].ms_plain = Float64(perf_counter_ns() - t1) / 1e6
        rows[i].got_plain = or_none(g1_route_destination(b))

    var splits = List[String]()
    for r in rows:
        var seen = False
        for x in splits:
            if x == r.split:
                seen = True
        if not seen:
            splits.append(r.split.copy())
    var kinds: List[String] = ["bare", "room"]
    for sk in range(len(splits) * 2):
        var s = splits[sk // 2].copy()
        var kind = kinds[sk % 2].copy()
        var n = 0
        var first_plain = 0
        var first_gate = 0
        var both = 0
        var two_n = 0
        var two_ok = 0
        var extra = 0
        var paid = 0
        for r in rows:
            if r.split != s or r.state != kind:
                continue
            n += 1
            if r.got_plain == r.first or (r.then != "none" and r.got_plain == r.then):
                first_plain += 1   # one place of the sentence, as before B2
            if r.got_first == r.first:
                first_gate += 1
            if r.got_first == r.first and r.got_then == r.then:
                both += 1
            if r.asked_pair:
                paid += 1
            if r.then != "none":
                two_n += 1
                if r.got_first == r.first and r.got_then == r.then:
                    two_ok += 1
            elif r.got_then != "none":
                extra += 1
        print(s + ", " + kind + " state:", n, "sentences | a named place routed before B2:", first_plain, "| first place right now:", first_gate,
              "| both right:", both, "/", n, "| two-place sentences:", two_ok, "/", two_n,
              "| a second place invented:", extra, "/", n - two_n, "| second request asked:", paid)
    var g = 0.0
    var pl = 0.0
    var pr = 0.0
    var npair = 0
    for r in rows:
        g += r.ms_gate
        pl += r.ms_plain
        if r.asked_pair:
            pr += r.ms_pair
            npair += 1
    print("destination request, mean:", Int(pl / Float64(len(rows))), "ms before,", Int(g / Float64(len(rows))), "ms with the gate question; the second request:",
          Int(pr / Float64(max(npair, 1))), "ms, asked for", npair, "of", len(rows))
    print("wrong:")
    for r in rows:
        if r.got_first != r.first or r.got_then != r.then:
            print("   ", r.split, r.state, "| want", r.first, "then", r.then, "| got", r.got_first, "then", r.got_then, "( two places", round(r.two_p, 2),
                  ", pair", r.pair, ") | before B2:", r.got_plain, "|", r.text)
    if out_path != "":
        var tsv = String("split\tstate\tfirst\tthen\tlang\tgot_plain\tgot_first\tgot_then\ttwo_p\tpair\tms_plain\tms_gate\tms_pair\ttext\n")
        for r in rows:
            tsv += (r.split + "\t" + r.state + "\t" + r.first + "\t" + r.then + "\t" + r.lang + "\t" + r.got_plain + "\t" + r.got_first + "\t" + r.got_then
                    + "\t" + String(round(r.two_p, 3)) + "\t" + r.pair + "\t" + String(Int(r.ms_plain)) + "\t" + String(Int(r.ms_gate)) + "\t"
                    + String(Int(r.ms_pair)) + "\t" + r.text + "\n")
        with open(out_path, "w") as f:
            f.write(tsv)
        print("answers written to", out_path)
