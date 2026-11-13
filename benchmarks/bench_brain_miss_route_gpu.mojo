# +--------------------------------------------------------------------------+ #
# | A sentence that matched no command: a movement to sketch, or a reply? (d1)
# +--------------------------------------------------------------------------+ #
"""The miss route's bench (docs/ROBOT_BRAIN_PLAN.md, step B2).

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_brain_miss_route_gpu.mojo -o build/bench_miss
    ./build/bench_miss [out.tsv] [tests/brain/miss_route.tsv] [d1 spec, default native-text]

When the voice loop's decision finds no command for a sentence addressed to the
robot, it asks d1 for a reward sketch (`g1_spec_questions`), whose slot 0
always names a quantity. Each labelled sentence of the set (`movement`: a
movement of the robot's own body; `reply`: a question, something to say, an
action on an object) is sent as the loop sends it, the instruction alone, and
routed three ways:

    slots      as before B2: a reply only when slot 0 answers `none`
    slot >= t  a reply when slot 0's probability is under t (the best t on the
               tune rows, then applied to the held rows)
    kind       `g1_add_miss_kind`'s question: a reply when P(movement) < 0.5

Printed per split (tune, held) and want (movement, reply): sentences routed
right, the request's latency with and without the kind question. `out.tsv`
gets every answer.
"""

from std.sys import argv
from std.time import perf_counter_ns

from noeira.ai.jev import JevQuestions, JevAnswers
from noeira.core.bytes import string_from_bytes
from noeira.core.concurrent.thread import sleep_us
from noeira.envs.robots.g1_spec import (
    g1_spec_questions, g1_spec_prompt, g1_add_miss_kind, g1_miss_movement_p, G1_Q_MISS_KIND, G1_MISS_MOVE_P,
)
from noeira.io.fileio import read_file_bytes
from noeira.llm.d1.service import D1Service


@fieldwise_init
struct Row(Copyable, Movable):
    var split: String
    var want: String
    var lang: String
    var text: String
    var slot0: String
    var slot0_p: Float64
    var kind: String
    var p_move: Float64
    var ms_with: Float64
    var ms_without: Float64


def ask(mut d1: D1Service, text: String, ref q: JevQuestions) raises -> JevAnswers:
    d1.start_text(g1_spec_prompt(text), q)
    while not d1.poll():
        _ = sleep_us(1000)
    return d1.result()


def count(ref rows: List[Row], split: String, want: String, rule: Int, t: Float64) -> Tuple[Int, Int]:
    """(routed right, rows) for one rule: 0 = slots, 1 = slot 0 >= t, 2 = kind."""
    var ok = 0
    var n = 0
    for r in rows:
        if r.split != split or r.want != want:
            continue
        n += 1
        var moves = r.slot0 != "none"
        if rule == 1:
            moves = r.slot0 != "none" and r.slot0_p >= t
        elif rule == 2:
            moves = r.p_move >= G1_MISS_MOVE_P
        if moves == (r.want == "movement"):
            ok += 1
    return (ok, n)


def main() raises:
    var args = argv()
    var out_path = String(args[1]) if len(args) > 1 else String("")
    var set_path = String(args[2]) if len(args) > 2 else String("tests/brain/miss_route.tsv")
    var spec = String(args[3]) if len(args) > 3 else String("native-text")
    var rows = List[Row]()
    for l in string_from_bytes(read_file_bytes(set_path)).split("\n"):
        var line = String(l)
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var c = List[String]()
        for x in line.split("\t"):
            c.append(String(x))
        if len(c) != 4:
            raise Error("miss_route: a row with " + String(len(c)) + " columns")
        rows.append(Row(c[0], c[1], c[2], c[3], String(""), 0.0, String(""), 0.0, 0.0, 0.0))
    print("miss route:", len(rows), "sentences; d1", spec)

    var q_plain = g1_spec_questions(True)
    var q_kind = g1_spec_questions(True)
    g1_add_miss_kind(q_kind, True)
    var d1 = D1Service.from_spec(spec)
    _ = ask(d1, String("Hello."), q_kind)  # loads, warms
    _ = ask(d1, String("Hello."), q_plain)
    for i in range(len(rows)):
        var t0 = perf_counter_ns()
        var a = ask(d1, rows[i].text, q_kind)
        rows[i].ms_with = Float64(perf_counter_ns() - t0) / 1e6
        rows[i].slot0 = a.choice(String("spec_q0"))
        rows[i].slot0_p = a.probability(String("spec_q0"), rows[i].slot0)
        rows[i].kind = a.choice(String(G1_Q_MISS_KIND))
        rows[i].p_move = g1_miss_movement_p(a)
        var t1 = perf_counter_ns()
        var b = ask(d1, rows[i].text, q_plain)
        rows[i].ms_without = Float64(perf_counter_ns() - t1) / 1e6
        if b.choice(String("spec_q0")) != rows[i].slot0:
            print("  !! slot 0 differs with the kind question asked:", rows[i].text)

    # the best slot-0 threshold on the tune rows
    var best_t = 0.0
    var best_ok = -1
    for k in range(1, 20):
        var t = Float64(k) * 0.05
        var ok = count(rows, String("tune"), String("movement"), 1, t)[0] + count(rows, String("tune"), String("reply"), 1, t)[0]
        if ok > best_ok:
            best_ok = ok
            best_t = t
    var splits: List[String] = ["tune", "held"]
    var names: List[String] = ["slots (before)", "slot 0 >= " + String(best_t), "kind question"]
    for s in splits:
        print(s + ":")
        for rule in range(3):
            var m = count(rows, s, String("movement"), rule, best_t)
            var r = count(rows, s, String("reply"), rule, best_t)
            print("   ", names[rule], "| movement", m[0], "/", m[1], "| reply", r[0], "/", r[1], "| all", m[0] + r[0], "/", m[1] + r[1])
    var w = 0.0
    var wo = 0.0
    for r in rows:
        w += r.ms_with
        wo += r.ms_without
    print("request, mean:", Int(wo / Float64(len(rows))), "ms without the kind question,", Int(w / Float64(len(rows))), "ms with")
    print("wrong under the kind question:")
    for r in rows:
        if (r.p_move >= G1_MISS_MOVE_P) != (r.want == "movement"):
            print("   ", r.split, r.want, "|", r.text, "->", r.kind, "P(movement)", round(r.p_move, 2), "| slot 0", r.slot0, round(r.slot0_p, 2))
    if out_path != "":
        var tsv = String("split\twant\tlang\tslot0\tslot0_p\tkind\tp_movement\tms_with\tms_without\ttext\n")
        for r in rows:
            tsv += (r.split + "\t" + r.want + "\t" + r.lang + "\t" + r.slot0 + "\t" + String(round(r.slot0_p, 3)) + "\t" + r.kind
                    + "\t" + String(round(r.p_move, 3)) + "\t" + String(Int(r.ms_with)) + "\t" + String(Int(r.ms_without)) + "\t" + r.text + "\n")
        with open(out_path, "w") as f:
            f.write(tsv)
        print("answers written to", out_path)
