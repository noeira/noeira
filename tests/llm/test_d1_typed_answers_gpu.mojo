"""d1's typed answers in Mojo (`D1.system_one`) against d1-2B's own code
(docs/NN_MODEL_PORTS_PLAN.md §12.9, T2 onwards).

    ./test_d1_typed_answers_gpu [--w8a8] [d1 dir] [ref dir]
      defaults: ~/.cache/noeira/local-ai/d1/d1-2B and ~/.cache/noeira/d1/typed_ref

The reference is `tools/d1/dump_d1_typed_reference.py --forward` (fp32, the
Orin): per case each question's distributions as d1's `run` reads them, and
the answers made from them. Every case whose questions are all read by now is
asked here (the others are named and skipped; picture cases wait for T5):

1. RAW: each slot's distribution (`D1.last_probs`) — an untyped question's
   tempered softmax, a typed slot's raw one — within the bound of d1's.
2. ANSWERS: the same picks (a choice, a score level, a multiple choice's set
   of choices), every probability within the bound, the same input tokens.

T2 reads `multiple` choices: each option a yes/no slot, raw, then the
`multiple` table. The cases mix them with untyped questions on JSON and text
states, so a temperature applied to a typed slot (or left off an untyped one)
moves the raw distributions.

T3 reads estimates: one slot over the grid's values (the tokenizer's numbers
for an integer grid, codes otherwise: digits up to 10 values, option codes
above), raw; the answer's median, interval and unit EXACTLY d1's — the values
are the grid's, read at the `estimate` table's levels, which move four of the
case's answers against the plain 0.1 / 0.5 / 0.9. Mutants caught: the levels
not applied; the grid's values themselves are gated bit for bit on the host
(`test_d1_typed_prompt_reference`, item 9). A plain sum for `math.fsum` moves
nothing here (it only matters where a cumulative share sits exactly on a
level): not gated.

`--w8a8`: `D1.load(w8a8=True)`: the same picks, probabilities within W8A8_BOUND,
estimates within W8A8_GRID_SHARE of their range.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.json import load_json, parse_json, JsonDoc, J_NUMBER, J_ARRAY, J_OBJECT, J_NULL
from noeira.llm.d1 import D1, parse_questions, Q_MULTI, Q_ESTIMATE
from noeira.llm.d1.prompt import py_json_dumps
from noeira.llm.lfm3.model import LF_BF16


comptime PROB_BOUND: Float64 = 3e-3 if LF_BF16 else 1e-4
comptime W8A8_BOUND: Float64 = 0.1
comptime W8A8_GRID_SHARE: Float64 = 0.2
"""`--w8a8`: an estimate's median and interval within this share of its range
of d1's fp32 values (the fp32 build: exactly). Measured 0.13, 0.10 and 0.06 on
three versions of the case (the Orin, 2026-11-23: the row's other questions
change the context): `tenth`'s interval, 31 values over 0-3 with max p 0.35,
moved by 0.4 under W8A8's 0.058 in raw probabilities — its p90 is read at a
0.77 share, where a broad distribution's cumulative mass crosses several bins;
the flat 128-value grids (`free`, `big`, max p 0.03) moved 1-3 bins."""


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for x in s.as_bytes():
        b.append(x)
    return b^


def main() raises:
    var args = argv()
    var w8a8 = False
    var pos = List[String]()
    for i in range(1, len(args)):
        if String(args[i]) == "--w8a8":
            w8a8 = True
        else:
            pos.append(String(args[i]))
    var home = getenv("HOME")
    var model_dir = pos[0] if len(pos) > 0 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var ref_dir = pos[1] if len(pos) > 1 else home + "/.cache/noeira/d1/typed_ref"
    var bound = W8A8_BOUND if w8a8 else PROB_BOUND
    var fails = 0
    print("d1 typed answers in Mojo vs d1-2B's own code", "(LFM3_BF16)" if LF_BF16 else "(fp32)", "W8A8" if w8a8 else "")

    var doc = load_json(ref_dir + "/cases.json")
    var cases = doc.field(doc.root(), "cases")
    var t0 = perf_counter_ns()
    var d1 = D1.load(DeviceContext(), model_dir, w8a8=w8a8)
    print("  d1 loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var asked = 0
    var multi_asked = 0
    var estimates_asked = 0
    for ci in range(doc.size(cases)):
        var name = doc.key_at(cases, ci)
        var c = doc.at(cases, ci)
        var fw = doc.field(c, "forward")
        if fw < 0 or doc.kind_of(fw) == J_NULL:
            print("    (", name, ": no forward in the reference — skipped )")
            continue
        if doc.integer(doc.field(c, "pictures")) > 0:
            print("    (", name, ": pictures — T5 )")
            continue
        var is_json = doc.boolean(doc.field(c, "state_is_json"))
        var snode = doc.field(c, "state")
        var state = py_json_dumps(doc, snode) if is_json else doc.string(snode)
        var qnode = doc.field(c, "questions")
        var qjson = py_json_dumps(doc, qnode)
        var qs = parse_questions(doc, qnode, d1.record)
        var t1 = perf_counter_ns()
        var got_s = String("")
        var refused = String("")
        try:
            got_s = d1.system_one(state, is_json, qjson)
        except e:
            refused = String(e)
        if refused.byte_length() > 0:
            if "not yet read" in refused:
                print("    (", name, ": refused, not yet read —", qs[0].name, "... )")
                continue
            raise Error(refused)
        var ms = Float64(perf_counter_ns() - t1) / 1e6
        asked += 1

        # 1. raw distributions
        var raw = doc.field(fw, "raw")
        var worst_raw = 0.0
        var shape_ok = True
        for qi in range(len(qs)):
            var r = doc.field(raw, qs[qi].name)
            ref mine = d1.last_probs[qi]
            if qs[qi].kind == Q_ESTIMATE:
                estimates_asked += 1
            var before = worst_raw
            worst_raw = 0.0
            if qs[qi].kind == Q_MULTI:
                multi_asked += 1
                shape_ok = shape_ok and len(mine) == doc.size(r)
                for k in range(min(len(mine), doc.size(r))):
                    worst_raw = max(worst_raw, abs(mine[k][0] - doc.number(doc.at(r, k))))
            elif doc.kind_of(r) == J_ARRAY and doc.size(r) > 0 and doc.kind_of(doc.at(r, 0)) == J_NUMBER:
                shape_ok = shape_ok and len(mine) == 1 and len(mine[0]) == doc.size(r)
                for k in range(min(len(mine[0]), doc.size(r))):
                    worst_raw = max(worst_raw, abs(mine[0][k] - doc.number(doc.at(r, k))))
            else:
                shape_ok = False
                print("    ", name, "/", qs[qi].name, ": a raw read this gate does not compare yet")
            if worst_raw >= bound:
                var line = String("")
                for k in range(len(mine)):
                    line += String(mine[k][0])[byte=0:6] + " "
                print("    ", name, "/", qs[qi].name, ": raw |d| =", worst_raw, " mojo [", line, "]")
            worst_raw = max(worst_raw, before)

        # 2. answers
        var got = parse_json(_bytes(got_s))
        var ga = got.field(got.root(), "answers")
        var wa = doc.field(fw, "answers")
        var worst = 0.0
        var worst_share = 0.0
        var same = got.size(ga) == doc.size(wa)
        for i in range(doc.size(wa)):
            var qn = doc.key_at(wa, i)
            var w = doc.at(wa, i)
            var g = got.field(ga, qn)
            if g < 0:
                same = False
                continue
            var kind = doc.string(doc.field(w, "type"))
            if kind == "noul":
                worst = max(worst, abs(got.number(got.field(g, "noul")) - doc.number(doc.field(w, "noul"))))
                continue
            if kind == "number" and doc.field(w, "interval") >= 0:  # an estimate: values read off the grid
                var wi = doc.field(w, "interval")
                var gi = got.field(g, "interval")
                var wun = doc.field(w, "unit")
                var gun = got.field(g, "unit")
                var unit_ok = (doc.kind_of(wun) == J_NULL) == (got.kind_of(gun) == J_NULL)
                if unit_ok and doc.kind_of(wun) != J_NULL:
                    unit_ok = doc.string(wun) == got.string(gun)
                var span = 1.0
                for qq in qs:
                    if qq.name == qn:
                        span = qq.grid.high - qq.grid.low
                var share = 0.0
                if gi >= 0 and got.size(gi) == 2:
                    share = max(abs(got.number(got.field(g, "number")) - doc.number(doc.field(w, "number"))),
                                abs(got.number(got.at(gi, 0)) - doc.number(doc.at(wi, 0))),
                                abs(got.number(got.at(gi, 1)) - doc.number(doc.at(wi, 1)))) / span
                worst_share = max(worst_share, share)
                if gi < 0 or got.size(gi) != 2 or not unit_ok or share > (W8A8_GRID_SHARE if w8a8 else 0.0):
                    same = False
                    print("    ", name, "/", qn, ": estimate", got.number(got.field(g, "number")), "vs",
                          doc.number(doc.field(w, "number")), ", worst share of the range", share)
                continue
            var wc = doc.field(w, "choices")
            if wc >= 0:  # a multiple choice
                var gc = got.field(g, "choices")
                if gc < 0 or got.size(gc) != doc.size(wc):
                    same = False
                    print("    ", name, "/", qn, ": choices", got.size(gc) if gc >= 0 else -1, "vs", doc.size(wc))
                else:
                    for k in range(doc.size(wc)):
                        if got.string(got.at(gc, k)) != doc.string(doc.at(wc, k)):
                            same = False
            elif kind == "choice" and got.string(got.field(g, "choice")) != doc.string(doc.field(w, "choice")):
                same = False
            elif kind == "score" and got.number(got.field(g, "score")) != doc.number(doc.field(w, "score")):
                same = False
            var wp = doc.field(w, "probabilities")
            var gp = got.field(g, "probabilities")
            for k in range(doc.size(wp)):
                var gk = got.field(gp, doc.key_at(wp, k))
                if gk < 0:
                    same = False
                    continue
                worst = max(worst, abs(got.number(gk) - doc.number(doc.at(wp, k))))
        var wu = doc.integer(doc.field(doc.field(fw, "usage"), "input_tokens"))
        var gu = got.integer(got.field(got.field(got.root(), "usage"), "input_tokens"))
        check(fails, shape_ok and worst_raw < bound, name + ": raw slot distributions, worst |d| = " + String(worst_raw))
        check(fails, same and worst < bound and wu == gu,
              name + ": " + String(doc.size(wa)) + " answers, same picks, worst probability |d| = " + String(worst)
              + ", " + String(gu) + " tokens (" + String(ms) + " ms)"
              + (", estimates off by " + String(worst_share) + " of their range" if worst_share > 0.0 else ""))
    check(fails, asked >= 3 and multi_asked >= 4 and estimates_asked >= 5,
          String(asked) + " cases asked, " + String(multi_asked) + " multiple choices and " + String(estimates_asked)
          + " estimates among them")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
