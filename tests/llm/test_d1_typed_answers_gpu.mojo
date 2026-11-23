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

T4 reads `text`, `number` and `date` spans: each read's two slots against the
row's state positions (`pointer.mojo`); its raw found within the bound, its
text, span and JSON path EXACTLY d1's, and the answer's (found through
`found:<type>`). T6: a number's or a date's VALUE (and a number's unit),
parsed from that text by d1's rules, EXACTLY d1's.
Mutants caught: the pointer without 1/sqrt(D), found read off position 1
instead of the BOS, invalid positions scored, whitespace kept at a span's
edge, a span not clipped to its JSON string (`label`: its first token " \"("
starts on the quote). Not gated: the span caps (512 / 32 tokens; no reference
read comes near 32).

T5 reads `box` and `boxes` over a picture (the reference's own pixels,
`<case>_image0.npy`): each slot raw within the bound (presence, and every
edge's 1,000-value distribution), every drawn box's edges EXACTLY d1's, found
and presence within the bound, confidence within CONF_BOUND, the count and
`at_least` equal. The case holds a box, boxes of 16 / 4 / 1 blocks drawing
none or one, and two drawing several ("Every picture on the wall." 3, "Every
book." 6 of max 6, so `at_least`). Mutants caught: no reading-order sort,
presence not kept non-increasing, the `found:box` and `box` tables skipped,
the Halton draws in one base. Not gated: the areas' products fused into the
IoU's sum (it moves an IoU's last bit, which matters only at exactly 0.5).

`--w8a8`: `D1.load(w8a8=True)`: the same picks, probabilities within W8A8_BOUND,
estimates within W8A8_GRID_SHARE of their range.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.json import load_json, parse_json, JsonDoc, J_NUMBER, J_ARRAY, J_OBJECT, J_NULL
from noeira.llm.d1 import D1, parse_questions, Q_MULTI, Q_ESTIMATE, Q_READ
from noeira.llm.d1.prompt import py_json_dumps
from noeira.llm.lfm3 import Picture
from noeira.io.npy import load_npy_f64, npy_header_of
from noeira.llm.lfm3.model import LF_BF16


comptime PROB_BOUND: Float64 = 3e-3 if LF_BF16 else 1e-4
comptime CONF_BOUND: Float64 = 0.02
"""A box's confidence counts 512 Halton draws through its edges' CDFs: one
draw crossing IoU 0.5 under fp32 noise moves it by 1/512 before the `box`
table (slopes up to ~1.3). Measured in the result line."""

comptime W8A8_BOUND: Float64 = 0.1
comptime W8A8_CONF: Float64 = 0.2
"""`--w8a8`: a box's confidence within this of d1's fp32 one. Measured 0.096
(the books, 2026-11-23): W8A8 moves the edge CDFs the 512 draws go through."""
comptime W8A8_EDGE: Float64 = 0.05
"""`--w8a8`: a drawn box's largest edge difference with d1's fp32 box, at
most, as a share of the picture (fp32: the same edges exactly). Edges, not
IoU: a sliver box (one book drawn 0.242-0.249 tall) shares nothing with its
W8A8 self (0.275-0.276) after a 0.03 shift. A `boxes` answer is matched as a
set — each of d1's boxes against our nearest — since neighbours can swap
reading order. Measured 0.033 (that book; the others <= 0.006), the Orin
2026-11-23; in the result line."""
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


def _f5(x: Float64) -> String:
    var t = String(x)
    return String(t[byte=0:min(5, t.byte_length())])


def rect_iou(ref a: JsonDoc, na: Int, ref b: JsonDoc, nb: Int) raises -> Float64:
    var at = a.number(a.field(na, "top"))
    var ab = a.number(a.field(na, "bottom"))
    var al = a.number(a.field(na, "left"))
    var ar = a.number(a.field(na, "right"))
    var bt = b.number(b.field(nb, "top"))
    var bb = b.number(b.field(nb, "bottom"))
    var bl = b.number(b.field(nb, "left"))
    var br = b.number(b.field(nb, "right"))
    var ov = max(0.0, min(ab, bb) - max(at, bt)) * max(0.0, min(ar, br) - max(al, bl))
    return ov / ((ab - at) * (ar - al) + (bb - bt) * (br - bl) - ov)


def edge_gap(ref a: JsonDoc, na: Int, ref b: JsonDoc, nb: Int) raises -> Float64:
    var keys: List[String] = ["top", "bottom", "left", "right"]
    var g = 0.0
    for k in keys:
        g = max(g, abs(a.number(a.field(na, k)) - b.number(b.field(nb, k))))
    return g


def same_rect(ref a: JsonDoc, na: Int, ref b: JsonDoc, nb: Int, w8a8: Bool, mut worst_edge: Float64) raises -> Bool:
    """Both null, or the same four edges exactly (a median bin, or a crossed
    pair's middle); under `--w8a8` edges within W8A8_EDGE."""
    if a.kind_of(na) == J_NULL or b.kind_of(nb) == J_NULL:
        return a.kind_of(na) == J_NULL and b.kind_of(nb) == J_NULL
    var gap = edge_gap(a, na, b, nb)
    worst_edge = max(worst_edge, gap)
    if w8a8:
        return gap <= W8A8_EDGE
    var keys: List[String] = ["top", "bottom", "left", "right"]
    for k in keys:
        if a.number(a.field(na, k)) != b.number(b.field(nb, k)):
            return False
    return True


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
    var d1 = D1.load(DeviceContext(), model_dir, vision=True, w8a8=w8a8)
    print("  d1 loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var asked = 0
    var multi_asked = 0
    var estimates_asked = 0
    var reads_asked = 0
    var reads_found = 0
    var values_checked = 0
    var boxes_asked = 0
    var boxes_drawn = 0
    for ci in range(doc.size(cases)):
        var name = doc.key_at(cases, ci)
        var c = doc.at(cases, ci)
        var fw = doc.field(c, "forward")
        if fw < 0 or doc.kind_of(fw) == J_NULL:
            print("    (", name, ": no forward in the reference — skipped )")
            continue
        var pics = List[Picture]()
        for k in range(doc.integer(doc.field(c, "pictures"))):
            var path = ref_dir + "/" + name + "_image" + String(k) + ".npy"
            var hd = npy_header_of(path)
            var px = load_npy_f64(path)
            var rgb = List[UInt8](capacity=len(px))
            for v in px:
                rgb.append(UInt8(Int(v)))
            pics.append(Picture(rgb^, hd.rows, hd.cols // 3))
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
            got_s = d1.system_one(state, is_json, qjson, pics)
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
            if qs[qi].kind == Q_READ:
                reads_asked += 1
                ref rd = d1.last_reads[qi]
                var wt = doc.field(r, "text")
                var ws = doc.field(r, "span")
                var wpth = doc.field(r, "path")
                worst_raw = max(worst_raw, abs(rd.found - doc.number(doc.field(r, "found"))))
                var ok = (doc.kind_of(wt) == J_NULL) == (not rd.has_text)
                if ok and rd.has_text:
                    reads_found += 1
                    ok = (rd.text == doc.string(wt) and rd.start == doc.integer(doc.at(ws, 0))
                          and rd.end == doc.integer(doc.at(ws, 1)))
                ok = ok and ((wpth >= 0) == rd.has_path) and (wpth < 0 or doc.string(wpth) == rd.path)
                if not ok:
                    shape_ok = False
                    print("    ", name, "/", qs[qi].name, ": read", repr(rd.text), rd.start, rd.end, rd.path,
                          "vs d1", repr(doc.string(wt)) if doc.kind_of(wt) != J_NULL else String("None"))
            elif qs[qi].kind == Q_MULTI:
                multi_asked += 1
                shape_ok = shape_ok and len(mine) == doc.size(r)
                for k in range(min(len(mine), doc.size(r))):
                    worst_raw = max(worst_raw, abs(mine[k][0] - doc.number(doc.at(r, k))))
            elif doc.kind_of(r) == J_ARRAY and doc.size(r) > 0 and doc.kind_of(doc.at(r, 0)) == J_ARRAY:
                boxes_asked += 1  # a box / boxes: per slot, presence [yes, no] then each edge's distribution
                shape_ok = shape_ok and len(mine) == doc.size(r)
                for k in range(min(len(mine), doc.size(r))):
                    var rk = doc.at(r, k)
                    shape_ok = shape_ok and len(mine[k]) == doc.size(rk)
                    for j in range(min(len(mine[k]), doc.size(rk))):
                        worst_raw = max(worst_raw, abs(mine[k][j] - doc.number(doc.at(rk, j))))
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
        var worst_conf = 0.0
        var worst_edge = 0.0
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
            if kind == "box":
                worst = max(worst, abs(got.number(got.field(g, "found")) - doc.number(doc.field(w, "found"))))
                worst_conf = max(worst_conf, abs(got.number(got.field(g, "confidence")) - doc.number(doc.field(w, "confidence"))))
                var wb = doc.field(w, "box")
                var gb = got.field(g, "box")
                if (doc.kind_of(wb) == J_NULL) != (got.kind_of(gb) == J_NULL) or not same_rect(doc, wb, got, gb, w8a8, worst_edge):
                    same = False
                    print("    ", name, "/", qn, ": box differs")
                elif doc.kind_of(wb) != J_NULL:
                    boxes_drawn += 1
                continue
            if kind == "boxes":
                var wl = doc.field(w, "boxes")
                var gl = got.field(g, "boxes")
                var wpr = doc.field(w, "presence")
                var gpr = got.field(g, "presence")
                var ok = (got.size(gl) == doc.size(wl) and got.integer(got.field(g, "count")) == doc.integer(doc.field(w, "count"))
                          and got.boolean(got.field(g, "at_least")) == doc.boolean(doc.field(w, "at_least"))
                          and got.size(gpr) == doc.size(wpr))
                if ok:
                    for k in range(doc.size(wpr)):
                        worst = max(worst, abs(got.number(got.at(gpr, k)) - doc.number(doc.at(wpr, k))))
                    for k in range(doc.size(wl)):
                        var wk = doc.at(wl, k)
                        var gk = got.at(gl, k)
                        if w8a8:  # neighbours can swap reading order under W8A8: d1's box against our nearest
                            var best = 2.0
                            for j in range(got.size(gl)):
                                var gap_j = edge_gap(doc, doc.field(wk, "box"), got, got.field(got.at(gl, j), "box"))
                                if gap_j < best:
                                    best = gap_j
                                    gk = got.at(gl, j)
                        worst = max(worst, abs(got.number(got.field(gk, "found")) - doc.number(doc.field(wk, "found"))))
                        worst_conf = max(worst_conf, abs(got.number(got.field(gk, "confidence"))
                                                         - doc.number(doc.field(wk, "confidence"))))
                        ok = ok and same_rect(doc, doc.field(wk, "box"), got, got.field(gk, "box"), w8a8, worst_edge)
                        boxes_drawn += 1
                if not ok:
                    same = False
                    print("    ", name, "/", qn, ": boxes differ (", got.size(gl), "vs", doc.size(wl), ")")
                    for k in range(max(got.size(gl), doc.size(wl))):
                        var line = String("       ")
                        for side in range(2):
                            var ok_side = k < (doc.size(wl) if side == 0 else got.size(gl))
                            if not ok_side:
                                line += " -                              "
                                continue
                            var bx = doc.field(doc.at(wl, k), "box") if side == 0 else got.field(got.at(gl, k), "box")
                            ref dd = doc if side == 0 else got
                            line += (String(" d1 " if side == 0 else "  mojo ") + _f5(dd.number(dd.field(bx, "top"))) + " "
                                     + _f5(dd.number(dd.field(bx, "bottom"))) + " " + _f5(dd.number(dd.field(bx, "left")))
                                     + " " + _f5(dd.number(dd.field(bx, "right"))))
                        print(line)
                continue
            if (kind == "text" or kind == "number" or kind == "date") and doc.field(w, "found") >= 0:  # a read
                worst = max(worst, abs(got.number(got.field(g, "found")) - doc.number(doc.field(w, "found"))))
                var tkey = String("text")
                var tw = doc.field(w, tkey)
                var tg = got.field(g, tkey)
                var sw = doc.field(w, "span")
                var sg = got.field(g, "span")
                var pw = doc.field(w, "path")
                var pg = got.field(g, "path")
                var ok = (doc.kind_of(tw) == J_NULL) == (got.kind_of(tg) == J_NULL)
                if ok and doc.kind_of(tw) != J_NULL:
                    ok = (doc.string(tw) == got.string(tg) and got.kind_of(sg) != J_NULL
                          and doc.integer(doc.at(sw, 0)) == got.integer(got.at(sg, 0))
                          and doc.integer(doc.at(sw, 1)) == got.integer(got.at(sg, 1)))
                ok = ok and ((pw >= 0) == (pg >= 0)) and (pw < 0 or doc.string(pw) == got.string(pg))
                if kind != "text":  # the value parsed from the text (T6), and a number's unit
                    values_checked += 1
                    var vw = doc.field(w, kind)
                    var vg = got.field(g, kind)
                    var v_ok = (doc.kind_of(vw) == J_NULL) == (got.kind_of(vg) == J_NULL)
                    if v_ok and doc.kind_of(vw) != J_NULL:
                        v_ok = (doc.number(vw) == got.number(vg)) if kind == "number" else (doc.string(vw) == got.string(vg))
                    if kind == "number":
                        var uw = doc.field(w, "unit")
                        var ug = got.field(g, "unit")
                        v_ok = v_ok and ((doc.kind_of(uw) == J_NULL) == (got.kind_of(ug) == J_NULL))
                        if v_ok and doc.kind_of(uw) != J_NULL:
                            v_ok = doc.string(uw) == got.string(ug)
                    if not v_ok:
                        ok = False
                        print("    ", name, "/", qn, ": parsed value or unit differs")
                if not ok:
                    same = False
                    print("    ", name, "/", qn, ": answer's read differs")
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
        check(fails, same and worst < bound and worst_conf < (W8A8_CONF if w8a8 else CONF_BOUND) and wu == gu,
              name + ": " + String(doc.size(wa)) + " answers, same picks, worst probability |d| = " + String(worst)
              + ", " + String(gu) + " tokens (" + String(ms) + " ms)"
              + (", estimates off by " + String(worst_share) + " of their range" if worst_share > 0.0 else "")
              + (", box confidence |d| = " + String(worst_conf) if worst_conf > 0.0 else "")
              + (", box edges off by " + String(worst_edge) if worst_edge > 0.0 else ""))
    check(fails, asked >= 8 and multi_asked >= 5 and estimates_asked >= 6 and reads_asked >= 15 and reads_found >= 8 and values_checked >= 7
          and boxes_asked >= 4,
          String(asked) + " cases asked: " + String(multi_asked) + " multiple choices, " + String(estimates_asked)
          + " estimates, " + String(reads_asked) + " reads (" + String(reads_found) + " found something; "
          + String(values_checked) + " number / date values parsed), " + String(boxes_asked) + " box questions ("
          + String(boxes_drawn) + " boxes drawn)")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
