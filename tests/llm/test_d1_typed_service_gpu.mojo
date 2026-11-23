"""d1's typed answers through the API the brain uses: `JevQuestions`' builders
in, `D1Service` (d1 on its own thread behind `DecisionService`), `JevAnswers`'
accessors out (docs/NN_MODEL_PORTS_PLAN.md §12.9, T7).

    build on the Orin: build/jetson_build.sh t tests/llm/test_d1_typed_service_gpu.mojo

The typed reference's cases (`tools/d1/dump_d1_typed_reference.py --forward`,
`~/.cache/noeira/d1/typed_ref`): reads over a JSON room state, estimates,
multiple choices on a text state, a box and five `boxes` over a picture.

1. BUILDERS: every reference question is built with the builder of its type
   (`choices`, `read`, `estimate`, `box`, `boxes`, and the untyped three) from
   its own fields, and the JSON the builders write equals the reference's
   question, field for field.
2. SERVICE: each case asked through `D1Service` (`start`, `start_text`,
   `start_seeing` with the reference's own pixels), polled without blocking.
3. ACCESSORS: every answer read through `JevAnswers` equals d1's own —
   `choices` and `probability`, `found`, `has_value`, `text`, `number`,
   `date`, `unit`, `span`, `interval`, `box`, `boxes`, `count`, `at_least` —
   discrete values exactly, probabilities within the bound.
"""

from std.os import getenv
from std.time import perf_counter_ns

from noeira.ai.jev import JevAnswers, JevQuestions, JevBox
from noeira.ai.picture import Picture
from noeira.ai.local_service import SVC_LOADING, SVC_READY
from noeira.core.concurrent.thread import sleep_us
from noeira.io.json import JsonDoc, JsonWriter, load_json, parse_json, J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT
from noeira.io.npy import load_npy_f64, npy_header_of
from noeira.llm.d1.prompt import py_json_dumps
from noeira.llm.d1.service import D1Service
from noeira.llm.lfm3.model import LF_BF16


comptime PROB_BOUND: Float64 = 3e-3 if LF_BF16 else 1e-4
comptime CASES: List[String] = ["reads_room", "reads_text", "estimates", "multiple_text", "box_one", "boxes"]


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


def same_json(ref a: JsonDoc, na: Int, ref b: JsonDoc, nb: Int) raises -> Bool:
    """Equal JSON values: objects as key sets, numbers by value."""
    var ka = a.kind_of(na)
    if ka != b.kind_of(nb):
        return False
    if ka == J_NULL:
        return True
    if ka == J_BOOL:
        return a.boolean(na) == b.boolean(nb)
    if ka == J_NUMBER:
        return a.number(na) == b.number(nb)
    if ka == J_STRING:
        return a.string(na) == b.string(nb)
    if a.size(na) != b.size(nb):
        return False
    for i in range(a.size(na)):
        if ka == J_ARRAY:
            if not same_json(a, a.at(na, i), b, b.at(nb, i)):
                return False
        else:
            var k = a.key_at(na, i)
            var nb2 = b.field(nb, k)
            if nb2 < 0 or not same_json(a, a.at(na, i), b, nb2):
                return False
    return True


def build(ref doc: JsonDoc, qn: Int) raises -> JevQuestions:
    """The reference's questions, each through the builder of its type."""
    var q = JevQuestions()
    for i in range(doc.size(qn)):
        var id = doc.key_at(qn, i)
        var e = doc.at(qn, i)
        var kind = doc.string(doc.field(e, "type"))
        var instr = doc.string(doc.field(e, "instructions"))
        var crit = doc.field(e, "criteria")
        var un = doc.field(e, "unit")
        var unit = doc.string(un) if un >= 0 else String("")
        if kind == "noul":
            q.noul(id, instr)
        elif kind == "score":
            var levels = List[String]()
            for k in range(doc.size(crit)):
                levels.append(doc.string(doc.at(crit, k)))
            q.score(id, instr, levels)
        elif kind == "choice":
            var opts = List[String]()
            var descs = List[String]()
            for k in range(doc.size(crit)):
                opts.append(doc.key_at(crit, k))
                descs.append(doc.string(doc.at(crit, k)))
            var mult = doc.field(e, "multiple")
            if mult >= 0 and doc.boolean(mult):
                q.choices(id, instr, opts, descs)
            else:
                q.choice(id, instr, opts, descs)
        elif kind == "number" and doc.field(e, "range") >= 0:
            var r = doc.field(e, "range")
            var st = doc.field(e, "step")
            q.estimate(id, instr, doc.number(doc.at(r, 0)), doc.number(doc.at(r, 1)),
                       doc.number(st) if st >= 0 else 0.0, unit)
        elif kind == "text" or kind == "number" or kind == "date":
            q.read(id, instr, kind, unit)
        elif kind == "box":
            q.box(id, instr)
        elif kind == "boxes":
            var mx = doc.field(e, "max")
            q.boxes(id, instr, doc.integer(mx) if mx >= 0 else 0)
        else:
            raise Error("no builder for " + kind)
    return q^


def main() raises:
    var fails = 0
    var home = getenv("HOME")
    var ref_dir = home + "/.cache/noeira/d1/typed_ref"
    print("d1 typed answers through JevQuestions / D1Service / JevAnswers", "(LFM3_BF16)" if LF_BF16 else "(fp32)")
    var doc = load_json(ref_dir + "/cases.json")
    var cases = doc.field(doc.root(), "cases")

    var t0 = perf_counter_ns()
    var dec = D1Service.from_spec(String("native"))
    while dec.state() == SVC_LOADING and perf_counter_ns() - t0 < 300_000_000_000:
        _ = sleep_us(10_000)
    check(fails, dec.state() == SVC_READY, String("READY after ") + String((perf_counter_ns() - t0) // 1_000_000) + " ms")

    var n_built = 0
    var n_acc = 0
    for name in materialize[CASES]():
        var c = doc.field(cases, name)
        var qn = doc.field(c, "questions")
        var fw = doc.field(c, "forward")

        # 1. builders
        var q = build(doc, qn)
        var w = JsonWriter()
        q.write_to_json(w)
        var bd = parse_json(_bytes(w.done()))
        var built_ok = same_json(bd, bd.root(), doc, qn)
        n_built += doc.size(qn)
        check(fails, built_ok, name + ": " + String(doc.size(qn)) + " questions built equal the reference's JSON")

        # 2. the service
        var is_json = doc.boolean(doc.field(c, "state_is_json"))
        var snode = doc.field(c, "state")
        var state = py_json_dumps(doc, snode) if is_json else doc.string(snode)
        var pics = List[Picture]()
        for k in range(doc.integer(doc.field(c, "pictures"))):
            var path = ref_dir + "/" + name + "_image" + String(k) + ".npy"
            var hd = npy_header_of(path)
            var px = load_npy_f64(path)
            var rgb = List[UInt8](capacity=len(px))
            for v in px:
                rgb.append(UInt8(Int(v)))
            pics.append(Picture(rgb^, hd.rows, hd.cols // 3))
        var t1 = perf_counter_ns()
        if len(pics) > 0:
            dec.start_seeing(state, is_json, q, pics)
        elif is_json:
            dec.start(state, q)
        else:
            dec.start_text(state, q)
        while not dec.poll():
            _ = sleep_us(2_000)
        var a = dec.result()
        var ms = (perf_counter_ns() - t1) // 1_000_000

        # 3. accessors against d1's answers
        var wa = doc.field(fw, "answers")
        var ok = True
        var worst = 0.0
        for i in range(doc.size(wa)):
            var id = doc.key_at(wa, i)
            var e = doc.at(wa, i)
            var kind = doc.string(doc.field(e, "type"))
            n_acc += 1
            if kind == "noul":
                worst = max(worst, abs(a.noul(id) - doc.number(doc.field(e, "noul"))))
            elif kind == "choice" and doc.field(e, "choices") >= 0:
                var got = a.choices(id)
                var wc = doc.field(e, "choices")
                ok = ok and len(got) == doc.size(wc)
                for k in range(min(len(got), doc.size(wc))):
                    ok = ok and got[k] == doc.string(doc.at(wc, k))
                var wp = doc.field(e, "probabilities")
                for k in range(doc.size(wp)):
                    worst = max(worst, abs(a.probability(id, doc.key_at(wp, k)) - doc.number(doc.at(wp, k))))
            elif kind == "choice":
                ok = ok and a.choice(id) == doc.string(doc.field(e, "choice"))
                worst = max(worst, abs(a.confidence(id) - doc.number(doc.field(e, "confidence"))))
            elif kind == "number" and doc.field(e, "interval") >= 0:
                var iv = a.interval(id)
                var wi = doc.field(e, "interval")
                ok = ok and a.number(id) == doc.number(doc.field(e, "number"))
                ok = ok and iv[0] == doc.number(doc.at(wi, 0)) and iv[1] == doc.number(doc.at(wi, 1))
                var wu = doc.field(e, "unit")
                ok = ok and a.unit(id) == (doc.string(wu) if doc.kind_of(wu) == J_STRING else String(""))
            elif kind == "text" or kind == "number" or kind == "date":
                worst = max(worst, abs(a.found(id) - doc.number(doc.field(e, "found"))))
                var wv = doc.field(e, kind)
                ok = ok and a.has_value(id) == (doc.kind_of(wv) != J_NULL)
                var ws = doc.field(e, "span")
                var sp = a.span(id)
                if doc.kind_of(ws) == J_NULL:
                    ok = ok and sp[0] == -1
                else:
                    ok = ok and sp[0] == doc.integer(doc.at(ws, 0)) and sp[1] == doc.integer(doc.at(ws, 1))
                var wt = doc.field(e, "text")
                ok = ok and a.text(id) == (doc.string(wt) if doc.kind_of(wt) == J_STRING else String(""))
                if kind == "number":
                    if doc.kind_of(wv) != J_NULL:
                        ok = ok and a.number(id) == doc.number(wv)
                    var wu = doc.field(e, "unit")
                    ok = ok and a.unit(id) == (doc.string(wu) if doc.kind_of(wu) == J_STRING else String(""))
                if kind == "date":
                    ok = ok and a.date(id) == (doc.string(wv) if doc.kind_of(wv) == J_STRING else String(""))
            elif kind == "box":
                var b = a.box(id)
                var wb = doc.field(e, "box")
                worst = max(worst, abs(b.found - doc.number(doc.field(e, "found"))))
                ok = ok and b.drawn == (doc.kind_of(wb) != J_NULL)
                if b.drawn:
                    ok = ok and b.top == doc.number(doc.field(wb, "top")) and b.bottom == doc.number(doc.field(wb, "bottom"))
                    ok = ok and b.left == doc.number(doc.field(wb, "left")) and b.right == doc.number(doc.field(wb, "right"))
                    ok = ok and b.confidence == doc.number(doc.field(e, "confidence"))
            elif kind == "boxes":
                var bs = a.boxes(id)
                var wl = doc.field(e, "boxes")
                ok = ok and len(bs) == doc.size(wl) and a.count(id) == doc.integer(doc.field(e, "count"))
                ok = ok and a.at_least(id) == doc.boolean(doc.field(e, "at_least"))
                for k in range(min(len(bs), doc.size(wl))):
                    var we = doc.at(wl, k)
                    var wb = doc.field(we, "box")
                    ok = ok and bs[k].top == doc.number(doc.field(wb, "top")) and bs[k].left == doc.number(doc.field(wb, "left"))
                    ok = ok and bs[k].bottom == doc.number(doc.field(wb, "bottom")) and bs[k].right == doc.number(doc.field(wb, "right"))
                    ok = ok and bs[k].confidence == doc.number(doc.field(we, "confidence"))
                    worst = max(worst, abs(bs[k].found - doc.number(doc.field(we, "found"))))
            else:
                ok = False
                print("    ", name, "/", id, ": no accessor check for", kind)
        check(fails, ok and worst < PROB_BOUND, name + ": " + String(doc.size(wa)) + " answers read through JevAnswers "
              + "equal d1's (worst probability |d| = " + String(worst) + ", " + String(ms) + " ms through the service)")
    check(fails, n_built >= 25 and n_acc >= 25, String(n_built) + " questions built, " + String(n_acc) + " answers read")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
