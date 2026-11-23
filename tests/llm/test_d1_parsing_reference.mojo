"""d1's parser in Mojo (`noeira.llm.d1.parsing`, on `noeira.text.pyre`) against
d1-2B's own `parsing.py` and Python's `re` (docs/NN_MODEL_PORTS_PLAN.md §12.9,
T6). Host only, no model.

    pixi run mojo run -I . tests/llm/test_d1_parsing_reference.mojo

Needs `~/.cache/noeira/d1/parsing_ref.json` (`tools/d1/dump_d1_parsing_reference.py`,
the Orin's d1ref-venv).

1. PATTERNS: the module-level patterns the port builds equal d1's, as text.
2. ENGINE: every pattern (module-level and inline) on every probe text — each
   `re.finditer` match's group spans and `re.fullmatch`'s — equal `re`'s.
3. FOLD: `_fold` of every date / number text and context, and of every 13th
   code point of Unicode, equals d1's.
4. `written_unit`, `parse_number` (None, an int or a float, the float bit for
   bit) and `parse_date` (value and ambiguity) equal d1's on every case.
"""

from std.os import getenv
from std.time import perf_counter_ns

from noeira.io.json import JsonDoc, load_json, J_NULL, J_STRING
from noeira.text.pyre import PyRe, PyMatch
from noeira.llm.d1.parsing import D1Parser, cps, text_of, fold


def _cut(s: String, n: Int) -> String:
    """The first `n` characters (a byte slice could end inside one, or past the end)."""
    var out = String("")
    var k = 0
    for c in s.codepoints():
        if k == n:
            break
        out += String(c)
        k += 1
    return out^


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok:", what)
    else:
        print("  FAIL:", what)
        fails += 1


def main() raises:
    var home = getenv("HOME")
    var doc = load_json(home + "/.cache/noeira/d1/parsing_ref.json")
    var root = doc.root()
    var fails = 0
    var t0 = perf_counter_ns()
    var P = D1Parser()
    print("d1's parser in Mojo vs parsing.py: built in", (perf_counter_ns() - t0) // 1_000_000, "ms")

    # 1. patterns
    var pn = doc.field(root, "patterns")
    var p_ok = 0
    for i in range(doc.size(pn)):
        var name = doc.key_at(pn, i)
        var mine = P.p.get(name)
        if mine and mine.value() == doc.string(doc.at(pn, i)):
            p_ok += 1
        else:
            print("    pattern", name, "differs")
            if mine:
                print("      mojo:", _cut(mine.value(), 200))
            print("      d1  :", _cut(doc.string(doc.at(pn, i)), 200))
    check(fails, p_ok == doc.size(pn), String(p_ok) + " / " + String(doc.size(pn)) + " module patterns equal d1's")

    # 2. the engine
    var texts_n = doc.field(root, "probe_texts")
    var texts = List[List[UInt32]]()
    for i in range(doc.size(texts_n)):
        texts.append(cps(doc.string(doc.at(texts_n, i))))
    var probes = doc.field(root, "probes")
    var e_ok = 0
    var e_n = 0
    var t1 = perf_counter_ns()
    for i in range(doc.size(probes)):
        var name = doc.key_at(probes, i)
        var pr = doc.at(probes, i)
        var re = PyRe.compile(doc.string(doc.field(pr, "pattern")), doc.boolean(doc.field(pr, "ignorecase")))
        var rows = doc.field(pr, "rows")
        var bad = 0
        for k in range(doc.size(rows)):
            var row = doc.at(rows, k)
            var want = doc.field(row, "finditer")
            var got = re.finditer(texts[k])
            var good = len(got) == doc.size(want)
            for j in range(min(len(got), doc.size(want))):
                var w = doc.at(want, j)
                for g in range(doc.size(w)):
                    var sp = doc.at(w, g)
                    good = good and got[j].spans[2 * g] == doc.integer(doc.at(sp, 0)) and got[j].spans[2 * g + 1] == doc.integer(doc.at(sp, 1))
            var wf = doc.field(row, "fullmatch")
            var gf = re.fullmatch(texts[k])
            if doc.kind_of(wf) == J_NULL:
                good = good and not gf
            else:
                good = good and Bool(gf)
                if gf:
                    for g in range(doc.size(wf)):
                        var sp = doc.at(wf, g)
                        good = good and gf.value().spans[2 * g] == doc.integer(doc.at(sp, 0)) and gf.value().spans[2 * g + 1] == doc.integer(doc.at(sp, 1))
            e_n += 1
            if good:
                e_ok += 1
            else:
                bad += 1
                if bad <= 2:
                    print("    pattern", name, "on", _cut(repr(doc.string(doc.at(texts_n, k))), 80), ":",
                          len(got), "matches vs", doc.size(want))
    check(fails, e_ok == e_n, String(e_ok) + " / " + String(e_n) + " (pattern, text) pairs match as re does ("
          + String(doc.size(probes)) + " patterns, " + String((perf_counter_ns() - t1) // 1_000_000) + " ms)")

    # 3. fold
    var fo = doc.field(root, "folds")
    var f_ok = 0
    for i in range(doc.size(fo)):
        var e = doc.at(fo, i)
        if text_of(fold(cps(doc.string(doc.field(e, "text"))))) == doc.string(doc.field(e, "folded")):
            f_ok += 1
        else:
            print("    fold of", _cut(repr(doc.string(doc.field(e, "text"))), 60), "differs")
    check(fails, f_ok == doc.size(fo), String(f_ok) + " / " + String(doc.size(fo)) + " folds equal d1's")

    # 4. units, numbers, dates
    var un = doc.field(root, "written_units")
    var u_ok = 0
    for i in range(doc.size(un)):
        var e = doc.at(un, i)
        var got = P.written_unit(cps(doc.string(doc.field(e, "text"))))
        var w = doc.field(e, "unit")
        var good = (doc.kind_of(w) == J_NULL) == (not got[0])
        if good and got[0]:
            good = got[1] == doc.string(w)
        if good:
            u_ok += 1
        else:
            print("    written_unit", repr(doc.string(doc.field(e, "text"))), "->", got[1])
    check(fails, u_ok == doc.size(un), String(u_ok) + " / " + String(doc.size(un)) + " written units equal d1's")

    var nn = doc.field(root, "numbers")
    var n_ok = 0
    var n_bad = 0
    var t2 = perf_counter_ns()
    for i in range(doc.size(nn)):
        var e = doc.at(nn, i)
        var un2 = doc.field(e, "unit")
        var has_unit = doc.kind_of(un2) == J_STRING
        var got = P.parse_number(cps(doc.string(doc.field(e, "text"))), cps(doc.string(doc.field(e, "context"))),
                                 doc.string(un2) if has_unit else String(""), has_unit)
        var kind = doc.string(doc.field(e, "kind"))
        var good: Bool
        if kind == "none":
            good = got.kind == 0
        elif kind == "int":
            good = got.kind == 1 and got.i == doc.integer(doc.field(e, "value"))
        else:
            good = got.kind == 2 and got.f == doc.number(doc.field(e, "value"))
        if good:
            n_ok += 1
        else:
            n_bad += 1
            if n_bad <= 12:
                print("    parse_number(", repr(doc.string(doc.field(e, "text"))), ",",
                      repr(doc.string(doc.field(e, "context"))), ", unit", doc.string(un2) if has_unit else String("None"),
                      ") d1", kind, doc.number(doc.field(e, "value")) if kind != "none" else 0.0, "mojo", got.kind, got.i, got.f)
    check(fails, n_ok == doc.size(nn), String(n_ok) + " / " + String(doc.size(nn)) + " parse_number results equal d1's ("
          + String((perf_counter_ns() - t2) // 1_000_000) + " ms)")

    var dn = doc.field(root, "dates")
    var d_ok = 0
    var d_bad = 0
    for i in range(doc.size(dn)):
        var e = doc.at(dn, i)
        var got = P.parse_date(cps(doc.string(doc.field(e, "text"))), cps(doc.string(doc.field(e, "context"))))
        var w = doc.field(e, "value")
        var good = (doc.kind_of(w) == J_NULL) == (not got.found) and got.ambiguous == doc.boolean(doc.field(e, "ambiguous"))
        if good and got.found:
            good = got.iso == doc.string(w)
        if good:
            d_ok += 1
        else:
            d_bad += 1
            if d_bad <= 12:
                print("    parse_date(", repr(doc.string(doc.field(e, "text"))), ",", repr(doc.string(doc.field(e, "context"))),
                      ") d1", doc.string(w) if doc.kind_of(w) != J_NULL else String("None"),
                      doc.boolean(doc.field(e, "ambiguous")), "mojo", got.iso if got.found else String("None"), got.ambiguous)
    check(fails, d_ok == doc.size(dn), String(d_ok) + " / " + String(doc.size(dn)) + " parse_date results equal d1's")

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
