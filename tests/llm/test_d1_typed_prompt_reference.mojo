"""d1's typed questions laid out by the Mojo port against d1-2B's own code
(docs/NN_MODEL_PORTS_PLAN.md §12.9, step T1). Host only, no model.

    pixi run mojo run -I . tests/llm/test_d1_typed_prompt_reference.mojo

Needs `~/.cache/noeira/d1/tokenizer.json` and the reference dumped on the Orin by
`tools/d1/dump_d1_typed_reference.py`, copied to `~/.cache/noeira/d1/typed_ref`
(`cases.json`, `config.json`, `calibration.json`).

For every case (multiple choices, reads over text and JSON states, estimates on
number and code grids, a box, boxes of 16 / 4 / 1 blocks, a 2-row mix):
1. the checkpoint's record (`D1Record.load`) equals what d1's engine serves;
2. the prefix equals d1's (a picture case takes its media markup from d1's prefix);
3. each question's suffix and slot offsets equal d1's (`suffix_layout`);
4. each slot's readout (its token groups, or a pointer end) equals d1's;
5. the packed rows — ids, each token's question, state length, slots — equal d1's;
6. each row's state positions — per token span, JSON string paths, pointer
   width — equal d1's (`state_positions`);
7. every text d1 tokenized gives its offset mapping exactly (`encode_offsets`);
8. the calibration tables' `through` and `level` equal d1's over a sweep.

Mutants run against it (11-23): each of these fails it — a box's `after`
newline, `plain_offsets`' boundaries, `:g` zeros, edges as codes, unquoted
`multiple` markers, BOS outside the pointer mask, RFC 6901 escaping, the
first knot's flat end, a cut character's offsets. Not reachable with LFM3's
tokenizer, so not gated: `state_positions`' whitespace test on a token's far
side (`full_text[hi:b]`) — the state is always followed by "\n\n\nQUESTION:",
and no split joins a newline to the letters after it.
"""

from std.os import getenv
from std.os.path import exists

from noeira.io.json import JsonDoc, load_json, J_NULL
from noeira.text.bpe import ByteLevelBPE
from noeira.llm.d1.prompt import (
    D1Record, D1Question, parse_questions, py_json_dumps, prefix_text, suffix_layout, question_slots, pack_rows,
    plain, SLOT_CODES, SLOT_START, SLOT_END, BOS, IM_START, HEADER,
)
from noeira.llm.d1.positions import state_positions
from noeira.llm.d1.calibration import D1Calibration


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok:", what)
    else:
        print("  FAIL:", what)
        fails += 1


def ints(ref doc: JsonDoc, node: Int) raises -> List[Int]:
    var out = List[Int]()
    for i in range(doc.size(node)):
        out.append(doc.integer(doc.at(node, i)))
    return out^


def same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def main() raises:
    var home = getenv("HOME")
    var dir = home + "/.cache/noeira/d1"
    var ref_dir = dir + "/typed_ref"
    if not exists(ref_dir + "/cases.json"):
        raise Error("no " + ref_dir + "/cases.json: dump it with tools/d1/dump_d1_typed_reference.py")
    var fails = 0
    var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
    var doc = load_json(ref_dir + "/cases.json")
    var root = doc.root()

    # 1. the record
    var rec = D1Record.load(ref_dir + "/config.json")
    var rn = doc.field(root, "record")
    check(fails, rec.codes == doc.integer(doc.field(rn, "d1_codes")) and rec.numbers == doc.boolean(doc.field(rn, "d1_numbers"))
          and rec.prompt == doc.integer(doc.field(rn, "d1_prompt")) and rec.box_edges == doc.string(doc.field(rn, "d1_box_edges"))
          and rec.set_size == doc.integer(doc.field(rn, "set_size")) and rec.attention == doc.string(doc.field(rn, "attention")),
          "record: codes " + String(rec.codes) + ", numbers " + String(rec.numbers) + ", prompt " + String(rec.prompt)
          + ", edges " + rec.box_edges + ", set " + String(rec.set_size) + ", " + rec.attention)

    var tables = doc.field(root, "tables")
    var cases = doc.field(root, "cases")
    var n_pre = 0
    var ok_pre = 0
    var n_suf = 0
    var ok_suf = 0
    var n_slot = 0
    var ok_slot = 0
    var n_row = 0
    var ok_row = 0
    var n_pos = 0
    var ok_pos = 0
    var n_off = 0
    var ok_off = 0
    for ci in range(doc.size(cases)):
        var name = doc.key_at(cases, ci)
        var c = doc.at(cases, ci)
        var is_json = doc.boolean(doc.field(c, "state_is_json"))
        var snode = doc.field(c, "state")
        var block = py_json_dumps(doc, snode) if is_json else doc.string(snode)
        var qs = parse_questions(doc, doc.field(c, "questions"), rec)

        # 2. prefix
        var want_pre = doc.string(doc.field(c, "prefix"))
        var media = String("")
        if doc.integer(doc.field(c, "pictures")) > 0:
            var head = String(BOS) + IM_START + "user\n"
            var rest = plain(tok, block + "\n\n") + "\n" + HEADER
            var wb = want_pre.byte_length()
            if want_pre.startswith(head) and want_pre.endswith(rest):
                media = String(want_pre[byte=head.byte_length():wb - rest.byte_length()])
        var prefix = prefix_text(tok, block, media)
        n_pre += 1
        if prefix == want_pre:
            ok_pre += 1
        else:
            print("    ", name, ": prefix differs (", prefix.byte_length(), "vs", want_pre.byte_length(), "bytes)")

        # 3-4. suffixes, slot offsets, slots
        var sufs = doc.field(c, "suffixes")
        var offs_n = doc.field(c, "slot_offsets")
        var slots_n = doc.field(c, "slots")
        var suffixes = List[String]()
        var offsets = List[List[Int]]()
        for qi in range(len(qs)):
            var lay = suffix_layout(tok, qs[qi], rec)
            n_suf += 1
            var want_s = doc.string(doc.at(sufs, qi))
            if lay[0] == want_s and same(lay[1], ints(doc, doc.at(offs_n, qi))):
                ok_suf += 1
            else:
                print("    ", name, "/", qs[qi].name, ": suffix", "SAME" if lay[0] == want_s else "DIFFERS",
                      ", offsets", "same" if same(lay[1], ints(doc, doc.at(offs_n, qi))) else "differ")
                if lay[0] != want_s:
                    print("       mojo  :", repr(lay[0])[codepoint=0:min(300, repr(lay[0]).count_codepoints())])
                    print("       d1    :", repr(want_s)[codepoint=0:min(300, repr(want_s).count_codepoints())])
            suffixes.append(lay[0].copy())
            offsets.append(lay[1].copy())
            var sl = question_slots(tok, qs[qi], rec)
            var ws = doc.at(slots_n, qi)
            var good = len(sl) == doc.size(ws)
            for k in range(min(len(sl), doc.size(ws))):
                var w = doc.at(ws, k)
                var pn = doc.field(w, "position")
                if pn >= 0:
                    var want_kind = SLOT_START if doc.string(pn) == "start" else SLOT_END
                    good = good and sl[k].kind == want_kind
                else:
                    var t = doc.at(tables, doc.integer(doc.field(w, "codes")))
                    good = good and sl[k].kind == SLOT_CODES and len(sl[k].groups) == doc.size(t)
                    if good:
                        for g in range(doc.size(t)):
                            if not same(sl[k].groups[g], ints(doc, doc.at(t, g))):
                                good = False
                                break
            n_slot += 1
            if good:
                ok_slot += 1
            else:
                print("    ", name, "/", qs[qi].name, ": slots differ (", len(sl), "vs", doc.size(ws), ")")

        # 5-6. rows and state positions
        var rows = pack_rows(tok, prefix, suffixes, rec.set_size, offsets)
        var wrows = doc.field(c, "rows")
        if len(rows) != doc.size(wrows):
            print("    ", name, ":", len(rows), "rows against d1's", doc.size(wrows))
            n_row += 1
            continue
        for r in range(len(rows)):
            ref row = rows[r]
            var w = doc.at(wrows, r)
            var good = same(row.ids, ints(doc, doc.field(w, "ids"))) and same(row.groups, ints(doc, doc.field(w, "question")))
            good = good and row.state_len == doc.integer(doc.field(w, "state_len")) and row.first == doc.integer(doc.field(w, "first"))
            var we = doc.field(w, "ends")
            var flat = List[Int]()
            for qi in range(doc.size(we)):
                for v in ints(doc, doc.at(we, qi)):
                    flat.append(v)
            good = good and same(row.slots, flat)
            n_row += 1
            if good:
                ok_row += 1
            else:
                print("    ", name, "row", r, ": ids", same(row.ids, ints(doc, doc.field(w, "ids"))), "groups",
                      same(row.groups, ints(doc, doc.field(w, "question"))), "state", row.state_len, "vs",
                      doc.integer(doc.field(w, "state_len")), "slots", same(row.slots, flat))
            var wp = doc.field(w, "positions")
            if wp < 0 or doc.kind_of(wp) == J_NULL:
                continue
            var full = prefix + suffixes[row.first]
            var enc = tok.encode_offsets(full)
            var sp = state_positions(tok, block, is_json, prefix, full, enc[1], enc[2])
            var po = doc.field(wp, "offsets")
            var pgood = sp.text == doc.string(doc.field(wp, "text")) and len(sp.starts) == doc.size(po)
            var bad = -1
            if pgood:
                for k in range(len(sp.starts)):
                    var o = doc.at(po, k)
                    var ok: Bool
                    if doc.kind_of(o) == J_NULL:
                        ok = sp.starts[k] < 0
                    else:
                        ok = sp.starts[k] == doc.integer(doc.at(o, 0)) and sp.ends[k] == doc.integer(doc.at(o, 1))
                    if not ok:
                        pgood = False
                        bad = k
                        break
            var pp = doc.field(wp, "paths")
            pgood = pgood and len(sp.paths) == doc.size(pp)
            if pgood:
                for k in range(len(sp.paths)):
                    var o = doc.at(pp, k)
                    if (sp.paths[k].start != doc.integer(doc.at(o, 0)) or sp.paths[k].end != doc.integer(doc.at(o, 1))
                            or sp.paths[k].path != doc.string(doc.at(o, 2))):
                        pgood = False
                        print("       path", k, sp.paths[k].path, sp.paths[k].start, sp.paths[k].end)
                        break
            var mask = doc.field(wp, "pointer_mask")
            pgood = pgood and sp.pointer_width() == doc.size(mask)
            if pgood:
                for k in range(doc.size(mask)):
                    if sp.valid(k) != doc.boolean(doc.at(mask, k)):
                        pgood = False
                        break
            n_pos += 1
            if pgood:
                ok_pos += 1
            else:
                print("    ", name, "row", r, ": state positions differ (token", bad, ", paths", len(sp.paths), "vs",
                      doc.size(pp), ", width", sp.pointer_width(), "vs", doc.size(mask), ")")

        # 7. offsets
        var on = doc.field(c, "offsets")
        for k in range(doc.size(on)):
            var e = doc.at(on, k)
            var got = tok.encode_offsets(doc.string(doc.field(e, "text")))
            var wo = doc.field(e, "offsets")
            var good = same(got[0], ints(doc, doc.field(e, "ids"))) and doc.size(wo) == len(got[1])
            if good:
                for t in range(len(got[1])):
                    var o = doc.at(wo, t)
                    if got[1][t] != doc.integer(doc.at(o, 0)) or got[2][t] != doc.integer(doc.at(o, 1)):
                        good = False
                        break
            n_off += 1
            if good:
                ok_off += 1
            else:
                print("    ", name, ": offsets of text", k, "differ")

    check(fails, ok_pre == n_pre, String(ok_pre) + " / " + String(n_pre) + " prefixes equal d1's")
    check(fails, ok_suf == n_suf, String(ok_suf) + " / " + String(n_suf) + " suffixes and slot offsets equal d1's")
    check(fails, ok_slot == n_slot, String(ok_slot) + " / " + String(n_slot) + " questions' slot readouts equal d1's")
    check(fails, ok_row == n_row, String(ok_row) + " / " + String(n_row) + " packed rows equal d1's (ids, questions, state, slots)")
    check(fails, n_pos > 0 and ok_pos == n_pos, String(ok_pos) + " / " + String(n_pos) + " rows' state positions equal d1's")
    check(fails, ok_off == n_off, String(ok_off) + " / " + String(n_off) + " texts' offset mappings equal d1's")

    # 8. calibration
    var cal = D1Calibration.load(ref_dir + "/calibration.json")
    var cn = doc.field(root, "calibration")
    var thr = doc.field(cn, "through")
    var n_t = 0
    var ok_t = 0
    for i in range(doc.size(thr)):
        var tname = doc.key_at(thr, i)
        var pts = doc.at(thr, i)
        for k in range(doc.size(pts)):
            var p = doc.at(pts, k)
            n_t += 1
            if cal.through(tname, doc.number(doc.at(p, 0))) == doc.number(doc.at(p, 1)):
                ok_t += 1
    var lv = doc.field(cn, "level")
    for k in range(doc.size(lv)):
        var p = doc.at(lv, k)
        n_t += 1
        if cal.level("estimate", doc.number(doc.at(p, 0))) == doc.number(doc.at(p, 1)):
            ok_t += 1
    check(fails, n_t > 0 and ok_t == n_t and cal.default == doc.number(doc.field(rn, "temperature")),
          String(ok_t) + " / " + String(n_t) + " calibration reads equal d1's exactly (" + String(len(cal.names))
          + " tables, temperature " + String(cal.default) + ")")

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
