"""`noeira.text.bpe.ByteLevelBPE` against Hugging Face `tokenizers`, on LFM3's
`tokenizer.json` (d1-2B's; every LFM3 model shares it).

    pixi run mojo run -I . tests/text/test_bpe_reference.mojo

Needs `~/.cache/noeira/d1/tokenizer.json` and the reference
(`pixi run -e detect python tools/text/dump_bpe_reference.py`), and for the
packing gate d1's reference rows (`tools/d1/dump_d1_reference.py`, copied to
`~/.cache/noeira/d1/ref`).

1. Every reference string -> EXACTLY `tokenizers`' ids (75 strings aimed at each
   branch of the split regex, non-ASCII text, added tokens inside text, and d1's
   own prompt texts).
2. Every id list -> EXACTLY `tokenizers`' decoded text.
3. d1's packed rows rebuilt from its prompt texts — the first question with the
   state, then "QUESTION:\\n" + each further one, 16 to a row — equal the rows
   d1 itself built, id for id.
"""

from std.os import getenv
from std.os.path import exists
from std.time import perf_counter_ns

from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.text.bpe import ByteLevelBPE


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok:", what)
    else:
        print("  FAIL:", what)
        fails += 1


def show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(min(len(ids), 24)):
        s += String(ids[i]) + ("," if i + 1 < len(ids) else "")
    return s + ("...]" if len(ids) > 24 else "]")


def main() raises:
    var home = getenv("HOME")
    var dir = home + "/.cache/noeira/d1"
    var fails = 0
    var t0 = perf_counter_ns()
    var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
    print("ByteLevelBPE vs tokenizers (LFM3): loaded", tok.vocab_size(), "ids,", len(tok.merges), "merges,",
          len(tok.added), "added tokens in", (perf_counter_ns() - t0) // 1_000_000, "ms")

    # 1-2. the reference strings
    var doc = load_json(dir + "/bpe_cases.json")
    var cases = doc.field(doc.root(), "cases")
    var n = doc.size(cases)
    var enc_ok = 0
    var dec_ok = 0
    var n_ids = 0
    var t1 = perf_counter_ns()
    for i in range(n):
        var c = doc.at(cases, i)
        var text = doc.string(doc.field(c, "text"))
        var want = List[Int]()
        var ids_node = doc.field(c, "ids")
        for k in range(doc.size(ids_node)):
            want.append(doc.integer(doc.at(ids_node, k)))
        n_ids += len(want)
        var got = tok.encode(text)
        var same = len(got) == len(want)
        if same:
            for k in range(len(got)):
                if got[k] != want[k]:
                    same = False
                    break
        if same:
            enc_ok += 1
        else:
            print("    case", i, repr(text)[codepoint=0:min(80, repr(text).count_codepoints())])
            print("      tokenizers", show(want))
            print("      mojo      ", show(got))
        var dec = tok.decode(want)
        if dec == doc.string(doc.field(c, "decoded")):
            dec_ok += 1
        else:
            print("    decode case", i, ":", repr(dec)[codepoint=0:min(80, repr(dec).count_codepoints())])
    var ms = Float64(perf_counter_ns() - t1) / 1e6
    check(fails, enc_ok == n, String(enc_ok) + " / " + String(n) + " strings encode to tokenizers' ids exactly ("
          + String(n_ids) + " ids, " + String(ms) + " ms)")
    check(fails, dec_ok == n, String(dec_ok) + " / " + String(n) + " id lists decode to tokenizers' text exactly")

    # 3. d1's packed rows
    var names: List[String] = ["room_en", "single_fr", "many_noul"]
    var rows_ok = 0
    var rows = 0
    for name in names:
        var case_dir = dir + "/ref/" + name
        if not exists(case_dir + "/prompt.json"):
            print("    (no", case_dir, "— skipped)")
            continue
        var pd = load_json(case_dir + "/prompt.json")
        var prefix = pd.string(pd.field(pd.root(), "prefix"))
        var suf = pd.field(pd.root(), "suffixes")
        var r = 0
        var q0 = 0
        while q0 < pd.size(suf):
            var ids = tok.encode(prefix + pd.string(pd.at(suf, q0)))
            for q in range(q0 + 1, min(q0 + 16, pd.size(suf))):
                for id in tok.encode(String("QUESTION:\n") + pd.string(pd.at(suf, q))):
                    ids.append(id)
            var want = load_npy_f64(case_dir + "/row" + String(r) + "_ids.npy")
            var same = len(want) == len(ids)
            if same:
                for k in range(len(ids)):
                    if Int(want[k]) != ids[k]:
                        same = False
                        break
            rows += 1
            if same:
                rows_ok += 1
            else:
                print("    ", name, "row", r, ":", len(ids), "ids against d1's", len(want))
            r += 1
            q0 += 16
    check(fails, rows > 0 and rows_ok == rows, String(rows_ok) + " / " + String(rows)
          + " of d1's packed rows rebuilt id for id")

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
