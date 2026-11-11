"""CLIP's BPE in Mojo (`noeira.text.clip_bpe`) against Hugging Face `tokenizers`.

    pixi run -e detect python tools/text/dump_clip_bpe_reference.py
    pixi run mojo run -I . tests/text/test_clip_bpe_reference.mojo [tokenizer.json] [reference json]

Every string's 32 ids and mask exact (normalisation, CLIP's split, byte level,
`</w>` merges, the framing and the padding).
"""

from std.os import getenv, listdir
from std.sys import argv

from noeira.io.json import load_json
from noeira.text.clip_bpe import ClipBPE


def main() raises:
    var args = argv()
    var tok_path: String
    if len(args) > 1:
        tok_path = String(args[1])
    else:
        var snaps = getenv("HOME") + "/.cache/huggingface/hub/models--facebook--sam3/snapshots"
        tok_path = String("")
        for d in listdir(snaps):
            tok_path = snaps + "/" + String(d) + "/tokenizer.json"
    var ref_path = String(args[2]) if len(args) > 2 else getenv("HOME") + "/.cache/noeira/sam3/clip_bpe.json"
    var tok = ClipBPE.load(tok_path)
    var doc = load_json(ref_path)
    var cases = doc.field(doc.root(), "cases")
    var fails = 0
    for i in range(doc.size(cases)):
        var c = doc.at(cases, i)
        var text = doc.string(doc.field(c, "text"))
        var want = doc.field(c, "ids")
        var wm = doc.field(c, "mask")
        var got = tok.encode(text, 32)
        var ok = True
        for k in range(32):
            if got.ids[k] != doc.integer(doc.at(want, k)) or got.mask[k] != doc.integer(doc.at(wm, k)):
                ok = False
        if not ok:
            fails += 1
            var gs = String("")
            var ws = String("")
            for k in range(got.length + 1):
                gs += String(got.ids[k]) + " "
                ws += String(doc.integer(doc.at(want, k))) + " "
            print("  FAIL", repr(text), "\n     got ", gs, "\n     want", ws)
    print("  ", doc.size(cases) - fails, "/", doc.size(cases), "strings: ids and mask exact")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
