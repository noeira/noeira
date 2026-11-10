# +--------------------------------------------------------------------------+ #
# | espeak-ng from Mojo, against phonemizer, kokoro-onnx and misaki
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/audio/espeak.mojo` and `noeira/audio/kokoro/g2p.mojo`.

    python3 tools/kokoro/fetch_espeak_ng.py                          # the library, once
    ~/.cache/noeira/local-ai/tts-venv/bin/python tools/kokoro/dump_g2p_reference.py
    pixi run mojo run -I . tests/audio/test_espeak_reference.mojo [cases.json]

For each of the reference's sentences (English and French: punctuation of
every kind, several sentences on a line, several lines, numbers and decimals,
apostrophes, an English word in French), EXACT string equality of:

- `EspeakNG.phonemize` against `phonemizer.phonemize(text, lang,
  preserve_punctuation=True, with_stress=True)` (`raw`);
- the three Kokoro modes against kokoro-onnx (`deployment`), misaki's
  `EspeakG2P` (`misaki`) and misaki's `EspeakFallback` (`misaki_en`).

And the library is the reference's: same espeak-ng version.

Falsification — each must change at least one sentence, else the comparison
could not see it: punctuation removed instead of preserved; the tie on where
the deployment has none; the British voice for American text; language-switch
flags kept in the misaki mode.
"""

from std.os import getenv
from std.sys import argv

from noeira.io.json import load_json
from noeira.audio.espeak import EspeakNG
from noeira.audio.kokoro.g2p import (
    kokoro_phonemes, KK_G2P_DEPLOYMENT, KK_G2P_MISAKI, KK_G2P_MISAKI_EN,
)


def show(s: String) -> String:
    return "'" + s.replace("\n", "\\n") + "'"


def main() raises:
    var args = argv()
    var path = String(args[1]) if len(args) >= 2 else getenv("HOME") + "/.cache/noeira/kokoro/g2p_cases.json"
    var doc = load_json(path)
    var root = doc.root()
    var meta = doc.field(root, "meta")
    var cases = doc.field(root, "cases")
    var want_version = doc.string(doc.field(meta, "espeak_ng"))
    var vocab_str = doc.string(doc.field(meta, "vocab"))
    var vocab = Dict[String, Int]()
    for sl in vocab_str.codepoint_slices():
        vocab[String(sl)] = 1

    var es = EspeakNG.open()
    print("espeak-ng from Mojo vs phonemizer / kokoro-onnx / misaki (" + path + ")")
    var fails = 0
    var v_ok = es.version == want_version
    print("  " + ("PASS" if v_ok else "FAIL") + "  library espeak-ng " + es.version
          + " (reference " + want_version + "), data " + es.data_path)
    if not v_ok:
        fails += 1

    var n = doc.size(cases)
    var compared = 0
    var diff_punct = 0
    var diff_tie = 0
    var diff_voice = 0
    var diff_flags = 0
    var filtered = 0
    for k in range(n):
        var c = doc.at(cases, k)
        var lang = doc.string(doc.field(c, "lang"))
        var text = doc.string(doc.field(c, "text"))
        var checks = List[Tuple[String, String, String]]()
        var raw = es.phonemize(text, lang)
        checks.append((String("raw"), raw, doc.string(doc.field(c, "raw"))))
        var dep = kokoro_phonemes(es, text, lang, KK_G2P_DEPLOYMENT, vocab)
        checks.append((String("deployment"), dep, doc.string(doc.field(c, "deployment"))))
        checks.append((String("misaki"), kokoro_phonemes(es, text, lang, KK_G2P_MISAKI, vocab),
                       doc.string(doc.field(c, "misaki"))))
        if doc.field(c, "misaki_en") >= 0:
            checks.append((String("misaki_en"), kokoro_phonemes(es, text, lang, KK_G2P_MISAKI_EN, vocab),
                           doc.string(doc.field(c, "misaki_en"))))
        var case_ok = True
        for j in range(len(checks)):
            compared += 1
            if checks[j][1] != checks[j][2]:
                case_ok = False
                print("  FAIL  [" + lang + "] " + checks[j][0] + " of " + show(text))
                print("          ours " + show(checks[j][1]))
                print("        theirs " + show(checks[j][2]))
        if case_ok:
            print("  PASS  [" + lang + "] " + show(text) + " -> " + show(dep))
        else:
            fails += 1
        if raw != dep:
            filtered += 1

        # falsification probes on the same sentence
        var lines: List[String] = [text]
        var no_punct = es.backend_phonemize(lines, lang, String(""), False, False)
        var np = String("")
        for j in range(len(no_punct)):
            np += no_punct[j]
        if np != raw:
            diff_punct += 1
        if es.phonemize(text, lang, String("^")) != raw:
            diff_tie += 1
        if lang == "en-us" and es.phonemize(text, String("en-gb")) != raw:
            diff_voice += 1
        if es.phonemize(text, lang, String("^"), False) != es.phonemize(text, lang, String("^"), True):
            diff_flags += 1

    if compared == 0:
        raise Error("gate: nothing compared — this run proved nothing")
    print("  " + String(compared) + " strings compared over " + String(n) + " sentences; the vocabulary filter"
          + " changed " + String(filtered) + " of them")
    var probes = List[Tuple[String, Int]]()
    probes.append((String("punctuation removed instead of preserved"), diff_punct))
    probes.append((String("a tie where the deployment has none"), diff_tie))
    probes.append((String("the British voice for American text"), diff_voice))
    probes.append((String("language-switch flags kept vs removed"), diff_flags))
    for j in range(len(probes)):
        var ok = probes[j][1] > 0
        print("  " + ("PASS" if ok else "FAIL") + "  defect: " + probes[j][0] + " changes "
              + String(probes[j][1]) + " sentence(s)" + ("" if ok else "  NOT SEEN"))
        if not ok:
            fails += 1
    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("espeak gate failed")
