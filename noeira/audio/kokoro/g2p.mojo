# +--------------------------------------------------------------------------+ #
# | Kokoro-82M — text to its phonemes, through espeak-ng
# +--------------------------------------------------------------------------+ #
"""Three ways to turn text into Kokoro phonemes with espeak-ng
(docs/NN_MODEL_PORTS_PLAN.md §11.3, option A), each reproducing a Python path
exactly (`tests/audio/test_espeak_reference.mojo`):

| mode | reproduces | what it does after espeak |
|---|---|---|
| `KK_G2P_DEPLOYMENT` | kokoro-onnx 0.6.1 (the robot today) | nothing: characters outside Kokoro's vocabulary dropped |
| `KK_G2P_MISAKI` | upstream Kokoro for every language but en / ja / zh (`misaki.espeak.EspeakG2P`) | `^` tie, language-switch flags removed, diphthongs to Kokoro's letters, « » kept as “ ” |
| `KK_G2P_MISAKI_EN` | upstream Kokoro's espeak mapping for English (`misaki.espeak.EspeakFallback`, applied to the whole sentence) | as above, plus American English without length marks, flap as `T` |

What differs in practice (measured on 28 sentences): the deployment hands
Kokoro raw espeak diphthongs (`aɪ`, `oʊ`, `dʒ`) where upstream gives the
single letters the model was trained on (`I`, `O`, `ʤ`); it DROPS « » (not in
the vocabulary); and for an English word in French ("le football") it keeps
espeak's language-switch flags, so the model is fed the characters `(en)` and
`(fr)`.

The defaults (`kokoro_default_g2p`) were chosen by a ROUND TRIP — text, Kokoro
in Mojo, then the Mojo Parakeet, 17 robot sentences in each mode (2026-11-10):

| | French (9) | English (8) |
|---|---|---|
| `KK_G2P_MISAKI` / `_MISAKI_EN` | 9 / 9 | 6 / 8: the word "I" is LOST twice |
| `KK_G2P_DEPLOYMENT` | 8 / 9: "le football" comes back "le N football, fre" (the flags, spoken) | 8 / 8 |

(numbers written as digits by Parakeet counted as equal). So French and the
other espeak languages take upstream's mapping, English keeps the deployment's.
The lost "I": misaki maps espeak's `aɪ` to the letter `I`, and a lone `I`
without a stress mark — which misaki's lexicon would have given it — is
swallowed by the model. Upstream's English proper is that lexicon; the espeak
mapping is only what upstream falls back to for words the lexicon lacks.
"""

from noeira.audio.espeak import EspeakNG, py_isspace, py_strip


comptime KK_G2P_DEPLOYMENT: Int = 0
comptime KK_G2P_MISAKI: Int = 1
comptime KK_G2P_MISAKI_EN: Int = 2


def kokoro_language(voice: String) raises -> String:
    """espeak's language for a Kokoro voice, from its first letter (Kokoro's
    naming: `af_heart` American English, `ff_siwis` French, ...)."""
    var k = String(voice[byte=0:1])
    if k == "a":
        return String("en-us")
    if k == "b":
        return String("en-gb")
    if k == "f":
        return String("fr-fr")
    if k == "e":
        return String("es")
    if k == "i":
        return String("it")
    if k == "p":
        return String("pt-br")
    if k == "h":
        return String("hi")
    raise Error("kokoro: no espeak language for voice '" + voice + "' (Japanese and Chinese need misaki, not espeak)")


def kokoro_default_g2p(language: String) -> Int:
    """The deployment's mapping for English, upstream's for the rest (module
    docstring: the round trip that chose them)."""
    return KK_G2P_DEPLOYMENT if language.startswith("en-") else KK_G2P_MISAKI


def _misaki_g2p(mut es: EspeakNG, text: String, language: String) raises -> String:
    """`misaki.espeak.EspeakG2P(language)(text)`, version None."""
    var t = text.replace("«", "“").replace("»", "”")
    t = t.replace("(", "«").replace(")", "»")
    var lines: List[String] = [t]
    var out = es.backend_phonemize(lines, language, String("^"), True)
    if len(out) == 0:
        return String("")
    var ps = py_strip(out[0])
    # misaki's e2m table, in `sorted(...)` order.
    ps = ps.replace("a^ɪ", "I").replace("a^ʊ", "W").replace("d^z", "ʣ").replace("d^ʒ", "ʤ")
    ps = ps.replace("e^ɪ", "A").replace("o^ʊ", "O").replace("s^s", "S").replace("t^s", "ʦ")
    ps = ps.replace("t^ʃ", "ʧ").replace("ɔ^ɪ", "Y").replace("ə^ʊ", "Q")
    ps = ps.replace("^", "").replace("-", "")
    return ps.replace("«", "(").replace("»", ")")


def _syllabic(ps: String) -> String:
    """`re.sub(r'(\\S)\\u0329', r'ᵊ\\1', ps).replace(chr(809), '')`."""
    var cps = List[Int]()
    var strs = List[String]()
    for c in ps.codepoints():
        cps.append(Int(c))
    for sl in ps.codepoint_slices():
        strs.append(String(sl))
    var out = String("")
    var i = 0
    while i < len(cps):
        if i + 1 < len(cps) and cps[i + 1] == 0x329 and not py_isspace(cps[i]):
            out += "ᵊ" + strs[i]
            i += 2
            continue
        if cps[i] != 0x329:
            out += strs[i]
        i += 1
    return out^


def _misaki_en(mut es: EspeakNG, text: String, language: String) raises -> String:
    """`misaki.espeak.EspeakFallback(british)` applied to `text` (version None)."""
    var british = language == "en-gb"
    var lines: List[String] = [text]
    var out = es.backend_phonemize(lines, language, String("^"), False)
    if len(out) == 0:
        return String("")
    var ps = py_strip(out[0])
    # E2M, longest first; equal lengths in the table's own order.
    ps = ps.replace("ʔˌn̩", "ʔn")
    ps = ps.replace("ʔn̩", "ʔn").replace("a^ɪ", "I").replace("a^ʊ", "W").replace("d^ʒ", "ʤ")
    ps = ps.replace("e^ɪ", "A").replace("t^ʃ", "ʧ").replace("ɔ^ɪ", "Y").replace("ə^l", "ᵊl")
    ps = ps.replace("ʲo", "jo").replace("ʲə", "jə")
    ps = ps.replace("e", "A").replace("ʲ", "").replace("ɚ", "əɹ").replace("r", "ɹ")
    ps = ps.replace("x", "k").replace("ç", "k").replace("ɐ", "ə").replace("ɬ", "l").replace("̃", "")
    ps = _syllabic(ps)
    if british:
        ps = ps.replace("e^ə", "ɛː").replace("iə", "ɪə").replace("ə^ʊ", "Q")
    else:
        ps = ps.replace("o^ʊ", "O").replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ").replace("ɪə", "iə").replace("ː", "")
    ps = ps.replace("o", "ɔ")
    ps = ps.replace("ɾ", "T").replace("ʔ", "t")
    return ps.replace("^", "")


def kokoro_phonemes(
    mut es: EspeakNG, text: String, language: String, mode: Int,
    ref vocab: Dict[String, Int],
) raises -> String:
    """`text` in Kokoro's phoneme alphabet, through `mode` (module docstring).
    The deployment's mode drops characters outside `vocab` here, as kokoro-onnx
    does; the misaki modes leave that to the model's id lookup, as upstream."""
    if mode == KK_G2P_DEPLOYMENT:
        var raw = es.phonemize(py_strip(text), language)
        var kept = String("")
        for sl in raw.codepoint_slices():
            var ch = String(sl)
            if ch in vocab:
                kept += ch
        return py_strip(kept)
    if mode == KK_G2P_MISAKI:
        return _misaki_g2p(es, text, language)
    if mode == KK_G2P_MISAKI_EN:
        return _misaki_en(es, text, language)
    raise Error("kokoro: unknown G2P mode " + String(mode))
