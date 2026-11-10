#!/usr/bin/env python3
"""Text -> phonemes as the deployment and as upstream Kokoro do it, for the
Mojo espeak-ng binding's gate (docs/NN_MODEL_PORTS_PLAN.md §11.3, option A).

    ~/.cache/noeira/local-ai/tts-venv/bin/python tools/kokoro/dump_g2p_reference.py \
        [--out ~/.cache/noeira/kokoro/g2p_cases.json]

⚠ RUN IT WITH THE DEPLOYMENT'S PYTHON, not a pixi environment: the reference is
what the robot runs today — kokoro-onnx 0.6.1, phonemizer 3.4.0 (whose
punctuation handling keeps a decimal comma inside a number) and the
espeak-ng 1.52.0 that `espeakng_loader` 0.2.4 bundles. The same three versions
are on the Orin (checked 2026-11-10).

Two modes, both recorded for every sentence:

- `deployment`: `kokoro_onnx.Tokenizer.phonemize(text, lang)` — phonemizer with
  `preserve_punctuation=True, with_stress=True`, default separators, the
  'keep-flags' language-switch policy, then every character outside Kokoro's
  vocabulary dropped and the result stripped. `raw` is phonemizer's output
  before that filter.
- `misaki`: upstream Kokoro's path for every language but English, Japanese and
  Chinese — `misaki.espeak.EspeakG2P(language)`: the same phonemizer call with a
  `^` tie and 'remove-flags', and misaki's symbol replacements (needs misaki in
  `~/.cache/noeira/kokoro/ref-site`, as `_reference.py` sets up).
- `misaki_en` (English only): misaki's `EspeakFallback` — the espeak mapping
  upstream uses for English words its lexicons do not know — applied to the
  whole sentence: `^` tie, 'keep-flags', Kokoro's diphthong letters, and for
  American English no length marks.

The sentences exercise what the binding must reproduce: punctuation kept and
restored (commas, ? !, quotes, parentheses, ellipsis, dashes, a line made only
of punctuation), several sentences on one line (espeak returns them in pieces),
several lines, numbers and decimals, apostrophes and hyphens, and English words
in French text (espeak's language switches).
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import espeakng_loader
import phonemizer
from kokoro_onnx.tokenizer import Tokenizer

sys.path.append(os.path.expanduser("~/.cache/noeira/kokoro/ref-site"))
from misaki.espeak import EspeakFallback, EspeakG2P  # noqa: E402

SENTENCES = {
    "en-us": [
        "I am going to the fridge.",
        "Sorry, I cannot do that.",
        "Hello! How are you? I'm fine, thanks.",
        "Walk to the door; then stop: wait for me.",
        "The battery is at 19.5 percent, and it costs $3,500.",
        "She said \"go to the kitchen\" (now) and left...",
        "Well — that's it… I think.",
        "It's 10:30. Mr. Smith isn't here, isn't he?",
        "Turn left, turn right, then go forward three meters.",
        "first line\nsecond line, with a comma\n\nthird line!",
        "?!",
        "OK.",
        "I can't reach the cup; it's too far away.",
        "Raise your right hand, please.",
    ],
    "fr-fr": [
        "Je vais au frigo.",
        "Bonjour, comment puis-je vous aider ?",
        "Désolé, je ne peux pas faire ça.",
        "Il fait 19,5 degrés, et ça coûte 3,50 euros.",
        "Va près du canapé ; puis attends-moi : je reviens.",
        "Elle a dit « va dans la cuisine » (maintenant) et elle est partie...",
        "J'aime le football et le week-end.",
        "L'eau est froide. Peut-être qu'il pleut ? Oui !",
        "M. Dupont n'est pas là, n'est-ce pas ?",
        "Tourne à gauche, puis à droite — avance de trois mètres.",
        "Où est la porte d'entrée ?",
        "première ligne\ndeuxième ligne, avec une virgule",
        "Ah… bon.",
        "Je suis Noé, ton robot.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/kokoro/g2p_cases.json"))
    args = ap.parse_args()

    tok = Tokenizer()
    version = None
    cases = []
    for lang, texts in SENTENCES.items():
        g2p = EspeakG2P(language=lang)
        fallback = EspeakFallback(british=False) if lang == "en-us" else None
        for text in texts:
            raw = phonemizer.phonemize(text.strip(), lang, preserve_punctuation=True, with_stress=True)
            deployment = tok.phonemize(text, lang)
            misaki, _ = g2p(text)
            case = {"lang": lang, "text": text, "raw": raw, "deployment": deployment, "misaki": misaki}
            line = f"{lang}  {text!r}\n    deployment {deployment!r}\n    misaki     {misaki!r}"
            if fallback is not None:
                misaki_en, _ = fallback(SimpleNamespace(text=text))
                case["misaki_en"] = misaki_en or ""
                line += f"\n    misaki_en  {case['misaki_en']!r}"
            cases.append(case)
            print(line)
        if version is None:
            version = g2p.backend.version()

    meta = {
        "espeak_ng": ".".join(str(v) for v in version),
        "library": espeakng_loader.get_library_path(),
        "phonemizer": phonemizer.__version__,
        "vocab": "".join(sorted(tok.vocab)),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "cases": cases}, f, ensure_ascii=False, indent=1)
    differ = sum(c["deployment"] != c["misaki"] for c in cases)
    print(f"{len(cases)} cases -> {args.out}; espeak-ng {meta['espeak_ng']}, phonemizer {meta['phonemizer']};"
          f" the two modes differ on {differ}")


if __name__ == "__main__":
    main()
