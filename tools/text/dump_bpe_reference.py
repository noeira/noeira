#!/usr/bin/env python3
"""Text -> ids (and ids -> text) by Hugging Face `tokenizers`, for the Mojo
byte-level BPE's gate (`tests/text/test_bpe_reference.mojo`).

    pixi run -e detect python tools/text/dump_bpe_reference.py \
        [--tokenizer ~/.cache/noeira/d1/tokenizer.json] [--d1-ref ~/.cache/noeira/d1/ref] \
        [--out ~/.cache/noeira/d1/bpe_cases.json]

The strings exercise every branch of LFM3's split regex (contractions in any case,
an optional leading character before letters, 1-3 digit runs, punctuation with a
leading space and trailing newlines, whitespace runs with and without newlines,
at the end and in the middle), non-ASCII letters and numbers, combining marks,
U+200B (which d1 inserts into special strings), emoji (4-byte UTF-8), added
tokens inside text — and d1's own prompt texts from its reference dumps.
"""

import argparse
import glob
import json
import os

from tokenizers import Tokenizer

STRESS = [
    "Hello world",
    "I'm here, you're there; IT'S fine, we'll see, they'd go, she's OK, I've been, don't, CAN'T",
    "'s 'S 't 'T 're 'RE 've 'VE 'm 'M 'll 'LL 'd 'D 'x ''",
    "12345 1,234.56 3.14159e10 007 2026-11-10 10:30 19,5 °C 100%",
    "a  b   c\t\td\n\n\n  e  \r\n f   ",
    "trailing spaces   ",
    "   leading spaces",
    "\n\nnewlines first\n",
    "tabs\there\t\tand\t\t\tthere",
    "...!!!??? --- *** ### (parenthèses) [brackets] {braces} «guillemets» \"quotes\"",
    "Je vais au frigo, puis je reviens près du canapé. Où est la porte d'entrée ?",
    "Ça coûte 3,50 € — l'œuvre de l'été à Noël.",
    "été (combining acute)",
    "zero​width <​|im_start|> space",
    "Emoji 🤖🏠🍕 and 你好世界 and مرحبا and Привет мир and Ελληνικά",
    "Digits ٣٤٥ and ⅫⅢ and ²³ and ½",
    "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "<|pad|><|pad|>text<|endoftext|>",
    "not a <|special|> token < | im_start | >",
    '{\n  "robot": {\n    "at": "door",\n    "facing": "kitchen"\n  },\n  "heard": "Can you go?"\n}',
    "def f(x):\n    return x ** 2  # square\n\n\nprint(f(3))\n",
    "aɪɐm ɡˌoʊɪŋ tə ðə fɹˈɪdʒ",
    "A" * 300,
    "word " * 50,
    " ",
    "",
    "\r\n\r\n",
    "x y z　w",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/.cache/noeira/d1/tokenizer.json"))
    ap.add_argument("--d1-ref", default=os.path.expanduser("~/.cache/noeira/d1/ref"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/bpe_cases.json"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    texts = list(STRESS)
    for p in sorted(glob.glob(os.path.join(args.d1_ref, "*", "prompt.json"))):
        pr = json.load(open(p))
        texts.append(pr["prefix"])
        texts.extend(pr["suffixes"])
        texts.append(pr["prefix"] + pr["suffixes"][0])
        texts.extend("QUESTION:\n" + s for s in pr["suffixes"][1:])
    cases = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        cases.append({"text": t, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
    json.dump({"tokenizers": __import__("tokenizers").__version__, "cases": cases},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(cases)} cases -> {args.out} (tokenizers {__import__('tokenizers').__version__}); "
          f"{sum(len(c['ids']) for c in cases)} ids")


if __name__ == "__main__":
    main()
