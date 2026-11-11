#!/usr/bin/env python3
"""CLIP's tokenizer (SAM 3's, `facebook/sam3`) as Hugging Face `tokenizers` runs
it: the reference `tests/text/test_clip_bpe_reference.mojo` gates
`noeira/text/clip_bpe.mojo` against.

    pixi run -e detect python tools/text/dump_clip_bpe_reference.py [--out ~/.cache/noeira/sam3/clip_bpe.json]

Each string's ids and attention mask, padded to 32 as `Sam3Processor` pads them.
"""

import argparse
import json
import os

STRINGS = [
    "door", "sofa", "table", "fridge", "bed", "Fridge Door", "a red chair", "the kitchen counter",
    "Canapé", "CANAPÉ", "réfrigérateur", "naïve café", "ÀÉÎÕÜ àéîõü",
    "it's 2 cats!!", "don't", "we'll they're I've I'm he'd", "'quoted' words", "''s",
    "x-ray 3.14 and 1,000", "123 456", "  multiple   spaces\tand\nnewlines ", "C3PO-R2D2",
    "¿qué?", "emoji 🙂 ok", "日本の部屋", "the sofa's cushion", "...", "a/b\\c", "#1 & #2",
    "<|startoftext|>door<|endoftext|>", "supercalifragilisticexpialidocious",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/sam3/clip_bpe.json"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/sam3")
    out = []
    for s in STRINGS:
        t = tok(s, padding="max_length", max_length=32)
        out.append({"text": s, "ids": t["input_ids"], "mask": t["attention_mask"]})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"tokenizer": type(tok).__name__, "cases": out}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(len(out), "strings ->", args.out)


if __name__ == "__main__":
    main()
