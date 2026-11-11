"""NYU Depth v2's labelled frames as an RF-DETR fine-tuning set (COCO format, polygon
masks) — the room words COCO lacks (door) beside the ones it has.

    pixi run -e detect ~/.cache/noeira/venvs/detcand/bin/python tools/perception/nyu_to_coco.py \\
        --mat ~/.cache/noeira/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat \\
        --out ~/.cache/noeira/rfdetr/nyu_coco [--words fridge,table,door,sofa] [--valid 0.15] [--seed 0]

THE EVAL STAYS HELD OUT: the 524 frames `detector_eval.py` scores (seed 0, 120 per
word, 120 negatives — the same `select`) are never written; the other 925 split
into train / valid (`--valid`, seeded). So the fine-tuned model is judged on the
SAME 524 frames as every detector before it (SPATIAL_PERCEPTION_PLAN P3 table).

Classes: the words with NYU's class ids (`detector_eval.WORDS`), category ids 1..N
in that order. An instance is (label == class) & (instance == k), kept from
`--min-pixels` (200; the eval's "difficult" rule is 0.5 % of the frame = 1 536 px,
but small true instances still teach); its mask as polygons
(`cv2.findContours`, external, each part of >= 3 points), its box from the mask.

Writes `<out>/{train,valid}/` with the frames as PNG and `_annotations.coco.json`
(the layout `rfdetr`'s `train(dataset_dir=...)` reads), and `<out>/split.json`.
"""

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_eval import WORDS, select, load  # noqa: E402


def polygons(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        c = c.reshape(-1, 2)
        if len(c) >= 3:
            out.append([float(v) for xy in c for v in (xy[0] + 0.5, xy[1] + 0.5)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", default="fridge,table,door,sofa")
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=200)
    a = ap.parse_args()
    words = [w.strip() for w in a.words.split(",")]
    for w in words:
        if w not in WORDS:
            raise SystemExit(f"'{w}' has no NYU class id in detector_eval.WORDS")
    out = os.path.expanduser(a.out)
    f = h5py.File(os.path.expanduser(a.mat), "r")
    held, _ = select(f, 120, 120, 0)
    held = set(held)
    n = f["images"].shape[0]
    rest = [i for i in range(n) if i not in held]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(rest)
    nv = int(round(len(rest) * a.valid))
    split = {"valid": sorted(rest[:nv]), "train": sorted(rest[nv:]), "held_out_eval": sorted(held)}
    cats = [{"id": k + 1, "name": w, "supercategory": "room"} for k, w in enumerate(words)]
    totals = {}
    for part in ("train", "valid"):
        d = os.path.join(out, part)
        os.makedirs(d, exist_ok=True)
        images, anns = [], []
        counts = {w: 0 for w in words}
        for i in split[part]:
            img, lab, ins, _ = load(f, i)
            name = f"nyu_{i:04d}.png"
            Image.fromarray(img).save(os.path.join(d, name))
            images.append({"id": i, "file_name": name, "height": int(img.shape[0]), "width": int(img.shape[1])})
            for k, w in enumerate(words):
                m = lab == WORDS[w]
                if not m.any():
                    continue
                for inst in np.unique(ins[m]):
                    mk = m & (ins == inst)
                    area = int(mk.sum())
                    if area < a.min_pixels:
                        continue
                    seg = polygons(mk)
                    if not seg:
                        continue
                    ys, xs = np.nonzero(mk)
                    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
                    anns.append({"id": len(anns) + 1, "image_id": i, "category_id": k + 1, "segmentation": seg,
                                 "area": float(area), "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
                    counts[w] += 1
        with open(os.path.join(d, "_annotations.coco.json"), "w") as fh:
            json.dump({"images": images, "annotations": anns, "categories": cats}, fh)
        totals[part] = {"images": len(images), "instances": counts}
        print(f"{part}: {len(images)} frames, instances {counts}")
    with open(os.path.join(out, "split.json"), "w") as fh:
        json.dump({**split, "words": words, "totals": totals}, fh)
    print(f"held out (the eval's {len(held)} frames) never written; wrote {out}")


if __name__ == "__main__":
    main()
