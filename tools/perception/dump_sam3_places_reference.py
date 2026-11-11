"""The 3D places SAM 3's masks give on a real RGB-D frame — the reference the Mojo
detector service (`noeira.perception.detector`) is gated against.

    pixi run -e detect python tools/perception/dump_sam3_places_reference.py \\
        --mat ~/.cache/noeira/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat --index 921 \\
        [--ref ~/.cache/noeira/sam3/ref]

Reads NYU Depth v2's labelled frame `index` (640 x 480 Kinect RGB with the depth
REGISTERED to it, metres) and, for every word under `<ref>/words/`, the masks
transformers' SAM 3 kept (`post_masks.npy`, from `dump_sam3_reference.py` on the
SAME picture — checked: the frame's RGB must equal `<ref>/image.npy`). Writes into
`<ref>`:

    depth.npy        [480, 640] float32, metres, 0 = no depth
    places.json      {"intrinsics": [fx, fy, cx, cy],
                      "words": {word: {"sam3": [[x, y, z] | null per kept mask],
                                       "truth": [[x, y, z] per NYU instance] | null}}}

A place is `detector_eval.centroid3d`'s rule: the per-axis MEDIAN of the mask's
depth points (a mask edge bleeding onto the wall behind drags a mean by metres),
null under 20 points. Coordinates are OpenCV's camera frame (x right, y down, z
forward) — the Mojo side's sensor frame is REP-103 (x forward, y left, z up):
`(z, -x, -y)`. `truth` (the frame's NYU instances of the word's class, the eval's
class ids) is printed beside, not gated: the detector's own place error.
"""

import argparse
import json
import os

import h5py
import numpy as np

FX, FY, CX, CY = 518.857901, 519.469611, 325.582449, 253.736166  # detector_eval.py's
CLASS = {"fridge": 17, "table": 19, "door": 28, "sofa": 83}
MIN_POINTS = 20


def centroid3d(mask, dep):
    v, u = np.nonzero(mask & (dep > 0))
    if len(v) < MIN_POINTS:
        return None
    z = dep[v, u].astype(np.float64)
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return [float(np.median(x)), float(np.median(y)), float(np.median(z))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--ref", default=os.path.expanduser("~/.cache/noeira/sam3/ref"))
    a = ap.parse_args()
    f = h5py.File(os.path.expanduser(a.mat), "r")
    i = a.index
    img = np.ascontiguousarray(f["images"][i].transpose(2, 1, 0))
    dep = np.ascontiguousarray(f["depths"][i].T).astype(np.float32)
    lab = f["labels"][i].T
    ins = f["instances"][i].T
    ref_img = np.load(os.path.join(a.ref, "image.npy"))
    if ref_img.shape != img.shape or not np.array_equal(ref_img, img):
        raise SystemExit(f"frame {i} is not the picture in {a.ref}/image.npy")
    np.save(os.path.join(a.ref, "depth.npy"), dep)
    out = {"intrinsics": [FX, FY, CX, CY], "index": i, "words": {}}
    wdir = os.path.join(a.ref, "words")
    for d in sorted(os.listdir(wdir)):
        word = d.replace("_", " ")
        masks = np.load(os.path.join(wdir, d, "post_masks.npy"))  # [n, H, W]
        sam = [centroid3d(m > 0.5, dep) for m in masks.reshape(-1, *dep.shape)] if masks.size else []
        truth = None
        if word in CLASS:
            m = lab == CLASS[word]
            truth = [centroid3d(m & (ins == k), dep) for k in np.unique(ins[m])] if m.any() else []
        out["words"][word] = {"sam3": sam, "truth": truth}
        print(f"{word:12s} sam3 {sam}  truth {truth}")
    with open(os.path.join(a.ref, "places.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(a.ref, "depth.npy"), "and places.json")


if __name__ == "__main__":
    main()
