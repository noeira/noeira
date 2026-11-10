"""Open-vocabulary detectors on real indoor photos: word -> box -> mask -> 3D place.

    pixi run -e detect python tools/perception/detector_eval.py \\
        --mat ~/.cache/noeira/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat \\
        --out runs/p3/detector_eval [--per-word 120] [--negatives 120] [--seed 0]

SPATIAL_PERCEPTION_PLAN P3 step 2: the oracle labels of step 1 are replaced by a
detector, and the detector is chosen HERE, on real photos — the simulator's flat
colours cannot judge it. NYU Depth v2's labelled set (1 449 Kinect RGB-D frames
of real rooms, per-pixel class + instance labels) carries the demo's four place
words: fridge (NYU "refridgerator", 17), table (19), door (28), sofa (83).

Per detector, per word, over a seeded stratified subset (images where the word
is present + images where none of the four is):

  AP50          box average precision at IoU 0.5 (instance level, VOC-style
                all-point interpolation), over every score the detector emits
  P / R @ F1*   precision and recall at the score threshold that maximises F1
  mask IoU      SAM (box-prompted) on the best-scoring detection matched to a
                true instance, against that instance's mask — median
  3D error      the word's PLACE: the 3D centroid of the predicted mask's depth
                points against the true instance's, camera frame, metres —
                median, and the fraction within 0.3 m. THE number P3 needs.
  latency       per image on this machine (MPS when available)

Detectors (`--detectors`): OWLv2 (google/owlv2-base-patch16-ensemble) and
Grounding DINO (IDEA-Research/grounding-dino-tiny), each prompted with the four
words, with SAM (facebook/sam-vit-base) making masks from their boxes; and SAM 3
(facebook/sam3, gated behind Meta's licence), which segments each word's
instances itself — its own masks are scored.

Ground truth: an instance is (label == class) & (instance == k); instances under
0.5 % of the image (blurred far fragments, mislabelled specks) are ignored as
"difficult" — neither a miss nor a false positive if detected.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

WORDS = {"fridge": 17, "table": 19, "door": 28, "sofa": 83}
# NYU Depth v2's colour intrinsics (the toolbox's camera_params.m).
FX, FY, CX, CY = 518.857901, 519.469611, 325.582449, 253.736166
MIN_FRAC = 0.005


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(f, i):
    img = np.ascontiguousarray(f["images"][i].transpose(2, 1, 0))  # 480 x 640 x 3
    lab = f["labels"][i].T
    ins = f["instances"][i].T
    dep = f["depths"][i].T
    return img, lab, ins, dep


def instances_of(lab, ins, cls):
    """[(mask, difficult)] for the class."""
    out = []
    m = lab == cls
    if not m.any():
        return out
    for k in np.unique(ins[m]):
        mk = m & (ins == k)
        out.append((mk, mk.mean() < MIN_FRAC))
    return out


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def centroid3d(mask, dep):
    v, u = np.nonzero(mask & (dep > 0))
    if len(v) < 20:
        return None
    z = dep[v, u]
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    # The median, not the mean: a mask edge bleeding onto the wall behind
    # drags a mean by metres.
    return np.array([np.median(x), np.median(y), np.median(z)])


def ap_voc(scores, tps, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return float("nan"), 0.0, 0.0, 0.0
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[o]
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    b = int(np.argmax(f1))
    sorted_scores = np.asarray(scores)[o]
    return ap, float(prec[b]), float(rec[b]), float(sorted_scores[b])


def _cast(inp, dev, dtype):
    """Move a processor's batch to the device, floating tensors in `dtype`."""
    return {k: (v.to(dev, dtype) if torch.is_floating_point(v) else v.to(dev)) for k, v in inp.items()}


class Owl:
    name = "owlv2-base"

    def __init__(self, dev, dtype=torch.float32):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.p = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.m = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev, dtype).eval()
        self.dev = dev
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img, words):
        pil = Image.fromarray(img)
        inp = _cast(self.p(text=[[f"a photo of a {w}" for w in words]], images=pil, return_tensors="pt"),
                    self.dev, self.dtype)
        out = self.m(**inp)
        # OWLv2 pads to a square: boxes are relative to the padded side.
        s = max(img.shape[0], img.shape[1])
        res = self.p.post_process_grounded_object_detection(out, threshold=0.01, target_sizes=torch.tensor([[s, s]]))[0]
        dets = []
        for b, sc, l in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(),
                            res["labels"].cpu().numpy()):
            dets.append((words[int(l)], float(sc), b.astype(np.float64)))
        return dets


class GDino:
    name = "grounding-dino-tiny"

    def __init__(self, dev, dtype=torch.float32):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        mid = "IDEA-Research/grounding-dino-tiny"
        self.p = AutoProcessor.from_pretrained(mid)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev, dtype).eval()
        self.dev = dev
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img, words):
        pil = Image.fromarray(img)
        text = ". ".join(words) + "."
        inp = _cast(self.p(images=pil, text=text, return_tensors="pt"), self.dev, self.dtype)
        out = self.m(**inp)
        res = self.p.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=0.05, text_threshold=0.05,
            target_sizes=[img.shape[:2]],
        )[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = []
        for b, sc, l in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), labels):
            lw = str(l).strip()
            # A phrase can come back merged ("table sofa"): credit each word in it.
            for w in words:
                if w in lw.split():
                    dets.append((w, float(sc), b.astype(np.float64)))
        return dets


class Sam3:
    """SAM 3 (facebook/sam3, Meta's licence): promptable CONCEPT segmentation —
    a noun phrase in, every instance's mask, box and score out. Its OWN masks
    are scored (not SAM-base on its boxes). The image is encoded once and the
    four words are run against the cached vision embeddings."""

    name = "sam3"

    def __init__(self, dev, dtype=torch.float32):
        from transformers import Sam3Model, Sam3Processor
        self.p = Sam3Processor.from_pretrained("facebook/sam3")
        self.m = Sam3Model.from_pretrained("facebook/sam3").to(dev, dtype).eval()
        self.dev = dev
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img, words):
        pil = Image.fromarray(img)
        inp = self.p(images=pil, return_tensors="pt")
        pv = inp["pixel_values"].to(self.dev, self.dtype)
        ve = self.m.get_vision_features(pixel_values=pv)
        dets = []
        for w in words:
            ti = self.p(text=w, return_tensors="pt")
            out = self.m(vision_embeds=ve, input_ids=ti["input_ids"].to(self.dev),
                         attention_mask=ti["attention_mask"].to(self.dev))
            res = self.p.post_process_instance_segmentation(
                out, threshold=0.05, mask_threshold=0.5, target_sizes=[img.shape[:2]]
            )[0]
            for b, sc, mk in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(),
                                 res["masks"].cpu().numpy()):
                dets.append((w, float(sc), b.astype(np.float64), mk.astype(bool)))
        return dets


class Sam:
    def __init__(self, dev, dtype=torch.float32):
        from transformers import SamModel, SamProcessor
        self.p = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.m = SamModel.from_pretrained("facebook/sam-vit-base").to(dev, dtype).eval()
        self.dev = dev
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img, box):
        pil = Image.fromarray(img)
        inp = self.p(pil, input_boxes=[[[float(v) for v in box]]], return_tensors="pt")
        # MPS has no float64 (the processor builds some tensors in it): every
        # floating tensor goes in the model's dtype.
        inp = _cast(inp, self.dev, self.dtype)
        out = self.m(**inp, multimask_output=False)
        masks = self.p.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
        )
        return masks[0][0, 0].numpy().astype(bool)


def select(f, per_word, negatives, seed):
    rng = np.random.default_rng(seed)
    n = f["labels"].shape[0]
    present = {w: [] for w in WORDS}
    none = []
    for i in range(n):
        lab = f["labels"][i]
        any_ = False
        for w, c in WORDS.items():
            m = lab == c
            if m.mean() >= MIN_FRAC:
                present[w].append(i)
                any_ = True
        if not any_:
            none.append(i)
    chosen = set()
    for w in WORDS:
        p = present[w]
        chosen.update(rng.choice(p, size=min(per_word, len(p)), replace=False).tolist())
    chosen.update(rng.choice(none, size=min(negatives, len(none)), replace=False).tolist())
    counts = {w: len(present[w]) for w in WORDS}
    return sorted(chosen), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-word", type=int, default=120)
    ap.add_argument("--negatives", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N selected images")
    ap.add_argument("--detectors", default="owlv2,gdino", help="comma list of owlv2, gdino, sam3")
    ap.add_argument("--tag", default="", help="suffix for report.json (separate runs per detector)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = device()
    import h5py  # only the dataset needs it (the latency script imports this module)
    f = h5py.File(os.path.expanduser(a.mat), "r")
    idx, counts = select(f, a.per_word, a.negatives, a.seed)
    if a.limit:
        idx = idx[: a.limit]
    print(f"device {dev}; images with the word (>= {MIN_FRAC:.1%} of the frame): {counts}; evaluated {len(idx)}")
    words = list(WORDS)
    table = {"owlv2": Owl, "gdino": GDino, "sam3": Sam3}
    chosen = [table[d.strip()] for d in a.detectors.split(",")]
    sam = Sam(dev) if any(D is not Sam3 for D in chosen) else None
    report = {"images": len(idx), "present_counts": counts, "selection_seed": a.seed, "detectors": {}}
    for Det in chosen:
        det = Det(dev)
        recs = {w: {"scores": [], "tps": [], "n_gt": 0, "mask_iou": [], "err3d": []} for w in words}
        t_det = 0.0
        for k, i in enumerate(idx):
            img, lab, ins, dep = load(f, i)
            t0 = time.perf_counter()
            dets = det(img, words)
            t_det += time.perf_counter() - t0
            for w in words:
                gts = instances_of(lab, ins, WORDS[w])
                gt_boxes = [box_of(m) for m, _ in gts]
                recs[w]["n_gt"] += sum(1 for _, d in gts if not d)
                used = [False] * len(gts)
                best_match = None
                mine = [(d[1], d[2], d[3] if len(d) > 3 else None) for d in dets if d[0] == w]
                for sc, b, own in sorted(mine, key=lambda t: -t[0]):
                    j_best, o_best = -1, 0.5
                    for j, gb in enumerate(gt_boxes):
                        o = iou(b, gb)
                        if o >= o_best and not used[j]:
                            j_best, o_best = j, o
                    if j_best >= 0:
                        used[j_best] = True
                        if gts[j_best][1]:
                            continue  # difficult: neither TP nor FP
                        recs[w]["scores"].append(sc)
                        recs[w]["tps"].append(1)
                        if best_match is None:
                            best_match = (b, gts[j_best][0], own)
                    else:
                        recs[w]["scores"].append(sc)
                        recs[w]["tps"].append(0)
                if best_match is not None:
                    pm = best_match[2] if best_match[2] is not None else sam(img, best_match[0])
                    gm = best_match[1]
                    inter = (pm & gm).sum()
                    uni = (pm | gm).sum()
                    recs[w]["mask_iou"].append(inter / uni if uni else 0.0)
                    cp = centroid3d(pm, dep)
                    cg = centroid3d(gm, dep)
                    if cp is not None and cg is not None:
                        recs[w]["err3d"].append(float(np.linalg.norm(cp - cg)))
            if (k + 1) % 50 == 0:
                print(f"  {det.name}: {k + 1}/{len(idx)} images, {t_det / (k + 1) * 1000:.0f} ms/image")
        rows = {}
        for w in words:
            r = recs[w]
            apv, p, rc, thr = ap_voc(r["scores"], r["tps"], r["n_gt"])
            e = np.asarray(r["err3d"])
            rows[w] = {
                "n_gt": r["n_gt"], "AP50": apv, "P_at_F1": p, "R_at_F1": rc, "thr_at_F1": thr,
                "mask_iou_median": float(np.median(r["mask_iou"])) if r["mask_iou"] else float("nan"),
                "err3d_median_m": float(np.median(e)) if len(e) else float("nan"),
                "err3d_within_0.3m": float(np.mean(e < 0.3)) if len(e) else float("nan"),
                "n_3d": int(len(e)),
            }
        report["detectors"][det.name] = {"ms_per_image": t_det / max(1, len(idx)) * 1000, "words": rows}
        print(f"\n{det.name}  ({t_det / max(1, len(idx)) * 1000:.0f} ms/image on {dev})")
        print(f"  {'word':7s} {'GT':>4s} {'AP50':>6s} {'P@F1':>6s} {'R@F1':>6s} {'thr':>5s} {'maskIoU':>8s} {'3D med':>7s} {'<0.3m':>6s} {'n3D':>4s}")
        for w in words:
            x = rows[w]
            print(f"  {w:7s} {x['n_gt']:4d} {x['AP50']:6.3f} {x['P_at_F1']:6.3f} {x['R_at_F1']:6.3f} {x['thr_at_F1']:5.2f} "
                  f"{x['mask_iou_median']:8.3f} {x['err3d_median_m']:7.3f} {x['err3d_within_0.3m']:6.2f} {x['n_3d']:4d}")
        del det
        if dev == "mps":
            torch.mps.empty_cache()
    path = os.path.join(a.out, f"report{('_' + a.tag) if a.tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
