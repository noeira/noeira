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

Real-time candidates (2026-11-11, `--detectors`; they need `ultralytics` and
`rfdetr`, kept OUT of the pixi env: a venv on top of it,
`python -m venv --system-site-packages ~/.cache/noeira/venvs/detcand` from the
`detect` env's python, then `pip install ultralytics rfdetr`):
  yoloe-26{n,s,m,l}   YOLOE-26-seg (Ultralytics, AGPL-3.0): open vocabulary, the
                      four words as text prompts, its own masks
  yolo26{n,s,m}       YOLO26-seg (AGPL-3.0), COCO's 80 classes: fridge ->
                      refrigerator, table -> dining table, sofa -> couch; NO door
  rfdetr-seg-{n,s,m}  RF-DETR-Seg (Roboflow, Apache-2.0), COCO classes, as yolo26
  omdet-turbo         OmDet-Turbo Swin-T (Apache-2.0): open vocabulary, BOXES —
                      masked by SAM-base like OWLv2
Each candidate's input is the RGB frame as is; scores down to 0.01 for AP.

Box-only places (`err3d_box`): beside the mask's place, the median of the depth
points in the CENTRAL HALF of the matched box (no mask at all) — does a robot's
place need masks, or does the box's middle do?

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


def _nms_on_cpu():
    """The env's torchvision has no MPS kernel for NMS (YOLO26's and OmDet's
    post-processing): run it on the CPU (a few hundred boxes) and hand the
    indices back on the boxes' device."""
    import torchvision
    from torchvision.ops import boxes as tb
    nms = tb.nms
    if getattr(nms, "_on_cpu", False):
        return

    def cpu_nms(b, s, t):
        return nms(b.float().cpu(), s.float().cpu(), t).to(b.device)  # (the CPU kernel has no fp16)

    cpu_nms._on_cpu = True
    tb.nms = cpu_nms
    torchvision.ops.nms = cpu_nms


COCO_OF = {"fridge": "refrigerator", "table": "dining table", "sofa": "couch", "door": None}


class Yoloe:
    """YOLOE-26-seg, the words as text prompts (embedded once per word list)."""

    def __init__(self, dev, size="s"):
        from ultralytics import YOLOE
        _nms_on_cpu()
        self.name = f"yoloe-26{size}-seg"
        self.m = YOLOE(os.path.expanduser(f"~/.cache/noeira/weights/ultra/yoloe-26{size}-seg.pt"))
        self.dev = dev
        self.words = None

    def __call__(self, img, words):
        if self.words != list(words):
            self.m.set_classes(list(words), self.m.get_text_pe(list(words)))
            self.words = list(words)
        # ultralytics reads a numpy image as BGR (OpenCV)
        r = self.m.predict(np.ascontiguousarray(img[..., ::-1]), conf=0.01, verbose=False, device=self.dev,
                           retina_masks=True)[0]
        return _ultra_dets(r, lambda c: self.words[c])


def _ultra_dets(r, word_of):
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    boxes = r.boxes.xyxy.float().cpu().numpy()
    conf = r.boxes.conf.float().cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
    for k in range(len(boxes)):
        w = word_of(int(cls[k]))
        if w is None:
            continue
        mk = masks[k] if masks is not None else None
        dets.append((w, float(conf[k]), boxes[k].astype(np.float64), mk))
    return dets


class Yolo26:
    """YOLO26-seg on COCO's classes (`COCO_OF`); no door."""

    def __init__(self, dev, size="s"):
        from ultralytics import YOLO
        _nms_on_cpu()
        self.name = f"yolo26{size}-seg"
        self.m = YOLO(os.path.expanduser(f"~/.cache/noeira/weights/ultra/yolo26{size}-seg.pt"))
        self.dev = dev

    def __call__(self, img, words):
        names = self.m.names
        back = {COCO_OF[w]: w for w in words if COCO_OF.get(w)}
        r = self.m.predict(np.ascontiguousarray(img[..., ::-1]), conf=0.01, verbose=False, device=self.dev,
                           retina_masks=True)[0]
        return _ultra_dets(r, lambda c: back.get(names[c]))


class RfDetrSeg:
    """RF-DETR-Seg on COCO's classes (`COCO_OF`); no door."""

    def __init__(self, dev, size="s"):
        import rfdetr
        cls = {"n": rfdetr.RFDETRSegNano, "s": rfdetr.RFDETRSegSmall, "m": rfdetr.RFDETRSegMedium}[size]
        self.name = f"rfdetr-seg-{size}"
        # RFDETR_DEVICE=cpu: the CPU reference (MPS samples the deformable attention with other rounding)
        rd = os.environ.get("RFDETR_DEVICE")
        self.m = cls(device=rd) if rd else cls()
        if rd:
            self.name += "-" + rd
        from rfdetr.assets.coco_classes import COCO_CLASSES  # {COCO id: name}
        self.names = COCO_CLASSES

    def __call__(self, img, words):
        back = {COCO_OF[w]: w for w in words if COCO_OF.get(w)}
        d = self.m.predict(Image.fromarray(img), threshold=0.01)
        dets = []
        for k in range(len(d.xyxy)):
            w = back.get(self.names.get(int(d.class_id[k]), ""))
            if w is None:
                continue
            mk = d.mask[k].astype(bool) if d.mask is not None else None
            dets.append((w, float(d.confidence[k]), d.xyxy[k].astype(np.float64), mk))
        return dets


class OmDet:
    """OmDet-Turbo Swin-T: open vocabulary, boxes (SAM-base masks them)."""

    name = "omdet-turbo"

    def __init__(self, dev, dtype=torch.float32):
        from transformers import AutoProcessor, OmDetTurboForObjectDetection
        _nms_on_cpu()
        mid = "omlab/omdet-turbo-swin-tiny-hf"
        self.p = AutoProcessor.from_pretrained(mid)
        self.m = OmDetTurboForObjectDetection.from_pretrained(mid).to(dev, dtype).eval()
        self.dev = dev
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img, words):
        pil = Image.fromarray(img)
        inp = _cast(self.p(pil, text=list(words), return_tensors="pt"), self.dev, self.dtype)
        out = self.m(**inp)
        res = self.p.post_process_grounded_object_detection(
            out, text_labels=[list(words)], target_sizes=[img.shape[:2]], threshold=0.01, nms_threshold=0.5,
        )[0]
        labels = res.get("text_labels", res.get("classes"))
        dets = []
        for b, sc, l in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), labels):
            if str(l) in words:
                dets.append((str(l), float(sc), b.astype(np.float64)))
        return dets


class Precomputed:
    """Detections a separate program wrote (`--detectors precomputed:<dir>`):
    `<dir>/nyu_<i>.det` per frame, from `--export-frames` (the Mojo RF-DETR,
    `tools/perception/rfdetr_frames.mojo`). Format: `[u32 count]` then per
    detection `[u32 COCO id][f32 score][f32 x0 y0 x1 y1][u32 h][u32 w][mask bits,
    row-major, LSB first]`; the COCO ids map back to the words (`COCO_OF`)."""

    def __init__(self, dev, path=""):
        self.dir = os.path.expanduser(path)
        self.name = "precomputed-" + os.path.basename(self.dir.rstrip("/"))
        self.frame = None
        from rfdetr.assets.coco_classes import COCO_CLASSES
        self.id_of = {name: cid for cid, name in COCO_CLASSES.items()}

    def __call__(self, img, words):
        b = open(os.path.join(self.dir, f"nyu_{self.frame:04d}.det"), "rb").read()
        word_of = {self.id_of[COCO_OF[w]]: w for w in words if COCO_OF.get(w)}
        n = int(np.frombuffer(b, "<u4", 1, 0)[0])
        at, dets = 4, []
        for _ in range(n):
            cid = int(np.frombuffer(b, "<u4", 1, at)[0])
            sc = float(np.frombuffer(b, "<f4", 1, at + 4)[0])
            box = np.frombuffer(b, "<f4", 4, at + 8).astype(np.float64)
            h, w = (int(v) for v in np.frombuffer(b, "<u4", 2, at + 24))
            nb = (h * w + 7) // 8
            bits = np.unpackbits(np.frombuffer(b, np.uint8, nb, at + 32), bitorder="little")[: h * w]
            at += 32 + nb
            if cid in word_of:
                dets.append((word_of[cid], sc, box, bits.reshape(h, w).astype(bool)))
        return dets


class RfDetrFt:
    """A FINE-TUNED RF-DETR-Seg-Small (`--detectors rfdetr-ft:<checkpoint>`,
    `finetune_rfdetr.py`): its own classes, named in the checkpoint; a word
    scores the detections of the class of the same name."""

    def __init__(self, dev, path=""):
        import rfdetr
        p = os.path.expanduser(path)
        self.name = "rfdetr-ft-" + os.path.basename(os.path.dirname(p))
        rd = os.environ.get("RFDETR_DEVICE")
        self.m = rfdetr.RFDETRSegSmall(pretrain_weights=p, **({"device": rd} if rd else {}))
        names = self.m.class_names
        self.names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)

    def __call__(self, img, words):
        d = self.m.predict(Image.fromarray(img), threshold=0.01)
        dets = []
        for k in range(len(d.xyxy)):
            w = self.names.get(int(d.class_id[k]), "")
            if w not in words:
                continue
            mk = d.mask[k].astype(bool) if d.mask is not None else None
            dets.append((w, float(d.confidence[k]), d.xyxy[k].astype(np.float64), mk))
        return dets


def box_centre_mask(box, shape):
    """The central half of a box (each side shrunk by a quarter of its extent)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    m = np.zeros(shape, dtype=bool)
    a, b = int(max(0, y0 + h / 4)), int(min(shape[0], y1 - h / 4))
    c, d = int(max(0, x0 + w / 4)), int(min(shape[1], x1 - w / 4))
    m[a:b, c:d] = True
    return m


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
    ap.add_argument("--detectors", default="owlv2,gdino",
                    help="comma list of owlv2, gdino, sam3, omdet-turbo, yoloe-26{n,s,m,l}, yolo26{n,s,m}, rfdetr-seg-{n,s,m}")
    ap.add_argument("--tag", default="", help="suffix for report.json (separate runs per detector)")
    ap.add_argument("--export-frames", default="",
                    help="write the selected frames as <dir>/nyu_<i>.npy (uint8 HWC) + frames.txt, then stop")
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
    if a.export_frames:
        d = os.path.expanduser(a.export_frames)
        os.makedirs(d, exist_ok=True)
        for i in idx:
            img, _, _, _ = load(f, i)
            np.save(os.path.join(d, f"nyu_{i:04d}.npy"), img)
        with open(os.path.join(d, "frames.txt"), "w") as fh:
            fh.write("\n".join(str(i) for i in idx) + "\n")
        print(f"exported {len(idx)} frames to {d}")
        return
    from functools import partial
    table = {"owlv2": Owl, "gdino": GDino, "sam3": Sam3, "omdet-turbo": OmDet}
    for z in "nsml":
        table[f"yoloe-26{z}"] = partial(Yoloe, size=z)
    for z in "nsm":
        table[f"yolo26{z}"] = partial(Yolo26, size=z)
        table[f"rfdetr-seg-{z}"] = partial(RfDetrSeg, size=z)
    def _pick(d):
        if d.startswith("precomputed:"):
            return partial(Precomputed, path=d.split(":", 1)[1])
        if d.startswith("rfdetr-ft:"):
            return partial(RfDetrFt, path=d.split(":", 1)[1])
        return table[d.strip()]

    chosen = [_pick(d) for d in a.detectors.split(",")]
    sam = Sam(dev) if any(D in (Owl, GDino, OmDet) for D in chosen) else None
    report = {"images": len(idx), "present_counts": counts, "selection_seed": a.seed, "detectors": {}}
    for Det in chosen:
        det = Det(dev)
        recs = {w: {"scores": [], "tps": [], "n_gt": 0, "mask_iou": [], "err3d": [], "err3d_box": []} for w in words}
        t_det = 0.0
        for k, i in enumerate(idx):
            img, lab, ins, dep = load(f, i)
            if hasattr(det, "frame"):
                det.frame = i
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
                    cb = centroid3d(box_centre_mask(best_match[0], dep.shape), dep)
                    if cb is not None and cg is not None:
                        recs[w]["err3d_box"].append(float(np.linalg.norm(cb - cg)))
            if (k + 1) % 50 == 0:
                print(f"  {det.name}: {k + 1}/{len(idx)} images, {t_det / (k + 1) * 1000:.0f} ms/image")
        rows = {}
        for w in words:
            r = recs[w]
            apv, p, rc, thr = ap_voc(r["scores"], r["tps"], r["n_gt"])
            e = np.asarray(r["err3d"])
            eb = np.asarray(r["err3d_box"])
            rows[w] = {
                "n_gt": r["n_gt"], "AP50": apv, "P_at_F1": p, "R_at_F1": rc, "thr_at_F1": thr,
                "mask_iou_median": float(np.median(r["mask_iou"])) if r["mask_iou"] else float("nan"),
                "err3d_median_m": float(np.median(e)) if len(e) else float("nan"),
                "err3d_within_0.3m": float(np.mean(e < 0.3)) if len(e) else float("nan"),
                "n_3d": int(len(e)),
                "err3d_box_median_m": float(np.median(eb)) if len(eb) else float("nan"),
                "err3d_box_within_0.3m": float(np.mean(eb < 0.3)) if len(eb) else float("nan"),
            }
        report["detectors"][det.name] = {"ms_per_image": t_det / max(1, len(idx)) * 1000, "words": rows}
        print(f"\n{det.name}  ({t_det / max(1, len(idx)) * 1000:.0f} ms/image on {dev})")
        print(f"  {'word':7s} {'GT':>4s} {'AP50':>6s} {'P@F1':>6s} {'R@F1':>6s} {'thr':>5s} {'maskIoU':>8s} {'3D med':>7s} {'<0.3m':>6s} {'n3D':>4s} {'box3D':>6s} {'<0.3m':>6s}")
        for w in words:
            x = rows[w]
            print(f"  {w:7s} {x['n_gt']:4d} {x['AP50']:6.3f} {x['P_at_F1']:6.3f} {x['R_at_F1']:6.3f} {x['thr_at_F1']:5.2f} "
                  f"{x['mask_iou_median']:8.3f} {x['err3d_median_m']:7.3f} {x['err3d_within_0.3m']:6.2f} {x['n_3d']:4d} "
                  f"{x['err3d_box_median_m']:6.3f} {x['err3d_box_within_0.3m']:6.2f}")
        del det
        if dev == "mps":
            torch.mps.empty_cache()
    path = os.path.join(a.out, f"report{('_' + a.tag) if a.tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
