"""Detector LATENCY on a target device (the Orin, the laptop), on a few real photos.

    pixi run -e detect python tools/perception/detector_latency.py --export-nyu 20 \\
        --mat ~/.cache/noeira/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat --images DIR
    python tools/perception/detector_latency.py --images DIR --out FILE.json \\
        [--detectors owlv2,sam3] [--dtype fp32|fp16] [--warmup 3]

The ACCURACY is `detector_eval.py`'s (the laptop, 524 NYU frames); this measures
what the same models cost where they would run, and checks the target gives the
same answers: per image and word, the top detection's score and box are written
to `--out`, so two devices' files can be compared (`--compare A.json B.json`).

Timed per image, after `--warmup` images: OWLv2 (one forward for the four words)
and SAM-base on the best box of each word found — the pair a deployment would
run; and SAM 3 (the image encoded once, the four words against it). CUDA times
are synchronised; peak device memory is reported where the backend exposes it.

The real-time candidates (2026-11-11; `detector_eval.py`'s, in the detcand
venv): `yoloe-26{n,s,m,l}`, `yolo26{n,s,m}`, `rfdetr-seg-{n,s,m}` (their own
masks) and `omdet-turbo` (+ SAM-base on its best boxes, as OWLv2). `--dtype
fp16` runs the Ultralytics models with `half`; `--trt` exports them to a
TensorRT fp16 engine first (YOLOE with the four words baked in: the vocabulary
is fixed per engine) — what a Jetson deployment runs.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_eval import WORDS, Owl, Sam, Sam3, OmDet, RfDetrSeg, Yoloe, Yolo26, device  # noqa: E402


class _Ultra:
    """An Ultralytics model timed as deployed: fp16 (`half`), or a TensorRT fp16
    engine exported once (`trt`; YOLOE's text prompts baked in)."""

    def __init__(self, base, half, trt, words):
        from ultralytics import YOLO
        self.name = base.name + ("-trt" if trt else ("-fp16" if half else ""))
        self.base = base
        self.half = half
        self.engine = None
        if isinstance(base, Yoloe):
            base.m.set_classes(list(words), base.m.get_text_pe(list(words)))
            base.words = list(words)
        if trt:
            path = base.m.export(format="engine", half=True, imgsz=640, device=0, verbose=False)
            self.engine = YOLO(path, task="segment")

    def __call__(self, img, words):
        from detector_eval import _ultra_dets, COCO_OF
        m = self.engine if self.engine is not None else self.base.m
        r = m.predict(np.ascontiguousarray(img[..., ::-1]), conf=0.01, verbose=False, device=0,
                      half=self.half or self.engine is not None, retina_masks=True)[0]
        if isinstance(self.base, Yoloe):
            return _ultra_dets(r, lambda c: list(words)[c])
        back = {COCO_OF[w]: w for w in words if COCO_OF.get(w)}
        names = m.names
        return _ultra_dets(r, lambda c: back.get(names[c]))


def make(name, dev, dtype, trt, words):
    """(detector, SAM-base or None) for `name`."""
    if name == "owlv2":
        return Owl(dev, dtype), Sam(dev, dtype)
    if name == "sam3":
        return Sam3(dev, dtype), None
    if name == "omdet-turbo":
        return OmDet(dev, dtype), Sam(dev, dtype)
    if name.startswith("rfdetr-seg-"):
        return RfDetrSeg(dev, name[-1]), None
    if name.startswith("yoloe-26"):
        return _Ultra(Yoloe(dev, name[-1]), dtype == torch.float16, trt, words), None
    if name.startswith("yolo26"):
        return _Ultra(Yolo26(dev, name[-1]), dtype == torch.float16, trt, words), None
    raise SystemExit(f"unknown detector {name}")


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def export_nyu(mat, out_dir, n, seed=0):
    import h5py
    os.makedirs(out_dir, exist_ok=True)
    f = h5py.File(os.path.expanduser(mat), "r")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(f["images"].shape[0], size=n, replace=False).tolist())
    for i in idx:
        img = np.ascontiguousarray(f["images"][i].transpose(2, 1, 0))
        Image.fromarray(img).save(os.path.join(out_dir, f"nyu_{i:04d}.png"))
    print(f"exported {len(idx)} frames to {out_dir}")


def top_per_word(dets):
    best = {}
    for d in dets:
        w, sc, b = d[0], d[1], d[2]
        if w not in best or sc > best[w][0]:
            best[w] = (sc, [float(v) for v in b])
    return best


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    for det in a["answers"]:
        if det not in b["answers"]:
            continue
        n = same = 0
        dscore = []
        for img, wa in a["answers"][det].items():
            wb = b["answers"][det].get(img, {})
            for w in WORDS:
                ta = wa.get(w)
                tb = wb.get(w)
                ha = ta is not None and ta[0] >= 0.3
                hb = tb is not None and tb[0] >= 0.3
                n += 1
                if ha == hb:
                    same += 1
                if ta is not None and tb is not None:
                    dscore.append(abs(ta[0] - tb[0]))
        print(f"{det}: same found/not-found at score 0.3 in {same}/{n} (image, word) pairs; "
              f"top-score |diff| median {np.median(dscore):.4f}, max {np.max(dscore):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--detectors", default="owlv2,sam3")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--export-nyu", type=int, default=0)
    ap.add_argument("--mat")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--trt", action="store_true", help="Ultralytics models as TensorRT fp16 engines")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if a.export_nyu:
        export_nyu(a.mat, a.images, a.export_nyu)
        return
    dev = device()
    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    paths = sorted(glob.glob(os.path.join(a.images, "*.png")))
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    words = list(WORDS)
    report = {"device": dev, "dtype": a.dtype, "images": len(imgs), "torch": torch.__version__,
              "timings_ms": {}, "answers": {}}
    if dev == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    for name in a.detectors.split(","):
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_load = time.perf_counter()
        det, sam = make(name, dev, dtype, a.trt, words)
        name = getattr(det, "name", name)  # e.g. yoloe-26s-seg-trt
        load_s = time.perf_counter() - t_load
        t_det = []
        t_sam = []
        answers = {}
        for k, (p, img) in enumerate(zip(paths, imgs)):
            sync(dev)
            t0 = time.perf_counter()
            dets = det(img, words)
            sync(dev)
            t1 = time.perf_counter()
            best = top_per_word(dets)
            if sam is not None:
                for w, (sc, b) in best.items():
                    if sc >= 0.3:
                        _ = sam(img, np.asarray(b))
            sync(dev)
            t2 = time.perf_counter()
            answers[os.path.basename(p)] = best
            if k >= a.warmup:
                t_det.append((t1 - t0) * 1e3)
                t_sam.append((t2 - t1) * 1e3)
        row = {
            "load_s": load_s,
            "detector_ms_median": float(np.median(t_det)),
            "detector_ms_p90": float(np.percentile(t_det, 90)),
            "masks_ms_median": float(np.median(t_sam)) if sam is not None else 0.0,
            "total_ms_median": float(np.median(np.asarray(t_det) + np.asarray(t_sam))),
            "timed_images": len(t_det),
        }
        if dev == "cuda":
            row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        report["timings_ms"][name] = row
        report["answers"][name] = answers
        print(f"{name} [{dev} {a.dtype}]: detector {row['detector_ms_median']:.0f} ms (p90 "
              f"{row['detector_ms_p90']:.0f}), masks {row['masks_ms_median']:.0f} ms, total "
              f"{row['total_ms_median']:.0f} ms per image, 4 words"
              + (f", peak {row['peak_mem_gb']:.2f} GB" if 'peak_mem_gb' in row else "")
              + f", load {load_s:.0f} s")
        del det, sam
        if dev == "cuda":
            torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
