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
from detector_eval import WORDS, Owl, Sam, Sam3, device  # noqa: E402


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
        if name == "owlv2":
            det = Owl(dev, dtype)
            sam = Sam(dev, dtype)
        elif name == "sam3":
            det = Sam3(dev, dtype)
            sam = None
        else:
            raise SystemExit(f"unknown detector {name}")
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
