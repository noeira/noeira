#!/usr/bin/env python3
"""facebook/sam3's IMAGE DETECTOR alone, in bf16: what `noeira.perception.sam3`
loads on a machine without room for the 6.4 GB fp32 video checkpoint (the Orin).

    pixi run -e detect python tools/perception/extract_sam3_detector.py [--out ~/.cache/noeira/sam3/detector-bf16]

Keeps `detector_model.*` (848M parameters; the tracker, `tracker_*`, is dropped),
rounds each tensor to bf16 (nearest even), and writes `<out>/model.safetensors`
(~1.7 GB) with the checkpoint's `tokenizer.json` beside it. The Mojo loader widens
bf16 back to fp32 on upload. Rounding the weights moves the outputs: gate it
against the fp32 reference (`test_sam3_reference_gpu.mojo` reports the drift).
"""

import argparse
import glob
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/sam3/detector-bf16"))
    args = ap.parse_args()
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/"))[0]
    os.makedirs(args.out, exist_ok=True)
    out = {}
    with safe_open(os.path.join(snap, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("detector_model."):
                out[k] = f.get_tensor(k).to(torch.bfloat16).contiguous()
    save_file(out, os.path.join(args.out, "model.safetensors"))
    shutil.copy(os.path.join(snap, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    n = sum(t.numel() for t in out.values())
    print(f"{len(out)} tensors, {n / 1e6:.0f}M parameters -> {args.out}")


if __name__ == "__main__":
    main()
