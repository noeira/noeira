"""An RF-DETR checkpoint (`rf-detr-seg-small.pt`, Roboflow, Apache-2.0) as the
safetensors file `noeira.perception.rfdetr` loads.

    pixi run -e detect python -I tools/perception/extract_rfdetr.py \\
        [--pt ~/.roboflow/models/rf-detr-seg-small.pt] [--out ~/.cache/noeira/rfdetr/seg-small/model.safetensors]

The checkpoint is a pickle: read with `weights_only=True` (tensors only, no code
runs). Its `model` state dict is written as is (fp32, the same key names), the
training-only parts included — the loader takes what inference reads. The
`rfdetr` package downloads the .pt on first use (`RFDETRSegSmall()`); this tool
does not.
"""

import argparse
import os

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.expanduser("~/.roboflow/models/rf-detr-seg-small.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/rfdetr/seg-small/model.safetensors"))
    a = ap.parse_args()
    ck = torch.load(a.pt, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    out = {k: v.detach().to(torch.float32).contiguous() for k, v in sd.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_file(out, a.out)
    n = sum(v.numel() for v in out.values())
    print(f"wrote {a.out}: {len(out)} tensors, {n / 1e6:.1f} M parameters")


if __name__ == "__main__":
    main()
