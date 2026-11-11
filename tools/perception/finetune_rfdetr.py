"""Fine-tune RF-DETR-Seg-Small on a COCO-format set (`nyu_to_coco.py`) with
Roboflow's own trainer — the training stays in Python, the inference is the Mojo
port (`noeira.perception.rfdetr`).

    pixi run -e detect ~/.cache/noeira/venvs/detcand/bin/python tools/perception/finetune_rfdetr.py \\
        --data ~/.cache/noeira/rfdetr/nyu_coco --out ~/.cache/noeira/rfdetr/ft_nyu3 \\
        [--epochs 1] [--batch 4] [--accum 4] [--device mps|cuda|cpu] [--lr 1e-4]

Starts from the COCO checkpoint (`rf-detr-seg-small.pt`); the class head is
re-sized to the set's categories (`rfdetr` aligns `num_classes` to them), so the
fine-tuned model knows ONLY those classes — COCO's 80 are not kept unless the set
contains them. ⚠ On the 16 GB laptop: ONE GPU process at a time.
"""

import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    from rfdetr import RFDETRSegSmall

    m = RFDETRSegSmall()
    kw = dict(dataset_dir=os.path.expanduser(a.data), output_dir=os.path.expanduser(a.out), epochs=a.epochs,
              batch_size=a.batch, grad_accum_steps=a.accum, lr=a.lr)
    if a.device:
        kw["device"] = a.device
    t0 = time.time()
    m.train(**kw)
    print(f"trained {a.epochs} epoch(s) in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
