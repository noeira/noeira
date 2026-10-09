"""The control for the GPT back-to-back crash: the same step, built and run
by Python, with LayerNorm and attention each as noeira's kernel or as MAX's
composite.

Each configuration runs in its own process (a CUDA fault is sticky): `--sync`
steps with the loss copied back after each, then `--extra` more, either back
to back (every output kept until one final synchronisation) or synchronised,
and prints a digest of every buffer. Back to back and synchronised must leave
the same digest.

    pixi run python -m noeira_max.train_from_mojo.probe_back_to_back [--device gpu]
        [--sync 50] [--extra 200] [--layers 2] [--only LN,ATTN,MODE]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

# The Mojo program's GPT (train_gpt.mojo): vocab, seq, dim, heads; batch.
VOCAB, SEQ, DIM, HEADS, BATCH = 65, 16, 64, 2, 4
LR = 3e-4


def one(device: str, ln: str, attn: str, mode: str, sync: int, extra: int, layers: int) -> None:
    from max.driver import CPU, Accelerator, Buffer
    from max.dtype import DType
    from max.graph import DeviceRef, TensorType

    from noeira_max.autodiff.models import gpt
    from noeira_max.autodiff.optim import AdamW
    from noeira_max.autodiff.train import CompiledStep, build_train_step
    from noeira_max.train_from_mojo import py_grad

    config = gpt.Config(vocab=VOCAB, seq=SEQ, dim=DIM, heads=HEADS, layers=layers,
                        layer_norm=ln, attention=attn)
    data = py_grad.gpt_problem([VOCAB, SEQ, DIM, HEADS, layers], BATCH, 0)
    dev_obj = Accelerator() if device == "gpu" else CPU()
    dev = DeviceRef.from_device(dev_obj)
    opt = AdamW(lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    step = build_train_step(
        lambda p, i, t: gpt.loss(p, i, t, config), data["init"], opt,
        [TensorType(DType.int64, data["x"].shape, dev), TensorType(DType.int64, data["y"].shape, dev)],
        dev, name=f"gpt_b2b_{ln}_{attn}",
    )
    compiled = CompiledStep(step, data["init"], opt, dev_obj)
    x = Buffer.from_numpy(data["x"]).to(dev_obj)
    y = Buffer.from_numpy(data["y"]).to(dev_obj)
    loss = [float(compiled(x, y)[0].to_numpy().item()) for _ in range(sync)]
    kept = []
    for _ in range(extra):
        out = compiled(x, y)
        if mode == "sync":
            out[0].to_numpy()
        else:
            kept.append(out)
    state = [*compiled.host_params().values(), *compiled.host_state().values()]  # synchronises
    print(f"{ln:9s} {attn:9s} {mode:4s} loss {loss[0]:.6f} -> {loss[-1]:.6f}; "
          f"{extra} more: digest {py_grad.digest(state)}; finite {all(np.isfinite(a).all() for a in state)}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sync", type=int, default=50)
    ap.add_argument("--extra", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--only", default="", help="LN,ATTN,MODE: run this one configuration here")
    a = ap.parse_args()
    if a.only:
        ln, attn, mode = a.only.split(",")
        one(a.device, ln, attn, mode, a.sync, a.extra, a.layers)
        return
    for ln in ("kernel", "composite"):
        for attn in ("kernel", "composite"):
            for mode in ("sync", "b2b"):
                cmd = [sys.executable, "-m", "noeira_max.train_from_mojo.probe_back_to_back",
                       "--device", a.device, "--sync", str(a.sync), "--extra", str(a.extra),
                       "--layers", str(a.layers), "--only", f"{ln},{attn},{mode}"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                lines = [l for l in r.stdout.splitlines() if l.strip()]
                if r.returncode == 0 and lines:
                    print(lines[-1], flush=True)
                else:
                    err = [l for l in (r.stdout + r.stderr).splitlines() if "rror" in l][:3]
                    print(f"{ln:9s} {attn:9s} {mode:4s} EXIT {r.returncode}: {' | '.join(err)[:400]}", flush=True)


if __name__ == "__main__":
    main()
