"""Compile time against depth: the GPT forward alone (loss only)
and the whole train step (forward, backward, AdamW).

MAX caches compiled models by graph BODY (the graph's name is not in the key)
and also caches compiled kernels. Each graph here therefore carries a fresh
constant (a nonce, multiplied by zero), so every compile is of a new graph;
the kernel cache stays warm after the first one, which is the cost of
iterating on a model. The first compile of a process also pays for the
kernels themselves, so the first row is reported separately.

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/compile_scaling.py --device cpu --depths 1 2
    AUTODIFF_ENV=nvidia noeira_max/autodiff/run.sh noeira_max/autodiff/bench/compile_scaling.py
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff.models import gpt
from noeira_max.autodiff.optim import AdamW, WarmupCosine
from noeira_max.autodiff.train import build_train_step


def with_nonce(loss_fn, dev: DeviceRef):
    """``loss_fn`` plus ``nonce * 0``: the same value, a new graph body."""
    nonce = random.random()

    def wrapped(*args):
        loss = loss_fn(*args)
        return loss + ops.constant(nonce, DType.float32, dev) * 0.0

    wrapped.nonce = nonce
    return wrapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 12])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = Accelerator() if args.device == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])
    tokens = TensorType(DType.int64, [args.batch, args.seq], dev)
    rows = []
    for depth in args.depths:
        cfg = gpt.Config(seq=args.seq, dim=args.dim, heads=args.heads, layers=depth)
        init = gpt.init(cfg, np.random.default_rng(0), dtype=np.float32)
        names = list(init)
        param_types = [TensorType(DType.float32, init[n].shape, dev) for n in names]
        loss = with_nonce(lambda p, x, y: gpt.loss(p, x, y, cfg), dev)

        with Graph(f"forward{depth}", input_types=param_types + [tokens, tokens]) as fwd:
            p = dict(zip(names, fwd.inputs[: len(names)]))
            fwd.output(loss(p, *fwd.inputs[len(names):]))
        opt = AdamW(lr=1e-3, schedule=WarmupCosine(100, 5000, 0.1), clip_norm=1.0,
                    betas=(0.9, 0.99), weight_decay=0.1, decay=gpt.decays)
        start = time.perf_counter()
        step = build_train_step(loss, init, opt, [tokens, tokens], dev, name=f"step{depth}")
        build_s = time.perf_counter() - start

        row = {"layers": depth, "build_step_s": build_s}
        for kind, graph in (("forward", fwd), ("train_step", step.graph)):
            start = time.perf_counter()
            session.compile(graph)
            row[f"{kind}_compile_s"] = time.perf_counter() - start
        rows.append(row)
        print(f"  {depth:2d} layers: forward {row['forward_compile_s']:6.1f} s, "
              f"train step {row['train_step_compile_s']:6.1f} s "
              f"(graph built in {build_s:.1f} s)", flush=True)
    print("RESULT " + json.dumps({
        "device": str(device), "dim": args.dim,
        "note": "row 0 includes first-use kernel compilation", "rows": rows,
    }))


if __name__ == "__main__":
    main()
