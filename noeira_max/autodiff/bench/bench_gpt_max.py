"""The torch twin's char-GPT recipe as ONE compiled MAX train step (plan M2).

Same recipe as ``tools/nn/torch_nn_reference.py gpt``: 6 layers x 384, 6
heads, sequence 256, batch 64, dropout 0.2, tied head, AdamW 1e-3
(0.9, 0.99) with weight decay 0.1 on the matrices but the positions, global
clip 1, warmup 100 + cosine to 0.1, pre-LN, tanh GELU. Forward, backward
(``value_and_grad``) and the AdamW update are one graph; parameters, moments,
the step counter and the RNG seed are buffers updated in place; the batch is
drawn in the graph from the device-resident corpus. Every call has the same
inputs, so the step can be captured and replayed.

Differences from the twin, by construction: the RNG streams (init, batches,
dropout) are not torch's; biases start at zero; attention is unfused
(softmax of the materialised scores), where the twin's SDPA is flash
attention — compare with ``torch_twin_math.py`` for the same math.

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/bench_gpt_max.py --device cpu --layers 1 ...
    AUTODIFF_ENV=nvidia noeira_max/autodiff/run.sh noeira_max/autodiff/bench/bench_gpt_max.py --mode capture

Prints a ``RESULT {json}`` line like the twin's.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, TensorType

from noeira_max.autodiff.models import data, gpt
from noeira_max.autodiff.optim import AdamW, WarmupCosine
from noeira_max.autodiff.train import CompiledStep, build_train_step

CAPTURE_KEY = 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--mode", choices=["execute", "capture"], default="execute")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=5000, help="schedule length")
    ap.add_argument("--bench-steps", type=int, default=50,
                    help="time this many steps after as many warmup steps (the twin's flag)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = Accelerator() if args.device == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    cfg = gpt.Config(seq=args.seq, dim=args.dim, heads=args.heads,
                     layers=args.layers, dropout=args.dropout)
    init = gpt.init(cfg, np.random.default_rng(args.seed), dtype=np.float32)
    _, tokens = data.shakespeare_vocab()
    corpus = tokens[: len(tokens) * 9 // 10]  # the twin's train split
    opt = AdamW(
        lr=1e-3, schedule=WarmupCosine(100, args.iters, 0.1), clip_norm=1.0,
        betas=(0.9, 0.99), weight_decay=0.1, decay=gpt.decays,
    )

    start = time.perf_counter()
    step = build_train_step(
        lambda p, c: gpt.loss(p, *gpt.sample_batch(c, args.batch, cfg.seq), cfg),
        init, opt, [TensorType(DType.int64, [len(corpus)], dev)], dev,
        uses_seed=True, name=f"gpt{args.layers}x{args.dim}_train_step",
    )
    build_s = time.perf_counter() - start
    compiled = CompiledStep(step, init, opt, device, seed=args.seed)
    corpus_buffer = Buffer.from_numpy(corpus).to(device)
    inputs = [*compiled.buffers, corpus_buffer]
    print(f"[max gpt] {args.layers}x{args.dim} on {device}: graph built in "
          f"{build_s:.1f} s, compiled in {compiled.compile_seconds:.1f} s", flush=True)

    out = dict(model="gpt", framework="max", mode=args.mode, device=str(device),
               layers=args.layers, dim=args.dim, batch=args.batch, seq=args.seq,
               dropout=args.dropout, build_s=build_s, compile_s=compiled.compile_seconds)

    last: list[Buffer] = []  # the loss of the last step run, in either mode

    def run_steps(n: int) -> None:
        if args.mode == "capture":
            for _ in range(n):
                compiled.model.replay(CAPTURE_KEY, *inputs)
            last[:] = outputs[:1]  # every replay writes the captured outputs
            return
        for _ in range(n):
            last[:] = compiled(corpus_buffer)[:1]

    # Warm up (the first execution, and the capture, happen here).
    first = compiled(corpus_buffer)[0].to_numpy().item()
    if args.mode == "capture":
        outputs = compiled.model.capture(CAPTURE_KEY, *inputs)
    run_steps(max(0, args.bench_steps - 1))
    device.synchronize()

    start = time.perf_counter()
    run_steps(args.bench_steps)
    device.synchronize()
    elapsed = time.perf_counter() - start
    out["ms_per_step"] = 1000.0 * elapsed / args.bench_steps

    # Per-step latency, synchronised every step (no pipelining across steps).
    samples = []
    for _ in range(min(20, args.bench_steps)):
        t = time.perf_counter()
        run_steps(1)
        device.synchronize()
        samples.append(1000.0 * (time.perf_counter() - t))
    out["median_synced_ms"] = statistics.median(samples)
    out["first_loss"] = first
    # The same step count in both modes: execute and capture must agree here.
    out["loss"] = last[0].to_numpy().item()
    # The device counter proves the replays ran the step: anything short of
    # the executions issued means a replay did nothing. (The capture itself
    # records without running the step: on the 5090 both modes count 420/420.)
    out["steps_done"] = int(compiled.host_state()["step"].item())
    out["steps_issued"] = 1 + max(0, args.bench_steps - 1) + args.bench_steps + len(samples)
    print(f"steady train step {out['ms_per_step']:.3f} ms over {args.bench_steps} steps "
          f"(median synced {out['median_synced_ms']:.3f} ms); loss {first:.4f} -> "
          f"{out['loss']:.4f} after {out['steps_done']} steps", flush=True)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
