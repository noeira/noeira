"""One MLP train step (forward, backward, Adam) as one compiled MAX graph, at
the shapes of noeira's RL networks: what MAX compiles a small training step
into, and what it costs per step.

- ``--shape sac``: a SAC critic, 23 -> 256 -> 256 -> 1 (observation 17 and
  action 6), ReLU, batch 256;
- ``--shape ppo``: a PPO network, 8 -> 64 -> 64 -> 4, tanh, batch 64;
- or ``--dims 23,256,256,1 --act relu --batch 256``.

The loss is the mean squared error against a fixed target, so the step has a
training graph's structure (GEMMs forward and backward, the activation and its
derivative, the loss terms, the update of every parameter) without an RL
algorithm around it. Adam is ``AdamW`` with no weight decay, in float32, with
its step counter in a device buffer. Every step takes the same inputs, so
``--mode capture`` records the step once as a CUDA graph and replays it.

Prints one ``RESULT {json}`` line, also written to ``--json``:

- ``compile_s``: the compile time. ``--cold`` adds a constant that makes the
  graph new to MAX's compile cache; without it, a graph compiled before
  loads in about a second.
- ``ms_per_step`` (steps back to back) and ``median_synced_ms``.
- ``first_loss_rel_err``: the first step's loss against a float64 NumPy
  forward of the same weights and batch. About 1e-7 means float32 GEMMs,
  1e-5 to 1e-3 TF32 somewhere, and anything larger a wrong result.
- ``kernels``: the graph's kernels as MAX lists them
  (``Model.kernel_summaries``), which shows what it fused: an
  ``Epilogue(mo.matmul, ...)`` entry is a GEMM with elementwise work in its
  epilogue.

Kernels per step, their GEMM paths, and memsets come from nsys, through
``nsys_summary.py``. Both shapes, both modes and the profiles, on one GPU:

    noeira_max/autodiff/bench/run_5090.sh mlp

One run:

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/mlp_step.py --shape sac --mode capture
    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/mlp_step.py --shape ppo --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops

from noeira_max.autodiff._ops import sum_all
from noeira_max.autodiff.models import mlp
from noeira_max.autodiff.models.common import dense
from noeira_max.autodiff.optim import AdamW
from noeira_max.autodiff.train import CompiledStep, build_train_step

# name: (layer widths, hidden activation, batch)
SHAPES = {
    "sac": ((23, 256, 256, 1), "relu", 256),
    "ppo": ((8, 64, 64, 4), "tanh", 64),
}
CAPTURE_KEY = 11


def forward(p: dict, x: TensorValue, depth: int, act: str) -> TensorValue:
    h = x
    for i in range(depth):
        h = dense(h, p[f"w{i}"], p[f"b{i}"])
        if i < depth - 1:
            h = ops.relu(h) if act == "relu" else ops.tanh(h)
    return h


def mse(p: dict, x: TensorValue, y: TensorValue, depth: int, act: str) -> TensorValue:
    d = forward(p, x, depth, act) - y
    return sum_all(d * d) * (1.0 / (int(y.shape[0]) * int(y.shape[1])))


def numpy_mse(p: dict, x: np.ndarray, y: np.ndarray, depth: int, act: str) -> float:
    """The same loss in float64, from the same float32 weights and batch."""
    h = x.astype(np.float64)
    for i in range(depth):
        h = h @ p[f"w{i}"].astype(np.float64) + p[f"b{i}"].astype(np.float64)
        if i < depth - 1:
            h = np.maximum(h, 0.0) if act == "relu" else np.tanh(h)
    return float(np.mean((h - y.astype(np.float64)) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shape", choices=sorted(SHAPES), default="sac")
    ap.add_argument("--dims", help="layer widths, e.g. 23,256,256,1 (overrides the shape's)")
    ap.add_argument("--act", choices=["relu", "tanh"], help="hidden activation (overrides the shape's)")
    ap.add_argument("--batch", type=int, help="batch size (overrides the shape's)")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--mode", choices=["execute", "capture"], default="execute")
    ap.add_argument("--bench-steps", type=int, default=200,
                    help="time this many steps after as many warmup steps")
    ap.add_argument("--cold", action="store_true", help="make the graph new to MAX's compile cache")
    ap.add_argument("--json", type=Path, help="also write the RESULT here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dims, act, batch = SHAPES[args.shape]
    if args.dims:
        dims = tuple(int(d) for d in args.dims.split(","))
    act = args.act or act
    batch = args.batch or batch
    depth = len(dims) - 1

    device = Accelerator() if args.device == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    rng = np.random.default_rng(args.seed)
    init = mlp.init(rng, dims=dims, dtype=np.float32)
    x = rng.standard_normal((batch, dims[0])).astype(np.float32)
    y = rng.standard_normal((batch, dims[-1])).astype(np.float32)
    opt = AdamW(lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    nonce = random.random()

    def loss_fn(p, xv, yv):  # noqa: ANN001, ANN202
        loss = mse(p, xv, yv, depth, act)
        if args.cold:  # the same value, a graph body MAX's cache has not seen
            loss = loss + ops.constant(nonce, DType.float32, dev) * 0.0
        return loss

    name = f"mlp_{'x'.join(map(str, dims))}_{act}_b{batch}_train_step"
    start = time.perf_counter()
    step = build_train_step(
        loss_fn, init, opt,
        [TensorType(DType.float32, x.shape, dev), TensorType(DType.float32, y.shape, dev)],
        dev, name=name,
    )
    build_s = time.perf_counter() - start
    compiled = CompiledStep(step, init, opt, device)
    x_buffer = Buffer.from_numpy(x).to(device)
    y_buffer = Buffer.from_numpy(y).to(device)
    inputs = [*compiled.buffers, x_buffer, y_buffer]
    print(f"[max mlp] {'-'.join(map(str, dims))} {act} batch {batch} on {device}: graph built in "
          f"{build_s:.2f} s, compiled in {compiled.compile_seconds:.1f} s", flush=True)

    last: list[Buffer] = []  # the loss of the last step run, in either mode

    def run_steps(n: int) -> None:
        if args.mode == "capture":
            for _ in range(n):
                compiled.model.replay(CAPTURE_KEY, *inputs)
            last[:] = outputs[:1]  # every replay writes the captured outputs
            return
        for _ in range(n):
            last[:] = compiled(x_buffer, y_buffer)[:1]

    # The first step runs from the initial weights: its loss is checked.
    first = compiled(x_buffer, y_buffer)[0].to_numpy().item()
    reference = numpy_mse(init, x, y, depth, act)
    if args.mode == "capture":
        try:
            outputs = compiled.model.capture(CAPTURE_KEY, *inputs)
        except Exception as e:  # CPU and Metal have no device graphs
            raise SystemExit(f"capture failed on {device}: {e}") from e
    run_steps(max(0, args.bench_steps - 1))
    device.synchronize()
    start = time.perf_counter()
    run_steps(args.bench_steps)
    device.synchronize()
    elapsed = time.perf_counter() - start
    samples = []
    for _ in range(min(200, args.bench_steps)):
        t = time.perf_counter()
        run_steps(1)
        device.synchronize()
        samples.append(1000.0 * (time.perf_counter() - t))

    kernels = Counter(k for k in compiled.model.kernel_summaries if not k.startswith("index."))
    out = dict(
        model="mlp", framework="max", mode=args.mode, device=str(device),
        dims=list(dims), act=act, batch=batch, cold=args.cold,
        build_s=build_s, compile_s=compiled.compile_seconds,
        ms_per_step=1000.0 * elapsed / args.bench_steps,
        median_synced_ms=statistics.median(samples),
        first_loss=first, numpy_loss=reference,
        first_loss_rel_err=abs(first - reference) / abs(reference),
        loss=last[0].to_numpy().item(),
        # The device counter proves the replays ran the step.
        steps_done=int(compiled.host_state()["step"].item()),
        steps_issued=1 + max(0, args.bench_steps - 1) + args.bench_steps + len(samples),
        kernel_count=sum(kernels.values()),
        kernels=dict(sorted(kernels.items())),
    )
    print(f"steady train step {out['ms_per_step']:.4f} ms over {args.bench_steps} steps "
          f"(median synced {out['median_synced_ms']:.4f} ms); first loss {first:.6f} "
          f"(float64 NumPy {reference:.6f}, relative error {out['first_loss_rel_err']:.1e}); "
          f"{out['kernel_count']} kernels in the graph; {out['steps_done']} steps done", flush=True)
    for kernel, count in out["kernels"].items():
        print(f"  {count:3d} x {kernel}", flush=True)
    line = json.dumps(out)
    print("RESULT " + line, flush=True)
    if args.json:
        args.json.write_text(line + "\n")


if __name__ == "__main__":
    main()
