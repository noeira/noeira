"""Exports a train step for the Mojo driver (plan §4, M3).

Writes, into OUT_DIR:

- ``step.mef``: the compiled train step (forward, backward, AdamW).
- ``step.inputs``: every input's initial bytes, in model input order
  (parameters, optimizer state, seed, data), each at a 64-byte boundary.
- ``step.manifest``: little-endian int64s: ``n``, the step counter's input
  index (``-1`` if none), then per input ``dtype rank dims... offset
  nbytes``, dtypes as the C API's ``M_Dtype``.
- ``python_reference.txt``: the same compiled model driven from Python for
  ``--steps`` steps from the same bytes: one loss per line, then the median
  step time. The Mojo driver must reproduce the losses.

    noeira_max/autodiff/run.sh noeira_max/autodiff/capi/export_step.py OUT_DIR [--tiny]

``--tiny`` exports a one-buffer step (``b += x``) instead: the per-call floor
of each driver, with no model in it.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff.models import data, gpt
from noeira_max.autodiff.optim import AdamW, WarmupCosine
from noeira_max.autodiff.train import build_train_step

# The C API's M_Dtype values (include/max/c/types.h).
M_DTYPE = {
    np.dtype(np.float32): 17 | (1 << 6),
    np.dtype(np.float64): 18 | (1 << 6),
    np.dtype(np.int64): (6 << 1) | (1 << 7) | 1,
    np.dtype(np.uint64): (6 << 1) | (1 << 7),
}
ALIGN = 64


def gpt_step(args, dev: DeviceRef):
    """The bench recipe at CPU size: the step samples its own batch."""
    cfg = gpt.Config(seq=args.seq, dim=args.dim, heads=args.heads,
                     layers=args.layers, dropout=0.1)
    params = gpt.init(cfg, np.random.default_rng(0), dtype=np.float32)
    _, tokens = data.shakespeare_vocab()
    corpus = tokens[: len(tokens) * 9 // 10]
    opt = AdamW(lr=3e-3, schedule=WarmupCosine(20, 1000, 0.1), clip_norm=1.0,
                betas=(0.9, 0.99), weight_decay=0.1, decay=gpt.decays)
    step = build_train_step(
        lambda p, c: gpt.loss(p, *gpt.sample_batch(c, args.batch, cfg.seq), cfg),
        params, opt, [TensorType(DType.int64, [len(corpus)], dev)], dev,
        uses_seed=True, name="capi_gpt_step",
    )
    state = opt.init(params)
    inputs = [params[n] for n in step.param_names] + [state[n] for n in step.state_names]
    counter = len(step.param_names) + step.state_names.index("step")
    inputs += [np.array([1234], np.uint64), corpus]
    return step.graph, inputs, counter


def tiny_step(dev: DeviceRef):
    """One in-place update, ``b += x``, and ``sum(b)`` out."""
    types = [BufferType(DType.float32, [8], dev), TensorType(DType.float32, [8], dev)]
    with Graph("capi_tiny_step", input_types=types) as g:
        b, x = g.inputs
        new = ops.buffer_load(b) + x
        ops.buffer_store(b, new)
        g.output(ops.sum(new, axis=0))
    return g, [np.zeros(8, np.float32), np.ones(8, np.float32)], -1


def write_inputs(out: Path, inputs: list[np.ndarray], counter: int) -> None:
    header, blob = [len(inputs), counter], bytearray()
    for a in inputs:
        a = np.ascontiguousarray(a)
        blob += b"\0" * (-len(blob) % ALIGN)
        header += [M_DTYPE[a.dtype], a.ndim, *a.shape, len(blob), a.nbytes]
        blob += a.tobytes()
    (out / "step.inputs").write_bytes(bytes(blob))
    np.array(header, np.int64).tofile(out / "step.manifest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = DeviceRef.CPU()
    graph, inputs, counter = tiny_step(dev) if args.tiny else gpt_step(args, dev)
    session = InferenceSession(devices=[CPU()])
    start = time.perf_counter()
    compiled = session.compile(graph)
    print(f"compiled in {time.perf_counter() - start:.1f} s", flush=True)
    compiled.export_mef(args.out / "step.mef")
    write_inputs(args.out, inputs, counter)

    # The reference: the same compiled model, driven from Python, from the
    # same bytes (copied: the step updates its buffers in place).
    model = session.init(compiled)
    print("inputs:", [s.name for s in model.input_metadata][:3], "...",
          f"({len(model.input_metadata)} in total)", flush=True)
    buffers = [Buffer.from_numpy(np.array(a, copy=True)) for a in inputs]
    losses, times = [], []
    for _ in range(args.steps):
        start = time.perf_counter()
        out = model.execute(*buffers)
        times.append(time.perf_counter() - start)
        losses.append(out[0].to_numpy().item())
    lines = [repr(v) for v in losses] + [f"median_step_us {1e6 * statistics.median(times):.1f}"]
    (args.out / "python_reference.txt").write_text("\n".join(lines) + "\n")
    print(f"python: {args.steps} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          f"median {1e6 * statistics.median(times):.1f} us/step", flush=True)


if __name__ == "__main__":
    main()
