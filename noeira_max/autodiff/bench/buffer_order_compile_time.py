"""Compile time against the number of in-place buffer updates, and their order.

No autodiff involved: ``N`` buffers, each loaded, updated elementwise and
stored. "interleaved" emits load b1, store b1, load b2, store b2, ...;
"phased" emits every load, then every store. A train step updates every
parameter and every optimizer moment in place, so N is in the hundreds.

Measured on an Apple M1 CPU with MAX 26.6: interleaved 7.7 s at
N = 10, 24.9 s at 30, over 240 s at 40; phased 7.4 s at 10, 11.1 s at 170.

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/buffer_order_compile_time.py
    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/buffer_order_compile_time.py --device gpu

Each compile runs in a subprocess with a time limit; a fresh constant per run
keeps the compile cache from answering instead of the compiler.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time


def compile_one(pattern: str, n: int, device_kind: str) -> None:
    from max.driver import CPU, Accelerator, accelerator_count
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

    device = Accelerator() if device_kind == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    nonce = random.random()
    types = [BufferType(DType.float32, [64, 64], dev) for _ in range(n)]
    types.append(TensorType(DType.float32, [64, 64], dev))
    with Graph(f"buffers_{pattern}_{n}", input_types=types) as g:
        buffers, x = g.inputs[:n], g.inputs[n]
        x = x + ops.constant(nonce, DType.float32, dev) * 0.0
        if pattern == "interleaved":
            for b in buffers:
                ops.buffer_store(b, ops.buffer_load(b) * 0.9 + x)
        else:
            loaded = [ops.buffer_load(b) for b in buffers]
            for b, v in zip(buffers, loaded):
                ops.buffer_store(b, v * 0.9 + x)
        g.output(x)
    start = time.perf_counter()
    InferenceSession(devices=[device]).compile(g)
    print(f"{time.perf_counter() - start:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--counts", type=int, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--limit", type=float, default=240.0, help="seconds per compile")
    ap.add_argument("--one", nargs=2, metavar=("PATTERN", "N"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        compile_one(args.one[0], int(args.one[1]), args.device)
        return

    print(f"{'N':>4}  {'interleaved':>12}  {'phased':>8}")
    for n in args.counts:
        cells = []
        for pattern in ("interleaved", "phased"):
            cmd = [sys.executable, __file__, "--device", args.device, "--one", pattern, str(n)]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.limit)
                cells.append(f"{float(out.stdout.split()[-1]):.1f} s")
            except subprocess.TimeoutExpired:
                cells.append(f"> {args.limit:.0f} s")
            except (ValueError, IndexError):
                cells.append("error")
        print(f"{n:>4}  {cells[0]:>12}  {cells[1]:>8}", flush=True)


if __name__ == "__main__":
    main()
