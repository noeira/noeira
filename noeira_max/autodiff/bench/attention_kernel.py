"""Composite against kernel-backed causal attention: speed and
memory of ``value_and_grad`` of ``sum(attn(qkv) * w)``, at the GPT
benchmark's shape (B 64, T 256, 6 heads of 64).

- composite: ``gpt.causal_attention`` (split, heads, scores materialised,
  causal bias, softmax, matmul, merge) and the rules of every op in it;
- kernel: noeira's fused attention as the ``noeira_attention_fwd`` / ``_bwd``
  custom ops (``kernels/attention.mojo``), the output and log-sum-exp passed
  from one to the other.

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/attention_kernel.py --device gpu

Prints one ``RESULT {json}`` line per variant: milliseconds per call
(pipelined, and the median of synchronised calls), the compiled model's
kernel count and buffer allocations (``Model.kernel_summaries``), and the
compile time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff import value_and_grad
from noeira_max.autodiff._ops import sum_all
from noeira_max.autodiff.models import gpt
from noeira_max.autodiff.models.common import attention_kernel


def build(kind: str, cfg: gpt.Config, batch: int, dev: DeviceRef) -> Graph:
    f32 = DType.float32
    types = [TensorType(f32, [batch, cfg.seq, 3 * cfg.dim], dev),
             TensorType(f32, [batch, cfg.seq, cfg.dim], dev)]
    with Graph(f"attn_bench_{kind}", input_types=types) as g:
        qkv, w = (v.tensor for v in g.inputs)

        def loss(qkv):  # noqa: ANN001, ANN202
            if kind == "kernel":
                return sum_all(attention_kernel(qkv, cfg.heads) * w)
            return sum_all(gpt.causal_attention(qkv, cfg) * w)

        value, grad = value_and_grad(loss)(qkv)
        g.output(value, grad)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--calls", type=int, default=50)
    args = ap.parse_args()

    device = Accelerator() if args.device == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])
    rng = np.random.default_rng(0)
    cfg = gpt.Config(seq=args.seq, dim=args.dim, heads=args.heads)
    arrays = [rng.standard_normal((args.batch, args.seq, 3 * args.dim)).astype(np.float32),
              rng.standard_normal((args.batch, args.seq, args.dim)).astype(np.float32)]
    outputs = {}
    for kind in ("composite", "kernel"):
        start = time.perf_counter()
        model = session.load(build(kind, cfg, args.batch, dev))
        compile_s = time.perf_counter() - start
        inputs = [Buffer.from_numpy(a).to(device) for a in arrays]
        for _ in range(5):
            model.execute(*inputs)
        device.synchronize()
        start = time.perf_counter()
        for _ in range(args.calls):
            model.execute(*inputs)
        device.synchronize()
        pipelined = 1000 * (time.perf_counter() - start) / args.calls
        synced = []
        for _ in range(min(20, args.calls)):
            t = time.perf_counter()
            model.execute(*inputs)
            device.synchronize()
            synced.append(1000 * (time.perf_counter() - t))
        kernels = Counter(k for k in model.kernel_summaries if not k.startswith("index."))
        outputs[kind] = [o.to_numpy() for o in model.execute(*inputs)]
        print("RESULT " + json.dumps(dict(
            variant=kind, device=str(device), batch=args.batch, seq=args.seq, dim=args.dim,
            heads=args.heads, compile_s=round(compile_s, 2),
            ms=round(pipelined, 4), median_synced_ms=round(statistics.median(synced), 4),
            kernels=sum(kernels.values()), buffer_allocs=kernels.get("mgp.buffer.alloc", 0),
        )), flush=True)
    for name, a, b in zip(("loss", "dqkv"), outputs["kernel"], outputs["composite"]):
        err = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))
        print(f"  kernel vs composite, {name}: max relative difference {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
