"""Composite against kernel-backed layer norm (plan M4): speed and memory of
``value_and_grad`` of ``sum(LN(x) * w)``, at the GPT benchmark's shape.

- composite: ``ops.layer_norm`` and its rule (``rules/nn.py``), which
  recomputes the mean and rstd in the backward from MAX ops;
- kernel: the ``noeira_layer_norm_fwd`` / ``_bwd`` custom ops
  (``kernels/layer_norm.mojo``), the residuals passed from one to the other.

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/layer_norm_kernel.py [--device gpu]

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
from noeira_max.autodiff.models.common import layer_norm_kernel

EPS = float(np.float32(1e-5))


def build(kind: str, shape: list[int], dev: DeviceRef) -> Graph:
    d = shape[-1]
    f32 = DType.float32
    types = [TensorType(f32, shape, dev), TensorType(f32, [d], dev),
             TensorType(f32, [d], dev), TensorType(f32, shape, dev)]
    with Graph(f"ln_bench_{kind}", input_types=types) as g:
        x, gamma, beta, w = (v.tensor for v in g.inputs)

        def loss(x, gamma, beta):  # noqa: ANN001, ANN202
            norm = layer_norm_kernel if kind == "kernel" else ops.layer_norm
            return sum_all(norm(x, gamma, beta, EPS) * w)

        value, grads = value_and_grad(loss, argnums=(0, 1, 2))(x, gamma, beta)
        g.output(value, *grads)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--shape", type=int, nargs="+", default=[64, 256, 384])
    ap.add_argument("--calls", type=int, default=50)
    args = ap.parse_args()

    device = Accelerator() if args.device == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])
    rng = np.random.default_rng(0)
    d = args.shape[-1]
    arrays = [rng.standard_normal(args.shape).astype(np.float32), rng.standard_normal(d).astype(np.float32),
              rng.standard_normal(d).astype(np.float32), rng.standard_normal(args.shape).astype(np.float32)]
    outputs = {}
    for kind in ("composite", "kernel"):
        start = time.perf_counter()
        model = session.load(build(kind, args.shape, dev))
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
            variant=kind, device=str(device), shape=args.shape, compile_s=round(compile_s, 2),
            ms=round(pipelined, 4), median_synced_ms=round(statistics.median(synced), 4),
            kernels=sum(kernels.values()), buffer_allocs=kernels.get("mgp.buffer.alloc", 0),
        )), flush=True)
    for name, a, b in zip(("loss", "dx", "dgamma", "dbeta"), outputs["kernel"], outputs["composite"]):
        err = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))
        print(f"  kernel vs composite, {name}: max relative difference {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
