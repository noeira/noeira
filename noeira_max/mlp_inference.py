"""Configurable MLP inference on MAX, with timing primitives for interop benchmarking.

Inference only (training out of scope for v1). The graph is built + compiled once in
``__init__``; the per-step hot path is just ``model.execute`` + (optional) host<->device
copies. Weights are random (Kaiming) baked-in constants since we only measure latency.

Design notes for the benchmark
------------------------------
The Mojo driver pays one Mojo->Python crossing per method call. To separate "MAX work"
from "interop tax", the ``bench_*`` methods each run ``iters`` iterations *inside one
Python call* (so a single crossing is amortized to ~0), and return total seconds. The
Mojo side then *also* measures the per-call crossing floor (``noop`` in a Mojo loop) and
the realistic per-call cost (``infer`` in a Mojo loop). Comparing the two attributions
isolates the bridge overhead on the real path.

All GPU timings call ``drv.synchronize()`` before stopping the clock so we measure
completed compute, not async dispatch. ``device="gpu"`` resolves to Metal on Apple and
CUDA on NVIDIA via MAX's driver — identical Python, different backend.
"""

import time

import numpy as np

from max.driver import CPU, Accelerator, accelerator_count, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _parse_hidden(hidden):
    """Accept hidden dims as a list/tuple or a "256,256" string (Mojo-friendly)."""
    if hidden is None:
        return []
    if isinstance(hidden, str):
        return [int(t) for t in hidden.split(",") if t.strip() != ""]
    return [int(h) for h in hidden]


class MLPInference:
    """A compiled MAX MLP with configurable dims/batch/device + timing primitives.

    Parameters
    ----------
    input_dim, output_dim, batch : int
        Input feature width, output width, and (fixed) batch size of the graph.
    hidden : list[int] | str
        Hidden layer widths, e.g. ``[256, 256]`` or the string ``"256,256"``.
    device : str
        ``"gpu"`` (Metal/CUDA, falls back to CPU if no accelerator) or ``"cpu"``.
    seed : int
        RNG seed for the (inference-only) random weights and input.
    """

    def __init__(self, input_dim, hidden, output_dim, batch,
                 device="gpu", seed=0):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.batch = int(batch)
        self.hidden = _parse_hidden(hidden)

        want_gpu = (str(device).lower() == "gpu") and accelerator_count() > 0
        self.on_gpu = bool(want_gpu)
        self.dref = DeviceRef.GPU() if want_gpu else DeviceRef.CPU()
        self.drv = Accelerator() if want_gpu else CPU()
        self.host = CPU()

        rng = np.random.default_rng(int(seed))
        self._dims = [self.input_dim] + self.hidden + [self.output_dim]
        weights = []
        for i in range(len(self._dims) - 1):
            fan_in, fan_out = self._dims[i], self._dims[i + 1]
            scale = np.sqrt(2.0 / fan_in)  # Kaiming
            w = (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)
            b = np.zeros((fan_out,), dtype=np.float32)
            weights.append((w, b))

        in_type = TensorType(
            DType.float32, shape=[self.batch, self.input_dim], device=self.dref
        )
        with Graph("mlp_b%d_in%d" % (self.batch, self.input_dim),
                   input_types=[in_type]) as g:
            h = g.inputs[0]
            n = len(weights)
            for i, (w, b) in enumerate(weights):
                wt = ops.constant(w, DType.float32, device=self.dref)
                bt = ops.constant(b, DType.float32, device=self.dref)
                h = h @ wt + bt
                if i < n - 1:  # ReLU on all but the output layer
                    h = ops.relu(h)
            g.output(h)

        session = InferenceSession(devices=[self.drv])
        t0 = time.perf_counter()
        self.compiled = session.compile(g)  # kept for export_mef (capi_mojo/bench)
        self.model = session.init(self.compiled)
        self.compile_seconds = time.perf_counter() - t0

        # Persistent host + device inputs reused for steady-state timing.
        self._host_input = rng.standard_normal(
            (self.batch, self.input_dim)
        ).astype(np.float32)
        self._dev_input = Buffer.from_numpy(self._host_input).to(self.drv)
        self.synchronize()

    # ------------------------------------------------------------------ utils
    def synchronize(self):
        self.drv.synchronize()

    def info(self):
        return "MLP %s | batch=%d | device=%s | params=%d | compile=%.4fs" % (
            "->".join(str(d) for d in self._dims),
            self.batch,
            "gpu" if self.on_gpu else "cpu",
            self.num_params(),
            self.compile_seconds,
        )

    def num_params(self):
        total = 0
        for i in range(len(self._dims) - 1):
            total += self._dims[i] * self._dims[i + 1] + self._dims[i + 1]
        return int(total)

    def noop(self):
        """Cheapest possible call: the Mojo->Python interop floor."""
        return 0

    def make_host_input(self):
        """Return the persistent host input (numpy [batch, input_dim])."""
        return self._host_input

    # ----------------------------------------------------- single-shot stages
    def to_device(self, x_np):
        xt = Buffer.from_numpy(
            np.ascontiguousarray(x_np, dtype=np.float32)
        ).to(self.drv)
        self.synchronize()
        return xt

    def compute(self, xt):
        out = self.model.execute(xt)[0]
        self.synchronize()
        return out

    def to_host(self, out):
        h = out.to(self.host) if self.on_gpu else out
        return h.to_numpy()

    def infer(self, x_np):
        """Full realistic call: H2D + execute + D2H, returns numpy output."""
        xt = Buffer.from_numpy(
            np.ascontiguousarray(x_np, dtype=np.float32)
        ).to(self.drv)
        out = self.model.execute(xt)[0]
        h = out.to(self.host) if self.on_gpu else out
        return h.to_numpy()

    # ------------------------------------------------ amortized python benches
    # Each runs `iters` iterations inside ONE Python call (interop ~ free) and
    # returns total wall seconds. Divide by iters on the caller side.
    def bench_compute(self, iters):
        """Pure device compute: execute on an already-on-device input."""
        iters = int(iters)
        xt = self._dev_input
        self.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = self.model.execute(xt)[0]
        self.synchronize()
        return time.perf_counter() - t0

    def bench_h2d(self, iters):
        """Host->device transfer of the input."""
        iters = int(iters)
        x = self._host_input
        self.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = Buffer.from_numpy(x).to(self.drv)
        self.synchronize()
        return time.perf_counter() - t0

    def bench_d2h(self, iters):
        """Device->host transfer of the output."""
        iters = int(iters)
        out = self.model.execute(self._dev_input)[0]
        self.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            h = out.to(self.host) if self.on_gpu else out
            _ = h.to_numpy()
        self.synchronize()
        return time.perf_counter() - t0

    def bench_full(self, iters):
        """End-to-end H2D + execute + D2H, all inside Python (no per-iter interop)."""
        iters = int(iters)
        x = self._host_input
        self.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            xt = Buffer.from_numpy(x).to(self.drv)
            out = self.model.execute(xt)[0]
            h = out.to(self.host) if self.on_gpu else out
            _ = h.to_numpy()
        self.synchronize()
        return time.perf_counter() - t0

    def bench_noop(self, iters):
        """Python-side loop over noop() — reference floor (not interop, just loop)."""
        iters = int(iters)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.noop()
        return time.perf_counter() - t0
