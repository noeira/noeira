"""Kernel-backed rules (plan M4): Mojo custom-op pairs whose VJP rule hands
the forward's residuals to the backward kernel (``rules/custom.py``, RFC
§6.3). LayerNorm (residuals: mean, rstd), and causal attention with noeira's
fused kernels (residuals: the output and each row's log-sum-exp).

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_custom_op -v
"""

from __future__ import annotations

import unittest

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff import _graph, value_and_grad
from noeira_max.autodiff._ops import sum_all
from noeira_max.autodiff.models import gpt
from noeira_max.autodiff.models.common import attention_kernel, layer_norm_kernel
from noeira_max.autodiff.rules import custom

SHAPE = (2, 3, 5)
# ops.layer_norm's epsilon is a float32 constant even for float64 input
# (results doc §5.4): use the same value on both sides.
EPS = float(np.float32(1e-5))
_SESSION = None


def session() -> InferenceSession:
    global _SESSION
    _SESSION = _SESSION or InferenceSession(devices=[CPU()])
    return _SESSION


def build(kind: str, dtype: DType = DType.float64, dev: DeviceRef | None = None) -> Graph:
    """``L = sum(LN(x) * w)``; outputs ``L``, ``y``, ``dL/dx``, ``dL/dgamma``,
    ``dL/dbeta``. ``kind``: ``composite`` (``ops.layer_norm``) or ``kernel``."""
    dev = dev or DeviceRef.CPU()
    d = SHAPE[-1]
    types = [TensorType(dtype, SHAPE, dev), TensorType(dtype, [d], dev),
             TensorType(dtype, [d], dev), TensorType(dtype, SHAPE, dev)]
    with Graph(f"layer_norm_{kind}_{dtype}_{dev.device_type.value}", input_types=types) as g:
        x, gamma, beta, w = (v.tensor for v in g.inputs)

        def loss(x, gamma, beta):  # noqa: ANN001, ANN202
            if kind == "kernel":
                y = layer_norm_kernel(x, gamma, beta, EPS)
            else:
                y = ops.layer_norm(x, gamma, beta, EPS)
            return sum_all(y * w), y

        (value, y), grads = value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, gamma, beta)
        g.output(value, y, *grads)
    return g


def inputs(seed: int, dtype=np.float64) -> list[np.ndarray]:  # noqa: ANN001
    rng = np.random.default_rng(seed)
    d = SHAPE[-1]
    return [rng.standard_normal(SHAPE).astype(dtype) * 2 + 0.5,
            rng.standard_normal(d).astype(dtype),
            rng.standard_normal(d).astype(dtype),
            rng.standard_normal(SHAPE).astype(dtype)]


def run(model, arrays, device=None) -> list[np.ndarray]:  # noqa: ANN001
    buffers = [Buffer.from_numpy(a.copy()) for a in arrays]
    if device is not None:
        buffers = [b.to(device) for b in buffers]
    return [o.to_numpy() for o in model.execute(*buffers)]


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


class KernelBackedLayerNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel_graph = build("kernel")
        cls.kernel = session().load(cls.kernel_graph)
        cls.composite = session().load(build("composite"))

    def test_the_graph_uses_the_kernel_pair(self):
        names = [_graph.op_name(op) for op in _graph.block_ops(self.kernel_graph)]
        self.assertEqual(names.count("mo.custom"), 2)  # forward and backward
        self.assertNotIn("mo.reduce.layer_norm", names)

    def test_matches_the_composite_rule(self):
        for seed in range(3):
            arrays = inputs(seed)
            got, want = run(self.kernel, arrays), run(self.composite, arrays)
            for name, a, b in zip(("loss", "y", "dx", "dgamma", "dbeta"), got, want):
                self.assertLess(rel(a, b), 1e-12, f"{name}, seed {seed}")

    def test_gradcheck(self):
        """Central differences of L along random directions, in float64."""
        arrays = inputs(7)
        _, _, *grads = run(self.kernel, arrays)
        rng = np.random.default_rng(11)
        h = 1e-6
        for i in range(3):  # x, gamma, beta
            for _ in range(3):
                direction = rng.standard_normal(arrays[i].shape)
                up = [a.copy() for a in arrays]
                down = [a.copy() for a in arrays]
                up[i] = up[i] + h * direction
                down[i] = down[i] - h * direction
                fd = (run(self.kernel, up)[0] - run(self.kernel, down)[0]).item() / (2 * h)
                analytic = float(np.sum(grads[i] * direction))
                self.assertLess(abs(fd - analytic) / max(abs(analytic), 1e-8), 1e-6,
                                f"input {i}: fd {fd} vs analytic {analytic}")

    def test_float32(self):
        k32 = session().load(build("kernel", DType.float32))
        c32 = session().load(build("composite", DType.float32))
        arrays = inputs(3, np.float32)
        for name, a, b in zip(("loss", "y", "dx", "dgamma", "dbeta"), run(k32, arrays), run(c32, arrays)):
            self.assertLess(rel(a, b), 1e-5, name)

    @unittest.skipUnless(accelerator_count(), "no accelerator")
    def test_gpu_float32(self):
        """The kernels' GPU path (one block per row) against the composite rule
        on the same GPU."""
        gpu = Accelerator()
        gpu_session = InferenceSession(devices=[gpu])
        dev = DeviceRef.from_device(gpu)
        kernel = gpu_session.load(build("kernel", DType.float32, dev))
        composite = gpu_session.load(build("composite", DType.float32, dev))
        arrays = inputs(5, np.float32)
        for name, a, b in zip(("loss", "y", "dx", "dgamma", "dbeta"),
                              run(kernel, arrays, gpu), run(composite, arrays, gpu)):
            self.assertLess(rel(a, b), 1e-5, name)

    def test_a_wrong_rule_is_caught(self):
        """Vacuity: the same checks fail when the VJP drops dbeta's sum."""
        good = custom._BY_SYMBOL["noeira_layer_norm_fwd"]

        def wrong(ctx):  # noqa: ANN001, ANN202
            dx, dgamma, dbeta, none = good(ctx)
            return [dx, dgamma, dbeta * 0.5, none]

        custom._BY_SYMBOL["noeira_layer_norm_fwd"] = wrong
        try:
            mutant = session().load(build("kernel"))
        finally:
            custom._BY_SYMBOL["noeira_layer_norm_fwd"] = good
        arrays = inputs(0)
        got, want = run(mutant, arrays), run(self.composite, arrays)
        self.assertGreater(rel(got[4], want[4]), 0.1)


# Head dim 64. noeira's GPU kernels split it over a tile's threads, which
# needs a multiple of 32 with NVIDIA's tiles (16 with Apple's). And its scale,
# 1/8, is a float32: MAX's fused CPU attention, which the float64 composite
# compiles to, holds the scale in float32 (test_max_findings).
ATTN = gpt.Config(seq=8, dim=128, heads=2)
ATTN_SHAPE = (2, ATTN.seq, 3 * ATTN.dim)


def build_attention(kind: str, dtype: DType = DType.float64, dev: DeviceRef | None = None) -> Graph:
    """``L = sum(attn(qkv) * w)``; outputs ``L``, ``y``, ``dL/dqkv``."""
    dev = dev or DeviceRef.CPU()
    out_shape = (ATTN_SHAPE[0], ATTN.seq, ATTN.dim)
    types = [TensorType(dtype, ATTN_SHAPE, dev), TensorType(dtype, out_shape, dev)]
    with Graph(f"attention_{kind}_{dtype}_{dev.device_type.value}", input_types=types) as g:
        qkv, w = (v.tensor for v in g.inputs)

        def loss(qkv):  # noqa: ANN001, ANN202
            if kind == "kernel":
                y = attention_kernel(qkv, ATTN.heads)
            else:
                y = gpt.causal_attention(qkv, ATTN)
            return sum_all(y * w), y

        (value, y), grads = value_and_grad(loss, has_aux=True)(qkv)
        g.output(value, y, grads)
    return g


def attention_inputs(seed: int, dtype=np.float64) -> list[np.ndarray]:  # noqa: ANN001
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(ATTN_SHAPE).astype(dtype),
            rng.standard_normal((ATTN_SHAPE[0], ATTN.seq, ATTN.dim)).astype(dtype)]


class KernelBackedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel_graph = build_attention("kernel")
        cls.kernel = session().load(cls.kernel_graph)
        cls.composite = session().load(build_attention("composite"))

    def test_the_graph_uses_the_kernel_pair(self):
        names = [_graph.op_name(op) for op in _graph.block_ops(self.kernel_graph)]
        self.assertEqual(names.count("mo.custom"), 2)
        self.assertNotIn("rmo.mo.reduce.softmax", names)

    def test_matches_the_composite_rule(self):
        for seed in range(3):
            arrays = attention_inputs(seed)
            got, want = run(self.kernel, arrays), run(self.composite, arrays)
            for name, a, b in zip(("loss", "y", "dqkv"), got, want):
                self.assertLess(rel(a, b), 1e-12, f"{name}, seed {seed}")

    def test_gradcheck(self):
        arrays = attention_inputs(7)
        _, _, grad = run(self.kernel, arrays)
        rng = np.random.default_rng(13)
        h = 1e-6
        for _ in range(4):
            direction = rng.standard_normal(ATTN_SHAPE)
            up = [arrays[0] + h * direction, arrays[1]]
            down = [arrays[0] - h * direction, arrays[1]]
            fd = (run(self.kernel, up)[0] - run(self.kernel, down)[0]).item() / (2 * h)
            analytic = float(np.sum(grad * direction))
            self.assertLess(abs(fd - analytic) / max(abs(analytic), 1e-8), 1e-6)

    @unittest.skipUnless(accelerator_count(), "no accelerator")
    def test_gpu_float32(self):
        """noeira's fused attention kernels against the composite form, on the
        same GPU."""
        gpu = Accelerator()
        gpu_session = InferenceSession(devices=[gpu])
        dev = DeviceRef.from_device(gpu)
        kernel = gpu_session.load(build_attention("kernel", DType.float32, dev))
        composite = gpu_session.load(build_attention("composite", DType.float32, dev))
        arrays = attention_inputs(5, np.float32)
        for name, a, b in zip(("loss", "y", "dqkv"), run(kernel, arrays, gpu), run(composite, arrays, gpu)):
            self.assertLess(rel(a, b), 1e-4, name)

    def test_a_wrong_rule_is_caught(self):
        good = custom._BY_SYMBOL["noeira_attention_fwd"]

        def wrong(ctx):  # noqa: ANN001, ANN202
            (dqkv,) = good(ctx)
            return [dqkv * 0.5]

        custom._BY_SYMBOL["noeira_attention_fwd"] = wrong
        try:
            mutant = session().load(build_attention("kernel"))
        finally:
            custom._BY_SYMBOL["noeira_attention_fwd"] = good
        arrays = attention_inputs(0)
        self.assertGreater(rel(run(mutant, arrays)[2], run(self.composite, arrays)[2]), 0.1)


if __name__ == "__main__":
    unittest.main()
