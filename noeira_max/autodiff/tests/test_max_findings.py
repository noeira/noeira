"""MAX 26.6 behaviours the prototype found, pinned so a release that changes
them is noticed. Each test asserts the CORRECT result and is marked
``expectedFailure`` while MAX still gets it wrong: an "unexpected success"
means the release fixed it, and the workaround can go.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_max_findings -v
"""

from __future__ import annotations

import unittest

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

DEV = DeviceRef.CPU()
_SESSION = InferenceSession(devices=[CPU()])


def _run(graph, *arrays):
    return [b.to_numpy() for b in _SESSION.load(graph).execute(*map(Buffer.from_numpy, arrays))]


def _masked_attention(dtype, np_dtype, batch, heads, form):
    t, d = 8, 4
    shape = [batch, heads, t, d]
    with Graph(f"attn_{form}", input_types=[TensorType(dtype, shape, DEV)] * 3) as g:
        q, k, v = g.inputs
        scores = ops.matmul(q, ops.transpose(k, -1, -2)) * 0.5
        tril = np.tril(np.ones((t, t), bool))
        if form == "select":
            mask = ops.constant(tril, DType.bool, DEV)
            masked = ops.where(mask, scores, ops.constant(-np.inf, dtype, DEV))
        else:
            bias = np.where(tril, 0.0, -np.inf).astype(np_dtype)
            masked = scores + ops.constant(bias, dtype, DEV)
        g.output(ops.matmul(ops.softmax(masked), v))
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal(shape).astype(np_dtype) for _ in range(3))
    (y,) = _run(g, q, k, v)
    s = np.where(tril, (q @ np.swapaxes(k, -1, -2)) * 0.5, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True))
    return y, (w / w.sum(-1, keepdims=True)) @ v


class MaskedAttentionTest(unittest.TestCase):
    """``softmax(where(mask, q @ k^T * c, -inf)) @ v``: all NaN on CPU once
    the batch dims exceed 1 (B = H = 2), float32 and float64 alike; correct
    with B = H = 1, with an additive -inf bias, with a finite fill, or when
    the scores are also a graph output (which blocks the fusion)."""

    @unittest.expectedFailure
    def test_select_mask_with_batch_dims(self):
        y, ref = _masked_attention(DType.float32, np.float32, 2, 2, "select")
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)

    def test_select_mask_without_batch_dims(self):
        y, ref = _masked_attention(DType.float32, np.float32, 1, 1, "select")
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)

    def test_additive_mask_with_batch_dims(self):  # the workaround models use
        y, ref = _masked_attention(DType.float32, np.float32, 2, 2, "bias")
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)


class Float64PrecisionTest(unittest.TestCase):
    """Float64 kernels that are only float32-accurate."""

    @unittest.expectedFailure
    def test_mean_scale_is_exact(self):
        # 3 * float32(1/3) = 1.0000000298023224, which MAX returns.
        with Graph("mean", input_types=[TensorType(DType.float64, [1, 3], DEV)]) as g:
            g.output(ops.mean(g.inputs[0], axis=1))
        (m,) = _run(g, np.ones((1, 3)))
        self.assertEqual(m.item(), 1.0)

    @unittest.expectedFailure
    def test_erf_is_float64_accurate(self):
        from math import erf

        x = np.linspace(-3, 3, 13)
        with Graph("erf", input_types=[TensorType(DType.float64, [13], DEV)]) as g:
            g.output(ops.erf(g.inputs[0]))
        (y,) = _run(g, x)
        np.testing.assert_allclose(y, [erf(v) for v in x], rtol=1e-12)


class SplitTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_split_accepts_symbolic_sizes(self):
        # Annotated Sequence[DimLike], but it calls int() on every size.
        with Graph("split", input_types=[TensorType(DType.float64, ["n", "m"], DEV)]) as g:
            x = g.inputs[0]
            ops.split(x, [x.shape[0]], axis=0)


if __name__ == "__main__":
    unittest.main()
