"""Semantics of the transform itself, beyond any single rule.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_structural -v
"""

from __future__ import annotations

import unittest

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import compilation
from max.experimental import functional as F
from max.experimental.sharding import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff import grad, value_and_grad, vjp
from noeira_max.autodiff._ops import sum_all

F64 = DType.float64
DEV = DeviceRef.CPU()
_SESSION = InferenceSession(devices=[CPU()])


def _run(graph: Graph, *arrays: np.ndarray) -> list[np.ndarray]:
    model = _SESSION.load(graph)
    return [b.to_numpy() for b in model.execute(*map(Buffer.from_numpy, arrays))]


def _f64(*shapes) -> list[TensorType]:
    return [TensorType(F64, s, DEV) for s in shapes]


class TransformTest(unittest.TestCase):
    rng = np.random.default_rng(0)

    def test_closed_over_value_is_a_constant(self):
        # h is built from x BEFORE fun runs: jax.grad treats it as a constant,
        # so d/dx sum(x * h) = h, not 4x.
        with Graph("closure", input_types=_f64([3, 2])) as g:
            (x,) = g.inputs
            h = x * 2.0
            g.output(grad(lambda x: sum_all(x * h))(x))
        xv = self.rng.standard_normal((3, 2))
        (dx,) = _run(g, xv)
        np.testing.assert_allclose(dx, 2.0 * xv, rtol=1e-12)

    def test_same_value_twice_gets_separate_partials(self):
        with Graph("twice", input_types=_f64([3])) as g:
            (x,) = g.inputs
            g.output(*grad(lambda a, b: sum_all(a * a * b), argnums=(0, 1))(x, x))
        xv = self.rng.standard_normal(3)
        da, db = _run(g, xv)
        np.testing.assert_allclose(da, 2 * xv * xv, rtol=1e-12)  # 2ab
        np.testing.assert_allclose(db, xv * xv, rtol=1e-12)  # a^2

    def test_unreachable_input_gets_zeros(self):
        with Graph("unused", input_types=_f64([2], [4])) as g:
            x, y = g.inputs
            g.output(*grad(lambda x, y: sum_all(x * x), argnums=(0, 1))(x, y))
        dx, dy = _run(g, np.ones(2), np.ones(4))
        np.testing.assert_array_equal(dy, np.zeros(4))
        np.testing.assert_allclose(dx, 2 * np.ones(2))

    def test_side_path_without_a_rule_is_never_visited(self):
        # cumsum has no rule; it depends on x but not the output, so it must
        # not stop the transform.
        def loss(x):
            _ = ops.cumsum(x, axis=0)
            _ = ops.argmax(x, axis=0)
            return sum_all(ops.exp(x))

        with Graph("side", input_types=_f64([4])) as g:
            (x,) = g.inputs
            g.output(grad(loss)(x))
        xv = self.rng.standard_normal(4)
        (dx,) = _run(g, xv)
        np.testing.assert_allclose(dx, np.exp(xv), rtol=1e-12)

    def test_unknown_op_on_the_path_names_the_op(self):
        with Graph("unknown", input_types=_f64([4])):
            x = Graph.current.inputs[0]
            with self.assertRaisesRegex(NotImplementedError, "rmo.mo.cumsum"):
                grad(lambda x: sum_all(ops.cumsum(x, axis=0)))(x)

    def test_vjp_with_a_non_scalar_output(self):
        with Graph("vjp", input_types=_f64([3, 2], [3, 2])) as g:
            x, ct = g.inputs
            y, vjp_fn = vjp(lambda x: ops.tanh(x), x)
            (dx,) = vjp_fn(ct)
            g.output(y, dx)
        xv, cv = self.rng.standard_normal((2, 3, 2))
        y, dx = _run(g, xv, cv)
        np.testing.assert_allclose(dx, cv * (1 - np.tanh(xv) ** 2), rtol=1e-12)

    def test_non_scalar_output_is_refused(self):
        with Graph("nonscalar", input_types=_f64([3])):
            x = Graph.current.inputs[0]
            with self.assertRaisesRegex(TypeError, "scalar"):
                grad(lambda x: ops.exp(x))(x)

    def test_pytree_of_parameters(self):
        def loss(params, x):
            h = ops.tanh(ops.matmul(x, params["w1"]) + params["b1"])
            return sum_all(ops.matmul(h, params["w2"]))

        with Graph("pytree", input_types=_f64([2, 3], [3, 4], [4], [4, 1])) as g:
            x, w1, b1, w2 = g.inputs
            params = {"w1": w1, "b1": b1, "w2": w2}
            value, grads = value_and_grad(loss)(params, x)
            g.output(value, grads["w1"], grads["b1"], grads["w2"])
        xv = self.rng.standard_normal((2, 3))
        w1v, b1v, w2v = (self.rng.standard_normal(s) for s in [(3, 4), (4,), (4, 1)])
        _, gw1, gb1, gw2 = _run(g, xv, w1v, b1v, w2v)
        h = np.tanh(xv @ w1v + b1v)
        dpre = (np.ones((2, 1)) @ w2v.T) * (1 - h * h)
        np.testing.assert_allclose(gw2, h.T @ np.ones((2, 1)), rtol=1e-12)
        np.testing.assert_allclose(gb1, dpre.sum(0), rtol=1e-12)
        np.testing.assert_allclose(gw1, xv.T @ dpre, rtol=1e-12)


class ExperimentalTensorTest(unittest.TestCase):
    """The user-facing path of ``max.experimental``: tensors traced and
    compiled by ``compilation.compile``, gradient taken inside the trace."""

    def test_value_and_grad_inside_compile(self):
        def step(x, w):
            def loss(w):
                y = F.relu(x @ w)
                return F.sum(F.sum(y * y, axis=1), axis=0)

            return value_and_grad(loss)(w)

        run = compilation.compile(step)(
            TensorLayout(F64, ["n", 4], CPU()), TensorLayout(F64, [4, 3], CPU())
        )
        rng = np.random.default_rng(1)
        xv, wv = rng.standard_normal((5, 4)), rng.standard_normal((4, 3))
        value, gw = run(Tensor.from_dlpack(xv), Tensor.from_dlpack(wv))
        h = xv @ wv
        np.testing.assert_allclose(
            value.to_numpy().item(), (np.maximum(h, 0) ** 2).sum(), rtol=1e-12
        )
        np.testing.assert_allclose(
            gw.to_numpy(), xv.T @ (2 * np.maximum(h, 0)), rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
