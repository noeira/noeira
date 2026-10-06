"""The parity tests' reference graphs, built with ``max.graph`` in Python.

``test_parity.mojo`` builds each case again in Mojo, through the generated
ops, and runs both MEFs through ``maxrt`` on the same inputs. Inputs are
``input0``, ``input1``, ... in the order of ``INPUTS[case]``.
"""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

F32 = DType.float32
INPUTS = {
    "elementwise": [[4, 8], [4, 8]],
    "matmul": [[4, 8], [8, 3]],
    "reshape_transpose": [[2, 3, 4]],
    "reductions": [[4, 8]],
    "softmax": [[4, 8]],
    "mlp": [[16, 17], [17, 32], [32], [32, 6], [6]],
    "generated_only": [[4, 8]],
}


def _graph(case: str, body) -> Graph:  # noqa: ANN001
    types = [TensorType(F32, shape, DeviceRef.CPU()) for shape in INPUTS[case]]
    with Graph(f"{case}_python", input_types=types) as g:
        g.output(*body(*(v.tensor for v in g.inputs)))
    return g


def elementwise(x, y):  # noqa: ANN001, ANN201
    return [(x + y) * x - y, ops.relu(x), ops.tanh(x), ops.exp(x), ops.sqrt(ops.exp(x)),
            x / ops.exp(y)]


def matmul(x, w):  # noqa: ANN001, ANN201
    return [x @ w]


def reshape_transpose(x):  # noqa: ANN001, ANN201
    return [ops.transpose(ops.reshape(x, [6, 4]), 0, 1)]


def reductions(x):  # noqa: ANN001, ANN201
    return [ops.sum(x, axis=-1), ops.max(x, axis=0), ops.mean(x, axis=1)]


def softmax(x):  # noqa: ANN001, ANN201
    return [ops.softmax(x, axis=-1)]


def mlp(x, w1, b1, w2, b2):  # noqa: ANN001, ANN201
    return [ops.relu(x @ w1 + b1) @ w2 + b2]


def generated_only(x):  # noqa: ANN001, ANN201
    values, indices = ops.top_k(x, 3, axis=-1)
    return [values, indices, ops.cumsum(x, axis=1), ops.erf(x)]


def build(case: str, path: str) -> float:
    """Builds ``case`` with ``max.graph``, compiles it for the CPU and
    exports it to ``path``; returns the compile seconds."""
    import time

    graph = _graph(case, globals()[case])
    session = InferenceSession(devices=[CPU()])
    start = time.perf_counter()
    compiled = session.compile(graph)
    seconds = time.perf_counter() - start
    compiled.export_mef(path)
    return seconds
