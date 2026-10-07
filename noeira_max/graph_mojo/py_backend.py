"""The Python half of ``max_graph_gen``'s ``PythonBackend``.

The generated Mojo ops hand over each op as its stub class name and its
named arguments, each tagged with a kind (``tensor``, ``index``, ``shape``,
...). This module turns them into the objects MAX's op constructors take and
adds the op with ``Graph._add_op_generated``: the call ``max.graph.ops``
itself makes. It also creates graphs, provides constants (they live in the
``mo`` dialect, not ``rmo``), and compiles a finished graph to a MEF for the
Mojo binding.
"""

from __future__ import annotations

import time
from pathlib import Path

from max._core.dialects import builtin, kgen, rmo
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, Shape, StaticDim, TensorType, ops
from max.graph.type import _ChainType


def _device(name: str) -> DeviceRef:
    return DeviceRef.GPU() if name == "gpu" else DeviceRef.CPU()


def tensor_type(dtype: str, dims: list, device: str, buffer: bool = False) -> TensorType | BufferType:
    """``dims``: ints (static) and strings (symbolic). ``buffer``: a
    ``BufferType`` instead, a mutable graph input."""
    kind = BufferType if buffer else TensorType
    return kind(getattr(DType, dtype), Shape(dims), _device(device))


def describe(value) -> tuple | None:  # noqa: ANN001
    """``(dtype, dims, device, is_buffer)`` of a graph value, or ``None`` for a
    chain."""
    t = value.type
    if not isinstance(t, (TensorType, BufferType)):
        return None
    dims = [int(d) if isinstance(d, StaticDim) else str(d) for d in t.shape]
    # Not `t.device.device_type == DeviceKind.GPU`: in MAX 26.6 that is True
    # for a CPU device too (DeviceKind.CPU == DeviceKind.GPU).
    device = "gpu" if t.device.is_gpu() else "cpu"
    return (t.dtype.name, dims, device, isinstance(t, BufferType))


def _convert(kind: str, payload):  # noqa: ANN001, ANN202
    if kind == "result_tensor":
        return tensor_type(*payload[:3])
    if kind == "result_chain":
        return _ChainType()
    if kind == "param_decls":
        return kgen.ParamDeclArrayAttr([])
    if kind == "index":
        return builtin.IntegerAttr(builtin.IndexType(), payload)
    if kind == "bool_attr":
        return builtin.BoolAttr(payload)
    if kind == "string_attr":
        return builtin.StringAttr(payload)
    if kind == "shape":
        return Shape(payload)
    if kind == "dtype":
        return getattr(DType, payload)
    return payload  # operands, int, bool, str


def add_op(graph: Graph, cls: str, names: list, kinds: list, payloads: list) -> list:
    """Adds the ``rmo`` op ``cls`` to ``graph``; returns ``[(value, describe(value))]``."""
    kwargs = {n: _convert(k, p) for n, k, p in zip(names, kinds, payloads)}
    values = graph._add_op_generated(getattr(rmo, cls), **kwargs)
    return [(v, describe(v)) for v in values]


def new_graph(name: str, input_types: list) -> Graph:
    """A graph with the given input types, entered (current) until ``finish``."""
    graph = Graph(name, input_types=[tensor_type(*t) for t in input_types])
    graph.__enter__()
    return graph


def current_graph() -> Graph:
    return Graph.current


def chain(graph: Graph, buffer) -> tuple:  # noqa: ANN001
    """The chain that orders ``buffer``'s device's loads and stores: a load or
    store takes it and returns the next one (``set_chain``), as
    ``max.graph.ops.buffer_load`` / ``buffer_store`` do."""
    return (graph.device_chains[buffer.device], None)


def set_chain(graph: Graph, buffer, chain) -> None:  # noqa: ANN001
    graph.device_chains[buffer.device] = chain


def inputs(graph: Graph) -> list:
    return [(v, describe(v)) for v in graph.inputs]


def constant(values: list, dtype: str, dims: list, device: str) -> tuple:
    import numpy as np

    array = np.array(values, dtype=getattr(DType, dtype).to_numpy()).reshape(dims)
    # A rank-0 constant goes in as a Python number, as max.graph makes one from
    # `x * 0.5`: a 0-d array would come out as shape [1].
    value = ops.constant(array.item() if not dims else array, getattr(DType, dtype), _device(device))
    return (value, describe(value))


def custom(graph: Graph, kernels: str, symbol: str, values: list, out_types: list,
           parameters: dict | None = None) -> list:
    """The Mojo custom op ``symbol`` from the kernel package ``kernels``
    (made available to ``graph`` once). Custom ops are ``mo.custom``, with no
    typed ``rmo`` class: they come from here, not from the generated layer."""
    loaded = graph.__dict__.setdefault("_autodiff_kernel_paths", set())
    if kernels not in loaded:
        graph._import_kernels([Path(kernels)])
        loaded.add(kernels)
    results = ops.custom(
        symbol, device=values[0].device, values=list(values),
        out_types=[tensor_type(*t[:3]) for t in out_types],
        parameters=parameters or None,
    )
    return [(v.tensor, describe(v.tensor)) for v in results]


def cast(value, dtype: str) -> tuple:  # noqa: ANN001
    result = ops.cast(value, getattr(DType, dtype))
    return (result, describe(result))


def finish(graph: Graph, values: list) -> None:
    graph.output(*values)
    graph.__exit__(None, None, None)


def compile_to_mef(graph: Graph, path: str, device: str) -> float:
    """Compiles ``graph`` and exports it to ``path``; returns the compile seconds."""
    session = InferenceSession(devices=[Accelerator() if device == "gpu" else CPU()])
    start = time.perf_counter()
    compiled = session.compile(graph)
    seconds = time.perf_counter() - start
    compiled.export_mef(path)
    return seconds
