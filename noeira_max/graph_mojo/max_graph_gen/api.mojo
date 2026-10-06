"""The hand-written layer over the generated ops: a `Graph` to build in,
the functions behind `Value`'s operators, the ops whose result type must be
computed (as `max.graph.ops` computes it in Python), constants, and
`Graph.compile`, which hands the graph to the `maxrt` binding.

    var g = Graph("mlp", [TensorType(DType.float32, [64, 17])])
    var x = g.inputs()[0].copy()
    var h = relu(x @ w1 + b1)
    g.output([h @ w2 + b2])
    var model = g.compile(Runtime(), "/tmp/mlp.mef")

Ops are added to the graph `max.graph` considers current: the innermost
`Graph` not yet output, as in Python.
"""

from std.python import Python, PythonObject

from maxrt import Model, Runtime

from . import ops
from .backend import (
    GLUE,
    Dim,
    PythonBackend,
    TensorType,
    Value,
    _py_dims,
    _py_type,
    value_from_py,
    values_from_py,
)


struct Graph(Movable):
    """A graph under construction, current until `output`."""

    var backend: PythonBackend
    var compile_seconds: Float64

    def __init__(out self, name: String, input_types: List[TensorType]) raises:
        var builtins = Python.import_module("builtins")
        var types = builtins.list()
        for i in range(len(input_types)):
            _ = types.append(_py_type(input_types[i]))
        var glue = Python.import_module(GLUE)
        self.backend = PythonBackend(glue.new_graph(PythonObject(name), types))
        self.compile_seconds = 0.0

    def inputs(self) raises -> List[Value]:
        return values_from_py(self.backend.glue.inputs(self.backend.graph))

    def output(mut self, values: List[Value]) raises:
        var builtins = Python.import_module("builtins")
        var handles = builtins.list()
        for i in range(len(values)):
            _ = handles.append(values[i].handle)
        self.backend.glue.finish(self.backend.graph, handles)

    def compile(mut self, rt: Runtime, path: String, device: String = "cpu") raises -> Model:
        """Compiles in Python, exports the MEF to `path`, and loads it with
        `maxrt`: from here on, no Python is involved."""
        self.compile_seconds = Float64(
            py=self.backend.glue.compile_to_mef(self.backend.graph, PythonObject(path), PythonObject(device))
        )
        return rt.load(path)


def _current() raises -> PythonBackend:
    return PythonBackend.current()


def _axis(x: Value, axis: Int) -> Int:
    return axis + x.type.rank() if axis < 0 else axis


# Ops whose result type the stub infers: the generated function is enough.


def add(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.add(b, x, y)


def sub(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.sub(b, x, y)


def mul(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.mul(b, x, y)


def div(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.div(b, x, y)


def matmul(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.matmul(b, x, y)


def cast(x: Value, dtype: DType) raises -> Value:
    var b = _current()
    return ops.mo_cast(b, x, dtype)


def reshape(x: Value, shape: List[Dim]) raises -> Value:
    var b = _current()
    return ops.reshape(b, x, shape)


# Ops whose stub needs the result type: computed here, as max.graph does.


def relu(x: Value) raises -> Value:
    var b = _current()
    return ops.mo_relu(b, x.type, x)


def tanh(x: Value) raises -> Value:
    var b = _current()
    return ops.mo_tanh(b, x.type, x)


def exp(x: Value) raises -> Value:
    var b = _current()
    return ops.mo_exp(b, x.type, x)


def sqrt(x: Value) raises -> Value:
    var b = _current()
    return ops.mo_sqrt(b, x.type, x)


def softmax(x: Value, axis: Int = -1) raises -> Value:
    var b = _current()
    return ops.mo_reduce_softmax(b, x.type, x, _axis(x, axis))


def _reduced(x: Value, axis: Int) -> TensorType:
    var t = x.type.copy()
    t.shape[_axis(x, axis)] = Dim(1)
    return t^


def reduce_sum(x: Value, axis: Int = -1) raises -> Value:
    """Sums along `axis`, which is kept with size 1 (as `max.graph.ops.sum`)."""
    var b = _current()
    return ops.mo_reduce_add(b, _reduced(x, axis), x, _axis(x, axis))


def reduce_max(x: Value, axis: Int = -1) raises -> Value:
    var b = _current()
    return ops.mo_reduce_max(b, _reduced(x, axis), x, _axis(x, axis))


def reduce_mean(x: Value, axis: Int = -1) raises -> Value:
    var b = _current()
    return ops.mo_reduce_mean(b, _reduced(x, axis), x, _axis(x, axis))


def transpose(x: Value, axis_1: Int, axis_2: Int) raises -> Value:
    """Swaps two axes (as `max.graph.ops.transpose`: a permutation constant
    on the host, and the result type computed here)."""
    var a1 = _axis(x, axis_1)
    var a2 = _axis(x, axis_2)
    var perm = List[Float64]()
    var t = x.type.copy()
    for i in range(x.type.rank()):
        var j = a2 if i == a1 else (a1 if i == a2 else i)
        perm.append(Float64(j))
        t.shape[i] = x.type.shape[j].copy()
    var b = _current()
    return ops.mo_transpose(
        b, t, x, constant(perm, DType.int64, [Dim(x.type.rank())], "cpu")
    )


def constant(values: List[Float64], dtype: DType, shape: List[Dim], device: String) raises -> Value:
    """A constant tensor. Constants are `mo` ops, not `rmo`: they come from
    the Python backend, not from the generated layer."""
    var builtins = Python.import_module("builtins")
    var py_values = builtins.list()
    for i in range(len(values)):
        _ = py_values.append(PythonObject(values[i]))
    var glue = Python.import_module(GLUE)
    return value_from_py(
        glue.constant(py_values, PythonObject(String(dtype)), _py_dims(shape), PythonObject(device))
    )
