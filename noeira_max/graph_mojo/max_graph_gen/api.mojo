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


def pow(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.pow(b, x, y)


def cast(x: Value, dtype: DType) raises -> Value:
    """`x` converted to `dtype`. As constants, this comes from the Python
    backend: `max.graph.ops.cast` emits the `mo` dialect's `mo.cast`, while
    the generated `ops.mo_cast` emits `rmo.mo.cast`, a different op that
    computes the same thing."""
    var glue = Python.import_module(GLUE)
    return value_from_py(glue.cast(x.handle, PythonObject(String(dtype))))


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


def gelu_tanh(x: Value) raises -> Value:
    """GELU with the tanh approximation (`ops.gelu(x, approximate="tanh")`)."""
    var b = _current()
    return ops.mo_gelu_tanh(b, x.type, x)


def negative(x: Value) raises -> Value:
    var b = _current()
    return ops.mo_negative(b, x.type, x)


def equal(x: Value, y: Value) raises -> Value:
    var b = _current()
    return ops.equal(b, x, y)


def logsoftmax(x: Value, axis: Int = -1) raises -> Value:
    var b = _current()
    return ops.mo_reduce_logsoftmax(b, x.type, x, _axis(x, axis))


def gather(x: Value, indices: Value, axis: Int = 0) raises -> Value:
    """`x`'s slices along `axis` at `indices`: the result's shape is `x`'s
    with that axis replaced by `indices`' shape."""
    var a = _axis(x, axis)
    var t = TensorType(x.type.dtype, List[Dim](), x.type.device)
    for i in range(a):
        t.shape.append(x.type.shape[i].copy())
    for i in range(indices.type.rank()):
        t.shape.append(indices.type.shape[i].copy())
    for i in range(a + 1, x.type.rank()):
        t.shape.append(x.type.shape[i].copy())
    var b = _current()
    return ops.mo_gather(b, t, x, indices, a)


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


def scalar(value: Float64, like: Value) raises -> Value:
    """A rank-0 constant with `like`'s dtype and device: what `max.graph`
    makes of a Python number in `x * 0.5`."""
    return constant([value], like.type.dtype, List[Dim](), like.type.device)


# Mutable buffers. A load or store takes the chain that orders its device's
# buffer accesses and returns the next one; `max.graph` keeps that chain on
# the graph, and so do these (as `ops.buffer_load` / `ops.buffer_store`).


def buffer_load(buffer: Value) raises -> Value:
    """The buffer's current contents, as a value-semantic tensor."""
    var b = _current()
    var glue = Python.import_module(GLUE)
    var chain = value_from_py(glue.chain(b.graph, buffer.handle))
    var t = buffer.type.copy()
    t.is_buffer = False
    var results = ops.mo_mutable_load(b, t, buffer, chain)
    glue.set_chain(b.graph, buffer.handle, results[1].handle)
    return results[0].copy()


def buffer_store(buffer: Value, value: Value) raises:
    """Overwrites the buffer with `value`, in program order with every other
    load and store of its device."""
    var b = _current()
    var glue = Python.import_module(GLUE)
    var chain = value_from_py(glue.chain(b.graph, buffer.handle))
    var next = ops.mo_mutable_store(b, buffer, value, chain)
    glue.set_chain(b.graph, buffer.handle, next.handle)


def custom(
    kernels: String, symbol: String, values: List[Value], out_types: List[TensorType],
    parameter_names: List[String] = List[String](), parameter_values: List[Int] = List[Int](),
) raises -> List[Value]:
    """The Mojo custom op `symbol` (a kernel registered with
    `@extensibility.register`) from the kernel package at `kernels`, with
    the kernel's compile-time integer parameters, if any."""
    var b = _current()
    var builtins = Python.import_module("builtins")
    var handles = builtins.list()
    for i in range(len(values)):
        _ = handles.append(values[i].handle)
    var types = builtins.list()
    for i in range(len(out_types)):
        _ = types.append(_py_type(out_types[i]))
    var parameters = builtins.dict()
    for i in range(len(parameter_names)):
        parameters[PythonObject(parameter_names[i])] = PythonObject(parameter_values[i])
    var glue = Python.import_module(GLUE)
    return values_from_py(
        glue.custom(b.graph, PythonObject(kernels), PythonObject(symbol), handles, types, parameters)
    )
