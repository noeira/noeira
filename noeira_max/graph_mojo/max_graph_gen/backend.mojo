"""The hand-written half of the generated graph builder: dimensions, tensor
types, values, the argument list every generated op fills, and the backends.

`ops.mojo` is generated from MAX's op stubs. Each of its functions fills an
`OpArgs`: the op's stub class and MLIR name, and its arguments in the stub's
order, each named and tagged with a kind. It then calls `GraphBackend.add_op`.
A backend turns the arguments into an op:

- `PythonBackend` forwards to `max.graph`'s own `Graph._add_op_generated`
  through Mojo's Python interop (`noeira_max/graph_mojo/py_backend.py`). In
  MAX 26.6 this is the only way to build a graph.
- `CBackend` is the native backend's interface. MAX has no C API for
  building graphs, so it raises, naming the calls it would make.
"""

from std.python import Python, PythonObject

from . import api


comptime NO_DTYPE = DType.bool
"""A placeholder where no dtype applies (an argument of another kind, a
chain's type): Mojo's `DType` has no "invalid"."""


struct Dim(Copyable, Movable, Writable):
    """A tensor dimension: static (`Dim(64)`, or just `64`) or symbolic
    (`Dim.symbolic("batch")`)."""

    var size: Int
    """The static size; -1 when symbolic."""
    var name: String
    """The symbolic name; empty when static."""

    @implicit
    def __init__(out self, size: Int):
        self.size = size
        self.name = String("")

    @staticmethod
    def symbolic(name: String) -> Dim:
        var d = Dim(-1)
        d.name = name
        return d^

    def is_static(self) -> Bool:
        return self.name.byte_length() == 0

    def write_to(self, mut writer: Some[Writer]):
        if self.is_static():
            writer.write(self.size)
        else:
            writer.write(self.name)


struct TensorType(Copyable, Movable, Writable):
    """A tensor's dtype, shape and device (`"cpu"` or `"gpu"`). With
    `is_buffer`, a mutable buffer (`max.graph.BufferType`): a graph input
    the graph reads with `buffer_load` and writes with `buffer_store`."""

    var dtype: DType
    var shape: List[Dim]
    var device: String
    var is_buffer: Bool

    def __init__(
        out self, dtype: DType, var shape: List[Dim], device: String = "cpu", is_buffer: Bool = False
    ):
        self.dtype = dtype
        self.shape = shape^
        self.device = device
        self.is_buffer = is_buffer

    def rank(self) -> Int:
        return len(self.shape)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("BufferType(" if self.is_buffer else "TensorType(", self.dtype, ", [")
        for i in range(len(self.shape)):
            if i > 0:
                writer.write(", ")
            writer.write(self.shape[i])
        writer.write("], ", self.device, ")")


struct Value(Copyable, Movable):
    """A value in a graph under construction: the backend's handle, and its
    type (a chain has none)."""

    var handle: PythonObject
    """`PythonBackend`: the `max.graph` value. A native backend would keep
    its own handle here."""
    var type: TensorType
    var is_chain: Bool

    def __init__(out self, handle: PythonObject, var type: TensorType, is_chain: Bool = False):
        self.handle = handle
        self.type = type^
        self.is_chain = is_chain

    # Operators add to the current graph (`api`): the same broadcasting ops
    # as `max.graph`'s operators.

    def __add__(self, rhs: Value) raises -> Value:
        return api.add(self, rhs)

    def __sub__(self, rhs: Value) raises -> Value:
        return api.sub(self, rhs)

    def __mul__(self, rhs: Value) raises -> Value:
        return api.mul(self, rhs)

    def __truediv__(self, rhs: Value) raises -> Value:
        return api.div(self, rhs)

    def __matmul__(self, rhs: Value) raises -> Value:
        return api.matmul(self, rhs)


struct Arg(Copyable, Movable):
    """One named argument of an op, tagged with its kind. Only the payload
    field of that kind is meaningful."""

    var name: String
    var kind: String
    """`tensor` (an operand), `result_tensor`, `result_chain`,
    `param_decls`, `index`, `int`, `bool`, `bool_attr`, `string`,
    `string_attr`, `shape` or `dtype`."""
    var value: Optional[Value]
    var type: Optional[TensorType]
    var int_value: Int
    var bool_value: Bool
    var string_value: String
    var dims: List[Dim]
    var dtype: DType

    def __init__(out self, name: String, kind: String):
        self.name = name
        self.kind = kind
        self.value = None
        self.type = None
        self.int_value = 0
        self.bool_value = False
        self.string_value = String("")
        self.dims = List[Dim]()
        self.dtype = NO_DTYPE


struct OpArgs(Movable):
    """An op to add: its stub class, its MLIR name, and its arguments in the
    stub constructor's order."""

    var cls: String
    var mlir_name: String
    var args: List[Arg]

    def __init__(out self, cls: String, mlir_name: String):
        self.cls = cls
        self.mlir_name = mlir_name
        self.args = List[Arg]()

    def operand(mut self, name: String, value: Value):
        var a = Arg(name, "tensor")
        a.value = value.copy()
        self.args.append(a^)

    def result(mut self, name: String, type: TensorType):
        var a = Arg(name, "result_tensor")
        a.type = type.copy()
        self.args.append(a^)

    def result_chain(mut self, name: String):
        self.args.append(Arg(name, "result_chain"))

    def param_decls(mut self, name: String):
        """The op's output parameter declarations: always empty here, as
        `max.graph` fills them for a graph without symbolic parameters."""
        self.args.append(Arg(name, "param_decls"))

    def attr_index(mut self, name: String, value: Int):
        var a = Arg(name, "index")
        a.int_value = value
        self.args.append(a^)

    def attr_int(mut self, name: String, value: Int):
        var a = Arg(name, "int")
        a.int_value = value
        self.args.append(a^)

    def attr_bool(mut self, name: String, value: Bool):
        var a = Arg(name, "bool")
        a.bool_value = value
        self.args.append(a^)

    def attr_bool_attr(mut self, name: String, value: Bool):
        var a = Arg(name, "bool_attr")
        a.bool_value = value
        self.args.append(a^)

    def attr_string(mut self, name: String, value: String):
        var a = Arg(name, "string")
        a.string_value = value
        self.args.append(a^)

    def attr_string_attr(mut self, name: String, value: String):
        var a = Arg(name, "string_attr")
        a.string_value = value
        self.args.append(a^)

    def attr_shape(mut self, name: String, dims: List[Dim]):
        var a = Arg(name, "shape")
        a.dims = dims.copy()
        self.args.append(a^)

    def attr_dtype(mut self, name: String, dtype: DType):
        var a = Arg(name, "dtype")
        a.dtype = dtype
        self.args.append(a^)


trait GraphBackend:
    """What the generated ops call: add one op, return its results."""

    def add_op(mut self, var args: OpArgs) raises -> List[Value]:
        ...


# --- PythonBackend ---------------------------------------------------------

comptime GLUE = "noeira_max.graph_mojo.py_backend"


def _py_dims(dims: List[Dim]) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    for i in range(len(dims)):
        if dims[i].is_static():
            _ = out.append(PythonObject(dims[i].size))
        else:
            _ = out.append(PythonObject(dims[i].name))
    return out


def _py_type(t: TensorType) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    _ = out.append(PythonObject(String(t.dtype)))
    _ = out.append(_py_dims(t.shape))
    _ = out.append(PythonObject(t.device))
    _ = out.append(PythonObject(t.is_buffer))
    return out


def dtype_from_name(name: String) raises -> DType:
    """A `max.dtype.DType` name as a Mojo `DType`."""
    if name == "float32":
        return DType.float32
    if name == "float64":
        return DType.float64
    if name == "float16":
        return DType.float16
    if name == "bfloat16":
        return DType.bfloat16
    if name == "int64":
        return DType.int64
    if name == "int32":
        return DType.int32
    if name == "int16":
        return DType.int16
    if name == "int8":
        return DType.int8
    if name == "uint64":
        return DType.uint64
    if name == "uint32":
        return DType.uint32
    if name == "uint8":
        return DType.uint8
    if name == "bool":
        return DType.bool
    raise Error("max_graph_gen: no Mojo DType for MAX dtype " + name)


def value_from_py(pair: PythonObject) raises -> Value:
    """A `(value, describe(value))` pair from the glue as a `Value`."""
    var desc = pair[1]
    if desc is None:
        return Value(pair[0], TensorType(NO_DTYPE, List[Dim]()), is_chain=True)
    var builtins = Python.import_module("builtins")
    var dims = List[Dim]()
    var py_dims = desc[1]
    for i in range(Int(py=builtins.len(py_dims))):
        var d = py_dims[i]
        if Bool(py=builtins.isinstance(d, builtins.int)):
            dims.append(Dim(Int(py=d)))
        else:
            dims.append(Dim.symbolic(String(d)))
    var type = TensorType(
        dtype_from_name(String(desc[0])), dims^, String(desc[2]), is_buffer=Bool(py=desc[3])
    )
    return Value(pair[0], type^)


def values_from_py(pairs: PythonObject) raises -> List[Value]:
    var builtins = Python.import_module("builtins")
    var out = List[Value]()
    for i in range(Int(py=builtins.len(pairs))):
        out.append(value_from_py(pairs[i]))
    return out^


struct PythonBackend(GraphBackend, Copyable, Movable):
    """Builds into a `max.graph.Graph`, through Python."""

    var graph: PythonObject
    var glue: PythonObject

    def __init__(out self, graph: PythonObject) raises:
        self.graph = graph
        self.glue = Python.import_module(GLUE)

    @staticmethod
    def current() raises -> PythonBackend:
        """The graph `max.graph` considers current (the innermost open one)."""
        return PythonBackend(Python.import_module(GLUE).current_graph())

    def add_op(mut self, var args: OpArgs) raises -> List[Value]:
        var builtins = Python.import_module("builtins")
        var names = builtins.list()
        var kinds = builtins.list()
        var payloads = builtins.list()
        for i in range(len(args.args)):
            ref a = args.args[i]
            _ = names.append(PythonObject(a.name))
            _ = kinds.append(PythonObject(a.kind))
            _ = payloads.append(self._payload(a))
        return values_from_py(
            self.glue.add_op(self.graph, PythonObject(args.cls), names, kinds, payloads)
        )

    def _payload(self, a: Arg) raises -> PythonObject:
        if a.kind == "tensor":
            return a.value.value().handle
        if a.kind == "result_tensor":
            return _py_type(a.type.value())
        if a.kind == "index" or a.kind == "int":
            return PythonObject(a.int_value)
        if a.kind == "bool" or a.kind == "bool_attr":
            return PythonObject(a.bool_value)
        if a.kind == "string" or a.kind == "string_attr":
            return PythonObject(a.string_value)
        if a.kind == "shape":
            return _py_dims(a.dims)
        if a.kind == "dtype":
            return PythonObject(String(a.dtype))
        return Python.none()  # result_chain, param_decls: the glue makes them


# --- CBackend --------------------------------------------------------------


struct CBackend(GraphBackend, Movable):
    """The native backend's interface, for a C graph-building API that MAX
    26.6 does not have. It would map one `add_op` to:

        M_Graph *M_newGraph(const char *name, const M_TensorSpec **inputs, ...);
        M_Value **M_addOp(M_Graph *, const char *mlir_name,
                          M_Value **operands, size_t n_operands,
                          const M_Attr **attrs, size_t n_attrs,
                          const M_TensorSpec **result_types, size_t n_results,
                          M_Status *);
        void M_graphOutput(M_Graph *, M_Value **values, size_t n, M_Status *);
        M_AsyncCompiledModel *M_compileGraph(M_RuntimeContext *, M_Graph *, M_Status *);

    with `OpArgs.mlir_name` as the op name, and the kinds of `Arg` as the
    attribute encodings. No Python, and the result is a model the `maxrt`
    binding already runs.
    """

    def __init__(out self):
        pass

    def add_op(mut self, var args: OpArgs) raises -> List[Value]:
        raise Error(
            "max_graph_gen.CBackend: MAX 26.6 has no C API to build graphs"
            " (would call M_addOp for " + args.mlir_name + ")"
        )
