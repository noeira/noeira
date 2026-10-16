"""`Tape` and `FTensor` — the functional mode's surface.

```mojo
var tape = Tape()                           # or Tape(DeviceContext()): GPU
var w = tape.param[Kaiming](4, 16)          # [IN, OUT], like `Linear`
var b = tape.zeros(1, 16)
var x = tape.input(batch, 4, xs)           # shapes are runtime values
var h = (x @ w + b).relu()                 # operators record on the tape
var loss = ((h - y) ** 2).mean()
loss.backward()
```

An `FTensor` is a HANDLE: the tape it lives on (shared, reference-counted)
and its node id. Copying one copies the handle, never the data. Every
operator appends a node, so the network is whatever code ran — loops, `if`s
and shapes chosen at run time included.
"""

from std.memory import ArcPointer
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.module import Module
from noeira.nn.core.param import (
    ParamVisitor,
    ParamVisitorRef,
    IsParamVisitorRef,
    ParamWalkable,
)
from .gpu import gpu_copy
from .module_op import make_entry
from .tape import (
    TapeData,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_SCALE,
    OP_SHIFT,
    OP_DIVC,
    OP_RSUBC,
    OP_SUM,
    OP_MEAN,
    EW_RELU,
    EW_TANH,
    EW_SIGMOID,
    EW_GELU,
    EW_GELU_TANH,
    EW_MISH,
    EW_SWISH,
    EW_ELU,
)


struct Tape(ImplicitlyCopyable, ParamWalkable):
    """A handle on one tape. Copies share it.

    `ParamWalkable`: nn's optimizers and checkpoint walks take a tape as
    they take a model — `adam.step["cpu"](tape)` updates the tape's own
    parameters AND every module handed to it, in one step."""

    var data: ArcPointer[TapeData]

    def __init__(out self):
        """A CPU tape."""
        self.data = ArcPointer(TapeData())

    def __init__(out self, ctx: DeviceContext):
        """A GPU tape: every value and gradient lives on `ctx`'s device."""
        self.data = ArcPointer(TapeData(ctx))

    def on_gpu(self) -> Bool:
        return self.data[].on_gpu()

    # ── comptime nn modules ──────────────────────────────────────────────
    def module[M: Module, //, B: Int](self, var m: M) -> FModule[M]:
        """Hand `m` to the tape (it OWNS it from here) as an op for inputs
        `[B, M.IN_DIMS[0]]`. Make `m` for the tape's device."""
        var e = make_entry[M, B](m^, self.data[].on_gpu())
        return FModule[M](self.data, self.data[].add_module(e^))

    # ── ParamWalkable ────────────────────────────────────────────────────
    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Only through a `ParamVisitorRef` (what `walk_params`, `Adam.step`
        and `SGD.step` drive): the tape's parameters have RUNTIME sizes,
        which a concrete visitor's `visit[target, N]` cannot take."""
        comptime if conforms_to(V, IsParamVisitorRef):
            ref r = Pointer(to=visitor).unsafe_bitcast[ParamVisitorRef]()[]
            self.data[].walk_params(r, ctx, prefix)
        else:
            raise Error(
                "Tape.for_each_param: drive the walk through a"
                " ParamVisitorRef (walk_params / Adam.step / SGD.step)"
            )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        self.data[].zero_grad()

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        comptime if conforms_to(V, IsParamVisitorRef):
            ref r = Pointer(to=visitor).unsafe_bitcast[ParamVisitorRef]()[]
            self.data[].walk_states(r, ctx, prefix)
        else:
            raise Error("Tape.for_each_state: drive it through a ParamVisitorRef")

    def synchronize(self) raises:
        if self.data[].ctx:
            self.data[].dctx().synchronize()

    def _wrap(self, id: Int) -> FTensor:
        return FTensor(self.data, id)

    # ── parameters (persist across `reset`) ──────────────────────────────
    def param[INIT: Initializer](self, rows: Int, cols: Int) raises -> FTensor:
        """A `[rows, cols]` parameter initialised as the comptime nn
        initialises a `Linear` weight of fan_in `rows`, fan_out `cols`."""
        var i = self.data[].param(rows, cols)
        INIT.init_weight["cpu"](self.data[].nodes[i].val, rows * cols, rows, cols, None)
        self.data[].upload_val(i)
        return self._wrap(i)

    def param(
        self, rows: Int, cols: Int, values: List[Scalar[DT]]
    ) raises -> FTensor:
        if len(values) != rows * cols:
            raise Error("Tape.param: values do not fill the shape")
        var i = self.data[].param(rows, cols)
        for k in range(rows * cols):
            self.data[].nodes[i].val.data[k] = values[k]
        self.data[].upload_val(i)
        return self._wrap(i)

    def zeros(self, rows: Int, cols: Int) raises -> FTensor:
        """A zero-initialised parameter (a bias)."""
        var i = self.data[].param(rows, cols)
        for k in range(rows * cols):
            self.data[].nodes[i].val.data[k] = 0
        self.data[].upload_val(i)
        return self._wrap(i)

    # ── inputs (dropped by `reset`) ──────────────────────────────────────
    def input(
        self,
        rows: Int,
        cols: Int,
        values: List[Scalar[DT]],
        requires_grad: Bool = False,
    ) raises -> FTensor:
        if len(values) < rows * cols:
            raise Error("Tape.input: values do not fill the shape")
        var i = self.data[].leaf(rows, cols, requires_grad)
        self.data[]._host_fit(i, False)
        var p = self.data[].val_ptr(i)
        for k in range(rows * cols):
            p[unsafe_offset=k] = values[k]
        self.data[].upload_val(i)
        return self._wrap(i)

    def input_from(
        self,
        rows: Int,
        cols: Int,
        src: Pointer[Scalar[DT], ImmutAnyOrigin],
        requires_grad: Bool = False,
    ) raises -> FTensor:
        """`input` from raw host memory (an env's observation buffer, a
        `Tensor`'s `data.unsafe_ptr()`), copied in."""
        var i = self.data[].leaf(rows, cols, requires_grad)
        self.data[]._host_fit(i, False)
        var p = self.data[].val_ptr(i)
        for k in range(rows * cols):
            p[unsafe_offset=k] = src[unsafe_offset=k]
        self.data[].upload_val(i)
        return self._wrap(i)

    def input_device(
        self,
        rows: Int,
        cols: Int,
        src: DeviceBuffer[DT],
        requires_grad: Bool = False,
    ) raises -> FTensor:
        """`input` from memory already on the tape's device (a GPU env's
        observation batch): one device copy, no host round trip."""
        if not self.data[].ctx:
            raise Error("Tape.input_device: a CPU tape")
        var i = self.data[].leaf(rows, cols, requires_grad)
        var ctx = self.data[].dctx()
        gpu_copy(ctx, self.data[].dv(i), src, rows * cols)
        return self._wrap(i)

    def scalar(self, value: Scalar[DT]) raises -> FTensor:
        var i = self.data[].leaf(1, 1, False)
        self.data[]._host_fit(i, False)
        self.data[].val_ptr(i)[unsafe_offset=0] = value
        self.data[].upload_val(i)
        return self._wrap(i)

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        """Forget every activation; keep parameters and every buffer. Call
        once per step: the next step then allocates nothing."""
        self.data[].reset()

    def zero_grad(self) raises:
        self.data[].zero_grad()

    def params(self) -> List[FTensor]:
        var out = List[FTensor]()
        for i in range(self.data[].n_params):
            out.append(self._wrap(i))
        return out^

    def num_nodes(self) -> Int:
        return self.data[].n_live

    def sgd_step(self, lr: Scalar[DT]) raises:
        """`p -= lr · grad` on every parameter."""
        self.data[].sgd_step(lr)


struct FTensor(ImplicitlyCopyable, Writable):
    var tape: ArcPointer[TapeData]
    var id: Int

    def __init__(out self, tape: ArcPointer[TapeData], id: Int):
        self.tape = tape
        self.id = id

    def _new(self, id: Int) -> Self:
        return Self(self.tape, id)

    # ── shape and values ─────────────────────────────────────────────────
    def rows(self) -> Int:
        return self.tape[].nodes[self.id].rows

    def cols(self) -> Int:
        return self.tape[].nodes[self.id].cols

    def size(self) -> Int:
        return self.rows() * self.cols()

    def values(self) raises -> List[Scalar[DT]]:
        """Every value, row-major, on the host (a GPU tape synchronises)."""
        self.tape[].download_val(self.id)
        var out = List[Scalar[DT]]()
        var p = self.tape[].val_ptr(self.id)
        for k in range(self.size()):
            out.append(p[unsafe_offset=k])
        return out^

    def grads(self) raises -> List[Scalar[DT]]:
        """Every gradient element (after `backward`), on the host."""
        self.tape[].download_grad(self.id)
        var out = List[Scalar[DT]]()
        var p = self.tape[].grad_ptr(self.id)
        for k in range(self.size()):
            out.append(p[unsafe_offset=k])
        return out^

    def __getitem__(self, i: Int, j: Int) raises -> Scalar[DT]:
        return self.value(i * self.cols() + j)

    def item(self) raises -> Scalar[DT]:
        if self.size() != 1:
            raise Error("item: not a [1, 1] tensor")
        return self.value(0)

    def value(self, k: Int) raises -> Scalar[DT]:
        """Flat (row-major) element `k`. On a GPU tape this downloads the
        whole tensor: read many through `values()`."""
        self.tape[].download_val(self.id)
        return self.tape[].val_ptr(self.id)[unsafe_offset=k]

    def grad(self, k: Int) raises -> Scalar[DT]:
        """Flat element `k` of the gradient (after `backward`)."""
        self.tape[].download_grad(self.id)
        return self.tape[].grad_ptr(self.id)[unsafe_offset=k]

    def set_value(self, k: Int, v: Scalar[DT]) raises:
        self.tape[].download_val(self.id)
        self.tape[].val_ptr(self.id)[unsafe_offset=k] = v
        self.tape[].upload_val(self.id)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FTensor[", self.rows(), ", ", self.cols(), "](")
        try:
            var vals = self.values()
            var n = min(len(vals), 8)
            for k in range(n):
                if k > 0:
                    writer.write(", ")
                writer.write(vals[k])
            if len(vals) > n:
                writer.write(", …")
        except:
            writer.write("<unreadable>")
        writer.write(")")

    # ── tensor ⊕ tensor (numpy broadcasting) ─────────────────────────────
    def __add__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_ADD](self.id, other.id))

    def __sub__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_SUB](self.id, other.id))

    def __mul__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_MUL](self.id, other.id))

    def __truediv__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_DIV](self.id, other.id))

    def __matmul__(self, other: Self) raises -> Self:
        return self._new(self.tape[].matmul(self.id, other.id))

    # ── tensor ⊕ scalar ──────────────────────────────────────────────────
    def __add__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_SHIFT](self.id, c))

    def __radd__(self, c: Scalar[DT]) raises -> Self:
        return self + c

    def __sub__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_SHIFT](self.id, -c))

    def __rsub__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_RSUBC](self.id, c))

    def __mul__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_SCALE](self.id, c))

    def __rmul__(self, c: Scalar[DT]) raises -> Self:
        return self * c

    def __truediv__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_DIVC](self.id, c))

    def __neg__(self) raises -> Self:
        return self * -1.0

    def __pow__(self, p: Int) raises -> Self:
        """Integer powers by repeated product (`x ** 2` is `x * x`)."""
        if p < 1:
            raise Error("__pow__: only positive integer powers")
        var out = self
        for _ in range(p - 1):
            out = out * self
        return out

    # ── shape ops, reductions ────────────────────────────────────────────
    def T(self) raises -> Self:
        """The transpose (`x.T()`: Mojo has no properties)."""
        return self._new(self.tape[].transpose(self.id))

    def sum(self) raises -> Self:
        return self._new(self.tape[].reduce[OP_SUM](self.id))

    def mean(self) raises -> Self:
        return self._new(self.tape[].reduce[OP_MEAN](self.id))

    def sum_rows(self) raises -> Self:
        """`[R, C] -> [1, C]`."""
        return self._new(self.tape[].sum_rows(self.id))

    # ── activations (the comptime nn's `ElementOp`s) ─────────────────────
    def relu(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_RELU))

    def tanh(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_TANH))

    def sigmoid(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_SIGMOID))

    def gelu(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_GELU))

    def gelu_tanh(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_GELU_TANH))

    def mish(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_MISH))

    def swish(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_SWISH))

    def elu(self) raises -> Self:
        return self._new(self.tape[].elementwise(self.id, EW_ELU))

    # ── autodiff ─────────────────────────────────────────────────────────
    def backward(self) raises:
        self.tape[].backward(self.id)


struct FModule[M: Module](ImplicitlyCopyable):
    """A comptime nn module the tape owns. `m(x)` records it as one op;
    `m.get()` reaches the module itself (its parameters, a checkpoint)."""

    var tape: ArcPointer[TapeData]
    var id: Int

    def __init__(out self, tape: ArcPointer[TapeData], id: Int):
        self.tape = tape
        self.id = id

    def __call__(self, x: FTensor) raises -> FTensor:
        return FTensor(self.tape, self.tape[].module_call(self.id, x.id))

    def get(mut self) -> ref[self] Self.M:
        """The module, owned by the tape. The reference's origin is THIS
        handle, so the handle — and the tape it keeps alive — outlives every
        use of it. (An untracked pointer did not: Mojo ends a value at its
        last use, the handle died right after `get()`, it was the tape's
        last owner, and the next read was of freed memory.)"""
        return self.tape[].mods[self.id].state.unsafe_bitcast[
            Self.M
        ]().unsafe_origin_cast[origin_of(self)]()[]
