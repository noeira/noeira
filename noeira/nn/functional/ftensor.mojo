"""`Tape` and `FTensor` — the functional mode's surface.

```mojo
var tape = Tape()
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

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
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


struct Tape(ImplicitlyCopyable):
    """A handle on one tape. Copies share it."""

    var data: ArcPointer[TapeData]

    def __init__(out self):
        self.data = ArcPointer(TapeData())

    def _wrap(self, id: Int) -> FTensor:
        return FTensor(self.data, id)

    # ── parameters (persist across `reset`) ──────────────────────────────
    def param[INIT: Initializer](self, rows: Int, cols: Int) raises -> FTensor:
        """A `[rows, cols]` parameter initialised as the comptime nn
        initialises a `Linear` weight of fan_in `rows`, fan_out `cols`."""
        var i = self.data[].param(rows, cols)
        INIT.init_weight["cpu"](self.data[].nodes[i].val, rows * cols, rows, cols, None)
        return self._wrap(i)

    def param(
        self, rows: Int, cols: Int, values: List[Scalar[DT]]
    ) raises -> FTensor:
        if len(values) != rows * cols:
            raise Error("Tape.param: values do not fill the shape")
        var i = self.data[].param(rows, cols)
        for k in range(rows * cols):
            self.data[].nodes[i].val.data[k] = values[k]
        return self._wrap(i)

    def zeros(self, rows: Int, cols: Int) raises -> FTensor:
        """A zero-initialised parameter (a bias)."""
        var i = self.data[].param(rows, cols)
        for k in range(rows * cols):
            self.data[].nodes[i].val.data[k] = 0
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
        var p = self.data[].val_ptr(i)
        for k in range(rows * cols):
            p[unsafe_offset=k] = values[k]
        return self._wrap(i)

    def input_from(
        self,
        rows: Int,
        cols: Int,
        src: Pointer[Scalar[DT], ImmutAnyOrigin],
        requires_grad: Bool = False,
    ) -> FTensor:
        """`input` from raw memory (an env's observation buffer, a `Tensor`'s
        `data.unsafe_ptr()`), copied in."""
        var i = self.data[].leaf(rows, cols, requires_grad)
        var p = self.data[].val_ptr(i)
        for k in range(rows * cols):
            p[unsafe_offset=k] = src[unsafe_offset=k]
        return self._wrap(i)

    def scalar(self, value: Scalar[DT]) -> FTensor:
        var i = self.data[].leaf(1, 1, False)
        self.data[].val_ptr(i)[unsafe_offset=0] = value
        return self._wrap(i)

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self):
        """Forget every activation; keep parameters and every buffer. Call
        once per step: the next step then allocates nothing."""
        self.data[].reset()

    def zero_grad(self):
        self.data[].zero_grad()

    def params(self) -> List[FTensor]:
        var out = List[FTensor]()
        for i in range(self.data[].n_params):
            out.append(self._wrap(i))
        return out^

    def num_nodes(self) -> Int:
        return self.data[].n_live

    def sgd_step(self, lr: Scalar[DT]):
        """`p -= lr · grad` on every parameter."""
        for i in range(self.data[].n_params):
            if self.data[].nodes[i].rg:
                var v = self.data[].val_ptr(i)
                var g = self.data[].grad_ptr(i)
                for k in range(self.data[].nodes[i].size()):
                    v[unsafe_offset=k] -= lr * g[unsafe_offset=k]


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

    def __getitem__(self, i: Int, j: Int) -> Scalar[DT]:
        return self.tape[].val_ptr(self.id)[unsafe_offset=i * self.cols() + j]

    def item(self) raises -> Scalar[DT]:
        if self.size() != 1:
            raise Error("item: not a [1, 1] tensor")
        return self.tape[].val_ptr(self.id)[unsafe_offset=0]

    def value(self, k: Int) -> Scalar[DT]:
        """Flat (row-major) element `k`."""
        return self.tape[].val_ptr(self.id)[unsafe_offset=k]

    def grad(self, k: Int) -> Scalar[DT]:
        """Flat element `k` of the gradient (after `backward`)."""
        return self.tape[].grad_ptr(self.id)[unsafe_offset=k]

    def set_value(self, k: Int, v: Scalar[DT]):
        self.tape[].val_ptr(self.id)[unsafe_offset=k] = v

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FTensor[", self.rows(), ", ", self.cols(), "](")
        var n = min(self.size(), 8)
        for k in range(n):
            if k > 0:
                writer.write(", ")
            writer.write(self.value(k))
        if self.size() > n:
            writer.write(", …")
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
    def __add__(self, c: Scalar[DT]) -> Self:
        return self._new(self.tape[].scalar_op[OP_SHIFT](self.id, c))

    def __radd__(self, c: Scalar[DT]) -> Self:
        return self + c

    def __sub__(self, c: Scalar[DT]) -> Self:
        return self._new(self.tape[].scalar_op[OP_SHIFT](self.id, -c))

    def __rsub__(self, c: Scalar[DT]) -> Self:
        return self._new(self.tape[].scalar_op[OP_RSUBC](self.id, c))

    def __mul__(self, c: Scalar[DT]) -> Self:
        return self._new(self.tape[].scalar_op[OP_SCALE](self.id, c))

    def __rmul__(self, c: Scalar[DT]) -> Self:
        return self * c

    def __truediv__(self, c: Scalar[DT]) -> Self:
        return self._new(self.tape[].scalar_op[OP_DIVC](self.id, c))

    def __neg__(self) -> Self:
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
    def T(self) -> Self:
        """The transpose (`x.T()`: Mojo has no properties)."""
        return self._new(self.tape[].transpose(self.id))

    def sum(self) -> Self:
        return self._new(self.tape[].reduce[OP_SUM](self.id))

    def mean(self) -> Self:
        return self._new(self.tape[].reduce[OP_MEAN](self.id))

    def sum_rows(self) -> Self:
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
