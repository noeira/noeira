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
from noeira.cuda.graph import CUDAGraph
from .gpu import gpu_copy
from .module_op import make_entry
from .nd import nd_reshape, nd_permute, nd_reduce, nd_narrow, nd_concat, nd_gather
from .extra import ex_clamp, ex_softmax
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
    RED_SUM,
    RED_MEAN,
    RED_MAX,
    EW_RELU,
    EW_TANH,
    EW_SIGMOID,
    EW_GELU,
    EW_GELU_TANH,
    EW_MISH,
    EW_SWISH,
    EW_ELU,
    EW_EXP,
    EW_LOG,
    EW_SQRT,
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
    def module[M: Module, //, B: Int](
        self, var m: M, name: String = String("")
    ) raises -> FModule[M]:
        """Hand `m` to the tape (it OWNS it from here) as an op for inputs
        `[B, M.IN_DIMS[0]]`. Make `m` for the tape's device. `name` (default
        `m<index>`) prefixes its parameters: `<name>.weight`, …"""
        var e = make_entry[M, B](m^, self.data[].on_gpu())
        return FModule[M](self.data, self.data[].add_module(e^, name))

    # ── gradient mode ────────────────────────────────────────────────────
    def no_grad(self) -> NoGrad:
        """`with tape.no_grad(): …` — inference. Nodes recorded inside do not
        require grad: no gradient buffers, no saved pre-activations, no
        backward, and a module may be called any number of times."""
        return NoGrad(self.data)

    def set_grad_enabled(self, on: Bool):
        self.data[].grad_on = on

    def grad_enabled(self) -> Bool:
        return self.data[].grad_on

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

    # ── CUDA graph capture (NVIDIA) ──────────────────────────────────────
    #
    # A steady-state step re-records the same nodes into the same slots, so
    # every buffer address is the same each step: the step can be captured
    # ONCE and replayed. Run it eagerly at least once first (slots, scratch
    # and cuBLASLt plans are created on the first call; capture forbids
    # allocation). Inside the captured step: no host reads (`item`,
    # `values`) and no host inputs — feed `input_device` from a buffer you
    # refill between replays.
    #
    # The OPTIMIZER can be inside the step once it is ADOPTED:
    # `adam.adopt["gpu"](tape, ctx)` packs the tape's parameters and its
    # modules' into nn's arena, whose step is one kernel and whose bias
    # correction β^t advances ON THE DEVICE, per replay. An un-adopted Adam
    # bakes the capture step's host `1 - β^t` into the graph and replays it
    # (`tests/experimental/functional/test_functional_capture_adam.mojo`, control arm D).
    #
    #     adam.adopt["gpu"](tape, ctx)    # once, after every param/module
    #     step(tape)                      # eager: allocates
    #     tape.capture_begin(); step(tape); tape.capture_end()  # records, then runs it once
    #     for …: refill(src); tape.replay()   # forward, backward, Adam
    #
    # Off NVIDIA `capture_begin/end` run the step eagerly and `captured()`
    # stays False: check it and run the step yourself.

    def capture_begin(self) raises:
        if not self.data[].ctx:
            raise Error("Tape.capture_begin: a CPU tape")
        self.data[].graph = CUDAGraph(self.data[].dctx())
        self.data[].graph.value().begin_capture()

    def capture_end(self, run: Bool = True) raises:
        """End the capture and, by default, REPLAY once. Stream capture
        RECORDS the step without executing it, so the step written between
        `capture_begin` and `capture_end` has not happened yet: a training
        step captured and not run silently drops one optimizer update (seen:
        a captured Adam run trained on batches 0, 2, 3, … and diverged from
        eager by 13 %). `run=False` only records."""
        self.data[].graph.value().end_capture()
        if run and self.captured():
            self.replay()

    def captured(self) -> Bool:
        return Bool(self.data[].graph) and self.data[].graph.value().is_captured()

    def replay(self) raises:
        """Re-run the captured step on the context's own stream: ordered
        after whatever was enqueued before (an input refill) and before what
        comes after (a read, an optimizer step) with no extra sync."""
        if not self.captured():
            raise Error(
                "Tape.replay: nothing captured (capture is NVIDIA-only —"
                " run the step eagerly when `captured()` is False)"
            )
        self.data[].graph.value().replay_on_mojo_stream()

    def synchronize(self) raises:
        if self.data[].ctx:
            self.data[].dctx().synchronize()

    def _wrap(self, id: Int) -> FTensor:
        return FTensor(self.data, id)

    def param_names(self) -> List[String]:
        var out = List[String]()
        for i in range(self.data[].n_params):
            out.append(self.data[].nodes[i].name)
        return out^

    # ── parameters (persist across `reset`) ──────────────────────────────
    def param[INIT: Initializer](
        self, rows: Int, cols: Int, name: String = String("")
    ) raises -> FTensor:
        """A `[rows, cols]` parameter initialised as the comptime nn
        initialises a `Linear` weight of fan_in `rows`, fan_out `cols`.
        `name` keys checkpoints (default `p<index>`): name what you save."""
        var i = self.data[].param(rows, cols, name)
        INIT.init_weight["cpu"](self.data[].nodes[i].val, rows * cols, rows, cols, None)
        self.data[].upload_val(i)
        return self._wrap(i)

    def param(
        self, rows: Int, cols: Int, values: List[Scalar[DT]],
        name: String = String(""),
    ) raises -> FTensor:
        if len(values) != rows * cols:
            raise Error("Tape.param: values do not fill the shape")
        var i = self.data[].param(rows, cols, name)
        for k in range(rows * cols):
            self.data[].nodes[i].val.data[k] = values[k]
        self.data[].upload_val(i)
        return self._wrap(i)

    def zeros(
        self, rows: Int, cols: Int, name: String = String("")
    ) raises -> FTensor:
        """A zero-initialised parameter (a bias)."""
        var i = self.data[].param(rows, cols, name)
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

    def input(
        self,
        shape: List[Int],
        values: List[Scalar[DT]],
        requires_grad: Bool = False,
    ) raises -> FTensor:
        """An N-d input (rank 1 to 4), row-major `values`."""
        var r = len(shape)
        if r < 1 or r > 4:
            raise Error("Tape.input: rank 1 to 4")
        var d = List[Int](length=4, fill=1)
        for i in range(r):
            d[4 - r + i] = shape[i]
        var t = self.input(d[0] * d[1] * d[2], d[3], values, requires_grad)
        self.data[].nodes[t.id].set_dims(d[0], d[1], d[2], d[3], r)
        return t

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
    var gen: Int
    """The tape's step when this handle was made (`TapeData.gen`)."""

    def __init__(out self, tape: ArcPointer[TapeData], id: Int):
        self.tape = tape
        self.id = id
        self.gen = tape[].gen

    def _new(self, id: Int) -> Self:
        return Self(self.tape, id)

    def _vid(self) raises -> Int:
        """`id`, checked: a node from before the last `reset` has had its
        slot handed to another node, so reading it would read THAT node.
        Parameters persist and are always valid."""
        if self.id >= self.tape[].n_params and self.gen != self.tape[].gen:
            raise Error(
                "FTensor: recorded before the last reset() — its slot now"
                " holds another node (keep parameters, re-record activations)"
            )
        return self.id

    # ── shape and values ─────────────────────────────────────────────────
    def rows(self) -> Int:
        return self.tape[].nodes[self.id].rows

    def cols(self) -> Int:
        return self.tape[].nodes[self.id].cols

    def size(self) -> Int:
        return self.rows() * self.cols()

    def shape(self) -> List[Int]:
        return self.tape[].nodes[self.id].shape()

    def rank(self) -> Int:
        return self.tape[].nodes[self.id].rank

    # ── N-d (nd.mojo) ────────────────────────────────────────────────────
    def reshape(self, shape: List[Int]) raises -> Self:
        """Same elements, row-major, another shape (one `-1` inferred). A
        copy: the tape's nodes own their storage."""
        return self._new(nd_reshape(self.tape[], self._vid(), shape))

    def permute(self, perm: List[Int]) raises -> Self:
        """Output axis `i` is input axis `perm[i]` (torch's `permute`)."""
        return self._new(nd_permute(self.tape[], self._vid(), perm))

    def transpose(self, i: Int, j: Int) raises -> Self:
        var r = self.rank()
        var perm = List[Int]()
        for k in range(r):
            perm.append(k)
        var a = i + r if i < 0 else i
        var b = j + r if j < 0 else j
        if a < 0 or a >= r or b < 0 or b >= r:
            raise Error("transpose: axis out of range")
        perm[a] = b
        perm[b] = a
        return self.permute(perm)

    def sum(self, axis: Int, keepdim: Bool = False) raises -> Self:
        return self._new(nd_reduce(self.tape[], self._vid(), axis, RED_SUM, keepdim))

    def mean(self, axis: Int, keepdim: Bool = False) raises -> Self:
        return self._new(nd_reduce(self.tape[], self._vid(), axis, RED_MEAN, keepdim))

    def max(self, axis: Int, keepdim: Bool = False) raises -> Self:
        """Gradient to the FIRST maximum of each slice."""
        return self._new(nd_reduce(self.tape[], self._vid(), axis, RED_MAX, keepdim))

    def narrow(self, axis: Int, start: Int, length: Int) raises -> Self:
        """`length` entries of `axis` from `start` (torch's `narrow`)."""
        return self._new(nd_narrow(self.tape[], self._vid(), axis, start, length))

    def values(self) raises -> List[Scalar[DT]]:
        """Every value, row-major, on the host (a GPU tape synchronises)."""
        self.tape[].download_val(self._vid())
        var out = List[Scalar[DT]]()
        var p = self.tape[].val_ptr(self._vid())
        for k in range(self.size()):
            out.append(p[unsafe_offset=k])
        return out^

    def grads(self) raises -> List[Scalar[DT]]:
        """Every gradient element (after `backward`), on the host."""
        self.tape[].download_grad(self._vid())
        var out = List[Scalar[DT]]()
        var p = self.tape[].grad_ptr(self._vid())
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
        self.tape[].download_val(self._vid())
        return self.tape[].val_ptr(self._vid())[unsafe_offset=k]

    def grad(self, k: Int) raises -> Scalar[DT]:
        """Flat element `k` of the gradient (after `backward`)."""
        self.tape[].download_grad(self._vid())
        return self.tape[].grad_ptr(self._vid())[unsafe_offset=k]

    def set_value(self, k: Int, v: Scalar[DT]) raises:
        self.tape[].download_val(self._vid())
        self.tape[].val_ptr(self._vid())[unsafe_offset=k] = v
        self.tape[].upload_val(self._vid())

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
        return self._new(self.tape[].binary[OP_ADD](self._vid(), other._vid()))

    def __sub__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_SUB](self._vid(), other._vid()))

    def __mul__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_MUL](self._vid(), other._vid()))

    def __truediv__(self, other: Self) raises -> Self:
        return self._new(self.tape[].binary[OP_DIV](self._vid(), other._vid()))

    def __matmul__(self, other: Self) raises -> Self:
        return self._new(self.tape[].matmul(self._vid(), other._vid()))

    # ── tensor ⊕ scalar ──────────────────────────────────────────────────
    def __add__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_SHIFT](self._vid(), c))

    def __radd__(self, c: Scalar[DT]) raises -> Self:
        return self + c

    def __sub__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_SHIFT](self._vid(), -c))

    def __rsub__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_RSUBC](self._vid(), c))

    def __mul__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_SCALE](self._vid(), c))

    def __rmul__(self, c: Scalar[DT]) raises -> Self:
        return self * c

    def __truediv__(self, c: Scalar[DT]) raises -> Self:
        return self._new(self.tape[].scalar_op[OP_DIVC](self._vid(), c))

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
        """The transpose of the LAST two axes (`x.T()`: Mojo has no
        properties); the rank-2 kernel for a matrix."""
        if self.rank() > 2:
            return self.transpose(-2, -1)
        return self._new(self.tape[].transpose(self._vid()))

    def sum(self) raises -> Self:
        return self._new(self.tape[].reduce[OP_SUM](self._vid()))

    def mean(self) raises -> Self:
        return self._new(self.tape[].reduce[OP_MEAN](self._vid()))

    def sum_rows(self) raises -> Self:
        """`[R, C] -> [1, C]`."""
        return self._new(self.tape[].sum_rows(self._vid()))

    # ── activations (the comptime nn's `ElementOp`s) ─────────────────────
    def relu(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_RELU))

    def tanh(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_TANH))

    def sigmoid(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_SIGMOID))

    def gelu(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_GELU))

    def gelu_tanh(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_GELU_TANH))

    def mish(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_MISH))

    def swish(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_SWISH))

    def elu(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_ELU))

    def exp(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_EXP))

    def log(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_LOG))

    def sqrt(self) raises -> Self:
        return self._new(self.tape[].elementwise(self._vid(), EW_SQRT))

    def clamp(self, lo: Scalar[DT], hi: Scalar[DT]) raises -> Self:
        """Gradient passes inside [lo, hi], bounds included (torch's)."""
        return self._new(ex_clamp(self.tape[], self._vid(), lo, hi))

    def softmax(self) raises -> Self:
        """Over the LAST axis (`permute` first for another)."""
        return self._new(ex_softmax(self.tape[], self._vid(), False))

    def log_softmax(self) raises -> Self:
        """Over the LAST axis."""
        return self._new(ex_softmax(self.tape[], self._vid(), True))

    # ── autodiff ─────────────────────────────────────────────────────────
    def backward(self) raises:
        self.tape[].backward(self._vid())


struct FModule[M: Module](ImplicitlyCopyable):
    """A comptime nn module the tape owns. `m(x)` records it as one op;
    `m.get()` reaches the module itself (its parameters, a checkpoint)."""

    var tape: ArcPointer[TapeData]
    var id: Int

    def __init__(out self, tape: ArcPointer[TapeData], id: Int):
        self.tape = tape
        self.id = id

    def __call__(self, x: FTensor) raises -> FTensor:
        return FTensor(self.tape, self.tape[].module_call(self.id, x._vid()))

    def get(mut self) -> ref[self] Self.M:
        """The module, owned by the tape. The reference's origin is THIS
        handle, so the handle — and the tape it keeps alive — outlives every
        use of it. (An untracked pointer did not: Mojo ends a value at its
        last use, the handle died right after `get()`, it was the tape's
        last owner, and the next read was of freed memory.)"""
        return self.tape[].mods[self.id].state.unsafe_bitcast[
            Self.M
        ]().unsafe_origin_cast[origin_of(self)]()[]


struct NoGrad:
    """The guard `Tape.no_grad()` returns: gradients off inside the `with`,
    restored to what they were on the way out."""

    var tape: ArcPointer[TapeData]
    var was_on: Bool

    def __init__(out self, tape: ArcPointer[TapeData]):
        self.tape = tape
        self.was_on = tape[].grad_on

    def __enter__(self):
        self.tape[].grad_on = False

    def __exit__(self):
        self.tape[].grad_on = self.was_on

    def __exit__(self, err: Error) -> Bool:
        self.tape[].grad_on = self.was_on
        return False
