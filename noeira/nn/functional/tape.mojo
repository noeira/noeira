"""TapeData — the engine behind the functional mode: a list of nodes whose
shapes are RUNTIME values, recorded as the ops run, replayed backwards.

The comptime nn composes each module's VJP at compile time: `Sequential`
knows its children's types, so the type IS the tape. A network wired at run
time has no type to carry that structure, so this records it: one `Node` per
op (op tag, input ids, saved value), and `backward` walks them in reverse,
dispatching on the tag.

Mojo has no trait objects and no storable closures, so the op set is CLOSED:
a new op is a tag, a forward branch and a backward branch here. Elementwise
activations are the exception that stays open-ish: the per-element math is
the comptime nn's own `ElementOp` structs (`ReLUOp`, `TanhOp`, …), so both
modes run one definition of each activation and its derivative.

Shapes are rank 2 (`rows`, `cols`); a scalar is `[1, 1]`. Binary ops
broadcast numpy-style along any axis of extent 1 — a bias is `[1, N]`.

Lifecycle: parameters are the first `n_params` nodes and persist; `reset`
drops every node after them but KEEPS the node structs, so the next step
reuses their buffers and a steady-state step allocates nothing.

DEVICE: a tape built with a `DeviceContext` keeps every value and gradient
on the GPU (`Tensor.dev`) and runs each op as the runtime-sized kernels of
`gpu.mojo`; without one it runs on the CPU (`Tensor.data`). The choice is a
RUNTIME branch per op, so one binary serves both. Host copies happen only
when asked for (`download_val`, inputs, parameter init).

Values are fp32 (`DT`).
"""

from std.math import max as fmax
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, CPU_SIMD_W
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.cpu_gemm import gemm_cpu_ptr
from noeira.nn.core.element_op import ElementOp
from noeira.nn.primitives.ops.relu_op import ReLUOp
from noeira.nn.primitives.ops.tanh_op import TanhOp
from noeira.nn.primitives.ops.sigmoid_op import SigmoidOp
from noeira.nn.primitives.ops.gelu_op import GELUOp
from noeira.nn.primitives.ops.gelu_tanh_op import GELUTanhOp
from noeira.nn.primitives.ops.mish_op import MishOp
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.nn.primitives.ops.elu_op import ELUOp
from noeira.nn.core.param import ParamVisitorRef
from noeira.nn.core.walkers import join_name
from .opcodes import *
from .module_op import ModEntry
from .gpu import (
    gpu_binary,
    gpu_binary_bwd,
    gpu_scalar,
    gpu_scalar_bwd,
    gpu_ew,
    gpu_ew_bwd,
    gpu_reduce,
    gpu_reduce_bwd,
    gpu_sum_rows,
    gpu_sum_rows_bwd,
    gpu_transpose,
    gpu_axpy,
    gpu_copy,
    gpu_matmul,
    gpu_matmul_bwd_a,
    gpu_matmul_bwd_b,
    bin_bwd_part_size,
)


comptime Ptr = Pointer[Scalar[DT], MutAnyOrigin]
comptime DBuf = DeviceBuffer[DT]


struct Node(Movable):
    var val: Tensor
    var grad: Tensor
    var m: Tensor
    var v: Tensor
    """A parameter's optimizer moments, allocated by the optimizer on its
    first step (nn's `Adam` via `ParamVisitorRef`); empty otherwise."""
    var rows: Int
    var cols: Int
    var op: Int
    var sub: Int
    var a: Int
    var b: Int
    var c: Scalar[DT]
    var rg: Bool
    """Requires grad: a parameter, an input asked for it, or anything
    computed from one. `backward` skips every node without it."""
    var gfresh: Bool
    """GPU backward: no gradient has reached this node yet, so its grad
    buffer holds garbage — the first contribution WRITES it, later ones add.
    Never set on a parameter (their gradients accumulate across steps)."""

    def __init__(out self):
        self.val = Tensor()
        self.grad = Tensor()
        self.m = Tensor()
        self.v = Tensor()
        self.rows = 0
        self.cols = 0
        self.op = OP_LEAF
        self.sub = 0
        self.a = -1
        self.b = -1
        self.c = 0
        self.rg = False
        self.gfresh = False

    def size(self) -> Int:
        return self.rows * self.cols


struct TapeData(Movable):
    var nodes: List[Node]
    var n_live: Int
    var n_params: Int
    var ctx: Optional[DeviceContext]
    var s1: Tensor
    var s2: Tensor
    var s3: Tensor
    var s4: Tensor
    """Backward scratch (a transposed operand, a product before it
    is accumulated). Grown on demand, reused by every matmul."""

    var mods: List[ModEntry]
    """Comptime nn modules the tape owns (`module_op.mojo`)."""

    def __init__(out self, ctx: Optional[DeviceContext] = None):
        self.nodes = List[Node]()
        self.n_live = 0
        self.n_params = 0
        self.ctx = ctx
        self.s1 = Tensor()
        self.s2 = Tensor()
        self.s3 = Tensor()
        self.s4 = Tensor()
        self.mods = List[ModEntry]()

    def __deinit__(deinit self):
        for i in range(len(self.mods)):
            self.mods[i].drop(self.mods[i].state)

    # ── storage ──────────────────────────────────────────────────────────
    def on_gpu(self) -> Bool:
        return Bool(self.ctx)

    def dctx(self) -> DeviceContext:
        """The device context, by value: an argument next to `self.dv(…)`
        in one call, where a borrow of `self.ctx` would alias the `mut
        self` that `dv` takes."""
        return self.ctx.value()

    def val_ptr(mut self, i: Int) -> Ptr:
        return rebind[Ptr](self.nodes[i].val.data.unsafe_ptr())

    def grad_ptr(mut self, i: Int) -> Ptr:
        return rebind[Ptr](self.nodes[i].grad.data.unsafe_ptr())

    def dv(mut self, i: Int) raises -> DBuf:
        """Node `i`'s device value, exactly its size (the buffer itself may
        be larger: slots only grow)."""
        return self.nodes[i].val.dev.value().create_sub_buffer[DT](
            0, self.nodes[i].size()
        )

    def dg(mut self, i: Int) raises -> DBuf:
        return self.nodes[i].grad.dev.value().create_sub_buffer[DT](
            0, self.nodes[i].size()
        )

    def _scratch1(mut self, n: Int) raises -> DBuf:
        var c = self.dctx()
        self.s1.ensure_gpu(c, n)
        return self.s1.dev.value().create_sub_buffer[DT](0, n)

    def _scratch2(mut self, n: Int) raises -> DBuf:
        var c = self.dctx()
        self.s2.ensure_gpu(c, n)
        return self.s2.dev.value().create_sub_buffer[DT](0, n)

    def _scratch3(mut self, n: Int) raises -> DBuf:
        var c = self.dctx()
        self.s3.ensure_gpu(c, n)
        return self.s3.dev.value().create_sub_buffer[DT](0, n)

    def _take(mut self, i: Int) -> Bool:
        """Whether a contribution to node `i`'s gradient must ACCUMULATE
        (False: it is the first, so it writes); marks `i` as written."""
        var acc = not self.nodes[i].gfresh
        self.nodes[i].gfresh = False
        return acc

    def _host_fit(mut self, i: Int, grad: Bool):
        var n = self.nodes[i].size()
        if grad:
            while len(self.nodes[i].grad.data) < n:
                self.nodes[i].grad.data.append(0)
        else:
            while len(self.nodes[i].val.data) < n:
                self.nodes[i].val.data.append(0)

    def upload_val(mut self, i: Int) raises:
        """Host `val.data[:size]` -> device. No-op on a CPU tape."""
        if not self.ctx:
            return
        var c = self.dctx()
        c.enqueue_copy(self.dv(i), self.nodes[i].val.data.unsafe_ptr())

    def download_val(mut self, i: Int) raises:
        """Device -> host `val.data[:size]` (synchronises). No-op on CPU."""
        if not self.ctx:
            return
        self._host_fit(i, False)
        var c = self.dctx()
        var src = self.dv(i)
        c.enqueue_copy(self.nodes[i].val.data.unsafe_ptr(), src)
        c.synchronize()

    def download_grad(mut self, i: Int) raises:
        if not self.ctx:
            return
        self._host_fit(i, True)
        var c = self.dctx()
        var src = self.dg(i)
        c.enqueue_copy(self.nodes[i].grad.data.unsafe_ptr(), src)
        c.synchronize()

    def new_node(
        mut self, rows: Int, cols: Int, op: Int, a: Int, b: Int, rg: Bool
    ) raises -> Int:
        """The next node: a reused slot left by `reset` when there is one
        (its buffers grow only if this step's shape is larger), else a new
        one. The value buffer is NOT cleared — every op writes all of it."""
        var i = self.n_live
        if i == len(self.nodes):
            self.nodes.append(Node())
        var gpu = self.on_gpu()
        var c: Optional[DeviceContext] = None
        if gpu:
            c = self.dctx()
        ref n = self.nodes[i]
        n.rows = rows
        n.cols = cols
        n.op = op
        n.sub = 0
        n.a = a
        n.b = b
        n.c = 0
        n.rg = rg
        n.gfresh = False
        if gpu:
            n.val.ensure_gpu(c.value(), rows * cols)
            if rg:
                n.grad.ensure_gpu(c.value(), rows * cols)
        else:
            n.val.ensure(rows * cols)
        self.n_live = i + 1
        return i

    def leaf(mut self, rows: Int, cols: Int, rg: Bool) raises -> Int:
        var i = self.new_node(rows, cols, OP_LEAF, -1, -1, rg)
        if rg and not self.ctx:
            self.nodes[i].grad.ensure(rows * cols)
        return i

    def param(mut self, rows: Int, cols: Int) raises -> Int:
        """A parameter: a persistent leaf that requires grad. Every
        parameter must exist before the first activation, so they stay the
        contiguous prefix `reset` keeps. Its host `val.data` is sized for
        the caller to fill, then `upload_val`."""
        if self.n_live != self.n_params:
            raise Error(
                "Tape.param: create every parameter before the first"
                " activation (or call reset() first)"
            )
        var i = self.leaf(rows, cols, True)
        self._host_fit(i, False)
        if self.ctx:
            fill_dev(self.dg(i), rows * cols, Scalar[DT](0), self.dctx())
        else:
            for k in range(rows * cols):
                self.nodes[i].grad.data[k] = 0
        self.n_params = self.n_live
        return i

    def reset(mut self):
        """Drop every activation; parameters, and every buffer, stay."""
        self.n_live = self.n_params
        for i in range(len(self.mods)):
            self.mods[i].used = False

    def zero_grad(mut self) raises:
        var c = self.ctx
        for j in range(len(self.mods)):
            self.mods[j].zero(self.mods[j].state, c)
        for i in range(self.n_params):
            if self.nodes[i].rg:
                if self.ctx:
                    fill_dev(
                        self.dg(i), self.nodes[i].size(), Scalar[DT](0),
                        self.dctx(),
                    )
                else:
                    var g = self.grad_ptr(i)
                    for k in range(self.nodes[i].size()):
                        g[unsafe_offset=k] = 0

    def sgd_step(mut self, lr: Scalar[DT]) raises:
        """`p -= lr · grad` on every parameter."""
        for i in range(self.n_params):
            if not self.nodes[i].rg:
                continue
            var n = self.nodes[i].size()
            if self.ctx:
                gpu_axpy(self.dctx(), self.dv(i), self.dg(i), n, -lr)
            else:
                var v = self.val_ptr(i)
                var g = self.grad_ptr(i)
                for k in range(n):
                    v[unsafe_offset=k] -= lr * g[unsafe_offset=k]

    # ── comptime modules ─────────────────────────────────────────────────
    def _vref(mut self, i: Int) -> ref[MutAnyOrigin] Tensor:
        """Node `i`'s value storage, origin erased — a module call needs two
        node storages of the one list at once (`TensorPack.__getitem__`'s
        idiom: the wildcard pins `self`)."""
        return self.nodes[i].val

    def _gref(mut self, i: Int) -> ref[MutAnyOrigin] Tensor:
        return self.nodes[i].grad

    def _s4ref(mut self) -> ref[MutAnyOrigin] Tensor:
        return self.s4

    def add_module(mut self, var e: ModEntry) -> Int:
        self.mods.append(e^)
        return len(self.mods) - 1

    def module_call(mut self, mid: Int, a: Int) raises -> Int:
        var b = self.mods[mid].batch
        var ind = self.mods[mid].in_dim
        if self.nodes[a].rows != b or self.nodes[a].cols != ind:
            raise Error(
                String("module ", mid, ": built for [", b, ", ", ind,
                       "], called on [", self.nodes[a].rows, ", ",
                       self.nodes[a].cols, "]")
            )
        if self.mods[mid].used:
            raise Error(
                String("module ", mid, ": called twice since reset() — its"
                       " caches hold one call; share weights with native ops")
            )
        self.mods[mid].used = True
        var o = self.new_node(b, self.mods[mid].out_dim, OP_MODULE, a, -1, True)
        self.nodes[o].sub = mid
        var f = self.mods[mid].fwd
        var st = self.mods[mid].state
        var c = self.ctx
        f(st, self._vref(a), self._vref(o), c)
        return o

    def _module_vjp(mut self, o: Int, a: Int) raises:
        """The module's `vjp` WRITES its input gradient. Straight into the
        input's gradient when it is the first to arrive (GPU, still fresh);
        otherwise into scratch, then added."""
        var mid = self.nodes[o].sub
        var f = self.mods[mid].vjp
        var st = self.mods[mid].state
        var c = self.ctx
        var direct = Bool(c) and self.nodes[a].rg and self.nodes[a].gfresh
        if direct:
            self.nodes[a].gfresh = False
            f(st, self._vref(a), self._gref(o), self._gref(a), c)
            return
        f(st, self._vref(a), self._gref(o), self._s4ref(), c)
        if not self.nodes[a].rg:
            return
        var n = self.nodes[a].size()
        if self.ctx:
            var src = self.s4.dev.value().create_sub_buffer[DT](0, n)
            gpu_axpy(self.dctx(), self.dg(a), src, n, 1.0)
            self.nodes[a].gfresh = False
        else:
            var ga = self.grad_ptr(a)
            var gs = self.s4.data.unsafe_ptr()
            for k in range(n):
                ga[unsafe_offset=k] += gs[unsafe_offset=k]

    def walk_params(
        mut self, mut r: ParamVisitorRef, ctx: Optional[DeviceContext],
        prefix: String,
    ) raises:
        """Every trainable parameter, through the erased visitor nn's
        optimizers and checkpoints drive: native ones as `p<i>` (weight
        decay on matrices, not on `[1, n]` rows), then each module's own
        walk under `m<j>`."""
        for i in range(self.n_params):
            if not self.nodes[i].rg:
                continue
            ref nd = self.nodes[i]
            r.call(
                r.state, join_name(prefix, String("p", i)), nd.val, nd.grad,
                nd.m, nd.v, nd.size(), nd.rows > 1, ctx,
            )
        for j in range(len(self.mods)):
            var w = self.mods[j].walk
            w(self.mods[j].state, r, ctx, join_name(prefix, String("m", j)))

    def walk_states(
        mut self, mut r: ParamVisitorRef, ctx: Optional[DeviceContext],
        prefix: String,
    ) raises:
        for j in range(len(self.mods)):
            var w = self.mods[j].walk_state
            w(self.mods[j].state, r, ctx, join_name(prefix, String("m", j)))

    # ── forward ops ──────────────────────────────────────────────────────
    def binary[OP: Int](mut self, a: Int, b: Int) raises -> Int:
        var ra = self.nodes[a].rows
        var ca = self.nodes[a].cols
        var rb = self.nodes[b].rows
        var cb = self.nodes[b].cols
        if not (
            (ra == rb or ra == 1 or rb == 1) and (ca == cb or ca == 1 or cb == 1)
        ):
            raise Error(
                String("shapes [", ra, ", ", ca, "] and [", rb, ", ", cb,
                       "] do not broadcast")
            )
        var r = fmax(ra, rb)
        var c = fmax(ca, cb)
        var o = self.new_node(
            r, c, OP, a, b, self.nodes[a].rg or self.nodes[b].rg
        )
        if self.ctx:
            gpu_binary[OP](
                self.dctx(), self.dv(o), self.dv(a), self.dv(b),
                r, c, ra, ca, rb, cb,
            )
            return o
        var pa = self.val_ptr(a)
        var pb = self.val_ptr(b)
        var po = self.val_ptr(o)
        if ra == rb and ca == cb:
            for k in range(r * c):
                po[unsafe_offset=k] = bin_op[OP](
                    pa[unsafe_offset=k], pb[unsafe_offset=k]
                )
        else:
            for i in range(r):
                var ia = (i if ra > 1 else 0) * ca
                var ib = (i if rb > 1 else 0) * cb
                for j in range(c):
                    po[unsafe_offset=i * c + j] = bin_op[OP](
                        pa[unsafe_offset=ia + (j if ca > 1 else 0)],
                        pb[unsafe_offset=ib + (j if cb > 1 else 0)],
                    )
        return o

    def matmul(mut self, a: Int, b: Int) raises -> Int:
        var m = self.nodes[a].rows
        var k = self.nodes[a].cols
        var n = self.nodes[b].cols
        if self.nodes[b].rows != k:
            raise Error(
                String("matmul: [", m, ", ", k, "] @ [", self.nodes[b].rows,
                       ", ", n, "]")
            )
        var o = self.new_node(
            m, n, OP_MATMUL, a, b, self.nodes[a].rg or self.nodes[b].rg
        )
        if self.ctx:
            gpu_matmul(
                self.dctx(), self.dv(o), self.dv(a), self.dv(b), m, n, k
            )
            return o
        gemm_cpu_ptr[False, False](
            m, n, k, 1.0,
            rebind[Pointer[Float32, ImmutAnyOrigin]](self.val_ptr(a)), k,
            rebind[Pointer[Float32, ImmutAnyOrigin]](self.val_ptr(b)), n,
            0.0,
            rebind[Pointer[Float32, MutAnyOrigin]](self.val_ptr(o)), n,
        )
        return o

    def scalar_op[OP: Int](mut self, a: Int, c: Scalar[DT]) raises -> Int:
        var o = self.new_node(
            self.nodes[a].rows, self.nodes[a].cols, OP, a, -1, self.nodes[a].rg
        )
        self.nodes[o].c = c
        if self.ctx:
            gpu_scalar[OP](
                self.dctx(), self.dv(o), self.dv(a), self.nodes[o].size(), c
            )
            return o
        var pa = self.val_ptr(a)
        var po = self.val_ptr(o)
        for k in range(self.nodes[o].size()):
            var x = pa[unsafe_offset=k]
            comptime if OP == OP_SCALE:
                po[unsafe_offset=k] = x * c
            elif OP == OP_SHIFT:
                po[unsafe_offset=k] = x + c
            elif OP == OP_DIVC:
                po[unsafe_offset=k] = x / c
            else:
                po[unsafe_offset=k] = c - x
        return o

    def elementwise(mut self, a: Int, sub: Int) raises -> Int:
        var o = self.new_node(
            self.nodes[a].rows, self.nodes[a].cols, OP_EW, a, -1,
            self.nodes[a].rg,
        )
        self.nodes[o].sub = sub
        if sub == EW_RELU:
            self._ew_fwd_one[ReLUOp](a, o)
        elif sub == EW_TANH:
            self._ew_fwd_one[TanhOp](a, o)
        elif sub == EW_SIGMOID:
            self._ew_fwd_one[SigmoidOp](a, o)
        elif sub == EW_GELU:
            self._ew_fwd_one[GELUOp](a, o)
        elif sub == EW_GELU_TANH:
            self._ew_fwd_one[GELUTanhOp](a, o)
        elif sub == EW_MISH:
            self._ew_fwd_one[MishOp](a, o)
        elif sub == EW_SWISH:
            self._ew_fwd_one[SwishOp](a, o)
        elif sub == EW_ELU:
            self._ew_fwd_one[ELUOp](a, o)
        else:
            raise Error(String("unknown elementwise op ", sub))
        return o

    def _ew_fwd_one[OP: ElementOp](mut self, a: Int, o: Int) raises:
        var n = self.nodes[o].size()
        if self.ctx:
            gpu_ew[OP](self.dctx(), self.dv(o), self.dv(a), n)
        else:
            _ew_fwd[OP](self.val_ptr(a), self.val_ptr(o), n)

    def reduce[OP: Int](mut self, a: Int) raises -> Int:
        """`OP_SUM` / `OP_MEAN` over every element. CPU: in index order —
        the order `mse_forward` sums in. GPU: one block, fixed tree order."""
        var o = self.new_node(1, 1, OP, a, -1, self.nodes[a].rg)
        var n = self.nodes[a].size()
        if self.ctx:
            gpu_reduce[OP == OP_MEAN](
                self.dctx(), self.dv(o), self.dv(a), n
            )
            return o
        var pa = self.val_ptr(a)
        var s: Scalar[DT] = 0
        for k in range(n):
            s += pa[unsafe_offset=k]
        comptime if OP == OP_MEAN:
            s = s / Scalar[DT](n)
        self.val_ptr(o)[unsafe_offset=0] = s
        return o

    def sum_rows(mut self, a: Int) raises -> Int:
        var r = self.nodes[a].rows
        var c = self.nodes[a].cols
        var o = self.new_node(1, c, OP_SUM_ROWS, a, -1, self.nodes[a].rg)
        if self.ctx:
            gpu_sum_rows(self.dctx(), self.dv(o), self.dv(a), r, c)
            return o
        var pa = self.val_ptr(a)
        var po = self.val_ptr(o)
        for j in range(c):
            po[unsafe_offset=j] = 0
        for i in range(r):
            for j in range(c):
                po[unsafe_offset=j] += pa[unsafe_offset=i * c + j]
        return o

    def transpose(mut self, a: Int) raises -> Int:
        var r = self.nodes[a].rows
        var c = self.nodes[a].cols
        var o = self.new_node(c, r, OP_TRANSPOSE, a, -1, self.nodes[a].rg)
        if self.ctx:
            gpu_transpose(self.dctx(), self.dv(o), self.dv(a), r, c)
            return o
        var pa = self.val_ptr(a)
        var po = self.val_ptr(o)
        for i in range(r):
            for j in range(c):
                po[unsafe_offset=j * r + i] = pa[unsafe_offset=i * c + j]
        return o

    # ── backward ─────────────────────────────────────────────────────────
    def backward(mut self, loss: Int) raises:
        """d loss / d every node that requires grad. Parameter gradients
        ACCUMULATE (call `zero_grad` between steps); every other node's
        gradient is cleared first."""
        if self.nodes[loss].size() != 1:
            raise Error("backward: the loss must be a [1, 1] scalar")
        if not self.nodes[loss].rg:
            raise Error("backward: the loss does not depend on any parameter")
        for i in range(self.n_params, loss + 1):
            if not self.nodes[i].rg:
                continue
            var n = self.nodes[i].size()
            if self.ctx:
                self.nodes[i].gfresh = True
            else:
                self.nodes[i].grad.ensure(n)
                var g = self.grad_ptr(i)
                for k in range(n):
                    g[unsafe_offset=k] = 0
        if self.ctx:
            fill_dev(self.dg(loss), 1, Scalar[DT](1), self.dctx())
            self.nodes[loss].gfresh = False
        else:
            self.grad_ptr(loss)[unsafe_offset=0] = 1
        var i = loss
        while i >= self.n_params:
            ref nd = self.nodes[i]
            # A node no gradient reached (GPU: still fresh) contributes zero.
            if nd.rg and nd.op != OP_LEAF and not nd.gfresh:
                self._vjp(i)
            i -= 1
        if self.ctx:
            # Leaves that asked for a gradient but are not on the loss's
            # path: their gradient is zero, not garbage.
            for j in range(self.n_params, loss + 1):
                if self.nodes[j].rg and self.nodes[j].gfresh:
                    fill_dev(
                        self.dg(j), self.nodes[j].size(), Scalar[DT](0),
                        self.dctx(),
                    )
                    self.nodes[j].gfresh = False

    def _vjp(mut self, o: Int) raises:
        var op = self.nodes[o].op
        var a = self.nodes[o].a
        var b = self.nodes[o].b
        if op == OP_ADD:
            self._bin_vjp[OP_ADD](o, a, b)
        elif op == OP_SUB:
            self._bin_vjp[OP_SUB](o, a, b)
        elif op == OP_MUL:
            self._bin_vjp[OP_MUL](o, a, b)
        elif op == OP_DIV:
            self._bin_vjp[OP_DIV](o, a, b)
        elif op == OP_MATMUL:
            self._matmul_vjp(o, a, b)
        elif op == OP_SCALE:
            self._scalar_vjp[OP_SCALE](o, a)
        elif op == OP_SHIFT:
            self._scalar_vjp[OP_SHIFT](o, a)
        elif op == OP_DIVC:
            self._scalar_vjp[OP_DIVC](o, a)
        elif op == OP_RSUBC:
            self._scalar_vjp[OP_RSUBC](o, a)
        elif op == OP_EW:
            self._ew_vjp(o, a)
        elif op == OP_MODULE:
            self._module_vjp(o, a)
        elif op == OP_SUM or op == OP_MEAN:
            if not self.nodes[a].rg:
                return
            var n = self.nodes[a].size()
            if self.ctx:
                var div = n if op == OP_MEAN else 1
                if self._take(a):
                    gpu_reduce_bwd[True](self.dctx(), self.dg(a), self.dg(o), n, div)
                else:
                    gpu_reduce_bwd[False](self.dctx(), self.dg(a), self.dg(o), n, div)
                return
            var g = self.grad_ptr(o)[unsafe_offset=0]
            if op == OP_MEAN:
                g = g / Scalar[DT](n)
            var ga = self.grad_ptr(a)
            for k in range(n):
                ga[unsafe_offset=k] += g
        elif op == OP_SUM_ROWS:
            if not self.nodes[a].rg:
                return
            var r = self.nodes[a].rows
            var c = self.nodes[a].cols
            if self.ctx:
                if self._take(a):
                    gpu_sum_rows_bwd[True](self.dctx(), self.dg(a), self.dg(o), r, c)
                else:
                    gpu_sum_rows_bwd[False](self.dctx(), self.dg(a), self.dg(o), r, c)
                return
            var go = self.grad_ptr(o)
            var ga = self.grad_ptr(a)
            for i in range(r):
                for j in range(c):
                    ga[unsafe_offset=i * c + j] += go[unsafe_offset=j]
        elif op == OP_TRANSPOSE:
            if not self.nodes[a].rg:
                return
            var r = self.nodes[a].rows
            var c = self.nodes[a].cols
            if self.ctx:
                # go is [c, r]; its transpose lands in ga [r, c].
                if self._take(a):
                    gpu_transpose[True](self.dctx(), self.dg(a), self.dg(o), c, r)
                else:
                    gpu_transpose[False](self.dctx(), self.dg(a), self.dg(o), c, r)
                return
            var go = self.grad_ptr(o)
            var ga = self.grad_ptr(a)
            for i in range(r):
                for j in range(c):
                    ga[unsafe_offset=i * c + j] += go[unsafe_offset=j * r + i]
        else:
            raise Error(String("backward: unknown op ", op))

    def _bin_vjp[OP: Int](mut self, o: Int, a: Int, b: Int) raises:
        """Each input's gradient, summed over the axes it was broadcast
        along. `a` and `b` may be the same node (`x * x`): both terms then
        land in the one buffer, which is the product rule."""
        var r = self.nodes[o].rows
        var c = self.nodes[o].cols
        var ra = self.nodes[a].rows
        var ca = self.nodes[a].cols
        var rb = self.nodes[b].rows
        var cb = self.nodes[b].cols
        var want_a = self.nodes[a].rg
        var want_b = self.nodes[b].rg
        if self.ctx:
            if want_a:
                var part = self._scratch3(bin_bwd_part_size(r, ra * ca))
                if self._take(a):
                    gpu_binary_bwd[OP, 0, True](
                        self.dctx(), self.dg(a), self.dg(o), self.dv(a),
                        self.dv(b), r, c, ra, ca, rb, cb, part,
                    )
                else:
                    gpu_binary_bwd[OP, 0, False](
                        self.dctx(), self.dg(a), self.dg(o), self.dv(a),
                        self.dv(b), r, c, ra, ca, rb, cb, part,
                    )
            if want_b:
                var part = self._scratch3(bin_bwd_part_size(r, rb * cb))
                if self._take(b):
                    gpu_binary_bwd[OP, 1, True](
                        self.dctx(), self.dg(b), self.dg(o), self.dv(a),
                        self.dv(b), r, c, ra, ca, rb, cb, part,
                    )
                else:
                    gpu_binary_bwd[OP, 1, False](
                        self.dctx(), self.dg(b), self.dg(o), self.dv(a),
                        self.dv(b), r, c, ra, ca, rb, cb, part,
                    )
            return
        var go = self.grad_ptr(o)
        var pa = self.val_ptr(a)
        var pb = self.val_ptr(b)
        var ga = self.grad_ptr(a) if want_a else go
        var gb = self.grad_ptr(b) if want_b else go
        for i in range(r):
            var ia = (i if ra > 1 else 0) * ca
            var ib = (i if rb > 1 else 0) * cb
            for j in range(c):
                var ka = ia + (j if ca > 1 else 0)
                var kb = ib + (j if cb > 1 else 0)
                var g = go[unsafe_offset=i * c + j]
                comptime if OP == OP_ADD:
                    if want_a:
                        ga[unsafe_offset=ka] += g
                    if want_b:
                        gb[unsafe_offset=kb] += g
                elif OP == OP_SUB:
                    if want_a:
                        ga[unsafe_offset=ka] += g
                    if want_b:
                        gb[unsafe_offset=kb] -= g
                elif OP == OP_MUL:
                    var x = pa[unsafe_offset=ka]
                    var y = pb[unsafe_offset=kb]
                    if want_a:
                        ga[unsafe_offset=ka] += g * y
                    if want_b:
                        gb[unsafe_offset=kb] += g * x
                else:
                    var x = pa[unsafe_offset=ka]
                    var y = pb[unsafe_offset=kb]
                    if want_a:
                        ga[unsafe_offset=ka] += g / y
                    if want_b:
                        gb[unsafe_offset=kb] -= g * x / (y * y)

    def _matmul_vjp(mut self, o: Int, a: Int, b: Int) raises:
        """`C = A @ B`: dA += dC @ Bᵀ, dB += Aᵀ @ dC — on the CPU the same
        two cblas calls (and for dB the same accumulate-into-grad form) as
        `Linear.vjp`'s CPU path."""
        var m = self.nodes[a].rows
        var k = self.nodes[a].cols
        var n = self.nodes[b].cols
        if self.ctx:
            if self.nodes[a].rg:
                if self._take(a):
                    gpu_matmul_bwd_a[True](
                        self.dctx(), self.dg(a), self.dg(o), self.dv(b),
                        m, n, k, self._scratch2(m * k),
                    )
                else:
                    gpu_matmul_bwd_a[False](
                        self.dctx(), self.dg(a), self.dg(o), self.dv(b),
                        m, n, k, self._scratch2(m * k),
                    )
            if self.nodes[b].rg:
                if self._take(b):
                    gpu_matmul_bwd_b[True](
                        self.dctx(), self.dg(b), self.dv(a), self.dg(o),
                        m, n, k, self._scratch1(m * k), self._scratch2(k * n),
                    )
                else:
                    gpu_matmul_bwd_b[False](
                        self.dctx(), self.dg(b), self.dv(a), self.dg(o),
                        m, n, k, self._scratch1(m * k), self._scratch2(k * n),
                    )
            return
        var go = rebind[Pointer[Float32, ImmutAnyOrigin]](self.grad_ptr(o))
        if self.nodes[a].rg:
            gemm_cpu_ptr[False, True](
                m, k, n, 1.0,
                go, n,
                rebind[Pointer[Float32, ImmutAnyOrigin]](self.val_ptr(b)), n,
                1.0,
                rebind[Pointer[Float32, MutAnyOrigin]](self.grad_ptr(a)), k,
            )
        if self.nodes[b].rg:
            gemm_cpu_ptr[True, False](
                k, n, m, 1.0,
                rebind[Pointer[Float32, ImmutAnyOrigin]](self.val_ptr(a)), k,
                go, n,
                1.0,
                rebind[Pointer[Float32, MutAnyOrigin]](self.grad_ptr(b)), n,
            )

    def _scalar_vjp[OP: Int](mut self, o: Int, a: Int) raises:
        if not self.nodes[a].rg:
            return
        var c = self.nodes[o].c
        var n = self.nodes[o].size()
        if self.ctx:
            if self._take(a):
                gpu_scalar_bwd[OP, True](self.dctx(), self.dg(a), self.dg(o), n, c)
            else:
                gpu_scalar_bwd[OP, False](self.dctx(), self.dg(a), self.dg(o), n, c)
            return
        var go = self.grad_ptr(o)
        var ga = self.grad_ptr(a)
        for k in range(n):
            var g = go[unsafe_offset=k]
            comptime if OP == OP_SCALE:
                ga[unsafe_offset=k] += g * c
            elif OP == OP_SHIFT:
                ga[unsafe_offset=k] += g
            elif OP == OP_DIVC:
                ga[unsafe_offset=k] += g / c
            else:
                ga[unsafe_offset=k] -= g

    def _ew_vjp(mut self, o: Int, a: Int) raises:
        if not self.nodes[a].rg:
            return
        var sub = self.nodes[o].sub
        if sub == EW_RELU:
            self._ew_bwd_one[ReLUOp](o, a)
        elif sub == EW_TANH:
            self._ew_bwd_one[TanhOp](o, a)
        elif sub == EW_SIGMOID:
            self._ew_bwd_one[SigmoidOp](o, a)
        elif sub == EW_GELU:
            self._ew_bwd_one[GELUOp](o, a)
        elif sub == EW_GELU_TANH:
            self._ew_bwd_one[GELUTanhOp](o, a)
        elif sub == EW_MISH:
            self._ew_bwd_one[MishOp](o, a)
        elif sub == EW_SWISH:
            self._ew_bwd_one[SwishOp](o, a)
        elif sub == EW_ELU:
            self._ew_bwd_one[ELUOp](o, a)
        else:
            raise Error(String("unknown elementwise op ", sub))

    def _ew_bwd_one[OP: ElementOp](mut self, o: Int, a: Int) raises:
        var n = self.nodes[o].size()
        if self.ctx:
            if self._take(a):
                gpu_ew_bwd[OP, True](
                    self.dctx(), self.dg(a), self.dg(o), self.dv(a),
                    self.dv(o), n,
                )
            else:
                gpu_ew_bwd[OP, False](
                    self.dctx(), self.dg(a), self.dg(o), self.dv(a),
                    self.dv(o), n,
                )
        else:
            _ew_bwd[OP](
                self.val_ptr(a), self.val_ptr(o), self.grad_ptr(o),
                self.grad_ptr(a), n,
            )


# ── elementwise loops: `Elementwise`'s CPU loop shape over the shared math ─
def _ew_fwd[OP: ElementOp](x: Ptr, y: Ptr, n: Int):
    var k = 0
    while k + CPU_SIMD_W <= n:
        y.unsafe_store(k, OP.forward_simd[CPU_SIMD_W](
            x.unsafe_load[width=CPU_SIMD_W](k)
        ))
        k += CPU_SIMD_W
    while k < n:
        y[unsafe_offset=k] = OP.forward_scalar(x[unsafe_offset=k])
        k += 1


def _ew_bwd[OP: ElementOp](x: Ptr, y: Ptr, go: Ptr, ga: Ptr, n: Int):
    """`ga += f'(c) ⊙ go`, with `c` = y for an output-cache op and x for an
    input-cache one (`ElementOp.owns_cache`)."""
    var k = 0
    while k + CPU_SIMD_W <= n:
        var g = go.unsafe_load[width=CPU_SIMD_W](k)
        var cv: SIMD[DT, CPU_SIMD_W]
        comptime if OP.owns_cache:
            cv = y.unsafe_load[width=CPU_SIMD_W](k)
        else:
            cv = x.unsafe_load[width=CPU_SIMD_W](k)
        ga.unsafe_store(
            k,
            ga.unsafe_load[width=CPU_SIMD_W](k)
            + OP.backward_simd[CPU_SIMD_W](cv, g),
        )
        k += CPU_SIMD_W
    while k < n:
        var cs: Scalar[DT]
        comptime if OP.owns_cache:
            cs = y[unsafe_offset=k]
        else:
            cs = x[unsafe_offset=k]
        ga[unsafe_offset=k] += OP.backward_scalar(cs, go[unsafe_offset=k])
        k += 1
