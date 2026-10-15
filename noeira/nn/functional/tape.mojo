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

CPU only. Values are fp32 (`DT`).
"""

from std.math import max as fmax

from noeira.nn.constants import DT, CPU_SIMD_W
from noeira.nn.core.tensor import Tensor
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


comptime Ptr = Pointer[Scalar[DT], MutAnyOrigin]

# ── op tags ──────────────────────────────────────────────────────────────
comptime OP_LEAF = 0
comptime OP_ADD = 1
comptime OP_SUB = 2
comptime OP_MUL = 3
comptime OP_DIV = 4
comptime OP_MATMUL = 5
comptime OP_SCALE = 6  # x * c
comptime OP_SHIFT = 7  # x + c
comptime OP_DIVC = 8  # x / c
comptime OP_RSUBC = 9  # c - x
comptime OP_EW = 10  # elementwise activation, `sub` = EW_* tag
comptime OP_SUM = 11  # all elements -> [1, 1]
comptime OP_MEAN = 12  # all elements -> [1, 1]
comptime OP_TRANSPOSE = 13
comptime OP_SUM_ROWS = 14  # [R, C] -> [1, C]

# ── elementwise activations: the comptime nn's `ElementOp`s ─────────────
comptime EW_RELU = 0
comptime EW_TANH = 1
comptime EW_SIGMOID = 2
comptime EW_GELU = 3
comptime EW_GELU_TANH = 4
comptime EW_MISH = 5
comptime EW_SWISH = 6
comptime EW_ELU = 7


struct Node(Movable):
    var val: Tensor
    var grad: Tensor
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

    def __init__(out self):
        self.val = Tensor()
        self.grad = Tensor()
        self.rows = 0
        self.cols = 0
        self.op = OP_LEAF
        self.sub = 0
        self.a = -1
        self.b = -1
        self.c = 0
        self.rg = False

    def size(self) -> Int:
        return self.rows * self.cols


@always_inline
def _bin[OP: Int](x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
    comptime if OP == OP_ADD:
        return x + y
    elif OP == OP_SUB:
        return x - y
    elif OP == OP_MUL:
        return x * y
    else:
        return x / y


struct TapeData(Movable):
    var nodes: List[Node]
    var n_live: Int
    var n_params: Int

    def __init__(out self):
        self.nodes = List[Node]()
        self.n_live = 0
        self.n_params = 0

    # ── storage ──────────────────────────────────────────────────────────
    def val_ptr(mut self, i: Int) -> Ptr:
        return rebind[Ptr](self.nodes[i].val.data.unsafe_ptr())

    def grad_ptr(mut self, i: Int) -> Ptr:
        return rebind[Ptr](self.nodes[i].grad.data.unsafe_ptr())

    def new_node(
        mut self, rows: Int, cols: Int, op: Int, a: Int, b: Int, rg: Bool
    ) -> Int:
        """The next node: a reused slot left by `reset` when there is one
        (its buffers grow only if this step's shape is larger), else a new
        one. The value buffer is NOT cleared — every op writes all of it."""
        var i = self.n_live
        if i == len(self.nodes):
            self.nodes.append(Node())
        ref n = self.nodes[i]
        n.rows = rows
        n.cols = cols
        n.op = op
        n.sub = 0
        n.a = a
        n.b = b
        n.c = 0
        n.rg = rg
        n.val.ensure(rows * cols)
        self.n_live = i + 1
        return i

    def leaf(
        mut self, rows: Int, cols: Int, rg: Bool
    ) -> Int:
        var i = self.new_node(rows, cols, OP_LEAF, -1, -1, rg)
        if rg:
            self.nodes[i].grad.ensure(rows * cols)
        return i

    def param(mut self, rows: Int, cols: Int) raises -> Int:
        """A parameter: a persistent leaf that requires grad. Every
        parameter must exist before the first activation, so they stay the
        contiguous prefix `reset` keeps."""
        if self.n_live != self.n_params:
            raise Error(
                "Tape.param: create every parameter before the first"
                " activation (or call reset() first)"
            )
        var i = self.leaf(rows, cols, True)
        for k in range(rows * cols):
            self.nodes[i].grad.data[k] = 0
        self.n_params = self.n_live
        return i

    def reset(mut self):
        """Drop every activation; parameters, and every buffer, stay."""
        self.n_live = self.n_params

    def zero_grad(mut self):
        for i in range(self.n_params):
            if self.nodes[i].rg:
                var g = self.grad_ptr(i)
                for k in range(self.nodes[i].size()):
                    g[unsafe_offset=k] = 0

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
        var pa = self.val_ptr(a)
        var pb = self.val_ptr(b)
        var po = self.val_ptr(o)
        if ra == rb and ca == cb:
            for k in range(r * c):
                po[unsafe_offset=k] = _bin[OP](
                    pa[unsafe_offset=k], pb[unsafe_offset=k]
                )
        else:
            for i in range(r):
                var ia = (i if ra > 1 else 0) * ca
                var ib = (i if rb > 1 else 0) * cb
                for j in range(c):
                    po[unsafe_offset=i * c + j] = _bin[OP](
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
        gemm_cpu_ptr[False, False](
            m, n, k, 1.0,
            rebind[Pointer[Float32, ImmutAnyOrigin]](self.val_ptr(a)), k,
            rebind[Pointer[Float32, ImmutAnyOrigin]](self.val_ptr(b)), n,
            0.0,
            rebind[Pointer[Float32, MutAnyOrigin]](self.val_ptr(o)), n,
        )
        return o

    def scalar_op[OP: Int](mut self, a: Int, c: Scalar[DT]) -> Int:
        var o = self.new_node(
            self.nodes[a].rows, self.nodes[a].cols, OP, a, -1, self.nodes[a].rg
        )
        self.nodes[o].c = c
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
        var n = self.nodes[o].size()
        var x = self.val_ptr(a)
        var y = self.val_ptr(o)
        if sub == EW_RELU:
            _ew_fwd[ReLUOp](x, y, n)
        elif sub == EW_TANH:
            _ew_fwd[TanhOp](x, y, n)
        elif sub == EW_SIGMOID:
            _ew_fwd[SigmoidOp](x, y, n)
        elif sub == EW_GELU:
            _ew_fwd[GELUOp](x, y, n)
        elif sub == EW_GELU_TANH:
            _ew_fwd[GELUTanhOp](x, y, n)
        elif sub == EW_MISH:
            _ew_fwd[MishOp](x, y, n)
        elif sub == EW_SWISH:
            _ew_fwd[SwishOp](x, y, n)
        elif sub == EW_ELU:
            _ew_fwd[ELUOp](x, y, n)
        else:
            raise Error(String("unknown elementwise op ", sub))
        return o

    def reduce[OP: Int](mut self, a: Int) -> Int:
        """`OP_SUM` / `OP_MEAN` over every element, in index order — the
        order `mse_forward` sums in."""
        var o = self.new_node(1, 1, OP, a, -1, self.nodes[a].rg)
        var n = self.nodes[a].size()
        var pa = self.val_ptr(a)
        var s: Scalar[DT] = 0
        for k in range(n):
            s += pa[unsafe_offset=k]
        comptime if OP == OP_MEAN:
            s = s / Scalar[DT](n)
        self.val_ptr(o)[unsafe_offset=0] = s
        return o

    def sum_rows(mut self, a: Int) -> Int:
        var r = self.nodes[a].rows
        var c = self.nodes[a].cols
        var o = self.new_node(1, c, OP_SUM_ROWS, a, -1, self.nodes[a].rg)
        var pa = self.val_ptr(a)
        var po = self.val_ptr(o)
        for j in range(c):
            po[unsafe_offset=j] = 0
        for i in range(r):
            for j in range(c):
                po[unsafe_offset=j] += pa[unsafe_offset=i * c + j]
        return o

    def transpose(mut self, a: Int) -> Int:
        var r = self.nodes[a].rows
        var c = self.nodes[a].cols
        var o = self.new_node(c, r, OP_TRANSPOSE, a, -1, self.nodes[a].rg)
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
            if self.nodes[i].rg:
                var n = self.nodes[i].size()
                self.nodes[i].grad.ensure(n)
                var g = self.grad_ptr(i)
                for k in range(n):
                    g[unsafe_offset=k] = 0
        self.grad_ptr(loss)[unsafe_offset=0] = 1
        var i = loss
        while i >= self.n_params:
            if self.nodes[i].rg and self.nodes[i].op != OP_LEAF:
                self._vjp(i)
            i -= 1

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
        elif op == OP_SCALE or op == OP_SHIFT or op == OP_DIVC or op == OP_RSUBC:
            self._scalar_vjp(o, a)
        elif op == OP_EW:
            self._ew_vjp(o, a)
        elif op == OP_SUM or op == OP_MEAN:
            if not self.nodes[a].rg:
                return
            var g = self.grad_ptr(o)[unsafe_offset=0]
            if op == OP_MEAN:
                g = g / Scalar[DT](self.nodes[a].size())
            var ga = self.grad_ptr(a)
            for k in range(self.nodes[a].size()):
                ga[unsafe_offset=k] += g
        elif op == OP_SUM_ROWS:
            if not self.nodes[a].rg:
                return
            var r = self.nodes[a].rows
            var c = self.nodes[a].cols
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
            var go = self.grad_ptr(o)
            var ga = self.grad_ptr(a)
            for i in range(r):
                for j in range(c):
                    ga[unsafe_offset=i * c + j] += go[unsafe_offset=j * r + i]
        else:
            raise Error(String("backward: unknown op ", op))

    def _bin_vjp[OP: Int](mut self, o: Int, a: Int, b: Int):
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
        """`C = A @ B`: dA += dC @ Bᵀ, dB += Aᵀ @ dC — the same two cblas
        calls (and for dB the same accumulate-into-grad form) as
        `Linear.vjp`'s CPU path."""
        var m = self.nodes[a].rows
        var k = self.nodes[a].cols
        var n = self.nodes[b].cols
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

    def _scalar_vjp(mut self, o: Int, a: Int):
        if not self.nodes[a].rg:
            return
        var op = self.nodes[o].op
        var c = self.nodes[o].c
        var go = self.grad_ptr(o)
        var ga = self.grad_ptr(a)
        for k in range(self.nodes[o].size()):
            var g = go[unsafe_offset=k]
            if op == OP_SCALE:
                ga[unsafe_offset=k] += g * c
            elif op == OP_SHIFT:
                ga[unsafe_offset=k] += g
            elif op == OP_DIVC:
                ga[unsafe_offset=k] += g / c
            else:
                ga[unsafe_offset=k] -= g

    def _ew_vjp(mut self, o: Int, a: Int) raises:
        if not self.nodes[a].rg:
            return
        var n = self.nodes[o].size()
        var x = self.val_ptr(a)
        var y = self.val_ptr(o)
        var go = self.grad_ptr(o)
        var ga = self.grad_ptr(a)
        var sub = self.nodes[o].sub
        if sub == EW_RELU:
            _ew_bwd[ReLUOp](x, y, go, ga, n)
        elif sub == EW_TANH:
            _ew_bwd[TanhOp](x, y, go, ga, n)
        elif sub == EW_SIGMOID:
            _ew_bwd[SigmoidOp](x, y, go, ga, n)
        elif sub == EW_GELU:
            _ew_bwd[GELUOp](x, y, go, ga, n)
        elif sub == EW_GELU_TANH:
            _ew_bwd[GELUTanhOp](x, y, go, ga, n)
        elif sub == EW_MISH:
            _ew_bwd[MishOp](x, y, go, ga, n)
        elif sub == EW_SWISH:
            _ew_bwd[SwishOp](x, y, go, ga, n)
        elif sub == EW_ELU:
            _ew_bwd[ELUOp](x, y, go, ga, n)
        else:
            raise Error(String("unknown elementwise op ", sub))


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
