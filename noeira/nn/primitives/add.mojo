"""Add — binary (ARITY=2) elementwise Module conformer (CPU + GPU). z = a + b.

Each forward/vjp branches `comptime if target == "cpu"` (tracked `TileTensor`
over `.data`) `else` (device `LayoutTensor` via `lt_gpu` + a naive kernel).
The storage surface (`ref/mut Tensor`, `TensorRefs`) is identical on both
targets; the only GPU erasure is the kernel-arg `MutAnyOrigin`. Params are
`Param` (two `Tensor`s, cpu+dev).

LIFETIME NOTE: a pack subscript (`inputs[k]`) returns a TEMPORARY ref. Building
a view from `inputs[k].data` directly and using it LATER dangles (the temporary
dies at the end of the statement; a later op clobbers the stack). So each body
first binds the element to a named `ref` (`ref in0 = inputs[0]`) that lives for
the whole function, then builds views from that.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


def _add_fwd_kernel[
    M: Int, ADT: DType = DT
](
    a: Tile[ADT, M],
    b: Tile[ADT, M],
    o: Tile[ADT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        o[i] = a[i] + b[i]


def _copy_kernel[
    M: Int, ADT: DType = DT
](
    src: Tile[ADT, M],
    dst: Tile[ADT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        dst[i] = src[i]


# ── Add (binary) ───────────────────────────────────────────────────────
struct Add[DIM_: Int, ADT: DType = DT](Module):
    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype (AMP). Add is dtype-TRANSPARENT (elementwise sum,
    # grad = copy to both inputs) → carries ACT_DT through unchanged. ACT_DT ==
    # DT (default) → byte-identical to the fp32 path.
    comptime ACT_DT = Self.ADT

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref a0 = inputs[0]
        ref a1 = inputs[1]
        comptime if target == "cpu":
            out.ensure(M)
            var a = TileTensor(a0.data, row_major[M]())
            var b = TileTensor(a1.data, row_major[M]())
            var o = TileTensor(out.data, row_major[M]())
            for i in range(M):
                o[i] = a[i] + b[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, M)
            var al = a0.tt["gpu", M]()
            var bl = a1.tt["gpu", M]()
            var ol = out.tt["gpu", M]()
            c.enqueue_function[_add_fwd_kernel[M, Self.ACT_DT]](
                al, bl, ol, grid_dim=(M + 255) // 256, block_dim=256
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[2, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref g0 = grad_inputs[0]
        ref g1 = grad_inputs[1]
        comptime if target == "cpu":
            g0.ensure(M)
            g1.ensure(M)
            var go = TileTensor(grad_output.data, row_major[M]())
            var gi0 = TileTensor(g0.data, row_major[M]())
            var gi1 = TileTensor(g1.data, row_major[M]())
            for i in range(M):
                gi0[i] = go[i]
                gi1[i] = go[i]
        else:
            var c = ctx.value()
            g0.ensure_gpu(c, M)
            g1.ensure_gpu(c, M)
            var gol = grad_output.tt["gpu", M]()
            var gi0l = g0.tt["gpu", M]()
            var gi1l = g1.tt["gpu", M]()
            comptime nblk = (M + 255) // 256
            c.enqueue_function[_copy_kernel[M, Self.ACT_DT]](
                gol, gi0l, grid_dim=nblk, block_dim=256
            )
            c.enqueue_function[_copy_kernel[M, Self.ACT_DT]](
                gol, gi1l, grid_dim=nblk, block_dim=256
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
