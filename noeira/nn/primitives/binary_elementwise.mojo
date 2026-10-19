"""BinaryElementwise[DIM, OP] — generic 2-input elementwise op (storage surface).

The binary twin of `Elementwise`. Reuses the legacy `BinaryElementOp` trait +
`ops/` structs VERBATIM (DT-only), so the per-lane math is bit-identical. The
storage `vjp` gets BOTH inputs as `forward_input`, so the per-element carry
(`cache`) is RECOMPUTED in backward (`OP.cache_scalar(x,y)`) — no cache field.

    comptime BinaryElemMin[DIM] = BinaryElementwise[DIM, BinaryElemMinOp]   # SAC twin-min
    comptime BinarySub[DIM]     = BinaryElementwise[DIM, BinarySubOp]

Backward: c = OP.cache(x,y) ; gi0 = OP.backward_x(c, go) ; gi1 = OP.backward_y(c, go).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, CPU_SIMD_W, TPB
from noeira.nn.core.binary_element_op import BinaryElementOp
from noeira.nn.primitives.ops.binary_elem_min_op import BinaryElemMinOp
from noeira.nn.primitives.ops.binary_sub_op import BinarySubOp
from noeira.nn.primitives.ops.binary_add_op import BinaryAddOp
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


def _be_fwd_kernel[
    M: Int, OP: BinaryElementOp
](
    a: Tile[DT, M],
    b: Tile[DT, M],
    o: Tile[DT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        o[i] = OP.forward_scalar(
            rebind[Scalar[DT]](a[i]), rebind[Scalar[DT]](b[i])
        )


def _be_bwd_kernel[
    M: Int, OP: BinaryElementOp
](
    a: Tile[DT, M],
    b: Tile[DT, M],
    go: Tile[DT, M],
    gi0: Tile[DT, M],
    gi1: Tile[DT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        var c = OP.cache_scalar(
            rebind[Scalar[DT]](a[i]), rebind[Scalar[DT]](b[i])
        )
        var g = rebind[Scalar[DT]](go[i])
        gi0[i] = OP.backward_scalar_x(c, g)
        gi1[i] = OP.backward_scalar_y(c, g)


struct BinaryElementwise[DIM_: Int, OP: BinaryElementOp](Module):
    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    @staticmethod
    def display_label() -> String:
        return Self.OP.display_label()

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
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref a = inputs[0]
        ref b = inputs[1]
        comptime if target == "cpu":
            out.ensure(M)
            var ap = a.data.unsafe_ptr()
            var bp = b.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= M:
                op.unsafe_store(
                    k,
                    Self.OP.forward_simd[CPU_SIMD_W](
                        ap.unsafe_load[width=CPU_SIMD_W](k),
                        bp.unsafe_load[width=CPU_SIMD_W](k),
                    ),
                )
                k += CPU_SIMD_W
            while k < M:
                op[unsafe_offset=k] = Self.OP.forward_scalar(ap[unsafe_offset=k], bp[unsafe_offset=k])
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_be_fwd_kernel[M, Self.OP]](
                a.tt["gpu", M](),
                b.tt["gpu", M](),
                out.tt["gpu", M](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref a = forward_input[0]
        ref b = forward_input[1]
        ref gi0 = grad_inputs[0]
        ref gi1 = grad_inputs[1]
        comptime if target == "cpu":
            gi0.ensure(M)
            gi1.ensure(M)
            var ap = a.data.unsafe_ptr()
            var bp = b.data.unsafe_ptr()
            var gp = grad_output.data.unsafe_ptr()
            var g0 = gi0.data.unsafe_ptr()
            var g1 = gi1.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= M:
                var cc = Self.OP.cache_simd[CPU_SIMD_W](
                    ap.unsafe_load[width=CPU_SIMD_W](k), bp.unsafe_load[width=CPU_SIMD_W](k)
                )
                var gv = gp.unsafe_load[width=CPU_SIMD_W](k)
                g0.unsafe_store(k, Self.OP.backward_simd_x[CPU_SIMD_W](cc, gv))
                g1.unsafe_store(k, Self.OP.backward_simd_y[CPU_SIMD_W](cc, gv))
                k += CPU_SIMD_W
            while k < M:
                var cc = Self.OP.cache_scalar(ap[unsafe_offset=k], bp[unsafe_offset=k])
                g0[unsafe_offset=k] = Self.OP.backward_scalar_x(cc, gp[unsafe_offset=k])
                g1[unsafe_offset=k] = Self.OP.backward_scalar_y(cc, gp[unsafe_offset=k])
                k += 1
        else:
            var c = ctx.value()
            gi0.ensure_gpu(c, M)
            gi1.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_be_bwd_kernel[M, Self.OP]](
                a.tt["gpu", M](),
                b.tt["gpu", M](),
                grad_output.tt["gpu", M](),
                gi0.tt["gpu", M](),
                gi1.tt["gpu", M](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).


comptime BinaryElemMin[DIM: Int] = BinaryElementwise[DIM, BinaryElemMinOp]
comptime BinarySub[DIM: Int] = BinaryElementwise[DIM, BinarySubOp]
comptime BinaryElemAdd[DIM: Int] = BinaryElementwise[DIM, BinaryAddOp]
