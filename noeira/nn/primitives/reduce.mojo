"""Reduction Modules — `Reduce[DIM, OP: ReduceOp]` + `Sum` / `Mean` aliases.

Transformed from legacy `nn.primitives.reduce` (surface-only change). The CPU
loops + the two GPU kernels are carried over verbatim. `Sum` / `Mean` collapse
to one-line aliases over `Reduce[DIM, OP]`, with the per-op math factored into
`SumOp` / `MeanOp` (`ReduceOp` trait). The legacy split the `ReduceOp` trait
and the two op impls into `core/reduce_op.mojo` + `primitives/ops/{sum,mean}_op`;
here they're inlined into this single file (the storage surface has no `ops/`
subpackage) — the `scale_factor` math is identical.

Forward:  `out[b, 0] = OP.scale_factor[DIM]() · Σ_d input[b, d]`
Backward: `grad_in[b, d] = OP.scale_factor[DIM]() · grad_out[b, 0]`
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# ReduceOp trait + SumOp / MeanOp impls (verbatim math from legacy
# core/reduce_op.mojo + primitives/ops/{sum,mean}_op.mojo, inlined).
# ──────────────────────────────────────────────────────────────────────
trait ReduceOp(Movable & Deinitable):
    """Marker trait — linear-reduction op providing a comptime scale factor."""

    @staticmethod
    def scale_factor[DIM: Int]() -> Scalar[DT]:
        ...


struct SumOp(ReduceOp):
    """Sum reduction: `out = Σ x`, `grad_in[d] = grad_out`."""

    @staticmethod
    def scale_factor[DIM: Int]() -> Scalar[DT]:
        return Scalar[DT](1.0)


struct MeanOp(ReduceOp):
    """Mean reduction: `out = (1/DIM)·Σ x`, `grad_in[d] = grad_out / DIM`."""

    @staticmethod
    def scale_factor[DIM: Int]() -> Scalar[DT]:
        return Scalar[DT](1.0) / Scalar[DT](DIM)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — OP supplies the scale factor via `OP.scale_factor[DIM]()`.
# Inside top-level kernel functions, must use bare `OP` (not `Self.OP`):
# see `feedback_mojo_kernel_op_param_scope`.
# ──────────────────────────────────────────────────────────────────────


def _reduce_forward_kernel[
    BATCH: Int, DIM: Int, OP: ReduceOp,
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, 1],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var acc: Scalar[DT] = 0.0
        for d in range(DIM):
            acc += rebind[Scalar[DT]](input[b, d])
        output[b, 0] = acc * OP.scale_factor[DIM]()


def _reduce_broadcast_kernel[
    BATCH: Int, DIM: Int, OP: ReduceOp,
](
    grad_output: Tile[DT, BATCH, 1],
    grad_input: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var d = idx % DIM
        grad_input[b, d] = (
            rebind[Scalar[DT]](grad_output[b, 0]) * OP.scale_factor[DIM]()
        )


# ──────────────────────────────────────────────────────────────────────
# Reduce[DIM, OP] — shared body for every linear reduction.
# ──────────────────────────────────────────────────────────────────────


struct Reduce[DIM_: Int, OP: ReduceOp](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B)
            var in_t = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var out_t = TileTensor(out.data, row_major[B, 1]())
            var scale = Self.OP.scale_factor[Self.DIM_]()
            for b in range(B):
                var acc: Scalar[DT] = 0.0
                for d in range(Self.DIM_):
                    acc += in_t[b, d]
                out_t[b, 0] = acc * scale
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[
                _reduce_forward_kernel[B, Self.DIM_, Self.OP]
            ](
                in0.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, 1](),
                grid_dim=n_blocks,
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var go_t = TileTensor(grad_output.data, row_major[B, 1]())
            var gi_t = TileTensor(gin.data, row_major[B, Self.DIM_]())
            var scale = Self.OP.scale_factor[Self.DIM_]()
            for b in range(B):
                var go_scaled = go_t[b, 0] * scale
                for d in range(Self.DIM_):
                    gi_t[b, d] = go_scaled
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime total = B * Self.DIM_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _reduce_broadcast_kernel[B, Self.DIM_, Self.OP]
            ](
                grad_output.tt["gpu", B, 1](),
                gin.tt["gpu", B, Self.DIM_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).


# ──────────────────────────────────────────────────────────────────────
# Leaf aliases.
# ──────────────────────────────────────────────────────────────────────


comptime Sum[DIM: Int] = Reduce[DIM, SumOp]
comptime Mean[DIM: Int] = Reduce[DIM, MeanOp]
