"""Modulate[DIM] — AdaLN-zero affine modulation (ARITY=3, storage surface).

    y = x * (1 + scale) + shift

Three separate inputs (the graph passes one pointer per slot — no concatenation):
    inputs[0] = x      [BATCH, DIM]
    inputs[1] = scale  [BATCH, DIM]
    inputs[2] = shift  [BATCH, DIM]

Gradients:
    grad_x[i]     = grad_out[i] * (1 + scale[i])
    grad_scale[i] = grad_out[i] * x[i]
    grad_shift[i] = grad_out[i]

Transformed from legacy `nn.primitives.Modulate` (surface-only change). Cache
(leaf-owned): x and scale (needed by the x- and scale-grads). PARAM-free. Used
inside LeWM's ConditionalTransformerBlock. The CPU loop + the two GPU kernels
are carried over verbatim.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


def _modulate_forward_kernel[
    BATCH: Int, DIM: Int,
](
    x: Tile[DT, BATCH, DIM],
    scale: Tile[DT, BATCH, DIM],
    shift: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache_x: Tile[DT, BATCH, DIM],
    cache_scale: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var xv = rebind[Scalar[DT]](x[b, i])
    var sv = rebind[Scalar[DT]](scale[b, i])
    var shv = rebind[Scalar[DT]](shift[b, i])
    cache_x[b, i] = xv
    cache_scale[b, i] = sv
    output[b, i] = xv * (Scalar[DT](1.0) + sv) + shv


def _modulate_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    cache_x: Tile[DT, BATCH, DIM],
    cache_scale: Tile[DT, BATCH, DIM],
    grad_x: Tile[DT, BATCH, DIM],
    grad_scale: Tile[DT, BATCH, DIM],
    grad_shift: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var go = rebind[Scalar[DT]](grad_output[b, i])
    var xv = rebind[Scalar[DT]](cache_x[b, i])
    var sv = rebind[Scalar[DT]](cache_scale[b, i])
    grad_x[b, i] = go * (Scalar[DT](1.0) + sv)
    grad_scale[b, i] = go * xv
    grad_shift[b, i] = go


struct Modulate[DIM_: Int](Module):
    comptime ARITY = 3
    comptime IN_DIMS = Array[Int, 3](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var cache_x: Tensor  # [BATCH, DIM]
    var cache_scale: Tensor  # [BATCH, DIM]

    def __init__(out self):
        self.cache_x = Tensor()
        self.cache_scale = Tensor()

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
        inputs: TensorRefs[3, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        ref scale = inputs[1]
        ref shift = inputs[2]
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            self.cache_x.ensure(B * Self.DIM_)
            self.cache_scale.ensure(B * Self.DIM_)
            var x_t = TileTensor(x.data, row_major[B, Self.DIM_]())
            var s_t = TileTensor(scale.data, row_major[B, Self.DIM_]())
            var sh_t = TileTensor(shift.data, row_major[B, Self.DIM_]())
            var out_t = TileTensor(out.data, row_major[B, Self.DIM_]())
            var cx = TileTensor(self.cache_x.data, row_major[B, Self.DIM_]())
            var cs = TileTensor(self.cache_scale.data, row_major[B, Self.DIM_]())
            for b in range(B):
                for i in range(Self.DIM_):
                    var xv = x_t[b, i]
                    var sv = s_t[b, i]
                    cx[b, i] = xv
                    cs[b, i] = sv
                    out_t[b, i] = xv * (Scalar[DT](1.0) + sv) + sh_t[b, i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_x.ensure_gpu(c, B * Self.DIM_)
            self.cache_scale.ensure_gpu(c, B * Self.DIM_)
            comptime lay = RowMajor[B, Self.DIM_]
            comptime n_blocks = (B * Self.DIM_ + TPB - 1) // TPB
            c.enqueue_function[_modulate_forward_kernel[B, Self.DIM_]](
                x.tt["gpu", B, Self.DIM_](),
                scale.tt["gpu", B, Self.DIM_](),
                shift.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
                self.cache_x.tt["gpu", B, Self.DIM_](),
                self.cache_scale.tt["gpu", B, Self.DIM_](),
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
        forward_input: TensorRefs[3, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[3, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gx = grad_inputs[0]
        ref gs = grad_inputs[1]
        ref gsh = grad_inputs[2]
        comptime if target == "cpu":
            gx.ensure(B * Self.DIM_)
            gs.ensure(B * Self.DIM_)
            gsh.ensure(B * Self.DIM_)
            var go_t = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var cx = TileTensor(self.cache_x.data, row_major[B, Self.DIM_]())
            var cs = TileTensor(self.cache_scale.data, row_major[B, Self.DIM_]())
            var gx_t = TileTensor(gx.data, row_major[B, Self.DIM_]())
            var gs_t = TileTensor(gs.data, row_major[B, Self.DIM_]())
            var gsh_t = TileTensor(gsh.data, row_major[B, Self.DIM_]())
            for b in range(B):
                for i in range(Self.DIM_):
                    var g = go_t[b, i]
                    gx_t[b, i] = g * (Scalar[DT](1.0) + cs[b, i])
                    gs_t[b, i] = g * cx[b, i]
                    gsh_t[b, i] = g
        else:
            var c = ctx.value()
            gx.ensure_gpu(c, B * Self.DIM_)
            gs.ensure_gpu(c, B * Self.DIM_)
            gsh.ensure_gpu(c, B * Self.DIM_)
            comptime lay = RowMajor[B, Self.DIM_]
            comptime n_blocks = (B * Self.DIM_ + TPB - 1) // TPB
            c.enqueue_function[_modulate_backward_kernel[B, Self.DIM_]](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.cache_x.tt["gpu", B, Self.DIM_](),
                self.cache_scale.tt["gpu", B, Self.DIM_](),
                gx.tt["gpu", B, Self.DIM_](),
                gs.tt["gpu", B, Self.DIM_](),
                gsh.tt["gpu", B, Self.DIM_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf → no-op). No polyak_from (no Params).
