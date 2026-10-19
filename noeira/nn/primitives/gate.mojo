"""Gate[DIM] — AdaLN-zero residual gate (ARITY=3, storage surface).

    y = x + gate * branch

Three separate inputs:
    inputs[0] = x       [BATCH, DIM]   (residual stream)
    inputs[1] = gate    [BATCH, DIM]   (modulation gate from AdaLN)
    inputs[2] = branch  [BATCH, DIM]   (sub-layer output)

Gradients:
    grad_x[i]      = grad_out[i]
    grad_gate[i]   = grad_out[i] * branch[i]
    grad_branch[i] = grad_out[i] * gate[i]

Transformed from legacy `nn.primitives.Gate` (surface-only change). Cache (leaf-
owned): gate and branch. PARAM-free. With AdaLN zero-init the gate starts at 0 →
block is identity at init (the LeWM correctness invariant). The CPU loop + the
two GPU kernels are carried over verbatim.
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


def _gate_forward_kernel[
    BATCH: Int, DIM: Int,
](
    x: Tile[DT, BATCH, DIM],
    gate: Tile[DT, BATCH, DIM],
    branch: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache_gate: Tile[DT, BATCH, DIM],
    cache_branch: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var xv = rebind[Scalar[DT]](x[b, i])
    var gv = rebind[Scalar[DT]](gate[b, i])
    var brv = rebind[Scalar[DT]](branch[b, i])
    cache_gate[b, i] = gv
    cache_branch[b, i] = brv
    output[b, i] = xv + gv * brv


def _gate_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    cache_gate: Tile[DT, BATCH, DIM],
    cache_branch: Tile[DT, BATCH, DIM],
    grad_x: Tile[DT, BATCH, DIM],
    grad_gate: Tile[DT, BATCH, DIM],
    grad_branch: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var go = rebind[Scalar[DT]](grad_output[b, i])
    var gv = rebind[Scalar[DT]](cache_gate[b, i])
    var brv = rebind[Scalar[DT]](cache_branch[b, i])
    grad_x[b, i] = go
    grad_gate[b, i] = go * brv
    grad_branch[b, i] = go * gv


struct Gate[DIM_: Int](Module):
    comptime ARITY = 3
    comptime IN_DIMS = Array[Int, 3](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var cache_gate: Tensor  # [BATCH, DIM]
    var cache_branch: Tensor  # [BATCH, DIM]

    def __init__(out self):
        self.cache_gate = Tensor()
        self.cache_branch = Tensor()

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
        ref gate = inputs[1]
        ref branch = inputs[2]
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            self.cache_gate.ensure(B * Self.DIM_)
            self.cache_branch.ensure(B * Self.DIM_)
            var x_t = TileTensor(x.data, row_major[B, Self.DIM_]())
            var g_t = TileTensor(gate.data, row_major[B, Self.DIM_]())
            var br_t = TileTensor(branch.data, row_major[B, Self.DIM_]())
            var out_t = TileTensor(out.data, row_major[B, Self.DIM_]())
            var cg = TileTensor(self.cache_gate.data, row_major[B, Self.DIM_]())
            var cb = TileTensor(
                self.cache_branch.data, row_major[B, Self.DIM_]()
            )
            for b in range(B):
                for i in range(Self.DIM_):
                    var gv = g_t[b, i]
                    var brv = br_t[b, i]
                    cg[b, i] = gv
                    cb[b, i] = brv
                    out_t[b, i] = x_t[b, i] + gv * brv
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_gate.ensure_gpu(c, B * Self.DIM_)
            self.cache_branch.ensure_gpu(c, B * Self.DIM_)
            comptime lay = RowMajor[B, Self.DIM_]
            comptime n_blocks = (B * Self.DIM_ + TPB - 1) // TPB
            c.enqueue_function[_gate_forward_kernel[B, Self.DIM_]](
                x.tt["gpu", B, Self.DIM_](),
                gate.tt["gpu", B, Self.DIM_](),
                branch.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
                self.cache_gate.tt["gpu", B, Self.DIM_](),
                self.cache_branch.tt["gpu", B, Self.DIM_](),
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
        ref gg = grad_inputs[1]
        ref gbr = grad_inputs[2]
        comptime if target == "cpu":
            gx.ensure(B * Self.DIM_)
            gg.ensure(B * Self.DIM_)
            gbr.ensure(B * Self.DIM_)
            var go_t = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var cg = TileTensor(self.cache_gate.data, row_major[B, Self.DIM_]())
            var cb = TileTensor(
                self.cache_branch.data, row_major[B, Self.DIM_]()
            )
            var gx_t = TileTensor(gx.data, row_major[B, Self.DIM_]())
            var gg_t = TileTensor(gg.data, row_major[B, Self.DIM_]())
            var gbr_t = TileTensor(gbr.data, row_major[B, Self.DIM_]())
            for b in range(B):
                for i in range(Self.DIM_):
                    var g = go_t[b, i]
                    gx_t[b, i] = g
                    gg_t[b, i] = g * cb[b, i]
                    gbr_t[b, i] = g * cg[b, i]
        else:
            var c = ctx.value()
            gx.ensure_gpu(c, B * Self.DIM_)
            gg.ensure_gpu(c, B * Self.DIM_)
            gbr.ensure_gpu(c, B * Self.DIM_)
            comptime lay = RowMajor[B, Self.DIM_]
            comptime n_blocks = (B * Self.DIM_ + TPB - 1) // TPB
            c.enqueue_function[_gate_backward_kernel[B, Self.DIM_]](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.cache_gate.tt["gpu", B, Self.DIM_](),
                self.cache_branch.tt["gpu", B, Self.DIM_](),
                gx.tt["gpu", B, Self.DIM_](),
                gg.tt["gpu", B, Self.DIM_](),
                gbr.tt["gpu", B, Self.DIM_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf → no-op). No polyak_from (no Params).
