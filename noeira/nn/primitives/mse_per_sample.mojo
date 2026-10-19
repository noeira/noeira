"""MSEPerSample[DIM] — per-row mean squared error, a graph-node loss (storage).

    out[b, 0] = (1/DIM) * sum_i (a[b,i] - b[b,i])^2

ARITY=2, OUT_DIM=1, PARAM-free. Transformed from legacy
`nn.primitives.MSEPerSample` (surface-only change). Produces a (B,1) per-sample
loss so it can be the single output of a ComputeGraph (the SAC actor-loss
pattern): the Step block then reduces to the batch mean and seeds grad=1/B.
The CPU loop + the two GPU kernels are carried over verbatim.

Cache (leaf-owned): diff = a-b (DIM per row).
Backward (c = grad_out·2/DIM):
    grad_a[b,i] =  c · (a-b);   grad_b[b,i] = -c · (a-b)
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


def _mps_forward_kernel[BATCH: Int, DIM: Int](
    a: Tile[DT, BATCH, DIM],
    b: Tile[DT, BATCH, DIM],
    o: Tile[DT, BATCH, 1],
    diff: Tile[DT, BATCH, DIM],
):
    var r = Int(global_idx.x)
    if r >= BATCH:
        return
    var s = Scalar[DT](0)
    for i in range(DIM):
        var d = rebind[Scalar[DT]](a[r, i]) - rebind[Scalar[DT]](b[r, i])
        diff[r, i] = d
        s += d * d
    o[r, 0] = s / Scalar[DT](DIM)


def _mps_backward_kernel[BATCH: Int, DIM: Int](
    go: Tile[DT, BATCH, 1],
    diff: Tile[DT, BATCH, DIM],
    ga: Tile[DT, BATCH, DIM],
    gb: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var r = idx // DIM
    var i = idx % DIM
    var c = rebind[Scalar[DT]](go[r, 0]) * Scalar[DT](2.0 / Float64(DIM))
    var v = c * rebind[Scalar[DT]](diff[r, i])
    ga[r, i] = v
    gb[r, i] = -v


struct MSEPerSample[DIM_: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.DIM_)
    comptime OUT_DIM = 1

    var cache_diff: Tensor  # [BATCH, DIM]

    def __init__(out self):
        self.cache_diff = Tensor()

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
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref a = inputs[0]
        ref b = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            self.cache_diff.ensure(B * Self.DIM_)
            var a_t = TileTensor(a.data, row_major[B, Self.DIM_]())
            var b_t = TileTensor(b.data, row_major[B, Self.DIM_]())
            var o_t = TileTensor(out.data, row_major[B, 1]())
            var diff_t = TileTensor(
                self.cache_diff.data, row_major[B, Self.DIM_]()
            )
            for r in range(B):
                var s = Scalar[DT](0)
                for i in range(Self.DIM_):
                    var d = a_t[r, i] - b_t[r, i]
                    diff_t[r, i] = d
                    s += d * d
                o_t[r, 0] = s / Scalar[DT](Self.DIM_)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            self.cache_diff.ensure_gpu(c, B * Self.DIM_)
            comptime lay2 = RowMajor[B, Self.DIM_]
            comptime lay1 = RowMajor[B, 1]
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_mps_forward_kernel[B, Self.DIM_]](
                a.tt["gpu", B, Self.DIM_](),
                b.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, 1](),
                self.cache_diff.tt["gpu", B, Self.DIM_](),
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
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref ga = grad_inputs[0]
        ref gb = grad_inputs[1]
        comptime if target == "cpu":
            ga.ensure(B * Self.DIM_)
            gb.ensure(B * Self.DIM_)
            var go_t = TileTensor(grad_output.data, row_major[B, 1]())
            var diff_t = TileTensor(
                self.cache_diff.data, row_major[B, Self.DIM_]()
            )
            var ga_t = TileTensor(ga.data, row_major[B, Self.DIM_]())
            var gb_t = TileTensor(gb.data, row_major[B, Self.DIM_]())
            for r in range(B):
                var c = go_t[r, 0] * Scalar[DT](2.0 / Float64(Self.DIM_))
                for i in range(Self.DIM_):
                    var v = c * diff_t[r, i]
                    ga_t[r, i] = v
                    gb_t[r, i] = -v
        else:
            var c = ctx.value()
            ga.ensure_gpu(c, B * Self.DIM_)
            gb.ensure_gpu(c, B * Self.DIM_)
            comptime lay2 = RowMajor[B, Self.DIM_]
            comptime lay1 = RowMajor[B, 1]
            comptime n_blocks = (B * Self.DIM_ + TPB - 1) // TPB
            c.enqueue_function[_mps_backward_kernel[B, Self.DIM_]](
                grad_output.tt["gpu", B, 1](),
                self.cache_diff.tt["gpu", B, Self.DIM_](),
                ga.tt["gpu", B, Self.DIM_](),
                gb.tt["gpu", B, Self.DIM_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf → no-op). No polyak_from (no Params).
