"""DuelingHead[NA] — Dueling DQN aggregation as a leaf Module (storage surface).

Input  shape: [B, 1 + NA]  — first column V(s), remaining NA columns A(s, ·).
Output shape: [B, NA]      — Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

Drops the average-advantage offset so the network learns identifiable
V and A streams (Wang et al. 2016).

Backward:
    grad_in[b, 0]     = Σ_a grad_out[b, a]
    grad_in[b, 1 + a] = grad_out[b, a] − (1 / NA) · Σ_a grad_out[b, a]

Use as: `Sequential[backbone..., Linear[H, 1 + NA], DuelingHead[NA]]`.
Pure architectural swap — no params, no cache. CPU + GPU.

Transformed from legacy `nn.primitives.DuelingHead` (surface-only change). The
CPU aggregation loops and the two GPU kernels (combine / grad) are carried over
verbatim.
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


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _dueling_combine_kernel[
    BATCH: Int, NA: Int,
](
    raw_in: Tile[DT, BATCH, NA + 1],
    q_out: Tile[DT, BATCH, NA],
):
    """`q_out[b, a] = raw_in[b, 0] + raw_in[b, 1 + a] − mean_a raw_in[b, 1 + a]`.
    1 thread per BATCH row, scalar inner loop over NA.
    """
    var b = Int(global_idx.x)
    if b < BATCH:
        var v = rebind[Scalar[DT]](raw_in[b, 0])
        var mean_a: Scalar[DT] = 0.0
        for a in range(NA):
            mean_a = mean_a + rebind[Scalar[DT]](raw_in[b, 1 + a])
        mean_a = mean_a * (Scalar[DT](1.0) / Scalar[DT](NA))
        for a in range(NA):
            var adv = rebind[Scalar[DT]](raw_in[b, 1 + a])
            q_out[b, a] = v + (adv - mean_a)


def _dueling_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_out: Tile[DT, BATCH, NA],
    grad_in: Tile[DT, BATCH, NA + 1],
):
    """`grad_in[b, 0] = Σ_a grad_out[b, a]`,
    `grad_in[b, 1 + a] = grad_out[b, a] − (1 / NA) · Σ_a grad_out[b, a]`.
    1 thread per BATCH row.
    """
    var b = Int(global_idx.x)
    if b < BATCH:
        var sum_dq: Scalar[DT] = 0.0
        for a in range(NA):
            sum_dq = sum_dq + rebind[Scalar[DT]](grad_out[b, a])
        grad_in[b, 0] = sum_dq
        var inv = Scalar[DT](1.0) / Scalar[DT](NA)
        for a in range(NA):
            grad_in[b, 1 + a] = (
                rebind[Scalar[DT]](grad_out[b, a]) - inv * sum_dq
            )


struct DuelingHead[NA: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.NA + 1)
    comptime OUT_DIM: Int = Self.NA

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params) but accepted
        for Sequential.make[target, INIT] uniformity."""
        comptime assert target == "cpu" or target == "gpu", (
            "DuelingHead: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "DuelingHead: NA must be > 0"
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
            out.ensure(B * Self.OUT_DIM)
            var input = TileTensor(
                in0.data, row_major[B, Self.NA + 1]()
            )
            var output_v = TileTensor(out.data, row_major[B, Self.NA]())
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(B):
                var v = input[b, 0]
                var sum_a: Scalar[DT] = 0.0
                for a in range(Self.NA):
                    sum_a = sum_a + input[b, 1 + a]
                var mean_a = sum_a * inv
                for a in range(Self.NA):
                    output_v[b, a] = v + (input[b, 1 + a] - mean_a)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime n_blocks = (B + TPB - 1) // TPB
            comptime kernel = _dueling_combine_kernel[B, Self.NA]
            c.enqueue_function[kernel](
                in0.tt["gpu", B, Self.NA + 1](),
                out.tt["gpu", B, Self.NA](),
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
            gin.ensure(B * (Self.NA + 1))
            var grad_output_v = TileTensor(
                grad_output.data, row_major[B, Self.NA]()
            )
            var grad_input_v = TileTensor(
                gin.data, row_major[B, Self.NA + 1]()
            )
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(B):
                var sum_dq: Scalar[DT] = 0.0
                for a in range(Self.NA):
                    sum_dq = sum_dq + grad_output_v[b, a]
                grad_input_v[b, 0] = sum_dq
                for a in range(Self.NA):
                    grad_input_v[b, 1 + a] = (
                        grad_output_v[b, a] - inv * sum_dq
                    )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * (Self.NA + 1))
            comptime n_blocks = (B + TPB - 1) // TPB
            comptime kernel = _dueling_grad_kernel[B, Self.NA]
            c.enqueue_function[kernel](
                grad_output.tt["gpu", B, Self.NA](),
                gin.tt["gpu", B, Self.NA + 1](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
