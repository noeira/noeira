"""ReduceMax[NA] — per-row max reduction `[B, NA] → [B, 1]` (storage surface).

Transformed from legacy `nn.primitives.reduce_max` (surface-only change). The CPU
loops + the two GPU kernels are carried over verbatim.

**Forward-only primitive** (target-Y path for DQN — gradient never flows through
`max_a Q_target(s', a)`). The `Module` trait still requires a `vjp`; we zero-fill
`grad_input` (mirrors the `gather_cols` / `StopGrad` pattern). Callers that need a
gradient through max should use a different op.

Non-linear reduction — doesn't fit the `Reduce[DIM, OP]` template (linear only).
ARITY 1, no params, no cache.

Forward:  `out[b, 0] = max_a input[b, a]`
Backward: `grad_input[b, a] = 0` for all (b, a)
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
def _reduce_max_forward_kernel[
    BATCH: Int, NA: Int,
](
    input: Tile[DT, BATCH, NA],
    output: Tile[DT, BATCH, 1],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var best: Scalar[DT] = rebind[Scalar[DT]](input[b, 0])
        for a in range(1, NA):
            var v = rebind[Scalar[DT]](input[b, a])
            if v > best:
                best = v
        output[b, 0] = best


def _reduce_max_zero_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_input: Tile[DT, BATCH, NA],
):
    var idx = Int(global_idx.x)
    var total = BATCH * NA
    if idx < total:
        var b = idx // NA
        var a = idx % NA
        grad_input[b, a] = Scalar[DT](0.0)


struct ReduceMax[NA_: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.NA_)
    comptime OUT_DIM: Int = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert Self.NA_ > 0, "ReduceMax: NA must be > 0"
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
            var in_t = TileTensor(in0.data, row_major[B, Self.NA_]())
            var out_t = TileTensor(out.data, row_major[B, 1]())
            for b in range(B):
                var best: Scalar[DT] = in_t[b, 0]
                for a in range(1, Self.NA_):
                    var v: Scalar[DT] = in_t[b, a]
                    if v > best:
                        best = v
                out_t[b, 0] = best
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_reduce_max_forward_kernel[B, Self.NA_]](
                in0.tt["gpu", B, Self.NA_](),
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
        """Forward-only op: zero-fill grad_input regardless of grad_output."""
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.NA_)
            var gi_t = TileTensor(gin.data, row_major[B, Self.NA_]())
            for b in range(B):
                for a in range(Self.NA_):
                    gi_t[b, a] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.NA_)
            comptime total = B * Self.NA_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_reduce_max_zero_grad_kernel[B, Self.NA_]](
                gin.tt["gpu", B, Self.NA_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
