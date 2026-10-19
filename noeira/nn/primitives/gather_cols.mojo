"""GatherCols[NA] — per-row gather by integer index (storage surface).

Transformed from legacy `nn.primitives.GatherCols` (surface-only change). The
GPU forward + two zero-fill vjp kernels are carried over verbatim.

Two inputs: `values` ([B, NA]) and `idx` ([B, 1] holding integer column indices
stored as `Scalar[DT]`). Output: `out[b, 0] = values[b, Int(idx[b, 0])]`.

**Forward-only semantics** — vjp zero-fills both grad_values and grad_idx; the
gradient never flows through a gather-by-discrete-index in the DQN topology (the
surrounding block owns the scatter that builds grad_q_all). ARITY 2, no params,
no cache.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


def _dims2(d0: Int, d1: Int) -> Array[Int, 2]:
    var a = Array[Int, 2](fill=0)
    a[0] = d0
    a[1] = d1
    return a^


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _gather_cols_forward_kernel[
    BATCH: Int, NA: Int,
](
    values: Tile[DT, BATCH, NA],
    idx: Tile[DT, BATCH, 1],
    output: Tile[DT, BATCH, 1],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var a = Int(rebind[Scalar[DT]](idx[b, 0]))
        output[b, 0] = rebind[Scalar[DT]](values[b, a])


def _gather_cols_zero_values_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_values: Tile[DT, BATCH, NA],
):
    var lin = Int(global_idx.x)
    var total = BATCH * NA
    if lin < total:
        var b = lin // NA
        var k = lin % NA
        grad_values[b, k] = Scalar[DT](0.0)


def _gather_cols_zero_idx_grad_kernel[
    BATCH: Int,
](
    grad_idx: Tile[DT, BATCH, 1],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        grad_idx[b, 0] = Scalar[DT](0.0)


struct GatherCols[NA_: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = _dims2(Self.NA_, 1)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert Self.NA_ > 0, "GatherCols: NA must be > 0"
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref values = inputs[0]
        ref idx = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            var v_p = values.data.unsafe_ptr()
            var i_p = idx.data.unsafe_ptr()
            for b in range(B):
                var a = Int(i_p[unsafe_offset=b])
                out.data[b] = v_p[unsafe_offset=b * Self.NA_ + a]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_gather_cols_forward_kernel[B, Self.NA_]](
                values.tt["gpu", B, Self.NA_](),
                idx.tt["gpu", B, 1](),
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
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Forward-only op: both grad_values and grad_idx zero-fill."""
        ref gv = grad_inputs[0]
        ref gi = grad_inputs[1]
        comptime if target == "cpu":
            gv.ensure(B * Self.NA_)
            gi.ensure(B)
            for b in range(B):
                for k in range(Self.NA_):
                    gv.data[b * Self.NA_ + k] = Scalar[DT](0.0)
                gi.data[b] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            gv.ensure_gpu(c, B * Self.NA_)
            gi.ensure_gpu(c, B)
            comptime values_total = B * Self.NA_
            comptime values_blocks = (values_total + TPB - 1) // TPB
            c.enqueue_function[
                _gather_cols_zero_values_grad_kernel[B, Self.NA_]
            ](
                gv.tt["gpu", B, Self.NA_](),
                grid_dim=values_blocks,
                block_dim=TPB,
            )
            comptime idx_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_gather_cols_zero_idx_grad_kernel[B]](
                gi.tt["gpu", B, 1](),
                grid_dim=idx_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
