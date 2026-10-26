"""Slice[IN, START, END] — extracts column range `[START, END)` (storage surface).

Transformed from legacy `nn.primitives.Slice` (surface-only change). The CPU
loops + the two GPU kernels are carried over verbatim.

Forward:  out[b, j] = in[b, START + j]   for j in [0, OUT)
Backward: grad_in[b, k] = grad_out[b, k - START] for k in [START, START+OUT),
          0 elsewhere — zero-fills the rest so ComputeGraph's scatter-add into a
          shared predecessor `_grad_out_buf` interleaves correctly with parallel
          slicers (e.g. the q1/q2/log_prob unpack in `SACActorLoss`).

No params, no cache. Conforms to `Module`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _slice_forward_kernel[
    BATCH: Int, IN: Int, START: Int, OUT: Int, ADT: DType = DT,
](
    input: Tile[ADT, BATCH, IN],
    output: Tile[ADT, BATCH, OUT],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var j = idx % OUT
        output[b, j] = rebind[Scalar[ADT]](input[b, START + j])


def _slice_backward_kernel[
    BATCH: Int, IN: Int, START: Int, OUT: Int, ADT: DType = DT,
](
    grad_output: Tile[ADT, BATCH, OUT],
    grad_input: Tile[ADT, BATCH, IN],
):
    # Zero the whole grad_input and scatter the slice in. One thread
    # per [b, k] over the FULL input shape — k in [START, START+OUT)
    # gets grad_output, the rest gets 0.
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var k = idx % IN
        var zero: Scalar[ADT] = 0.0
        if k >= START and k < START + OUT:
            grad_input[b, k] = rebind[Scalar[ADT]](grad_output[b, k - START])
        else:
            grad_input[b, k] = zero


struct Slice[IN_: Int, START_: Int, END_: Int, ADT: DType = DT](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.END_ - Self.START_
    # Activation-flow dtype (AMP). Slice is dtype-TRANSPARENT — a pure copy +
    # zero-fill with no math/cast — so it carries ACT_DT through unchanged.
    # ACT_DT == DT (default) → byte-identical to the fp32 path.
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return String("Slice")

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert Self.START_ >= 0, "Slice.START must be >= 0"
        comptime assert Self.END_ > Self.START_, "Slice.END must be > START"
        comptime assert Self.END_ <= Self.IN_, "Slice.END must be <= IN_DIM"
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            var in_t = TileTensor(in0.data, row_major[B, Self.IN_]())
            var out_t = TileTensor(out.data, row_major[B, Self.OUT_DIM]())
            for b in range(B):
                for j in range(Self.OUT_DIM):
                    out_t[b, j] = in_t[b, Self.START_ + j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime n_blocks = (B * Self.OUT_DIM + TPB - 1) // TPB
            c.enqueue_function[
                _slice_forward_kernel[
                    B, Self.IN_, Self.START_, Self.OUT_DIM, Self.ACT_DT
                ]
            ](
                in0.tt["gpu", B, Self.IN_](),
                out.tt["gpu", B, Self.OUT_DIM](),
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
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_)
            var go_t = TileTensor(grad_output.data, row_major[B, Self.OUT_DIM]())
            var gi_t = TileTensor(gin.data, row_major[B, Self.IN_]())
            # Zero whole grad_input first; scatter the slice in afterward.
            # Zeros required for ComputeGraph scatter-add: when multiple
            # slicers share a predecessor, each writes its slice range and
            # leaves the rest at 0 so the scatter-add sums correctly.
            for b in range(B):
                for k in range(Self.IN_):
                    gi_t[b, k] = Scalar[Self.ACT_DT](0.0)
            for b in range(B):
                for j in range(Self.OUT_DIM):
                    gi_t[b, Self.START_ + j] = go_t[b, j]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            comptime n_blocks = (B * Self.IN_ + TPB - 1) // TPB
            c.enqueue_function[
                _slice_backward_kernel[
                    B, Self.IN_, Self.START_, Self.OUT_DIM, Self.ACT_DT
                ]
            ](
                grad_output.tt["gpu", B, Self.OUT_DIM](),
                gin.tt["gpu", B, Self.IN_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
