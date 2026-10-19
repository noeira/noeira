"""Concat[*DIMS] — variadic feature-axis concatenation (storage surface).

  inputs (in0 [BATCH, DIMS[0]], in1 [BATCH, DIMS[1]], …)  →  [BATCH, ΣDIMS]
  out[b, off_i : off_i+DIMS[i]] = in_i[b]      off_i = Σ_{j<i} DIMS[j]  (comptime)

`ARITY = DIMS.size` (≥ 2), `OUT_DIM = Σ DIMS[i]`. Mirrors the legacy variadic
`Concat[*DIMS]`; the storage `TensorRefs[ARITY]` input pack makes the N-ary form
as clean as a binary one (the legacy framework only had `Concat2` here, which
forced 3+-input concats to be expressed as a chain — this restores the variadic
primitive). The SAC critic's `concat(state, action)` is just N=2 (`Concat2`,
the parametric alias below); tdmpc2 / redq_ofe graphs use N=2…4.

No params, no cache (backward is a pure slice-split of grad_output). GPU forward
launches N small slab-copy kernels (one per input); backward symmetric.

Backward: grad_in_i[b, :] = grad_output[b, off_i : off_i+DIMS[i]].
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


# ── comptime variadic helpers (mirror Parallel / legacy concat) ─────────
def _total_dim[*DIMS: Int]() -> Int:
    var s = 0
    comptime for i in range(DIMS.size):
        s += DIMS[i]
    return s


def _cum_offset[index: Int, *DIMS: Int]() -> Int:
    var s = 0
    comptime for j in range(index):
        s += DIMS[j]
    return s


def _build_in_dims[*DIMS: Int]() -> Array[Int, DIMS.size]:
    var d = Array[Int, DIMS.size](fill=0)
    comptime for k in range(DIMS.size):
        d[k] = DIMS[k]
    return d^


# ── per-slab copy kernels (one launch per input; legacy-style) ──────────
def _concat_copy_in_kernel[
    BATCH: Int, SRC_DIM: Int, OUT_DIM: Int, DST_OFF: Int, ADT: DType = DT
](
    src: Tile[ADT, BATCH, SRC_DIM],
    output: Tile[ADT, BATCH, OUT_DIM],
):
    var idx = Int(global_idx.x)
    var total = BATCH * SRC_DIM
    if idx < total:
        var b = idx // SRC_DIM
        var d = idx % SRC_DIM
        output[b, DST_OFF + d] = rebind[Scalar[ADT]](src[b, d])


def _concat_copy_out_kernel[
    BATCH: Int, DST_DIM: Int, OUT_DIM: Int, SRC_OFF: Int, ADT: DType = DT
](
    grad_output: Tile[ADT, BATCH, OUT_DIM],
    grad_in: Tile[ADT, BATCH, DST_DIM],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DST_DIM
    if idx < total:
        var b = idx // DST_DIM
        var d = idx % DST_DIM
        grad_in[b, d] = rebind[Scalar[ADT]](grad_output[b, SRC_OFF + d])


struct Concat[*DIMS: Int, ADT: DType = DT](Module):
    comptime ARITY = Self.DIMS.size
    comptime IN_DIMS = _build_in_dims[*Self.DIMS]()
    comptime OUT_DIM = _total_dim[*Self.DIMS]()
    # Activation-flow dtype (AMP). Concat is dtype-TRANSPARENT (pure slab copy /
    # slice-split grad) → carries ACT_DT through unchanged. ACT_DT == DT
    # (default, keyword param after the variadic *DIMS) → byte-identical fp32.
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return String("Concat")

    def __init__(out self):
        comptime assert Self.DIMS.size >= 2, "Concat: needs at least 2 inputs"

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime OUT = Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(B * OUT)
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                ref in_i = inputs[i]
                for b in range(B):
                    for d in range(D):
                        out.data[b * OUT + OFF + d] = in_i.data[b * D + d]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * OUT)
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                comptime nblk = (B * D + TPB - 1) // TPB
                ref in_i = inputs[i]
                c.enqueue_function[
                    _concat_copy_in_kernel[B, D, OUT, OFF, Self.ACT_DT]
                ](
                    in_i.tt["gpu", B, D](),
                    out.tt["gpu", B, OUT](),
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
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime OUT = Self.OUT_DIM
        comptime if target == "cpu":
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                ref gi = grad_inputs[i]
                gi.ensure(B * D)
                for b in range(B):
                    for d in range(D):
                        gi.data[b * D + d] = grad_output.data[b * OUT + OFF + d]
        else:
            var c = ctx.value()
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                comptime nblk = (B * D + TPB - 1) // TPB
                ref gi = grad_inputs[i]
                gi.ensure_gpu(c, B * D)
                c.enqueue_function[
                    _concat_copy_out_kernel[B, D, OUT, OFF, Self.ACT_DT]
                ](
                    grad_output.tt["gpu", B, OUT](),
                    gi.tt["gpu", B, D](),
                    grid_dim=nblk,
                    block_dim=TPB,
                )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).


# ── Concat2[D0, D1] — parametric alias kept for call-site compatibility ──
# The legacy storage surface exposed a fixed binary `Concat2`; it is now just
# the N=2 instance of the variadic `Concat` (mirrors `BranchConcat = Parallel`).
comptime Concat2[D0_: Int, D1_: Int] = Concat[D0_, D1_]
