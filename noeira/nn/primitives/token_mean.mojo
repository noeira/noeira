"""TokenMean[SEQ_LEN, DIM] — mean-pool over the sequence axis (storage surface).

Transformed from legacy `nn.primitives.TokenMean` (surface-only change). The CPU
loops + the two GPU kernels (forward reduce / backward broadcast) are carried
over verbatim.

Each sample is `(SEQ_LEN, DIM)` laid out row-major (token-major); the op
averages over the `SEQ_LEN` tokens to produce a single `DIM` vector:

    out[b, d] = (1/SEQ_LEN) * sum_t in[b, t*DIM + d]

IN_DIM = SEQ_LEN*DIM, OUT_DIM = DIM; no params, no cache. Backward broadcasts
the upstream gradient evenly across tokens:

    grad_in[b, t*DIM + d] = grad_out[b, d] / SEQ_LEN

Used by ViT to collapse patch tokens to a single class vector before the
classification head.
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


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _token_mean_fwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int
](
    src: Tile[DT, BATCH, SEQ * DIM],
    dst: Tile[DT, BATCH, DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * DIM
    if gid >= total:
        return
    var b = gid // DIM
    var d = gid % DIM
    var acc: Scalar[DT] = 0.0
    for t in range(SEQ):
        acc += rebind[Scalar[DT]](src[b, t * DIM + d])
    dst[b, d] = acc / Scalar[DT](Float32(SEQ))


def _token_mean_bwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int
](
    go: Tile[DT, BATCH, DIM],
    gi: Tile[DT, BATCH, SEQ * DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (SEQ * DIM)
    var rem = gid % (SEQ * DIM)
    var d = rem % DIM
    gi[b, rem] = rebind[Scalar[DT]](go[b, d]) / Scalar[DT](Float32(SEQ))


struct TokenMean[SEQ_LEN_: Int, DIM_: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.SEQ_LEN_ * Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert Self.SEQ_LEN_ > 0 and Self.DIM_ > 0, (
            "TokenMean: SEQ_LEN, DIM must be > 0"
        )
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
        comptime SD = Self.SEQ_LEN_ * Self.DIM_
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            var in_v = TileTensor(in0.data, row_major[B, SD]())
            var out_v = TileTensor(out.data, row_major[B, Self.DIM_]())
            var inv_seq: Scalar[DT] = 1.0 / Float32(Self.SEQ_LEN_)
            for b in range(B):
                for d in range(Self.DIM_):
                    var acc: Scalar[DT] = 0.0
                    for t in range(Self.SEQ_LEN_):
                        acc += in_v[b, t * Self.DIM_ + d]
                    out_v[b, d] = acc * inv_seq
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            comptime l_in = RowMajor[B, SD]
            comptime l_out = RowMajor[B, Self.DIM_]
            comptime total = B * Self.DIM_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _token_mean_fwd_kernel[B, Self.SEQ_LEN_, Self.DIM_]
            ](
                in0.tt["gpu", B, SD](),
                out.tt["gpu", B, Self.DIM_](),
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
        comptime SD = Self.SEQ_LEN_ * Self.DIM_
        comptime if target == "cpu":
            gin.ensure(B * SD)
            var go_v = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_v = TileTensor(gin.data, row_major[B, SD]())
            var inv_seq: Scalar[DT] = 1.0 / Float32(Self.SEQ_LEN_)
            for b in range(B):
                for t in range(Self.SEQ_LEN_):
                    for d in range(Self.DIM_):
                        gi_v[b, t * Self.DIM_ + d] = go_v[b, d] * inv_seq
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * SD)
            comptime l_out = RowMajor[B, Self.DIM_]
            comptime l_in = RowMajor[B, SD]
            comptime total = B * SD
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _token_mean_bwd_kernel[B, Self.SEQ_LEN_, Self.DIM_]
            ](
                grad_output.tt["gpu", B, Self.DIM_](),
                gin.tt["gpu", B, SD](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
