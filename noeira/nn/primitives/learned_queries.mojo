"""LearnedQueries[IGNORE_DIM, N, D] — a learned constant token sequence.

Transformed from legacy `nn.primitives.LearnedQueries` (surface-only change; the
CPU loops and the 3 GPU kernels are carried over verbatim).

A `N·D` parameter, the same for every row of the batch — `N` learned query
tokens of width `D`. The (only) input is read for its BATCH count *and
nothing else*; its values are ignored and its gradient is zero:

    out[b, i] = queries[i]                         (∀ b)
    grad_input[b, ·] = 0
    grad_queries[i]  = sum_b grad_out[b, i]         (tokens shared → reduce)

This is the DETR/LeWM decoder's "fixed set of learnable query tokens, one
per target patch". It differs from `LearnedTokens` (which *concatenates*
learned tokens onto a real input stream): here the queries ARE the whole
stream, and the carrier input exists only so the op can live as a graph
node hanging off an upstream tensor (e.g. the encoder `emb`) to inherit
the batch dimension. `IGNORE_DIM` is that carrier's width.

Param is weight-decay-exempt (like `LearnedTokens`) and INIT-filled (NOT
zero — zero queries would make every patch identical and unrecoverable by
symmetry). CPU + GPU.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.initializer import Normal
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


comptime LQ_RTPB = 64  # reduction block size for the param-grad batch sum


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _lq_forward_kernel[
    BATCH: Int, OUT_DIM: Int
](
    param: Tile[DT, OUT_DIM],
    output: Tile[DT, BATCH, OUT_DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_DIM:
        return
    var i = idx % OUT_DIM
    output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](param.ptr[unsafe_offset=i])


def _lq_grad_input_zero_kernel[
    N: Int
](
    grad_input: Tile[DT, N],
):
    var idx = Int(global_idx.x)
    if idx < N:
        grad_input.ptr[unsafe_offset=idx] = Scalar[DT](0.0)


def _lq_grad_param_kernel[
    BATCH: Int, OUT_DIM: Int
](
    grad_output: Tile[DT, BATCH, OUT_DIM],
    grad_param: Tile[DT, OUT_DIM],
):
    # One block per param element; threads reduce over the batch.
    var col = Int(block_idx.x)
    if col >= OUT_DIM:
        return
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        acc += rebind[Scalar[DT]](grad_output.ptr[unsafe_offset=bi * OUT_DIM + col])
        bi += LQ_RTPB
    var total = block.sum[block_size=LQ_RTPB, broadcast=False](val=acc)
    if t == 0:
        grad_param.ptr[unsafe_offset=col] = rebind[Scalar[DT]](grad_param.ptr[unsafe_offset=col]) + total[0]


struct LearnedQueries[
    IGNORE_DIM: Int, N: Int, D: Int, EMB_STD: Float64 = 1.0
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IGNORE_DIM)
    comptime OUT_DIM = Self.N * Self.D
    comptime Q_SIZE = Self.N * Self.D

    var queries: Param["queries", False, Self.Q_SIZE]

    def __init__(out self):
        comptime assert Self.N > 0 and Self.D > 0, (
            "LearnedQueries: N, D must be > 0"
        )
        self.queries = Param["queries", False, Self.Q_SIZE]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "LearnedQueries: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.queries = Param["queries", False, Self.Q_SIZE].make[target](ctx)
        # ⚠⚠ N(0, 1) — `torch.nn.Embedding`'s default — NOT THE GRAPH'S `INIT`.
        # These rows are DETR's query / cls / additional-position embeddings
        # (`detr_vae.py:54,68,76`, all `nn.Embedding`), and the reference
        # draws them at unit scale. Under the graph's Kaiming rule they came
        # out at std sqrt(2/D) = 0.088 for D=256, and at that scale the query
        # rows are a perturbation on a near-uniform cross-attention softmax:
        # every chunk position attends to the same memory average, the decoder
        # output is IDENTICAL across the 40 positions (measured 2026-09-20,
        # `libero_act_inspect`: queries spread 0.087, decoder output spread
        # 0.0002 at rms 1.0, fresh and after training), and the gradient back
        # to the rows is proportionally tiny, so training never grows them.
        # Every ACT fit on LIBERO drove with the chunk's MEAN per state
        # (0-31/200 against a vision-free control at 144/200) because of this
        # line. The optimizer's INIT still names the rule for the rest of the
        # graph; an embedding is not a weight matrix and has no fan-in.
        Normal[0.0, Self.EMB_STD].init_weight[target](
            m.queries.val, Self.Q_SIZE, Self.D, Self.D, ctx
        )
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            var o_v = TileTensor(out.data, row_major[B, Self.OUT_DIM]())
            var q = TileTensor(self.queries.val.data, row_major[Self.OUT_DIM]())
            for b in range(B):
                for i in range(Self.OUT_DIM):
                    o_v[b, i] = q[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime lp = RowMajor[Self.OUT_DIM]
            comptime lbo = RowMajor[B, Self.OUT_DIM]
            comptime n_blocks = (B * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _lq_forward_kernel[B, Self.OUT_DIM]
            c.enqueue_function[kernel](
                self.queries.val.tt["gpu", Self.OUT_DIM](),
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime gi_total = B * Self.IGNORE_DIM
        comptime if target == "cpu":
            gin.ensure(gi_total)
            # grad_input = 0 (queries don't depend on the carrier input).
            for k in range(gi_total):
                gin.data[k] = Scalar[DT](0.0)
            var go = TileTensor(grad_output.data, row_major[B, Self.OUT_DIM]())
            var gq = TileTensor(self.queries.grd.data, row_major[Self.OUT_DIM]())
            for b in range(B):
                for i in range(Self.OUT_DIM):
                    gq[i] += go[b, i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, gi_total)
            comptime zblocks = (gi_total + TPB - 1) // TPB
            c.enqueue_function[_lq_grad_input_zero_kernel[gi_total]](
                gin.tt["gpu", gi_total](),
                grid_dim=zblocks,
                block_dim=TPB,
            )
            comptime lbo = RowMajor[B, Self.OUT_DIM]
            comptime lp = RowMajor[Self.OUT_DIM]
            comptime gpk = _lq_grad_param_kernel[B, Self.OUT_DIM]
            c.enqueue_function[gpk](
                grad_output.tt["gpu", B, Self.OUT_DIM](),
                self.queries.grd.tt["gpu", Self.OUT_DIM](),
                grid_dim=Self.OUT_DIM,
                block_dim=LQ_RTPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `queries` Param field).

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.Q_SIZE](
            self.queries.val, src.queries.val, tau, ctx
        )
