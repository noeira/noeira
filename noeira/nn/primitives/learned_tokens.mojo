"""LearnedTokens[N_IN, N_NEW, D, PREPEND] — concatenate learned tokens.

Transformed from legacy `nn.primitives.LearnedTokens` (surface-only change; the
CPU loops and the 3 GPU kernels are carried over verbatim).

Dreamer 4's encoder prepends learned latent tokens to the projected patches;
the decoder appends learned patch-query tokens to the up-projected latents.
This leaf does that concat with the learned tokens as its (only) parameter,
shared across the whole B·T batch (every frame gets the same learned tokens):

    PREPEND : out = [ learned(N_NEW) ‖ input(N_IN) ]   (encoder latents)
    else    : out = [ input(N_IN) ‖ learned(N_NEW) ]   (decoder queries)

IN_DIM = N_IN·D, OUT_DIM = (N_IN+N_NEW)·D, param = N_NEW·D. The param grad is
batch-reduced (the tokens are shared): grad_tokens[k] = Σ_bt grad_out[bt, new+k].
CPU + GPU.

INIT_STD: when > 0 the tokens are N(0, INIT_STD) (the ViT CLS/query convention,
std≈0.02, independent of fan-in); otherwise the leaf-supplied `INIT` fills them.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.math import sqrt as fsqrt, log, cos, sin
from std.random import random_float64
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


comptime LT_RTPB = 64  # reduction block size for the param-grad batch sum
comptime _LT_TWO_PI: Float64 = 6.283185307179586


def _lt_fill_normal(mut tok: Tensor, n_elems: Int, std: Float64):
    """N(0, std) Box-Muller fill — the ViT convention for learned CLS/query
    tokens (std≈0.02), independent of fan-in. fan_in=1 Kaiming would give
    std≈1.4, a constant that swamps the per-image attention signal and
    collapses the readout (see LeWMEncoderCLS)."""
    var i = 0
    while i < n_elems:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(-2.0 * log(u1))
        tok.data[i] = Scalar[DT](std * r * cos(_LT_TWO_PI * u2))
        i += 1
        if i < n_elems:
            tok.data[i] = Scalar[DT](std * r * sin(_LT_TWO_PI * u2))
            i += 1


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _lt_forward_kernel[
    BATCH: Int, IN_N: Int, NEW_N: Int, OUT_DIM: Int, NEW_OFF: Int, IN_OFF: Int
](
    input: Tile[DT, BATCH, IN_N],
    param: Tile[DT, NEW_N],
    output: Tile[DT, BATCH, OUT_DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_DIM:
        return
    var bt = idx // OUT_DIM
    var pos = idx % OUT_DIM
    if pos >= NEW_OFF and pos < NEW_OFF + NEW_N:
        output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](param.ptr[unsafe_offset=pos - NEW_OFF])
    else:
        output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
            input.ptr[unsafe_offset=bt * IN_N + (pos - IN_OFF)]
        )


def _lt_grad_input_kernel[
    BATCH: Int, IN_N: Int, OUT_DIM: Int, IN_OFF: Int
](
    grad_output: Tile[DT, BATCH, OUT_DIM],
    grad_input: Tile[DT, BATCH, IN_N],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * IN_N:
        return
    var bt = idx // IN_N
    var k = idx % IN_N
    grad_input.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        grad_output.ptr[unsafe_offset=bt * OUT_DIM + IN_OFF + k]
    )


def _lt_grad_param_kernel[
    BATCH: Int, NEW_N: Int, OUT_DIM: Int, NEW_OFF: Int
](
    grad_output: Tile[DT, BATCH, OUT_DIM],
    grad_param: Tile[DT, NEW_N],
):
    # One block per param element; threads reduce over the batch.
    var col = Int(block_idx.x)
    if col >= NEW_N:
        return
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        acc += rebind[Scalar[DT]](
            grad_output.ptr[unsafe_offset=bi * OUT_DIM + NEW_OFF + col]
        )
        bi += LT_RTPB
    var total = block.sum[block_size=LT_RTPB, broadcast=False](val=acc)
    if t == 0:
        grad_param.ptr[unsafe_offset=col] = rebind[Scalar[DT]](grad_param.ptr[unsafe_offset=col]) + total[0]


struct LearnedTokens[
    N_IN: Int, N_NEW: Int, D: Int, PREPEND: Bool, INIT_STD: Float64 = 0.0
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.N_IN * Self.D)
    comptime OUT_DIM = (Self.N_IN + Self.N_NEW) * Self.D
    comptime NEW_N: Int = Self.N_NEW * Self.D
    comptime IN_N: Int = Self.N_IN * Self.D
    comptime NEW_OFF: Int = 0 if Self.PREPEND else Self.IN_N
    comptime IN_OFF: Int = Self.NEW_N if Self.PREPEND else 0

    var tokens: Param["tokens", False, Self.NEW_N]

    def __init__(out self):
        self.tokens = Param["tokens", False, Self.NEW_N]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "LearnedTokens: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.tokens = Param["tokens", False, Self.NEW_N].make[target](ctx)
        comptime if Self.INIT_STD > 0.0:
            _lt_fill_normal(m.tokens.val, Self.NEW_N, Self.INIT_STD)
            comptime if target == "gpu":
                m.tokens.val.upload(ctx.value())
        else:
            INIT.init_weight[target](
                m.tokens.val, Self.NEW_N, Self.N_NEW, Self.D, ctx
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
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            var inp = TileTensor(in0.data, row_major[B, Self.IN_N]())
            var o_v = TileTensor(out.data, row_major[B, Self.OUT_DIM]())
            var tok = TileTensor(self.tokens.val.data, row_major[Self.NEW_N]())
            for bt in range(B):
                for k in range(Self.NEW_N):
                    o_v[bt, Self.NEW_OFF + k] = tok[k]
                for k in range(Self.IN_N):
                    o_v[bt, Self.IN_OFF + k] = inp[bt, k]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime lbi = RowMajor[B, Self.IN_N]
            comptime lbo = RowMajor[B, Self.OUT_DIM]
            comptime lp = RowMajor[Self.NEW_N]
            comptime n_blocks = (B * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _lt_forward_kernel[
                B, Self.IN_N, Self.NEW_N, Self.OUT_DIM,
                Self.NEW_OFF, Self.IN_OFF,
            ]
            c.enqueue_function[kernel](
                in0.tt["gpu", B, Self.IN_N](),
                self.tokens.val.tt["gpu", Self.NEW_N](),
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
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_N)
            var go = TileTensor(grad_output.data, row_major[B, Self.OUT_DIM]())
            var gi = TileTensor(gin.data, row_major[B, Self.IN_N]())
            for bt in range(B):
                for k in range(Self.IN_N):
                    gi[bt, k] = go[bt, Self.IN_OFF + k]
            var gtok = TileTensor(
                self.tokens.grd.data, row_major[Self.NEW_N]()
            )
            for bt in range(B):
                for k in range(Self.NEW_N):
                    gtok[k] += go[bt, Self.NEW_OFF + k]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_N)
            comptime lbi = RowMajor[B, Self.IN_N]
            comptime lbo = RowMajor[B, Self.OUT_DIM]
            comptime lp = RowMajor[Self.NEW_N]
            comptime gin_blocks = (B * Self.IN_N + TPB - 1) // TPB
            comptime gik = _lt_grad_input_kernel[
                B, Self.IN_N, Self.OUT_DIM, Self.IN_OFF
            ]
            c.enqueue_function[gik](
                grad_output.tt["gpu", B, Self.OUT_DIM](),
                gin.tt["gpu", B, Self.IN_N](),
                grid_dim=gin_blocks,
                block_dim=TPB,
            )
            comptime gpk = _lt_grad_param_kernel[
                B, Self.NEW_N, Self.OUT_DIM, Self.NEW_OFF
            ]
            c.enqueue_function[gpk](
                grad_output.tt["gpu", B, Self.OUT_DIM](),
                self.tokens.grd.tt["gpu", Self.NEW_N](),
                grid_dim=Self.NEW_N,
                block_dim=LT_RTPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `tokens` Param field).

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.NEW_N](
            self.tokens.val, src.tokens.val, tau, ctx
        )
