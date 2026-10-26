"""RepeatKVHeads[SEQ, N_KV, REP, HEAD_DIM] — broadcast K/V heads for GQA.

Grouped-query attention gives Q more heads than K and V: SmolLM2 has 15 query
heads over 5 key/value heads. Attention itself still needs one K/V head per Q
head, so each K/V head is reused by `REP = N_HEADS // N_KV` query heads. This
leaf materialises that reuse, turning `[B, SEQ * N_KV * HEAD_DIM]` into
`[B, SEQ * N_KV*REP * HEAD_DIM]` so an ordinary attention core can consume it.

    out[b, t, g, :] = in[b, t, g // REP, :]        g in [0, N_KV*REP)

## ⚠ Which query heads share a K/V head

`g // REP`, not `g % N_KV`. Both are legal-looking groupings of the same head
count and both produce identically shaped tensors; only one matches the file.
HuggingFace's `repeat_kv` is

    x[:, :, None, :, :].expand(B, N_KV, REP, S, D).reshape(B, N_KV*REP, S, D)

which lays the REP copies of kv head `k` at output heads `k*REP … k*REP+REP-1`
— i.e. output head `g` reads kv head `g // REP`. Choosing the interleaved
`g % N_KV` instead pairs every query head with the wrong key head: same shapes,
finite numbers, silently different model.

## Backward

Forward is a pure broadcast, so the VJP is the matching SUM — each input head
collects the gradient of all REP output heads that read it. Getting this wrong
in the obvious way (copying one instead of summing REP) scales the K/V
gradients by 1/REP and shows up only as slow training.

Param-free, cache-free. CPU + GPU.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


def _rkv_fwd_kernel[
    BATCH: Int, SEQ: Int, N_KV: Int, REP: Int, HD: Int
](
    src: Tile[DT, BATCH, SEQ * N_KV * HD],
    dst: Tile[DT, BATCH, SEQ * N_KV * REP * HD],
):
    comptime IN_W = N_KV * HD
    comptime OUT_W = N_KV * REP * HD
    comptime TOTAL = BATCH * SEQ * OUT_W
    var idx = Int(global_idx.x)
    if idx >= TOTAL:
        return
    var d = idx % HD
    var r1 = idx // HD
    var g = r1 % (N_KV * REP)
    var r2 = r1 // (N_KV * REP)
    var t = r2 % SEQ
    var b = r2 // SEQ
    var kv = g // REP
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = b * SEQ * IN_W + t * IN_W + kv * HD + d]
    )


def _rkv_bwd_kernel[
    BATCH: Int, SEQ: Int, N_KV: Int, REP: Int, HD: Int
](
    go: Tile[DT, BATCH, SEQ * N_KV * REP * HD],
    gi: Tile[DT, BATCH, SEQ * N_KV * HD],
):
    comptime IN_W = N_KV * HD
    comptime OUT_W = N_KV * REP * HD
    comptime TOTAL = BATCH * SEQ * IN_W
    var idx = Int(global_idx.x)
    if idx >= TOTAL:
        return
    var d = idx % HD
    var r1 = idx // HD
    var kv = r1 % N_KV
    var r2 = r1 // N_KV
    var t = r2 % SEQ
    var b = r2 // SEQ
    var acc = Scalar[DT](0)
    var base = b * SEQ * OUT_W + t * OUT_W + kv * REP * HD + d
    for r in range(REP):
        acc += rebind[Scalar[DT]](go.ptr[unsafe_offset = base + r * HD])
    gi.ptr[unsafe_offset=idx] = acc


struct RepeatKVHeads[SEQ: Int, N_KV: Int, REP: Int, HEAD_DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_W: Int = Self.N_KV * Self.HEAD_DIM
    comptime OUT_W: Int = Self.N_KV * Self.REP * Self.HEAD_DIM
    comptime IN_N: Int = Self.SEQ * Self.IN_W
    comptime OUT_N: Int = Self.SEQ * Self.OUT_W
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_N)
    comptime OUT_DIM: Int = Self.OUT_N

    def __init__(out self):
        comptime assert Self.REP >= 1, "RepeatKVHeads: REP must be >= 1"

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_N)
            var xp = x.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            for b in range(B):
                for t in range(Self.SEQ):
                    var ib = b * Self.IN_N + t * Self.IN_W
                    var ob = b * Self.OUT_N + t * Self.OUT_W
                    for kv in range(Self.N_KV):
                        for r in range(Self.REP):
                            var g = kv * Self.REP + r
                            for d in range(Self.HEAD_DIM):
                                op[
                                    unsafe_offset = ob + g * Self.HEAD_DIM + d
                                ] = xp[
                                    unsafe_offset = ib + kv * Self.HEAD_DIM + d
                                ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_N)
            comptime TOTAL = B * Self.OUT_N
            comptime n_blocks = (TOTAL + TPB - 1) // TPB
            c.enqueue_function[
                _rkv_fwd_kernel[
                    B, Self.SEQ, Self.N_KV, Self.REP, Self.HEAD_DIM
                ]
            ](
                x.tt["gpu", B, Self.IN_N](),
                out.tt["gpu", B, Self.OUT_N](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
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
            var gp = grad_output.data.unsafe_ptr()
            var ip = gin.data.unsafe_ptr()
            for b in range(B):
                for t in range(Self.SEQ):
                    var ib = b * Self.IN_N + t * Self.IN_W
                    var ob = b * Self.OUT_N + t * Self.OUT_W
                    for kv in range(Self.N_KV):
                        for d in range(Self.HEAD_DIM):
                            var acc = Scalar[DT](0)
                            for r in range(Self.REP):
                                var g = kv * Self.REP + r
                                acc += gp[
                                    unsafe_offset = ob + g * Self.HEAD_DIM + d
                                ]
                            ip[unsafe_offset = ib + kv * Self.HEAD_DIM + d] = acc
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_N)
            comptime TOTAL = B * Self.IN_N
            comptime n_blocks = (TOTAL + TPB - 1) // TPB
            c.enqueue_function[
                _rkv_bwd_kernel[
                    B, Self.SEQ, Self.N_KV, Self.REP, Self.HEAD_DIM
                ]
            ](
                grad_output.tt["gpu", B, Self.OUT_N](),
                gin.tt["gpu", B, Self.IN_N](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass
