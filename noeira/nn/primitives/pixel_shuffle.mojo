"""PixelShuffle[GRID, EMBED, SCALE] — fold a patch grid's space into channels.

SmolVLM's connector shrinks a `GRID x GRID` patch grid by `SCALE` in each
direction and pays for it in width, turning `[B, GRID^2, EMBED]` into
`[B, (GRID/SCALE)^2, EMBED*SCALE^2]`. At SmolVLA's numbers that is
1024 tokens x 768 -> **64 tokens x 12288**, which is what makes two or three
camera images affordable as a prefix.

Param-free: a pure permutation of the elements.

## The mapping, and why it is written closed-form

The reference (`transformers/models/smolvlm`, `SmolVLMConnector.pixel_shuffle`)
expresses it as five chained `view`/`permute`/`reshape` calls. Composing them:

    out[b, t, c] = in[b, h*GRID + w, e]
        t = (h / SCALE) * (GRID / SCALE) + (w / SCALE)          (integer div)
        c = (h % SCALE) * (SCALE*EMBED) + (w % SCALE) * EMBED + e

⚠ Every step in that chain is a reshape or a transpose, so **every wrong
composition of them is still shape-legal** — the output has the right size and
finite values and simply scrambles which patch went where. The closed form above
was checked against numpy replaying the reference's exact op sequence, at
(GRID, EMBED, SCALE) = (32, 768, 4), (4, 6, 2) and (6, 5, 3), before being
written here; `tests/nn/test_pixel_shuffle.mojo` re-derives it independently by
simulating the five steps rather than reusing this formula.

Backward is the inverse permutation. A permutation is orthogonal, so the adjoint
identity `<f(x), y> == <x, vjp(y)>` holds exactly and is the gate.
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


def _ps_fwd_kernel[
    BATCH: Int, GRID: Int, EMBED: Int, SCALE: Int, N: Int
](
    src: Tile[DT, BATCH, N],
    dst: Tile[DT, BATCH, N],
):
    comptime OG = GRID // SCALE
    comptime OC = EMBED * SCALE * SCALE
    var idx = Int(global_idx.x)
    if idx >= BATCH * N:
        return
    var b = idx // N
    var r = idx % N
    var c = r % OC
    var t = r // OC
    var h_hi = t // OG
    var w_hi = t % OG
    var h_lo = c // (SCALE * EMBED)
    var rem = c % (SCALE * EMBED)
    var w_lo = rem // EMBED
    var e = rem % EMBED
    var h = h_hi * SCALE + h_lo
    var w = w_hi * SCALE + w_lo
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = b * N + (h * GRID + w) * EMBED + e]
    )


def _ps_bwd_kernel[
    BATCH: Int, GRID: Int, EMBED: Int, SCALE: Int, N: Int
](
    go: Tile[DT, BATCH, N],
    gi: Tile[DT, BATCH, N],
):
    comptime OG = GRID // SCALE
    comptime OC = EMBED * SCALE * SCALE
    var idx = Int(global_idx.x)
    if idx >= BATCH * N:
        return
    var b = idx // N
    var r = idx % N
    var e = r % EMBED
    var hw = r // EMBED
    var w = hw % GRID
    var h = hw // GRID
    var t = (h // SCALE) * OG + (w // SCALE)
    var c = (h % SCALE) * (SCALE * EMBED) + (w % SCALE) * EMBED + e
    gi.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        go.ptr[unsafe_offset = b * N + t * OC + c]
    )


struct PixelShuffle[GRID: Int, EMBED: Int, SCALE: Int](Module):
    comptime ARITY: Int = 1
    comptime OG: Int = Self.GRID // Self.SCALE
    comptime OUT_TOKENS: Int = Self.OG * Self.OG
    comptime OUT_CHAN: Int = Self.EMBED * Self.SCALE * Self.SCALE
    comptime N: Int = Self.GRID * Self.GRID * Self.EMBED
    comptime IN_DIMS = Array[Int, 1](fill=Self.N)
    comptime OUT_DIM: Int = Self.N

    def __init__(out self):
        comptime assert Self.GRID % Self.SCALE == 0, (
            "PixelShuffle: GRID must be divisible by SCALE"
        )

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
            out.ensure(B * Self.N)
            var xp = x.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            for b in range(B):
                for h in range(Self.GRID):
                    for w in range(Self.GRID):
                        var t = (h // Self.SCALE) * Self.OG + (w // Self.SCALE)
                        var cb = (h % Self.SCALE) * (
                            Self.SCALE * Self.EMBED
                        ) + (w % Self.SCALE) * Self.EMBED
                        var ib = b * Self.N + (h * Self.GRID + w) * Self.EMBED
                        var ob = b * Self.N + t * Self.OUT_CHAN + cb
                        for e in range(Self.EMBED):
                            op[unsafe_offset = ob + e] = xp[
                                unsafe_offset = ib + e
                            ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.N)
            comptime n_blocks = (B * Self.N + TPB - 1) // TPB
            c.enqueue_function[
                _ps_fwd_kernel[
                    B, Self.GRID, Self.EMBED, Self.SCALE, Self.N
                ]
            ](
                x.tt["gpu", B, Self.N](),
                out.tt["gpu", B, Self.N](),
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
            gin.ensure(B * Self.N)
            var gp = grad_output.data.unsafe_ptr()
            var ip = gin.data.unsafe_ptr()
            for b in range(B):
                for h in range(Self.GRID):
                    for w in range(Self.GRID):
                        var t = (h // Self.SCALE) * Self.OG + (w // Self.SCALE)
                        var cb = (h % Self.SCALE) * (
                            Self.SCALE * Self.EMBED
                        ) + (w % Self.SCALE) * Self.EMBED
                        var ib = b * Self.N + (h * Self.GRID + w) * Self.EMBED
                        var ob = b * Self.N + t * Self.OUT_CHAN + cb
                        for e in range(Self.EMBED):
                            ip[unsafe_offset = ib + e] = gp[
                                unsafe_offset = ob + e
                            ]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.N)
            comptime n_blocks = (B * Self.N + TPB - 1) // TPB
            c.enqueue_function[
                _ps_bwd_kernel[
                    B, Self.GRID, Self.EMBED, Self.SCALE, Self.N
                ]
            ](
                grad_output.tt["gpu", B, Self.N](),
                gin.tt["gpu", B, Self.N](),
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
