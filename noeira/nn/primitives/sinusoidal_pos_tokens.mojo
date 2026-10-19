"""Fixed sinusoidal position tokens as a graph VALUE.

    SinusoidalPos1DTokens[IGNORE_DIM, SEQ, DIM]     ACT's CVAE-encoder table
    SinusoidalPos2DTokens[IGNORE_DIM, DIM, OH, OW]  DETR's image-feature table

Both are `LearnedQueries` with the parameter replaced by a constant: one
`N*D` table, the same for every row of the batch, produced from a carrier input
that is read for its BATCH count and nothing else (`grad_input = 0`, no params).

**Why a value and not an add.** In DETR the positional embedding is not folded
into the token stream once — it is carried alongside and added to q and k, but
not v, inside EVERY layer (`transformer.py:with_pos_embed`). So `pos` has to
exist as a tensor the graph can route to each layer's conditioning input, which
an additive `x + pos` node cannot provide.

## The 1-D table (`detr_vae.py:23 get_sinusoid_encoding_table`)

    angle[t, j] = t / 10000^(2*(j // 2) / d_hid)
    table[t, 2i] = sin(angle),  table[t, 2i+1] = cos(angle)

⚠ NOT `SinusoidalPosAdd[T, S, D]` at `S = 1`: that primitive adds SEPARABLE
time + space positions, contributing `sinusoid(0)` on top — which is 0 on even
indices and **1 on every odd index**, a constant offset ACT's table lacks.

## The 2-D table (`position_encoding.py:PositionEmbeddingSine`, normalize=True)

`num_pos_feats = DIM // 2`, half the channels encoding the row and half the
column. Both axes are `cumsum`ed from 1, divided by the last value, and scaled
by `2*pi`:

    y[h] = (h + 1) / (OH + 1e-6) * 2*pi        x[w] = (w + 1) / (OW + 1e-6) * 2*pi
    dim_t[k] = 10000^(2*(k // 2) / num_pos_feats)
    pos_y[h, k] = sin(y/dim_t[k]) if k even else cos(y/dim_t[k])   (same for x)
    out[:, h, w] = concat(pos_y[h], pos_x[w])                       — y FIRST

⚠ Three things here are easy to get wrong and all produce a plausible field:
the **+1 offset** (a `cumsum` starts at 1, not 0), the **y-before-x**
concatenation order (`torch.cat((pos_y, pos_x), dim=3)`), and the `1e-6` in the
denominator (which makes the last row/column *slightly* less than `2*pi`).

Emitted in **token-major** `[OH*OW, DIM]` — the layout the transformer consumes
after `Transpose2D` turns the conv feature map from NCHW into tokens. The
reference produces NCHW and permutes later; doing it once here means no
transpose node on the position path.
"""

from std.math import cos, exp, log, pi, sin
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


comptime SINUSOID_BASE_1D: Float64 = 10000.0
comptime DETR_POS_EPS: Float64 = 1e-6
comptime DETR_POS_SCALE: Float64 = 2.0 * pi


# ── table builders (host, comptime-shaped; exposed so a gate can check the
#    table itself rather than only its effect downstream) ──────────────────


def sinusoid_1d_entry(t: Int, j: Int, d_hid: Int) -> Float64:
    """One entry of ACT's `get_sinusoid_encoding_table`."""
    var k = Float64(j // 2)
    var angle = Float64(t) * exp(
        -(2.0 * k) / Float64(d_hid) * log(SINUSOID_BASE_1D)
    )
    return sin(angle) if j % 2 == 0 else cos(angle)


def detr_pos2d_entry(h: Int, w: Int, j: Int, oh: Int, ow: Int, dim: Int) -> Float64:
    """One entry of `PositionEmbeddingSine(dim//2, normalize=True)`, in
    token-major `[h*OW + w, j]` order. `j < dim/2` is the ROW half."""
    var npf = dim // 2
    var is_y = j < npf
    var k = j if is_y else j - npf
    # cumsum starts at 1; normalized by the last value (+eps) and scaled by 2pi.
    var embed = (
        (Float64(h) + 1.0) / (Float64(oh) + DETR_POS_EPS) * DETR_POS_SCALE
    ) if is_y else (
        (Float64(w) + 1.0) / (Float64(ow) + DETR_POS_EPS) * DETR_POS_SCALE
    )
    var dim_t = exp(
        Float64(2 * (k // 2)) / Float64(npf) * log(SINUSOID_BASE_1D)
    )
    var v = embed / dim_t
    return sin(v) if k % 2 == 0 else cos(v)


# ── shared kernels ───────────────────────────────────────────────────────


def _spt_broadcast_kernel[
    BATCH: Int, OUT_DIM: Int
](
    table: Tile[DT, OUT_DIM],
    output: Tile[DT, BATCH, OUT_DIM],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_DIM:
        return
    output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        table.ptr[unsafe_offset=idx % OUT_DIM]
    )


def _spt_zero_kernel[
    N: Int
](grad_input: Tile[DT, N]):
    var idx = Int(global_idx.x)
    if idx < N:
        grad_input.ptr[unsafe_offset=idx] = Scalar[DT](0.0)


# ── 1-D (ACT CVAE encoder) ───────────────────────────────────────────────


struct SinusoidalPos1DTokens[IGNORE_DIM: Int, SEQ: Int, DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IGNORE_DIM)
    comptime OUT_DIM = Self.SEQ * Self.DIM

    var table: Tensor

    def __init__(out self):
        self.table = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.table = move.table^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SinusoidalPos1DTokens: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.table = Tensor.alloc(Self.OUT_DIM)
        for t in range(Self.SEQ):
            for j in range(Self.DIM):
                m.table.data[t * Self.DIM + j] = Scalar[DT](
                    sinusoid_1d_entry(t, j, Self.DIM)
                )
        comptime if target != "cpu":
            m.table.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        _spt_forward[target, B, Self.OUT_DIM](self.table, out, ctx)

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
        _spt_zero[target, B * Self.IGNORE_DIM](gin, ctx)


# ── 2-D (DETR image features) ────────────────────────────────────────────


struct SinusoidalPos2DTokens[
    IGNORE_DIM: Int, DIM: Int, OH: Int, OW: Int, N_REPEAT: Int = 1
](Module):
    """`N_REPEAT` tiles the SAME `OH*OW` table back to back.

    ⚠ That tiling is the reference's behaviour, not a shortcut. ACT concatenates
    its cameras along the feature map's WIDTH (`src = torch.cat(all_cam_features,
    axis=3)`) and concatenates their position embeddings the same way — but each
    camera's embedding was computed independently on its own `OH x OW` map, so
    they are IDENTICAL. Camera 0's token (h, w) and camera 1's token (h, w)
    therefore receive the same positional vector, and the model cannot tell the
    cameras apart by position. Attention is permutation-equivariant over the
    memory as long as each token keeps its own embedding, so our per-camera token
    ORDER differs from the reference's interleaved-by-row order and the function
    is unchanged.
    """

    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IGNORE_DIM)
    comptime OUT_DIM = Self.N_REPEAT * Self.OH * Self.OW * Self.DIM

    var table: Tensor

    def __init__(out self):
        comptime assert Self.DIM % 2 == 0, (
            "SinusoidalPos2DTokens: DIM must be even (half row, half column)"
        )
        self.table = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.table = move.table^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SinusoidalPos2DTokens: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.table = Tensor.alloc(Self.OUT_DIM)
        comptime BLOCK = Self.OH * Self.OW * Self.DIM
        for h in range(Self.OH):
            for w in range(Self.OW):
                var tok = h * Self.OW + w
                for j in range(Self.DIM):
                    var v = Scalar[DT](
                        detr_pos2d_entry(h, w, j, Self.OH, Self.OW, Self.DIM)
                    )
                    for r in range(Self.N_REPEAT):
                        m.table.data[r * BLOCK + tok * Self.DIM + j] = v
        comptime if target != "cpu":
            m.table.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        _spt_forward[target, B, Self.OUT_DIM](self.table, out, ctx)

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
        _spt_zero[target, B * Self.IGNORE_DIM](gin, ctx)


# ── shared bodies ────────────────────────────────────────────────────────


def _spt_forward[
    target: StaticString, B: Int, OUT_DIM: Int
](mut table: Tensor, mut out: Tensor, ctx: Optional[DeviceContext]) raises:
    comptime if target == "cpu":
        out.ensure(B * OUT_DIM)
        var o_v = TileTensor(out.data, row_major[B, OUT_DIM]())
        var t_v = TileTensor(table.data, row_major[OUT_DIM]())
        for b in range(B):
            for i in range(OUT_DIM):
                o_v[b, i] = t_v[i]
    else:
        var c = ctx.value()
        out.ensure_gpu(c, B * OUT_DIM)
        c.enqueue_function[_spt_broadcast_kernel[B, OUT_DIM]](
            table.tt["gpu", OUT_DIM](),
            out.tt["gpu", B, OUT_DIM](),
            grid_dim=(B * OUT_DIM + TPB - 1) // TPB,
            block_dim=TPB,
        )


def _spt_zero[
    target: StaticString, N: Int
](mut gin: Tensor, ctx: Optional[DeviceContext]) raises:
    """The table is constant, so the carrier input's gradient is exactly zero.
    Written rather than left alone: the caller's grad slot is reused across
    nodes, and an accumulating graph would add whatever the previous node put
    there."""
    comptime if target == "cpu":
        gin.ensure(N)
        for k in range(N):
            gin.data[k] = Scalar[DT](0.0)
    else:
        var c = ctx.value()
        gin.ensure_gpu(c, N)
        c.enqueue_function[_spt_zero_kernel[N]](
            gin.tt["gpu", N](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )


# ══════════════════════════════════════════════════════════════════════════
# ZeroTokens
# ══════════════════════════════════════════════════════════════════════════


struct ZeroTokens[IGNORE_DIM: Int, N: Int, D: Int](Module):
    """`torch.zeros_like(query_embed)` — the DETR decoder's initial target.

    A constant zero block of `N*D`, batch-broadcast from a carrier input that is
    read for its BATCH count only (`grad_input = 0`, no params).

    Spelled as its own primitive rather than as a `Scale` node with the
    multiplier left at 0: a `Scale` would be one forgotten
    `set_node_attr["tgt0", "multiplier"](0.0)` away from starting the decoder
    from the query embedding instead of from zero — which trains, converges to
    something, and is not ACT.
    """

    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IGNORE_DIM)
    comptime OUT_DIM = Self.N * Self.D

    def __init__(out self):
        comptime assert Self.N > 0 and Self.D > 0, (
            "ZeroTokens: N, D must be > 0"
        )

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ZeroTokens: target must be 'cpu' or 'gpu'"
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
        _spt_zero[target, B * Self.OUT_DIM](out, ctx)

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
        _spt_zero[target, B * Self.IGNORE_DIM](gin, ctx)
