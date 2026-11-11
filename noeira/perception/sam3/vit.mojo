# +--------------------------------------------------------------------------+ #
# | SAM 3 — the vision backbone (ViT-L/14 at 1008, windowed + global RoPE)
# +--------------------------------------------------------------------------+ #
"""`Sam3ViT`: SAM 3's image encoder, from the checkpoint's
`detector_model.vision_encoder.backbone.*`, verified against transformers'
`Sam3ViTModel` (`tools/perception/dump_sam3_reference.py`).

    var vit = Sam3ViT(ctx, sam3_checkpoint())
    var feats = vit.forward(pixel_values, taps)          # [5184, 1024]: the 72x72 grid, row-major

## The forward (`Sam3ViTModel`)

`pixel_values` [3, 1008, 1008] -> 14x14 patches (a stride-14 conv, no bias:
a GEMM on the patches, `c ky kx` order) -> + the 24x24 pretraining position
embedding TILED 3x3 (not interpolated) -> `layer_norm` (BEFORE the layers) ->
32 pre-norm layers:

    x = x + o_proj( attn( rope( qkv( LN1(x) ) ) ) ),   x = x + fc2( gelu( fc1( LN2(x) ) ) )

Layers 7, 15, 23, 31 attend over the whole grid; the others inside 24x24
windows (9 of 576 tokens, row-major). 16 heads x 64, biases everywhere, exact
GELU, LayerNorm eps 1e-6. No norm after the last layer.

## RoPE (`Sam3ViTRotaryEmbedding`, `rotate_pairwise`)

Axial, PAIRWISE (channels 2i and 2i+1 turn together — not rotate-half): 16
frequencies `1 / 1e4^(2i / 32)`; pair i < 16 turns by `col * s * f_i`, pair
16 + i by `row * s * f_i` (the COLUMN first: transformers stacks (w, h)). In a
window `row, col` are the window's own (0..23) and `s = 1`; on the global layers
they are the grid's (0..71) and `s = 24 / 72` — fractional positions.

## Memory

A global layer's scores are `16 x 5184^2` floats (1.7 GB): they are taken ONE
HEAD at a time (107 MB). A window's are `16 x 576^2` (21 MB).

## `debug` (falsification only)

Bit 1: the RoPE's axes swapped. Bit 2: every layer windowed (no global
attention). Bit 4: the position embedding stretched over the grid instead of
tiled.
"""

from std.math import cos, sin, pow
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.llm.lfm3.model import attention_probs, attention_apply, split_heads
from .ops import (
    Sam3Weights, TPB, Ptr, blocks, linear, layer_norm, gemm_tb, add_kernel, gelu_kernel, rope_pairs_kernel,
    put_head_kernel, mha, linear_into, flash_self, USE_FLASH, bias_add, mlp_residual,
    SAM3_BF16, linear_into_b, flash_self_bf16, qkv_rope_bf16_kernel, layer_norm_bf16, bias_add_perm, mlp_residual_b,
)


comptime VIT_PREFIX = "detector_model.vision_encoder.backbone."
comptime IMG: Int = 1008
comptime PATCH: Int = 14
comptime GRID: Int = IMG // PATCH
"""72."""
comptime NTOK: Int = GRID * GRID
"""5184."""
comptime DIM: Int = 1024
comptime HEADS: Int = 16
comptime HD: Int = 64
comptime FF: Int = 4736
comptime LAYERS: Int = 32
comptime WIN: Int = 24
comptime NWIN: Int = (GRID // WIN) * (GRID // WIN)
comptime PRE: Int = 24
"""The pretraining grid (336 / 14)."""
comptime EPS: Float32 = 1e-6
comptime THETA: Float64 = 1e4


def is_global(li: Int) -> Bool:
    return li == 7 or li == 15 or li == 23 or li == 31


def patchify_kernel(px: Ptr, dst: Ptr, debug: Int64):
    """`dst[n, c 196 + ky 14 + kx] = px[c, 14 r + ky, 14 col + kx]`, n = r 72 + col.
    Grid `(ceil(588 / TPB), NTOK)`."""
    var e = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    if e >= 3 * PATCH * PATCH:
        return
    var c = e // (PATCH * PATCH)
    var rem = e - c * PATCH * PATCH
    var ky = rem // PATCH
    var kx = rem - ky * PATCH
    var r = n // GRID
    var col = n - r * GRID
    dst[unsafe_offset=n * 3 * PATCH * PATCH + e] = px[unsafe_offset=(c * IMG + r * PATCH + ky) * IMG + col * PATCH + kx]


def pos_tile_kernel(x: Ptr, pos: Ptr, debug: Int64):
    """`x[n, :] += pos[(r % 24) 24 + (col % 24), :]` — the pretraining grid TILED.
    `debug` 4: stretched (`r 24 / 72`) instead. Grid `(ceil(D / TPB), NTOK)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    if d >= DIM:
        return
    var r = n // GRID
    var col = n - r * GRID
    var pr = r % PRE
    var pc = col % PRE
    if debug == 4:
        pr = r * PRE // GRID
        pc = col * PRE // GRID
    x[unsafe_offset=n * DIM + d] = x[unsafe_offset=n * DIM + d] + pos[unsafe_offset=(pr * PRE + pc) * DIM + d]


def window_kernel(src: Ptr, dst: Ptr, inverse: Int64):
    """Raster tokens <-> window order (window (wr, wc) row-major, then its 24x24
    tokens row-major). `inverse` 0: dst = windows of src; 1: dst = raster of src
    (windowed). Grid `(ceil(D / TPB), NTOK)`, one row per block row (the WINDOW
    index on the windowed side)."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var m = Int(block_idx.y)
    if d >= DIM:
        return
    var w = m // (WIN * WIN)
    var i = m - w * WIN * WIN
    var wr = w // (GRID // WIN)
    var wc = w - wr * (GRID // WIN)
    var r = wr * WIN + i // WIN
    var col = wc * WIN + (i % WIN)
    var n = r * GRID + col
    if inverse == 0:
        dst[unsafe_offset=m * DIM + d] = src[unsafe_offset=n * DIM + d]
    else:
        dst[unsafe_offset=n * DIM + d] = src[unsafe_offset=m * DIM + d]


struct Sam3ViT(Movable):
    var wts: Sam3Weights
    var rope_win: Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]
    var rope_glob: Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]
    var dummy: DeviceBuffer[DType.int32]
    var debug: Int

    def __init__(out self, var ctx: DeviceContext, path: String, debug: Int = 0) raises:
        self.wts = Sam3Weights(ctx^, path, VIT_PREFIX)
        self.debug = debug
        self.rope_win = _rope_table(self.wts.ctx, True, debug)
        self.rope_glob = _rope_table(self.wts.ctx, False, debug)
        self.dummy = self.wts.ctx.enqueue_create_buffer[DType.int32](1)
        comptime if USE_FLASH:
            for li in range(LAYERS):
                var p = String("layers.") + String(li) + ".attention."
                var parts = List[String]()
                parts.append(p + "q_proj")
                parts.append(p + "k_proj")
                parts.append(p + "v_proj")
                self.wts.concat(p + "qkv_proj", parts)

    def set_debug(mut self, debug: Int) raises:
        self.debug = debug
        self.rope_win = _rope_table(self.wts.ctx, True, debug)
        self.rope_glob = _rope_table(self.wts.ctx, False, debug)

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def forward(
        mut self, pixel_values: DeviceBuffer[DT], mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False,
    ) raises -> DeviceBuffer[DT]:
        """`pixel_values` [3, 1008, 1008] -> [5184, 1024]. `taps` (with
        `keep_taps`): the embedding, the pre-norm, then every layer's output."""
        ref ctx = self.wts.ctx
        var patches = self.new(NTOK * 3 * PATCH * PATCH)
        ctx.enqueue_function[patchify_kernel](
            pixel_values, patches, Int64(self.debug), grid_dim=(blocks(3 * PATCH * PATCH), NTOK), block_dim=TPB,
        )
        var x = linear(self.wts, "embeddings.patch_embeddings.projection", patches, NTOK, 3 * PATCH * PATCH, bias=False)
        ctx.enqueue_function[pos_tile_kernel](
            x, self.wts.w("embeddings.position_embeddings"), Int64(self.debug & 4),
            grid_dim=(blocks(DIM), NTOK), block_dim=TPB,
        )
        if keep_taps:
            taps.append(_copy(ctx, x, NTOK * DIM))
        layer_norm(self.wts, "layer_norm", x, x, NTOK, DIM, EPS)
        if keep_taps:
            taps.append(_copy(ctx, x, NTOK * DIM))
        var sc_win = self.new(HEADS * WIN * WIN * WIN * WIN)
        for li in range(LAYERS):
            var p = String("layers.") + String(li) + "."
            var glob = is_global(li) and (self.debug & 2) == 0
            var hw: DeviceBuffer[DT]
            var hb: DeviceBuffer[DType.bfloat16]
            comptime if SAM3_BF16:
                # bf16: LN1 written as the GEMM's bf16 operand, in window order (no permutation pass)
                hw = self.new(1)
                hb = ctx.enqueue_create_buffer[DType.bfloat16](NTOK * DIM)
                layer_norm_bf16(self.wts, p + "layer_norm1", x, hb, NTOK, DIM, EPS, GRID, 0 if glob else WIN)
            else:
                hb = ctx.enqueue_create_buffer[DType.bfloat16](1)
                var h = self.new(NTOK * DIM)
                layer_norm(self.wts, p + "layer_norm1", x, h, NTOK, DIM, EPS)
                if glob:
                    hw = h
                else:
                    hw = self.new(NTOK * DIM)
                    ctx.enqueue_function[window_kernel](h, hw, Int64(0), grid_dim=(blocks(DIM), NTOK), block_dim=TPB)
            var c_t = self.rope_glob[0] if glob else self.rope_win[0]
            var s_t = self.rope_glob[1] if glob else self.rope_win[1]
            var a = self.new(NTOK * DIM)
            var o = self.new(NTOK * DIM)
            comptime if SAM3_BF16:
                # bf16: the q | k | v GEMM without bias, then bias + RoPE + bf16 in one pass,
                # the bf16 fused attention, its bf16 output straight into o_proj's GEMM
                var qkv = self.new(3 * NTOK * DIM)
                linear_into_b(self.wts, p + "attention.qkv_proj", hb, NTOK, DIM, qkv, bias=False)
                var qb = ctx.enqueue_create_buffer[DType.bfloat16](3 * NTOK * DIM)
                ctx.enqueue_function[qkv_rope_bf16_kernel](
                    qkv, self.wts.w(p + "attention.qkv_proj.bias"), c_t, s_t, qb, Int64(HEADS), Int64(HD // 2),
                    grid_dim=(blocks(3 * DIM // 2), NTOK), block_dim=TPB,
                )
                var ab = ctx.enqueue_create_buffer[DType.bfloat16](NTOK * DIM)
                if glob:
                    flash_self_bf16[NTOK, HEADS, HD, True](ctx, qb, ab)
                else:
                    flash_self_bf16[WIN * WIN, HEADS, HD, True](ctx, qb, ab, NWIN)
                linear_into_b(self.wts, p + "attention.o_proj", ab, NTOK, DIM, o, bias=False)
            elif USE_FLASH:
                # one GEMM: rows [q | k | v] (the fused kernel's interleaved input); windows are
                # 576 consecutive rows already (hw is in window order), nine samples
                var qkv = linear(self.wts, p + "attention.qkv_proj", hw, NTOK, DIM)
                ctx.enqueue_function[rope_pairs_kernel](
                    qkv, c_t, s_t, Int64(HEADS), Int64(HD // 2), Int64(3 * DIM),
                    grid_dim=(blocks(HEADS * HD // 2), NTOK), block_dim=TPB,
                )
                ctx.enqueue_function[rope_pairs_kernel](
                    qkv.create_sub_buffer[DT](DIM, 3 * NTOK * DIM - DIM), c_t, s_t, Int64(HEADS), Int64(HD // 2),
                    Int64(3 * DIM), grid_dim=(blocks(HEADS * HD // 2), NTOK), block_dim=TPB,
                )
                if glob:
                    flash_self[NTOK, HEADS, HD, True](ctx, qkv, a)
                else:
                    flash_self[WIN * WIN, HEADS, HD, True](ctx, qkv, a, NWIN)
                linear_into(self.wts, p + "attention.o_proj", a, NTOK, DIM, o, bias=False)
            else:
                var q = linear(self.wts, p + "attention.q_proj", hw, NTOK, DIM)
                var k = linear(self.wts, p + "attention.k_proj", hw, NTOK, DIM)
                var v = linear(self.wts, p + "attention.v_proj", hw, NTOK, DIM)
                ctx.enqueue_function[rope_pairs_kernel](
                    q, c_t, s_t, Int64(HEADS), Int64(HD // 2), Int64(DIM),
                    grid_dim=(blocks(HEADS * HD // 2), NTOK), block_dim=TPB,
                )
                ctx.enqueue_function[rope_pairs_kernel](
                    k, c_t, s_t, Int64(HEADS), Int64(HD // 2), Int64(DIM),
                    grid_dim=(blocks(HEADS * HD // 2), NTOK), block_dim=TPB,
                )
                if glob:
                    # one head at a time (`ops.mha`): all sixteen would hold 1.7 GB of scores
                    a = mha(ctx, q, k, v, NTOK, NTOK, HEADS, HD, NTOK)
                else:
                    var wt = WIN * WIN
                    for w in range(NWIN):
                        var off = w * wt * DIM
                        attention_probs(
                            ctx, q.create_sub_buffer[DT](off, wt * DIM), k.create_sub_buffer[DT](off, wt * DIM),
                            sc_win, self.dummy, wt, HEADS, HEADS, HD, 3, 0, 0,
                        )
                        var aw = a.create_sub_buffer[DT](off, wt * DIM)
                        attention_apply(ctx, sc_win, v.create_sub_buffer[DT](off, wt * DIM), aw, False, aw, wt, HEADS, HEADS, HD)
                linear_into(self.wts, p + "attention.o_proj", a, NTOK, DIM, o, bias=False)
            # o_proj's bias folded into the residual add (a bias is per column: the window
            # permutation commutes with it)
            comptime if SAM3_BF16:
                # the residual add reads the window-order result through the same map
                bias_add_perm(self.wts, p + "attention.o_proj", x, o, NTOK, DIM, GRID, 0 if glob else WIN)
                var h2b = ctx.enqueue_create_buffer[DType.bfloat16](NTOK * DIM)
                layer_norm_bf16(self.wts, p + "layer_norm2", x, h2b, NTOK, DIM, EPS)
                mlp_residual_b[1](self.wts, p + "mlp.fc1", p + "mlp.fc2", h2b, NTOK, DIM, FF, x)
            else:
                if not glob:
                    var orr = self.new(NTOK * DIM)
                    ctx.enqueue_function[window_kernel](o, orr, Int64(1), grid_dim=(blocks(DIM), NTOK), block_dim=TPB)
                    o = orr
                bias_add(self.wts, p + "attention.o_proj", x, o, NTOK, DIM)
                var h2 = self.new(NTOK * DIM)
                layer_norm(self.wts, p + "layer_norm2", x, h2, NTOK, DIM, EPS)
                mlp_residual[1](self.wts, p + "mlp.fc1", p + "mlp.fc2", h2, NTOK, DIM, FF, x)
            if keep_taps:
                taps.append(_copy(ctx, x, NTOK * DIM))
        return x^


def _rope_table(ctx: DeviceContext, windowed: Bool, debug: Int) raises -> Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]:
    """cos / sin `[NTOK, 32]` in the order the tokens are in at the attention
    (window order when `windowed`, raster otherwise); module note."""
    comptime PAIRS = HD // 2
    comptime Q = PAIRS // 2
    var inv = List[Float32](length=Q, fill=0.0)
    for i in range(Q):
        inv[i] = Float32(1.0) / Float32(pow(THETA, Float64(Float32(2 * i) / Float32(PAIRS))))
    var scale = Float32(1.0) if windowed else Float32(WIN) / Float32(GRID)
    var hc = ctx.enqueue_create_host_buffer[DT](NTOK * PAIRS)
    var hs = ctx.enqueue_create_host_buffer[DT](NTOK * PAIRS)
    for m in range(NTOK):
        var r: Int
        var col: Int
        if windowed:
            var i = m % (WIN * WIN)
            r = i // WIN
            col = i % WIN
        else:
            r = m // GRID
            col = m % GRID
        if (debug & 1) != 0:
            var t = r
            r = col
            col = t
        var pc = Float32(col) * scale
        var pr = Float32(r) * scale
        for i in range(PAIRS):
            var a = (pc if i < Q else pr) * inv[i if i < Q else i - Q]
            hc[m * PAIRS + i] = Float32(cos(Float64(a)))
            hs[m * PAIRS + i] = Float32(sin(Float64(a)))
    var dc = ctx.enqueue_create_buffer[DT](NTOK * PAIRS)
    var ds = ctx.enqueue_create_buffer[DT](NTOK * PAIRS)
    ctx.enqueue_copy(dc, hc)
    ctx.enqueue_copy(ds, hs)
    return (dc^, ds^)


def _copy(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DT]:
    var b = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(b, x.create_sub_buffer[DT](0, n))
    return b^
