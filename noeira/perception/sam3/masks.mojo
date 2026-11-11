# +--------------------------------------------------------------------------+ #
# | SAM 3 — the mask decoder (pixel decoder, 200 masks, semantic map)
# +--------------------------------------------------------------------------+ #
"""`Sam3MaskDecoder`: `Sam3MaskDecoder` from `detector_model.mask_decoder.*`.

    var md = Sam3MaskDecoder(ctx, sam3_checkpoint())
    var m = md.forward(queries, fpn288, fpn144, fused, text, text_len)
    # m.masks [200, 288 x 288] logits, m.semantic [288 x 288] (device, token-major)

1. `fused' = fused + prompt_cross_attn(LN(fused), text, text)` (the padding
   masked; LayerNorm eps 1e-5).
2. The pixel decoder from `fused'` (the 72 x 72 level) up: nearest x2, + the
   neck's 144 level, 3x3 conv, GroupNorm(8), ReLU; again with the 288 level.
   (`conv_layers.2` / `norms.2` exist in the checkpoint and are never used: the
   forward iterates over two finer levels.)
3. `masks = mask_embedder(queries) @ instance_projection(pixels)^T` (a 3-layer
   ReLU MLP; a 1x1 conv), `semantic = semantic_projection(pixels)` (1 channel:
   a dot product per pixel, not a GEMM — `max_matmul` is wrong at N = 1 off
   NVIDIA).

Token-major throughout, like the neck.
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.primitives import block, warp

from noeira.nn.constants import DT
from .ops import (
    Sam3Weights, TPB, Ptr, blocks, linear, layer_norm, mha, gemm_tb, add_kernel, relu_kernel, _d,
)
from .neck import conv3x3, conv3x3_taps


comptime MASK_PREFIX = "detector_model.mask_decoder."
comptime C: Int = 256
comptime NHM: Int = 8
comptime HDM: Int = 32
comptime GROUPS: Int = 8
comptime EPS: Float32 = 1e-5
comptime T_LEN: Int = 32
comptime NQ: Int = 200


@fieldwise_init
struct Sam3Masks(Movable):
    var masks: DeviceBuffer[DT]
    """[200, h w] mask logits at the finest level (288 x 288 for 1008 input)."""
    var semantic: DeviceBuffer[DT]
    """[h w] semantic logits."""
    var h: Int
    var w: Int


def upsample2_add_kernel(coarse: Ptr, fine: Ptr, dst: Ptr, wf_n: Int64):
    """`dst[r, c] = coarse[r // 2, c // 2] + fine[r, c]` (nearest x2, token-major,
    C channels). Grid `(H W, ceil(C / TPB))` of the fine grid (positions on x:
    CUDA caps grid y at 65 535)."""
    var ch = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.x)
    if ch >= C:
        return
    var wf = Int(wf_n)
    var r = _d(n, wf)
    var c = n - r * wf
    var m = (r // 2) * (wf // 2) + c // 2
    dst[unsafe_offset=n * C + ch] = coarse[unsafe_offset=m * C + ch] + fine[unsafe_offset=n * C + ch]


def group_stats_kernel(x: Ptr, stats: Ptr, n_n: Int64):
    """Per group g (block g): mean and 1 / sqrt(var + eps) over its 32 channels
    and every position — two passes. Grid `GROUPS`."""
    var g = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = Int(n_n)
    comptime GC = C // GROUPS
    var total = n * GC
    var s = Scalar[DT](0)
    var i = tid
    while i < total:
        var p = i // GC
        s += x[unsafe_offset=p * C + g * GC + (i - p * GC)]
        i += TPB
    var mean = block.sum[block_size=TPB, broadcast=True](val=s) / Scalar[DT](total)
    var q = Scalar[DT](0)
    i = tid
    while i < total:
        var p = i // GC
        var d = x[unsafe_offset=p * C + g * GC + (i - p * GC)] - mean
        q += d * d
        i += TPB
    var var_ = block.sum[block_size=TPB, broadcast=True](val=q) / Scalar[DT](total)
    if tid == 0:
        stats[unsafe_offset=2 * g] = mean
        stats[unsafe_offset=2 * g + 1] = Scalar[DT](1) / sqrt(var_ + Scalar[DT](EPS))


def group_norm_relu_kernel(x: Ptr, stats: Ptr, w: Ptr, b: Ptr, n_n: Int64):
    """`relu((x - mean_g) inv_g w + b)` in place. Grid `ceil(N C / TPB)`."""
    var i = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if i >= Int(n_n) * C:
        return
    var ch = i % C
    var g = ch // (C // GROUPS)
    var v = (x[unsafe_offset=i] - stats[unsafe_offset=2 * g]) * stats[unsafe_offset=2 * g + 1] * w[unsafe_offset=ch] + b[unsafe_offset=ch]
    x[unsafe_offset=i] = max(v, Scalar[DT](0))


def channel_dot_kernel(x: Ptr, w: Ptr, b: Ptr, dst: Ptr, n_n: Int64):
    """`dst[n] = x[n, :] . w + b[0]` — a 1-channel 1x1 conv. A warp per row."""
    var tid = Int(thread_idx.x)
    var n = Int(block_idx.x) * 8 + (tid >> 5)
    var lane = tid & 31
    var s = Scalar[DT](0)
    if n < Int(n_n):
        var ch = lane
        while ch < C:
            s += x[unsafe_offset=n * C + ch] * w[unsafe_offset=ch]
            ch += 32
    s = warp.sum(s)
    if lane == 0 and n < Int(n_n):
        dst[unsafe_offset=n] = s + b[unsafe_offset=0]


struct Sam3MaskDecoder(Movable):
    var wts: Sam3Weights
    var taps: List[DeviceBuffer[DT]]
    var debug: Int
    """Falsification only: bit 1, no prompt cross-attention; bit 2, the
    GroupNorm as a LayerNorm per position (each pixel normalised alone)."""

    def __init__(out self, var ctx: DeviceContext, path: String) raises:
        self.wts = Sam3Weights(ctx^, path, MASK_PREFIX)
        self.taps = List[DeviceBuffer[DT]]()
        for i in range(2):
            self.taps.append(conv3x3_taps(self.wts.ctx, self.wts.w(String("pixel_decoder.conv_layers.") + String(i) + ".weight")))
        self.debug = 0

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def forward(
        mut self, queries: DeviceBuffer[DT], fpn288: DeviceBuffer[DT], fpn144: DeviceBuffer[DT], fused: DeviceBuffer[DT],
        text: DeviceBuffer[DT], text_len: Int, h0: Int = 288, w0: Int = 288,
    ) raises -> Sam3Masks:
        """`queries` [200, 256]; the neck's levels token-major (288 then 144);
        `fused` the DETR encoder's [72 x 72, 256]; `text` [32, 256]."""
        ref ctx = self.wts.ctx
        var h2 = h0 // 4
        var w2 = w0 // 4
        var n2 = h2 * w2
        # 1. the prompt cross-attention
        var enc = self.new(n2 * C)
        ctx.enqueue_copy(enc, fused.create_sub_buffer[DT](0, n2 * C))
        if (self.debug & 1) == 0:
            var ln = self.new(n2 * C)
            layer_norm(self.wts, "prompt_cross_attn_norm", fused, ln, n2, C, EPS)
            var q = linear(self.wts, "prompt_cross_attn.q_proj", ln, n2, C)
            var k = linear(self.wts, "prompt_cross_attn.k_proj", text, T_LEN, C)
            var v = linear(self.wts, "prompt_cross_attn.v_proj", text, T_LEN, C)
            var a = mha(ctx, q, k, v, n2, T_LEN, NHM, HDM, text_len)
            var o = linear(self.wts, "prompt_cross_attn.o_proj", a, n2, C)
            ctx.enqueue_function[add_kernel](enc, o, Int64(n2 * C), grid_dim=blocks(n2 * C), block_dim=TPB)
        # 2. the pixel decoder: 72 -> 144 -> 288
        var prev = enc
        var hh = h2
        var ww = w2
        for stage in range(2):
            var fine = fpn144 if stage == 0 else fpn288
            hh *= 2
            ww *= 2
            var n = hh * ww
            var s = self.new(n * C)
            ctx.enqueue_function[upsample2_add_kernel](prev, fine, s, Int64(ww), grid_dim=(n, blocks(C)), block_dim=TPB)
            var cv = conv3x3(ctx, s, hh, ww, self.taps[stage], self.wts.w(String("pixel_decoder.conv_layers.") + String(stage) + ".bias"))
            var nw = self.wts.w(String("pixel_decoder.norms.") + String(stage) + ".weight")
            var nb = self.wts.w(String("pixel_decoder.norms.") + String(stage) + ".bias")
            if (self.debug & 2) != 0:
                var y = self.new(n * C)
                layer_norm(self.wts, String("pixel_decoder.norms.") + String(stage), cv, y, n, C, EPS)
                ctx.enqueue_function[relu_kernel](y, Int64(n * C), grid_dim=blocks(n * C), block_dim=TPB)
                prev = y
            else:
                var stats = self.new(2 * GROUPS)
                ctx.enqueue_function[group_stats_kernel](cv, stats, Int64(n), grid_dim=GROUPS, block_dim=TPB)
                ctx.enqueue_function[group_norm_relu_kernel](cv, stats, nw, nb, Int64(n), grid_dim=blocks(n * C), block_dim=TPB)
                prev = cv
        var npx = hh * ww
        # 3. masks and the semantic map
        var inst = linear(self.wts, "instance_projection", prev, npx, C)
        var e1 = linear(self.wts, "mask_embedder.layers.0", queries, NQ, C)
        ctx.enqueue_function[relu_kernel](e1, Int64(NQ * C), grid_dim=blocks(NQ * C), block_dim=TPB)
        var e2 = linear(self.wts, "mask_embedder.layers.1", e1, NQ, C)
        ctx.enqueue_function[relu_kernel](e2, Int64(NQ * C), grid_dim=blocks(NQ * C), block_dim=TPB)
        var emb = linear(self.wts, "mask_embedder.layers.2", e2, NQ, C)
        var masks = self.new(NQ * npx)
        gemm_tb(ctx, masks, emb, inst, NQ, npx, C)
        var sem = self.new(npx)
        ctx.enqueue_function[channel_dot_kernel](
            prev, self.wts.w("semantic_projection.weight"), self.wts.w("semantic_projection.bias"), sem, Int64(npx),
            grid_dim=(npx + 7) // 8, block_dim=256,
        )
        return Sam3Masks(masks^, sem^, hh, ww)
