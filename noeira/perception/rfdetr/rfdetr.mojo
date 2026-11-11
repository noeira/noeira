# +--------------------------------------------------------------------------+ #
# | RF-DETR-Seg-Small — a picture in, COCO instances (box, mask) out
# +--------------------------------------------------------------------------+ #
"""`RfDetr`: what Roboflow's `RFDETRSegSmall().predict(picture, threshold)` does
(`rfdetr` 1.12, Apache-2.0), in Mojo, from its checkpoint as safetensors
(`tools/perception/extract_rfdetr.py`); gated stage by stage against
`tools/perception/dump_rfdetr_reference.py` (tests/perception/test_rfdetr_reference_gpu.mojo).

    var det = RfDetr(DeviceContext())                  # 34 M parameters, fp32
    for inst in det.detect(picture, 0.3):
        inst.label (COCO category id: 72 tv, 82 refrigerator, 67 dining table, 63 couch, 65 bed, ...)
        inst.score; inst.box (x0, y0, x1, y1 pixels); inst.mask (H x W, 0 / 1)

A CLOSED vocabulary (COCO's 80 classes, no door): the permissive real-time
alternative to an open-vocabulary detector (SPATIAL_PERCEPTION_PLAN P3, 11-11);
the words a robot needs beyond COCO come by fine-tuning.

## The forward (Seg-Small: 384 input, patch 12, 2 x 2 windows; token-major)

1. **Pixels**: the RGB24 picture as float / 255, STRETCHED to 384 x 384 (torch
   bilinear, `align_corners=False`, no antialias — the aspect is not kept), then
   `(v - mean) / std` (ImageNet's).
2. **Backbone** (windowed DINOv2-S/12, 384 wide, 6 heads): the 12x12 patch conv
   (a GEMM) -> 32 x 32 tokens; per 16 x 16 window a CLS copy, + the position
   embedding -> [4 x 257, 384]; 12 pre-norm blocks with layer scale, attention
   inside each window except blocks 3, 6, 9 (all 1028 tokens, the four CLS
   copies included), exact GELU, LayerNorm eps 1e-6. The outputs of blocks 2,
   5, 8, 11 go through the final LayerNorm, lose their CLS rows and are put back
   in raster order, side by side: [1024, 1536].
3. **Projector**: C2f (1x1 to 256, split 128 | 128, three 3x3 128 -> 128 pairs
   on the second half WITHOUT a residual, concat 640, 1x1 to 256), each conv
   bias-free and followed by a channel LayerNorm (eps 1e-6) + SiLU; then a
   LayerNorm -> `src` [1024, 256], the decoder's memory.
4. **Two-stage** (group 0): Linear + LayerNorm (1e-5) -> 91 class logits per
   token; the 100 tokens with the highest max logit (descending); a 3-layer MLP
   on them -> deltas on the fixed anchors ((x + 0.5) / 32, (y + 0.5) / 32, 0.05,
   0.05): `xy = d_xy wh_a + xy_a`, `wh = exp(d_wh) wh_a`; the learned
   `refpoint_embed[:100]` composed onto those the same way -> the reference
   boxes; the content queries `query_feat[:100]`.
5. **Decoder** (4 post-norm layers; the references and `query_pos =
   ref_point_head(sine(ref))` fixed — "lite" refinement): self-attention (8 x
   32, q = k = x + pos), deformable cross-attention on the memory (16 heads x 2
   points, `ops.msda_kernel`), FFN 2048 ReLU; each layer's output through the
   decoder's LayerNorm is `hs[l]`.
6. **Heads** on `hs[3]`: 91 logits (index = COCO category id, sigmoid scores);
   boxes = a 3-layer MLP's deltas on the references (same rule).
7. **Masks**: `src` upsampled to 96 x 96 (bilinear), 4 blocks (depthwise 3x3,
   LayerNorm 1e-6, Linear, GELU, + residual), a 1x1 conv; the queries through an
   MLP block (LayerNorm 1e-5, 256 -> 1024 -> 256 GELU, + residual) and a Linear;
   `masks = queries . pixels + bias` -> [100, 96 x 96] logits.
8. **Post-processing** (host): sigmoid over the 100 x 91, kept above the
   threshold in descending order (stable), at most 100; boxes cxcywh -> xyxy in
   the picture's pixels (clamped); masks resized 96 -> the picture (bilinear)
   and cut at logit 0.

`debug` (falsification only): bit 1, every backbone block windowed (no global
attention); bit 2, the deformable offsets' x and y swapped; bit 4, the C2f
bottlenecks WITH a residual.
"""

from std.math import exp, sin, cos, pow
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.ai.picture import Picture
from noeira.nn.constants import DT
from noeira.vision.resize_pad import torch_bilinear_chw
from noeira.perception.sam3.ops import (
    Sam3Weights, TPB, blocks, linear, linear_into, layer_norm, gemm_tb, add_kernel, add_into_kernel,
    gelu_kernel, relu_kernel, bias_kernel, mha,
)
from .ops import (
    IMG, PATCH, GRID, NTOK, NWIN, WTOK, BB_DIM, IPtr,
    patchify_kernel, window_embed_kernel, unwindow_kernel, scale_add_kernel, silu_kernel, copy_cols_kernel,
    gather_rows_kernel, add_scalar_kernel, msda_kernel, dwconv3x3_kernel, upsample_bilinear_kernel,
    conv_taps, conv3x3_tm,
)


comptime EMB = "backbone.0.encoder.encoder.embeddings."
comptime LAYER = "backbone.0.encoder.encoder.encoder.layer."
comptime BB_LN = "backbone.0.encoder.encoder.layernorm"
comptime PROJ = "backbone.0.projector.stages.0."
comptime LAYERS: Int = 12
comptime HEADS: Int = 6
comptime BB_FF: Int = 1536
comptime BB_EPS: Float32 = 1e-6
comptime D: Int = 256
comptime NQ: Int = 100
comptime NCLS: Int = 91
comptime DEC_LAYERS: Int = 4
comptime DEC_FF: Int = 2048
comptime TR_EPS: Float32 = 1e-5
comptime MASK: Int = IMG // 4
"""96: the masks' side."""
comptime C2F_HALF: Int = 128


def is_global(i: Int) -> Bool:
    return i == 3 or i == 6 or i == 9


def is_tap(i: Int) -> Int:
    """The tap index of block `i` (2, 5, 8, 11 -> 0..3), else -1."""
    if i == 2:
        return 0
    if i == 5:
        return 1
    if i == 8:
        return 2
    if i == 11:
        return 3
    return -1


@fieldwise_init
struct RfDetrInstance(Copyable, Movable):
    var label: Int
    """The COCO category id (1 person ... 90 toothbrush)."""
    var score: Float32
    var box: List[Float32]
    """x0, y0, x1, y1 in the picture's pixels."""
    var mask: List[UInt8]
    """[H x W] 0 / 1, the picture's size."""
    var query: Int


struct RfDetrRaw(Movable):
    """The network's outputs before post-processing (host copies)."""
    var logits: List[Float32]
    """[100, 91]."""
    var boxes: List[Float32]
    """[100, 4] cxcywh, normalised."""
    var masks: DeviceBuffer[DT]
    """[100, 96 x 96] logits (device)."""
    var hs: List[DeviceBuffer[DT]]
    """The 4 normalised decoder outputs [100, 256]."""

    def __init__(out self, var logits: List[Float32], var boxes: List[Float32], var masks: DeviceBuffer[DT], var hs: List[DeviceBuffer[DT]]):
        self.logits = logits^
        self.boxes = boxes^
        self.masks = masks^
        self.hs = hs^


struct RfDetr(Movable):
    var wts: Sam3Weights
    var taps: Dict[String, DeviceBuffer[DT]]
    """The C2f 3x3 convs' weights per tap, [9, 128, 128]."""
    var refpoint: List[Float32]
    """`refpoint_embed[:100]` [100, 4] (host)."""
    var debug: Int
    # what the last forward kept, for the stage gates
    var enc_logits: List[Float32]
    var dec_ref: List[Float32]
    var query_pos: DeviceBuffer[DT]
    var dec_layers: List[DeviceBuffer[DT]]

    def __init__(out self, var ctx: DeviceContext, path: String = String(""), debug: Int = 0) raises:
        var p = path if path.byte_length() > 0 else rfdetr_checkpoint()
        self.wts = Sam3Weights(ctx^, p, String(""))
        self.debug = debug
        self.taps = Dict[String, DeviceBuffer[DT]]()
        for k in range(3):
            for cv in range(1, 3):
                var name = PROJ + "0.m." + String(k) + ".cv" + String(cv) + ".conv"
                self.taps[name] = conv_taps(self.wts.ctx, self.wts.w(name + ".weight"), C2F_HALF, C2F_HALF)
        self.refpoint = List[Float32]()
        self.enc_logits = List[Float32]()
        self.dec_ref = List[Float32]()
        self.query_pos = self.wts.ctx.enqueue_create_buffer[DT](1)
        self.dec_layers = List[DeviceBuffer[DT]]()
        self.refpoint = self._down(self.wts.w("refpoint_embed.weight"), NQ * 4)  # (methods need every field set)

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def _down(self, x: DeviceBuffer[DT], n: Int) raises -> List[Float32]:
        var h = self.wts.ctx.enqueue_create_host_buffer[DT](n)
        self.wts.ctx.enqueue_copy(h, x.create_sub_buffer[DT](0, n))
        self.wts.ctx.synchronize()
        var out = List[Float32](capacity=n)
        for i in range(n):
            out.append(h[i])
        return out^

    def _up(self, v: List[Float32]) raises -> DeviceBuffer[DT]:
        var h = self.wts.ctx.enqueue_create_host_buffer[DT](len(v))
        for i in range(len(v)):
            h[i] = v[i]
        var d = self.new(len(v))
        self.wts.ctx.enqueue_copy(d, h)
        return d^

    # ── 1. pixels ─────────────────────────────────────────────────────────

    def pixel_values(self, pic: Picture) raises -> DeviceBuffer[DT]:
        """[3, 384, 384]: the network's input."""
        var chw = List[Float32](length=3 * pic.height * pic.width, fill=0.0)
        var hw = pic.height * pic.width
        for i in range(hw):
            for c in range(3):
                chw[c * hw + i] = Float32(pic.rgb[i * 3 + c]) / Float32(255.0)
        var r = List[Float32]()
        torch_bilinear_chw(chw, 3, pic.height, pic.width, r, IMG, IMG)
        var mean: List[Float32] = [0.485, 0.456, 0.406]
        var std: List[Float32] = [0.229, 0.224, 0.225]
        for c in range(3):
            for i in range(IMG * IMG):
                r[c * IMG * IMG + i] = (r[c * IMG * IMG + i] - mean[c]) / std[c]
        return self._up(r)

    # ── 2-3. backbone + projector ─────────────────────────────────────────

    def _ln_silu(mut self, name: String, x: DeviceBuffer[DT], rows: Int, cols: Int) raises:
        layer_norm(self.wts, name, x, x, rows, cols, BB_EPS)
        self.wts.ctx.enqueue_function[silu_kernel](x, Int64(rows * cols), grid_dim=blocks(rows * cols), block_dim=TPB)

    def backbone(
        mut self, px: DeviceBuffer[DT], mut keep: List[DeviceBuffer[DT]], keep_taps: Bool = False,
    ) raises -> DeviceBuffer[DT]:
        """[3, 384, 384] -> `src` [1024, 256]. `keep` (with `keep_taps`): the
        windowed embeddings, every block's output, then the [1024, 1536] taps."""
        ref ctx = self.wts.ctx
        var patches = self.new(NTOK * 3 * PATCH * PATCH)
        ctx.enqueue_function[patchify_kernel](px, patches, grid_dim=(blocks(3 * PATCH * PATCH), NTOK), block_dim=TPB)
        var tok = linear(self.wts, EMB + "patch_embeddings.projection", patches, NTOK, 3 * PATCH * PATCH)
        var nrow = NWIN * NWIN * WTOK
        var x = self.new(nrow * BB_DIM)
        ctx.enqueue_function[window_embed_kernel](
            tok, self.wts.w(EMB + "cls_token"), self.wts.w(EMB + "position_embeddings"), x,
            grid_dim=(blocks(BB_DIM), nrow), block_dim=TPB,
        )
        if keep_taps:
            keep.append(_copy(ctx, x, nrow * BB_DIM))
        var cat = self.new(NTOK * 4 * BB_DIM)
        for li in range(LAYERS):
            var p = LAYER + String(li) + "."
            var h = self.new(nrow * BB_DIM)
            layer_norm(self.wts, p + "norm1", x, h, nrow, BB_DIM, BB_EPS)
            var q = linear(self.wts, p + "attention.attention.query", h, nrow, BB_DIM)
            var k = linear(self.wts, p + "attention.attention.key", h, nrow, BB_DIM)
            var v = linear(self.wts, p + "attention.attention.value", h, nrow, BB_DIM)
            var a: DeviceBuffer[DT]
            if is_global(li) and (self.debug & 1) == 0:
                a = mha(ctx, q, k, v, nrow, nrow, HEADS, BB_DIM // HEADS, nrow)
            else:
                a = self.new(nrow * BB_DIM)
                for w in range(NWIN * NWIN):
                    var off = w * WTOK * BB_DIM
                    var aw = mha(
                        ctx, q.create_sub_buffer[DT](off, WTOK * BB_DIM), k.create_sub_buffer[DT](off, WTOK * BB_DIM),
                        v.create_sub_buffer[DT](off, WTOK * BB_DIM), WTOK, WTOK, HEADS, BB_DIM // HEADS, WTOK,
                    )
                    ctx.enqueue_copy(a.create_sub_buffer[DT](off, WTOK * BB_DIM), aw)
            var o = linear(self.wts, p + "attention.output.dense", a, nrow, BB_DIM)
            ctx.enqueue_function[scale_add_kernel](
                x, o, self.wts.w(p + "layer_scale1.lambda1"), Int64(BB_DIM), grid_dim=(nrow, blocks(BB_DIM)), block_dim=TPB,
            )
            var h2 = self.new(nrow * BB_DIM)
            layer_norm(self.wts, p + "norm2", x, h2, nrow, BB_DIM, BB_EPS)
            var f1 = linear(self.wts, p + "mlp.fc1", h2, nrow, BB_DIM)
            ctx.enqueue_function[gelu_kernel](f1, Int64(nrow * BB_FF), grid_dim=blocks(nrow * BB_FF), block_dim=TPB)
            var f2 = linear(self.wts, p + "mlp.fc2", f1, nrow, BB_FF)
            ctx.enqueue_function[scale_add_kernel](
                x, f2, self.wts.w(p + "layer_scale2.lambda1"), Int64(BB_DIM), grid_dim=(nrow, blocks(BB_DIM)), block_dim=TPB,
            )
            if keep_taps:
                keep.append(_copy(ctx, x, nrow * BB_DIM))
            var t = is_tap(li)
            if t >= 0:
                var ln = self.new(nrow * BB_DIM)
                layer_norm(self.wts, BB_LN, x, ln, nrow, BB_DIM, BB_EPS)
                ctx.enqueue_function[unwindow_kernel](
                    ln, cat, Int64(4 * BB_DIM), Int64(t * BB_DIM), grid_dim=(blocks(BB_DIM), NTOK), block_dim=TPB,
                )
        if keep_taps:
            keep.append(_copy(ctx, cat, NTOK * 4 * BB_DIM))
        return self.projector(cat)

    def projector(mut self, cat: DeviceBuffer[DT]) raises -> DeviceBuffer[DT]:
        """[1024, 1536] -> [1024, 256] (C2f + LayerNorm)."""
        ref ctx = self.wts.ctx
        var y = linear(self.wts, PROJ + "0.cv1.conv", cat, NTOK, 4 * BB_DIM, bias=False)
        self._ln_silu(PROJ + "0.cv1.bn", y, NTOK, 2 * C2F_HALF)
        var cc = self.new(NTOK * 5 * C2F_HALF)
        ctx.enqueue_function[copy_cols_kernel](
            y, Int64(2 * C2F_HALF), Int64(0), cc, Int64(5 * C2F_HALF), Int64(0), Int64(2 * C2F_HALF),
            grid_dim=(NTOK, blocks(2 * C2F_HALF)), block_dim=TPB,
        )
        var z = self.new(NTOK * C2F_HALF)
        ctx.enqueue_function[copy_cols_kernel](
            y, Int64(2 * C2F_HALF), Int64(C2F_HALF), z, Int64(C2F_HALF), Int64(0), Int64(C2F_HALF),
            grid_dim=(NTOK, blocks(C2F_HALF)), block_dim=TPB,
        )
        for k in range(3):
            var base = PROJ + "0.m." + String(k) + "."
            var z1 = conv3x3_tm(ctx, z, GRID, GRID, C2F_HALF, C2F_HALF, self.taps[base + "cv1.conv"])
            self._ln_silu(base + "cv1.bn", z1, NTOK, C2F_HALF)
            var z2 = conv3x3_tm(ctx, z1, GRID, GRID, C2F_HALF, C2F_HALF, self.taps[base + "cv2.conv"])
            self._ln_silu(base + "cv2.bn", z2, NTOK, C2F_HALF)
            if (self.debug & 4) != 0:
                ctx.enqueue_function[add_kernel](z2, z, Int64(NTOK * C2F_HALF), grid_dim=blocks(NTOK * C2F_HALF), block_dim=TPB)
            ctx.enqueue_function[copy_cols_kernel](
                z2, Int64(C2F_HALF), Int64(0), cc, Int64(5 * C2F_HALF), Int64((2 + k) * C2F_HALF), Int64(C2F_HALF),
                grid_dim=(NTOK, blocks(C2F_HALF)), block_dim=TPB,
            )
            z = z2
        var out = linear(self.wts, PROJ + "0.cv2.conv", cc, NTOK, 5 * C2F_HALF, bias=False)
        self._ln_silu(PROJ + "0.cv2.bn", out, NTOK, D)
        var src = self.new(NTOK * D)
        layer_norm(self.wts, PROJ + "1", out, src, NTOK, D, BB_EPS)
        return src^

    # ── 4-6. two-stage, decoder, heads ───────────────────────────────────

    def _mlp(mut self, name: String, x: DeviceBuffer[DT], t: Int, k: Int, layers: Int) raises -> DeviceBuffer[DT]:
        """`MLP`: ReLU between layers, none after the last."""
        var h = x
        var kk = k
        for i in range(layers):
            var y = linear(self.wts, name + ".layers." + String(i), h, t, kk)
            var n = len(y) // t
            if i < layers - 1:
                self.wts.ctx.enqueue_function[relu_kernel](y, Int64(t * n), grid_dim=blocks(t * n), block_dim=TPB)
            h = y
            kk = n
        return h^

    def _rows_linear(mut self, name: String, row0: Int, n: Int, x: DeviceBuffer[DT], t: Int, k: Int) raises -> DeviceBuffer[DT]:
        """`x @ W[row0 : row0 + n]^T + b[row0 : row0 + n]` (a slice of a packed
        q | k | v projection)."""
        var y = self.new(t * n)
        gemm_tb(self.wts.ctx, y, x, self.wts.w(name + "_weight").create_sub_buffer[DT](row0 * k, n * k), t, n, k)
        self.wts.ctx.enqueue_function[bias_kernel](
            y, self.wts.w(name + "_bias").create_sub_buffer[DT](row0, n), Int64(n), Int64(t),
            grid_dim=(t, blocks(n)), block_dim=TPB,
        )
        return y^

    def _ln_new(mut self, name: String, x: DeviceBuffer[DT], rows: Int, cols: Int, eps: Float32) raises -> DeviceBuffer[DT]:
        var y = self.new(rows * cols)
        layer_norm(self.wts, name, x, y, rows, cols, eps)
        return y^

    def _sum(self, a: DeviceBuffer[DT], b: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DT]:
        var s = self.new(n)
        self.wts.ctx.enqueue_function[add_into_kernel](s, a, b, Int64(n), grid_dim=blocks(n), block_dim=TPB)
        return s^

    def decode(mut self, src: DeviceBuffer[DT]) raises -> RfDetrRaw:
        """`src` [1024, 256] -> logits, boxes, mask logits, the decoder outputs."""
        ref ctx = self.wts.ctx
        comptime T = "transformer."
        # two-stage
        var pre = linear(self.wts, T + "enc_output.0", src, NTOK, D)
        var normed = self._ln_new(T + "enc_output_norm.0", pre, NTOK, D, TR_EPS)
        var cls = linear(self.wts, T + "enc_out_class_embed.0", normed, NTOK, D)
        self.enc_logits = self._down(cls, NTOK * NCLS)
        var best = List[Float32](capacity=NTOK)
        for n in range(NTOK):
            var m = self.enc_logits[n * NCLS]
            for c in range(1, NCLS):
                m = max(m, self.enc_logits[n * NCLS + c])
            best.append(m)
        var top = _top_k(best, NQ)
        var hidx = ctx.enqueue_create_host_buffer[DType.int32](NQ)
        for i in range(NQ):
            hidx[i] = Int32(top[i])
        var didx = ctx.enqueue_create_buffer[DType.int32](NQ)
        ctx.enqueue_copy(didx, hidx)
        var tgt_ts = self.new(NQ * D)
        ctx.enqueue_function[gather_rows_kernel](normed, didx, tgt_ts, Int64(D), grid_dim=(NQ, blocks(D)), block_dim=TPB)
        var dts = self._down(self._mlp(T + "enc_out_bbox_embed.0", tgt_ts, NQ, D, 3), NQ * 4)
        var refb = List[Float32](capacity=NQ * 4)
        for i in range(NQ):
            var n = top[i]
            var ax = (Float32(n % GRID) + Float32(0.5)) / Float32(GRID)
            var ay = (Float32(n // GRID) + Float32(0.5)) / Float32(GRID)
            var aw = Float32(0.05)
            var ah = Float32(0.05)
            # the proposal, then the learned refpoint composed onto it (both bbox_reparam)
            var tx = dts[i * 4] * aw + ax
            var ty = dts[i * 4 + 1] * ah + ay
            var tw = exp(dts[i * 4 + 2]) * aw
            var th = exp(dts[i * 4 + 3]) * ah
            refb.append(self.refpoint[i * 4] * tw + tx)
            refb.append(self.refpoint[i * 4 + 1] * th + ty)
            refb.append(exp(self.refpoint[i * 4 + 2]) * tw)
            refb.append(exp(self.refpoint[i * 4 + 3]) * th)
        self.dec_ref = refb.copy()
        var dref = self._up(refb)
        self.query_pos = self._mlp(T + "decoder.ref_point_head", self._up(_sine(refb)), NQ, 2 * D, 2)
        # decoder
        var out = self.new(NQ * D)
        ctx.enqueue_copy(out, self.wts.w("query_feat.weight").create_sub_buffer[DT](0, NQ * D))
        self.dec_layers = List[DeviceBuffer[DT]]()
        var hs = List[DeviceBuffer[DT]]()
        for l in range(DEC_LAYERS):
            var p = T + "decoder.layers." + String(l) + "."
            var qk = self._sum(out, self.query_pos, NQ * D)
            var q = self._rows_linear(p + "self_attn.in_proj", 0, D, qk, NQ, D)
            var k = self._rows_linear(p + "self_attn.in_proj", D, D, qk, NQ, D)
            var v = self._rows_linear(p + "self_attn.in_proj", 2 * D, D, out, NQ, D)
            var sa = mha(ctx, q, k, v, NQ, NQ, 8, D // 8, NQ)
            var so = linear(self.wts, p + "self_attn.out_proj", sa, NQ, D)
            var out1 = self._ln_new(p + "norm1", self._sum(out, so, NQ * D), NQ, D, TR_EPS)
            var query = self._sum(out1, self.query_pos, NQ * D)
            var value = linear(self.wts, p + "cross_attn.value_proj", src, NTOK, D)
            var attw = linear(self.wts, p + "cross_attn.attention_weights", query, NQ, D)
            var offs = linear(self.wts, p + "cross_attn.sampling_offsets", query, NQ, D)
            var ca = self.new(NQ * D)
            ctx.enqueue_function[msda_kernel](
                value, offs, attw, dref, ca, Int64(GRID), Int64(GRID), Int64(1 if (self.debug & 2) != 0 else 0),
                grid_dim=(NQ, 16), block_dim=16,
            )
            var co = linear(self.wts, p + "cross_attn.output_proj", ca, NQ, D)
            var out2 = self._ln_new(p + "norm2", self._sum(out1, co, NQ * D), NQ, D, TR_EPS)
            var f1 = linear(self.wts, p + "linear1", out2, NQ, D)
            ctx.enqueue_function[relu_kernel](f1, Int64(NQ * DEC_FF), grid_dim=blocks(NQ * DEC_FF), block_dim=TPB)
            var f2 = linear(self.wts, p + "linear2", f1, NQ, DEC_FF)
            out = self._ln_new(p + "norm3", self._sum(out2, f2, NQ * D), NQ, D, TR_EPS)
            self.dec_layers.append(_copy(ctx, out, NQ * D))
            hs.append(self._ln_new(T + "decoder.norm", out, NQ, D, TR_EPS))
        # heads
        var logits = self._down(linear(self.wts, "class_embed", hs[DEC_LAYERS - 1], NQ, D), NQ * NCLS)
        var dl = self._down(self._mlp("bbox_embed", hs[DEC_LAYERS - 1], NQ, D, 3), NQ * 4)
        var boxes = List[Float32](capacity=NQ * 4)
        for i in range(NQ):
            boxes.append(dl[i * 4] * refb[i * 4 + 2] + refb[i * 4])
            boxes.append(dl[i * 4 + 1] * refb[i * 4 + 3] + refb[i * 4 + 1])
            boxes.append(exp(dl[i * 4 + 2]) * refb[i * 4 + 2])
            boxes.append(exp(dl[i * 4 + 3]) * refb[i * 4 + 3])
        var masks = self.mask_head(src, hs[DEC_LAYERS - 1])
        return RfDetrRaw(logits^, boxes^, masks^, hs^)

    # ── 7. masks ──────────────────────────────────────────────────────────

    def mask_head(mut self, src: DeviceBuffer[DT], queries: DeviceBuffer[DT]) raises -> DeviceBuffer[DT]:
        """[100, 96 x 96] mask logits."""
        ref ctx = self.wts.ctx
        comptime S = "segmentation_head."
        comptime NPX = MASK * MASK
        var x = self.new(NPX * D)
        ctx.enqueue_function[upsample_bilinear_kernel](
            src, x, Int64(GRID), Int64(GRID), Int64(MASK), Int64(MASK), Int64(D), grid_dim=(NPX, blocks(D)), block_dim=TPB,
        )
        for b in range(DEC_LAYERS):
            var p = S + "blocks." + String(b) + "."
            var dw = self.new(NPX * D)
            ctx.enqueue_function[dwconv3x3_kernel](
                x, self.wts.w(p + "dwconv.weight"), self.wts.w(p + "dwconv.bias"), dw, Int64(MASK), Int64(MASK), Int64(D),
                grid_dim=(NPX, blocks(D)), block_dim=TPB,
            )
            layer_norm(self.wts, p + "norm", dw, dw, NPX, D, BB_EPS)
            var pw = linear(self.wts, p + "pwconv1", dw, NPX, D)
            ctx.enqueue_function[gelu_kernel](pw, Int64(NPX * D), grid_dim=blocks(NPX * D), block_dim=TPB)
            ctx.enqueue_function[add_kernel](x, pw, Int64(NPX * D), grid_dim=blocks(NPX * D), block_dim=TPB)
        var pix = linear(self.wts, S + "spatial_features_proj", x, NPX, D)
        var qn = self._ln_new(S + "query_features_block.norm_in", queries, NQ, D, TR_EPS)
        var h1 = linear(self.wts, S + "query_features_block.layers.0", qn, NQ, D)
        ctx.enqueue_function[gelu_kernel](h1, Int64(NQ * 4 * D), grid_dim=blocks(NQ * 4 * D), block_dim=TPB)
        var h2 = linear(self.wts, S + "query_features_block.layers.2", h1, NQ, 4 * D)
        var qf = linear(self.wts, S + "query_features_proj", self._sum(queries, h2, NQ * D), NQ, D)
        var masks = self.new(NQ * NPX)
        gemm_tb(ctx, masks, qf, pix, NQ, NPX, D)
        ctx.enqueue_function[add_scalar_kernel](masks, self.wts.w(S + "bias"), Int64(NQ * NPX), grid_dim=blocks(NQ * NPX), block_dim=TPB)
        return masks^

    # ── 8. the whole detector ─────────────────────────────────────────────

    def forward(mut self, pic: Picture) raises -> RfDetrRaw:
        var none = List[DeviceBuffer[DT]]()
        return self.decode(self.backbone(self.pixel_values(pic), none))

    def detect(mut self, pic: Picture, threshold: Float32 = 0.3) raises -> List[RfDetrInstance]:
        var raw = self.forward(pic)
        return self.post_process(raw, pic.height, pic.width, threshold)

    def post_process(
        mut self, ref raw: RfDetrRaw, height: Int, width: Int, threshold: Float32,
    ) raises -> List[RfDetrInstance]:
        # (query, class) pairs above the threshold, descending score, stable (the lower flat index first)
        var flat = List[Int]()
        var prob = List[Float32]()
        for i in range(NQ * NCLS):
            var s = Float32(1.0) / (Float32(1.0) + exp(-raw.logits[i]))
            if s > threshold:
                flat.append(i)
                prob.append(s)
        for i in range(1, len(flat)):  # insertion sort: a handful kept
            var j = i
            while j > 0 and prob[j - 1] < prob[j]:
                var tp = prob[j - 1]
                prob[j - 1] = prob[j]
                prob[j] = tp
                var tf = flat[j - 1]
                flat[j - 1] = flat[j]
                flat[j] = tf
                j -= 1
        var out = List[RfDetrInstance]()
        comptime NPX = MASK * MASK
        ref ctx = self.wts.ctx
        var hbuf = ctx.enqueue_create_host_buffer[DT](NPX)
        for k in range(min(len(flat), NQ)):
            var q = flat[k] // NCLS
            var label = flat[k] - q * NCLS
            var cx = raw.boxes[q * 4]
            var cy = raw.boxes[q * 4 + 1]
            var w = max(raw.boxes[q * 4 + 2], Float32(0))
            var h = max(raw.boxes[q * 4 + 3], Float32(0))
            var W = Float32(width)
            var H = Float32(height)
            var box = List[Float32](capacity=4)
            box.append(min(max((cx - Float32(0.5) * w) * W, Float32(0)), W))
            box.append(min(max((cy - Float32(0.5) * h) * H, Float32(0)), H))
            box.append(min(max((cx + Float32(0.5) * w) * W, Float32(0)), W))
            box.append(min(max((cy + Float32(0.5) * h) * H, Float32(0)), H))
            ctx.enqueue_copy(hbuf, raw.masks.create_sub_buffer[DT](q * NPX, NPX))
            ctx.synchronize()
            var lg = List[Float32](capacity=NPX)
            for i in range(NPX):
                lg.append(hbuf[i])
            var up = List[Float32]()
            torch_bilinear_chw(lg, 1, MASK, MASK, up, height, width)
            var mask = List[UInt8](capacity=height * width)
            for v in up:
                mask.append(UInt8(1) if v > 0.0 else UInt8(0))
            out.append(RfDetrInstance(label, prob[k], box^, mask^, q))
        return out^


def rfdetr_checkpoint() raises -> String:
    """`RFDETR_CHECKPOINT`, else `~/.cache/noeira/rfdetr/seg-small/model.safetensors`."""
    var env = getenv("RFDETR_CHECKPOINT")
    if env.byte_length() > 0:
        return env
    return getenv("HOME") + "/.cache/noeira/rfdetr/seg-small/model.safetensors"


def _top_k(v: List[Float32], k: Int) -> List[Int]:
    """The indices of the `k` largest, descending; ties to the lower index."""
    var idx = List[Int]()
    var used = List[Bool](length=len(v), fill=False)
    for _ in range(k):
        var b = -1
        for i in range(len(v)):
            if not used[i] and (b < 0 or v[i] > v[b]):
                b = i
        used[b] = True
        idx.append(b)
    return idx^


def _sine(ref_boxes: List[Float32]) -> List[Float32]:
    """`gen_sineembed_for_position(ref, 128)`: per box its y, x, w, h encodings,
    `a_i = c 2 pi / 1e4^(2 (i // 2) / 128)`, sin at even i, cos at odd -> [100, 512]."""
    comptime NF = 128
    var dim_t = List[Float32](capacity=NF)
    for i in range(NF):
        dim_t.append(pow(Float32(10000.0), Float32(2 * (i // 2)) / Float32(NF)))
    var scale = Float32(6.283185307179586)
    var order: List[Int] = [1, 0, 2, 3]
    var out = List[Float32](capacity=len(ref_boxes) // 4 * 4 * NF)
    for q in range(len(ref_boxes) // 4):
        for s in range(4):
            var c = ref_boxes[q * 4 + order[s]] * scale
            for i in range(NF):
                var a = c / dim_t[i]
                out.append(sin(a) if i % 2 == 0 else cos(a))
    return out^


def _copy(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DT]:
    var b = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(b, x.create_sub_buffer[DT](0, n))
    return b^
