# +--------------------------------------------------------------------------+ #
# | SAM 3 — the whole detector: a picture and a word in, instances out
# +--------------------------------------------------------------------------+ #
"""`Sam3`: what `Sam3Processor` + `Sam3Model` + `post_process_instance_segmentation`
do for a TEXT prompt, in Mojo (`tools/perception/detector_eval.py`'s use).

    var sam = Sam3(DeviceContext())                      # ~3.4 GB of fp32 weights on the device
    var img = sam.encode_image(picture)                  # once per picture (the backbone + neck)
    for word in ["door", "sofa"]:
        for inst in sam.segment(img, word):              # once per word
            inst.score; inst.box (x0, y0, x1, y1 pixels); inst.mask (H x W, 0/1)

1. **Pixels**: the RGB24 picture resized to 1008 x 1008 with torch's uint8
   antialiased bilinear (`noeira.vision.preprocess.torch_bilinear_aa_u8`, bit-
   exact to the processor's), then `(v - 127.5) / 127.5`, channels first.
2. **Image** (`encode_image`): the backbone (`Sam3ViT`), the neck (`Sam3Neck`);
   the 288 / 144 / 72 levels are kept (the 36 one is never used for a text
   prompt) with the 72 level's sine position encoding.
3. **Word** (`segment`): CLIP's tokens (`noeira.text.clip_bpe`, 32), the text
   features (`Sam3Text`), the DETR encoder on the 72 level, the decoder and the
   scoring; `score = sigmoid(logit) sigmoid(presence)`, kept above `threshold`;
   the boxes scaled to the picture; only if one is kept, the mask decoder, each
   kept mask's sigmoid resized to the picture (torch's float bilinear,
   `align_corners=False`) and cut at `mask_threshold`.
"""

from std.math import exp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.ai.picture import Picture
from noeira.nn.constants import DT
from noeira.text.clip_bpe import ClipBPE
from noeira.vision.preprocess import torch_bilinear_aa_u8
from noeira.vision.resize_pad import torch_bilinear_chw
from .ops import sam3_checkpoint
from .vit import Sam3ViT, IMG
from .neck import Sam3Neck, FpnLevel, position_encoding
from .text import Sam3Text
from .detr import Sam3DetrEncoder
from .decoder import Sam3Decoder
from .masks import Sam3MaskDecoder


@fieldwise_init
struct Sam3Instance(Copyable, Movable):
    var score: Float32
    var box: List[Float32]
    """x0, y0, x1, y1 in the picture's pixels."""
    var mask: List[UInt8]
    """[H x W] 0 / 1, the picture's size."""
    var query: Int


@fieldwise_init
struct Sam3Image(Movable):
    var levels: List[FpnLevel]
    """288, 144, 72 (token-major, 256 channels)."""
    var pos72: DeviceBuffer[DT]
    var height: Int
    var width: Int


struct Sam3(Movable):
    var vit: Sam3ViT
    var neck: Sam3Neck
    var text: Sam3Text
    var enc: Sam3DetrEncoder
    var dec: Sam3Decoder
    var masks: Sam3MaskDecoder
    var tok: ClipBPE

    def __init__(out self, var ctx: DeviceContext, path: String = String("")) raises:
        var p = path if path.byte_length() > 0 else sam3_checkpoint()
        var dir = String(p[byte=0:p.rfind("/") + 1])
        self.tok = ClipBPE.load(dir + "tokenizer.json")
        self.vit = Sam3ViT(ctx, p)
        self.neck = Sam3Neck(ctx, p)
        self.text = Sam3Text(ctx, p)
        self.enc = Sam3DetrEncoder(ctx, p)
        self.dec = Sam3Decoder(ctx, p)
        self.masks = Sam3MaskDecoder(ctx^, p)

    def pixel_values(self, pic: Picture) raises -> DeviceBuffer[DT]:
        """[3, 1008, 1008], the processor's `pixel_values`."""
        var r = List[UInt8]()
        torch_bilinear_aa_u8(pic.rgb, pic.width, pic.height, 3, r, IMG, IMG)
        ref ctx = self.vit.wts.ctx
        var h = ctx.enqueue_create_host_buffer[DT](3 * IMG * IMG)
        for i in range(IMG * IMG):
            for c in range(3):
                h[c * IMG * IMG + i] = (Float32(r[i * 3 + c]) - Float32(127.5)) / Float32(127.5)
        var d = ctx.enqueue_create_buffer[DT](3 * IMG * IMG)
        ctx.enqueue_copy(d, h)
        return d^

    def encode_image(mut self, pic: Picture) raises -> Sam3Image:
        var none = List[DeviceBuffer[DT]]()
        var feats = self.vit.forward(self.pixel_values(pic), none)
        var lv = self.neck.forward(feats)
        var keep = List[FpnLevel]()
        for i in range(3):
            keep.append(lv[i].copy())
        var pe = position_encoding(lv[2].h, lv[2].w)
        ref ctx = self.vit.wts.ctx
        var hp = ctx.enqueue_create_host_buffer[DT](len(pe))
        for i in range(len(pe)):
            hp[i] = pe[i]
        var dp = ctx.enqueue_create_buffer[DT](len(pe))
        ctx.enqueue_copy(dp, hp)
        return Sam3Image(keep^, dp^, pic.height, pic.width)

    def segment(
        mut self, img: Sam3Image, word: String, threshold: Float32 = 0.3, mask_threshold: Float32 = 0.5,
    ) raises -> List[Sam3Instance]:
        var toks = self.tok.encode(word, 32)
        var none = List[DeviceBuffer[DT]]()
        var tf = self.text.forward(toks, none)
        ref l72 = img.levels[2]
        var fused = self.enc.forward(l72.x, img.pos72, l72.h * l72.w, tf, toks.length, none)
        var det = self.dec.forward(fused, img.pos72, tf, toks.length)
        var pres = Float32(1.0) / (Float32(1.0) + exp(-det.presence))
        var kept = List[Int]()
        var scores = List[Float32]()
        for q in range(len(det.logits)):
            var s = Float32(1.0) / (Float32(1.0) + exp(-det.logits[q])) * pres
            if s > threshold:
                kept.append(q)
                scores.append(s)
        var out = List[Sam3Instance]()
        if len(kept) == 0:
            return out^
        var m = self.masks.forward(det.queries, img.levels[0].x, img.levels[1].x, fused, tf, toks.length, img.levels[0].h, img.levels[0].w)
        var mh = m.h
        var mw = m.w
        ref ctx = self.vit.wts.ctx
        var hbuf = ctx.enqueue_create_host_buffer[DT](mh * mw)
        for k in range(len(kept)):
            var q = kept[k]
            ctx.enqueue_copy(hbuf, m.masks.create_sub_buffer[DT](q * mh * mw, mh * mw))
            ctx.synchronize()
            var prob = List[Float32](capacity=mh * mw)
            for i in range(mh * mw):
                prob.append(Float32(1.0) / (Float32(1.0) + exp(-hbuf[i])))
            var up = List[Float32]()
            torch_bilinear_chw(prob, 1, mh, mw, up, img.height, img.width)
            var mask = List[UInt8](capacity=img.height * img.width)
            for v in up:
                mask.append(UInt8(1) if v > mask_threshold else UInt8(0))
            var box = List[Float32](capacity=4)
            box.append(det.boxes[q * 4] * Float32(img.width))
            box.append(det.boxes[q * 4 + 1] * Float32(img.height))
            box.append(det.boxes[q * 4 + 2] * Float32(img.width))
            box.append(det.boxes[q * 4 + 3] * Float32(img.height))
            out.append(Sam3Instance(scores[k], box^, mask^, q))
        return out^
