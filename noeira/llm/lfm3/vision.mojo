# +--------------------------------------------------------------------------+ #
# | LFM3-VL — the vision tower and its projector (pictures into the backbone)
# +--------------------------------------------------------------------------+ #
"""`Lfm3Vision` — LFM3-VL's picture encoder, run from the same checkpoint as
the text backbone (d1-2B, LFM3-VL-2B), verified against d1's own code
(docs/NN_MODEL_PORTS_PLAN.md §12.5, step 6).

    var vis = Lfm3Vision.load(lm.ctx, dir)
    var feats = vis.encode(picture, taps)                   # [N / 4, 2048]: one row per <|image|> token
    var h = lm.forward(ids, groups, state_len, taps2, image_feats=Optional(feats), image_token=vis.cfg.image_token)

## The tower (`Lfm3VlVisionModel`)

A picture `[H, W, 3]` uint8 becomes 16x16 patches (each repeated twice in time:
1536 values), in MERGE-BLOCK order (`vision_kernels.mojo`); `patchifier`
(linear, no bias), then `embedding_norm` (RMSNorm). 14 pre-norm layers (NOT the
text backbone's sandwich):

    x = x + OP( operator_norm(x) ),  x = x + w2( silu(w1 n) * w3 n ),  n = ffn_norm(x)

OP = a gated depthwise 3x3 conv over the patch grid (`in_proj` -> B, C, x;
`C * conv(B * x)`; `out_proj`), or bidirectional attention over the picture
(16 heads, 8 KV heads, head_dim 64, RMSNorm per head on q and k, NO output
gate, `out_proj`) with AXIAL 2-D RoPE: theta 1e6, 16 frequencies `1 / theta^(2i
/ 32)`; slot i < 16 turns by `row * f_i`, slot 16 + i by `col * f_i`, and
rotate-half pairs slot j with j + 32. No final norm.

## The projector

`LayerNorm(1024)` (eps 1e-6, with bias), four consecutive tokens (one 2x2
block) as one 4096-wide row, `linear_1` (+ bias), exact GELU, `linear_2`
(+ bias) to the backbone's 2048: the features of the `<|image|>` tokens.

## Pictures it takes

The processor resizes with `smart_resize` (bicubic) to multiples of 32 inside
[min, max] pixels; a picture ALREADY there passes unchanged (its bicubic resize
is the identity), and that is the only kind taken here: anything else raises.
Bring camera frames to such a size (640x480, 512x384, ...). d1's own cap
(`d1_vision_max_pixels`, 1 638 400) applies first, as in its `media.cap_pixels`.
The attention holds `16 N^2` floats for N patches (92 MB at 640x480).

## `debug` (falsification only)

Bit 1: patches in RASTER order. Bit 2: the conv's neighbours taken in token
order. Bit 4: the RoPE axes swapped (rows turn the column slots).
"""

from std.math import cos, sin, pow
from std.memory import unsafe_memcpy
from std.sys import default_accelerator
from std.sys.defines import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.ai.picture import Picture
from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm, cublas_gemm_lp
from noeira.io.json import load_json
from noeira.io.safetensors import SafeTensors, ST_BF16
from .kernels import (
    TPB, bf16_to_f32_kernel, rms_kernel, rms_bf16_kernel, add_kernel, rope_kernel, swiglu_kernel,
    swiglu_bf16_kernel, f32_to_bf16_kernel,
)
from .vision_kernels import patchify_kernel, conv2d_gated_kernel, layer_norm_kernel, bias_kernel, gelu_kernel
from .model import LF_BF16, LF_TF32, attention_probs, attention_apply, _blocks


comptime LF_VIS_TF32: Bool = is_defined["LFM3_VIS_TF32"]()
comptime VIS_BF16: Bool = LF_BF16 and not LF_VIS_TF32
"""The tower's GEMMs on bf16 operands (the bf16 build's default) — or, with
`-D LFM3_VIS_TF32=1`, fp32 activations and TF32 GEMMs (weights widened per GEMM)."""
comptime VIS_TF32: Bool = LF_TF32 or LF_VIS_TF32

comptime VT_CONV: Int = 0
comptime VT_ATTN: Int = 1
comptime _ALIGN: Int = 64


struct Lfm3VisionConfig(Copyable, Movable):
    var hidden: Int
    var heads: Int
    var kv_heads: Int
    var head_dim: Int
    var ff: Int
    var patch: Int
    var tpatch: Int
    var merge: Int
    var theta: Float64
    var eps: Float32
    var layer_types: List[Int]
    var proj_hidden: Int
    var text_hidden: Int
    var proj_eps: Float32
    var image_token: Int
    var vision_start: Int
    var vision_end: Int
    var min_pixels: Int
    var max_pixels: Int
    """d1's own cap (`d1_vision_max_pixels`) when the checkpoint records one,
    else the processor's `longest_edge`."""

    def __init__(out self):
        self.hidden = 0
        self.heads = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.ff = 0
        self.patch = 16
        self.tpatch = 2
        self.merge = 2
        self.theta = 1e6
        self.eps = 1e-5
        self.layer_types = List[Int]()
        self.proj_hidden = 0
        self.text_hidden = 0
        self.proj_eps = 1e-6
        self.image_token = -1
        self.vision_start = -1
        self.vision_end = -1
        self.min_pixels = 0
        self.max_pixels = 0

    @staticmethod
    def load(dir: String) raises -> Lfm3VisionConfig:
        """`config.json` (its `vision_config` and the projector's keys) and
        `processor_config.json` (the pixel range)."""
        var doc = load_json(dir + "/config.json")
        var r = doc.root()
        var vc = doc.field(r, "vision_config")
        if vc < 0:
            raise Error("lfm3-vl: '" + dir + "/config.json' has no vision_config (a text-only checkpoint)")
        var c = Lfm3VisionConfig()
        c.hidden = doc.integer(doc.field(vc, "hidden_size"))
        c.heads = doc.integer(doc.field(vc, "num_attention_heads"))
        c.kv_heads = doc.integer(doc.field(vc, "num_key_value_heads"))
        c.head_dim = c.hidden // c.heads
        c.ff = doc.integer(doc.field(vc, "intermediate_size"))
        c.patch = doc.integer(doc.field(vc, "patch_size"))
        c.tpatch = doc.integer(doc.field(vc, "temporal_patch_size"))
        c.merge = doc.integer(doc.field(vc, "spatial_merge_size"))
        c.eps = Float32(doc.number(doc.field(vc, "norm_eps")))
        if doc.integer(doc.field(vc, "conv_kernel_size")) != 3 or c.merge != 2 or doc.integer(doc.field(r, "downsample_factor")) != 2:
            raise Error("lfm3-vl: only a 3x3 conv and a 2x2 merge are implemented")
        var rp = doc.field(vc, "rope_parameters")
        if doc.string(doc.field(rp, "rope_type")) != "axial":
            raise Error("lfm3-vl: only the axial RoPE is implemented")
        c.theta = doc.number(doc.field(rp, "rope_theta"))
        var lt = doc.field(vc, "layer_types")
        for i in range(doc.size(lt)):
            var s = doc.string(doc.at(lt, i))
            if s == "conv":
                c.layer_types.append(VT_CONV)
            elif s == "full_attention":
                c.layer_types.append(VT_ATTN)
            else:
                raise Error("lfm3-vl: unknown vision layer type '" + s + "'")
        if doc.string(doc.field(r, "projector_hidden_act")) != "gelu" or not doc.boolean(doc.field(r, "projector_bias")):
            raise Error("lfm3-vl: only a biased GELU projector is implemented")
        c.proj_hidden = doc.integer(doc.field(r, "projector_hidden_size"))
        c.proj_eps = Float32(doc.number(doc.field(r, "projector_layer_norm_eps")))
        c.text_hidden = doc.integer(doc.field(doc.field(r, "text_config"), "hidden_size"))
        c.image_token = doc.integer(doc.field(r, "image_token_id"))
        c.vision_start = doc.integer(doc.field(r, "vision_start_token_id"))
        c.vision_end = doc.integer(doc.field(r, "vision_end_token_id"))
        var pc = load_json(dir + "/processor_config.json")
        var ip = pc.field(pc.root(), "image_processor")
        var size = pc.field(ip, "size")
        c.min_pixels = pc.integer(pc.field(size, "shortest_edge"))
        c.max_pixels = pc.integer(pc.field(size, "longest_edge"))
        var cap = doc.field(r, "d1_vision_max_pixels")
        if cap >= 0:
            c.max_pixels = min(c.max_pixels, doc.integer(cap))
        return c^


struct Lfm3Vision(Movable):
    var ctx: DeviceContext
    var cfg: Lfm3VisionConfig
    var arena: DeviceBuffer[DType.uint16]
    var index: Dict[String, Int]
    """`tower.<name>` / `proj.<name>` -> its slot in `dev`."""
    var dev: List[DeviceBuffer[DType.uint16]]
    var offs: List[Int]
    var used: List[Bool]
    var names: List[String]
    var scratch: DeviceBuffer[DT]
    var dummy: DeviceBuffer[DType.int32]
    """The attention's `groups` argument, unread in mode 3."""
    var debug: Int

    def __init__(
        out self, var ctx: DeviceContext, var cfg: Lfm3VisionConfig, var arena: DeviceBuffer[DType.uint16],
        var index: Dict[String, Int], var dev: List[DeviceBuffer[DType.uint16]], var offs: List[Int],
        var names: List[String], var scratch: DeviceBuffer[DT],
    ) raises:
        self.dummy = ctx.enqueue_create_buffer[DType.int32](1)
        self.ctx = ctx^
        self.cfg = cfg^
        self.arena = arena^
        self.index = index^
        self.dev = dev^
        self.offs = offs^
        self.used = List[Bool](length=len(self.dev), fill=False)
        self.names = names^
        self.scratch = scratch^
        self.debug = 0

    @staticmethod
    def load(var ctx: DeviceContext, dir: String) raises -> Lfm3Vision:
        """The tower (`model.vision_tower.*`) and the projector
        (`model.multi_modal_projector.*`) of `dir/model.safetensors` into one
        bf16 device arena (the same layout as `Lfm3Text.load`)."""
        var cfg = Lfm3VisionConfig.load(dir)
        var st = SafeTensors(dir + "/model.safetensors")
        var prefixes: List[String] = ["model.vision_tower.", "model.multi_modal_projector."]
        var shorts: List[String] = ["tower.", "proj."]
        var picked = List[Int]()
        var short_of = List[String]()
        var total = 0
        var largest = 0
        for i in range(st.size()):
            var name = st.names[i]
            for p in range(len(prefixes)):
                if name.startswith(prefixes[p]):
                    if st.dtypes[i] != ST_BF16:
                        raise Error("lfm3-vl: " + name + " is not bf16")
                    var n = st.numel(name)
                    picked.append(i)
                    short_of.append(shorts[p] + String(name[byte=prefixes[p].byte_length():]))
                    total += (n + _ALIGN - 1) // _ALIGN * _ALIGN
                    largest = max(largest, n)
        if len(picked) == 0:
            raise Error("lfm3-vl: no vision tensors in '" + dir + "/model.safetensors'")
        var arena = ctx.enqueue_create_buffer[DType.uint16](total)
        var staging = ctx.enqueue_create_host_buffer[DType.uint16](total)
        ctx.synchronize()
        var index = Dict[String, Int]()
        var dev = List[DeviceBuffer[DType.uint16]]()
        var offs = List[Int]()
        var names = List[String]()
        var off = 0
        for j in range(len(picked)):
            var raw = st.read_raw(st.names[picked[j]])
            var n = len(raw) // 2
            unsafe_memcpy(dest=staging.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_offset(off * 2), src=raw.unsafe_ptr(), count=len(raw))
            index[short_of[j]] = len(dev)
            dev.append(arena.create_sub_buffer[DType.uint16](off, n))
            offs.append(off)
            names.append(short_of[j])
            off += (n + _ALIGN - 1) // _ALIGN * _ALIGN
        ctx.enqueue_copy(arena, staging)
        ctx.synchronize()
        _ = staging^
        var scratch = ctx.enqueue_create_buffer[DT](largest)
        return Lfm3Vision(ctx^, cfg^, arena^, index^, dev^, offs^, names^, scratch^)

    # ── plumbing ─────────────────────────────────────────────────────────

    def w(mut self, name: String) raises -> DeviceBuffer[DType.uint16]:
        var i = self.index.get(name)
        if not i:
            raise Error("lfm3-vl: no tensor '" + name + "'")
        self.used[i.value()] = True
        return self.dev[i.value()]

    def unused(self) -> List[String]:
        var out = List[String]()
        for i in range(len(self.used)):
            if not self.used[i]:
                out.append(self.names[i])
        return out^

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.ctx.enqueue_create_buffer[DT](max(n, 1))

    def linear(
        mut self, name: String, x: DeviceBuffer[DT], t: Int, k: Int,
        xb: Optional[DeviceBuffer[DType.bfloat16]] = None,
    ) raises -> DeviceBuffer[DT]:
        """`x[t, k] @ W^T`, W the bf16 `name.weight` [n, k]: widened into the
        scratch, or (LFM3_BF16) on the tensor cores in place (`xb`: `x`
        already rounded)."""
        var wb = self.w(name + ".weight")
        var n = len(wb) // k
        var y = self.new(t * n)
        comptime if VIS_BF16:
            var xbb: DeviceBuffer[DType.bfloat16]
            if xb:
                xbb = xb.value()
            else:
                xbb = self.ctx.enqueue_create_buffer[DType.bfloat16](t * k)
                self.ctx.enqueue_function[f32_to_bf16_kernel](x, xbb, Int64(t * k), grid_dim=_blocks(t * k), block_dim=TPB)
            var i = self.index.get(name + ".weight").value()
            var wv = DeviceBuffer[DType.bfloat16](
                self.ctx, self.arena.unsafe_ptr().bitcast[Scalar[DType.bfloat16]]() + self.offs[i], len(wb), owning=False,
            )
            cublas_gemm_lp[False, True, DType.bfloat16, DT](self.ctx, y, xbb, wv, t, n, k, 0.0)
            return y^
        var wf = self.scratch.create_sub_buffer[DT](0, n * k)
        self.ctx.enqueue_function[bf16_to_f32_kernel](wb, wf, Int64(n * k), grid_dim=_blocks(n * k), block_dim=TPB)
        _gemm_tb(self.ctx, y, x, wf, t, n, k)
        return y^

    def rms_b(
        mut self, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, dim: Int,
    ) raises -> Optional[DeviceBuffer[DType.bfloat16]]:
        comptime if VIS_BF16:
            var yb = self.ctx.enqueue_create_buffer[DType.bfloat16](rows * dim)
            self.ctx.enqueue_function[rms_bf16_kernel](
                x, y, yb, self.w(name + ".weight"), Int64(dim), self.cfg.eps, grid_dim=rows, block_dim=TPB,
            )
            return Optional(yb^)
        self.rms(name, x, y, rows, dim)
        return None

    def rms(mut self, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, dim: Int) raises:
        self.ctx.enqueue_function[rms_kernel](
            x, y, self.w(name + ".weight"), Int64(dim), self.cfg.eps, grid_dim=rows, block_dim=TPB,
        )

    def check(self, pic: Picture) raises:
        """Raise unless the processor would take `pic` unchanged (module doc)."""
        var f = self.cfg.patch * self.cfg.merge
        var px = pic.height * pic.width
        if pic.height % f != 0 or pic.width % f != 0 or px < self.cfg.min_pixels or px > self.cfg.max_pixels:
            raise Error(
                "lfm3-vl: a " + String(pic.width) + "x" + String(pic.height) + " picture would be resampled (bicubic, "
                "not implemented): give one whose sides are multiples of " + String(f) + " with "
                + String(self.cfg.min_pixels) + " to " + String(self.cfg.max_pixels) + " pixels"
            )

    def tokens(self, pic: Picture) -> Int:
        """The `<|image|>` tokens `pic` takes: one per 2x2 block of patches."""
        return (pic.height // self.cfg.patch) * (pic.width // self.cfg.patch) // (self.cfg.merge * self.cfg.merge)

    def markup(self, pic: Picture, tok_image: String, tok_start: String, tok_end: String) -> String:
        """The picture's text in a prompt, as d1's processor expands it."""
        var s = tok_start
        for _ in range(self.tokens(pic)):
            s += tok_image
        return s + tok_end

    def rope_table(self, gh: Int, gw: Int) raises -> Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]:
        """cos / sin `[N, head_dim / 2]` of the axial RoPE, tokens in merge-block
        order: slot i < head_dim / 4 at angle `row * f_i`, the next ones `col * f_i`
        (the angle in fp32, as torch's)."""
        var half = self.cfg.head_dim // 2
        var q = half // 2
        var n = gh * gw
        var inv = List[Float32](length=q, fill=0.0)
        for i in range(q):
            inv[i] = Float32(1.0) / Float32(pow(self.cfg.theta, Float64(Float32(2 * i) / Float32(half))))
        var hc = self.ctx.enqueue_create_host_buffer[DT](n * half)
        var hs = self.ctx.enqueue_create_host_buffer[DT](n * half)
        for t in range(n):
            var mw = t & 1
            var mh = (t >> 1) & 1
            var blk = t >> 2
            var bh = blk // (gw // 2)
            var row = bh * 2 + mh
            var col = (blk - bh * (gw // 2)) * 2 + mw
            if (self.debug & 4) != 0:
                var tmp = row
                row = col
                col = tmp
            for i in range(half):
                var a = Float32(row if i < q else col) * inv[i if i < q else i - q]
                hc[t * half + i] = Float32(cos(Float64(a)))
                hs[t * half + i] = Float32(sin(Float64(a)))
        var dc = self.new(n * half)
        var ds = self.new(n * half)
        self.ctx.enqueue_copy(dc, hc)
        self.ctx.enqueue_copy(ds, hs)
        return (dc^, ds^)

    # ── the forward ──────────────────────────────────────────────────────

    def patchify(self, pic: Picture) raises -> DeviceBuffer[DT]:
        """`pixel_values` [N, 3 T P^2], as the processor writes them."""
        self.check(pic)
        var p = self.cfg.patch
        var gh = pic.height // p
        var gw = pic.width // p
        var n = gh * gw
        var per = 3 * self.cfg.tpatch * p * p
        var himg = self.ctx.enqueue_create_host_buffer[DType.uint8](len(pic.rgb))
        unsafe_memcpy(dest=himg.unsafe_ptr(), src=pic.rgb.unsafe_ptr(), count=len(pic.rgb))
        var dimg = self.ctx.enqueue_create_buffer[DType.uint8](len(pic.rgb))
        self.ctx.enqueue_copy(dimg, himg)
        var px = self.new(n * per)
        self.ctx.enqueue_function[patchify_kernel](
            dimg, px, Int64(pic.width), Int64(gw), Int64(p), Int64(self.cfg.tpatch), Int64(self.debug & 1),
            grid_dim=(_blocks(per), n), block_dim=TPB,
        )
        return px^

    def encode(mut self, pic: Picture, mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False) raises -> DeviceBuffer[DT]:
        """The picture's `<|image|>` features `[N / 4, text hidden]`. `taps`
        (with `keep_taps`): the embedding, each layer's output, the projector's."""
        var c = self.cfg.copy()
        var p = c.patch
        var gh = pic.height // p
        var gw = pic.width // p
        var n = gh * gw
        var dm = c.hidden
        var px = self.patchify(pic)
        var x = self.linear("tower.patchifier", px, n, 3 * c.tpatch * p * p)
        self.rms("tower.embedding_norm", x, x, n, dm)
        if keep_taps:
            taps.append(_copy(self.ctx, x, n * dm))
        var rt = self.rope_table(gh, gw)
        var sc = self.new(c.heads * n * n)
        for li in range(len(c.layer_types)):
            var pre = String("tower.layers.") + String(li) + "."
            var hn = self.new(n * dm)
            var hnb = self.rms_b(pre + "operator_norm", x, hn, n, dm)
            var o: DeviceBuffer[DT]
            if c.layer_types[li] == VT_CONV:
                var bcx = self.linear(pre + "conv.in_proj", hn, n, dm, hnb)
                var z = self.new(n * dm)
                self.ctx.enqueue_function[conv2d_gated_kernel](
                    bcx, self.w(pre + "conv.spatial_conv.weight"), z, Int64(dm), Int64(gh), Int64(gw),
                    Int64(self.debug & 2), grid_dim=(_blocks(dm), n), block_dim=TPB,
                )
                o = self.linear(pre + "conv.out_proj", z, n, dm)
            else:
                var hd = c.head_dim
                var q = self.linear(pre + "self_attn.q_proj", hn, n, dm, hnb)
                var k = self.linear(pre + "self_attn.k_proj", hn, n, dm, hnb)
                var v = self.linear(pre + "self_attn.v_proj", hn, n, dm, hnb)
                self.rms(pre + "self_attn.q_layernorm", q, q, n * c.heads, hd)
                self.rms(pre + "self_attn.k_layernorm", k, k, n * c.kv_heads, hd)
                self.ctx.enqueue_function[rope_kernel](
                    q, rt[0], rt[1], Int64(c.heads), Int64(hd // 2), grid_dim=(_blocks(c.heads * hd // 2), n), block_dim=TPB,
                )
                self.ctx.enqueue_function[rope_kernel](
                    k, rt[0], rt[1], Int64(c.kv_heads), Int64(hd // 2), grid_dim=(_blocks(c.kv_heads * hd // 2), n), block_dim=TPB,
                )
                attention_probs(self.ctx, q, k, sc, self.dummy, n, c.heads, c.kv_heads, hd, 3, 0, 0)
                var a = self.new(n * dm)
                attention_apply(self.ctx, sc, v, v, False, a, n, c.heads, c.kv_heads, hd)
                o = self.linear(pre + "self_attn.out_proj", a, n, dm)
            self.ctx.enqueue_function[add_kernel](x, o, Int64(n * dm), grid_dim=_blocks(n * dm), block_dim=TPB)
            var fnrm = self.new(n * dm)
            var fnb = self.rms_b(pre + "ffn_norm", x, fnrm, n, dm)
            var a1 = self.linear(pre + "feed_forward.w1", fnrm, n, dm, fnb)
            var a3 = self.linear(pre + "feed_forward.w3", fnrm, n, dm, fnb)
            var swb = Optional[DeviceBuffer[DType.bfloat16]](None)
            comptime if VIS_BF16:
                var b = self.ctx.enqueue_create_buffer[DType.bfloat16](n * c.ff)
                self.ctx.enqueue_function[swiglu_bf16_kernel](a1, a3, b, Int64(n * c.ff), grid_dim=_blocks(n * c.ff), block_dim=TPB)
                swb = Optional(b^)
            if not swb:
                self.ctx.enqueue_function[swiglu_kernel](a1, a3, a1, Int64(n * c.ff), grid_dim=_blocks(n * c.ff), block_dim=TPB)
            var f = self.linear(pre + "feed_forward.w2", a1, n, c.ff, swb)
            self.ctx.enqueue_function[add_kernel](x, f, Int64(n * dm), grid_dim=_blocks(n * dm), block_dim=TPB)
            if keep_taps:
                taps.append(_copy(self.ctx, x, n * dm))
        # the projector
        var ln = self.new(n * dm)
        self.ctx.enqueue_function[layer_norm_kernel](
            x, ln, self.w("proj.layer_norm.weight"), self.w("proj.layer_norm.bias"), Int64(dm), c.proj_eps,
            grid_dim=n, block_dim=TPB,
        )
        var m = n // (c.merge * c.merge)
        var wide = dm * c.merge * c.merge
        var h1 = self.linear("proj.linear_1", ln, m, wide)
        self.ctx.enqueue_function[bias_kernel](
            h1, self.w("proj.linear_1.bias"), Int64(c.proj_hidden), Int64(m * c.proj_hidden),
            grid_dim=_blocks(m * c.proj_hidden), block_dim=TPB,
        )
        self.ctx.enqueue_function[gelu_kernel](h1, Int64(m * c.proj_hidden), grid_dim=_blocks(m * c.proj_hidden), block_dim=TPB)
        var out = self.linear("proj.linear_2", h1, m, c.proj_hidden)
        self.ctx.enqueue_function[bias_kernel](
            out, self.w("proj.linear_2.bias"), Int64(c.text_hidden), Int64(m * c.text_hidden),
            grid_dim=_blocks(m * c.text_hidden), block_dim=TPB,
        )
        if keep_taps:
            taps.append(_copy(self.ctx, out, m * c.text_hidden))
        return out^

    def encode_all(mut self, pics: List[Picture]) raises -> DeviceBuffer[DT]:
        """Every picture's features, back to back in order (the `<|image|>`
        tokens of a prompt that holds them in this order)."""
        var total = 0
        for pic in pics:
            total += self.tokens(pic)
        var th = self.cfg.text_hidden
        var out = self.new(total * th)
        var at = 0
        var none = List[DeviceBuffer[DT]]()
        for pic in pics:
            var f = self.encode(pic, none)
            var m = self.tokens(pic)
            self.ctx.enqueue_copy(out.create_sub_buffer[DT](at * th, m * th), f.create_sub_buffer[DT](0, m * th))
            at += m
        return out^


def _copy(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DT]:
    var b = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(b, x.create_sub_buffer[DT](0, n))
    return b^


def _gemm_tb(ctx: DeviceContext, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int) raises:
    """`dst[m, n] = a[m, k] @ b[n, k]^T` in fp32 (TF32 with VIS_TF32; NVIDIA: cuBLAS; elsewhere MAX's matmul)."""
    comptime if default_accelerator().is_nvidia_gpu():
        cublas_gemm[False, True, VIS_TF32](ctx, dst, a, b, m, n, k, 0.0)
    else:
        var out_buf = dst
        var av = TileTensor(a, row_major(m, k))
        var ov = TileTensor(out_buf, row_major(m, n))
        var bv = TileTensor(b, row_major(n, k))
        max_matmul[transpose_b=True, target="gpu"](ov, av, bv, ctx)
