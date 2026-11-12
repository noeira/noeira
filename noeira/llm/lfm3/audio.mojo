# +--------------------------------------------------------------------------+ #
# | LFM3-Omni — the audio encoder and its projector (heard clips into the backbone)
# +--------------------------------------------------------------------------+ #
"""`Lfm3Audio` — LFM3-Omni's (and d1-Omni's) audio-input path, run from the same
checkpoint as the text backbone, verified against d1-Omni-2B's own code
(docs/NN_MODEL_PORTS_PLAN.md §12.6).

    var aud = Lfm3Audio.load(lm.ctx, dir)
    var feats = aud.encode_wave(samples, taps)              # [ceil(F / 8), 2048]: one row per <|audio|> token
    var h = lm.forward(ids, groups, state_len, taps2, audio_feats=Optional(feats), audio_token=aud.cfg.audio_token)

## The path (`modeling_lfm3_omni.py`)

1. The log-mel front end (`noeira.audio.lfm3_mel`, CPU): 16 kHz -> `[128, F]`.
2. `Lfm3OmniDownsampler`: `conv_in` (1 -> 256 channels, 3x3 stride 2) + SiLU;
   `conv_mid` (depthwise 3x3 stride 2) + `lin_mid` (1x1) + SiLU; `conv_end` +
   `lin_end` + SiLU, the same; flatten (channel, mel) = 256 x 16 = 4096 per time
   step; `lin_out` (1x1, 4096 -> 768). 128 mel bins x 8 frames -> one vector.
   Its padding masks are the identity for ONE clip (each stage's length is
   exactly its output size), and clips are encoded one at a time here.
3. An LFM2 stack (`audio_encoder.lfm`): 17 PRE-norm layers (not the text
   backbone's sandwich), no embedding:

       x = x + OP( operator_norm(x) ),  x = x + w2( silu(w1 n) * w3 n ),  n = ffn_norm(x)

   OP = the short conv (kernel 3, causal) or attention: 8 heads, 4 KV heads,
   head_dim 96, RMSNorm per head on q and k, RoPE (theta 1e6, the whole head),
   NO output gate, `out_proj`, under a SLIDING-WINDOW CAUSAL mask of 32 tokens
   (`i - 32 < j <= i`, 2.56 s). Then `embedding_norm` (the final norm).
4. `Lfm3OmniProjector`: LayerNorm(768) — torch's default eps 1e-5, NOT the
   config's `projector_layer_norm_eps` (1e-6, the vision projector's) —
   `linear_1` 768 -> 2048 (+ bias), exact GELU, `linear_2` 2048 -> 2048 (+ bias).

## The markup (d1's processor, not the config's names)

A clip is `<|vision_start|>` + one `<|audio|>` per token + `<|vision_end|>`
(125009 / 125031 / 125010) at the head of the user turn, like a picture. The
config's `audio_start_token_id` (125012) is not what d1's processor writes.

## `debug` (falsification only)

Bit 1: the attention causal without the window. Bit 2: the attention
bidirectional. Bit 4: the flatten mel-major (feature `m C + c`). Bit 8: no RoPE.
"""

from std.math import cos, sin, pow
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.audio.lfm3_mel import Lfm3LogMel
from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm_lp
from noeira.io.json import load_json
from noeira.io.safetensors import SafeTensors, ST_BF16
from .kernels import (
    TPB, bf16_to_f32_kernel, rms_kernel, rms_bf16_kernel, add_kernel, short_conv_kernel, rope_kernel,
    swiglu_kernel, swiglu_bf16_kernel, f32_to_bf16_kernel,
)
from .vision_kernels import layer_norm_kernel, bias_kernel, gelu_kernel
from .audio_kernels import audio_conv_in_kernel, audio_dw_conv_kernel, bias_silu_kernel, audio_flatten_kernel
from .model import LF_BF16, attention_probs, attention_apply, _blocks
from .vision import _gemm_tb, _copy


comptime AT_CONV: Int = 0
comptime AT_ATTN: Int = 1
comptime _ALIGN: Int = 64
comptime PROJ_LN_EPS: Float32 = 1e-5
"""`nn.LayerNorm(768)`'s default (module note)."""


struct Lfm3AudioConfig(Copyable, Movable):
    var hidden: Int
    var channels: Int
    var mels: Int
    var heads: Int
    var kv_heads: Int
    var head_dim: Int
    var conv_k: Int
    var window: Int
    var theta: Float64
    var eps: Float32
    var layer_types: List[Int]
    var text_hidden: Int
    var audio_token: Int

    def __init__(out self):
        self.hidden = 0
        self.channels = 0
        self.mels = 0
        self.heads = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.conv_k = 0
        self.window = 0
        self.theta = 1e6
        self.eps = 1e-5
        self.layer_types = List[Int]()
        self.text_hidden = 0
        self.audio_token = -1

    @staticmethod
    def load(dir: String) raises -> Lfm3AudioConfig:
        """`config.json`'s `audio_encoder_config` (and its LFM2 `text_config`)."""
        var doc = load_json(dir + "/config.json")
        var r = doc.root()
        var ac = doc.field(r, "audio_encoder_config")
        if ac < 0:
            raise Error("lfm3-omni: '" + dir + "/config.json' has no audio_encoder_config (not an omni checkpoint)")
        var lc = doc.field(ac, "text_config")
        if doc.string(doc.field(lc, "model_type")) != "lfm2":
            raise Error("lfm3-omni: only an LFM2 audio encoder is implemented")
        var c = Lfm3AudioConfig()
        c.hidden = doc.integer(doc.field(ac, "hidden_size"))
        c.channels = doc.integer(doc.field(ac, "conv_channels"))
        c.mels = doc.integer(doc.field(ac, "num_mel_bins"))
        c.window = doc.integer(doc.field(ac, "sliding_window"))
        c.heads = doc.integer(doc.field(lc, "num_attention_heads"))
        c.kv_heads = doc.integer(doc.field(lc, "num_key_value_heads"))
        c.head_dim = c.hidden // c.heads
        c.conv_k = doc.integer(doc.field(lc, "conv_L_cache"))
        c.eps = Float32(doc.number(doc.field(lc, "norm_eps")))
        var rp = doc.field(lc, "rope_parameters")
        if doc.string(doc.field(rp, "rope_type")) != "default":
            raise Error("lfm3-omni: only the default RoPE is implemented in the audio encoder")
        c.theta = doc.number(doc.field(rp, "rope_theta"))
        var lt = doc.field(lc, "layer_types")
        for i in range(doc.size(lt)):
            var s = doc.string(doc.at(lt, i))
            if s == "conv":
                c.layer_types.append(AT_CONV)
            elif s == "full_attention":
                c.layer_types.append(AT_ATTN)
            else:
                raise Error("lfm3-omni: unknown audio layer type '" + s + "'")
        if doc.boolean(doc.field(lc, "conv_bias")):
            raise Error("lfm3-omni: a biased short conv is not implemented")
        if c.mels % 8 != 0:
            raise Error("lfm3-omni: the mel bins must be a multiple of 8 (three stride-2 convolutions)")
        c.text_hidden = doc.integer(doc.field(doc.field(r, "text_config"), "hidden_size"))
        c.audio_token = doc.integer(doc.field(r, "audio_token_id"))
        return c^


struct Lfm3Audio(Movable):
    var ctx: DeviceContext
    var cfg: Lfm3AudioConfig
    var mel: Lfm3LogMel
    var arena: DeviceBuffer[DType.uint16]
    var index: Dict[String, Int]
    """`down.<name>` / `enc.<name>` / `proj.<name>` -> its slot in `dev`."""
    var dev: List[DeviceBuffer[DType.uint16]]
    var offs: List[Int]
    var used: List[Bool]
    var names: List[String]
    var scratch: DeviceBuffer[DT]
    var dummy: DeviceBuffer[DType.int32]
    """The attention's `groups` argument, unread outside mode 0."""
    var ff: Int
    var debug: Int

    def __init__(
        out self, var ctx: DeviceContext, var cfg: Lfm3AudioConfig, var arena: DeviceBuffer[DType.uint16],
        var index: Dict[String, Int], var dev: List[DeviceBuffer[DType.uint16]], var offs: List[Int],
        var names: List[String], var scratch: DeviceBuffer[DT], ff: Int,
    ) raises:
        self.dummy = ctx.enqueue_create_buffer[DType.int32](1)
        self.ctx = ctx^
        self.cfg = cfg^
        self.mel = Lfm3LogMel()
        self.arena = arena^
        self.index = index^
        self.dev = dev^
        self.offs = offs^
        self.used = List[Bool](length=len(self.dev), fill=False)
        self.names = names^
        self.scratch = scratch^
        self.ff = ff
        self.debug = 0

    @staticmethod
    def load(var ctx: DeviceContext, dir: String) raises -> Lfm3Audio:
        """The downsampler, the encoder and the projector of
        `dir/model.safetensors` into one bf16 device arena (`Lfm3Text.load`'s layout)."""
        var cfg = Lfm3AudioConfig.load(dir)
        var st = SafeTensors(dir + "/model.safetensors")
        var prefixes: List[String] = ["model.audio_encoder.downsample.", "model.audio_encoder.lfm.", "model.audio_projector."]
        var shorts: List[String] = ["down.", "enc.", "proj."]
        var picked = List[Int]()
        var short_of = List[String]()
        var total = 0
        var largest = 0
        var ff = 0
        for i in range(st.size()):
            var name = st.names[i]
            for p in range(len(prefixes)):
                if name.startswith(prefixes[p]):
                    if st.dtypes[i] != ST_BF16:
                        raise Error("lfm3-omni: " + name + " is not bf16")
                    var n = st.numel(name)
                    picked.append(i)
                    short_of.append(shorts[p] + String(name[byte=prefixes[p].byte_length():]))
                    total += (n + _ALIGN - 1) // _ALIGN * _ALIGN
                    largest = max(largest, n)
                    if name.endswith("layers.0.feed_forward.w1.weight"):
                        ff = n // cfg.hidden  # LFM2 sizes its FFN from `intermediate_size` by a rule: read it
        if len(picked) == 0:
            raise Error("lfm3-omni: no audio tensors in '" + dir + "/model.safetensors'")
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
        return Lfm3Audio(ctx^, cfg^, arena^, index^, dev^, offs^, names^, scratch^, ff)

    # ── plumbing ─────────────────────────────────────────────────────────

    def w(mut self, name: String) raises -> DeviceBuffer[DType.uint16]:
        var i = self.index.get(name)
        if not i:
            raise Error("lfm3-omni: no tensor '" + name + "'")
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
        """`x[t, k] @ W^T`, W the bf16 `name.weight` ([n, k], or a 1x1
        convolution's `[n, k, 1(, 1)]`): widened into the scratch, or
        (LFM3_BF16) on the tensor cores in place."""
        var wb = self.w(name + ".weight")
        var n = len(wb) // k
        var y = self.new(t * n)
        comptime if LF_BF16:
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

    def bias(mut self, name: String, x: DeviceBuffer[DT], rows: Int, cols: Int) raises:
        self.ctx.enqueue_function[bias_kernel](
            x, self.w(name + ".bias"), Int64(cols), Int64(rows * cols), grid_dim=_blocks(rows * cols), block_dim=TPB,
        )

    def bias_silu(mut self, name: String, x: DeviceBuffer[DT], rows: Int, cols: Int) raises:
        self.ctx.enqueue_function[bias_silu_kernel](
            x, self.w(name + ".bias"), Int64(cols), Int64(rows * cols), grid_dim=_blocks(rows * cols), block_dim=TPB,
        )

    def rms_b(
        mut self, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, dim: Int,
    ) raises -> Optional[DeviceBuffer[DType.bfloat16]]:
        comptime if LF_BF16:
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

    def rope_table(self, t: Int) raises -> Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]:
        """cos / sin `[t, head_dim / 2]`: frequencies `1 / theta^(2i / head_dim)`
        over the whole head, the angle `pos * freq` in fp32 (torch's). Bit 8: no
        rotation."""
        var half = self.cfg.head_dim // 2
        var hc = self.ctx.enqueue_create_host_buffer[DT](t * half)
        var hs = self.ctx.enqueue_create_host_buffer[DT](t * half)
        var inv = List[Float32](length=half, fill=0.0)
        for i in range(half):
            inv[i] = Float32(1.0) / Float32(pow(self.cfg.theta, Float64(Float32(2 * i) / Float32(self.cfg.head_dim))))
        for pos in range(t):
            for i in range(half):
                var a = Float32(pos) * inv[i] if (self.debug & 8) == 0 else Float32(0)
                hc[pos * half + i] = Float32(cos(Float64(a)))
                hs[pos * half + i] = Float32(sin(Float64(a)))
        var dc = self.new(t * half)
        var ds = self.new(t * half)
        self.ctx.enqueue_copy(dc, hc)
        self.ctx.enqueue_copy(ds, hs)
        return (dc^, ds^)

    # ── the clip ─────────────────────────────────────────────────────────

    def tokens(self, samples: Int) -> Int:
        """The `<|audio|>` tokens a clip of `samples` (16 kHz) takes."""
        return self.mel.tokens(samples)

    def markup(self, samples: Int, tok_audio: String, tok_start: String, tok_end: String) -> String:
        """The clip's text in a prompt, as d1's processor expands it."""
        var s = tok_start
        for _ in range(self.tokens(samples)):
            s += tok_audio
        return s + tok_end

    def encode_wave(mut self, ref samples: List[Float32], mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False) raises -> DeviceBuffer[DT]:
        """A 16 kHz mono clip's `<|audio|>` features (the front end, then `encode`)."""
        var feats = self.mel.extract(samples)
        return self.encode(feats, self.mel.frames(len(samples)), taps, keep_taps)

    def encode_all(mut self, clips: List[List[Float32]]) raises -> DeviceBuffer[DT]:
        """Every clip's features, back to back in order (the `<|audio|>` tokens
        of a prompt that holds them in this order)."""
        var total = 0
        for ref c in clips:
            total += self.tokens(len(c))
        var th = self.cfg.text_hidden
        var out = self.new(total * th)
        var at = 0
        var none = List[DeviceBuffer[DT]]()
        for ref c in clips:
            var f = self.encode_wave(c, none)
            var m = self.tokens(len(c))
            self.ctx.enqueue_copy(out.create_sub_buffer[DT](at * th, m * th), f.create_sub_buffer[DT](0, m * th))
            at += m
        return out^

    # ── the forward ──────────────────────────────────────────────────────

    def encode(
        mut self, ref mel: List[Float32], frames: Int, mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False,
    ) raises -> DeviceBuffer[DT]:
        """`mel` `[mels, frames]` (the front end's) -> `[ceil(frames / 8), text
        hidden]`. `taps` (with `keep_taps`): the downsampler's output, each
        encoder layer's, the final norm's, the projector's."""
        var c = self.cfg.copy()
        var ch = c.channels
        var dm = c.hidden
        if len(mel) != c.mels * frames:
            raise Error("lfm3-omni: " + String(len(mel)) + " mel values for " + String(c.mels) + " x " + String(frames))
        var hm = self.ctx.enqueue_create_host_buffer[DT](len(mel))
        for i in range(len(mel)):
            hm[i] = mel[i]
        var dmel = self.new(len(mel))
        self.ctx.enqueue_copy(dmel, hm)
        # the downsampler, position-major [mel, time, C]
        var h1 = (c.mels + 1) // 2
        var w1 = (frames + 1) // 2
        var a1 = self.new(h1 * w1 * ch)
        self.ctx.enqueue_function[audio_conv_in_kernel](
            dmel, self.w("down.conv_in.weight"), self.w("down.conv_in.bias"), a1, Int64(c.mels), Int64(frames), Int64(ch),
            grid_dim=(_blocks(ch), w1, h1), block_dim=TPB,
        )
        var h2 = (h1 + 1) // 2
        var w2 = (w1 + 1) // 2
        var a2 = self.new(h2 * w2 * ch)
        self.ctx.enqueue_function[audio_dw_conv_kernel](
            a1, self.w("down.conv_mid.weight"), self.w("down.conv_mid.bias"), a2, Int64(h1), Int64(w1), Int64(ch),
            grid_dim=(_blocks(ch), w2, h2), block_dim=TPB,
        )
        var b2 = self.linear("down.lin_mid", a2, h2 * w2, ch)
        self.bias_silu("down.lin_mid", b2, h2 * w2, ch)
        var h3 = (h2 + 1) // 2
        var t = (w2 + 1) // 2
        var a3 = self.new(h3 * t * ch)
        self.ctx.enqueue_function[audio_dw_conv_kernel](
            b2, self.w("down.conv_end.weight"), self.w("down.conv_end.bias"), a3, Int64(h2), Int64(w2), Int64(ch),
            grid_dim=(_blocks(ch), t, h3), block_dim=TPB,
        )
        var b3 = self.linear("down.lin_end", a3, h3 * t, ch)
        self.bias_silu("down.lin_end", b3, h3 * t, ch)
        var flat = self.new(t * ch * h3)
        self.ctx.enqueue_function[audio_flatten_kernel](
            b3, flat, Int64(h3), Int64(t), Int64(ch), Int64(1 if (self.debug & 4) != 0 else 0),
            grid_dim=(_blocks(ch), t, h3), block_dim=TPB,
        )
        var x = self.linear("down.lin_out", flat, t, ch * h3)
        self.bias("down.lin_out", x, t, dm)
        if keep_taps:
            taps.append(_copy(self.ctx, x, t * dm))
        # the LFM2 encoder
        var rt = self.rope_table(t)
        var sc = self.new(c.heads * t * t)
        var mode = 3 if (self.debug & 2) != 0 else (2 if (self.debug & 1) != 0 else 1)
        for li in range(len(c.layer_types)):
            var pre = String("enc.layers.") + String(li) + "."
            var hn = self.new(t * dm)
            var hnb = self.rms_b(pre + "operator_norm", x, hn, t, dm)
            var o: DeviceBuffer[DT]
            if c.layer_types[li] == AT_CONV:
                var bcx = self.linear(pre + "conv.in_proj", hn, t, dm, hnb)
                var z = self.new(t * dm)
                self.ctx.enqueue_function[short_conv_kernel](
                    bcx, self.w(pre + "conv.conv.weight"), z, Int64(dm), Int64(c.conv_k),
                    grid_dim=(_blocks(dm), t), block_dim=TPB,
                )
                o = self.linear(pre + "conv.out_proj", z, t, dm)
            else:
                var hd = c.head_dim
                var q = self.linear(pre + "self_attn.q_proj", hn, t, dm, hnb)
                var k = self.linear(pre + "self_attn.k_proj", hn, t, dm, hnb)
                var v = self.linear(pre + "self_attn.v_proj", hn, t, dm, hnb)
                self.rms(pre + "self_attn.q_layernorm", q, q, t * c.heads, hd)
                self.rms(pre + "self_attn.k_layernorm", k, k, t * c.kv_heads, hd)
                self.ctx.enqueue_function[rope_kernel](
                    q, rt[0], rt[1], Int64(c.heads), Int64(hd // 2), grid_dim=(_blocks(c.heads * hd // 2), t), block_dim=TPB,
                )
                self.ctx.enqueue_function[rope_kernel](
                    k, rt[0], rt[1], Int64(c.kv_heads), Int64(hd // 2), grid_dim=(_blocks(c.kv_heads * hd // 2), t), block_dim=TPB,
                )
                attention_probs(self.ctx, q, k, sc, self.dummy, t, c.heads, c.kv_heads, hd, mode, 0, c.window)
                var a = self.new(t * dm)
                attention_apply(self.ctx, sc, v, v, False, a, t, c.heads, c.kv_heads, hd)
                o = self.linear(pre + "self_attn.out_proj", a, t, dm)
            self.ctx.enqueue_function[add_kernel](x, o, Int64(t * dm), grid_dim=_blocks(t * dm), block_dim=TPB)
            var fnrm = self.new(t * dm)
            var fnb = self.rms_b(pre + "ffn_norm", x, fnrm, t, dm)
            var a1f = self.linear(pre + "feed_forward.w1", fnrm, t, dm, fnb)
            var a3f = self.linear(pre + "feed_forward.w3", fnrm, t, dm, fnb)
            var ff = self.ff
            var swb = Optional[DeviceBuffer[DType.bfloat16]](None)
            comptime if LF_BF16:
                var bb = self.ctx.enqueue_create_buffer[DType.bfloat16](t * ff)
                self.ctx.enqueue_function[swiglu_bf16_kernel](a1f, a3f, bb, Int64(t * ff), grid_dim=_blocks(t * ff), block_dim=TPB)
                swb = Optional(bb^)
            if not swb:
                self.ctx.enqueue_function[swiglu_kernel](a1f, a3f, a1f, Int64(t * ff), grid_dim=_blocks(t * ff), block_dim=TPB)
            var f = self.linear(pre + "feed_forward.w2", a1f, t, ff, swb)
            self.ctx.enqueue_function[add_kernel](x, f, Int64(t * dm), grid_dim=_blocks(t * dm), block_dim=TPB)
            if keep_taps:
                taps.append(_copy(self.ctx, x, t * dm))
        var xe = self.new(t * dm)
        self.rms("enc.embedding_norm", x, xe, t, dm)
        if keep_taps:
            taps.append(_copy(self.ctx, xe, t * dm))
        # the projector
        var th = c.text_hidden
        var ln = self.new(t * dm)
        self.ctx.enqueue_function[layer_norm_kernel](
            xe, ln, self.w("proj.layer_norm.weight"), self.w("proj.layer_norm.bias"), Int64(dm), PROJ_LN_EPS,
            grid_dim=t, block_dim=TPB,
        )
        var p1 = self.linear("proj.linear_1", ln, t, dm)
        self.bias("proj.linear_1", p1, t, th)
        self.ctx.enqueue_function[gelu_kernel](p1, Int64(t * th), grid_dim=_blocks(t * th), block_dim=TPB)
        var out = self.linear("proj.linear_2", p1, t, th)
        self.bias("proj.linear_2", out, t, th)
        if keep_taps:
            taps.append(_copy(self.ctx, out, t * th))
        return out^
