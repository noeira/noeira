# +--------------------------------------------------------------------------+ #
# | LFM3 — the language backbone on a device (d1, LFM3 text / VL / omni)
# +--------------------------------------------------------------------------+ #
"""`Lfm3Text` — the text decoder every LFM3 model shares, run from its
checkpoint (docs/NN_MODEL_PORTS_PLAN.md §12.1 has the architecture, verified
against d1's own code).

    var lm = Lfm3Text.load(DeviceContext(), dir)              # bf16 weights on the device
    var h = lm.forward(ids, groups, state_len)                # [T, 2048] after the final norm
    var logz = lm.slot_logz(h, len(ids), slots)               # log-softmax over the vocabulary at each slot

## The forward, per layer (`use_sandwich_norm`)

    h = x + post_operator_norm( OP( operator_norm(x) ) )      OP = short conv | attention
    x = h + post_ffn_norm( w2( silu(w1 n) * w3 n ) ),  n = ffn_norm(h)

then `embedding_norm` (the FINAL norm, despite its name). The short conv:
`in_proj` -> B, C, x; `C * dwconv_causal_7(B * x)`; `out_proj`. The attention:
q/k/v/gate projections, RMSNorm per head on q and k, rotate-half RoPE (sliding
layers: theta 1e4 on the whole head; full layers: theta 1e6 on HALF the
frequency slots, the "zero tail" — dims 0-31 with 64-95), softmax(q k / sqrt
128) under the layer's mask, `* sigmoid(gate)` per head, `o_proj`.

## Masks (d1's `set` attention)

Full-attention layers read the row under `set`: the first `state_len` tokens
see the whole row, and a question's tokens see the state and their OWN
question (`groups`: -1 on the state, k on question k), both ways. Sliding
layers: causal, 256 keys. Conv layers: causal by construction. A plain causal
LM (`state_len = 0`, no groups) is `causal_full = True`.

## Precision

Activations fp32. Weights stay bf16 on the device (the checkpoint's own type:
nothing is lost against an fp32 reference, which is the bf16 weights upcast)
and are widened into a scratch buffer for each GEMM. The GEMMs are fp32
(`-D LFM3_TF32=1`: TF32 on NVIDIA). A bf16-activation build is the next step.

## `debug` (falsification only)

Bit 1: the full layers causal instead of `set`. Bit 2: full RoPE on the full
layers (no zero tail). Bit 4: no output gate.
"""

from std.math import cos, sin, pow
from std.memory import unsafe_memcpy
from std.sys import default_accelerator
from std.sys.defines import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm
from noeira.io.json import load_json
from noeira.io.safetensors import SafeTensors, ST_BF16
from .kernels import (
    TPB, bf16_to_f32_kernel, embed_kernel, rms_kernel, add_kernel, short_conv_kernel, rope_kernel,
    scores_kernel, softmax_rows_kernel, attn_out_kernel, swiglu_kernel, gather_rows_kernel,
    lm_head_kernel, log_softmax_rows_kernel,
)


comptime LF_TF32: Bool = is_defined["LFM3_TF32"]()
comptime LT_CONV: Int = 0
comptime LT_FULL: Int = 1
comptime LT_SLIDING: Int = 2
comptime _ALIGN: Int = 64


def _blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


struct Lfm3Config(Copyable, Movable):
    var hidden: Int
    var heads: Int
    var kv_heads: Int
    var head_dim: Int
    var ff: Int
    var vocab: Int
    var conv_k: Int
    var window: Int
    var eps: Float32
    var layer_types: List[Int]
    var theta_full: Float64
    var prf_full: Float64
    var theta_sliding: Float64
    var prf_sliding: Float64

    def __init__(out self):
        self.hidden = 0
        self.heads = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.ff = 0
        self.vocab = 0
        self.conv_k = 0
        self.window = 0
        self.eps = 1e-5
        self.layer_types = List[Int]()
        self.theta_full = 1e6
        self.prf_full = 1.0
        self.theta_sliding = 1e4
        self.prf_sliding = 1.0

    @staticmethod
    def load(path: String) raises -> Lfm3Config:
        """`config.json`: its `text_config` when there is one (VL / omni / d1)."""
        var doc = load_json(path)
        var c = doc.root()
        var tc = doc.field(c, "text_config")
        if tc >= 0:
            c = tc
        var cfg = Lfm3Config()
        cfg.hidden = doc.integer(doc.field(c, "hidden_size"))
        cfg.heads = doc.integer(doc.field(c, "num_attention_heads"))
        cfg.kv_heads = doc.integer(doc.field(c, "num_key_value_heads"))
        var hd = doc.field(c, "head_dim")
        cfg.head_dim = doc.integer(hd) if hd >= 0 else cfg.hidden // cfg.heads
        cfg.ff = doc.integer(doc.field(c, "intermediate_size"))
        cfg.vocab = doc.integer(doc.field(c, "vocab_size"))
        cfg.conv_k = doc.integer(doc.field(c, "conv_L_cache"))
        cfg.window = doc.integer(doc.field(c, "sliding_window"))
        cfg.eps = Float32(doc.number(doc.field(c, "norm_eps")))
        var lt = doc.field(c, "layer_types")
        for i in range(doc.size(lt)):
            var s = doc.string(doc.at(lt, i))
            if s == "conv":
                cfg.layer_types.append(LT_CONV)
            elif s == "full_attention":
                cfg.layer_types.append(LT_FULL)
            elif s == "sliding_attention":
                cfg.layer_types.append(LT_SLIDING)
            else:
                raise Error("lfm3: unknown layer type '" + s + "'")
        var rp = doc.field(c, "rope_parameters")
        if rp >= 0:
            var f = doc.field(rp, "full_attention")
            if f >= 0:
                cfg.theta_full = doc.number(doc.field(f, "rope_theta"))
                var p = doc.field(f, "partial_rotary_factor")
                cfg.prf_full = doc.number(p) if p >= 0 else 1.0
            var s = doc.field(rp, "sliding_attention")
            if s >= 0:
                cfg.theta_sliding = doc.number(doc.field(s, "rope_theta"))
                var p = doc.field(s, "partial_rotary_factor")
                cfg.prf_sliding = doc.number(p) if p >= 0 else 1.0
        return cfg^


struct Lfm3Text(Movable):
    var ctx: DeviceContext
    var cfg: Lfm3Config
    var arena: DeviceBuffer[DType.uint16]
    var index: Dict[String, Int]
    """Tensor name (without the prefix) -> its slot in `dev`."""
    var dev: List[DeviceBuffer[DType.uint16]]
    var used: List[Bool]
    var names: List[String]
    var scratch: DeviceBuffer[DT]
    """A weight widened to fp32 for its GEMM (the largest weight's size)."""
    var debug: Int

    def __init__(
        out self, var ctx: DeviceContext, var cfg: Lfm3Config, var arena: DeviceBuffer[DType.uint16],
        var index: Dict[String, Int], var dev: List[DeviceBuffer[DType.uint16]], var names: List[String],
        var scratch: DeviceBuffer[DT],
    ):
        self.ctx = ctx^
        self.cfg = cfg^
        self.arena = arena^
        self.index = index^
        self.dev = dev^
        self.used = List[Bool](length=len(self.dev), fill=False)
        self.names = names^
        self.scratch = scratch^
        self.debug = 0

    @staticmethod
    def load(var ctx: DeviceContext, dir: String) raises -> Lfm3Text:
        """`config.json` and `model.safetensors` from `dir`: every language-model
        tensor (`model.language_model.*`, or `model.*` for a text-only model)
        into one bf16 device arena; the vision / audio parts are skipped."""
        var cfg = Lfm3Config.load(dir + "/config.json")
        var st = SafeTensors(dir + "/model.safetensors")
        var prefix = String("model.language_model.")
        if not st.has(prefix + "embed_tokens.weight"):
            prefix = String("model.")
        var picked = List[Int]()
        var total = 0
        var largest = 0
        for i in range(st.size()):
            var name = st.names[i]
            if not name.startswith(prefix) or (prefix == "model." and name.startswith("model.vision")):
                continue
            if st.dtypes[i] != ST_BF16:
                raise Error("lfm3: " + name + " is not bf16")
            var n = st.numel(name)
            picked.append(i)
            total += (n + _ALIGN - 1) // _ALIGN * _ALIGN
            if not name.endswith("embed_tokens.weight"):
                largest = max(largest, n)
        var arena = ctx.enqueue_create_buffer[DType.uint16](total)
        var staging = ctx.enqueue_create_host_buffer[DType.uint16](total)
        ctx.synchronize()
        var index = Dict[String, Int]()
        var dev = List[DeviceBuffer[DType.uint16]]()
        var names = List[String]()
        var off = 0
        for i in picked:
            var name = st.names[i]
            var raw = st.read_raw(name)
            var n = len(raw) // 2
            unsafe_memcpy(dest=staging.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_offset(off * 2), src=raw.unsafe_ptr(), count=len(raw))
            var short = String(name[byte=prefix.byte_length():])
            index[short] = len(dev)
            dev.append(arena.create_sub_buffer[DType.uint16](off, n))
            names.append(short)
            off += (n + _ALIGN - 1) // _ALIGN * _ALIGN
        ctx.enqueue_copy(arena, staging)
        ctx.synchronize()
        _ = staging^
        var scratch = ctx.enqueue_create_buffer[DT](largest)
        return Lfm3Text(ctx^, cfg^, arena^, index^, dev^, names^, scratch^)

    # ── plumbing ─────────────────────────────────────────────────────────

    def w(mut self, name: String) raises -> DeviceBuffer[DType.uint16]:
        var i = self.index.get(name)
        if not i:
            raise Error("lfm3: no tensor '" + name + "'")
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

    def gemm_tb(self, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int) raises:
        """`dst[m, n] = a[m, k] @ b[n, k]^T` (fp32; TF32 on NVIDIA with LFM3_TF32)."""
        comptime if default_accelerator().is_nvidia_gpu():
            cublas_gemm[False, True, LF_TF32](self.ctx, dst, a, b, m, n, k, 0.0)
        else:
            var out_buf = dst
            var av = TileTensor(a, row_major(m, k))
            var ov = TileTensor(out_buf, row_major(m, n))
            var bv = TileTensor(b, row_major(n, k))
            max_matmul[transpose_b=True, target="gpu"](ov, av, bv, self.ctx)

    def linear(mut self, name: String, x: DeviceBuffer[DT], t: Int, k: Int) raises -> DeviceBuffer[DT]:
        """`x[t, k] @ W^T` for the bf16 weight `name` ([n, k]), widened into the
        scratch first."""
        var wb = self.w(name + ".weight")
        var n = len(wb) // k
        var wf = self.scratch.create_sub_buffer[DT](0, n * k)
        self.ctx.enqueue_function[bf16_to_f32_kernel](wb, wf, Int64(n * k), grid_dim=_blocks(n * k), block_dim=TPB)
        var y = self.new(t * n)
        self.gemm_tb(y, x, wf, t, n, k)
        return y^

    def rms(mut self, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, dim: Int) raises:
        self.ctx.enqueue_function[rms_kernel](
            x, y, self.w(name + ".weight"), Int64(dim), self.cfg.eps, grid_dim=rows, block_dim=TPB,
        )

    def rope_tables(self, t: Int, theta: Float64, prf: Float64) raises -> Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]:
        """cos / sin `[t, head_dim / 2]`, as torch computes them in fp32: the
        frequencies `1 / theta^(2i / head_dim)` for i < prf head_dim / 2 and 0
        after (the zero tail), the angle `pos * freq` in fp32."""
        var half = self.cfg.head_dim // 2
        var rot = Int(prf * Float64(self.cfg.head_dim)) // 2
        var hc = self.ctx.enqueue_create_host_buffer[DT](t * half)
        var hs = self.ctx.enqueue_create_host_buffer[DT](t * half)
        var inv = List[Float32](length=half, fill=0.0)
        for i in range(rot):
            var p = Float32(pow(theta, Float64(Float32(2 * i) / Float32(self.cfg.head_dim))))
            inv[i] = Float32(1.0) / p
        for pos in range(t):
            for i in range(half):
                var a = Float32(pos) * inv[i]
                hc[pos * half + i] = Float32(cos(Float64(a)))
                hs[pos * half + i] = Float32(sin(Float64(a)))
        var dc = self.new(t * half)
        var ds = self.new(t * half)
        self.ctx.enqueue_copy(dc, hc)
        self.ctx.enqueue_copy(ds, hs)
        return (dc^, ds^)

    # ── the forward ──────────────────────────────────────────────────────

    def forward(
        mut self, ids: List[Int], groups: List[Int], state_len: Int,
        mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False, causal_full: Bool = False,
    ) raises -> DeviceBuffer[DT]:
        """`[T, hidden]` after the final norm. `taps` (with `keep_taps`): the
        embedding, then each layer's output."""
        var t = len(ids)
        var dm = self.cfg.hidden
        var hd = self.cfg.head_dim
        var nh = self.cfg.heads
        var nkv = self.cfg.kv_heads
        var hid = self.ctx.enqueue_create_host_buffer[DType.int32](t)
        var hg = self.ctx.enqueue_create_host_buffer[DType.int32](t)
        for i in range(t):
            hid[i] = Int32(ids[i])
            hg[i] = Int32(groups[i]) if len(groups) == t else Int32(-1)
        var did = self.ctx.enqueue_create_buffer[DType.int32](t)
        var dgr = self.ctx.enqueue_create_buffer[DType.int32](t)
        self.ctx.enqueue_copy(did, hid)
        self.ctx.enqueue_copy(dgr, hg)
        var x = self.new(t * dm)
        self.ctx.enqueue_function[embed_kernel](
            did, self.w("embed_tokens.weight"), x, Int64(dm), grid_dim=(_blocks(dm), t), block_dim=TPB,
        )
        if keep_taps:
            taps.append(self._copy(x, t * dm))
        var full_prf = 1.0 if (self.debug & 2) != 0 else self.cfg.prf_full
        var rf = self.rope_tables(t, self.cfg.theta_full, full_prf)
        var rs = self.rope_tables(t, self.cfg.theta_sliding, self.cfg.prf_sliding)
        var sc = self.new(nh * t * t)
        var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
        for li in range(len(self.cfg.layer_types)):
            var p = String("layers.") + String(li) + "."
            var kind = self.cfg.layer_types[li]
            var hn = self.new(t * dm)
            self.rms(p + "operator_norm", x, hn, t, dm)
            var o: DeviceBuffer[DT]
            if kind == LT_CONV:
                var bcx = self.linear(p + "conv.in_proj", hn, t, dm)
                var z = self.new(t * dm)
                self.ctx.enqueue_function[short_conv_kernel](
                    bcx, self.w(p + "conv.conv.weight"), z, Int64(dm), Int64(self.cfg.conv_k),
                    grid_dim=(_blocks(dm), t), block_dim=TPB,
                )
                o = self.linear(p + "conv.out_proj", z, t, dm)
            else:
                var q = self.linear(p + "self_attn.q_proj", hn, t, dm)
                var k = self.linear(p + "self_attn.k_proj", hn, t, dm)
                var v = self.linear(p + "self_attn.v_proj", hn, t, dm)
                var g = self.linear(p + "self_attn.gate_proj", hn, t, dm)
                self.rms(p + "self_attn.q_norm", q, q, t * nh, hd)
                self.rms(p + "self_attn.k_norm", k, k, t * nkv, hd)
                var cos_t = rf[0] if kind == LT_FULL else rs[0]
                var sin_t = rf[1] if kind == LT_FULL else rs[1]
                self.ctx.enqueue_function[rope_kernel](
                    q, cos_t, sin_t, Int64(nh), Int64(hd // 2), grid_dim=(_blocks(nh * hd // 2), t), block_dim=TPB,
                )
                self.ctx.enqueue_function[rope_kernel](
                    k, cos_t, sin_t, Int64(nkv), Int64(hd // 2), grid_dim=(_blocks(nkv * hd // 2), t), block_dim=TPB,
                )
                var mode = 1 if kind == LT_SLIDING else (2 if (causal_full or (self.debug & 1) != 0) else 0)
                self.ctx.enqueue_function[scores_kernel](
                    q, k, sc, dgr, Int64(t), Int64(nh), Int64(nkv), Int64(hd), scale, Int64(mode),
                    Int64(state_len), Int64(self.cfg.window), grid_dim=(_blocks(t), t, nh), block_dim=TPB,
                )
                self.ctx.enqueue_function[softmax_rows_kernel](sc, Int64(t), grid_dim=nh * t, block_dim=TPB)
                if (self.debug & 4) != 0:
                    g = self.new(t * nh)
                    g.enqueue_fill(30.0)  # sigmoid = 1: the gate off
                var a = self.new(t * dm)
                self.ctx.enqueue_function[attn_out_kernel](
                    sc, v, g, a, Int64(t), Int64(nh), Int64(nkv), Int64(hd),
                    grid_dim=(_blocks(nh * hd), t), block_dim=TPB,
                )
                o = self.linear(p + "self_attn.o_proj", a, t, dm)
            self.rms(p + "post_operator_norm", o, o, t, dm)
            self.ctx.enqueue_function[add_kernel](x, o, Int64(t * dm), grid_dim=_blocks(t * dm), block_dim=TPB)
            var fnrm = self.new(t * dm)
            self.rms(p + "ffn_norm", x, fnrm, t, dm)
            var a1 = self.linear(p + "feed_forward.w1", fnrm, t, dm)
            var a3 = self.linear(p + "feed_forward.w3", fnrm, t, dm)
            var ff = self.cfg.ff
            self.ctx.enqueue_function[swiglu_kernel](a1, a3, a1, Int64(t * ff), grid_dim=_blocks(t * ff), block_dim=TPB)
            var f = self.linear(p + "feed_forward.w2", a1, t, ff)
            self.rms(p + "post_ffn_norm", f, f, t, dm)
            self.ctx.enqueue_function[add_kernel](x, f, Int64(t * dm), grid_dim=_blocks(t * dm), block_dim=TPB)
            if keep_taps:
                taps.append(self._copy(x, t * dm))
        var out = self.new(t * dm)
        self.rms("embedding_norm", x, out, t, dm)
        return out^

    def slot_logz(mut self, h: DeviceBuffer[DT], t: Int, slots: List[Int]) raises -> List[Float32]:
        """`log_softmax(h[slot] @ embed^T)` for each slot: `[S, vocab]` on the host."""
        var s = len(slots)
        var dm = self.cfg.hidden
        var v = self.cfg.vocab
        var hs = self.ctx.enqueue_create_host_buffer[DType.int32](s)
        for i in range(s):
            hs[i] = Int32(slots[i])
        var ds = self.ctx.enqueue_create_buffer[DType.int32](s)
        self.ctx.enqueue_copy(ds, hs)
        var rows = self.new(s * dm)
        self.ctx.enqueue_function[gather_rows_kernel](h, ds, rows, Int64(dm), grid_dim=(_blocks(dm), s), block_dim=TPB)
        var logits = self.new(s * v)
        self.ctx.enqueue_function[lm_head_kernel](
            rows, self.w("embed_tokens.weight"), logits, Int64(v), Int64(dm), grid_dim=((v + 7) // 8, s), block_dim=256,
        )
        self.ctx.enqueue_function[log_softmax_rows_kernel](logits, Int64(v), grid_dim=s, block_dim=TPB)
        return self.download(logits, s * v)

    def _copy(self, x: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DT]:
        var b = self.new(n)
        self.ctx.enqueue_copy(b, x.create_sub_buffer[DT](0, n))
        return b^

    def download(self, x: DeviceBuffer[DT], n: Int) raises -> List[Float32]:
        var h = self.ctx.enqueue_create_host_buffer[DT](n)
        self.ctx.enqueue_copy(h, x.create_sub_buffer[DT](0, n))
        self.ctx.synchronize()
        var out = List[Float32](capacity=n)
        for i in range(n):
            out.append(h[i])
        return out^
