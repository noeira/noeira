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
nothing is lost against an fp32 reference, which is the bf16 weights upcast).
Default: each weight is widened into a scratch buffer and the GEMM is fp32
(`-D LFM3_TF32=1`: TF32 on NVIDIA) — the build that PROVES the code. With
`-D LFM3_BF16=1` (NVIDIA) a GEMM rounds its activations to bf16 and runs on the
tensor cores against the weights in place (`cublas_gemm_lp`, fp32
accumulation and output) — what torch and llama.cpp run d1 in. Its GEMM
operands are rounded where they are made (`rms_b`, `swiglu_bf16_kernel`), once
per input shared by several linears.

The attention on NVIDIA: `q k^T` and `P v` as cuBLAS GEMMs, one per KV head
(fp32; TF32 in the TF32 and bf16 builds, `LF_ATTN_TF32`), the mask and softmax
in one kernel (`attn_probs`, `attn_apply`). It was a dot product per thread:
59 % of a 685-token call on the Orin (634 ms), now ~20 ms.

## `debug` (falsification only)

Bit 1: the full layers causal instead of `set`. Bit 2: full RoPE on the full
layers (no zero tail). Bit 4: no output gate.
"""

from std.math import cos, sin, pow
from std.random import random_float64, seed as set_seed
from std.memory import unsafe_memcpy
from std.sys import default_accelerator
from std.sys.defines import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm, cublas_gemm_lp
from noeira.io.json import load_json
from noeira.io.safetensors import SafeTensors, ST_BF16
from .kernels import (
    TPB, bf16_to_f32_kernel, embed_kernel, rms_kernel, add_kernel, short_conv_kernel, rope_kernel,
    scores_kernel, softmax_rows_kernel, attn_out_kernel, swiglu_kernel, gather_rows_kernel,
    lm_head_kernel, log_softmax_rows_kernel, f32_to_bf16_kernel, split_heads_kernel, masked_softmax_kernel,
    merge_heads_gate_kernel, LM_SLOTS, rms_bf16_kernel, swiglu_bf16_kernel,
)
from .vision_kernels import put_rows_kernel
from .int8_kernels import (
    QPtr, q8_quantize_kernel, q8_gemv_kernel, q8_dequant_bf16_kernel, q8_dequant_f32_kernel, q8_embed_kernel,
    Q8_GEMV_MAX, Q8_RPW,
)


comptime LF_TF32: Bool = is_defined["LFM3_TF32"]()
comptime LF_BF16: Bool = is_defined["LFM3_BF16"]() and default_accelerator().is_nvidia_gpu()
comptime LF_ATTN_TF32: Bool = LF_TF32 or LF_BF16
"""The attention's two GEMMs (q k^T, P v) in TF32 in the fast builds; fp32
in the default one. TF32 keeps 10 mantissa bits to bf16's 7, and the scores
are not rounded to bf16 at all."""
comptime LF_INT8: Bool = is_defined["LFM3_INT8"]()
"""`-D LFM3_INT8=1`: the chat loads its weights as int8 (`Lfm3Text.quantize_int8`;
with LFM3_BF16 on the Orin). A default, not a switch inside the backbone."""
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


def q8_cols(cfg: Lfm3Config, name: String) -> Int:
    """The input width `k` of a weight `quantize_int8` / `load(int8=True)`
    quantizes (`[n, k]`), or 0: the linears (`*_proj`, the MLP's w1 / w2 / w3)
    and the tied table. Norms and the depthwise conv stay bf16 (tiny)."""
    if name.endswith("feed_forward.w2.weight"):
        return cfg.ff
    if (
        name == "embed_tokens.weight" or name.endswith("_proj.weight") or name.endswith("feed_forward.w1.weight")
        or name.endswith("feed_forward.w3.weight")
    ):
        return cfg.hidden
    return 0


struct Lfm3Text(Movable):
    var ctx: DeviceContext
    var cfg: Lfm3Config
    var arena: DeviceBuffer[DType.uint16]
    var index: Dict[String, Int]
    """Tensor name (without the prefix) -> its slot in `dev`."""
    var dev: List[DeviceBuffer[DType.uint16]]
    var offs: List[Int]
    """Each tensor's offset in the arena (elements)."""
    var used: List[Bool]
    var names: List[String]
    var scratch: DeviceBuffer[DT]
    """A weight widened to fp32 for its GEMM (the largest weight's size)."""
    var debug: Int
    var training: Bool
    """`forward` keeps what the backward needs (`act`, `act_x_last`) — see `train.mojo`."""
    var act: List[Dict[String, DeviceBuffer[DT]]]
    var act_x_last: List[DeviceBuffer[DT]]
    var cur: Dict[String, DeviceBuffer[DT]]
    var lora_index: Dict[String, Int]
    """Adapted linear (`layers.<i>.<module>`) -> its slot in the lists below."""
    var lora_a: List[DeviceBuffer[DT]]
    var lora_b: List[DeviceBuffer[DT]]
    var lora_da: List[DeviceBuffer[DT]]
    var lora_db: List[DeviceBuffer[DT]]
    var lora_shape: List[Tuple[Int, Int]]
    """(in, out) of each adapted linear."""
    var lora_rank: Int
    var lora_scale: Float32
    var q8_on: Bool
    """`quantize_int8` ran: the linears and the LM head read int8 weights (`int8_kernels.mojo`)."""
    var q8: DeviceBuffer[DType.int8]
    var q8_scale: DeviceBuffer[DT]
    var q8_index: Dict[String, Int]
    """Weight name -> its slot in the lists below."""
    var q8_off: List[Int]
    """Each weight's offset in `q8` (elements, 64-aligned)."""
    var q8_soff: List[Int]
    """Its scales' offset in `q8_scale`."""
    var q8_rows: List[Int]
    var q8_cols: List[Int]
    var q8_bf: DeviceBuffer[DType.bfloat16]
    """A weight dequantized to bf16 for a chunk's GEMM (LFM3_BF16; the fp32 build uses `scratch`)."""

    def __init__(
        out self, var ctx: DeviceContext, var cfg: Lfm3Config, var arena: DeviceBuffer[DType.uint16],
        var index: Dict[String, Int], var dev: List[DeviceBuffer[DType.uint16]], var offs: List[Int],
        var names: List[String], var scratch: DeviceBuffer[DT],
    ) raises:
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
        self.training = False
        self.act = List[Dict[String, DeviceBuffer[DT]]]()
        self.act_x_last = List[DeviceBuffer[DT]]()
        self.cur = Dict[String, DeviceBuffer[DT]]()
        self.lora_index = Dict[String, Int]()
        self.lora_a = List[DeviceBuffer[DT]]()
        self.lora_b = List[DeviceBuffer[DT]]()
        self.lora_da = List[DeviceBuffer[DT]]()
        self.lora_db = List[DeviceBuffer[DT]]()
        self.lora_shape = List[Tuple[Int, Int]]()
        self.lora_rank = 0
        self.lora_scale = 0.0
        self.q8_on = False
        self.q8 = self.ctx.enqueue_create_buffer[DType.int8](1)
        self.q8_scale = self.ctx.enqueue_create_buffer[DT](1)
        self.q8_index = Dict[String, Int]()
        self.q8_off = List[Int]()
        self.q8_soff = List[Int]()
        self.q8_rows = List[Int]()
        self.q8_cols = List[Int]()
        self.q8_bf = self.ctx.enqueue_create_buffer[DType.bfloat16](1)

    @staticmethod
    def load(var ctx: DeviceContext, dir: String, int8: Bool = False, q8_levels: Int = 127) raises -> Lfm3Text:
        """`config.json` and `model.safetensors` from `dir`: every language-model
        tensor (`model.language_model.*`, or `model.*` for a text-only model)
        into one bf16 device arena; the vision / audio parts are skipped.
        `int8`: the linears and the tied table go straight to int8 (one tensor
        at a time through a staging buffer) and are NOT kept in bf16 — half the
        memory; inference only (`quantize_int8` keeps both). `q8_levels`: the
        grid's half-width, 127 (falsification: 7 is a 4-bit grid)."""
        var cfg = Lfm3Config.load(dir + "/config.json")
        var st = SafeTensors(dir + "/model.safetensors")
        var prefix = String("model.language_model.")
        if not st.has(prefix + "embed_tokens.weight"):
            prefix = String("model.")
        var picked = List[Int]()
        var quant = List[Int]()
        var total = 0
        var largest = 0
        for i in range(st.size()):
            var name = st.names[i]
            if not name.startswith(prefix) or (prefix == "model." and name.startswith("model.vision")):
                continue
            if st.dtypes[i] != ST_BF16:
                raise Error("lfm3: " + name + " is not bf16")
            var n = st.numel(name)
            if not name.endswith("embed_tokens.weight"):
                largest = max(largest, n)
            if int8 and q8_cols(cfg, String(name[byte=prefix.byte_length():])) > 0:
                quant.append(i)
                continue
            picked.append(i)
            total += (n + _ALIGN - 1) // _ALIGN * _ALIGN
        var arena = ctx.enqueue_create_buffer[DType.uint16](max(total, 1))
        var staging = ctx.enqueue_create_host_buffer[DType.uint16](max(total, 1))
        ctx.synchronize()
        var index = Dict[String, Int]()
        var dev = List[DeviceBuffer[DType.uint16]]()
        var offs = List[Int]()
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
            offs.append(off)
            names.append(short)
            off += (n + _ALIGN - 1) // _ALIGN * _ALIGN
        ctx.enqueue_copy(arena, staging)
        ctx.synchronize()
        _ = staging^
        var scratch = ctx.enqueue_create_buffer[DT](largest)
        var lm = Lfm3Text(ctx^, cfg^, arena^, index^, dev^, offs^, names^, scratch^)
        if int8:
            var qn = List[String]()
            for i in quant:
                qn.append(st.names[i])
            lm._load_q8(st, prefix, qn, q8_levels)
        return lm^

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

    def gemm_acc(
        self, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int,
        alpha: Float32, beta: Float32,
    ) raises:
        """`dst = alpha a[m, k] @ b[n, k]^T + beta dst` (NVIDIA: the LoRA path)."""
        comptime if default_accelerator().is_nvidia_gpu():
            cublas_gemm[False, True, LF_TF32](self.ctx, dst, a, b, m, n, k, beta, alpha)
        else:
            raise Error("lfm3: LoRA and training run on NVIDIA (cuBLAS) only")

    def load_lora(mut self, path: String, rank: Int, scale: Float32) raises:
        """Adapters from a safetensors file of `layers.<i>.<module>.lora_A`
        [rank, in] and `.lora_B` [out, rank] (fp32), PEFT's convention:
        `y = x W^T + scale (x A^T) B^T`. Gradients start at zero."""
        var st = SafeTensors(path)
        self.lora_rank = rank
        self.lora_scale = scale
        for i in range(st.size()):
            var name = st.names[i]
            if not name.endswith(".lora_A"):
                continue
            var key = String(name[byte=0:name.byte_length() - 7])
            if key + ".weight" not in self.index:
                raise Error("lfm3: an adapter for '" + key + "', which is not a weight here")
            var sa = st.shape(name)
            var sb = st.shape(key + ".lora_B")
            if sa[0] != rank or sb[1] != rank:
                raise Error("lfm3: adapter '" + key + "' is not rank " + String(rank))
            self.lora_index[key] = len(self.lora_a)
            self.lora_a.append(self._upload(st.read_f32(name)))
            self.lora_b.append(self._upload(st.read_f32(key + ".lora_B")))
            var da = self.new(rank * sa[1])
            da.enqueue_fill(0.0)
            var db = self.new(sb[0] * rank)
            db.enqueue_fill(0.0)
            self.lora_da.append(da^)
            self.lora_db.append(db^)
            self.lora_shape.append((sa[1], sb[0]))

    def init_lora(mut self, rank: Int, alpha: Float32, seed: Int = 0) raises:
        """Fresh adapters on every linear Liquid's recipe adapts (q k v o,
        in_proj out_proj, w1 w2 w3), PEFT's initialisation: A uniform in
        +-1/sqrt(in) (kaiming_uniform with a = sqrt 5), B ZERO — the adapted
        model starts exactly as the base one. Scale `alpha / rank`."""
        set_seed(seed)
        self.lora_index = Dict[String, Int]()
        self.lora_a = List[DeviceBuffer[DT]]()
        self.lora_b = List[DeviceBuffer[DT]]()
        self.lora_da = List[DeviceBuffer[DT]]()
        self.lora_db = List[DeviceBuffer[DT]]()
        self.lora_shape = List[Tuple[Int, Int]]()
        self.lora_rank = rank
        self.lora_scale = alpha / Float32(rank)
        var dm = self.cfg.hidden
        for li in range(len(self.cfg.layer_types)):
            var p = String("layers.") + String(li) + "."
            var mods = List[String]()
            if self.cfg.layer_types[li] == LT_CONV:
                mods.append(p + "conv.in_proj")
                mods.append(p + "conv.out_proj")
            else:
                for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
                    mods.append(p + String(m))
            for m in ["feed_forward.w1", "feed_forward.w3", "feed_forward.w2"]:
                mods.append(p + String(m))
            for key in mods:
                var n_el = len(self.dev[self.index[key + ".weight"]])
                var fan_in = self.cfg.ff if key.endswith("w2") else dm
                var fan_out = n_el // fan_in
                var bound = 1.0 / (Float64(fan_in) ** 0.5)
                var a = List[Float32](capacity=rank * fan_in)
                for _ in range(rank * fan_in):
                    a.append(Float32((random_float64() * 2.0 - 1.0) * bound))
                self.lora_index[key] = len(self.lora_a)
                self.lora_a.append(self._upload(a))
                var b = self.new(fan_out * rank)
                b.enqueue_fill(0.0)
                self.lora_b.append(b^)
                var da = self.new(rank * fan_in)
                da.enqueue_fill(0.0)
                var db = self.new(fan_out * rank)
                db.enqueue_fill(0.0)
                self.lora_da.append(da^)
                self.lora_db.append(db^)
                self.lora_shape.append((fan_in, fan_out))

    def _upload(self, v: List[Float32]) raises -> DeviceBuffer[DT]:
        var h = self.ctx.enqueue_create_host_buffer[DT](len(v))
        for i in range(len(v)):
            h[i] = v[i]
        var d = self.new(len(v))
        self.ctx.enqueue_copy(d, h)
        return d^

    def linear(
        mut self, name: String, x: DeviceBuffer[DT], t: Int, k: Int,
        xb: Optional[DeviceBuffer[DType.bfloat16]] = None,
    ) raises -> DeviceBuffer[DT]:
        """`x[t, k] @ W^T` (+ the adapter's term). `xb`: `x` already rounded to
        bf16 (LFM3_BF16; `rms_b`), shared by the linears that read one input."""
        var y = self._linear_base(name, x, t, k, xb)
        var li = self.lora_index.get(name)
        if li:
            var j = li.value()
            var r = self.lora_rank
            var n = self.lora_shape[j][1]
            var xa = self.new(t * r)
            self.gemm_acc(xa, x, self.lora_a[j], t, r, k, 1.0, 0.0)
            self.gemm_acc(y, xa, self.lora_b[j], t, n, r, self.lora_scale, 1.0)
            if self.training:
                self.cur["xa:" + name] = xa
        return y^

    def _linear_base(
        mut self, name: String, x: DeviceBuffer[DT], t: Int, k: Int,
        xb: Optional[DeviceBuffer[DType.bfloat16]] = None,
    ) raises -> DeviceBuffer[DT]:
        """`x[t, k] @ W^T` for the bf16 weight `name` ([n, k]): widened into the
        scratch first, or (LFM3_BF16) against the weight in place; after
        `quantize_int8`, from its int8 copy (`_linear_q8`)."""
        if self.q8_on:
            var qi = self.q8_index.get(name + ".weight")
            if qi:
                return self._linear_q8(qi.value(), x, t, k, xb)
        var wb = self.w(name + ".weight")
        var n = len(wb) // k
        comptime if LF_BF16:
            var xbb: DeviceBuffer[DType.bfloat16]
            if xb:
                xbb = xb.value()
            else:
                xbb = self.to_bf16(x, t * k)
            var yb = self.new(t * n)
            cublas_gemm_lp[False, True, DType.bfloat16, DT](self.ctx, yb, xbb, self.weight_bf16(name), t, n, k, 0.0)
            return yb^
        var wf = self.scratch.create_sub_buffer[DT](0, n * k)
        self.ctx.enqueue_function[bf16_to_f32_kernel](wb, wf, Int64(n * k), grid_dim=_blocks(n * k), block_dim=TPB)
        var y = self.new(t * n)
        self.gemm_tb(y, x, wf, t, n, k)
        return y^

    # ── int8 weights ─────────────────────────────────────────────────────

    def _q8_layout(mut self, names: List[String], numels: List[Int]) raises -> Int:
        """Slots, offsets and shapes for the int8 copies of `names`; allocates
        `q8`, `q8_scale` and the bf16 dequantization scratch. Returns the
        largest weight's size (elements)."""
        var total = 0
        var rows_total = 0
        var largest = 0
        var largest_lin = 0
        for i in range(len(names)):
            var k = q8_cols(self.cfg, names[i])
            var n = numels[i] // k
            if n * k != numels[i] or k % 512 != 0 or n % Q8_RPW != 0:
                raise Error("lfm3 int8: " + names[i] + " is not [n, " + String(k) + "] with k a multiple of 512 and n of " + String(Q8_RPW))
            self.q8_index[names[i]] = len(self.q8_off)
            self.q8_off.append(total)
            self.q8_soff.append(rows_total)
            self.q8_rows.append(n)
            self.q8_cols.append(k)
            total += (n * k + _ALIGN - 1) // _ALIGN * _ALIGN
            rows_total += n
            largest = max(largest, n * k)
            if names[i] != "embed_tokens.weight":
                largest_lin = max(largest_lin, n * k)
        self.q8 = self.ctx.enqueue_create_buffer[DType.int8](total)
        self.q8_scale = self.ctx.enqueue_create_buffer[DT](rows_total)
        comptime if LF_BF16:
            self.q8_bf = self.ctx.enqueue_create_buffer[DType.bfloat16](largest_lin)
        return largest

    def _q8_from(mut self, j: Int, w: DeviceBuffer[DType.uint16], levels: Int = 127) raises:
        """Quantize the bf16 `w` into int8 slot `j`."""
        var n = self.q8_rows[j]
        var k = self.q8_cols[j]
        self.ctx.enqueue_function[q8_quantize_kernel](
            w, self.q8.create_sub_buffer[DType.int8](self.q8_off[j], n * k),
            self.q8_scale.create_sub_buffer[DT](self.q8_soff[j], n), Int64(k), Int64(levels), grid_dim=n, block_dim=TPB,
        )

    def _load_q8(mut self, ref st: SafeTensors, prefix: String, full_names: List[String], levels: Int = 127) raises:
        """`load(int8=True)`: each tensor read, uploaded to a bf16 staging buffer,
        quantized, one at a time (the bf16 copy never stays)."""
        var names = List[String]()
        var numels = List[Int]()
        for nm in full_names:
            names.append(String(nm[byte=prefix.byte_length():]))
            numels.append(st.numel(nm))
        var largest = self._q8_layout(names, numels)
        var dstage = self.ctx.enqueue_create_buffer[DType.uint16](largest)
        for i in range(len(names)):
            var raw = st.read_raw(full_names[i])
            var n = len(raw) // 2
            var h = self.ctx.enqueue_create_host_buffer[DType.uint16](n)
            self.ctx.synchronize()
            unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=raw.unsafe_ptr(), count=len(raw))
            var d = dstage.create_sub_buffer[DType.uint16](0, n)
            self.ctx.enqueue_copy(d, h)
            self._q8_from(self.q8_index.get(names[i]).value(), d, levels)
            self.ctx.synchronize()  # the staging buffers are reused / freed
        self.q8_on = True

    def q8_bytes(self) -> Int:
        """Device bytes of the int8 weights and their scales (0 before int8)."""
        if not self.q8_on:
            return 0
        return len(self.q8) + 4 * len(self.q8_scale)

    def bf16_bytes(self) -> Int:
        return 2 * len(self.arena)

    def embed_rows(mut self, ids: DeviceBuffer[DType.int32], x: DeviceBuffer[DT], t: Int) raises:
        """`x[t, :]` = the embeddings of `ids`: from the bf16 table, or the int8
        one when the model was loaded int8 (no bf16 table then)."""
        var dm = self.cfg.hidden
        if not self.index.get("embed_tokens.weight") and self.q8_on:
            var j = self.q8_index.get("embed_tokens.weight").value()
            self.ctx.enqueue_function[q8_embed_kernel](
                ids, self.q8.create_sub_buffer[DType.int8](self.q8_off[j], self.q8_rows[j] * dm),
                self.q8_scale.create_sub_buffer[DT](self.q8_soff[j], self.q8_rows[j]), x, Int64(dm),
                grid_dim=(_blocks(dm), t), block_dim=TPB,
            )
            return
        self.ctx.enqueue_function[embed_kernel](
            ids, self.w("embed_tokens.weight"), x, Int64(dm), grid_dim=(_blocks(dm), t), block_dim=TPB,
        )

    def quantize_int8(mut self) raises:
        """Every linear's weight and the tied table to int8, one scale per row
        (`int8_kernels.mojo`), on the device from the bf16 arena — which STAYS
        (the embedding lookup reads it; the backward reads the bf16 weights).
        `load(int8=True)` is the same quantization without the bf16 copy."""
        var names = List[String]()
        var numels = List[Int]()
        for i in range(len(self.names)):
            if q8_cols(self.cfg, self.names[i]) > 0:
                names.append(self.names[i])
                numels.append(len(self.dev[i]))
        _ = self._q8_layout(names, numels)
        for nm in names:
            self._q8_from(self.q8_index.get(nm).value(), self.w(nm))
        self.ctx.synchronize()
        self.q8_on = True

    def _q8_gemv(self, j: Int, x: DeviceBuffer[DT], y: DeviceBuffer[DT], t: Int) raises:
        """`y[t, n] = x[t, k] @ W^T` from weight slot `j`'s int8 rows, `t` <= Q8_GEMV_MAX."""
        var n = self.q8_rows[j]
        var k = self.q8_cols[j]
        var q = self.q8.create_sub_buffer[DType.int8](self.q8_off[j], n * k)
        var sc = self.q8_scale.create_sub_buffer[DT](self.q8_soff[j], n)
        var g = (n + 8 * Q8_RPW - 1) // (8 * Q8_RPW)
        if t == 1:
            self.ctx.enqueue_function[q8_gemv_kernel[1]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
        elif t == 2:
            self.ctx.enqueue_function[q8_gemv_kernel[2]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
        elif t == 3:  # 2 + 1 (SIMD widths are powers of two)
            self.ctx.enqueue_function[q8_gemv_kernel[2]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
            self.ctx.enqueue_function[q8_gemv_kernel[1]](
                x.create_sub_buffer[DT](2 * k, k), q, sc, y.create_sub_buffer[DT](2 * n, n), Int64(n), Int64(k),
                grid_dim=g, block_dim=256,
            )
        else:
            self.ctx.enqueue_function[q8_gemv_kernel[4]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)

    def _linear_q8(
        mut self, j: Int, x: DeviceBuffer[DT], t: Int, k: Int, xb: Optional[DeviceBuffer[DType.bfloat16]],
    ) raises -> DeviceBuffer[DT]:
        """The int8 route: the GEMV for a few rows; for a chunk, the weight
        dequantized (bf16 / fp32) and the build's GEMM."""
        var n = self.q8_rows[j]
        if k != self.q8_cols[j]:
            raise Error("lfm3 int8: a [" + String(t) + ", " + String(k) + "] input for a weight of " + String(self.q8_cols[j]) + " columns")
        var y = self.new(t * n)
        if t <= Q8_GEMV_MAX:
            self._q8_gemv(j, x, y, t)
            return y^
        var q = self.q8.create_sub_buffer[DType.int8](self.q8_off[j], n * k)
        var sc = self.q8_scale.create_sub_buffer[DT](self.q8_soff[j], n)
        comptime if LF_BF16:
            var wb = self.q8_bf.create_sub_buffer[DType.bfloat16](0, n * k)
            self.ctx.enqueue_function[q8_dequant_bf16_kernel](q, sc, wb, Int64(k), grid_dim=(_blocks(k // 16), n), block_dim=TPB)
            var xbb: DeviceBuffer[DType.bfloat16]
            if xb:
                xbb = xb.value()
            else:
                xbb = self.to_bf16(x, t * k)
            cublas_gemm_lp[False, True, DType.bfloat16, DT](self.ctx, y, xbb, wb, t, n, k, 0.0)
            return y^
        var wf = self.scratch.create_sub_buffer[DT](0, n * k)
        self.ctx.enqueue_function[q8_dequant_f32_kernel](q, sc, wf, Int64(k), grid_dim=(_blocks(k // 16), n), block_dim=TPB)
        self.gemm_tb(y, x, wf, t, n, k)
        return y^

    def weight_bf16(mut self, name: String) raises -> DeviceBuffer[DType.bfloat16]:
        """The bf16 weight `name` in place, typed for cuBLAS: a non-owning view
        of its slice of the arena (the forward's and the backward's GEMMs)."""
        var wb = self.w(name + ".weight")
        var i = self.index.get(name + ".weight").value()
        return DeviceBuffer[DType.bfloat16](
            self.ctx, self.arena.unsafe_ptr().bitcast[Scalar[DType.bfloat16]]() + self.offs[i], len(wb), owning=False,
        )

    def to_bf16(self, x: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DType.bfloat16]:
        """`x` rounded to bf16 (nearest even), for a tensor-core GEMM operand."""
        var xb = self.ctx.enqueue_create_buffer[DType.bfloat16](n)
        self.ctx.enqueue_function[f32_to_bf16_kernel](x, xb, Int64(n), grid_dim=_blocks(n), block_dim=TPB)
        return xb^

    def rms_b(
        mut self, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, dim: Int,
    ) raises -> Optional[DeviceBuffer[DType.bfloat16]]:
        """`rms`, and (LFM3_BF16) its output rounded to bf16 by the same kernel,
        for the GEMMs that read it; None in the other builds."""
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

    def attn_probs(
        self, q: DeviceBuffer[DT], k: DeviceBuffer[DT], sc: DeviceBuffer[DT], groups: DeviceBuffer[DType.int32],
        t: Int, mode: Int, state_len: Int,
    ) raises:
        attention_probs(
            self.ctx, q, k, sc, groups, t, self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim, mode, state_len,
            self.cfg.window,
        )

    def attn_apply(
        self, sc: DeviceBuffer[DT], v: DeviceBuffer[DT], gate: DeviceBuffer[DT], dst: DeviceBuffer[DT], t: Int,
    ) raises:
        attention_apply(self.ctx, sc, v, gate, True, dst, t, self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim)

    def rope_tables(
        self, t: Int, theta: Float64, prf: Float64, neg_sin: Bool = False,
    ) raises -> Tuple[DeviceBuffer[DT], DeviceBuffer[DT]]:
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
                var sv = Float32(sin(Float64(a)))
                hs[pos * half + i] = -sv if neg_sin else sv  # -sin: the rotation's transpose (its VJP)
        var dc = self.new(t * half)
        var ds = self.new(t * half)
        self.ctx.enqueue_copy(dc, hc)
        self.ctx.enqueue_copy(ds, hs)
        return (dc^, ds^)

    # ── the forward ──────────────────────────────────────────────────────

    def forward(
        mut self, ids: List[Int], groups: List[Int], state_len: Int,
        mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False, causal_full: Bool = False,
        image_feats: Optional[DeviceBuffer[DT]] = None, image_token: Int = -1,
        audio_feats: Optional[DeviceBuffer[DT]] = None, audio_token: Int = -1,
    ) raises -> DeviceBuffer[DT]:
        """`[T, hidden]` after the final norm. `taps` (with `keep_taps`): the
        embedding, then each layer's output. `image_feats` (LFM3-VL,
        `Lfm3Vision.encode_all`): one row per `image_token` in `ids`, in order,
        written over those tokens' embeddings (transformers' `masked_scatter`);
        `audio_feats` (LFM3-Omni, `Lfm3Audio.encode_all`) the same over the
        `audio_token`s."""
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
        self.embed_rows(did, x, t)
        if image_feats:
            self._scatter(x, ids, image_feats.value(), image_token, "image")
        if audio_feats:
            self._scatter(x, ids, audio_feats.value(), audio_token, "audio")
        if keep_taps:
            taps.append(self._copy(x, t * dm))
        var full_prf = 1.0 if (self.debug & 2) != 0 else self.cfg.prf_full
        var rf = self.rope_tables(t, self.cfg.theta_full, full_prf)
        var rs = self.rope_tables(t, self.cfg.theta_sliding, self.cfg.prf_sliding)
        var sc = self.new(nh * t * t)
        if self.training:
            self.act = List[Dict[String, DeviceBuffer[DT]]]()
            self.act_x_last = List[DeviceBuffer[DT]]()
        for li in range(len(self.cfg.layer_types)):
            var p = String("layers.") + String(li) + "."
            var kind = self.cfg.layer_types[li]
            self.cur = Dict[String, DeviceBuffer[DT]]()
            if self.training:
                self.cur["x_in"] = self._copy(x, t * dm)
            var hn = self.new(t * dm)
            var hnb = self.rms_b(p + "operator_norm", x, hn, t, dm)
            if self.training:
                self.cur["hn"] = hn
            var o: DeviceBuffer[DT]
            if kind == LT_CONV:
                var bcx = self.linear(p + "conv.in_proj", hn, t, dm, hnb)
                var z = self.new(t * dm)
                self.ctx.enqueue_function[short_conv_kernel](
                    bcx, self.w(p + "conv.conv.weight"), z, Int64(dm), Int64(self.cfg.conv_k),
                    grid_dim=(_blocks(dm), t), block_dim=TPB,
                )
                if self.training:
                    self.cur["bcx"] = bcx
                    self.cur["z"] = z
                o = self.linear(p + "conv.out_proj", z, t, dm)
            else:
                var q = self.linear(p + "self_attn.q_proj", hn, t, dm, hnb)
                var k = self.linear(p + "self_attn.k_proj", hn, t, dm, hnb)
                var v = self.linear(p + "self_attn.v_proj", hn, t, dm, hnb)
                var g = self.linear(p + "self_attn.gate_proj", hn, t, dm, hnb)
                if self.training:
                    self.cur["q_lin"] = self._copy(q, t * dm)
                    self.cur["k_lin"] = self._copy(k, t * nkv * hd)
                    self.cur["v"] = v
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
                self.attn_probs(q, k, sc, dgr, t, mode, state_len)
                if self.training:
                    self.cur["q"] = q
                    self.cur["k"] = k
                    self.cur["P"] = self._copy(sc, nh * t * t)
                    self.cur["g"] = g
                    var ones = self.new(t * nh)
                    ones.enqueue_fill(30.0)  # sigmoid(30) = 1 - 9e-14: the output before its gate
                    var a_pre = self.new(t * dm)
                    self.attn_apply(sc, v, ones, a_pre, t)
                    self.cur["a_pre"] = a_pre
                if (self.debug & 4) != 0:
                    g = self.new(t * nh)
                    g.enqueue_fill(30.0)  # sigmoid = 1: the gate off
                var a = self.new(t * dm)
                self.attn_apply(sc, v, g, a, t)
                if self.training:
                    self.cur["a"] = a
                o = self.linear(p + "self_attn.o_proj", a, t, dm)
            if self.training:
                self.cur["o"] = self._copy(o, t * dm)
            self.rms(p + "post_operator_norm", o, o, t, dm)
            self.ctx.enqueue_function[add_kernel](x, o, Int64(t * dm), grid_dim=_blocks(t * dm), block_dim=TPB)
            if self.training:
                self.cur["h_mid"] = self._copy(x, t * dm)
            var fnrm = self.new(t * dm)
            var fnb = self.rms_b(p + "ffn_norm", x, fnrm, t, dm)
            if self.training:
                self.cur["fn"] = fnrm
            var a1 = self.linear(p + "feed_forward.w1", fnrm, t, dm, fnb)
            var a3 = self.linear(p + "feed_forward.w3", fnrm, t, dm, fnb)
            var ff = self.cfg.ff
            if self.training:
                self.cur["a1"] = self._copy(a1, t * ff)
                self.cur["a3"] = a3
            var swb = Optional[DeviceBuffer[DType.bfloat16]](None)
            comptime if LF_BF16:
                if not self.training and len(self.lora_index) == 0 and not self.q8_on:  # nothing reads silu(a1) a3 in fp32 (the int8 GEMV does)
                    var b = self.ctx.enqueue_create_buffer[DType.bfloat16](t * ff)
                    self.ctx.enqueue_function[swiglu_bf16_kernel](a1, a3, b, Int64(t * ff), grid_dim=_blocks(t * ff), block_dim=TPB)
                    swb = Optional(b^)
            if not swb:
                self.ctx.enqueue_function[swiglu_kernel](a1, a3, a1, Int64(t * ff), grid_dim=_blocks(t * ff), block_dim=TPB)
            if self.training:
                self.cur["sw"] = a1
            var f = self.linear(p + "feed_forward.w2", a1, t, ff, swb)
            if self.training:
                self.cur["f"] = self._copy(f, t * dm)
            self.rms(p + "post_ffn_norm", f, f, t, dm)
            self.ctx.enqueue_function[add_kernel](x, f, Int64(t * dm), grid_dim=_blocks(t * dm), block_dim=TPB)
            if keep_taps:
                taps.append(self._copy(x, t * dm))
            if self.training:
                self.act.append(self.cur^)
                self.cur = Dict[String, DeviceBuffer[DT]]()
        if self.training:
            self.act_x_last.append(self._copy(x, t * dm))
        var out = self.new(t * dm)
        self.rms("embedding_norm", x, out, t, dm)
        return out^

    def _scatter(self, x: DeviceBuffer[DT], ids: List[Int], feats: DeviceBuffer[DT], token: Int, what: String) raises:
        """`x[i] = feats[k]` for the k-th `token` in `ids` (a media encoder's rows
        over its placeholders' embeddings)."""
        var dm = self.cfg.hidden
        var at = List[Int]()
        for i in range(len(ids)):
            if ids[i] == token:
                at.append(i)
        if len(at) * dm != len(feats):
            raise Error(
                "lfm3: " + String(len(at)) + " " + what + " tokens in the prompt for " + String(len(feats) // dm)
                + " rows of " + what + " features"
            )
        if len(at) == 0:
            return
        var hp = self.ctx.enqueue_create_host_buffer[DType.int32](len(at))
        for i in range(len(at)):
            hp[i] = Int32(at[i])
        var dp = self.ctx.enqueue_create_buffer[DType.int32](len(at))
        self.ctx.enqueue_copy(dp, hp)
        self.ctx.enqueue_function[put_rows_kernel](
            feats, dp, x, Int64(dm), grid_dim=(_blocks(dm), len(at)), block_dim=TPB,
        )

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
        var rows = self.slot_rows(s)
        self.ctx.enqueue_function[gather_rows_kernel](h, ds, rows, Int64(dm), grid_dim=(_blocks(dm), s), block_dim=TPB)
        var logits = self.new(s * v)
        self.lm_head(rows, logits, s)
        self.ctx.enqueue_function[log_softmax_rows_kernel](logits, Int64(v), grid_dim=s, block_dim=TPB)
        return self.download(logits, s * v)

    def slot_rows(self, s: Int) raises -> DeviceBuffer[DT]:
        """A zeroed `[S, hidden]` buffer padded to a multiple of LM_SLOTS rows:
        what `lm_head` reads the slots' hidden states from."""
        var n = (s + LM_SLOTS - 1) // LM_SLOTS * LM_SLOTS * self.cfg.hidden
        var rows = self.new(n)
        rows.enqueue_fill(0.0)
        return rows^

    def lm_head(mut self, rows: DeviceBuffer[DT], logits: DeviceBuffer[DT], s: Int) raises:
        """`logits[s, :] = rows[s] @ embed^T` for the first `s` rows of `rows`
        (from `slot_rows`): one pass over the table per LM_SLOTS slots, the last
        pass with the fewest slots (1, 2, 4, 8 or 16) that covers what is left."""
        var dm = self.cfg.hidden
        var v = self.cfg.vocab
        if self.q8_on:
            var qi = self.q8_index.get("embed_tokens.weight")
            if qi and (s <= Q8_GEMV_MAX or not self.index.get("embed_tokens.weight")):
                var s0 = 0
                while s0 < s:  # Q8_GEMV_MAX rows a pass over the int8 table
                    var c = min(Q8_GEMV_MAX, s - s0)
                    self._q8_gemv(
                        qi.value(), rows.create_sub_buffer[DT](s0 * dm, c * dm), logits.create_sub_buffer[DT](s0 * v, c * v), c,
                    )
                    s0 += c
                return
        var table = self.w("embed_tokens.weight")
        var s0 = 0
        while s0 < s:
            var left = s - s0
            var r = rows.create_sub_buffer[DT](s0 * dm, LM_SLOTS * dm)  # padded: always LM_SLOTS rows from s0
            var lg = logits.create_sub_buffer[DT](s0 * v, min(left, LM_SLOTS) * v)
            var ns = Int64(min(left, LM_SLOTS))
            if left >= 16:
                self.ctx.enqueue_function[lm_head_kernel[16]](r, table, lg, Int64(v), Int64(dm), ns, grid_dim=(v + 7) // 8, block_dim=256)
            elif left > 4:
                self.ctx.enqueue_function[lm_head_kernel[8]](r, table, lg, Int64(v), Int64(dm), ns, grid_dim=(v + 7) // 8, block_dim=256)
            elif left > 2:
                self.ctx.enqueue_function[lm_head_kernel[4]](r, table, lg, Int64(v), Int64(dm), ns, grid_dim=(v + 7) // 8, block_dim=256)
            elif left == 2:
                self.ctx.enqueue_function[lm_head_kernel[2]](r, table, lg, Int64(v), Int64(dm), ns, grid_dim=(v + 7) // 8, block_dim=256)
            else:
                self.ctx.enqueue_function[lm_head_kernel[1]](r, table, lg, Int64(v), Int64(dm), ns, grid_dim=(v + 7) // 8, block_dim=256)
            s0 += LM_SLOTS

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


# ── the attention, shared by the text backbone and the vision tower ─────

def split_heads(ctx: DeviceContext, x: DeviceBuffer[DT], t: Int, heads: Int, hd: Int) raises -> DeviceBuffer[DT]:
    var y = ctx.enqueue_create_buffer[DT](heads * t * hd)
    ctx.enqueue_function[split_heads_kernel](
        x, y, Int64(t), Int64(heads), Int64(hd), grid_dim=(_blocks(heads * hd), t), block_dim=TPB,
    )
    return y^


def attention_probs(
    ctx: DeviceContext, q: DeviceBuffer[DT], k: DeviceBuffer[DT], sc: DeviceBuffer[DT],
    groups: DeviceBuffer[DType.int32], t: Int, nh: Int, nkv: Int, hd: Int, mode: Int, state_len: Int, window: Int,
) raises:
    """`sc[h, i, :]` = the attention probabilities of query `i` in head `h`
    (0 where `i` may not see `j`: `kernels._sees`, mode 3 = every token sees
    every token), from `q[T, nh, hd]` and `k[T, nkv, hd]` after their norms and
    RoPE; scale `1 / sqrt(hd)`. NVIDIA: one GEMM per KV head (its `rep` query
    heads stacked, `[rep T, hd] @ [T, hd]^T`), then the mask and the softmax in
    one kernel. Elsewhere: a dot product per thread. `groups` is read in mode 0
    only (any int32 buffer otherwise)."""
    var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
    comptime if default_accelerator().is_nvidia_gpu():
        var rep = nh // nkv
        var qh = split_heads(ctx, q, t, nh, hd)
        var kh = split_heads(ctx, k, t, nkv, hd)
        for kv in range(nkv):
            cublas_gemm[False, True, LF_ATTN_TF32](
                ctx, sc.create_sub_buffer[DT](kv * rep * t * t, rep * t * t),
                qh.create_sub_buffer[DT](kv * rep * t * hd, rep * t * hd),
                kh.create_sub_buffer[DT](kv * t * hd, t * hd), rep * t, t, hd, 0.0,
            )
        ctx.enqueue_function[masked_softmax_kernel](
            sc, groups, Int64(t), scale, Int64(mode), Int64(state_len), Int64(window),
            grid_dim=(t, nh), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[scores_kernel](
            q, k, sc, groups, Int64(t), Int64(nh), Int64(nkv), Int64(hd), scale, Int64(mode),
            Int64(state_len), Int64(window), grid_dim=(_blocks(t), t, nh), block_dim=TPB,
        )
        ctx.enqueue_function[softmax_rows_kernel](sc, Int64(t), grid_dim=nh * t, block_dim=TPB)


def attention_apply(
    ctx: DeviceContext, sc: DeviceBuffer[DT], v: DeviceBuffer[DT], gate: DeviceBuffer[DT], has_gate: Bool,
    dst: DeviceBuffer[DT], t: Int, nh: Int, nkv: Int, hd: Int,
) raises:
    """`dst[i, h, :] = g * sum_j sc[h, i, j] v[j, h / rep, :]`, `g` =
    `sigmoid(gate[i, h])` (`has_gate`) or 1 (`gate` then unread). NVIDIA: one
    GEMM per KV head (`[rep T, T] @ [T, hd]`), then the gate."""
    comptime if default_accelerator().is_nvidia_gpu():
        var rep = nh // nkv
        var vh = split_heads(ctx, v, t, nkv, hd)
        var oh = ctx.enqueue_create_buffer[DT](nh * t * hd)
        for kv in range(nkv):
            cublas_gemm[False, False, LF_ATTN_TF32](
                ctx, oh.create_sub_buffer[DT](kv * rep * t * hd, rep * t * hd),
                sc.create_sub_buffer[DT](kv * rep * t * t, rep * t * t),
                vh.create_sub_buffer[DT](kv * t * hd, t * hd), rep * t, hd, t, 0.0,
            )
        ctx.enqueue_function[merge_heads_gate_kernel](
            oh, gate, dst, Int64(t), Int64(nh), Int64(hd), Int64(1 if has_gate else 0),
            grid_dim=(_blocks(nh * hd), t), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[attn_out_kernel](
            sc, v, gate, dst, Int64(t), Int64(nh), Int64(nkv), Int64(hd), Int64(1 if has_gate else 0),
            grid_dim=(_blocks(nh * hd), t), block_dim=TPB,
        )
