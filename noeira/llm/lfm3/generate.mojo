# +--------------------------------------------------------------------------+ #
# | LFM3 generation — a forward over new tokens on a cache of the earlier ones
# +--------------------------------------------------------------------------+ #
"""What makes the LFM3 backbone a chat model (docs/NN_MODEL_PORTS_PLAN.md §12.7):
the tokens before the chunk are not recomputed; each layer reads what it needs
of them from an `Lfm3Cache`.

    var cache = Lfm3Cache.make(lm, 4096)              # capacity in tokens
    var h = forward_cached(lm, prompt_ids, cache)     # prefill: one chunk
    var logits = last_logits(lm, h, len(prompt_ids))  # [vocab] for the next token
    h = forward_cached(lm, [token], cache)            # each next token: a chunk of 1

What a layer needs of the past (Liquid's `modeling_lfm3.py`, `DynamicCache`):
- a short-conv layer: its last K - 1 = 6 rows of `in_proj`'s `[B | C | x]` (the
  causal depthwise conv reads the six tokens before). Kept as rows of `bcx` so
  the products are the ones the full pass takes, in the same order;
- an attention layer: every earlier key (after `k_norm` and RoPE at its
  absolute position) and value. A sliding layer reads only the last `window`
  of them; the cache keeps all of them anyway (the mask is applied when they
  are read), 45 KB a token over the 11 attention layers in fp32.

A chunk at position `p` gives the rows one causal pass over the whole sequence
gives at `p..p+n-1` (`Lfm3Text.forward(..., causal_full=True)`, the gate's
"cache = recompute" check). The MLP and the norms are the forward's own helpers.

`Lfm3Cache.debug` (falsification, the gate's mutants): 1 = every chunk's RoPE
positions from 0 (the offset forgotten; a uniform shift would be invisible, RoPE
being relative), 2 = no sliding window on the cache, 4 = the conv history never
rolled.
"""

from std.sys import default_accelerator
from max.gpu.host import DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm
from .kernels import (
    TPB, embed_kernel, add_kernel, rope_kernel, swiglu_kernel, swiglu_bf16_kernel, merge_heads_gate_kernel,
)
from .generate_kernels import (
    conv_cached_kernel, conv_history_kernel, put_kv_kernel, cached_attention_kernel, masked_softmax_offset_kernel,
    MAX_KEYS,
)
from .model import Lfm3Text, LT_CONV, LT_FULL, LT_SLIDING, LF_BF16, LF_ATTN_TF32, split_heads


def _blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


struct Lfm3Cache(Movable):
    var cap: Int
    """Tokens it holds at most."""
    var pos: Int
    """Tokens in it: the next chunk's first position."""
    var slot: List[Int]
    """Layer -> its slot in `k` / `v` (attention) or `hist` (conv)."""
    var k: List[DeviceBuffer[DT]]
    """`[kv_heads, cap, head_dim]` per attention layer."""
    var v: List[DeviceBuffer[DT]]
    var hist: List[DeviceBuffer[DT]]
    """`[K - 1, 3 hidden]` per conv layer: the last K - 1 rows of `[B | C | x]`."""
    var hist_next: DeviceBuffer[DT]
    var cos_full: DeviceBuffer[DT]
    """`[cap, head_dim / 2]` RoPE tables at every position, full and sliding layers."""
    var sin_full: DeviceBuffer[DT]
    var cos_sliding: DeviceBuffer[DT]
    var sin_sliding: DeviceBuffer[DT]
    var debug: Int

    def __init__(
        out self, cap: Int, var slot: List[Int], var k: List[DeviceBuffer[DT]], var v: List[DeviceBuffer[DT]],
        var hist: List[DeviceBuffer[DT]], var hist_next: DeviceBuffer[DT], var cos_full: DeviceBuffer[DT],
        var sin_full: DeviceBuffer[DT], var cos_sliding: DeviceBuffer[DT], var sin_sliding: DeviceBuffer[DT],
    ):
        self.cap = cap
        self.pos = 0
        self.slot = slot^
        self.k = k^
        self.v = v^
        self.hist = hist^
        self.hist_next = hist_next^
        self.cos_full = cos_full^
        self.sin_full = sin_full^
        self.cos_sliding = cos_sliding^
        self.sin_sliding = sin_sliding^
        self.debug = 0

    @staticmethod
    def make(mut lm: Lfm3Text, cap: Int) raises -> Lfm3Cache:
        """An empty cache for `cap` tokens (at most `MAX_KEYS`)."""
        if cap > MAX_KEYS:
            raise Error("lfm3: a cache of " + String(cap) + " tokens; the attention kernel reads at most " + String(MAX_KEYS))
        var c = lm.cfg.copy()
        var kvw = c.kv_heads * cap * c.head_dim
        var slot = List[Int]()
        var k = List[DeviceBuffer[DT]]()
        var v = List[DeviceBuffer[DT]]()
        var hist = List[DeviceBuffer[DT]]()
        for kind in c.layer_types:
            if kind == LT_CONV:
                slot.append(len(hist))
                hist.append(lm.new((c.conv_k - 1) * 3 * c.hidden))
            else:
                slot.append(len(k))
                k.append(lm.new(kvw))
                v.append(lm.new(kvw))
        var rf = lm.rope_tables(cap, c.theta_full, c.prf_full)
        var rs = lm.rope_tables(cap, c.theta_sliding, c.prf_sliding)
        var out = Lfm3Cache(
            cap, slot^, k^, v^, hist^, lm.new((c.conv_k - 1) * 3 * c.hidden), rf[0], rf[1], rs[0], rs[1],
        )
        out.reset()
        return out^

    def reset(mut self) raises:
        """Forget every token: the conv histories back to zeros (the keys and
        values past `pos` are never read)."""
        self.pos = 0
        for h in self.hist:
            h.enqueue_fill(0.0)

    def left(self) -> Int:
        return self.cap - self.pos


def forward_cached(
    mut lm: Lfm3Text, ids: List[Int], mut cache: Lfm3Cache, mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False,
) raises -> DeviceBuffer[DT]:
    """`[n, hidden]` after the final norm for the `n` new tokens `ids`, at
    positions `cache.pos..`; their keys, values and conv rows go into the cache.
    `taps` (with `keep_taps`): the embedding, then each layer's output."""
    var n = len(ids)
    if n == 0:
        raise Error("lfm3: an empty chunk")
    if cache.pos + n > cache.cap:
        raise Error(
            "lfm3: the cache is full (" + String(cache.pos) + " + " + String(n) + " tokens > " + String(cache.cap) + ")"
        )
    var p = cache.pos
    var dm = lm.cfg.hidden
    var hd = lm.cfg.head_dim
    var half = hd // 2
    var nh = lm.cfg.heads
    var nkv = lm.cfg.kv_heads
    var kk = lm.cfg.conv_k
    var hid = lm.ctx.enqueue_create_host_buffer[DType.int32](n)
    for i in range(n):
        hid[i] = Int32(ids[i])
    var did = lm.ctx.enqueue_create_buffer[DType.int32](n)
    lm.ctx.enqueue_copy(did, hid)
    var x = lm.new(n * dm)
    lm.embed_rows(did, x, n)
    if keep_taps:
        taps.append(lm._copy(x, n * dm))
    var rp = 0 if (cache.debug & 1) != 0 else p  # mutant 1: every chunk's positions from 0
    var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
    for li in range(len(lm.cfg.layer_types)):
        var pre = String("layers.") + String(li) + "."
        var kind = lm.cfg.layer_types[li]
        var s = cache.slot[li]
        var hn = lm.new(n * dm)
        var hnb = lm.rms_b(pre + "operator_norm", x, hn, n, dm)
        var o: DeviceBuffer[DT]
        if kind == LT_CONV:
            var bcx = lm.linear(pre + "conv.in_proj", hn, n, dm, hnb)
            var z = lm.new(n * dm)
            lm.ctx.enqueue_function[conv_cached_kernel](
                bcx, cache.hist[s], lm.w(pre + "conv.conv.weight"), z, Int64(dm), Int64(kk),
                grid_dim=(_blocks(dm), n), block_dim=TPB,
            )
            if (cache.debug & 4) == 0:
                lm.ctx.enqueue_function[conv_history_kernel](
                    bcx, cache.hist[s], cache.hist_next, Int64(n), Int64(3 * dm), Int64(kk),
                    grid_dim=(_blocks(3 * dm), kk - 1), block_dim=TPB,
                )
                lm.ctx.enqueue_copy(cache.hist[s], cache.hist_next)
            o = lm.linear(pre + "conv.out_proj", z, n, dm)
        else:
            var q = lm.linear(pre + "self_attn.q_proj", hn, n, dm, hnb)
            var k = lm.linear(pre + "self_attn.k_proj", hn, n, dm, hnb)
            var v = lm.linear(pre + "self_attn.v_proj", hn, n, dm, hnb)
            var g = lm.linear(pre + "self_attn.gate_proj", hn, n, dm, hnb)
            lm.rms(pre + "self_attn.q_norm", q, q, n * nh, hd)
            lm.rms(pre + "self_attn.k_norm", k, k, n * nkv, hd)
            var full = kind == LT_FULL
            var cos_t = (cache.cos_full if full else cache.cos_sliding).create_sub_buffer[DT](rp * half, n * half)
            var sin_t = (cache.sin_full if full else cache.sin_sliding).create_sub_buffer[DT](rp * half, n * half)
            lm.ctx.enqueue_function[rope_kernel](
                q, cos_t, sin_t, Int64(nh), Int64(half), grid_dim=(_blocks(nh * half), n), block_dim=TPB,
            )
            lm.ctx.enqueue_function[rope_kernel](
                k, cos_t, sin_t, Int64(nkv), Int64(half), grid_dim=(_blocks(nkv * half), n), block_dim=TPB,
            )
            lm.ctx.enqueue_function[put_kv_kernel](
                k, cache.k[s], Int64(p), Int64(cache.cap), Int64(nkv), Int64(hd),
                grid_dim=(_blocks(nkv * hd), n), block_dim=TPB,
            )
            lm.ctx.enqueue_function[put_kv_kernel](
                v, cache.v[s], Int64(p), Int64(cache.cap), Int64(nkv), Int64(hd),
                grid_dim=(_blocks(nkv * hd), n), block_dim=TPB,
            )
            var window = lm.cfg.window if (kind == LT_SLIDING and (cache.debug & 2) == 0) else 0
            var a = lm.new(n * dm)
            _attend(lm, q, cache.k[s], cache.v[s], g, a, n, p, cache.cap, window, scale)
            o = lm.linear(pre + "self_attn.o_proj", a, n, dm)
        lm.rms(pre + "post_operator_norm", o, o, n, dm)
        lm.ctx.enqueue_function[add_kernel](x, o, Int64(n * dm), grid_dim=_blocks(n * dm), block_dim=TPB)
        _mlp(lm, pre, x, n)
        if keep_taps:
            taps.append(lm._copy(x, n * dm))
    cache.pos += n
    var out = lm.new(n * dm)
    lm.rms("embedding_norm", x, out, n, dm)
    return out^


comptime GEMM_ATTN_MIN: Int = 8
"""Chunks of at least this many tokens (a prefill, a new turn) attend with
tensor-core GEMMs on NVIDIA; shorter ones (decoding) with the per-query kernel.
The per-query kernel is linear in the keys for one query, but a 1024-token
prefill through it took 1.5 s on the Orin."""


def _attend(
    mut lm: Lfm3Text, q: DeviceBuffer[DT], kc: DeviceBuffer[DT], vc: DeviceBuffer[DT], g: DeviceBuffer[DT],
    dst: DeviceBuffer[DT], n: Int, p: Int, cap: Int, window: Int, scale: Float32,
) raises:
    """`dst[n, nh hd]`: the chunk's queries (positions p..p+n-1) against the
    cached keys 0..p+n-1, through the output gate. Two routes, one mask
    (`cached_attention_kernel` / `masked_softmax_offset_kernel` write the same
    rule): per query (decoding), or per KV head as GEMMs `[rep n, hd] @ [L,
    hd]^T` then `[rep n, L] @ [L, hd]` (the cache is heads-major, so each KV
    head's keys are one contiguous operand)."""
    var nh = lm.cfg.heads
    var nkv = lm.cfg.kv_heads
    var hd = lm.cfg.head_dim
    comptime if default_accelerator().is_nvidia_gpu():
        if n >= GEMM_ATTN_MIN:
            var l = p + n
            var rep = nh // nkv
            var qh = split_heads(lm.ctx, q, n, nh, hd)
            var sc = lm.new(nh * n * l)
            var oh = lm.new(nh * n * hd)
            for kv in range(nkv):
                cublas_gemm[False, True, LF_ATTN_TF32](
                    lm.ctx, sc.create_sub_buffer[DT](kv * rep * n * l, rep * n * l),
                    qh.create_sub_buffer[DT](kv * rep * n * hd, rep * n * hd),
                    kc.create_sub_buffer[DT](kv * cap * hd, l * hd), rep * n, l, hd, 0.0,
                )
            lm.ctx.enqueue_function[masked_softmax_offset_kernel](
                sc, Int64(n), Int64(l), Int64(p), Int64(window), scale, grid_dim=(n, nh), block_dim=TPB,
            )
            for kv in range(nkv):
                cublas_gemm[False, False, LF_ATTN_TF32](
                    lm.ctx, oh.create_sub_buffer[DT](kv * rep * n * hd, rep * n * hd),
                    sc.create_sub_buffer[DT](kv * rep * n * l, rep * n * l),
                    vc.create_sub_buffer[DT](kv * cap * hd, l * hd), rep * n, hd, l, 0.0,
                )
            lm.ctx.enqueue_function[merge_heads_gate_kernel](
                oh, g, dst, Int64(n), Int64(nh), Int64(hd), Int64(1), grid_dim=(_blocks(nh * hd), n), block_dim=TPB,
            )
            return
    lm.ctx.enqueue_function[cached_attention_kernel](
        q, kc, vc, g, dst, Int64(p), Int64(nh), Int64(nkv), Int64(hd), Int64(cap), Int64(window), scale,
        grid_dim=(n, nh), block_dim=TPB,
    )


def forward_cached(mut lm: Lfm3Text, ids: List[Int], mut cache: Lfm3Cache) raises -> DeviceBuffer[DT]:
    var taps = List[DeviceBuffer[DT]]()
    return forward_cached(lm, ids, cache, taps)


def _mlp(mut lm: Lfm3Text, pre: String, x: DeviceBuffer[DT], n: Int) raises:
    """`x += post_ffn_norm(w2(silu(w1 h) * w3 h))`, `h = ffn_norm(x)`: the
    forward's MLP at inference (no training captures)."""
    var dm = lm.cfg.hidden
    var ff = lm.cfg.ff
    var fnrm = lm.new(n * dm)
    var fnb = lm.rms_b(pre + "ffn_norm", x, fnrm, n, dm)
    var a1 = lm.linear(pre + "feed_forward.w1", fnrm, n, dm, fnb)
    var a3 = lm.linear(pre + "feed_forward.w3", fnrm, n, dm, fnb)
    var swb = Optional[DeviceBuffer[DType.bfloat16]](None)
    comptime if LF_BF16:
        if len(lm.lora_index) == 0 and not lm.q8_on:  # the int8 GEMV reads the fp32 product
            var b = lm.ctx.enqueue_create_buffer[DType.bfloat16](n * ff)
            lm.ctx.enqueue_function[swiglu_bf16_kernel](a1, a3, b, Int64(n * ff), grid_dim=_blocks(n * ff), block_dim=TPB)
            swb = Optional(b^)
    if not swb:
        lm.ctx.enqueue_function[swiglu_kernel](a1, a3, a1, Int64(n * ff), grid_dim=_blocks(n * ff), block_dim=TPB)
    var f = lm.linear(pre + "feed_forward.w2", a1, n, ff, swb)
    lm.rms(pre + "post_ffn_norm", f, f, n, dm)
    lm.ctx.enqueue_function[add_kernel](x, f, Int64(n * dm), grid_dim=_blocks(n * dm), block_dim=TPB)


def last_logits(mut lm: Lfm3Text, h: DeviceBuffer[DT], n: Int) raises -> List[Float32]:
    """The LM head's raw logits `[vocab]` at the chunk's last row: the next token's."""
    var dm = lm.cfg.hidden
    var v = lm.cfg.vocab
    var rows = lm.slot_rows(1)
    lm.ctx.enqueue_copy(rows.create_sub_buffer[DT](0, dm), h.create_sub_buffer[DT]((n - 1) * dm, dm))
    var logits = lm.new(v)
    lm.lm_head(rows, logits, 1)
    return lm.download(logits, v)
