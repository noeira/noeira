# +--------------------------------------------------------------------------+ #
# | SAM 3 — the text encoder (CLIP's, 24 layers) and SAM 3's projection
# +--------------------------------------------------------------------------+ #
"""`Sam3Text`: `Sam3Model.get_text_features` — CLIP's text transformer
(`detector_model.text_encoder.text_model.*`) then SAM 3's `text_projection`
(1024 -> 256, with bias) on EVERY token.

    var text = Sam3Text(ctx, sam3_checkpoint())
    var toks = tokenizer.encode("a red chair", 32)     # noeira.text.clip_bpe
    var f = text.forward(toks)                         # [32, 256]

The forward: token + position embedding (32 positions), 24 pre-norm layers
(LayerNorm eps 1e-5; 16 heads x 64 with biases; exact GELU, as SAM 3's
config sets — CLIP's own is quick-GELU), `final_layer_norm`. The mask is
causal AND hides the padding keys (`kernels._sees` mode 4): a padded query
row still reads every real token, so its output is defined (the reference's
too). CLIP's own `text_projection` (1024 -> 512, its pooled output) is not
used by SAM 3 and is not applied.
"""

from std.math import exp
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.llm.lfm3.model import attention_probs, attention_apply
from noeira.text.clip_bpe import ClipTokens
from .ops import Sam3Weights, TPB, Ptr, blocks, linear, layer_norm, add_kernel, gelu_kernel


comptime TEXT_PREFIX = "detector_model.text_"
comptime T_LEN: Int = 32
comptime T_DIM: Int = 1024
comptime T_HEADS: Int = 16
comptime T_HD: Int = 64
comptime T_FF: Int = 4096
comptime T_LAYERS: Int = 24
comptime T_EPS: Float32 = 1e-5
comptime TEXT_OUT: Int = 256
comptime _TM = "encoder.text_model."


def text_embed_kernel(ids: Pointer[Scalar[DType.int32], MutAnyOrigin], tok: Ptr, pos: Ptr, dst: Ptr):
    """`dst[t] = tok[ids[t]] + pos[t]`. Grid `(ceil(D / TPB), T)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if d < T_DIM:
        dst[unsafe_offset=t * T_DIM + d] = tok[unsafe_offset=Int(ids[unsafe_offset=t]) * T_DIM + d] + pos[unsafe_offset=t * T_DIM + d]


def quick_gelu_kernel(x: Ptr, n_n: Int64):
    """`x sigmoid(1.702 x)` in place — CLIP's default, NOT SAM 3's (falsification)."""
    var i = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if i < Int(n_n):
        var v = x[unsafe_offset=i]
        x[unsafe_offset=i] = v / (Scalar[DT](1) + exp(Scalar[DT](-1.702) * v))


struct Sam3Text(Movable):
    var wts: Sam3Weights
    var dummy: DeviceBuffer[DType.int32]
    var debug: Int
    """Falsification only: bit 1, plain causal (the padding keys visible);
    bit 2, quick-GELU (CLIP's own activation) instead of SAM 3's GELU."""

    def __init__(out self, var ctx: DeviceContext, path: String) raises:
        self.wts = Sam3Weights(ctx^, path, TEXT_PREFIX)
        self.dummy = self.wts.ctx.enqueue_create_buffer[DType.int32](1)
        self.debug = 0

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def forward(mut self, toks: ClipTokens, mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False) raises -> DeviceBuffer[DT]:
        """The 32 tokens' features [32, 256]. `taps` (with `keep_taps`): each
        layer's output, then `final_layer_norm`'s."""
        ref ctx = self.wts.ctx
        if len(toks.ids) != T_LEN:
            raise Error("sam3 text: " + String(len(toks.ids)) + " ids, not " + String(T_LEN))
        var hid = ctx.enqueue_create_host_buffer[DType.int32](T_LEN)
        for i in range(T_LEN):
            hid[i] = Int32(toks.ids[i])
        var did = ctx.enqueue_create_buffer[DType.int32](T_LEN)
        ctx.enqueue_copy(did, hid)
        var x = self.new(T_LEN * T_DIM)
        ctx.enqueue_function[text_embed_kernel](
            did, self.wts.w(_TM + "embeddings.token_embedding.weight"),
            self.wts.w(_TM + "embeddings.position_embedding.weight"), x, grid_dim=(blocks(T_DIM), T_LEN), block_dim=TPB,
        )
        var mode = 2 if (self.debug & 1) != 0 else 4
        var sc = self.new(T_HEADS * T_LEN * T_LEN)
        for li in range(T_LAYERS):
            var p = _TM + "encoder.layers." + String(li) + "."
            var h = self.new(T_LEN * T_DIM)
            layer_norm(self.wts, p + "layer_norm1", x, h, T_LEN, T_DIM, T_EPS)
            var q = linear(self.wts, p + "self_attn.q_proj", h, T_LEN, T_DIM)
            var k = linear(self.wts, p + "self_attn.k_proj", h, T_LEN, T_DIM)
            var v = linear(self.wts, p + "self_attn.v_proj", h, T_LEN, T_DIM)
            attention_probs(ctx, q, k, sc, self.dummy, T_LEN, T_HEADS, T_HEADS, T_HD, mode, toks.length, 0)
            var a = self.new(T_LEN * T_DIM)
            attention_apply(ctx, sc, v, a, False, a, T_LEN, T_HEADS, T_HEADS, T_HD)
            var o = linear(self.wts, p + "self_attn.out_proj", a, T_LEN, T_DIM)
            ctx.enqueue_function[add_kernel](x, o, Int64(T_LEN * T_DIM), grid_dim=blocks(T_LEN * T_DIM), block_dim=TPB)
            var h2 = self.new(T_LEN * T_DIM)
            layer_norm(self.wts, p + "layer_norm2", x, h2, T_LEN, T_DIM, T_EPS)
            var f1 = linear(self.wts, p + "mlp.fc1", h2, T_LEN, T_DIM)
            if (self.debug & 2) != 0:
                ctx.enqueue_function[quick_gelu_kernel](f1, Int64(T_LEN * T_FF), grid_dim=blocks(T_LEN * T_FF), block_dim=TPB)
            else:
                ctx.enqueue_function[gelu_kernel](f1, Int64(T_LEN * T_FF), grid_dim=blocks(T_LEN * T_FF), block_dim=TPB)
            var f2 = linear(self.wts, p + "mlp.fc2", f1, T_LEN, T_FF)
            ctx.enqueue_function[add_kernel](x, f2, Int64(T_LEN * T_DIM), grid_dim=blocks(T_LEN * T_DIM), block_dim=TPB)
            if keep_taps:
                taps.append(_copy(ctx, x, T_LEN * T_DIM))
        var fin = self.new(T_LEN * T_DIM)
        layer_norm(self.wts, _TM + "final_layer_norm", x, fin, T_LEN, T_DIM, T_EPS)
        if keep_taps:
            taps.append(_copy(ctx, fin, T_LEN * T_DIM))
        return linear(self.wts, "projection", fin, T_LEN, T_DIM)


def _copy(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> DeviceBuffer[DT]:
    var b = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(b, x.create_sub_buffer[DT](0, n))
    return b^
