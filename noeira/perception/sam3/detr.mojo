# +--------------------------------------------------------------------------+ #
# | SAM 3 — the DETR encoder (image tokens fused with the prompt)
# +--------------------------------------------------------------------------+ #
"""`Sam3DetrEncoder`: `Sam3DetrEncoder` from `detector_model.detr_encoder.*`.

    var enc = Sam3DetrEncoder(ctx, sam3_checkpoint())
    var fused = enc.forward(level72, pos72, text_features, text_len)     # [5184, 256]

On the neck's 72 x 72 level (token-major) with its sine position encoding and
the prompt's 32 features (the first `text_len` real), 6 layers:

    x = x + self_attn(q = k = LN1(x) + pos, v = LN1(x))
    x = x + cross_attn(q = LN2(x), k = v = text)          # the padding keys masked
    x = x + fc2(relu(fc1(LN3(x))))

8 heads x 32 with biases; the LayerNorms are torch's DEFAULT eps 1e-5 (the
config's 1e-6 is not passed to them). The self-attention holds a 5184 x 5184
head at a time (107 MB).
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from .ops import (
    Sam3Weights, TPB, blocks, linear, layer_norm, mha, add_kernel, add_into_kernel, relu_kernel,
    linear_into, flash_self, USE_FLASH, linear_residual, mlp_residual,
    SAM3_BF16, flash_self_bf16, bias_act_bf16_kernel, linear_into_b, bias_add,
)


comptime ENC_PREFIX = "detector_model.detr_encoder."
comptime D_MODEL: Int = 256
comptime D_HEADS: Int = 8
comptime D_HD: Int = 32
comptime D_FF: Int = 2048
comptime D_LAYERS: Int = 6
comptime D_EPS: Float32 = 1e-5
comptime T_LEN: Int = 32
comptime FLASH_N: Int = 72 * 72
"""The fused self-attention's token count (compile-time in the kernel)."""


struct Sam3DetrEncoder(Movable):
    var wts: Sam3Weights
    var debug: Int
    """Falsification only: bit 1, the text's padding keys visible; bit 2, the
    position added to the self-attention's VALUES too."""

    def __init__(out self, var ctx: DeviceContext, path: String) raises:
        self.wts = Sam3Weights(ctx^, path, ENC_PREFIX)
        self.debug = 0

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def forward(
        mut self, vision: DeviceBuffer[DT], pos: DeviceBuffer[DT], n: Int, text: DeviceBuffer[DT], text_len: Int,
        mut taps: List[DeviceBuffer[DT]], keep_taps: Bool = False,
    ) raises -> DeviceBuffer[DT]:
        """`vision`, `pos` [n, 256] token-major; `text` [32, 256]. -> [n, 256]."""
        ref ctx = self.wts.ctx
        var x = self.new(n * D_MODEL)
        ctx.enqueue_copy(x, vision.create_sub_buffer[DT](0, n * D_MODEL))
        for li in range(D_LAYERS):
            var p = String("layers.") + String(li) + "."
            var h = self.new(n * D_MODEL)
            layer_norm(self.wts, p + "layer_norm1", x, h, n, D_MODEL, D_EPS)
            var hp = self.new(n * D_MODEL)
            ctx.enqueue_function[add_into_kernel](hp, h, pos, Int64(n * D_MODEL), grid_dim=blocks(n * D_MODEL), block_dim=TPB)
            comptime if SAM3_BF16:
                # bf16: the three GEMMs (TF32: 256 wide) without bias, then bias + bf16 into one
                # [Q | K | V] buffer, the bf16 fused attention, its output straight into o_proj
                if n != FLASH_N:
                    raise Error("sam3 detr: the fused attention is built for 72 x 72 tokens, not " + String(n))
                var qf = self.new(3 * n * D_MODEL)
                var qb = ctx.enqueue_create_buffer[DType.bfloat16](3 * n * D_MODEL)
                var names: List[String] = ["q_proj", "k_proj", "v_proj"]
                for part in range(3):
                    var nm = p + "self_attn." + names[part]
                    var src = hp if part < 2 or (self.debug & 2) != 0 else h
                    var slot = qf.create_sub_buffer[DT](part * n * D_MODEL, n * D_MODEL)
                    linear_into(self.wts, nm, src, n, D_MODEL, slot, bias=False)
                    ctx.enqueue_function[bias_act_bf16_kernel[0]](
                        slot, self.wts.w(nm + ".bias"), qb.create_sub_buffer[DType.bfloat16](part * n * D_MODEL, n * D_MODEL),
                        Int64(D_MODEL), grid_dim=(n, blocks(D_MODEL)), block_dim=TPB,
                    )
                var ab = ctx.enqueue_create_buffer[DType.bfloat16](n * D_MODEL)
                flash_self_bf16[FLASH_N, D_HEADS, D_HD](ctx, qb, ab)
                var o = self.new(n * D_MODEL)
                linear_into_b(self.wts, p + "self_attn.o_proj", ab, n, D_MODEL, o, bias=False)
                bias_add(self.wts, p + "self_attn.o_proj", x, o, n, D_MODEL)
            else:
                var qkv = self.new(3 * n * D_MODEL)
                var q = qkv.create_sub_buffer[DT](0, n * D_MODEL)
                var k = qkv.create_sub_buffer[DT](n * D_MODEL, n * D_MODEL)
                var v = qkv.create_sub_buffer[DT](2 * n * D_MODEL, n * D_MODEL)
                linear_into(self.wts, p + "self_attn.q_proj", hp, n, D_MODEL, q)
                linear_into(self.wts, p + "self_attn.k_proj", hp, n, D_MODEL, k)
                linear_into(self.wts, p + "self_attn.v_proj", hp if (self.debug & 2) != 0 else h, n, D_MODEL, v)
                var a: DeviceBuffer[DT]
                comptime if USE_FLASH:
                    if n != FLASH_N:
                        raise Error("sam3 detr: the fused attention is built for 72 x 72 tokens, not " + String(n))
                    a = self.new(n * D_MODEL)
                    flash_self[FLASH_N, D_HEADS, D_HD](ctx, qkv, a)
                else:
                    a = mha(ctx, q, k, v, n, n, D_HEADS, D_HD, n)
                linear_residual(self.wts, p + "self_attn.o_proj", a, n, D_MODEL, x, D_MODEL)
            var h2 = self.new(n * D_MODEL)
            layer_norm(self.wts, p + "layer_norm2", x, h2, n, D_MODEL, D_EPS)
            var cq = linear(self.wts, p + "cross_attn.q_proj", h2, n, D_MODEL)
            var ck = linear(self.wts, p + "cross_attn.k_proj", text, T_LEN, D_MODEL)
            var cv = linear(self.wts, p + "cross_attn.v_proj", text, T_LEN, D_MODEL)
            var ca = mha(ctx, cq, ck, cv, n, T_LEN, D_HEADS, D_HD, T_LEN if (self.debug & 1) != 0 else text_len)
            linear_residual(self.wts, p + "cross_attn.o_proj", ca, n, D_MODEL, x, D_MODEL)
            var h3 = self.new(n * D_MODEL)
            layer_norm(self.wts, p + "layer_norm3", x, h3, n, D_MODEL, D_EPS)
            mlp_residual[2](self.wts, p + "mlp.fc1", p + "mlp.fc2", h3, n, D_MODEL, D_FF, x)
            if keep_taps:
                var c = self.new(n * D_MODEL)
                ctx.enqueue_copy(c, x)
                taps.append(c^)
        return x^
