# +--------------------------------------------------------------------------+ #
# | LFM3 — the backward pass, for LoRA on a frozen base
# +--------------------------------------------------------------------------+ #
"""Gradients of a loss at the answer slots with respect to the LoRA adapters
(and the input embeddings), on NVIDIA. The forward is `Lfm3Text.forward` itself
with `training = True` — one forward, not a second copy — which keeps each
layer's activations in `lm.act`.

    lm.load_lora(path, 16, 2.0)                    # or adapters of your own
    lm.training = True
    var h = lm.forward(ids, groups, state_len, taps)
    var lt = slot_loss(lm, h, len(ids), slots, targets)   # loss, d final
    var d_embeds = backward(lm, lt[1], len(ids))          # LoRA grads accumulate in lm.lora_da / lora_db

The loss: `-sum_s log_softmax(lm_head(h[slot_s]))[target_s]`, cross-entropy
over the vocabulary at each slot (the readout d1 reads its answers from). The
base weights are frozen: each linear's backward is `dx = dy W` (W widened as in
the forward) plus, where an adapter sits, `dB += s dy^T (x A^T)`,
`dA += (s dy B)^T x`, `dx += (s dy B) A`.
"""

from std.math import exp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm
from .kernels import (
    TPB, rope_kernel, gather_rows_kernel, lm_head_kernel, log_softmax_rows_kernel, bf16_to_f32_kernel,
)
from .kernels_bwd import (
    rms_bwd_kernel, swiglu_bwd_kernel, short_conv_bwd_kernel, gate_bwd_kernel, attn_dp_kernel,
    softmax_bwd_rows_kernel, attn_dq_kernel, attn_dkv_kernel, lm_head_bwd_kernel, scatter_rows_kernel, axpy_kernel,
    adamw_kernel,
)
from .model import Lfm3Text, LT_CONV, LT_FULL, LT_SLIDING, LF_TF32


def _blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


def _gemm[TA: Bool, TB: Bool](
    mut lm: Lfm3Text, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT],
    m: Int, n: Int, k: Int, alpha: Float32, beta: Float32,
) raises:
    cublas_gemm[TA, TB, LF_TF32](lm.ctx, dst, a, b, m, n, k, beta, alpha)


def _rms_bwd(mut lm: Lfm3Text, name: String, x: DeviceBuffer[DT], dy: DeviceBuffer[DT], rows: Int, dim: Int) raises -> DeviceBuffer[DT]:
    var dx = lm.new(rows * dim)
    lm.ctx.enqueue_function[rms_bwd_kernel](
        x, lm.w(name + ".weight"), dy, dx, Int64(dim), lm.cfg.eps, grid_dim=rows, block_dim=TPB,
    )
    return dx^


def _add(mut lm: Lfm3Text, dst: DeviceBuffer[DT], src: DeviceBuffer[DT], n: Int) raises:
    lm.ctx.enqueue_function[axpy_kernel](dst, src, Int64(n), Float32(1.0), grid_dim=_blocks(n), block_dim=TPB)


def linear_bwd(
    mut lm: Lfm3Text, layer: Int, name: String, x: DeviceBuffer[DT], dy: DeviceBuffer[DT], t: Int, k: Int,
) raises -> DeviceBuffer[DT]:
    """`dx` of `y = x W^T (+ s (x A^T) B^T)`, `x [t, k]`, `dy [t, n]`; the
    adapter's gradients accumulate."""
    var wb = lm.w(name + ".weight")
    var n = len(wb) // k
    var wf = lm.scratch.create_sub_buffer[DT](0, n * k)
    lm.ctx.enqueue_function[bf16_to_f32_kernel](wb, wf, Int64(n * k), grid_dim=_blocks(n * k), block_dim=TPB)
    var dx = lm.new(t * k)
    _gemm[False, False](lm, dx, dy, wf, t, k, n, 1.0, 0.0)
    var li = lm.lora_index.get(name)
    if li:
        var j = li.value()
        var r = lm.lora_rank
        var s = lm.lora_scale
        # handles (reference-counted): the same device memory, borrowed apart from `lm`
        var xa = lm.act[layer]["xa:" + name]
        var a_j = lm.lora_a[j]
        var b_j = lm.lora_b[j]
        var da_j = lm.lora_da[j]
        var db_j = lm.lora_db[j]
        _gemm[True, False](lm, db_j, dy, xa, n, r, t, s, 1.0)        # dB += s dy^T xa
        var dxa = lm.new(t * r)
        _gemm[False, False](lm, dxa, dy, b_j, t, r, n, s, 0.0)       # dxa = s dy B
        _gemm[True, False](lm, da_j, dxa, x, r, k, t, 1.0, 1.0)      # dA += dxa^T x
        _gemm[False, False](lm, dx, dxa, a_j, t, k, r, 1.0, 1.0)     # dx += dxa A
    return dx^


def slot_loss(
    mut lm: Lfm3Text, h: DeviceBuffer[DT], t: Int, slots: List[Int], targets: List[Int],
) raises -> Tuple[Float64, DeviceBuffer[DT]]:
    """The loss and its gradient with respect to the final norm's output [t, hidden]."""
    var s = len(slots)
    var dm = lm.cfg.hidden
    var v = lm.cfg.vocab
    var hs = lm.ctx.enqueue_create_host_buffer[DType.int32](s)
    for i in range(s):
        hs[i] = Int32(slots[i])
    var ds = lm.ctx.enqueue_create_buffer[DType.int32](s)
    lm.ctx.enqueue_copy(ds, hs)
    var rows = lm.new(s * dm)
    lm.ctx.enqueue_function[gather_rows_kernel](h, ds, rows, Int64(dm), grid_dim=(_blocks(dm), s), block_dim=TPB)
    var logz = lm.new(s * v)
    lm.ctx.enqueue_function[lm_head_kernel](
        rows, lm.w("embed_tokens.weight"), logz, Int64(v), Int64(dm), grid_dim=((v + 7) // 8, s), block_dim=256,
    )
    lm.ctx.enqueue_function[log_softmax_rows_kernel](logz, Int64(v), grid_dim=s, block_dim=TPB)
    var lz = lm.download(logz, s * v)
    # d logits = softmax - onehot(target), per slot
    var hd = lm.ctx.enqueue_create_host_buffer[DT](s * v)
    var loss = 0.0
    for i in range(s):
        loss -= Float64(lz[i * v + targets[i]])
        for k in range(v):
            hd[i * v + k] = exp(lz[i * v + k])
        hd[i * v + targets[i]] = hd[i * v + targets[i]] - 1.0
    var dl = lm.new(s * v)
    lm.ctx.enqueue_copy(dl, hd)
    var dh = lm.new(s * dm)
    lm.ctx.enqueue_function[lm_head_bwd_kernel](
        dl, lm.w("embed_tokens.weight"), dh, Int64(v), Int64(dm), grid_dim=(_blocks(dm), s), block_dim=TPB,
    )
    var dfinal = lm.new(t * dm)
    dfinal.enqueue_fill(0.0)
    lm.ctx.enqueue_function[scatter_rows_kernel](dh, ds, dfinal, Int64(dm), grid_dim=(_blocks(dm), s), block_dim=TPB)
    return (loss, dfinal^)


def backward(mut lm: Lfm3Text, dfinal: DeviceBuffer[DT], t: Int) raises -> DeviceBuffer[DT]:
    """Through the final norm and every layer, in reverse, from the cache of
    the last `forward` with `training = True`. Returns d loss / d embeddings."""
    var dm = lm.cfg.hidden
    var hd = lm.cfg.head_dim
    var nh = lm.cfg.heads
    var nkv = lm.cfg.kv_heads
    var ff = lm.cfg.ff
    if hd != 128:
        raise Error("lfm3: the gate's backward is written for head_dim 128")
    if len(lm.act) != len(lm.cfg.layer_types):
        raise Error("lfm3: backward needs a forward with training = True first")
    var x_last = lm.act_x_last[0]
    var dx = _rms_bwd(lm, "embedding_norm", x_last, dfinal, t, dm)
    # `lm.debug` (falsification only): 8 = the softmax's VJP skipped (dS = dP), 16 = the
    # RoPE gradient rotated the wrong way, 32 = q's per-head norm skipped in the backward.
    var neg = (lm.debug & 16) == 0
    var rf = lm.rope_tables(t, lm.cfg.theta_full, lm.cfg.prf_full, neg_sin=neg)
    var rs = lm.rope_tables(t, lm.cfg.theta_sliding, lm.cfg.prf_sliding, neg_sin=neg)
    var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
    var li = len(lm.cfg.layer_types) - 1
    while li >= 0:
        var p = String("layers.") + String(li) + "."
        var kind = lm.cfg.layer_types[li]
        var c = lm.act[li].copy()  # this layer's cached handles, borrowed apart from `lm`
        # x_out = h_mid + post_ffn_norm(f), f = w2(silu(w1 fn) * w3 fn), fn = ffn_norm(h_mid)
        var df = _rms_bwd(lm, p + "post_ffn_norm", c["f"], dx, t, dm)
        var dsw = linear_bwd(lm, li, p + "feed_forward.w2", c["sw"], df, t, ff)
        var da1 = lm.new(t * ff)
        var da3 = lm.new(t * ff)
        lm.ctx.enqueue_function[swiglu_bwd_kernel](
            c["a1"], c["a3"], dsw, da1, da3, Int64(t * ff), grid_dim=_blocks(t * ff), block_dim=TPB,
        )
        var dfn = linear_bwd(lm, li, p + "feed_forward.w1", c["fn"], da1, t, dm)
        var dfn3 = linear_bwd(lm, li, p + "feed_forward.w3", c["fn"], da3, t, dm)
        _add(lm, dfn, dfn3, t * dm)
        var dh_mid = _rms_bwd(lm, p + "ffn_norm", c["h_mid"], dfn, t, dm)
        _add(lm, dh_mid, dx, t * dm)
        # h_mid = x_in + post_operator_norm(OP(operator_norm(x_in)))
        var do_ = _rms_bwd(lm, p + "post_operator_norm", c["o"], dh_mid, t, dm)
        var dhn: DeviceBuffer[DT]
        if kind == LT_CONV:
            var dz = linear_bwd(lm, li, p + "conv.out_proj", c["z"], do_, t, dm)
            var dbcx = lm.new(t * 3 * dm)
            lm.ctx.enqueue_function[short_conv_bwd_kernel](
                c["bcx"], lm.w(p + "conv.conv.weight"), dz, dbcx, Int64(t), Int64(dm), Int64(lm.cfg.conv_k),
                grid_dim=(_blocks(dm), t), block_dim=TPB,
            )
            dhn = linear_bwd(lm, li, p + "conv.in_proj", c["hn"], dbcx, t, dm)
        else:
            var da = linear_bwd(lm, li, p + "self_attn.o_proj", c["a"], do_, t, dm)
            var da_pre = lm.new(t * dm)
            var dg = lm.new(t * nh)
            lm.ctx.enqueue_function[gate_bwd_kernel](
                c["a_pre"], c["g"], da, da_pre, dg, Int64(nh), Int64(hd), grid_dim=t * nh, block_dim=128,
            )
            var dsm = lm.new(nh * t * t)
            lm.ctx.enqueue_function[attn_dp_kernel](
                da_pre, c["v"], c["P"], dsm, Int64(t), Int64(nh), Int64(nkv), Int64(hd),
                grid_dim=(_blocks(t), t, nh), block_dim=TPB,
            )
            if (lm.debug & 8) == 0:
                lm.ctx.enqueue_function[softmax_bwd_rows_kernel](
                    c["P"], dsm, Int64(t), scale, grid_dim=nh * t, block_dim=TPB,
                )
            var dq = lm.new(t * dm)
            lm.ctx.enqueue_function[attn_dq_kernel](
                dsm, c["k"], dq, Int64(t), Int64(nh), Int64(nkv), Int64(hd), grid_dim=(_blocks(nh * hd), t), block_dim=TPB,
            )
            var dk = lm.new(t * nkv * hd)
            var dv = lm.new(t * nkv * hd)
            lm.ctx.enqueue_function[attn_dkv_kernel](
                dsm, c["P"], c["q"], da_pre, dk, dv, Int64(t), Int64(nh), Int64(nkv), Int64(hd),
                grid_dim=(_blocks(nkv * hd), t), block_dim=TPB,
            )
            var cos_t = rf[0] if kind == LT_FULL else rs[0]
            var sin_t = rf[1] if kind == LT_FULL else rs[1]
            lm.ctx.enqueue_function[rope_kernel](
                dq, cos_t, sin_t, Int64(nh), Int64(hd // 2), grid_dim=(_blocks(nh * hd // 2), t), block_dim=TPB,
            )
            lm.ctx.enqueue_function[rope_kernel](
                dk, cos_t, sin_t, Int64(nkv), Int64(hd // 2), grid_dim=(_blocks(nkv * hd // 2), t), block_dim=TPB,
            )
            var dq_lin = dq if (lm.debug & 32) != 0 else _rms_bwd(lm, p + "self_attn.q_norm", c["q_lin"], dq, t * nh, hd)
            var dk_lin = _rms_bwd(lm, p + "self_attn.k_norm", c["k_lin"], dk, t * nkv, hd)
            dhn = linear_bwd(lm, li, p + "self_attn.q_proj", c["hn"], dq_lin, t, dm)
            var d2 = linear_bwd(lm, li, p + "self_attn.k_proj", c["hn"], dk_lin, t, dm)
            var d3 = linear_bwd(lm, li, p + "self_attn.v_proj", c["hn"], dv, t, dm)
            var d4 = linear_bwd(lm, li, p + "self_attn.gate_proj", c["hn"], dg, t, dm)
            _add(lm, dhn, d2, t * dm)
            _add(lm, dhn, d3, t * dm)
            _add(lm, dhn, d4, t * dm)
        var dx_in = _rms_bwd(lm, p + "operator_norm", c["x_in"], dhn, t, dm)
        _add(lm, dx_in, dh_mid, t * dm)
        dx = dx_in^
        li -= 1
    return dx^


struct LoraAdamW(Movable):
    """AdamW over every adapter of an `Lfm3Text` (its `lora_a` / `lora_b`),
    the gradients zeroed by each step. Liquid's cookbook: lr 2e-4."""

    var lr: Float32
    var b1: Float32
    var b2: Float32
    var eps: Float32
    var wd: Float32
    var t: Int
    var m: List[DeviceBuffer[DT]]
    var v: List[DeviceBuffer[DT]]

    def __init__(out self, mut lm: Lfm3Text, lr: Float32 = 2e-4, wd: Float32 = 0.0) raises:
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
        self.wd = wd
        self.t = 0
        self.m = List[DeviceBuffer[DT]]()
        self.v = List[DeviceBuffer[DT]]()
        for j in range(len(lm.lora_a)):
            for which in range(2):
                var n = len(lm.lora_a[j]) if which == 0 else len(lm.lora_b[j])
                var mb = lm.new(n)
                mb.enqueue_fill(0.0)
                var vb = lm.new(n)
                vb.enqueue_fill(0.0)
                self.m.append(mb^)
                self.v.append(vb^)

    def step(mut self, mut lm: Lfm3Text) raises:
        self.t += 1
        var c1 = Float32(1.0 - Float64(self.b1) ** self.t)
        var c2 = Float32(1.0 - Float64(self.b2) ** self.t)
        for j in range(len(lm.lora_a)):
            for which in range(2):
                var p = lm.lora_a[j] if which == 0 else lm.lora_b[j]
                var g = lm.lora_da[j] if which == 0 else lm.lora_db[j]
                var n = len(p)
                lm.ctx.enqueue_function[adamw_kernel](
                    p, g, self.m[2 * j + which], self.v[2 * j + which], Int64(n), self.lr, self.b1, self.b2,
                    self.eps, self.wd, c1, c2, grid_dim=_blocks(n), block_dim=TPB,
                )


def train_step(
    mut lm: Lfm3Text, mut opt: LoraAdamW, ids: List[Int], groups: List[Int], state_len: Int,
    slots: List[Int], targets: List[Int],
) raises -> Float64:
    """One row: forward (training), the slots' cross-entropy, backward, AdamW.
    Returns the loss before the step."""
    lm.training = True
    var taps = List[DeviceBuffer[DT]]()
    var h = lm.forward(ids, groups, state_len, taps)
    var lt = slot_loss(lm, h, len(ids), slots, targets)
    _ = backward(lm, lt[1], len(ids))
    opt.step(lm)
    lm.training = False
    lm.act = List[Dict[String, DeviceBuffer[DT]]]()  # free the activations
    return lt[0]
