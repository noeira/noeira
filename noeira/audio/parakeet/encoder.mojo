# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — the FastConformer encoder, at a fixed maximum length
# +--------------------------------------------------------------------------+ #
"""Mel frames -> encoder frames, for `ParakeetTDT`'s weights.

    var ws = PkEncoderWork[801]()             # 8 s of audio: 801 mel frames, 101 encoder frames
    # ... fill ws.feats ([801, 128], zero beyond the valid mel frames) ...
    var t = pk_encode[801](model, ws, n_mel)  # valid encoder frames; ws.proj is [101, 640]

## Fixed maximum, runtime length (docs/NN_MODEL_PORTS_PLAN.md §9.3)

Every buffer and every `Linear` / `LayerNorm` call is sized by the COMPILED
maximum (`F_MAX` mel frames, hence `T` encoder frames); the clip's real length is
a runtime number. That is the reference's own padded path — a clip batched with
a longer one — and it stays exact for the valid frames because of three masks,
all reproduced here:

1. **Subsampling**: after every convolution, rows beyond the current valid
   length are zeroed (the length shrinks `n -> (n - 1) // 2 + 1` per stride-2
   conv). Without it the next 3x3 conv would read padding as signal.
2. **Attention**: a valid frame attends to valid frames only.
3. **Conformer conv**: rows beyond the valid length are zeroed before the
   kernel-9 depthwise conv, which would otherwise pull them in.

Feed-forwards and norms are per-frame, so padded rows compute something and
nothing reads it. Outputs beyond the valid length are NOT meaningful.

## Relative positions

Transformer-XL attention: `logit[i, j] = ((q_i + u) . k_j + (q_i + v) . r_{i-j}) / sqrt(128)`
with `r = rel_k(sinusoid)`. The table holds offsets `T-1 .. -(T-1)`, so offset
`i - j` is row `j - i + T - 1`, whatever `T` the table was built for — which is
why a compiled maximum does not change the valid frames' result.

CPU only for now, written for exactness against the reference rather than
speed: the GEMMs go through `Linear` (Accelerate on macOS), the rest is plain
loops. `tests/audio/test_parakeet_reference.mojo` gates every stage.
"""

from std.math import exp, sin, cos, sqrt

from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from .config import (
    PK_MELS, PK_D, PK_FF, PK_HEADS, PK_HEAD_DIM, PK_LAYERS, PK_CONV_K,
    PK_SUB_CH, PK_SUB_FREQ, PK_SUB_FLAT, PK_DEC, PK_BN_EPS, pk_sub_len,
)
from .weights import ParakeetTDT


def pk_pos_table(t: Int) -> List[Float32]:
    """The relative positional encoding for `t` frames: `[2t - 1, D]`, row `p`
    is offset `t - 1 - p`, sin and cos interleaved. Float32 arithmetic in the
    reference's order (`inv_freq` in float32, then the angle)."""
    var out = List[Float32](length=(2 * t - 1) * PK_D, fill=0.0)
    for p in range(2 * t - 1):
        var position = Float32(t - 1 - p)
        for m in range(PK_D // 2):
            var e = Float32(2 * m) / Float32(PK_D)
            var inv_freq: Float32 = 1.0 / (Float32(10000.0) ** e)
            var angle = position * inv_freq
            out[p * PK_D + 2 * m] = sin(angle)
            out[p * PK_D + 2 * m + 1] = cos(angle)
    return out^


struct PkEncoderWork[F_MAX: Int](Movable):
    """The encoder's buffers at `F_MAX` mel frames. `feats` is the input,
    `x` the hidden state (`[T, 1024]`) after `pk_subsample` and each
    `pk_block`, `proj` the encoder projection (`[T, 640]`)."""

    comptime F1: Int = pk_sub_len(Self.F_MAX)
    comptime F2: Int = pk_sub_len(Self.F1)
    comptime T: Int = pk_sub_len(Self.F2)
    comptime P: Int = 2 * Self.T - 1
    comptime W1: Int = PK_MELS // 2
    comptime W2: Int = PK_MELS // 4

    var feats: Tensor
    var a1: Tensor
    var b1: Tensor
    var c1: Tensor
    var b2: Tensor
    var c2: Tensor
    var flat: Tensor
    var x: Tensor
    var h: Tensor
    var y: Tensor
    var ff: Tensor
    var q: Tensor
    var k: Tensor
    var v: Tensor
    var att: Tensor
    var pos: Tensor
    var relk: Tensor
    var glu: Tensor
    var conv: Tensor
    var proj: Tensor
    var scores: List[Float32]

    def __init__(out self):
        comptime assert PK_SUB_FREQ == PK_MELS // 8, "three stride-2 convs leave MELS / 8 bins"
        self.feats = Tensor.alloc(Self.F_MAX * PK_MELS)
        self.a1 = Tensor.alloc(Self.F1 * Self.W1 * PK_SUB_CH)
        self.b1 = Tensor.alloc(Self.F2 * Self.W2 * PK_SUB_CH)
        self.c1 = Tensor.alloc(Self.F2 * Self.W2 * PK_SUB_CH)
        self.b2 = Tensor.alloc(Self.T * PK_SUB_FLAT)
        self.c2 = Tensor.alloc(Self.T * PK_SUB_FLAT)
        self.flat = Tensor.alloc(Self.T * PK_SUB_FLAT)
        self.x = Tensor.alloc(Self.T * PK_D)
        self.h = Tensor.alloc(Self.T * PK_D)
        self.y = Tensor.alloc(Self.T * PK_D)
        self.ff = Tensor.alloc(Self.T * PK_FF)
        self.q = Tensor.alloc(Self.T * PK_D)
        self.k = Tensor.alloc(Self.T * PK_D)
        self.v = Tensor.alloc(Self.T * PK_D)
        self.att = Tensor.alloc(Self.T * PK_D)
        self.pos = Tensor.alloc(Self.P * PK_D)
        var table = pk_pos_table(Self.T)
        for i in range(len(table)):
            self.pos.data[i] = table[i]
        self.relk = Tensor.alloc(Self.P * PK_D)
        self.glu = Tensor.alloc(Self.T * 2 * PK_D)
        self.conv = Tensor.alloc(Self.T * PK_D)
        self.proj = Tensor.alloc(Self.T * PK_DEC)
        self.scores = List[Float32](length=Self.T, fill=0.0)


# ── elementwise pieces ─────────────────────────────────────────────────────


def _silu(mut t: Tensor, n: Int):
    for i in range(n):
        var v = t.data[i]
        t.data[i] = v / (1.0 + exp(-v))


def _relu_rows(mut t: Tensor, row: Int, valid: Int, rows: Int):
    """ReLU on the first `valid` rows of `[rows, row]`, zero on the others."""
    for i in range(valid * row):
        if t.data[i] < 0.0:
            t.data[i] = 0.0
    for i in range(valid * row, rows * row):
        t.data[i] = 0.0


def _add_scaled(mut x: Tensor, ref y: Tensor, n: Int, scale: Float32):
    for i in range(n):
        x.data[i] += scale * y.data[i]


# ── subsampling convolutions (channel-last, CPU) ───────────────────────────


def _conv3x3_s2_first(
    ref src: Tensor, src_h: Int, src_w: Int,
    ref w: Tensor, ref b: Tensor,
    mut dst: Tensor, dst_h: Int, dst_w: Int, valid: Int,
):
    """`src [src_h, src_w]` (one channel) -> `dst [dst_h, dst_w, C]`: 3x3,
    stride 2, padding 1, then ReLU. Rows beyond `valid` are zero."""
    comptime C = PK_SUB_CH
    var p = List[Float32](length=9, fill=0.0)
    for t in range(valid):
        for x in range(dst_w):
            for kh in range(3):
                var sy = 2 * t - 1 + kh
                for kw in range(3):
                    var sx = 2 * x - 1 + kw
                    var inside = sy >= 0 and sy < src_h and sx >= 0 and sx < src_w
                    p[kh * 3 + kw] = src.data[sy * src_w + sx] if inside else Float32(0.0)
            var base = (t * dst_w + x) * C
            for c in range(C):
                var acc = b.data[c]
                for j in range(9):
                    acc += p[j] * w.data[c * 9 + j]
                dst.data[base + c] = acc if acc > 0.0 else Float32(0.0)
    for i in range(valid * dst_w * C, dst_h * dst_w * C):
        dst.data[i] = 0.0


def _dw3x3_s2(
    ref src: Tensor, src_h: Int, src_w: Int,
    ref w: Tensor, ref b: Tensor,
    mut dst: Tensor, dst_h: Int, dst_w: Int, valid: Int,
):
    """Depthwise 3x3, stride 2, padding 1, channel-last: `src [src_h, src_w, C]`
    -> `dst [dst_h, dst_w, C]`. Rows beyond `valid` are zero."""
    comptime C = PK_SUB_CH
    for t in range(valid):
        for x in range(dst_w):
            var base = (t * dst_w + x) * C
            for c in range(C):
                dst.data[base + c] = b.data[c]
            for kh in range(3):
                var sy = 2 * t - 1 + kh
                if sy < 0 or sy >= src_h:
                    continue
                for kw in range(3):
                    var sx = 2 * x - 1 + kw
                    if sx < 0 or sx >= src_w:
                        continue
                    var sbase = (sy * src_w + sx) * C
                    var j = kh * 3 + kw
                    for c in range(C):
                        dst.data[base + c] += src.data[sbase + c] * w.data[c * 9 + j]
    for i in range(valid * dst_w * C, dst_h * dst_w * C):
        dst.data[i] = 0.0


def _zero_rows(mut t: Tensor, row: Int, valid: Int, rows: Int):
    for i in range(valid * row, rows * row):
        t.data[i] = 0.0


def pk_subsample[F_MAX: Int](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX], n_mel: Int
) raises -> Int:
    """`ws.feats` (`[F_MAX, 128]`, `n_mel` valid rows, zero beyond) -> `ws.x`
    (`[T, 1024]`). Returns the valid encoder frames."""
    comptime W = PkEncoderWork[F_MAX]
    comptime C = PK_SUB_CH
    if n_mel < 1 or n_mel > F_MAX:
        raise Error("pk_subsample: n_mel out of range for the compiled maximum")
    var len1 = pk_sub_len(n_mel)
    var len2 = pk_sub_len(len1)
    var len3 = pk_sub_len(len2)

    _conv3x3_s2_first(
        ws.feats, F_MAX, PK_MELS, m.sub.conv0.weight.val, m.sub.conv0.bias.val,
        ws.a1, W.F1, W.W1, len1,
    )
    _dw3x3_s2(
        ws.a1, W.F1, W.W1, m.sub.dw1.weight.val, m.sub.dw1.bias.val,
        ws.b1, W.F2, W.W2, len2,
    )
    m.sub.pw1.forward["cpu", W.F2 * W.W2](TensorRefs[1](ws.b1), ws.c1)
    _relu_rows(ws.c1, W.W2 * C, len2, W.F2)
    _dw3x3_s2(
        ws.c1, W.F2, W.W2, m.sub.dw2.weight.val, m.sub.dw2.bias.val,
        ws.b2, W.T, PK_SUB_FREQ, len3,
    )
    m.sub.pw2.forward["cpu", W.T * PK_SUB_FREQ](TensorRefs[1](ws.b2), ws.c2)
    _relu_rows(ws.c2, PK_SUB_FREQ * C, len3, W.T)

    # [T, freq, C] -> [T, C * freq]: the reference flattens channel-major.
    for t in range(W.T):
        for f in range(PK_SUB_FREQ):
            var src = (t * PK_SUB_FREQ + f) * C
            for c in range(C):
                ws.flat.data[t * PK_SUB_FLAT + c * PK_SUB_FREQ + f] = ws.c2.data[src + c]
    m.sub.linear.forward["cpu", W.T](TensorRefs[1](ws.flat), ws.x)
    return len3


# ── one conformer block ────────────────────────────────────────────────────


def _rel_attention(
    ref q: Tensor, ref k: Tensor, ref v: Tensor, ref relk: Tensor,
    ref bias_u: Tensor, ref bias_v: Tensor,
    mut out: Tensor, mut scores: List[Float32], t_max: Int, valid: Int,
):
    """Relative-position multi-head attention over the first `valid` of
    `t_max` frames; `out` rows beyond `valid` are zero."""
    comptime D = PK_D
    comptime HD = PK_HEAD_DIM
    var scale = Float32(1.0 / sqrt(Float64(HD)))
    var qu = List[Float32](length=HD, fill=0.0)
    var qv = List[Float32](length=HD, fill=0.0)
    for i in range(t_max * D):
        out.data[i] = 0.0
    for h in range(PK_HEADS):
        var ho = h * HD
        for i in range(valid):
            for d in range(HD):
                var qd = q.data[i * D + ho + d]
                qu[d] = qd + bias_u.data[ho + d]
                qv[d] = qd + bias_v.data[ho + d]
            var top = Float32(-3.0e38)
            for j in range(valid):
                var kb = j * D + ho
                var rb = (j - i + t_max - 1) * D + ho
                var ac = Float32(0.0)
                var bd = Float32(0.0)
                for d in range(HD):
                    ac += qu[d] * k.data[kb + d]
                    bd += qv[d] * relk.data[rb + d]
                var s = ac * scale + bd * scale
                scores[j] = s
                if s > top:
                    top = s
            var total = Float32(0.0)
            for j in range(valid):
                scores[j] = exp(scores[j] - top)
                total += scores[j]
            var ob = i * D + ho
            for j in range(valid):
                var wj = scores[j] / total
                var vb = j * D + ho
                for d in range(HD):
                    out.data[ob + d] += wj * v.data[vb + d]


def _glu_rows(ref src: Tensor, mut dst: Tensor, valid: Int, rows: Int):
    """`dst[t, c] = src[t, c] * sigmoid(src[t, D + c])` on the valid rows, zero
    on the others (the mask the depthwise conv needs)."""
    comptime D = PK_D
    for t in range(valid):
        for c in range(D):
            var g = src.data[t * 2 * D + D + c]
            dst.data[t * D + c] = src.data[t * 2 * D + c] / (1.0 + exp(-g))
    for i in range(valid * D, rows * D):
        dst.data[i] = 0.0


def _dw1d_bn_silu(
    ref src: Tensor, ref w: Tensor,
    ref gamma: Tensor, ref beta: Tensor, ref mean: Tensor, ref variance: Tensor,
    mut dst: Tensor, valid: Int, rows: Int,
):
    """Depthwise Conv1d over time (kernel 9, padding 4, no bias), BatchNorm with
    the running statistics, SiLU. `src` rows beyond `valid` are zero; so are
    `dst`'s."""
    comptime D = PK_D
    comptime K = PK_CONV_K
    for t in range(valid):
        var ob = t * D
        for c in range(D):
            dst.data[ob + c] = 0.0
        for j in range(K):
            var tt = t - K // 2 + j
            if tt < 0 or tt >= valid:
                continue
            var sb = tt * D
            for c in range(D):
                dst.data[ob + c] += src.data[sb + c] * w.data[c * K + j]
        for c in range(D):
            var n = (dst.data[ob + c] - mean.data[c]) / sqrt(
                variance.data[c] + Float32(PK_BN_EPS)
            ) * gamma.data[c] + beta.data[c]
            dst.data[ob + c] = n / (1.0 + exp(-n))
    for i in range(valid * D, rows * D):
        dst.data[i] = 0.0


def pk_block[F_MAX: Int](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX], layer: Int, valid: Int
) raises:
    """Apply conformer block `layer` to `ws.x` in place (`valid` of `T` frames)."""
    comptime T = PkEncoderWork[F_MAX].T
    comptime P = PkEncoderWork[F_MAX].P
    comptime D = PK_D

    # 1. first feed-forward, half-weighted
    m.layers[layer].norm_ff1.forward["cpu", T](TensorRefs[1](ws.x), ws.h)
    m.layers[layer].ff1_a.forward["cpu", T](TensorRefs[1](ws.h), ws.ff)
    _silu(ws.ff, T * PK_FF)
    m.layers[layer].ff1_b.forward["cpu", T](TensorRefs[1](ws.ff), ws.y)
    _add_scaled(ws.x, ws.y, T * D, 0.5)

    # 2. relative-position self-attention
    m.layers[layer].norm_att.forward["cpu", T](TensorRefs[1](ws.x), ws.h)
    m.layers[layer].q.forward["cpu", T](TensorRefs[1](ws.h), ws.q)
    m.layers[layer].k.forward["cpu", T](TensorRefs[1](ws.h), ws.k)
    m.layers[layer].v.forward["cpu", T](TensorRefs[1](ws.h), ws.v)
    m.layers[layer].rel_k.forward["cpu", P](TensorRefs[1](ws.pos), ws.relk)
    _rel_attention(
        ws.q, ws.k, ws.v, ws.relk,
        m.layers[layer].bias_u.val, m.layers[layer].bias_v.val,
        ws.att, ws.scores, T, valid,
    )
    m.layers[layer].o.forward["cpu", T](TensorRefs[1](ws.att), ws.y)
    _add_scaled(ws.x, ws.y, T * D, 1.0)

    # 3. convolution module
    m.layers[layer].norm_conv.forward["cpu", T](TensorRefs[1](ws.x), ws.h)
    m.layers[layer].pw1.forward["cpu", T](TensorRefs[1](ws.h), ws.glu)
    _glu_rows(ws.glu, ws.h, valid, T)
    _dw1d_bn_silu(
        ws.h, m.layers[layer].dw.weight.val,
        m.layers[layer].bn.gamma.val, m.layers[layer].bn.beta.val,
        m.layers[layer].bn.running_mean.t, m.layers[layer].bn.running_var.t,
        ws.conv, valid, T,
    )
    m.layers[layer].pw2.forward["cpu", T](TensorRefs[1](ws.conv), ws.y)
    _add_scaled(ws.x, ws.y, T * D, 1.0)

    # 4. second feed-forward, half-weighted
    m.layers[layer].norm_ff2.forward["cpu", T](TensorRefs[1](ws.x), ws.h)
    m.layers[layer].ff2_a.forward["cpu", T](TensorRefs[1](ws.h), ws.ff)
    _silu(ws.ff, T * PK_FF)
    m.layers[layer].ff2_b.forward["cpu", T](TensorRefs[1](ws.ff), ws.y)
    _add_scaled(ws.x, ws.y, T * D, 0.5)

    # 5. output norm
    m.layers[layer].norm_out.forward["cpu", T](TensorRefs[1](ws.x), ws.h)
    for i in range(T * D):
        ws.x.data[i] = ws.h.data[i]


def pk_encode[F_MAX: Int](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX], n_mel: Int
) raises -> Int:
    """The whole encoder: `ws.feats` -> `ws.x` (`[T, 1024]`) and `ws.proj`
    (`[T, 640]`, what the joint network reads). Returns the valid frames."""
    comptime T = PkEncoderWork[F_MAX].T
    var valid = pk_subsample[F_MAX](m, ws, n_mel)
    for layer in range(PK_LAYERS):
        pk_block[F_MAX](m, ws, layer, valid)
    m.enc_proj.forward["cpu", T](TensorRefs[1](ws.x), ws.proj)
    return valid
