# +--------------------------------------------------------------------------+ #
# | Kokoro-82M — phonemes to durations
# +--------------------------------------------------------------------------+ #
"""The first half of `KModel.forward_with_tokens`: PL-BERT, then how long each
phoneme lasts.

    bert      = ALBERT(input_ids)                      [T, 768]
    d_en      = bert_encoder(bert)                     [T, 512]
    dur_enc   = DurationEncoder(d_en, style)           [T, 640]
    dur_lstm  = predictor.lstm(dur_enc)                [T, 512]
    duration  = sum(sigmoid(duration_proj(dur_lstm))) / speed
    pred_dur  = max(round(duration), 1)                frames per phoneme

`input_ids` is `0, the phoneme ids, 0`; `style` is the PROSODY half of the
voice's style row (columns 128-255 of `ref_s`; the decoder takes the other).

ALBERT here is one transformer layer whose weights are applied twelve times
(`num_hidden_groups = 1`): embeddings 128 wide (word + position + token type 0,
LayerNorm eps 1e-12), a linear to 768, then 12 x [attention (12 heads of 64),
LayerNorm(x + .), FFN 768 -> 2048 -> 768 with `gelu_new`, LayerNorm(. + .)].

The duration encoder is three (bidirectional LSTM 640 -> 512, AdaLayerNorm,
style concatenated back to 640). AdaLayerNorm is a LayerNorm (eps 1e-5, no
affine of its own) whose scale and shift come from the style:
`(1 + gamma) * norm(x) + beta`, `[gamma | beta] = fc(style)`.

⚠ `pred_dur` sets every later shape (frames, samples), so it is the one output
of this file that is gated for EQUALITY, not within a bound. torch rounds
halves to even; so does `_round_half_even`.
"""

from std.math import exp, floor, sqrt, tanh

from noeira.nn.core.tensor import Tensor
from .ops import (
    kk_linear, kk_layer_norm, kk_layer_norm_affine, kk_bilstm, kk_conv1d,
    kk_adain_resblk,
)
from .weights import KokoroWeights


comptime KK_BERT_DIM: Int = 768
comptime KK_BERT_EMB: Int = 128
comptime KK_BERT_HEADS: Int = 12
comptime KK_BERT_LAYERS: Int = 12
comptime KK_HIDDEN: Int = 512
comptime KK_STYLE: Int = 128
comptime KK_MAX_DUR: Int = 50
comptime KK_CONTEXT: Int = 512


def _gelu_new(x: Float32) -> Float32:
    """`0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`."""
    return 0.5 * x * (1.0 + tanh(Float32(0.7978845608028654) * (x + Float32(0.044715) * x * x * x)))


def _round_half_even(x: Float32) -> Int:
    """`torch.round`: halves go to the even neighbour."""
    var f = floor(x)
    var d = x - f
    var i = Int(f)
    if d > 0.5:
        return i + 1
    if d < 0.5:
        return i
    return i if i % 2 == 0 else i + 1


def kk_albert(mut w: KokoroWeights, ref ids: List[Int], mut out: Tensor) raises:
    """`out[T, 768]` = PL-BERT's last hidden state for `ids` (no padding)."""
    comptime D = KK_BERT_DIM
    comptime E = KK_BERT_EMB
    comptime HD = D // KK_BERT_HEADS
    var t = len(ids)
    if t > KK_CONTEXT:
        raise Error("kokoro: " + String(t) + " tokens, the model's context is " + String(KK_CONTEXT))
    var word = w.at("bert.embeddings.word_embeddings.weight")
    var pos = w.at("bert.embeddings.position_embeddings.weight")
    var typ = w.at("bert.embeddings.token_type_embeddings.weight")
    var emb = Tensor.alloc(t * E)
    for i in range(t):
        for d in range(E):
            emb.data[i * E + d] = (
                w.tensors[word].data[ids[i] * E + d] + w.tensors[typ].data[d]
                + w.tensors[pos].data[i * E + d]
            )
    kk_layer_norm_affine(
        w, "bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.bias",
        emb, t, 1e-12,
    )
    var x = Tensor.alloc(t * D)
    kk_linear(w, "bert.encoder.embedding_hidden_mapping_in", emb, t, x)

    var layer = String("bert.encoder.albert_layer_groups.0.albert_layers.0.")
    var q = Tensor.alloc(t * D)
    var k = Tensor.alloc(t * D)
    var v = Tensor.alloc(t * D)
    var ctx = Tensor.alloc(t * D)
    var proj = Tensor.alloc(t * D)
    var mid = Tensor.alloc(t * 2048)
    var ffo = Tensor.alloc(t * D)
    var scores = List[Float32](length=t, fill=0.0)
    var scale = Float32(1.0 / sqrt(Float64(HD)))
    for _ in range(KK_BERT_LAYERS):
        kk_linear(w, layer + "attention.query", x, t, q)
        kk_linear(w, layer + "attention.key", x, t, k)
        kk_linear(w, layer + "attention.value", x, t, v)
        for i in range(t * D):
            ctx.data[i] = 0.0
        for h in range(KK_BERT_HEADS):
            var ho = h * HD
            for i in range(t):
                var top = Float32(-3.0e38)
                for j in range(t):
                    var acc = Float32(0.0)
                    for d in range(HD):
                        acc += q.data[i * D + ho + d] * k.data[j * D + ho + d]
                    scores[j] = acc * scale
                    if scores[j] > top:
                        top = scores[j]
                var total = Float32(0.0)
                for j in range(t):
                    scores[j] = exp(scores[j] - top)
                    total += scores[j]
                for j in range(t):
                    var p = scores[j] / total
                    for d in range(HD):
                        ctx.data[i * D + ho + d] += p * v.data[j * D + ho + d]
        kk_linear(w, layer + "attention.dense", ctx, t, proj)
        for i in range(t * D):
            proj.data[i] += x.data[i]
        kk_layer_norm_affine(
            w, layer + "attention.LayerNorm.weight", layer + "attention.LayerNorm.bias",
            proj, t, 1e-12,
        )
        kk_linear(w, layer + "ffn", proj, t, mid)
        for i in range(t * 2048):
            mid.data[i] = _gelu_new(mid.data[i])
        kk_linear(w, layer + "ffn_output", mid, t, ffo)
        for i in range(t * D):
            x.data[i] = ffo.data[i] + proj.data[i]
        kk_layer_norm_affine(
            w, layer + "full_layer_layer_norm.weight", layer + "full_layer_layer_norm.bias",
            x, t, 1e-12,
        )
    out.ensure(t * D)
    for i in range(t * D):
        out.data[i] = x.data[i]


struct KokoroProsody(Movable):
    """What `kk_durations` computed, stage by stage (the gate reads them all)."""

    var tokens: Int
    var bert: Tensor
    """`[T, 768]`"""
    var d_en: Tensor
    """`[T, 512]`"""
    var dur_enc: Tensor
    """`[T, 640]`: the duration encoder's output, style columns last."""
    var dur_lstm: Tensor
    """`[T, 512]`"""
    var duration: List[Float32]
    var pred_dur: List[Int]
    var frames: Int

    def __init__(out self):
        self.tokens = 0
        self.bert = Tensor()
        self.d_en = Tensor()
        self.dur_enc = Tensor()
        self.dur_lstm = Tensor()
        self.duration = List[Float32]()
        self.pred_dur = List[Int]()
        self.frames = 0


def kk_durations(
    mut w: KokoroWeights, ref ids: List[Int], ref ref_s: List[Float32],
    speed: Float32 = 1.0,
) raises -> KokoroProsody:
    """`ids` (boundary zeros included) and the voice's 256-wide style row ->
    the prosody stages and the frames per phoneme."""
    comptime H = KK_HIDDEN
    comptime S = KK_STYLE
    comptime W = H + S
    if len(ref_s) != 2 * S:
        raise Error("kokoro: a style row is 256 values")
    var t = len(ids)
    var p = KokoroProsody()
    p.tokens = t
    kk_albert(w, ids, p.bert)
    kk_linear(w, "bert_encoder", p.bert, t, p.d_en)

    # Duration encoder: [d_en | style], then 3 x (bi-LSTM, AdaLayerNorm, | style).
    var x = Tensor.alloc(t * W)
    for i in range(t):
        for d in range(H):
            x.data[i * W + d] = p.d_en.data[i * H + d]
        for d in range(S):
            x.data[i * W + H + d] = ref_s[S + d]
    var style = Tensor.alloc(S)
    for d in range(S):
        style.data[d] = ref_s[S + d]
    var lstm_out = Tensor.alloc(t * H)
    var gb = Tensor.alloc(2 * H)
    for block in range(3):
        var lstm = String("predictor.text_encoder.lstms." + String(2 * block))
        var ada = String("predictor.text_encoder.lstms." + String(2 * block + 1))
        _ = kk_bilstm(w, lstm, x, t, lstm_out)
        kk_linear(w, ada + ".fc", style, 1, gb)
        kk_layer_norm(lstm_out, t, H, 1e-5)
        for i in range(t):
            for d in range(H):
                x.data[i * W + d] = (1.0 + gb.data[d]) * lstm_out.data[i * H + d] + gb.data[H + d]
            # the style columns of `x` are unchanged: concatenated back as they were
    p.dur_enc.ensure(t * W)
    for i in range(t * W):
        p.dur_enc.data[i] = x.data[i]

    _ = kk_bilstm(w, "predictor.lstm", p.dur_enc, t, p.dur_lstm)
    var logits = Tensor.alloc(t * KK_MAX_DUR)
    kk_linear(w, "predictor.duration_proj.linear_layer", p.dur_lstm, t, logits)
    for i in range(t):
        var total = Float32(0.0)
        for d in range(KK_MAX_DUR):
            total += 1.0 / (1.0 + exp(-logits.data[i * KK_MAX_DUR + d]))
        var dur = total / speed
        var frames = _round_half_even(dur)
        if frames < 1:
            frames = 1
        p.duration.append(dur)
        p.pred_dur.append(frames)
        p.frames += frames
    return p^


# ── alignment, text encoder, pitch and energy ──────────────────────────────


struct KokoroAcoustic(Movable):
    """What `kk_acoustic` computed: everything the decoder reads. Sequences are
    channels-last (`[length, channels]`)."""

    var frames: Int
    var t_en: Tensor
    """`[T, 512]`: the text encoder."""
    var en: Tensor
    """`[F, 640]`: the duration encoder's output, one row per frame."""
    var asr: Tensor
    """`[F, 512]`: the text encoder's output, one row per frame."""
    var f0: Tensor
    """`[2F]`: predicted pitch."""
    var n: Tensor
    """`[2F]`: predicted energy."""

    def __init__(out self):
        self.frames = 0
        self.t_en = Tensor()
        self.en = Tensor()
        self.asr = Tensor()
        self.f0 = Tensor()
        self.n = Tensor()


def kk_text_encoder(mut w: KokoroWeights, ref ids: List[Int], mut out: Tensor) raises:
    """`TextEncoder`: embedding, three (Conv1d k 5, LayerNorm over channels,
    LeakyReLU 0.2), a bidirectional LSTM. `out[T, 512]`."""
    comptime H = KK_HIDDEN
    var t = len(ids)
    var emb = w.at("text_encoder.embedding.weight")
    var x = Tensor.alloc(t * H)
    for i in range(t):
        for d in range(H):
            x.data[i * H + d] = w.tensors[emb].data[ids[i] * H + d]
    var y = Tensor()
    for i in range(3):
        var cnn = String("text_encoder.cnn." + String(i))
        _ = kk_conv1d(w, cnn + ".0", x, t, y, 1, 2)
        kk_layer_norm_affine(w, cnn + ".1.gamma", cnn + ".1.beta", y, t, 1e-5)
        for k in range(t * H):
            x.data[k] = y.data[k] if y.data[k] >= 0.0 else y.data[k] * Float32(0.2)
    _ = kk_bilstm(w, "text_encoder.lstm", x, t, out)


def _predict_curve(
    mut w: KokoroWeights, which: String, shared: Tensor, frames: Int,
    style: Tensor, mut out: Tensor,
) raises:
    """The F0 or N head: three `AdainResBlk1d` (the middle one doubles the
    length) and a 1x1 convolution to one channel. `out[2 frames]`."""
    var a = Tensor()
    var b = Tensor()
    var l = kk_adain_resblk(w, "predictor." + which + ".0", shared, frames, style, a)
    l = kk_adain_resblk(w, "predictor." + which + ".1", a, l, style, b)
    l = kk_adain_resblk(w, "predictor." + which + ".2", b, l, style, a)
    _ = kk_conv1d(w, "predictor." + which + "_proj", a, l, out)


def kk_acoustic(
    mut w: KokoroWeights, ref ids: List[Int], ref ref_s: List[Float32],
    ref p: KokoroProsody,
) raises -> KokoroAcoustic:
    """From the durations: both encoders' outputs repeated per frame, and the
    pitch and energy curves (at twice the frame rate)."""
    comptime H = KK_HIDDEN
    comptime S = KK_STYLE
    comptime W = H + S
    var t = len(ids)
    var a = KokoroAcoustic()
    a.frames = p.frames
    kk_text_encoder(w, ids, a.t_en)

    # The alignment: frame f belongs to the token whose duration covers it.
    a.en = Tensor.alloc(p.frames * W)
    a.asr = Tensor.alloc(p.frames * H)
    var f = 0
    for i in range(t):
        for _ in range(p.pred_dur[i]):
            for d in range(W):
                a.en.data[f * W + d] = p.dur_enc.data[i * W + d]
            for d in range(H):
                a.asr.data[f * H + d] = a.t_en.data[i * H + d]
            f += 1

    var style = Tensor.alloc(S)
    for d in range(S):
        style.data[d] = ref_s[S + d]
    var shared = Tensor()
    _ = kk_bilstm(w, "predictor.shared", a.en, p.frames, shared)
    _predict_curve(w, "F0", shared, p.frames, style, a.f0)
    _predict_curve(w, "N", shared, p.frames, style, a.n)
    return a^
