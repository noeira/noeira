# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — greedy token-and-duration decoding
# +--------------------------------------------------------------------------+ #
"""Encoder frames -> tokens, the way `ParakeetForTDT.generate` does it.

A transducer walks the encoder frames with a frame pointer. At each step the
joint network reads ONE encoder frame and the prediction network's output, and
predicts a token AND a duration:

    logits = joint(relu(encoder_proj[frame] + decoder_out))      8193 tokens + 5 durations
    token, duration = argmax of each part
    a blank with duration 0 advances by 1 (progress is forced)
    frame += duration; stop when frame reaches the clip's valid frames

The prediction network (embedding -> 2-layer LSTM -> linear) runs only when the
last token was NOT blank: after a blank its output and its LSTM state are
reused unchanged. It starts from the blank token with a zero state.

The trace (`tokens`, `durations`, the leading start token included) is what the
reference returns, so the gate compares it element for element.
"""

from std.math import exp, tanh

from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from .config import (
    PK_DEC, PK_DEC_LAYERS, PK_VOCAB, PK_BLANK, PK_JOINT, PK_MAX_SYMBOLS,
)
from .weights import ParakeetTDT


comptime PK_EOS: Int = 3
"""`<|endoftext|>`: `generation_config.json` names it `eos_token_id`, so the
reference's generation loop stops if the model ever emits it."""


struct PkTrace(Movable):
    var tokens: List[Int]
    var durations: List[Int]

    def __init__(out self):
        self.tokens = List[Int]()
        self.durations = List[Int]()

    def text_ids(self) -> List[Int]:
        """The emitted tokens without the start token."""
        var out = List[Int]()
        for i in range(1, len(self.tokens)):
            out.append(self.tokens[i])
        return out^


struct PkDecodeWork(Movable):
    """The prediction network's state and the decoder's scratch."""

    var x: Tensor
    var g_ih: Tensor
    var g_hh: Tensor
    var h: List[Tensor]
    var c: List[Tensor]
    var dec: Tensor
    var z: Tensor
    var logits: Tensor

    def __init__(out self):
        self.x = Tensor.alloc(PK_DEC)
        self.g_ih = Tensor.alloc(4 * PK_DEC)
        self.g_hh = Tensor.alloc(4 * PK_DEC)
        self.h = List[Tensor]()
        self.c = List[Tensor]()
        for _ in range(PK_DEC_LAYERS):
            self.h.append(Tensor.alloc(PK_DEC))
            self.c.append(Tensor.alloc(PK_DEC))
        self.dec = Tensor.alloc(PK_DEC)
        self.z = Tensor.alloc(PK_DEC)
        self.logits = Tensor.alloc(PK_JOINT)

    def reset(mut self):
        for l in range(PK_DEC_LAYERS):
            for i in range(PK_DEC):
                self.h[l].data[i] = 0.0
                self.c[l].data[i] = 0.0


def _sigmoid(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


def _predict(mut m: ParakeetTDT, mut w: PkDecodeWork, token: Int) raises:
    """Advance the prediction network by `token`: `w.dec` and the LSTM state."""
    comptime H = PK_DEC
    for i in range(H):
        w.x.data[i] = m.embedding.weight.val.data[token * H + i]
    for l in range(PK_DEC_LAYERS):
        m.lstm[l].ih.forward["cpu", 1](TensorRefs[1](w.x), w.g_ih)
        m.lstm[l].hh.forward["cpu", 1](TensorRefs[1](w.h[l]), w.g_hh)
        for i in range(H):
            var gi = _sigmoid(w.g_ih.data[i] + w.g_hh.data[i])
            var gf = _sigmoid(w.g_ih.data[H + i] + w.g_hh.data[H + i])
            var gg = tanh(w.g_ih.data[2 * H + i] + w.g_hh.data[2 * H + i])
            var go = _sigmoid(w.g_ih.data[3 * H + i] + w.g_hh.data[3 * H + i])
            var cn = gf * w.c[l].data[i] + gi * gg
            w.c[l].data[i] = cn
            w.h[l].data[i] = go * tanh(cn)
        for i in range(H):
            w.x.data[i] = w.h[l].data[i]
    m.dec_proj.forward["cpu", 1](TensorRefs[1](w.x), w.dec)


def _argmax(ref t: Tensor, start: Int, end: Int) -> Int:
    """Index (from `start`) of the first largest value in `[start, end)`."""
    var best = start
    for i in range(start + 1, end):
        if t.data[i] > t.data[best]:
            best = i
    return best - start


def pk_greedy_decode(
    mut m: ParakeetTDT, ref enc_proj: Tensor, valid: Int, t_max: Int,
    mut w: PkDecodeWork,
) raises -> PkTrace:
    """Greedy TDT decoding over the first `valid` rows of `enc_proj`
    (`[t_max, 640]`). `t_max` only bounds the number of steps, as in the
    reference (`max_symbols_per_step` x the padded length)."""
    comptime H = PK_DEC
    var trace = PkTrace()
    trace.tokens.append(PK_BLANK)
    trace.durations.append(0)
    if valid <= 0:
        return trace^
    w.reset()
    var last = PK_BLANK
    var started = False
    var frame = 0
    var max_len = PK_MAX_SYMBOLS * t_max
    while True:
        if not (started and last == PK_BLANK):
            _predict(m, w, last)
            started = True
        for i in range(H):
            var s = enc_proj.data[frame * H + i] + w.dec.data[i]
            w.z.data[i] = s if s > 0.0 else Float32(0.0)
        m.joint.forward["cpu", 1](TensorRefs[1](w.z), w.logits)
        var token = _argmax(w.logits, 0, PK_VOCAB)
        var duration = _argmax(w.logits, PK_VOCAB, PK_JOINT)
        if token == PK_BLANK and duration == 0:
            duration = 1
        trace.tokens.append(token)
        trace.durations.append(duration)
        frame += duration
        last = token
        if frame >= valid or token == PK_EOS or len(trace.tokens) >= max_len:
            break
    return trace^
