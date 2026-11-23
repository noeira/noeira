# +--------------------------------------------------------------------------+ #
# | d1 — a read's pointer: its start and end slots over the state's tokens
# +--------------------------------------------------------------------------+ #
"""d1's `pointer.py` readout, transcribed (docs/NN_MODEL_PORTS_PLAN.md §12.9,
T4). Parameter-free: a read's `start` and `end` slots are each scored against
the row's state positions,

    logits[t] = h[slot] . h[t] / sqrt(D)      t < width, -inf where not valid

(`position_logits`; `h` the backbone's output after its final norm, `width`
up to the last state position, position 0 — the BOS — standing for nothing).
`spans`: `found = 1 - softmax(start)[0]`; the span is the best
`logp_start[s] + logp_end[e]` over 1 <= s <= e < s + cap; nothing below
`found` 0.5. `read`: the token span as characters of the state text, trimmed
of whitespace, clipped to the one JSON string it falls in (then its path, and
the text unescaped).

On the host, in Float64 (d1 runs it in float32): a row has a few hundred
state positions and a read two slots.
"""

from std.math import exp, log, sqrt, expm1

from .positions import StatePositions, json_text
from .pystr import str_of, py_isspace


comptime SPAN_TEXT: Int = 512
comptime SPAN_NUMBER: Int = 32
"""A read's longest span in tokens (`SPAN_CAPS`): text 512, a number or a date 32."""
comptime NEG_INF: Float64 = -Float64.MAX


@fieldwise_init
struct D1Read(Copyable, Movable):
    """What a read's slots point at (`StatePositions.read` with `found`)."""

    var found: Float64
    """Raw P(something is there): 1 - softmax(start)[0]."""
    var has_text: Bool
    var text: String
    var start: Int
    var end: Int
    """The span in the state text, code points; -1 for none."""
    var has_path: Bool
    var path: String

    @staticmethod
    def nothing(found: Float64) -> D1Read:
        return D1Read(found, False, String(""), -1, -1, False, String(""))


def position_logits(hh: List[Float32], d: Int, slot: Int, sp: StatePositions) -> List[Float64]:
    """`pointer.position_logits` for one slot: `hh` the row's hidden states
    `[T, d]` from position 0 (at least up to `slot` and `sp.pointer_width()`)."""
    var width = sp.pointer_width()
    var scale = 1.0 / sqrt(Float64(d))
    var out = List[Float64](capacity=width)
    var q0 = slot * d
    for t in range(width):
        if not sp.valid(t):
            out.append(NEG_INF)
            continue
        var k0 = t * d
        var acc = 0.0
        for j in range(d):
            acc += Float64(hh[q0 + j]) * Float64(hh[k0 + j])
        out.append(acc * scale)
    return out^


def _log_softmax(x: List[Float64]) -> List[Float64]:
    var m = NEG_INF
    for v in x:
        m = max(m, v)
    var tot = 0.0
    for v in x:
        if v > NEG_INF:
            tot += exp(v - m)
    var lz = m + log(tot)
    var out = List[Float64](capacity=len(x))
    for v in x:
        out.append(v - lz if v > NEG_INF else NEG_INF)
    return out^


def spans(start_logits: List[Float64], end_logits: List[Float64], max_tokens: Int) -> Tuple[Int, Int, Float64]:
    """`pointer.spans` for one read: (first, last, found); (0, 0) when found is
    below 0.5 or there is only the BOS."""
    var ls = _log_softmax(start_logits)
    var le = _log_softmax(end_logits)
    var found = -expm1(ls[0]) if ls[0] > NEG_INF else 1.0
    var t = len(ls)
    if t == 1 or found < 0.5:
        return (0, 0, found)
    var cap = min(max_tokens, t - 1)
    var best = NEG_INF
    var first = 1
    var last = 1
    for s in range(1, t):
        if ls[s] == NEG_INF:
            continue
        var emax = NEG_INF
        var eat = s
        for e in range(s, min(s + cap, t)):
            if le[e] > emax:  # the first of equal maxima, as max_pool1d keeps it
                emax = le[e]
                eat = e
        var score = ls[s] + emax
        if score > best:  # argmax: the first of equal maxima
            best = score
            first = s
            last = eat
    return (first, last, found)


def read(sp: StatePositions, first: Int, last: Int, max_tokens: Int, found: Float64) raises -> D1Read:
    """`StatePositions.read`: the span's characters of the state text."""
    if first == 0 and last == 0:
        return D1Read.nothing(found)
    if not (0 < first and first <= last and last < len(sp.starts)) or last - first >= max_tokens:
        raise Error("d1: invalid pointer endpoints")
    if sp.starts[first] < 0 or sp.starts[last] < 0:
        raise Error("d1: pointer leaves the state text")
    var a = sp.starts[first]
    var b = sp.ends[last]
    ref text = sp.text_cps
    while a < b and py_isspace(text[a]):
        a += 1
    while b > a and py_isspace(text[b - 1]):
        b -= 1
    var held = -1
    var n_held = 0
    for k in range(len(sp.paths)):
        if sp.paths[k].start < b and a < sp.paths[k].end:
            held = k
            n_held += 1
    if n_held == 1:
        a = max(a, sp.paths[held].start)
        b = min(b, sp.paths[held].end)
        return D1Read(found, True, json_text(str_of(text, a, b)), a, b, True, sp.paths[held].path)
    return D1Read(found, True, str_of(text, a, b), a, b, False, String(""))
