# +--------------------------------------------------------------------------+ #
# | d1 — a row's state positions: which tokens a read may point at
# +--------------------------------------------------------------------------+ #
"""d1's `pointer.state_positions`, `prompt.plain_offsets` and `json_spans`,
transcribed (docs/NN_MODEL_PORTS_PLAN.md §12.9). A read's `start` and `end`
slots point at the row's tokens; these say which tokens are the caller's state
text and which characters each one covers.

Every offset here is a CODE POINT index (Python `str`), as d1's are: the spans
a read answers index the state's text as Python renders it.

- `plain_offsets`: `plain`'s escaping of special-token strings, with each
  escaped character's boundary mapped back to the original text.
- `state_positions`: per token of the prompt, `[start, end)` in the state text
  or none (template, picture and question tokens; position 0, the BOS, stands
  for "nothing"), and for a JSON state where each string value sits
  (`string_paths`, RFC 6901 paths).
- `json_text`: a slice of a rendered JSON string as the characters it encodes.
"""

from noeira.text.bpe import ByteLevelBPE

from .prompt import HEADER
from .pystr import cps_of, str_of, cp_len, blank, py_isspace


comptime BREAK_CP: UInt32 = 0x200B


def plain_offsets(ref tok: ByteLevelBPE, text: String) raises -> Tuple[List[UInt32], List[Int]]:
    """`prompt.plain_offsets`: `text` with every special token's string broken
    by U+200B after its first character (again until none is left), and every
    character boundary of the result mapped to the original text."""
    var cur = cps_of(text)
    var bounds = List[Int](capacity=len(cur) + 1)
    for i in range(len(cur) + 1):
        bounds.append(i)
    var specials = List[List[UInt32]]()
    for k in range(len(tok.added)):  # longest first, as `_specials_cached` sorts them
        if tok.added_special[k] and tok.added[k].count_codepoints() > 1 and BREAK_CP not in cps_of(tok.added[k]):
            specials.append(cps_of(tok.added[k]))
    while True:
        var starts = List[Int]()
        var lens = List[Int]()
        var i = 0
        while i < len(cur):
            var hit = -1
            for k in range(len(specials)):
                ref sp = specials[k]
                if i + len(sp) > len(cur):
                    continue
                var ok = True
                for j in range(len(sp)):
                    if cur[i + j] != sp[j]:
                        ok = False
                        break
                if ok:
                    hit = k
                    break
            if hit >= 0:
                starts.append(i)
                lens.append(len(specials[hit]))
                i += len(specials[hit])
            else:
                i += 1
        if len(starts) == 0:
            return (cur^, bounds^)
        for m in reversed(range(len(starts))):
            var at = starts[m] + 1
            bounds.insert(at, bounds[at])
        var out = List[UInt32](capacity=len(cur) + len(starts))
        var prev = 0
        for m in range(len(starts)):
            for j in range(prev, starts[m] + 1):
                out.append(cur[j])
            out.append(BREAK_CP)
            prev = starts[m] + 1
        for j in range(prev, len(cur)):
            out.append(cur[j])
        cur = out^


# ── JSON string spans ─────────────────────────────────────────────────────

@fieldwise_init
struct JsonSpan(Copyable, Movable):
    var start: Int
    var end: Int
    """The string's characters, quotes excluded."""
    var path: String
    """RFC 6901: `/key/0/inner`, `~` as `~0` and `/` as `~1`."""


def _skip(t: List[UInt32], at: Int) -> Int:
    var i = at
    while i < len(t) and py_isspace(t[i]):
        i += 1
    return i


def _hex4(t: List[UInt32], i: Int) raises -> Int:
    if i + 4 > len(t):
        raise Error("json: a cut \\u escape")
    var v = 0
    for k in range(4):
        var c = Int(t[i + k])
        var d: Int
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        else:
            raise Error("json: a bad \\u escape")
        v = v * 16 + d
    return v


def _string_at(t: List[UInt32], at: Int) raises -> Tuple[String, Int]:
    """A JSON string starting at its opening quote `at`: its value and the index
    after its closing quote (`json.decoder.scanstring`, strict)."""
    if at >= len(t) or t[at] != 34:
        raise Error("json: a string was expected")
    var out = List[UInt32]()
    var i = at + 1
    while True:
        if i >= len(t):
            raise Error("json: an unterminated string")
        var c = t[i]
        if c == 34:
            break
        if c < 0x20:
            raise Error("json: a control character in a string")
        if c != 92:
            out.append(c)
            i += 1
            continue
        if i + 1 >= len(t):
            raise Error("json: a cut escape")
        var e = t[i + 1]
        if e == 34 or e == 92 or e == 47:
            out.append(e)
        elif e == 98:
            out.append(8)
        elif e == 102:
            out.append(12)
        elif e == 110:
            out.append(10)
        elif e == 114:
            out.append(13)
        elif e == 116:
            out.append(9)
        elif e == 117:
            var u = _hex4(t, i + 2)
            if u >= 0xD800 and u <= 0xDBFF and i + 11 < len(t) and t[i + 6] == 92 and t[i + 7] == 117:
                var lo = _hex4(t, i + 8)
                if lo >= 0xDC00 and lo <= 0xDFFF:
                    out.append(UInt32(0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00)))
                    i += 12
                    continue
            if u >= 0xD800 and u <= 0xDFFF:
                raise Error("json: a lone surrogate")
            out.append(UInt32(u))
            i += 6
            continue
        else:
            raise Error("json: a bad escape")
        i += 2
    return (str_of(out, 0, len(out)), i + 1)


def _scalar_end(t: List[UInt32], at: Int) raises -> Int:
    """The end of a number, `true`, `false` or `null` starting at `at`."""
    var i = at
    if i < len(t) and t[i] == 45:  # -
        i += 1
    var word_start = i
    while i < len(t) and ((t[i] >= 97 and t[i] <= 122) or (t[i] >= 65 and t[i] <= 90)):
        i += 1
    if i > word_start:
        return i
    while i < len(t) and ((t[i] >= 48 and t[i] <= 57) or t[i] == 46 or t[i] == 101 or t[i] == 69 or t[i] == 43
                          or t[i] == 45):
        i += 1
    if i == at:
        raise Error("json: a value was expected")
    return i


def _value(t: List[UInt32], at0: Int, path: String, mut out: List[JsonSpan]) raises -> Int:
    var at = _skip(t, at0)
    if at >= len(t):
        raise Error("json: a value was expected")
    if t[at] == 123:  # {
        at = _skip(t, at + 1)
        while t[at] != 125:
            var kv = _string_at(t, at)
            var end = _skip(t, kv[1])
            if t[end] != 58:
                raise Error("invalid JSON object")
            var name = kv[0].replace("~", "~0").replace("/", "~1")
            at = _skip(t, _value(t, end + 1, path + "/" + name, out))
            if t[at] == 44:
                at = _skip(t, at + 1)
            elif t[at] != 125:
                raise Error("invalid JSON object separator")
        return at + 1
    if t[at] == 91:  # [
        at = _skip(t, at + 1)
        var index = 0
        while t[at] != 93:
            at = _skip(t, _value(t, at, path + "/" + String(index), out))
            index += 1
            if t[at] == 44:
                at = _skip(t, at + 1)
            elif t[at] != 93:
                raise Error("invalid JSON array separator")
        return at + 1
    if t[at] == 34:
        var sv = _string_at(t, at)
        out.append(JsonSpan(at + 1, sv[1] - 1, path))
        return sv[1]
    return _scalar_end(t, at)


def string_paths(text: List[UInt32]) raises -> List[JsonSpan]:
    """`json_spans.string_paths`: every string value's span in rendered JSON,
    quotes excluded, with its path."""
    var out = List[JsonSpan]()
    var end = _value(text, 0, String(""), out)
    if _skip(text, end) != len(text):
        raise Error("trailing JSON text")
    return out^


def json_text(escaped: String) -> String:
    """`json_spans.json_text`: a slice of a rendered JSON string as the
    characters it encodes; unchanged when it cuts an escape."""
    var t = List[UInt32]()
    t.append(34)
    for c in cps_of(escaped):
        t.append(c)
    t.append(34)
    try:
        var sv = _string_at(t, 0)
        if sv[1] != len(t):
            return escaped
        return sv[0]
    except:
        return escaped


# ── state positions ───────────────────────────────────────────────────────

struct StatePositions(Movable):
    """`pointer.StatePositions`: per prompt token, its `[start, end)` in the
    state text (`start` -1: not a text position)."""

    var text: String
    var text_cps: List[UInt32]
    var starts: List[Int]
    var ends: List[Int]
    var paths: List[JsonSpan]

    def __init__(out self, var text: String, var starts: List[Int], var ends: List[Int], var paths: List[JsonSpan]):
        self.text_cps = cps_of(text)
        self.text = text^
        self.starts = starts^
        self.ends = ends^
        self.paths = paths^

    def valid(self, i: Int) -> Bool:
        return i == 0 or self.starts[i] >= 0

    def pointer_width(self) -> Int:
        """`len(pointer_mask)`: up to and including the last valid position."""
        var last = 0
        for i in range(len(self.starts)):
            if self.valid(i):
                last = i
        return last + 1


def state_positions(
    ref tok: ByteLevelBPE, state_text: String, state_is_json: Bool, prefix: String, full_text: String,
    starts: List[Int], ends: List[Int],
) raises -> StatePositions:
    """`pointer.state_positions`: `state_text` is the state as the prompt renders
    it without its blank line (`prompt.state_text`: the text itself, or the
    indented JSON); `full_text` the row's tokenized text, starting with
    `prefix`; `starts` / `ends` its tokens' offsets (`encode_offsets`)."""
    var full = cps_of(full_text)
    var pref = cps_of(prefix)
    if len(pref) > len(full):
        raise Error("d1: the tokenized prompt must start with the state prefix")
    for i in range(len(pref)):
        if pref[i] != full[i]:
            raise Error("d1: the tokenized prompt must start with the state prefix")
    var esc = plain_offsets(tok, state_text + "\n\n")
    ref escaped = esc[0]
    ref bounds = esc[1]
    var tail = escaped.copy()
    for c in cps_of(String("\n") + HEADER):
        tail.append(c)
    if len(tail) > len(pref):
        raise Error("d1: state rendering differs from the prompt prefix")
    var start = len(pref) - len(tail)
    for i in range(len(tail)):
        if pref[start + i] != tail[i]:
            raise Error("d1: state rendering differs from the prompt prefix")
    var text = cps_of(state_text)
    var tl = len(text)
    var br = 0  # bisect_right(bounds, len(text))
    var lo_b = 0
    var hi_b = len(bounds)
    while lo_b < hi_b:
        var mid = (lo_b + hi_b) // 2
        if tl < bounds[mid]:
            hi_b = mid
        else:
            lo_b = mid + 1
    br = lo_b
    var stop = start + br - 1
    var s_out = List[Int](capacity=len(starts))
    var e_out = List[Int](capacity=len(starts))
    for k in range(len(starts)):
        var a = starts[k]
        var b = ends[k]
        var sp_a = -1
        var sp_b = -1
        var lo = max(a, start)
        var hi = min(b, stop)
        if lo < hi and blank(full, a, lo) and blank(full, hi, b):
            var left = bounds[lo - start]
            var right = bounds[hi - start]
            if left < right and right <= tl and not blank(text, left, right):
                sp_a = left
                sp_b = right
        s_out.append(sp_a)
        e_out.append(sp_b)
    if len(s_out) == 0 or s_out[0] >= 0:
        raise Error("d1: position 0 must be a template token for the nothing readout")
    var paths = List[JsonSpan]()
    if state_is_json:
        paths = string_paths(text)
    return StatePositions(state_text, s_out^, e_out^, paths^)
