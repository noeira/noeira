# +--------------------------------------------------------------------------+ #
# | A small JSON reader
# +--------------------------------------------------------------------------+ #
"""Enough JSON to read a dataset's `meta/info.json` without Python.

    var doc = parse_json(text)
    var fps = doc.number(doc.field(doc.root(), "fps"))
    var feats = doc.field(doc.root(), "features")
    for i in range(doc.size(feats)):
        var key = doc.key_at(feats, i)
        ...

## Shape

The document is a FLAT node table, not a tree of boxed values: `kind[i]`,
plus an index into `nums` / `strs` / a `(start, count)` slice of `children`.
Mojo has no cheap recursive sum type, and a `Variant`-of-`List` tree costs a
heap allocation per node and an explicit copy at every access. Node ids are
plain `Int`s and `-1` means "absent", so a missing key is a value rather than
an exception — which is what makes `field(...)` chainable.

## Scope

Full JSON *values*: objects, arrays, strings with `\\u` escapes, numbers,
`true`/`false`/`null`. Numbers are all `Float64`; `info.json` carries integers
well inside 2^53, and `integer(...)` raises rather than truncating anything
that is not exactly integral, so a silent narrowing cannot happen.

Not supported: streaming, comments, trailing commas, duplicate-key policy
(the LAST duplicate wins, matching Python's `json`).
"""


comptime J_NULL = 0
comptime J_BOOL = 1
comptime J_NUMBER = 2
comptime J_STRING = 3
comptime J_ARRAY = 4
comptime J_OBJECT = 5


def kind_name(k: Int) -> String:
    if k == J_NULL: return String("null")
    if k == J_BOOL: return String("bool")
    if k == J_NUMBER: return String("number")
    if k == J_STRING: return String("string")
    if k == J_ARRAY: return String("array")
    if k == J_OBJECT: return String("object")
    return String("<none>")


struct JsonDoc(Movable):
    var kind: List[Int]
    var num: List[Float64]
    """Indexed by node id; also holds 0/1 for booleans."""
    var text: List[String]
    """Indexed by node id; the decoded value of a string node."""
    var child_start: List[Int]
    var child_count: List[Int]
    var children: List[Int]
    var child_key: List[String]
    """Parallel to `children`; empty for array elements."""

    def __init__(out self):
        self.kind = List[Int]()
        self.num = List[Float64]()
        self.text = List[String]()
        self.child_start = List[Int]()
        self.child_count = List[Int]()
        self.children = List[Int]()
        self.child_key = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind^
        self.num = move.num^
        self.text = move.text^
        self.child_start = move.child_start^
        self.child_count = move.child_count^
        self.children = move.children^
        self.child_key = move.child_key^

    def root(self) raises -> Int:
        if len(self.kind) == 0:
            raise Error("json: empty document")
        return 0

    def kind_of(self, node: Int) -> Int:
        if node < 0 or node >= len(self.kind):
            return -1
        return self.kind[node]

    def size(self, node: Int) -> Int:
        """Element / member count; 0 for a scalar or an absent node."""
        if node < 0 or node >= len(self.kind):
            return 0
        return self.child_count[node]

    def at(self, node: Int, i: Int) -> Int:
        if node < 0 or node >= len(self.kind):
            return -1
        if i < 0 or i >= self.child_count[node]:
            return -1
        return self.children[self.child_start[node] + i]

    def key_at(self, node: Int, i: Int) raises -> String:
        if node < 0 or node >= len(self.kind) or i < 0 or i >= self.child_count[node]:
            raise Error("json: key_at out of range")
        return String(self.child_key[self.child_start[node] + i])

    def field(self, node: Int, name: String) -> Int:
        """The value for `name`, or -1. The LAST duplicate wins."""
        if node < 0 or node >= len(self.kind) or self.kind[node] != J_OBJECT:
            return -1
        var found = -1
        for i in range(self.child_count[node]):
            if self.child_key[self.child_start[node] + i] == name:
                found = self.children[self.child_start[node] + i]
        return found

    def number(self, node: Int) raises -> Float64:
        if self.kind_of(node) != J_NUMBER:
            raise Error(
                "json: expected a number, found "
                + kind_name(self.kind_of(node))
            )
        return self.num[node]

    def integer(self, node: Int) raises -> Int:
        var v = self.number(node)
        var i = Int(v)
        if Float64(i) != v:
            raise Error("json: " + String(v) + " is not an integer")
        return i

    def string(self, node: Int) raises -> String:
        if self.kind_of(node) != J_STRING:
            raise Error(
                "json: expected a string, found "
                + kind_name(self.kind_of(node))
            )
        return String(self.text[node])

    def boolean(self, node: Int) raises -> Bool:
        if self.kind_of(node) != J_BOOL:
            raise Error(
                "json: expected a bool, found " + kind_name(self.kind_of(node))
            )
        return self.num[node] != 0.0

    def _new(mut self, k: Int) -> Int:
        self.kind.append(k)
        self.num.append(0.0)
        self.text.append(String(""))
        self.child_start.append(0)
        self.child_count.append(0)
        return len(self.kind) - 1


struct _Scanner(Movable):
    var b: List[UInt8]
    var pos: Int

    def __init__(out self, var b: List[UInt8]):
        self.b = b^
        self.pos = 0

    def __init__(out self, *, deinit move: Self):
        self.b = move.b^
        self.pos = move.pos

    def peek(self) -> Int:
        if self.pos >= len(self.b):
            return -1
        return Int(self.b[self.pos])

    def next(mut self) raises -> Int:
        if self.pos >= len(self.b):
            raise Error("json: unexpected end of input")
        var c = Int(self.b[self.pos])
        self.pos += 1
        return c

    def skip_ws(mut self):
        while self.pos < len(self.b):
            var c = Int(self.b[self.pos])
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                return

    def expect(mut self, c: Int) raises:
        var got = self.next()
        if got != c:
            raise Error(
                "json: expected '" + chr(c) + "' at byte " + String(self.pos - 1)
                + ", found '" + chr(got) + "'"
            )

    def literal(mut self, word: String) raises:
        for i in range(word.byte_length()):
            if self.next() != Int(word.as_bytes()[i]):
                raise Error(
                    "json: malformed literal near byte " + String(self.pos)
                )


def _hex4(mut s: _Scanner) raises -> Int:
    var v = 0
    for _ in range(4):
        var c = s.next()
        var d: Int
        if c >= 0x30 and c <= 0x39:
            d = c - 0x30
        elif c >= 0x61 and c <= 0x66:
            d = c - 0x61 + 10
        elif c >= 0x41 and c <= 0x46:
            d = c - 0x41 + 10
        else:
            raise Error("json: bad \\u escape")
        v = v * 16 + d
    return v


def _put_utf8(mut out: List[UInt8], cp: Int):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def _parse_string(mut s: _Scanner) raises -> String:
    """⚠ BYTES IN, BYTES OUT — never `chr(byte)`. A raw UTF-8 byte above 0x7F
    passed through `chr` becomes the CODEPOINT of that value and is re-encoded
    as two bytes, so "à" (C3 A0) read back as "Ã " — the defect
    `core/bytes.mojo` documents, and it lived here until a French transcript
    came back from Whisper through `noeira/ai/speech.mojo`. Raw bytes are
    copied verbatim; only a `\\u` escape is decoded, to UTF-8.

    ⚠ `\\u0000` TRUNCATES the value: the result is built through a
    NUL-terminated constructor. No payload this tree reads carries one.
    """
    s.expect(0x22)  # "
    var out = List[UInt8]()
    while True:
        var c = s.next()
        if c == 0x22:
            out.append(0)
            return String(unsafe_from_utf8_ptr=out.unsafe_ptr())
        if c != 0x5C:  # backslash
            out.append(UInt8(c))
            continue
        var e = s.next()
        if e == 0x22: out.append(0x22)
        elif e == 0x5C: out.append(0x5C)
        elif e == 0x2F: out.append(0x2F)
        elif e == 0x62: out.append(0x08)
        elif e == 0x66: out.append(0x0C)
        elif e == 0x6E: out.append(0x0A)
        elif e == 0x72: out.append(0x0D)
        elif e == 0x74: out.append(0x09)
        elif e == 0x75:
            var cp = _hex4(s)
            # Surrogate pair: a high surrogate must be followed by \uDC00-\uDFFF.
            if cp >= 0xD800 and cp <= 0xDBFF:
                if s.peek() == 0x5C:
                    _ = s.next()
                    s.expect(0x75)
                    var lo = _hex4(s)
                    if lo >= 0xDC00 and lo <= 0xDFFF:
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    else:
                        raise Error("json: unpaired surrogate escape")
                else:
                    raise Error("json: unpaired surrogate escape")
            _put_utf8(out, cp)
        else:
            raise Error("json: unknown escape '\\" + chr(e) + "'")


def _parse_number(mut s: _Scanner) raises -> Float64:
    var start = s.pos
    if s.peek() == 0x2D:
        _ = s.next()
    while True:
        var c = s.peek()
        if (
            (c >= 0x30 and c <= 0x39)
            or c == 0x2E  # .
            or c == 0x65 or c == 0x45  # e E
            or c == 0x2B or c == 0x2D  # + -
        ):
            _ = s.next()
        else:
            break
    if s.pos == start:
        raise Error("json: expected a number at byte " + String(start))
    # ⚠ `chr` PER BYTE IS SAFE **HERE ONLY**, and this note exists so nobody
    # "fixes" it again. A JSON NUMBER is `[-+0-9.eE]` by the grammar — every
    # byte is below 128, so the codepoint and the byte coincide. The same
    # spelling corrupts anything non-ASCII (see `core/bytes.mojo`), which is
    # why string literals in this file already use
    # `String(unsafe_from_utf8_ptr=)` and must keep doing so.
    var lit = String("")
    for i in range(start, s.pos):
        lit += chr(Int(s.b[i]))
    return atof(lit)


def _parse_value(mut s: _Scanner, mut doc: JsonDoc, depth: Int) raises -> Int:
    if depth > 256:
        raise Error("json: nesting deeper than 256 levels")
    s.skip_ws()
    var c = s.peek()
    if c < 0:
        raise Error("json: unexpected end of input")

    if c == 0x7B:  # {
        _ = s.next()
        var node = doc._new(J_OBJECT)
        var kids = List[Int]()
        var keys = List[String]()
        s.skip_ws()
        if s.peek() == 0x7D:
            _ = s.next()
        else:
            while True:
                s.skip_ws()
                keys.append(_parse_string(s))
                s.skip_ws()
                s.expect(0x3A)  # :
                kids.append(_parse_value(s, doc, depth + 1))
                s.skip_ws()
                var d = s.next()
                if d == 0x7D:
                    break
                if d != 0x2C:  # ,
                    raise Error(
                        "json: expected ',' or '}' at byte " + String(s.pos - 1)
                    )
        doc.child_start[node] = len(doc.children)
        doc.child_count[node] = len(kids)
        for i in range(len(kids)):
            doc.children.append(kids[i])
            doc.child_key.append(String(keys[i]))
        return node

    if c == 0x5B:  # [
        _ = s.next()
        var node = doc._new(J_ARRAY)
        var kids = List[Int]()
        s.skip_ws()
        if s.peek() == 0x5D:
            _ = s.next()
        else:
            while True:
                kids.append(_parse_value(s, doc, depth + 1))
                s.skip_ws()
                var d = s.next()
                if d == 0x5D:
                    break
                if d != 0x2C:
                    raise Error(
                        "json: expected ',' or ']' at byte " + String(s.pos - 1)
                    )
        doc.child_start[node] = len(doc.children)
        doc.child_count[node] = len(kids)
        for i in range(len(kids)):
            doc.children.append(kids[i])
            doc.child_key.append(String(""))
        return node

    if c == 0x22:  # "
        var v = _parse_string(s)
        var node = doc._new(J_STRING)
        doc.text[node] = v^
        return node

    if c == 0x74:  # true
        s.literal(String("true"))
        var node = doc._new(J_BOOL)
        doc.num[node] = 1.0
        return node

    if c == 0x66:  # false
        s.literal(String("false"))
        var node = doc._new(J_BOOL)
        doc.num[node] = 0.0
        return node

    if c == 0x6E:  # null
        s.literal(String("null"))
        return doc._new(J_NULL)

    var v = _parse_number(s)
    var node = doc._new(J_NUMBER)
    doc.num[node] = v
    return node


def parse_json(var text: List[UInt8]) raises -> JsonDoc:
    var doc = JsonDoc()
    var s = _Scanner(text^)
    _ = _parse_value(s, doc, 0)
    s.skip_ws()
    if s.pos != len(s.b):
        raise Error(
            "json: " + String(len(s.b) - s.pos) + " trailing bytes after the"
            " top-level value"
        )
    return doc^


def load_json(path: String) raises -> JsonDoc:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return parse_json(b^)


# ═══════════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════════
#
# The reader above replaced `json.loads`; this replaces `json.dumps`, which is
# the other half of what `core/logger.mojo` and `data/remote.mojo` needed
# Python for. A builder rather than a value tree, for the same reason the
# reader is a flat table: every payload here is written once, in order, and
# never inspected — a tree would cost an allocation per node to serialise it
# immediately.
#
#     var w = JsonWriter()
#     w.begin_object()
#     w.key("run_id"); w.string(run_id)
#     w.key("metrics"); w.begin_array()
#     ...
#     w.end_array()
#     w.end_object()
#     var payload = w.done()


struct JsonWriter(Movable):
    """Append-only JSON serialiser.

    ⚠ IT DOES NOT VALIDATE STRUCTURE. `end_object` after `begin_array` writes
    `]`-where-`}`-belongs and produces invalid JSON. The call sites here write
    fixed shapes a few lines long, so a validating writer would be paying for
    a mistake that a test on the emitted text catches once and for all.

    What it DOES enforce is the part that breaks silently: escaping, and the
    refusal to write a non-finite number.
    """

    var _out: String
    var _first: List[Bool]
    """One flag per open container: whether the next item is its first."""
    var _after_key: Bool
    """⚠ THE COMMA BELONGS TO THE MEMBER, NOT THE VALUE. `key()` writes the
    separator and then `"name":`; the value that follows must write none, or
    every object comes out as `{"a":,1}`. One flag is the whole rule."""

    def __init__(out self):
        self._out = String("")
        self._first = List[Bool]()
        self._after_key = False

    def __init__(out self, *, deinit move: Self):
        self._out = move._out^
        self._first = move._first^
        self._after_key = move._after_key

    def _sep(mut self):
        if self._after_key:
            self._after_key = False
            return
        if len(self._first) > 0:
            if self._first[len(self._first) - 1]:
                self._first[len(self._first) - 1] = False
            else:
                self._out += ","

    def begin_object(mut self):
        self._sep()
        self._out += "{"
        self._first.append(True)

    def end_object(mut self):
        self._out += "}"
        _ = self._first.pop()

    def begin_array(mut self):
        self._sep()
        self._out += "["
        self._first.append(True)

    def end_array(mut self):
        self._out += "]"
        _ = self._first.pop()

    def key(mut self, name: String) raises:
        self._sep()
        self._out += json_quote(name)
        self._out += ":"
        self._after_key = True

    def string(mut self, v: String) raises:
        self._sep()
        self._out += json_quote(v)

    def integer(mut self, v: Int):
        self._sep()
        self._out += String(v)

    def number(mut self, v: Float64) raises:
        """⚠ JSON HAS NO NaN AND NO Infinity. Python's `json.dumps` emits
        `NaN` / `Infinity` by default, which is not JSON and which a strict
        server rejects; raising here is the honest translation."""
        from std.math import isnan, isinf

        if isnan(v) or isinf(v):
            raise Error("json: " + String(v) + " has no JSON representation")
        self._sep()
        self._out += String(v)

    def boolean(mut self, v: Bool):
        self._sep()
        self._out += "true" if v else "false"

    def null(mut self):
        self._sep()
        self._out += "null"

    def raw(mut self, json_text: String) raises:
        """Splice an already-serialised JSON value in as the next item.

        For values the caller holds as text: a tool's JSON Schema, a model's
        tool-call arguments, an assistant turn replayed verbatim
        (`noeira/ai/`). ⚠ NOT VALIDATED beyond being non-empty — an invalid
        fragment makes the whole payload invalid. Parse it first
        (`parse_json`) when it came from a person rather than a server.
        """
        if json_text.byte_length() == 0:
            raise Error("json: raw() of an empty fragment")
        self._sep()
        self._out += json_text

    def member(mut self, name: String, v: String) raises:
        """`"name": "value"` — the shape most call sites actually write."""
        self.key(name)
        self.string(v)

    def member(mut self, name: String, v: Int) raises:
        self.key(name)
        self.integer(v)

    def member(mut self, name: String, v: Float64) raises:
        self.key(name)
        self.number(v)

    def done(mut self) raises -> String:
        if len(self._first) != 0:
            raise Error(
                "json: " + String(len(self._first)) + " container(s) still open"
            )
        return self._out.copy()


def json_quote(s: String) raises -> String:
    """`s` as a JSON string literal, quotes included.

    ⚠ ESCAPING IS WHERE A HAND-ROLLED WRITER FAILS, and it fails on data
    rather than on code: a run name with a quote in it, a Windows path with a
    backslash, a metric name carrying a tab. Every byte below 0x20 must become
    `\\uXXXX` — a raw one inside a string is invalid JSON, not merely ugly.

    ⚠ IT BUILDS BYTES, NOT A STRING, and that is not a micro-optimisation.
    An earlier version appended `s[byte = i : i + 1]` per byte, which ABORTS
    the process on any multi-byte character — Mojo's byte slicing asserts a
    codepoint boundary. A metric name with an accent in it would have taken
    the training run down. Bytes have no such rule; the result is reassembled
    once at the end.
    """
    var out = List[UInt8]()
    out.append(UInt8(0x22))  # opening quote
    var b = s.as_bytes()
    for i in range(s.byte_length()):
        var c = Int(b[i])
        if c == 0x22:  # "
            out.append(UInt8(0x5C))
            out.append(UInt8(0x22))
        elif c == 0x5C:  # backslash
            out.append(UInt8(0x5C))
            out.append(UInt8(0x5C))
        elif c == 0x0A:
            out.append(UInt8(0x5C))
            out.append(UInt8(0x6E))  # "n"
        elif c == 0x0D:
            out.append(UInt8(0x5C))
            out.append(UInt8(0x72))  # "r"
        elif c == 0x09:
            out.append(UInt8(0x5C))
            out.append(UInt8(0x74))  # "t"
        elif c == 0x08:
            out.append(UInt8(0x5C))
            out.append(UInt8(0x62))  # "b"
        elif c == 0x0C:
            out.append(UInt8(0x5C))
            out.append(UInt8(0x66))  # "f"
        elif c < 0x20:
            var hexd = String("0123456789abcdef").as_bytes()
            out.append(UInt8(0x5C))
            out.append(UInt8(0x75))  # "u"
            out.append(UInt8(0x30))  # "0"
            out.append(UInt8(0x30))  # "0"
            out.append(hexd[(c >> 4) & 0xF])
            out.append(hexd[c & 0xF])
        else:
            # UTF-8 lead and continuation bytes pass through untouched: JSON
            # is defined over Unicode text, and \u-escaping non-ASCII is
            # optional. Copying the bytes keeps the encoding intact without
            # this function having to decode anything.
            out.append(b[i])
    out.append(UInt8(0x22))  # closing quote
    out.append(UInt8(0))
    return String(unsafe_from_utf8_ptr=out.unsafe_ptr())


def _dump_node(ref doc: JsonDoc, node: Int, mut w: JsonWriter) raises:
    var k = doc.kind_of(node)
    if k == J_OBJECT:
        w.begin_object()
        for i in range(doc.size(node)):
            w.key(doc.key_at(node, i))
            _dump_node(doc, doc.at(node, i), w)
        w.end_object()
    elif k == J_ARRAY:
        w.begin_array()
        for i in range(doc.size(node)):
            _dump_node(doc, doc.at(node, i), w)
        w.end_array()
    elif k == J_STRING:
        w.string(doc.string(node))
    elif k == J_NUMBER:
        # An integral value goes back out as an integer: `{"n": 3}` must not
        # come back as `{"n": 3.0}`, which a strict schema (`"type":
        # "integer"`) rejects.
        var v = doc.number(node)
        var i = Int(v)
        if Float64(i) == v and v < 9007199254740992.0 and v > -9007199254740992.0:
            w.integer(i)
        else:
            w.number(v)
    elif k == J_BOOL:
        w.boolean(doc.boolean(node))
    elif k == J_NULL:
        w.null()
    else:
        raise Error("json: dump_json of an absent node")


def dump_json(ref doc: JsonDoc, node: Int) raises -> String:
    """The sub-tree at `node`, serialised back to compact JSON text.

    The reader keeps decoded values, not source spans, so this re-writes
    rather than slices: key ORDER is preserved, whitespace is not, and a
    number that was integral comes back without a fraction. Used to hand a
    model's tool-call arguments to the caller as one JSON string.
    """
    var w = JsonWriter()
    _dump_node(doc, node, w)
    return w.done()
