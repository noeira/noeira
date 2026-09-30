# +--------------------------------------------------------------------------+ #
# | The JSON reader keeps non-ASCII text byte-exact
# +--------------------------------------------------------------------------+ #
"""Gate `parse_json`'s string decoding against BYTE LITERALS.

    pixi run mojo run -I . tests/io/test_json_read_utf8.mojo

⚠ THE REFERENCE IS A LIST OF BYTES, never a string that went through the
reader — `core/bytes.mojo` records how a per-byte `chr` hid for months
between two arms of a store-vs-store gate that shared it. The reader did
exactly that until a French Whisper transcript read back "Ã " for "à".

Covers: raw 2-, 3- and 4-byte UTF-8 passed through, the same characters as
`\\u` escapes (including a surrogate pair), a key with an accent, and the
writer -> reader round trip of that text.
"""

from noeira.io.json import JsonWriter, parse_json


def _b(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(s.byte_length()):
        out.append(s.as_bytes()[i])
    return out^


def _same(got: String, want: List[UInt8], what: String) raises:
    var g = _b(got)
    if g != want:
        var dump = String("")
        for i in range(len(g)):
            dump += hex(Int(g[i])) + " "
        raise Error("FAIL " + what + ": got bytes " + dump)


def main() raises:
    # à = C3 A0, 日 = E6 97 A5, ✓ = E2 9C 93, 😀 = F0 9F 98 80
    var want: List[UInt8] = [
        0xC3, 0xA0, 0x20, 0xE6, 0x97, 0xA5, 0x20, 0xE2, 0x9C, 0x93, 0x20,
        0xF0, 0x9F, 0x98, 0x80,
    ]
    # Raw UTF-8 in the document, keyed by an accented key "clé" (63 6C C3 A9).
    var raw: List[UInt8] = [0x7B, 0x22, 0x63, 0x6C, 0xC3, 0xA9, 0x22, 0x3A, 0x22]
    for i in range(len(want)):
        raw.append(want[i])
    raw.append(0x22)
    raw.append(0x7D)
    var d = parse_json(raw^)
    var key: List[UInt8] = [0x63, 0x6C, 0xC3, 0xA9]
    _same(d.key_at(d.root(), 0), key, "accented key")
    _same(d.string(d.at(d.root(), 0)), want, "raw UTF-8 value")

    # The same text as escapes.
    var esc = parse_json(_b('{"v":"\\u00e0 \\u65e5 \\u2713 \\ud83d\\ude00"}'))
    _same(esc.string(esc.field(esc.root(), "v")), want, "\\u escapes")

    # Writer -> reader.
    var text = d.string(d.at(d.root(), 0))
    var w = JsonWriter()
    w.begin_object()
    w.member("t", text)
    w.end_object()
    var back = parse_json(_b(w.done()))
    _same(back.string(back.field(back.root(), "t")), want, "writer round trip")
    print("=== json read utf8: 4 checks passed ===")
