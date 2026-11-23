# +--------------------------------------------------------------------------+ #
# | d1 — Python string semantics the prompt code relies on
# +--------------------------------------------------------------------------+ #
"""Code-point views of a `String` and Python's notion of whitespace, so d1's
character offsets (Python `str` indices) and its `.strip()` tests can be
transcribed as they are written (docs/NN_MODEL_PORTS_PLAN.md §12.9)."""


def cps_of(s: String) -> List[UInt32]:
    """`s` as code points (Python's `list(s)`)."""
    var out = List[UInt32](capacity=s.byte_length())
    for c in s.codepoints():
        out.append(c.to_u32())
    return out^


def str_of(cps: List[UInt32], a: Int, b: Int) -> String:
    """`s[a:b]` of the code points `cps`."""
    var out = String("")
    for i in range(max(a, 0), min(b, len(cps))):
        out += String(Codepoint(unsafe_unchecked_codepoint=cps[i]))
    return out^


def cp_len(s: String) -> Int:
    return s.count_codepoints()


def py_isspace(c: UInt32) -> Bool:
    """`str.isspace()` for one character (Unicode White_Space plus the ASCII
    separators 0x1C-0x1F, as CPython has it); U+200B is NOT whitespace."""
    if c == 0x20 or (c >= 0x09 and c <= 0x0D) or (c >= 0x1C and c <= 0x1F):
        return True
    if c < 0x85:
        return False
    return (c == 0x85 or c == 0xA0 or c == 0x1680 or (c >= 0x2000 and c <= 0x200A) or c == 0x2028 or c == 0x2029
            or c == 0x202F or c == 0x205F or c == 0x3000)


def blank(cps: List[UInt32], a: Int, b: Int) -> Bool:
    """`not s[a:b].strip()`: empty or whitespace only."""
    for i in range(max(a, 0), min(b, len(cps))):
        if not py_isspace(cps[i]):
            return False
    return True


def py_rstrip(s: String) -> String:
    """`s.rstrip()`."""
    var c = cps_of(s)
    var e = len(c)
    while e > 0 and py_isspace(c[e - 1]):
        e -= 1
    return str_of(c, 0, e)
