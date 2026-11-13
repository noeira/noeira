# +--------------------------------------------------------------------------+ #
# | A model's reply as text a voice can say
# +--------------------------------------------------------------------------+ #
"""`spoken_text(reply)`: what goes to a speech synthesiser and to the log.

A reply model writes for a screen: an emoji after a joke, `*emphasis*`, a
stray pictograph. A synthesiser reads those aloud or stumbles on them (Kokoro's
G2P has no word for U+1F604), so they are removed HERE, by rule, whatever the
prompt asked: 8 to 10 of 172 replies carried an emoji with LFM3-1.7B on the
room demo's prompts (docs/ROBOT_BRAIN_PLAN.md, B1).

Removed: emoji and pictographs (U+1F000 and above, U+2190-21FF, U+2600-27BF,
U+2B00-2BFF), the variation selector U+FE0F and the joiner U+200D that glue
them, `*` and backticks. Runs of spaces left behind collapse to one, a space
left before `.,!?;:` at the end goes, and the ends are trimmed. Letters,
accents, digits and punctuation are untouched.
"""


def spoken_text(reply: String) -> String:
    var b = List[UInt8]()
    for x in reply.as_bytes():
        b.append(x)
    var out = List[UInt8]()
    var n = len(b)
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c >= 0xF0 and i + 3 < n:
            # four bytes: U+10000 and above. U+1F000.. (F0 9F ..) are the pictographs.
            if c == 0xF0 and Int(b[i + 1]) >= 0x9F:
                i += 4
                continue
            for k in range(4):
                out.append(b[i + k])
            i += 4
            continue
        if c >= 0xE0 and i + 2 < n:
            var d = Int(b[i + 1])
            var e = Int(b[i + 2])
            var drop = False
            if c == 0xE2:
                # U+2190-21FF arrows, U+2600-27BF symbols and dingbats, U+2B00-2BFF, U+200D
                drop = (d == 0x86 or d == 0x87 or (d >= 0x98 and d <= 0x9E) or d == 0xAC or d == 0xAD
                        or (d == 0x80 and e == 0x8D))
            elif c == 0xEF:
                drop = d == 0xB8 and e == 0x8F  # U+FE0F
            if not drop:
                out.append(b[i])
                out.append(b[i + 1])
                out.append(b[i + 2])
            i += 3
            continue
        if c == 42 or c == 96:  # * and `
            i += 1
            continue
        if c == 32 and len(out) > 0 and Int(out[len(out) - 1]) == 32:
            i += 1
            continue
        out.append(b[i])
        i += 1
    # the ends
    while len(out) > 0 and (Int(out[len(out) - 1]) == 32 or Int(out[len(out) - 1]) == 10):
        _ = out.pop()
    var start = 0
    while start < len(out) and (Int(out[start]) == 32 or Int(out[start]) == 10):
        start += 1
    var t = List[UInt8]()
    for k in range(start, len(out)):
        t.append(out[k])
    return String(unsafe_from_utf8=Span(t))
