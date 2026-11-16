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


def spoken_sentence_end(text: String, start: Int) -> Int:
    """Where the first finished sentence of `text[start:]` ends (a byte index
    past its last mark), or -1 while none has finished. For a reply spoken
    sentence by sentence while it is still being written.

    A sentence is finished by `.`, `!`, `?` or `…` (with the marks, quotes and
    brackets right after them: `?!`, `...`, `.»`, `!)`) FOLLOWED BY A BLANK, or
    by a line break. ⚠ THE BLANK IS WHAT WAITS FOR THE NEXT TOKEN: `2.5`,
    `3,5 m.` at the very end and `Noé.` with nothing after it are not
    finished yet, as the next token can still be `5` or `com`. The end of the
    reply says the rest (`spoken_text` of what is left). French puts a blank
    BEFORE `!` and `?` (« Oui ! »): the blank after the mark still closes it.
    A piece of blanks only is not a sentence. `M. Dupont` cuts after `M.`
    (rare in a robot's reply; it is said in two pieces, not lost)."""
    var b = text.as_bytes()
    var n = len(b)
    var i = start
    var seen_word = False
    while i < n:
        var c = Int(b[i])
        if c == 10:
            if seen_word:
                return i + 1
            i += 1
            continue
        var mark = c == 46 or c == 33 or c == 63          # . ! ?
        var mlen = 1
        if c == 0xE2 and i + 2 < n and Int(b[i + 1]) == 0x80 and Int(b[i + 2]) == 0xA6:
            mark = True                                   # … U+2026
            mlen = 3
        if mark and seen_word:
            var j = i + mlen
            # more marks, closing quotes and brackets: `?!`, `...`, `."`, `.»` (C2 BB), `!)`
            while j < n:
                var d = Int(b[j])
                if d == 46 or d == 33 or d == 63 or d == 34 or d == 39 or d == 41 or d == 93:
                    j += 1
                elif d == 0xC2 and j + 1 < n and Int(b[j + 1]) == 0xBB:
                    j += 2
                elif d == 0xE2 and j + 2 < n and Int(b[j + 1]) == 0x80 and Int(b[j + 2]) == 0xA6:
                    j += 3
                else:
                    break
            if j < n and (Int(b[j]) == 32 or Int(b[j]) == 10 or Int(b[j]) == 9):
                return j
            if j >= n:
                return -1   # the mark is the last thing written: wait for what comes after it
            i = j
            continue
        if c != 32 and c != 9 and not mark:
            seen_word = True
        i += mlen if mark else 1
    return -1


def spoken_clause_end(text: String, start: Int, min_bytes: Int) -> Int:
    """Where the first clause of `text[start:]` ends (a byte index past its `,`
    `;` or `:`), or -1: a mark FOLLOWED BY A BLANK (so `4,5` and `10:30` are
    not cut) after at least `min_bytes` bytes of text (so « Oui, » is not a
    piece of its own: too short to cover the synthesis of the next piece).
    For the FIRST piece of a reply only (a one-sentence reply would otherwise
    wait for its whole sentence); a sentence's end is `spoken_sentence_end`."""
    var b = text.as_bytes()
    var n = len(b)
    var i = start
    while i < n and (Int(b[i]) == 32 or Int(b[i]) == 10 or Int(b[i]) == 9):
        i += 1
    var first = i
    while i < n:
        var c = Int(b[i])
        if (c == 44 or c == 59 or c == 58) and i - first + 1 >= min_bytes:
            if i + 1 < n and (Int(b[i + 1]) == 32 or Int(b[i + 1]) == 10 or Int(b[i + 1]) == 9):
                return i + 1
        i += 1
    return -1
