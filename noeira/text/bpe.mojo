# +--------------------------------------------------------------------------+ #
# | Byte-level BPE — Hugging Face `tokenizers` files, text -> ids -> text
# +--------------------------------------------------------------------------+ #
"""`ByteLevelBPE.load(tokenizer.json)` — the tokenizer of the LFM3 family (d1,
LFM3 text / VL / omni), and of any byte-level BPE whose pre-tokenizer is the
same split. No Python.

    var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
    var ids = tok.encode("<|startoftext|><|im_start|>user\\nHello")   # specials matched literally
    var text = tok.decode(ids)

What `encode` does, in `tokenizers`' order:

1. **Added tokens first.** Every `added_tokens` content (138 for LFM3: `<|im_start|>`,
   `<|pad|>`, ...) is found LITERALLY in the text (longest first) and becomes its id;
   the text between them goes on. (d1 inserts U+200B into a special string that
   appears in caller text, so it no longer matches — that is d1's business.)
2. **The split** (`Split`, `Isolated`): the regex
   `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
   hand-coded alternative by alternative in `_match` (leftmost-first, each with
   the backtracking the regex would do — read the comments there). Any other
   `pre_tokenizer` is refused at load.
3. **Byte level**: each piece's UTF-8 bytes become GPT-2's printable characters.
4. **BPE** with `ignore_merges`: a piece that is a whole vocabulary entry is that
   id; otherwise its bytes are merged, lowest rank first.

No normalizer (LFM3 has none; a file with one is refused).
"""

from noeira.io.json import load_json, J_ARRAY, J_NULL
from .unicode_classes import is_letter, is_number, is_space


comptime _KEY: Int = 1 << 18
"""Pair key = a * _KEY + b; ids must stay below it (LFM3: 125 033)."""


def _bytes_to_unicode() -> List[Int]:
    """GPT-2's byte -> printable code point."""
    var keep = List[Bool](length=256, fill=False)
    for b in range(33, 127):
        keep[b] = True
    for b in range(161, 173):
        keep[b] = True
    for b in range(174, 256):
        keep[b] = True
    var out = List[Int](length=256, fill=0)
    var n = 0
    for b in range(256):
        if keep[b]:
            out[b] = b
        else:
            out[b] = 256 + n
            n += 1
    return out^


def _utf8(cp: Int) -> String:
    var b = List[UInt8]()
    if cp < 0x80:
        b.append(UInt8(cp))
    elif cp < 0x800:
        b.append(UInt8(0xC0 | (cp >> 6)))
        b.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        b.append(UInt8(0xE0 | (cp >> 12)))
        b.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        b.append(UInt8(0x80 | (cp & 0x3F)))
    return String(unsafe_from_utf8=Span(b))


struct ByteLevelBPE(Movable):
    var vocab: Dict[String, Int]
    """Byte-level string -> id (the model's vocabulary, not the added tokens)."""
    var pieces: List[String]
    """id -> its string (byte-level for the vocabulary, plain for added tokens)."""
    var is_added: List[Bool]
    var merges: Dict[Int, Int]
    """(a, b) -> rank * _KEY + merged id."""
    var added: List[String]
    var added_ids: List[Int]
    """Added tokens, longest first."""
    var added_special: List[Bool]
    """Whether each added token is `special` (what a caller's text must not
    turn into: d1's `plain` breaks those, not `<think>`)."""
    var byte_char: List[String]
    """Byte -> its byte-level character (UTF-8)."""
    var char_byte: Dict[Int, Int]
    var ignore_merges: Bool
    var added_lt: Bool
    """Every added token starts with '<': only a '<' needs the literal search."""
    var _cache: Dict[String, List[Int]]

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.pieces = List[String]()
        self.is_added = List[Bool]()
        self.merges = Dict[Int, Int]()
        self.added = List[String]()
        self.added_ids = List[Int]()
        self.added_special = List[Bool]()
        self.byte_char = List[String]()
        self.char_byte = Dict[Int, Int]()
        self.ignore_merges = False
        self.added_lt = True
        self._cache = Dict[String, List[Int]]()

    @staticmethod
    def load(path: String) raises -> Self:
        var doc = load_json(path)
        var root = doc.root()
        var t = Self()
        var norm = doc.field(root, "normalizer")
        if norm >= 0 and doc.kind_of(norm) != J_NULL:
            raise Error("bpe: " + path + " has a normalizer; only none is implemented")
        var pre = doc.field(root, "pre_tokenizer")
        var split_ok = False
        if pre >= 0:
            var seq = doc.field(pre, "pretokenizers")
            if seq >= 0:
                for i in range(doc.size(seq)):
                    var p = doc.at(seq, i)
                    var kind = doc.string(doc.field(p, "type"))
                    if kind == "Split":
                        var pat = doc.string(doc.field(doc.field(p, "pattern"), "Regex"))
                        if pat != String(_SPLIT):
                            raise Error("bpe: a Split pattern other than LFM3's: " + pat)
                        split_ok = True
                    elif kind == "ByteLevel":
                        var reg = doc.field(p, "use_regex")
                        if reg >= 0 and doc.boolean(reg):
                            raise Error("bpe: ByteLevel with its own regex is not implemented")
                    else:
                        raise Error("bpe: pre-tokenizer '" + kind + "' is not implemented")
        if not split_ok:
            raise Error("bpe: " + path + " has no LFM3 Split pre-tokenizer")

        var m = doc.field(root, "model")
        if doc.string(doc.field(m, "type")) != "BPE":
            raise Error("bpe: model type is not BPE")
        var im = doc.field(m, "ignore_merges")
        t.ignore_merges = im >= 0 and doc.boolean(im)
        var v = doc.field(m, "vocab")
        var n_vocab = doc.size(v)
        var top = 0
        for i in range(n_vocab):
            top = max(top, doc.integer(doc.at(v, i)))
        var added = doc.field(root, "added_tokens")
        for i in range(doc.size(added)):
            top = max(top, doc.integer(doc.field(doc.at(added, i), "id")))
        if top >= _KEY:
            raise Error("bpe: ids up to " + String(top) + " do not fit the pair key")
        t.pieces = List[String](length=top + 1, fill=String(""))
        t.is_added = List[Bool](length=top + 1, fill=False)
        for i in range(n_vocab):
            var s = doc.key_at(v, i)
            var id = doc.integer(doc.at(v, i))
            t.vocab[s] = id
            t.pieces[id] = s
        var mg = doc.field(m, "merges")
        for r in range(doc.size(mg)):
            var e = doc.at(mg, r)
            var a: String
            var b: String
            if doc.kind_of(e) == J_ARRAY:
                a = doc.string(doc.at(e, 0))
                b = doc.string(doc.at(e, 1))
            else:
                var parts = doc.string(e).split(" ", 1)
                a = String(parts[0])
                b = String(parts[1])
            var ia = t.vocab.get(a)
            var ib = t.vocab.get(b)
            var ic = t.vocab.get(a + b)
            if not ia or not ib or not ic:
                raise Error("bpe: merge " + String(r) + " names a piece outside the vocabulary")
            var key = ia.value() * _KEY + ib.value()
            if key not in t.merges:
                t.merges[key] = r * _KEY + ic.value()
        # added tokens, longest first
        var order = List[Int]()
        for i in range(doc.size(added)):
            order.append(i)
        var lens = List[Int]()
        for i in range(doc.size(added)):
            lens.append(doc.string(doc.field(doc.at(added, i), "content")).byte_length())
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if lens[order[j]] > lens[order[i]]:
                    var tmp = order[i]
                    order[i] = order[j]
                    order[j] = tmp
        for i in order:
            var a = doc.at(added, i)
            var c = doc.string(doc.field(a, "content"))
            var id = doc.integer(doc.field(a, "id"))
            t.added.append(c)
            t.added_ids.append(id)
            var sp = doc.field(a, "special")
            t.added_special.append(sp >= 0 and doc.boolean(sp))
            t.pieces[id] = c
            t.is_added[id] = True
            if c.byte_length() == 0 or c.as_bytes()[0] != 60:
                t.added_lt = False
        var bu = _bytes_to_unicode()
        for b in range(256):
            t.byte_char.append(_utf8(bu[b]))
            t.char_byte[bu[b]] = b
        return t^

    def vocab_size(self) -> Int:
        return len(self.pieces)

    def token_id(self, piece: String) -> Int:
        """An added token's or a vocabulary entry's id; -1 for none."""
        for i in range(len(self.added)):
            if self.added[i] == piece:
                return self.added_ids[i]
        var v = self.vocab.get(piece)
        return v.value() if v else -1

    # ── encode ─────────────────────────────────────────────────────────

    def encode(mut self, text: String) raises -> List[Int]:
        var out = List[Int]()
        var b = text.as_bytes()
        var n = len(b)
        var start = 0
        var i = 0
        while i < n:
            var hit = -1
            if b[i] == 60 or not self.added_lt:  # every LFM3 added token starts with '<'
                hit = self._added_at(b, i)
            if hit >= 0:
                if i > start:
                    self._ordinary(String(text[byte=start:i]), out)
                out.append(self.added_ids[hit])
                i += self.added[hit].byte_length()
                start = i
                continue
            i += 1
        if start < n:
            self._ordinary(String(text[byte=start:n]), out)
        return out^

    def _added_at(self, b: Span[UInt8, _], i: Int) -> Int:
        for k in range(len(self.added)):
            var a = self.added[k].as_bytes()
            if i + len(a) > len(b):
                continue
            var ok = True
            for j in range(len(a)):
                if b[i + j] != a[j]:
                    ok = False
                    break
            if ok:
                return k
        return -1

    def _ordinary(mut self, s: String, mut out: List[Int]) raises:
        var cps = List[UInt32]()
        var offs = List[Int]()
        var o = 0
        for c in s.codepoints():
            cps.append(c.to_u32())
            offs.append(o)
            o += c.utf8_byte_length()
        offs.append(o)
        var i = 0
        while i < len(cps):
            var e = _match(cps, i)
            self._piece(String(s[byte=offs[i]:offs[e]]), out)
            i = e

    def _piece(mut self, piece: String, mut out: List[Int]) raises:
        var mapped = String("")
        for x in piece.as_bytes():
            mapped += self.byte_char[Int(x)]
        if self.ignore_merges:
            var v = self.vocab.get(mapped)
            if v:
                out.append(v.value())
                return
        var c = self._cache.get(mapped)
        if c:
            for id in c.value():
                out.append(id)
            return
        var sym = List[Int]()
        for x in piece.as_bytes():
            var v = self.vocab.get(self.byte_char[Int(x)])
            if not v:
                raise Error("bpe: byte " + String(Int(x)) + " has no vocabulary entry")
            sym.append(v.value())
        while len(sym) > 1:
            var best = -1
            var best_rank = Int.MAX
            var best_id = 0
            for j in range(len(sym) - 1):
                var mv = self.merges.get(sym[j] * _KEY + sym[j + 1])
                if mv:
                    var rank = mv.value() // _KEY
                    if rank < best_rank:
                        best_rank = rank
                        best = j
                        best_id = mv.value() % _KEY
            if best < 0:
                break
            sym[best] = best_id
            _ = sym.pop(best + 1)
        for id in sym:
            out.append(id)
        self._cache[mapped] = sym^

    # ── decode ─────────────────────────────────────────────────────────

    def decode(self, ids: List[Int], skip_added: Bool = False) -> String:
        """Ids back to text: added tokens as their content (or skipped), the
        rest from byte-level characters to bytes. Invalid UTF-8 (a cut
        multi-byte character) is replaced with U+FFFD."""
        var bytes = List[UInt8]()
        for id in ids:
            if id < 0 or id >= len(self.pieces):
                continue
            if self.is_added[id]:
                if not skip_added:
                    for x in self.pieces[id].as_bytes():
                        bytes.append(x)
                continue
            for c in self.pieces[id].codepoints():
                var v = self.char_byte.get(Int(c.to_u32()))
                if v:
                    bytes.append(UInt8(v.value()))
        return String(from_utf8_lossy=Span(bytes))


comptime _SPLIT = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
"""LFM3's Split regex, exactly as `tokenizer.json` holds it."""


@always_inline
def _nl(c: UInt32) -> Bool:
    return c == 10 or c == 13


@always_inline
def _other(c: UInt32) -> Bool:
    """`[^\\s\\p{L}\\p{N}]`."""
    return not is_space(c) and not is_letter(c) and not is_number(c)


@always_inline
def _lower(c: UInt32) -> UInt32:
    return c + 32 if c >= 65 and c <= 90 else c


def _match(cps: List[UInt32], i: Int) -> Int:
    """End (exclusive) of the split regex's match at `i`."""
    var n = len(cps)
    var c = cps[i]
    # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == 39 and i + 1 < n:
        var c1 = _lower(cps[i + 1])
        if c1 == 115 or c1 == 116:  # s t
            return i + 2
        if i + 2 < n:
            var c2 = _lower(cps[i + 2])
            if (c1 == 114 and c2 == 101) or (c1 == 118 and c2 == 101) or (c1 == 108 and c2 == 108):  # re ve ll
                return i + 3
        if c1 == 109 or c1 == 100:  # m d
            return i + 2
    # 2. [^\r\n\p{L}\p{N}]?\p{L}+ : one optional non-newline, non-letter, non-number, then letters
    var j = -1
    if not _nl(c) and not is_letter(c) and not is_number(c) and i + 1 < n and is_letter(cps[i + 1]):
        j = i + 1
    elif is_letter(c):
        j = i
    if j >= 0:
        var k = j
        while k < n and is_letter(cps[k]):
            k += 1
        return k
    # 3. \p{N}{1,3}
    if is_number(c):
        var k = i
        while k < n and k - i < 3 and is_number(cps[k]):
            k += 1
        return k
    # 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*' : an optional ASCII space, other characters, then newlines
    j = -1
    if c == 32 and i + 1 < n and _other(cps[i + 1]):
        j = i + 1
    elif _other(c):
        j = i
    if j >= 0:
        var k = j
        while k < n and _other(cps[k]):
            k += 1
        while k < n and _nl(cps[k]):
            k += 1
        return k
    # whitespace: the run [i, r)
    var r = i
    while r < n and is_space(cps[r]):
        r += 1
    if r > i:
        # 5. \s*[\r\n]+ : the greedy \s* backs off to the run's LAST newline; [\r\n]+ takes it
        var last = -1
        for p in range(i, r):
            if _nl(cps[p]):
                last = p
        if last >= 0:
            return last + 1
        # 6. \s+(?!\S) : the whole run at the end of the text, else all but its last character
        if r == n:
            return r
        if r - 1 > i:
            return r - 1
        # 7. \s+
        return r
    return i + 1  # unreachable: every character is a letter, number, other or space
