# +--------------------------------------------------------------------------+ #
# | CLIP's BPE — the text side of SAM 3 (and of CLIP itself), no Python
# +--------------------------------------------------------------------------+ #
"""`ClipBPE.load(tokenizer.json)`: CLIP's tokenizer as Hugging Face `tokenizers`
runs it (SAM 3's `facebook/sam3`), checked against that file's own
configuration at load — anything else is refused.

    var tok = ClipBPE.load(dir + "/tokenizer.json")
    var t = tok.encode("a red chair", 32)      # t.ids [32]: <|startoftext|> ... <|endoftext|>, padded; t.mask

What `encode` does, in `tokenizers`' order:

1. **Normalise**: NFC (taken as given: typed text is composed; a decomposed
   accent is NOT recomposed here), every run of whitespace to one space,
   lowercase (ASCII and Latin-1; other scripts pass unchanged).
2. **Split** (`invert`: the matches are the pieces, what lies between them is
   dropped): `<|startoftext|>`, `<|endoftext|>`, the contractions `'s 't 're 've
   'm 'll 'd`, a run of letters, ONE digit, a run of anything that is neither
   space, letter nor digit — leftmost-first, in that order.
3. **Byte level**: each piece's UTF-8 bytes as GPT-2's printable characters.
   (`ByteLevel`'s own regex runs again on each piece and leaves CLIP's pieces
   whole; the gate checks that against `tokenizers`.)
4. **BPE** with CLIP's `</w>`: the piece's last symbol carries the end-of-word
   suffix, then merges, lowest rank first.
5. `<|startoftext|>` + ids + `<|endoftext|>`, padded with `<|endoftext|>` to
   `max_len` (SAM 3: 32); `mask` marks the real ones.
"""

from noeira.io.json import load_json, J_ARRAY
from .bpe import _bytes_to_unicode, _utf8
from .unicode_classes import is_letter, is_number, is_space


comptime _KEY: Int = 1 << 17
comptime CLIP_SPLIT = "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
"""CLIP's Split regex, as `tokenizer.json` holds it."""
comptime _SOT = "<|startoftext|>"
comptime _EOT = "<|endoftext|>"
comptime _EOW = "</w>"


@fieldwise_init
struct ClipTokens(Copyable, Movable):
    var ids: List[Int]
    var mask: List[Int]
    var length: Int
    """Real tokens (`<|startoftext|>` and `<|endoftext|>` included)."""


struct ClipBPE(Movable):
    var vocab: Dict[String, Int]
    var merges: Dict[Int, Int]
    """(a, b) -> rank * _KEY + merged id."""
    var byte_char: List[String]
    var sot: Int
    var eot: Int
    var _cache: Dict[String, List[Int]]

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.merges = Dict[Int, Int]()
        self.byte_char = List[String]()
        self.sot = -1
        self.eot = -1
        self._cache = Dict[String, List[Int]]()

    @staticmethod
    def load(path: String) raises -> Self:
        var doc = load_json(path)
        var root = doc.root()
        # the normalizer: NFC, \s+ -> " ", Lowercase — nothing else
        var norm = doc.field(root, "normalizer")
        var kinds = String("")
        var ns = doc.field(norm, "normalizers")
        for i in range(doc.size(ns)):
            kinds += doc.string(doc.field(doc.at(ns, i), "type")) + ","
        if kinds != "NFC,Replace,Lowercase,":
            raise Error("clip_bpe: normalizers " + kinds + " — only CLIP's NFC,Replace,Lowercase")
        var pre = doc.field(root, "pre_tokenizer")
        var seq = doc.field(pre, "pretokenizers")
        if doc.size(seq) != 2:
            raise Error("clip_bpe: not CLIP's two pre-tokenizers")
        var sp = doc.at(seq, 0)
        if doc.string(doc.field(sp, "type")) != "Split" or doc.string(doc.field(doc.field(sp, "pattern"), "Regex")) != String(CLIP_SPLIT):
            raise Error("clip_bpe: a Split other than CLIP's")
        if not doc.boolean(doc.field(sp, "invert")):
            raise Error("clip_bpe: CLIP's Split keeps the matches (invert)")
        if doc.string(doc.field(doc.at(seq, 1), "type")) != "ByteLevel":
            raise Error("clip_bpe: the second pre-tokenizer is not ByteLevel")
        var m = doc.field(root, "model")
        if doc.string(doc.field(m, "type")) != "BPE" or doc.string(doc.field(m, "end_of_word_suffix")) != _EOW:
            raise Error("clip_bpe: not a BPE with the </w> suffix")
        var t = Self()
        var v = doc.field(m, "vocab")
        for i in range(doc.size(v)):
            var id = doc.integer(doc.at(v, i))
            if id >= _KEY:
                raise Error("clip_bpe: id " + String(id) + " does not fit the pair key")
            t.vocab[doc.key_at(v, i)] = id
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
                raise Error("clip_bpe: merge " + String(r) + " names a piece outside the vocabulary")
            var key = ia.value() * _KEY + ib.value()
            if key not in t.merges:
                t.merges[key] = r * _KEY + ic.value()
        var added = doc.field(root, "added_tokens")
        for i in range(doc.size(added)):
            var a = doc.at(added, i)
            var c = doc.string(doc.field(a, "content"))
            if c == _SOT:
                t.sot = doc.integer(doc.field(a, "id"))
            elif c == _EOT:
                t.eot = doc.integer(doc.field(a, "id"))
        if t.sot < 0 or t.eot < 0:
            raise Error("clip_bpe: no <|startoftext|> / <|endoftext|>")
        var bu = _bytes_to_unicode()
        for b in range(256):
            t.byte_char.append(_utf8(bu[b]))
        return t^

    def encode(mut self, text: String, max_len: Int = 32) raises -> ClipTokens:
        """`<|startoftext|>` + the text's ids + `<|endoftext|>`, padded to
        `max_len` with `<|endoftext|>` (raises if longer)."""
        var cps = _normalise(text)
        var ids = List[Int]()
        ids.append(self.sot)
        var i = 0
        while i < len(cps):
            var e = _clip_match(cps, i)
            if e < 0:
                i += 1  # whitespace: between the pieces, dropped
                continue
            var piece = String("")
            for k in range(i, e):
                piece += _utf8_any(cps[k])
            self._piece(piece, ids)
            i = e
        ids.append(self.eot)
        var n = len(ids)
        if n > max_len:
            raise Error("clip_bpe: " + String(n) + " tokens for at most " + String(max_len))
        var mask = List[Int](length=max_len, fill=0)
        for k in range(n):
            mask[k] = 1
        while len(ids) < max_len:
            ids.append(self.eot)
        return ClipTokens(ids^, mask^, n)

    def _piece(mut self, piece: String, mut out: List[Int]) raises:
        if piece == _SOT:
            out.append(self.sot)
            return
        if piece == _EOT:
            out.append(self.eot)
            return
        var c = self._cache.get(piece)
        if c:
            for id in c.value():
                out.append(id)
            return
        var chars = List[String]()
        for x in piece.as_bytes():
            chars.append(self.byte_char[Int(x)])
        chars[len(chars) - 1] = chars[len(chars) - 1] + _EOW
        var sym = List[Int]()
        for s in chars:
            var v = self.vocab.get(s)
            if not v:
                raise Error("clip_bpe: '" + s + "' has no vocabulary entry")
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
        self._cache[piece] = sym^


def _utf8_any(cp: UInt32) -> String:
    """A code point as UTF-8, 1 to 4 bytes (`bpe._utf8` stops at 3: it only
    writes GPT-2's byte characters, all below U+0144; an emoji is 4)."""
    var c = Int(cp)
    var b = List[UInt8]()
    if c < 0x80:
        b.append(UInt8(c))
    elif c < 0x800:
        b.append(UInt8(0xC0 | (c >> 6)))
        b.append(UInt8(0x80 | (c & 0x3F)))
    elif c < 0x10000:
        b.append(UInt8(0xE0 | (c >> 12)))
        b.append(UInt8(0x80 | ((c >> 6) & 0x3F)))
        b.append(UInt8(0x80 | (c & 0x3F)))
    else:
        b.append(UInt8(0xF0 | (c >> 18)))
        b.append(UInt8(0x80 | ((c >> 12) & 0x3F)))
        b.append(UInt8(0x80 | ((c >> 6) & 0x3F)))
        b.append(UInt8(0x80 | (c & 0x3F)))
    return String(unsafe_from_utf8=Span(b))


def _lower(c: UInt32) -> UInt32:
    """ASCII and Latin-1 upper case to lower (U+00D7 x is not a letter)."""
    if (c >= 65 and c <= 90) or (c >= 0xC0 and c <= 0xDE and c != 0xD7):
        return c + 32
    return c


def _normalise(text: String) -> List[UInt32]:
    """Whitespace runs to one space, lowercase (NFC taken as given)."""
    var out = List[UInt32]()
    var in_space = False
    for ch in text.codepoints():
        var c = ch.to_u32()
        if is_space(c):
            if not in_space:
                out.append(32)
            in_space = True
            continue
        in_space = False
        out.append(_lower(c))
    return out^


def _starts(cps: List[UInt32], i: Int, s: String) -> Bool:
    var k = i
    for b in s.as_bytes():
        if k >= len(cps) or cps[k] != UInt32(b):
            return False
        k += 1
    return True


def _clip_match(cps: List[UInt32], i: Int) -> Int:
    """End of CLIP's split match at `i`, or -1 when none starts there (a space).
    Leftmost-first over the alternatives, in the regex's order."""
    if _starts(cps, i, _SOT):
        return i + 15
    if _starts(cps, i, _EOT):
        return i + 13
    if cps[i] == 39 and i + 1 < len(cps):  # '
        var n = cps[i + 1]
        if n == 115 or n == 116 or n == 109 or n == 100:  # 's 't 'm 'd
            return i + 2
        if i + 2 < len(cps):
            var n2 = cps[i + 2]
            if (n == 114 and n2 == 101) or (n == 118 and n2 == 101) or (n == 108 and n2 == 108):  # 're 've 'll
                return i + 3
    var c = cps[i]
    if is_letter(c):
        var e = i + 1
        while e < len(cps) and is_letter(cps[e]):
            e += 1
        return e
    if is_number(c):
        return i + 1
    if not is_space(c):
        var e = i + 1
        while e < len(cps) and not is_space(cps[e]) and not is_letter(cps[e]) and not is_number(cps[e]):
            e += 1
        return e
    return -1
