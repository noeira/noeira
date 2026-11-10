# +--------------------------------------------------------------------------+ #
# | espeak-ng from Mojo: text to IPA phonemes
# +--------------------------------------------------------------------------+ #
"""libespeak-ng through FFI, and `phonemizer`'s espeak backend ported on top.

    var es = EspeakNG.open()                        # ~/.cache/noeira/espeak-ng
    var ps = es.phonemize("Bonjour, ça va ?", "fr-fr")
    # 'bɔ̃ʒˈuʁ, sa vˈa ? '   (as phonemizer.phonemize(..., preserve_punctuation=True, with_stress=True))

Why both layers: espeak-ng's C call (`espeak_TextToPhonemes`) is what turns
words into phonemes, but it DROPS punctuation and returns a sentence in pieces.
What text-to-speech models were trained on is `phonemizer`'s post-processing of
it — punctuation cut out of the text before espeak sees it and put back after,
phone separators removed, stress marks kept, multi-character phonemes tied or
not, language-switch flags kept or removed. This file reproduces phonemizer
3.4.0 (the deployment's version, on the Mac and the Orin) step for step:

    phonemize(text)           phonemizer.phonemize: split lines, drop blank ones,
                              backend, join with "\\n"
    backend_phonemize(lines)  EspeakBackend.phonemize: cut punctuation
                              (`_preserve`), espeak per chunk (`_line`), restore
                              (`_restore`)

Gate: `tests/audio/test_espeak_reference.mojo`, 28 English and French
sentences against phonemizer itself, exact string equality.

## The library

`EspeakNG.open()` loads, in order: the paths given, `$NOEIRA_ESPEAK_LIB` /
`$NOEIRA_ESPEAK_DATA`, then `~/.cache/noeira/espeak-ng/` — what
`tools/kokoro/fetch_espeak_ng.py` installs: espeak-ng 1.52.0, byte-identical to
the deployment's. A system espeak-ng is NOT used by default: dictionaries
change between releases (the Orin's is 1.49).

⚠ espeak-ng keeps PROCESS-GLOBAL state (the voice, its buffers). One
`EspeakNG` per process, used from one thread; switching language is
`set_voice`. espeak-ng is GPL-3.0: it is loaded at run time, never linked
or vendored.
"""

from std.ffi import OwnedDLHandle, c_char
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget

from noeira.io.hf import noeira_cache


comptime ESPEAK_LIB_NAME = "libespeak-ng.dylib" if CompilationTarget.is_macos() else "libespeak-ng.so"
comptime _AUDIO_OUTPUT_SYNCHRONOUS: Int32 = 0x02
comptime _CHARS_UTF8: Int32 = 1
comptime _PHONEMES_IPA: Int = 0x02
comptime _PHONEMES_TIE: Int = 0x80
comptime COMBINING_TIE = "͡"
"""What espeak puts inside a multi-character phoneme in tie mode."""

comptime _MARKS_OTHER = ";:!?¡¿—…\"«»“”(){}[]"
"""phonemizer's default punctuation marks, minus the decimal separators."""
comptime _MARKS_DECIMAL = ",."
"""Marks that are punctuation unless they sit BETWEEN two digits (19,5)."""

comptime _B = 0
comptime _I = 1
comptime _E = 2
comptime _A = 3


# ── Python's string semantics, where they differ from Mojo's ──────────────


def py_isspace(c: Int) -> Bool:
    """`str.isspace()` for one code point: what Python's `\\s` and `strip()` use."""
    if c >= 9 and c <= 13:
        return True
    if c >= 28 and c <= 32:
        return True
    if c == 133 or c == 160 or c == 5760:
        return True
    if c >= 8192 and c <= 8202:
        return True
    return c == 8232 or c == 8233 or c == 8239 or c == 8287 or c == 12288


struct _Chars(Movable):
    """A string as code points and their UTF-8 slices."""

    var cps: List[Int]
    var strs: List[String]

    def __init__(out self, s: String):
        self.cps = List[Int]()
        self.strs = List[String]()
        for c in s.codepoints():
            self.cps.append(Int(c))
        for sl in s.codepoint_slices():
            self.strs.append(String(sl))

    def join(self, a: Int, b: Int) -> String:
        var out = String("")
        for i in range(a, b):
            out += self.strs[i]
        return out^


def py_strip(s: String) -> String:
    """`str.strip()`: Unicode whitespace off both ends."""
    var c = _Chars(s)
    var a = 0
    var b = len(c.cps)
    while a < b and py_isspace(c.cps[a]):
        a += 1
    while b > a and py_isspace(c.cps[b - 1]):
        b -= 1
    return c.join(a, b)


def _strip_char(s: String, ch: String) -> String:
    """`str.strip(ch)` for a one-byte `ch`."""
    var a = 0
    var b = s.byte_length()
    while a < b and String(s[byte=a : a + 1]) == ch:
        a += 1
    while b > a and String(s[byte=b - 1 : b]) == ch:
        b -= 1
    return String(s[byte=a:b])


def _drop_last_byte(s: String) -> String:
    return String(s[byte=0 : s.byte_length() - 1])


# ── phonemizer's Punctuation (preserve / restore) ──────────────────────────


@fieldwise_init
struct _Mark(Copyable, Movable):
    var index: Int
    """The input LINE the mark came from."""
    var mark: String
    var position: Int
    """_B, _I, _E or _A."""


def _is_mark(c: _Chars, i: Int) -> Bool:
    """Whether code point `i` is a punctuation mark for phonemizer's regex
    `(\\s*(?:[others]|(?<![0-9])[,.]|[,.](?![0-9]))+\\s*)+`."""
    var s = c.strs[i]
    if String(_MARKS_OTHER).find(s) >= 0:
        return True
    if s == "," or s == ".":
        var digit_before = i > 0 and c.cps[i - 1] >= 48 and c.cps[i - 1] <= 57
        var digit_after = i + 1 < len(c.cps) and c.cps[i + 1] >= 48 and c.cps[i + 1] <= 57
        return not (digit_before and digit_after)
    return False


def _mark_runs(c: _Chars) -> List[Tuple[Int, Int]]:
    """What `re.finditer` matches, left to right, as code-point ranges: every
    maximal run of whitespace and marks that holds at least one mark."""
    var out = List[Tuple[Int, Int]]()
    var n = len(c.cps)
    var i = 0
    while i < n:
        if not (py_isspace(c.cps[i]) or _is_mark(c, i)):
            i += 1
            continue
        var j = i
        var has_mark = False
        while j < n and (py_isspace(c.cps[j]) or _is_mark(c, j)):
            if _is_mark(c, j):
                has_mark = True
            j += 1
        if has_mark:
            out.append((i, j))
        i = j
    return out^


def _find_marks(line: String) -> List[String]:
    """The texts of `_mark_runs`."""
    var c = _Chars(line)
    var runs = _mark_runs(c)
    var out = List[String]()
    for k in range(len(runs)):
        out.append(c.join(runs[k][0], runs[k][1]))
    return out^


def _remove_marks(line: String) -> String:
    """`Punctuation.remove`: every run replaced by one space, then stripped."""
    var c = _Chars(line)
    var runs = _mark_runs(c)
    var out = String("")
    var at = 0
    for k in range(len(runs)):
        out += c.join(at, runs[k][0]) + " "
        at = runs[k][1]
    out += c.join(at, len(c.cps))
    return py_strip(out)


def _preserve(lines: List[String], mut chunks: List[String], mut marks: List[_Mark]):
    """`Punctuation.preserve`: each line cut at its punctuation; the chunks
    (empty ones dropped) and the marks with where they go."""
    var all = List[String]()
    for num in range(len(lines)):
        var line = lines[num]
        var found = _find_marks(line)
        if len(found) == 0:
            all.append(line)
            continue
        if len(found) == 1 and found[0] == line:
            marks.append(_Mark(num, line, _A))
            continue
        var these = List[_Mark]()
        for k in range(len(found)):
            var pos = _I
            if k == 0 and line.startswith(found[k]):
                pos = _B
            elif k == len(found) - 1 and line.endswith(found[k]):
                pos = _E
            these.append(_Mark(num, found[k], pos))
        var rest = line
        for k in range(len(these)):
            var at = rest.find(these[k].mark)
            if at < 0:
                all.append(rest)
                rest = String("")
            else:
                all.append(String(rest[byte=0:at]))
                var tail = String(rest[byte = at + these[k].mark.byte_length() :])
                rest = tail^
        all.append(rest)
        for k in range(len(these)):
            marks.append(these[k].copy())
    for k in range(len(all)):
        if all[k].byte_length() > 0:
            chunks.append(all[k])


def _restore(var text: List[String], var marks: List[_Mark]) -> List[String]:
    """`Punctuation.restore` with the default separator (word " ") and
    `strip=False`, line for line."""
    var out = List[String]()
    var pos = 0
    var t0 = 0
    var m0 = 0
    while t0 < len(text) or m0 < len(marks):
        if m0 >= len(marks):
            for k in range(t0, len(text)):
                var line = text[k]
                if not line.endswith(" "):
                    line += " "
                out.append(line)
            t0 = len(text)
        elif t0 >= len(text):
            var joined = String("")
            for k in range(m0, len(marks)):
                joined += marks[k].mark
            out.append(joined)
            m0 = len(marks)
        else:
            ref cur = marks[m0]
            if cur.index == pos:
                var mark = cur.mark
                var position = cur.position
                m0 += 1
                if text[t0].endswith(" "):
                    text[t0] = _drop_last_byte(text[t0])
                if position == _B:
                    text[t0] = mark + text[t0]
                elif position == _E:
                    out.append(text[t0] + mark + (String("") if mark.endswith(" ") else String(" ")))
                    t0 += 1
                    pos += 1
                elif position == _A:
                    out.append(mark + (String("") if mark.endswith(" ") else String(" ")))
                    pos += 1
                else:
                    if len(text) - t0 == 1:
                        text[t0] = text[t0] + mark
                    else:
                        var first = text[t0]
                        t0 += 1
                        text[t0] = first + mark + text[t0]
            else:
                out.append(text[t0])
                t0 += 1
                pos += 1
    return out^


# ── one chunk through espeak (EspeakBackend._postprocess_line) ─────────────


def _collapse_underscores(s: String) -> String:
    """`re.sub(r'_+', '_', s)` then `re.sub(r'_ ', ' ', s)`."""
    var a = String("")
    var prev_us = False
    for sl in s.codepoint_slices():
        var ch = String(sl)
        if ch == "_":
            if not prev_us:
                a += ch
            prev_us = True
        else:
            a += ch
            prev_us = False
    return a.replace("_ ", " ")


def _remove_flags(s: String) -> String:
    """`re.sub(r'\\(.+?\\)', '', s)`: a language-switch flag such as `(en)`."""
    var out = String("")
    var c = _Chars(s)
    var n = len(c.cps)
    var i = 0
    while i < n:
        if c.strs[i] == "(":
            var j = i + 2
            while j < n and c.strs[j] != ")":
                j += 1
            if j < n:
                i = j + 1
                continue
        out += c.strs[i]
        i += 1
    return out^


def _postprocess_line(raw: String, tie: String, remove_flags: Bool) -> String:
    var line = py_strip(raw).replace("\n", " ").replace("  ", " ")
    line = _collapse_underscores(line)
    if remove_flags:
        line = _remove_flags(line)
    if line.byte_length() == 0:
        return String("")
    var out = String("")
    for w in line.split(" "):
        var word = py_strip(String(w))
        if tie.byte_length() == 0:
            word += "_"
            word = word.replace("_", "")
        elif tie != COMBINING_TIE:
            word = word.replace(COMBINING_TIE, tie)
        else:
            word = word.replace("_", "")
        out += word + " "
    return out^


# ── the library ────────────────────────────────────────────────────────────


def _cstr(addr: Int) -> String:
    if addr == 0:
        return String("")
    return String(unsafe_from_utf8_ptr=Pointer[c_char, MutUntrackedOrigin](unsafe_from_address=addr))


def _word_at(addr: Int) -> Int:
    return Pointer[Int, MutUntrackedOrigin](unsafe_from_address=addr)[]


def espeak_default_paths() raises -> Tuple[String, String]:
    """`(library, data)`: `$NOEIRA_ESPEAK_LIB` / `$NOEIRA_ESPEAK_DATA` if set,
    else `~/.cache/noeira/espeak-ng/` (`tools/kokoro/fetch_espeak_ng.py`)."""
    var lib = getenv("NOEIRA_ESPEAK_LIB")
    var data = getenv("NOEIRA_ESPEAK_DATA")
    var dir = noeira_cache() + "/espeak-ng"
    if lib.byte_length() == 0:
        lib = dir + "/" + ESPEAK_LIB_NAME
    if data.byte_length() == 0:
        data = dir + "/espeak-ng-data"
    return (lib, data)


struct EspeakNG(Movable):
    var _lib: OwnedDLHandle
    var version: String
    var data_path: String
    var language: String
    """The voice's language code, as `set_voice` was given it."""

    def __init__(out self, var lib: OwnedDLHandle, var version: String, var data_path: String):
        self._lib = lib^
        self.version = version^
        self.data_path = data_path^
        self.language = String("")

    @staticmethod
    def open(var lib_path: String = String(""), var data_path: String = String("")) raises -> Self:
        """Load and initialise espeak-ng (module docstring for the search)."""
        var defaults = espeak_default_paths()
        if lib_path.byte_length() == 0:
            lib_path = defaults[0]
        if data_path.byte_length() == 0:
            data_path = defaults[1]
        if not exists(lib_path) or not exists(data_path):
            raise Error(
                "espeak-ng: no library at " + lib_path + " or no data at " + data_path
                + ". Install them: python3 tools/kokoro/fetch_espeak_ng.py"
            )
        var lib = OwnedDLHandle(lib_path)
        var path_c = data_path + "\0"
        var rate = lib.get_function[Int32]("espeak_Initialize")(
            _AUDIO_OUTPUT_SYNCHRONOUS, Int32(0), Int(path_c.unsafe_ptr()), Int32(0)
        )
        _ = path_c^
        if rate <= 0:
            raise Error("espeak-ng: espeak_Initialize failed (" + String(rate) + ") with data " + data_path)
        var slot = List[Int](length=1, fill=0)
        var v = _cstr(lib.get_function[Int]("espeak_Info")(Int(slot.unsafe_ptr())))
        _ = slot^
        return Self(lib^, py_strip(v), data_path^)

    def voices(self) raises -> List[Tuple[String, String]]:
        """`(language, identifier)` of every voice, in espeak's order (most
        relevant first within a language)."""
        var out = List[Tuple[String, String]]()
        var arr = self._lib.get_function[Int]("espeak_ListVoices")(0)
        if arr == 0:
            raise Error("espeak-ng: espeak_ListVoices returned nothing")
        var k = 0
        while True:
            var v = _word_at(arr + 8 * k)
            if v == 0:
                break
            # struct espeak_VOICE { const char *name, *languages, *identifier; ... }
            # `languages` is a priority byte, then the first language code.
            var langs = _word_at(v + 8)
            var ident = _cstr(_word_at(v + 16))
            out.append((_cstr(langs + 1), ident^))
            k += 1
        return out^

    def set_voice(mut self, language: String) raises:
        """Select the first non-mbrola voice for `language` (`"en-us"`,
        `"fr-fr"`, ...), as phonemizer's `EspeakWrapper.set_voice` does."""
        var vs = self.voices()
        var ident = String("")
        for i in range(len(vs)):
            if vs[i][1].replace("\\", "/").startswith("mb/"):
                continue
            if vs[i][0] == language:
                ident = vs[i][1]
                break
        if ident.byte_length() == 0:
            raise Error("espeak-ng: invalid voice code \"" + language + "\"")
        var name_c = ident + "\0"
        var rc = self._lib.get_function[Int32]("espeak_SetVoiceByName")(Int(name_c.unsafe_ptr()))
        _ = name_c^
        if rc != 0:
            raise Error("espeak-ng: failed to load voice \"" + language + "\" (" + ident + ")")
        self.language = language

    def text_to_phonemes(self, text: String, tie: Bool) raises -> String:
        """espeak's IPA for `text`, its pieces joined by spaces: phonemes
        separated by `_`, or with U+0361 inside multi-letter phonemes when
        `tie` (phonemizer's `EspeakWrapper.text_to_phonemes`)."""
        if self.language.byte_length() == 0:
            raise Error("espeak-ng: no voice set")
        var mode: Int
        if tie:
            mode = _PHONEMES_IPA | _PHONEMES_TIE | (0x361 << 8)
        else:
            mode = (95 << 8) | _PHONEMES_IPA
        var buf = List[UInt8]()
        for b in text.as_bytes():
            buf.append(b)
        buf.append(0)
        var slot = List[Int](length=1, fill=Int(buf.unsafe_ptr()))
        var slot_addr = Int(slot.unsafe_ptr())
        var f = self._lib.get_function[Int]("espeak_TextToPhonemes")
        var out = String("")
        var first = True
        while _word_at(slot_addr) != 0:
            var piece = _cstr(f(slot_addr, _CHARS_UTF8, Int32(mode)))
            if piece.byte_length() > 0:
                if not first:
                    out += " "
                out += piece
                first = False
        _ = slot^
        _ = buf^
        return out^

    # ── phonemizer ───────────────────────────────────────────────────────

    def backend_phonemize(
        mut self, lines: List[String], language: String, tie: String = String(""),
        remove_flags: Bool = False, preserve_punctuation: Bool = True,
    ) raises -> List[String]:
        """`EspeakBackend(language, preserve_punctuation, with_stress=True,
        tie, language_switch).phonemize(lines)` with the default separator and
        `strip=False`. `tie` is "" (none) or the one character to put inside
        multi-letter phonemes. `preserve_punctuation=False` is phonemizer's
        removal (marks become spaces) — the gate's falsification uses it."""
        if self.language != language:
            self.set_voice(language)
        var chunks = List[String]()
        var marks = List[_Mark]()
        if preserve_punctuation:
            _preserve(lines, chunks, marks)
        else:
            for k in range(len(lines)):
                chunks.append(_remove_marks(lines[k]))
        var phonemized = List[String]()
        for k in range(len(chunks)):
            var raw = self.text_to_phonemes(chunks[k], tie.byte_length() > 0)
            phonemized.append(_postprocess_line(raw, tie, remove_flags))
        if not preserve_punctuation:
            return phonemized^
        return _restore(phonemized^, marks^)

    def phonemize(
        mut self, text: String, language: String, tie: String = String(""),
        remove_flags: Bool = False,
    ) raises -> String:
        """`phonemizer.phonemize(text, language, preserve_punctuation=True,
        with_stress=True, tie=..., language_switch=...)`: lines split on
        `\\n`, blank ones dropped, the result's lines joined with `\\n`."""
        var lines = List[String]()
        for l in _strip_char(text, "\n").split("\n"):
            var line = _strip_char(String(l), "\n")
            if py_strip(line).byte_length() > 0:
                lines.append(line^)
        if len(lines) == 0:
            return String("")
        var out = self.backend_phonemize(lines, language, tie, remove_flags)
        var joined = String("")
        for k in range(len(out)):
            if k > 0:
                joined += "\n"
            joined += out[k]
        return joined^
