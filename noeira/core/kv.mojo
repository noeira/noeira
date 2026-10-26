"""`key=value` text — the shared reader for this tree's own file formats.

Several formats here are a flat block of `key=value` lines with one or more
REPEATING keys: `data/manifest.mojo` (`column=`), checkpoint v2, `core/dotenv`,
and now `tasks/spec.mojo` (`slot=`, `region=`, `active=`, `init=`). The parsing
is four lines each and has been written three times.

⚠ THIS IS THE HOME FOR NEW READERS, NOT YET A MIGRATION. `data/manifest._split`
and `core/dotenv._split_lines` are the two older private copies. Moving them
here is a separate change on the DATA path — the manifest lives inside every
`.h5` this tree has written — and is not worth bundling into a task-layer
commit. **Do not add a fourth copy.**

## ⚠ BYTE-WISE ON PURPOSE

Every one of these formats is ASCII by construction. Splitting by codepoint
would make a stray UTF-8 byte a slicing error somewhere far from the line that
carried it; `_a_byte_slice_of_a_string_asserts_a_codepoint_boundary` is the
recorded shape. The one field that is legitimately human text — a task's
`language=` instruction — is carried through as bytes and never split.
"""


def split_on(s: String, sep: String) -> List[String]:

    """Every field of `s` between single-byte `sep`. Empty fields are kept.

    ⚠⚠ BYTES, NOT `chr` PER BYTE. `cur += chr(Int(bytes[i]))` is the obvious
    spelling and it CORRUPTS any value above 127: `chr` yields the CODEPOINT
    of that byte value, which re-encodes as two bytes, so "日本語" comes back
    as mojibake. This reader carried that bug for as long as it has existed —
    every non-ASCII value in every manifest read back wrong — and it surfaced
    only when a byte-exact task string was finally put through it.
    """
    var out = List[String]()
    var cur = List[UInt8]()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    for i in range(len(bytes)):
        if bytes[i] == sb:
            cur.append(0)
            out.append(String(unsafe_from_utf8_ptr=cur.unsafe_ptr()))
            cur = List[UInt8]()
        else:
            cur.append(bytes[i])
    cur.append(0)
    out.append(String(unsafe_from_utf8_ptr=cur.unsafe_ptr()))
    return out^


def split_once(s: String, sep: String) -> List[String]:
    """`[head, tail]` around the FIRST `sep`, or an EMPTY list if absent.

    ⚠ SPLIT-ONCE MATTERS BECAUSE THE VALUE MAY CONTAIN THE SEPARATOR. A task's
    `language=Put the brick in the box, then stop` has no `=`, but a region's
    `region=table:site:table_surface:-0.1,-0.15,0.1,0.15` is split on `:` by a
    caller that must not lose the negative numbers to a greedy split.


    ⚠⚠ BYTES, NOT `chr` PER BYTE. `cur += chr(Int(bytes[i]))` is the obvious
    spelling and it CORRUPTS any value above 127: `chr` yields the CODEPOINT
    of that byte value, which re-encodes as two bytes, so "日本語" comes back
    as mojibake. This reader carried that bug for as long as it has existed —
    every non-ASCII value in every manifest read back wrong — and it surfaced
    only when a byte-exact task string was finally put through it.
    """
    var out = List[String]()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    var cut = -1
    for i in range(len(bytes)):
        if bytes[i] == sb:
            cut = i
            break
    if cut < 0:
        return out^
    var head = List[UInt8]()
    for i in range(cut):
        head.append(bytes[i])
    head.append(0)
    var tail = List[UInt8]()
    for i in range(cut + 1, len(bytes)):
        tail.append(bytes[i])
    tail.append(0)
    out.append(String(unsafe_from_utf8_ptr=head.unsafe_ptr()))
    out.append(String(unsafe_from_utf8_ptr=tail.unsafe_ptr()))
    return out^


struct KvLine(Copyable, ImplicitlyCopyable, Movable):
    """One `key=value` line, plus the 1-based line number it came from.

    ⚠ THE LINE NUMBER IS NOT DECORATION. These files are hand-authored, so a
    diagnostic that cannot say WHERE costs the author a manual scan of a
    file whose whole point is that a human writes it.
    """

    var key: String
    var value: String
    var lineno: Int

    def __init__(out self, key: String, value: String, lineno: Int):
        self.key = key
        self.value = value
        self.lineno = lineno


def kv_lines(text: String, what: String) raises -> List[KvLine]:
    """Every `key=value` line of `text`, in order. Blank and `#` lines skipped.

    `what` names the format in diagnostics ("task spec", "family spec").

    ⚠ RAISES ON A LINE WITH NO `=`. It does NOT decide what to do with an
    unknown KEY — that is the caller's policy, and the two callers here differ:
    `data/manifest.mojo` ignores unknown keys so a store written by a newer
    build stays readable, while `tasks/spec.mojo` REFUSES them, because a
    silently dropped `goal=` is a task that always succeeds. Keeping that
    decision out of here is what lets both be right.
    """
    var out = List[KvLine]()
    var lines = split_on(text, String("\n"))
    for i in range(len(lines)):
        var line = String(lines[i].strip())
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var kv = split_once(line, String("="))
        if len(kv) != 2:
            raise Error(
                what + ": line " + String(i + 1)
                + " has no '=': '" + line + "'"
            )
        var key = String(kv[0].strip())
        if key.byte_length() == 0:
            raise Error(
                what + ": line " + String(i + 1) + " has an empty key: '"
                + line + "'"
            )
        out.append(KvLine(key, String(kv[1].strip()), i + 1))
    return out^


# =============================================================================
# Writing — the other half of the format
# =============================================================================
#
# ⚠⚠ THIS MODULE STILL IMPORTS NOTHING, AND THAT IS DELIBERATE. Writing a file
# is `write_text_atomic(path, kv_write(...))` at the CALLER, not a `kv_write_file`
# here. `tasks/spec.mojo` already notes that it is imported by modules it must
# not import back; keeping this leaf dependency-free is what lets anything in the
# tree read and write the format without dragging `io/` in behind it.
#
# ⚠ ONE KNOWN SITE IS DELIBERATELY NOT MIGRATED: `nn/core/save_scalar.mojo`
# emits `prefix=value` lines for checkpoint v2. It is on the checkpoint hot
# path, it has typed emitters per scalar kind, and every `.ckpt` in the tree is
# in its format — so it is the same "separate change on the DATA path" this
# file's header already reserves for `data/manifest._split`. Recorded so the
# next reader does not rediscover it as an oversight.


def _reject(what: String, field: String, key: String, why: String) raises:
    raise Error(
        what + ": refusing to write " + field + " '" + key + "' — " + why
    )


def _check(what: String, field: String, s: String, is_key: Bool) raises:
    """Refuse anything this format cannot read back as it was written.

    ⚠⚠ THE RULE IS ROUND-TRIP, NOT A CHARACTER BLACKLIST, AND THAT IS WHY IT IS
    SPELLED `s != s.strip()`. `kv_lines` strips the line, then the key, then the
    value; so a value with a trailing space, a tab, or a stray `\\r` comes back
    SHORTER than it went in — silently, and only for the one field that had it.
    Comparing against `strip()` is the exact condition, and it cannot drift out
    of step with the reader the way an enumerated list of characters would.

    ⚠ A NEWLINE IS THE ONE THAT CORRUPTS THE FILE RATHER THAN THE FIELD. A
    value carrying `\\n` writes two lines; the second has no `=`, so the whole
    file stops parsing at a line the author never wrote. `tag=` is free human
    text pasted by a person and is exactly where this will arrive.

    ⚠ `=` IN A VALUE IS FINE. `split_once` cuts at the FIRST one, which is what
    lets `outcome=success_rate=0.82` and `region=table:site:...` work. Only a
    KEY may not contain it — that would move the cut.
    """
    if is_key and s.byte_length() == 0:
        _reject(what, String("key"), s, String("it is empty"))
    if s.find("\n") >= 0:
        _reject(
            what,
            String("key") if is_key else String("value"),
            s,
            String("it contains a newline, which would split the record"),
        )
    if s != String(s.strip()):
        _reject(
            what,
            String("key") if is_key else String("value"),
            s,
            String(
                "it has leading or trailing whitespace, which the reader"
                " strips — it would not round-trip"
            ),
        )
    if is_key:
        if s.find("=") >= 0:
            _reject(what, String("key"), s, String("it contains '='"))
        if s.startswith("#"):
            _reject(
                what, String("key"), s,
                String("it would read back as a comment"),
            )


struct KvWriter(Movable):
    """Accumulates `key=value` lines and refuses the ones that cannot be read
    back. `done()` is the file's text, newline-terminated.

    ⚠⚠ VALIDATION IS AT `add`, NOT AT `done`. The diagnostic has to name the
    pair the caller was holding; a check deferred to the end reports a bad
    `tag=` from a `RunContext.close()` fifty lines away from the `set_tag` that
    accepted it.

    ⚠ REPEATING KEYS ARE THE POINT, NOT AN ACCIDENT. `artifact=`, `config=` and
    `region=` all appear many times in one file; nothing here deduplicates.
    """

    var _out: String
    var _what: String
    var _n: Int

    def __init__(out self, what: String = String("kv")):
        self._out = String("")
        self._what = what
        self._n = 0

    def __init__(out self, *, deinit move: Self):
        self._out = move._out^
        self._what = move._what^
        self._n = move._n

    def add(mut self, key: String, value: String) raises:
        _check(self._what, String("key"), key, True)
        _check(self._what, String("value"), value, False)
        self._out += key + "=" + value + "\n"
        self._n += 1

    def comment(mut self, text: String) raises:
        """A `#` line. ⚠ It does not survive a read/write round trip — nothing
        in `kv_lines` returns comments — so use it for a file header a human
        reads, never for anything a program must recover."""
        if text.find("\n") >= 0:
            _reject(
                self._what, String("comment"), text,
                String("it contains a newline"),
            )
        self._out += "# " + text + "\n"

    def count(self) -> Int:
        """Lines added. Diagnostics and gates."""
        return self._n

    def done(deinit self) -> String:
        return self._out^


def kv_write(lines: List[KvLine], what: String = String("kv")) raises -> String:
    """Every line's `key=value`, in order, newline-terminated.

    ⚠ `lineno` IS IGNORED, AND IT HAS TO BE. It records where a line CAME FROM;
    honouring it on the way out would mean padding a file with blank lines to
    reach a number that was only ever true of the file it was read from.
    """
    var w = KvWriter(what)
    for i in range(len(lines)):
        w.add(lines[i].key, lines[i].value)
    return w^.done()
