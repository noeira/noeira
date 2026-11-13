# +--------------------------------------------------------------------------+ #
# | BB-1 — the grounded-answers set: its files, its states, its scoring rule
# +--------------------------------------------------------------------------+ #
"""The brain's first bench (docs/ROBOT_BRAIN_PLAN.md, step B1): questions a
person asks the robot whose answers are in its state, and the rule each reply
is scored by. Written by `tools/brain/make_bb1.py` into `tests/brain/`.

    var set = BB1Set.load("tests/brain")
    var ws = set.scenarios[0].state(Body.g1_room("Noé"))      # the robot's situation
    reply_passes("Le frigo est à ma droite.", item.must, item.forbid)

THE RULE (`reply_passes`), on the reply in lower case:
- `must`: groups joined by " && "; a group is alternatives joined by "|";
  every group needs one alternative found. "-" = no requirement.
- `forbid`: alternatives joined by "|"; one found fails the reply.
- an alternative is a phrase, found only on WORD BOUNDARIES ("non" is not in
  "annoncer", "table" is in "la table."), or `num:<v>`: a number written in
  digits, "3,5" and "3.5" alike, and "3" is not found in "3.5".
- an empty reply fails.

⚠ A RULE, NOT A JUDGE. It reads surface words, so a right answer in unusual
words fails and the lists are generous on purpose. Its use is the DIFFERENCE
between two prompts on the same questions, and the mutant rows (`where_mut`:
two places swapped, the old truth must now fail) are what show the score
follows the state.
"""

from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes
from .geometry import PlaceXY
from .state import WorldState, Body


# ── the scoring rule ──────────────────────────────────────────────────────


def normalise(text: String) -> List[UInt8]:
    """Lower case (ASCII and the accented capitals of Latin-1), typographic
    apostrophes to ', no-break spaces to spaces."""
    var b = List[UInt8]()
    for x in text.as_bytes():
        b.append(x)
    var out = List[UInt8]()
    var i = 0
    var n = len(b)
    while i < n:
        var c = Int(b[i])
        if c >= 65 and c <= 90:
            out.append(UInt8(c + 32))
        elif c == 0xC3 and i + 1 < n:
            var d = Int(b[i + 1])
            out.append(UInt8(c))
            out.append(UInt8(d + 0x20) if d >= 0x80 and d <= 0x9E and d != 0x97 else UInt8(d))
            i += 1
        elif c == 0xC2 and i + 1 < n and Int(b[i + 1]) == 0xA0:
            out.append(32)
            i += 1
        elif c == 0xE2 and i + 2 < n and Int(b[i + 1]) == 0x80 and (Int(b[i + 2]) == 0x99 or Int(b[i + 2]) == 0x98):
            out.append(39)
            i += 2
        elif c == 0xE2 and i + 2 < n and Int(b[i + 1]) == 0x80 and Int(b[i + 2]) == 0xAF:
            out.append(32)
            i += 2
        else:
            out.append(UInt8(c))
        i += 1
    return out^


def _wordish(c: UInt8) -> Bool:
    var x = Int(c)
    return (x >= 97 and x <= 122) or (x >= 48 and x <= 57) or x >= 0x80


def has_phrase(ref text: List[UInt8], ref pat: List[UInt8]) -> Bool:
    """`pat` in `text` (both normalised), not glued to a letter or a digit on
    a side where `pat` itself ends in one."""
    var n = len(text)
    var m = len(pat)
    if m == 0 or m > n:
        return False
    for i in range(n - m + 1):
        var same = True
        for k in range(m):
            if text[i + k] != pat[k]:
                same = False
                break
        if not same:
            continue
        if _wordish(pat[0]) and i > 0 and _wordish(text[i - 1]):
            continue
        if _wordish(pat[m - 1]) and i + m < n and _wordish(text[i + m]):
            continue
        return True
    return False


def numbers_in(ref text: List[UInt8]) -> List[Float64]:
    """Every number written in digits: `3`, `3.5`, `3,5` (a `.` or `,` counts
    only between two digits)."""
    var out = List[Float64]()
    var n = len(text)
    var i = 0
    while i < n:
        var c = Int(text[i])
        if c < 48 or c > 57:
            i += 1
            continue
        var v = 0.0
        while i < n and Int(text[i]) >= 48 and Int(text[i]) <= 57:
            v = v * 10.0 + Float64(Int(text[i]) - 48)
            i += 1
        if i + 1 < n and (Int(text[i]) == 46 or Int(text[i]) == 44) and Int(text[i + 1]) >= 48 and Int(text[i + 1]) <= 57:
            i += 1
            var scale = 0.1
            while i < n and Int(text[i]) >= 48 and Int(text[i]) <= 57:
                v += scale * Float64(Int(text[i]) - 48)
                scale *= 0.1
                i += 1
        out.append(v)
    return out^


def _alt_found(ref text: List[UInt8], alt: String) raises -> Bool:
    if alt.startswith("num:"):
        var want = Float64(String(alt[byte=4:]))
        for v in numbers_in(text):
            if abs(v - want) < 1e-6:
                return True
        return False
    var pat = normalise(alt)
    return has_phrase(text, pat)


def _any_found(ref text: List[UInt8], alts: String) raises -> Bool:
    for a in alts.split("|"):
        var alt = String(String(a).strip())
        if alt.byte_length() > 0 and _alt_found(text, alt):
            return True
    return False


def reply_passes(reply: String, must: String, forbid: String) raises -> Bool:
    """The module note's rule. "-" or "" = nothing asked on that side."""
    var text = normalise(reply)
    var blank = True
    for c in text:
        if Int(c) > 32:
            blank = False
            break
    if blank:
        return False
    if must != "-" and must.byte_length() > 0:
        for g in must.split(" && "):
            if not _any_found(text, String(g)):
                return False
    if forbid != "-" and forbid.byte_length() > 0:
        if _any_found(text, forbid):
            return False
    return True


comptime _EN_ONLY = "the|was|is|are|i'm|you|my|and|have|this|it's|i"
comptime _FR_ONLY = "le|la|les|est|je|suis|une|et|pas|que|dans|mon|ma|à|j'ai|c'est"


def reply_in_language(reply: String, lang: String) raises -> Bool:
    """Whether a reply to a question in `lang` (`fr` or `en`) is free of the
    OTHER language's commonest words: a reply in French has no "the", "is" or
    "my"; a reply in English no "le", "est" or "je". Any other `lang`: True.
    A coarse test for one failure, the robot answering in the wrong language."""
    var text = normalise(reply)
    if lang == "fr":
        return not _any_found(text, String(_EN_ONLY))
    if lang == "en":
        return not _any_found(text, String(_FR_ONLY))
    return True


# ── the files ─────────────────────────────────────────────────────────────


def _rows(path: String, columns: Int) raises -> List[List[String]]:
    var out = List[List[String]]()
    var txt = string_from_bytes(read_file_bytes(path))
    var n = 0
    for l in txt.split("\n"):
        n += 1
        var line = String(l)
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var cols = List[String]()
        for c in line.split("\t"):
            cols.append(String(c))
        if len(cols) != columns:
            raise Error(path + " line " + String(n) + ": " + String(len(cols)) + " columns, " + String(columns) + " expected")
        out.append(cols^)
    return out^


def _list(s: String, sep: String) -> List[String]:
    var out = List[String]()
    if s == "-" or s.byte_length() == 0:
        return out^
    for p in s.split(sep):
        out.append(String(p))
    return out^


struct BB1Scenario(Copyable, Movable):
    var sid: String
    var x: Float64
    var y: Float64
    var yaw: Float64
    var doing: String
    var going: String
    var at: String
    var arrivals: List[String]
    """Places arrived at, oldest first, the current one (`at`) last when there is one."""
    var places: List[PlaceXY]
    var ref_lines: String
    """`geometry.place_line` of every place, joined by " ; " (the reference's)."""
    var ref_nearest: String
    var ref_before: List[String]
    var mutant_of: String
    """The scenario this one swaps two places of; "-" for none."""

    def __init__(out self, ref c: List[String]) raises:
        self.sid = c[0].copy()
        self.x = Float64(c[1])
        self.y = Float64(c[2])
        self.yaw = Float64(c[3])
        self.doing = c[4].copy()
        self.going = c[5].copy()
        self.at = c[6].copy()
        self.arrivals = _list(c[7], ",")
        self.places = List[PlaceXY]()
        for p in c[8].split(","):
            var f = String(p).split(":")
            self.places.append(PlaceXY(String(f[0]), Float64(String(f[1])), Float64(String(f[2]))))
        self.ref_lines = c[9].copy()
        self.ref_nearest = c[10].copy()
        self.ref_before = _list(c[11], ",")
        self.mutant_of = c[12].copy()

    def state(self, var body: Body, language: String = String(""), t0: Int = 1_000_000) -> WorldState:
        """The scenario as a `WorldState`, its history replayed through
        `observe_motion` one second an arrival. `language`: the conversation's,
        when the caller knows it (`WorldState.language`)."""
        var ws = WorldState(body^, String("a furnished room"))
        ws.places = self.places.copy()
        ws.language = language.copy()
        if language == "fr":
            # what the room demo gives its robot (`--lang fr`)
            ws.call_place(String("fridge"), String("fr"), String("le frigo"))
            ws.call_place(String("door"), String("fr"), String("la porte"))
            ws.call_place(String("table"), String("fr"), String("la table"))
            ws.call_place(String("sofa"), String("fr"), String("le canapé"))
        var t = t0
        for i in range(len(self.arrivals)):
            # it sets off for the place, then it is there
            ws.observe_motion(String("goto:") + self.arrivals[i], self.arrivals[i], String("none"), t)
            ws.observe_motion(String("goto:") + self.arrivals[i], String("none"), self.arrivals[i], t + 1)
            t += 2
        # now: still there (`at`), or gone from it (`at` = none) toward `going`
        ws.observe_motion(self.doing, self.going, self.at, t)
        ws.observe_pose(self.x, self.y, self.yaw)
        return ws^


struct BB1Item(Copyable, Movable):
    var id: String
    var sid: String
    var cat: String
    var lang: String
    var question: String
    var must: String
    var forbid: String
    var alt_must: String
    var alt_forbid: String

    def __init__(out self, ref c: List[String]):
        self.id = c[0].copy()
        self.sid = c[1].copy()
        self.cat = c[2].copy()
        self.lang = c[3].copy()
        self.question = c[4].copy()
        self.must = c[5].copy()
        self.forbid = c[6].copy()
        self.alt_must = c[7].copy()
        self.alt_forbid = c[8].copy()


struct BB1Set(Movable):
    var scenarios: List[BB1Scenario]
    var items: List[BB1Item]

    def __init__(out self, var scenarios: List[BB1Scenario], var items: List[BB1Item]):
        self.scenarios = scenarios^
        self.items = items^

    @staticmethod
    def load(dir: String) raises -> BB1Set:
        var sc = List[BB1Scenario]()
        for r in _rows(dir + "/bb1_scenarios.tsv", 13):
            sc.append(BB1Scenario(r))
        var it = List[BB1Item]()
        for r in _rows(dir + "/bb1_items.tsv", 9):
            it.append(BB1Item(r))
        return BB1Set(sc^, it^)

    def scenario(self, sid: String) raises -> Int:
        for i in range(len(self.scenarios)):
            if self.scenarios[i].sid == sid:
                return i
        raise Error("bb1: no scenario '" + sid + "'")
