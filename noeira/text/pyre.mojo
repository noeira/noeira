# +--------------------------------------------------------------------------+ #
# | pyre — a backtracking matcher for the subset of Python `re` d1's parser uses
# +--------------------------------------------------------------------------+ #
"""Python's `re` semantics on code points, for transcribing code that is
written with regular expressions (d1's `parsing.py`, docs/NN_MODEL_PORTS_PLAN.md
§12.9 T6) with its patterns kept VERBATIM — gated against `re` itself on the
same patterns and texts (`tests/llm/test_d1_parsing_reference.mojo`).

    var re = PyRe.compile("(?<!\\d)(\\d{1,4})([/.-])(\\d{1,2})\\2(\\d{2,4})(?!\\d)")
    var m = re.fullmatch(cps_of("20/10/2026"))       # Optional[PyMatch]
    m.value().text(cps, 1)                            # "20"

Supported: literals, `.` (not a newline), classes `[...]` / `[^...]` with
ranges and `\\d \\s \\w` inside, `\\d \\D \\s \\S \\w \\W \\b \\B`, `\\uXXXX`, `\\xXX`,
`\\t \\n \\r \\f \\v`, other escaped characters as themselves; greedy and lazy
`* + ? {m} {m,} {m,n}` (a `{` that is no quantifier is a literal); capturing
groups, `(?:...)`, `(?i:...)`; alternation; `(?=...)`, `(?!...)`, and
FIXED-WIDTH `(?<=...)`, `(?<!...)`; back-references `\\1`-`\\9`; `^` (the
text's start), `$` (its end, or before a final newline); the whole-pattern
IGNORECASE flag. Python's Unicode classes (`noeira.text.pyunicode_tables`):
`\\d` = `str.isdecimal`, `\\w` = `str.isalnum` or `_`, `\\s` = `str.isspace`.

⚠ Case-insensitive matching compares `str.lower()` of one character to one
character (a class tests a character and its lowercase): enough for d1's
patterns, whose letters are all lowercase or ASCII; not Python's full
case-folding tables.
"""

from noeira.text.pyunicode_tables import (
    DECIMAL_N, DECIMAL_LO, DECIMAL_HI, WORD_N, WORD_LO, WORD_HI, LOWER_N, LOWER_FROM, LOWER_TO,
)


# ── character classes ─────────────────────────────────────────────────────

def _in_ranges[N: Int](cp: UInt32, lo: Array[UInt32, N], hi: Array[UInt32, N], n: Int) -> Bool:
    var a = 0
    var b = n
    while a < b:
        var m = (a + b) // 2
        if cp > hi[m]:
            a = m + 1
        elif cp < lo[m]:
            b = m
        else:
            return True
    return False


def py_isdecimal(cp: UInt32) -> Bool:
    """`re`'s `\\d` (Unicode Nd)."""
    if cp < 0x80:
        return cp >= 48 and cp <= 57
    return _in_ranges(cp, materialize[DECIMAL_LO](), materialize[DECIMAL_HI](), DECIMAL_N)


def decimal_value(cp: UInt32) -> Int:
    """The digit's value (`int()` of one Nd character), -1 for none."""
    if cp >= 48 and cp <= 57:
        return Int(cp) - 48
    var lo = materialize[DECIMAL_LO]()
    var hi = materialize[DECIMAL_HI]()
    for i in range(DECIMAL_N):
        if cp >= lo[i] and cp <= hi[i]:
            return Int(cp - lo[i]) % 10
    return -1


def py_isword(cp: UInt32) -> Bool:
    """`re`'s `\\w`: `str.isalnum()` or `_`."""
    if cp < 0x80:
        return (cp >= 48 and cp <= 57) or (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122) or cp == 95
    return _in_ranges(cp, materialize[WORD_LO](), materialize[WORD_HI](), WORD_N)


def py_isspace_cp(cp: UInt32) -> Bool:
    if cp == 0x20 or (cp >= 0x09 and cp <= 0x0D) or (cp >= 0x1C and cp <= 0x1F):
        return True
    if cp < 0x85:
        return False
    return (cp == 0x85 or cp == 0xA0 or cp == 0x1680 or (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028
            or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000)


def py_lower(cp: UInt32) -> UInt32:
    """`str.lower()` of one character when it is one character (else the character)."""
    if cp < 0x80:
        return cp + 32 if cp >= 65 and cp <= 90 else cp
    var fr = materialize[LOWER_FROM]()
    var to = materialize[LOWER_TO]()
    var a = 0
    var b = LOWER_N
    while a < b:
        var m = (a + b) // 2
        if fr[m] < cp:
            a = m + 1
        else:
            b = m
    if a < LOWER_N and fr[a] == cp:
        return to[a]
    return cp


# ── the program ───────────────────────────────────────────────────────────

comptime OP_CHAR = 0
comptime OP_ANY = 1
comptime OP_CLASS = 2
comptime OP_SPLIT = 3
"""Try `a`, then `b`."""
comptime OP_JMP = 4
comptime OP_SAVE = 5
comptime OP_MATCH = 6
comptime OP_BOL = 7
comptime OP_EOL = 8
comptime OP_WORDB = 9
comptime OP_NWORDB = 10
comptime OP_BACKREF = 11
comptime OP_LOOK = 12
"""`a`: the sub-program; `b`: 0 ahead, 1 behind; `neg`; `c`: a look-behind's width."""


@fieldwise_init
struct Inst(Copyable, Movable):
    var op: Int
    var a: Int
    var b: Int
    var c: UInt32
    var icase: Bool
    var neg: Bool


@fieldwise_init
struct CharClass(Copyable, Movable):
    var neg: Bool
    var lo: List[UInt32]
    var hi: List[UInt32]
    var d: Bool
    var s: Bool
    var w: Bool
    var nd: Bool
    var ns: Bool
    var nw: Bool

    def has(self, c: UInt32) -> Bool:
        var hit = False
        for i in range(len(self.lo)):
            if c >= self.lo[i] and c <= self.hi[i]:
                hit = True
                break
        if not hit:
            hit = ((self.d and py_isdecimal(c)) or (self.s and py_isspace_cp(c)) or (self.w and py_isword(c))
                   or (self.nd and not py_isdecimal(c)) or (self.ns and not py_isspace_cp(c))
                   or (self.nw and not py_isword(c)))
        return hit != self.neg


# ── the parse tree ────────────────────────────────────────────────────────

comptime N_CHAR = 0
comptime N_ANY = 1
comptime N_CLASS = 2
comptime N_CAT = 3
comptime N_ALT = 4
comptime N_REPEAT = 5
comptime N_GROUP = 6
comptime N_LOOK = 7
comptime N_ASSERT = 8
comptime N_BACKREF = 9
comptime INF = 1 << 30


@fieldwise_init
struct Node(Copyable, Movable):
    var kind: Int
    var kids: List[Int]
    var c: UInt32
    var lo: Int
    var hi: Int
    var greedy: Bool
    var group: Int
    var icase: Bool
    var neg: Bool
    var behind: Bool
    var op: Int
    """N_ASSERT: OP_BOL, OP_EOL, OP_WORDB or OP_NWORDB."""


def _node(kind: Int) -> Node:
    return Node(kind, List[Int](), 0, 0, 0, True, -1, False, False, False, 0)


struct _Parser:
    var p: List[UInt32]
    var i: Int
    var nodes: List[Node]
    var classes: List[CharClass]
    var ngroups: Int

    def __init__(out self, var p: List[UInt32]):
        self.p = p^
        self.i = 0
        self.nodes = List[Node]()
        self.classes = List[CharClass]()
        self.ngroups = 0

    def add(mut self, var n: Node) -> Int:
        self.nodes.append(n^)
        return len(self.nodes) - 1

    def peek(self) -> Int:
        return Int(self.p[self.i]) if self.i < len(self.p) else -1

    def alt(mut self, icase: Bool) raises -> Int:
        var branches: List[Int] = [self.cat(icase)]
        while self.peek() == 124:  # |
            self.i += 1
            branches.append(self.cat(icase))
        if len(branches) == 1:
            return branches[0]
        var n = _node(N_ALT)
        n.kids = branches^
        return self.add(n^)

    def cat(mut self, icase: Bool) raises -> Int:
        var items = List[Int]()
        while self.i < len(self.p) and self.peek() != 124 and self.peek() != 41:
            items.append(self.repeat(icase))
        var n = _node(N_CAT)
        n.kids = items^
        return self.add(n^)

    def _number(mut self) -> Int:
        var v = -1
        while self.i < len(self.p) and self.p[self.i] >= 48 and self.p[self.i] <= 57:
            v = (0 if v < 0 else v * 10) + Int(self.p[self.i]) - 48
            self.i += 1
        return v

    def repeat(mut self, icase: Bool) raises -> Int:
        var atom = self.atom(icase)
        while True:
            var c = self.peek()
            var lo = -1
            var hi = -1
            if c == 42:  # *
                lo = 0
                hi = INF
                self.i += 1
            elif c == 43:  # +
                lo = 1
                hi = INF
                self.i += 1
            elif c == 63:  # ?
                lo = 0
                hi = 1
                self.i += 1
            elif c == 123:  # {m}, {m,}, {m,n}; otherwise a literal brace
                var save = self.i
                self.i += 1
                var m = self._number()
                if self.peek() == 125 and m >= 0:
                    self.i += 1
                    lo = m
                    hi = m
                elif self.peek() == 44:
                    self.i += 1
                    var n2 = self._number()
                    if self.peek() == 125:
                        self.i += 1
                        lo = m if m >= 0 else 0
                        hi = n2 if n2 >= 0 else INF
                    else:
                        self.i = save
                else:
                    self.i = save
                if lo < 0:
                    return atom
            else:
                return atom
            var greedy = True
            if self.peek() == 63:
                greedy = False
                self.i += 1
            var n = _node(N_REPEAT)
            n.kids = [atom]
            n.lo = lo
            n.hi = hi
            n.greedy = greedy
            atom = self.add(n^)

    def _escape_char(mut self) raises -> UInt32:
        """After a backslash: `\\uXXXX`, `\\xXX`, `\\t` ... or the character itself."""
        var c = self.p[self.i]
        self.i += 1
        if c == 117 or c == 120:  # u, x
            var nd = 4 if c == 117 else 2
            var v = 0
            for _ in range(nd):
                var h = Int(self.p[self.i])
                self.i += 1
                v = v * 16 + (h - 48 if h <= 57 else (h - 87 if h >= 97 else h - 55))
            return UInt32(v)
        if c == 116:
            return 9
        if c == 110:
            return 10
        if c == 114:
            return 13
        if c == 102:
            return 12
        if c == 118:
            return 11
        return c

    def atom(mut self, icase: Bool) raises -> Int:
        var c = self.p[self.i]
        if c == 40:  # (
            self.i += 1
            if self.peek() == 63:  # ?
                self.i += 1
                var k = self.peek()
                if k == 58:  # (?:
                    self.i += 1
                    var inner = self.alt(icase)
                    self._close()
                    return inner
                if k == 105 and self.i + 1 < len(self.p) and self.p[self.i + 1] == 58:  # (?i:
                    self.i += 2
                    var inner = self.alt(True)
                    self._close()
                    return inner
                var behind = False
                if k == 60:  # <
                    behind = True
                    self.i += 1
                    k = self.peek()
                if k == 61 or k == 33:  # = !
                    self.i += 1
                    var inner = self.alt(icase)
                    self._close()
                    var n = _node(N_LOOK)
                    n.kids = [inner]
                    n.neg = k == 33
                    n.behind = behind
                    return self.add(n^)
                raise Error("pyre: unsupported group syntax")
            self.ngroups += 1
            var g = self.ngroups
            var inner = self.alt(icase)
            self._close()
            var n = _node(N_GROUP)
            n.kids = [inner]
            n.group = g
            return self.add(n^)
        if c == 91:  # [
            return self._class(icase)
        if c == 46:  # .
            self.i += 1
            return self.add(_node(N_ANY))
        if c == 94 or c == 36:  # ^ $
            self.i += 1
            var n = _node(N_ASSERT)
            n.op = OP_BOL if c == 94 else OP_EOL
            return self.add(n^)
        if c == 92:  # \
            self.i += 1
            var e = self.p[self.i]
            if e == 98 or e == 66:  # \b \B
                self.i += 1
                var n = _node(N_ASSERT)
                n.op = OP_WORDB if e == 98 else OP_NWORDB
                return self.add(n^)
            if e >= 49 and e <= 57:  # \1-\9
                self.i += 1
                var n = _node(N_BACKREF)
                n.group = Int(e) - 48
                n.icase = icase
                return self.add(n^)
            if e == 100 or e == 68 or e == 115 or e == 83 or e == 119 or e == 87:  # \d \D \s \S \w \W
                self.i += 1
                var cc = CharClass(False, List[UInt32](), List[UInt32](), e == 100, e == 115, e == 119, e == 68,
                                   e == 83, e == 87)
                self.classes.append(cc^)
                var n = _node(N_CLASS)
                n.group = len(self.classes) - 1
                n.icase = icase
                return self.add(n^)
            var ch = self._escape_char()
            var n = _node(N_CHAR)
            n.c = ch
            n.icase = icase
            return self.add(n^)
        self.i += 1
        var n = _node(N_CHAR)
        n.c = c
        n.icase = icase
        return self.add(n^)

    def _close(mut self) raises:
        if self.peek() != 41:
            raise Error("pyre: missing )")
        self.i += 1

    def _class(mut self, icase: Bool) raises -> Int:
        self.i += 1  # [
        var cc = CharClass(False, List[UInt32](), List[UInt32](), False, False, False, False, False, False)
        if self.peek() == 94:
            cc.neg = True
            self.i += 1
        var first = True
        while True:
            if self.i >= len(self.p):
                raise Error("pyre: unterminated class")
            var c = self.p[self.i]
            if c == 93 and not first:  # ]
                self.i += 1
                break
            first = False
            var lo: UInt32
            if c == 92:
                self.i += 1
                var e = self.p[self.i]
                if e == 100:
                    cc.d = True
                    self.i += 1
                    continue
                if e == 115:
                    cc.s = True
                    self.i += 1
                    continue
                if e == 119:
                    cc.w = True
                    self.i += 1
                    continue
                if e == 68:
                    cc.nd = True
                    self.i += 1
                    continue
                if e == 83:
                    cc.ns = True
                    self.i += 1
                    continue
                if e == 87:
                    cc.nw = True
                    self.i += 1
                    continue
                lo = self._escape_char()
            else:
                lo = c
                self.i += 1
            var hi = lo
            if self.peek() == 45 and self.i + 1 < len(self.p) and self.p[self.i + 1] != 93:  # a-z
                self.i += 1
                var c2 = self.p[self.i]
                if c2 == 92:
                    self.i += 1
                    hi = self._escape_char()
                else:
                    hi = c2
                    self.i += 1
            cc.lo.append(lo)
            cc.hi.append(hi)
        self.classes.append(cc^)
        var n = _node(N_CLASS)
        n.group = len(self.classes) - 1
        n.icase = icase
        return self.add(n^)


# ── compiling the tree ────────────────────────────────────────────────────

struct PyMatch(Copyable, Movable):
    """Group spans in code points; -1 for a group that did not take part."""

    var spans: List[Int]

    def __init__(out self, var spans: List[Int]):
        self.spans = spans^

    def start(self, g: Int = 0) -> Int:
        return self.spans[2 * g]

    def end(self, g: Int = 0) -> Int:
        return self.spans[2 * g + 1]

    def has(self, g: Int) -> Bool:
        return self.spans[2 * g] >= 0

    def text(self, t: List[UInt32], g: Int = 0) -> String:
        """The group's text, "" when it did not take part (`has` tells None from "")."""
        var out = String("")
        if not self.has(g):
            return out
        for i in range(self.start(g), self.end(g)):
            out += String(Codepoint(unsafe_unchecked_codepoint=t[i]))
        return out^


struct PyRe(Movable):
    var progs: List[List[Inst]]
    """Program 0 is the pattern; the others are its look-arounds."""
    var classes: List[CharClass]
    var ngroups: Int
    var pattern: String

    def __init__(out self, var progs: List[List[Inst]], var classes: List[CharClass], ngroups: Int, var pattern: String):
        self.progs = progs^
        self.classes = classes^
        self.ngroups = ngroups
        self.pattern = pattern^

    @staticmethod
    def compile(pattern: String, ignorecase: Bool = False) raises -> PyRe:
        var cps = List[UInt32]()
        for c in pattern.codepoints():
            cps.append(c.to_u32())
        var ps = _Parser(cps^)
        var root = ps.alt(ignorecase)
        if ps.i != len(ps.p):
            raise Error("pyre: unbalanced ) in " + pattern)
        var progs = List[List[Inst]]()
        progs.append(List[Inst]())
        _emit(ps.nodes, root, progs, 0)
        progs[0].append(Inst(OP_MATCH, 0, 0, 0, False, False))
        var classes = ps.classes.copy()
        return PyRe(progs^, classes^, ps.ngroups, pattern)

    # ── running ──

    def _run(self, prog: Int, pc0: Int, pos0: Int, t: List[UInt32], mut caps: List[Int], end_at: Int) -> Int:
        """The match's end from `pc0` at `pos0`, -1 for none (backtracking:
        a SPLIT tries its first branch recursively). `end_at` >= 0: MATCH only
        there (`fullmatch`)."""
        var pc = pc0
        var pos = pos0
        ref code = self.progs[prog]
        while True:
            ref ins = code[pc]
            var op = ins.op
            if op == OP_CHAR:
                if pos >= len(t):
                    return -1
                var x = t[pos]
                if x != ins.c and not (ins.icase and py_lower(x) == py_lower(ins.c)):
                    return -1
                pos += 1
                pc += 1
            elif op == OP_ANY:
                if pos >= len(t) or t[pos] == 10:
                    return -1
                pos += 1
                pc += 1
            elif op == OP_CLASS:
                if pos >= len(t):
                    return -1
                ref cc = self.classes[ins.a]
                var x = t[pos]
                if not (cc.has(x) or (ins.icase and cc.has(py_lower(x)))):
                    return -1
                pos += 1
                pc += 1
            elif op == OP_SPLIT:
                var saved = caps.copy()
                var r = self._run(prog, ins.a, pos, t, caps, end_at)
                if r >= 0:
                    return r
                caps = saved^
                pc = ins.b
            elif op == OP_JMP:
                pc = ins.a
            elif op == OP_SAVE:
                caps[ins.a] = pos
                pc += 1
            elif op == OP_MATCH:
                if prog == 0 and end_at >= 0 and pos != end_at:
                    return -1
                return pos
            elif op == OP_BOL:
                if pos != 0:
                    return -1
                pc += 1
            elif op == OP_EOL:
                if not (pos == len(t) or (pos == len(t) - 1 and t[pos] == 10)):
                    return -1
                pc += 1
            elif op == OP_WORDB or op == OP_NWORDB:
                var before = pos > 0 and py_isword(t[pos - 1])
                var after = pos < len(t) and py_isword(t[pos])
                if (before != after) != (op == OP_WORDB):
                    return -1
                pc += 1
            elif op == OP_BACKREF:
                var a = caps[2 * ins.a]
                var b = caps[2 * ins.a + 1]
                if a < 0 or b < 0:
                    return -1
                for k in range(b - a):
                    if pos + k >= len(t):
                        return -1
                    var x = t[pos + k]
                    var y = t[a + k]
                    if x != y and not (ins.icase and py_lower(x) == py_lower(y)):
                        return -1
                pos += b - a
                pc += 1
            else:  # OP_LOOK
                var ok: Bool
                var sub_caps = caps.copy()
                if ins.b == 1:
                    var w = Int(ins.c)
                    ok = pos - w >= 0 and self._run(ins.a, 0, pos - w, t, sub_caps, -1) == pos
                else:
                    ok = self._run(ins.a, 0, pos, t, sub_caps, -1) >= 0
                if ok == ins.neg:
                    return -1
                if ok and not ins.neg:
                    caps = sub_caps^
                pc += 1

    def _at(self, t: List[UInt32], start: Int, end_at: Int) -> Optional[PyMatch]:
        var caps = List[Int](length=2 * (self.ngroups + 1), fill=-1)
        var e = self._run(0, 0, start, t, caps, end_at)
        if e < 0:
            return None
        caps[0] = start
        caps[1] = e
        return PyMatch(caps^)

    def search(self, t: List[UInt32], start: Int = 0) -> Optional[PyMatch]:
        """`re.search`: the leftmost match at or after `start`."""
        for s in range(start, len(t) + 1):
            var m = self._at(t, s, -1)
            if m:
                return m^
        return None

    def match(self, t: List[UInt32]) -> Optional[PyMatch]:
        """`re.match`: a match at the start."""
        return self._at(t, 0, -1)

    def fullmatch(self, t: List[UInt32]) -> Optional[PyMatch]:
        """`re.fullmatch`: a match of the whole text."""
        return self._at(t, 0, len(t))

    def finditer(self, t: List[UInt32]) -> List[PyMatch]:
        """`re.finditer`: the non-overlapping matches, left to right (after an
        empty match the search moves on by one)."""
        var out = List[PyMatch]()
        var s = 0
        while s <= len(t):
            var m = self.search(t, s)
            if not m:
                break
            var mm = m.value().copy()
            s = mm.end() if mm.end() > mm.start() else mm.end() + 1
            out.append(mm^)
        return out^

    def sub(self, t: List[UInt32], repl: String) -> List[UInt32]:
        """`re.sub(pattern, repl, text)` with a literal `repl`."""
        var r = List[UInt32]()
        for c in repl.codepoints():
            r.append(c.to_u32())
        var out = List[UInt32]()
        var at = 0
        for ref m in self.finditer(t):
            for i in range(at, m.start()):
                out.append(t[i])
            for c in r:
                out.append(c)
            at = m.end()
        for i in range(at, len(t)):
            out.append(t[i])
        return out^


def _width(nodes: List[Node], n: Int) raises -> Int:
    """A look-behind's fixed width (Python refuses a variable one)."""
    ref nd = nodes[n]
    if nd.kind == N_CHAR or nd.kind == N_ANY or nd.kind == N_CLASS:
        return 1
    if nd.kind == N_ASSERT or nd.kind == N_LOOK:
        return 0
    if nd.kind == N_GROUP:
        return _width(nodes, nd.kids[0])
    if nd.kind == N_CAT:
        var w = 0
        for k in nd.kids:
            w += _width(nodes, k)
        return w
    if nd.kind == N_ALT:
        var w = _width(nodes, nd.kids[0])
        for k in nd.kids:
            if _width(nodes, k) != w:
                raise Error("pyre: look-behind requires fixed-width pattern")
        return w
    if nd.kind == N_REPEAT and nd.lo == nd.hi:
        return nd.lo * _width(nodes, nd.kids[0])
    raise Error("pyre: look-behind requires fixed-width pattern")


def _emit(nodes: List[Node], n: Int, mut progs: List[List[Inst]], p: Int) raises:
    ref nd = nodes[n]
    var k = nd.kind
    if k == N_CHAR:
        progs[p].append(Inst(OP_CHAR, 0, 0, nd.c, nd.icase, False))
    elif k == N_ANY:
        progs[p].append(Inst(OP_ANY, 0, 0, 0, False, False))
    elif k == N_CLASS:
        progs[p].append(Inst(OP_CLASS, nd.group, 0, 0, nd.icase, False))
    elif k == N_ASSERT:
        progs[p].append(Inst(nd.op, 0, 0, 0, False, False))
    elif k == N_BACKREF:
        progs[p].append(Inst(OP_BACKREF, nd.group, 0, 0, nd.icase, False))
    elif k == N_CAT:
        for kid in nd.kids:
            _emit(nodes, kid, progs, p)
    elif k == N_GROUP:
        progs[p].append(Inst(OP_SAVE, 2 * nd.group, 0, 0, False, False))
        _emit(nodes, nd.kids[0], progs, p)
        progs[p].append(Inst(OP_SAVE, 2 * nd.group + 1, 0, 0, False, False))
    elif k == N_ALT:
        var jumps = List[Int]()
        for i in range(len(nd.kids)):
            if i + 1 < len(nd.kids):
                var split = len(progs[p])
                progs[p].append(Inst(OP_SPLIT, split + 1, -1, 0, False, False))
                _emit(nodes, nd.kids[i], progs, p)
                jumps.append(len(progs[p]))
                progs[p].append(Inst(OP_JMP, -1, 0, 0, False, False))
                progs[p][split].b = len(progs[p])
            else:
                _emit(nodes, nd.kids[i], progs, p)
        for j in jumps:
            progs[p][j].a = len(progs[p])
    elif k == N_REPEAT:
        var kid = nd.kids[0]
        for _ in range(nd.lo):
            _emit(nodes, kid, progs, p)
        if nd.hi == INF:
            var top = len(progs[p])
            progs[p].append(Inst(OP_SPLIT, -1, -1, 0, False, False))
            var body = len(progs[p])
            _emit(nodes, kid, progs, p)
            progs[p].append(Inst(OP_JMP, top, 0, 0, False, False))
            var out = len(progs[p])
            progs[p][top].a = body if nd.greedy else out
            progs[p][top].b = out if nd.greedy else body
        else:
            var splits = List[Int]()
            for _ in range(nd.hi - nd.lo):
                var sp = len(progs[p])
                progs[p].append(Inst(OP_SPLIT, -1, -1, 0, False, False))
                splits.append(sp)
                _emit(nodes, kid, progs, p)
            var out = len(progs[p])
            for sp in splits:
                progs[p][sp].a = sp + 1 if nd.greedy else out
                progs[p][sp].b = out if nd.greedy else sp + 1
    else:  # N_LOOK
        var sub = len(progs)
        progs.append(List[Inst]())
        _emit(nodes, nd.kids[0], progs, sub)
        progs[sub].append(Inst(OP_MATCH, 0, 0, 0, False, False))
        var w = _width(nodes, nd.kids[0]) if nd.behind else 0
        progs[p].append(Inst(OP_LOOK, sub, 1 if nd.behind else 0, UInt32(w), False, nd.neg))
