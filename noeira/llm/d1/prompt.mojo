# +--------------------------------------------------------------------------+ #
# | d1 — System One's prompts, packing and readout ids (no model)
# +--------------------------------------------------------------------------+ #
"""d1's `prompt.py` / `attention.pack_set` / `prompt.readout_ids`, transcribed
from d1-2B's own code (docs/NN_MODEL_PORTS_PLAN.md §12.2, §12.9): the three
untyped questions (`noul`, `choice`, `score`) and the typed ones — a `multiple`
choice, a read (`text` / `number` / `date`), an estimate (`number` with a
`range`), a `box` and `boxes`.

    prefix  = "<|startoftext|><|im_start|>user\\n" + plain(state block) + "\\n" + "QUESTION:\\n"
    suffix  = plain(question block) + "<|im_end|>\\n<|im_start|>assistant\\n<think></think>" [+ markers]
    row     = enc(prefix + suffix_1) + enc("QUESTION:\\n" + suffix_k) ...   (16 questions a row)

An untyped question has one answer slot, its last token. A typed one writes
named markers after the assistant header (`start:` `end:`, `present:` `top:` …,
`"option":`), each slot the token holding a marker's colon (`marked_suffix`),
placed by its offset from the question's last token (`suffix_layout`).

The state block: a string as it is, any other JSON value as Python's
`json.dumps(value, ensure_ascii=False, indent=2)`; then a blank line.
⚠ `JsonDoc` keeps numbers as Float64, so an integral float (`3.0`) is written
`3` where Python keeps `3.0`; send such numbers as text if it matters.

`plain` breaks every SPECIAL token's string found in caller text with U+200B
after its first character, so a state cannot open or close a turn.

What a checkpoint records in its `config.json` (`d1_codes`, `d1_numbers`,
`d1_prompt`, `d1_box_edges`, `d1_set_size`, `d1_attention`) is a `D1Record`;
its default is d1-2B's.
"""

from std.math import floor, log10

from noeira.io.json import JsonDoc, J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT, load_json
from noeira.text.bpe import ByteLevelBPE
from noeira.text.unicode_classes import is_letter
from .pystr import cps_of, cp_len, blank, py_rstrip


comptime Q_NOUL: Int = 0
comptime Q_CHOICE: Int = 1
comptime Q_SCORE: Int = 2
comptime Q_MULTI: Int = 3
"""A choice with `multiple: true`: one yes/no slot per option."""
comptime Q_READ: Int = 4
"""A `text`, `number` or `date` read from the state (`read_type`): `start` and `end` slots."""
comptime Q_ESTIMATE: Int = 5
"""A `number` with a `range`: one slot over the grid's values."""
comptime Q_BOX: Int = 6
comptime Q_BOXES: Int = 7

comptime SLOT_CODES: Int = 0
"""A slot read through the LM head over its groups of option tokens."""
comptime SLOT_START: Int = 1
comptime SLOT_END: Int = 2
"""A read's slots: a pointer over the state's positions, not the vocabulary."""

comptime BOS = "<|startoftext|>"
comptime IM_START = "<|im_start|>"
comptime IM_END = "<|im_end|>"
comptime LEAD = "<think></think>"
comptime HEADER = "QUESTION:\n"
comptime BREAK = "​"
comptime SET_SIZE: Int = 16
comptime INSTANCES: Int = 16
"""A `boxes` question's blocks without `max` (`box.INSTANCES`), the most prompt 1 takes."""
comptime MAX_INSTANCES: Int = 64
"""The most blocks a `max` asks of a prompt-2 checkpoint (`box.MAX_INSTANCES`)."""
comptime GRID_MAX: Int = 128
"""An estimate's grid is at most this many values (`Estimate.grid`)."""


@fieldwise_init
struct D1Record(Copyable, Movable):
    """What a checkpoint records in its `config.json` and `runner._serve`
    reads: the option codes' scheme, numbers as tokens, the prompt version,
    how a box's edges are read, the set size and the attention mode."""

    var codes: Int
    var numbers: Bool
    var prompt: Int
    var box_edges: String
    var set_size: Int
    var attention: String

    @staticmethod
    def d1_2b() -> D1Record:
        """d1-2B's record (`config.json` at a6bce774e6; prompt and edges defaulted)."""
        return D1Record(2, True, 1, String("median"), 16, String("set"))

    @staticmethod
    def load(config_path: String) raises -> D1Record:
        """`config.json`'s `d1_*` fields with `runner._serve`'s defaults (codes 1,
        no numbers, prompt 1, median edges, set size 4, causal). Only `set`
        attention is implemented."""
        var doc = load_json(config_path)
        var r = doc.root()
        var rec = D1Record(1, False, 1, String("median"), 4, String("causal"))
        if doc.field(r, "d1_codes") >= 0:
            rec.codes = doc.integer(doc.field(r, "d1_codes"))
        if doc.field(r, "d1_numbers") >= 0:
            rec.numbers = doc.boolean(doc.field(r, "d1_numbers"))
        if doc.field(r, "d1_prompt") >= 0:
            rec.prompt = doc.integer(doc.field(r, "d1_prompt"))
        if doc.field(r, "d1_box_edges") >= 0 and doc.kind_of(doc.field(r, "d1_box_edges")) == J_STRING:
            rec.box_edges = doc.string(doc.field(r, "d1_box_edges"))
        if doc.field(r, "d1_set_size") >= 0:
            rec.set_size = doc.integer(doc.field(r, "d1_set_size"))
        if doc.field(r, "d1_attention") >= 0:
            rec.attention = doc.string(doc.field(r, "d1_attention"))
        if rec.codes != 1 and rec.codes != 2:
            raise Error("d1: d1_codes " + String(rec.codes) + " is not 1 or 2")
        if rec.prompt != 1 and rec.prompt != 2:
            raise Error("d1: d1_prompt " + String(rec.prompt) + " is not 1 or 2")
        if rec.box_edges != "median" and rec.box_edges != "interp":
            raise Error("d1: d1_box_edges '" + rec.box_edges + "' is not median or interp")
        if rec.set_size < 2 or rec.set_size > 128:
            raise Error("d1: d1_set_size " + String(rec.set_size) + " is not 2 to 128")
        if rec.attention != "set":
            raise Error("d1: attention '" + rec.attention + "': only `set` is implemented")
        return rec^

    def most_blocks(self) -> Int:
        """The most blocks a `boxes` question's `max` asks (`prompt.most_blocks`)."""
        return INSTANCES if self.prompt == 1 else MAX_INSTANCES


@fieldwise_init
struct D1Grid(Copyable, Movable):
    """`grid.Grid`: `size` equally spaced values from `low` to `high`."""

    var low: Float64
    var high: Float64
    var size: Int

    def step(self) -> Float64:
        return (self.high - self.low) / Float64(self.size - 1)

    def values(self) -> List[Float64]:
        """`low + i * step` for all but the last, which is `high` itself."""
        var out = List[Float64](capacity=self.size)
        var st = self.step()
        for i in range(self.size - 1):
            out.append(self.low + Float64(i) * st)
        out.append(self.high)
        return out^


@fieldwise_init
struct D1Question(Copyable, Movable):
    var name: String
    var kind: Int
    var instructions: String
    var labels: List[String]
    """Choice / multiple: the option names (criteria keys, in order). Score: the levels."""
    var descs: List[String]
    """Choice / multiple: each option's description ("" for none). Noul: [true, false] when given."""
    var read_type: String
    """Q_READ: "text", "number" or "date"."""
    var unit: String
    var has_unit: Bool
    """A read's or an estimate's `unit` (None when not `has_unit`)."""
    var grid: D1Grid
    """Q_ESTIMATE: its grid over `range`."""
    var blocks: Int
    """Q_BOXES: the blocks read (`max`, else INSTANCES)."""

    @staticmethod
    def plain_kind(var name: String, kind: Int, var instructions: String, var labels: List[String],
                   var descs: List[String]) -> D1Question:
        return D1Question(name^, kind, instructions^, labels^, descs^, String(""), String(""), False,
                          D1Grid(0.0, 1.0, 2), 0)


def _py_round(x: Float64) -> Int:
    """Python's `round(x)`: half to even."""
    var f = floor(x)
    var d = x - f
    var n = Int(f)
    if d > 0.5 or (d == 0.5 and n % 2 != 0):
        n += 1
    return n


def _isclose(a: Float64, b: Float64, abs_tol: Float64 = 0.0) -> Bool:
    """`math.isclose` (rel_tol 1e-9)."""
    return abs(a - b) <= max(1e-9 * max(abs(a), abs(b)), abs_tol)


def parse_questions(ref doc: JsonDoc, node: Int, rec: D1Record = D1Record.d1_2b()) raises -> List[D1Question]:
    """`{name: {type, instructions, criteria, ...}}` — the System One API's (and
    `JevQuestions.write_to_json`'s) question map, checked as `check_question` and
    `as_question` check it. Spoken instructions (a media dict) are refused."""
    var out = List[D1Question]()
    for i in range(doc.size(node)):
        var name = doc.key_at(node, i)
        var q = doc.at(node, i)
        var tnode = doc.field(q, "type")
        var kind_s = doc.string(tnode) if tnode >= 0 else String("choice")
        var inode = doc.field(q, "instructions")
        if inode < 0 or doc.kind_of(inode) != J_STRING:
            raise Error("d1: question '" + name + "': \"instructions\" is the question as text (spoken ones are not implemented)")
        var instr = doc.string(inode)
        if instr.strip().byte_length() == 0:
            raise Error("d1: question '" + name + "': empty instructions")
        var crit = doc.field(q, "criteria")
        var mult = doc.field(q, "multiple")
        var multiple = mult >= 0 and doc.kind_of(mult) == J_BOOL and doc.boolean(mult)
        if multiple and kind_s != "choice":
            raise Error("d1: question '" + name + "': \"multiple\" belongs to a choice")
        var maxn = doc.field(q, "max")
        if maxn >= 0 and kind_s != "boxes":
            raise Error("d1: question '" + name + "': \"max\" belongs to boxes")
        var rng = doc.field(q, "range")
        var stp = doc.field(q, "step")
        if stp >= 0 and rng < 0:
            raise Error("d1: question '" + name + "': \"step\" needs a \"range\"")
        if rng >= 0 and kind_s != "number":
            raise Error("d1: question '" + name + "': an estimate is {\"type\": \"number\", \"range\": [low, high]}")
        var unode = doc.field(q, "unit")
        var has_unit = unode >= 0 and doc.kind_of(unode) == J_STRING
        var unit = doc.string(unode) if has_unit else String("")
        var labels = List[String]()
        var descs = List[String]()
        if kind_s == "noul":
            if crit >= 0 and doc.kind_of(crit) == J_OBJECT:
                descs.append(_text_or_none(doc, doc.field(crit, "true")))
                descs.append(_text_or_none(doc, doc.field(crit, "false")))
            elif crit >= 0 and doc.kind_of(crit) != J_NULL:
                raise Error("d1: noul '" + name + "': criteria is {\"true\": ..., \"false\": ...}")
            out.append(D1Question.plain_kind(name, Q_NOUL, instr, labels^, descs^))
        elif kind_s == "score":
            if crit < 0 or doc.kind_of(crit) != J_ARRAY:
                raise Error("d1: score '" + name + "': criteria lists 2 to 10 levels")
            for k in range(doc.size(crit)):
                labels.append(doc.string(doc.at(crit, k)))
            if len(labels) < 2 or len(labels) > 10:
                raise Error("d1: score '" + name + "' needs 2 to 10 levels")
            out.append(D1Question.plain_kind(name, Q_SCORE, instr, labels^, descs^))
        elif kind_s == "choice":
            if crit < 0 or doc.kind_of(crit) != J_OBJECT or doc.size(crit) == 0:
                raise Error("d1: choice '" + name + "': criteria names the options, {name: description}")
            for k in range(doc.size(crit)):
                labels.append(doc.key_at(crit, k))
                var d = doc.at(crit, k)
                descs.append(doc.string(d) if doc.kind_of(d) == J_STRING else String(""))
            out.append(D1Question.plain_kind(name, Q_MULTI if multiple else Q_CHOICE, instr, labels^, descs^))
        elif kind_s == "number" and rng >= 0:
            if doc.kind_of(rng) != J_ARRAY or doc.size(rng) != 2:
                raise Error("d1: estimate '" + name + "': range is [low, high]")
            var low = doc.number(doc.at(rng, 0))
            var high = doc.number(doc.at(rng, 1))
            var size = GRID_MAX
            if stp >= 0:
                var step = doc.number(stp)
                if not low < high or not step > 0 or not _isclose((high - low) / step, Float64(_py_round((high - low) / step))):
                    raise Error("d1: estimate '" + name + "': its range runs low to high and its step, above 0, divides it")
                size = min(GRID_MAX, _py_round((high - low) / step) + 1)
            elif not low < high:
                raise Error("d1: estimate '" + name + "': its range runs low to high")
            out.append(D1Question(name, Q_ESTIMATE, instr, labels^, descs^, String("number"), unit, has_unit,
                                  D1Grid(low, high, size), 0))
        elif kind_s == "text" or kind_s == "number" or kind_s == "date":
            out.append(D1Question(name, Q_READ, instr, labels^, descs^, kind_s, unit, has_unit, D1Grid(0.0, 1.0, 2), 0))
        elif kind_s == "box":
            out.append(D1Question.plain_kind(name, Q_BOX, instr, labels^, descs^))
        elif kind_s == "boxes":
            var blocks = INSTANCES
            if maxn >= 0:
                if doc.kind_of(maxn) != J_NUMBER:
                    raise Error("d1: boxes '" + name + "': max is a whole number")
                var mv = doc.number(maxn)
                blocks = Int(mv)
                if Float64(blocks) != mv or blocks < 1 or blocks > rec.most_blocks():
                    raise Error("d1: boxes '" + name + "': max is a whole number from 1 to " + String(rec.most_blocks()))
            var bq = D1Question.plain_kind(name, Q_BOXES, instr, labels^, descs^)
            bq.blocks = blocks
            out.append(bq^)
        else:
            raise Error("d1: question type '" + kind_s + "' is not one of noul, choice, score, text, number, date, box, boxes ('" + name + "')")
    return out^


def _text_or_none(ref doc: JsonDoc, node: Int) raises -> String:
    if node < 0 or doc.kind_of(node) == J_NULL:
        return String("None")  # Python's f"{None}"
    return doc.string(node)


# ── Python's json.dumps(value, ensure_ascii=False, indent=2) ─────────────

def _py_str(s: String) -> String:
    var out = String('"')
    for c in s.codepoints():
        var u = Int(c.to_u32())
        if u == 34:
            out += '\\"'
        elif u == 92:
            out += "\\\\"
        elif u == 10:
            out += "\\n"
        elif u == 13:
            out += "\\r"
        elif u == 9:
            out += "\\t"
        elif u == 8:
            out += "\\b"
        elif u == 12:
            out += "\\f"
        elif u < 32:
            var hexd = "0123456789abcdef"
            out += "\\u00" + String(hexd[byte=u >> 4]) + String(hexd[byte=u & 15])
        else:
            out += String(c)
    return out + '"'


def _py_num(v: Float64) -> String:
    if v == Float64(Int(v)) and abs(v) < 1e16:
        return String(Int(v))
    return String(v)


def py_json_dumps(ref doc: JsonDoc, node: Int, level: Int = 0) raises -> String:
    var k = doc.kind_of(node)
    if k == J_NULL:
        return String("null")
    if k == J_BOOL:
        return String("true") if doc.boolean(node) else String("false")
    if k == J_NUMBER:
        return _py_num(doc.number(node))
    if k == J_STRING:
        return _py_str(doc.string(node))
    var n = doc.size(node)
    var open = "[" if k == J_ARRAY else "{"
    var close = "]" if k == J_ARRAY else "}"
    if n == 0:
        return String(open) + close
    var pad = String(" ") * (2 * (level + 1))
    var out = String(open) + "\n"
    for i in range(n):
        out += pad
        if k == J_OBJECT:
            out += _py_str(doc.key_at(node, i)) + ": "
        out += py_json_dumps(doc, doc.at(node, i), level + 1)
        out += ",\n" if i + 1 < n else "\n"
    return out + String(" ") * (2 * level) + close


# ── plain ────────────────────────────────────────────────────────────────

def plain(ref tok: ByteLevelBPE, text: String) -> String:
    """Every special token's string in `text` broken by U+200B after its first
    character (d1's `prompt.plain`): leftmost, longest first, again until none."""
    var cur = text
    while True:
        var b = cur.as_bytes()
        var out = String("")
        var found = False
        var i = 0
        var start = 0
        while i < len(b):
            var hit = -1
            for k in range(len(tok.added)):
                if not tok.added_special[k] or tok.added[k].byte_length() < 2:
                    continue
                var a = tok.added[k].as_bytes()
                if i + len(a) > len(b):
                    continue
                var ok = True
                for j in range(len(a)):
                    if b[i + j] != a[j]:
                        ok = False
                        break
                if ok:
                    hit = k
                    break
            if hit >= 0:
                var a = tok.added[hit]
                var first = 1  # every special string here starts with an ASCII character
                out += String(cur[byte=start:i]) + String(a[byte=0:first]) + BREAK + String(a[byte=first:])
                i += a.byte_length()
                start = i
                found = True
            else:
                i += 1
        if not found:
            return cur
        out += String(cur[byte=start:])
        cur = out


# ── codes and numbers ─────────────────────────────────────────────────────

def option_codes(labels: List[String], scheme: Int = 2) -> List[String]:
    """Native letters when every label already is one, else A..Z, else the
    `scheme`'s codes: 00, 01, .. (1) or A..Z then AA, AB, .. (2)."""
    var all_letters = len(labels) > 0
    for l in labels:
        var s = String(l.strip())
        if s.count_codepoints() != 1:
            all_letters = False
            break
        for c in s.codepoints():
            if not is_letter(c.to_u32()):
                all_letters = False
    var out = List[String]()
    if all_letters:
        for l in labels:
            out.append(String(l.strip()))
        return out^
    var letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(labels) > 26 and scheme != 2:
        for i in range(len(labels)):
            out.append(String("0") + String(i) if i < 10 else String(i))
        return out^
    for i in range(len(labels)):
        if i < 26:
            out.append(String(letters[byte=i]))
        else:
            var j = i - 26
            out.append(String(letters[byte=j // 26]) + String(letters[byte=j % 26]))
    return out^


def _single(mut tok: ByteLevelBPE, text: String) raises -> Int:
    var ids = tok.encode(text)
    return ids[0] if len(ids) == 1 else -1


def aliases(mut tok: ByteLevelBPE, labels: List[String], scheme: Int = 2) raises -> Tuple[List[String], List[Int]]:
    """Each option's distinct single-token code and its id (`prompt.aliases`):
    its own code when that is one unused token, else the first free one of the
    fallback pool."""
    var codes = option_codes(labels, scheme)
    var pool = List[String]()
    var letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    var lower = "abcdefghijklmnopqrstuvwxyz"
    for i in range(26):
        pool.append(String(letters[byte=i]))
    for i in range(100):
        pool.append(String("0") + String(i) if i < 10 else String(i))
    for i in range(26):
        pool.append(String(lower[byte=i]))
    for i in range(200):
        pool.append(String("#") + String(i))
    for a in range(26):
        for b in range(26):
            pool.append(String(letters[byte=a]) + String(letters[byte=b]))
    var used = List[Int]()
    var out_c = List[String]()
    var out_i = List[Int]()
    for code in codes:
        var id = _single(tok, code)
        if id >= 0 and id not in used:
            out_c.append(code)
            out_i.append(id)
            used.append(id)
            continue
        var taken = False
        for raw in pool:
            var rid = _single(tok, raw)
            if rid >= 0 and rid not in used:
                out_c.append(raw)
                out_i.append(rid)
                used.append(rid)
                taken = True
                break
        if not taken:
            raise Error("d1: no single-token alias left for " + String(len(codes)) + " options")
    return (out_c^, out_i^)


def py_g(v: Float64) -> String:
    """Python's `f"{v:g}"`: six significant digits, trailing zeros dropped,
    exponent form below 1e-4 and from 1e6. ⚠ The digits come from a scaled
    float: a value whose seventh digit is an exact tie in decimal may round
    unlike CPython's correctly rounded formatting (not met by d1's grids)."""
    if v == 0.0:
        return String("0")
    var neg = v < 0.0
    var a = abs(v)
    var e = Int(floor(log10(a)))
    var m = 0
    for _ in range(3):
        var k = e - 5
        var scaled = a / (10.0 ** Float64(k)) if k >= 0 else a * (10.0 ** Float64(-k))
        m = _py_round(scaled)
        if m >= 1000000:
            e += 1
        elif m < 100000:
            e -= 1
        else:
            break
    var digits = String(m)
    var out: String
    if e >= -4 and e < 6:
        var ip: String
        var fp: String
        if e >= 0:
            ip = String(digits[byte=0:e + 1])
            fp = String(digits[byte=e + 1:])
        else:
            ip = String("0")
            fp = String("0") * (-e - 1) + digits
        var n = fp.byte_length()
        while n > 0 and fp.as_bytes()[n - 1] == 48:
            n -= 1
        out = ip if n == 0 else ip + "." + String(fp[byte=0:n])
    else:
        var fp = String(digits[byte=1:])
        var n = fp.byte_length()
        while n > 0 and fp.as_bytes()[n - 1] == 48:
            n -= 1
        var mant = String(digits[byte=0:1]) + ("" if n == 0 else "." + String(fp[byte=0:n]))
        var ae = abs(e)
        out = mant + "e" + ("+" if e >= 0 else "-") + ("0" if ae < 10 else "") + String(ae)
    return String("-") + out if neg else out^


def _ids(mut tok: ByteLevelBPE, texts: List[String]) raises -> List[Int]:
    """The single-token encodings among `texts`, without repeats (`prompt._ids`)."""
    var out = List[Int]()
    for t in texts:
        var id = _single(tok, t)
        if id >= 0 and id not in out:
            out.append(id)
    return out^


def number_groups(mut tok: ByteLevelBPE, g: D1Grid, scale: Int) raises -> List[List[Int]]:
    """`prompt.number_ids`: each grid value times `scale` as the tokenizer
    writes it, plain and after a space, one token a form; EMPTY (Python's None)
    when a value is no integer or takes more than one token."""
    var out = List[List[Int]]()
    for value in g.values():
        var x = value * Float64(scale)
        var n = _py_round(x)
        if not _isclose(Float64(n), x, 1e-6):
            return List[List[Int]]()
        var group = _ids(tok, [String(n), String(" ") + String(n)])
        if len(group) == 0 or len(tok.encode(String(n))) != 1:
            return List[List[Int]]()
        out.append(group^)
    return out^


# ── slots ─────────────────────────────────────────────────────────────────

@fieldwise_init
struct D1Slot(Copyable, Movable):
    """One answer position's readout (`prompt.question_slots`)."""

    var kind: Int
    """SLOT_CODES, SLOT_START or SLOT_END."""
    var groups: List[List[Int]]
    """SLOT_CODES: per option, the token ids whose best log-probability is its score."""
    var number: Bool
    """A grid read as the tokenizer's numbers (`Number`), not as codes."""


comptime YES_FORMS: List[String] = ["yes", "Yes", "YES"]
comptime NO_FORMS: List[String] = ["no", "No", "NO"]


def _noul_slot(mut tok: ByteLevelBPE) raises -> D1Slot:
    var yes = _ids(tok, materialize[YES_FORMS]())
    var no = _ids(tok, materialize[NO_FORMS]())
    if len(yes) == 0 or len(no) == 0:
        raise Error("d1: tokenizer has no single-token yes/no")
    var groups = List[List[Int]]()
    groups.append(yes^)
    groups.append(no^)
    return D1Slot(SLOT_CODES, groups^, False)


def _grid_labels(g: D1Grid) -> List[String]:
    """`grid_question`'s option keys: "0" .. "size-1"."""
    var out = List[String]()
    for i in range(g.size):
        out.append(String(i))
    return out^


def grid_slot(mut tok: ByteLevelBPE, g: D1Grid, scale: Int, rec: D1Record) raises -> D1Slot:
    """`prompt.grid_slot`: the tokenizer's numbers where the checkpoint reads them
    so and every value is one token, else the grid's codes — digits for up to
    10 values (`Score`), option codes above (`Choice`)."""
    if rec.numbers:
        var ng = number_groups(tok, g, scale)
        if len(ng) > 0:
            return D1Slot(SLOT_CODES, ng^, True)
    var groups = List[List[Int]]()
    if g.size <= 10:
        for i in range(g.size):
            var d = _ids(tok, [String(i)])
            if len(d) == 0:
                raise Error("d1: a grid of " + String(g.size) + " values needs single-token digits")
            groups.append(d^)
        return D1Slot(SLOT_CODES, groups^, False)
    var al = aliases(tok, _grid_labels(g), rec.codes)
    for i in range(g.size):
        var gr: List[Int] = [al[1][i]]
        for e in _ids(tok, [String(" ") + al[0][i]]):
            if e != al[1][i]:
                gr.append(e)
        groups.append(gr^)
    return D1Slot(SLOT_CODES, groups^, False)


def edge_grid(mut tok: ByteLevelBPE, rec: D1Record) raises -> Tuple[D1Grid, Int]:
    """`box.edge_grid`: the thousandths 0-999 as numbers where the checkpoint
    reads numbers and the tokenizer writes each in one token, else 128 codes."""
    var th = D1Grid(0.0, 0.999, 1000)
    if rec.numbers and len(number_groups(tok, th, 1000)) > 0:
        return (th^, 1000)
    return (D1Grid(0.0, 1.0, 128), 1)


def question_slots(mut tok: ByteLevelBPE, q: D1Question, rec: D1Record = D1Record.d1_2b()) raises -> List[D1Slot]:
    """Each answer position's readout, in the question's order."""
    var out = List[D1Slot]()
    if q.kind == Q_READ:
        out.append(D1Slot(SLOT_START, List[List[Int]](), False))
        out.append(D1Slot(SLOT_END, List[List[Int]](), False))
    elif q.kind == Q_ESTIMATE:
        out.append(grid_slot(tok, q.grid, 1, rec))
    elif q.kind == Q_BOX or q.kind == Q_BOXES:
        var eg = edge_grid(tok, rec)
        var edge = grid_slot(tok, eg[0], eg[1], rec)
        var yes = _noul_slot(tok)
        var blocks = 1 if q.kind == Q_BOX else q.blocks
        for _ in range(blocks):
            out.append(yes.copy())
            for _ in range(4):
                out.append(edge.copy())
    elif q.kind == Q_MULTI:
        var yes = _noul_slot(tok)
        for _ in range(len(q.labels)):
            out.append(yes.copy())
    else:
        out.append(D1Slot(SLOT_CODES, readout_ids(tok, q, rec), False))
    return out^


# ── the question blocks ──────────────────────────────────────────────────

def _edge_legend(mut tok: ByteLevelBPE, rec: D1Record) raises -> String:
    var eg = edge_grid(tok, rec)
    if eg[1] > 1:
        return String("Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top "
                      "edge, left and right from its left edge.")
    var codes = aliases(tok, _grid_labels(eg[0]), rec.codes)[0].copy()
    return (String("Each edge is a code on a ") + String(eg[0].size) + "-value scale: " + codes[0]
            + " is 0 (top or left), " + codes[len(codes) - 1]
            + " is 1 (bottom or right). Values are equally spaced.")


def question_block(mut tok: ByteLevelBPE, q: D1Question, rec: D1Record = D1Record.d1_2b()) raises -> String:
    var said = q.instructions
    if q.kind == Q_NOUL:
        var extra = String("")
        if len(q.descs) == 2:
            extra = String("\nYes: ") + q.descs[0] + "\nNo: " + q.descs[1]
        return said + extra + "\n\nReply with yes or no only."
    if q.kind == Q_SCORE:
        var legend = String("")
        for i in range(len(q.labels)):
            legend += String(i) + " " + q.labels[i] + ("\n" if i + 1 < len(q.labels) else "")
        return (said + "\n\n" + legend + "\n\nReply with a single digit 0-"
                + String(len(q.labels) - 1) + " only.")
    if q.kind == Q_MULTI:
        var lines = String("")
        for i in range(len(q.labels)):
            var text = q.descs[i] if q.descs[i].byte_length() > 0 else q.labels[i]
            lines += _py_str(q.labels[i]) + ": " + text + ("\n" if i + 1 < len(q.labels) else "")
        return said + "\n\nOptions:\n" + lines + "\n\nReply with yes or no for each option."
    if q.kind == Q_READ:
        var unit = String("")
        if q.read_type == "number" and q.has_unit and q.unit.byte_length() > 0:
            unit = String(" (unit: ") + q.unit + ")"
        return said + "\n\nRead the " + q.read_type + unit + " from the state, or nothing if it is absent."
    if q.kind == Q_ESTIMATE:
        var unit = String(" ") + q.unit if q.has_unit else String("")
        var slot = grid_slot(tok, q.grid, 1, rec)
        if slot.number:
            return (said + "\n\nEstimate from " + py_g(q.grid.low) + " to " + py_g(q.grid.high) + unit
                    + ", in steps of " + py_g(q.grid.step()) + ". Reply with the number only.")
        var codes = List[String]()
        if q.grid.size <= 10:
            for i in range(q.grid.size):
                codes.append(String(i))
        else:
            codes = aliases(tok, _grid_labels(q.grid), rec.codes)[0].copy()
        return (said + "\n\nEstimate from " + py_g(q.grid.low) + " to " + py_g(q.grid.high) + unit + " on "
                + String(q.grid.size) + " equally spaced values, step " + py_g(q.grid.step()) + ": " + codes[0]
                + " is " + py_g(q.grid.low) + ", " + codes[len(codes) - 1] + " is " + py_g(q.grid.high)
                + ". Reply with the code only.")
    if q.kind == Q_BOX:
        var eg = edge_grid(tok, rec)
        var one = String("one number") if eg[1] > 1 else String("one code")
        return (said + "\n\nPresence is yes or no. " + _edge_legend(tok, rec) + " Reply with yes or no at present, then "
                + one + " at top, bottom, left and right.")
    if q.kind == Q_BOXES:
        var most = String(INSTANCES) + " at most, " if rec.prompt == 1 else String("")
        return (said + "\n\nEvery one of them, " + most + "in reading order (left edge, then top). Present k is yes "
                + "when there are at least k. " + _edge_legend(tok, rec) + " Reply with yes or no at each present, "
                + "then its top, bottom, left and right.")
    var al = aliases(tok, q.labels, rec.codes)
    var lines = String("")
    for i in range(len(q.labels)):
        var text = q.descs[i] if q.descs[i].byte_length() > 0 else q.labels[i].replace("_", " ")
        lines += al[0][i] + " " + text + ("\n" if i + 1 < len(q.labels) else "")
    return said + "\n\nOptions:\n" + lines + "\n\nReply with the option code only."


comptime EDGES: List[String] = ["top", "bottom", "left", "right"]


def marked_suffix(mut tok: ByteLevelBPE, body: String, names: List[String], after: String = "") raises -> Tuple[String, List[Int]]:
    """`prompt.marked_suffix`: the question, the assistant header and one
    `name:` marker a line; each marker's slot is the token holding its colon,
    as an offset from the suffix's last token. `after`: whitespace appended
    when the tokenizer writes the last colon and it as one token."""
    if len(names) == 0:
        raise Error("d1: slot names must be distinct nonempty single lines")
    for i in range(len(names)):
        if names[i].byte_length() == 0 or "\n" in names[i]:
            raise Error("d1: slot names must be distinct nonempty single lines")
        for j in range(i):
            if names[j] == names[i]:
                raise Error("d1: slot names must be distinct nonempty single lines")
    var text = plain(tok, body) + IM_END + "\n" + IM_START + "assistant\n" + LEAD
    var ends = List[Int]()
    for i in range(len(names)):
        text += ("\n" if i > 0 else "") + plain(tok, names[i]) + ":"
        ends.append(cp_len(text))
    if after.byte_length() > 0:
        var t2 = tok.encode(text + after)
        if py_rstrip(tok.decode([t2[len(t2) - 1]])).endswith(":"):
            text += after
    var enc = tok.encode_offsets(text)
    ref ids = enc[0]
    var cps = cps_of(text)
    var slots = List[Int]()
    for chars in ends:
        var at = -1
        var count = 0
        for i in range(len(ids)):
            if enc[1][i] <= chars - 1 and chars - 1 < enc[2][i]:
                at = i
                count += 1
        if count != 1:
            raise Error("d1: a slot marker must have an exact colon token position")
        if not blank(cps, chars, enc[2][at]) or not py_rstrip(tok.decode([ids[at]])).endswith(":"):
            raise Error("d1: a slot marker token must end with its colon and optional whitespace")
        slots.append(at - len(ids) + 1)
    return (text^, slots^)


def suffix_layout(mut tok: ByteLevelBPE, q: D1Question, rec: D1Record = D1Record.d1_2b()) raises -> Tuple[String, List[Int]]:
    """The question's suffix and each slot's token offset from its last token
    (`prompt.suffix_layout`); an untyped question has one slot, at 0."""
    var body = question_block(tok, q, rec)
    if q.kind == Q_READ:
        return marked_suffix(tok, body, ["start", "end"])
    if q.kind == Q_BOX:
        var names: List[String] = ["present"]
        for e in materialize[EDGES]():
            names.append(e)
        return marked_suffix(tok, body, names)
    if q.kind == Q_BOXES:
        var names = List[String]()
        var edges = materialize[EDGES]()
        for k in range(1, q.blocks + 1):
            names.append(String("present ") + String(k))
            for e in edges:
                names.append(e + " " + String(k))
        var after = String("\n") if q.blocks < INSTANCES or rec.prompt > 1 else String("")
        return marked_suffix(tok, body, names, after)
    if q.kind == Q_MULTI:
        var names = List[String]()
        for l in q.labels:
            names.append(_py_str(l))
        return marked_suffix(tok, body, names)
    return (plain(tok, body) + IM_END + "\n" + IM_START + "assistant\n" + LEAD, [0])


def suffix_text(mut tok: ByteLevelBPE, q: D1Question, rec: D1Record = D1Record.d1_2b()) raises -> String:
    return suffix_layout(tok, q, rec)[0].copy()


def prefix_text(ref tok: ByteLevelBPE, state_block: String, media: String = "") -> String:
    """`state_block`: the state as text, or Python's indented JSON — WITHOUT the
    blank line (added here). `media`: the pictures' markup (special tokens kept:
    it is the processor's, not the caller's text), at the head of the turn."""
    return String(BOS) + IM_START + "user\n" + media + plain(tok, state_block + "\n\n") + "\n" + HEADER


# ── packing and readout ids ──────────────────────────────────────────────

@fieldwise_init
struct D1Row(Copyable, Movable):
    var ids: List[Int]
    var groups: List[Int]
    """-1 on the state, k (1..) on question k of the row."""
    var state_len: Int
    var slots: List[Int]
    """Every answer position of the row, question by question, each question's in its order."""
    var first: Int
    """The row's first question, an index into the request's questions."""
    var slot_q: List[Int]
    """Each slot's question, an index into the request's questions."""


def pack_rows(
    mut tok: ByteLevelBPE, prefix: String, suffixes: List[String], set_size: Int = SET_SIZE,
    offsets: List[List[Int]] = List[List[Int]](),
) raises -> List[D1Row]:
    """`attention.pack_set`, `set_size` questions a row after one copy of the
    state; `offsets`: each question's slots from its last token (`suffix_layout`),
    one slot at 0 each when empty."""
    if len(offsets) > 0 and len(offsets) != len(suffixes):
        raise Error("d1: one slot layout is required per question")
    var rows = List[D1Row]()
    var k0 = 0
    while k0 < len(suffixes):
        var ids = tok.encode(prefix + suffixes[k0])
        var state_len = len(ids) - len(tok.encode(suffixes[k0]))
        var groups = List[Int]()
        for i in range(len(ids)):
            groups.append(-1 if i < state_len else 1)
        var slots = List[Int]()
        var slot_q = List[Int]()
        _place(offsets, k0, len(ids) - state_len, len(ids) - 1, slots, slot_q)
        var g = 2
        for k in range(k0 + 1, min(k0 + set_size, len(suffixes))):
            var own = tok.encode(String(HEADER) + suffixes[k])
            for id in own:
                ids.append(id)
                groups.append(g)
            _place(offsets, k, len(own), len(ids) - 1, slots, slot_q)
            g += 1
        rows.append(D1Row(ids^, groups^, state_len, slots^, k0, slot_q^))
        k0 += set_size
    return rows^


def _place(offsets: List[List[Int]], k: Int, length: Int, end: Int, mut slots: List[Int], mut slot_q: List[Int]) raises:
    """Question `k`'s slots, ending at `end` (`pack_set`'s `positions`)."""
    if len(offsets) == 0:
        slots.append(end)
        slot_q.append(k)
        return
    ref own = offsets[k]
    if len(own) == 0 or own[len(own) - 1] != 0:
        raise Error("d1: slots must be increasing positions inside their question, ending at its last token")
    for i in range(len(own)):
        if own[i] <= -length or own[i] > 0 or (i > 0 and own[i] <= own[i - 1]):
            raise Error("d1: slots must be increasing positions inside their question, ending at its last token")
        slots.append(end + own[i])
        slot_q.append(k)


def readout_ids(mut tok: ByteLevelBPE, q: D1Question, rec: D1Record = D1Record.d1_2b()) raises -> List[List[Int]]:
    """Per option, the token ids whose best log-probability is its score (an
    untyped question's one slot; `question_slots` gives every slot's)."""
    var groups = List[List[Int]]()
    if q.kind == Q_NOUL:
        return _noul_slot(tok).groups.copy()
    if q.kind == Q_SCORE:
        for i in range(len(q.labels)):
            var d = _ids(tok, [String(i)])
            if len(d) == 0:
                raise Error("d1: a score with " + String(len(q.labels)) + " levels needs single-token digits")
            groups.append(d^)
        return groups^
    if q.kind != Q_CHOICE:
        raise Error("d1: readout_ids reads an untyped question's one slot; '" + q.name + "' has its own slots")
    var al = aliases(tok, q.labels, rec.codes)
    for i in range(len(q.labels)):
        var g: List[Int] = [al[1][i]]
        for e in _ids(tok, [String(" ") + al[0][i]]):
            if e != al[1][i]:
                g.append(e)
        groups.append(g^)
    return groups^
