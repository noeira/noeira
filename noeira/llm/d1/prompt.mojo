# +--------------------------------------------------------------------------+ #
# | d1 — System One's prompts, packing and readout ids (no model)
# +--------------------------------------------------------------------------+ #
"""d1's `prompt.py` / `attention.pack_set` / `prompt.readout_ids`, for the three
UNTYPED questions a robot's decisions use — yes/no (`noul`), `choice`, `score`
— transcribed from d1-2B's own code (docs/NN_MODEL_PORTS_PLAN.md §12.2).
Typed reads (text, number, date, box) and `multiple` choices are refused.

    prefix  = "<|startoftext|><|im_start|>user\\n" + plain(state block) + "\\n" + "QUESTION:\\n"
    suffix  = plain(question block) + "<|im_end|>\\n<|im_start|>assistant\\n<think></think>"
    row     = enc(prefix + suffix_1) + enc("QUESTION:\\n" + suffix_k) ...   (16 questions a row)

The state block: a string as it is, any other JSON value as Python's
`json.dumps(value, ensure_ascii=False, indent=2)`; then a blank line.
⚠ `JsonDoc` keeps numbers as Float64, so an integral float (`3.0`) is written
`3` where Python keeps `3.0`; send such numbers as text if it matters.

`plain` breaks every SPECIAL token's string found in caller text with U+200B
after its first character, so a state cannot open or close a turn.
"""

from noeira.io.json import JsonDoc, J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT
from noeira.text.bpe import ByteLevelBPE
from noeira.text.unicode_classes import is_letter


comptime Q_NOUL: Int = 0
comptime Q_CHOICE: Int = 1
comptime Q_SCORE: Int = 2

comptime BOS = "<|startoftext|>"
comptime IM_START = "<|im_start|>"
comptime IM_END = "<|im_end|>"
comptime LEAD = "<think></think>"
comptime HEADER = "QUESTION:\n"
comptime BREAK = "​"
comptime SET_SIZE: Int = 16


@fieldwise_init
struct D1Question(Copyable, Movable):
    var name: String
    var kind: Int
    var instructions: String
    var labels: List[String]
    """Choice: the option names (criteria keys, in order). Score: the levels."""
    var descs: List[String]
    """Choice: each option's description ("" for none). Noul: [true, false] when given."""


def parse_questions(ref doc: JsonDoc, node: Int) raises -> List[D1Question]:
    """`{name: {type, instructions, criteria}}` — the System One API's (and
    `JevQuestions.write_to_json`'s) question map."""
    var out = List[D1Question]()
    for i in range(doc.size(node)):
        var name = doc.key_at(node, i)
        var q = doc.at(node, i)
        var tnode = doc.field(q, "type")
        var kind_s = doc.string(tnode) if tnode >= 0 else String("choice")
        var instr = doc.string(doc.field(q, "instructions"))
        var crit = doc.field(q, "criteria")
        var labels = List[String]()
        var descs = List[String]()
        var kind: Int
        if kind_s == "noul":
            kind = Q_NOUL
            if crit >= 0 and doc.kind_of(crit) == J_OBJECT:
                descs.append(_text_or_none(doc, doc.field(crit, "true")))
                descs.append(_text_or_none(doc, doc.field(crit, "false")))
        elif kind_s == "score":
            kind = Q_SCORE
            for k in range(doc.size(crit)):
                labels.append(doc.string(doc.at(crit, k)))
            if len(labels) < 2 or len(labels) > 10:
                raise Error("d1: score '" + name + "' needs 2 to 10 levels")
        elif kind_s == "choice":
            kind = Q_CHOICE
            var mult = doc.field(q, "multiple")
            if mult >= 0 and doc.boolean(mult):
                raise Error("d1: a multiple choice is not implemented ('" + name + "')")
            if crit < 0 or doc.kind_of(crit) != J_OBJECT:
                raise Error("d1: choice '" + name + "': criteria names the options, {name: description}")
            for k in range(doc.size(crit)):
                labels.append(doc.key_at(crit, k))
                var d = doc.at(crit, k)
                descs.append(doc.string(d) if doc.kind_of(d) == J_STRING else String(""))
        else:
            raise Error("d1: question type '" + kind_s + "' is not implemented ('" + name + "')")
        out.append(D1Question(name, kind, instr, labels^, descs^))
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


# ── the question blocks ──────────────────────────────────────────────────

def option_codes(labels: List[String]) -> List[String]:
    """d1_codes = 2: the labels themselves when each is one letter, else A..Z,
    then AA, AB, ..."""
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


def aliases(mut tok: ByteLevelBPE, labels: List[String]) raises -> Tuple[List[String], List[Int]]:
    """Each option's distinct single-token code and its id (`prompt.aliases`):
    its own code when that is one unused token, else the first free one of the
    fallback pool."""
    var codes = option_codes(labels)
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


def question_block(mut tok: ByteLevelBPE, q: D1Question) raises -> String:
    if q.kind == Q_NOUL:
        var extra = String("")
        if len(q.descs) == 2:
            extra = String("\nYes: ") + q.descs[0] + "\nNo: " + q.descs[1]
        return q.instructions + extra + "\n\nReply with yes or no only."
    if q.kind == Q_SCORE:
        var legend = String("")
        for i in range(len(q.labels)):
            legend += String(i) + " " + q.labels[i] + ("\n" if i + 1 < len(q.labels) else "")
        return (q.instructions + "\n\n" + legend + "\n\nReply with a single digit 0-"
                + String(len(q.labels) - 1) + " only.")
    var al = aliases(tok, q.labels)
    var lines = String("")
    for i in range(len(q.labels)):
        var text = q.descs[i] if q.descs[i].byte_length() > 0 else q.labels[i].replace("_", " ")
        lines += al[0][i] + " " + text + ("\n" if i + 1 < len(q.labels) else "")
    return q.instructions + "\n\nOptions:\n" + lines + "\n\nReply with the option code only."


def suffix_text(mut tok: ByteLevelBPE, q: D1Question) raises -> String:
    return plain(tok, question_block(tok, q)) + IM_END + "\n" + IM_START + "assistant\n" + LEAD


def prefix_text(ref tok: ByteLevelBPE, state_block: String) -> String:
    """`state_block`: the state as text, or Python's indented JSON — WITHOUT the
    blank line (added here)."""
    return String(BOS) + IM_START + "user\n" + plain(tok, state_block + "\n\n") + "\n" + HEADER


# ── packing and readout ids ──────────────────────────────────────────────

@fieldwise_init
struct D1Row(Copyable, Movable):
    var ids: List[Int]
    var groups: List[Int]
    """-1 on the state, k (1..) on question k of the row."""
    var state_len: Int
    var slots: List[Int]
    var first: Int
    """The row's first question, an index into the request's questions."""


def pack_rows(mut tok: ByteLevelBPE, prefix: String, suffixes: List[String], set_size: Int = SET_SIZE) raises -> List[D1Row]:
    """`attention.pack_set`, `set_size` questions a row after one copy of the state."""
    var rows = List[D1Row]()
    var k0 = 0
    while k0 < len(suffixes):
        var ids = tok.encode(prefix + suffixes[k0])
        var state_len = len(ids) - len(tok.encode(suffixes[k0]))
        var groups = List[Int]()
        for i in range(len(ids)):
            groups.append(-1 if i < state_len else 1)
        var slots: List[Int] = [len(ids) - 1]
        var g = 2
        for k in range(k0 + 1, min(k0 + set_size, len(suffixes))):
            for id in tok.encode(String(HEADER) + suffixes[k]):
                ids.append(id)
                groups.append(g)
            slots.append(len(ids) - 1)
            g += 1
        rows.append(D1Row(ids^, groups^, state_len, slots^, k0))
        k0 += set_size
    return rows^


def _ids(mut tok: ByteLevelBPE, texts: List[String]) raises -> List[Int]:
    """The single-token encodings among `texts`, without repeats (`prompt._ids`)."""
    var out = List[Int]()
    for t in texts:
        var id = _single(tok, t)
        if id >= 0 and id not in out:
            out.append(id)
    return out^


def readout_ids(mut tok: ByteLevelBPE, q: D1Question) raises -> List[List[Int]]:
    """Per option, the token ids whose best log-probability is its score."""
    var groups = List[List[Int]]()
    if q.kind == Q_NOUL:
        groups.append(_ids(tok, ["yes", "Yes", "YES"]))
        groups.append(_ids(tok, ["no", "No", "NO"]))
        return groups^
    if q.kind == Q_SCORE:
        for i in range(len(q.labels)):
            groups.append(_ids(tok, [String(i)]))
        return groups^
    var al = aliases(tok, q.labels)
    for i in range(len(q.labels)):
        var g: List[Int] = [al[1][i]]
        for e in _ids(tok, [String(" ") + al[0][i]]):
            if e != al[1][i]:
                g.append(e)
        groups.append(g^)
    return groups^
