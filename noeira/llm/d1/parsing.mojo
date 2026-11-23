# +--------------------------------------------------------------------------+ #
# | d1 — strict readings of written values: d1's parsing.py
# +--------------------------------------------------------------------------+ #
"""d1-2B's `parsing.py`, transcribed with its regular expressions VERBATIM
(`noeira.text.pyre`; docs/NN_MODEL_PORTS_PLAN.md §12.9, T6). A read's text
becomes a value here: a `number` read's magnitude, a `date` read's ISO date.
No arithmetic, unit conversion or invented date component, as d1 has it.

- `written_unit(text)`: an explicitly written percent or currency marker.
- `parse_number(text, context, unit)`: the written magnitude, its currency
  and percent sign set apart and its scale word applied; the decimal mark from
  the first evidence that speaks (the span's currency, the document's other
  numbers, its currencies, a declared locale). None (`ParsedNumber.none`) when
  the writing does not settle one value.
- `parse_date(text, context)`: ISO 8601 and an "unresolved ambiguity" flag;
  numeric dates ordered by a declared format, a locale or the document's other
  dates; month names and weekdays in a dozen languages, accents folded; CJK
  dates; a time kept only when written.

The module-level patterns are built as Python builds them (the same tables,
the same sort by length, `re.escape`), and the gate compares the pattern
strings themselves before the parses (`tests/llm/test_d1_parsing_reference.mojo`).
`int(value)` / `float(value)` of the exact decimal: an integer within Int64,
else a float — correctly rounded when the mantissa is below 2^53 and the
exponent within 22 (one exact division), else through `Float64(String)`.
"""

from std.math import isfinite

from noeira.text.pyre import PyRe, PyMatch, py_isdecimal, decimal_value, py_lower, py_isspace_cp
from noeira.text.pyunicode_tables import (
    DIGIT_N, DIGIT_LO, DIGIT_HI, FOLD_N, FOLD_FROM, FOLD_OFF, FOLD_OUT,
)


# ── code-point strings ────────────────────────────────────────────────────

comptime Cps = List[UInt32]


def cps(s: String) -> Cps:
    var out = Cps()
    for c in s.codepoints():
        out.append(c.to_u32())
    return out^


def text_of(t: Cps, a: Int = 0, b: Int = -1) -> String:
    var e = len(t) if b < 0 else min(b, len(t))
    var out = String("")
    for i in range(max(a, 0), e):
        out += String(Codepoint(unsafe_unchecked_codepoint=t[i]))
    return out^


def _slice(t: Cps, a: Int, b: Int) -> Cps:
    var out = Cps()
    for i in range(max(a, 0), min(b, len(t))):
        out.append(t[i])
    return out^


def _strip(t: Cps) -> Cps:
    var a = 0
    var b = len(t)
    while a < b and py_isspace_cp(t[a]):
        a += 1
    while b > a and py_isspace_cp(t[b - 1]):
        b -= 1
    return _slice(t, a, b)


def _rstrip(t: Cps) -> Cps:
    var b = len(t)
    while b > 0 and py_isspace_cp(t[b - 1]):
        b -= 1
    return _slice(t, 0, b)


def _cat(a: Cps, b: Cps) -> Cps:
    var out = a.copy()
    for x in b:
        out.append(x)
    return out^


def _lower(t: Cps) -> Cps:
    """`str.lower()` / `str.casefold()` for d1's keys (one character to one)."""
    var out = Cps()
    for x in t:
        out.append(py_lower(x))
    return out^


def _isdigit(cp: UInt32) -> Bool:
    if cp >= 48 and cp <= 57:
        return True
    if cp < 0x80:
        return False
    var lo = materialize[DIGIT_LO]()
    var hi = materialize[DIGIT_HI]()
    for i in range(DIGIT_N):
        if cp >= lo[i] and cp <= hi[i]:
            return True
    return False


def _all_digits(t: Cps) -> Bool:
    """`str.isdigit()`."""
    if len(t) == 0:
        return False
    for x in t:
        if not _isdigit(x):
            return False
    return True


def _int(t: Cps) raises -> Int:
    """`int(s)` for decimal digits (any script), raising as Python does otherwise."""
    var s = _strip(t)
    if len(s) == 0:
        raise Error("ValueError")
    var v = 0
    for x in s:
        var d = decimal_value(x)
        if d < 0:
            raise Error("ValueError")
        v = v * 10 + d
    return v


def fold(t: Cps) -> Cps:
    """d1's `_fold`: `casefold()`, NFKD, combining marks dropped — per character
    (`pyunicode_tables.FOLD_*`)."""
    var fr = materialize[FOLD_FROM]()
    var off = materialize[FOLD_OFF]()
    var outt = materialize[FOLD_OUT]()
    var out = Cps()
    for x in t:
        if x < 0x80:
            out.append(x + 32 if x >= 65 and x <= 90 else x)
            continue
        if x >= 0xAC00 and x <= 0xD7A3:  # a Hangul syllable: L V (T), Unicode's arithmetic
            var si = Int(x) - 0xAC00
            out.append(UInt32(0x1100 + si // 588))
            out.append(UInt32(0x1161 + (si % 588) // 28))
            if si % 28 != 0:
                out.append(UInt32(0x11A7 + si % 28))
            continue
        var a = 0
        var b = FOLD_N
        while a < b:
            var m = (a + b) // 2
            if fr[m] < x:
                a = m + 1
            else:
                b = m
        if a < FOLD_N and fr[a] == x:
            for k in range(Int(off[a]), Int(off[a + 1])):
                out.append(outt[k])
        else:
            out.append(x)
    return out^


def _split_ws(t: Cps) -> List[Cps]:
    """`str.split()`."""
    var out = List[Cps]()
    var cur = Cps()
    for x in t:
        if py_isspace_cp(x):
            if len(cur) > 0:
                out.append(cur^)
                cur = Cps()
        else:
            cur.append(x)
    if len(cur) > 0:
        out.append(cur^)
    return out^


def _re_escape(s: String) -> String:
    """`re.escape` (Python 3.7+: the special characters only)."""
    var special: List[Int] = [40, 41, 91, 93, 123, 125, 63, 42, 43, 45, 124, 94, 36, 92, 46, 38, 126, 35, 32, 9, 10,
                              13, 11, 12]
    var out = String("")
    for c in s.codepoints():
        if Int(c.to_u32()) in special:
            out += "\\"
        out += String(c)
    return out^


def _sort_by_len_desc(xs: List[String]) -> List[String]:
    """`sorted(xs, key=len, reverse=True)`: stable, length in characters."""
    var out = List[String]()
    for x in xs:
        var n = x.count_codepoints()
        var j = len(out)
        while j > 0 and out[j - 1].count_codepoints() < n:
            j -= 1
        out.insert(j, x)
    return out^


# ── the tables ────────────────────────────────────────────────────────────

comptime _MONTHS: List[String] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
]
comptime _WEEKDAYS = """monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun
lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag sonntag lunes
martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca
quarta quinta sexta"""
comptime _CUR_DOT: List[String] = ["$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY", "RMB",
                                   "元", "HKD", "SGD"]
comptime _CUR_COMMA: List[String] = ["R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY"]
comptime _CUR_WHOLE: List[String] = ["¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP"]
comptime _CUR_BOTH: List[String] = ["€", "EUR", "CHF"]

comptime _DAY = "(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)"
comptime _MONTH = "(?:mm?|month|mois|monat|mes|mese)"
comptime _YEAR = "(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)"


@fieldwise_init
struct ParsedNumber(Copyable, Movable):
    """`parse_number`'s result: None, an int or a float."""

    var kind: Int
    """0 None, 1 int, 2 float."""
    var i: Int
    var f: Float64

    @staticmethod
    def none() -> ParsedNumber:
        return ParsedNumber(0, 0, 0.0)


@fieldwise_init
struct ParsedDate(Copyable, Movable):
    var found: Bool
    var iso: String
    var ambiguous: Bool


struct D1Parser(Movable):
    """The compiled patterns and tables; build once (`D1Parser()`), then parse."""

    var months: Dict[String, Int]
    var month_keys: List[String]
    var weekdays: List[String]
    var names: Dict[String, String]
    """A currency's casefold -> its written form."""
    var marks: Dict[String, Int]
    """A currency's casefold -> 0 ".", 1 ",", 2 "" (whole units), 3 None (both in use)."""
    var scales: Dict[String, Int]
    """A scale word -> its power of ten."""
    var month_name: String
    var currency: String
    var p: Dict[String, String]
    """Every pattern's text by its d1 name (what the gate compares)."""
    var re_cjk: PyRe
    var re_numeric: PyRe
    var re_word: PyRe
    var re_currency: PyRe
    var re_scale: PyRe
    var re_suffix: PyRe
    var re_counted: PyRe
    var re_grouped: PyRe
    var re_short: PyRe
    var re_repeated: PyRe
    var re_locale: PyRe
    var re_xml: PyRe
    var re_cur_either: PyRe
    var re_number_chars: PyRe
    var re_spacer: PyRe
    var re_spaced: PyRe
    var re_lakh: PyRe
    var re_rupee: PyRe
    var re_group_dot: PyRe
    var re_group_comma: PyRe
    var re_fmt_dmy: PyRe
    var re_fmt_mdy: PyRe
    var re_date_locale: PyRe
    var re_timed: PyRe
    var re_ordinal: PyRe
    var re_ordinal2: PyRe
    var re_articles: PyRe
    var re_commas: PyRe

    def __init__(out self) raises:
        self.months = Dict[String, Int]()
        self.month_keys = List[String]()
        var abbr_keys = List[String]()
        var abbr = Dict[String, List[Int]]()
        var ms = materialize[_MONTHS]()
        for m in range(len(ms)):
            for name in ms[m].split(" "):
                var n = String(name)
                if n not in self.months:
                    self.month_keys.append(n)
                self.months[n] = m + 1
        for k in self.month_keys.copy():
            var pre = text_of(cps(k), 0, 3)
            if pre not in abbr:
                abbr[pre] = List[Int]()
                abbr_keys.append(pre)
            if self.months[k] not in abbr[pre]:
                abbr[pre].append(self.months[k])
        for pre in abbr_keys:
            if len(abbr[pre]) == 1:
                if pre not in self.months:
                    self.month_keys.append(pre)
                self.months[pre] = abbr[pre][0]
        var extra: List[String] = ["janv", "fevr", "juil", "sept"]
        var extra_m: List[Int] = [1, 2, 7, 9]
        for i in range(4):
            if extra[i] not in self.months:
                self.month_keys.append(extra[i])
            self.months[extra[i]] = extra_m[i]
        self.weekdays = List[String]()
        for w in String(_WEEKDAYS).split():
            self.weekdays.append(String(w))

        self.names = Dict[String, String]()
        self.marks = Dict[String, Int]()
        var order = List[String]()
        var groups: List[List[String]] = [materialize[_CUR_DOT](), materialize[_CUR_COMMA](), materialize[_CUR_WHOLE](),
                                          materialize[_CUR_BOTH]()]
        for g in range(4):
            for c in groups[g]:
                var key = text_of(_lower(cps(c)))
                if key not in self.names:
                    order.append(key)
                self.names[key] = c
                self.marks[key] = g
        var written = List[String]()
        for k in order:
            written.append(self.names[k])
        var alts = List[String]()
        for c in _sort_by_len_desc(written):
            var first = cps(c)[0]
            var ascii_alpha = (first >= 65 and first <= 90) or (first >= 97 and first <= 122)
            alts.append(String("\\b") + _re_escape(c) + "(?![a-z])" if ascii_alpha else _re_escape(c))
        self.currency = String("(?i:") + String("|").join(alts) + ")"

        self.scales = Dict[String, Int]()
        var scale_keys = List[String]()
        var sw: List[List[String]] = [
            ["thousand", "mil", "tausend", "тыс", "千"], ["万", "萬"], ["lakh", "lakhs"], ["crore", "crores"],
            ["million", "millions", "mio", "mln", "millón", "millones", "milhão", "milhões", "milione", "milioni", "млн"],
            ["亿", "億"], ["billion", "billions", "milliard", "milliards", "milliarde", "milliarden", "mrd", "miliardo",
                         "miliardi", "млрд"], ["trillion"],
        ]
        var sp: List[Int] = [3, 4, 5, 7, 6, 8, 9, 12]
        for g in range(len(sw)):
            for w in sw[g]:
                if w not in self.scales:
                    scale_keys.append(w)
                self.scales[w] = sp[g]
        var esc = List[String]()
        for k in scale_keys:
            esc.append(_re_escape(k))

        var mk = List[String]()
        for k in self.month_keys:
            mk.append(k)
        self.month_name = String("|").join(_sort_by_len_desc(mk))

        self.p = Dict[String, String]()
        self.p["CJK_DATE"] = "(\\d{4})\\s*[年년]\\s*(\\d{1,2})\\s*[月월]\\s*(\\d{1,2})\\s*[日일]"
        self.p["NUMERIC_DATE"] = "(?<!\\d)(\\d{1,4})([/.-])(\\d{1,2})\\2(\\d{2,4})(?!\\d)"
        self.p["WORD_DATE"] = (String("\\b(?:\\d{1,2}\\w*[ .]+(?:de\\s+)?(?:") + self.month_name
                               + ")[ ,.]+(?:de\\s+)?\\d{4}|(?:" + self.month_name + ")[ .]+\\d{1,2}\\w*[ ,.]+\\d{4})\\b")
        self.p["_CURRENCY"] = self.currency
        self.p["_SCALE"] = String("\\s*(") + String("|").join(_sort_by_len_desc(esc)) + ")\\.?$"
        self.p["_SUFFIX"] = "(?<=\\d)(k|m|bn|b)$"
        self.p["_COUNTED"] = ("\\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|"
                              + "downloads)\\b")
        self.p["_GROUPED"] = "(?<![\\d.,])\\d{1,3}(?:([,. \\u00a0\\u202f'’])\\d{3})+(?!\\1)([.,])\\d+(?![\\d.,])"
        self.p["_SHORT"] = "(?<![\\d.,])\\d+([.,])\\d{1,2}(?![\\d.,])"
        self.p["_REPEATED"] = "(?<![\\d.,])\\d{1,3}(?:([.,])\\d{3}){2,}(?![\\d.,])"
        self.p["_LOCALE"] = ("\\b(?:locale|language|lang)[\"\\']?\\s*[=:]\\s*[\"\\']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)"
                             + "(?:[-_][a-z]{2})?\\b")
        self.p["DATE_FORMAT_DMY"] = String("\\b") + _DAY + "\\s*[/.-]\\s*" + _MONTH + "\\s*[/.-]\\s*" + _YEAR + "\\b"
        self.p["DATE_FORMAT_MDY"] = String("\\b") + _MONTH + "\\s*[/.-]\\s*" + _DAY + "\\s*[/.-]\\s*" + _YEAR + "\\b"

        self.re_cjk = PyRe.compile(self.p["CJK_DATE"])
        self.re_numeric = PyRe.compile(self.p["NUMERIC_DATE"])
        self.re_word = PyRe.compile(self.p["WORD_DATE"])
        self.re_currency = PyRe.compile(self.currency)
        self.re_scale = PyRe.compile(self.p["_SCALE"], ignorecase=True)
        self.re_suffix = PyRe.compile(self.p["_SUFFIX"], ignorecase=True)
        self.re_counted = PyRe.compile(self.p["_COUNTED"], ignorecase=True)
        self.re_grouped = PyRe.compile(self.p["_GROUPED"])
        self.re_short = PyRe.compile(self.p["_SHORT"])
        self.re_repeated = PyRe.compile(self.p["_REPEATED"])
        self.re_locale = PyRe.compile(self.p["_LOCALE"])
        self.re_xml = PyRe.compile("<\\?[^>]*\\?>")
        self.re_cur_either = PyRe.compile(String("(") + self.currency + ")\\s*(.*)|(.*?)\\s*(" + self.currency + ")")
        self.re_number_chars = PyRe.compile("\\d[\\d., '’]*")
        self.re_spacer = PyRe.compile("[ '’]")
        self.re_spaced = PyRe.compile("(\\d{1,3}(?:[ '’]\\d{3})+)([.,]\\d+)?")
        self.re_lakh = PyRe.compile("\\d{1,2}(?:,\\d{2})+,\\d{3}(?:\\.\\d+)?")
        self.re_rupee = PyRe.compile("₹|\\bINR\\b|\\bRs\\b|lakh|crore", ignorecase=True)
        self.re_group_dot = PyRe.compile("\\d{1,3}(?:\\.\\d{3})+,\\d+")  # group ".", point ","
        self.re_group_comma = PyRe.compile("\\d{1,3}(?:,\\d{3})+\\.\\d+")  # group ",", point "."
        self.re_fmt_dmy = PyRe.compile(self.p["DATE_FORMAT_DMY"])
        self.re_fmt_mdy = PyRe.compile(self.p["DATE_FORMAT_MDY"])
        self.re_date_locale = PyRe.compile(
            "\\b(?:locale|language|lang|langue|sprache|idioma|lingua)[\"\\']?\\s*[=:]\\s*[\"\\']?"
            + "(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\\b")
        self.re_timed = PyRe.compile("(.+?)[T\\s]+(\\d{1,2}:\\d{2}(?::\\d{2}(?:\\.\\d+)?)?(?:Z|[+-]\\d{2}:\\d{2})?)")
        self.re_ordinal = PyRe.compile("(?<=\\d)(?:st|nd|rd|th|er)\\b")
        self.re_ordinal2 = PyRe.compile("(?<=\\d)\\.?(?:o|a|°)\\b")
        self.re_articles = PyRe.compile("\\b(?:de|of|the|le|el|il|den)\\b")
        self.re_commas = PyRe.compile("[,.]")

    # ── numbers ──

    def written_unit(self, text: Cps) -> Tuple[Bool, String]:
        """`parsing.written_unit`: "%" when the text ends with one, else its one
        currency marker in its written form; (False, "") for none."""
        var r = _rstrip(text)
        if len(r) > 0 and r[len(r) - 1] == 37:
            return (True, String("%"))
        var ms = self.re_currency.finditer(text)
        if len(ms) == 1:
            var key = text_of(_lower(_slice(text, ms[0].start(), ms[0].end())))
            try:
                return (True, self.names[key])
            except:
                return (False, String(""))
        return (False, String(""))

    def _currency_marks(self, text: Cps) -> List[Int]:
        """The decimal marks of the currencies `text` writes (a currency of both says nothing)."""
        var out = List[Int]()
        for ref m in self.re_currency.finditer(text):
            var key = text_of(_lower(_slice(text, m.start(), m.end())))
            var mk = self.marks.get(key)
            if mk and mk.value() != 3 and mk.value() not in out:
                out.append(mk.value())
        return out^

    def _decimal_mark(self, written: Cps, context0: Cps) -> Int:
        """`_decimal_mark`: 0 ".", 1 ",", 2 "" (whole units), -1 None."""
        var context = self.re_xml.sub(context0, "")
        var numbers = List[Int]()
        for ref m in self.re_grouped.finditer(context):
            var k = 0 if context[m.start(2)] == 46 else 1
            if k not in numbers:
                numbers.append(k)
        for ref m in self.re_short.finditer(context):
            var k = 0 if context[m.start(1)] == 46 else 1
            if k not in numbers:
                numbers.append(k)
        for ref m in self.re_repeated.finditer(context):
            var k = 1 if context[m.start(1)] == 46 else 0
            if k not in numbers:
                numbers.append(k)
        var locales = List[Int]()
        var folded = _lower(context)
        for ref m in self.re_locale.finditer(folded):
            var lang = text_of(folded, m.start(1), m.end(1))
            var k = 0 if (lang == "en" or lang == "ja" or lang == "zh" or lang == "ko") else 1
            if k not in locales:
                locales.append(k)
        var tiers: List[List[Int]] = [self._currency_marks(written), numbers^, self._currency_marks(context), locales^]
        for ref t in tiers:
            if len(t) > 0:
                return t[0] if len(t) == 1 else -1
        return -1

    def _digits(self, text: Cps, written: Cps, context: Cps) -> Tuple[Bool, Cps]:
        """`_digits`: the number as plain digits and a point, or (False, ...)."""
        if not self.re_number_chars.fullmatch(text):
            return (False, Cps())
        if self.re_spacer.search(text):
            var g = self.re_spaced.fullmatch(text)
            if not g:
                return (False, Cps())
            var gm = g.value().copy()
            var head = self.re_spacer.sub(_slice(text, gm.start(1), gm.end(1)), "")
            var tail = _slice(text, gm.start(2), gm.end(2)) if gm.has(2) else Cps()
            return self._digits(_cat(head, tail), written, context)
        var both = _cat(_cat(written, cps("\n")), context)
        if self.re_lakh.fullmatch(text) and self.re_rupee.search(both):
            var out = Cps()
            for x in text:
                if x != 44:
                    out.append(x)
            return (True, out^)
        var has_dot = 46 in text
        var has_comma = 44 in text
        if has_dot and has_comma:
            var last_dot = -1
            var last_comma = -1
            for i in range(len(text)):
                if text[i] == 46:
                    last_dot = i
                elif text[i] == 44:
                    last_comma = i
            var point: UInt32 = 46 if last_dot > last_comma else 44
            var ok = self.re_group_comma.fullmatch(text) if point == 46 else self.re_group_dot.fullmatch(text)
            if not ok:
                return (False, Cps())
            var out = Cps()
            for x in text:
                if x == point:
                    out.append(46)
                elif x != 44 and x != 46:
                    out.append(x)
            return (True, out^)
        if not has_dot and not has_comma:
            return (True, text.copy())
        var mark: UInt32 = 46 if has_dot else 44
        var parts = List[Cps]()
        var cur = Cps()
        for x in text:
            if x == mark:
                parts.append(cur^)
                cur = Cps()
            else:
                cur.append(x)
        parts.append(cur^)
        for ref pt in parts:
            if len(pt) == 0:
                return (False, Cps())
        var point = self._decimal_mark(written, context)
        var mark_k = 0 if mark == 46 else 1
        if len(parts) > 2 or (point != -1 and point != mark_k):  # the mark groups: complete threes
            var ok = len(parts[0]) >= 1 and len(parts[0]) <= 3
            for i in range(1, len(parts)):
                ok = ok and len(parts[i]) == 3
            if not ok:
                return (False, Cps())
            var out = Cps()
            for ref pt in parts:
                for x in pt:
                    out.append(x)
            return (True, out^)
        if point == -1 and len(parts[1]) == 3 and len(parts[0]) <= 3:  # 1,250: unsettled
            return (False, Cps())
        var out = parts[0].copy()
        out.append(46)
        for x in parts[1]:
            out.append(x)
        return (True, out^)

    def parse_number(self, text0: Cps, context0: Cps, unit: String = "", has_unit: Bool = False) raises -> ParsedNumber:
        """`parsing.parse_number(text, context, unit=unit)`."""
        var context = context0.copy()
        if has_unit and unit.byte_length() > 0 and self.re_currency.fullmatch(_strip(cps(unit))):
            context = _cat(context, _cat(cps("\n"), cps(unit)))
        var written = text0.copy()
        var t = Cps()
        for x in _strip(text0):
            if x == 0x2212:
                t.append(45)
            elif x == 0xA0 or x == 0x202F:
                t.append(32)
            else:
                t.append(x)
        if len(t) > 0 and t[len(t) - 1] == 37:
            t = _slice(t, 0, len(t) - 1)
        t = _rstrip(t)
        var negative = len(t) > 0 and t[0] == 40 and t[len(t) - 1] == 41
        if negative:
            t = _strip(_slice(t, 1, len(t) - 1))
        var sign: UInt32 = 0
        var has_currency = False
        for _ in range(2):
            if not has_currency:
                var m = self.re_cur_either.fullmatch(t)
                if m:
                    var mm = m.value().copy()
                    has_currency = True
                    if mm.has(1) and mm.end(1) > mm.start(1):
                        t = _strip(_slice(t, mm.start(2), mm.end(2)))
                    else:
                        t = _strip(_slice(t, mm.start(3), mm.end(3)))
            if sign == 0 and len(t) > 0 and (t[0] == 45 or t[0] == 43):
                sign = t[0]
                t = _strip(_slice(t, 1, len(t)))
        if negative and sign != 0:
            return ParsedNumber.none()
        var scale = 0  # a power of ten
        var scale_ok = True
        var w = self.re_scale.search(t)
        if w:
            var wm = w.value().copy()
            var word = text_of(t, wm.start(1), wm.end(1))
            if word == "Billion":
                scale_ok = False
            else:
                scale = self.scales[text_of(_lower(cps(word)))]
            t = _slice(t, 0, wm.start())
        else:
            var sfx = self.re_suffix.search(t)
            if sfx:
                var sm = sfx.value().copy()
                var both = _cat(_cat(written, cps("\n")), context)
                if not (self.re_currency.search(both) or self.re_counted.search(context)):
                    return ParsedNumber.none()
                var s = text_of(_lower(_slice(t, sm.start(1), sm.end(1))))
                scale = 3 if s == "k" else (6 if s == "m" else 9)
                t = _slice(t, 0, sm.start())
        var dg = self._digits(t, written, context)
        if not dg[0] or not scale_ok:
            return ParsedNumber.none()
        return _decimal_value(dg[1], negative or sign == 45, scale)

    # ── dates ──

    def date_order(self, context: Cps, folded: Cps) -> Int:
        """`date_order`: 0 "dmy", 1 "mdy", -1 None."""
        var explicit = List[Int]()
        if self.re_fmt_dmy.search(folded):
            explicit.append(0)
        if self.re_fmt_mdy.search(folded):
            explicit.append(1)
        if len(explicit) > 0:
            return explicit[0] if len(explicit) == 1 else -1
        var locales = List[Int]()
        for ref m in self.re_date_locale.finditer(folded):
            var loc = text_of(folded, m.start(1), m.end(1)).replace("_", "-")
            var k = 1 if (loc == "en" or loc == "en-us") else 0
            if k not in locales:
                locales.append(k)
        if len(locales) > 0:
            return locales[0] if len(locales) == 1 else -1
        var orders = List[Int]()
        for ref m in self.re_numeric.finditer(context):
            try:
                var a = _int(_slice(context, m.start(1), m.end(1)))
                var b = _int(_slice(context, m.start(3), m.end(3)))
                var year = _int(_slice(context, m.start(4), m.end(4)))
                if m.end(4) - m.start(4) != 4 or a > 31 or b > 31 or a < 1 or b < 1:
                    continue
                if _valid(year, b, a) and ((a > 12) != (b > 12)):
                    if 0 not in orders:
                        orders.append(0)
                if _valid(year, a, b) and ((a > 12) != (b > 12)):
                    if 1 not in orders:
                        orders.append(1)
            except:
                continue
        return orders[0] if len(orders) == 1 else -1

    def parse_date(self, text: Cps, context: Cps) raises -> ParsedDate:
        """`parsing.parse_date`: (ISO value, unresolved ambiguity)."""
        var folded = fold(context)
        var order = self.date_order(context, folded)
        return self._parse_date(text, context, folded, order)

    def _parse_date(self, text0: Cps, context: Cps, folded: Cps, order: Int) raises -> ParsedDate:
        var text = _strip(text0)
        if len(text) == 0:
            return ParsedDate(False, String(""), False)
        var timed = self.re_timed.fullmatch(text)
        if timed:
            var tm = timed.value().copy()
            var d = self._parse_date(_slice(text, tm.start(1), tm.end(1)), context, folded, order)
            if not d.found:
                return ParsedDate(False, String(""), d.ambiguous)
            var clock = _slice(text, tm.start(2), tm.end(2))
            if clock[1] == 58:
                clock.insert(0, 48)
            var iso = _iso_datetime(d.iso, clock)
            if iso.byte_length() == 0:
                return ParsedDate(False, String(""), False)
            return ParsedDate(True, iso, False)
        var numeric = self.re_numeric.fullmatch(text)
        if numeric:
            var nm = numeric.value().copy()
            var a = _int(_slice(text, nm.start(1), nm.end(1)))
            var b = _int(_slice(text, nm.start(3), nm.end(3)))
            var c = _int(_slice(text, nm.start(4), nm.end(4)))
            var cand = List[Tuple[Int, Int, Int]]()  # (year, month, day)
            if nm.end(1) - nm.start(1) == 4:
                cand.append((a, b, c))
            elif nm.end(4) - nm.start(4) == 4:
                _candidates(cand, c, a, b, order)
            else:
                if nm.end(4) - nm.start(4) != 2:
                    return ParsedDate(False, String(""), False)
                var centuries = List[Int]()
                var complete = List[Cps]()
                for ref m in self.re_numeric.finditer(context):
                    if m.end(1) - m.start(1) == 4 or m.end(4) - m.start(4) == 4:
                        complete.append(_slice(context, m.start(), m.end()))
                for ref m in self.re_word.finditer(folded):
                    complete.append(_slice(folded, m.start(), m.end()))
                for ref cnd in complete:
                    var v = self._parse_date(cnd, context, folded, order)
                    if v.found and not v.ambiguous:
                        var cent = Int(String(v.iso[byte=0:4])) // 100 * 100
                        if cent not in centuries:
                            centuries.append(cent)
                if len(centuries) != 1:
                    return ParsedDate(False, String(""), False)
                _candidates(cand, centuries[0] + c, a, b, order)
            var values = List[String]()
            for ref y in cand:
                if _valid(y[0], y[1], y[2]):
                    var iso = _iso_date(y[0], y[1], y[2])
                    if iso not in values:
                        values.append(iso)
            if len(values) > 1:
                return ParsedDate(False, String(""), True)
            if len(values) == 0:
                return ParsedDate(False, String(""), False)
            return ParsedDate(True, values[0], False)
        var cjk = self.re_cjk.fullmatch(text)
        if cjk:
            var cm = cjk.value().copy()
            var y = _int(_slice(text, cm.start(1), cm.end(1)))
            var mo = _int(_slice(text, cm.start(2), cm.end(2)))
            var dd = _int(_slice(text, cm.start(3), cm.end(3)))
            if not _valid(y, mo, dd):
                return ParsedDate(False, String(""), False)
            return ParsedDate(True, _iso_date(y, mo, dd), False)
        var words = fold(text)
        words = self.re_ordinal.sub(words, "")
        words = self.re_ordinal2.sub(words, "")
        words = self.re_articles.sub(words, " ")
        var parts = _split_ws(self.re_commas.sub(words, " "))
        if len(parts) > 0:
            var first = text_of(parts[0])
            if first.endswith("-feira"):
                var cut = first.byte_length() - 6
                var head = String(first[byte=0:cut])
                first = head^
            if first in self.weekdays:
                _ = parts.pop(0)
        if len(parts) != 3:
            return ParsedDate(False, String(""), False)
        var month: Int
        var day: Cps
        var year: Cps
        var w0 = text_of(parts[0])
        var w1 = text_of(parts[1])
        if w0 in self.months:
            month = self.months[w0]
            day = parts[1].copy()
            year = parts[2].copy()
        elif w1 in self.months:
            day = parts[0].copy()
            month = self.months[w1]
            year = parts[2].copy()
        else:
            return ParsedDate(False, String(""), False)
        if not _all_digits(day) or not _all_digits(year) or len(year) != 4:
            return ParsedDate(False, String(""), False)
        try:
            var yi = _int(year)
            var di = _int(day)
            if not _valid(yi, month, di):
                return ParsedDate(False, String(""), False)
            return ParsedDate(True, _iso_date(yi, month, di), False)
        except:
            return ParsedDate(False, String(""), False)


# ── helpers ───────────────────────────────────────────────────────────────

def _candidates(mut out: List[Tuple[Int, Int, Int]], year: Int, a: Int, b: Int, order: Int):
    """`_candidates`: (year, month, day) for a/b/year — day first, month first, or as `order` says."""
    if order == -1 or order == 0:
        out.append((year, b, a))
    if order == -1 or order == 1:
        out.append((year, a, b))


def _leap(y: Int) -> Bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _valid(y: Int, m: Int, d: Int) -> Bool:
    """`datetime.date(y, m, d)` would not raise."""
    if y < 1 or y > 9999 or m < 1 or m > 12 or d < 1:
        return False
    var dim: List[Int] = [31, 29 if _leap(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return d <= dim[m - 1]


def _pad(v: Int, n: Int) -> String:
    var s = String(v)
    while s.byte_length() < n:
        s = String("0") + s
    return s^


def _iso_date(y: Int, m: Int, d: Int) -> String:
    return _pad(y, 4) + "-" + _pad(m, 2) + "-" + _pad(d, 2)


def _ascii_int(t: Cps, a: Int, b: Int) -> Int:
    """An ASCII digit run's value, -1 when it is not one (`fromisoformat` reads ASCII only)."""
    if b <= a:
        return -1
    var v = 0
    for i in range(a, b):
        if t[i] < 48 or t[i] > 57:
            return -1
        v = v * 10 + Int(t[i]) - 48
    return v


def _iso_datetime(date: String, clock: Cps) -> String:
    """`datetime.fromisoformat(date + "T" + clock).isoformat()` for the clocks
    `parse_date` reads (HH:MM[:SS[.f+]][Z|±HH:MM]); "" where Python raises."""
    var n = len(clock)
    var hh = _ascii_int(clock, 0, 2)
    if n < 5 or clock[2] != 58:
        return String("")
    var mi = _ascii_int(clock, 3, 5)
    var i = 5
    var ss = 0
    var us = 0
    if i < n and clock[i] == 58:
        ss = _ascii_int(clock, i + 1, i + 3)
        i += 3
        if i < n and clock[i] == 46:
            var j = i + 1
            while j < n and clock[j] >= 48 and clock[j] <= 57:
                j += 1
            if j == i + 1:
                return String("")
            var nd = min(6, j - i - 1)
            us = _ascii_int(clock, i + 1, i + 1 + nd)
            for _ in range(6 - nd):
                us *= 10
            i = j
    var tz = String("")
    if i < n:
        if clock[i] == 90 and i + 1 == n:  # Z
            tz = String("+00:00")
        elif (clock[i] == 43 or clock[i] == 45) and i + 6 == n and clock[i + 3] == 58:
            var th = _ascii_int(clock, i + 1, i + 3)
            var tmn = _ascii_int(clock, i + 4, i + 6)
            if th < 0 or tmn < 0 or tmn > 59 or th > 23:
                return String("")
            var zero = th == 0 and tmn == 0
            tz = String("+" if (clock[i] == 43 or zero) else "-") + _pad(th, 2) + ":" + _pad(tmn, 2)
        else:
            return String("")
    if hh < 0 or mi < 0 or ss < 0 or us < 0 or hh > 23 or mi > 59 or ss > 59:
        return String("")
    var out = date + "T" + _pad(hh, 2) + ":" + _pad(mi, 2) + ":" + _pad(ss, 2)
    if us > 0:
        out += "." + _pad(us, 6)
    return out + tz


def _decimal_value(digits: Cps, negative: Bool, scale: Int) raises -> ParsedNumber:
    """`Decimal(sign + digits) * 10**scale`: an int when integral, else its float."""
    var mant = 0
    var frac = 0
    var seen_point = False
    var big = False
    var nz = 0  # significant digits
    for x in digits:
        if x == 46:
            seen_point = True
            continue
        var d = decimal_value(x)
        if d < 0:
            return ParsedNumber.none()
        if mant > 0 or d > 0:
            nz += 1
        if nz > 18:
            big = True
        mant = mant * 10 + d
        if seen_point:
            frac += 1
    var e = scale - frac
    while e < 0 and mant % 10 == 0 and mant != 0:  # trailing zeros of the fraction
        mant //= 10
        e += 1
    if mant == 0:
        return ParsedNumber(1, 0, 0.0)
    if e >= 0 and not big:
        var v = mant
        var ok = True
        for _ in range(e):
            if v > 922337203685477580:
                ok = False
                break
            v *= 10
        if ok:
            return ParsedNumber(1, -v if negative else v, 0.0)
    var f: Float64
    if not big and mant < (1 << 53) and e < 0 and e >= -22:
        f = Float64(mant) / (10.0 ** Float64(-e))  # exact operands: one correctly rounded division
    else:
        var s = String(mant) + "e" + String(e)
        f = Float64(s)
    if not isfinite(f):
        return ParsedNumber.none()
    return ParsedNumber(2, 0, -f if negative else f)
