# +--------------------------------------------------------------------------+ #
# | Memory across restarts — rebuilt from the robot's own log, by rule
# +--------------------------------------------------------------------------+ #
"""What the robot knows when its process starts again (docs/ROBOT_BRAIN_PLAN.md
§6, step B3).

    var ws = WorldState(Body.g1_room("Noé"), "a furnished room")
    restore_memory(ws, robot_dir("noe"), epoch_seconds())   # BEFORE open_log
    ws.open_log(robot_dir("noe"), epoch_seconds())

    ws.remembered:
      "Earlier today, before you were restarted: You did: walk to the table, then walk to the
       door. The last things you were asked or told: "Va à la table, puis à la porte."."
      "Yesterday: You did: walk to the fridge. You tried and failed: walk to the sofa (the sofa
       was not reached after 60 seconds). ..."

THE LOG IS THE MEMORY; EVERYTHING ELSE IS DERIVED FROM IT, HERE, AT START.
Nothing is written at shutdown, so a process that is killed loses nothing: the
events were on disk as they happened (`EventLog.append`). At start the last
days' files are read, each day is folded into a few lines BY RULE
(`digest_lines`: no model, so a restart cannot reword a fact into another one),
and the lines go into the state for the reply view. A past day's lines are also
written to `<dir>/digest/<date>.md` for a person to read; that file is rewritten
from the log whenever it is read back, never the other way round.

WHAT A DAY'S DIGEST SAYS: the tasks that were done, in order (a repeat is
counted, not listed again); the tasks that failed and why; how many were
interrupted; and the last things it was asked or told, word for word. ⚠ The places it "went to" are the places of its
DONE TASKS, not every region its body crossed: the robot's start once drifted
through the fridge's region, the log truthfully held an arrival there, and the
reply model built a story on it.

DAYS ARE UTC DAYS, as the log's files are: a session a little after midnight
local time is "yesterday" a little longer than a person would say. A torn last
line (a power cut in the middle of a write) is skipped; a damaged line anywhere
else is reported in `ws.memory_error` and that day is left out.

`save_places` / `load_places`: the places the robot knows, with their names
in people's languages, in `<dir>/places.json`.
"""

from std.os import listdir, makedirs
from std.os.path import exists

from noeira.core.bytes import string_from_bytes
from noeira.core.run import date_utc
from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.io.json import JsonWriter, parse_json
from .events import (
    BrainEvent, parse_event, EV_HEARD, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED,
)
from .bb1 import normalise, has_phrase
from .geometry import PlaceXY
from .state import WorldState


comptime MEM_DAYS: Int = 3
"""Days of log read back at start, the current one included."""
comptime MEM_TASKS: Int = 6
"""Different done tasks named per day; earlier ones are counted."""
comptime MEM_HEARD: Int = 3
"""Utterances quoted per day, the last ones."""
comptime MEM_HEARD_CHARS: Int = 120
"""An utterance longer than this is cut (on a character boundary)."""


def days_from_civil(y_in: Int, m: Int, d: Int) -> Int:
    """Days since the Unix epoch for a date (Hinnant's `days_from_civil`,
    the inverse of `noeira.core.run.civil_from_days`)."""
    var y = y_in - 1 if m <= 2 else y_in
    var era = (y if y >= 0 else y - 399) // 400
    var yoe = y - era * 400
    var doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def date_days(date: String) raises -> Int:
    """`2026-11-13` -> days since the epoch."""
    var p = date.split("-")
    if len(p) != 3:
        raise Error("brain memory: not a date: '" + date + "'")
    return days_from_civil(Int(String(p[0])), Int(String(p[1])), Int(String(p[2])))


def when_words(days_ago: Int, date: String) -> String:
    """How the reply model is told WHEN: `Earlier today, before you were
    restarted`, `Yesterday`, `3 days ago (2026-11-10)`."""
    if days_ago <= 0:
        return String("Earlier today, before you were restarted")
    if days_ago == 1:
        return String("Yesterday")
    return String(days_ago) + " days ago (" + date + ")"


def _cut(text: String, max_bytes: Int) -> String:
    """`text`, or its first `max_bytes` bytes back to a character boundary, with `...`."""
    if text.byte_length() <= max_bytes:
        return text.copy()
    var b = List[UInt8]()
    for x in text.as_bytes():
        b.append(x)
    var n = max_bytes
    while n > 0 and (Int(b[n]) & 0xC0) == 0x80:
        n -= 1
    var out = List[UInt8]()
    for i in range(n):
        out.append(b[i])
    return String(unsafe_from_utf8=Span(out)) + "..."


def digest_lines(ref events: List[BrainEvent]) -> List[String]:
    """A stretch of the log as a few sentences, by rule (the module note).
    Empty when nothing in it is worth a memory."""
    var done = List[String]()
    var done_n = List[Int]()
    var failed = List[String]()
    var cancelled = 0
    var heard = List[String]()
    var heard_n = 0
    for e in events:
        if e.kind == EV_TASK_DONE:
            var at = -1
            for i in range(len(done)):
                if done[i] == e.text:
                    at = i
            if at >= 0:
                done_n[at] += 1
            else:
                done.append(e.text.copy())
                done_n.append(1)
        elif e.kind == EV_TASK_FAILED:
            var cut = e.text.find(" | ")
            if cut >= 0:
                failed.append(String(e.text[byte=0:cut]) + " (" + String(e.text[byte = cut + 3 :]) + ")")
            else:
                failed.append(e.text.copy())
        elif e.kind == EV_TASK_CANCELLED:
            cancelled += 1
        elif e.kind == EV_HEARD:
            heard_n += 1
            heard.append(_cut(e.text, MEM_HEARD_CHARS))
    var lines = List[String]()
    if len(done) > 0:
        var t = String("You did: ")
        var first = max(0, len(done) - MEM_TASKS)
        if first > 0:
            t += String(first) + " other tasks, then "
        for i in range(first, len(done)):
            if i > first:
                t += "; "
            t += done[i]
            if done_n[i] > 1:
                t += " (" + String(done_n[i]) + " times)"
        lines.append(t + ".")
    if len(failed) > 0:
        var t = String("You tried and failed: ")
        for i in range(len(failed)):
            if i > 0:
                t += "; "
            t += failed[i]
        lines.append(t + ".")
    if cancelled > 0:
        lines.append(String(cancelled) + (" task was" if cancelled == 1 else " tasks were") + " interrupted by a new request.")
    if heard_n > 0:
        # ⚠ WHO SPOKE IS IN THE SENTENCE. As "the last things you heard", a quoted "Je m'appelle Denis."
        # was read by the reply model as its own name; and a count ("spoken to 4 times") became "I talked
        # to four people" (the restart gate's first run). No count, and the wording says they were said
        # TO the robot. (It still reads a quoted "je" as itself: who people are is B4's structured fact.)
        var t = String("The last things you were asked or told: ")
        var first = max(0, len(heard) - MEM_HEARD)
        for i in range(first, len(heard)):
            if i > first:
                t += "; "
            t += "\"" + heard[i] + "\""
        lines.append(t + ".")
    return lines^


def read_log(path: String) raises -> List[BrainEvent]:
    """One log file's events. A last line that does not parse is a torn write
    and is skipped; a damaged line before it raises with its number."""
    var txt = string_from_bytes(read_file_bytes(path))
    var raw = List[String]()
    for l in txt.split("\n"):
        var line = String(l)
        if line.byte_length() > 0:
            raw.append(line^)
    var out = List[BrainEvent]()
    for i in range(len(raw)):
        try:
            out.append(parse_event(raw[i]))
        except e:
            if i == len(raw) - 1:
                break
            raise Error("brain log " + path + " line " + String(i + 1) + ": " + String(e))
    return out^


def log_days(dir: String) raises -> List[String]:
    """The dates that have a log file in `<dir>/log`, oldest first; empty
    when the folder does not exist."""
    var out = List[String]()
    if not exists(dir + "/log"):
        return out^
    for name in listdir(dir + "/log"):
        var n = String(name)
        if n.endswith(".jsonl") and n.byte_length() == 16:
            out.append(String(n[byte=0:10]))
    for i in range(1, len(out)):
        var j = i
        while j > 0 and out[j - 1] > out[j]:
            var t = out[j].copy()
            out[j] = out[j - 1].copy()
            out[j - 1] = t^
            j -= 1
    return out^


def restore_memory(mut ws: WorldState, dir: String, now: Int):
    """Fill `ws.remembered` from the last `MEM_DAYS` days of `<dir>/log`, the
    most recent first, and write each PAST day's digest file. Call it before
    `open_log`, so that today's file holds only earlier sessions. ⚠ Does not
    raise: what cannot be read is reported in `ws.memory_error`."""
    ws.memory_on = True
    ws.remembered = List[String]()
    ws.remembered_ago = List[Int]()
    ws.memory_error = String("")
    var today = date_utc(now)
    try:
        var today_days = date_days(today)
        var days = log_days(dir)
        var i = len(days) - 1
        var read = 0
        while i >= 0 and read < MEM_DAYS:
            var d = days[i].copy()
            i -= 1
            var ago = today_days - date_days(d)
            if ago < 0:
                continue   # a file from the future: a clock that was wrong
            read += 1
            try:
                var events = read_log(dir + "/log/" + d + ".jsonl")
                var lines = digest_lines(events)
                if ago > 0:
                    makedirs(dir + "/digest", exist_ok=True)
                    var md = String("# ") + d + "\n\n"
                    for l in lines:
                        md += "- " + l + "\n"
                    write_text_atomic(dir + "/digest/" + d + ".md", md)
                if len(lines) > 0:
                    var t = when_words(ago, d) + ":"
                    for l in lines:
                        t += " " + l
                    ws.remembered.append(t^)
                    ws.remembered_ago.append(ago)
            except e:
                ws.memory_error = String(e)
    except e:
        ws.memory_error = String(e)


@fieldwise_init
struct TimeRef(Copyable, Movable):
    """A stretch of the past a question names: `lo` to `hi` days ago."""

    var lo: Int
    var hi: Int
    var words: String
    """In English, for the reply model: `yesterday`, `last week`."""


def _says(ref text: List[UInt8], phrases: String) -> Bool:
    for p in phrases.split("|"):
        var pat = normalise(String(p))
        if has_phrase(text, pat):
            return True
    return False


def time_refs(question: String) -> List[TimeRef]:
    """The days a question is about, from its time words, French and English:
    earlier today, yesterday, the day before, last week, longer ago. Empty
    when it names no time.

    ⚠ WHICH DAY "hier" IS, IS ARITHMETIC, AND IT STAYS ON OUR SIDE, as the
    place geometry does. Given two days of memory at once, the reply model
    told yesterday's walk as this morning's, and asked about last week it
    moved yesterday there: 8 to 10 of 14 questions right. A word list is
    coarse, and it is the same coarse the recogniser's language is pinned to."""
    var text = normalise(question)
    var out = List[TimeRef]()
    var before_yesterday = _says(text, String("avant-hier|avant hier|day before yesterday"))
    if before_yesterday:
        out.append(TimeRef(2, 2, String("the day before yesterday")))
    elif _says(text, String("hier|yesterday")):
        out.append(TimeRef(1, 1, String("yesterday")))
    if _says(text, String(
        "aujourd'hui|today|ce matin|this morning|cet après-midi|cet apres-midi|this afternoon|tout à l'heure|tout a l'heure"
        "|plus tôt|plus tot|earlier|avant de redémarrer|before you were restarted|before the restart|tantôt"
    )):
        out.append(TimeRef(0, 0, String("earlier today")))
    if _says(text, String(
        "semaine dernière|semaine derniere|semaine passée|last week|il y a une semaine|a week ago|une semaine"
    )):
        out.append(TimeRef(3, 14, String("last week")))
    if _says(text, String(
        "mois dernier|last month|il y a un mois|a month ago|l'année dernière|l'an dernier|last year|il y a longtemps|long ago"
    )):
        out.append(TimeRef(15, 1000000, String("that long ago")))
    return out^


def focused_memory(ref ws: WorldState) -> List[String]:
    """The memory lines for the utterance being answered (`ws.asked`): when it
    names a time, only that time's lines, or a line saying the log has nothing
    for it; when it names none, every line and the edge of the memory."""
    var out = List[String]()
    var refs = time_refs(ws.asked)
    if len(refs) == 0:
        if len(ws.remembered) == 0:
            out.append(String("Nothing: you remember no earlier session and no earlier day."))
            return out^
        for m in ws.remembered:
            out.append(m.copy())
        out.append(String("Nothing else: you remember no other day."))
        return out^
    for r in refs:
        var found = False
        for i in range(len(ws.remembered)):
            if ws.remembered_ago[i] >= r.lo and ws.remembered_ago[i] <= r.hi:
                out.append(ws.remembered[i].copy())
                found = True
        if not found:
            out.append("About " + r.words + ": nothing. Your log has no entry for " + r.words + ", so you do not know what you did then.")
    return out^


def save_places(ref ws: WorldState, dir: String) raises:
    """`<dir>/places.json`: the places the robot knows and what they are
    called, written whole and atomically."""
    var w = JsonWriter()
    w.begin_object()
    w.key("places")
    w.begin_array()
    for p in ws.places:
        w.begin_object()
        w.member("name", p.name)
        w.key("x")
        w.number(p.x)
        w.key("y")
        w.number(p.y)
        w.key("words")
        w.begin_object()
        for pw in ws.place_words:
            if pw.place == p.name:
                w.member(pw.language, pw.word)
        w.end_object()
        w.end_object()
    w.end_array()
    w.end_object()
    makedirs(dir, exist_ok=True)
    write_text_atomic(dir + "/places.json", w.done() + "\n")


def load_places(mut ws: WorldState, dir: String) raises -> Int:
    """Replace `ws.places` and their words by `<dir>/places.json`'s; the
    number of places read, 0 (and nothing changed) when there is no file."""
    var path = dir + "/places.json"
    if not exists(path):
        return 0
    var doc = parse_json(read_file_bytes(path))
    var arr = doc.field(doc.root(), "places")
    ws.places = List[PlaceXY]()
    ws.place_words.clear()
    for i in range(doc.size(arr)):
        var p = doc.at(arr, i)
        var name = doc.string(doc.field(p, "name"))
        ws.places.append(PlaceXY(name.copy(), doc.number(doc.field(p, "x")), doc.number(doc.field(p, "y"))))
        var words = doc.field(p, "words")
        for k in range(doc.size(words)):
            ws.call_place(name, doc.key_at(words, k), doc.string(doc.at(words, k)))
    return len(ws.places)
