# +--------------------------------------------------------------------------+ #
# | What happened to the robot, one line an event, never rewritten
# +--------------------------------------------------------------------------+ #
"""The robot's record: every event of a day appended to one file.

    var log = EventLog(robot_dir("noe"))            # ~/.noeira/robots/noe
    log.append(BrainEvent(epoch_seconds(), EV_HEARD, "Va à la table", "voice"))
    var day = load_events(log.path_for(epoch))      # the file, back as events

    ~/.noeira/robots/<name>/log/2026-11-13.jsonl    (UTC dates)
    {"t":1794000000,"at":"2026-11-13T09:20:00Z","kind":"heard","source":"voice","text":"Va à la table"}

⚠ APPEND-ONLY, AND THAT IS THE DESIGN (docs/ROBOT_BRAIN_PLAN.md §5.3, §8). The
state in RAM, the daily digest and, later, a model's consolidation are all
DERIVED from this file and can be rebuilt from it. Nothing here edits or
removes a line: a consolidation that could rewrite its own source is a
telephone game with no way back.

The kinds the state's reducer reads are below. An unknown kind is written and
read back like the others and the reducer ignores it, so a newer binary's log
does not break an older one.
"""

from std.os import getenv, makedirs

from noeira.core.bytes import string_from_bytes
from noeira.core.run import iso8601_utc, date_utc
from noeira.io.fileio import read_file_bytes
from noeira.io.json import JsonWriter, parse_json


comptime EV_SESSION = "session"
"""A process started; `text` = the body's name."""
comptime EV_HEARD = "heard"
"""An utterance addressed or not; `text` = what was transcribed."""
comptime EV_SAID = "said"
"""The robot spoke; `text` = what it said."""
comptime EV_BEGAN = "began"
"""A command started; `text` = its name ("squat", "spin_left")."""
comptime EV_GOING = "going"
"""The robot set off for a place; `text` = the place."""
comptime EV_ARRIVED = "arrived"
"""The robot arrived; `text` = the place."""
comptime EV_TASK_STARTED = "task_started"
"""The executive took a task; `text` = its steps in words."""
comptime EV_STEP_STARTED = "step_started"
comptime EV_STEP_DONE = "step_done"
"""`text` = the step in words ("walk to the table")."""
comptime EV_TASK_DONE = "task_done"
comptime EV_TASK_FAILED = "task_failed"
comptime EV_TASK_CANCELLED = "task_cancelled"
"""`text` = the task in words, then " | " and why."""


@fieldwise_init
struct BrainEvent(Copyable, Movable):
    var t: Int
    """Seconds since the Unix epoch (`noeira.core.run.epoch_seconds`)."""
    var kind: String
    var text: String
    var source: String
    """Who reported it: `voice`, `nav`, `keyboard`, `demo`, a test's name."""


def event_json(e: BrainEvent) raises -> String:
    """One event as its log line (no newline). `at` repeats `t` for a reader;
    `t` is what is read back."""
    var w = JsonWriter()
    w.begin_object()
    w.key("t")
    w.integer(e.t)
    w.member("at", iso8601_utc(e.t))
    w.member("kind", e.kind)
    w.member("source", e.source)
    w.member("text", e.text)
    w.end_object()
    return w.done()


def parse_event(line: String) raises -> BrainEvent:
    var b = List[UInt8]()
    for x in line.as_bytes():
        b.append(x)
    var doc = parse_json(b^)
    var root = doc.root()
    var src = doc.field(root, "source")
    return BrainEvent(
        doc.integer(doc.field(root, "t")),
        doc.string(doc.field(root, "kind")),
        doc.string(doc.field(root, "text")),
        doc.string(src) if src >= 0 else String(""),
    )


def load_events(path: String) raises -> List[BrainEvent]:
    """Every event of one log file, in the file's order. A line that does not
    parse raises with its number: a damaged record is not skipped silently."""
    var out = List[BrainEvent]()
    var txt = string_from_bytes(read_file_bytes(path))
    var n = 0
    for l in txt.split("\n"):
        n += 1
        var line = String(l)
        if line.byte_length() == 0:
            continue
        try:
            out.append(parse_event(line))
        except e:
            raise Error("brain log " + path + " line " + String(n) + ": " + String(e))
    return out^


def robot_dir(name: String) -> String:
    """`~/.noeira/robots/<name>`: one robot's memory on this machine."""
    return getenv("HOME") + "/.noeira/robots/" + name


struct EventLog(Movable):
    var dir: String
    """The robot's folder; the files are `<dir>/log/<UTC date>.jsonl`."""
    var made: Bool

    def __init__(out self, var dir: String):
        self.dir = dir^
        self.made = False

    def path_for(self, t: Int) -> String:
        return self.dir + "/log/" + date_utc(t) + ".jsonl"

    def append(mut self, e: BrainEvent) raises:
        """Write one line and close the file: an event is on disk when this
        returns, whatever happens to the process next."""
        if not self.made:
            makedirs(self.dir + "/log", exist_ok=True)
            self.made = True
        var line = event_json(e) + "\n"
        with open(self.path_for(e.t), "a") as f:
            f.write(line)
