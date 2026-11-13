# +--------------------------------------------------------------------------+ #
# | WorldState — what the robot is, where it is, what it is doing, what happened
# +--------------------------------------------------------------------------+ #
"""The one state every model of the robot reads (docs/ROBOT_BRAIN_PLAN.md §5.1).

    var ws = WorldState(Body.g1_room("Noé"), "a furnished room")
    ws.places = [PlaceXY("fridge", 3.0, 0.0), PlaceXY("table", 0.0, -2.0)]
    ws.open_log(robot_dir("noe"))                  # optional: events also go to disk

    # once a frame, from whoever owns the robot: the facts as they are NOW
    ws.observe_pose(x, y, yaw_deg)
    ws.observe_motion(doing, walking_to, at, now)  # a change becomes an event
    ws.heard("Où est le frigo ?", now)
    ws.said("Devant moi, à trois mètres.", now)

    var view = reply_view(ws)                      # noeira.brain.views: the reply model's prompt

⚠ THE CALLER REPORTS FACTS, THE STATE KEEPS THE HISTORY. `observe_motion` is
given what is true this frame and records the TRANSITIONS itself (`began`,
`going`, `arrived`): a caller that had to remember to log each change would log
some of them, and the robot's memory of its own day would have the gaps.

⚠ NOTHING HERE RAISES OR BLOCKS: it is called from a render loop that also
steps the physics. A log file that cannot be written is reported in
`log_error` and the event still enters the state.

What is NOT here yet (plan §7): objects seen and their last-seen time (B5),
people (B4), the task stack (B2), reading a day back at start (B3).
"""

from .events import (
    BrainEvent, EventLog, EV_SESSION, EV_HEARD, EV_SAID, EV_BEGAN, EV_GOING, EV_ARRIVED,
)

comptime STEP_PENDING: Int = 0
comptime STEP_RUNNING: Int = 1
comptime STEP_DONE: Int = 2
comptime STEP_FAILED: Int = 3

comptime TASK_RUNNING: Int = 1
comptime TASK_DONE: Int = 2
comptime TASK_FAILED: Int = 3
comptime TASK_CANCELLED: Int = 4

comptime STEP_TIMEOUT_S: Float64 = 60.0
"""How long a walk to a place may take before its step has failed. The room's
longest measured walk is about 20 s (`BFM_ZERO_ROOM_PLAN.md` R5b)."""
comptime STEP_HOLD_S: Float64 = 3.0
"""How long a step with no place (a posture, a gesture) runs before the next."""
from .geometry import PlaceXY, PlaceAround, places_around, place_lines, nearest_place


comptime WS_EVENTS_N: Int = 256
"""Events kept in RAM; the oldest go first (the file keeps them all)."""
comptime WS_TRAIL_N: Int = 4
"""Places in `trail`, most recent first (RB-1's reducer rule)."""


struct Body(Copyable, Movable):
    """What this robot is and what it can do, in words a model reads. The
    reply view's "can / cannot" lines are generated from it, so a body that
    cannot walk says so because `can` has no walking, not because a prompt was
    edited."""

    var name: String
    var description: String
    """After "You are <name>, ": `a small humanoid robot with ...`."""
    var can: List[String]
    """Each completes "You can ...": `walk to a place of the room`."""
    var cannot: List[String]
    """Things people ask for that it cannot do, each completing "You cannot ..."."""

    def __init__(out self, var name: String, var description: String):
        self.name = name^
        self.description = description^
        self.can = List[String]()
        self.cannot = List[String]()

    @staticmethod
    def g1_room(name: String) -> Body:
        """The simulated G1 of the room demo: BFM-Zero's bank and the navigator."""
        var b = Body(name, String("a small humanoid robot with two legs, two arms and a head"))
        b.can.append(String("walk to a place of the room"))
        b.can.append(String("turn"))
        b.can.append(String("squat"))
        b.can.append(String("raise your arms"))
        b.can.append(String("talk and answer questions"))
        b.cannot.append(String("pick things up or carry them"))
        b.cannot.append(String("open the door"))
        b.cannot.append(String("sit down"))
        return b^


@fieldwise_init
struct PlaceWord(Copyable, Movable):
    """What a place is called in one language: (`table`, `fr`, `la table`)."""

    var place: String
    var language: String
    var word: String


struct Step(Copyable, Movable):
    """One command of the body inside a task (`noeira.brain.tasks`)."""

    var command: String
    """As the body takes it: `goto:table`, `squat`."""
    var label: String
    """In words, for the models: `walk to the table`."""
    var place: String
    """The place the step ends at; "" for a step that ends after `hold_s`."""
    var timeout_s: Float64
    var hold_s: Float64
    var status: Int
    var t_start: Float64

    def __init__(out self, var command: String, var label: String, var place: String):
        self.command = command^
        self.label = label^
        self.place = place^
        self.timeout_s = STEP_TIMEOUT_S
        self.hold_s = STEP_HOLD_S
        self.status = STEP_PENDING
        self.t_start = 0.0

    @staticmethod
    def goto(place: String) -> Step:
        """Walk to a place: done on arrival, failed after `timeout_s`."""
        return Step("goto:" + place, "walk to the " + place, place.copy())

    @staticmethod
    def gesture(name: String) -> Step:
        """A posture or a gesture of the body's own list: done after `hold_s`."""
        return Step(name.copy(), name.replace("_", " "), String(""))


struct Task(Copyable, Movable):
    """What the robot was asked, as steps to run in order."""

    var said: String
    """The request as it was heard; "" for a task given by code."""
    var steps: List[Step]
    var at: Int
    """The step in hand."""
    var status: Int
    var reason: String
    """Why it failed or was cancelled; "" otherwise."""
    var t_start: Float64
    var t_end: Float64

    def __init__(out self, var said: String):
        self.said = said^
        self.steps = List[Step]()
        self.at = 0
        self.status = TASK_RUNNING
        self.reason = String("")
        self.t_start = 0.0
        self.t_end = 0.0

    def describe(self) -> String:
        """`walk to the table, then walk to the door`."""
        var t = String("")
        for i in range(len(self.steps)):
            if i > 0:
                t += ", then "
            t += self.steps[i].label
        return t


def _or_none(s: String) -> String:
    return String("") if s == "none" else s


struct WorldState(Movable):
    var body: Body
    var where: String
    """After "You are in ": `a furnished room`."""
    var language: String
    """The language of the conversation when the caller knows it (`fr`, `en`,
    ...: the one pinned on the recogniser); "" = unknown, and the reply model
    is told to answer in the language it was addressed in. Measured (BB-1):
    told only that, LFM3-1.7B answered French questions in English when the
    answer was a fact written in English."""
    var places: List[PlaceXY]
    """The places the robot knows, in the room's frame (from the scene, a map
    or a detector: the caller's source)."""

    var place_words: List[PlaceWord]
    """The places' names in the languages people speak to the robot
    (`call_place`). The places are named in English in the state; the reply
    view gives the conversation's words beside them. Measured (BB-1): without
    them LFM3-1.7B called the table "le tableau" in 2 of 16 French answers."""

    var pose_known: Bool
    var x: Float64
    var y: Float64
    var yaw_deg: Float64

    var doing: String
    """The command running now; "" for none."""
    var walking_to: String
    """The place being walked to now; "" for none."""
    var at: String
    """The place the robot stands in now; "" for none."""

    var has_task: Bool
    var task: Task
    """The task in hand (`has_task`), run by `noeira.brain.tasks`."""
    var has_last_task: Bool
    var last_task: Task
    """The last task that ended: done, failed or cancelled."""

    var memory_on: Bool
    """`noeira.brain.memory.restore_memory` was called: the reply view then
    says what the robot remembers from before this session, or that it
    remembers nothing."""
    var remembered: List[String]
    """One line per earlier stretch of the log, the most recent first:
    `Yesterday: You did: ...`."""
    var remembered_ago: List[Int]
    """For each line of `remembered`, how many days ago its day was (0 = an
    earlier session of today)."""
    var asked: String
    """The utterance being answered, when the caller has it. The reply view
    selects the memory by its time words (`noeira.brain.memory.focused_memory`);
    "" = no selection."""
    var memory_error: String
    """What `restore_memory` could not read; "" when nothing."""

    var events: List[BrainEvent]
    """The last `WS_EVENTS_N` events, oldest first."""
    var log: Optional[EventLog]
    var log_error: String
    """The last failure to write the log; "" when none."""

    def __init__(out self, var body: Body, var where: String):
        self.body = body^
        self.where = where^
        self.language = String("")
        self.places = List[PlaceXY]()
        self.place_words = List[PlaceWord]()
        self.pose_known = False
        self.x = 0.0
        self.y = 0.0
        self.yaw_deg = 0.0
        self.doing = String("")
        self.walking_to = String("")
        self.at = String("")
        self.has_task = False
        self.task = Task(String(""))
        self.has_last_task = False
        self.last_task = Task(String(""))
        self.memory_on = False
        self.remembered = List[String]()
        self.remembered_ago = List[Int]()
        self.asked = String("")
        self.memory_error = String("")
        self.events = List[BrainEvent]()
        self.log = None
        self.log_error = String("")

    # ── the record ────────────────────────────────────────────────────

    def open_log(mut self, var dir: String, now: Int):
        """Events also go to `<dir>/log/<date>.jsonl` from here on; the
        session's start is the first one."""
        self.log = Optional(EventLog(dir^))
        var name = self.body.name.copy()
        self.record(EV_SESSION, name, String("brain"), now)

    def record(mut self, kind: String, text: String, source: String, now: Int):
        """One event into the state, and onto disk when a log is open."""
        var e = BrainEvent(now, kind, text, source)
        if self.log:
            try:
                self.log.value().append(e)
            except err:
                self.log_error = String(err)
        self.events.append(e^)
        while len(self.events) > WS_EVENTS_N:
            _ = self.events.pop(0)

    def heard(mut self, text: String, now: Int):
        self.record(EV_HEARD, text, String("voice"), now)

    def said(mut self, text: String, now: Int):
        self.record(EV_SAID, text, String("voice"), now)

    def call_place(mut self, place: String, language: String, word: String):
        """`place` is called `word` in `language`: ("table", "fr", "la table")."""
        self.place_words.append(PlaceWord(place, language, word))

    def word_for(self, place: String, language: String) -> String:
        """What to call `place` when speaking `language`: its word if one was
        given (`la table`), else `the <place>`."""
        for w in self.place_words:
            if w.place == place and w.language == language:
                return w.word.copy()
        return "the " + place

    def words_in(self, language: String) -> List[String]:
        """`place = word` for every known place that has a word in `language`."""
        var out = List[String]()
        for p in self.places:
            for w in self.place_words:
                if w.place == p.name and w.language == language:
                    out.append(p.name + " = " + w.word)
                    break
        return out^

    # ── the facts of this frame ───────────────────────────────────────

    def observe_pose(mut self, x: Float64, y: Float64, yaw_deg: Float64):
        self.pose_known = True
        self.x = x
        self.y = y
        self.yaw_deg = yaw_deg

    def observe_motion(mut self, doing: String, walking_to: String, at: String, now: Int):
        """What is true now ("" or `none` for nothing). Each CHANGE is
        recorded: a new command (`began`), a new destination (`going`), a
        place reached (`arrived`)."""
        var d = _or_none(doing)
        var w = _or_none(walking_to)
        var a = _or_none(at)
        if d != self.doing:
            self.doing = d
            if d != "":
                self.record(EV_BEGAN, d, String("motion"), now)
        if w != self.walking_to:
            self.walking_to = w
            if w != "":
                self.record(EV_GOING, w, String("motion"), now)
        if a != self.at:
            self.at = a
            if a != "":
                self.record(EV_ARRIVED, a, String("motion"), now)

    # ── what is derived ───────────────────────────────────────────────

    def around(self) -> List[PlaceAround]:
        """Each known place from where the robot stands; empty without a pose."""
        if not self.pose_known:
            return List[PlaceAround]()
        return places_around(self.places, self.x, self.y, self.yaw_deg)

    def around_lines(self) -> List[String]:
        """`G1Context.around`: the decision model's lines (`geometry.place_line`)."""
        if not self.pose_known:
            return List[String]()
        return place_lines(self.places, self.x, self.y, self.yaw_deg)

    def nearest(self) -> String:
        """The closest known place; "" without a pose or a place."""
        if not self.pose_known:
            return String("")
        return nearest_place(self.places, self.x, self.y)

    def trail(self) -> List[String]:
        """Places arrived at, most recent first, a repeat of the head not
        re-added, at most `WS_TRAIL_N`."""
        var all = List[String]()
        for e in self.events:
            if e.kind == EV_ARRIVED and (len(all) == 0 or all[0] != e.text):
                all.insert(0, e.text.copy())
        var out = List[String]()
        for i in range(min(len(all), WS_TRAIL_N)):
            out.append(all[i].copy())
        return out^

    def been_before(self) -> List[String]:
        """The trail without the place the robot stands in now: where it WAS."""
        var t = self.trail()
        if len(t) > 0 and self.at != "" and t[0] == self.at:
            _ = t.pop(0)
        return t^

    def last_of(self, kind: String, n: Int) -> List[String]:
        """The last `n` events of `kind`, most recent first."""
        var out = List[String]()
        var i = len(self.events) - 1
        while i >= 0 and len(out) < n:
            if self.events[i].kind == kind:
                out.append(self.events[i].text.copy())
            i -= 1
        return out^
