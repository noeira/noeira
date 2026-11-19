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

Movable objects (the crate) are `objects`, one `ObjectXY` record each, written
by `saw_object`, `told_object`, `object_missing`, `picked_up` and `put_down`
(`noeira.brain.objects` says how). The reply view does not read them yet.
"""

from .events import (
    BrainEvent, EventLog, EV_SESSION, EV_HEARD, EV_SAID, EV_BEGAN, EV_GOING, EV_ARRIVED,
    EV_MISSED, EV_SAW, EV_TOLD_WHERE, EV_LOST, EV_PICKED, EV_PUT,
)
from .objects import (
    ObjectXY, object_event_text, OBJ_FLOOR, OBJ_HELD, OBJ_PUT, HOW_SEEN, HOW_TOLD, OBJ_GAP_S,
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


struct Skill(Copyable, Movable):
    """One thing the body can (or cannot) do, in the words each reader needs:
    the reply view's English line, the French clause code says, the command the
    body runs, and the words that name it in a question (`noeira.brain.answers`)."""

    var en: String
    """Completes "You can ..." / "You cannot ..." (the reply view)."""
    var fr: String
    """Completes « Je peux … » / « je ne peux pas … »: `danser`, `m'asseoir`."""
    var command: String
    """What the body runs when it is asked « Tu peux … ? » or told to: a bank
    row or a named motion. "" = the decision says how (a place, a direction),
    or there is nothing to run (a limit, talking)."""
    var forms: List[String]
    """Words that name it, matched as phrases on normalised text (`bb1.has_phrase`):
    `danse`, `danser`, `dance`. Empty = never matched (asked about, the reply
    model answers)."""
    var to_places: Bool
    """Walking to a place: « aller au frigo, à la porte… », the places listed."""
    var motion: Bool
    """A named motion (`Body.add_motion`): the list of skills names two of them, not all."""

    def __init__(out self, var en: String, var fr: String, var command: String, var forms: List[String],
                 to_places: Bool = False, motion: Bool = False):
        self.en = en^
        self.fr = fr^
        self.command = command^
        self.forms = forms^
        self.to_places = to_places
        self.motion = motion


def known_motion(name: String) -> Bool:
    """Whether `motion_skill` has words for `name`."""
    return (name == "dance" or name == "wave" or name == "dab" or name == "stretch" or name == "jump"
            or name == "boxing" or name == "backflip")


def motion_skill(name: String) -> Skill:
    """A named motion of the body (SuperMimic's room: `wave`, `dab`, `dance`,
    `stretch`, `jump`, `boxing`, `backflip`), with its words. A name not in
    this table is run by its own name and named by it."""
    if name == "dance":
        return Skill(String("dance"), String("danser"), name, ["danse", "danser", "dance"])
    if name == "wave":
        return Skill(String("wave hello"), String("faire coucou"), name,
                     ["coucou", "salue", "saluer", "fais signe", "faire signe", "wave"])
    if name == "dab":
        return Skill(String("do a dab"), String("faire un dab"), name, ["dab"])
    if name == "stretch":
        return Skill(String("stretch"), String("m'étirer"), name, ["étire", "étirer", "stretch"])
    if name == "jump":
        return Skill(String("jump"), String("sauter"), name, ["saute", "sauter", "jump"])
    if name == "boxing":
        return Skill(String("shadow-box"), String("faire de la boxe"), name,
                     ["boxe", "boxer", "boxing", "shadow box", "shadow-box"])
    if name == "backflip":
        return Skill(String("do a backflip"), String("faire un salto"), name,
                     ["salto", "saut périlleux", "backflip", "back flip"])
    return Skill(name, name, name, [name])


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
    var skills: List[Skill]
    """The `can` lines that code answers for (`answers.capabilities_answer`,
    `answers.can_answer`), each with its words. Added by `add_skill`, which
    writes the `can` line too: one list, so the two cannot disagree."""
    var limits: List[Skill]
    """The same for `cannot` (`add_limit`)."""

    def __init__(out self, var name: String, var description: String):
        self.name = name^
        self.description = description^
        self.can = List[String]()
        self.cannot = List[String]()
        self.skills = List[Skill]()
        self.limits = List[Skill]()

    def add_skill(mut self, var s: Skill):
        """A thing it can do: its `can` line and its words."""
        self.can.append(s.en)
        self.skills.append(s^)

    def add_limit(mut self, var s: Skill):
        """A thing people ask for that it cannot do: its `cannot` line and its words."""
        self.cannot.append(s.en)
        self.limits.append(s^)

    def add_motion(mut self, name: String):
        """A named motion (`motion_skill`), after the movements and before
        talking (a skill with no command and no words): « danser » reads
        among the movements.

        ⚠ IT CHANGES THE REPLY VIEW'S `can` LINE, which BB-1 measured with
        `g1_room`'s five. Only a body that has the motions adds them."""
        var s = motion_skill(name)
        s.motion = True
        var at = len(self.skills)
        for i in range(len(self.skills)):
            if self.skills[i].command == "" and not self.skills[i].to_places and len(self.skills[i].forms) == 0:
                at = i
                break
        # ⚠ `can` and `skills` in the same order, so the view and the answers list them alike
        var k = len(self.can) - (len(self.skills) - at)
        self.can.insert(k, s.en)
        self.skills.insert(at, s^)

    @staticmethod
    def g1_room(name: String) -> Body:
        """The simulated G1 of the room demo: BFM-Zero's bank and the navigator.

        ⚠ THE `can` / `cannot` LINES ARE BB-1'S: the English words are the ones
        it was measured with (docs/ROBOT_BRAIN_PLAN.md §11.2)."""
        var b = Body(name, String("a small humanoid robot with two legs, two arms and a head"))
        b.add_skill(Skill(String("walk to a place of the room"), String("aller"), String(""), List[String](), True))
        b.add_skill(Skill(String("turn"), String("tourner"), String(""), ["tourner", "tourne", "turn"]))
        b.add_skill(Skill(String("squat"), String("faire un squat"), String("squat"),
                          ["squat", "accroupir", "accroupis", "accroupi"]))
        b.add_skill(Skill(String("raise your arms"), String("lever les bras"), String("both_hands_up"),
                          ["lever les bras", "lève les bras", "lever tes bras", "lève tes bras", "lever les deux bras",
                           "lève les deux bras", "raise your arms", "raise both arms", "arms up", "hands up"]))
        b.add_skill(Skill(String("talk and answer questions"), String("parler et répondre à tes questions"), String(""),
                          List[String]()))
        b.add_limit(Skill(String("pick things up or carry them"), String("prendre ou porter des objets"), String(""),
                          ["prendre", "porter", "attraper", "ramasser", "pick up", "pick things up", "carry", "grab"]))
        b.add_limit(Skill(String("open the door"), String("ouvrir la porte"), String(""),
                          ["ouvrir", "ouvre", "open"]))
        b.add_limit(Skill(String("sit down"), String("m'asseoir"), String(""),
                          ["asseoir", "assieds", "assois", "sit", "sit down"]))
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
    """As the body takes it: `goto:table`, `squat`, `pick:crate`, `put:crate:table`."""
    var label: String
    """In words, for the models: `walk to the table`."""
    var place: String
    """The place the step ends at; "" for a step that ends after `hold_s`.
    For a `put:` step, the place the object is to be put at."""
    var obj: String
    """The object a `pick:` or `put:` step is about; "" for the others. Such a
    step ends on the object's record (`noeira.brain.tasks`), not on `at`."""
    var timeout_s: Float64
    var hold_s: Float64
    var status: Int
    var t_start: Float64

    def __init__(out self, var command: String, var label: String, var place: String):
        self.command = command^
        self.label = label^
        self.place = place^
        self.obj = String("")
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

    @staticmethod
    def pick(obj: String) -> Step:
        """Pick an object up (`pick:crate`): done when it is held, failed
        after `timeout_s`, or at once when the state does not know where it is."""
        var s = Step("pick:" + obj, "pick up the " + obj, String(""))
        s.obj = obj.copy()
        return s^

    @staticmethod
    def put(obj: String, place: String) -> Step:
        """Put an object at a place (`put:crate:table`): done when it was put
        down there, failed after `timeout_s`. The body decides whether that
        includes walking there and picking it up first."""
        var s = Step("put:" + obj + ":" + place, "put the " + obj + " at the " + place, place.copy())
        s.obj = obj.copy()
        return s^


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
    them LFM3-1.7B called the table "le tableau" in 2 of 16 French answers.
    An object's words are kept here too, under its name (`crate` = `la caisse`)."""

    var objects: List[ObjectXY]
    """The movable objects the robot knows (`add_object`), one record each,
    where each is believed to be now (`noeira.brain.objects`)."""

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

    var speaker: String
    """Who is talking to the robot, by voice (`noeira.brain.people`); "" when
    the voice is unknown or no voice was heard yet."""
    var speaker_sure: Bool
    """The voice IS `speaker`'s; False = it only resembles it."""
    var speaker_score: Float64
    var speaker_since: Int
    """Epoch seconds: when `speaker` first gave their name."""
    var speaker_said_name: String
    """The name the utterance being answered gives for its speaker, before
    its voice is known; "" otherwise."""
    var speaker_pending: Bool
    """The utterance being answered has been heard and its voice is still
    being compared: who spoke is not known yet."""
    var people_on: Bool
    """A people store is in use: the reply view says who is talking, or that
    the robot does not know."""
    var people_known: List[String]
    """The names of the people the robot knows."""

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
        self.objects = List[ObjectXY]()
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
        self.speaker = String("")
        self.speaker_sure = False
        self.speaker_score = 0.0
        self.speaker_since = 0
        self.speaker_said_name = String("")
        self.speaker_pending = False
        self.people_on = False
        self.people_known = List[String]()
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

    def missed(mut self, seconds: Float64, now: Int):
        """Speech the robot did not listen to (`EV_MISSED`): its length only."""
        self.record(EV_MISSED, String(round(seconds, 1)) + String(" s"), String("voice"), now)

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

    # ── movable objects (`noeira.brain.objects`) ──────────────────────

    def object_index(self, name: String) -> Int:
        """The object called `name` in `objects`; -1 when it was not added."""
        for i in range(len(self.objects)):
            if self.objects[i].name == name:
                return i
        return -1

    def place_index(self, name: String) -> Int:
        """The place called `name` in `places`; -1 when there is none."""
        for i in range(len(self.places)):
            if self.places[i].name == name:
                return i
        return -1

    def add_object(mut self, var o: ObjectXY):
        """An object the robot is to know, with its size. One of the same name
        is replaced (pose and all)."""
        var i = self.object_index(o.name)
        if i >= 0:
            self.objects[i] = o^
        else:
            self.objects.append(o^)

    def _log_object(mut self, i: Int, kind: String, source: String, t: Float64, now: Int):
        self.objects[i].mark_logged(t)
        var text = object_event_text(self.objects[i])
        self.record(kind, text, source, now)

    def saw_object(mut self, name: String, x: Float64, y: Float64, yaw_deg: Float64, score: Float64,
                   source: String, t: Float64, now: Int) -> Bool:
        """A sighting: the object's pose REPLACES the record's (`how` = seen).
        Call it as often as the object is seen; a `saw` event is written only
        when the sighting is news (the module note). False = no object of that
        name was added: the size a fit was made with is the state's, so an
        undeclared object is not taken."""
        var i = self.object_index(name)
        if i < 0:
            return False
        var news = (
            self.objects[i].how != HOW_SEEN
            or self.objects[i].missing
            or self.objects[i].t_seen < 0.0
            or t - self.objects[i].t_seen > OBJ_GAP_S
        )
        self.objects[i].set_pose(x, y, yaw_deg)
        self.objects[i].how = String(HOW_SEEN)
        self.objects[i].source = source.copy()
        self.objects[i].score = score
        self.objects[i].missing = False
        self.objects[i].t_seen = t
        self.objects[i].now_seen = now
        self.objects[i].times_seen += 1
        if self.objects[i].where == OBJ_HELD:
            return True   # its pose follows the hands; written when it is put down
        var moved = self.objects[i].moved_since_logged()
        if moved and self.objects[i].where == OBJ_PUT:
            # it is no longer where the robot put it: someone moved it
            self.objects[i].where = OBJ_FLOOR
            self.objects[i].put_at = String("")
        if news or moved or self.objects[i].t_logged < 0.0:
            self._log_object(i, EV_SAW, source, t, now)
        return True

    def told_object(mut self, name: String, x: Float64, y: Float64, yaw_deg: Float64,
                    source: String, t: Float64, now: Int) -> Bool:
        """Where the object is, given rather than seen (`how` = told): a trials
        table, a scene file, a person. On the floor, at no place. False = not added."""
        var i = self.object_index(name)
        if i < 0:
            return False
        self.objects[i].set_pose(x, y, yaw_deg)
        self.objects[i].how = String(HOW_TOLD)
        self.objects[i].source = source.copy()
        self.objects[i].score = 0.0
        self.objects[i].missing = False
        self.objects[i].where = OBJ_FLOOR
        self.objects[i].put_at = String("")
        self.objects[i].t_seen = t
        self.objects[i].now_seen = now
        self._log_object(i, EV_TOLD_WHERE, source, t, now)
        return True

    def object_missing(mut self, name: String, t: Float64, now: Int) -> Bool:
        """The robot looked where the record puts the object and did not find
        it: `missing`, and a `lost` event with the pose it was believed at. A
        later sighting clears it. Only the first report is written. False =
        not added or no pose to have looked at."""
        var i = self.object_index(name)
        if i < 0 or not self.objects[i].has_pose:
            return False
        if self.objects[i].missing:
            return True
        self.objects[i].missing = True
        self._log_object(i, EV_LOST, String("brain"), t, now)
        return True

    def picked_up(mut self, name: String, t: Float64, now: Int) -> Bool:
        """The hands have the object (`OBJ_HELD`). False = not added."""
        var i = self.object_index(name)
        if i < 0:
            return False
        self.objects[i].where = OBJ_HELD
        self.objects[i].put_at = String("")
        self.objects[i].missing = False
        self._log_object(i, EV_PICKED, String("body"), t, now)
        return True

    def put_down(mut self, name: String, place: String, x: Float64, y: Float64, yaw_deg: Float64,
                 t: Float64, now: Int) -> Bool:
        """The object was let go at (x, y, heading): put at `place`
        (`OBJ_PUT`), or on the floor at no place when `place` is "". The pose
        is the body's estimate; the next sightings refine it. False = not added."""
        var i = self.object_index(name)
        if i < 0:
            return False
        self.objects[i].set_pose(x, y, yaw_deg)
        self.objects[i].where = OBJ_PUT if place != "" else OBJ_FLOOR
        self.objects[i].put_at = place.copy()
        self.objects[i].missing = False
        self.objects[i].now_seen = now
        self._log_object(i, EV_PUT, String("body"), t, now)
        return True

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
