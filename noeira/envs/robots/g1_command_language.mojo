# +--------------------------------------------------------------------------+ #
# | Language onto the command bank — one question, one decision rule
# +--------------------------------------------------------------------------+ #
"""Turn a sentence into a bank command, or refuse.

Two programs need this: `bfm_zero_say.mojo` (one shot, from the shell) and
`bfm_zero_voice_demo.mojo` (in a render loop). They must ask the IDENTICAL
question, and that is not a style preference — §12.54 measured the wording
moving `P(none)` by half on a correct instruction. Two copies would be two
different systems wearing one name.

THE DECISION RULE, AND WHY IT IS NOT THE ARGMAX
===============================================
§12.54, all measured against the live service:

- **The bar is `P(none)`, not the top-1 confidence.** "crouch down low"
  returned `crouch` 0.56 with `squat` 0.38 — the mass split between two
  CORRECT answers, because their bands are adjacent. A top-1 threshold
  punishes options for resembling each other rather than the instruction for
  being unclear. Abstention asks "does this robot HAVE such a command", and
  that is `P(none)`.

- **`none` winning the argmax does not mean the instruction is impossible.**
  "spin around fast" split 0.42 `spin_left` / 0.14 `spin_right`, so `none` at
  0.44 won while 0.56 of the mass sat on commands the robot has.
  Near-duplicates divide their own vote; `none` does not have to beat the
  field, only the sum of it. So: take the best REAL option, refuse on
  `P(none)`.

- **A second, lower bar on the best real option** catches a distribution
  spread flat across many commands — "it is a robot command but I cannot tell
  which", also not something to act on.

THE STATE IS HALF THE QUESTION
==============================
`g1_command_state` builds the JSON the questions are answered against:
`instruction`, `doing`, `doing_since`, `did_before`, `last_arm_raised`. Before
it, the demo sent the transcript as free text and "plus vite" after a walk was
refused — correctly, because the request did not say what was to be done
faster. A relative instruction is an edit to a command, and an edit needs
something to edit. §10.1 of `docs/BFM_ZERO_NEXT_LEVEL.md`.

TWO EXTRA QUESTIONS, BOTH NEARLY FREE
=====================================
Jev answers every question against the same state, so more questions cost
almost nothing — that is the shape of its cost model.

- `needs_world`: this robot has no object interaction and no navigation to a
  landmark. Rewording the choice to stop it over-picking `none` created the
  opposite error ("walk to the fridge and open it" went from refused to
  `walk` at 0.92), and this is what catches it. The caller decides whether to
  refuse or to do the part it can AND SAY SO.

- `addressed`: for an always-listening demo, "is this an instruction to the
  robot at all?" ⚠ A WAKE WORD CANNOT DO THIS JOB WELL. Whisper renders
  "Robot" as *Robo*, *Roberto*, *Robots*, with varying punctuation and
  across languages — and the French utterances this project produces would
  need their own spellings. Asking costs nothing and generalises.
"""

from noeira.ai.jev import JevQuestions, JevAnswers
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.io.json import JsonWriter

comptime G1_Q_COMMAND: String = "command"
# ⚠ THREE COMMANDS IN ONE CALL. Jev answers every question against the same
# state, so asking for a second and third step costs almost nothing beyond
# repeating the option list — that is the shape of its cost model, and it is
# what makes decomposition cheap here. "Marche un mètre et tourne sur
# toi-même" is two steps, and a single-choice reading of it has to throw one
# of them away.
comptime G1_Q_CMD2: String = "command2"
comptime G1_Q_CMD3: String = "command3"
comptime G1_Q_EXTENT: String = "extent"
comptime G1_MAX_STEPS: Int = 3
comptime G1_Q_NEEDS_WORLD: String = "needs_world"
comptime G1_Q_ADDRESSED: String = "addressed"
# ⚠ A DESTINATION IS NOT A COMMAND, AND IT MUST NOT REACH THE SPEC PATH.
# `z = E_rho[r(s) B(s)]` is an expectation over the state pool, and of the 463
# privileged numbers exactly ONE is a world quantity (`root_height`); the other
# 462 are in the heading frame. So `r(s) = -||root_xy - fridge_xy||` is not a
# hard reward, it is a MEANINGLESS one — it returns a unit-norm `z` and a robot
# that does something confident and wrong (§3.2 of BFM_ZERO_NEXT_LEVEL.md,
# §12.60). A destination therefore has to be routed to a NAVIGATOR before the
# cache-miss path can claim it, which is why this question is answered in the
# same call and checked before `P(none)`.
comptime G1_Q_DESTINATION: String = "destination"
# The yes/no gate of the REQUEST wording; see `g1_add_request_destination`.
comptime G1_Q_IS_REQUEST: String = "is_request"
comptime G1_OPT_NONE: String = "none"
# ⚠ `talk` IS AN OPTION, NOT A REFUSAL. "Bonjour, comment vas-tu ?" used to
# come back "not addressed to the robot" — technically true and completely
# wrong in effect: a robot that cannot say hello reads as one that is not
# listening. Offering it as a choice lets the same call that picks a movement
# decide that no movement was wanted, which is cheaper and more accurate than
# a second classifier.
comptime G1_OPT_TALK: String = "talk"

# ── aliases: more WORDS than there are behaviours ─────────────────────────
# ⚠ AN ALIAS IS NOT A BANK ENTRY, and the distinction is the point. A bank
# entry is a distinct behaviour that passed four gates; an alias is a word
# people use for one that already exists. "Baisse les bras" came back refused
# at P(none) 0.47 in a real session — not because the robot cannot lower its
# arms, but because `stand` ships with them at 0.78/0.78 and nobody calls
# that "arms down".
#
# Adding `arms_down` to the BANK would have been dishonest: it would fail the
# scaffold gate, which exists precisely to reject a command that does nothing
# its scaffold does not already do. Adding it here is honest — the word maps
# to a behaviour that genuinely has that property.
comptime G1_N_ALIAS: Int = 3


def g1_alias_name(i: Int) -> String:
    if i == 0:
        return String("arms_down")
    if i == 1:
        return String("stop")
    return String("turn_around")


def g1_alias_target(i: Int) -> String:
    if i == 0:
        return String("stand")
    if i == 1:
        return String("stand")
    return String("spin_left")


def g1_alias_desc(i: Int) -> String:
    if i == 0:
        return String("stand still with both arms hanging down at the sides")
    if i == 1:
        return String("stop moving and stand still")
    return String("turn on the spot to face the other way")


def g1_resolve(name: String) -> String:
    """An alias to the bank command it means, or the name unchanged."""
    for i in range(G1_N_ALIAS):
        if g1_alias_name(i) == name:
            return g1_alias_target(i)
    return name


# ── the conversational state ──────────────────────────────────────────────
# ⚠ THE STATE IS WHY A RELATIVE INSTRUCTION CAN WORK AT ALL. The demo used
# to send `"instruction: " + heard` as free text, and §10.1 of
# `docs/BFM_ZERO_NEXT_LEVEL.md` is what that cost: the user said "plus vite"
# after a walk and got a refusal. Jev was not failing to remember — THERE WAS
# NOTHING TO REMEMBER WITH. "plus vite" has no referent unless the state says
# what is being done faster, and no wording fixes a state that omits the
# subject of the sentence.
#
# So the state carries what the robot is doing, how long it has been doing it,
# what it did before that, and which arm was last raised. Nothing else: this
# is a classifier over a closed option set, and every field is one the model
# has to use to resolve a word like "encore" or "l'autre".
comptime G1_N_RECENT: Int = 3


def g1_arm_of(name: String) -> String:
    """Which arm a bank command raises: "left", "right", "both" or "".

    This is what makes "l'autre bras" answerable. It is derived from the
    bank's own names rather than stored, so a new arm command needs one line
    here and nothing else.
    """
    if name == "right_hand_up" or name == "walk_right_hand_up":
        return String("right")
    if name == "left_hand_up":
        return String("left")
    if (
        name == "both_hands_up"
        or name == "walk_both_hands_up"
        or name == "arms_wide"
    ):
        return String("both")
    return String("")


def g1_since_word(seconds: Float64) -> String:
    """How long the current command has been running, AS A WORD.

    ⚠ NOT A NUMBER, AND THIS IS NOT STYLE. Jev's own docs list arithmetic,
    counting and numeric comparison as failure modes, so a threshold belongs
    on our side of the wire (`docs/SYSTEM_ONE_ASSESSMENT.md`, and the same
    rule the `extent` question already follows). Sending `4.2` would ask the
    model to compare it against a bound it was never told.
    """
    if seconds < 1.5:
        return String("just_started")
    if seconds < 6.0:
        return String("a_moment")
    return String("a_while")


comptime G1_TRAIL_N: Int = 4
"""Places in `places_visited_recently`, most recent first."""
comptime G1_HEARD_N: Int = 3
"""Utterances in `recently_heard`, most recent first."""
comptime G1_LOG_N: Int = 64
"""Events `G1Context.log` keeps; the oldest go first."""


@fieldwise_init
struct G1Event(Copyable, Movable):
    """One thing the robot remembers: `arrived` (text = the place) or `heard`
    (text = an earlier utterance). `t_s` is the caller's clock in seconds;
    only its order is read."""

    var kind: String
    var text: String
    var t_s: Float64


struct G1Context(Copyable, Movable):
    """What the robot is doing, for the next instruction to be relative to."""

    var doing: String
    """The command running now; "" before anything has been asked."""
    var since_s: Float64
    """Seconds since `doing` started. Bucketed by `g1_since_word` on the way
    out — the caller keeps the clock, this keeps the word."""
    var recent: List[String]
    """MOST RECENT FIRST, `doing` excluded, at most `G1_N_RECENT`."""
    var places: Bool
    """Send the three place fields below. OFF by default: a caller with no
    world (the bank viewer, `g1say`) and the measured Jev path send exactly
    the state they always sent."""
    var walking_to: String
    """The destination being walked to now, "" for none."""
    var at: String
    """The destination the robot stands in now, "" for none."""
    var arrived: String
    """The destination it last arrived at, "" for none."""
    var geometry: Bool
    """Send `places_around_you` (`around`) and `places_visited_recently` (the
    trail, from `log`). OFF by default. RB-1's compact S1 (`s1c`): the largest
    effect that bench measured, spatial 19 -> 48/70 on hosted Jev
    (`docs/ROOM_BENCH_RB1_PLAN.md`, 2026-11-05/06)."""
    var around: List[String]
    """One line per place, rendered by the caller from the robot's pose:
    `"table: left, 3.5 m, next to sofa"`. The caller owns the geometry (and
    its buckets); this sends the words."""
    var memory: Bool
    """Send `recently_heard` (the last `G1_HEARD_N` utterances from `log`).
    OFF by default."""
    var log: List[G1Event]
    """Arrivals and earlier utterances, oldest first (`arrived_at`,
    `heard_said`). Read by the trail, `recently_heard` and `g1_other_one`."""

    def __init__(out self):
        self.doing = String("")
        self.since_s = 0.0
        self.recent = List[String]()
        self.places = False
        self.walking_to = String("")
        self.at = String("")
        self.arrived = String("")
        self.geometry = False
        self.around = List[String]()
        self.memory = False
        self.log = List[G1Event]()

    def _remember(mut self, kind: String, text: String, t_s: Float64):
        self.log.append(G1Event(kind, text, t_s))
        while len(self.log) > G1_LOG_N:
            _ = self.log.pop(0)

    def arrived_at(mut self, place: String, t_s: Float64):
        """The robot has just arrived at `place`."""
        self._remember(String("arrived"), place, t_s)

    def heard_said(mut self, utterance: String, t_s: Float64):
        """An utterance, logged AFTER the decision it was the instruction of:
        `recently_heard` is what was said before, never the sentence being
        decided."""
        self._remember(String("heard"), utterance, t_s)

    def trail(self) -> List[String]:
        """Arrivals, most recent first, a repeat of the head not re-added, at
        most `G1_TRAIL_N` (RB-1's reducer rule, `rb1_reduce`)."""
        var all = List[String]()
        for e in self.log:
            if e.kind == "arrived" and (len(all) == 0 or all[0] != e.text):
                all.insert(0, e.text)
        var out = List[String]()
        for i in range(min(len(all), G1_TRAIL_N)):
            out.append(all[i])
        return out^

    def heard_recent(self) -> List[String]:
        """The last `G1_HEARD_N` utterances, most recent first."""
        var out = List[String]()
        var i = len(self.log) - 1
        while i >= 0 and len(out) < G1_HEARD_N:
            if self.log[i].kind == "heard":
                out.append(self.log[i].text)
            i -= 1
        return out^

    def began(mut self, name: String):
        """Record that `name` has just started.

        ⚠ A REPEAT IS NOT A HISTORY ENTRY. Holding `walk` for thirty seconds
        while the user says "plus vite" twice must not push `walk` into
        `recent` three times and evict everything that came before it.
        """
        if self.doing != "" and self.doing != name:
            self.recent.insert(0, self.doing.copy())
            while len(self.recent) > G1_N_RECENT:
                _ = self.recent.pop()
        self.doing = name.copy()
        self.since_s = 0.0

    def last_arm(self) -> String:
        """The arm most recently raised, current command first."""
        var a = g1_arm_of(self.doing)
        if a != "":
            return a
        for i in range(len(self.recent)):
            a = g1_arm_of(self.recent[i])
            if a != "":
                return a
        return String("")


def g1_command_state(instruction: String, ref ctx: G1Context) raises -> String:
    """The JSON state for `JevClient.decide` / `.start`.

    ⚠ ONE COPY, for the same reason `g1_command_instruction` is one copy. The
    viewer and `bfm_zero_say.mojo` must send the same SHAPE as well as the
    same questions — a CLI that omits `doing` is asking a different question
    than the demo it stands in for, and its answer cannot be carried back.
    That has now cost this project three separate sessions.
    """
    var w = JsonWriter()
    w.begin_object()
    w.member("instruction", instruction)
    # ⚠ "nothing" RATHER THAN AN ABSENT FIELD. An option list that is always
    # the same length and a state whose keys are always present are what make
    # the answers comparable across calls; a missing key reads as a different
    # question.
    # ⚠ THE KEY'S NAME IS PART OF THE QUESTION. With the place fields, Kev-9B
    # read `doing: goto:sofa` + "the fridge is white" as a request (gate 0.93,
    # sofa 0.94) and `robot_is_doing: goto:sofa` as none (gate 0.06) — the
    # name the request wording was tuned with.
    w.member(
        "robot_is_doing" if ctx.places else "doing",
        ctx.doing if ctx.doing != "" else String("nothing"),
    )
    w.member("doing_since", g1_since_word(ctx.since_s))
    w.key("did_before")
    w.begin_array()
    for i in range(len(ctx.recent)):
        w.string(ctx.recent[i])
    w.end_array()
    w.member("last_arm_raised", ctx.last_arm() if ctx.last_arm() != "" else String("none"))
    # ⚠ THE REQUEST WORDING RESOLVES "back" AND "instead" AGAINST THESE, so
    # they are sent with it; the names are the ones that wording was tuned on.
    if ctx.places:
        w.member("currently_walking_to", ctx.walking_to if ctx.walking_to != "" else String("none"))
        w.member("currently_at", ctx.at if ctx.at != "" else String("none"))
        w.member("last_arrived_at", ctx.arrived if ctx.arrived != "" else String("none"))
    # ⚠ THESE KEYS AND THEIR ORDER ARE RB-1's `s1c` (+ mem-2's `recently_heard`),
    # byte for byte: `room_bench --check-state` compares this writer with the
    # bench's. Each key is present whenever its switch is on, empty or not.
    if ctx.geometry:
        w.key("places_around_you")
        w.begin_array()
        for line in ctx.around:
            w.string(line)
        w.end_array()
        w.key("places_visited_recently")
        w.begin_array()
        for t in ctx.trail():
            w.string(t)
        w.end_array()
    if ctx.memory:
        w.key("recently_heard")
        w.begin_array()
        for h in ctx.heard_recent():
            w.string(h)
        w.end_array()
    w.end_object()
    return w.done()


def g1_command_instruction() -> String:
    """⚠ THE ONE COPY OF THE WORDING.

    The first version ended "Choose `none` if the instruction asks for
    something not on the list", which reads as an invitation: "show me your
    left hand" came back `none` 0.52 against `left_hand_up` 0.48. Making
    `none` explicitly the exception took the same phrase to `left_hand_up`
    0.90 with `P(none)` 0.04. That single edit is larger than any threshold
    in this file.
    """
    # ⚠ "THE FIRST THING" IS LOAD-BEARING. Without it, a two-part
    # instruction returns whichever part the model finds most salient:
    # "marche un mètre puis tourne sur toi-même" came back `turn_around`,
    # and "accroupis-toi puis lève les deux bras et marche" came back
    # `walk_both_hands_up` — steps two and three merged into one. The
    # chain questions only work if this one is anchored.
    return String(
        "A humanoid robot can perform exactly the commands listed, and"
        " nothing else. Which command is the FIRST thing the instruction"
        " asks the robot to do? If the instruction asks for several things"
        " in sequence, answer with the first one only. It does not have to"
        " match exactly, only be the closest thing the robot can do. Pick"
        " `none` ONLY when no command on the list is even approximately"
        " what was asked."
        # ⚠ ADDITIVE, AND DELIBERATELY SO. §12.54 measured this wording
        # moving `P(none)` by half on a correct instruction, so the relative
        # clause is appended rather than woven in, and `--doing` on
        # `bfm_zero_say.mojo` exists to re-measure the absolute phrasings
        # against it.
        " The state also says what the robot is doing NOW, what it did"
        " before, and which arm it last raised. An instruction may be"
        " relative to that rather than self-contained — \"faster\","
        " \"again\", \"the other arm\", \"stop\" — and then the answer is"
        " the command that RESULTS from applying it to what the robot is"
        " already doing, not the command that describes the words."
    )


def g1_add_request_destination(
    mut q: JevQuestions, destinations: List[String]
) raises:
    """The destination asked as a REQUEST, and its yes/no gate
    (`G1_Q_IS_REQUEST`); `g1_route_destination` reads both."""
    var dops = List[String]()
    var ddesc = List[String]()
    for i in range(len(destinations)):
        dops.append(destinations[i])
        ddesc.append(
            String("the instruction asks the robot to go to the ")
            + destinations[i]
        )
    dops.append(String(G1_OPT_NONE))
    ddesc.append(String(
        "no request to go anywhere: a statement or a question about a"
        " place, a negation or a cancellation, thanks or a greeting, or"
        " a movement without a destination"
    ))
    # ⚠ A PLACE THAT IS MENTIONED IS NOT A PLACE THAT IS ASKED FOR.
    # On a local Jev-compatible model (Kev-9B) the default destination
    # question (`g1_command_questions`) answered "the door is made of wood" -> door and "where is the
    # door?" -> door: a keyword match, six wrong places in 24 traps.
    # Asking whether the sentence REQUESTS a place, and asking it a
    # second time as a yes/no question, took that set to 0 wrong
    # places (22/24), and a held-out set to 21/24, 0 wrong. Those
    # numbers were measured on this wording and these option texts;
    # an edit to either needs a re-measure.
    q.choice(
        String(G1_Q_DESTINATION),
        String(
            "Does the instruction REQUEST that the robot go to one of"
            " these places, and if so which one? Only a command or a"
            " stated need asks the robot to move (\"go to the table\","
            " \"I'm thirsty\" = the fridge). A sentence that only"
            " mentions or describes a place, or asks a question about"
            " it, is not a request: choose none. A negation or a"
            " cancellation (\"don't go to X\", \"not X\", \"cancel"
            " X\") is not a request to go to X: choose none. If the"
            " robot is already walking somewhere, a request for another"
            " place replaces it."
        ),
        dops, ddesc,
    )
    q.noul(
        String(G1_Q_IS_REQUEST),
        String(
            "Is the instruction a request for the robot to go to a"
            " place, rather than a statement, a question, a negation,"
            " a cancellation, thanks or a greeting?"
        ),
    )


def g1_destination_questions(destinations: List[String]) raises -> JevQuestions:
    """The FIRST of two requests (`G1VoiceConfig.request_wording`): the
    destination and its gate only. ⚠ ON A LOCAL MODEL EVERY QUESTION IS A ROW
    OF ITS OWN, and the command question is the longest one by far (28
    options); on Kev-9B on the Orin this request is ~0.9 s against ~4 s for
    the whole set. A goto is answered from it alone; the commands are asked
    only when it routes nothing.

    ⚠ THE DESTINATION IS THEREFORE DECIDED AHEAD OF `addressed` AND `talk`,
    and that is wanted: the gate already asks whether the sentence is a
    request rather than thanks, a greeting or a remark, and on Kev-9B
    `addressed` refused "go to the fridge" itself (0.45) while the robot was
    walking elsewhere."""
    var q = JevQuestions()
    g1_add_request_destination(q, destinations)
    return q^


def g1_route_destination(
    ref ans: JevAnswers,
    min_dest: Float64 = 0.40,
    max_dest_none: Float64 = 0.25,
    min_request: Float64 = 0.5,
) raises -> String:
    """The destination the request wording routes, or "": a pick other than
    `none`, top >= `min_dest`, P(none) <= `max_dest_none`, and the gate >=
    `min_request` — the rule that wording was measured with."""
    var pick = ans.choice(String(G1_Q_DESTINATION))
    if (
        pick != G1_OPT_NONE
        and ans.probability(String(G1_Q_DESTINATION), pick) >= min_dest
        and ans.probability(String(G1_Q_DESTINATION), String(G1_OPT_NONE)) <= max_dest_none
        and ans.noul(String(G1_Q_IS_REQUEST)) >= min_request
    ):
        return pick
    return String("")


comptime G1_Q_IS_GO_MEMORY: String = "is_go_memory"
comptime G1_Q_MEMORY_REF: String = "memory_ref"


def g1_add_memory_reference(mut q: JevQuestions) raises:
    """Which REMEMBERED place a request means: RB-1's memory parser route
    (`docs/ROOM_BENCH_RB1_PLAN.md`, 2026-11-06). The model names the reference;
    code computes the place from `G1Context.log` (`g1_route_memory`).

    ⚠ THE WORDING AND THE OPTIONS ARE THE MEASURED ONES (room_bench `mem2p`,
    frozen bb824a6d): on the held-out memory set "the other one" went 2 -> 8/8
    (hosted Jev) and 0 -> 4/8 (d1), every routed answer right. Five options
    are asked although only `other_one` is acted on (`g1_route_memory`)."""
    q.noul(
        String(G1_Q_IS_GO_MEMORY),
        String(
            "Is the instruction telling the robot to move on to a place now, even if it"
            " does not name the place (back, on, the next one, the other one), rather"
            " than a question, a remark, or a story about what happened?"
        ),
    )
    var opts = List[String]()
    var descs = List[String]()
    opts.append(String("previous_stop"))
    descs.append(String("go back to the place the robot was at just before the one it is at now"))
    opts.append(String("starting_point"))
    descs.append(String("go back to the first place the robot went to, where it started"))
    opts.append(String("other_one"))
    descs.append(String("go to the other place, of two mentioned earlier, that the robot did not take"))
    opts.append(String("next_errand"))
    descs.append(String("go on to the next place of an errand list that is not done yet"))
    opts.append(String(G1_OPT_NONE))
    descs.append(String("none of these: the instruction names its place, or asks for no remembered place"))
    q.choice(
        String(G1_Q_MEMORY_REF),
        String("Which remembered place does the instruction ask the robot to go to?"),
        opts, descs,
    )


def g1_place_mentions(
    text: String, places: List[String], words: List[String]
) -> List[String]:
    """The places `text` names, in the order they first appear: each place's
    own name or one of its `words[i]` (comma-separated, "" for none), whole
    words, a plural `s` allowed, lowercased. Read on the robot's MEMORY (what
    was heard), never on the instruction being decided — on the instruction
    this is the keyword baseline the model is measured against."""
    var clean = String(" ")
    for cp in text.lower().codepoint_slices():
        var c = String(cp)
        if c == "," or c == "." or c == "!" or c == "?" or c == ":" or c == ";" or c == "'" or c == "’" or c == "\"" or c == "-":
            clean += " "
        else:
            clean += c
    clean += " "
    var at = List[Int]()
    var names = List[String]()
    for i in range(len(places)):
        var forms = List[String]()
        forms.append(places[i])
        if i < len(words) and words[i] != "":
            for w in words[i].split(","):
                forms.append(String(w))
        var first = -1
        for w in forms:
            for f in [String(" ") + w + " ", String(" ") + w + "s "]:
                var k = clean.find(f)
                if k >= 0 and (first < 0 or k < first):
                    first = k
        if first >= 0:
            var j = len(at)
            while j > 0 and at[j - 1] > first:
                j -= 1
            at.insert(j, first)
            names.insert(j, places[i])
    return names^


def g1_other_one(
    ref ctx: G1Context, places: List[String], words: List[String]
) -> String:
    """"The other one": the latest heard utterance that names two places or
    more; of those, the ones that are not where the robot is, not where it is
    walking, and not arrived at since it was heard. Exactly one, or "" (the
    log does not settle it, and the caller falls through to the commands)."""
    var i = len(ctx.log) - 1
    while i >= 0:
        if ctx.log[i].kind == "heard":
            var named = g1_place_mentions(ctx.log[i].text, places, words)
            if len(named) >= 2:
                var left = List[String]()
                for p in named:
                    if p == ctx.at or p == ctx.walking_to:
                        continue
                    var taken = False
                    for j in range(i + 1, len(ctx.log)):
                        if ctx.log[j].kind == "arrived" and ctx.log[j].text == p:
                            taken = True
                    if not taken:
                        left.append(p)
                return left[0] if len(left) == 1 else String("")
        i -= 1
    return String("")


def g1_route_memory(
    ref ans: JevAnswers, ref ctx: G1Context, places: List[String],
    words: List[String], min_gate: Float64 = 0.5,
) raises -> String:
    """The place a memory reference routes, or "": the gate >= `min_gate`, the
    reference `other_one`, and `g1_other_one` settles it.

    ⚠ ONLY `other_one`. RB-1 measured the others too, and they stay off here:
    `previous_stop` sent the robot to the wrong place on both anchored items
    ("go back to where you were before the table": it has no anchor), and a
    robot with no errand list or object sightings has nothing for
    `next_errand` or a where-question to read."""
    if ans.noul(String(G1_Q_IS_GO_MEMORY)) < min_gate:
        return String("")
    if ans.choice(String(G1_Q_MEMORY_REF)) != "other_one":
        return String("")
    return g1_other_one(ctx, places, words)


# ── the triage (RB-2) ─────────────────────────────────────────────────────
# ⚠ WHAT THE SENTENCE ASKS, BEFORE WHERE IT POINTS. RB-1 ended with no model
# safe: "climb onto the table" went to the table, "go to the one on MY left"
# picked the robot's left, "did I ask you to go to the door?" walked. The
# destination question cannot see these, because each one does name a place.
# One more choice in the SAME first request names the kind of sentence, and
# code acts on it (`docs/ROOM_BENCH_RB2_PLAN.md` §6-7, 2026-11-09):
#   hosted Jev, `G1_TRIAGE_FULL`: held-out dangerous answers 17 -> 4 / 72,
#     RB-1's used sets 221 -> 248 / 276 exact, dangerous 20 -> 2;
#   d1, `G1_TRIAGE_NARROW`: held-out 19 -> 11, RB-1's sets 199 -> 204.
# The wordings below are the measured ones (room_bench `--state s2`, frozen
# e7c388dc / fefbc522); an edit needs a re-measure, on NEW held-out items.

comptime G1_TRIAGE_OFF: Int = 0
comptime G1_TRIAGE_FULL: Int = 1
"""Every kind acts, by probability MASS (hosted Jev)."""
comptime G1_TRIAGE_NARROW: Int = 2
"""Only the kinds d1 reads reliably act: top unsafe / speaker_side >= 0.4,
top act_here >= 0.5. d1's kinds are diffuse: on RB-1's sets `talk` and
`unsettled` refused plain spatial requests and the trail relation sent
anchored items to the previous stop, so those stay off for it."""

comptime G1_Q_KIND: String = "kind"
comptime G1_Q_TRAIL_REL: String = "trail_relation"
comptime G1_Q_TRAIL_ANCHOR: String = "trail_anchor"
comptime G1_KIND_TALK: String = "unsafe,speaker_side,unsettled,talk"
"""The kinds that move nothing: the robot answers, or asks back."""


def g1_add_triage(mut q: JevQuestions, name_desc: Bool = False) raises:
    """The triage choice, appended to the destination request
    (`g1_destination_questions`): no extra request. `name_desc`: each
    description starts with its option's name (d1, as `g1_command_questions`)."""
    var opts = List[String]()
    var descs = List[String]()
    opts.append(String("go_place"))
    descs.append(String(
        "it asks the robot to go to one of the places now, named, or described from the robot's own"
        " position or from the room, or the other one of two places just mentioned (recently_heard)"
    ))
    opts.append(String("remembered"))
    descs.append(String(
        "it asks the robot to go back to a place on its recent trail (places_visited_recently):"
        " where it was before or after somewhere, its last stop, where it started"
    ))
    opts.append(String("act_here"))
    descs.append(String(
        "it asks for a step, a turn, a gesture, a stop or a wait where the robot is, not for a place"
    ))
    opts.append(String("unsafe"))
    descs.append(String(
        "it asks for something a walking robot cannot or must not do: acrobatics (a flip or"
        " somersault, a cartwheel, a handstand, the splits), climbing, standing or jumping on or over"
        " furniture, sitting on top of something not made for sitting, lying down, pushing or lifting"
        " furniture. Going to sit on a seat is not unsafe"
    ))
    opts.append(String("speaker_side"))
    descs.append(String(
        "it picks the place by the SPEAKER's own left, right, front or back, which the robot does"
        " not know"
    ))
    opts.append(String("unsettled"))
    descs.append(String(
        "it asks the robot to go to a place the state cannot settle: one from a time the state does"
        " not cover (yesterday, this morning, last week: the trail has no dates), one someone else"
        " chose, or one that fits none of the places"
    ))
    opts.append(String("talk"))
    descs.append(String(
        "it asks the robot to do nothing: a question, a remark, a story about what happened, or what"
        " someone else is doing or will do"
    ))
    if name_desc:
        for i in range(len(opts)):
            descs[i] = opts[i] + ": " + descs[i]
    q.choice(
        String(G1_Q_KIND),
        String("What does the instruction ask of the robot, as the speaker finally means it?"),
        opts, descs,
    )


struct G1Triage(Copyable, Movable):
    """What the triage decided. `kind` is "" when it does not act (the
    destination and the commands decide as before)."""

    var kind: String
    """A `G1_KIND_TALK` kind, `remembered`, `act_here`, `go_place`, or ""."""
    var p: Float64
    """The probability `kind` acted on (the mass, for a talk kind under FULL)."""
    var top: String
    var top_p: Float64
    var talk_mass: Float64
    var place_mass: Float64
    """P(go_place + remembered + unsettled): the sentence is about going to a
    place. FULL only (0 under NARROW)."""

    def __init__(out self):
        self.kind = String("")
        self.p = 0.0
        self.top = String("")
        self.top_p = 0.0
        self.talk_mass = 0.0
        self.place_mass = 0.0

    def talks(self) -> Bool:
        for t in String(G1_KIND_TALK).split(","):
            if String(t) == self.kind:
                return True
        return False


def g1_route_triage(ref ans: JevAnswers, policy: Int) raises -> G1Triage:
    """The triage's decision under `policy` (`G1_TRIAGE_FULL` / `_NARROW`).

    ⚠ FULL READS MASSES, NOT THE TOP KIND ALONE: kinds that agree on what to do
    split their probability ("before the sofa" with no sofa on the trail:
    remembered .45 / unsettled .5, neither >= .5, and it fell through to
    `back`)."""
    var t = G1Triage()
    if policy == G1_TRIAGE_OFF:
        return t^
    var k = String(G1_Q_KIND)
    t.top = ans.choice(k)
    t.top_p = ans.probability(k, t.top)
    if policy == G1_TRIAGE_NARROW:
        if (t.top == "unsafe" or t.top == "speaker_side") and t.top_p >= 0.4:
            t.kind = t.top
            t.p = t.top_p
        elif t.top == "act_here" and t.top_p >= 0.5:
            t.kind = t.top
            t.p = t.top_p
        return t^
    for name in String(G1_KIND_TALK).split(","):
        t.talk_mass += ans.probability(k, String(name))
    t.place_mass = (ans.probability(k, String("go_place"))
                    + ans.probability(k, String("remembered"))
                    + ans.probability(k, String("unsettled")))
    if t.top == "remembered" and t.top_p >= 0.5:
        t.kind = t.top
        t.p = t.top_p
    elif t.talk_mass >= 0.5:
        # the talk kind named is the top one when it is a talk kind, else `talk`
        t.kind = String("talk") if (
            t.top == "go_place" or t.top == "remembered" or t.top == "act_here"
        ) else t.top
        t.p = t.talk_mass
    elif t.top_p >= 0.5:
        t.kind = t.top
        t.p = t.top_p
    return t^


def g1_trail_questions(places: List[String], name_desc: Bool = False) raises -> JevQuestions:
    """The trail request, asked only when the triage says `remembered`: the
    model names the relation and the landmark, `g1_resolve_trail` computes the
    place from `G1Context.log` (the ordering stays in code)."""
    var q = JevQuestions()
    var opts = List[String]()
    var descs = List[String]()
    opts.append(String("previous_stop"))
    descs.append(String(
        "the place the robot was at just before the one it is at now, or last left (\"where you were"
        " before\", \"your last stop\", \"the place before this one\")"
    ))
    opts.append(String("starting_point"))
    descs.append(String(
        "the FIRST place on the robot's trail (\"where you started\", \"the first place you went to\")"
    ))
    opts.append(String("before_landmark"))
    descs.append(String(
        "the place the robot was at just BEFORE a place the instruction names (also: the place between"
        " two visits to it)"
    ))
    opts.append(String("after_landmark"))
    descs.append(String("the place the robot went to just AFTER a place the instruction names"))
    opts.append(String("named"))
    descs.append(String("the place the instruction names itself, as the place to go back to"))
    if name_desc:
        for i in range(len(opts)):
            descs[i] = opts[i] + ": " + descs[i]
    q.choice(
        String(G1_Q_TRAIL_REL),
        String("Which place of its trail does the instruction send the robot back to?"),
        opts, descs,
    )
    var aops = places.copy()
    var adesc = List[String]()
    for p in places:
        adesc.append(String("the ") + p + " is the landmark (before or after the " + p + ")")
    aops.append(String("none"))
    adesc.append(String("the instruction names no landmark"))
    if name_desc:
        for i in range(len(aops)):
            adesc[i] = aops[i] + ": " + adesc[i]
    q.choice(
        String(G1_Q_TRAIL_ANCHOR),
        String("If the instruction places its target BEFORE or AFTER another place, which place is that?"),
        aops, adesc,
    )
    return q^


def g1_trail_arrivals(ref ctx: G1Context) -> List[String]:
    """Every arrival in the log, OLDEST first, a repeat in a row collapsed
    (uncapped, unlike `trail()`)."""
    var out = List[String]()
    for e in ctx.log:
        if e.kind == "arrived" and (len(out) == 0 or out[len(out) - 1] != e.text):
            out.append(e.text)
    return out^


def g1_resolve_trail(rel: String, anchor: String, ref ctx: G1Context) -> String:
    """The place a trail reference names, or "" when the log does not settle
    it (the caller asks back): `previous_stop` = the first trail entry that is
    not where the robot is; `starting_point` = the oldest trail entry, unless
    the robot is there; `before_landmark` / `after_landmark` = the arrival just
    before / after the LATEST arrival at `anchor`, unless that is where the
    robot is. `named` is the caller's (the destination the first request
    routed)."""
    var tr = ctx.trail()
    if rel == "previous_stop":
        for t in tr:
            if t != ctx.at:
                return t
        return String("")
    if rel == "starting_point":
        if len(tr) == 0:
            return String("")
        var first = tr[len(tr) - 1]
        return first if first != ctx.at else String("")
    if anchor == "none" or anchor == "":
        return String("")
    var arr = g1_trail_arrivals(ctx)
    var k = -1
    for i in range(len(arr)):
        if arr[i] == anchor:
            k = i
    if k < 0:
        return String("")
    if rel == "before_landmark":
        return arr[k - 1] if k > 0 and arr[k - 1] != ctx.at else String("")
    if rel == "after_landmark":
        return arr[k + 1] if k + 1 < len(arr) and arr[k + 1] != ctx.at else String("")
    return String("")


def g1_command_moves(ref bank: G1CommandBank, name: String) -> Bool:
    """Whether a command (or an alias, by its target) moves the robot: bank
    group `locomotion` or `combo`."""
    var i = bank.find(g1_resolve(name))
    if i < 0:
        return False
    var g = bank.group_at(i)
    return g == "locomotion" or g == "combo"


def _g1_command_options(
    ref bank: G1CommandBank,
    only: List[String],
    rotate: Int,
    mut options: List[String],
    mut descs: List[String],
    name_desc: Bool = False,
) raises:
    """The command question's options: the real ones (bank rows + aliases,
    `only` keeps just those named), rotated left by `rotate`, then `talk` and
    `none`. The one copy of that list and its descriptions.

    `name_desc`: each description starts with its option's name
    (`left_hand_up: arms: left_hand_height above 0.93, ...`). ⚠ A model that
    shows an option by its description ALONE never reads the name: d1
    (LiquidAI's `prompt.py`, `_option_line`, style `desc` by default) saw
    only the bank's predicates and picked the three plain-English aliases
    for clear commands (RB-1, 11-05). Its `name_desc` style writes exactly
    `name: description`, which this reproduces through the description."""
    var real = List[String]()
    var real_descs = List[String]()
    for i in range(bank.count()):
        real.append(bank.name_at(i))
        real_descs.append(bank.describe(i))
    # the aliases are offered as ordinary options; `g1_decide` resolves them
    for i in range(G1_N_ALIAS):
        real.append(g1_alias_name(i))
        real_descs.append(g1_alias_desc(i))
    if len(only) > 0:
        var kept = List[String]()
        var kept_descs = List[String]()
        for i in range(len(real)):
            for n in only:
                if n == real[i]:
                    kept.append(real[i])
                    kept_descs.append(real_descs[i])
                    break
        if len(kept) != len(only):
            raise Error("g1 command options: `only` names an option that does not exist")
        real = kept^
        real_descs = kept_descs^
    var n_real = len(real)
    for i in range(n_real):
        var j = (i + rotate) % n_real if n_real > 0 else 0
        if j < 0:
            j += n_real
        options.append(real[j])
        descs.append(real[j] + ": " + real_descs[j] if name_desc else real_descs[j])
    # `none` last and ALWAYS present: without it the model must pick
    # something, and "something" for an impossible request is a real command
    # the robot will actually run.
    options.append(String(G1_OPT_TALK))
    descs.append(String(
        "the speaker is talking TO the robot but not asking it to move —"
        " a greeting, a question, a remark. Answer in words, not motion."
    ))
    options.append(String(G1_OPT_NONE))
    descs.append(String("not one of these, or not a command for this robot"))


def g1_add_command_choice(
    mut q: JevQuestions,
    ref bank: G1CommandBank,
    id: String,
    only: List[String] = List[String](),
    rotate: Int = 0,
    name_desc: Bool = False,
) raises:
    """The command question again under another `id`, over `only` (with
    `talk` and `none`), for a caller that splits the option list across
    questions; `g1_decide(command_id=id)` reads it."""
    var options = List[String]()
    var descs = List[String]()
    _g1_command_options(bank, only, rotate, options, descs, name_desc)
    q.choice(id, g1_command_instruction(), options, descs)


def g1_command_questions(
    ref bank: G1CommandBank,
    with_addressed: Bool = False,
    destinations: List[String] = List[String](),
    dest_descs: List[String] = List[String](),
    with_chain: Bool = True,
    rotate: Int = 0,
    only: List[String] = List[String](),
    name_desc: Bool = False,
) raises -> JevQuestions:
    """The choice over the bank, plus the two cheap guards.

    `with_chain=False` drops the second and third command questions, for a
    caller that runs the first step only — on a local model each question is
    a row of its own, and those two are the longest.

    `rotate=k` rotates the real options (bank rows + aliases) left by k;
    `talk` and `none` stay last. 0, the default, is the request every measured
    number used. It exists for order-sensitivity tests: an answer that changes
    with the option order is a flip.

    `only`: offer just these real options (bank rows or aliases, kept in bank
    order), for a caller that splits the list (`g1_add_command_choice` asks
    the rest). Empty, the default, offers all of them.

    `name_desc`: each real option's description starts with its name
    (`_g1_command_options`). Off by default: every measured number used the
    descriptions alone.

    ⚠ The option descriptions are GENERATED from each compound
    (`bank.describe`). A hand-written blurb would drift from the terms the
    command was gated on, and the drift is invisible — the robot doing one
    thing while the model was told another.
    """
    var options = List[String]()
    var descs = List[String]()
    _g1_command_options(bank, only, rotate, options, descs, name_desc)

    var q = JevQuestions()
    q.choice(String(G1_Q_COMMAND), g1_command_instruction(), options, descs)
    # ⚠ THE SAME OPTION LIST, so a second step is chosen from exactly what
    # the robot can do. `none` on step 2 means the instruction was one
    # command, which is the common case and must stay cheap.
    if with_chain:
        q.choice(
            String(G1_Q_CMD2),
            String(
                "If the instruction asks the robot to do a SECOND thing after the"
                " first, which command is it? Pick `none` when the instruction"
                " asks for only one thing."
            ),
            options, descs,
        )
        q.choice(
            String(G1_Q_CMD3),
            String(
                "If the instruction asks for a THIRD thing after the second,"
                " which command is it? Pick `none` otherwise."
            ),
            options, descs,
        )
    # ⚠ THE EXTENT IS ORDERED, WHICH IS WHY IT IS A `score`. "a metre" and
    # "ten metres" are not unrelated labels, and the answer is the EXPECTED
    # level — a float between them — so "a couple of metres" lands between
    # the two rather than being forced onto one.
    var levels = List[String]()
    levels.append(String("barely at all — about half a metre, or one second"))
    levels.append(String("a little — about one metre, or two seconds"))
    levels.append(String("a moderate amount — about two metres, or three seconds"))
    levels.append(String("a long way — about five metres, or six seconds"))
    levels.append(String("a very long way — about ten metres, or twelve seconds"))
    q.score(
        String(G1_Q_EXTENT),
        String(
            "How far, or for how long, should the robot carry out the FIRST"
            " action? Judge from the instruction; when it says nothing about"
            " distance or duration, answer `a moderate amount`."
        ),
        levels,
    )
    q.noul(
        String(G1_Q_NEEDS_WORLD),
        String(
            "Does the instruction ask the robot to interact with an object,"
            " or to go to a particular place or thing?"
        ),
    )
    # ⚠ THE DESTINATION QUESTION IS ADDITIVE AND ABSENT BY DEFAULT, and that
    # is a safety property rather than tidiness. §12.54 measured the `command`
    # WORDING moving `P(none)` by half, and every question in the request is
    # part of the context the others are answered against — so a caller with
    # no destinations (the bare viewer, `g1say`) sends exactly the request it
    # sent before and its measured numbers still hold. Only a caller that HAS
    # a world to navigate pays for the extra option list.
    #
    # The options come from the caller because they are ITS scene: the room
    # session reads them from its `.task` `language=` lines, which keeps the
    # names it offers the model identical to the names its planner resolves —
    # the same rule that makes `bank.describe` generated rather than written.
    if len(destinations) > 0:
        if len(dest_descs) != 0 and len(dest_descs) != len(destinations):
            raise Error(
                "g1 destination: one description per destination, or none"
            )
        var dops = List[String]()
        var ddesc = List[String]()
        for i in range(len(destinations)):
            dops.append(destinations[i])
            ddesc.append(
                dest_descs[i] if len(dest_descs) > 0 else destinations[i]
            )
        # ⚠ `none` LAST AND ALWAYS, for the same reason the command question
        # has one: without it the model must name a place, and "lève le bras
        # droit" would acquire a destination.
        dops.append(String(G1_OPT_NONE))
        ddesc.append(String(
            "the instruction names no place or thing to go to"
        ))
        q.choice(
            String(G1_Q_DESTINATION),
            String(
                "Does the instruction tell the robot to GO somewhere, or to a"
                " particular object? If so, which one? Pick `none` when it"
                " asks for a movement or a posture rather than a destination."
            ),
            dops, ddesc,
        )
    if with_addressed:
        q.noul(
            String(G1_Q_ADDRESSED),
            # ⚠ "AIMED AT", NOT "AN INSTRUCTION". The first wording asked
            # whether this was "an instruction addressed to the robot, asking
            # it to do something now" — and a greeting is NOT an instruction,
            # so "Bonjour, comment vas-tu ?" came back 0.06 and was refused
            # before `talk` could ever be reached. The question this gate
            # needs to ask is whether the speaker is addressing the ROBOT at
            # all; what they want from it is what every other question is
            # for.
            String(
                "Is this speech aimed at the robot — an instruction, a"
                " question, or a greeting directed at it? Answer no for"
                " background conversation, for someone talking to another"
                " person, and for speech not aimed at the robot."
            ),
        )
    return q^


struct G1LangPick(Copyable, Movable):
    var name: String
    """The command to run, or "" when refused."""
    var best: String
    """The top real option ALWAYS, refused or not. `name` is what to act on;
    this is what to show — "refused, and it was leaning towards `walk`" is a
    far more useful line in a HUD or a log than a blank."""
    var conf: Float64
    var p_none: Float64
    var needs_world: Float64
    var addressed: Float64
    var reason: String
    """Empty when accepted; otherwise why not, in words for a HUD."""
    var talk: Bool
    """The speaker wanted an answer, not a movement. `name` is empty and
    `reason` is not set — this is a success with a different handler."""
    var destination: String
    """A place the instruction named, or "". Like `talk`, this is a SUCCESS
    WITH A DIFFERENT HANDLER: `name` is empty and a navigator takes over.
    Empty unless the caller offered destinations."""
    var dest_conf: Float64

    def __init__(out self):
        self.name = String("")
        self.best = String("")
        self.conf = 0.0
        self.p_none = 1.0
        self.needs_world = 0.0
        self.addressed = 1.0
        self.reason = String("")
        self.talk = False
        self.destination = String("")
        self.dest_conf = 0.0


def g1_command_phrase(name: String) -> String:
    """What the robot says when it starts a command.

    ⚠ IT USED TO SAY THE COMMAND'S NAME. `say "spin_left"` pronounces the
    underscore, takes 2.79 s for two syllables of content, and sounds like a
    machine reading a variable back. These are one line each and cost
    nothing — no second round trip before the robot acknowledges, and no
    tokens per command, which an LLM-written confirmation would spend on
    every single movement.

    English here because the tree is English; `--phrases FILE` overrides the
    table with `name=phrase` lines, and that file is data rather than source.
    """
    if name == "walk":
        return String("walking")
    if name == "run":
        return String("running")
    if name == "spin_left":
        return String("turning left")
    if name == "spin_right":
        return String("turning right")
    if name == "strafe_left":
        return String("stepping to my left")
    if name == "strafe_right":
        return String("stepping to my right")
    if name == "diagonal":
        return String("walking diagonally")
    if name == "stand":
        return String("standing still")
    if name == "squat":
        return String("squatting")
    if name == "crouch":
        return String("crouching")
    if name == "right_hand_up":
        return String("raising my right arm")
    if name == "left_hand_up":
        return String("raising my left arm")
    if name == "both_hands_up":
        return String("raising both arms")
    if name == "arms_wide":
        return String("arms out wide")
    if name == "look_left":
        return String("looking left")
    if name == "look_right":
        return String("looking right")
    if name == "walk_right_hand_up":
        return String("walking with my right arm up")
    if name == "walk_both_hands_up":
        return String("walking with both arms up")
    if name == "spin_arms_in":
        return String("spinning with my arms in")
    return name


struct G1ChainStep(Copyable, Movable):
    var name: String
    var conf: Float64

    def __init__(out self, name: String, conf: Float64):
        self.name = name
        self.conf = conf


def g1_extent_metres(level: Float64) -> Float64:
    """The `extent` score as a distance, interpolated between the levels it
    was described with. ⚠ Approximate by construction: the robot's distance
    is integrated from its own commanded velocity with no odometry, so "a
    metre" means about a metre."""
    var t = level
    if t < 0.0:
        t = 0.0
    if t > 4.0:
        t = 4.0
    var a = List[Float64]()
    a.append(0.5)
    a.append(1.0)
    a.append(2.0)
    a.append(5.0)
    a.append(10.0)
    var i = Int(t)
    if i >= 4:
        return a[4]
    return a[i] + (a[i + 1] - a[i]) * (t - Float64(i))


def g1_extent_seconds(level: Float64) -> Float64:
    var t = level
    if t < 0.0:
        t = 0.0
    if t > 4.0:
        t = 4.0
    var a = List[Float64]()
    a.append(1.0)
    a.append(2.0)
    a.append(3.0)
    a.append(6.0)
    a.append(12.0)
    var i = Int(t)
    if i >= 4:
        return a[4]
    return a[i] + (a[i + 1] - a[i]) * (t - Float64(i))


def g1_decide_chain(
    ref ans: JevAnswers,
    ref bank: G1CommandBank,
    min_step: Float64 = 0.35,
) raises -> List[G1ChainStep]:
    """Steps 2 and 3, in order. Step 1 comes from `g1_decide`.

    ⚠ A LATER STEP NEEDS A CLEARER ANSWER THAN THE FIRST, not a looser one.
    Errors compound down a chain and nothing checks them: a wrong second step
    runs after a correct first one and looks like the robot misunderstanding
    the whole sentence. When in doubt, do less.
    """
    var out = List[G1ChainStep]()
    var ids = List[String]()
    ids.append(String(G1_Q_CMD2))
    ids.append(String(G1_Q_CMD3))
    for k in range(len(ids)):
        var best = String("")
        var bconf = 0.0
        for i in range(bank.count() + G1_N_ALIAS):
            var nm = bank.name_at(i) if i < bank.count() else g1_alias_name(
                i - bank.count()
            )
            var p = ans.probability(ids[k], nm)
            if p > bconf:
                bconf = p
                best = g1_resolve(nm)
        var p_none = ans.probability(ids[k], String(G1_OPT_NONE))
        # ⚠ STOP AT THE FIRST GAP. A third step without a second is not a
        # chain, it is a misread — and running it would reorder the
        # instruction.
        if best == "" or bconf < min_step or p_none > bconf:
            break
        out.append(G1ChainStep(best, bconf))
    return out^


def g1_decide(
    ref ans: JevAnswers,
    ref bank: G1CommandBank,
    max_none: Float64 = 0.25,
    min_top: Float64 = 0.35,
    min_addressed: Float64 = 0.5,
    with_addressed: Bool = False,
    with_destination: Bool = False,
    min_world: Float64 = 0.5,
    min_dest: Float64 = 0.5,
    refuse_placeless: Bool = False,
    dest_decided: Bool = False,
    command_id: String = String(G1_Q_COMMAND),
) raises -> G1LangPick:
    """Apply the rule above. Returns an empty `name` with a `reason` set when
    the honest answer is to do nothing.

    `dest_decided`: the destination was asked in an EARLIER request
    (`g1_destination_questions`) and routed nothing, so these answers carry
    no destination question but `refuse_placeless` still applies.

    `command_id`: which command question to read (`g1_add_command_choice`);
    an option it does not offer reads as probability 0."""
    var r = G1LangPick()
    r.p_none = ans.probability(command_id, String(G1_OPT_NONE))
    r.needs_world = ans.noul(String(G1_Q_NEEDS_WORLD))
    if with_addressed:
        r.addressed = ans.noul(String(G1_Q_ADDRESSED))

    var p_talk = ans.probability(command_id, String(G1_OPT_TALK))

    # the best REAL option — never the argmax, see the header. Aliases
    # compete on equal terms and are resolved to their target afterwards.
    for i in range(bank.count() + G1_N_ALIAS):
        var nm = bank.name_at(i) if i < bank.count() else g1_alias_name(
            i - bank.count()
        )
        var p = ans.probability(command_id, nm)
        if p > r.conf:
            r.conf = p
            r.name = g1_resolve(nm)
            r.best = nm

    if with_addressed and r.addressed < min_addressed:
        r.reason = String("not addressed to the robot")
        r.name = String("")
        return r^
    # ⚠ `talk` COMPETES WITH THE MOVEMENTS, and only wins outright. A
    # greeting that the model half-reads as a command should move the robot,
    # not start a conversation — the movement is the demo, and a wrong answer
    # in words is more confusing than a wrong gesture.
    # ⚠ `talk` NEEDS A REAL BAR, not merely to beat `none`. A garbled
    # fragment — "Fera un." — won at 0.30 and started a conversation about
    # nothing. Beating the best movement is necessary; being confident is
    # also necessary.
    if p_talk > r.conf and p_talk > 0.50:
        r.talk = True
        r.name = String("")
        r.conf = p_talk
        return r^
    # ⚠⚠ THE DESTINATION IS CHECKED BEFORE `P(none)`, AND THE ORDER IS THE
    # WHOLE POINT. A refusal for `no such command` is what routes an
    # instruction to the cache-miss path (§12.60), and a destination must
    # NEVER get there: world position is not in the state pool, so a reward
    # over it returns a unit-norm `z` and a confidently wrong robot (§3.2).
    # "Walk to the fridge" has to reach a navigator, and if `P(none)` were
    # tested first it would instead reach a reward spec.
    #
    # It is also checked AFTER `talk`, because a greeting that happens to
    # mention a place is still a greeting, and after `addressed`, because
    # background conversation about the kitchen is not an instruction.
    #
    # ⚠ BOTH SIGNALS ARE REQUIRED. `needs_world` alone over-fires — §12.54
    # measured "walk to the fridge and open it" coming back `walk` at 0.92
    # with `needs_world` high, so the noul is sensitive but says nothing about
    # WHICH place. A named destination alone is not enough either: the
    # destination question has to pick something from a closed list, and
    # `none` is the only escape. Requiring both means a route happens when the
    # instruction is about going somewhere AND the somewhere is one this scene
    # has.
    if with_destination:
        var dpick = ans.choice(String(G1_Q_DESTINATION))
        var dconf = ans.probability(String(G1_Q_DESTINATION), dpick)
        if (dpick != G1_OPT_NONE and dconf >= min_dest
            and r.needs_world >= min_world):
            r.destination = dpick
            r.dest_conf = dconf
            # ⚠ `name` IS CLEARED. The caller must not run a bank command AND
            # hand a destination to a planner — the planner's first act is to
            # choose a command, and a stale one racing it is how the robot
            # walks off while being told where to walk.
            r.name = String("")
            return r^

    # ⚠ "GO TO THE KITCHEN" WHEN THE SCENE HAS NO KITCHEN. `needs_world` is
    # high and no destination matched, so without this the decision falls
    # through to the nearest MOVEMENT — `walk` at 0.97 in the room session's
    # measurement — and the robot sets off in an arbitrary direction.
    #
    # ⚠ WHETHER THAT IS RIGHT IS A PROPERTY OF THE SCENE, NOT OF THE
    # LANGUAGE, which is why this is opt-in rather than the default. §12.54's
    # rule is to do the part it can and NAME the part it cannot, and walking
    # while saying "but I have no destination" is exactly that — harmless in
    # an empty void, and walking into the fridge in a furnished room. The
    # caller knows which it has; this file cannot.
    #
    # ⚠⚠ AND THE REASON MUST NOT BE `no such command`, because that exact
    # string is what routes an instruction to the cache-miss path (§12.60). A
    # placeless "go somewhere" handed to a reward spec is §3.2's trap with an
    # extra step: world position is not in the state pool, so it would come
    # back a unit-norm `z` and a confidently wrong robot. Different cause,
    # different word, different route.
    if ((with_destination or dest_decided) and refuse_placeless and r.destination == ""
        and r.needs_world >= min_world):
        # ⚠ THE REASON STATES THE OBSERVATION, NOT AN INFERENCE. It first read
        # "names a place this scene does not have", which is true for "go to
        # the kitchen" and FALSE for "assieds-toi sur la chaise" — the chair IS
        # listed, and the destination question returned `none` because sitting
        # is not GOING (§6.4). Both arrive here identically and nothing in the
        # answer separates them, so the wording may not claim to know which.
        # Refusing is right in both cases; asserting an absent chair is not.
        r.reason = String("needs a destination, and none of this scene's matched")
        r.name = String("")
        return r^

    if r.p_none > max_none or r.name == "":
        r.reason = String("no such command")
        r.name = String("")
        return r^
    if r.conf < min_top:
        r.reason = String("no clear pick")
        r.name = String("")
        return r^
    # ⚠ belt and braces: a model that answered outside its own option list
    # would otherwise reach the robot.
    if bank.find(r.name) < 0:
        r.reason = String("not in the bank")
        r.name = String("")
    return r^
