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
    )


def g1_command_questions(
    ref bank: G1CommandBank, with_addressed: Bool = False
) raises -> JevQuestions:
    """The choice over the bank, plus the two cheap guards.

    ⚠ The option descriptions are GENERATED from each compound
    (`bank.describe`). A hand-written blurb would drift from the terms the
    command was gated on, and the drift is invisible — the robot doing one
    thing while the model was told another.
    """
    var options = List[String]()
    var descs = List[String]()
    for i in range(bank.count()):
        options.append(bank.name_at(i))
        descs.append(bank.describe(i))
    # the aliases are offered as ordinary options; `g1_decide` resolves them
    for i in range(G1_N_ALIAS):
        options.append(g1_alias_name(i))
        descs.append(g1_alias_desc(i))
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

    var q = JevQuestions()
    q.choice(String(G1_Q_COMMAND), g1_command_instruction(), options, descs)
    # ⚠ THE SAME OPTION LIST, so a second step is chosen from exactly what
    # the robot can do. `none` on step 2 means the instruction was one
    # command, which is the common case and must stay cheap.
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

    def __init__(out self):
        self.name = String("")
        self.best = String("")
        self.conf = 0.0
        self.p_none = 1.0
        self.needs_world = 0.0
        self.addressed = 1.0
        self.reason = String("")
        self.talk = False


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
) raises -> G1LangPick:
    """Apply the rule above. Returns an empty `name` with a `reason` set when
    the honest answer is to do nothing."""
    var r = G1LangPick()
    r.p_none = ans.probability(String(G1_Q_COMMAND), String(G1_OPT_NONE))
    r.needs_world = ans.noul(String(G1_Q_NEEDS_WORLD))
    if with_addressed:
        r.addressed = ans.noul(String(G1_Q_ADDRESSED))

    var p_talk = ans.probability(String(G1_Q_COMMAND), String(G1_OPT_TALK))

    # the best REAL option — never the argmax, see the header. Aliases
    # compete on equal terms and are resolved to their target afterwards.
    for i in range(bank.count() + G1_N_ALIAS):
        var nm = bank.name_at(i) if i < bank.count() else g1_alias_name(
            i - bank.count()
        )
        var p = ans.probability(String(G1_Q_COMMAND), nm)
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
