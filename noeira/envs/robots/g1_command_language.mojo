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
comptime G1_Q_NEEDS_WORLD: String = "needs_world"
comptime G1_Q_ADDRESSED: String = "addressed"
comptime G1_OPT_NONE: String = "none"

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
    return String(
        "A humanoid robot can perform exactly the commands listed, and"
        " nothing else. Pick the command that best matches what the"
        " instruction asks the robot to do with its body — it does not"
        " have to match exactly, only be the closest thing the robot can"
        " do. Pick `none` ONLY when no command on the list is even"
        " approximately what was asked."
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
    options.append(String(G1_OPT_NONE))
    descs.append(String("not one of these, or not a command for this robot"))

    var q = JevQuestions()
    q.choice(String(G1_Q_COMMAND), g1_command_instruction(), options, descs)
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
            String(
                "Is this an instruction addressed to the robot, asking it to"
                " do something now? Answer no for background conversation,"
                " silence, or speech not aimed at the robot."
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

    def __init__(out self):
        self.name = String("")
        self.best = String("")
        self.conf = 0.0
        self.p_none = 1.0
        self.needs_world = 0.0
        self.addressed = 1.0
        self.reason = String("")


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
