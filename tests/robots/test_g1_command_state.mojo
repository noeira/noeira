"""The conversational state Jev is asked against — a gate, because it is half
the question and it is silent when wrong.

    pixi run mojo run -I . tests/robots/test_g1_command_state.mojo

WHY THIS EXISTS
===============
"plus vite" after a walk was refused until the state carried `doing`. That
failure was indistinguishable from a bad wording, a bad threshold or a bad
transcript, and it cost a round of the wrong diagnosis (§10.1 of
`docs/BFM_ZERO_NEXT_LEVEL.md`). Three properties are worth a gate:

**Every key is always present.** A missing key is a different question, and
the answers to two different questions are not comparable across calls. So
`doing` says `"nothing"` and `last_arm_raised` says `"none"` rather than
being absent.

**A repeat is not a history entry.** Holding `walk` while the user says
"plus vite" twice must not push `walk` into `did_before` three times and
evict what came before it.

**⚠ THE DURATION IS A WORD.** Jev's own docs list numeric comparison as a
failure mode, so a `doing_since` that leaked a float would ask the model to
compare against a bound it was never told. The bucketing is ours and its
boundaries are pinned here.

Nothing in this file touches the network or the bank, so it runs in the smoke
tier: the parts that need the live service are measured through
`bfm_zero_say.mojo --doing`.
"""

from noeira.envs.robots.g1_command_language import (
    G1Context, g1_arm_of, g1_since_word, g1_command_state, G1_N_RECENT,
)


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()

    # ── g1_arm_of: derived from the bank's names, not stored ──────────
    print("-- g1_arm_of")
    t.truth(g1_arm_of(String("right_hand_up")) == "right", "right_hand_up -> right")
    t.truth(
        g1_arm_of(String("walk_right_hand_up")) == "right",
        "walk_right_hand_up -> right (the compound counts)",
    )
    t.truth(g1_arm_of(String("left_hand_up")) == "left", "left_hand_up -> left")
    t.truth(g1_arm_of(String("both_hands_up")) == "both", "both_hands_up -> both")
    t.truth(g1_arm_of(String("arms_wide")) == "both", "arms_wide -> both")
    t.truth(g1_arm_of(String("walk")) == "", "walk raises no arm")
    t.truth(g1_arm_of(String("squat")) == "", "squat raises no arm")
    # ⚠ an unknown name must be "" and not raise: the bank can gain a command
    # before this table knows about it, and the state must still build.
    t.truth(g1_arm_of(String("hopscotch")) == "", "an unknown command raises no arm")

    # ── g1_since_word: the boundaries, pinned ─────────────────────────
    print("-- g1_since_word")
    t.truth(g1_since_word(0.0) == "just_started", "0.0 s -> just_started")
    t.truth(g1_since_word(1.49) == "just_started", "just under 1.5 s -> just_started")
    t.truth(g1_since_word(1.5) == "a_moment", "1.5 s is the first a_moment")
    t.truth(g1_since_word(5.99) == "a_moment", "just under 6 s -> a_moment")
    t.truth(g1_since_word(6.0) == "a_while", "6 s is the first a_while")
    t.truth(g1_since_word(600.0) == "a_while", "ten minutes -> a_while")

    # ── G1Context.began: order, the repeat, the cap ───────────────────
    print("-- G1Context.began")
    var c = G1Context()
    t.truth(c.doing == "", "a fresh context is doing nothing")
    t.truth(len(c.recent) == 0, "a fresh context has no history")

    c.began(String("walk"))
    t.truth(c.doing == "walk", "began sets doing")
    t.truth(len(c.recent) == 0, "the FIRST command does not enter the history")
    t.truth(c.since_s == 0.0, "began resets the clock")

    c.since_s = 4.0
    c.began(String("walk"))
    t.truth(
        len(c.recent) == 0,
        "a REPEAT of the same command does not enter the history",
    )
    t.truth(c.since_s == 0.0, "a repeat still resets the clock")

    c.began(String("squat"))
    t.truth(c.doing == "squat", "the new command is current")
    t.truth(len(c.recent) == 1 and c.recent[0] == "walk", "the old one moved to history")

    c.began(String("spin_left"))
    t.truth(
        len(c.recent) == 2 and c.recent[0] == "squat" and c.recent[1] == "walk",
        "the history is MOST RECENT FIRST",
    )

    # ── the cap holds and evicts the oldest ───────────────────────────
    c.began(String("run"))
    c.began(String("back"))
    c.began(String("stand"))
    t.truth(len(c.recent) == G1_N_RECENT, "the history is capped at G1_N_RECENT")
    t.truth(c.recent[0] == "back", "the newest history entry is first")
    # walk, walk, squat, spin_left, run, back, stand -> [back, run, spin_left]:
    # `walk` and `squat` are gone, which is the end that should go.
    t.truth(
        c.recent[G1_N_RECENT - 1] == "spin_left",
        "the OLDEST entries are the ones evicted, not the newest",
    )
    var still_has_walk = False
    for i in range(len(c.recent)):
        if c.recent[i] == "walk":
            still_has_walk = True
    t.truth(not still_has_walk, "the first command has fallen off the history")

    # ── last_arm: current command first, then history ─────────────────
    print("-- last_arm")
    var a = G1Context()
    t.truth(a.last_arm() == "", "nothing done yet raises no arm")
    a.began(String("walk"))
    t.truth(a.last_arm() == "", "a locomotion command raises no arm")
    a.began(String("right_hand_up"))
    t.truth(a.last_arm() == "right", "the CURRENT command wins")
    a.began(String("walk"))
    t.truth(
        a.last_arm() == "right",
        "with no arm in the current command, the history answers",
    )
    a.began(String("left_hand_up"))
    a.began(String("squat"))
    t.truth(a.last_arm() == "left", "the MOST RECENT arm in the history wins")

    # ── the JSON: every key present, whatever the context ─────────────
    print("-- g1_command_state")
    var empty = G1Context()
    var s0 = g1_command_state(String("marche"), empty)
    t.truth(s0.find("\"instruction\":\"marche\"") >= 0, "the instruction is carried")
    t.truth(
        s0.find("\"doing\":\"nothing\"") >= 0,
        "an idle robot says `nothing`, it does not omit the key",
    )
    t.truth(
        s0.find("\"last_arm_raised\":\"none\"") >= 0,
        "no arm raised says `none`, it does not omit the key",
    )
    t.truth(s0.find("\"did_before\":[]") >= 0, "an empty history is an empty array")
    t.truth(s0.find("\"doing_since\"") >= 0, "doing_since is always present")
    # ⚠ the float must NOT reach the wire in any form
    t.truth(s0.find("0.0") < 0 and s0.find("since_s") < 0, "no raw duration on the wire")

    var full = G1Context()
    full.began(String("walk"))
    full.began(String("right_hand_up"))
    full.since_s = 8.0
    var s1 = g1_command_state(String("plus vite"), full)
    t.truth(s1.find("\"doing\":\"right_hand_up\"") >= 0, "doing names the live command")
    t.truth(s1.find("\"doing_since\":\"a_while\"") >= 0, "8 s bucketed to a_while")
    t.truth(s1.find("\"did_before\":[\"walk\"]") >= 0, "the history is a JSON array")
    t.truth(s1.find("\"last_arm_raised\":\"right\"") >= 0, "the arm is reported")

    # ⚠ a transcript can contain a quote or a backslash; the writer must
    # escape it or the whole request is malformed and the failure reads as a
    # service error.
    var q = G1Context()
    var s2 = g1_command_state(String("dis \"bonjour\" a\\ll"), q)
    t.truth(s2.find("\\\"bonjour\\\"") >= 0, "a quote in the transcript is escaped")
    t.truth(s2.find("a\\\\ll") >= 0, "a backslash in the transcript is escaped")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_g1_command_state: " + String(t.fails) + " failed")
