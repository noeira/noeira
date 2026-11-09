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
    g1_place_mentions, g1_other_one,
    g1_route_triage, g1_resolve_trail, g1_trail_arrivals, G1Triage,
    G1_TRIAGE_OFF, G1_TRIAGE_FULL, G1_TRIAGE_NARROW,
)
from noeira.ai.jev import JevAnswers
from noeira.io.json import parse_json


def _kind_answer(top: String, probs: String) raises -> JevAnswers:
    """A Jev answer to the triage alone: `probs` is the JSON body of `probabilities`."""
    var body = (String('{"model":"t","usage":{"input_tokens":1},"answers":{"kind":{"choice":"')
                + top + String('","probabilities":{') + probs + String("}}}}"))
    return JevAnswers(parse_json(List[UInt8](body.as_bytes())), 0.0)


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

    # ── the room state (RB-1's s1c + recently_heard) ─────────────────
    # ⚠ OFF, NOTHING CHANGES: the measured requests (8 Oct, RB-1's S0) must
    # stay byte-identical for every caller that does not ask for these keys.
    print("-- room state: geometry and memory")
    var off = G1Context()
    off.places = True
    off.arrived_at(String("fridge"), 1.0)
    off.heard_said(String("the fridge or the table"), 2.0)
    var s_off = g1_command_state(String("x"), off)
    t.truth(
        s_off.find("places_around_you") < 0 and s_off.find("recently_heard") < 0
        and s_off.find("places_visited_recently") < 0,
        "switches off: a log alone adds no key",
    )
    var on = G1Context()
    on.geometry = True
    on.memory = True
    var s_on = g1_command_state(String("x"), on)
    t.truth(
        s_on.find("\"places_around_you\":[],\"places_visited_recently\":[],\"recently_heard\":[]") >= 0,
        "switches on, nothing known: every key present, empty, in s1c's order",
    )
    on.around.append(String("table: left, 3.5 m, next to sofa"))
    on.arrived_at(String("door"), 1.0)
    on.arrived_at(String("fridge"), 2.0)
    on.arrived_at(String("table"), 3.0)
    on.arrived_at(String("table"), 4.0)
    on.arrived_at(String("sofa"), 5.0)
    on.arrived_at(String("door"), 6.0)
    var tr = on.trail()
    t.truth(
        len(tr) == 4 and tr[0] == "door" and tr[1] == "sofa" and tr[2] == "table" and tr[3] == "fridge",
        "trail: most recent first, a repeat not re-added, capped at 4 (the oldest door dropped)",
    )
    for h in ["a", "b", "c", "d"]:
        on.heard_said(String(h), 7.0)
    var hr = on.heard_recent()
    t.truth(len(hr) == 3 and hr[0] == "d" and hr[2] == "b", "recently_heard: the last 3, most recent first")
    var s_full = g1_command_state(String("x"), on)
    t.truth(
        s_full.find("\"places_around_you\":[\"table: left, 3.5 m, next to sofa\"]") >= 0
        and s_full.find("\"recently_heard\":[\"d\",\"c\",\"b\"]") >= 0,
        "the caller's place lines and the heard utterances are sent as written",
    )

    # ── "the other one", settled by the log ───────────────────────────
    print("-- g1_place_mentions, g1_other_one")
    var pl = List[String]()
    var wd = List[String]()
    for p in ["fridge", "door", "table", "sofa"]:
        pl.append(String(p))
    wd.append(String("frigo,réfrigérateur"))
    wd.append(String("porte"))
    wd.append(String(""))
    wd.append(String("canapé,couch"))
    var mn = g1_place_mentions(String("Le canapé, ou plutôt les tables ? Non: le frigo."), pl, wd)
    t.truth(
        len(mn) == 3 and mn[0] == "sofa" and mn[1] == "table" and mn[2] == "fridge",
        "mentions: in order of appearance, synonyms and plurals, punctuation ignored",
    )
    t.truth(len(g1_place_mentions(String("the doorbell rang"), pl, wd)) == 0, "whole words only: doorbell is no door")
    var o1 = G1Context()
    o1.heard_said(String("the fridge, or the table, go to the fridge"), 1.0)
    o1.arrived_at(String("fridge"), 2.0)
    t.truth(g1_other_one(o1, pl, wd) == "table", "arrived at one of two: the other")
    var o2 = G1Context()
    o2.heard_said(String("the fridge or the table"), 1.0)
    o2.arrived_at(String("fridge"), 2.0)
    o2.at = String("")
    t.truth(g1_other_one(o2, pl, wd) == "table", "arrived and left again: still taken")
    var o3 = G1Context()
    o3.heard_said(String("go to the sofa or the door"), 1.0)
    o3.walking_to = String("sofa")
    t.truth(g1_other_one(o3, pl, wd) == "door", "walking to one of two: the other")
    var o4 = G1Context()
    o4.heard_said(String("fridge, table or sofa"), 1.0)
    o4.arrived_at(String("sofa"), 2.0)
    t.truth(g1_other_one(o4, pl, wd) == "", "two left: not settled")
    var o5 = G1Context()
    o5.heard_said(String("the fridge or the table"), 1.0)
    o5.heard_said(String("salut"), 2.0)
    o5.arrived_at(String("table"), 3.0)
    t.truth(g1_other_one(o5, pl, wd) == "fridge", "the latest utterance naming two places, not the latest one")
    t.truth(g1_other_one(G1Context(), pl, wd) == "", "nothing heard: not settled")

    # ── the triage (RB-2): the two policies, as room_bench `s2` measured them ──
    print("-- g1_route_triage")
    var a_unsafe = _kind_answer(String("unsafe"), String('"unsafe":0.6,"go_place":0.3,"act_here":0.1'))
    var tf = g1_route_triage(a_unsafe, G1_TRIAGE_FULL)
    t.truth(tf.kind == "unsafe" and tf.talks() and abs(tf.p - 0.6) < 1e-9, "full: unsafe .6 -> talks, p = the talk mass")
    t.truth(g1_route_triage(a_unsafe, G1_TRIAGE_OFF).kind == "", "off: never acts")
    # ⚠ the mass rule: neither kind reaches .5, together they do
    var a_split = _kind_answer(String("unsettled"), String('"remembered":0.45,"unsettled":0.48,"talk":0.07'))
    var ts = g1_route_triage(a_split, G1_TRIAGE_FULL)
    t.truth(ts.kind == "unsettled" and ts.talks(), "full: remembered .45 / unsettled .48 + talk .07 -> unsettled by mass (no fall-through)")
    var a_below = _kind_answer(String("unsettled"), String('"remembered":0.45,"unsettled":0.48,"go_place":0.07'))
    t.truth(g1_route_triage(a_below, G1_TRIAGE_FULL).kind == "", "full: talk mass .48 < .5 and no top >= .5 -> does not act")
    t.truth(abs(ts.place_mass - 0.93) < 1e-9, "place mass = go_place + remembered + unsettled")
    var a_spl = _kind_answer(String("go_place"), String('"go_place":0.4,"speaker_side":0.35,"talk":0.25'))
    var tg = g1_route_triage(a_spl, G1_TRIAGE_FULL)
    t.truth(tg.kind == "talk" and tg.talks(), "full: top go_place .4, talk kinds .6 -> `talk`")
    var a_rem = _kind_answer(String("remembered"), String('"remembered":0.7,"unsettled":0.2,"talk":0.1'))
    t.truth(g1_route_triage(a_rem, G1_TRIAGE_FULL).kind == "remembered", "full: remembered .7 -> the trail request")
    t.truth(g1_route_triage(a_rem, G1_TRIAGE_NARROW).kind == "", "narrow: remembered never acts (d1)")
    var a_low = _kind_answer(String("speaker_side"), String('"speaker_side":0.42,"go_place":0.38,"talk":0.2'))
    t.truth(g1_route_triage(a_low, G1_TRIAGE_NARROW).kind == "speaker_side", "narrow: speaker_side .42 >= .4 acts")
    var a_tk = _kind_answer(String("talk"), String('"talk":0.9,"go_place":0.1'))
    var tn = g1_route_triage(a_tk, G1_TRIAGE_NARROW)
    t.truth(tn.kind == "" and tn.place_mass == 0.0, "narrow: talk never acts, no place mass")
    var a_act = _kind_answer(String("act_here"), String('"act_here":0.49,"go_place":0.51'))
    t.truth(g1_route_triage(a_act, G1_TRIAGE_NARROW).kind == "", "narrow: act_here .49 (not the top) does not act")

    # ── the trail, resolved from the log ──────────────────────────────
    print("-- g1_resolve_trail")
    var tc = G1Context()
    tc.arrived_at(String("fridge"), 1.0)
    tc.arrived_at(String("door"), 2.0)
    tc.heard_said(String("nice"), 3.0)
    tc.arrived_at(String("table"), 4.0)
    tc.arrived_at(String("sofa"), 5.0)
    tc.at = String("sofa")
    var arr = g1_trail_arrivals(tc)
    t.truth(len(arr) == 4 and arr[0] == "fridge" and arr[3] == "sofa", "arrivals: oldest first, heard skipped")
    t.truth(g1_resolve_trail(String("before_landmark"), String("door"), tc) == "fridge", "before the door -> fridge (not the previous stop)")
    t.truth(g1_resolve_trail(String("after_landmark"), String("fridge"), tc) == "door", "after the fridge -> door")
    t.truth(g1_resolve_trail(String("previous_stop"), String("none"), tc) == "table", "previous stop -> table")
    t.truth(g1_resolve_trail(String("starting_point"), String("none"), tc) == "fridge", "starting point -> fridge")
    t.truth(g1_resolve_trail(String("before_landmark"), String("shelf"), tc) == "", "a landmark not on the trail: not settled (ask)")
    t.truth(g1_resolve_trail(String("after_landmark"), String("table"), tc) == "", "after the table is here: not settled")
    t.truth(g1_resolve_trail(String("before_landmark"), String("none"), tc) == "", "before, no landmark: not settled")
    var tw = G1Context()
    tw.arrived_at(String("door"), 1.0)
    tw.arrived_at(String("fridge"), 2.0)
    tw.arrived_at(String("door"), 3.0)
    t.truth(g1_resolve_trail(String("before_landmark"), String("door"), tw) == "fridge",
            "between two visits to the door: the LATEST visit's predecessor")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_g1_command_state: " + String(t.fails) + " failed")
