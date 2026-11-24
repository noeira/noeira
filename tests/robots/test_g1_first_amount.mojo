"""A command's amount, by code (`g1_parse_amount`, `g1_first_amount`,
`g1_step_end`, `g1_step_done`; docs/NN_MODEL_PORTS_PLAN.md §12.9.7).

    pixi run mojo run -I . tests/robots/test_g1_first_amount.mojo

WHY THIS EXISTS
===============
The voice demo takes "Avance de 3 mètres" as three metres from these
functions, not from d1: on 60 held-out commands with an amount they read 58
right, against 18 / 44 for the `extent` level. That was measured with the
bench's Python (`extent_score.py`'s `parse_amount` and `find_amount`, plus the
clause split); the robot runs this Mojo port. Every `_first` and `_parse` case
below is the Python's own output on the same text, including what it cannot
read on purpose: a unit the transcript dropped ("90" in "Tourne de 90 à
gauche."), kilometres, "Go 12m forward." (a glued unit), and "a few" steps
read as 1 step. An amount after "puis" / "then" is a later action's and
gives none. `_step` checks how an amount becomes the step's distance and
clock, with the clamps; `_done` that a step with a distance still ends on its
clock (a blocked robot used to hold its chain for ever).
"""

from noeira.envs.robots.g1_command_language import (
    G1Amount, g1_parse_amount, g1_first_amount, g1_step_end, g1_step_done,
)


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _first(mut fails: Int, said: String, value: Float64, unit: String, a: Int, b: Int) raises:
    var got = g1_first_amount(said)
    if got.unit != unit or not _close(got.value, value) or got.start != a or got.end != b:
        print("  FAIL first", "'" + said + "'", "->", got.value, got.unit, got.start, got.end,
              "| want", value, unit, a, b)
        fails += 1


def _parse(mut fails: Int, span: String, said: String, value: Float64, unit: String) raises:
    var got = g1_parse_amount(span, said)
    if got.unit != unit or not _close(got.value, value):
        print("  FAIL parse", "'" + span + "' in '" + said + "'", "->", got.value, got.unit, "| want", value, unit)
        fails += 1


def _step(mut fails: Int, loco: Bool, extent: Float64, value: Float64, unit: String, d: Float64, s: Float64):
    var got = g1_step_end(loco, extent, G1Amount(value, unit, -1, -1))
    if not _close(got[0], d) or not _close(got[1], s):
        print("  FAIL step", loco, extent, value, unit, "->", got[0], got[1], "| want", d, s)
        fails += 1


def _done(mut fails: Int, dist: Float64, end_dist: Float64, now: Int, end_at: Int, want: Bool):
    if g1_step_done(dist, end_dist, now, end_at) != want:
        print("  FAIL done", dist, end_dist, now, end_at, "| want", want)
        fails += 1


def main() raises:
    var fails = 0
    _first(fails, "Avance de 6 mètres.", 6.0, "m", 7, 18)
    _first(fails, "Recule d'un mètre.", 1.0, "m", 7, 17)
    _first(fails, "Marche 2,75 m vers la porte.", 2.75, "m", 7, 13)
    _first(fails, "Avance de un mètre cinquante.", 1.5, "m", 7, 28)
    _first(fails, "Marche 1 m 20 tout droit.", 1.2, "m", 7, 13)
    _first(fails, "Fais 3 mètres et demi.", 3.5, "m", 5, 21)
    _first(fails, "Avance d'une demi-mètre.", 0.0, "", -1, -1)
    _first(fails, "Go forward half a meter.", 0.5, "m", 11, 23)
    _first(fails, "Walk two and a half meters.", 2.5, "m", 5, 26)
    _first(fails, "Move 25 centimeters left.", 0.25, "m", 5, 19)
    _first(fails, "Reste assis pendant 2 minutes.", 120.0, "s", 12, 29)
    _first(fails, "Wait for one minute.", 60.0, "s", 5, 19)
    _first(fails, "Lève le bras pendant quinze secondes.", 15.0, "s", 13, 36)
    _first(fails, "Hold it for 0,5 secondes.", 0.5, "s", 8, 24)
    _first(fails, "Tourne de 120 degrés.", 120.0, "deg", 7, 20)
    _first(fails, "Turn 45 DEGREES left.", 45.0, "deg", 5, 15)
    _first(fails, "Tourne de quatre-vingt-dix degrés.", 90.0, "deg", 7, 33)
    _first(fails, "Turn one hundred and eighty degrees.", 180.0, "deg", 5, 35)
    _first(fails, "Tourne à droite, puis avance de 4 mètres.", 0.0, "", -1, -1)
    _first(fails, "Raise your hand, then walk 2 meters.", 0.0, "", -1, -1)
    _first(fails, "Lève le bras et ensuite recule de 1 mètre.", 0.0, "", -1, -1)
    _first(fails, "Walk 3 meters, then turn 90 degrees.", 3.0, "m", 5, 13)
    _first(fails, "Avance de 2 mètres puis tourne.", 2.0, "m", 7, 18)
    _first(fails, "After that, walk 5 meters.", 0.0, "", -1, -1)
    _first(fails, "Avance un peu.", 0.0, "", -1, -1)
    _first(fails, "Il y a 2 personnes.", 0.0, "", -1, -1)
    _first(fails, "Go 12m forward.", 0.0, "", -1, -1)
    _first(fails, "Recule de trois mètres.", 3.0, "m", 7, 22)
    _first(fails, "Avance de 1.5 m.", 1.5, "m", 7, 15)
    _first(fails, "Marche pendant 30 secondes.", 30.0, "s", 7, 26)
    _first(fails, "WALK 4 METERS.", 4.0, "m", 5, 13)
    _first(fails, "Avance de 7 mètres et demi, puis arrête-toi.", 7.5, "m", 7, 26)
    _first(fails, "Élève-toi de 2 mètres.", 2.0, "m", 10, 21)
    _first(fails, "", 0.0, "", -1, -1)
    _parse(fails, "3 mètres", "Avance de 3 mètres.", 3.0, "m")
    _parse(fails, "90", "Tourne de 90 degrés.", 90.0, "deg")
    _parse(fails, "90", "Tourne de 90 à gauche.", 0.0, "")
    _parse(fails, "un peu", "Avance un peu.", 0.0, "")
    _parse(fails, "a few", "Go back a few steps.", 1.0, "steps")
    _parse(fails, "10 kilomètres", "J'ai couru 10 kilomètres.", 0.0, "")
    _parse(fails, "une minute", "Attends une minute.", 60.0, "s")
    _parse(fails, "half a minute", "Wait half a minute.", 30.0, "s")
    _parse(fails, "1 mètre 80", "Il mesure 1 mètre 80.", 1.8, "m")
    _parse(fails, "deux mètres et demi", "Avance de deux mètres et demi.", 2.5, "m")
    _parse(fails, "2 heures", "Dans 2 heures.", 7200.0, "s")
    _parse(fails, "5 pas", "Fais 5 pas.", 5.0, "steps")
    _parse(fails, "ten seconds", "Wait ten seconds.", 10.0, "s")
    _parse(fails, "3", "Avance de 3, mètres.", 3.0, "m")
    _parse(fails, "4 meters", "Walk 4 meters.", 4.0, "m")
    _parse(fails, "", "Avance.", 0.0, "")
    _parse(fails, "un mètre cinquante", "Avance d'un mètre cinquante.", 1.5, "m")
    _parse(fails, "une demi-heure", "Attends une demi-heure.", 1800.0, "s")
    # how the step ends: (metres, 0 = on the clock; seconds)
    _step(fails, True, 2.0, 3.0, "m", 3.0, 16.0)       # 3 m, the clock extent gives 3 m (4 s x 3 + 4)
    _step(fails, True, 2.0, 25.0, "m", 10.0, 40.0)     # clamped to ten metres
    _step(fails, True, 2.0, 0.05, "m", 0.1, 7.0)       # clamped up to 10 cm
    _step(fails, True, 2.0, 5.0, "s", 0.0, 5.0)        # "Avance pendant 5 secondes": the clock alone
    _step(fails, False, 2.0, 120.0, "s", 0.0, 12.0)    # two minutes, clamped to twelve seconds
    _step(fails, False, 2.0, 0.2, "s", 0.0, 0.5)
    _step(fails, True, 2.0, 90.0, "deg", 2.0, 13.0)    # degrees unused: the extent level, as before
    _step(fails, True, 1.0, 0.0, "", 1.0, 10.0)        # no amount: as before
    _step(fails, False, 2.0, 2.0, "m", 0.0, 3.0)       # metres for a posture: the extent's seconds
    _step(fails, False, 0.0, 0.0, "", 0.0, 1.0)
    _step(fails, True, 2.0, 4.0, "steps", 2.0, 13.0)   # steps unused
    # when the step is over: the distance OR the clock (step_dist, end_dist, now, end_at)
    _done(fails, 1.0, 3.0, 5, 10, False)    # walking, neither yet
    _done(fails, 3.0, 3.0, 5, 10, True)     # the distance covered
    _done(fails, 0.2, 3.0, 10, 10, True)    # blocked: the clock ends it anyway
    _done(fails, 0.0, 0.0, 5, 10, False)    # a clock step, not yet
    _done(fails, 9.0, 0.0, 5, 10, False)    # a clock step: distance means nothing
    _done(fails, 0.0, 0.0, 11, 10, True)
    if fails > 0:
        raise Error(String(fails) + " amount case(s) differ")
    print("test_g1_first_amount: 34 first + 18 parse + 11 step + 6 done cases PASS")
