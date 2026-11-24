"""The amount cut out of a sentence before the command request
(`g1_find_amount`, `g1_strip_amount`; docs/NN_MODEL_PORTS_PLAN.md §12.9.5).

    pixi run mojo run -I . tests/robots/test_g1_strip_amount.mojo

WHY THIS EXISTS
===============
An amount makes d1 refuse the command: "Avance de 3 mètres." came back `none`
where "Avance." is a walk, and cutting the amount out first took the held-out
commands with an amount from 13 / 40 accepted to 21 / 40. That was measured
with the bench's Python (`re`); the robot runs this Mojo port of it. So every
case below is the bench function's own output (span in code points, and the
tidied sentence), including the ones it gets wrong on purpose: an amount with
no unit ("Tourne de 90 à gauche.") and a glued unit ("12m") are left in, and
"à 2 mètres devant toi" loses its distance, as measured.
"""

from noeira.envs.robots.g1_command_language import g1_find_amount, g1_strip_amount


def _case(mut fails: Int, said: String, want: String, a: Int, b: Int) raises:
    var at = g1_find_amount(said)
    var got = g1_strip_amount(said)
    if at[0] != a or at[1] != b or got != want:
        print("  FAIL", "'" + said + "'", "->", "'" + got + "'", at[0], at[1], "| want", "'" + want + "'", a, b)
        fails += 1


def main() raises:
    var fails = 0
    _case(fails, "Avance de 3 mètres.", "Avance.", 7, 18)
    _case(fails, "Recule d'un mètre.", "Recule.", 7, 17)
    _case(fails, "Recul de 2,50 m.", "Recul.", 6, 15)
    _case(fails, "Fais 1 mètre et demi vers la droite.", "Fais vers la droite.", 5, 20)
    _case(fails, "Avance de deux mètres et demi.", "Avance.", 7, 29)
    _case(fails, "Lève le bras gauche pendant 5 secondes.", "Lève le bras gauche.", 20, 38)
    _case(fails, "Garde les bras écartés une minute.", "Garde les bras écartés.", 23, 33)
    _case(fails, "Tourne de 90 degrés à droite.", "Tourne à droite.", 7, 19)
    _case(fails, "Tourne de 90 à gauche.", "Tourne de 90 à gauche.", -1, -1)
    _case(fails, "Lève le bras, puis avance de 4 mètres.", "Lève le bras, puis avance.", 26, 37)
    _case(fails, "Walk forward 3 meters.", "Walk forward.", 13, 21)
    _case(fails, "Go back two and a half meters.", "Go back.", 8, 29)
    _case(fails, "Move 50 centimeters forward.", "Move forward.", 5, 19)
    _case(fails, "Keep your arms out wide for half a minute.", "Keep your arms out wide.", 24, 41)
    _case(fails, "Turn around 180 degrees.", "Turn around.", 12, 23)
    _case(fails, "Walk 4 meters and then turn right.", "Walk and then turn right.", 5, 13)
    _case(fails, "Run 6 meters then stop.", "Run then stop.", 4, 12)
    _case(fails, "Avance un peu.", "Avance un peu.", -1, -1)
    _case(fails, "J'ai 2 questions, regarde à gauche.", "J'ai 2 questions, regarde à gauche.", -1, -1)
    _case(fails, "Il est 5 heures, avance.", "Il est 5 heures, avance.", -1, -1)
    _case(fails, "It's 4 o'clock, walk forward.", "It's 4 o'clock, walk forward.", -1, -1)
    _case(fails, "Go to the fridge.", "Go to the fridge.", -1, -1)
    _case(fails, "", "", -1, -1)
    _case(fails, "3 mètres.", ".", 0, 8)
    _case(fails, "Pendant 10 secondes, lève le bras.", "Lève le bras.", 0, 19)
    _case(fails, "WALK FORWARD 2 METERS.", "WALK FORWARD.", 13, 21)
    _case(fails, "avance de 1.5 m", "Avance", 7, 15)
    _case(fails, "Avance  de  3 mètres  ,  puis tourne.", "Avance, puis tourne.", 8, 20)
    _case(fails, "Marche 1 m 50 vers la porte.", "Marche vers la porte.", 7, 13)
    _case(fails, "Attends dix secondes.", "Attends.", 8, 20)
    _case(fails, "Go 12m forward.", "Go 12m forward.", -1, -1)
    _case(fails, "Reculez de 30 cm.", "Reculez.", 8, 16)
    _case(fails, "Tourne de quatre-vingt-dix degrés.", "Tourne.", 7, 33)
    _case(fails, "Il y a un trou à 2 mètres devant toi.", "Il y a un trou à devant toi.", 17, 25)
    _case(fails, "élève-toi de 1 mètre.", "Élève-toi.", 10, 20)
    if fails > 0:
        raise Error(String(fails) + " strip-amount case(s) differ from the bench's")
    print("test_g1_strip_amount: 35 cases PASS")
