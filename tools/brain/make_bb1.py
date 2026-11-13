"""BB-1, the brain's grounded-answers set: scenarios, questions and how each reply is scored.

    python3 tools/brain/make_bb1.py            # writes tests/brain/bb1_scenarios.tsv and bb1_items.tsv
    python3 tools/brain/make_bb1.py --seed 7 --out /tmp/bb1_fresh     # other situations, same questions

docs/ROBOT_BRAIN_PLAN.md, step B1. A scenario is a robot's situation in a room (pose, what it is doing,
where it has been). An item is one question a person asks, in French or English, with the rule its reply
is scored by. The two files are committed: the Mojo test and the bench read them.

THIS FILE IS THE REFERENCE for `noeira/brain/geometry.mojo` and the state's reducers: the `ref_*` columns
(place lines, nearest place, where the robot was before) are computed here, with this file's own
arithmetic, and `tests/brain/test_brain_state.mojo` holds the Mojo code to them. The same numbers decide
what a right answer is, so a geometry bug cannot hide behind its own expected value.

Scenarios are drawn with a fixed seed and REJECTED when a fact sits near an edge (a bearing within 12
degrees of a sector boundary, a distance within 6 cm of a rounding boundary, two places almost equally
near): a question about an edge case has no answer a rule can score.

Scoring rule (columns `must`, `forbid`):
    must    groups joined by " && "; a group is alternatives joined by "|"; every group needs one hit
    forbid  alternatives joined by "|"; any hit fails the reply
    an alternative is a phrase (matched on word boundaries, case-insensitive) or `num:<value>` (a number
    written in digits in the reply, "3,5" and "3.5" alike)
Mutant rows (`cat` = where_mut) swap two places of a scenario: `must`/`forbid` are the truth of the swapped
room, `alt_must`/`alt_forbid` the truth of the room before the swap, which a reply that reads the state
must now FAIL.
"""

import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "tests", "brain")

NEAR_M = 2.5
NEXT_TO_M = 3.0

LAYOUTS = {
    "A": [("fridge", 3.0, 1.5), ("door", -3.2, 2.0), ("table", 1.0, -2.5), ("sofa", -2.0, -2.8)],
    "B": [("fridge", -3.0, -1.0), ("door", 0.5, 3.3), ("table", 2.8, -0.5), ("sofa", -0.5, -3.0)],
    "C": [("fridge", 1.5, 3.0), ("door", 3.3, -2.0), ("table", -2.5, 0.8), ("sofa", -1.0, -3.0)],
}

WORDS = {
    "fridge": "frigo|réfrigérateur|refrigerateur|fridge|refrigerator",
    "door": "porte|door",
    "table": "table",
    "sofa": "canapé|canape|sofa|couch",
}
FR = {"fridge": "le frigo", "door": "la porte", "table": "la table", "sofa": "le canapé"}
EN = {"fridge": "the fridge", "door": "the door", "table": "the table", "sofa": "the sofa"}
FR_AT = {"fridge": "au frigo", "door": "à la porte", "table": "à la table", "sofa": "au canapé"}

DIR = {
    "ahead": "devant|en face|tout droit|ahead|in front|straight",
    "left": "gauche|côté gauche|my left|your left|the left|left side",
    "right": "droite|mon droit|côté droit|my right|your right|the right|right side",
    "behind": "derrière|derriere|dans mon dos|behind",
}
YES = "oui|yes|je peux|i can|bien sûr|of course|sure|je sais|absolument|certainly|tout à fait"
REFUSE = ("ne peux pas|peux pas|ne sais pas|cannot|can't|can not|unable|pas capable|not able|incapable|"
          "malheureusement|unfortunately|désolé|désolée|desole|sorry|impossible")
# "no", or saying it lacks the ability. NOT "I don't know": a robot asked to climb stairs cannot, it does
# not ignore whether it can.
NO = ("ne peux pas|peux pas|cannot|can't|can not|unable|pas capable|not able|incapable|impossible|non|no|"
      "don't have the ability|do not have the ability|don't have any way|don't have a way|have no way|"
      "n'ai pas la capacité|n'ai pas les moyens|isn't one of|is not one of|not one of|don't think i can|"
      "ne pense pas pouvoir|don't know how|do not know how")
# a "yes, but I cannot X" is a yes: only a plain refusal fails a can_yes
REFUSAL_ONLY = "non|no|désolé|désolée|desole|sorry|malheureusement|unfortunately"
AFFIRM = "oui|yes|bien sûr|of course|avec plaisir|certainly|d'accord|okay|ok"
UNKNOWN = ("ne sais pas|sais pas|connais pas|don't know|do not know|pas de|aucun|aucune|n'y a pas|there is no|there's no|"
           "not sure|ne vois pas|don't see|do not see|j'ignore|not aware|ne connais pas|pas d'information|"
           "no information|inconnu|unknown|pas à ma connaissance|not that i know|n'ai pas|don't have|"
           "do not have|no tv|no television|no plant|non|no")

NUM_FR = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def wrap(a):
    r = a - 360.0 * math.floor(a / 360.0)
    return r - 360.0 if r > 180.0 else r


def sector(rel):
    if -45.0 <= rel <= 45.0:
        return "ahead"
    if 45.0 < rel <= 135.0:
        return "left"
    if -135.0 <= rel < -45.0:
        return "right"
    return "behind"


def around(places, x, y, yaw):
    """Per place: (name, bearing, rounded distance, near/far, next_to, rel, d)."""
    out = []
    for name, px, py in places:
        d = math.hypot(px - x, py - y)
        rel = wrap(math.degrees(math.atan2(py - y, px - x)) - yaw)
        best, bd = "none", 1e9
        for other, ox, oy in places:
            if other == name:
                continue
            dd = math.hypot(ox - px, oy - py)
            if dd < bd:
                best, bd = other, dd
        out.append((name, sector(rel), math.floor(d * 2.0 + 0.5) / 2.0, "near" if d < NEAR_M else "far",
                    best if bd <= NEXT_TO_M else "none", rel, d))
    return out


def line(p):
    t = "%s: %s, %s m" % (p[0], p[1], repr(float(p[2])))
    if p[4] != "none":
        t += ", next to " + p[4]
    return t


def clear_of_edges(places, x, y, yaw, at):
    ds = []
    for p in around(places, x, y, yaw):
        if p[0] == at:
            ds.append(p[6])
            continue
        for edge in (45.0, 135.0, -45.0, -135.0, 180.0, -180.0):
            if abs(p[5] - edge) < 12.0:
                return False
        frac = (p[6] * 2.0) % 1.0
        if abs(frac - 0.5) < 0.12:  # 6 cm from a rounding boundary
            return False
        if abs(p[6] - NEAR_M) < 0.05 or p[6] < 0.9:
            return False
        ds.append(p[6])
    ds.sort()
    return ds[1] - ds[0] >= 0.75


def num_alts(v, lang):
    """How a distance or a count may be written."""
    whole = int(v)
    words = NUM_FR if lang == "fr" else NUM_EN
    if float(whole) == v:
        return "num:%d|%s" % (whole, words[whole])
    half = ("%s mètres et demi|%s et demi" % (words[whole], words[whole]) if lang == "fr"
            else "%s and a half|%s metres and a half|%s meters and a half" % ((words[whole],) * 3))
    return "num:%s|%s" % (repr(v), half)


def other_words(keep):
    return "|".join(WORDS[k] for k in WORDS if k not in keep)


def make(seed=20261113, out=OUT):
    rng = random.Random(seed)
    scenarios, items = [], []

    def scenario(layout, kind):
        places = LAYOUTS[layout]
        names = [p[0] for p in places]
        for _ in range(100000):
            at, going, doing, arrivals, task = "none", "none", "stand", [], "-"
            if kind == "idle" or kind == "failed":
                x, y, yaw = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-180.0, 180.0)
                if kind == "failed":
                    # it set off for a place and never got there: standing in the room, the task failed
                    task = "failed:" + rng.choice(names)
            elif kind == "at" or kind == "done2":
                at = rng.choice(names)
                px, py = [(p[1], p[2]) for p in places if p[0] == at][0]
                # 0.7 m from the place's centre, on the room's side, facing it
                back = math.atan2(-py, -px) + rng.uniform(-0.4, 0.4)
                x, y = px + 0.7 * math.cos(back), py + 0.7 * math.sin(back)
                yaw = wrap(math.degrees(back) + 180.0)
                prev = rng.sample([n for n in names if n != at], rng.choice([1, 2]))
                arrivals = prev[::-1] + [at]  # oldest first
                doing = "goto:" + at
                task = "done:" + at
                if kind == "done2":
                    # a task of two steps, both done: the place before, then this one
                    task = "done:" + arrivals[-2] + ">" + at
            else:  # walking, or `two`: walking on the first step of a task of two
                going = rng.choice(names)
                was = rng.choice([n for n in names if n != going])
                x, y = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
                px, py = [(p[1], p[2]) for p in places if p[0] == going][0]
                yaw = wrap(math.degrees(math.atan2(py - y, px - x)) + rng.uniform(-15.0, 15.0))
                arrivals = [was]
                doing = "goto:" + going
                task = "run:" + going + "@0"
                if kind == "two":
                    # the first step of two: after this place, another one
                    then = rng.choice([n for n in names if n != going and n != was])
                    task = "run:" + going + ">" + then + "@0"
            if clear_of_edges(places, x, y, yaw, at):
                x, y, yaw = round(x, 3), round(y, 3), round(yaw, 2)
                if clear_of_edges(places, x, y, yaw, at):
                    return dict(layout=layout, places=places, x=x, y=y, yaw=yaw, doing=doing, going=going, at=at,
                                arrivals=arrivals, task=task)
        raise RuntimeError("no scenario for " + layout + " " + kind)

    def add_scenario(s):
        s["sid"] = "s%02d" % len(scenarios)
        scenarios.append(s)
        return s

    def item(s, cat, lang, q, must, forbid="", alt_must="", alt_forbid=""):
        items.append(("i%03d" % len(items), s["sid"], cat, lang, q, must, forbid, alt_must, alt_forbid))

    def where_truth(s, name, lang):
        p = [a for a in around(s["places"], s["x"], s["y"], s["yaw"]) if a[0] == name][0]
        forbid = "|".join(DIR[k] for k in DIR if k != p[1])
        return p, DIR[p[1]], forbid

    plan = [("A", "idle"), ("B", "idle"), ("A", "at"), ("B", "at"), ("C", "at"), ("A", "walking"),
            ("B", "walking"), ("C", "walking")]
    for layout, kind in plan:
        s = add_scenario(scenario(layout, kind))
        names = [p[0] for p in s["places"] if p[0] != s["at"]]
        asked = rng.sample(names, 2)
        for lang in ("fr", "en"):
            art = FR if lang == "fr" else EN
            for name in asked:
                p, must, forbid = where_truth(s, name, lang)
                q = "Où est %s ?" % art[name] if lang == "fr" else "Where is %s?" % art[name]
                item(s, "where_dir", lang, q, must, forbid)
            name = [n for n in names if n not in asked][0]
            p, _, _ = where_truth(s, name, lang)
            q = ("À quelle distance est %s ?" % art[name] if lang == "fr" else "How far is %s?" % art[name])
            half = "et demi|and a half|,5|.5" if float(int(p[2])) == p[2] else ""
            item(s, "where_dist", lang, q, num_alts(p[2], lang), half)
            ar = around(s["places"], s["x"], s["y"], s["yaw"])
            nearest = min(ar, key=lambda a: a[6])[0]
            q = ("Quel est l'endroit le plus proche de toi ?" if lang == "fr"
                 else "Which place is the closest to you?")
            item(s, "nearest", lang, q, WORDS[nearest], other_words([nearest]))
            if kind == "walking":
                q = "Que fais-tu en ce moment ?" if lang == "fr" else "What are you doing right now?"
                item(s, "doing", lang, q, WORDS[s["going"]], other_words([s["going"], s["arrivals"][-1]]))
            if kind == "at":
                q = "Où es-tu en ce moment ?" if lang == "fr" else "Where are you right now?"
                item(s, "doing", lang, q, WORDS[s["at"]])
            if kind != "idle":
                before = s["arrivals"][-2] if kind == "at" else s["arrivals"][-1]
                q = "Où étais-tu juste avant ?" if lang == "fr" else "Where were you just before?"
                item(s, "before", lang, q, WORDS[before])
        # the mutant: the first asked place swapped with a place in ANOTHER sector
        base = around(s["places"], s["x"], s["y"], s["yaw"])
        a = asked[0]
        sec = {p[0]: p[1] for p in base}
        partners = [n for n in names if n != a and sec[n] != sec[a]]
        if partners:
            b = partners[0]
            xy = {p[0]: (p[1], p[2]) for p in s["places"]}
            swapped = [(n, xy[b][0], xy[b][1]) if n == a else (n, xy[a][0], xy[a][1]) if n == b else (n, px, py)
                       for n, px, py in s["places"]]
            m = add_scenario(dict(s, places=swapped, mutant_of=s["sid"]))
            for lang in ("fr", "en"):
                art = FR if lang == "fr" else EN
                _, must, forbid = where_truth(m, a, lang)
                _, alt_must, alt_forbid = where_truth(s, a, lang)
                q = "Où est %s ?" % art[a] if lang == "fr" else "Where is %s?" % art[a]
                item(m, "where_mut", lang, q, must, forbid, alt_must, alt_forbid)

    # questions whose answer does not depend on the pose: asked in two scenarios
    general = {
        "can_yes": [("Est-ce que tu sais marcher ?", "Can you walk?"),
                    ("Peux-tu lever les bras ?", "Can you raise your arms?"),
                    ("Est-ce que tu sais t'accroupir ?", "Are you able to squat?")],
        "can_no": [("Peux-tu ramasser la tasse sur la table ?", "Can you pick up the cup on the table?"),
                   ("Peux-tu ouvrir la porte ?", "Can you open the door?"),
                   ("Peux-tu t'asseoir sur le canapé ?", "Can you sit on the sofa?"),
                   ("Est-ce que tu peux sauter ?", "Can you jump?"),
                   ("Peux-tu monter les escaliers ?", "Can you climb the stairs?"),
                   ("Peux-tu me préparer un café ?", "Can you make me a coffee?")],
        "unknown": [("Où est la télévision ?", "Where is the television?"),
                    ("Où sont mes clés ?", "Where are my keys?"),
                    ("Y a-t-il une plante dans la pièce ?", "Is there a plant in the room?")],
        "self": [("As-tu des jambes ?", "Do you have legs?"),
                 ("Combien de bras as-tu ?", "How many arms do you have?"),
                 ("Comment t'appelles-tu ?", "What is your name?")],
        "chat": [("Bonjour, comment vas-tu ?", "Hello, how are you?"),
                 ("Raconte-moi une blague.", "Tell me a joke.")],
    }
    for s in (scenarios[0], [x for x in scenarios if x["at"] != "none"][0]):
        for cat, qs in general.items():
            for k, (qf, qe) in enumerate(qs):
                for lang, q in (("fr", qf), ("en", qe)):
                    if cat == "can_yes":
                        item(s, cat, lang, q, YES, REFUSAL_ONLY)
                    elif cat == "can_no":
                        item(s, cat, lang, q, NO, AFFIRM)
                    elif cat == "unknown":
                        # saying it does not know is the answer, whatever else the reply adds
                        item(s, cat, lang, q, UNKNOWN, "" if k < 2 else "oui|yes")
                    elif cat == "self":
                        must = [YES, num_alts(2.0, lang), "noé|noe"][k]
                        item(s, cat, lang, q, must, REFUSAL_ONLY if k == 0 else "")
                    else:
                        item(s, cat, lang, q, "", "ne peux pas|peux pas|cannot|can't|unable|pas capable")

    # HELD OUT: other questions of the same kinds, never looked at while a prompt's wording is chosen. The
    # view's "can / cannot" sentence was reworded once on the can_no rows above; these say what it is worth.
    held = {
        "held_can_no": [("Est-ce que tu sais nager ?", "Can you swim?"),
                        ("Peux-tu voler ?", "Can you fly?"),
                        ("Peux-tu conduire une voiture ?", "Can you drive a car?"),
                        ("Peux-tu porter mon sac ?", "Can you carry my bag?"),
                        ("Peux-tu fermer la fenêtre ?", "Can you close the window?"),
                        ("Peux-tu t'allonger par terre ?", "Can you lie down on the floor?"),
                        ("Est-ce que tu peux écrire une lettre ?", "Can you write a letter?"),
                        ("Peux-tu grimper sur la table ?", "Can you climb onto the table?")],
        "held_can_yes": [("Peux-tu tourner sur toi-même ?", "Can you turn around?"),
                         ("Es-tu capable de te déplacer dans la pièce ?", "Are you able to move around the room?"),
                         ("Est-ce que tu peux me parler ?", "Can you talk to me?")],
        "held_unknown": [("Où est mon téléphone ?", "Where is my phone?"),
                         ("Y a-t-il un chat dans la pièce ?", "Is there a cat in the room?")],
    }
    s = [x for x in scenarios if x["going"] != "none"][0]
    for cat, qs in held.items():
        for k, (qf, qe) in enumerate(qs):
            for lang, q in (("fr", qf), ("en", qe)):
                if cat == "held_can_no":
                    item(s, cat, lang, q, NO, AFFIRM)
                elif cat == "held_can_yes":
                    item(s, cat, lang, q, YES, REFUSAL_ONLY)
                else:
                    item(s, cat, lang, q, UNKNOWN, "" if k == 0 else "oui|yes")

    # TASKS (B2): what the executive's record lets the robot answer
    YES_DONE = "oui|yes|terminé|terminée|fini|finie|done|finished|c'est fait|completed"
    NOT_DONE = "non|no|pas encore|not yet|pas fini|pas terminé|not finished|not done|en cours|still|encore|in progress"
    NOT_REACHED = ("non|no|pas réussi|could not|couldn't|n'ai pas|did not|didn't|pas arrivé|pas atteint|failed|échoué|"
                   "not reached|unable|pas pu|wasn't able|was not able")
    for layout, kind in (("A", "two"), ("B", "two"), ("C", "done2"), ("A", "done2"), ("B", "failed"), ("C", "failed")):
        s = add_scenario(scenario(layout, kind))
        for lang in ("fr", "en"):
            art = FR if lang == "fr" else EN
            if kind == "two":
                first, then = s["task"][4:-2].split(">")
                q = "Que feras-tu après ça ?" if lang == "fr" else "What will you do after that?"
                item(s, "task_next", lang, q, WORDS[then], other_words([then, first]))
                q = "As-tu fini ce que je t'ai demandé ?" if lang == "fr" else "Have you finished what I asked?"
                item(s, "task_state", lang, q, NOT_DONE, "oui|yes")
            elif kind == "done2":
                q = "As-tu fini ce que je t'ai demandé ?" if lang == "fr" else "Have you finished what I asked?"
                item(s, "task_state", lang, q, YES_DONE, "non|no|pas encore|not yet")
            else:
                place = s["task"][7:]
                q = "Es-tu arrivé %s ?" % FR_AT[place] if lang == "fr" else "Did you reach %s?" % art[place]
                item(s, "task_state", lang, q, NOT_REACHED, "oui|yes")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "bb1_scenarios.tsv"), "w", encoding="utf-8") as f:
        f.write("# sid\tx\ty\tyaw\tdoing\tgoing\tat\tarrivals\tplaces\tref_lines\tref_nearest\tref_before\tmutant_of\ttask\n")
        for s in scenarios:
            ar = around(s["places"], s["x"], s["y"], s["yaw"])
            nearest = min(ar, key=lambda a: a[6])[0]
            trail = []
            for a in s["arrivals"]:
                if not trail or trail[0] != a:
                    trail.insert(0, a)
            if trail and s["at"] != "none" and trail[0] == s["at"]:
                trail = trail[1:]
            f.write("\t".join([
                s["sid"], repr(s["x"]), repr(s["y"]), repr(s["yaw"]), s["doing"], s["going"], s["at"],
                ",".join(s["arrivals"]) or "-",
                ",".join("%s:%s:%s" % (n, repr(float(px)), repr(float(py))) for n, px, py in s["places"]),
                " ; ".join(line(p) for p in ar), nearest, ",".join(trail[:4]) or "-", s.get("mutant_of", "-"),
                s["task"],
            ]) + "\n")
    with open(os.path.join(out, "bb1_items.tsv"), "w", encoding="utf-8") as f:
        f.write("# id\tsid\tcat\tlang\tquestion\tmust\tforbid\talt_must\talt_forbid\n")
        for it in items:
            f.write("\t".join(x if x else "-" for x in it) + "\n")
    cats = {}
    for it in items:
        cats[it[2]] = cats.get(it[2], 0) + 1
    print(len(scenarios), "scenarios,", len(items), "items:", cats)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261113, help="the committed set is the default seed's")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    make(a.seed, a.out)
