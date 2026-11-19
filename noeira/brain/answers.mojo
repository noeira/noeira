# +--------------------------------------------------------------------------+ #
# | Answers the robot computes — said as they are, the reply model not asked
# +--------------------------------------------------------------------------+ #
"""A question whose answer the state holds EXACTLY is answered by code
(docs/ROBOT_BRAIN_PLAN.md §16, §17):

    ws.asked = heard                       # the utterance being answered
    var ct = computed_turn(ws)             # ct.text "" = the model replies
    voice.set_reply_override(ct.text, ct.final, ct.decides, ct.command)

Six questions today:

- who the speaker is (`noeira.brain.people.identity_answer`, B4);
- whether the robot has finished what it was asked (`task_done_answer`);
- where a movable object is (`object_answer`, from `ws.objects`);
- an order to carry one (`carry_turn`): a task of `pick:` / `put:` steps
  when the body carries it, a no otherwise (« Porte la caisse à la table »
  must not send the robot walking to the table);
- what it can do (`capabilities_answer`, from `Body.skills`);
- whether it can do one thing, or an order to do it (`can_turn`): a limit is a
  no, a skill with a command is run (`ct.command`), and both SETTLE the turn
  (`ct.decides` False: d1 is not asked further).

And how long the model's own reply may be: `reply_sentences(ws)` (one
sentence; two for a joke or a story), for the voice loop's
`set_reply_sentences`, which stops the reply there.

⚠ WHY NOT THE REPLY MODEL. With "Your last task: walk to the table, then walk
to the door. It is done." in its prompt, LFM3-1.7B answered « As-tu fini ce
que je t'ai demandé ? » with « Non, j'ai encore besoin de te dire où est la
table et la cuisine pour continuer. » (the room demo on the Orin). The task
record is exact; a sentence about it is not left to a 1.7 B model.

⚠ "" WHEN THE STATE CANNOT SAY. A spoken gesture ("squat") is not a task yet:
it cancels the task in hand and leaves no record of its own. So a cancelled
task, a gesture in progress, or no task at all give "" — the model replies, as
before — rather than a confident sentence about the wrong request.
"""

from .bb1 import normalise, has_phrase
from .events import EV_BEGAN, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED
from .geometry import PlaceXY, places_around
from .objects import OBJ_HELD, OBJ_PUT, HOW_TOLD, HOW_REMEMBERED
from .people import asks_identity, identity_answer
from .state import WorldState, Skill, TASK_DONE, TASK_FAILED


def _plain(text: String) -> List[UInt8]:
    """Normalised, with hyphens as spaces: a recogniser writes « as-tu » and
    « as tu » alike."""
    var t = normalise(text)
    for i in range(len(t)):
        if t[i] == 45:
            t[i] = 32
    return t^


def _any(ref text: List[UInt8], forms: List[String]) -> Bool:
    for f in forms:
        var pat = _plain(f)
        if has_phrase(text, pat):
            return True
    return False


def _gesture_since_last_task(ref ws: WorldState) -> Bool:
    """A command that is not a task (a spoken "squat") began after the last
    task ended, as far back as the state's events go."""
    var i = len(ws.events) - 1
    while i >= 0:
        var k = ws.events[i].kind.copy()
        if k == EV_TASK_DONE or k == EV_TASK_FAILED or k == EV_TASK_CANCELLED:
            return False
        if k == EV_BEGAN:
            var c = ws.events[i].text.copy()
            if c != "stand" and not c.startswith("goto"):
                return True
        i -= 1
    return False


def asks_task_done(text: String) -> Bool:
    """Whether a sentence asks the robot if it has finished (French and
    English).

    An inverted form is a question by itself (« as-tu fini », "have you
    finished"). A plain one (« tu as fini ») counts only with a question mark
    or « est-ce que »: « Quand tu as fini, va à la porte. » is a request."""
    var t = _plain(text)
    var inverted: List[String] = [
        "as-tu fini", "as-tu terminé", "as-tu réussi", "avez-vous fini", "avez-vous terminé", "avez-vous réussi",
        "est-ce fini", "est-ce terminé", "est-ce que c'est fait", "est-ce que c'est fini", "est-ce que c'est terminé",
        "have you finished", "have you completed", "are you finished", "are you done", "did you finish",
        "did you complete", "is it done", "is it finished",
    ]
    if _any(t, inverted):
        return True
    var plain: List[String] = [
        "tu as fini", "tu as terminé", "tu as réussi", "t'as fini", "t'as terminé", "vous avez fini", "vous avez terminé",
        "c'est fini", "c'est terminé", "you have finished", "you are done", "you finished", "it is done",
    ]
    if not _any(t, plain):
        return False
    var marks: List[String] = ["?", "est-ce que", "est-ce qu'"]
    return _any(t, marks)


def task_done_answer(ref ws: WorldState) -> String:
    """The exact sentence that answers "have you finished?" from the task
    record, in the conversation's language (French, else English); "" when
    the record cannot say (module note).

    ⚠ A TEMPLATE, like `tasks.notice_text` and `people.identity_answer`."""
    var fr = ws.language == "fr"
    if ws.has_task:
        var n = len(ws.task.steps)
        var k = ws.task.at
        if n > 1:
            if fr:
                return "Pas encore : j'en suis à l'étape " + String(k + 1) + " sur " + String(n) + "."
            return "Not yet: I am on step " + String(k + 1) + " of " + String(n) + "."
        var place = ws.task.steps[k].place.copy()
        if place != "":
            var word = ws.word_for(place, ws.language)
            if fr:
                return "Pas encore, je suis en route vers " + word + "."
            return "Not yet, I am on my way to " + word + "."
        return String("Pas encore, c'est en cours.") if fr else String("Not yet, I am still doing it.")
    if not ws.has_last_task:
        return String("")
    # a gesture asked after the last task ended is what the person means, and it has no record
    if _gesture_since_last_task(ws):
        return String("")
    if ws.last_task.status == TASK_DONE:
        return String("Oui, j'ai fini ce que tu m'as demandé.") if fr else String("Yes, I have finished what you asked.")
    if ws.last_task.status == TASK_FAILED:
        var k = ws.last_task.at
        var place = String("")
        if k < len(ws.last_task.steps):
            place = ws.last_task.steps[k].place.copy()
        if place != "":
            var word = ws.word_for(place, ws.language)
            if fr:
                return "Non, je n'ai pas réussi à atteindre " + word + "."
            return "No, I could not reach " + word + "."
        return String("Non, je n'ai pas su le faire.") if fr else String("No, I could not do it.")
    return String("")


def computed_reply_final(ref ws: WorldState) -> Bool:
    """Whether `computed_reply(ws)` is the answer for good: False while the
    voice behind « Comment je m'appelle ? » is still being checked (its answer
    is then « Un instant… », and the next frames give the name). For
    `G1VoiceLoopOf.set_reply_override`'s `final`: the loop's `computed_first`
    says only a final answer."""
    return not (ws.people_on and asks_identity(ws.asked) and ws.speaker_pending)


def computed_reply(ref ws: WorldState) -> String:
    """What to say to `ws.asked` when code holds the answer; "" = the reply
    model answers. For `G1VoiceLoopOf.set_reply_override`, once a frame.
    `computed_turn` gives the same text with what the decision may still do."""
    return computed_turn(ws).text


# ── what the body can do (« Tu peux faire quoi ? », « Tu peux danser ? ») ──


def _letter(c: UInt8) -> Bool:
    return (c >= 97 and c <= 122) or (c >= 48 and c <= 57) or c >= 128


def _starts(ref t: List[UInt8], at: Int, pat: List[UInt8]) -> Bool:
    """`pat` at `at` in `t`, ending at a word's end."""
    var m = len(pat)
    if m == 0 or at + m > len(t):
        return False
    for k in range(m):
        if t[at + k] != pat[k]:
            return False
    return not (_letter(pat[m - 1]) and at + m < len(t) and _letter(t[at + m]))


def _head(ref t: List[UInt8], robot: String) -> Int:
    """Where the sentence's own words begin: after a call (« Noé, », « hé »)
    and a filler (« allez », « maintenant »), and the punctuation between."""
    var fillers: List[String] = [
        "hé", "hey", "eh", "allez", "alors", "bon", "ok", "okay", "d'accord", "maintenant", "now", "please",
        "vas y", "s'il te plaît", "s'il te plait", "stp",
    ]
    var name = _plain(robot)
    var i = 0
    var moved = True
    while moved:
        while i < len(t) and (t[i] == 32 or t[i] == 44 or t[i] == 46 or t[i] == 33 or t[i] == 58):
            i += 1
        moved = False
        if len(name) > 0 and _starts(t, i, name.copy()):
            i += len(name)
            moved = True
            continue
        for f in fillers:
            var p = _plain(f)
            if _starts(t, i, p.copy()):
                i += len(p)
                moved = True
                break
    return i


def asks_capabilities(text: String) -> Bool:
    """Whether a sentence asks the robot what it can do (French and English)."""
    var t = _plain(text)
    var forms: List[String] = [
        "tu peux faire quoi", "tu sais faire quoi", "tu sais faire quelles choses", "qu'est-ce que tu sais faire",
        "qu'est-ce que tu peux faire", "qu'est ce que tu sais faire", "qu'est ce que tu peux faire",
        "que sais-tu faire", "que peux-tu faire", "que savez-vous faire", "que pouvez-vous faire",
        "vous pouvez faire quoi", "vous savez faire quoi", "c'est quoi tes capacités", "quelles sont tes capacités",
        "what can you do", "what are you able to do", "what do you know how to do", "what can you do for me",
    ]
    return _any(t, forms)


def asks_can(text: String) -> Bool:
    """Whether a sentence asks the robot whether it can do something: an
    inverted form is a question by itself (« peux-tu », "can you"); a plain one
    (« tu peux ») counts with a question mark or « est-ce que », as in
    `asks_task_done`."""
    var t = _plain(text)
    var inverted: List[String] = [
        "peux-tu", "pourrais-tu", "sais-tu", "pouvez-vous", "pourriez-vous", "savez-vous",
        "can you", "could you", "are you able to",
    ]
    if _any(t, inverted):
        return True
    var plain: List[String] = ["tu peux", "tu pourrais", "tu sais", "vous pouvez", "vous pourriez", "vous savez", "tu es capable"]
    if not _any(t, plain):
        return False
    var marks: List[String] = ["?", "est-ce que", "est-ce qu'"]
    return _any(t, marks)


def _named(ref t: List[UInt8], skills: List[Skill]) -> Tuple[Int, Int]:
    """The skill a sentence names, by its longest matching form (« saut
    périlleux » is the backflip, not « saut »): (index, form length), (-1, 0)."""
    var best = -1
    var best_n = 0
    for i in range(len(skills)):
        for f in skills[i].forms:
            var p = _plain(f)
            if len(p) > best_n and has_phrase(t, p):
                best = i
                best_n = len(p)
    return (best, best_n)


def _then_or_place(ref ws: WorldState, ref t: List[UInt8]) -> Bool:
    """The sentence goes on to a place or a next step (« fais un squat puis va
    à la porte »): the decision reads it, not a rule."""
    var seq: List[String] = ["puis", "ensuite", "après", "et après", "then", "after that", "afterwards"]
    if _any(t, seq):
        return True
    for w in ws.place_words:
        var p = _plain(w.word)
        if has_phrase(t, p):
            return True
    for pl in ws.places:
        var p = _plain(pl.name)
        if has_phrase(t, p):
            return True
    return False


def _a_fr(word: String) -> String:
    """« à » before a French place word: au frigo, à la porte, à l'étagère, aux toilettes."""
    if word.startswith("le "):
        return "au " + String(word[byte=3:])
    if word.startswith("les "):
        return "aux " + String(word[byte=4:])
    return "à " + word


def _list_fr(items: List[String], last: String) -> String:
    var s = String("")
    for i in range(len(items)):
        if i > 0:
            s += String(" ") + last + String(" ") if i == len(items) - 1 else String(", ")
        s += items[i]
    return s


def capabilities_answer(ref ws: WorldState) -> String:
    """« Je peux aller au frigo, à la porte…, tourner, faire un squat, lever les
    bras, et des figures comme danser ou faire un salto. » from the body's
    skills, in the conversation's language (French, else English).

    ⚠ A TEMPLATE, LIKE `task_done_answer`. Asked « Tu peux faire quoi ? » in the
    live room (2026-11-18), d1 refused it as « no clear pick » and nothing was
    said; asked « Tu peux danser ? » of a body that dances, the reply model
    answered « je suis un robot sans capacité de mouvement ».

    ⚠ SHORT, BECAUSE THE ROBOT IS DEAF WHILE IT SPEAKS. Every skill listed was
    14 s of speech on the SuperMimic body, and the replay lost the next two
    sentences under it (the echo gate, round 42). The motions are named by two
    examples; talking, which the person is doing, is not listed."""
    var fr = ws.language == "fr"
    var items = List[String]()
    var motions = List[Int]()
    for i in range(len(ws.body.skills)):
        ref s = ws.body.skills[i]
        if s.motion:
            motions.append(i)
        elif s.to_places and len(ws.places) > 0:
            var ps = List[String]()
            for p in ws.places:
                ps.append(_a_fr(ws.word_for(p.name, String("fr"))) if fr else ws.word_for(p.name, ws.language))
            items.append((String("aller ") if fr else String("walk to ")) + _list_fr(ps, String("ou") if fr else String("or")))
        elif s.talking():
            continue
        elif fr:
            items.append(s.fr)
        else:
            items.append(s.en.replace(" your ", " my "))
    if len(motions) > 0:
        # the two named: the first of these the body has, then its own order
        var prefer: List[String] = ["dance", "backflip", "boxing", "jump", "wave", "dab", "stretch"]
        var picked = List[Int]()
        for want in prefer:
            for i in motions:
                if ws.body.skills[i].command == want and len(picked) < 2:
                    picked.append(i)
        for i in motions:
            var seen = False
            for k in picked:
                if k == i:
                    seen = True
            if not seen and len(picked) < 2:
                picked.append(i)
        var shown = List[String]()
        for i in picked:
            shown.append(ws.body.skills[i].fr if fr else ws.body.skills[i].en)
        if fr:
            items.append(String("des figures comme ") + _list_fr(shown, String("ou")))
        else:
            items.append(String("moves such as: ") + _list_fr(shown, String("or")))
    if len(items) == 0:
        return String("")
    if fr:
        return "Je peux " + _list_fr(items, String("et")) + "."
    return "I can " + _list_fr(items, String("and")) + "."


struct ComputedTurn(Copyable, Movable):
    """What code says to an utterance, and what the decision may still do
    (`G1VoiceLoopOf.set_reply_override`)."""

    var text: String
    """The answer; "" = none (the reply model answers, the decision decides)."""
    var final: Bool
    """`computed_reply_final`."""
    var decides: Bool
    """False: the answer settles the turn and d1's requests are dropped (a
    limit, the list of skills, a skill run by `command`)."""
    var command: String
    """What the body runs with the answer ("" = nothing)."""

    def __init__(out self, var text: String, final: Bool = True, decides: Bool = True, var command: String = String("")):
        self.text = text^
        self.final = final
        self.decides = decides
        self.command = command^


def can_turn(ref ws: WorldState) -> ComputedTurn:
    """« Tu peux danser ? » / « Danse ! » / « Tu peux ouvrir la porte ? »: what
    the body's skills say (`Body.skills`, `Body.limits`).

    - asked whether it can, or told to, do a skill that has a `command`: done,
      « Oui, regarde ! » / « D'accord ! », d1 not asked further;
    - asked whether it can do a limit: « Non, je ne peux pas ouvrir la porte. »;
    - anything else (a skill the decision runs — a place, a direction —, a
      thing not in either list, a sentence that goes on to a place or a next
      step): no answer, the decision and the reply model as before."""
    var none = ComputedTurn(String(""))
    var fr = ws.language == "fr"
    var t = _plain(ws.asked)
    var asking = asks_can(ws.asked)
    var sk = _named(t, ws.body.skills)
    var lm = _named(t, ws.body.limits)
    if asking and lm[0] >= 0 and lm[1] > sk[1]:
        var l = ws.body.limits[lm[0]].copy()
        if fr:
            return ComputedTurn("Non, je ne peux pas " + l.fr + ".", True, False)
        return ComputedTurn("No, I cannot " + l.en + ".", True, False)
    if sk[0] < 0 or ws.body.skills[sk[0]].command == "" or _then_or_place(ws, t):
        return none^
    var s = ws.body.skills[sk[0]].copy()
    if asking:
        return ComputedTurn(String("Oui, regarde !") if fr else String("Yes, watch!"), True, False, s.command)
    # told to: the sentence's own first word names it (« Danse ! »), or « fais … » does (« Fais un dab. »)
    var h = _head(t, ws.body.name)
    var told = False
    for f in s.forms:
        if _starts(t, h, _plain(f)):
            told = True
    var dos: List[String] = ["fais", "fait", "refais", "do", "go"]
    for d in dos:
        if _starts(t, h, _plain(d)):
            told = True
    if told:
        return ComputedTurn(String("D'accord !") if fr else String("Okay!"), True, False, s.command)
    return none^


comptime REPLY_SENTENCES: Int = 1
"""Sentences a reply may say (`reply_sentences`): the robot is deaf while it
speaks, and the replays lost the sentences said over its second one (§17.6)."""
comptime REPLY_SENTENCES_MORE: Int = 2
"""When the person asked for more than an answer: a joke is a question and its
punchline. Three let LFM3 add « — Mais je ne suis pas sûr de ce qui est une
bonne blague ici… », and « Il fait beau aujourd'hui », said over it, was
heard as « Aujourd'hui. » (round 43)."""


def asks_for_more(text: String) -> Bool:
    """Whether a sentence asks for more than one sentence's answer: a joke, a
    story, an explanation, a description (French and English)."""
    var t = _plain(text)
    var forms: List[String] = [
        "raconte", "racontes", "racontez", "raconter", "blague", "blagues", "histoire", "histoires",
        "explique", "expliques", "expliquez", "expliquer", "décris", "décrire", "décrivez",
        "dis-m'en plus", "dites-m'en plus", "parle-moi de", "parlez-moi de", "en détail",
        "tell me a", "tell me about", "tell me more", "joke", "jokes", "story", "stories", "explain", "describe",
    ]
    return _any(t, forms)


def reply_sentences(ref ws: WorldState) -> Int:
    """How many sentences the reply to `ws.asked` may say, for the voice loop's
    `set_reply_sentences`: `REPLY_SENTENCES`, or `REPLY_SENTENCES_MORE` when
    the person asked for more (`asks_for_more`)."""
    return REPLY_SENTENCES_MORE if asks_for_more(ws.asked) else REPLY_SENTENCES


# ── movable objects (« Où est la caisse ? », « Porte la caisse à la table. ») ──


def _no_article(word: String) -> String:
    """`la caisse` -> `caisse`, `l'étagère` -> `étagère`, `the crate` -> `crate`."""
    var arts: List[String] = ["le ", "la ", "les ", "un ", "une ", "the ", "a "]
    for a in arts:
        if word.startswith(a):
            return String(word[byte=a.byte_length():])
    if word.startswith("l'"):
        return String(word[byte=2:])
    return word.copy()


def _forms_of(ws: WorldState, name: String) -> List[String]:
    """The words that name a place or an object: its name, and each of its
    words with and without the article."""
    var out: List[String] = [name.replace("_", " ")]
    for w in ws.place_words:
        if w.place == name:
            out.append(w.word.copy())
            out.append(_no_article(w.word))
    return out^


def object_named(ws: WorldState, text: String) -> Int:
    """The object a sentence names (`ws.objects`), by its longest form; -1 for none."""
    var t = _plain(text)
    var best = -1
    var best_n = 0
    for i in range(len(ws.objects)):
        var forms = _forms_of(ws, ws.objects[i].name)
        forms.append(ws.objects[i].label.copy())
        for f in forms:
            var p = _plain(f)
            if len(p) > best_n and has_phrase(t, p):
                best = i
                best_n = len(p)
    return best


def place_named(ws: WorldState, text: String) -> String:
    """The place a sentence names (`ws.places`), by its longest form; "" for none."""
    var t = _plain(text)
    var best = String("")
    var best_n = 0
    for pl in ws.places:
        for f in _forms_of(ws, pl.name):
            var p = _plain(f)
            if len(p) > best_n and has_phrase(t, p):
                best = pl.name.copy()
                best_n = len(p)
    return best


def asks_where(text: String) -> Bool:
    """Whether a sentence asks where something is (French and English)."""
    var t = _plain(text)
    var forms: List[String] = [
        "où est", "où se trouve", "où sont", "où tu as mis", "où as-tu mis", "où tu l'as mis", "où l'as-tu mis",
        "c'est où", "where is", "where's", "where are", "where did you put",
    ]
    return _any(t, forms)


def _cap(s: String) -> String:
    """The first letter upper case (ASCII): `la caisse` -> `La caisse`."""
    if s.byte_length() == 0:
        return s.copy()
    var b = s.as_bytes()
    if b[0] >= 97 and b[0] <= 122:
        return chr(Int(b[0]) - 32) + String(s[byte=1:])
    return s.copy()


def _pres_de(word: String) -> String:
    """« près de » before a French noun phrase: près de la table, près du frigo, près des chaises."""
    if word.startswith("le "):
        return "près du " + String(word[byte=3:])
    if word.startswith("les "):
        return "près des " + String(word[byte=4:])
    return "près de " + word


def _metres_fr(d: Float64) -> String:
    """`2 mètres`, `2,5 mètres`, `0,5 mètre`: a distance already rounded to 0.5 m."""
    var whole = Int(d)
    var n = String(whole) if Float64(whole) == d else String(d).replace(".", ",")
    return n + (String(" mètre") if d < 2.0 else String(" mètres"))


def _metres_en(d: Float64) -> String:
    var whole = Int(d)
    return (String(whole) if Float64(whole) == d else String(d)) + " m"


def object_answer(ref ws: WorldState, i: Int) -> String:
    """Where object `i` is, as the record has it, in the conversation's
    language (French, else English): « La caisse est devant moi, à 2 mètres,
    près de la table. », « La caisse est dans mes mains. », « Je ne sais pas
    où est la caisse. ». A told or remembered pose says so (« D'après ce
    qu'on m'a dit, … », « Si je me souviens bien, … »).

    ⚠ A TEMPLATE, like `task_done_answer`: the record is exact, and the
    places' arithmetic is `geometry`'s, from where the robot stands."""
    var fr = ws.language == "fr"
    ref o = ws.objects[i]
    var word = ws.word_for(o.name, ws.language)
    if o.missing or not o.has_pose:
        return ("Je ne sais pas où est " + word + ".") if fr else ("I do not know where " + word + " is.")
    if o.where == OBJ_HELD:
        # the hands know: no "I was told"
        return _cap(word) + (String(" est dans mes mains.") if fr else String(" is in my hands."))
    var where = String("")
    if o.where == OBJ_PUT and o.put_at != "":
        var pw = ws.word_for(o.put_at, ws.language)
        where = _pres_de(pw) if fr else ("next to " + pw)
    else:
        var entries: List[PlaceXY] = [PlaceXY(o.name, o.x, o.y)]
        for p in ws.places:
            entries.append(p.copy())
        var a = places_around(entries, ws.x, ws.y, ws.yaw_deg)[0].copy()
        var parts = List[String]()
        if ws.pose_known:
            if a.distance_m < 1.0:
                parts.append(String("juste à côté de moi") if fr else String("right next to me"))
            else:
                var side = String("")
                if a.bearing == "ahead":
                    side = String("devant moi") if fr else String("ahead of me")
                elif a.bearing == "left":
                    side = String("à ma gauche") if fr else String("on my left")
                elif a.bearing == "right":
                    side = String("à ma droite") if fr else String("on my right")
                else:
                    side = String("derrière moi") if fr else String("behind me")
                parts.append(side)
                parts.append(("à " + _metres_fr(a.distance_m)) if fr else (_metres_en(a.distance_m) + " away"))
        if a.next_to != "none":
            var nw = ws.word_for(a.next_to, ws.language)
            parts.append(_pres_de(nw) if fr else ("next to " + nw))
        if len(parts) == 0:
            return ("Je ne sais pas où est " + word + ".") if fr else ("I do not know where " + word + " is.")
        for k in range(len(parts)):
            if k > 0:
                where += ", "
            where += parts[k]
    var said = (word + " est " + where) if fr else (word + " is " + where)
    if o.how == HOW_TOLD:
        return ("D'après ce qu'on m'a dit, " + said + ".") if fr else ("I was told " + said + ".")
    if o.how == HOW_REMEMBERED:
        return ("Si je me souviens bien, " + said + ".") if fr else ("As far as I remember, " + said + ".")
    return _cap(said) + "."


def object_turn(ref ws: WorldState) -> ComputedTurn:
    """« Où est la caisse ? »: `object_answer`, the turn settled. "" when the
    sentence asks for no object the state knows."""
    if not asks_where(ws.asked):
        return ComputedTurn(String(""))
    var i = object_named(ws, ws.asked)
    if i < 0:
        return ComputedTurn(String(""))
    return ComputedTurn(object_answer(ws, i), True, False)


def _verb_of(ref ws: WorldState, ref t: List[UInt8]) -> Int:
    """0 none, 1 take (prends, ramasse, pick up), 2 carry (porte, apporte,
    bring), 3 put (pose, mets, put): the verb of an order about an object.

    ⚠ A FRENCH IMPERATIVE COUNTS AT THE SENTENCE'S HEAD ONLY: « porte » is
    also the door (« La caisse est près de la porte, prends-la. » is a take,
    not a carry), « range » a row. The infinitives (« tu peux porter … ? »)
    and the English verbs count anywhere."""
    var h = _head(t, ws.body.name)
    var put_head: List[String] = ["pose", "dépose", "mets", "range", "place"]
    var put_any: List[String] = ["poser", "déposer", "mettre", "ranger", "put", "set down"]
    var carry_head: List[String] = ["porte", "apporte", "rapporte", "amène", "ramène", "emmène", "emporte", "déplace"]
    var carry_any: List[String] = [
        "porter", "apporter", "rapporter", "amener", "ramener", "emmener", "emporter", "déplacer", "carry", "bring", "move", "take",
    ]
    var take_head: List[String] = ["prends", "ramasse", "attrape", "soulève"]
    var take_any: List[String] = ["prendre", "ramasser", "attraper", "soulever", "pick up", "grab", "lift"]
    for f in put_head:
        if _starts(t, h, _plain(f)):
            return 3
    for f in carry_head:
        if _starts(t, h, _plain(f)):
            return 2
    for f in take_head:
        if _starts(t, h, _plain(f)):
            return 1
    if _any(t, put_any):
        return 3
    if _any(t, carry_any):
        return 2
    if _any(t, take_any):
        return 1
    return 0


def carry_turn(ref ws: WorldState) -> ComputedTurn:
    """« Porte la caisse à la table. », « Prends la caisse. », « Tu peux poser
    la caisse près du frigo ? »: an order (or a request) about an object the
    state knows, by its verb.

    - with a place: the task `task:put:<object>:<place>`;
    - without one, taken or carried: `task:pick:<object>`; put down with no
      place: no answer (the decision reads it);
    - a body that does not carry it (`Body.can_carry`): « Non, je ne peux pas
      porter la caisse. », the turn settled. ⚠ Left to the decision, « Porte
      la caisse à la table » named a place, and a place is a walk.

    The answer is « D'accord ! » / « Oui ! »: the executive's notices say how
    it ends (`noeira.brain.tasks`)."""
    var none = ComputedTurn(String(""))
    var t = _plain(ws.asked)
    var verb = _verb_of(ws, t)
    if verb == 0:
        return none^
    var i = object_named(ws, ws.asked)
    if i < 0:
        return none^
    var fr = ws.language == "fr"
    var obj = ws.objects[i].name.copy()
    var place = place_named(ws, ws.asked)
    var asking = asks_can(ws.asked)
    if not ws.body.can_carry(obj):
        var word = ws.word_for(obj, ws.language)
        var v_fr = String("prendre") if verb == 1 else (String("poser") if verb == 3 else String("porter"))
        var v_en = String("pick up") if verb == 1 else (String("put down") if verb == 3 else String("carry"))
        if fr:
            return ComputedTurn("Non, je ne peux pas " + v_fr + " " + word + ".", True, False)
        return ComputedTurn("No, I cannot " + v_en + " " + word + ".", True, False)
    var cmd = String("")
    if place != "":
        cmd = "task:put:" + obj + ":" + place
    elif verb == 3:
        return none^
    else:
        cmd = "task:pick:" + obj
    var yes = (String("Oui !") if fr else String("Yes!")) if asking else (String("D'accord !") if fr else String("Okay!"))
    return ComputedTurn(yes, True, False, cmd)


def computed_turn(ref ws: WorldState) -> ComputedTurn:
    """What code answers to `ws.asked`, and what the decision may still do;
    `text` "" = nothing computed. In order: who the speaker is, whether the task
    is finished, where an object is, an order to carry one, what the robot can
    do, whether it can do one thing."""
    if ws.people_on and asks_identity(ws.asked):
        return ComputedTurn(identity_answer(ws), computed_reply_final(ws))
    if asks_task_done(ws.asked):
        return ComputedTurn(task_done_answer(ws))
    if len(ws.objects) > 0:
        var ot = object_turn(ws)
        if ot.text != "":
            return ot^
        var cr = carry_turn(ws)
        if cr.text != "":
            return cr^
    if asks_capabilities(ws.asked):
        return ComputedTurn(capabilities_answer(ws), True, False)
    return can_turn(ws)
