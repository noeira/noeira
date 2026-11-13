# +--------------------------------------------------------------------------+ #
# | Views — the state as the text one model reads
# +--------------------------------------------------------------------------+ #
"""`reply_view(ws)`: the system prompt of the model that speaks for the robot,
written from the `WorldState` every turn (docs/ROBOT_BRAIN_PLAN.md §5.2).

    var v = reply_view(ws)
    llm.start(messages, v.text())

    You are Noé, a small humanoid robot with two legs, two arms and a head. You are in a
    furnished room and you talk with the people around you.
    Reply in ONE short sentence, in French, in plain words: no emoji. Be warm and brief.
    You can: walk to a place of the room; turn; squat; raise your arms; talk and answer questions.
    That list is complete. Asked whether you can do anything else, answer no, kindly. In particular
    you cannot: pick things up or carry them; open the door; sit down.
    Answer questions about the room and about yourself ONLY from the facts below. ...

    Facts, true right now:
    - You are standing at the table.
    - Places you know, from where you stand: fridge: ahead of you, 3 m; door: on your left, 4 m; table: here; sofa: behind you, 4.5 m.
    - In French these places are: fridge = le frigo; door = la porte; table = la table; sofa = le canapé.
    - The nearest place to you: table.
    - Before this you were at: sofa, then door.
    - You know of no other place or object in this room.

TWO PARTS, FOR THE PREFIX CACHE. `stable` (who the robot is, what it can do)
changes when the body changes; `live` (the facts) changes every turn. The
stable part comes first so that a model's cache can keep it.

⚠ THE WORDING IS PART OF THE MODEL'S INPUT, AND THIS IS ITS ONE COPY. It is
measured by `benchmarks/bench_brain_grounded_gpu.mojo` (BB-1); change a
sentence, re-run the bench. `REPLY_VIEW_SENTENCES` is the same facts as plain
sentences, kept so the bench can compare the two.

⚠ THE ARITHMETIC IS DONE HERE (`geometry`): the model is given "left, 4 m" and
"the nearest place", never coordinates.

The decision model's view is not here yet: it is `g1_command_state`
(`noeira/envs/robots/g1_command_language.mojo`), whose `around` lines the
state already writes (`WorldState.around_lines`).
"""

from .geometry import PlaceAround
from .state import WorldState


comptime REPLY_VIEW_LINES: Int = 0
comptime REPLY_VIEW_SENTENCES: Int = 1


@fieldwise_init
struct ReplyView(Copyable, Movable):
    var stable: String
    var live: String

    def text(self) -> String:
        return self.stable + "\n\n" + self.live


def metres(d: Float64) -> String:
    """`3 m`, `3.5 m`: a distance already rounded to 0.5 m, without `.0`."""
    var whole = Int(d)
    if Float64(whole) == d:
        return String(whole) + " m"
    return String(d) + " m"


def _join(items: List[String], sep: String) -> String:
    var t = String("")
    for i in range(len(items)):
        if i > 0:
            t += sep
        t += items[i]
    return t


def _side(bearing: String) -> String:
    """`ahead` -> "ahead of you" ...: the bearing as a phrase."""
    if bearing == "ahead":
        return String("ahead of you")
    if bearing == "left":
        return String("on your left")
    if bearing == "right":
        return String("on your right")
    return String("behind you")


def language_name(code: String) -> String:
    """`fr` -> "French"; an unknown or empty code -> "the language you were
    addressed in" (the model then has to tell by itself)."""
    if code == "fr":
        return String("French")
    if code == "en":
        return String("English")
    if code == "de":
        return String("German")
    if code == "es":
        return String("Spanish")
    if code == "it":
        return String("Italian")
    return String("the language you were addressed in")


def reply_stable(ref ws: WorldState) -> String:
    """Who the robot is and what it can do: the part a cache keeps."""
    var t = "You are " + ws.body.name + ", " + ws.body.description + ". You are in " + ws.where
    t += " and you talk with the people around you.\n"
    t += "Reply in ONE short sentence, in " + language_name(ws.language) + ", in plain words: no emoji. Be warm and brief.\n"
    t += "You can: " + _join(ws.body.can, "; ") + ".\n"
    t += "That list is complete. Asked whether you can do anything else, answer no, kindly."
    if len(ws.body.cannot) > 0:
        t += " In particular you cannot: " + _join(ws.body.cannot, "; ") + "."
    t += "\n"
    t += "Answer questions about the room and about yourself ONLY from the facts below. "
    t += "If the answer is not there, say you do not know."
    return t


def _now_line(ref ws: WorldState) -> String:
    if ws.walking_to != "":
        return "You are walking to the " + ws.walking_to + "."
    var t = String("You are standing in the room, not at a named place.")
    if ws.at != "":
        t = "You are standing at the " + ws.at + "."
    if ws.doing != "" and not ws.doing.startswith("goto") and ws.doing != "stand":
        t += " Your current movement: " + ws.doing.replace("_", " ") + "."
    return t


def reply_live(ref ws: WorldState, style: Int = REPLY_VIEW_LINES) -> String:
    """The facts of this moment."""
    var around = ws.around()
    var before = ws.been_before()
    var nearest = ws.nearest()
    if style == REPLY_VIEW_SENTENCES:
        var t = "Right now: " + _now_line(ws)
        for p in around:
            if p.name == ws.at:
                continue
            t += " The " + p.name + " is " + _side(p.bearing) + ", " + metres(p.distance_m) + " away."
        var words_s = ws.words_in(ws.language)
        if len(words_s) > 0:
            t += " In " + language_name(ws.language) + " these places are: " + _join(words_s, "; ") + "."
        if nearest != "":
            t += " The nearest place to you is the " + nearest + "."
        if len(before) == 1:
            t += " Before this you were at the " + before[0] + "."
        elif len(before) > 1:
            t += " Before this you were at the " + before[0] + ", and before that at the " + before[1] + "."
        else:
            t += " You have not been anywhere else yet."
        t += " You know of no other place or object in this room."
        return t
    var t = String("Facts, true right now:\n- ") + _now_line(ws) + "\n"
    if len(around) > 0:
        var items = List[String]()
        for p in around:
            if p.name == ws.at:
                items.append(p.name + ": here")
            else:
                # ⚠ "on your right", never a bare "right": the model read "table: right, 3.5 m" as
                # "right next to me" (BB-1's mutant rows)
                items.append(p.name + ": " + _side(p.bearing) + ", " + metres(p.distance_m))
        t += "- Places you know, from where you stand: " + _join(items, "; ") + ".\n"
    var words = ws.words_in(ws.language)
    if len(words) > 0:
        t += "- In " + language_name(ws.language) + " these places are: " + _join(words, "; ") + ".\n"
    if nearest != "":
        t += "- The nearest place to you: " + nearest + ".\n"
    if len(before) > 0:
        t += "- Before this you were at: " + _join(before, ", then ") + ".\n"
    else:
        t += "- You have not been anywhere else yet.\n"
    t += "- You know of no other place or object in this room."
    return t


def reply_view(ref ws: WorldState, style: Int = REPLY_VIEW_LINES) -> ReplyView:
    return ReplyView(reply_stable(ws), reply_live(ws, style))
