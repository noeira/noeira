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
    - You have no task right now. Your last task: walk to the table. It is done.
    - Places you know, from where you stand: fridge: ahead of you, 3 m; door: on your left, 4 m; table: here; sofa: behind you, 4.5 m.
    - In French these places are: fridge = le frigo; door = la porte; table = la table; sofa = le canapé.
    - The nearest place to you: table.
    - Before this you were at: sofa, then door.
    - You know of no other place or object in this room.
    - You are talking with a person whose name is Denis: you recognise their voice.

TWO PARTS, FOR THE PREFIX CACHE. `stable` (who the robot is, what it can do)
changes when the body changes; `live` (the facts) changes every turn. The
stable part comes first so that a model's cache can keep it.

⚠ WHO IS SPEAKING IS THE LAST FACT, FOR THE SAME CACHE. A model's cache keeps a
prompt up to its first changed token (`Lfm3Chat.keep_system`), and the voice
loop computes the prompt while the person is still speaking
(`ChatService.prepare`). The speaker's line reads "still being checked" until
the voice has been compared, so it changes during most turns: as the FIRST
fact it left 6 replies of 17 in the room demo with a third of their prompt in
the cache (docs/NN_MODEL_PORTS_PLAN.md §12.8.5).

⚠ AND THE OTHER LINES STAY WHERE THEY WERE. Ordering all of them by how often
they change (names, task, position, places around, speaker) was tried and
scored one item lower on each bench: BB-1's task questions 14 / 16 for 15,
held out 25 / 26 for 26, the restart gate 11 / 14 for 12 on one scenario.

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
from .state import WorldState, TASK_DONE, TASK_FAILED
from .memory import focused_memory
from .people import asks_identity, identity_answer


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


def speaker_sentence(ref ws: WorldState) -> String:
    """Who the robot is talking with, from `noeira.brain.people`; "" for a
    robot without a people store.

    ⚠ "A PERSON WHOSE NAME IS", NEVER A BARE QUOTE. Given the remembered
    sentence « Je m'appelle Denis. », the reply model answered that ITS name
    was Denis (the restart gate, B3). The name is stated as the person's."""
    if not ws.people_on:
        return String("")
    var t = String("You do not know who is talking to you: nobody has told you their name yet.")
    if ws.speaker_said_name != "":
        return "The person talking to you has just told you their name: " + ws.speaker_said_name + ". That is the person's name, not yours."
    if ws.speaker_pending:
        t = String("The voice of the person talking to you is still being checked: you do not know yet who it is.")
    elif ws.speaker != "" and ws.speaker_sure:
        t = "You are talking with a person whose name is " + ws.speaker + ": you recognise their voice."
    elif ws.speaker != "":
        t = ("The voice of the person talking to you resembles the voice of " + ws.speaker
             + ", a person you know, but you are not sure it is " + ws.speaker + ".")
    elif len(ws.people_known) > 0:
        # ⚠ NO NAME HERE. With "the people you know by name: Denis" in this line, the reply model greeted
        # a stranger with « je ne reconnais pas la voix de Denis »: whom the robot knows is not for a
        # voice it does not know.
        t = String("You do not recognise the voice of the person talking to you: you do not know who it is.")
    if asks_identity(ws.asked):
        # the answer is computed, not composed (`people.identity_answer`)
        t += " The person is asking who they are. Answer with exactly this sentence and nothing else: \"" + identity_answer(ws) + "\""
    return t


def task_sentences(ref ws: WorldState) -> String:
    """The task in hand, or the last one and how it ended, in words (from
    `noeira.brain.tasks`' record): what "what are you doing?", "what next?"
    and "did you get there?" are answered from."""
    if ws.has_task:
        var n = len(ws.task.steps)
        var k = ws.task.at
        if n == 1:
            return "Your task: " + ws.task.describe() + ". It is in progress."
        var t = "Your task: " + ws.task.describe() + ". You are on step " + String(k + 1) + " of " + String(n) + ": " + ws.task.steps[k].label + "."
        if k + 1 < n:
            t += " Next: " + ws.task.steps[k + 1].label + "."
        else:
            t += " It is the last step."
        return t
    if not ws.has_last_task:
        return String("You have no task right now.")
    var t = "You have no task right now. Your last task: " + ws.last_task.describe() + "."
    if ws.last_task.status == TASK_DONE:
        return t + " It is done."
    if ws.last_task.status == TASK_FAILED:
        return t + " It failed: " + ws.last_task.reason + "."
    return t + " It was cancelled: " + ws.last_task.reason + "."


def memory_section(ref ws: WorldState) -> String:
    """What the log says of earlier sessions and days (`noeira.brain.memory`),
    as a section of its own after the facts; "" for a robot without a memory.

    ⚠ APART FROM "true right now", AND WITH ITS EDGE SAID. Mixed into the
    facts, yesterday's walk was told as this morning's; and asked about last
    week, the model moved yesterday there (the restart gate,
    docs/ROBOT_BRAIN_PLAN.md B3)."""
    if not ws.memory_on:
        return String("")
    var t = String("\n\nWhat you remember from before this session (your own log, by day):")
    # when the utterance names a time ("hier", "this morning"), only that time's lines
    for m in focused_memory(ws):
        t += "\n- " + m
    return t


def reply_live(ref ws: WorldState, style: Int = REPLY_VIEW_LINES) -> String:
    """The facts of this moment."""
    var around = ws.around()
    var before = ws.been_before()
    var nearest = ws.nearest()
    if style == REPLY_VIEW_SENTENCES:
        var t = "Right now: " + _now_line(ws) + " " + task_sentences(ws)
        if ws.people_on:
            t += " " + speaker_sentence(ws)
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
        return t + memory_section(ws)
    var t = String("Facts, true right now:\n")
    t += "- " + _now_line(ws) + "\n- " + task_sentences(ws) + "\n"
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
    if ws.people_on:
        t += "\n- " + speaker_sentence(ws)  # last: the line that changes most (module note)
    return t + memory_section(ws)


def reply_view(ref ws: WorldState, style: Int = REPLY_VIEW_LINES) -> ReplyView:
    return ReplyView(reply_stable(ws), reply_live(ws, style))
