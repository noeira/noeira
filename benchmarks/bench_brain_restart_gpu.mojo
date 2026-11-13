# +--------------------------------------------------------------------------+ #
# | The restart gate: a new process, questions about the old ones (LFM3-1.7B)
# +--------------------------------------------------------------------------+ #
"""Does the robot still know what it did, after its process is gone?
(docs/ROBOT_BRAIN_PLAN.md, step B3.)

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_brain_restart_gpu.mojo -o build/bench_restart
    ./build/bench_restart write <robot dir> [a|b]            # process 1, then it exits
    ./build/bench_restart ask <robot dir> [a|b] [out.tsv] [LFM3-1.7B dir]   # process 2
    ./build/bench_restart ask <an empty dir> ...             # the mutant: nothing to remember

`write` lives two sessions in `<robot dir>` with a clock of its own
(2026-11-12 and the morning of 2026-11-13), through the real state, executive
and log, and saves the places. Two scenarios, the same shape with other
content (`b` is the held-out one: the memory's wording was adjusted on `a`):

    a   yesterday: "Je m'appelle Denis.", walked to the fridge (done), the sofa not reached (failed)
        this morning: "Va à la table, puis à la porte." -> both steps done
    b   yesterday: "Je m'appelle Claire.", walked to the sofa (done), the door not reached (failed)
        this morning: "Va au frigo, puis à la table." -> both steps done

`ask` is another process, on the afternoon of 2026-11-13. It builds a state
with NO place given, reads the places and the memory back from the folder
(`load_places`, `restore_memory`, `People.load`), and asks the reply model
eighteen questions (nine in French, nine in English) as the voice loop does, scored by rule
(`noeira.brain.bb1.reply_passes`): what it did yesterday (yesterday's place and
none of this morning's), whether it reached the place it failed to reach, what
it did this morning, whether it has been to this morning's second place, the
speaker's name (as the speaker's, not its own; the people store was read back
too, and the question comes with the person's voice), what was asked this
morning, where a place is; and two it must NOT answer: what it did last week,
and the name of a voice it has never heard.

On an empty folder only the last kind can pass: that run shows the answers of
the first come from the folder.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.brain import (
    WorldState, Body, PlaceXY, Task, Step, give_task, tick_tasks, restore_memory, save_places, load_places,
    reply_view, places_around, People, asks_identity, identity_answer,
)
from noeira.brain.bb1 import reply_passes, reply_in_language
from noeira.llm.lfm3 import Lfm3Chat, ChatSettings, answer_of
from noeira.text.spoken import spoken_text


comptime DAY: Int = 20770 * 86400
"""2026-11-13T00:00:00Z: `write`'s today, and `ask`'s."""

comptime YES = "oui|yes|bien sûr|of course|absolument|tout à fait|indeed|en effet"
comptime FORGOT = "ne sais pas|sais pas|souviens pas|pas de souvenir|aucun souvenir|don't remember|do not remember|no memory|don't know|do not know|not sure"
comptime FAILED = "non|no|pas réussi|could not|couldn't|failed|échoué|not reached|pas atteint|did not manage|didn't manage|didn't make it|pas pu|did not reach|didn't reach|wasn't able|was not able|unable"
comptime NOTHING = FORGOT + "|rien|nothing|n'ai pas|don't have|do not have|pas d'information|no information|no record|aucune information"


def walk(mut ws: WorldState, place: String, clock: Float64, now: Int):
    """The step in hand, from its start to the arrival at `place`."""
    _ = tick_tasks(ws, clock, now)
    ws.observe_motion(String("goto:") + place, place, String("none"), now)
    ws.observe_motion(String("goto:") + place, String("none"), place, now + 5)
    _ = tick_tasks(ws, clock + 5.0, now + 5)
    _ = tick_tasks(ws, clock + 5.1, now + 5)


@fieldwise_init
struct Scenario(Copyable, Movable):
    var person: String
    var y_done: String
    var y_failed: String
    var t_first: String
    var t_second: String

    @staticmethod
    def named(which: String) -> Scenario:
        if which == "b":
            return Scenario("Claire", "sofa", "door", "fridge", "table")
        return Scenario("Denis", "fridge", "sofa", "table", "door")


def fr_to(place: String) -> String:
    """`au frigo`, `à la porte`: "to the <place>" in French."""
    if place == "fridge":
        return String("au frigo")
    if place == "door":
        return String("à la porte")
    if place == "table":
        return String("à la table")
    return String("au canapé")


def words_of(place: String) -> String:
    """How a reply may name a place, French or English."""
    if place == "fridge":
        return String("frigo|fridge|réfrigérateur|refrigerateur|refrigerator")
    if place == "door":
        return String("porte|door")
    if place == "table":
        return String("table")
    return String("canapé|canape|sofa|couch")


def write(dir: String, sc: Scenario) raises:
    var y0 = DAY - 86400 + 10 * 3600      # yesterday, 10:00 UTC
    var y = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    y.places = [PlaceXY("fridge", 3.0, 1.5), PlaceXY("door", -3.2, 2.0), PlaceXY("table", 1.0, -2.5), PlaceXY("sofa", -2.0, -2.8)]
    y.call_place(String("fridge"), String("fr"), String("le frigo"))
    y.call_place(String("door"), String("fr"), String("la porte"))
    y.call_place(String("table"), String("fr"), String("la table"))
    y.call_place(String("sofa"), String("fr"), String("le canapé"))
    restore_memory(y, dir, y0)
    y.open_log(dir, y0)
    save_places(y, dir)
    y.heard(String("Bonjour."), y0 + 10)
    y.said(String("Bonjour !"), y0 + 11)
    y.heard("Je m'appelle " + sc.person + ".", y0 + 20)
    # the person gives their name: the people store keeps it with the voice that said it
    var people = People(dir.copy())
    y.people_on = True
    people.heard(y, "Je m'appelle " + sc.person + ".", y0 + 20)
    var their_voice: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    people.voice(y, their_voice, y0 + 22)
    if people.error != "":
        raise Error("the people store was not written: " + people.error)
    y.said("Enchanté, " + sc.person + ".", y0 + 21)
    y.heard("Va " + fr_to(sc.y_done) + ".", y0 + 30)
    var a = Task(String(""))
    a.steps.append(Step.goto(sc.y_done))
    give_task(y, a^, 0.0, y0 + 31)
    walk(y, sc.y_done, 0.0, y0 + 31)
    y.heard("Va " + fr_to(sc.y_failed) + ".", y0 + 60)
    var b = Task(String(""))
    b.steps.append(Step.goto(sc.y_failed))
    give_task(y, b^, 30.0, y0 + 61)
    _ = tick_tasks(y, 30.0, y0 + 61)
    y.observe_motion(String("goto:") + sc.y_failed, sc.y_failed, String("none"), y0 + 61)
    _ = tick_tasks(y, 91.0, y0 + 122)          # past the 60 s deadline
    y.said(String("Je n'ai pas réussi."), y0 + 122)
    if y.log_error != "":
        raise Error("the log was not written: " + y.log_error)

    var m0 = DAY + 9 * 3600               # this morning, 09:00 UTC: another state, as another process has
    var m = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    restore_memory(m, dir, m0)
    m.open_log(dir, m0)
    m.heard("Va " + fr_to(sc.t_first) + ", puis " + fr_to(sc.t_second) + ".", m0 + 10)
    var c = Task(String(""))
    c.steps.append(Step.goto(sc.t_first))
    c.steps.append(Step.goto(sc.t_second))
    give_task(m, c^, 0.0, m0 + 11)
    walk(m, sc.t_first, 0.0, m0 + 11)
    walk(m, sc.t_second, 10.0, m0 + 21)
    m.said(String("C'est fait."), m0 + 27)
    print("written:", dir, "| this morning's process remembered", len(m.remembered), "line(s) of yesterday")


struct Q(Copyable, Movable):
    var cat: String
    var lang: String
    var text: String
    var must: String
    var forbid: String
    var stranger: Bool
    """Asked by a voice the robot has never heard."""

    def __init__(out self, cat: String, lang: String, text: String, must: String, forbid: String, stranger: Bool = False):
        self.cat = cat.copy()
        self.lang = lang.copy()
        self.text = text.copy()
        self.must = must.copy()
        self.forbid = forbid.copy()
        self.stranger = stranger


def side_words(bearing: String) -> String:
    if bearing == "ahead":
        return String("devant|en face|tout droit|ahead|in front|straight")
    if bearing == "left":
        return String("gauche|côté gauche|my left|your left|the left|left side")
    if bearing == "right":
        return String("droite|mon droit|côté droit|my right|your right|the right|right side")
    return String("derrière|derriere|dans mon dos|behind")


def ask(dir: String, sc: Scenario, out_path: String, model_dir: String) raises:
    var now = DAY + 15 * 3600
    var ws = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    var n_places = load_places(ws, dir)
    restore_memory(ws, dir, now)
    ws.observe_pose(0.0, 0.0, 90.0)
    ws.observe_motion(String("stand"), String("none"), String("none"), now)
    print("restart gate:", dir, "|", n_places, "places read back,", len(ws.remembered), "line(s) remembered",
          ("| " + ws.memory_error) if ws.memory_error != "" else String(""))
    for l in ws.remembered:
        print("   ", l)
    var asked_place = sc.t_first.copy()      # a place question: read back from places.json
    var side = String("ahead")
    for p in places_around(ws.places, 0.0, 0.0, 90.0):
        if p.name == asked_place:
            side = p.bearing.copy()
    var en: List[String] = ["the " + sc.y_failed, "the " + sc.t_second, "the " + asked_place]
    var today_both = words_of(sc.t_first) + " && " + words_of(sc.t_second)
    var all_places = String("frigo|fridge|réfrigérateur|refrigerator|table|canapé|sofa|porte|door")
    # a name question is right only when the name is the PERSON's: "je m'appelle <name>" is the robot naming itself
    var self_named = "je m'appelle " + sc.person + "|my name is " + sc.person + "|i am " + sc.person + "|je suis " + sc.person + "|i'm " + sc.person
    var qs = List[Q]()
    # yesterday's own places, and none of this morning's
    qs.append(Q("yesterday", "fr", "Qu'as-tu fait hier ?", words_of(sc.y_done), words_of(sc.t_first) + "|" + words_of(sc.t_second)))
    qs.append(Q("yesterday", "en", "What did you do yesterday?", words_of(sc.y_done), words_of(sc.t_first) + "|" + words_of(sc.t_second)))
    qs.append(Q("failed", "fr", "Hier, as-tu réussi à aller " + fr_to(sc.y_failed) + " ?", String(FAILED), String(YES) + "|" + String(FORGOT)))
    qs.append(Q("failed", "en", "Yesterday, did you manage to get to " + en[0] + "?", String(FAILED), String(YES) + "|" + String(FORGOT)))
    qs.append(Q("this_morning", "fr", "Qu'as-tu fait ce matin ?", today_both, String("")))
    qs.append(Q("this_morning", "en", "What did you do this morning?", today_both, String("")))
    # a "yes" is the answer; "yes ... I don't know if I went back since" is still a yes
    qs.append(Q("been_there", "fr", "Es-tu déjà allé " + fr_to(sc.t_second) + " aujourd'hui ?", String(YES), String("non|no")))
    qs.append(Q("been_there", "en", "Have you been to " + en[1] + " today?", String(YES), String("non|no")))
    qs.append(Q("name", "fr", "Comment je m'appelle ?", sc.person.copy(), self_named))
    qs.append(Q("name", "en", "What is my name?", sc.person.copy(), self_named))
    qs.append(Q("asked", "fr", "Qu'est-ce que je t'ai demandé ce matin ?", today_both, String("")))
    qs.append(Q("asked", "en", "What did I ask you this morning?", today_both, String("")))
    qs.append(Q("place", "fr", "Où est " + ws.word_for(asked_place, String("fr")) + " ?", side_words(side), String("")))
    qs.append(Q("place", "en", "Where is " + en[2] + "?", side_words(side), String("")))
    qs.append(Q("too_old", "fr", "Qu'as-tu fait la semaine dernière ?", String(NOTHING), all_places))
    qs.append(Q("too_old", "en", "What did you do last week?", String(NOTHING), all_places))
    # the same question from a voice the robot has never heard: it must not hand out the name it knows
    var unknown_voice = String(FORGOT) + "|ne reconnais pas|don't recognise|don't recognize|do not recognise|do not recognize|ne te connais pas|ne vous connais pas"
    qs.append(Q("stranger", "fr", "Comment je m'appelle ?", unknown_voice, sc.person.copy(), True))
    qs.append(Q("stranger", "en", "What is my name?", unknown_voice, sc.person.copy(), True))

    # who is talking: the store read back from the folder; every question comes with a voice, the
    # person's own (a little different from the one stored: cosine 0.9) or a stranger's
    var people = People.load(dir)
    people.dir = String("")          # asking must not write to the folder
    ws.people_on = True
    ws.people_known = people.names()
    var near_voice: List[Float32] = [0.9, 0.0, 0.43589, 0.0]
    var other_voice: List[Float32] = [0.0, 1.0, 0.0, 0.0]
    print("    people read back:", len(people.list), ("| " + people.error) if people.error != "" else String(""))
    var chat = Lfm3Chat.load(DeviceContext(), model_dir, 4096, True)
    var settings = ChatSettings()
    settings.max_tokens = 120
    var tsv = String("cat\tlang\tpass\tin_lang\tsys_tokens\tms\tquestion\treply\n")
    var ok_n = 0
    var lang_n = 0
    var known_ok = 0
    var known_n = 0
    var total_ms = 0.0
    var sys_tokens = 0
    for q in qs:
        ws.language = q.lang.copy()
        ws.asked = q.text.copy()     # the utterance being answered: its time words select the memory
        people.heard(ws, q.text, now)
        if q.stranger:
            people.voice(ws, other_voice, now)
        else:
            people.voice(ws, near_voice, now)
        var system = reply_view(ws).text()
        chat.reset(system)
        sys_tokens = len(chat.prompt_ids())
        chat.user(q.text)
        var t0 = perf_counter_ns()
        chat.start(settings)
        while chat.next():
            pass
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        total_ms += ms
        var reply = spoken_text(answer_of(chat.reply_text()))
        if asks_identity(q.text):
            # as the demo does (`G1VoiceLoop.set_reply_override`): who the speaker is, is computed, never composed
            reply = identity_answer(ws)
        var ok = reply_passes(reply, q.must, q.forbid)
        var in_lang = reply_in_language(reply, q.lang)
        if ok:
            ok_n += 1
        if in_lang:
            lang_n += 1
        if q.cat != "too_old" and q.cat != "stranger":
            known_n += 1
            if ok:
                known_ok += 1
        print("  PASS" if ok else "  fail", q.cat, q.lang, "|", q.text, "->", reply)
        tsv += (q.cat + "\t" + q.lang + "\t" + ("1" if ok else "0") + "\t" + ("1" if in_lang else "0") + "\t" + String(sys_tokens) + "\t"
                + String(Int(ms)) + "\t" + q.text + "\t" + reply.replace("\t", " ").replace("\n", " ") + "\n")
    print("questions about earlier sessions:", known_ok, "/", known_n, "| all:", ok_n, "/", len(qs), "| in the question's language:", lang_n, "/", len(qs))
    print("system prompt", sys_tokens, "tokens; reply", Int(total_ms / Float64(len(qs))), "ms (mean)")
    if out_path != "":
        with open(out_path, "w") as f:
            f.write(tsv)


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: bench_restart write <robot dir> [a|b] | ask <robot dir> [a|b] [out.tsv] [LFM3-1.7B dir]")
    var sc = Scenario.named(String(args[3]) if len(args) > 3 else String("a"))
    if String(args[1]) == "write":
        write(String(args[2]), sc)
    else:
        ask(
            String(args[2]), sc,
            String(args[4]) if len(args) > 4 else String(""),
            String(args[5]) if len(args) > 5 else getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B",
        )
