"""`talk_first`: a sentence the triage reads as talk is answered after the first request, and the commands request still follows.

    pixi run mojo run -I . tests/robots/test_g1_voice_loop_talk_first.mojo      (CPU, no microphone, about 1 s)

WHY THIS EXISTS. A chat turn paid both decision requests before its reply
(~0.5 + 0.7 s on the Orin, docs/NN_MODEL_PORTS_PLAN.md §12.8.6). With
`talk_first` the reply starts after the first request. But d1's triage named
two stops `talk` in the room bench's logs ("no, not the table, stop" 0.51),
so the commands request is NOT skipped: it runs once the reply is out, and
its decision acts as before. What this pins:

1. A greeting the triage reads as talk at 0.95: the reply starts after ONE
   request, and the commands request is asked when the reply is out, with the
   state the first one had. Its `talk` answer says nothing more.
2. A stop the triage read as talk at 0.75: a reply, then the stop command
   still comes out of the commands request.
3. Talk at 0.55, under the bar: the order of today (commands, then reply).
4. `talk_first` off: the order of today even at 0.95.
5. Code holds the answer: the order of today, and the answer said is the finished one.
6. One reply a decision.
7. (the start-up fault, not `talk_first`) a model that failed to load is named by `failed()`,
   and the listening line says so instead of "every model ready".

`computed_first` (docs/NN_MODEL_PORTS_PLAN.md §12.8.11): a computed answer is
said while d1 still decides, and the decision still acts.

8. « As-tu fini ? », the commands answer `talk`: the answer comes out BEFORE
   either of d1's answers, both requests are still asked, the reply model is
   never asked, and d1's `talk` says nothing more.
9. The same, d1 picks `stand` (the `--see` replays): the answer, then the stop.
10. Not final yet (the voice check behind « Comment je m'appelle ? »): nothing
    said until it is, then the final answer, once.
11. The previous turn's answer, still set while the loop decides the new
    utterance (a caller that takes the words a poll late): never said.
12. (S3 item 2) the two-place request carries no triage `kind`; the
    destination request keeps it.
13. (`tts_no_play_gated`, a replay's echo gate) a piece "playing" in
    silence counts as speaking only when gated: the microphone is then muted
    as it is live.

A computed answer that SETTLES the turn (`set_reply_override`'s `decides`
False, `noeira.brain.answers.can_turn`, live room 2026-11-18):

14. « Tu peux ouvrir la porte ? »: the answer at once, d1's request in flight
    dropped (cancelled, its answer never read), nothing else said or run, the
    loop idle.
15. « Tu peux danser ? » with the command `dance`: ONE event, the command,
    carrying the answer it was said with; d1 dropped the same way.
16. (`reply_refused`) an addressed sentence d1 refuses (« no clear pick »)
    gets a reply from the model; off, it stays silent as before.

The loop is the real `G1VoiceLoopOf`, its microphone closed, a stand-in
recogniser handing it the sentence, a stand-in decision answering from a
script, and a stand-in reply model.
"""

from std.sys import exit
from std.time import perf_counter_ns

from noeira.ai.chat import ChatMessage, ChatResponse, ChatService, ToolSpec
from noeira.ai.jev import DecisionService, JevAnswers, JevQuestions
from noeira.ai.speech import SpeechToTextService, TextToSpeechService, Transcript
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import G1Context, G1_TRIAGE_NARROW
from noeira.envs.robots.g1_voice_loop import (
    G1VoiceConfig, G1VoiceLoopOf, VL_ST_STT, VL_ST_IDLE, VL_COMMAND, VL_TALK,
)
from noeira.io.json import JsonWriter, parse_json
from noeira.io.wav import WavAudio


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8](capacity=s.byte_length())
    for x in s.as_bytes():
        b.append(x)
    return b^


# ── the stand-ins ─────────────────────────────────────────────────────────

struct TextStt(Movable, SpeechToTextService):
    """Answers `text` at once."""

    var text: String

    def __init__(out self):
        self.text = String("")

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        return Self()

    def set_language(mut self, lang: String):
        pass

    def set_hint(mut self, vocab: String) -> Bool:
        return False

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, ref audio: WavAudio) raises:
        pass

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return True

    def result(mut self) raises -> Transcript:
        return Transcript(self.text, 50.0)

    def cancel(mut self) raises:
        pass


struct ScriptDecision(Movable, DecisionService):
    """Answers each request with the next JSON of `script`, after `polls`
    polls. Records which request it was: `D` (it asks the triage `kind`) or
    `C` (it asks the `command`), and the state it was asked with."""

    var script: List[String]
    var asked: String
    var states: List[String]
    var wait: Int
    var polls: Int
    var broken: Bool
    """Its model "failed to load" (`load_failed`)."""
    var cancelled: Int

    def __init__(out self):
        self.broken = False
        self.cancelled = 0
        self.script = List[String]()
        self.asked = String("")
        self.states = List[String]()
        self.wait = 0
        self.polls = 2

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def load_failed(self) -> Bool:
        return self.broken

    def warm_up(mut self):
        pass

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        var w = JsonWriter()
        questions.write_to_json(w)
        var qs = w.done()
        self.asked += String("D") if qs.find("\"kind\"") >= 0 else (String("C") if qs.find("\"command\"") >= 0 else String("?"))
        self.states.append(state_json)
        self.wait = self.polls

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        self.asked += String("S")
        self.wait = self.polls

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        self.wait -= 1
        return self.wait <= 0

    def result(mut self) raises -> JevAnswers:
        if len(self.script) == 0:
            raise Error("test: no answer left in the script")
        var j = self.script.pop(0)
        return JevAnswers(parse_json(_bytes(j)), 500.0)

    def cancel(mut self) raises:
        self.cancelled += 1


struct ScriptChat(Movable, ChatService):
    """A reply that takes `polls` polls."""

    var asked: Int
    var wait: Int
    var polls: Int

    def __init__(out self):
        self.asked = 0
        self.wait = 0
        self.polls = 3

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        return Self()

    def set_reply_limits(mut self, max_tokens: Int) raises:
        pass

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(
        mut self,
        ref messages: List[ChatMessage],
        system: String = String(""),
        tools: List[ToolSpec] = List[ToolSpec](),
        stream: Bool = True,
    ) raises:
        self.asked += 1
        self.wait = self.polls

    def poll(mut self, timeout_ms: Int = 0) raises -> String:
        self.wait -= 1
        return String("")

    def done(self) -> Bool:
        return self.wait <= 0

    def result(mut self) raises -> ChatResponse:
        var r = ChatResponse()
        r.text = String("Bonjour !")
        r.latency_ms = 300.0
        return r^

    def cancel(mut self) raises:
        pass


struct NoTts(Movable, TextToSpeechService):
    def __init__(out self):
        pass

    @staticmethod
    def from_spec(spec: String, voice: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, text: String) raises:
        raise Error("test: muted")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return True

    def result_wav_bytes(mut self) raises -> List[UInt8]:
        raise Error("test: muted")

    def cancel(mut self) raises:
        pass


# ── the answers ───────────────────────────────────────────────────────────

def first_answer(talk_p: Float64) -> String:
    """The destination request: no place, not a request, triage top `talk`."""
    var rest = (1.0 - talk_p) / 2.0
    return (String('{"model":"script","usage":{"input_tokens":819},"answers":{')
            + String('"destination":{"choice":"none","probabilities":{"none":0.92,"fridge":0.08}},')
            + String('"is_request":{"noul":0.08},')
            + String('"kind":{"choice":"talk","probabilities":{"talk":') + String(talk_p)
            + String(',"act_here":') + String(rest) + String(',"go_place":') + String(rest) + String('}}}}'))


def command_answer(choice: String) -> String:
    """The commands request: `choice` at 0.9, addressed."""
    return (String('{"model":"script","usage":{"input_tokens":1228},"answers":{')
            + String('"command":{"choice":"') + choice + String('","probabilities":{"') + choice
            + String('":0.9,"none":0.02}},')
            + String('"needs_world":{"noul":0.05},"addressed":{"noul":0.95},"extent":{"score":1.0}}}'))


struct Run(Movable):
    var said: String
    """The last thing said."""
    var order: String
    """Events in the order they came out: `T` a reply spoken, `C<name>` a command."""
    var asked: String
    var replies: Int
    var notes: List[String]
    var same_state: Bool
    var idle: Bool

    def __init__(out self):
        self.said = String("")
        self.order = String("")
        self.asked = String("")
        self.replies = 0
        self.notes = List[String]()
        self.same_state = False
        self.idle = False


def make_loop(talk_first: Float64, ref bank: G1CommandBank, dest_chain: Bool = False) raises -> G1VoiceLoopOf[TextStt, NoTts, ScriptDecision, ScriptChat]:
    var cfg = G1VoiceConfig()
    cfg.dest_chain = dest_chain
    cfg.mute = True
    cfg.request_wording = True
    cfg.chain = False
    cfg.triage = G1_TRIAGE_NARROW
    cfg.name_desc = True
    cfg.talk_first = talk_first
    var places: List[String] = ["fridge", "door", "table", "sofa"]
    for p in places:
        cfg.destinations.append(p)
        cfg.dest_descs.append(String("the ") + p)
        cfg.dest_words.append(String(""))
    var loop = G1VoiceLoopOf[TextStt, NoTts, ScriptDecision, ScriptChat](cfg^, bank)
    loop.mic_on = False
    return loop^


def run(talk_first: Float64, talk_p: Float64, second: String, ref bank: G1CommandBank, computed: String = String(""), pending: String = String(""), ready_at: Int = 0) raises -> Run:
    """`computed`: what the caller's code would say (`set_reply_override`, set before every poll as the demo does);
    `pending` before poll `ready_at` (the voice check still running)."""
    var loop = make_loop(talk_first, bank)
    loop.stt.text = String("Bonjour, comment vas-tu ?")
    loop.jev.script.append(first_answer(talk_p))
    loop.jev.script.append(command_answer(second))
    loop.state = VL_ST_STT
    var ctx = G1Context()
    var r = Run()
    for i in range(40):
        if computed != "":
            loop.set_reply_override(pending if i < ready_at else computed)
        var ev = loop.poll(ctx, bank)
        if ev.kind == VL_TALK:
            r.order += String("T")
            r.said = ev.text
        elif ev.kind == VL_COMMAND:
            r.order += String("C") + ev.name
        var n = loop.take_note()
        if n != "":
            print("     ", n)
            if not n.startswith("[heard]"):
                r.notes.append(n)
    r.asked = loop.jev.asked
    r.replies = loop.llm.asked
    r.same_state = len(loop.jev.states) == 2 and loop.jev.states[0] == loop.jev.states[1]
    r.idle = loop.state == VL_ST_IDLE
    return r^


struct CRun(Movable):
    var order: String
    """`T` the computed answer, `t` one of d1's answers consumed, `C<name>` a command."""
    var said: List[String]
    var notes: List[String]
    var asked: String
    var replies: Int
    var cmd_said: String
    """A command event's `said`."""
    var cancelled: Int
    var idle: Bool
    var left: Int
    """d1's answers never read."""

    def __init__(out self):
        self.order = String("")
        self.said = List[String]()
        self.notes = List[String]()
        self.asked = String("")
        self.replies = 0
        self.cmd_said = String("")
        self.cancelled = 0
        self.idle = False
        self.left = 0


def run_computed(
    second: String, ref bank: G1CommandBank, final_at: Int, stale: Bool = False, take_from: Int = 0,
    decides: Bool = True, command: String = String(""), answer: String = String("Oui, j'ai fini ce que tu m'as demandé."),
    heard: String = String("As-tu fini ce que je t'ai demandé ?"),
) raises -> CRun:
    """« As-tu fini ? » with `computed_first`, driven as the demo drives the loop: before each poll the caller
    sets its computed answer from the words it has TAKEN (`take_heard` after each poll). `final_at`: the first
    poll from which the answer is final (« Un instant. » and final=False before). `stale`: the previous turn's
    answer is set until the caller has taken the new words, and nothing after (they are not a computed question).
    `take_from`: the caller takes the words from that poll on (a caller a poll late)."""
    var loop = make_loop(0.7, bank)
    loop.cfg.computed_first = True
    loop.stt.text = heard
    loop.jev.script.append(first_answer(0.5))
    loop.jev.script.append(command_answer(second))
    loop.state = VL_ST_STT
    var ctx = G1Context()
    var r = CRun()
    var taken = False
    var answers_left = 2
    for i in range(40):
        if stale:
            loop.set_reply_override(String("") if taken else String("Oui, j'ai fini ce que tu m'as demandé."))
        elif taken:
            if i < final_at:
                loop.set_reply_override(String("Un instant."), False)
            else:
                loop.set_reply_override(answer, True, decides, command)
        var ev = loop.poll(ctx, bank)
        if len(loop.jev.script) < answers_left:
            answers_left = len(loop.jev.script)
            r.order += String("t")
        if ev.kind == VL_TALK:
            r.order += String("T")
            r.said.append(ev.text)
        elif ev.kind == VL_COMMAND:
            r.order += String("C") + ev.name
            r.cmd_said = ev.said
        if i >= take_from and loop.take_heard() != "":
            taken = True
        var n = loop.take_note()
        if n != "":
            print("     ", n)
            if not n.startswith("[heard]"):
                r.notes.append(n)
    r.asked = loop.jev.asked
    r.replies = loop.llm.asked
    r.cancelled = loop.jev.cancelled
    r.idle = loop.state == VL_ST_IDLE
    r.left = len(loop.jev.script)
    return r^


def refused_answer() -> String:
    """The commands request: addressed, nothing clear (`squat` 0.30, under `min_top` 0.35)."""
    return (String('{"model":"script","usage":{"input_tokens":1228},"answers":{')
            + String('"command":{"choice":"squat","probabilities":{"squat":0.30,"crouch":0.28,"none":0.10}},')
            + String('"needs_world":{"noul":0.05},"addressed":{"noul":0.80},"extent":{"score":1.0}}}'))


def main() raises:
    var fails = 0
    var bank = G1CommandBank.load(String("g1_command_bank.txt"))

    print("1. a greeting, talk 0.95, talk_first 0.7")
    var a = run(0.7, 0.95, String("talk"), bank)
    check(fails, a.asked == "DC", "both requests asked, the destination first: " + a.asked)
    check(fails, a.order == "T" and a.replies == 1, "one reply, and it came out before the commands answer: " + a.order
          + " (" + String(a.replies) + " asked)")
    var reply_first = len(a.notes) >= 2 and a.notes[0].startswith("[talk-first]") and a.notes[1].startswith("[said]")
    check(fails, reply_first, "the reply was started from the first request's answer, then said")
    var last = a.notes[len(a.notes) - 1] if len(a.notes) > 0 else String("")
    check(fails, last.find("replied already") >= 0, "the commands answer `talk` adds nothing: " + last)
    check(fails, a.same_state, "the commands request was asked with the first request's state")
    check(fails, a.idle, "and the loop is idle again")

    print("2. a stop the triage read as talk 0.75")
    var b = run(0.7, 0.75, String("stand"), bank)
    check(fails, b.asked == "DC", "both requests asked: " + b.asked)
    check(fails, b.order == "TCstand", "the reply, then the stop: " + b.order)

    print("3. talk 0.55, under the bar")
    var c = run(0.7, 0.55, String("talk"), bank)
    check(fails, c.asked == "DC" and c.order == "T" and c.replies == 1, "commands first, then the reply: " + c.asked + " " + c.order)
    check(fails, len(c.notes) > 1 and c.notes[0].startswith("[talk-first] talk 0.55 < 0.70") and c.notes[1].startswith("[talk] "),
          "as today, and the log says why: " + (c.notes[0] if len(c.notes) > 0 else String("")))
    var c2 = run(0.7, 0.55, String("stand"), bank)
    check(fails, c2.order == "Cstand" and c2.replies == 0, "and a stop is only a stop: " + c2.order)

    print("4. talk_first off, talk 0.95")
    var d = run(0.0, 0.95, String("talk"), bank)
    check(fails, d.asked == "DC" and d.order == "T" and len(d.notes) > 0 and d.notes[0].startswith("[talk] "),
          "commands first, then the reply: " + d.asked + " " + d.order)

    print("5. code holds the answer (« Comment je m'appelle ? »), talk 0.95")
    # the voice check ends at poll 4: after the first request's answer (poll 2), as the commands' answer comes (poll 4)
    var e = run(0.7, 0.95, String("talk"), bank, String("Tu t'appelles Denis."), String("Un instant."), 4)
    check(fails, e.asked == "DC" and e.order == "T" and e.replies == 0, "commands first, then the computed answer, no model reply: "
          + e.asked + " " + e.order)
    check(fails, len(e.notes) > 2 and e.notes[0].find("a computed answer") >= 0 and e.notes[2].find("(computed)") >= 0,
          "the log says why: " + (e.notes[0] if len(e.notes) > 0 else String("")))
    check(fails, e.said == "Tu t'appelles Denis.", "and the answer said is the one the finished check gives: " + e.said)

    print("6. one reply a decision")
    # the commands request ended in another route that replies (the miss route's "a question -> reply")
    var f = make_loop(0.7, bank)
    f.heard = String("Raconte-moi une blague.")
    f.replied = True
    f._reply(String(""), String("[spec] a question -> reply"))
    check(fails, f.llm.asked == 0 and f.state == VL_ST_IDLE, "a second reply is not started, and the loop is idle")
    check(fails, f.take_note().find("replied already") >= 0, "and the log says so")
    f.replied = False
    f._reply(String(""), String("[spec] a question -> reply"))
    check(fails, f.llm.asked == 1, "the same call replies when nothing was said yet")

    print("7. a model that failed to load")
    var g = make_loop(0.7, bank)
    check(fails, g.failed() == "", "nothing failed: " + g.failed())
    g.jev.broken = True
    check(fails, g.failed() == " decide", "d1 failed: named as `decide`: '" + g.failed() + "'")
    g.mic_on = True
    g.wait_what = String(" decide")     # it was loading at the previous poll, as d1 is in the demo
    g.cfg.mic_input = String("-f lavfi -i anullsrc=r=16000:cl=mono -t 1")
    var ctx7 = G1Context()
    _ = g.poll(ctx7, bank)
    var line = g.take_note()
    g.stop()
    check(fails, line.find("FAILED TO LOAD: decide") >= 0 and line.find("every model ready") < 0,
          "the listening line names it: " + line)

    print("8. computed_first: « As-tu fini ? », d1 answers `talk`")
    var h = run_computed(String("talk"), bank, 0)
    check(fails, h.order == "Ttt" and len(h.said) == 1,
          "the answer first, then d1's two answers, and nothing more said: " + h.order)
    check(fails, h.asked == "DC" and h.replies == 0, "both requests asked, the reply model never: " + h.asked
          + " (" + String(h.replies) + ")")
    check(fails, len(h.notes) > 0 and h.notes[0].find("(computed, before the decision)") >= 0,
          "the log says it was said before the decision: " + (h.notes[0] if len(h.notes) > 0 else String("")))
    var hl = h.notes[len(h.notes) - 1] if len(h.notes) > 0 else String("")
    check(fails, hl.find("replied already") >= 0, "d1's `talk` adds nothing: " + hl)

    print("9. computed_first: d1 picks `stand` (the --see replays)")
    var k = run_computed(String("stand"), bank, 0)
    check(fails, k.order == "TttCstand", "the answer, then the stop still acts: " + k.order)

    print("10. computed_first: the answer is final only at poll 4")
    var m = run_computed(String("talk"), bank, 4)
    check(fails, len(m.said) == 1 and m.said[0] == "Oui, j'ai fini ce que tu m'as demandé.",
          "said once, the final answer: " + String(len(m.said)) + " " + (m.said[0] if len(m.said) > 0 else String("")))
    check(fails, m.order.startswith("t"), "not before it was final (d1's first answer came first): " + m.order)

    print("11. computed_first: the previous turn's answer, still set")
    # a caller that takes the words a poll late: at poll 1 the loop decides and the answer set is the last turn's
    var st = run_computed(String("talk"), bank, 0, True, 2)
    check(fails, len(st.said) == 1 and st.said[0] == "Bonjour !" and st.replies == 1,
          "never said: the model replied to the new words: " + (st.said[0] if len(st.said) > 0 else String("nothing")))

    print("12. the two-place request without the triage `kind`")
    var tp = make_loop(0.7, bank, True)
    var w = JsonWriter()
    tp.chain_q.write_to_json(w)
    var qs = w.done()
    check(fails, qs.find("\"kind\"") < 0 and qs.find("destination_then") >= 0,
          "its questions: no `kind`, `destination_then` there")
    var w2 = JsonWriter()
    tp.dest_q.write_to_json(w2)
    check(fails, w2.done().find("\"kind\"") >= 0, "the destination request keeps it")

    print("13. a silent piece and the echo gate")
    var sg = make_loop(0.7, bank)
    sg.cfg.tts_no_play = True
    sg.play_until = perf_counter_ns() + 2_000_000_000
    check(fails, not sg.speaking(), "--tts-no-play: a silent piece does not mute the microphone")
    sg.cfg.tts_no_play_gated = True
    check(fails, sg.speaking(), "--tts-no-play-gated: it does, as a piece played live")
    sg.play_until = 0
    check(fails, not sg.speaking(), "and not once it has ended")
    sg.tts_queue.append(String("Une phrase."))
    check(fails, sg.speaking(), "a piece still to synthesise counts, as live")

    print("14. a computed answer that settles the turn: « Tu peux ouvrir la porte ? »")
    var nd = run_computed(String("talk"), bank, 0, decides=False, answer=String("Non, je ne peux pas ouvrir la porte."),
                          heard=String("Tu peux ouvrir la porte ?"))
    check(fails, nd.order == "T" and len(nd.said) == 1 and nd.said[0] == "Non, je ne peux pas ouvrir la porte.",
          "the answer, and nothing else: " + nd.order)
    check(fails, nd.asked == "D" and nd.cancelled == 1 and nd.left == 2 and nd.replies == 0 and nd.idle,
          "the first request dropped unread, the second never asked, the model never, the loop idle: "
          + nd.asked + " cancelled " + String(nd.cancelled) + " left " + String(nd.left))
    check(fails, len(nd.notes) > 0 and nd.notes[0].find("the turn is settled") >= 0, "the log says so")

    print("15. the same with its command: « Tu peux danser ? »")
    var dn = run_computed(String("talk"), bank, 0, decides=False, command=String("dance"), answer=String("Oui, regarde !"),
                          heard=String("Tu peux danser ?"))
    check(fails, dn.order == "Cdance" and dn.cmd_said == "Oui, regarde !" and len(dn.said) == 0,
          "one event, the command, with the answer it was said with: " + dn.order + " / " + dn.cmd_said)
    check(fails, dn.asked == "D" and dn.cancelled == 1 and dn.idle, "d1 dropped the same way: " + dn.asked)
    check(fails, len(dn.notes) > 0 and dn.notes[0].find("runs dance") >= 0, "the log names the command")
    var kd = run_computed(String("stand"), bank, 0)
    check(fails, kd.order == "TttCstand" and kd.cancelled == 0, "`decides` left True: case 9's order, nothing dropped")

    print("16. an addressed sentence d1 refuses: a reply (`reply_refused`)")
    for on in range(2):
        var rf = make_loop(0.7, bank)
        rf.cfg.reply_refused = on == 1
        rf.stt.text = String("Tu peux faire quoi ?")
        rf.jev.script.append(first_answer(0.5))
        rf.jev.script.append(refused_answer())
        rf.state = VL_ST_STT
        var rctx = G1Context()
        var talked = 0
        var refused_note = False
        for _ in range(40):
            var rev = rf.poll(rctx, bank)
            if rev.kind == VL_TALK:
                talked += 1
            var rn = rf.take_note()
            if rn.find("[refused] no clear pick") >= 0:
                refused_note = True
        if on == 1:
            check(fails, refused_note and talked == 1 and rf.llm.asked == 1, "on: refused, then the model's reply, once")
        else:
            check(fails, refused_note and talked == 0 and rf.llm.asked == 0, "off: refused and silent, as before")

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
