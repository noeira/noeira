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

The loop is the real `G1VoiceLoopOf`, its microphone closed, a stand-in
recogniser handing it the sentence, a stand-in decision answering from a
script, and a stand-in reply model.
"""

from std.sys import exit

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

    def __init__(out self):
        self.broken = False
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
        pass


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


def make_loop(talk_first: Float64, ref bank: G1CommandBank) raises -> G1VoiceLoopOf[TextStt, NoTts, ScriptDecision, ScriptChat]:
    var cfg = G1VoiceConfig()
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

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
