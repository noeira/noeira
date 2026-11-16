"""The reply said sentence by sentence while it is written (docs/NN_MODEL_PORTS_PLAN.md §12.8.7, L2).

    pixi run mojo run -I . tests/robots/test_g1_voice_loop_stream.mojo      (CPU, no microphone, about 1 s)

WHY THIS EXISTS. The robot's spoken reply used to be synthesised once the reply
model had written all of it: « Je m'appelle Noé. Je suis un petit robot… »
waited ~0.9 s of writing before its first sound. Now each finished sentence
goes to the synthesiser while the next is written, and plays after the one
before it. What this pins:

1. `spoken_sentence_end`: where a sentence is finished (a mark FOLLOWED by a
   blank, or a line break), and where it is not yet (`2.5`, a mark that is the
   last thing written).
2. A reply streamed in pieces: its three sentences synthesised IN ORDER, ONE
   AT A TIME, the first one before the reply is finished; the note times the
   first audio; the event carries the whole answer.
3. `stream_reply` off: the whole reply, synthesised once it is written (as before).
4. The answer is the authority: an answer that does not start with what was
   streamed says nothing more, and the note says so.
5. A `say` from the caller cuts a streamed reply off: its waiting pieces are
   not said.
6. Muted: nothing synthesised.

The loop is the real `G1VoiceLoopOf` with `tts_no_play` (synthesised, played in
silence), its microphone closed; the reply model and the synthesiser are
stand-ins that record what they were given.
"""

from std.sys import exit

from noeira.ai.chat import ChatMessage, ChatResponse, ChatService, ToolSpec
from noeira.ai.jev import DecisionService, JevAnswers, JevQuestions
from noeira.ai.speech import SpeechToTextService, TextToSpeechService, Transcript
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import G1Context
from noeira.envs.robots.g1_voice_loop import G1VoiceConfig, G1VoiceLoopOf, VL_ST_IDLE, VL_TALK
from noeira.io.wav import WavAudio
from noeira.text.spoken import spoken_sentence_end


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


# ── the stand-ins ─────────────────────────────────────────────────────────

struct NoStt(Movable, SpeechToTextService):
    def __init__(out self):
        pass

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
        return Transcript(String(""), 0.0)

    def cancel(mut self) raises:
        pass


struct NoDecision(Movable, DecisionService):
    def __init__(out self):
        pass

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        raise Error("test: no decision here")

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        raise Error("test: no decision here")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return True

    def result(mut self) raises -> JevAnswers:
        raise Error("test: no decision here")

    def cancel(mut self) raises:
        pass


struct StreamChat(Movable, ChatService):
    """Streams `pieces`, one a poll, then is done; `answer` is what `result` says."""

    var pieces: List[String]
    var answer: String
    var at: Int
    var started: Bool

    def __init__(out self):
        self.pieces = List[String]()
        self.answer = String("")
        self.at = 0
        self.started = False

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
        self.at = 0
        self.started = True

    def poll(mut self, timeout_ms: Int = 0) raises -> String:
        if self.at < len(self.pieces):
            self.at += 1
            return self.pieces[self.at - 1]
        return String("")

    def done(self) -> Bool:
        """As the trait says: over, or none started."""
        return self.at >= len(self.pieces)

    def result(mut self) raises -> ChatResponse:
        self.started = False
        var r = ChatResponse()
        r.text = self.answer
        r.latency_ms = 400.0
        return r^

    def cancel(mut self) raises:
        self.started = False


struct RecTts(Movable, TextToSpeechService):
    """Records each text it is asked for; a synthesis takes `polls` polls and gives
    a WAV of no length. `overlap` counts a `start` while one is in flight."""

    var started: List[String]
    var busy: Int
    var polls: Int
    var overlap: Int
    var cancels: Int

    def __init__(out self):
        self.started = List[String]()
        self.busy = 0
        self.polls = 2
        self.overlap = 0
        self.cancels = 0

    @staticmethod
    def from_spec(spec: String, voice: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, text: String) raises:
        if self.busy > 0:
            self.overlap += 1
        self.started.append(text)
        self.busy = self.polls

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        self.busy -= 1
        return self.busy <= 0

    def result_wav_bytes(mut self) raises -> List[UInt8]:
        # a 44-byte PCM header (16 kHz, mono, 16-bit) and no samples: it "plays" for 0 s
        var h: List[UInt8] = [82, 73, 70, 70, 36, 0, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
                              128, 62, 0, 0, 0, 125, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 0, 0, 0]
        return h^

    def cancel(mut self) raises:
        self.busy = 0
        self.cancels += 1


comptime Loop = G1VoiceLoopOf[NoStt, RecTts, NoDecision, StreamChat]


def make_loop(stream: Bool, mute: Bool, ref bank: G1CommandBank) raises -> Loop:
    var cfg = G1VoiceConfig()
    cfg.mute = mute
    cfg.tts_spec = String("test")
    cfg.tts_no_play = True
    cfg.stream_reply = stream
    var loop = Loop(cfg^, bank)
    loop.mic_on = False
    return loop^


struct Run(Movable):
    var said: List[String]
    var first_before_done: Bool
    var notes: String
    var event: String
    var overlap: Int
    var idle: Bool

    def __init__(out self):
        self.said = List[String]()
        self.first_before_done = False
        self.notes = String("")
        self.event = String("")
        self.overlap = 0
        self.idle = False


def run(
    stream: Bool, mute: Bool, var pieces: List[String], answer: String, ref bank: G1CommandBank, say_at: Int = -1,
    tts_polls: Int = 2,
) raises -> Run:
    """A reply streamed as `pieces`; `say_at` >= 0: the caller says something at that poll; a synthesis takes
    `tts_polls` polls."""
    var loop = make_loop(stream, mute, bank)
    loop.tts.value().polls = tts_polls
    loop.llm.pieces = pieces^
    loop.llm.answer = answer
    loop.heard = String("Comment tu t'appelles ?")
    loop._reply(String(""), String("[talk] test"))
    var ctx = G1Context()
    var r = Run()
    for i in range(40):
        if i == say_at:
            loop.say(String("Je vais à la table."))
        var ev = loop.poll(ctx, bank)
        if len(loop.tts.value().started) > 0 and not loop.llm.done() and len(r.said) == 0:
            r.first_before_done = True
        if ev.kind == VL_TALK:
            r.event = ev.text
        var n = loop.take_note()
        if n != "":
            print("     ", n)
            r.notes += n + String("\n")
    r.said = loop.tts.value().started.copy()
    r.overlap = loop.tts.value().overlap
    r.idle = (loop.state == VL_ST_IDLE and not loop.tts_busy and len(loop.tts_queue) == 0
              and len(loop.tts_ready) == 0)
    return r^


def _join(ref l: List[String]) -> String:
    var s = String("")
    for x in l:
        s += String("[") + x + String("]")
    return s


def main() raises:
    var fails = 0
    var bank = G1CommandBank.load(String("g1_command_bank.txt"))

    print("1. where a sentence is finished")
    var t = String("Je m'appelle Noé. Je suis")
    check(fails, spoken_sentence_end(t, 0) == 18, "« Noé. » followed by a blank: " + String(spoken_sentence_end(t, 0)))
    check(fails, spoken_sentence_end(String("Je m'appelle Noé."), 0) == -1, "a mark that is the last thing written: not yet")
    check(fails, spoken_sentence_end(String("Le frigo est à 2.5 m"), 0) == -1, "`2.5` is not a sentence's end")
    check(fails, spoken_sentence_end(String("Le frigo est à 2.5 m. Bon"), 0) == 22, "but `m.` then a blank is")
    check(fails, spoken_sentence_end(String("Oui ! Bien sûr"), 0) == 5, "« Oui ! »: the blank after the mark")
    check(fails, spoken_sentence_end(String("Quoi ?! Non"), 0) == 7, "`?!` together")
    check(fails, spoken_sentence_end(String("Hmm... bon"), 0) == 6, "`...` together")
    var ell = String("Hmm") + String("…") + String(" bon")
    check(fails, spoken_sentence_end(ell, 0) == 6, "`…` (U+2026)")
    check(fails, spoken_sentence_end(String("Ligne un\nDeux"), 0) == 9, "a line break")
    check(fails, spoken_sentence_end(String(" . Et"), 0) == -1, "a mark with no word before it is not a sentence")
    check(fails, spoken_sentence_end(t, 18) == -1, "from `start`: the next one is not finished yet")

    print("2. a reply streamed in pieces")
    var p2: List[String] = ["Je m'app", "elle Noé.", " Je suis", " un petit robot", ". Et toi", " ?"]
    var a = run(True, False, p2^, String("Je m'appelle Noé. Je suis un petit robot. Et toi ?"), bank)
    check(fails, _join(a.said) == "[Je m'appelle Noé.][Je suis un petit robot.][Et toi ?]",
          "three sentences, in order: " + _join(a.said))
    check(fails, a.first_before_done, "the first was being synthesised before the reply was finished")
    check(fails, a.overlap == 0, "one synthesis at a time")
    check(fails, a.notes.find("[voice] first audio") >= 0, "the note times the first audio")
    check(fails, a.notes.find("said in 3 pieces") >= 0, "the [said] note counts the pieces")
    check(fails, a.event == "Je m'appelle Noé. Je suis un petit robot. Et toi ?", "the event carries the whole answer: " + a.event)
    check(fails, a.idle, "and everything is said at the end")

    print("2b. short sentences faster than the synthesiser: a backlog, still one at a time")
    var p2b: List[String] = ["Un.", " Deux.", " Trois.", " Quatre.", " Cinq.", " Six."]
    var a2 = run(True, False, p2b^, String("Un. Deux. Trois. Quatre. Cinq. Six."), bank, -1, 6)
    check(fails, _join(a2.said) == "[Un.][Deux.][Trois.][Quatre.][Cinq.][Six.]", "all six, in order: " + _join(a2.said))
    check(fails, a2.overlap == 0, "never two syntheses at once (" + String(a2.overlap) + " overlaps)")

    print("3. stream_reply off")
    var p3: List[String] = ["Je m'app", "elle Noé.", " Je suis", " un petit robot."]
    var b = run(False, False, p3^, String("Je m'appelle Noé. Je suis un petit robot."), bank)
    check(fails, _join(b.said) == "[Je m'appelle Noé. Je suis un petit robot.]", "the whole reply, once: " + _join(b.said))
    check(fails, not b.first_before_done, "nothing before the reply was finished")
    check(fails, b.notes.find("[voice] first audio") >= 0, "the first audio timed too (the A/B)")

    print("4. the answer is the authority")
    var p4: List[String] = ["Je m'appelle Noé.", " Je suis", " un robot."]
    var c = run(True, False, p4^, String("Bonjour, je suis Noé."), bank)
    check(fails, _join(c.said) == "[Je m'appelle Noé.]", "what was streamed and finished was said, nothing more: " + _join(c.said))
    check(fails, c.notes.find("not the answer's start") >= 0, "and the note says so")

    print("5. the caller says something: the waiting pieces go")
    var p5: List[String] = ["Un.", " Deux.", " Trois.", " Quatre.", " Cinq."]
    var d = run(True, False, p5^, String("Un. Deux. Trois. Quatre. Cinq."), bank, 3)
    var cut = len(d.said) > 0 and d.said[len(d.said) - 1] == "Je vais à la table."
    check(fails, cut, "the caller's sentence is the last said: " + _join(d.said))
    var before = 0
    for x in d.said:
        if x != "Je vais à la table.":
            before += 1
    check(fails, before <= 3 and before >= 1, "and only the pieces before it were sent (" + String(before) + ")")

    print("6. muted")
    var p6: List[String] = ["Un.", " Deux."]
    var e = run(True, True, p6^, String("Un. Deux."), bank)
    check(fails, len(e.said) == 0 and e.notes.find("first audio") < 0, "nothing synthesised: " + _join(e.said))
    check(fails, e.event == "Un. Deux.", "the event still carries the answer")

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
