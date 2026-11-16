"""The early start: a turn's transcription and first request begin at a pause, before the turn has ended (L3).

    pixi run mojo run -I . tests/robots/test_g1_voice_loop_early.mojo      (CPU, no microphone, about 1 s)

WHY THIS EXISTS. A turn ends after 0.6 s of quiet (`hang_s`): long enough to
tell a pause inside a sentence from its end. The transcription and d1's first
request waited for it. With `early_s` (0.3 s in the room demo) they start at
the pause, while the loop still listens (docs/NN_MODEL_PORTS_PLAN.md §12.8.7,
L3). What this pins:

1. At a pause: the segment in hand is transcribed, then the first request
   asked (the destination one, `request_wording`), and NOTHING is told to the
   caller: no `[heard]`, no `take_heard`, still recording.
2. The turn ends on that pause: the loop goes on from the request in flight
   (state DEST, its clock from when it was asked), and the TURN'S OWN SEGMENT
   is transcribed beside it; the same words: the caller is told what was heard
   (only then), and the answer may be used.
2c. Through `poll`: the poll that checks the words does not read the
   decision's answer (the caller sets its computed answer from the words in
   between: round 36 replied to « Où est le frigo ? » with the previous turn's).
2b. Different words (round 34: « S'il fait beau » at the pause, « Il fait
   beau » for the turn): the early request is dropped and the turn's own
   words asked again.
3. The person goes on: the early start is dropped (the request cancelled) and
   noted; the next pause starts another, and the turn ending on it keeps it.
4. The turn ends while it is still transcribing: dropped, the turn's end
   transcribes its own segment as before.
5. The turn ends on ANOTHER pause than the one started on: dropped, and the
   turn's end does it as before (no commit).
6. Too few letters (« . »): nothing asked; the turn's end does it as before.
7. Off (`early_s` 0): nothing starts at a pause.
8. One request a decision (no `request_wording`): the commands request, state JEV.

The loop is the real `G1VoiceLoopOf`, its microphone closed; the detector's
pause is set by hand (`G1Vad`'s own gate covers when it is set); the recogniser
and the decision are stand-ins that count what they are asked.
"""

from std.sys import exit

from noeira.ai.chat import ChatMessage, ChatResponse, ChatService, ToolSpec
from noeira.ai.jev import DecisionService, JevAnswers, JevQuestions
from noeira.ai.speech import SpeechToTextService, TextToSpeechService, Transcript
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import G1Context
from noeira.envs.robots.g1_voice_loop import (
    G1VoiceConfig, G1VoiceEvent, G1VoiceLoopOf, VL_ST_DEST, VL_ST_IDLE, VL_ST_JEV, VL_ST_REC, VL_ST_STT,
)
from noeira.io.json import JsonWriter
from noeira.io.wav import WavAudio


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


struct CountStt(Movable, SpeechToTextService):
    """Answers `text` after `polls` polls; counts starts and cancels."""

    var text: String
    var polls: Int
    var wait: Int
    var starts: Int
    var cancels: Int
    var samples: Int

    def __init__(out self):
        self.text = String("Va à la table.")
        self.polls = 2
        self.wait = 0
        self.starts = 0
        self.cancels = 0
        self.samples = 0

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
        self.starts += 1
        self.wait = self.polls
        self.samples = len(audio.samples)

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        self.wait -= 1
        return self.wait <= 0

    def result(mut self) raises -> Transcript:
        return Transcript(self.text, 60.0)

    def cancel(mut self) raises:
        self.cancels += 1


struct CountDecision(Movable, DecisionService):
    """Never answers; records `D` (destination request) or `C` (commands) per start, and cancels."""

    var asked: String
    var cancels: Int
    var polls: Int

    def __init__(out self):
        self.asked = String("")
        self.cancels = 0
        self.polls = 0

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        var w = JsonWriter()
        questions.write_to_json(w)
        var qs = w.done()
        self.asked += String("C") if qs.find("\"command\"") >= 0 else String("D")

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        self.asked += String("S")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        self.polls += 1
        return False

    def result(mut self) raises -> JevAnswers:
        raise Error("test: no answer")

    def cancel(mut self) raises:
        self.cancels += 1


struct NoChat(Movable, ChatService):
    def __init__(out self):
        pass

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
        pass

    def poll(mut self, timeout_ms: Int = 0) raises -> String:
        return String("")

    def done(self) -> Bool:
        return True

    def result(mut self) raises -> ChatResponse:
        return ChatResponse()

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


comptime Loop = G1VoiceLoopOf[CountStt, NoTts, CountDecision, NoChat]


def make_loop(early: Float64, two: Bool, ref bank: G1CommandBank) raises -> Loop:
    var cfg = G1VoiceConfig()
    cfg.mute = True
    cfg.early_s = early
    cfg.request_wording = two
    cfg.chain = not two
    if two:
        var places: List[String] = ["fridge", "door", "table", "sofa"]
        for p in places:
            cfg.destinations.append(p)
            cfg.dest_descs.append(String("the ") + p)
            cfg.dest_words.append(String(""))
    var loop = Loop(cfg^, bank)
    loop.mic_on = False
    loop.state = VL_ST_REC
    for _ in range(16000):
        loop.vad.seg.append(Int16(100))
    return loop^


def pause(mut loop: Loop, seq: Int):
    loop.vad.paused = True
    loop.vad.pause_seq = seq


def steps(mut loop: Loop, ref ctx: G1Context, n: Int) -> String:
    """`n` recording polls of the early start; the notes they left."""
    var notes = String("")
    for _ in range(n):
        loop._early_step(ctx)
        var t = loop.take_note()
        if t != "":
            print("     ", t)
            notes += t + String("\n")
    return notes^


def main() raises:
    var fails = 0
    var bank = G1CommandBank.load(String("g1_command_bank.txt"))
    var ctx = G1Context()

    print("1. at a pause: transcribed, then the first request asked; nothing told")
    var a = make_loop(0.3, True, bank)
    _ = steps(a, ctx, 2)
    check(fails, a.stt.starts == 0, "no pause yet: nothing started")
    pause(a, 1)
    var n1 = steps(a, ctx, 4)
    check(fails, a.stt.starts == 1 and a.stt.samples == 16000, "the segment in hand transcribed (" + String(a.stt.samples) + " samples)")
    check(fails, a.jev.asked == "D", "then the destination request asked: " + a.jev.asked)
    check(fails, a.state == VL_ST_REC and a.take_heard() == "" and n1.find("[heard]") < 0, "still recording, nothing heard by the caller")

    print("2. the turn ends on that pause")
    for _ in range(4000):
        a.vad.seg.append(Int16(0))              # the quiet that closed it
    var ev = G1VoiceEvent.none()
    var kept = a._early_commit(ev)
    check(fails, kept and a.state == VL_ST_DEST and a.early_verify, "kept: the loop goes on from the request in flight (DEST)")
    check(fails, a.t_wait == a.early_t_ask and a.jev.asked == "D", "its clock from when it was asked, and not asked again")
    check(fails, a.stt.starts == 2 and a.stt.samples == 20000, "the turn's own segment transcribed beside it (" + String(a.stt.samples) + " samples)")
    check(fails, a.take_heard() == "", "nothing told yet")
    var same = False
    for _ in range(4):
        if a._early_verified(ctx, ev):
            same = True
            break
    var n2 = a.take_note()
    print("     ", n2)
    check(fails, same and a.take_heard() == "Va à la table." and ev.text == "Va à la table." and n2.find("the same words") >= 0,
          "the same words: told now, the answer may be used")

    print("2c. the words checked in a poll of their own")
    var a3 = make_loop(0.3, True, bank)
    pause(a3, 1)
    _ = steps(a3, ctx, 4)
    var ev3c = G1VoiceEvent.none()
    _ = a3._early_commit(ev3c)
    a3.stt.wait = 0                             # the turn's transcript is in
    var p0 = a3.jev.polls
    var evp = a3.poll(ctx, bank)
    check(fails, not a3.early_verify and evp.text == "Va à la table." and a3.jev.polls == p0,
          "the poll that checks the words tells them, and does not read the answer (" + String(a3.jev.polls - p0) + " polls)")
    _ = a3.poll(ctx, bank)
    check(fails, a3.jev.polls == p0 + 1, "the next poll reads it")

    print("2b. the turn reads other words")
    var a2 = make_loop(0.3, True, bank)
    a2.stt.text = String("S'il fait beau.")
    pause(a2, 1)
    _ = steps(a2, ctx, 4)
    var ev2 = G1VoiceEvent.none()
    _ = a2._early_commit(ev2)
    a2.stt.text = String("Il fait beau.")
    var used = False
    for _ in range(4):
        if a2.early_verify and a2._early_verified(ctx, ev2):     # as `poll` calls it: while the check is due
            used = True
    var n2b = a2.take_note()
    print("     ", n2b)
    check(fails, not used and a2.jev.cancels == 1 and a2.jev.asked == "DD" and a2.state == VL_ST_DEST,
          "the early request dropped, the turn's words asked again: " + a2.jev.asked)
    check(fails, a2.take_heard() == "Il fait beau." and n2b.find("asked again") >= 0, "and the caller is told the turn's words")

    print("3. the person goes on: dropped; the next pause keeps")
    var b = make_loop(0.3, True, bank)
    pause(b, 1)
    _ = steps(b, ctx, 4)
    b.vad.paused = False                      # speech again
    var n3 = steps(b, ctx, 1)
    check(fails, b.jev.cancels == 1 and b.early == 0 and n3.find("[early] dropped") >= 0, "the request cancelled, and noted")
    pause(b, 2)
    _ = steps(b, ctx, 4)
    var ev3 = G1VoiceEvent.none()
    check(fails, b.stt.starts == 2 and b.jev.asked == "DD" and b._early_commit(ev3) and b.state == VL_ST_DEST,
          "the next pause starts again, and the turn ending on it keeps it: " + b.jev.asked)

    print("4. the turn ends while transcribing")
    var c = make_loop(0.3, True, bank)
    c.stt.polls = 10
    pause(c, 1)
    _ = steps(c, ctx, 2)
    var ev4 = G1VoiceEvent.none()
    check(fails, not c._early_commit(ev4) and c.stt.cancels == 1 and c.state == VL_ST_REC,
          "dropped: the turn's end transcribes its own segment as before")

    print("5. the turn ends on another pause")
    var d = make_loop(0.3, True, bank)
    pause(d, 1)
    _ = steps(d, ctx, 4)
    d.vad.pause_seq = 2                       # the speech resumed and paused again inside one read
    var ev5 = G1VoiceEvent.none()
    check(fails, not d._early_commit(ev5) and d.jev.cancels == 1 and d.state == VL_ST_REC,
          "not kept: cancelled, the turn's end does it as before")

    print("6. too few letters")
    var e = make_loop(0.3, True, bank)
    e.stt.text = String(" . ")
    pause(e, 1)
    _ = steps(e, ctx, 4)
    var ev6 = G1VoiceEvent.none()
    check(fails, e.jev.asked == "" and not e._early_commit(ev6) and e.early == 0, "nothing asked; the turn's end does it as before")

    print("7. off")
    var f = make_loop(0.0, True, bank)
    pause(f, 1)
    _ = steps(f, ctx, 4)
    var ev7 = G1VoiceEvent.none()
    check(fails, f.stt.starts == 0 and not f._early_commit(ev7), "nothing starts at a pause")

    print("8. one request a decision")
    var g = make_loop(0.3, False, bank)
    pause(g, 1)
    _ = steps(g, ctx, 4)
    var ev8 = G1VoiceEvent.none()
    check(fails, g.jev.asked == "C" and g._early_commit(ev8) and g.state == VL_ST_JEV, "the commands request, state JEV: " + g.jev.asked)

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
