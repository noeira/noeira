"""The voice loop's microphone with stand-in models: when it opens, and what becomes of speech it is too busy for.

    pixi run mojo run -I . tests/robots/test_g1_voice_loop_listen.mojo      (needs `ffmpeg`; real time, about 25 s)

WHY THIS EXISTS. Every headless run of the room demo lost its SECOND utterance,
or heard only its tail (« Je m'appelle Denis. » as « Denis. »). The detector
was not at fault (`test_g1_vad.mojo`); the loop was:

- it listened while its models were still loading. The first utterance was
  transcribed 6 s after it ended (the recogniser alone takes 80 ms), so the
  loop was still busy with it when the second arrived;
- speech that arrives while the loop is busy was dropped without a line, and
  when the loop came back to idle in the middle of it, the detector opened on
  what was left.

The loop here is the real `G1VoiceLoopOf`, on a real capture process replaying
a synthetic recording (`mic_input`), with four stand-in backends. The
recogniser "loads" for 2.5 s and takes 1.8 s over its first call, which ends
in the middle of the second utterance. It answers one letter, so the loop
returns to idle without asking for a decision.

1. While a backend loads, the loop says so, opens no capture and sends nothing.
2. It says when it listens, and the first utterance (4.8 s into the recording
   as the capture delivers it) is then heard whole.
3. The second utterance, half spoken when the loop is idle again: no segment
   made of its tail, and one line with its length.
4. The third and fourth are heard whole.
5. Each utterance listened to hands the reply's prompt to the reply model
   once, while it is being recorded (`ChatService.prepare`), never the one the
   loop was too busy for.

⚠ `-fflags nobuffer`, which the capture passes for a live device, makes ffmpeg
DROP the first 3.2 s of a file input (102 400 bytes, measured on ffmpeg 8.1):
a recording replayed through `mic_input` needs that much lead-in, and this one
has 6 s.
"""

from std.math import sin, pi
from std.sys import exit
from std.time import perf_counter_ns

from noeira.ai.chat import ChatMessage, ChatResponse, ChatService, ToolSpec
from noeira.ai.jev import DecisionService, JevAnswers, JevQuestions
from noeira.ai.speech import SpeechToTextService, TextToSpeechService, Transcript
from noeira.core.concurrent.thread import sleep_us
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import G1Context
from noeira.envs.robots.g1_voice_loop import G1VoiceConfig, G1VoiceLoopOf
from noeira.io.wav import WavAudio, save_wav


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


# ── the stand-ins ─────────────────────────────────────────────────────────

struct SlowStt(Movable, SpeechToTextService):
    """Loads for `load_s`, takes `first_s` over its first call and 0.1 s over
    the others, and answers `n1`, `n2`, ... (one letter: no decision follows)."""

    var t0: Int
    var load_s: Float64
    var first_s: Float64
    var calls: Int
    var early: Int
    """Calls made while it was still loading."""
    var t_call: Int
    var lens: List[Int]
    """Samples in each clip it was given."""

    def __init__(out self, load_s: Float64, first_s: Float64):
        self.t0 = perf_counter_ns()
        self.load_s = load_s
        self.first_s = first_s
        self.calls = 0
        self.early = 0
        self.t_call = 0
        self.lens = List[Int]()

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        """`<load seconds>:<first call seconds>`."""
        var p = spec.split(":")
        if len(p) != 2:
            raise Error("test: the stand-in recogniser takes '<load s>:<first call s>'")
        return Self(Float64(String(p[0])), Float64(String(p[1])))

    def set_language(mut self, lang: String):
        pass

    def set_hint(mut self, vocab: String) -> Bool:
        return False

    def loading(self) -> Bool:
        return Float64(perf_counter_ns() - self.t0) / 1e9 < self.load_s

    def warm_up(mut self):
        pass

    def start(mut self, ref audio: WavAudio) raises:
        if self.loading():
            self.early += 1
        self.calls += 1
        self.t_call = perf_counter_ns()
        self.lens.append(len(audio.samples))

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        var need = self.first_s if self.calls == 1 else 0.1
        return Float64(perf_counter_ns() - self.t_call) / 1e9 >= need

    def result(mut self) raises -> Transcript:
        return Transcript(String("n") + String(self.calls), 1.0)

    def cancel(mut self) raises:
        pass


struct NoDecision(Movable, DecisionService):
    var asked: Int

    def __init__(out self):
        self.asked = 0

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        self.asked += 1
        raise Error("test: no decision is expected")

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        self.asked += 1
        raise Error("test: no decision is expected")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return True

    def result(mut self) raises -> JevAnswers:
        raise Error("test: no decision is expected")

    def cancel(mut self) raises:
        pass


struct NoChat(Movable, ChatService):
    var asked: Int
    var prepared: Int
    """`prepare` calls: the loop hands the reply's prompt over ahead of the words."""
    var prepared_sys: String

    def __init__(out self):
        self.asked = 0
        self.prepared = 0
        self.prepared_sys = String("")

    def prepare(mut self, system: String) raises:
        self.prepared += 1
        self.prepared_sys = system

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
        raise Error("test: no reply is expected")

    def poll(mut self, timeout_ms: Int = 0) raises -> String:
        return String("")

    def done(self) -> Bool:
        return True

    def result(mut self) raises -> ChatResponse:
        raise Error("test: no reply is expected")

    def cancel(mut self) raises:
        pass


struct NoTts(Movable, TextToSpeechService):
    var asked: Int

    def __init__(out self):
        self.asked = 0

    @staticmethod
    def from_spec(spec: String, voice: String) raises -> Self:
        return Self()

    def loading(self) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, text: String) raises:
        raise Error("test: nothing is spoken")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return True

    def result_wav_bytes(mut self) raises -> List[UInt8]:
        raise Error("test: nothing is spoken")

    def cancel(mut self) raises:
        pass


# ── the recording ─────────────────────────────────────────────────────────

struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def unit(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 33) & 0xFFFFFF) / Float64(0x1000000) - 0.5


def room(mut out: List[Int16], seconds: Float64, mut rng: Rng):
    for _ in range(Int(seconds * 16000.0)):
        out.append(Int16(Int(rng.unit() * 110.0)))


def speech(mut out: List[Int16], seconds: Float64, mut rng: Rng):
    var n0 = len(out)
    for i in range(Int(seconds * 16000.0)):
        var v = 8000.0 * sin(2.0 * pi * 220.0 * Float64(n0 + i) / 16000.0) + rng.unit() * 110.0
        out.append(Int16(Int(v)))


def seconds_in(note: String) -> Float64:
    """The number after `[vad] ` in a "speech while busy" line; -1 when absent."""
    var p = note.split(" ")
    if len(p) < 2:
        return -1.0
    try:
        return Float64(String(p[1]))
    except:
        return -1.0


def main() raises:
    var fails = 0
    var LOAD_S = 2.5
    var FIRST_S = 1.8
    var UTT_S = 1.2

    var rng = Rng(20261113)
    var pcm = List[Int16]()
    room(pcm, 6.0, rng)
    speech(pcm, UTT_S, rng)         # A: 6.0 s
    room(pcm, 1.8, rng)
    speech(pcm, UTT_S, rng)         # B: 9.0 s, the first call ends in the middle of it
    room(pcm, 3.8, rng)
    speech(pcm, UTT_S, rng)         # C: 14.0 s
    room(pcm, 2.8, rng)
    speech(pcm, UTT_S, rng)         # D: 18.0 s
    room(pcm, 1.5, rng)
    var total_s = Float64(len(pcm)) / 16000.0
    var path = String("/tmp/noeira_test_voice_loop_listen.wav")
    var wav = WavAudio(16000, 1, pcm^)
    save_wav(path, wav)

    var cfg = G1VoiceConfig()
    cfg.stt_spec = String(LOAD_S) + ":" + String(FIRST_S)
    cfg.mic_input = "-re -i " + path
    var bank = G1CommandBank()
    var loop = G1VoiceLoopOf[SlowStt, NoTts, NoDecision, NoChat](cfg^, bank)
    loop.start()
    var ctx = G1Context()

    var notes = List[String]()
    var at = List[Float64]()
    var opened_early = False
    var taken = List[Float64]()
    var prep_at = List[Float64]()
    var t0 = perf_counter_ns()
    var run_s = LOAD_S + total_s - 3.2 + 2.0
    print("a", round(total_s, 1), "s recording, utterances of", UTT_S, "s at 6, 9, 14 and 18 s; running", round(run_s, 1), "s")
    while Float64(perf_counter_ns() - t0) / 1e9 < run_s:
        var prepared = loop.llm.prepared
        _ = loop.poll(ctx, bank)
        if loop.llm.prepared > prepared:
            prep_at.append(Float64(perf_counter_ns() - t0) / 1e9)
        if loop.stt.loading() and loop.mic:
            opened_early = True
        var ms = loop.take_missed()
        if ms > 0.0:
            taken.append(ms)
        var n = loop.take_note()
        if n != "":
            var el = Float64(perf_counter_ns() - t0) / 1e9
            print("   ", round(el, 2), "s", n)
            notes.append(n)
            at.append(el)
        _ = sleep_us(16_000)
    loop.stop()

    var waited = -1
    var listening = -1
    var missed = List[Int]()
    var sent = 0
    var sent_at = List[Float64]()
    var heard = 0
    var dead = String("")
    for i in range(len(notes)):
        if notes[i].startswith("[voice] not listening yet") and waited < 0:
            waited = i
        elif notes[i].startswith("[voice] listening"):
            listening = i
        elif notes[i].startswith("[vad] ") and notes[i].find("while busy") >= 0:
            missed.append(i)
        elif notes[i].startswith("[stt] "):
            sent += 1
            sent_at.append(at[i])
        elif notes[i].startswith("[heard] n"):
            heard += 1
        elif notes[i].startswith("[mic] ") and notes[i].find("input live") < 0:
            dead = notes[i].copy()
    if dead != "" and sent == 0:
        # (with clips sent, that line is only the recording's end)
        print("  the capture did not run (is `ffmpeg` on PATH?):", dead)

    print("1. while a backend loads")
    check(fails, waited == 0 and notes[0].find("hear") >= 0 and notes[0].find("decide") < 0,
          "the first line says what is still loading: " + (notes[0] if len(notes) > 0 else String("(no line)")))
    check(fails, not opened_early and loop.stt.early == 0, "no capture was opened and nothing was sent to the recogniser meanwhile")

    print("2. listening")
    check(fails, listening > waited and waited >= 0 and at[listening] >= LOAD_S and at[listening] < LOAD_S + 0.5,
          "it says when it listens, as the load ends: " + (String(round(at[listening], 2)) + " s" if listening >= 0 else String("never")))

    print("3. the utterance it was too busy for")
    check(fails, len(missed) == 1, String(len(missed)) + " report of speech while busy (1 expected)")
    if len(missed) == 1:
        var s = seconds_in(notes[missed[0]])
        check(fails, s > UTT_S - 0.2 and s < UTT_S + 0.2, "it gives the utterance's length: " + notes[missed[0]])
    check(fails, len(taken) == 1 and taken[0] > UTT_S - 0.2 and taken[0] < UTT_S + 0.2,
          "and the caller is handed it once, for its record (`take_missed`): "
          + (String(round(taken[0], 2)) + " s" if len(taken) == 1 else String(len(taken)) + " times"))
    check(fails, loop.stt.calls == 3 and sent == 3 and heard == 3,
          String(loop.stt.calls) + " clips reached the recogniser (3 expected: no clip made of the tail of the second)")

    print("4. every clip is a whole utterance")
    var whole = len(loop.stt.lens) == 3
    var lens = String("")
    for n in loop.stt.lens:
        var sec = Float64(n) / 16000.0
        lens += String(round(sec, 2)) + " s  "
        # the utterance, the pre-roll before it (0.35 s) and the quiet that ends it (0.6 s)
        if sec < UTT_S + 0.8 or sec > UTT_S + 1.1:
            whole = False
    check(fails, whole, "utterance + pre-roll + closing quiet, 2.0 to 2.3 s each: " + lens)
    check(fails, loop.jev.asked == 0 and loop.llm.asked == 0, "no decision and no reply were asked for (one-letter transcripts)")

    print("5. the reply's prompt is handed over while the person speaks")
    # once per utterance listened to, after it began and before it was sent to the recogniser: the only stretch
    # in which the robot's models have nothing to run (docs/NN_MODEL_PORTS_PLAN.md §12.8, S1)
    var during = len(prep_at) == len(sent_at)
    var when = String("")
    for k in range(min(len(prep_at), len(sent_at))):
        when += String(round(sent_at[k] - prep_at[k], 2)) + " s  "
        if prep_at[k] >= sent_at[k] or sent_at[k] - prep_at[k] > UTT_S + 1.2 or (k > 0 and prep_at[k] <= sent_at[k - 1]):
            during = False
    check(fails, during and loop.llm.prepared == 3, String(loop.llm.prepared) + " prompts handed over (3 expected), each before its clip was sent: "
          + when)
    check(fails, loop.llm.prepared_sys == loop.cfg.chat_sys and loop.llm.prepared_sys != "", "and it is the loop's reply prompt")

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
