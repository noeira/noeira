# +--------------------------------------------------------------------------+ #
# | The voice loop — speech to a decision, non-blocking, in one place
# +--------------------------------------------------------------------------+ #
"""Everything between a microphone and a decision, as one object a render loop
polls once a frame.

    var vl = G1VoiceLoop(cfg, bank)          # owns mic, TTS, STT, Jev, LLM
    while running:
        var ev = vl.poll(ctx)                # never blocks, NEVER raises
        if ev.kind == VL_COMMAND:   run_bank_row(ev.name); queue(ev.chain)
        elif ev.kind == VL_SPEC:    blend_into(ev.z)
        elif ev.kind == VL_WORLD:   navigator.go(ev.destination)
        elif ev.kind == VL_TALK:    pass                 # already spoken
        ui.label(vl.status_line(), vl.status_level())

## Why this is a library and not a block in the demo

Because a second copy of it would diverge, and this particular loop is where
every bug of §12.55 to §12.63 lived: the VAD floor tracking and its open/close
multipliers, the echo gate (three versions, two of them wrong), the deadlines
and the measured bounds they came from, the `no such command` -> reward-spec
route, and the conversational state that makes "plus vite" resolve. That list
is the changelog of ONE loop. `g1_command_language` is already a library for
the same reason — §12.54 measured its wording moving `P(none)` by half, and two
copies of a wording are two different systems wearing one name.

The room session (`noeira-77`) has a G1 in a furnished scene with an MPPI
planner; its binary needs this loop and must not reimplement it. Hence a
struct: whichever binary owns the room owns one of these, and the next VAD bug
is fixed once.

## THE SPLIT: this owns the AUDIO, the caller owns the ROBOT

⚠ Deliberate, and the boundary is where it is for a measured reason.

**Here**: the microphone, the voice-activity detector, the text-to-speech and
the echo gate between them. The gate and the speaking must live together —
splitting them is exactly how the gate raced (§12.56: two `say` shells shared
one flag file, the first to finish opened the microphone while the second was
still talking, and the robot re-heard its own "je tourne à gauche"). So the
caller never touches the voice; it calls `say`.

**The caller**: the latent, the blend, the bank index, the chain's distance and
clock, the locomotion timeout, the HUD. All of those read the ROBOT's state,
which this object knows nothing about. `poll` returns a decision; scheduling it
is the caller's business, and §12.60 is what happens when a decision assumes
something about the robot it is not entitled to.

## ⚠ `poll` DOES NOT RAISE

It is declared non-raising so the compiler enforces it. A render loop that
stops because an HTTP call returned 503 also stops the physics step, and
`poll()` returning True means "the call finished", not "it worked" — `result`
is what raises (§12.60). Every failure comes back as an event or a status
string instead.
"""

from std.time import perf_counter_ns

from noeira.ai.audio_io import MicCapture, rms, LocalVoice
from noeira.ai.chat import ChatClient, ChatMessage, ToolSpec
from noeira.ai.jev import JevClient, JevQuestions
from noeira.ai.speech import SpeechToText, STT_RAW
from noeira.io.wav import WavAudio
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import (
    g1_command_questions, g1_decide, g1_decide_chain, g1_command_state,
    g1_since_word, G1Context, G1LangPick, G1_Q_EXTENT,
    g1_destination_questions, g1_route_destination,
)
from noeira.envs.robots.g1_reward_vocab import G1Term
from noeira.envs.robots.g1_spec import (
    G1Pool, g1_spec_questions, g1_spec_prompt, g1_spec_from_answers,
    g1_spec_admit, g1_spec_bank_baseline, g1_spec_describe, g1_spec_record,
    G1_SPEC_D,
)

# ── the states ────────────────────────────────────────────────────────────
comptime VL_ST_IDLE: Int = 0
comptime VL_ST_REC: Int = 1
comptime VL_ST_STT: Int = 2
comptime VL_ST_JEV: Int = 3
comptime VL_ST_SPEC: Int = 4
comptime VL_ST_CHAT: Int = 5
comptime VL_ST_DEST: Int = 6
"""The first of the two requests `request_wording` makes: destination only."""

# ── the events ────────────────────────────────────────────────────────────
comptime VL_NONE: Int = 0
comptime VL_COMMAND: Int = 1
"""A bank row to run: `name`, `extent`, `chain`, and `needs_world` so the
caller can say the part it cannot do."""
comptime VL_SPEC: Int = 2
"""A novel latent the bank has no name for: `z` and `name` as its label."""
comptime VL_WORLD: Int = 3
"""A destination for a navigator: `destination`. ⚠ `name` is EMPTY — see
`g1_decide`, which clears it so a stale bank row cannot race the planner."""
comptime VL_TALK: Int = 4
"""A reply, already spoken. `text` is what was said."""
comptime VL_REFUSED: Int = 5
"""Nothing to do, and `text` says why — for a HUD, never spoken aloud."""

# ⚠ THE DEADLINES, FROM THIS PROJECT'S OWN MEASURED LATENCIES (§12.60). A
# session hung on "transcribing..." and the demo could not say whether it was
# slow or dead, because `HttpClient(120000, 10000)` with `retries = 2` lets one
# request sit for about six minutes. Whisper on the HF endpoint has returned
# 0.71, 0.79, 1.97, 4.99, 5.30 and 5.41 s across those sessions and Jev
# 0.25-0.61, so 20 s is ~4x the slowest transcription ever seen here and 12 s
# ~20x the slowest decision. A generative reply gets 30.
comptime VL_STT_DEADLINE_S: Float64 = 20.0
comptime VL_JEV_DEADLINE_S: Float64 = 12.0
comptime VL_CHAT_DEADLINE_S: Float64 = 30.0

# ⚠ AND THE TAIL IS A CAPTURE-BUFFER DRAIN, NOT A DURATION GUESS. `speaking()`
# goes false the instant playback ends, but ffmpeg's pipe is 64 KiB — about
# 2 s — so samples recorded WHILE the robot was talking are still queued behind
# it. A fixed window for a known buffer; the thing it replaced was an estimate
# of utterance length that was wrong in ORDER (§12.56: "spin_left" 2.79 s
# against "both_hands_up" 2.20 s, the longer word being the shorter sound).
comptime VL_ECHO_TAIL_S: Float64 = 0.45

comptime VL_PREROLL_S: Float64 = 0.35
"""Audio kept before the detector opens, so the first syllable survives."""
comptime VL_DRAIN_S: Float64 = 0.35

# ── the voice-activity detector, every constant measured ──────────────────
comptime VL_OPEN_MULT: Float64 = 5.0
comptime VL_CLOSE_MULT: Float64 = 2.5
comptime VL_OPEN_MIN: Float64 = 0.0060
comptime VL_OPEN_FRAMES: Int = 3
comptime VL_MIN_PEAK_MULT: Float64 = 1.6
comptime VL_FLOOR_MIN: Float64 = 0.00005


# ── the detector's arithmetic, as functions so it can be GATED ────────────
# ⚠ THIS WAS INLINE IN THE LOOP AND THEREFORE UNTESTABLE, which is the wrong
# place for the three lines with the worst history in this file. Both of
# §12.55's detector bugs live here.


def g1_vad_open_at(floor: Float64) -> Float64:
    """The level a segment opens at. ⚠ FLOORED, because `floor * 5` in a very
    quiet room is below the microphone's own noise and the detector would open
    on nothing."""
    var v = floor * VL_OPEN_MULT
    return v if v > VL_OPEN_MIN else VL_OPEN_MIN


def g1_vad_close_at(floor: Float64) -> Float64:
    """The level a segment closes below.

    ⚠ IT NEEDS A FLOOR OF ITS OWN, and this is the bug that cost a session.
    It was `floor * 2.5` alone, and the floor is an average over QUIET frames
    — so in a room whose between-word noise sits above that, a segment opens
    and NEVER CLOSES: one ran the full 10 s max-segment guard for a 3 s
    question, and another recorded 5.99 s for the word "you". Tying it to the
    opening threshold keeps the two in proportion whatever the room is doing.
    """
    var v = floor * VL_CLOSE_MULT
    var lo = VL_OPEN_MIN * 0.5
    return v if v > lo else lo


def g1_vad_floor_step(floor: Float64, level: Float64) -> Float64:
    """One EMA step of the tracked quiet level.

    ⚠ THE FLOOR IS TYPICAL QUIET, NOT THE QUIETEST INSTANT. Tracking the
    minimum put `close_at` below ordinary room noise, which is the same
    never-closing failure from the other direction. ⚠ And the CALLER must only
    call this on frames BELOW `g1_vad_open_at`, or speech raises the floor it
    is measured against and the detector deafens itself mid-sentence.
    """
    var f = floor * 0.98 + level * 0.02
    return f if f > VL_FLOOR_MIN else VL_FLOOR_MIN


@fieldwise_init
struct G1VoiceEvent(Copyable, Movable):
    var kind: Int
    var name: String
    """A bank row (VL_COMMAND) or a spec label (VL_SPEC)."""
    var text: String
    """The reply (VL_TALK), the reason (VL_REFUSED), or the transcript."""
    var destination: String
    var z: List[Float64]
    var conf: Float64
    var p_none: Float64
    var needs_world: Float64
    var extent: Float64
    """The `extent` score, already a float between its levels. The CALLER
    turns it into metres or seconds, because only the caller knows whether the
    row it is about to run is locomotion."""
    var chain: List[String]
    var chain_conf: List[Float64]

    @staticmethod
    def none() -> G1VoiceEvent:
        return G1VoiceEvent(
            VL_NONE, String(""), String(""), String(""), List[Float64](),
            0.0, 0.0, 0.0, 0.0, List[String](), List[Float64](),
        )


struct G1VoiceConfig(Copyable, Movable):
    """What the caller chooses. Defaults are the demo's measured ones."""

    var lang: String
    """ISO-639-1 pinned on the recogniser. ⚠ "" lets Whisper auto-detect from
    a one-word utterance, which is how "cours" came back "cool" (§12.56)."""
    var stt_spec: String
    """`hf`, `groq` or `local:<base_url>` (e.g. `local:http://127.0.0.1:8765/v1`).
    ⚠ Only a multipart backend takes a text vocabulary prompt; the HF endpoint
    RAISES on one, so `vocab` is sent to `groq` and `local:` only."""
    var vocab: String
    var vad: Bool
    var mute: Bool
    var hang_s: Float64
    var min_seg: Float64
    var max_seg: Float64
    var max_none: Float64
    var min_top: Float64
    var stt_dl: Float64
    var jev_dl: Float64
    var chat_dl: Float64
    var chat_sys: String
    var pool_path: String
    """"" disables the cache-miss path; the bank then refuses what it has no
    name for instead of writing a reward spec."""
    var cand_path: String
    var destinations: List[String]
    var dest_descs: List[String]
    var mic_dev: String
    var refuse_placeless: Bool
    """Refuse "go to the kitchen" when the scene has no kitchen, instead of
    walking in an arbitrary direction and saying so.

    ⚠ OFF BY DEFAULT, because whether a partial action is safe is a property
    of the SCENE. §12.54's rule is to do the part it can and name the part it
    cannot — harmless in an empty void, and walking into the fridge in a
    furnished room. A caller with obstacles sets this; a bare viewer does not.
    """
    var llm_spec: String
    """Passed to `ChatClient.from_spec` — `hf`, `openai`, `anthropic`, or a
    local openai-compatible base URL."""
    var request_wording: Bool
    """Ask the destination as a REQUEST with a yes/no gate, the wording
    measured on a local Jev-compatible model, in a request of its OWN
    (`g1_destination_questions`) — the commands are asked in a second request
    only when it routes nothing, so a goto costs the short request alone.
    ⚠ OFF BY DEFAULT: hosted Jev was measured on the default wording, in one
    request. The caller fills `G1Context`'s place fields."""
    var chain: Bool
    """Ask for a second and third step. A caller that runs only the first step
    turns it off, and a local model then answers two rows fewer."""

    def __init__(out self):
        self.lang = String("")
        self.stt_spec = String("hf")
        self.vocab = String("")
        self.vad = True
        self.mute = False
        self.hang_s = 0.6
        self.min_seg = 0.35
        self.max_seg = 10.0
        self.max_none = 0.25
        self.min_top = 0.35
        self.stt_dl = VL_STT_DEADLINE_S
        self.jev_dl = VL_JEV_DEADLINE_S
        self.chat_dl = VL_CHAT_DEADLINE_S
        self.chat_sys = String(
            "You are a small humanoid robot. Answer in one short sentence."
        )
        self.pool_path = String("")
        self.cand_path = String("")
        self.destinations = List[String]()
        self.dest_descs = List[String]()
        self.mic_dev = String("")
        self.refuse_placeless = False
        self.llm_spec = String("hf")
        self.request_wording = False
        self.chain = True


struct G1VoiceLoop(Movable):
    """The microphone, the detector, the recogniser, the decision and the
    voice — one object, polled once a frame."""

    var mic: MicCapture
    var voice: LocalVoice
    var stt: SpeechToText
    var jev: JevClient
    var llm: ChatClient
    var cfg: G1VoiceConfig
    var quest: JevQuestions
    var dest_q: JevQuestions
    """`request_wording` only: the first request. Empty otherwise."""
    var asked: String
    """The state the current decision was asked with, for its second
    request."""
    var spec_q: JevQuestions
    var has_pool: Bool
    var pool: G1Pool
    var zbase: List[Float64]
    var znew: List[Float64]

    var state: Int
    var heard: String
    var pick: G1LangPick
    var t_wait: Int
    var mute_until: Int
    var seg: List[Int16]
    var ring: List[Int16]
    var ring_max: Int
    var floor: Float64
    var level: Float64
    var peak: Float64
    var quiet_since: Int
    var above: Int
    var forced: Bool
    var mic_on: Bool
    var mic_dead: String
    var checked_silence: Bool
    var note: String
    """The last thing worth printing. The caller drains it with `take_note`,
    so this object never owns the log format."""

    def __init__(out self, var cfg: G1VoiceConfig, ref bank: G1CommandBank) raises:
        """⚠ RAISES, unlike `poll`. Opening the microphone, reading a 70 MB
        pool and resolving API keys are all things that should stop a program
        BEFORE its loop starts rather than be swallowed inside one."""
        self.cfg = cfg^
        self.jev = JevClient.from_env()
        # `local:<base_url>` is any OpenAI-compatible server on this machine
        # (Phonon-2's `phonon serve`: English only, `language` is ignored,
        # `prompt` is read as a hotword list — at most 25 words).
        if self.cfg.stt_spec.startswith("local:"):
            self.stt = SpeechToText.openai_compatible(
                String(self.cfg.stt_spec[byte=6:]), String("phonon-2")
            )
        elif self.cfg.stt_spec == "groq":
            self.stt = SpeechToText.groq()
        else:
            self.stt = SpeechToText.huggingface()
        if self.cfg.lang != "":
            self.stt.language = self.cfg.lang
        # ⚠ THE VOCABULARY GOES ONLY TO A BACKEND THAT TAKES ONE. HF's Whisper
        # accepts a prompt as token ids, never as text, and the client RAISES
        # rather than dropping it — which is right, because a silently dropped
        # hint and a hint that did not help are indistinguishable from outside
        # and that is exactly how "cours" was misdiagnosed (§12.56).
        if self.cfg.vocab != "" and self.stt.kind != STT_RAW:
            self.stt.prompt = self.cfg.vocab
        self.llm = ChatClient.from_spec(self.cfg.llm_spec)
        # ⚠ BOTH OF THESE ARE MEASURED AND BELONG WITH THE CLIENT. A reply is
        # one short sentence, so 120 tokens is generous — and Qwen THINKS
        # before answering unless told not to, which took a reply from
        # **10.8 s to 1.4 s** in the AI package's own tests. A robot that
        # pauses eleven seconds before saying hello is not having a
        # conversation, and a caller that had to remember these would
        # eventually not.
        self.llm.max_tokens = 120
        self.llm.extra(
            String("chat_template_kwargs"),
            String('{"enable_thinking": false}'),
        )
        self.voice = LocalVoice()
        # ⚠ `MicCapture.start` IS THE CONSTRUCTOR — it opens the device, so
        # there is no separate `start()` for it and the field cannot be a
        # not-yet-opened capture. A failure here raises out of `__init__`,
        # which is where a missing microphone should stop a program.
        self.mic = MicCapture.start(16000, self.cfg.mic_dev)
        self.dest_q = JevQuestions()
        if self.cfg.request_wording and len(self.cfg.destinations) > 0:
            # two requests: the destination first, then the commands WITHOUT it
            self.dest_q = g1_destination_questions(self.cfg.destinations)
            self.quest = g1_command_questions(
                bank, True, List[String](), List[String](), self.cfg.chain,
            )
        else:
            self.quest = g1_command_questions(
                bank, True, self.cfg.destinations, self.cfg.dest_descs,
                self.cfg.chain,
            )
        self.asked = String("")
        self.spec_q = g1_spec_questions(True)
        self.has_pool = self.cfg.pool_path != ""
        # ⚠ A ONE-ROW PLACEHOLDER when there is no pool: `G1Pool` is not
        # Optional-friendly at this size and a field must be initialised.
        self.pool = (
            G1Pool.load(self.cfg.pool_path) if self.has_pool
            else G1Pool(1, List[Float64](length=G1_SPEC_D, fill=0.0),
                        List[Float64](length=14, fill=0.0))
        )
        self.zbase = List[Float64]()
        if self.has_pool:
            # ⚠ PRECOMPUTED, because recomputing 20-odd baselines per miss is
            # 20 passes over 65 536 rows — about a second, mid-frame (§12.60).
            self.zbase = List[Float64](
                length=bank.count() * G1_SPEC_D, fill=0.0
            )
            g1_spec_bank_baseline(self.pool, bank, self.zbase)
        self.znew = List[Float64](length=G1_SPEC_D, fill=0.0)

        self.state = VL_ST_IDLE
        self.heard = String("")
        self.pick = G1LangPick()
        self.t_wait = perf_counter_ns()
        self.mute_until = perf_counter_ns()
        self.seg = List[Int16]()
        self.ring = List[Int16]()
        self.ring_max = Int(VL_PREROLL_S * 16000.0)
        self.floor = 0.0010
        self.level = 0.0
        self.peak = 0.0
        self.quiet_since = perf_counter_ns()
        self.above = 0
        self.forced = False
        self.mic_on = True
        self.mic_dead = String("")
        self.checked_silence = False
        self.note = String("")

    def __init__(out self, *, deinit move: Self):
        self.mic = move.mic^
        self.voice = move.voice^
        self.stt = move.stt^
        self.jev = move.jev^
        self.llm = move.llm^
        self.cfg = move.cfg^
        self.quest = move.quest^
        self.dest_q = move.dest_q^
        self.asked = move.asked^
        self.spec_q = move.spec_q^
        self.has_pool = move.has_pool
        self.pool = move.pool^
        self.zbase = move.zbase^
        self.znew = move.znew^
        self.state = move.state
        self.heard = move.heard^
        self.pick = move.pick^
        self.t_wait = move.t_wait
        self.mute_until = move.mute_until
        self.seg = move.seg^
        self.ring = move.ring^
        self.ring_max = move.ring_max
        self.floor = move.floor
        self.level = move.level
        self.peak = move.peak
        self.quiet_since = move.quiet_since
        self.above = move.above
        self.forced = move.forced
        self.mic_on = move.mic_on
        self.mic_dead = move.mic_dead^
        self.checked_silence = move.checked_silence
        self.note = move.note^

    # ── what the caller may do to it ──────────────────────────────────
    def start(mut self) raises:
        """Open the microphone and warm the connections. ⚠ `warm_up` costs
        19.5 ms against 0.16 ms warmed — one dropped frame now instead of one
        in the middle of an utterance."""
        self.jev.warm_up()
        self.stt.warm_up()
        # ⚠ `talk` needs its client warmed BEFORE the loop like the others, or
        # its first reply pays 19.5 ms inside one frame.
        self.llm.warm_up()

    def stop(mut self):
        try:
            self.mic.stop()
        except:
            pass
        try:
            self.voice.stop()
        except:
            pass

    def say(mut self, text: String):
        """Speak, and never raise. ⚠ THE ONLY WAY THE CALLER SHOULD SPEAK.
        The echo gate reads `voice.speaking()`, so a caller that owned its own
        TTS would reopen §12.56's race — two voices and one "done"."""
        try:
            self.voice.say(text)
        except:
            pass
        self.mute_until = perf_counter_ns() + _vl_ns(VL_ECHO_TAIL_S)

    def force(mut self):
        """TAB: end the current segment now, or start one. For a room too loud
        for the detector to close."""
        if self.state == VL_ST_IDLE:
            self.heard = String("")
            self.seg = self.ring.copy()
            self.ring = List[Int16]()
            self.peak = self.level
            self.quiet_since = perf_counter_ns()
            self.forced = False
            self.state = VL_ST_REC
        elif self.state == VL_ST_REC:
            self.forced = True

    def toggle_mic(mut self) -> String:
        """Close or reopen the microphone. Returns a line for the log.

        ⚠ IT ACTUALLY KILLS THE CAPTURE, and that is a privacy property rather
        than a convenience. An always-open mic in a room of people is a privacy
        problem before it is a false-trigger problem: everything said near the
        machine would otherwise be posted to a transcription service. `stop()`
        kills ffmpeg, so the OS recording indicator GOES OUT and nothing is
        captured at all — not captured-and-ignored. A version of this that
        only flipped a flag would look identical from the code and be a
        different promise to the room.
        """
        if self.mic_on:
            try:
                self.mic.stop()
            except:
                pass
            self.mic_on = False
            self.ring = List[Int16]()
            self.level = 0.0
            self.state = VL_ST_IDLE
            return String("[mic] closed")
        try:
            self.mic = MicCapture.start(16000, self.cfg.mic_dev)
            self.mic_on = True
            self.mic_dead = String("")
            # ⚠ and the detector starts over: a floor learned before the gap
            # describes a room that may have changed, and `checked_silence`
            # must re-run or a newly-muted device would never be reported.
            self.checked_silence = False
            self.peak = 0.0
            self.floor = 0.0010
            return String("[mic] reopened")
        except e:
            self.mic_dead = String(e)
            return String("[mic] ") + self.mic_dead

    def take_note(mut self) -> String:
        """The last log line, consumed. Empty when there is nothing new."""
        var n = self.note
        self.note = String("")
        return n

    def is_busy(self) -> Bool:
        """True while a transcription, decision or reply is in flight — the
        caller may want to suppress a keyboard command that would race it."""
        return self.state != VL_ST_IDLE

    def speaking(mut self) -> Bool:
        """⚠ A GATE THAT RAISES WOULD TAKE THE LOOP DOWN, and the safe default
        is to assume we ARE talking rather than open the microphone onto our
        own voice."""
        try:
            return self.voice.speaking()
        except:
            return True


    # ── the loop ──────────────────────────────────────────────────────
    def poll(
        mut self, ref ctx: G1Context, ref bank: G1CommandBank
    ) -> G1VoiceEvent:
        """One frame. ⚠ DOES NOT RAISE — see the module note.

        `ctx.since_s` is the CALLER's to set before calling: it is measured
        from the robot's own command clock, which this object does not have.
        """
        var ev = G1VoiceEvent.none()

        # ── the microphone, every frame ───────────────────────────────
        # ⚠ READ EVEN WHILE THE ROBOT SPEAKS. The pipe is 64 KiB, about 2 s,
        # and past that ffmpeg blocks and the device drops audio. The samples
        # are read and DISCARDED, never ringed, never levelled.
        if self.speaking():
            self.mute_until = perf_counter_ns() + _vl_ns(VL_ECHO_TAIL_S)
        var echoing = perf_counter_ns() < self.mute_until
        if self.mic_on and self.mic_dead == "":
            try:
                var pcm = self.mic.read()
                if echoing:
                    self.level = 0.0
                    # ⚠ AND DROP THE RING. It holds the last 0.35 s and is
                    # prepended to the next segment — which would hand Whisper
                    # the tail of the robot's own sentence as the first
                    # syllable of yours.
                    self.ring = List[Int16]()
                elif len(pcm) > 0:
                    self.level = rms(pcm)
                    if self.level > self.peak:
                        self.peak = self.level
                    if self.state == VL_ST_REC:
                        for i in range(len(pcm)):
                            self.seg.append(pcm[i])
                    else:
                        for i in range(len(pcm)):
                            self.ring.append(pcm[i])
                        if len(self.ring) > self.ring_max:
                            var keep = List[Int16]()
                            for i in range(len(self.ring) - self.ring_max,
                                           len(self.ring)):
                                keep.append(self.ring[i])
                            self.ring = keep^
                        # ⚠ THE FLOOR IS TYPICAL QUIET, NOT THE QUIETEST
                        # INSTANT. Tracking the minimum put the close
                        # threshold BELOW ordinary room noise, so a segment
                        # that opened never closed — a real session recorded
                        # 5.99 s for "you" and 7.06 s for "Cool. Run.",
                        # ending on the max-segment guard rather than on
                        # silence. Only frames BELOW the open threshold count,
                        # or speech raises the floor it is measured against
                        # and the detector deafens itself mid-sentence.
                        if self.level < g1_vad_open_at(self.floor):
                            self.floor = g1_vad_floor_step(
                                self.floor, self.level
                            )
                if not self.checked_silence and self.mic.seconds_read() > 1.0:
                    self.checked_silence = True
                    if self.mic.digital_silence(0.5):
                        self.mic_dead = String(
                            "mic is digital silence — muted, disabled, or "
                            "permission refused"
                        )
                        self.note = String("[mic] ") + self.mic_dead
                    else:
                        # ⚠ this used to say "noise floor" and print the PEAK
                        # — two different numbers, and the one it printed was
                        # the loudest thing in the first second.
                        self.note = (
                            String("[mic] input live — ")
                            + _vl_f2(self.mic.seconds_read())
                            + String(" s read, floor ") + _vl_f4(self.floor)
                            + String(" peak ") + _vl_f4(self.peak)
                        )
            except e:
                self.mic_dead = String(e)
                self.note = String("[mic] ") + self.mic_dead

        var open_at = g1_vad_open_at(self.floor)
        var close_at = g1_vad_close_at(self.floor)

        # ⚠ VAD OPENS ONLY FROM IDLE. While a transcription or a decision is
        # in flight, speech is still ringed but starts nothing — one call per
        # client, and a queue of half-heard commands is worse than a missed
        # one. The 1.2 s guard keeps the floor estimate from opening a segment
        # on its own first samples.
        if self.level > open_at and not echoing:
            self.above += 1
        else:
            self.above = 0
        var can_open = False
        try:
            can_open = self.mic.seconds_read() > 1.2
        except:
            can_open = False
        if (self.mic_on and self.mic_dead == "" and self.state == VL_ST_IDLE
            and self.cfg.vad and not echoing
            and self.above >= VL_OPEN_FRAMES and can_open):
            self.heard = String("")
            self.pick = G1LangPick()
            self.seg = self.ring.copy()
            self.ring = List[Int16]()
            self.peak = self.level
            self.quiet_since = perf_counter_ns()
            self.forced = False
            self.state = VL_ST_REC

        # ── recording ─────────────────────────────────────────────────
        if self.state == VL_ST_REC:
            if self.level > close_at:
                self.quiet_since = perf_counter_ns()
            var quiet_s = Float64(perf_counter_ns() - self.quiet_since) / 1e9
            var seg_s = Float64(len(self.seg)) / 16000.0
            # ⚠ THREE WAYS TO END, and the last two are not optional. Silence
            # is the normal one. A segment that never falls quiet (a fan, a
            # conversation across the room) would otherwise record for ever
            # and post a minute of audio; a forced end is what TAB is for.
            var over = seg_s > self.cfg.max_seg
            var done = self.forced or over or (quiet_s > self.cfg.hang_s)
            if done and seg_s > VL_DRAIN_S:
                self.forced = False
                # ⚠ a cough, a chair, a door. Below `min_seg` it is not
                # speech, and sending it costs a Whisper call to be told so.
                if seg_s < self.cfg.min_seg:
                    self.note = (String("[vad] dropped ") + _vl_f2(seg_s)
                                 + String(" s — under ")
                                 + _vl_f2(self.cfg.min_seg))
                    self.state = VL_ST_IDLE
                elif self.peak < open_at * VL_MIN_PEAK_MULT:
                    self.note = (String("[vad] dropped — peak ")
                                 + _vl_f4(self.peak) + String(" under ")
                                 + _vl_f4(open_at * VL_MIN_PEAK_MULT))
                    self.state = VL_ST_IDLE
                else:
                    try:
                        var audio = WavAudio(16000, 1, self.seg.copy())
                        self.stt.start(audio)
                        self.t_wait = perf_counter_ns()
                        self.note = (String("[stt] ") + String(len(self.seg))
                                     + String(" samples (") + _vl_f2(seg_s)
                                     + String(" s ), peak ")
                                     + _vl_f4(self.peak)
                                     + (String(" — max-seg") if over
                                        else String("")))
                        self.state = VL_ST_STT
                    except e:
                        self.note = String("[stt] could not start: ") + String(e)
                        self.state = VL_ST_IDLE

        # ── transcribing ──────────────────────────────────────────────
        elif self.state == VL_ST_STT:
            # ⚠ THE DEADLINE IS CHECKED BEFORE THE POLL, so a call that has
            # already blown its budget is cancelled rather than waited on for
            # one more frame.
            var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
            if el > self.cfg.stt_dl:
                self.note = (String("[stt] GAVE UP after ") + _vl_f2(el)
                             + String(" s — the endpoint never answered."))
                try:
                    self.stt.cancel()
                except:
                    pass
                self.state = VL_ST_IDLE
            else:
                var ready = False
                try:
                    ready = self.stt.poll()
                except:
                    ready = True
                if ready:
                    # ⚠ `poll` RETURNS TRUE ON FAILURE TOO — "the call
                    # finished", not "it worked" — and `result` is what
                    # raises. On failure `heard` stays empty and the
                    # letters guard below routes it back to IDLE, so no
                    # second exit path is needed.
                    var txt = String("")
                    var lat = 0.0
                    try:
                        var tr = self.stt.result()
                        txt = tr.text
                        lat = tr.latency_ms
                    except e:
                        self.note = String("[stt] FAILED: ") + String(e)
                    self.heard = txt
                    if lat > 0.0:
                        self.note = (String("[heard] ") + self.heard
                                     + String(" (") + _vl_f2(lat)
                                     + String(" ms )"))
                    # ⚠ Whisper returns "." or " " for a cough. Asking a
                    # model which of 23 commands a full stop means costs a
                    # call to be told none of them.
                    var letters = 0
                    for ch in self.heard.codepoints():
                        if ch.to_u32() > 64:
                            letters += 1
                    if letters < 3:
                        self.state = VL_ST_IDLE
                    else:
                        try:
                            self.asked = g1_command_state(self.heard, ctx)
                            if self._two_requests():
                                self.jev.start(self.asked, self.dest_q)
                                self.state = VL_ST_DEST
                            else:
                                self.jev.start(self.asked, self.quest)
                                self.state = VL_ST_JEV
                            self.t_wait = perf_counter_ns()
                        except e:
                            self.note = String("[jev] could not start: ") \
                                        + String(e)
                            self.state = VL_ST_IDLE
                        ev.text = self.heard

        # ── deciding: the destination request (`request_wording`) ──────
        elif self.state == VL_ST_DEST:
            var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
            if el > self.cfg.jev_dl:
                self.note = (String("[jev] GAVE UP after ") + _vl_f2(el)
                             + String(" s"))
                try:
                    self.jev.cancel()
                except:
                    pass
                self.state = VL_ST_IDLE
            else:
                var ready = False
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._destination()

        # ── deciding ──────────────────────────────────────────────────
        elif self.state == VL_ST_JEV:
            var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
            if el > self.cfg.jev_dl:
                self.note = (String("[jev] GAVE UP after ") + _vl_f2(el)
                             + String(" s"))
                try:
                    self.jev.cancel()
                except:
                    pass
                self.state = VL_ST_IDLE
            else:
                var ready = False
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._decide(ctx, bank)

        # ── the reward spec, on a cache miss ──────────────────────────
        elif self.state == VL_ST_SPEC:
            var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
            if el > self.cfg.jev_dl:
                self.note = (String("[spec] GAVE UP after ") + _vl_f2(el)
                             + String(" s"))
                try:
                    self.jev.cancel()
                except:
                    pass
                self.state = VL_ST_IDLE
            else:
                var ready = False
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._spec(bank)

        # ── the reply ─────────────────────────────────────────────────
        elif self.state == VL_ST_CHAT:
            var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
            if el > self.cfg.chat_dl:
                self.note = (String("[talk] GAVE UP after ") + _vl_f2(el)
                             + String(" s"))
                try:
                    self.llm.cancel()
                except:
                    pass
                self.state = VL_ST_IDLE
            else:
                try:
                    _ = self.llm.poll()
                except:
                    pass
                var done = False
                try:
                    done = self.llm.done()
                except:
                    done = True
                if done:
                    var rtxt = String("")
                    try:
                        var rep = self.llm.result()
                        rtxt = rep.text
                    except e:
                        self.note = String("[talk] FAILED: ") + String(e)
                    if rtxt != "":
                        self.note = String("[said] ") + rtxt
                        if not self.cfg.mute:
                            self.say(rtxt)
                        ev.kind = VL_TALK
                        ev.text = rtxt
                    self.state = VL_ST_IDLE

        return ev^

    def _two_requests(self) -> Bool:
        return self.cfg.request_wording and len(self.cfg.destinations) > 0

    def _destination(mut self) -> G1VoiceEvent:
        """The first request's answer: a goto, or the commands' request."""
        var ev = G1VoiceEvent.none()
        self.state = VL_ST_IDLE
        try:
            var ans = self.jev.result()
            var dest = g1_route_destination(ans)
            if dest != "":
                ev.kind = VL_WORLD
                ev.destination = dest
                ev.text = self.heard
                self.note = (String("[world] ") + dest + String(" ")
                             + _vl_f2(ans.probability(String("destination"), dest))
                             + String(" (") + _vl_f2(ans.latency_ms)
                             + String(" ms)"))
                return ev^
            # ⚠ THE SAME STATE, not a fresh one: the robot may have moved on
            # while the first request was answered, and the two answers must
            # be about the same moment.
            self.jev.start(self.asked, self.quest)
            self.t_wait = perf_counter_ns()
            self.state = VL_ST_JEV
        except e:
            self.note = String("[jev] FAILED: ") + String(e)
        return ev^

    def _decide(mut self, ref ctx: G1Context, ref bank: G1CommandBank) -> G1VoiceEvent:
        """The decision, and the four ways it can go."""
        var ev = G1VoiceEvent.none()
        var got = False
        try:
            var ans = self.jev.result()
            # with two requests the destination was decided already, and
            # these answers carry no destination question
            var two = self._two_requests()
            self.pick = g1_decide(
                ans, bank, self.cfg.max_none, self.cfg.min_top, 0.5, True,
                len(self.cfg.destinations) > 0 and not two, 0.5, 0.5,
                self.cfg.refuse_placeless, dest_decided=two,
            )
            ev.conf = self.pick.conf
            ev.p_none = self.pick.p_none
            ev.needs_world = self.pick.needs_world
            ev.text = self.heard
            if self.pick.destination != "":
                # ⚠ A DESTINATION IS A SUCCESS WITH A DIFFERENT HANDLER, and
                # `g1_decide` has already cleared `name` so a stale bank row
                # cannot race the planner.
                ev.kind = VL_WORLD
                ev.destination = self.pick.destination
                self.note = (String("[world] ") + self.pick.destination
                             + String(" ") + _vl_f2(self.pick.dest_conf))
            elif self.pick.name != "":
                ev.kind = VL_COMMAND
                ev.name = self.pick.name
                ev.extent = ans.score(String(G1_Q_EXTENT))
                if self.cfg.chain:
                    var rest = g1_decide_chain(ans, bank)
                    for si in range(len(rest)):
                        ev.chain.append(rest[si].name)
                        ev.chain_conf.append(rest[si].conf)
                self.note = (String("[pick] ") + self.pick.name + String(" ")
                             + _vl_f2(self.pick.conf) + String("  P(none) ")
                             + _vl_f2(self.pick.p_none) + String("  world ")
                             + _vl_f2(self.pick.needs_world))
            elif self.pick.talk:
                # ⚠ Nothing about the robot's motion changes — it goes on
                # doing whatever it was doing while it replies.
                try:
                    var msgs = List[ChatMessage]()
                    msgs.append(ChatMessage.user(self.heard))
                    self.llm.start(
                        msgs, self.cfg.chat_sys, List[ToolSpec](), False
                    )
                    self.t_wait = perf_counter_ns()
                    self.note = String("[talk] ") + _vl_f2(self.pick.conf)
                    self.state = VL_ST_CHAT
                    got = True
                except e:
                    self.note = String("[talk] could not start: ") + String(e)
            else:
                # ⚠ A REFUSAL FOR `no such command` IS A CACHE MISS: the bank
                # has no NAME for this and the reward vocabulary may still be
                # able to say it. Any other reason is a real refusal and must
                # NOT be routed there, or background speech starts generating
                # latents.
                if self.has_pool and self.pick.reason == "no such command":
                    try:
                        # ⚠ THE INSTRUCTION ALONE. Asking with `doing` in the
                        # state made one instruction give ESS 5089 while
                        # walking and 301 while standing (§12.62).
                        self.jev.start_text(
                            g1_spec_prompt(self.heard), self.spec_q
                        )
                        self.t_wait = perf_counter_ns()
                        self.note = (String("[miss] P(none) ")
                                     + _vl_f2(self.pick.p_none)
                                     + String(" — asking for a reward spec"))
                        self.state = VL_ST_SPEC
                        got = True
                    except e:
                        self.note = String("[miss] ") + String(e)
                if not got:
                    # ⚠ NEVER SPOKEN ALOUD. "no such command" is heard,
                    # transcribed and refused again — the loop that filled a
                    # whole session's log.
                    ev.kind = VL_REFUSED
                    ev.text = self.pick.reason
                    self.note = (String("[refused] ") + self.pick.reason
                                 + String("  P(none) ")
                                 + _vl_f2(self.pick.p_none)
                                 + String("  addressed ")
                                 + _vl_f2(self.pick.addressed)
                                 + String("  world ")
                                 + _vl_f2(self.pick.needs_world))
        except e:
            self.note = String("[jev] FAILED: ") + String(e)
        # ⚠ ONLY when the decision did not hand off. The talk and spec
        # branches move the state on, and an unconditional reset would drop
        # their request on the floor.
        if self.state == VL_ST_JEV:
            self.state = VL_ST_IDLE
        return ev^

    def _spec(mut self, ref bank: G1CommandBank) -> G1VoiceEvent:
        var ev = G1VoiceEvent.none()
        try:
            var sa = self.jev.result()
            var sterms = List[G1Term]()
            var n_scaf = g1_spec_from_answers(sa, self.pool, bank, sterms)
            if n_scaf < 0:
                self.note = String("[spec] the model named no quantity")
            else:
                var v = g1_spec_admit(
                    self.pool, bank, self.zbase, sterms, self.znew, n_scaf
                )
                var what = g1_spec_describe(sterms, n_scaf)
                self.note = (String("[spec] ") + what + String("  ESS ")
                             + String(Int(v.ess)))
                if v.ok:
                    ev.kind = VL_SPEC
                    ev.name = what
                    for k in range(G1_SPEC_D):
                        ev.z.append(self.znew[k])
                    # ⚠ RECORDED ONLY HERE, because this is the only place a
                    # spec is known to have actually RUN — a refused spec is
                    # not a promotion candidate.
                    if self.cfg.cand_path != "":
                        try:
                            if g1_spec_record(
                                self.cfg.cand_path, self.pool, sterms, n_scaf
                            ):
                                self.note += String(" (recorded)")
                        except:
                            pass
                    if not self.cfg.mute:
                        self.say(String("ok, ") + what)
                elif v.nearest >= 0:
                    # the spec IS a bank row: run the GATED one, which had
                    # four gates and a CEM refinement this baseline has not.
                    ev.kind = VL_COMMAND
                    ev.name = bank.name_at(v.nearest)
                    self.note += String(" -> ") + ev.name
                else:
                    ev.kind = VL_REFUSED
                    ev.text = v.reason
        except e:
            self.note = String("[spec] FAILED: ") + String(e)
        self.state = VL_ST_IDLE
        return ev^

    # ── what a HUD needs, without this owning the HUD ─────────────────
    def status_line(mut self) -> String:
        """⚠ WITH THE ELAPSED TIME AND THE BUDGET. "transcribing..." alone
        cannot distinguish a 5 s call from a dead one, and a session was lost
        to exactly that question (§12.60)."""
        if self.state == VL_ST_REC:
            return (String("LISTENING ")
                    + _vl_f2(Float64(len(self.seg)) / 16000.0) + String("s"))
        var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
        if self.state == VL_ST_STT:
            return (String("transcribing ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.stt_dl) + String("s"))
        if self.state == VL_ST_DEST:
            return (String("deciding: where? ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.jev_dl) + String("s"))
        if self.state == VL_ST_JEV:
            return (String("deciding ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.jev_dl) + String("s"))
        if self.state == VL_ST_SPEC:
            return (String("writing a reward ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.jev_dl) + String("s"))
        if self.state == VL_ST_CHAT:
            return (String("replying ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.chat_dl) + String("s"))
        if self.mic_dead != "":
            return String("MIC: ") + self.mic_dead
        if not self.mic_on:
            return String("mic muted (M)")
        return String("idle") if self.cfg.vad else String("idle — press TAB")

    def status_level(mut self) -> Int:
        """0 dim, 1 warn, 2 bad. The caller owns the colours."""
        if self.mic_dead != "":
            return 2
        if self.state == VL_ST_IDLE:
            return 0
        var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
        var budget = self.cfg.jev_dl
        if self.state == VL_ST_STT:
            budget = self.cfg.stt_dl
        elif self.state == VL_ST_CHAT:
            budget = self.cfg.chat_dl
        elif self.state == VL_ST_REC:
            return 1
        return 1 if el < budget * 0.5 else 2

    def meter(self) -> Float64:
        return self.level

    def mic_checked(self) -> Bool:
        """True once the first second of audio has been inspected for digital
        silence. ⚠ Until then "live" is not yet a claim this object can make,
        and a HUD that said so anyway would be asserting something it has not
        measured."""
        return self.checked_silence

    def mic_open(self) -> Bool:
        """False when M has CLOSED the device — not muted-in-software. The
        OS recording indicator is out and nothing is captured."""
        return self.mic_on

    def mic_error(self) -> String:
        """Why the microphone is unusable, in ffmpeg's own words, or "".
        ⚠ Carrying the REASON is the point: "grant Terminal microphone
        access" and "the device went away" are different problems, and a demo
        that printed neither cost a day (§12.55)."""
        return self.mic_dead

    def peak_level(self) -> Float64:
        """The loudest moment of the current segment, so a spike that came and
        went is still visible a frame later."""
        return self.peak

    def open_threshold(self) -> Float64:
        """What the level must exceed to start a segment. ⚠ A HUD that shows
        the level without this cannot answer "how far is my voice from
        starting a recording", which is the question a silent demo raises."""
        return g1_vad_open_at(self.floor)

    def noise_floor(self) -> Float64:
        """⚠ THE FLOOR, NOT THE PEAK. The first version of this printed the
        peak under the label "noise floor" — two different numbers, and the
        one it showed was the loudest thing in the first second."""
        return self.floor


def _vl_ns(seconds: Float64) -> Int:
    """Seconds to nanoseconds via milliseconds. ⚠ `Int(x * 1e9)` does not
    compile — the literal drags the product into a SIMD type `Int` will not
    take."""
    return Int(seconds * 1000.0) * 1_000_000


def _vl_f2(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 100.0 + 0.5)
    var f = String(h % 100)
    if h % 100 < 10:
        f = String("0") + f
    var b = String(h // 100) + String(".") + f
    return String("-") + b if neg else b


def _vl_f4(v: Float64) -> String:
    var h = Int((v if v > 0.0 else -v) * 10000.0 + 0.5)
    var f = String(h % 10000)
    while f.byte_length() < 4:
        f = String("0") + f
    return String(h // 10000) + String(".") + f
