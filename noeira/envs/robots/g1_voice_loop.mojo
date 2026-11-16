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

## The camera: the caller's frame, this loop's moment

A frame is ROBOT state, so the caller supplies it (`see`); this loop says WHEN
one is worth taking (`wants_view`: a segment opened, so a decision is coming,
and no frame has come in since — one render per utterance, not per tick).
With `cfg.see`, the frame of the UTTERANCE — taken when its segment opened,
what "in front of you" refers to — is sent with EVERY request of its decision
(the two requests must be about the same moment, as `asked` already is); the
reward-spec request reads the instruction alone and stays blind. Its age is
measured against the segment's opening, NOT the decision's start: the first
headless run waited 14.5 s for a recogniser still loading and the frame it
had was the right one. A frame taken more than `cfg.see_max_age_s` before the
segment opened is not sent (the decision is asked blind, and the note says so). A backend that cannot see (`JevClient`: the trait's default raises)
is asked blind and the loop stops trying.

## ⚠ `poll` DOES NOT RAISE

It is declared non-raising so the compiler enforces it. A render loop that
stops because an HTTP call returned 503 also stops the physics step, and
`poll()` returning True means "the call finished", not "it worked" — `result`
is what raises (§12.60). Every failure comes back as an event or a status
string instead.
"""

from std.time import perf_counter_ns

from noeira.ai.audio_io import MicCapture, LocalVoice, os_voice_for
from noeira.envs.robots.g1_vad import (
    G1Vad, G1VadEvent, g1_vad_open_at, g1_vad_close_at, g1_vad_floor_step,
    VL_PREROLL_S, VL_DRAIN_S, VL_OPEN_MULT, VL_CLOSE_MULT, VL_OPEN_MIN, VL_OPEN_FRAMES,
    VL_MIN_PEAK_MULT, VL_FLOOR_MIN, VL_FRAME,
)
from noeira.ai.chat import ChatClient, ChatMessage, ToolSpec, ChatService
from noeira.ai.jev import JevClient, JevQuestions, DecisionService
from noeira.ai.picture import Picture
from noeira.ai.speech import (
    SpeechToText, TextToSpeech, SpeechToTextService, TextToSpeechService,
)
from noeira.io.fileio import remove_file, write_file_atomic
from noeira.io.wav import WavAudio
from noeira.text.spoken import spoken_sentence_end, spoken_text
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import (
    g1_command_questions, g1_decide, g1_decide_chain, g1_command_state,
    g1_since_word, G1Context, G1LangPick, G1_Q_EXTENT,
    g1_destination_questions, g1_route_destination,
    g1_add_two_places_gate, g1_two_places_questions, g1_two_places_state, g1_route_two_places,
    G1_Q_IS_TWO_PLACES,
    g1_add_memory_reference, g1_route_memory, g1_other_one,
    g1_add_triage, g1_route_triage, g1_trail_questions, g1_resolve_trail,
    g1_command_moves, G1Triage, G1_TRIAGE_OFF, G1_TRIAGE_FULL,
    G1_Q_TRAIL_REL, G1_Q_TRAIL_ANCHOR,
)
from noeira.envs.robots.g1_reward_vocab import G1Term
from noeira.envs.robots.g1_spec import (
    G1Pool, g1_spec_questions, g1_spec_prompt, g1_spec_from_answers,
    g1_add_miss_kind, g1_miss_movement_p, G1_Q_MISS_KIND, G1_MISS_MOVE_P,
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
comptime VL_ST_TRAIL: Int = 7
"""`triage`: the trail request, after the triage said `remembered`."""
comptime VL_ST_CHAIN: Int = 8
"""`dest_chain`: the two-place request, after the gate said two places."""

comptime _VL_Q_CMD: Int = 0
comptime _VL_Q_DEST: Int = 1
comptime _VL_Q_TRAIL: Int = 2
comptime _VL_Q_MEM: Int = 3

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
comptime VL_TTS_DEADLINE_S: Float64 = 5.0
"""A reply synthesised later than this is spoken by the OS voice instead."""

# ⚠ AND THE TAIL IS A CAPTURE-BUFFER DRAIN, NOT A DURATION GUESS. `speaking()`
# goes false the instant playback ends, but ffmpeg's pipe is 64 KiB — about
# 2 s — so samples recorded WHILE the robot was talking are still queued behind
# it. A fixed window for a known buffer; the thing it replaced was an estimate
# of utterance length that was wrong in ORDER (§12.56: "spin_left" 2.79 s
# against "both_hands_up" 2.20 s, the longer word being the shorter sound).
comptime VL_ECHO_TAIL_S: Float64 = 0.45

# The detector (its constants, its thresholds, the frame-by-frame machine) is `g1_vad.mojo`; the
# names are imported above and still importable from here.


def g1_view_fresh(seg_t_ns: Int, view_t_ns: Int, max_age_s: Float64) -> Bool:
    """Whether a frame taken at `view_t_ns` may go with the utterance whose
    segment opened at `seg_t_ns`: taken after it opened (however long the
    recogniser then took), or at most `max_age_s` before."""
    return Float64(seg_t_ns - view_t_ns) / 1e9 <= max_age_s


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
    """A `VL_COMMAND`'s further steps (bank rows). For a `VL_WORLD` with
    `cfg.dest_chain`: the place to go on to after `destination`, if any."""
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
    """The reply model's system prompt. A caller with a `WorldState` rewrites
    it from the state (`set_reply_system`)."""
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
    """Passed to the reply backend's `from_spec`: for `ChatClient`, `hf`,
    `openai`, `anthropic`, or a local openai-compatible base URL; for
    `Lfm3ChatService`, `native` (LFM3-1.7B, int8) / `native-bf16` / `...:<dir>`."""
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
    var tts_spec: String
    """"" = the OS voice (`say` / `espeak`); `local:<base_url>` = an
    OpenAI-compatible `/audio/speech` server on this machine (Kokoro-82M,
    `kokoro_serve.py`). A failed or late synthesis falls back to the OS voice."""
    var tts_voice: String
    """"" picks by `lang`: Kokoro `af_heart` (en) / `ff_siwis` (fr), or
    `os_voice_for(lang)`. Otherwise the voice's name for the backend in use."""
    var tts_dl: Float64
    """Seconds a synthesis may take before the OS voice says it instead."""
    var stream_reply: Bool
    """With a synthesiser (`tts_spec`): the reply is said SENTENCE BY SENTENCE
    as the reply model writes it, each synthesised while the one before plays
    (docs/NN_MODEL_PORTS_PLAN.md §12.8.7, L2). On by default; off = the whole
    reply, synthesised once it is written (as before). The OS voice says the
    whole reply either way."""
    var tts_no_play: Bool
    """Synthesise but do not play: each piece "plays" for its WAV's length, in
    silence, and the microphone is never gated by it. For timing the voice on a
    headless replay (`--mic-input`): the note says when the first audio would
    have started."""
    var dec_spec: String
    """The decision backend's spec (`DEC.from_spec`): for `JevClient`, "" =
    hosted Jev or `JEV_URL`; for `D1Service`, `native` / `native:<dir>`."""
    var memory_route: Bool
    """`request_wording` only: the command request also asks which
    remembered place a request means (`g1_add_memory_reference`), and "the
    other one" goes to the place `G1Context.log` settles, ahead of the command
    decision (`g1_route_memory`). The caller keeps the log (`arrived_at`, and
    `heard_said` with `take_heard`). OFF by default."""
    var dest_words: List[String]
    """Per destination, other words for it in what is heard ("frigo,
    réfrigérateur"), comma-separated; "" for none. `g1_place_mentions`."""
    var triage: Int
    """`request_wording` only: RB-2's triage (`g1_add_triage`), one more
    choice in the first request naming WHAT the sentence asks.
    `G1_TRIAGE_FULL` (hosted Jev): unsafe, the speaker's side, a place the
    state cannot settle, or plain talk -> a spoken reply and NO motion;
    "back to where I was before X" -> the trail request, resolved from
    `G1Context.log`; a step or a turn -> no goto; a request for a place
    that routed no place is never turned into a walk. `G1_TRIAGE_NARROW`
    (d1): unsafe and the speaker's side only, and no goto for a step.
    OFF (`G1_TRIAGE_OFF`) by default. The caller keeps the log, as for
    `memory_route`."""
    var talk_first: Float64
    """`triage` only: when the first request routed no place and the triage's
    top kind is `talk` with at least this probability, the reply starts at
    once, and the commands request is asked once the reply is out: the robot
    answers ~0.7 s sooner (docs/NN_MODEL_PORTS_PLAN.md §12.8.6, S3 item 1).
    ⚠ THE COMMANDS REQUEST IS STILL ASKED, AND ITS DECISION STILL ACTS: d1
    named "no, not the table, stop" `talk` 0.51 and "non, plus le frigo, tu
    bouges pas" 0.58 (RB-1/RB-2 logs), so skipping it would have lost two
    stops. A `talk` answer there is not replied twice. 0 = off (the default:
    every bench measured the order commands, then reply)."""
    var dest_chain: Bool
    """`request_wording` only: "go to the table, then to the door". The
    destination request gets one yes/no question (are two places asked for,
    `g1_add_two_places_gate`); when it says yes for a routed place, one more
    request asks which two and in which order (`g1_two_places_questions`),
    and the `VL_WORLD` event carries the second in `chain`. OFF by default:
    RB-1 and RB-2 measured the destination request without the gate.
    Measured on d1 (`bench_brain_dest_chain_gpu`): +35 ms on the request,
    ~0.5 s more for a two-place sentence only."""
    var miss_kind: Bool
    """The spec request also asks what KIND of thing a sentence that matched
    no command wants (`g1_add_miss_kind`): a movement of the robot's own body
    goes on to the reward sketch; a question, or an action on an object, gets
    a spoken reply and no motion. OFF by default: the spec route's own
    measurements (§12.58) were made without it."""
    var name_desc: Bool
    """Each option's description starts with its name, in the command
    request and the triage (d1 reads an option by its description alone:
    RB-1's d1 arm). OFF by default: Jev was measured without."""
    var see: Bool
    """Decisions see the caller's latest frame (`see`, `wants_view`; the
    module note). Needs a backend that can see (`D1Service`). OFF by
    default: every bench so far measured blind decisions."""
    var see_max_age_s: Float64
    """A frame taken longer than this before its utterance began is not sent."""
    var mic_input: String
    """ffmpeg input arguments used verbatim instead of the device
    (`MicCapture.start`'s `input_args`): `-re -i clip.wav` runs the whole loop
    on a recording, headless. "" = the microphone (`mic_dev`). The recording
    starts when the loop first polls with every model loaded (`loading`; the
    `[voice] listening` line), not at construction. ⚠ Give it 4 s of room
    noise at its head: the capture's `-fflags nobuffer` makes ffmpeg drop the
    first 3.2 s of a FILE (`MicCapture`), and the detector takes its first
    second to learn the floor."""

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
        self.tts_spec = String("")
        self.tts_voice = String("")
        self.tts_dl = VL_TTS_DEADLINE_S
        self.stream_reply = True
        self.tts_no_play = False
        self.dec_spec = String("")
        self.memory_route = False
        self.dest_words = List[String]()
        self.triage = G1_TRIAGE_OFF
        self.talk_first = 0.0
        self.dest_chain = False
        self.miss_kind = False
        self.name_desc = False
        self.see = False
        self.see_max_age_s = 2.0
        self.mic_input = String("")


comptime G1VoiceLoop = G1VoiceLoopOf[SpeechToText, TextToSpeech, JevClient]
"""The loop on the HTTP clients: `stt_spec` / `tts_spec` / `dec_spec` choose
among hosted services and local servers at run time."""


struct G1VoiceLoopOf[
    STT: SpeechToTextService, TTS: TextToSpeechService, DEC: DecisionService, LLM: ChatService = ChatClient,
](Movable):
    """The microphone, the detector, the recogniser, the decision and the
    voice — one object, polled once a frame.

    `STT` / `TTS` / `DEC` / `LLM` are the backends, chosen at BUILD time
    (`G1VoiceLoop` is the HTTP ones); `cfg.stt_spec` / `cfg.tts_spec` /
    `cfg.dec_spec` / `cfg.llm_spec` go to their `from_spec`. `LLM` writes the
    spoken replies: `ChatClient` (HTTP, the default) or
    `noeira.llm.lfm3.service.Lfm3ChatService` (LFM3 in this process)."""

    var mic: Optional[MicCapture]
    """The capture process. ⚠ OPENED AT THE FIRST `poll`, NOT HERE AT
    CONSTRUCTION: a loop is built, then its caller loads a room for twenty
    seconds, and a capture started at construction has by then a backlog that
    comes back faster than real time (`g1_vad.mojo`'s note; in a headless run
    on a recording it ate the first utterances)."""
    var voice: LocalVoice
    var tts: Optional[Self.TTS]
    """`tts_spec` set: the synthesiser. Its WAV is played through `voice`, so
    `voice.speaking()` stays the one end signal."""
    var tts_busy: Bool
    """A reply is being synthesised: the robot is about to speak, and the
    echo gate counts it as speaking."""
    var tts_text: String
    """What is being synthesised, for the OS-voice fallback."""
    var tts_t0: Int
    var tts_queue: List[String]
    """Pieces of a reply waiting for the synthesiser (one synthesis at a time)."""
    var tts_ready: List[String]
    """Synthesised pieces waiting for the one playing to end: their WAV files."""
    var tts_ready_s: List[Float64]
    """Their lengths in seconds (`tts_no_play` plays them in silence that long)."""
    var play_until: Int
    """`tts_no_play`: when the piece "playing" ends (perf_counter_ns)."""
    var stream_raw: String
    """The reply's text streamed so far (`LLM.poll`)."""
    var stream_used: Int
    """Bytes of `stream_raw` already sent to be said."""
    var stream_parts: Int
    """Pieces of the current reply sent to be said."""
    var stream_cut: Bool
    """The caller said something over the streamed reply (`say`): the rest of
    the reply is not said."""
    var audio_wait: Bool
    """The reply's first audio has not started yet: the note says when it does."""
    var reply_t0: Int
    """When the reply model was asked (the first audio is timed from it)."""
    var stt: Self.STT
    var jev: Self.DEC
    var llm: Self.LLM
    var cfg: G1VoiceConfig
    var quest: JevQuestions
    var quest_mem: JevQuestions
    """`memory_route`: `quest` plus the memory reference, asked only when the
    log could settle "the other one" (otherwise its answer cannot route, and
    on a local model each question is a prompt of its own: ~300 ms on d1)."""
    var mem_asked: Bool
    """The current second request carries the memory reference."""
    var dest_q: JevQuestions
    """`request_wording` only: the first request. Empty otherwise."""
    var trail_q: JevQuestions
    """`triage`: the trail request. Empty otherwise."""
    var tri: G1Triage
    """The current decision's triage (the walk guard reads it in `_decide`)."""
    var trail_dest: String
    """The place the first request routed, for a trail answer `named`."""
    var chain_q: JevQuestions
    var chain_dest: String
    """The place the destination request routed, kept if the two-place
    request settles nothing."""
    var asked: String
    """The state the current decision was asked with, for its second
    request."""
    var dest_ms: Float64
    """`request_wording`: how long the first request of the current decision
    took, 0 when it was the only one. Shown in the decision's note."""
    var dest_tok: Int
    """The tokens that first request read (the note shows them with the time:
    a decision's wait is its tokens, docs/NN_MODEL_PORTS_PLAN.md §12.8)."""
    var then_decide: Bool
    """`talk_first`: a reply was started after the first request, and the
    commands request is asked as soon as it is out."""
    var replied: Bool
    """The current decision has replied already (`talk_first`): its commands'
    `talk` says nothing more."""
    var spec_q: JevQuestions
    var has_pool: Bool
    var pool: G1Pool
    var zbase: List[Float64]
    var znew: List[Float64]

    var state: Int
    var heard: String
    var heard_new: String
    """The utterance just sent for a decision, until `take_heard`."""
    var pick: G1LangPick
    var t_wait: Int
    var mute_until: Int
    var vad: G1Vad
    """The detector: levels, the pre-roll, the segment in hand (`g1_vad.mojo`)."""
    var seg_keep: List[Int16]
    """The utterance last sent to the recogniser, until `take_segment`."""
    var missed_s: Float64
    """Speech not listened to because the loop was busy, in seconds, until
    `take_missed`."""
    var reply_override: String
    """What to say INSTEAD of asking the reply model, when not empty
    (`set_reply_override`)."""
    var canned: String
    """A computed reply on its way out (one poll)."""
    var mic_on: Bool
    var mic_dead: String
    var checked_silence: Bool
    var t_built: Int
    """When the loop was built: the models have been loading since."""
    var wait_what: String
    """The backends still loading at the last poll, as printed ("" = none)."""
    var note: String
    """The last thing worth printing. The caller drains it with `take_note`,
    so this object never owns the log format."""
    var view: List[Picture]
    """The caller's latest frame (`see`): empty, or one picture."""
    var view_t: Int
    var view_due: Bool
    """A segment opened and no frame has come in since (`wants_view`)."""
    var prep_due: Bool
    """A segment opened: at the next poll the reply model is given the system
    prompt to compute while the person speaks (`ChatService.prepare`). The NEXT
    poll, so that a caller that writes the prompt from the robot's state
    (`set_reply_system`, set once the loop is busy) has written this moment's."""
    var seg_t: Int
    """When the current segment opened (a frame's age is measured to it)."""
    var asked_view: List[Picture]
    """The frame the current decision was asked with (empty: blind)."""
    var view_blind: Bool
    """The backend refused a picture once: asked blind from then on."""

    def __init__(out self, var cfg: G1VoiceConfig, ref bank: G1CommandBank) raises:
        """⚠ RAISES, unlike `poll`. Opening the microphone, reading a 70 MB
        pool and resolving API keys are all things that should stop a program
        BEFORE its loop starts rather than be swallowed inside one."""
        self.cfg = cfg^
        self.jev = Self.DEC.from_spec(self.cfg.dec_spec)
        # The backend reads its own spec. For the HTTP client, `local:<base_url>`
        # is any OpenAI-compatible server on this machine (Phonon-2's `phonon
        # serve`: English only, `language` is ignored, `prompt` is read as a
        # hotword list — at most 25 words).
        self.stt = Self.STT.from_spec(self.cfg.stt_spec)
        if self.cfg.lang != "":
            self.stt.set_language(self.cfg.lang)
        # ⚠ THE VOCABULARY GOES ONLY TO A BACKEND THAT TAKES ONE. HF's Whisper
        # accepts a prompt as token ids, never as text, and the client RAISES
        # rather than dropping it — which is right, because a silently dropped
        # hint and a hint that did not help are indistinguishable from outside
        # and that is exactly how "cours" was misdiagnosed (§12.56). So the
        # backend says whether it took it.
        if self.cfg.vocab != "":
            _ = self.stt.set_hint(self.cfg.vocab)
        self.llm = Self.LLM.from_spec(self.cfg.llm_spec)
        # ⚠ BOTH OF THESE ARE MEASURED AND BELONG WITH THE CLIENT. A reply is
        # one short sentence, so 120 tokens is generous — and Qwen THINKS
        # before answering unless told not to, which took a reply from
        # **10.8 s to 1.4 s** in the AI package's own tests. A robot that
        # pauses eleven seconds before saying hello is not having a
        # conversation, and a caller that had to remember these would
        # eventually not.
        self.llm.set_reply_limits(120)
        # ⚠ THE VOICE FOLLOWS THE REPLY'S LANGUAGE, not the system's — a
        # French Mac's default voice read the English demo with a French accent
        self.voice = LocalVoice(
            self.cfg.tts_voice if (self.cfg.tts_spec == ""
                                   and self.cfg.tts_voice != "")
            else os_voice_for(self.cfg.lang)
        )
        self.tts = None
        if self.cfg.tts_spec != "":
            var kv = self.cfg.tts_voice
            if kv == "":
                kv = String("ff_siwis") if self.cfg.lang == "fr" else String("af_heart")
            self.tts = Self.TTS.from_spec(self.cfg.tts_spec, kv)
        self.tts_busy = False
        self.tts_text = String("")
        self.tts_t0 = 0
        self.tts_queue = List[String]()
        self.tts_ready = List[String]()
        self.tts_ready_s = List[Float64]()
        self.play_until = 0
        self.stream_raw = String("")
        self.stream_used = 0
        self.stream_parts = 0
        self.stream_cut = False
        self.audio_wait = False
        self.reply_t0 = 0
        # ⚠ the capture is NOT opened here: `poll` opens it the first time it
        # runs (the `mic` field's note). A device that cannot be opened is
        # reported there, in `mic_dead` and the note, as a dead one is.
        self.mic = None
        self.vad = G1Vad(self.cfg.hang_s, self.cfg.min_seg, self.cfg.max_seg)
        self.dest_q = JevQuestions()
        self.quest_mem = JevQuestions()
        self.mem_asked = False
        self.trail_q = JevQuestions()
        self.tri = G1Triage()
        self.trail_dest = String("")
        self.chain_q = JevQuestions()
        self.chain_dest = String("")
        if self.cfg.request_wording and len(self.cfg.destinations) > 0:
            # two requests: the destination first, then the commands WITHOUT it
            self.dest_q = g1_destination_questions(self.cfg.destinations)
            if self.cfg.triage != G1_TRIAGE_OFF:
                g1_add_triage(self.dest_q, self.cfg.name_desc)
                self.trail_q = g1_trail_questions(self.cfg.destinations, self.cfg.name_desc)
            if self.cfg.dest_chain:
                g1_add_two_places_gate(self.dest_q)
                self.chain_q = g1_two_places_questions(self.cfg.destinations, self.cfg.name_desc)
            self.quest = g1_command_questions(
                bank, True, List[String](), List[String](), self.cfg.chain,
                name_desc=self.cfg.name_desc,
            )
            if self.cfg.memory_route:
                self.quest_mem = g1_command_questions(
                    bank, True, List[String](), List[String](), self.cfg.chain,
                    name_desc=self.cfg.name_desc,
                )
                g1_add_memory_reference(self.quest_mem)
        else:
            self.quest = g1_command_questions(
                bank, True, self.cfg.destinations, self.cfg.dest_descs,
                self.cfg.chain,
            )
        self.asked = String("")
        self.dest_ms = 0.0
        self.dest_tok = 0
        self.then_decide = False
        self.replied = False
        self.spec_q = g1_spec_questions(True)
        if self.cfg.miss_kind:
            g1_add_miss_kind(self.spec_q, self.cfg.name_desc)
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
        self.heard_new = String("")
        self.pick = G1LangPick()
        self.t_wait = perf_counter_ns()
        self.mute_until = perf_counter_ns()
        self.seg_keep = List[Int16]()
        self.missed_s = 0.0
        self.reply_override = String("")
        self.canned = String("")
        self.mic_on = True
        self.mic_dead = String("")
        self.checked_silence = False
        self.t_built = perf_counter_ns()
        self.wait_what = String("")
        self.note = String("")
        self.view = List[Picture]()
        self.view_t = 0
        self.view_due = False
        self.prep_due = False
        self.seg_t = 0
        self.asked_view = List[Picture]()
        self.view_blind = False

    def __init__(out self, *, deinit move: Self):
        self.mic = move.mic^
        self.voice = move.voice^
        self.tts = move.tts^
        self.tts_busy = move.tts_busy
        self.tts_text = move.tts_text^
        self.tts_t0 = move.tts_t0
        self.tts_queue = move.tts_queue^
        self.tts_ready = move.tts_ready^
        self.tts_ready_s = move.tts_ready_s^
        self.play_until = move.play_until
        self.stream_raw = move.stream_raw^
        self.stream_used = move.stream_used
        self.stream_parts = move.stream_parts
        self.stream_cut = move.stream_cut
        self.audio_wait = move.audio_wait
        self.reply_t0 = move.reply_t0
        self.stt = move.stt^
        self.jev = move.jev^
        self.llm = move.llm^
        self.cfg = move.cfg^
        self.quest = move.quest^
        self.quest_mem = move.quest_mem^
        self.mem_asked = move.mem_asked
        self.dest_q = move.dest_q^
        self.trail_q = move.trail_q^
        self.tri = move.tri^
        self.trail_dest = move.trail_dest^
        self.chain_q = move.chain_q^
        self.chain_dest = move.chain_dest^
        self.asked = move.asked^
        self.dest_ms = move.dest_ms
        self.dest_tok = move.dest_tok
        self.then_decide = move.then_decide
        self.replied = move.replied
        self.spec_q = move.spec_q^
        self.has_pool = move.has_pool
        self.pool = move.pool^
        self.zbase = move.zbase^
        self.znew = move.znew^
        self.state = move.state
        self.heard = move.heard^
        self.heard_new = move.heard_new^
        self.pick = move.pick^
        self.t_wait = move.t_wait
        self.mute_until = move.mute_until
        self.vad = move.vad^
        self.seg_keep = move.seg_keep^
        self.missed_s = move.missed_s
        self.reply_override = move.reply_override^
        self.canned = move.canned^
        self.mic_on = move.mic_on
        self.mic_dead = move.mic_dead^
        self.checked_silence = move.checked_silence
        self.t_built = move.t_built
        self.wait_what = move.wait_what^
        self.note = move.note^
        self.view = move.view^
        self.view_t = move.view_t
        self.view_due = move.view_due
        self.prep_due = move.prep_due
        self.seg_t = move.seg_t
        self.asked_view = move.asked_view^
        self.view_blind = move.view_blind

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
            if self.mic:
                self.mic.value().stop()
        except:
            pass
        self._tts_cancel()
        try:
            self.voice.stop()
        except:
            pass

    def say(mut self, text: String):
        """Speak, and never raise. ⚠ THE ONLY WAY THE CALLER SHOULD SPEAK.
        The echo gate reads `voice.speaking()`, so a caller that owned its own
        TTS would reopen §12.56's race — two voices and one "done".
        ⚠ IT ENDS A STREAMED REPLY: its sentences still to come are not said."""
        self.stream_cut = True
        self._speak(text, False)

    def _speak(mut self, text: String, reply: Bool):
        """`say`; `reply` = the first piece of the reply being timed (the note
        says when its audio starts), anything else stops that timing."""
        # ⚠ ONE UTTERANCE AT A TIME, as `LocalVoice` says: a new reply cuts
        # off the one playing AND the one still being synthesised, AND the
        # pieces of a streamed reply still waiting
        self._tts_cancel()
        if not reply:
            self.audio_wait = False
        try:
            self.voice.stop()
        except:
            pass
        if self.tts:
            self.tts_queue.append(text)
            self._tts_next()
            if self.tts_busy:
                self.mute_until = perf_counter_ns() + _vl_ns(VL_ECHO_TAIL_S)
                return
            self.tts_queue = List[String]()
        try:
            self.voice.say(text)
        except:
            pass
        self.mute_until = perf_counter_ns() + _vl_ns(VL_ECHO_TAIL_S)

    def say_more(mut self, text: String):
        """The next piece of what is being said (a streamed reply's next
        sentence): synthesised as soon as the synthesiser is free, played when
        the piece before it has ended. Cuts nothing off. With no synthesiser
        it is said by the OS voice when nothing is being said, else dropped
        with a note: the OS voice says one utterance at a time (a caller
        streams only with a synthesiser)."""
        if not self.tts:
            if not self.speaking():
                try:
                    self.voice.say(text)
                except:
                    pass
            else:
                self._add_note(String("[tts] no synthesiser: a piece not said: ") + text)
            return
        self.tts_queue.append(text)
        if not self.tts_busy:
            self._tts_next()

    def _tts_next(mut self):
        """Start synthesising the next queued piece. Never raises."""
        while len(self.tts_queue) > 0 and not self.tts_busy:
            var text = self.tts_queue.pop(0)
            try:
                self.tts.value().start(text)
                self.tts_busy = True
                self.tts_text = text
                self.tts_t0 = perf_counter_ns()
            except e:
                self.note = String("[tts] could not start: ") + String(e)

    def _tts_cancel(mut self):
        """Drop everything not yet heard: the synthesis in flight, the pieces
        waiting for it and the ones waiting to play."""
        if self.tts_busy:
            try:
                self.tts.value().cancel()
            except:
                pass
            self.tts_busy = False
        self.tts_queue = List[String]()
        for p in self.tts_ready:
            if p.byte_length() > 0:
                try:
                    remove_file(p)
                except:
                    pass
        self.tts_ready = List[String]()
        self.tts_ready_s = List[Float64]()
        self.play_until = 0

    def _playing(mut self) -> Bool:
        """A piece is playing (or, `tts_no_play`, would be)."""
        if self.cfg.tts_no_play:
            return perf_counter_ns() < self.play_until
        try:
            return self.voice.speaking()
        except:
            return True

    def _tts_fallback(mut self, why: String):
        """The synthesis in flight failed or was late: the OS voice says it if
        nothing else is being said; otherwise the piece is dropped (two voices
        at once would be worse than a missing sentence)."""
        self.note = why
        if len(self.tts_ready) == 0 and not self._playing():
            try:
                self.voice.say(self.tts_text)
            except:
                pass
        else:
            self._add_note(String("  (a piece not said)"))

    def _tts_step(mut self):
        """The voice, once a frame. Never raises: (1) the synthesis in flight —
        its WAV queued to play once in, or the OS voice when it fails or misses
        `tts_dl`; (2) the next piece to synthesise; (3) the next piece to play,
        once the one before it has ended."""
        if self.tts_busy:
            var el = Float64(perf_counter_ns() - self.tts_t0) / 1e9
            if el > self.cfg.tts_dl:
                try:
                    self.tts.value().cancel()
                except:
                    pass
                self.tts_busy = False
                self._tts_fallback(String("[tts] GAVE UP after ") + _vl_f2(el) + String(" s — OS voice"))
            else:
                var ready: Bool
                try:
                    ready = self.tts.value().poll()
                except:
                    ready = True
                if ready:
                    self.tts_busy = False
                    try:
                        var wav = self.tts.value().result_wav_bytes()
                        var path = String("")
                        if not self.cfg.tts_no_play:
                            path = (String("/tmp/noeira_tts_") + String(perf_counter_ns()) + String(".wav"))
                            write_file_atomic(path, wav)
                        self.tts_ready.append(path)
                        self.tts_ready_s.append(_vl_wav_seconds(wav))
                        self._add_note(String("[tts] ") + String(Int(el * 1000.0)) + String(" ms"))
                    except e:
                        self._tts_fallback(String("[tts] FAILED: ") + String(e) + String(" — OS voice"))
        if not self.tts_busy and len(self.tts_queue) > 0:
            self._tts_next()
        if len(self.tts_ready) > 0 and not self._playing():
            var path = self.tts_ready.pop(0)
            var secs = self.tts_ready_s.pop(0)
            if self.cfg.tts_no_play:
                self.play_until = perf_counter_ns() + Int(secs * 1e9)
            else:
                try:
                    self.voice.play(path)
                except e:
                    self._add_note(String("[tts] could not play: ") + String(e))
                self.mute_until = perf_counter_ns() + _vl_ns(VL_ECHO_TAIL_S)
            if self.audio_wait:
                self.audio_wait = False
                self._add_note(String("[voice] first audio ") + String(Int(Float64(perf_counter_ns() - self.reply_t0) / 1e6))
                               + String(" ms after the reply began (") + String(Int(secs * 1000.0)) + String(" ms of audio)"))

    def force(mut self):
        """TAB: end the current segment now, or start one. For a room too loud
        for the detector to close."""
        if self.state == VL_ST_IDLE:
            self.heard = String("")
            self.vad.open_now()
            self.state = VL_ST_REC
        elif self.state == VL_ST_REC:
            self.vad.forced = True

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
                if self.mic:
                    self.mic.value().stop()
            except:
                pass
            self.mic = None
            self.mic_on = False
            self.vad.restart()
            self.state = VL_ST_IDLE
            return String("[mic] closed")
        try:
            self.mic = Optional(MicCapture.start(16000, self.cfg.mic_dev, self.cfg.mic_input))
            self.mic_on = True
            self.mic_dead = String("")
            # ⚠ and the detector starts over: a floor learned before the gap
            # describes a room that may have changed, and `checked_silence`
            # must re-run or a newly-muted device would never be reported.
            self.checked_silence = False
            self.vad.restart()
            return String("[mic] reopened")
        except e:
            self.mic_dead = String(e)
            return String("[mic] ") + self.mic_dead

    def see(mut self, var pic: Picture):
        """The robot's latest frame (RGB24, sides multiples of 32 for d1's
        tower). Kept until the next one; read when a decision starts."""
        self.view = List[Picture]()
        self.view.append(pic^)
        self.view_t = perf_counter_ns()
        self.view_due = False

    def wants_view(self) -> Bool:
        """True from the moment a segment opens until a frame comes in
        (`see`): render one then, not every tick. False without `cfg.see`."""
        return self.cfg.see and self.view_due and not self.view_blind

    def set_reply_system(mut self, var text: String):
        """The reply model's system prompt from here on (`cfg.chat_sys`).
        For a caller whose prompt follows the robot's state
        (`noeira.brain.reply_view`): set it before `poll`, each frame or when
        the state changes. A reply already started keeps the one it had."""
        self.cfg.chat_sys = text^

    def set_reply_override(mut self, var text: String):
        """A reply the caller has COMPUTED for the utterance being handled: when
        a reply is due and this is not empty, it is said as it is and the reply
        model is not asked. "" (the default) = the model replies.

        For an answer that must be exact (`noeira.brain.answers.computed_reply`):
        who the speaker is, whether the robot has finished what it was asked.
        Told to say that sentence in its prompt, the reply model still answered
        « Je m'appelle Noé » when the question came through the no-command
        route and its hint. Set it before `poll`, each frame, as
        `set_reply_system`."""
        self.reply_override = text^

    def take_segment(mut self) -> List[Int16]:
        """The samples (16 kHz mono) of the utterance last sent to the
        recogniser, consumed; empty when none is waiting. For a caller that
        wants the VOICE as well as the words (`noeira.audio.speaker`,
        `noeira.brain.people`): take it when `take_heard` gives the words."""
        var s = self.seg_keep^
        self.seg_keep = List[Int16]()
        return s^

    def take_heard(mut self) -> String:
        """The utterance just sent for a decision, consumed; "" when none.
        For `G1Context.heard_said`: logged after its own state was built, so
        `recently_heard` never holds the sentence being decided."""
        var h = self.heard_new
        self.heard_new = String("")
        return h

    def take_missed(mut self) -> Float64:
        """Seconds of speech that began while the loop was busy and were not
        listened to, since the last call; 0 when none. For the caller's
        record (`noeira.brain.WorldState.missed`): the `[vad]` note says the
        same in the console, where nobody counts it."""
        var s = self.missed_s
        self.missed_s = 0.0
        return s

    def take_note(mut self) -> String:
        """The last log line, consumed. Empty when there is nothing new."""
        var n = self.note
        self.note = String("")
        return n

    def is_busy(self) -> Bool:
        """True while a transcription, decision or reply is in flight — the
        caller may want to suppress a keyboard command that would race it."""
        return self.state != VL_ST_IDLE

    def loading(self) -> String:
        """The backends still loading or warming up, by role ("" when none).

        ⚠ THE MICROPHONE STAYS CLOSED UNTIL THIS IS EMPTY, and that was the
        other half of the "lost utterance" (`g1_vad.mojo`'s note). The models
        load on their own threads while the loop already runs. An utterance
        spoken then is queued behind the load: its transcription came back
        6 s after it ended where the recogniser itself takes 80 ms, the loop
        was still deciding it when the next one arrived, and that one was
        lost. A robot that cannot answer yet should not be listening yet:
        the first `[voice] listening` line is when it can."""
        var w = String("")
        if self.stt.loading():
            w += " hear"
        if self.jev.loading():
            w += " decide"
        if self.llm.loading():
            w += " reply"
        if self.tts and self.tts.value().loading():
            w += " speak"
        return w^

    def failed(self) -> String:
        """The backends whose model failed to load, by role ("" when none).
        ⚠ `loading` is empty for them too: without this the loop said "every
        model ready" with no d1, and every decision then failed one by one."""
        var w = String("")
        if self.stt.load_failed():
            w += " hear"
        if self.jev.load_failed():
            w += " decide"
        if self.llm.load_failed():
            w += " reply"
        if self.tts and self.tts.value().load_failed():
            w += " speak"
        return w^

    def speaking(mut self) -> Bool:
        """⚠ A GATE THAT RAISES WOULD TAKE THE LOOP DOWN, and the safe default
        is to assume we ARE talking rather than open the microphone onto our
        own voice. A piece being synthesised or waiting to play counts: the
        robot is about to say it. `tts_no_play`: never (nothing is heard)."""
        if self.cfg.tts_no_play:
            return False
        if self.tts_busy or len(self.tts_queue) > 0 or len(self.tts_ready) > 0:
            return True
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
        self._tts_step()

        # ── the reply's prompt, computed while the person speaks ───────
        # ⚠ ONLY WHILE A SEGMENT IS BEING RECORDED: the GPU has no decision
        # and no reply to run then. Beside a decision it would only make the
        # decision later.
        if self.prep_due:
            self.prep_due = False
            if self.state == VL_ST_REC:
                try:
                    self.llm.prepare(self.cfg.chat_sys)
                except:
                    pass    # the prompt is then computed with its reply, as before

        # ── the microphone, every frame ───────────────────────────────
        # ⚠ READ EVEN WHILE THE ROBOT SPEAKS. The pipe is 64 KiB, about 2 s,
        # and past that ffmpeg blocks and the device drops audio. The samples
        # are read and DISCARDED, never ringed, never levelled.
        if self.speaking():
            self.mute_until = perf_counter_ns() + _vl_ns(VL_ECHO_TAIL_S)
        var echoing = perf_counter_ns() < self.mute_until
        # ⚠ THE STATE AT THE START OF THIS POLL: a segment that ends below
        # starts its transcription, whose own branch must wait for the next
        # poll (the chain after this block was `if REC ... elif STT ...`).
        var was_rec = self.state == VL_ST_REC
        var waiting = String("")
        if self.mic_on and self.mic_dead == "" and not self.mic:
            waiting = self.loading()
            if waiting != self.wait_what:
                # one line each time a backend comes in: the log then says how
                # long each took, which is what was missing to see this fault
                var el = _vl_f2(Float64(perf_counter_ns() - self.t_built) / 1e9)
                var broken = self.failed()
                if waiting == "" and broken != "":
                    self.note = (String("[voice] listening ") + el + String(" s after the loop was built — FAILED TO LOAD:")
                                 + broken + String(" (each call to it will fail and say why)"))
                elif waiting == "":
                    self.note = String("[voice] listening — every model ready ") + el + String(" s after the loop was built")
                else:
                    self.note = (String("[voice] not listening yet (") + el + String(" s) — still loading:") + waiting)
                self.wait_what = waiting.copy()
        if waiting != "":
            pass    # the microphone opens when the models can answer (`loading`)
        elif self.mic_on and self.mic_dead == "":
            try:
                if not self.mic:
                    # opened here, the first time the loop runs with its
                    # models loaded (the field's note, and `loading`)
                    self.mic = Optional(MicCapture.start(16000, self.cfg.mic_dev, self.cfg.mic_input))
                var pcm = self.mic.value().read()
                var secs = self.mic.value().seconds_read()
                var ve: G1VadEvent
                if echoing:
                    # ⚠ AND THE PRE-ROLL GOES WITH THEM (`G1Vad.skip`): it would
                    # hand the recogniser the tail of the robot's own sentence
                    # as the first syllable of yours.
                    ve = self.vad.skip(len(pcm))
                else:
                    # ⚠ VAD OPENS ONLY FROM IDLE. While a transcription or a
                    # decision is in flight, speech is still ringed but starts
                    # nothing — one call per client, and a queue of half-heard
                    # commands is worse than a missed one. The 1.2 s guard
                    # keeps the floor estimate from opening a segment on its
                    # own first samples.
                    var may_open = self.state == VL_ST_IDLE and self.cfg.vad and secs > 1.2
                    # ⚠ and what is spoken meanwhile is REPORTED, and never
                    # opened on its tail when the loop comes back mid-sentence
                    # (`G1Vad.feed`'s `busy`).
                    var busy = self.cfg.vad and self.state != VL_ST_IDLE and self.state != VL_ST_REC
                    ve = self.vad.feed(pcm, may_open, busy)
                if not self.checked_silence and secs > 1.0:
                    self.checked_silence = True
                    if self.mic.value().digital_silence(0.5):
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
                            + _vl_f2(secs)
                            + String(" s read, floor ") + _vl_f4(self.vad.floor)
                            + String(" peak ") + _vl_f4(self.vad.peak)
                        )
                # ── what the detector found in this poll's audio ──────
                if ve.opened and self.state == VL_ST_IDLE:
                    self.heard = String("")
                    self.pick = G1LangPick()
                    self.view_due = True
                    self.prep_due = True
                    self.seg_t = perf_counter_ns()
                    self.state = VL_ST_REC
                if ve.note != "":
                    self.note = ve.note
                self.missed_s += ve.missed_s
                if self.state == VL_ST_REC:
                    if ve.done:
                        try:
                            var audio = WavAudio(16000, 1, self.vad.seg.copy())
                            self.seg_keep = self.vad.seg.copy()
                            self.stt.start(audio)
                            self.t_wait = perf_counter_ns()
                            self.note = (String("[stt] ") + String(len(self.vad.seg))
                                         + String(" samples (") + _vl_f2(ve.seg_s)
                                         + String(" s ), peak ")
                                         + _vl_f4(self.vad.peak)
                                         + (String(" — max-seg") if ve.over
                                            else String("")))
                            self.state = VL_ST_STT
                        except e:
                            self.note = String("[stt] could not start: ") + String(e)
                            self.state = VL_ST_IDLE
                    elif not self.vad.recording:
                        self.state = VL_ST_IDLE     # the segment was dropped (its note says why)
            except e:
                self.mic_dead = String(e)
                self.note = String("[mic] ") + self.mic_dead
                if self.state == VL_ST_REC:
                    self.state = VL_ST_IDLE

        # ── transcribing ──────────────────────────────────────────────
        if was_rec:
            pass    # recording is handled with the microphone, above
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
                var ready: Bool
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
                            self._snapshot_view()
                            self.heard_new = self.heard
                            self.mem_asked = False
                            self.tri = G1Triage()
                            self.dest_ms = 0.0
                            self.dest_tok = 0
                            self.then_decide = False
                            self.replied = False
                            if self._two_requests():
                                self._ask(_VL_Q_DEST)
                                self.state = VL_ST_DEST
                            else:
                                self._ask(_VL_Q_CMD)
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
                var ready: Bool
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._destination(ctx)

        # ── deciding: the trail request (`triage`, "back to before X") ─
        elif self.state == VL_ST_TRAIL:
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
                var ready: Bool
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._trail(ctx)

        # ── deciding: which two places (`dest_chain`) ─────────────────
        elif self.state == VL_ST_CHAIN:
            var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
            if el > self.cfg.jev_dl:
                try:
                    self.jev.cancel()
                except:
                    pass
                ev = self._chain(True)
            else:
                var ready: Bool
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._chain(False)

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
                var ready: Bool
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
                var ready: Bool
                try:
                    ready = self.jev.poll()
                except:
                    ready = True
                if ready:
                    ev = self._spec(bank)

        # ── a computed reply (`set_reply_override`): out at once ───────
        elif self.state == VL_ST_CHAT and self.canned != "":
            var ctxt = spoken_text(self.canned)
            self.canned = String("")
            self._add_note(String("[said] ") + ctxt + String("  (computed)"))
            if not self.cfg.mute:
                self.say(ctxt)
            ev.kind = VL_TALK
            ev.text = ctxt
            self.state = VL_ST_IDLE
            self._after_reply()

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
                self._after_reply()
            else:
                var delta = String("")
                try:
                    delta = self.llm.poll()
                except:
                    pass
                # L2: the reply said sentence by sentence while it is written (`stream_reply`, a synthesiser)
                var streaming = self.cfg.stream_reply and Bool(self.tts) and not self.cfg.mute
                if streaming and delta.byte_length() > 0:
                    self.stream_raw += delta
                    self._say_sentences()
                var done = self.llm.done()
                if done:
                    var rtxt = String("")
                    var rtxt_raw = String("")
                    var rms_ = 0.0
                    var rcost = String("")
                    try:
                        var rep = self.llm.result()
                        # what a voice can say: no emoji, no markup (`noeira.text.spoken`)
                        rtxt = spoken_text(rep.text)
                        rtxt_raw = rep.text
                        rms_ = rep.latency_ms
                        if rep.input_tokens > 0:
                            # where the reply's time went: its first token (the prompt), then its tokens
                            rcost = (String("; first token ") + String(Int(rep.first_token_ms)) + String(" ms, ")
                                     + String(rep.input_tokens) + String(" tokens in, ") + String(rep.cached_tokens)
                                     + String(" cached, ") + String(rep.output_tokens) + String(" out"))
                    except e:
                        self.note = String("[talk] FAILED: ") + String(e)
                    if rtxt != "":
                        var pieces = String("")
                        if not self.cfg.mute:
                            if streaming and self.stream_parts > 0:
                                # what the stream had not finished as a sentence: the end of the answer
                                var rest = self._stream_rest(rtxt_raw)
                                if rest != "":
                                    self._say_part(rest)
                                pieces = String("; said in ") + String(self.stream_parts) + String(" pieces")
                            else:
                                self._speak(rtxt, True)
                        self._add_note(String("[said] ") + rtxt + String("  (reply ") + String(Int(rms_)) + String(" ms")
                                       + rcost + pieces + String(")"))
                        ev.kind = VL_TALK
                        ev.text = rtxt
                    self.state = VL_ST_IDLE
                    self._after_reply()

        return ev^

    def _snapshot_view(mut self):
        """The frame this decision is asked with: the latest, if `cfg.see`
        and it is fresh enough; otherwise none (and the note says why)."""
        self.asked_view = List[Picture]()
        if not self.cfg.see or self.view_blind:
            return
        if len(self.view) == 0:
            self._add_note(String("[see] no frame yet — asking blind"))
            return
        var age = Float64(self.seg_t - self.view_t) / 1e9
        if not g1_view_fresh(self.seg_t, self.view_t, self.cfg.see_max_age_s):
            self._add_note(String("[see] the frame is ") + _vl_f2(age) + String(" s older than the utterance — asking blind"))
            return
        self.asked_view.append(self.view[0].copy())

    def _add_note(mut self, msg: String):
        """`msg` after whatever this poll noted already (the caller prints
        `take_note` once a frame)."""
        if self.note == "":
            self.note = msg
        else:
            self.note += String("\n  ") + msg

    def _ask(mut self, which: Int) raises:
        """Start one request of the current decision (`_VL_Q_*`: which question
        set — by number, as `self`'s own fields cannot be passed beside a
        mutable `self`), with its frame if it has one. A backend that cannot
        see is asked blind, now and from then on."""
        if len(self.asked_view) > 0:
            try:
                if which == _VL_Q_DEST:
                    self.jev.start_seeing(self.asked, True, self.dest_q, self.asked_view)
                elif which == _VL_Q_TRAIL:
                    self.jev.start_seeing(self.asked, True, self.trail_q, self.asked_view)
                elif which == _VL_Q_MEM:
                    self.jev.start_seeing(self.asked, True, self.quest_mem, self.asked_view)
                else:
                    self.jev.start_seeing(self.asked, True, self.quest, self.asked_view)
                return
            except e:
                self._add_note(String("[see] the decision backend cannot see (") + String(e) + String(") — blind from now on"))
                self.view_blind = True
                self.asked_view = List[Picture]()
        if which == _VL_Q_DEST:
            self.jev.start(self.asked, self.dest_q)
        elif which == _VL_Q_TRAIL:
            self.jev.start(self.asked, self.trail_q)
        elif which == _VL_Q_MEM:
            self.jev.start(self.asked, self.quest_mem)
        else:
            self.jev.start(self.asked, self.quest)

    def _two_requests(self) -> Bool:
        return self.cfg.request_wording and len(self.cfg.destinations) > 0

    def _destination(mut self, ref ctx: G1Context) -> G1VoiceEvent:
        """The first request's answer: a goto, or the commands' request."""
        var ev = G1VoiceEvent.none()
        self.state = VL_ST_IDLE
        try:
            var ans = self.jev.result()
            var dest = g1_route_destination(ans)
            if self.cfg.triage != G1_TRIAGE_OFF:
                # ⚠ THE TRIAGE OVERRIDES A CONFIDENT DESTINATION: "climb onto the
                # table" routes the table, and it is the kind that says no
                self.tri = g1_route_triage(ans, self.cfg.triage)
                if self.tri.talks():
                    self._reply(
                        _vl_triage_hint(self.tri.kind),
                        String("[triage] ") + self.tri.kind + String(" ")
                        + _vl_f2(self.tri.p) + String(" -> reply, no motion")
                        + self._jev_ms(ans.latency_ms, ans.input_tokens),
                    )
                    return ev^
                if self.tri.kind == "remembered":
                    self.dest_ms = ans.latency_ms
                    self.dest_tok = ans.input_tokens
                    self.trail_dest = dest
                    self._ask(_VL_Q_TRAIL)
                    self.t_wait = perf_counter_ns()
                    self.state = VL_ST_TRAIL
                    return ev^
                if self.tri.kind == "act_here":
                    dest = String("")   # a step or a turn is not a place
            if dest != "":
                if self.cfg.dest_chain and ans.noul(String(G1_Q_IS_TWO_PLACES)) >= 0.5:
                    # two places asked for: which two, in which order, is a request of
                    # its own, against a state with nothing of the room in it
                    self.chain_dest = dest
                    self.dest_ms = ans.latency_ms
                    self.dest_tok = ans.input_tokens
                    self.jev.start(g1_two_places_state(self.heard), self.chain_q)
                    self.t_wait = perf_counter_ns()
                    self.state = VL_ST_CHAIN
                    return ev^
                ev.kind = VL_WORLD
                ev.destination = dest
                ev.text = self.heard
                self.note = (String("[world] ") + dest + String(" ")
                             + _vl_f2(ans.probability(String("destination"), dest))
                             + self._jev_ms(ans.latency_ms, ans.input_tokens))
                return ev^
            # ⚠ THE SAME STATE, not a fresh one: the robot may have moved on
            # while the first request was answered, and the two answers must
            # be about the same moment.
            self.mem_asked = self._settles_other_one(ctx)
            # ⚠ NOT WHEN CODE HOLDS THE ANSWER (`set_reply_override`): « Comment je m'appelle ? » was
            # answered « Un instant, je n'ai pas encore reconnu ta voix » 0.5 s in, the voice check still
            # running (e2e_b4_s1). A computed answer costs no model time: it waits for the decision, as before.
            if (self.cfg.talk_first > 0.0 and self.tri.top == "talk"
                    and self.tri.top_p >= self.cfg.talk_first and self.reply_override == ""):
                # the reply first, the commands request once it is out (the
                # field's note: that request is never skipped)
                var tnote = (String("[talk-first] talk ") + _vl_f2(self.tri.top_p)
                             + String(" -> reply now, the commands after it")
                             + self._jev_ms(ans.latency_ms, ans.input_tokens))
                self.dest_ms = ans.latency_ms
                self.dest_tok = ans.input_tokens
                self.then_decide = True
                self._reply(String(""), tnote)
                if self.state != VL_ST_CHAT:
                    self._after_reply()     # no reply could start: the commands now
                return ev^
            if self.cfg.talk_first > 0.0 and self.tri.top == "talk":
                # not taken: said, so a log shows where the bar sits against the room's sentences
                if self.reply_override != "":
                    self._add_note(String("[talk-first] talk ") + _vl_f2(self.tri.top_p)
                                   + String(", a computed answer: the commands first"))
                else:
                    self._add_note(String("[talk-first] talk ") + _vl_f2(self.tri.top_p) + String(" < ")
                                   + _vl_f2(self.cfg.talk_first) + String(": the commands first"))
            self.dest_ms = ans.latency_ms
            self.dest_tok = ans.input_tokens
            if self.mem_asked:
                self._ask(_VL_Q_MEM)
            else:
                self._ask(_VL_Q_CMD)
            self.t_wait = perf_counter_ns()
            self.state = VL_ST_JEV
        except e:
            self.note = String("[jev] FAILED: ") + String(e)
        return ev^

    def _chain(mut self, timed_out: Bool) -> G1VoiceEvent:
        """The two-place request's answer: the routed place and the one to go
        on to. ⚠ A request that fails, is late or settles nothing still walks:
        the destination request had routed a place, and that one is kept."""
        var ev = G1VoiceEvent.none()
        self.state = VL_ST_IDLE
        var dest = self.chain_dest
        var then = String("")
        var tail = String("")
        if timed_out:
            tail = String("  (the two-place request was late: one place)")
        else:
            try:
                var ans = self.jev.result()
                var pair = g1_route_two_places(ans)
                if pair[0] != "":
                    dest = pair[0]
                    then = pair[1]
                tail = self._jev_ms(ans.latency_ms, ans.input_tokens)
            except e:
                tail = String("  (the two-place request failed: one place) ") + String(e)
        ev.kind = VL_WORLD
        ev.destination = dest
        ev.text = self.heard
        if then != "":
            ev.chain.append(then.copy())
        self.note = (String("[world] ") + dest
                     + (String(", then ") + then if then != "" else String(" (one place)")) + tail)
        return ev^

    def _trail(mut self, ref ctx: G1Context) -> G1VoiceEvent:
        """The trail request's answer: the place the log settles, or a reply
        asking which place (never a guess)."""
        var ev = G1VoiceEvent.none()
        self.state = VL_ST_IDLE
        try:
            var ans = self.jev.result()
            var rel = ans.choice(String(G1_Q_TRAIL_REL))
            var anc = ans.choice(String(G1_Q_TRAIL_ANCHOR))
            var place = g1_resolve_trail(rel, anc, ctx)
            if rel == "named" and self.trail_dest != "":
                place = self.trail_dest
            if place != "":
                ev.kind = VL_WORLD
                ev.destination = place
                ev.text = self.heard
                self.note = (String("[trail] ") + rel + String(" (") + anc
                             + String(") -> ") + place + self._jev_ms(ans.latency_ms, ans.input_tokens))
            else:
                self._reply(
                    _vl_triage_hint(String("unsettled")),
                    String("[trail] ") + rel + String(" (") + anc
                    + String(") not settled by the log -> ask")
                    + self._jev_ms(ans.latency_ms, ans.input_tokens),
                )
        except e:
            self.note = String("[jev] FAILED: ") + String(e)
        return ev^

    def _say_part(mut self, piece: String):
        """One piece of the reply being written: the first cuts off whatever is
        being said (as `say` does), the next ones follow it. None once the
        caller said something over the reply (`stream_cut`)."""
        if self.stream_cut:
            return
        if self.stream_parts == 0:
            self._speak(piece, True)
        else:
            self.say_more(piece)
        self.stream_parts += 1

    def _say_sentences(mut self):
        """Every sentence the stream has finished since the last call goes to
        the voice (`spoken_sentence_end`, `spoken_text`)."""
        while True:
            var e = spoken_sentence_end(self.stream_raw, self.stream_used)
            if e < 0:
                return
            var piece = spoken_text(String(self.stream_raw[byte = self.stream_used : e]))
            self.stream_used = e
            if piece.byte_length() > 0:
                self._say_part(piece)

    def _stream_rest(mut self, answer: String) -> String:
        """What of `answer` (the reply model's whole answer, `result().text`)
        the streamed sentences did not say. ⚠ THE ANSWER IS THE AUTHORITY: when
        it does not start with what was streamed, nothing more is said and the
        note says so (half a reply heard twice would be worse)."""
        var said = String(String(self.stream_raw[byte = 0 : self.stream_used]).strip())
        var a = String(answer.strip())
        if not a.startswith(said):
            self._add_note(String("[talk] the streamed text is not the answer's start: the rest is not said"))
            return String("")
        return spoken_text(String(a[byte = said.byte_length() :]))

    def _reply(mut self, hint: String, note: String):
        """A spoken reply and nothing else: the robot's motion is unchanged.
        `hint` is added to the reply's system prompt ("" = a plain reply).

        ⚠ ONE REPLY A DECISION: after a `talk_first` reply, the commands request
        can still end in one (the miss route's "a question -> reply": the
        e2e replays said two jokes for « raconte-moi une blague »). It is not
        started; the note says so."""
        if self.replied:
            self.note = note + String("  (replied already: nothing more said)")
            self.state = VL_ST_IDLE
            return
        if self.reply_override != "":
            # a computed reply: said as it is, the model is not asked
            self.canned = self.reply_override
            self.t_wait = perf_counter_ns()
            self.note = note
            self.state = VL_ST_CHAT
            return
        try:
            var msgs = List[ChatMessage]()
            msgs.append(ChatMessage.user(self.heard))
            var sys = self.cfg.chat_sys
            if hint != "":
                sys += String(" ") + hint
            self.llm.start(msgs, sys, List[ToolSpec](), False)
            self.t_wait = perf_counter_ns()
            self.reply_t0 = self.t_wait
            self.audio_wait = not self.cfg.mute
            self.stream_raw = String("")
            self.stream_used = 0
            self.stream_parts = 0
            self.stream_cut = False
            self.note = note
            self.state = VL_ST_CHAT
        except e:
            self.note = String("[talk] could not start: ") + String(e)
            self.state = VL_ST_IDLE

    def _after_reply(mut self):
        """`talk_first`: the reply is out, or could not be had. The commands
        request of the same decision is asked now, with the state the first
        request was asked with (`asked`). Nothing when no reply came first."""
        if not self.then_decide:
            return
        self.then_decide = False
        self.replied = True
        try:
            self._ask(_VL_Q_MEM if self.mem_asked else _VL_Q_CMD)
            self.t_wait = perf_counter_ns()
            self.state = VL_ST_JEV
        except e:
            self._add_note(String("[jev] could not start: ") + String(e))
            self.state = VL_ST_IDLE

    def _settles_other_one(self, ref ctx: G1Context) -> Bool:
        """Whether the log names a place "the other one" could mean right now."""
        return self.cfg.memory_route and g1_other_one(
            ctx, self.cfg.destinations, self.cfg.dest_words
        ) != ""

    def _jev_ms(self, ms: Float64, tokens: Int = 0) -> String:
        """`  (jev 3412 ms, 512 tokens)`, or `  (jev 680 + 2290 ms, 512 + 890
        tokens)` after two requests: the wait a listener sees between the
        transcript and the robot, and the tokens the model read for it (0 = the
        backend did not say)."""
        var s = String("  (jev ")
        if self.dest_ms > 0.0:
            s += String(Int(self.dest_ms)) + String(" + ")
        s += String(Int(ms)) + String(" ms")
        if tokens > 0:
            s += String(", ")
            if self.dest_ms > 0.0:
                s += String(self.dest_tok) + String(" + ")
            s += String(tokens) + String(" tokens")
        if len(self.asked_view) > 0:
            s += String(", seeing ") + String(self.asked_view[0].width) + String("x") + String(self.asked_view[0].height)
        return s + String(")")

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
            # "the other one": a remembered place, settled by the log, ahead of
            # the commands (RB-1's parser route: the model names the reference)
            var mem = String("")
            if two and self.cfg.memory_route:
                if self.mem_asked:
                    mem = g1_route_memory(ans, ctx, self.cfg.destinations, self.cfg.dest_words)
            if (self.cfg.triage == G1_TRIAGE_FULL and two and mem == ""
                    and self.tri.place_mass >= 0.5 and self.pick.name != ""
                    and g1_command_moves(bank, self.pick.name)):
                # ⚠ A REQUEST FOR A PLACE THAT ROUTED NO PLACE IS NEVER A WALK:
                # "go where I told you yesterday" came back `walk` (RB-2)
                self._reply(
                    _vl_triage_hint(String("unsettled")),
                    String("[triage] a place, none routed: ") + self.pick.name
                    + String(" -> ask") + self._jev_ms(ans.latency_ms, ans.input_tokens),
                )
            elif mem != "":
                ev.kind = VL_WORLD
                ev.destination = mem
                self.note = (String("[memory] other one -> ") + mem
                             + self._jev_ms(ans.latency_ms, ans.input_tokens))
            elif self.pick.destination != "":
                # ⚠ A DESTINATION IS A SUCCESS WITH A DIFFERENT HANDLER, and
                # `g1_decide` has already cleared `name` so a stale bank row
                # cannot race the planner.
                ev.kind = VL_WORLD
                ev.destination = self.pick.destination
                self.note = (String("[world] ") + self.pick.destination
                             + String(" ") + _vl_f2(self.pick.dest_conf)
                             + self._jev_ms(ans.latency_ms, ans.input_tokens))
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
                             + _vl_f2(self.pick.needs_world)
                             + self._jev_ms(ans.latency_ms, ans.input_tokens))
            elif self.pick.talk and self.replied:
                # `talk_first`: the reply is out already, and one is enough
                self.note = (String("[talk] ") + _vl_f2(self.pick.conf) + String(" — replied already")
                             + self._jev_ms(ans.latency_ms, ans.input_tokens))
            elif self.pick.talk:
                # ⚠ Nothing about the robot's motion changes — it goes on
                # doing whatever it was doing while it replies.
                try:
                    var msgs = List[ChatMessage]()
                    msgs.append(ChatMessage.user(self.heard))
                    if self.reply_override != "":
                        self.canned = self.reply_override   # computed: the model is not asked
                    else:
                        self.llm.start(
                            msgs, self.cfg.chat_sys, List[ToolSpec](), False
                        )
                    self.t_wait = perf_counter_ns()
                    self.note = (String("[talk] ") + _vl_f2(self.pick.conf)
                                 + self._jev_ms(ans.latency_ms, ans.input_tokens))
                    self.state = VL_ST_CHAT
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
                                     + self._jev_ms(ans.latency_ms, ans.input_tokens)
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
                                 + _vl_f2(self.pick.needs_world)
                                 + self._jev_ms(ans.latency_ms, ans.input_tokens))
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
            if self.cfg.miss_kind:
                # a question, or an action on an object: words, not a latent
                var p_move = g1_miss_movement_p(sa)
                if p_move < G1_MISS_MOVE_P:
                    self._reply(
                        String("If the request is not something you can do, say so kindly; otherwise answer it."),
                        String("[spec] ") + sa.choice(String(G1_Q_MISS_KIND)) + String(", P(movement) ")
                        + _vl_f2(p_move) + String(" -> reply, no motion"),
                    )
                    return ev^
            var n_scaf = g1_spec_from_answers(sa, self.pool, bank, sterms)
            if n_scaf < 0:
                # ⚠ ADDRESSED, AND NOT A MOTION: "tell me a joke" came here (the
                # bank has no row for it and no reward can say it). Silence was
                # the old answer; a spoken reply is the right one, and the motion
                # is unchanged.
                self._reply(
                    String("If the request is not something you can do, say so kindly; otherwise answer it."),
                    String("[spec] the model named no quantity -> reply"),
                )
                return ev^
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
                    + _vl_f2(Float64(len(self.vad.seg)) / 16000.0) + String("s"))
        var el = Float64(perf_counter_ns() - self.t_wait) / 1e9
        if self.state == VL_ST_STT:
            return (String("transcribing ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.stt_dl) + String("s"))
        if self.state == VL_ST_DEST:
            return (String("deciding: where? ") + _vl_f2(el) + String("s / ")
                    + _vl_f2(self.cfg.jev_dl) + String("s"))
        if self.state == VL_ST_TRAIL:
            return (String("deciding: which place before? ") + _vl_f2(el)
                    + String("s / ") + _vl_f2(self.cfg.jev_dl) + String("s"))
        if self.state == VL_ST_CHAIN:
            return (String("deciding: which two places? ") + _vl_f2(el)
                    + String("s / ") + _vl_f2(self.cfg.jev_dl) + String("s"))
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
        return self.vad.level

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
        return self.vad.peak

    def open_threshold(self) -> Float64:
        """What the level must exceed to start a segment. ⚠ A HUD that shows
        the level without this cannot answer "how far is my voice from
        starting a recording", which is the question a silent demo raises."""
        return g1_vad_open_at(self.vad.floor)

    def noise_floor(self) -> Float64:
        """⚠ THE FLOOR, NOT THE PEAK. The first version of this printed the
        peak under the label "noise floor" — two different numbers, and the
        one it showed was the loudest thing in the first second."""
        return self.vad.floor


def _vl_triage_hint(kind: String) -> String:
    """What the reply is told when the triage stopped the motion."""
    if kind == "unsafe":
        return String(
            "You were asked for something you cannot do safely (climbing on"
            " furniture, acrobatics, moving furniture). Say briefly that you"
            " will not do it."
        )
    if kind == "speaker_side":
        return String(
            "The speaker named a place by THEIR own left, right, front or back,"
            " and you cannot tell where they stand. Ask briefly which place"
            " they mean."
        )
    if kind == "unsettled":
        return String(
            "You do not know which place the speaker means. Ask briefly which"
            " one."
        )
    return String("")


def _vl_wav_seconds(ref wav: List[UInt8]) -> Float64:
    """A PCM WAV's length from its header (channels, rate, bits) and size; 0 when
    the header is not one (a 44-byte RIFF header is assumed)."""
    if len(wav) < 44:
        return 0.0
    var ch = Int(wav[22]) | (Int(wav[23]) << 8)
    var rate = Int(wav[24]) | (Int(wav[25]) << 8) | (Int(wav[26]) << 16) | (Int(wav[27]) << 24)
    var bits = Int(wav[34]) | (Int(wav[35]) << 8)
    var per_s = rate * ch * (bits // 8)
    if per_s <= 0:
        return 0.0
    return Float64(len(wav) - 44) / Float64(per_s)


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
