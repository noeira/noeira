# +--------------------------------------------------------------------------+ #
# | Talk to the robot — speech, Jev and the bank, all inside one render loop
# +--------------------------------------------------------------------------+ #
"""Press TAB, say what you want, watch it happen.

    pixi run mojo build -I . -Xlinker -ld_classic \
        examples/g1/bfm_zero_voice_demo.mojo -o build/g1voice
    ./build/g1voice --ckpt runs/<id>/checkpoints/step_36000.ckpt

Needs `JEV_API_KEY` (or `TYPESAFE_API_KEY`) and `HF_TOKEN`. Keys, the sidebar
buttons and the §12.53 channel all still work, so the demo has a rehearsed
path as well as a live one.

## Everything is in one loop, and nothing blocks it

`noeira/ai/` grew `start` / `poll` / `result` on libcurl's non-blocking
interface, so a call in flight costs **under 0.5 ms a frame**. The robot keeps
executing its last command while it is being listened to, transcribed and
decided for — which is the whole reason this can be one program rather than
the writer-plus-reader split of §12.53.

    IDLE ──TAB──> RECORDING ──> TRANSCRIBING ──> DECIDING ──> acting ──> IDLE

`warm_up()` is called BEFORE the loop: the first poll on a cold connection
costs 19.5 ms, which is a dropped frame, and 0.16 ms after warming.

## ⚠ Push to talk, not an open mic, and the reason is not laziness

Whisper is FILE transcription, not streaming ASR. An always-open mic means
transcribing fixed windows for ever, and **a wake word does not fix that** —
you have to transcribe before you can detect "Robot", so a wake word saves
Jev calls, not Whisper calls. Hands-free needs local voice-activity
detection, and that needs a streaming capture primitive
(`record_wav(path, seconds)` is one fixed-duration ffmpeg call). That has
been requested; the state machine below does not change when it lands, only
what moves IDLE to RECORDING.

⚠ And `record_wav` BLOCKS for its whole duration. Four seconds of frozen
renderer is four seconds of frozen physics, so this spawns `ffmpeg` detached
and polls for the finished file — written to a `.part` and renamed, so the
poll can never see a half-written WAV.

## ⚠ Locomotion has no natural end

A posture command holds itself: `z` persists until something replaces it,
which is what "the robot stays in its pose" means for `squat` or `arms_wide`.
But `walk` walks for ever. Commands in the `locomotion` group therefore
return to `stand` after `--walk-seconds`. The bank carries a `group` per
entry, which is what that field is for.
"""

from std.math import abs
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.core.cont_action import ContAction
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.data.store import TrajectoryStore
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_D, G1_H, G1_L, G1_HB, g1_slerp_z,
)
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_channel import G1CommandChannel
from noeira.envs.robots.g1_reward_vocab import (
    G1_NVOC, g1_quantities, QV_BODY_H, QV_SPEED, QV_UPRIGHT, QV_SPEED_FWD,
    QV_LHAND_H, QV_RHAND_H, QV_YAW_RATE,
)
from noeira.render.sdl import Ptr
from noeira.render.types import Color
from noeira.render.ui import UI
from noeira.render.sdl.sdl_scancode import Scancode
from noeira.render.sdl.sdl_keyboard import get_keyboard_state
from noeira.io.fileio import (
    remove_file, file_size, read_file_bytes, write_text_atomic,
)
from noeira.core.bytes import string_from_bytes
from noeira.io.proc import run_system, quote_arg
from noeira.ai.audio_io import MicCapture, rms
from noeira.io.wav import WavAudio
from noeira.ai.jev import JevClient
from noeira.ai.speech import SpeechToText, STT_RAW
from noeira.envs.robots.g1_spec import (
    G1Pool, g1_spec_questions, g1_spec_from_answers, g1_spec_admit,
    g1_spec_bank_baseline, g1_spec_describe, g1_spec_prompt, g1_spec_record,
)
from noeira.envs.robots.g1_reward_vocab import G1Term
from noeira.envs.robots.g1_command_language import (
    g1_command_questions, g1_decide, G1LangPick, g1_command_phrase,
    g1_decide_chain, g1_extent_metres, g1_extent_seconds, G1_Q_EXTENT,
    G1Context, g1_command_state, g1_since_word,
)
from noeira.ai.chat import ChatClient, ChatMessage, ToolSpec

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime SIDEBAR_W: Int = 240
comptime UI_HEAD = Color(235, 240, 250, 255)
comptime UI_TXT = Color(205, 215, 232, 255)
comptime UI_DIM = Color(120, 132, 155, 255)
comptime UI_WARN = Color(232, 168, 96, 255)
comptime UI_OK = Color(120, 210, 150, 255)
comptime UI_BAD = Color(232, 108, 104, 255)

comptime ST_IDLE: Int = 0
comptime ST_REC: Int = 1
comptime ST_STT: Int = 2
comptime ST_JEV: Int = 3
comptime ST_CHAT: Int = 4
# ⚠ THE CACHE MISS GETS ITS OWN STATE, because it is a SECOND round trip and
# the loop cannot block for it any more than for the first. `P(none)` over
# the bank is a calibrated miss (§10.2); this is where the instruction gets a
# reward spec instead of a name.
comptime ST_SPEC: Int = 5

# ⚠ EVERY WAIT NEEDS A DEADLINE AND A VISIBLE CLOCK, and this had neither. A
# session hung on `transcribing...` and the honest answer to "is it slow or is
# it dead?" was that THE DEMO COULD NOT SAY — the HUD printed a STATE, not a
# FACT. That is the same defect as a "noise floor" that printed the peak and a
# language hint the HUD announced but the client dropped: an instrument that
# reports an intention instead of a measurement.
#
# It was most likely slow rather than dead. `HttpClient(120000, 10000)` gives
# a 120 s read timeout and `retries = 2`, so ONE hanging request can sit for
# about six minutes. From outside, six minutes is indistinguishable from
# forever.
#
# The bounds come from this project's own measured latencies, not from taste.
# Whisper on the HF endpoint across §12.55-12.60: 0.71, 0.79, 1.97, 4.99,
# 5.30, 5.41 s. Jev: 0.25-0.61 s. So 20 s is ~4x the slowest transcription
# ever seen here and 12 s is ~20x the slowest decision.
comptime STT_DEADLINE_S: Float64 = 20.0
comptime JEV_DEADLINE_S: Float64 = 12.0
# ⚠ THE CHAT REPLY GETS ITS OWN, AND A LONGER ONE. It is a generative call,
# not a constrained readout — tokens come out one at a time and a sentence is
# genuinely slower than a decision. But it is the same failure: `llm.done()`
# that never turns true leaves the demo waiting for ever, with the HUD saying
# nothing about it.
comptime CHAT_DEADLINE_S: Float64 = 30.0

# ⚠ A 4 s WINDOW DOES NOT CAPTURE 4 s. Measured: a 4.0 s recording arrived as
# 56 832 samples — 3.55 s — and Whisper returned "Lève les brl'air" for
# "lève les bras en l'air". The loss is at both ends and neither is the
# microphone's fault:
#
#   TAIL: the loop stops reading the instant a segment ends, while ffmpeg's
#   last packets are still in the pipe. DRAIN_S keeps reading past the end.
#
#   HEAD: a speaker starts on the keypress, not after it. PREROLL_S keeps a
#   ring of audio from BEFORE the key so the first syllable survives. This is
#   the same ring VAD will need, and for a sharper reason — a level detector
#   trips 100-300 ms after the onset, which is exactly where a wake word
#   lives.
comptime PREROLL_S: Float64 = 0.35
comptime DRAIN_S: Float64 = 0.35

# ── voice activity detection ──────────────────────────────────────────────
# ⚠ AN ABSOLUTE RMS THRESHOLD DOES NOT SURVIVE A ROOM CHANGE. This machine's
# noise floor measured 0.0004 against speech peaks of 0.024-0.037 — a factor
# of 60-90 — but a fan, a laptop under load or a different room moves the
# floor by more than the margin a fixed number would leave. So the floor is
# TRACKED while nobody is speaking, and the thresholds are multiples of it.
#
# ⚠ AND THEY ARE TWO DIFFERENT NUMBERS. One threshold chatters: a talker
# crosses it a dozen times a second between syllables and each crossing would
# open or close a segment. Opening is harder than staying open, and a segment
# ends only after HANG_S of quiet.
# ⚠ MEASURED AGAINST A REAL ROOM, NOT A QUIET ONE. A silent room gave a floor
# of 0.0004 against speech peaks of 0.024-0.037 — 60-90x, which made almost
# any multiplier look right. A working session gave ambient peaks of 0.0070
# against the same speech: a ratio of 4-8. These numbers fit the second case,
# because that is the one a demo happens in.
comptime VAD_OPEN_MULT: Float64 = 5.0
comptime VAD_CLOSE_MULT: Float64 = 2.5
# ⚠ RAISED FROM 0.0020 ON EVIDENCE. The floor is an average over quiet
# frames and keeps falling in a silent room, so this constant — not the
# multiplier — is what actually opens a segment when nothing is happening. At
# 0.0020 a faint sound of 0.0028 opened one and cost a Whisper call to be
# told it was "you". Speech in the same session peaked at 0.021-0.043.
comptime VAD_OPEN_MIN: Float64 = 0.0060
# ⚠ AND ONE FRAME IS NOT SPEECH. A single read above the threshold is a
# click, a chair, a keystroke. Requiring consecutive frames costs nothing,
# because the pre-roll ring already holds the onset that the wait discards.
comptime VAD_OPEN_FRAMES: Int = 3
# a segment whose loudest moment never rises meaningfully above the opening
# threshold was not speech either — dropped before it costs a call
comptime VAD_MIN_PEAK_MULT: Float64 = 1.6
comptime VAD_FLOOR_MIN: Float64 = 0.00005


# ⚠ THERE IS NO `_rec_start` ANY MORE, AND THE BUG IT DIED OF IS WORTH
# KEEPING. It spawned ffmpeg to write `<path>.part` and renamed on success —
# the atomic-write discipline the command channel uses. But **ffmpeg picks
# its output format from the file EXTENSION**, and `.part` is not a format:
#
#     Unable to choose an output format for '/tmp/....wav.part';
#     use a standard extension for the filename or specify the format
#
# It failed instantly, every time, and the demo reported "no wav — mic
# permission?" because that was the only explanation I had built. The mic was
# never involved. ⚠ AND I HAD SENT ITS STDERR TO /dev/null, so the one line
# that said exactly what was wrong was discarded on purpose. `-f wav` fixes
# it; MicCapture removes the spawn instead, and gives a live level meter for
# free — which is what tells you the microphone is alive BEFORE you speak.


def _f4(v: Float64) -> String:
    """⚠ FOUR DECIMALS FOR THE MIC, and it is not fussiness. A quiet room's
    noise floor is around 0.001 RMS, which at two decimals prints `0.00` —
    indistinguishable from a dead input, which is the one thing the meter
    exists to rule out. `digital_silence()` gives the verdict; this lets the
    number agree with it."""
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 10000.0 + 0.5)
    var f = String(h % 10000)
    while f.byte_length() < 4:
        f = String("0") + f
    var b = String(h // 10000) + String(".") + f
    return (String("-") + b) if neg else b


comptime SPEAK_FLAG: String = "/tmp/noeira_g1_speaking"
# ⚠ how long the room keeps ringing after `say` exits, plus whatever is still
# in ffmpeg's pipe. Small, but not zero.
comptime ECHO_TAIL_S: Float64 = 0.45


def _speaking() -> Bool:
    """True while `say` is still talking.

    ⚠ THIS WAS AN ESTIMATE AND THE ESTIMATE WAS NOT CLOSE. The first version
    predicted the duration from the text at ~14 characters a second: it gave
    1.24 s for "spin_left", which actually takes **2.79 s** — and
    "both_hands_up", half again as long in characters, takes **2.20 s**, so
    the estimate does not even order them correctly. `say` has a start-up
    cost that dominates a short word, and it pronounces an underscore.

    The robot went on hearing itself, transcribing "SpinLeft" and
    "SpinLift" back into its own decider, because the gate expired while the
    speaker was still going.

    `say` has no completion handle, so it gets one: the flag is created
    before the process and REMOVED BY THE SHELL when it exits. Exact, and the
    same trick as renaming a finished recording into place.
    """
    try:
        _ = file_size(String(SPEAK_FLAG))
        return True
    except:
        return False


comptime SPEAK_TEXT: String = "/tmp/noeira_g1_say.txt"


def _ns(seconds: Float64) -> Int:
    """Seconds to nanoseconds, via milliseconds.

    ⚠ `Int(x * 1e9)` does not compile here — the literal drags the product
    into a SIMD type that `Int` will not take. Going through an integer
    millisecond keeps it in Int arithmetic, and a millisecond is finer than
    anything this schedules."""
    return Int(seconds * 1000.0) * 1_000_000


def _say_async(text: String):
    """Speak in the background, and leave a flag behind that says so.

    ⚠ THE TEXT GOES THROUGH A FILE, NEVER THE COMMAND LINE. `quote_arg`
    REFUSES a string containing a single quote — a sound rule for a path and
    a fatal one for speech, because French is full of apostrophes. The first
    French phrase the robot tried to say was "je m'accroupis", and it took
    the whole demo down with it. `say -f` reads the text from a file, so
    nothing in it is ever interpreted by a shell: apostrophes, quotes,
    accents, newlines.

    ⚠ AND IT CANNOT RAISE. A confirmation is cosmetic; the robot, the
    physics and the microphone are not. Anything that goes wrong here is
    swallowed, because the alternative is what happened above.

    `say` blocks for as long as it speaks, so it cannot run in the loop. The
    flag is written HERE rather than inside the backgrounded shell, or there
    would be a window between the spawn and the flag appearing in which the
    gate is open and the microphone is already live.
    """
    # ⚠ NEVER START A SECOND UTTERANCE WHILE ONE IS RUNNING. The flag is a
    # single shared file: two overlapping `say` shells both remove it, and
    # the FIRST to finish opens the gate while the SECOND is still speaking.
    # That is how "je tourne à gauche" — the robot's own phrase for
    # `spin_left` — came back through Whisper and picked `spin_left` again,
    # after a chained step spoke while the previous confirmation was still
    # going. The confirmation is cosmetic and the HUD already shows it, so
    # the later one is simply dropped.
    if _speaking():
        return
    try:
        write_text_atomic(String(SPEAK_TEXT), text)
        _ = run_system("touch " + quote_arg(String(SPEAK_FLAG)))
        _ = run_system(
            "( say -f " + quote_arg(String(SPEAK_TEXT)) + " ; rm -f "
            + quote_arg(String(SPEAK_FLAG)) + " ) >/dev/null 2>&1 &"
        )
    except:
        # ⚠ and clear the flag, or the gate stays shut for ever and the
        # microphone never reopens.
        try:
            remove_file(String(SPEAK_FLAG))
        except:
            pass

comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) raises -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _f2(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 100.0 + 0.5)
    var f = String(h % 100)
    while f.byte_length() < 2:
        f = String("0") + f
    var b = String(h // 100) + String(".") + f
    return (String("-") + b) if neg else b


def main() raises:
    var ckpt = _flag(String("--ckpt"), String(""))
    var bank_path = _flag(String("--bank"), String("g1_command_bank.txt"))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var chan_path = _flag(String("--channel"), String("/tmp/g1_cmd"))
    var start_clip = atol(_flag(String("--start-clip"), String(13)))
    var fps = atol(_flag(String("--fps"), String(50)))
    var walk_s = Float64(String(_flag(String("--walk-seconds"), String("5"))))
    var max_none = Float64(String(_flag(String("--max-none"), String("0.25"))))
    var min_top = Float64(String(_flag(String("--min-top"), String("0.35"))))
    var mute = _has(String("--mute"))
    var mic_dev = _flag(String("--mic"), String(""))
    # ⚠ `--lang fr` PINS WHISPER'S LANGUAGE. Auto-detection on a short, quiet
    # clip is unreliable in a way that is invisible until it is absurd: a real
    # session's first utterance came back as "래위에 봐" — Korean — from a
    # French speaker. `SpeechToText.language`'s own docstring says it "avoids
    # a wrong-language transcript of a two-word command", which is exactly
    # the case here: robot commands are two words, and they are the hardest
    # thing to auto-detect from.
    var lang = _flag(String("--lang"), String(""))
    var llm_spec = _flag(String("--llm"), String("hf"))
    var phrase_file = _flag(String("--phrases"), String(""))
    var vad = not _has(String("--push-to-talk"))
    var hang_s = Float64(String(_flag(String("--hang"), String("0.7"))))
    var max_seg = Float64(String(_flag(String("--max-seg"), String("10"))))
    var min_seg = Float64(String(_flag(String("--min-seg"), String("0.45"))))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 72)
    print("BFM-Zero G1 — the command bank, driven")
    print("=" * 72)
    var bank = G1CommandBank.load(bank_path)
    print("  bank:", bank.count(), "commands from", bank_path)
    print("  channel:", chan_path, " (echo 'seq 1<newline>cmd squat' > it)")

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm sidecar beside the checkpoint: RAW inputs.")

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    # ⚠ THE BANK CARRIES ITS OWN START POSE, so nothing here opens the store.
    # It used to, for ONE row out of 441 131, and the 1.69 GB read put ~14 s
    # between launch and the first poll — long enough that commands sent in
    # the meantime were written, overwritten, and never seen. A writer that
    # waits for the `.ack` file is immune; one that fires and forgets is not.
    if bank.has_pose():
        for i in range(NQ):
            qp[i] = bank.qpos[i]
        for i in range(NV):
            qv[i] = bank.qvel[i]
        print("  start pose: from the bank")
    else:
        print("  start pose: NOT in the bank — opening the store (slow)")
        var store = TrajectoryStore(store_path)
        var rsi = G1RsiTable.from_store(store)
        var base = Int(rsi.ep_offset.data[start_clip]) * (G1_RSI_NQ + G1_RSI_NV)
        for i in range(NQ):
            qp[i] = Float64(rsi.rows.data[base + i])
        for i in range(NV):
            qv[i] = Float64(rsi.rows.data[base + G1_RSI_NQ + i])
    env.set_state(qp, qv)

    if not env.init_renderer(show_velocity=False):
        raise Error("no renderer available")
    env.set_ui_sidebar_width(SIDEBAR_W)
    # ⚠ the renderer's built-in HUD draws over the ENV viewport, not the
    # sidebar, so it sits on top of the robot. Safe to turn off: `render()`
    # draws the application widget list on both branches (the split that made
    # `set_show_hud(False)` stop killing `set_ui`), so the sidebar stays.
    env.renderer_set_show_hud(False)
    var delay_ms = 1000 // fps if fps > 0 else 0

    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var aobs = G1ActorObs()
    var quant = List[Float64](length=G1_NVOC, fill=0.0)

    # ── the z the policy is actually driven with ──────────────────────
    # `zpair` holds TWO rows: 0 is where the blend started, 1 is the target.
    # ⚠ ONE tensor with two row indices, not two tensors — `g1_slerp_z` takes
    # a source and two indices because Mojo refuses two `ref` arguments that
    # alias, which is the same shape §12.47 hit between knots.
    var zpair = Tensor.alloc(2 * D)
    var zcur = Tensor.alloc(D)
    var cur = bank.find(String("stand"))
    if cur < 0:
        cur = 0
    for k in range(D):
        zpair.data[k] = Scalar[DT](bank.z_at(cur, k))
        zpair.data[D + k] = Scalar[DT](bank.z_at(cur, k))
        zcur.data[k] = Scalar[DT](bank.z_at(cur, k))
    var blend_left = 0
    var blend_len = 0

    # ⚠ WARM BOTH CONNECTIONS BEFORE THE LOOP. The first poll on a cold TLS
    # connection costs 19.5 ms — a dropped frame — against 0.16 ms warmed.
    var jev = JevClient.from_env()
    # `--stt groq` is the escape hatch: Groq's Whisper takes a TEXT prompt,
    # which the HF endpoint cannot (see `--vocab` below).
    var stt_spec = _flag(String("--stt"), String("hf"))
    var stt = SpeechToText.groq() if stt_spec == "groq" else SpeechToText.huggingface()
    if lang != "":
        stt.language = lang
        print("  whisper language pinned to", lang, "(" + stt_spec + ")")
    # ⚠ BIAS THE DECODER TOWARD THE WORDS THIS ROBOT ANSWERS TO. "cours" came
    # back as "cool" twice in a row. The first suspect was the homophone; the
    # real one was that `language` was NEVER REACHING WHISPER on this backend
    # — the HF path posted the raw WAV and dropped both hints — so it was
    # auto-detecting from a one-word French utterance. That is fixed in the
    # client; pinning the language may be the whole fix, and the vocabulary
    # below only matters if it is not.
    #
    # ⚠ HF CANNOT TAKE A TEXT PROMPT — its Whisper pipeline accepts one only
    # as token ids, and the client RAISES rather than dropping it silently.
    # So the vocabulary is sent only on a backend that takes it, and the HUD
    # says which, instead of a hint that quietly does nothing.
    var vocab = _flag(String("--vocab"), String(
        "marche, cours, recule, tourne à gauche, tourne à droite,"
        " accroupis-toi, lève le bras droit, lève le bras gauche,"
        " lève les deux bras, écarte les bras, regarde à gauche,"
        " regarde à droite, arrête-toi, baisse les bras"
    ))
    if vocab != "":
        if stt.kind == STT_RAW:
            print("  vocabulary hint NOT sent —", stt_spec,
                  "takes no text prompt; pass --stt groq to use it")
        else:
            stt.prompt = vocab
            print("  vocabulary hint sent to the decoder")
    jev.warm_up()
    stt.warm_up()
    # ⚠ `talk` needs a client warmed BEFORE the loop like the others, or its
    # first reply pays 19.5 ms inside one frame.
    var llm = ChatClient.from_spec(llm_spec)
    llm.max_tokens = 120
    # ⚠ Qwen thinks before answering unless told not to, which took a reply
    # from 10.8 s to 1.4 s in the AI package's own tests. A robot that pauses
    # eleven seconds before saying hello is not having a conversation.
    llm.extra(String("chat_template_kwargs"), String('{"enable_thinking": false}'))
    llm.warm_up()
    var chat_sys = String(
        "You are a small humanoid robot in a simulator. Reply in ONE short"
        " sentence, in the language you were addressed in. You can walk,"
        " turn, squat and raise your arms, but you cannot pick things up or"
        " go anywhere in particular. Be warm and brief."
    )
    var quest = g1_command_questions(bank, True)
    print("  jev + whisper warmed")

    # ⚠ ONE capture for the whole session, not one per utterance. It is what
    # makes a live level meter possible, it is the foundation VAD needs, and
    # it removes a process spawn from the moment you press a key.
    var mic = MicCapture.start(16000, mic_dev)
    print("  mic open:", "default" if mic_dev == "" else mic_dev)
    var level = 0.0
    var peak = 0.0
    var seg = List[Int16]()
    var ring = List[Int16]()
    var ring_max = Int(PREROLL_S * 16000.0)
    # ⚠ `--phrases FILE` overrides the spoken phrases with `name=phrase`
    # lines. The built-in table is English because the tree is English; a
    # French session wants a French file, and that is data, not source.
    var ph_name = List[String]()
    var ph_text = List[String]()
    if phrase_file != "":
        var praw = string_from_bytes(read_file_bytes(phrase_file))
        var plines = praw.split("\n")
        for i in range(len(plines)):
            var pl = String(plines[i])
            if pl.byte_length() == 0 or pl.startswith("#"):
                continue
            var eq = pl.find("=")
            if eq <= 0:
                continue
            ph_name.append(String(pl[byte=0:eq]))
            ph_text.append(String(pl[byte=eq + 1:pl.byte_length()]))
        print("  phrases:", len(ph_name), "from", phrase_file)

    var mic_dead = String("")
    var checked_silence = False
    var mic_on = True
    var floor = 0.0010
    var quiet_since = perf_counter_ns()
    var prev_m = False
    var forced = False
    var above = 0
    # ── the queue ─────────────────────────────────────────────────────
    # ⚠ ONE STEP IS ACTIVE AT A TIME and the rest wait. A chain is not
    # played by scheduling three commands at fixed offsets: a locomotion
    # step ends on DISTANCE, which is not known in advance.
    var q_name = List[String]()
    var q_conf = List[Float64]()
    var step_end_dist = 0.0     # metres, 0 = not distance-terminated
    var step_end_at = perf_counter_ns()
    var step_dist = 0.0
    var extent = 2.0
    # ⚠ while this is in the future, the microphone is the robot's own voice
    var mute_until = perf_counter_ns()

    var state = ST_IDLE
    var heard = String("")
    var pick = G1LangPick()
    var t_rec = perf_counter_ns()
    # ⚠ when the CURRENT wait began. Reset on every transition into a waiting
    # state; read by both the HUD and the deadline, so they cannot disagree.
    var t_wait = perf_counter_ns()
    var t_cmd = perf_counter_ns()
    var prev_key = False

    # ── the cache-miss path, loaded up front ──────────────────────────
    # ⚠ AT STARTUP, NOT LAZILY. The sidecar is 70 MB and `G1Pool.__init__`
    # sorts 14 columns of 65 536; doing that on the first miss would freeze
    # the render loop — and with it the physics step — for about a second,
    # mid-demo. The demo already opens a 640 MB checkpoint, so one more read
    # here is invisible where a hitch would not be.
    #
    # ⚠ AND THE BANK BASELINES ARE PRECOMPUTED. They never change, and
    # recomputing all 20 per miss would cost 20 passes over the pool — about
    # a second. Precomputed, a miss costs ONE pass (~30 ms, one dropped
    # frame) plus 20 dot products.
    # ⚠ THE DEADLINES ARE FLAGS SO THE GIVE-UP PATH CAN BE EXERCISED. It
    # fires only when a call hangs, which is exactly the condition that
    # cannot be summoned on demand — so set a tiny one and any NORMAL call
    # blows it, which tests the recovery rather than the hang:
    #
    #     --stt-deadline 0.5     # speak; it must give up and keep running
    #
    # A recovery path that has never run is a guess, and this file has paid
    # for guesses four times over (§12.55's echo estimate, §12.56's dropped
    # language hint, §12.60's two duplicate criteria).
    var stt_dl = Float64(String(_flag(String("--stt-deadline"),
                                      String(STT_DEADLINE_S))))
    var jev_dl = Float64(String(_flag(String("--jev-deadline"),
                                      String(JEV_DEADLINE_S))))
    var chat_dl = Float64(String(_flag(String("--chat-deadline"),
                                       String(CHAT_DEADLINE_S))))
    var spec_pool_path = _flag(String("--pool"), String(""))
    var cand_path = _flag(String("--candidates"),
                          String("g1_spec_candidates.txt"))
    var has_pool = spec_pool_path != ""
    var pool = G1Pool(1, List[Float64](length=D, fill=0.0),
                      List[Float64](length=G1_NVOC, fill=0.0))
    var zbase = List[Float64]()
    if has_pool:
        pool = G1Pool.load(spec_pool_path)
        zbase = List[Float64](length=bank.count() * D, fill=0.0)
        g1_spec_bank_baseline(pool, bank, zbase)
        print("  spec path: ON —", pool.n, "pool rows from", spec_pool_path)
    else:
        print("  spec path: off (pass --pool g1_pool.bin to answer commands",
              "the bank has no name for)")
    var spec_q = g1_spec_questions(True)
    var znew = List[Float64](length=D, fill=0.0)

    var chan = G1CommandChannel(chan_path)
    # ⚠ WHAT THE ROBOT IS DOING, SO THE NEXT SENTENCE CAN BE RELATIVE TO IT.
    # Without this the demo sent Jev the transcript alone, and "plus vite"
    # after a walk was refused — correctly, because nothing in the request
    # said what was to be done faster (§10.1 of BFM_ZERO_NEXT_LEVEL.md).
    var ctx = G1Context()
    # ⚠ True while a novel `z` is driving and `cur` therefore does NOT
    # describe what the robot is doing — see the note at `spec_active = True`.
    var spec_active = False
    var last_event = String("(waiting)")
    var pending = -1
    var pending_blend = 25

    if vad:
        print("  HANDS-FREE: just speak. M mutes the mic, TAB forces a segment.")
    else:
        print("  push-to-talk: TAB to speak. M mutes the mic.")
    print("  keys 1-9 / a-i pick a command, SPACE = stand, ESC quits")
    print("-" * 72)

    var step = 0
    while env.is_renderer_open():
        if env.check_renderer_quit():
            break

        # ── the channel, polled every frame ───────────────────────────
        # An open/read/close of a few dozen bytes. The expensive thing —
        # speech, a model call — happened in another process.
        var msg = chan.poll()
        if msg.fresh:
            # ⚠ A `z` ON THE CHANNEL IS A COMMAND WITH NO BANK ENTRY — the
            # cache-miss path (`g1_spec.mojo`, §12.60). It arrives already
            # gated by the writer on ESS and the goal-term key, so the loop's
            # only job is to check the WIDTH: any 256 floats project onto the
            # sphere and drive a plausible robot, so a truncated row would be
            # obeyed rather than noticed.
            if len(msg.z) == D:
                for k in range(D):
                    zpair.data[k] = zcur.data[k]
                    zpair.data[D + k] = Scalar[DT](msg.z[k])
                blend_len = msg.blend if msg.blend > 0 else 1
                blend_left = blend_len
                # same cancellation as the voice path
                spec_active = True
                q_name = List[String]()
                q_conf = List[Float64]()
                # ⚠ `cur` STAYS PUT and `pending` is NOT set. A novel `z` has
                # no bank index, and `cur` is what the HUD, the keys and the
                # context all read as "the command running now". Pointing it
                # at a stale entry would make the next relative instruction
                # ("plus vite") resolve against a command the robot is not
                # doing (§12.56).
                ctx.began(String("spec"))
                t_cmd = perf_counter_ns()
                last_event = String("chan: z ") + msg.cmd
                chan.ack(msg.seq, msg.cmd, True)
            elif len(msg.z) > 0:
                # a short row is a half-written file, not a short command
                last_event = String("chan: BAD z (") + String(len(msg.z)) \
                             + String(" of ") + String(D) + String(")")
                chan.ack(msg.seq, msg.cmd, False)
            else:
                var idx = bank.find(msg.cmd)
                if idx >= 0:
                    pending = idx
                    pending_blend = msg.blend
                    last_event = String("chan: ") + msg.cmd
                    chan.ack(msg.seq, msg.cmd, True)
                else:
                    # ⚠ unknown: keep doing what we were doing, and SAY SO.
                    last_event = String("chan: ?") + msg.cmd
                    chan.ack(msg.seq, msg.cmd, False)

        # ── the voice state machine ───────────────────────────────────
        # Every branch is non-blocking. The policy above has already stepped,
        # so the robot goes on doing its last command throughout.
        # ── the microphone, every frame ───────────────────────────────
        # `read` never blocks: 0.08 ms steady state. It RAISES once ffmpeg
        # has exited, carrying ffmpeg's own reason — which is the difference
        # between "grant Terminal microphone access" and "the device went
        # away", and it is why this no longer guesses.
        # ⚠ still READ while the robot speaks — the pipe is 64 KiB, about 2 s,
        # and past that ffmpeg blocks and the device drops audio. The samples
        # are read and DISCARDED, never ringed, never levelled.
        if _speaking():
            mute_until = perf_counter_ns() + _ns(ECHO_TAIL_S)
        var echoing = perf_counter_ns() < mute_until
        if mic_on and mic_dead == "":
            try:
                var pcm = mic.read()
                if echoing:
                    level = 0.0
                    # ⚠ AND DROP THE RING. It holds the last 0.35 s of audio
                    # and is prepended to the next segment — which would
                    # hand Whisper the tail of the robot's own sentence as
                    # the first syllable of yours.
                    ring = List[Int16]()
                elif len(pcm) > 0:
                    level = rms(pcm)
                    if level > peak:
                        peak = level
                    if state == ST_REC:
                        for i in range(len(pcm)):
                            seg.append(pcm[i])
                    else:
                        for i in range(len(pcm)):
                            ring.append(pcm[i])
                        if len(ring) > ring_max:
                            var keep = List[Int16]()
                            for i in range(len(ring) - ring_max, len(ring)):
                                keep.append(ring[i])
                            ring = keep^
                        # ⚠ THE FLOOR IS TYPICAL QUIET, NOT THE QUIETEST
                        # INSTANT. Tracking the minimum put `close_at` BELOW
                        # ordinary room noise, so a segment that opened never
                        # closed: a real session recorded 5.99 s for "you"
                        # and 7.06 s for "Cool. Run.", ending on the
                        # max-segment guard rather than on silence. An
                        # average over quiet frames sits where the room
                        # actually is.
                        #
                        # Only frames BELOW the open threshold count, or
                        # speech would raise the floor it is measured against
                        # and the detector would deafen itself mid-sentence.
                        if level < floor * VAD_OPEN_MULT:
                            floor = floor * 0.98 + level * 0.02
                        if floor < VAD_FLOOR_MIN:
                            floor = VAD_FLOOR_MIN
                if not checked_silence and mic.seconds_read() > 1.0:
                    checked_silence = True
                    if mic.digital_silence(0.5):
                        mic_dead = String("mic is digital silence — muted, "
                                          "disabled, or permission refused")
                        print("  [mic]", mic_dead)
                    else:
                        # ⚠ this used to say "noise floor" and print the
                        # PEAK — two different numbers, and the one it
                        # printed was the loudest thing in the first second.
                        print("  [mic] input live —", _f2(mic.seconds_read()),
                              "s read, floor", _f4(floor), "peak", _f4(peak))
            except e:
                mic_dead = String(e)
                print("  [mic]", mic_dead)

        var open_at = floor * VAD_OPEN_MULT
        if open_at < VAD_OPEN_MIN:
            open_at = VAD_OPEN_MIN
        # ⚠ AND THE CLOSE THRESHOLD NEEDS A FLOOR OF ITS OWN. It was
        # `floor * 2.5`, and the floor is an average over QUIET frames, so in
        # a room whose between-word noise sits above that a segment opens and
        # never closes: one ran the full 10 s max-segment guard for a 3 s
        # question. Tying it to the opening threshold keeps the two in
        # proportion whatever the room is doing.
        var close_at = floor * VAD_CLOSE_MULT
        if close_at < VAD_OPEN_MIN * 0.5:
            close_at = VAD_OPEN_MIN * 0.5

        # ⚠ VAD OPENS ONLY FROM IDLE. While a transcription or a decision is
        # in flight, speech is still ringed but starts nothing — one call per
        # client, and a queue of half-heard commands is worse than a missed
        # one. The 1.2 s guard keeps the floor estimate from opening a
        # segment on its own first samples.
        if level > open_at and not echoing:
            above += 1
        else:
            above = 0
        if (mic_on and mic_dead == "" and state == ST_IDLE and vad
            and not echoing and above >= VAD_OPEN_FRAMES
            and mic.seconds_read() > 1.2):
            heard = String("")
            pick = G1LangPick()
            seg = ring.copy()
            ring = List[Int16]()
            peak = level
            t_rec = perf_counter_ns()
            quiet_since = perf_counter_ns()
            forced = False
            state = ST_REC
            last_event = String("listening...")

        if state == ST_REC:
            if level > close_at:
                quiet_since = perf_counter_ns()
            var quiet_s = Float64(perf_counter_ns() - quiet_since) / 1e9
            var seg_s = Float64(len(seg)) / 16000.0
            # ⚠ THREE WAYS TO END, and the last two are not optional. Silence
            # is the normal one — HANG_S bridges the gap between words without
            # feeling like a wait. A segment that never falls quiet (a fan, a
            # conversation across the room) would otherwise record for ever
            # and post a minute of audio to Whisper. A forced end is what TAB
            # is for when a room is too loud for the detector to close.
            var over = seg_s > max_seg
            var done = forced or over or (quiet_s > hang_s)
            if done and seg_s > DRAIN_S:
                forced = False
                # ⚠ a cough, a chair, a door. Below `min_seg` it is not
                # speech, and sending it costs a Whisper call to be told so.
                if seg_s < min_seg:
                    print("  [vad] dropped", _f2(seg_s), "s — under",
                          _f2(min_seg))
                    state = ST_IDLE
                elif peak < open_at * VAD_MIN_PEAK_MULT:
                    # never got meaningfully louder than the threshold that
                    # opened it
                    print("  [vad] dropped — peak", _f4(peak), "under",
                          _f4(open_at * VAD_MIN_PEAK_MULT))
                    state = ST_IDLE
                else:
                    var audio = WavAudio(16000, 1, seg.copy())
                    stt.start(audio)
                    t_wait = perf_counter_ns()
                    print("  [stt]", len(seg), "samples (", _f2(seg_s),
                          "s ), peak", _f4(peak),
                          "— max-seg" if over else "")
                    state = ST_STT

        elif state == ST_STT:
            # ⚠ CHECKED BEFORE THE POLL, so a call that has already blown its
            # budget is cancelled rather than waited on one more frame.
            var stt_el = Float64(perf_counter_ns() - t_wait) / 1e9
            if stt_el > stt_dl:
                print("  [stt] GAVE UP after", _f2(stt_el),
                      "s — the endpoint never answered. Say it again.")
                try:
                    stt.cancel()
                except:
                    # a cancel that fails must not take the renderer with it
                    pass
                last_event = String("stt timed out")
                state = ST_IDLE
            elif stt.poll():
                # ⚠ `poll` RETURNS TRUE ON FAILURE TOO — it means "the call
                # has finished", not "it worked" — and `result` is what
                # raises. Unwrapped, a 503 from the endpoint took the whole
                # renderer down, and with it the physics step.
                #
                # On failure `heard` stays empty, and the `letters < 3` guard
                # below already routes an empty transcript back to IDLE. So
                # this needs no second exit path.
                var txt = String("")
                var lat = 0.0
                try:
                    var tr = stt.result()
                    txt = tr.text
                    lat = tr.latency_ms
                except e:
                    print("  [stt] FAILED:", String(e))
                    last_event = String("stt failed")
                heard = txt
                if lat > 0.0:
                    print("  [heard]", heard, "(", lat, "ms )")
                # ⚠ Whisper returns "." or " " for a cough. Asking a model
                # which of 18 commands a full stop means costs a call to be
                # told none of them.
                var letters = 0
                for ch in heard.codepoints():
                    if ch.to_u32() > 64:
                        letters += 1
                if letters < 3:
                    print("  [stt] nothing said — skipped")
                    state = ST_IDLE
                else:
                    # ⚠ `start`, NOT `start_text` — a JSON state, so the
                    # questions can refer to `doing` / `did_before` /
                    # `last_arm_raised` by name.
                    ctx.since_s = Float64(perf_counter_ns() - t_cmd) * 1e-9
                    # ⚠ SHOW THE STATE. It is half the question now, and a
                    # wrong answer to a relative instruction cannot be read
                    # without it — "why did it walk?" is answerable only if
                    # the log says what it thought it was doing. The CLI
                    # prints the same thing; two views of one question is how
                    # the last three divergences happened.
                    print("  [ctx] doing", ctx.doing if ctx.doing != "" else
                          String("nothing"), "for",
                          g1_since_word(ctx.since_s), " arm",
                          ctx.last_arm() if ctx.last_arm() != "" else
                          String("none"))
                    jev.start(g1_command_state(heard, ctx), quest)
                    t_wait = perf_counter_ns()
                    state = ST_JEV

        elif state == ST_JEV:
            var jev_el = Float64(perf_counter_ns() - t_wait) / 1e9
            if jev_el > jev_dl:
                print("  [jev] GAVE UP after", _f2(jev_el),
                      "s — no decision came back.")
                try:
                    jev.cancel()
                except:
                    pass
                last_event = String("jev timed out")
                state = ST_IDLE
            elif jev.poll():
                var ans = jev.result()
                pick = g1_decide(ans, bank, max_none, min_top, 0.5, True)
                if pick.name != "":
                    var idx2 = bank.find(pick.name)
                    if idx2 >= 0:
                        pending = idx2
                        pending_blend = 25
                        last_event = String("voice: ") + pick.name
                        t_cmd = perf_counter_ns()
                        # the first step's extent comes from the utterance
                        extent = ans.score(String(G1_Q_EXTENT))
                        step_dist = 0.0
                        if bank.group_at(idx2) == "locomotion":
                            step_end_dist = g1_extent_metres(extent)
                            # ⚠ AND A CLOCK AS WELL. A robot that is blocked,
                            # or walking on the spot, would never reach its
                            # distance and the chain would hang for ever.
                            step_end_at = perf_counter_ns() + _ns(g1_extent_seconds(extent) * 3.0 + 4.0)
                        else:
                            step_end_dist = 0.0
                            step_end_at = perf_counter_ns() + _ns(g1_extent_seconds(extent))
                        var rest = g1_decide_chain(ans, bank)
                        q_name = List[String]()
                        q_conf = List[Float64]()
                        for si in range(len(rest)):
                            q_name.append(rest[si].name)
                            q_conf.append(rest[si].conf)
                        if len(q_name) > 0:
                            var qs = String("")
                            for si in range(len(q_name)):
                                qs += String(" -> ") + q_name[si]
                            print("  [chain]", len(q_name), "more:", qs,
                                  " extent", _f2(extent))
                        print("  [pick]", pick.name, _f2(pick.conf),
                              " P(none)", _f2(pick.p_none),
                              " world", _f2(pick.needs_world))
                        if not mute:
                            # ⚠ SAY THE PART IT CANNOT DO. A robot that
                            # silently performs 30 % of an instruction is
                            # worse than one that performs 30 % and says so.
                            var utter = g1_command_phrase(pick.name)
                            for pi in range(len(ph_name)):
                                if ph_name[pi] == pick.name:
                                    utter = ph_text[pi]
                            if pick.needs_world > 0.5:
                                utter = utter + String(
                                    ", but I have no object or destination")
                            _say_async(utter)
                            mute_until = perf_counter_ns() + _ns(ECHO_TAIL_S)
                elif pick.talk:
                    # ⚠ NOT a bank command: the speaker wanted an answer.
                    # Nothing about the robot's motion changes — it goes on
                    # doing whatever it was doing while it replies.
                    var msgs = List[ChatMessage]()
                    msgs.append(ChatMessage.user(heard))
                    llm.start(msgs, chat_sys, List[ToolSpec](), False)
                    t_wait = perf_counter_ns()
                    last_event = String("talking...")
                    print("  [talk]", _f2(pick.conf))
                    state = ST_CHAT
                else:
                    last_event = String("refused: ") + pick.reason
                    # ⚠ A REFUSAL FOR `no such command` IS A CACHE MISS.
                    # The bank has no NAME for this; the reward vocabulary may
                    # still be able to say it (§12.60). Any other reason —
                    # not addressed, no clear pick — is a real refusal and
                    # must not be routed here, or background speech would
                    # start generating latents.
                    if has_pool and pick.reason == "no such command":
                        print("  [miss] P(none)", _f2(pick.p_none),
                              "— no name for this; asking for a reward spec")
                        # ⚠ THE INSTRUCTION ALONE, not the conversational
                        # state — see `g1_spec_prompt`. Asking with `doing`
                        # in the state made the same instruction produce
                        # ESS 5089 while walking and 301 while standing.
                        jev.start_text(g1_spec_prompt(heard), spec_q)
                        t_wait = perf_counter_ns()
                        state = ST_SPEC
                    else:
                        print("  [refused]", pick.reason, " P(none)",
                              _f2(pick.p_none), " addressed",
                              _f2(pick.addressed))
                    # ⚠ A SPOKEN REFUSAL IS THE WORST THING TO SAY ALOUD.
                    # "no such command" is heard, transcribed, and refused
                    # again — the loop that filled a whole session's log. The
                    # HUD already says it; saying it too buys nothing and
                    # costs a Whisper call every time.
                # ⚠ ONLY when the decision did not hand off. The talk branch
                # sets ST_CHAT, and an unconditional reset here would drop
                # the reply on the floor — the request would run to
                # completion and nothing would ever read it.
                if state == ST_JEV:
                    state = ST_IDLE

        elif state == ST_SPEC:
            var sp_el = Float64(perf_counter_ns() - t_wait) / 1e9
            if sp_el > jev_dl:
                print("  [spec] GAVE UP after", _f2(sp_el), "s")
                try:
                    jev.cancel()
                except:
                    pass
                last_event = String("spec timed out")
                state = ST_IDLE
            elif jev.poll():
                var sa = jev.result()
                var sterms = List[G1Term]()
                var n_scaf = g1_spec_from_answers(sa, pool, bank, sterms)
                if n_scaf < 0:
                    print("  [spec] the model named no quantity — nothing done")
                    state = ST_IDLE
                else:
                    var v = g1_spec_admit(pool, bank, zbase, sterms, znew,
                                          n_scaf)
                    var what = g1_spec_describe(sterms, n_scaf)
                    print("  [spec]", what, " ESS", Int(v.ess))
                    if v.ok:
                        # ⚠ BLEND FROM WHERE WE ARE, like every other path:
                        # a novel `z` has no bank index, so `cur` stays put
                        # and `pending` is NOT set. `cur` is what the HUD,
                        # the keys and the context read as "running now", and
                        # pointing it at a stale entry would make the next
                        # "plus vite" resolve against the wrong command.
                        for k in range(D):
                            zpair.data[k] = zcur.data[k]
                            zpair.data[D + k] = Scalar[DT](znew[k])
                        blend_len = 25
                        blend_left = blend_len
                        ctx.began(String("spec"))
                        t_cmd = perf_counter_ns()
                        # ⚠ A NOVEL `z` MUST CANCEL THE PREVIOUS COMMAND'S
                        # TERMINATION, and leaving `cur` alone is exactly why
                        # it does not happen by itself. The locomotion
                        # timeout reads `bank.group_at(cur)`, so a foot-lift
                        # asked for while walking was overridden by the
                        # WALK's 5 s timeout a moment later — visible in the
                        # session log as `[spec] ... ESS 5089` followed
                        # immediately by `[timeout] locomotion -> stand`.
                        # ⚠ THE GUARD IS THE FLAG, NOT A FAR-FUTURE
                        # DEADLINE. Pushing `step_end_at` out would make
                        # `step_done` false for ever, and the timeout
                        # condition also reads it — so the NEXT `walk` would
                        # never end either. One flag, one meaning.
                        # ⚠ RECORD IT FOR OFFLINE PROMOTION. This is the
                        # only place a novel spec is known to have actually
                        # RUN, which is the signal worth recording — a spec
                        # that was refused is not a candidate.
                        if cand_path != "":
                            try:
                                if g1_spec_record(cand_path, pool, sterms,
                                                  n_scaf):
                                    print("  [spec] recorded a candidate")
                            except:
                                # a demo does not stop because a text file
                                # could not be written
                                pass
                        spec_active = True
                        # a chain queued by an earlier utterance is not what
                        # the robot was just asked for
                        q_name = List[String]()
                        q_conf = List[Float64]()
                        last_event = String("spec: ") + what
                        # ⚠ a novel command has no phrase in the table, so
                        # the robot says the QUANTITY it is about to move.
                        # Silence here reads as "it did not understand".
                        if not mute:
                            # ⚠ a novel command has no phrase in the table,
                            # so the robot says the QUANTITY it is about to
                            # move. Silence here reads as "it did not
                            # understand", which is the opposite of true.
                            _say_async(String("ok, ") + what)
                            mute_until = perf_counter_ns() + _ns(ECHO_TAIL_S)
                    elif v.nearest >= 0:
                        # the spec IS a bank entry: run the GATED row, which
                        # went through four gates and a CEM refinement this
                        # baseline has not.
                        print("   ", v.reason)
                        pending = v.nearest
                        pending_blend = 25
                        last_event = String("spec->") + bank.name_at(v.nearest)
                        if not mute:
                            var u3 = g1_command_phrase(bank.name_at(v.nearest))
                            for pi in range(len(ph_name)):
                                if ph_name[pi] == bank.name_at(v.nearest):
                                    u3 = ph_text[pi]
                            _say_async(u3)
                            mute_until = perf_counter_ns() + _ns(ECHO_TAIL_S)
                    else:
                        print("  [spec] REFUSED:", v.reason)
                    state = ST_IDLE

        elif state == ST_CHAT:
            # non-streaming, so `poll` returns "" throughout and `done`
            # is the signal. The robot keeps moving while it thinks.
            var ch_el = Float64(perf_counter_ns() - t_wait) / 1e9
            if ch_el > chat_dl:
                print("  [talk] GAVE UP after", _f2(ch_el),
                      "s — no reply came back.")
                try:
                    llm.cancel()
                except:
                    pass
                last_event = String("talk timed out")
                state = ST_IDLE
            else:
                _ = llm.poll()
            if state == ST_CHAT and llm.done():
                # ⚠ `result` RAISES on a failed call, and an unanswered
                # greeting must not take the renderer down.
                var rtxt = String("")
                try:
                    var rep = llm.result()
                    rtxt = rep.text
                except e:
                    print("  [talk] FAILED:", String(e))
                    last_event = String("talk failed")
                print("  [said]", rtxt)
                heard = rtxt
                if not mute and rtxt != "":
                    _say_async(rtxt)
                    mute_until = perf_counter_ns() + _ns(ECHO_TAIL_S)
                    last_event = String("replied")
                state = ST_IDLE

        # ── keys ──────────────────────────────────────────────────────
        # ⚠ SCANCODES, NOT KEYCODES — physical key POSITIONS, so this works
        # unchanged on AZERTY. The joystick learned that the hard way; the
        # row above QWERTY's `1..9` is the same row on AZERTY.
        var nkeys: Int32 = 0
        var kb = get_keyboard_state(Ptr(to=nkeys).as_unsafe_any_origin())
        for i in range(bank.count()):
            var sc = Int(Scancode.SCANCODE_1) + i if i < 9 else (
                Int(Scancode.SCANCODE_A) + (i - 9)
            )
            if kb[sc]:
                if pending != i:
                    last_event = String("key: ") + bank.name_at(i)
                pending = i
                pending_blend = 25
        # ⚠ EDGE-TRIGGERED. `get_keyboard_state` reports the key as HELD, so
        # a level test would start a new recording every frame it is down.
        var rec_key = kb[Int(Scancode.SCANCODE_TAB)]
        # ⚠ M CLOSES THE MICROPHONE, and it is not a convenience. An
        # always-open mic in a room of people is a privacy problem before it
        # is a false-trigger problem: everything said near this laptop would
        # otherwise be posted to a transcription service. `stop()` kills
        # ffmpeg, so the OS recording indicator goes out and nothing is
        # captured at all — not captured-and-ignored.
        var m_key = kb[Int(Scancode.SCANCODE_M)]
        if m_key and not prev_m:
            if mic_on:
                try:
                    mic.stop()
                except:
                    pass
                mic_on = False
                ring = List[Int16]()
                level = 0.0
                state = ST_IDLE
                last_event = String("mic CLOSED")
                print("  [mic] closed")
            else:
                try:
                    mic = MicCapture.start(16000, mic_dev)
                    mic_on = True
                    mic_dead = String("")
                    checked_silence = False
                    peak = 0.0
                    floor = 0.0010
                    last_event = String("mic open")
                    print("  [mic] reopened")
                except e:
                    mic_dead = String(e)
                    print("  [mic]", mic_dead)
        prev_m = m_key

        # TAB forces a segment to start, or ends the one in progress — the
        # manual override for a room too loud for the detector to close.
        if rec_key and not prev_key and mic_on and mic_dead == "":
            if state == ST_IDLE:
                heard = String("")
                pick = G1LangPick()
                seg = ring.copy()
                ring = List[Int16]()
                peak = 0.0
                t_rec = perf_counter_ns()
                quiet_since = perf_counter_ns()
                forced = False
                state = ST_REC
                last_event = String("listening...")
            elif state == ST_REC:
                forced = True
        prev_key = rec_key

        # ── the queue advances when the current step is done ──────────
        var step_done = (step_dist >= step_end_dist) if step_end_dist > 0.0 \
                        else (perf_counter_ns() >= step_end_at)
        if step_done and len(q_name) > 0 and state == ST_IDLE:
            var nxt = q_name[0]
            var ncf = q_conf[0]
            var rest_n = List[String]()
            var rest_c = List[Float64]()
            for i in range(1, len(q_name)):
                rest_n.append(q_name[i])
                rest_c.append(q_conf[i])
            q_name = rest_n^
            q_conf = rest_c^
            var ni = bank.find(nxt)
            if ni >= 0:
                pending = ni
                pending_blend = 25
                last_event = String("then: ") + nxt
                print("  [chain] ->", nxt, _f2(ncf))
                step_dist = 0.0
                # ⚠ a queued step gets the DEFAULT extent, not the first
                # step's. "Walk a metre and turn round" does not ask the
                # robot to turn round for a metre.
                if bank.group_at(ni) == "locomotion":
                    step_end_dist = 2.0
                    step_end_at = perf_counter_ns() + _ns(8.0)
                else:
                    step_end_dist = 0.0
                    step_end_at = perf_counter_ns() + _ns(3.0)
                if not mute:
                    var u2 = g1_command_phrase(nxt)
                    for pi in range(len(ph_name)):
                        if ph_name[pi] == nxt:
                            u2 = ph_text[pi]
                    _say_async(u2)
                    mute_until = perf_counter_ns() + _ns(ECHO_TAIL_S)

        # ⚠ LOCOMOTION HAS NO NATURAL END. A posture holds itself because `z`
        # persists; `walk` walks for ever. The bank's `group` is what decides.
        if (not spec_active and bank.group_at(cur) == "locomotion"
            and Float64(perf_counter_ns() - t_cmd) / 1e9 > walk_s
            and len(q_name) == 0 and step_done
            and state == ST_IDLE):
            var si2 = bank.find(String("stand"))
            if si2 >= 0 and cur != si2:
                pending = si2
                pending_blend = 20
                last_event = String("locomotion timed out -> stand")
                print("  [timeout] locomotion -> stand after", walk_s, "s")
                t_cmd = perf_counter_ns()

        if kb[Int(Scancode.SCANCODE_SPACE)]:
            var si = bank.find(String("stand"))
            if si >= 0 and pending != si:
                pending = si
                pending_blend = 15
                last_event = String("key: stand")

        if pending >= 0 and pending != cur:
            # start a blend from wherever we are NOW, not from the previous
            # target — a switch mid-blend would otherwise jump back.
            for k in range(D):
                zpair.data[k] = zcur.data[k]
                zpair.data[D + k] = Scalar[DT](bank.z_at(pending, k))
            blend_len = pending_blend if pending_blend > 0 else 1
            blend_left = blend_len
            cur = pending
            t_cmd = perf_counter_ns()
            # a bank command is running again, so `cur` is meaningful
            spec_active = False
            # ⚠ HERE, AND NOWHERE ELSE. This is the one site where the active
            # command changes — keys, channel, voice and chain steps all funnel
            # through it — so it is the only place the context can be kept
            # honest. Recording it at the decision instead would tell Jev the
            # robot is doing something it has not started, and a chain step
            # that never runs would enter the history.
            ctx.began(bank.name_at(cur))
            pending = -1
        elif pending == cur:
            pending = -1

        if blend_left > 0:
            var u = 1.0 - Float64(blend_left) / Float64(blend_len)
            g1_slerp_z[D](zpair, 0, 1, u, zcur, 0)
            blend_left -= 1

        # ── the policy ────────────────────────────────────────────────
        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
        for k in range(D):
            z1.data[k] = zcur.data[k]
        t.act[1](obs_t, z1, act_out)
        var a = ContAction[ACT]()
        for k in range(ACT):
            var v = Float64(act_out.data[k])
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            a.data[k] = v
        aobs.push(o, act_out)
        _ = env.step(a)
        g1_quantities(o, 0, quant, 0)
        # ⚠ DISTANCE IS INTEGRATED FROM THE POLICY'S OWN VELOCITY. There is
        # no odometry and nothing corrects it, so "one metre" means about one
        # metre — good enough for an instruction, not for anything that has
        # to be right. dt is the env's control step, not the frame time.
        var sp = quant[QV_SPEED_FWD]
        if sp < 0.0:
            sp = -sp
        step_dist += sp * 0.02

        # ── the UI ────────────────────────────────────────────────────
        var win_h = env.renderer_height()
        var ui = UI(
            env.renderer_mouse_x(), env.renderer_mouse_y(),
            env.renderer_take_click(),
        )
        ui.panel(0, 0, Float32(SIDEBAR_W), Float32(win_h))
        ui.label(12, 10, String("BFM-ZERO — TALK TO IT"), UI_HEAD, 1)
        ui.label(12, 26, String("just speak · M mute · TAB force"), UI_DIM, 1)
        # ⚠ THE STATE READOUT IS NOT DECORATION. Speech plus decision is
        # 1.5 s; without a visible state an audience sees a robot that moves
        # two seconds after you speak for no reason anyone can follow.
        var st_s = String("idle — press TAB")
        var st_c = UI_DIM
        if state == ST_REC:
            # the bar of the HUD is the segment so far, not a countdown —
            # there is no window to count down any more.
            var el = Float64(len(seg)) / 16000.0
            st_s = String("LISTENING ") + _f2(el) + String("s")
            st_c = UI_WARN
        elif state == ST_STT:
            # ⚠ WITH THE ELAPSED TIME AND THE BUDGET. "transcribing..." alone
            # cannot distinguish a 5 s call from a dead one, and a session was
            # lost to exactly that question.
            var e1 = Float64(perf_counter_ns() - t_wait) / 1e9
            st_s = String("transcribing ") + _f2(e1) + String("s / ") \
                   + _f2(stt_dl) + String("s")
            st_c = UI_WARN if e1 < stt_dl * 0.5 else UI_BAD
        elif state == ST_CHAT:
            var e3 = Float64(perf_counter_ns() - t_wait) / 1e9
            st_s = String("replying ") + _f2(e3) + String("s / ") \
                   + _f2(chat_dl) + String("s")
            st_c = UI_WARN if e3 < chat_dl * 0.5 else UI_BAD
        elif state == ST_JEV or state == ST_SPEC:
            var e2 = Float64(perf_counter_ns() - t_wait) / 1e9
            st_s = (String("deciding ") if state == ST_JEV
                    else String("writing a reward ")) + _f2(e2) \
                   + String("s / ") + _f2(jev_dl) + String("s")
            st_c = UI_WARN if e2 < jev_dl * 0.5 else UI_BAD
        ui.label(12, 40, st_s, st_c, 1)

        var hs = heard
        # ⚠ CODEPOINTS, NOT BYTES. The transcripts this project produces are
        # often French, and cutting mid-character would hand `draw_text` half
        # a UTF-8 sequence — which it degrades to two '?' rather than one.
        if hs.byte_length() > 27:
            var cut = String(heard[codepoint=0:27])
            hs = cut
        ui.label(12, 56, String("heard: ") + hs, UI_TXT, 1)
        if pick.name != "":
            ui.label(12, 70, String("-> ") + pick.name + String(" ")
                     + _f2(pick.conf), UI_OK, 1)
            if pick.needs_world > 0.5:
                ui.label(12, 84, String("(no object/destination)"), UI_WARN, 1)
        elif pick.reason != "":
            ui.label(12, 70, String("refused: ") + pick.reason, UI_WARN, 1)

        ui.label(12, 104, String("NOW"), UI_DIM, 1)
        ui.label(12, 120, bank.name_at(cur), UI_TXT, 1)
        ui.label(12, 134, String("grp  ") + bank.group_at(cur), UI_DIM, 1)
        ui.label(12, 148, String("hold ") + _f2(bank.hold_at(cur)), UI_DIM, 1)
        if blend_left > 0:
            ui.label(12, 162, String("blending ") + String(blend_left), UI_DIM, 1)

        ui.label(12, 182, String("MEASURED"), UI_DIM, 1)
        ui.label(12, 198, String("root  ") + _f2(quant[QV_BODY_H]), UI_TXT, 1)
        ui.label(12, 212, String("speed ") + _f2(quant[QV_SPEED]), UI_TXT, 1)
        ui.label(12, 226, String("up    ") + _f2(quant[QV_UPRIGHT]), UI_TXT, 1)
        ui.label(12, 240, String("yaw   ") + _f2(quant[QV_YAW_RATE]), UI_TXT, 1)
        ui.label(12, 254, String("hand L") + _f2(quant[QV_LHAND_H])
                 + String(" R") + _f2(quant[QV_RHAND_H]), UI_TXT, 1)

        # ⚠ NO COMMAND GRID. Eighteen buttons were how you drove this before
        # there was a microphone; they crowd the panel and invite clicking
        # rather than speaking. The keys still work for a rehearsed demo —
        # they are just not advertised here.
        var by = Float32(278)
        # ── the level meter ───────────────────────────────────────────
        # ⚠ THIS IS THE ANSWER TO "is the microphone working?", and it is
        # available BEFORE you speak. A bar that never moves is a dead input;
        # one that moves when you talk means everything upstream of Whisper
        # is fine. It cost a day to not have it.
        ui.label(12, by, String("MIC"), UI_DIM, 1)
        var mw = Float32(SIDEBAR_W - 24)
        ui.panel(12, by + 16, mw, 12)
        var lvl = level * 6.0              # speech sits low in a 0-1 RMS
        if lvl > 1.0:
            lvl = 1.0
        if lvl > 0.01:
            ui.panel(12, by + 16, mw * Float32(lvl), 12)
        # a thin peak marker — the loudest moment, so a spike that came and
        # went is still visible a frame later
        var pk = peak * 6.0
        if pk > 1.0:
            pk = 1.0
        if pk > 0.02:
            ui.panel(12 + mw * Float32(pk) - 2, by + 16, 2, 12)
        ui.label(12, by + 32, String("rms ") + _f4(level)
                 + String(" open>") + _f4(open_at), UI_DIM, 1)
        # the open threshold on the bar's own scale, so you can see how far a
        # voice is from starting a segment
        var thr = open_at * 6.0
        if thr > 1.0:
            thr = 1.0
        if mic_on and thr > 0.01:
            ui.panel(12 + mw * Float32(thr), by + 14, 1, 16)
        if not mic_on:
            ui.label(12, by + 48, String("MIC CLOSED — M opens"), UI_WARN, 1)
        elif mic_dead != "":
            var md = mic_dead
            if md.byte_length() > 26:
                var mcut = String(mic_dead[codepoint=0:26])
                md = mcut
            ui.label(12, by + 48, String("MIC: ") + md, UI_WARN, 1)
        elif not checked_silence:
            ui.label(12, by + 48, String("checking input..."), UI_DIM, 1)
        else:
            ui.label(12, by + 48, String("live · floor ") + _f4(floor),
                     UI_OK, 1)

        ui.label(12, by + 68, last_event, UI_DIM, 1)
        env.set_ui(ui.rects, ui.texts)

        env.render_frame()
        if delay_ms > 0:
            env.renderer_delay(delay_ms)
        step += 1

    env.close()
    print("  ", step, "steps")
