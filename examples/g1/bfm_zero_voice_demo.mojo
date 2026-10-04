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
# ⚠ `remove_file`, `file_size`, `write_text_atomic` and the whole of
# `noeira.io.proc` went with the flag-file echo gate — see the note above
# `_say`. Four imports fewer is the shape of the simplification: the gate used
# to be a /tmp file, a touch, a backgrounded shell and a `say -f` dance, and
# it is now one method call on `LocalVoice`.
from noeira.io.fileio import read_file_bytes
from noeira.core.bytes import string_from_bytes
from noeira.envs.robots.g1_voice_loop import (
    G1VoiceLoop, G1VoiceConfig, G1VoiceEvent,
    VL_COMMAND, VL_SPEC, VL_WORLD, VL_TALK, VL_REFUSED,
    VL_STT_DEADLINE_S, VL_JEV_DEADLINE_S, VL_CHAT_DEADLINE_S,
)
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


# ⚠ THE TAIL SURVIVES THE REWRITE, but for a different reason than it was
# first written — see the note at the gate. It is a CAPTURE-BUFFER DRAIN, not
# an estimate of how long an utterance takes.
comptime ECHO_TAIL_S: Float64 = 0.45


def _ns(seconds: Float64) -> Int:
    """Seconds to nanoseconds, via milliseconds.

    ⚠ `Int(x * 1e9)` does not compile here — the literal drags the product
    into a SIMD type that `Int` will not take. Going through an integer
    millisecond keeps it in Int arithmetic, and a millisecond is finer than
    anything this schedules."""
    return Int(seconds * 1000.0) * 1_000_000


# ⚠ THE ECHO GATE IS NOW `LocalVoice.speaking()` — AN EXACT SIGNAL, AND BOTH
# THINGS IT REPLACES WERE WRONG IN DIFFERENT WAYS (§12.55, §12.56).
#
# First it was an ESTIMATE of `say`'s duration from the text at ~14 characters
# a second. That gave 1.24 s for "spin_left", which takes **2.79 s**, while
# "both_hands_up" — half again as long in characters — takes **2.20 s**. The
# estimate does not even ORDER them correctly, because `say` has a start-up
# cost that dominates a short word and it pronounces the underscore.
#
# Then it was a FLAG FILE, which was exact per utterance and raced across
# them: two overlapping `say` shells share one file, both remove it, and the
# FIRST to finish opens the microphone while the SECOND is still audibly
# talking. That is how "je tourne à gauche" came back through Whisper and
# re-picked `spin_left`. The workaround was to DROP a second utterance while
# one was running — a cosmetic loss, accepted only because the race was worse.
#
# `LocalVoice` (noeira/ai/audio_io.mojo, commit ecc8f2657 from the
# `noeira/ai/` session) ends both: `speaking()` reads the child's stdout
# closing, so it is exact with no timer and no shared state, and `say()` cuts
# off the previous utterance itself — so the drop-the-second workaround goes
# too, and a chained step can speak over its predecessor as it should.
#
# It also takes ANY TEXT. `quote_arg` refuses a single quote, which is sound
# for a path and fatal for French — "je m'accroupis" took the whole demo down
# — so this file used to write the text to a file and call `say -f`. That is
# now inside `LocalVoice`, and the two /tmp paths it needed are gone.


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
    # ⚠ THE DEADLINES ARE FLAGS SO THE GIVE-UP PATH CAN BE EXERCISED. It
    # fires only when a call hangs, which is exactly the condition that cannot
    # be summoned on demand — so set a tiny one and any NORMAL call blows it,
    # which tests the RECOVERY rather than the hang:
    #
    #     --stt-deadline 0.5     # speak; it must give up and keep running
    #
    # The user ran exactly that: 0.5 was too harsh, every call gave up, the
    # demo kept running, and the path is therefore verified. ⚠ 2 s is below
    # the observed 5.4 s Whisper tail, so it clips slow calls; the default is
    # for use rather than for testing.
    var stt_dl = Float64(String(_flag(String("--stt-deadline"),
                                      String(VL_STT_DEADLINE_S))))
    var jev_dl = Float64(String(_flag(String("--jev-deadline"),
                                      String(VL_JEV_DEADLINE_S))))
    var chat_dl = Float64(String(_flag(String("--chat-deadline"),
                                       String(VL_CHAT_DEADLINE_S))))
    # ⚠ THE POOL IS OPT-IN AND LOADED BY THE LOOP, at construction rather than
    # on the first miss: the sidecar is 70 MB and `G1Pool.__init__` sorts 14
    # columns of 65 536, which mid-frame would freeze the render loop — and
    # with it the physics step — for about a second (§12.60).
    # ⚠ `--stt groq` IS THE ESCAPE HATCH for the vocabulary hint: only a
    # multipart backend takes a TEXT prompt, and the HF endpoint RAISES on one
    # (§12.56). The loop decides which to send it to.
    var stt_spec = _flag(String("--stt"), String("hf"))
    var spec_pool_path = _flag(String("--pool"), String(""))
    var cand_path = _flag(String("--candidates"),
                          String("g1_spec_candidates.txt"))

    # ⚠ THE WHOLE VOICE PATH IS ONE OBJECT NOW — `G1VoiceLoop`. It owns the
    # microphone, the detector, the recogniser, the decision, the reward-spec
    # route and the text-to-speech, because that set is where every bug of
    # §12.55 to §12.63 lived and a second copy of it would diverge. The room
    # session's binary owns one of these too, so the next detector fix is made
    # once (§12.64).
    #
    # This file keeps what only it knows: the latent, the blend, the bank
    # index, the chain's distance and clock, the locomotion timeout and the
    # HUD. `poll` returns a decision; scheduling it is the robot's business.
    var vcfg = G1VoiceConfig()
    vcfg.lang = lang
    vcfg.stt_spec = stt_spec
    vcfg.vad = vad
    vcfg.mute = mute
    vcfg.hang_s = hang_s
    vcfg.min_seg = min_seg
    vcfg.max_seg = max_seg
    vcfg.max_none = max_none
    vcfg.min_top = min_top
    vcfg.stt_dl = stt_dl
    vcfg.jev_dl = jev_dl
    vcfg.chat_dl = chat_dl
    vcfg.pool_path = spec_pool_path
    vcfg.cand_path = cand_path
    vcfg.mic_dev = mic_dev
    vcfg.llm_spec = llm_spec
    # ⚠ THE VOCABULARY IS SENT ONLY TO A BACKEND THAT TAKES ONE, and the loop
    # decides — HF's Whisper accepts a prompt as token ids, never text, and
    # the client RAISES on one. Passing it unconditionally is what made this
    # demo armed to crash at the first transcription (§12.56).
    vcfg.vocab = _flag(String("--vocab"), String(
        "marche, cours, recule, tourne à gauche, tourne à droite,"
        " accroupis-toi, lève le bras droit, lève le bras gauche,"
        " lève les deux bras, écarte les bras, regarde à gauche,"
        " regarde à droite, arrête-toi, baisse les bras"
    ))
    vcfg.chat_sys = String(
        "You are a small humanoid robot in a simulator. Reply in ONE short"
        " sentence, in the language you were addressed in. You can walk,"
        " turn, squat and raise your arms, but you cannot pick things up or"
        " go anywhere in particular. Be warm and brief."
    )
    var vl = G1VoiceLoop(vcfg^, bank)
    vl.start()
    print("  voice loop ready — mic", "default" if mic_dev == "" else mic_dev,
          "| spec path", "ON" if spec_pool_path != "" else "off")
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

    # ⚠ the microphone, the voice, the detector's state and the floor are all
    # inside `vl` now. What is left here is the KEY EDGE state, because the
    # keyboard belongs to the renderer and not to the voice path.
    var prev_m = False
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

        # ── the voice loop: one poll, five outcomes ───────────────────
        # ⚠ EVERYTHING BETWEEN THE MICROPHONE AND A DECISION IS IN
        # `G1VoiceLoop` NOW. What stays here is what only this file knows: the
        # latent, the blend, the bank index, the chain's distance and clock,
        # and the locomotion timeout. See §12.64 — the split is that the loop
        # owns the AUDIO and this owns the ROBOT.
        #
        # ⚠ `ctx.since_s` IS SET HERE, not in the loop, because it is measured
        # from the robot's own command clock which the loop does not have.
        ctx.since_s = Float64(perf_counter_ns() - t_cmd) * 1e-9
        var ev = vl.poll(ctx, bank)
        var vnote = vl.take_note()
        if vnote != "":
            print("  " + vnote)

        if ev.kind == VL_COMMAND:
            var idx2 = bank.find(ev.name)
            if idx2 >= 0:
                pending = idx2
                pending_blend = 25
                last_event = String("voice: ") + ev.name
                t_cmd = perf_counter_ns()
                # the first step's extent comes from the utterance
                extent = ev.extent
                step_dist = 0.0
                if bank.group_at(idx2) == "locomotion":
                    step_end_dist = g1_extent_metres(extent)
                    # ⚠ AND A CLOCK AS WELL. A robot that is blocked, or
                    # walking on the spot, would never reach its distance and
                    # the chain would hang for ever.
                    step_end_at = perf_counter_ns() + _ns(g1_extent_seconds(extent) * 3.0 + 4.0)
                else:
                    step_end_dist = 0.0
                    step_end_at = perf_counter_ns() + _ns(g1_extent_seconds(extent))
                q_name = List[String]()
                q_conf = List[Float64]()
                for si in range(len(ev.chain)):
                    q_name.append(ev.chain[si])
                    q_conf.append(ev.chain_conf[si])
                if len(q_name) > 0:
                    var qs = String("")
                    for si in range(len(q_name)):
                        qs += String(" -> ") + q_name[si]
                    print("  [chain]", len(q_name), "more:", qs,
                          " extent", _f2(extent))
                if not mute:
                    # ⚠ SAY THE PART IT CANNOT DO. A robot that silently
                    # performs 30 % of an instruction is worse than one that
                    # performs 30 % and says so.
                    var utter = g1_command_phrase(ev.name)
                    for pi in range(len(ph_name)):
                        if ph_name[pi] == ev.name:
                            utter = ph_text[pi]
                    if ev.needs_world > 0.5:
                        utter = utter + String(
                            ", but I have no object or destination")
                    vl.say(utter)

        elif ev.kind == VL_SPEC:
            # ⚠ BLEND FROM WHERE WE ARE, and leave `cur` ALONE: a novel `z`
            # has no bank index, and `cur` is what the HUD, the keys and the
            # context read as "running now". Pointing it at a stale entry
            # would make the next "plus vite" resolve against a command the
            # robot is not doing (§12.56).
            if len(ev.z) == D:
                for k in range(D):
                    zpair.data[k] = zcur.data[k]
                    zpair.data[D + k] = Scalar[DT](ev.z[k])
                blend_len = 25
                blend_left = blend_len
                ctx.began(String("spec"))
                t_cmd = perf_counter_ns()
                # ⚠ A NOVEL `z` MUST CANCEL THE PREVIOUS COMMAND'S
                # TERMINATION, and leaving `cur` alone is exactly why it does
                # not happen by itself: the locomotion timeout reads
                # `bank.group_at(cur)`, so a foot-lift asked for while walking
                # was overridden by the WALK's 5 s timeout a moment later. The
                # guard is the FLAG, not a far-future deadline — the timeout
                # also reads `step_done`, so pushing that out would stop the
                # NEXT walk from ever ending (§12.62).
                spec_active = True
                q_name = List[String]()
                q_conf = List[Float64]()
                last_event = String("spec: ") + ev.name

        elif ev.kind == VL_WORLD:
            # ⚠ A DESTINATION HAS NO HANDLER IN THIS BINARY, and saying so is
            # the point rather than a gap. `g1_decide` routes it here instead
            # of to the reward-spec path because world position is NOT in the
            # state pool — a reward over it returns a unit-norm `z` and a
            # confidently wrong robot (§3.2). The room session's binary owns a
            # planner and takes this event; this one has no scene, so it
            # reports and keeps doing what it was doing.
            last_event = String("world: ") + ev.destination
            print("  [world]", ev.destination,
                  "— no navigator in this binary; the room demo takes this.")
            if not mute:
                vl.say(String("I cannot go to the ") + ev.destination
                       + String(" from here"))

        elif ev.kind == VL_TALK:
            # already spoken by the loop, which owns the voice
            last_event = String("replied")

        elif ev.kind == VL_REFUSED:
            # ⚠ NEVER SPOKEN ALOUD: "no such command" is heard, transcribed
            # and refused again — the loop that filled a whole session's log.
            last_event = String("refused: ") + ev.text

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
        # a level test would fire every frame it is down.
        var rec_key = kb[Int(Scancode.SCANCODE_TAB)]
        var m_key = kb[Int(Scancode.SCANCODE_M)]
        if m_key and not prev_m:
            # ⚠ M really CLOSES the device — `vl.toggle_mic` kills ffmpeg, so
            # the OS recording indicator goes out and nothing is captured at
            # all, not captured-and-ignored. That is a privacy property.
            var mn = vl.toggle_mic()
            print("  " + mn)
            last_event = mn
        prev_m = m_key

        # TAB forces a segment to start, or ends the one in progress — the
        # manual override for a room too loud for the detector to close.
        if rec_key and not prev_key:
            vl.force()
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
                    # ⚠ THROUGH THE LOOP, which owns the echo gate — `say`
                    # sets the drain itself, so there is no second place that
                    # has to remember to.
                    vl.say(u2)

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
        # ⚠ THE LOOP OWNS THE WORDS, THIS FILE OWNS THE COLOURS. The status
        # text carries the ELAPSED TIME AND THE BUDGET — "transcribing 4.21s /
        # 20.00s" — because "transcribing..." alone cannot distinguish a 5 s
        # call from a dead one, and a session was lost to exactly that
        # question (§12.60).
        var st_s = vl.status_line()
        # ⚠ `slvl`, not `lvl` — the level METER below uses that name, and two
        # different levels under one name is how a HUD lies.
        var slvl = vl.status_level()
        var st_c = UI_DIM
        if slvl == 1:
            st_c = UI_WARN
        elif slvl == 2:
            st_c = UI_BAD
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
        var level = vl.meter()
        var peak = vl.peak_level()
        var open_at = vl.open_threshold()
        var mic_on = vl.mic_open()
        var mic_dead = vl.mic_error()
        var checked_silence = vl.mic_checked()
        var floor = vl.noise_floor()
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
