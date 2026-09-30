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
    G1_NVOC, g1_quantities, QV_BODY_H, QV_SPEED, QV_UPRIGHT,
    QV_LHAND_H, QV_RHAND_H, QV_YAW_RATE,
)
from noeira.render.sdl import Ptr
from noeira.render.types import Color
from noeira.render.ui import UI
from noeira.render.sdl.sdl_scancode import Scancode
from noeira.render.sdl.sdl_keyboard import get_keyboard_state
from noeira.io.fileio import remove_file, file_size
from noeira.io.proc import run_system, quote_arg
from noeira.ai.audio_io import MicCapture, rms
from noeira.io.wav import WavAudio
from noeira.ai.jev import JevClient
from noeira.ai.speech import SpeechToText
from noeira.envs.robots.g1_command_language import (
    g1_command_questions, g1_decide, G1LangPick,
)

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

comptime ST_IDLE: Int = 0
comptime ST_REC: Int = 1
comptime ST_STT: Int = 2
comptime ST_JEV: Int = 3

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


def _say_async(text: String) raises:
    """Speak in the background, and leave a flag behind that says so.

    `say` blocks for as long as it speaks, so it cannot run in the loop. The
    flag is written HERE rather than inside the backgrounded shell, or there
    would be a window between the spawn and the flag appearing in which the
    gate is open and the microphone is already live."""
    _ = run_system("touch " + quote_arg(String(SPEAK_FLAG)))
    _ = run_system(
        "( say " + quote_arg(text) + " ; rm -f "
        + quote_arg(String(SPEAK_FLAG)) + " ) >/dev/null 2>&1 &"
    )

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
    var stt = SpeechToText.huggingface()
    if lang != "":
        stt.language = lang
        print("  whisper language pinned to", lang)
    jev.warm_up()
    stt.warm_up()
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
    var mic_dead = String("")
    var checked_silence = False
    var mic_on = True
    var floor = 0.0010
    var quiet_since = perf_counter_ns()
    var prev_m = False
    var forced = False
    var above = 0
    # ⚠ while this is in the future, the microphone is the robot's own voice
    var mute_until = perf_counter_ns()

    var state = ST_IDLE
    var heard = String("")
    var pick = G1LangPick()
    var t_rec = perf_counter_ns()
    var t_cmd = perf_counter_ns()
    var prev_key = False

    var chan = G1CommandChannel(chan_path)
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
            mute_until = perf_counter_ns() + Int(ECHO_TAIL_S * 1e9)
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
        var close_at = floor * VAD_CLOSE_MULT

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
                    print("  [stt]", len(seg), "samples (", _f2(seg_s),
                          "s ), peak", _f4(peak),
                          "— max-seg" if over else "")
                    state = ST_STT

        elif state == ST_STT:
            if stt.poll():
                var tr = stt.result()
                heard = tr.text
                print("  [heard]", heard, "(", tr.latency_ms, "ms )")
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
                    jev.start_text(String("instruction: ") + heard, quest)
                    state = ST_JEV

        elif state == ST_JEV:
            if jev.poll():
                var ans = jev.result()
                pick = g1_decide(ans, bank, max_none, min_top, 0.5, True)
                if pick.name != "":
                    var idx2 = bank.find(pick.name)
                    if idx2 >= 0:
                        pending = idx2
                        pending_blend = 25
                        last_event = String("voice: ") + pick.name
                        t_cmd = perf_counter_ns()
                        print("  [pick]", pick.name, _f2(pick.conf),
                              " P(none)", _f2(pick.p_none),
                              " world", _f2(pick.needs_world))
                        if not mute:
                            # ⚠ SAY THE PART IT CANNOT DO. A robot that
                            # silently performs 30 % of an instruction is
                            # worse than one that performs 30 % and says so.
                            var utter = pick.name
                            if pick.needs_world > 0.5:
                                utter = pick.name + String(
                                    ", but I have no object or destination")
                            _say_async(utter)
                            mute_until = perf_counter_ns() + Int(
                                ECHO_TAIL_S * 1e9)
                else:
                    last_event = String("refused: ") + pick.reason
                    print("  [refused]", pick.reason, " P(none)",
                          _f2(pick.p_none), " addressed", _f2(pick.addressed))
                    # ⚠ A SPOKEN REFUSAL IS THE WORST THING TO SAY ALOUD.
                    # "no such command" is heard, transcribed, and refused
                    # again — the loop that filled a whole session's log. The
                    # HUD already says it; saying it too buys nothing and
                    # costs a Whisper call every time.
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

        # ⚠ LOCOMOTION HAS NO NATURAL END. A posture holds itself because `z`
        # persists; `walk` walks for ever. The bank's `group` is what decides.
        if (bank.group_at(cur) == "locomotion"
            and Float64(perf_counter_ns() - t_cmd) / 1e9 > walk_s
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
            st_s = String("transcribing...")
            st_c = UI_WARN
        elif state == ST_JEV:
            st_s = String("deciding...")
            st_c = UI_WARN
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
