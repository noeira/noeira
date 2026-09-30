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
from noeira.io.fileio import read_file_bytes, remove_file
from noeira.io.proc import run_system, quote_arg
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


def _rec_start(path: String, seconds: Float64) raises:
    """Spawn ffmpeg DETACHED and have the shell rename on success.

    ⚠ `audio_io.record_wav` blocks for the whole duration, and four seconds of
    frozen renderer is four seconds of frozen physics. ⚠ The rename is what
    makes the poll safe: without it the loop can read a WAV ffmpeg is still
    writing. Same discipline as the command channel's atomic writes.
    """
    try:
        remove_file(path)
    except:
        pass
    var part = path + String(".part")
    var cmd = (
        "( ffmpeg -hide_banner -loglevel error -y -f avfoundation"
        " -i " + quote_arg(String(":default"))
        + " -t " + String(seconds) + " -ac 1 -ar 16000 -sample_fmt s16 "
        + quote_arg(part) + " && mv " + quote_arg(part) + " "
        + quote_arg(path) + " ) >/dev/null 2>&1 &"
    )
    _ = run_system(cmd)


def _say_async(text: String) raises:
    """`say` blocks for as long as it speaks, so it goes to the background
    too. Fire and forget: a missed confirmation is not worth a dropped
    frame."""
    _ = run_system("say " + quote_arg(text) + " >/dev/null 2>&1 &")

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
    var rec_s = Float64(String(_flag(String("--record"), String("4"))))
    var walk_s = Float64(String(_flag(String("--walk-seconds"), String("5"))))
    var max_none = Float64(String(_flag(String("--max-none"), String("0.25"))))
    var min_top = Float64(String(_flag(String("--min-top"), String("0.35"))))
    var mute = _has(String("--mute"))
    var wav = String("/tmp/noeira_g1_voice.wav")
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
    jev.warm_up()
    stt.warm_up()
    var quest = g1_command_questions(bank, True)
    print("  jev + whisper warmed")

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

    print("  TAB = speak (", rec_s, "s), keys 1-9 / a-i pick, SPACE = stand, ESC quits")
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
        if state == ST_REC:
            # ffmpeg renames the file into place when it finishes, so its
            # mere existence means a COMPLETE wav.
            var ok_wav = False
            var bytes = List[UInt8]()
            try:
                bytes = read_file_bytes(wav)
                ok_wav = len(bytes) > 44        # past the WAV header
            except:
                pass
            if ok_wav:
                stt.start_wav_bytes(bytes)
                state = ST_STT
                print("  [stt] ", len(bytes), "bytes")
            elif Float64(perf_counter_ns() - t_rec) / 1e9 > rec_s + 6.0:
                # ⚠ ffmpeg can fail silently (no mic permission is the usual
                # one) and would otherwise leave the demo stuck in RECORDING
                # with no way back.
                heard = String("(no audio — mic permission?)")
                state = ST_IDLE
                print("  [rec] timed out — no wav. mic permission?")

        elif state == ST_STT:
            if stt.poll():
                var tr = stt.result()
                heard = tr.text
                print("  [heard]", heard, "(", tr.latency_ms, "ms )")
                if heard.byte_length() == 0:
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
                            if pick.needs_world > 0.5:
                                _say_async(pick.name
                                    + String(", but I have no object or destination"))
                            else:
                                _say_async(pick.name)
                else:
                    last_event = String("refused: ") + pick.reason
                    print("  [refused]", pick.reason, " P(none)",
                          _f2(pick.p_none), " addressed", _f2(pick.addressed))
                    if not mute:
                        _say_async(pick.reason)
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
        if rec_key and not prev_key and state == ST_IDLE:
            heard = String("")
            pick = G1LangPick()
            _rec_start(wav, rec_s)
            t_rec = perf_counter_ns()
            state = ST_REC
            last_event = String("listening...")
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
        ui.label(12, 26, String("TAB to speak, 18 commands"), UI_DIM, 1)
        # ⚠ THE STATE READOUT IS NOT DECORATION. Speech plus decision is
        # 1.5 s; without a visible state an audience sees a robot that moves
        # two seconds after you speak for no reason anyone can follow.
        var st_s = String("idle — press TAB")
        var st_c = UI_DIM
        if state == ST_REC:
            st_s = String("LISTENING...")
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

        var by = Float32(278)
        for i in range(bank.count()):
            var bx = Float32(12) if i % 2 == 0 else Float32(126)
            var byy = by + Float32((i // 2) * 24)
            if ui.button(bx, byy, 102, 20, bank.name_at(i), i == cur, 1):
                pending = i
                pending_blend = 25
                last_event = String("click: ") + bank.name_at(i)
        var ey = by + Float32(((bank.count() + 1) // 2) * 24) + 10
        ui.label(12, ey, last_event, UI_DIM, 1)
        env.set_ui(ui.rects, ui.texts)

        env.render_frame()
        if delay_ms > 0:
            env.renderer_delay(delay_ms)
        step += 1

    env.close()
    print("  ", step, "steps")
