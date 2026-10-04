# +--------------------------------------------------------------------------+ #
# | The bank, driven — keyboard, mouse, or anything that can write a file
# +--------------------------------------------------------------------------+ #
"""Play the 18 gated commands, and take orders over the channel.

    pixi run mojo build -I . -Xlinker -ld_classic \
        examples/g1/bfm_zero_bank_viewer.mojo -o build/g1play
    ./build/g1play --ckpt runs/<id>/checkpoints/step_36000.ckpt

    # from any other terminal, while it runs:
    echo "seq 1\\ncmd squat\\nblend 25" > /tmp/g1_cmd

No store, no `B`, no pool, no prompt: the bank already holds the `z` for
every command, so this loads a checkpoint and a text file and starts. The
latent search that made those vectors hold (§12.52) was paid for offline.

## What drives it

- **keys** 1-9 and a-i pick a command; SPACE is `stand`.
- **buttons** in the sidebar, one per bank entry.
- **the channel** — `--channel PATH`, polled every frame. Any process that
  can write a file can drive the robot: a chat client, speech-to-text, a
  language model, `echo`.

⚠ THE CHANNEL IS WHY THIS IS A SEPARATE PROGRAM FROM THE JOYSTICK. Speech is
~1.1 s and a constrained-readout model ~0.3 s; the loop has 20 ms. Nothing
slow may run inline, so the slow things run elsewhere and write a line. See
`noeira/envs/robots/g1_command_channel.mojo`.

## Transitions

A command switch slerps `z` over `blend` frames rather than snapping. The
`z`s are on the radius-16 sphere and two bank entries can be 70 degrees
apart; jumping between them makes the policy lurch, which reads as a bug in
the robot rather than in the driving. `g1_slerp_z` is the same routine
§12.47's trajectory optimisation uses between knots.

⚠ AN UNKNOWN COMMAND CHANGES NOTHING. `G1CommandBank.find` returns -1 for a
name that was rejected at build time or never existed, and this keeps doing
what it was doing and acks `unknown`. A language layer that guessed at the
nearest entry is how a robot ends up doing something nobody asked for.
"""

from std.math import abs
from std.sys import argv

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

    var chan = G1CommandChannel(chan_path)
    var last_event = String("(waiting)")
    var pending = -1
    var pending_blend = 25

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
        ui.label(12, 10, String("BFM-ZERO COMMAND BANK"), UI_HEAD, 1)
        ui.label(12, 26, String("18 gated commands"), UI_DIM, 1)
        ui.label(12, 38, String("z from the bank, no prompt"), UI_DIM, 1)

        ui.label(12, 58, String("NOW"), UI_DIM, 1)
        ui.label(12, 74, bank.name_at(cur), UI_TXT, 1)
        ui.label(12, 88, String("grp  ") + bank.group_at(cur), UI_DIM, 1)
        ui.label(12, 102, String("hold ") + _f2(bank.hold_at(cur)), UI_DIM, 1)
        if blend_left > 0:
            ui.label(12, 116, String("blending ") + String(blend_left), UI_DIM, 1)

        ui.label(12, 136, String("MEASURED"), UI_DIM, 1)
        ui.label(12, 152, String("root  ") + _f2(quant[QV_BODY_H]), UI_TXT, 1)
        ui.label(12, 166, String("speed ") + _f2(quant[QV_SPEED]), UI_TXT, 1)
        ui.label(12, 180, String("up    ") + _f2(quant[QV_UPRIGHT]), UI_TXT, 1)
        ui.label(12, 194, String("yaw   ") + _f2(quant[QV_YAW_RATE]), UI_TXT, 1)
        ui.label(12, 208, String("hand L") + _f2(quant[QV_LHAND_H])
                 + String(" R") + _f2(quant[QV_RHAND_H]), UI_TXT, 1)

        var by = Float32(232)
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
