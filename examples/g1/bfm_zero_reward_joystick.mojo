""""Drive the G1 by WRITING A REWARD, live, from the keyboard (§12.45).

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/bfm_zero_reward_joystick.mojo -o /tmp/g1joy
    pixi run /tmp/g1joy --ckpt runs/<id>/checkpoints/step_36000.ckpt

## ⚠ THIS IS NOT A CLIP PLAYER

`bfm_zero_policy_viewer.mojo` replays a motion: the prompt is
`z_t = project(B(reference row t+1))`, so the robot is being told to be a
recording. That shows a tracking controller. It does not show a behavioural
foundation model, because a tracking controller could do it too.

This does the thing only a BFM can. Because `M = F.B^T` is a successor
measure, the optimal policy for ANY reward `r` is `pi_z` at

    z = E_rho[ B(s).r(s) ]

— an identity, not an approximation. So a keypress changes the OBJECTIVE, a
reward nobody trained on, and the policy for it appears with no gradient step
and no data collection. The analogue of RLHF here is a matrix-vector product.

## How the loop stays interactive

`B(s)` over the state pool is computed ONCE, at startup: a `[POOL, D]`
matrix. A keypress then costs one `D x POOL` matvec plus a projection —
~1 MFLOP at the default sizes, microseconds. Nothing is retrained, reloaded
or re-encoded, which is why the command can follow the keyboard.

⚠ `backward_embed` takes its row count as a COMPTIME parameter, so `POOL` is
fixed at build time and the store's 441 131 rows are SUBSAMPLED into it on a
fixed stride. More rows is a better estimate of `E_rho`, not a different
objective.

## The reward

`local_body_vel` and `local_body_ang_vel` are in the HEADING frame (see
`unitree_g1_priv_obs.mojo`), which is exactly the frame a joystick command
wants: "forward" is the robot's forward, whichever way it is facing. Body 0
is the root, so the command reads six floats out of a vector `B` already
consumes:

    r(s) = exp(-||v_xy(s) - v*||^2 / SIGMA_V^2) * exp(-(w_z(s) - w*)^2 / SIGMA_W^2)

Bounded, non-negative, and peaked at the command. ⚠ The reward is computed on
the RAW rows, before the normaliser: `r` is a statement about physical
velocity, and `E_rho[B.r]` only means what it says if `r` does.

⚠ CPU PHYSICS AND A CPU POLICY. Same path as the viewer — `ctx=None`, the
float64 single-env. The GPU draws and nothing else.
"""

from std.math import exp, sqrt, acos, abs
from std.math import cos as _cos64, sin as _sin64
from std.sys import argv

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.core.cont_action import ContAction
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.z_sampler import z_from_reward
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
)
from noeira.envs.robots.unitree_g1_priv_obs import (
    G1_PRIV_OFF_VEL, G1_PRIV_OFF_ANGVEL,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
    UNITREE_G1_PRIV_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_D, G1_H, G1_L, G1_HB, g1_project_z,
)
from noeira.render.sdl.sdl_keycode import Keycode
from noeira.render.sdl.sdl_scancode import Scancode
from noeira.render.sdl.sdl_keyboard import get_keyboard_state
from noeira.render.sdl import c_int, Ptr
from noeira.render.ui import UI
from noeira.render.types import Color

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV

comptime POOL: Int = 4096

# The command features, as offsets into the packed 527-wide row. The
# privileged block starts where the state block ends; body 0 is the root.
comptime OFF_VX: Int = UNITREE_G1_STATE_DIM + G1_PRIV_OFF_VEL
comptime OFF_WZ: Int = UNITREE_G1_STATE_DIM + G1_PRIV_OFF_ANGVEL + 2

comptime SIGMA_V: Float64 = 0.5
comptime SIGMA_W: Float64 = 0.5

comptime SIDEBAR_W: Int = 240
# ⚠ THE TEXT BUDGET, IN CHARACTERS. `draw_text` advances 8*scale pixels per
# character, so at scale 1 a label starting at x=12 has (SIDEBAR_W-24)/8 = 27
# characters before it runs off the panel and into the 3D viewport. Nothing
# clips it for you — the text pass sets a FULL-WINDOW viewport on purpose so
# the HUD is not cut by the scene inset — so every string below is either
# short by construction or passed through `_fit`.
comptime SIDE_CHARS: Int = (SIDEBAR_W - 24) // 8
comptime PAD_W: Float32 = Float32(SIDEBAR_W - 24)
comptime PAD_H: Float32 = 150.0
comptime LOG_ROWS: Int = 6

# A stick, in units per 50 Hz control step. `RAMP` is how fast leaning on a key
# moves the command; `DECAY` is how fast letting go centres it. Decay is the
# faster of the two on purpose: "stop" should feel immediate, "go" should not.
comptime RAMP_V: Float64 = 0.030
comptime DECAY_V: Float64 = 0.060
comptime RAMP_W: Float64 = 0.050
comptime DECAY_W: Float64 = 0.100
comptime MAX_FWD: Float64 = 1.40
comptime MAX_BACK: Float64 = 0.80
comptime MAX_LAT: Float64 = 0.60
comptime MAX_YAW: Float64 = 1.60
# One click is worth a quarter second of holding the key: enough to feel like
# a control, small enough that the pad is still a fine adjustment.
comptime CLICK_V: Float64 = 0.25
comptime CLICK_W: Float64 = 0.40

comptime UI_HEAD = Color(235, 240, 250, 255)
comptime UI_TXT = Color(205, 215, 232, 255)
comptime UI_DIM = Color(120, 132, 155, 255)
comptime UI_LOG = Color(150, 200, 170, 255)
comptime UI_CAP = Color(44, 50, 66, 240)
comptime UI_CAP_ON = Color(210, 90, 70, 255)
comptime UI_RING = Color(60, 68, 88, 210)


def _fit(text: String, n: Int) -> String:
    """Truncate to `n` characters. See `SIDE_CHARS`: the alternative is a
    label bleeding over the robot, which is what shipped first."""
    if text.byte_length() <= n:
        return text
    return String(text[byte = 0 : n])


def _clamp(v: Float64, lo: Float64, hi: Float64) -> Float64:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _decay(v: Float64, rate: Float64) -> Float64:
    """Toward zero by `rate`, without crossing it."""
    if v > rate:
        return v - rate
    if v < -rate:
        return v + rate
    return 0.0


def _f2(v: Float64) -> String:
    """Two decimals, without pulling in a formatter."""
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 100.0 + 0.5)
    var ip = h // 100
    var fp = h % 100
    var frac = String(fp) if fp >= 10 else (String("0") + String(fp))
    var body = String(ip) + String(".") + frac
    return (String("-") + body) if neg else body


def _cap(
    mut ui: UI, x: Float32, y: Float32, w: Float32, label: String, on: Bool
) -> Bool:
    """One key cap: lit while the key is HELD, and clickable in its own right.

    ⚠ Returns the CLICK, not the held state — the caller already knows the
    latter. Menlo's pad works the same way: the caps are both a legend and a
    control, which is what makes the demo usable without a keyboard at all.
    """
    return ui.button(x, y, w, 20, label, on, 1)


def _drive_pad(
    mut ui: UI,
    x: Float32,
    y: Float32,
    cx: Float64,
    cy: Float64,
    cw: Float64,
    up: Bool, dn: Bool, lf: Bool, rt: Bool, sl: Bool, sr: Bool,
) -> List[Float64]:
    """The bottom-left stick: caps, a ring, and the command as a dot.

    The dot is the COMMAND, drawn at its fraction of the per-axis maximum, so
    the ring reads as a stick deflection rather than as a number. It is the
    only part of this window that shows `z` changing without reading a float.
    """
    ui.panel(x, y, PAD_W, PAD_H, Color(14, 16, 24, 205))
    var cxp = x + PAD_W * 0.5
    var cyp = y + 66.0

    # the ring, as 12 dots on a circle — `ui` has rects, not circles, and a
    # dotted ring reads better at this size than a square outline would
    var r = Float32(42.0)
    for i in range(12):
        var ang = Float32(i) * 0.5235988
        var dx = r * _cosf(ang)
        var dy = r * _sinf(ang)
        ui.panel(cxp + dx - 1.5, cyp + dy - 1.5, 3, 3, UI_RING)

    # the command dot
    var fx = Float32(cx / (MAX_FWD if cx >= 0.0 else MAX_BACK))
    var fy = Float32(cy / MAX_LAT)
    var px = cxp - fy * r
    var py = cyp - fx * r
    ui.panel(cxp - 4, cyp - 4, 8, 8, UI_RING)
    ui.panel(px - 5, py - 5, 10, 10, Color(235, 240, 250, 255))

    # the caps, laid out where the keys are
    # ⚠ THE PAD RETURNS DELTAS RATHER THAN WRITING THE COMMAND. A widget runs
    # inside this call and cannot reach the caller's `cx`; Mojo closures across
    # that boundary are painful enough that `ui.mojo` itself deferred drawing
    # for the same reason. So a click becomes a number the caller adds — which
    # also means a click and a held key go through exactly one code path.
    var d = List[Float64](length=3, fill=0.0)
    if _cap(ui, cxp - 14, y + 4, 28, String("^"), up):
        d[0] += CLICK_V
    if _cap(ui, cxp - 14, y + 112, 28, String("v"), dn):
        d[0] -= CLICK_V
    if _cap(ui, x + 6, cyp - 10, 28, String("<"), lf):
        d[2] += CLICK_W
    if _cap(ui, x + PAD_W - 34, cyp - 10, 28, String(">"), rt):
        d[2] -= CLICK_W
    if _cap(ui, x + 6, y + 4, 28, String("A"), sl):
        d[1] += CLICK_V
    if _cap(ui, x + PAD_W - 34, y + 4, 28, String("D"), sr):
        d[1] -= CLICK_V
    ui.label(x + 6, y + PAD_H - 14,
             String("^v drive  <> turn"), UI_DIM, 1)
    return d^


def _cosf(a: Float32) -> Float32:
    return Float32(_cos64(Float64(a)))


def _sinf(a: Float32) -> Float32:
    return Float32(_sin64(Float64(a)))


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


def main() raises:
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var fps = atol(_flag(String("--fps"), String("50")))
    var start_clip = atol(_flag(String("--clip"), String("13")))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")
    var delay_ms = 1000 // fps if fps > 0 else 0

    print("=" * 70)
    print("BFM-Zero G1 — the REWARD JOYSTICK (docs §12.45)")
    print("=" * 70)
    print("  up / down     drive forward / back   (also W / S)")
    print("  left / right  turn                  (also Q / E)")
    print("  A / D         strafe")
    print("  X             stop (the STAND objective)")
    print("  G             put the robot back on its feet")
    print("  Esc quit · 1-9 cameras · mouse orbits · the pad is CLICKABLE")
    print("  keys are read as SCANCODES, so the letters above are PHYSICAL")
    print("  positions — on AZERTY they fall on Z/S, A/E and Q/D.")
    print("-" * 70)

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm sidecar beside the checkpoint: RAW inputs.")

    # ── the state pool: E_rho is estimated on THESE rows ──────────────
    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var rsi = G1RsiTable.from_store(store)
    var n_rows = len(st) // UNITREE_G1_STATE_DIM
    if n_rows < POOL:
        raise Error("store has fewer rows than POOL")
    var stride = n_rows // POOL

    var pool = Tensor.alloc(POOL * OBS)
    # ⚠ The tail is zeroed ONCE and never written: `B` is
    # `BFMBNetFiltered[OBS, SP, ...]`, which SLICES the first 527 of a 928-wide
    # row. Handing it a 527-wide buffer instead would mis-stride every row
    # after the first — that is §12.38, which cost a whole run.
    for i in range(POOL * OBS):
        pool.data[i] = Scalar[DT](0.0)
    # the command features, kept RAW — the reward is about physical velocity
    var pvx = List[Float64](length=POOL, fill=0.0)
    var pvy = List[Float64](length=POOL, fill=0.0)
    var pwz = List[Float64](length=POOL, fill=0.0)
    for i in range(POOL):
        var r = i * stride
        for k in range(UNITREE_G1_STATE_DIM):
            pool.data[i * OBS + k] = Scalar[DT](
                st[r * UNITREE_G1_STATE_DIM + k]
            )
        for k in range(UNITREE_G1_PRIV_DIM):
            pool.data[i * OBS + UNITREE_G1_STATE_DIM + k] = Scalar[DT](
                pv[r * UNITREE_G1_PRIV_DIM + k]
            )
        pvx[i] = Float64(pool.data[i * OBS + OFF_VX + 0])
        pvy[i] = Float64(pool.data[i * OBS + OFF_VX + 1])
        pwz[i] = Float64(pool.data[i * OBS + OFF_WZ])
    if norm:
        norm.value().apply_rows(pool, POOL)

    var rewards_probe = List[Scalar[DT]](length=POOL, fill=Scalar[DT](0))
    var b_pool = Tensor()
    t.backward_embed[POOL](pool, b_pool)
    # `z_from_reward` takes flat Lists; copy once, then every keypress is a
    # matvec over this and nothing else touches the network again.
    var b_list = List[Scalar[DT]](length=POOL * D, fill=Scalar[DT](0))
    for i in range(POOL * D):
        b_list[i] = b_pool.data[i]
    print("  pool", POOL, "rows (stride", stride, "of", n_rows, ") — B encoded")

    # ── ⚠ THE VACUITY CHECK, and it belongs here rather than in a test ──
    # If every command produced the same `z`, this would be a placebo: the keys
    # would still respond, the HUD would still update, the robot would still
    # walk — and the objective would never have changed. `E_rho[B.r]` collapses
    # exactly that way when the pool has no spread in the command features, or
    # when SIGMA is so wide that every reward is ~1. Both are silent. So the
    # angles between three prompts get printed before the window opens: a
    # degenerate pool is visible on line one instead of after ten minutes of
    # driving something that was never listening.
    var probe_vx = List[Float64](length=3, fill=0.0)
    var probe_wz = List[Float64](length=3, fill=0.0)
    probe_vx[0] = 0.0                       # stand
    probe_vx[1] = 1.0                       # walk forward
    probe_wz[2] = 1.2                       # turn in place
    var pz = List[Scalar[DT]](length=3 * D, fill=Scalar[DT](0))
    for c in range(3):
        for i in range(POOL):
            var dx = pvx[i] - probe_vx[c]
            var dy = pvy[i]
            var dw = pwz[i] - probe_wz[c]
            rewards_probe[i] = Scalar[DT](
                exp(-(dx * dx + dy * dy) / (SIGMA_V * SIGMA_V))
                * exp(-(dw * dw) / (SIGMA_W * SIGMA_W))
            )
        var zc = z_from_reward[D](b_list, rewards_probe, POOL)
        for k in range(D):
            pz[c * D + k] = zc[k]
    var names = List[String]()
    names.append(String("stand"))
    names.append(String("forward"))
    names.append(String("turn"))
    var worst_deg = 1e9
    for c in range(3):
        for c2 in range(c + 1, 3):
            var dot = 0.0
            for k in range(D):
                dot += Float64(pz[c * D + k]) * Float64(pz[c2 * D + k])
            var cosang = dot / Float64(D)          # both on the radius-sqrt(D) sphere
            if cosang > 1.0:
                cosang = 1.0
            if cosang < -1.0:
                cosang = -1.0
            var deg = acos(cosang) * 180.0 / 3.14159265358979
            print("  prompt angle", names[c], "vs", names[c2], ":", deg, "deg")
            if deg < worst_deg:
                worst_deg = deg
    if worst_deg < 1.0:
        print("  ⚠⚠ THE PROMPTS ARE THE SAME VECTOR (worst", worst_deg,
              "deg). The joystick would be a placebo — check the pool's"
              " spread in the command features and SIGMA_V / SIGMA_W.")
    else:
        print("  prompts are distinct (worst separation", worst_deg, "deg)")

    # ── the env + the window ──────────────────────────────────────────
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    if not env.init_renderer():
        print("  ⚠ no renderer — nothing to drive. Exiting.")
        return
    env.renderer_set_show_hud(False)      # the sidebar says all of it, beside
    env.set_ui_sidebar_width(SIDEBAR_W)

    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var aobs = G1ActorObs()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var rewards = List[Scalar[DT]](length=POOL, fill=Scalar[DT](0))

    var reset_row = Int(rsi.ep_offset.data[start_clip])
    var want_respawn = True

    # the command, and the command the prompt was last built for
    var cx = 0.0
    var cy = 0.0
    var cw = 0.0
    var zx = 1e9
    var zy = 1e9
    var zw = 1e9
    var steps = 0
    var log = List[String]()

    # ⚠ SCANCODES, NOT KEYCODES, and the reason is not only AZERTY. A scancode
    # is a PHYSICAL key position, so the same code is the same key under every
    # layout — the block below is WASD on QWERTY and ZQSD on AZERTY without
    # branching on anything. It also sidesteps the event pump: `check_quit`
    # CLAIMS the right arrow for its pause-step binding and never forwards it,
    # so an arrow-driven joystick reading `take_key` would silently lose one of
    # its four directions. `get_keyboard_state` reads SDL's own array instead,
    # which is a snapshot of what is HELD rather than a queue of transitions —
    # exactly the right shape for a stick that ramps while you lean on it.
    var nkeys = c_int(0)
    var kb = get_keyboard_state(Ptr(to=nkeys).as_unsafe_any_origin())

    while env.is_renderer_open():
        if env.check_renderer_quit():
            break
        if want_respawn:
            var base = reset_row * (G1_RSI_NQ + G1_RSI_NV)
            for i in range(NQ):
                qp[i] = Float64(rsi.rows.data[base + i])
            for i in range(NV):
                qv[i] = Float64(rsi.rows.data[base + G1_RSI_NQ + i])
            env.set_state(qp, qv)
            # ⚠ the actor's 401 is part of the STATE. Leaving it stale would
            # feed the policy a history from before the teleport.
            aobs.reset()
            want_respawn = False

        # ── held keys -> a ramped command ─────────────────────────────
        var up = kb[Int(Scancode.SCANCODE_UP)] or kb[Int(Scancode.SCANCODE_W)]
        var dn = kb[Int(Scancode.SCANCODE_DOWN)] or kb[Int(Scancode.SCANCODE_S)]
        var lf = kb[Int(Scancode.SCANCODE_LEFT)] or kb[Int(Scancode.SCANCODE_Q)]
        var rt = kb[Int(Scancode.SCANCODE_RIGHT)] or kb[Int(Scancode.SCANCODE_E)]
        var sl = kb[Int(Scancode.SCANCODE_A)]
        var sr = kb[Int(Scancode.SCANCODE_D)]
        # ⚠ NOT space. `check_quit` claims space for its pause toggle, and
        # although scancode polling still SEES it, the renderer would pause
        # on the same press — one key doing two things, one of them
        # invisible. X is unclaimed.
        var halt = kb[Int(Scancode.SCANCODE_X)]

        # A stick, not a stepper: hold to lean, release and it centres. The
        # decay is what makes "let go" mean "stop" without a keypress.
        if up:
            cx += RAMP_V
        elif dn:
            cx -= RAMP_V
        else:
            cx = _decay(cx, DECAY_V)
        if sl:
            cy += RAMP_V
        elif sr:
            cy -= RAMP_V
        else:
            cy = _decay(cy, DECAY_V)
        if lf:
            cw += RAMP_W
        elif rt:
            cw -= RAMP_W
        else:
            cw = _decay(cw, DECAY_W)
        if halt:
            cx = 0.0
            cy = 0.0
            cw = 0.0

        var key = env.renderer_take_key()
        if key == Int(Keycode.SDLK_G):
            want_respawn = True

        cx = _clamp(cx, -MAX_BACK, MAX_FWD)
        cy = _clamp(cy, -MAX_LAT, MAX_LAT)
        cw = _clamp(cw, -MAX_YAW, MAX_YAW)

        # ── the whole point: a new objective, and its policy ───────────
        # Rebuild only on a MATERIAL change. The matvec is cheap enough to run
        # every frame, but rebuilding on float noise would make the printed
        # command log unreadable.
        if (
            abs(cx - zx) > 1e-3 or abs(cy - zy) > 1e-3 or abs(cw - zw) > 1e-3
        ):
            for i in range(POOL):
                var dx = pvx[i] - cx
                var dy = pvy[i] - cy
                var dw = pwz[i] - cw
                rewards[i] = Scalar[DT](
                    exp(-(dx * dx + dy * dy) / (SIGMA_V * SIGMA_V))
                    * exp(-(dw * dw) / (SIGMA_W * SIGMA_W))
                )
            var zl = z_from_reward[D](b_list, rewards, POOL)
            for k in range(D):
                z1.data[k] = zl[k]
            # `z_from_reward` already projects; belt-and-braces on the ONE
            # invariant §11 ranks first among the silent failures.
            g1_project_z[D](z1, 0)
            if (
                abs(cx - zx) > 0.08 or abs(cy - zy) > 0.08
                or abs(cw - zw) > 0.12
            ):
                # ⚠ 22 characters, not 43. The long form
                # ("set_velocity  vx .. · vy .. · vyaw ..") is wider than
                # the sidebar and every row of it ran over the scene.
                log.append(
                    String("vx ") + _f2(cx)
                    + String(" vy ") + _f2(cy)
                    + String(" w ") + _f2(cw)
                )
                if len(log) > LOG_ROWS:
                    var trimmed = List[String]()
                    for i in range(len(log) - LOG_ROWS, len(log)):
                        trimmed.append(log[i])
                    log = trimmed^
            zx = cx
            zy = cy
            zw = cw

        # ── act ───────────────────────────────────────────────────────
        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
        t.act[1](obs_t, z1, act_out)
        var a = ContAction[ACT]()
        for k in range(ACT):
            a.data[k] = _clamp(Float64(act_out.data[k]), -1.0, 1.0)
        aobs.push(o, act_out)
        _ = env.step(a)

        # what the robot is ACTUALLY doing, in the same heading frame the
        # command is written in — so the pad's two dots can be compared
        var mvx = Float64(o[OFF_VX + 0])
        var mvy = Float64(o[OFF_VX + 1])
        var mwz = Float64(o[OFF_WZ])

        # ── the UI ────────────────────────────────────────────────────
        var win_h = env.renderer_height()
        var ui = UI(
            env.renderer_mouse_x(), env.renderer_mouse_y(),
            env.renderer_take_click(),
        )

        # left sidebar — the menu
        ui.panel(0, 0, Float32(SIDEBAR_W), Float32(win_h))
        ui.label(12, 10, String("BFM-ZERO JOYSTICK"), UI_HEAD, 1)
        ui.label(12, 26, String("z = E[B(s) r(s)]"), UI_DIM, 1)
        ui.label(12, 38, String("no training, no gradient"), UI_DIM, 1)
        ui.label(12, 58, String("COMMAND"), UI_DIM, 1)
        ui.label(12, 74, String("vx    ") + _f2(cx) + String(" m/s"), UI_TXT, 1)
        ui.label(12, 88, String("vy    ") + _f2(cy) + String(" m/s"), UI_TXT, 1)
        ui.label(12, 102, String("vyaw  ") + _f2(cw) + String(" rad/s"), UI_TXT, 1)
        ui.label(12, 126, String("MEASURED"), UI_DIM, 1)
        ui.label(12, 142, String("vx    ") + _f2(mvx), UI_TXT, 1)
        ui.label(12, 156, String("vy    ") + _f2(mvy), UI_TXT, 1)
        ui.label(12, 170, String("vyaw  ") + _f2(mwz), UI_TXT, 1)

        var by = Float32(200)
        if ui.button(12, by, 96, 22, String("stand"), abs(cx) + abs(cy) + abs(cw) < 1e-3, 1):
            cx = 0.0
            cy = 0.0
            cw = 0.0
        if ui.button(114, by, 96, 22, String("walk"), False, 1):
            cx = 0.8
            cy = 0.0
            cw = 0.0
        if ui.button(12, by + 26, 96, 22, String("back"), False, 1):
            cx = -0.5
            cy = 0.0
            cw = 0.0
        if ui.button(114, by + 26, 96, 22, String("spin"), False, 1):
            cx = 0.0
            cy = 0.0
            cw = 1.2
        if ui.button(12, by + 56, 198, 22, String("respawn  (G)"), False, 1):
            want_respawn = True

        ui.label(12, by + 92, String("COMMAND LOG"), UI_DIM, 1)
        for i in range(len(log)):
            ui.label(12, by + 108 + Float32(i) * 12,
                     _fit(log[i], SIDE_CHARS), UI_LOG, 1)

        ui.label(12, Float32(win_h) - 32, String("steps ") + String(steps), UI_DIM, 1)
        ui.label(12, Float32(win_h) - 20, String("pool ") + String(POOL)
                 + String("  d ") + String(D), UI_DIM, 1)

        # ⚠ THE PAD LIVES IN THE SIDEBAR, not over the scene. Menlo's floats
        # on the render because that render is a video feed it cannot write
        # into; ours has a reserved column, and putting the pad there means
        # nothing the UI draws can ever sit on top of the robot. It also gives
        # the pad a hard width to fit — PAD_W is sized off SIDEBAR_W rather
        # than chosen, so the two cannot drift apart.
        var pad_y = Float32(win_h) - PAD_H - 64
        var pad = _drive_pad(
            ui, 12, pad_y, cx, cy, cw, up, dn, lf, rt, sl, sr,
        )
        cx += pad[0]
        cy += pad[1]
        cw += pad[2]
        # the pad's buttons are hit-tested inside `_drive_pad`; its Stop is the
        # only one that writes back, and it does it through this flag because a
        # widget cannot reach `cx` from in there.
        if ui.button(
            12, pad_y + PAD_H + 6, PAD_W, 22,
            String("STOP  (X)"), halt, 1,
        ):
            cx = 0.0
            cy = 0.0
            cw = 0.0

        env.set_ui(ui.rects, ui.texts)
        env.render_frame()
        env.renderer_delay(delay_ms)
        steps += 1

    env.close()
    print("-" * 70)
    print("  ", steps, "control steps, each one driven by a reward you wrote")
