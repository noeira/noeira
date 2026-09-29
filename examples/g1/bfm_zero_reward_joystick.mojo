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

from std.math import exp, sqrt, acos
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
    print("  W / X   forward velocity  -+ 0.25 m/s")
    print("  Q / E   strafe            -+ 0.25 m/s")
    print("  A / D   turn              -+ 0.40 rad/s")
    print("  Z       halt (zero command — the STAND objective)")
    print("  G       put the robot back on its feet")
    print("  Esc     quit.  1-9 cameras, Space pause, mouse orbits.")
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

    # ── the env ────────────────────────────────────────────────────────
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    if not env.init_renderer():
        print("  ⚠ no renderer — nothing to drive. Exiting.")
        return

    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var aobs = G1ActorObs()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var rewards = List[Scalar[DT]](length=POOL, fill=Scalar[DT](0))

    var reset_row = Int(rsi.ep_offset.data[start_clip])

    # ⚠ NOT a nested `def`. A closure over `env`/`qp`/`aobs` needs an explicit
    # capture convention in Mojo 1.1, and the respawn has to run once before the
    # first action anyway — so it is a flag consumed at the top of the loop.
    var want_respawn = True

    var cx = 0.0
    var cy = 0.0
    var cw = 0.0
    var dirty = True
    var steps = 0

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
        var key = env.renderer_take_key()
        if key == Int(Keycode.SDLK_W):
            cx += 0.25
            dirty = True
        elif key == Int(Keycode.SDLK_X):
            cx -= 0.25
            dirty = True
        elif key == Int(Keycode.SDLK_Q):
            cy += 0.25
            dirty = True
        elif key == Int(Keycode.SDLK_E):
            cy -= 0.25
            dirty = True
        elif key == Int(Keycode.SDLK_A):
            cw += 0.40
            dirty = True
        elif key == Int(Keycode.SDLK_D):
            cw -= 0.40
            dirty = True
        elif key == Int(Keycode.SDLK_Z):
            cx = 0.0
            cy = 0.0
            cw = 0.0
            dirty = True
        elif key == Int(Keycode.SDLK_G):
            want_respawn = True
        if cx > 1.5:
            cx = 1.5
        if cx < -1.0:
            cx = -1.0
        if cy > 1.0:
            cy = 1.0
        if cy < -1.0:
            cy = -1.0
        if cw > 1.6:
            cw = 1.6
        if cw < -1.6:
            cw = -1.6

        # ── the whole point: a new objective, and its policy ───────────
        if dirty:
            for i in range(POOL):
                var dx = pvx[i] - cx
                var dy = pvy[i] - cy
                var dw = pwz[i] - cw
                var rv = exp(-(dx * dx + dy * dy) / (SIGMA_V * SIGMA_V))
                var rw = exp(-(dw * dw) / (SIGMA_W * SIGMA_W))
                rewards[i] = Scalar[DT](rv * rw)
            var zl = z_from_reward[D](b_list, rewards, POOL)
            for k in range(D):
                z1.data[k] = zl[k]
            # `z_from_reward` already projects; this is belt-and-braces for the
            # ONE invariant §11 ranks first among the silent failures.
            g1_project_z[D](z1, 0)
            print("  cmd  vx", cx, " vy", cy, " wz", cw)
            var hud = List[String]()
            hud.append(
                String("cmd vx ") + String(cx) + String("  vy ") + String(cy)
                + String("  wz ") + String(cw)
            )
            hud.append(String("W/X fwd  Q/E strafe  A/D turn  Z halt  G reset"))
            env.set_hud_extra(hud)
            dirty = False

        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
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
        env.render_frame()
        env.renderer_delay(delay_ms)
        steps += 1

    env.close()
    print("-" * 70)
    print("  ", steps, "control steps driven by", steps, "rewards you wrote")
