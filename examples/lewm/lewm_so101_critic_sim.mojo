"""S3 — the SO-101 world model as a CRITIC on the pixel student, in the sim.

noeira-docs/SO101_LEWM_PLAN.md S3. The student (`pixel_student_probe_sim`'s
loop, unchanged: it acts every tick in target mode) proposes; every FS = 5
ticks — one world-model step — the critic

    1. renders the rig's two cameras (device tracer, 320 × 240, area-resized
       to 112 like the training store) and encodes the frame;
    2. builds K candidate offsets δ_k on the student's six action words
       (δ_0 = 0, the others N(0, σ²) per word);
    3. turns each into the model's action blocks: the student's CURRENT word
       + δ_k held for 5 ticks, through `target_step` from the last commanded
       target (the arm assumed to keep its lead), FS × 6 target changes,
       z-scored with the training run's `action_stats.txt`, repeated for
       `--h` model steps;
    4. rolls the predictor from the last 3 OBSERVED latents and the executed
       blocks between them (`LeWMRefRollout.rollout_ctx`), and scores each
       candidate against a REFERENCE PATH: the teacher's own run on the same
       scene, rendered at 112 (`ppo_state_probe_sim --demos` →
       `tower_demo_rerender --resize 112`), one latent every 5 ticks.
       Progress i* = the reference latent nearest the current one within
       `--window` steps ahead of the last i* (monotone); cost =
       Σ_h ‖ẑ_{t+h} − z_ref[i* + h]‖², h = 1 .. H;
    5. adds the winner's δ to the student's word on each of the next 5 ticks.

`--mode student` runs the student alone (the gate's baseline through this
code), `critic` picks the argmin, `random` picks a uniformly random candidate
(the control: does the RANKING help, or the noise?).

⚠ The reference path is an ORACLE in S3's sense: same scene, sim frames.
On the real arm a sim-made goal does not carry over with the v1 encoder
(S2: the twin gap after start-frame offset correction is ~4.6 model steps,
`lewm_so101_align`). S3 asks whether the critic can help at all.

Per episode it prints success (the brick within 0.045 m of the bowl, the
probe's `Near`), the tick it first held, and the first close after the jaw
opened — EMPTY (< −0.10 rad) or a grasp (a stall), the gate's
`empty_first.py` rule.

    pixi run -e apple mojo build -I . -Xlinker -ld_classic -D DAGGER_WINDOW \\
        -D DAGGER_PX_32 -D DAGGER_JOINT_VEL -D TASK_PPO_ACT_HIST=3 \\
        -D TASK_PPO_TARGET_OBS examples/lewm/lewm_so101_critic_sim.mojo -o critic
    critic --ckpt projects/so101-tower/policies/pixel_bowl_w15.ckpt \\
        --wm <run>/epoch_7 --stats <run>/action_stats.txt --ref ref.rendered.h5 \\
        --brick x,y --bowl x,y --mode critic [--k-sigma 0.3] [--h 3] \\
        [the probe's --lag-* flags]
"""

from std.os.path import exists
from std.sys import argv
from std.sys.defines import get_defined_int
from std.random import seed, random_float64, random_ui64
from std.math import sqrt, log, cos
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.ptr import mptr
from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.delta_action import target_step, ServoLag, TARGET_OBS
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import (
    StudentNet, N_CAMS, IN_DIM, ACT, RENDER, OVERHEAD_RENDER_W,
    OVERHEAD_RENDER_H, check_pixel_manifest,
    render_to_planes, joints_to_planes, joint_vels_to_planes, student_act,
    HIST_WORDS, act_hist_push, act_hist_to_planes, target_lead_to_planes,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, RIG_NPIX, RIG_CAM_W, RIG_CAM_H, make_tower_model,
    make_tower_renderer, tower_cameras, pack_camera_u8,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.data.store import TrajectoryStore
from noeira.experimental.lewm.ref_model import LeWMEncoderRef
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import LeWMRefRollout, REF_CTX
from noeira.experimental.lewm.so101_frames import AreaResize
from noeira.experimental.lewm.so101_data import FS, JOINTS, ACT_IN, CAMS

comptime E = Phyics3dEnv[So101TowerModel, So101TowerConfig, DT, False]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime R = 112
comptime EMB = 192
comptime FR = CAMS * 3 * R * R
comptime Enc = LeWMEncoderRef[CAMS * 3, R, 14, 192, 3, 12, EMB, 2048]
comptime K = get_defined_int["CRITIC_K", 16]()
"""Candidates per decision (δ_0 = 0 included)."""
comptime HMAX = get_defined_int["CRITIC_HMAX", 4]()
"""The most model steps `--h` may ask for (the rollout's buffer)."""
comptime HOR = REF_CTX - 1 + HMAX
comptime Roll = LeWMRefRollout["gpu", K, HOR, ACT_IN]
comptime POST_GOAL = 31
"""Ticks run after the brick first reaches the bowl."""


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _gauss() -> Float64:
    var u1 = max(random_float64(), 1e-12)
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


def _encode(
    mut enc: Enc, frame: List[UInt8], off: Int, ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    """One 2 × 3 × R × R u8 frame (at `off`) -> its latent, eval-mode BN."""
    var mean: List[Float64] = [0.485, 0.456, 0.406]
    var std: List[Float64] = [0.229, 0.224, 0.225]
    var x = Tensor.alloc(FR)
    for ch in range(CAMS * 3):
        var c = ch % 3
        for p in range(R * R):
            var v = Float64(Int(frame[off + ch * R * R + p])) / 255.0
            x.data[ch * R * R + p] = Scalar[DT]((v - mean[c]) / std[c])
    x.upload(ctx.value())
    var y = Tensor.alloc(EMB)
    enc.forward["gpu", 1](TensorRefs[1](x), y, ctx)
    ctx.value().synchronize()
    y.download(ctx.value())
    var z = List[Scalar[DT]](capacity=EMB)
    for d in range(EMB):
        z.append(y.data[d])
    return z^


def _d2(a: List[Scalar[DT]], ao: Int, b: List[Scalar[DT]], bo: Int) -> Float64:
    var s = 0.0
    for d in range(EMB):
        var x = Float64(a[ao + d]) - Float64(b[bo + d])
        s += x * x
    return s


def _ranks(v: List[Float64]) -> List[Float64]:
    var r = List[Float64](length=len(v), fill=0.0)
    for i in range(len(v)):
        var below = 0
        for j in range(len(v)):
            if v[j] < v[i] or (v[j] == v[i] and j < i):
                below += 1
        r[i] = Float64(below)
    return r^


def _spearman(a: List[Float64], b: List[Float64]) -> Float64:
    var ra = _ranks(a)
    var rb = _ranks(b)
    var n = Float64(len(a))
    var m = (n - 1.0) / 2.0
    var sab = 0.0
    var saa = 0.0
    var sbb = 0.0
    for i in range(len(a)):
        sab += (ra[i] - m) * (rb[i] - m)
        saa += (ra[i] - m) ** 2
        sbb += (rb[i] - m) ** 2
    return sab / max((saa * sbb) ** 0.5, 1e-12)


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var ckpt = _arg(args, "--ckpt", "projects/so101-tower/policies/pixel_bowl_w15.ckpt")
    var task = _arg(args, "--task", "so101_tower_cube_in_bowl")
    var seed0 = Int(_arg(args, "--seed", "1"))
    var episodes = Int(_arg(args, "--episodes", "1"))
    var brick_xy = _arg(args, "--brick", "")
    var bowl_xy = _arg(args, "--bowl", "")
    var wm = _arg(args, "--wm", "")
    var stats = _arg(args, "--stats", "")
    var ref_path = _arg(args, "--ref", "")
    var mode = _arg(args, "--mode", "critic")
    var sigma = Float64(_arg(args, "--k-sigma", "0.3"))
    var hz = Int(_arg(args, "--h", "3"))
    var window = Int(_arg(args, "--window", "8"))
    var rng = Int(_arg(args, "--rng", "0"))
    # every N-th decision: EXECUTE every candidate from the same sim state
    # (snapshot -> 5·h ticks -> render, encode -> restore) and compare the
    # model's ranking with the real one
    var rank_every = Int(_arg(args, "--rank-test", "0"))
    if mode != "student" and mode != "critic" and mode != "random":
        raise Error("--mode student|critic|random")
    if wm.byte_length() == 0 or stats.byte_length() == 0 or ref_path.byte_length() == 0:
        raise Error("--wm, --stats and --ref are required (also by --mode student)")
    if hz < 1 or hz > HMAX:
        raise Error("--h must be 1 .. " + String(HMAX) + " (-D CRITIC_HMAX)")
    seed(rng)

    var man_path = String(ckpt[byte = 0 : ckpt.byte_length() - 5]) + ".norm.json"
    if not exists(man_path):
        man_path = ckpt[byte = 0 : ckpt.rfind("/")] + "/norm.json"
    var man = check_pixel_manifest(man_path)
    if man.action != "target" or man.repeat != 1:
        raise Error("the critic expects a target-mode student acting every tick")

    var ctx = DeviceContext()
    var octx = Optional(ctx)
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    rd.upload_all(ctx)
    ctx.synchronize()
    var r_w = make_tower_renderer[1, RENDER, RENDER, 1](ctx, fmd, rm)
    var r_o = make_tower_renderer[1, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1](ctx, fmd, rm)
    # the world model's eyes: the training store's renderer and resampler
    var r_wm = make_tower_renderer[1](ctx, fmd, rm)
    var h_rgb = ctx.enqueue_create_host_buffer[RIG_DT](RIG_NPIX * 3)
    var full = List[UInt8](length=CAMS * 3 * RIG_NPIX, fill=UInt8(0))
    var resizer = AreaResize(3, RIG_CAM_H, RIG_CAM_W, R)

    var jadr = List[Int]()
    var jdadr = List[Int]()
    var a1 = 0
    var a2 = 0
    for i in range(len(fmd.joints)):
        jadr.append(a1)
        jdadr.append(a2)
        a1 += fmd.joints[i].nq
        a2 += fmd.joints[i].nv
    var qa = List[Int]()
    var da = List[Int]()
    var lo = List[Float64]()
    var hi = List[Float64]()
    for i in range(ACT):
        qa.append(jadr[fmd.actuators[i].joint_id])
        da.append(jdadr[fmd.actuators[i].joint_id])
        lo.append(fmd.actuators[i].ctrl_min)
        hi.append(fmd.actuators[i].ctrl_max)
    var brick = -1
    var bowl = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
        if String(fmd.body_names[b]) == "bowl_bowl":
            bowl = b

    var net = StudentNet.make["cpu", Kaiming](None)
    _ = load_params["cpu"](net, ckpt, None)
    var x = Tensor.alloc(IN_DIM)
    var y = Tensor.alloc(ACT)
    var xs = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))

    # ── the world model, its action normaliser, the reference path ─────────
    var enc = Enc.make["gpu", Kaiming](octx)
    var a_mean = List[Float64](length=JOINTS, fill=0.0)
    var a_std = List[Float64](length=JOINTS, fill=1.0)
    var z_ref = List[Scalar[DT]]()
    var n_ref = 0
    var roll = Roll(wm, octx)
    if True:
        var n_enc = load_ref["gpu"](enc, wm, String("emb.0."), octx)
        enc.set_attr["training"](Scalar[DT](0.0))
        var j = 0
        with open(stats, "r") as fh:
            for ln in fh.read().split("\n"):
                var s = String(ln.strip())
                if s.byte_length() == 0 or s.startswith("#"):
                    continue
                var w = s.split(" ")
                a_mean[j] = Float64(String(w[0]))
                a_std[j] = Float64(String(w[1]))
                j += 1
        var st = TrajectoryStore(ref_path)
        var ims = st.load_column[DType.uint8](String("images"), max_bytes=1 << 32)
        var n0 = Int(st.episodes.ep_len[0])
        var r = 0
        while r < n0:
            z_ref.extend(_encode(enc, ims, r * FR, octx))
            n_ref += 1
            r += FS
        print("critic: world model", wm, "(", n_enc, "encoder tensors ) | reference",
              ref_path, "|", n0, "ticks ->", n_ref, "latents | K", K, "| sigma", sigma,
              "| h", hz, "| window", window)
    print("critic: mode", mode, "| student", ckpt)

    var env = E(ctx)
    var lag = ServoLag.parse(1, _arg(args, "--lag-tau", ""), _arg(args, "--lag-delay", ""),
                             man.control_period_s)
    lag.set_limits(_arg(args, "--lag-vmax", ""), Float64(_arg(args, "--elbow-max", "0")))
    lag.set_per_joint(
        _arg(args, "--lag-tau-j", ""), _arg(args, "--lag-delay-j", ""),
        _arg(args, "--lag-vmax-j", ""),
    )
    lag.set_offset(_arg(args, "--lag-offset-j", ""))
    lag.set_period(_arg(args, "--lag-period", ""))

    var n_ok = 0
    var n_empty = 0
    for ep in range(episodes):
        _ = env.reset()
        var q0 = posed_qpos[So101TowerPlacement](
            task, String("so101_tower"), So101TowerConfig.SLOT_RADIUS,
            seed=UInt64(seed0 + ep),
        )
        # free slots: 0 = bowl, 1 = brick (the family's slot order)
        if bowl_xy.byte_length() > 0:
            var p = bowl_xy.split(",")
            q0[So101TowerPlacement.free_qadr(0)] = Float64(String(p[0]))
            q0[So101TowerPlacement.free_qadr(0) + 1] = Float64(String(p[1]))
        if brick_xy.byte_length() > 0:
            var p = brick_xy.split(",")
            q0[So101TowerPlacement.free_qadr(1)] = Float64(String(p[0]))
            q0[So101TowerPlacement.free_qadr(1) + 1] = Float64(String(p[1]))
        var v0 = List[Float64](length=NV, fill=0.0)
        _ = env.obs_at(q0, v0)
        var q_start = List[Float64](length=ACT, fill=0.0)
        for j in range(ACT):
            q_start[j] = Float64(env.d.qpos.data[qa[j]])
        lag.reset_lane(0, q_start, 0, 0.5, 0.5)
        var hist = List[Float64](length=HIST_WORDS, fill=0.0)
        var q = List[Float64](length=ACT, fill=0.0)
        var qd = List[Float64](length=ACT, fill=0.0)
        var a_ex = List[Float64](length=ACT, fill=0.0)
        var a_st = List[Float64](length=ACT, fill=0.0)
        var tgt_hold = List[Float64](length=ACT, fill=0.0)
        var tprev = q_start.copy()
        var lead = List[Float64](length=TARGET_OBS, fill=0.0)
        var delta = List[Float64](length=ACT, fill=0.0)
        # the critic's memory: observed latents and the executed blocks
        # between them (z-scored), the last REF_CTX of each
        var z_hist = List[List[Scalar[DT]]]()
        var a_hist = List[List[Scalar[DT]]]()
        var blk = List[Scalar[DT]](length=ACT_IN, fill=Scalar[DT](0))
        var i_star = 0
        var n_dec = 0
        var n_moved = 0
        var margin = 0.0
        var t_crit = 0
        # is the ranking signal above the model's own error? the chosen
        # candidate's 1-step prediction vs the latent observed 5 ticks later,
        # against the candidates' spread at h = 1
        var z_pred = List[Scalar[DT]]()
        var err = 0.0
        var n_err = 0
        var spread = 0.0
        var copy_err = 0.0
        var rk_rho = 0.0
        var rk_beats0 = 0
        var rk_regret = 0.0
        var rk_spread_act = 0.0
        var rk_spread_pred = 0.0
        var rk_err = 0.0
        var rk_n = 0
        # outcome
        var held_at = -1
        var opened = False
        var stall = 0
        var first_close = String("none")
        var z_desk = 1e9
        var rise = 0.0
        var z0 = Float64(env.d.xpos.data[brick * 3 + 2])
        var t_end = So101TowerConfig.MAX_STEPS
        for t in range(So101TowerConfig.MAX_STEPS):
            if t >= t_end:
                break
            for k in range(NQ):
                rd.qpos.data[k] = Scalar[RIG_DT](env.d.qpos.data[k])
            forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
            for k in range(N_CAMS):
                var rgb = List[Scalar[RIG_DT]]()
                var dd = List[Scalar[RIG_DT]]()
                var ss = List[Scalar[RIG_DT]]()
                if N_CAMS == 2 and k == 0:
                    r_o.cam = cams[k]
                    r_o.render_cpu(rd, rm, rgb, dd, ss)
                    render_to_planes(rgb, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, k, xs)
                else:
                    r_w.cam = cams[k]
                    r_w.render_cpu(rd, rm, rgb, dd, ss)
                    render_to_planes(rgb, RENDER, RENDER, k, xs)
            for j in range(ACT):
                q[j] = Float64(env.d.qpos.data[qa[j]])
                qd[j] = Float64(env.d.qvel.data[da[j]])
            joints_to_planes(q, xs)
            joint_vels_to_planes(qd, xs)
            act_hist_to_planes(hist, xs)
            for j in range(TARGET_OBS):
                lead[j] = tprev[j] - q[j]
            target_lead_to_planes(lead, xs)
            for k in range(IN_DIM):
                x.data[k] = xs[k]
            net.forward["cpu", 1](TensorRefs[1](x), y, None)
            for j in range(ACT):
                a_st[j] = Float64(student_act(y.data[j], j, man.gripper_sign))

            # ── the critic, once per model step ─────────────────────────────
            if mode != "student" and t % FS == 0:
                var tc = perf_counter_ns()
                rd.qpos.upload_resident(ctx)
                rd.xpos.upload_resident(ctx)
                rd.xquat.upload_resident(ctx)
                for slot in range(CAMS):
                    r_wm.render(ctx, rd, rm, cams[slot])
                    ctx.enqueue_copy(h_rgb, r_wm.rgb)
                    ctx.synchronize()
                    _ = pack_camera_u8(
                        mptr(h_rgb.unsafe_ptr()), 0,
                        rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](full.unsafe_ptr()),
                        slot * 3 * RIG_NPIX,
                    )
                var small = resizer.frames(full, CAMS)
                var z = _encode(enc, small, 0, octx)
                if len(z_pred) > 0:
                    err += sqrt(_d2(z, 0, z_pred, 0))
                    copy_err += sqrt(_d2(z, 0, z_hist[len(z_hist) - 1], 0))
                    n_err += 1
                if t > 0:
                    a_hist.append(blk.copy())
                z_hist.append(z.copy())
                if len(z_hist) > REF_CTX:
                    _ = z_hist.pop(0)
                if len(a_hist) > REF_CTX - 1:
                    _ = a_hist.pop(0)
                # progress along the reference: nearest within the window ahead
                var best_i = i_star
                var best_d = 1e30
                for i in range(i_star, min(n_ref, i_star + window + 1)):
                    var d2 = _d2(z, 0, z_ref, i * EMB)
                    if d2 < best_d:
                        best_d = d2
                        best_i = i
                i_star = best_i
                # candidates: offsets on the student's words, and their blocks
                var deltas = List[List[Float64]]()
                for c in range(K):
                    var dl = List[Float64](length=ACT, fill=0.0)
                    if c > 0:
                        for j in range(ACT):
                            dl[j] = sigma * _gauss()
                    deltas.append(dl^)
                var n_obs = len(z_hist)
                var acts = List[Scalar[DT]](length=K * HOR * ACT_IN, fill=Scalar[DT](0))
                for c in range(K):
                    for b in range(n_obs - 1):
                        for k in range(ACT_IN):
                            acts[(c * HOR + b) * ACT_IN + k] = a_hist[len(a_hist) - (n_obs - 1) + b][k]
                    var tp = tprev.copy()
                    var qq = q.copy()
                    for b in range(n_obs - 1, HOR):
                        for k in range(FS):
                            for j in range(ACT):
                                var aj = max(-1.0, min(1.0, a_st[j] + deltas[c][j]))
                                var tn = target_step(
                                    tp[j], qq[j], aj, j, lo[j], hi[j], man.delta_arm,
                                    man.delta_gripper, man.target_lead,
                                )
                                var dt = tn - tp[j]
                                qq[j] += dt
                                tp[j] = tn
                                acts[(c * HOR + b) * ACT_IN + k * JOINTS + j] = Scalar[DT](
                                    (dt - a_mean[j]) / a_std[j]
                                )
                var obs = List[Scalar[DT]](capacity=n_obs * EMB)
                for zz in z_hist:
                    obs.extend(zz.copy())
                var embs = roll.rollout_ctx(obs, n_obs, acts)
                var costs = List[Float64](length=K, fill=0.0)
                for c in range(K):
                    for h in range(hz):
                        var gi = min(n_ref - 1, i_star + 1 + h)
                        costs[c] += _d2(embs, (c * (HOR + 1) + n_obs + h) * EMB, z_ref, gi * EMB)
                if rank_every > 0 and n_dec % rank_every == 0:
                    var qs = List[Float64](length=NQ, fill=0.0)
                    var vs = List[Float64](length=NV, fill=0.0)
                    for k in range(NQ):
                        qs[k] = Float64(env.d.qpos.data[k])
                    for k in range(NV):
                        vs[k] = Float64(env.d.qvel.data[k])
                    var ly = lag.y.copy()
                    var lh = lag.hist.copy()
                    var lt = lag.tick
                    var real_cost = List[Float64](length=K, fill=0.0)
                    var z_act = List[Scalar[DT]](length=K * hz * EMB, fill=Scalar[DT](0))
                    for c in range(K):
                        env.set_state(qs, vs)
                        lag.y = ly.copy()
                        lag.hist = lh.copy()
                        lag.tick = lt
                        var tp = tprev.copy()
                        for h in range(hz):
                            for _k in range(FS):
                                var cact = ContAction[ACT]()
                                for j in range(ACT):
                                    var aj = max(-1.0, min(1.0, a_st[j] + deltas[c][j]))
                                    tp[j] = target_step(
                                        tp[j], Float64(env.d.qpos.data[qa[j]]), aj, j, lo[j], hi[j],
                                        man.delta_arm, man.delta_gripper, man.target_lead,
                                    )
                                    var tg = lag.apply(0, j, tp[j])
                                    cact.data[j] = (tg - 0.5 * (lo[j] + hi[j])) / (0.5 * (hi[j] - lo[j]))
                                lag.advance()
                                _ = env.step(cact)
                            for k in range(NQ):
                                rd.qpos.data[k] = Scalar[RIG_DT](env.d.qpos.data[k])
                            forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
                            rd.qpos.upload_resident(ctx)
                            rd.xpos.upload_resident(ctx)
                            rd.xquat.upload_resident(ctx)
                            for slot in range(CAMS):
                                r_wm.render(ctx, rd, rm, cams[slot])
                                ctx.enqueue_copy(h_rgb, r_wm.rgb)
                                ctx.synchronize()
                                _ = pack_camera_u8(
                                    mptr(h_rgb.unsafe_ptr()), 0,
                                    rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](full.unsafe_ptr()),
                                    slot * 3 * RIG_NPIX,
                                )
                            var zh = _encode(enc, resizer.frames(full, CAMS), 0, octx)
                            for d in range(EMB):
                                z_act[(c * hz + h) * EMB + d] = zh[d]
                            var gi = min(n_ref - 1, i_star + 1 + h)
                            real_cost[c] += _d2(zh, 0, z_ref, gi * EMB)
                    # restore the decision's state (and the render buffers' pose)
                    env.set_state(qs, vs)
                    lag.y = ly.copy()
                    lag.hist = lh.copy()
                    lag.tick = lt
                    for k in range(NQ):
                        rd.qpos.data[k] = Scalar[RIG_DT](env.d.qpos.data[k])
                    forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
                    var pbest = 0
                    var abest = 0
                    var amax = 0
                    for c in range(1, K):
                        if costs[c] < costs[pbest]:
                            pbest = c
                        if real_cost[c] < real_cost[abest]:
                            abest = c
                        if real_cost[c] > real_cost[amax]:
                            amax = c
                    rk_rho += _spearman(costs, real_cost)
                    if real_cost[pbest] < real_cost[0]:
                        rk_beats0 += 1
                    rk_regret += (real_cost[pbest] - real_cost[abest]) / max(
                        real_cost[amax] - real_cost[abest], 1e-9)
                    var b1 = n_obs * EMB
                    for c in range(K):
                        if c > 0:
                            rk_spread_act += sqrt(_d2(z_act, c * hz * EMB, z_act, 0)) / Float64(K - 1)
                            rk_spread_pred += sqrt(_d2(embs, c * (HOR + 1) * EMB + b1, embs, b1)) / Float64(K - 1)
                        rk_err += sqrt(_d2(embs, c * (HOR + 1) * EMB + b1, z_act, c * hz * EMB)) / Float64(K)
                    rk_n += 1
                var pick = 0
                if mode == "critic":
                    for c in range(1, K):
                        if costs[c] < costs[pick]:
                            pick = c
                else:
                    pick = Int(random_ui64(0, UInt64(K - 1)))
                var b1 = n_obs * EMB
                for c in range(1, K):
                    spread += sqrt(_d2(embs, c * (HOR + 1) * EMB + b1, embs, b1)) / Float64(K - 1)
                z_pred.clear()
                for d in range(EMB):
                    z_pred.append(embs[pick * (HOR + 1) * EMB + b1 + d])
                margin += costs[0] - costs[pick]
                n_dec += 1
                if pick != 0:
                    n_moved += 1
                for j in range(ACT):
                    delta[j] = deltas[pick][j]
                t_crit += Int(perf_counter_ns() - tc)

            for j in range(ACT):
                var a = max(-1.0, min(1.0, a_st[j] + delta[j]))
                a_ex[j] = a
                tgt_hold[j] = target_step(
                    tprev[j], q[j], a, j, lo[j], hi[j], man.delta_arm,
                    man.delta_gripper, man.target_lead,
                )
                var dt = tgt_hold[j] - tprev[j]
                blk[(t % FS) * JOINTS + j] = Scalar[DT]((dt - a_mean[j]) / a_std[j])
                tprev[j] = tgt_hold[j]
            act_hist_push(hist, a_ex)
            # the gate's first-close rule (`empty_first.py`)
            var g = q[ACT - 1]
            if g > 0.20:
                opened = True
                stall = 0
            elif opened and first_close == "none":
                if g < -0.10:
                    first_close = "EMPTY"
                elif g > -0.05 and tgt_hold[ACT - 1] < g - 0.1 and abs(qd[ACT - 1]) < 0.3:
                    stall += 1
                    if stall >= 10:
                        first_close = "grasp"
                else:
                    stall = 0
            var act = ContAction[ACT]()
            for j in range(ACT):
                var tg = lag.apply(0, j, tgt_hold[j])
                var mid = 0.5 * (lo[j] + hi[j])
                var half = 0.5 * (hi[j] - lo[j])
                act.data[j] = (tg - mid) / half
            lag.advance()
            _ = env.step(act)
            var zb = Float64(env.d.xpos.data[brick * 3 + 2])
            rise = max(rise, zb - z0)
            z_desk = min(z_desk, zb)
            var ex = Float64(env.d.xpos.data[brick * 3] - env.d.xpos.data[bowl * 3])
            var ey = Float64(env.d.xpos.data[brick * 3 + 1] - env.d.xpos.data[bowl * 3 + 1])
            var ez = Float64(env.d.xpos.data[brick * 3 + 2] - env.d.xpos.data[bowl * 3 + 2])
            if held_at < 0 and (ex * ex + ey * ey + ez * ez) ** 0.5 < 0.045:
                held_at = t
                t_end = t + POST_GOAL
        if held_at >= 0:
            n_ok += 1
        if first_close == "EMPTY":
            n_empty += 1
        var line = String("   episode ") + String(ep) + " | success " + String(held_at >= 0)
        line += " at tick " + String(held_at) + " | first close " + first_close
        line += " | rise " + String(Int(rise * 1000.0)) + " mm"
        if mode != "student":
            line += " | decisions " + String(n_dec) + ", moved " + String(n_moved)
            line += " | mean cost margin " + String(Float32(margin / Float64(max(n_dec, 1))))
            line += " | 1-step error " + String(Float32(err / Float64(max(n_err, 1))))
            line += " (copy-last " + String(Float32(copy_err / Float64(max(n_err, 1)))) + ")"
            line += " vs candidate spread " + String(Float32(spread / Float64(max(n_dec, 1))))
            line += " | ref progress " + String(i_star) + "/" + String(n_ref)
            line += " | " + String(Float32(Float64(t_crit) / 1e6 / Float64(max(n_dec, 1)))) + " ms/decision"
        print(line)
        if rk_n > 0:
            var nr = Float64(rk_n)
            print("     rank test (", rk_n, "decisions, every candidate executed ) | Spearman(model, sim cost)",
                  Float32(rk_rho / nr), "| model's pick beats delta 0 in the sim", rk_beats0, "/", rk_n,
                  "| regret (0 best .. 1 worst)", Float32(rk_regret / nr))
            print("     h=1: candidates' spread in the sim", Float32(rk_spread_act / nr), "| predicted",
                  Float32(rk_spread_pred / nr), "| model error per candidate", Float32(rk_err / nr))
    print("critic:", n_ok, "/", episodes, "reached the goal |", n_empty, "closed EMPTY first")
