# +--------------------------------------------------------------------------+ #
# | SuperMimic's object interface on the CPU: one clip at a time, traced
# +--------------------------------------------------------------------------+ #
"""A DIAGNOSTIC, not a score: the object interface (or the blind tracker) on
named carry clips, one lane on the CPU, with a row per step saying where the
box is against its reference, where the hands are against the box, and how
far the interface's action is from the tracker's own.

    pixi run mojo build -I . examples/g1/supermimic_object_trace_cpu.mojo -o build/sm_obj_trace
    pixi run ./build/sm_obj_trace --ckpt TRACKER.ckpt --store CARRY_50HZ.h5 --object-dir OBJECT_CSV_DIR \\
        --only NAME[,NAME…] --out-dir DIR [--interface INTERFACE.ckpt] [--gain 0.1]
        [--squeeze 40] [--ref-squeeze 0.05] [--ref-lower 1.0]

    --squeeze F      the scripted squeeze on the executed actions (`object_task.sm_squeeze_lane`), F newtons
    --ref-squeeze D  the reference's hands D metres toward the box while it is lifted (`sm_squeeze_reference`)
    --ref-lower S    … and DOWN by the share S of their height above the box's centre (the arms alone)
    The box's reference and the reward's reference grasp stay the ORIGINAL clip's under both edits.

Per clip, in `--out-dir`:

    NAME.trace.tsv   one row a step (columns in the header)
    NAME.qpos.tsv    one row a step: the simulator's qpos (robot 36, box 7),
                     then the reference box's pose (7) and the reference
                     robot's qpos (36): what a renderer needs

The protocol is `object_eval.sm_obj_benchmark`'s (the clip from its first
frame, no noise, no randomisation, to its end or a fall), and a step's box is
compared with the reference row the step aimed at, as there. The CPU and the
device lanes agree over a step and not over a long clip, so a clip's numbers
here can differ from the device benchmark's.

Without `--interface` the object encoder is at its zero start: the blind
tracker. Paper widths (the checkpoints'); mjlab's G1 with its box only.

⚠ A store of SEED motion is confidential: private compute only.
"""

from std.math import acos, sqrt
from std.os import makedirs
from std.sys import argv, get_defined_int

from noeira.core.bytes import string_from_bytes
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.heads import SMHeadLayout
from noeira.deep_agents.supermimic.kinematics import sm_fk_lane
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import sm_object_track
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_spec import SM_OBJ
from noeira.deep_agents.supermimic.object_task import (
    SM_OBJ_TERMS, SM_OREF, sm_obj_reward_lane, sm_obj_reward_params, sm_object_ref_rows, sm_squeeze_lane, sm_squeeze_reference,
    sm_squeeze_weights,
)
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBoxRobot, SMDims
from noeira.envs.robots.unitree_g1_mjlab_pd import g1m_action_scale, g1m_kp
from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime HO = get_defined_int["SM_OBJ_H", 1024]()
comptime Robot = G1MjlabBoxRobot
comptime D = SMDims[Robot]
comptime ARM0 = 15              # joints 15..28 are the two arms
comptime CARRY_RISE = 0.05      # `object_eval.SM_OBJ_LIFT`
# the training driver's widths, one lane
comptime C = SMConfig["cpu", Robot, 2048, 64, 2048, 1024, 512, 256, 2048, 1024, 512, 256, 256, 128, 1, 64, 16, 32]
comptime OC = SMObjConfig[C, HO, 256, 256, 256, 128]


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _object_poses(path: String, frames: Int) raises -> List[Float64]:
    """The box's pose rows of one clip, cut or padded (its last row repeated)
    to the clip's `frames`: [frames · 7]. (`supermimic_object_fit.mojo`'s.)"""
    var rows = List[Float64]()
    var first = True
    for line in string_from_bytes(read_file_bytes(path)).split("\n"):
        var t = String(String(line).strip())
        if t.byte_length() == 0:
            continue
        if first:               # the header
            first = False
            continue
        var n = 0
        for f in t.split(","):
            rows.append(Float64(String(String(f).strip())))
            n += 1
        if n != 7:
            raise Error(path + ": a row of " + String(n) + " values, expected 7")
    var have = len(rows) // 7
    if have < 2:
        raise Error(path + ": fewer than two rows")
    var out = List[Float64](length=frames * 7, fill=0.0)
    for f in range(frames):
        var s = min(f, have - 1) * 7
        for k in range(7):
            out[f * 7 + k] = rows[s + k]
    return out^


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var interface = _opt(args, String("--interface"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var object_dir = _opt(args, String("--object-dir"), String(""))
    var only = _opt(args, String("--only"), String(""))
    var out_dir = _opt(args, String("--out-dir"), String(""))
    var gain = Float64(_opt(args, String("--gain"), String(OC.DEF_GAIN)))
    var squeeze = Float64(_opt(args, String("--squeeze"), String("0")))
    var ref_squeeze = Float64(_opt(args, String("--ref-squeeze"), String("0")))
    var ref_lower = Float64(_opt(args, String("--ref-lower"), String("0")))
    if ckpt == "" or store == "" or object_dir == "" or only == "" or out_dir == "":
        raise Error("give --ckpt TRACKER.ckpt --store STORE.h5 --object-dir DIR --only NAME[,NAME…] --out-dir DIR [--interface CKPT]")
    var want = List[String]()
    for w in only.split(","):
        want.append(String(String(w).strip()))
    makedirs(out_dir, exist_ok=True)

    # the named clips, in the store's order: library rows and the box's poses
    var st = TrajectoryStore(store.copy())
    var refc = st.load_column[DType.float32](String("ref"))
    var rows = List[Scalar[DT]]()
    var lens = List[Int]()
    var names = List[String]()
    var poses = List[Float64]()
    var o = 0
    for e in range(st.n_episodes()):
        var l = Int(st.episodes.ep_len[e])
        var name = st.manifest.task_text(e)
        var hit = False
        for k in range(len(want)):
            if want[k] == name:
                hit = True
        if hit:
            for i in range(l * D.S):
                rows.append(refc[o * D.S + i])
            lens.append(l)
            names.append(name.copy())
            var p = _object_poses(object_dir + "/" + name + ".csv", l)
            for i in range(len(p)):
                poses.append(p[i])
        o += l
    if len(lens) != len(want):
        raise Error("the store holds " + String(len(lens)) + " of the " + String(len(want)) + " clips named")
    var track = sm_object_track(poses, lens)

    var base = SuperMimicTrainer[C].for_robot(Robot.SIM.create())
    var ctr = base.load_state(ckpt)
    var rows_ref = rows.copy()
    var track_ref = track.copy()
    var orefs = sm_object_ref_rows[Robot](base.tables, rows_ref, track_ref)
    var tr = SMObjTrainer[OC](base^)
    tr.gain = gain
    if interface != "":
        var ic = tr.load_state(interface)
        print("interface", interface, "at iteration", ic[0], "gain", gain, "| tracker at iteration", ctr[0])
    else:
        print("NO interface checkpoint: the BLIND tracker (iteration", ctr[0], ")")
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    var lib_rows = rows.copy()          # the ORIGINAL reference, for the dump
    if ref_squeeze > 0.0 or ref_lower > 0.0:
        rows = sm_squeeze_reference[Robot](tr.base.tables, rows, lens, track, SM_OBJ, ref_squeeze, ARM0, ref_lower)
        var moved = 0.0
        for i in range(len(rows)):
            moved = max(moved, abs(Float64(rows[i]) - Float64(lib_rows[i])))
        print("reference edited: hands", ref_squeeze, "m toward the box, lowered by", ref_lower, "of their height above its centre; largest joint change", moved, "rad")
    var sq_w = sm_squeeze_weights(lens, track, SM_OBJ)
    var ks = List[Scalar[DT]](length=Robot.NDOF, fill=1)
    for j in range(Robot.NDOF):
        ks[j] = Scalar[DT](g1m_kp(j) * g1m_action_scale(j))
    var sq_scratch = List[Scalar[DT]](length=D.S + 2 * D.FK_BLOCK + 6, fill=0)
    var rw = sm_obj_reward_params()
    var terms = List[Scalar[DT]](length=SM_OBJ_TERMS + 1, fill=0)
    var env = SMTrackingEnv[Robot](
        1, SMMotionLibrary[Robot](rows^, lens, q0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0,
        max_episode_steps=0,
    )
    var rb = SMReplayOf[Robot](1, 64)
    var hist = TensorImpl[DT].alloc(D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(D.GOAL)
    var s_t = TensorImpl[DT].alloc(D.S)
    var o_t = TensorImpl[DT].alloc(SM_OBJ)
    var oref_t = TensorImpl[DT].alloc(SM_OREF)
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    comptime HL = SMHeadLayout[Robot]
    var hb = List[Int](length=2, fill=0)
    for i in range(2):
        hb[i] = Int(tr.base.tables.head[HL.HANDS + i])

    for c in range(len(lens)):
        var f0 = env.lib.clip_start[c]
        var first: List[Int] = [f0]
        env.restart(first)
        env.terminations = True
        # the box at its reference state of the clip's first frame (the device env's reset)
        var bq = List[Float64]()
        var bv = List[Float64]()
        for k in range(7):
            bq.append(Float64(track[f0 * SM_OBJ + k]))
        for k in range(6):
            bv.append(Float64(track[f0 * SM_OBJ + 7 + k]))
        env.lanes[0].sim.set_box(bq, bv)
        var z0 = Float64(track[f0 * SM_OBJ + 2])
        var trace = String(
            "step\tframe\tref_carry\tref_rise\tbox_err\tbox_dz\tbox_dxy\tbox_tilt\tbox_rise\thand0_box\thand1_box\tref_hand0_box\tref_hand1_box\t"
            + "da_arm\tda_arm_max\tda_rest\troot_speed\tref_root_speed\troot_z\tbox_speed\treward\tr_pos\tr_ori\tr_hands\thand_z\tref_hand_z\n"
        )
        var dump = String("")
        var step = 0
        var failed = False
        while True:
            env.next_frame[0] = f0
            env.policy_inputs(hist, goals)
            var fr = env.lanes[0].frame
            for k in range(D.S):
                s_t.data[k] = env.lanes[0].s[k]
            for k in range(7):
                o_t.data[k] = Scalar[DT](env.lanes[0].sim.box_qpos(k))
            for k in range(6):
                o_t.data[7 + k] = Scalar[DT](env.lanes[0].sim.box_qvel(k))
            for k in range(SM_OREF):
                oref_t.data[k] = orefs[fr * SM_OREF + k]
            var a = tr.act[1](hist, goals, s_t, o_t, oref_t)
            var ab = tr.base.act[1](hist, goals)
            if squeeze > 0.0:
                sm_squeeze_lane[Robot](
                    ARM0, Scalar[DT](squeeze * sq_w[fr]), tr.base.tables.tree_ptr(), tr.base.tables.head_ptr(),
                    ks.unsafe_ptr().as_unsafe_any_origin(), s_t.data.unsafe_ptr().as_unsafe_any_origin(),
                    o_t.data.unsafe_ptr().as_unsafe_any_origin(), sq_scratch.unsafe_ptr().as_unsafe_any_origin(),
                    a.data.unsafe_ptr().as_unsafe_any_origin(),
                )
            var da = 0.0
            var da_max = 0.0
            var da_rest = 0.0
            for j in range(Robot.NDOF):
                var d = Float64(a.data[j]) - Float64(ab.data[j])
                if j >= ARM0:
                    da += d * d
                    da_max = max(da_max, abs(d))
                else:
                    da_rest += d * d
            _ = env.step(a.data, rb)
            step += 1
            if env.lanes[0].ep_len == 0:        # the clip ended (or the robot fell); the lane is back at its first frame
                failed = env.last_terminated[0]
                break
            # after the step: the box against the row the step aimed at, the hands against the box
            var sp = env.lanes[0].s.unsafe_ptr().as_unsafe_any_origin()
            var pos = fk.unsafe_ptr().as_unsafe_any_origin()
            sm_fk_lane[DT, D.NB](
                tr.base.tables.tree_ptr(), sp, sp.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7),
                pos.unsafe_offset(D.NB * 10),
            )
            var bx = List[Float64](length=7, fill=0.0)
            for k in range(7):
                bx[k] = env.lanes[0].sim.box_qpos(k)
            var rf = fr * SM_OREF
            var dx = bx[0] - Float64(orefs[rf])
            var dy = bx[1] - Float64(orefs[rf + 1])
            var dz = bx[2] - Float64(orefs[rf + 2])
            var dot = 0.0
            for k in range(4):
                dot += bx[3 + k] * Float64(orefs[rf + 3 + k])
            var hd = List[Float64](length=2, fill=0.0)
            var rhd = List[Float64](length=2, fill=0.0)
            for i in range(2):
                var h2 = 0.0
                var r2 = 0.0
                for k in range(3):
                    var dh = Float64(fk[hb[i] * 3 + k]) - bx[k]
                    h2 += dh * dh
                    r2 += Float64(orefs[rf + SM_OBJ + 3 * i + k]) * Float64(orefs[rf + SM_OBJ + 3 * i + k])
                hd[i] = sqrt(h2)
                rhd[i] = sqrt(r2)
            var vr = sqrt(
                Float64(env.lanes[0].s[D.NQ]) * Float64(env.lanes[0].s[D.NQ]) + Float64(env.lanes[0].s[D.NQ + 1]) * Float64(env.lanes[0].s[D.NQ + 1])
            )
            var rr = fr * D.S + D.NQ
            var vref = sqrt(Float64(lib_rows[rr]) * Float64(lib_rows[rr]) + Float64(lib_rows[rr + 1]) * Float64(lib_rows[rr + 1]))
            var bs = 0.0
            for k in range(3):
                bs += env.lanes[0].sim.box_qvel(k) * env.lanes[0].sim.box_qvel(k)
            var rise = Float64(orefs[rf + 2]) - z0
            # the object's three reward terms of this step, as the imagined rollout computes them
            for k in range(7):
                o_t.data[k] = Scalar[DT](bx[k])
            _ = sm_obj_reward_lane[DT, Robot](
                tr.base.tables.head_ptr(), pos, o_t.data.unsafe_ptr().as_unsafe_any_origin(),
                orefs.unsafe_ptr().unsafe_offset(rf).as_unsafe_any_origin(), rw.unsafe_ptr().as_unsafe_any_origin(),
                terms.unsafe_ptr().as_unsafe_any_origin(),
            )
            # the hands' mean height above the box's centre: simulated, and the reference's (its hand rows are in the box's frame)
            var hz = (Float64(fk[hb[0] * 3 + 2]) + Float64(fk[hb[1] * 3 + 2])) / 2.0 - bx[2]
            trace += (
                String(step) + "\t" + String(fr - f0) + "\t" + ("1" if rise >= CARRY_RISE else "0") + "\t" + String(rise) + "\t"
                + String(sqrt(dx * dx + dy * dy + dz * dz)) + "\t" + String(dz) + "\t" + String(sqrt(dx * dx + dy * dy)) + "\t"
                + String(2.0 * acos(min(abs(dot), 1.0))) + "\t" + String(bx[2] - z0) + "\t" + String(hd[0]) + "\t" + String(hd[1]) + "\t"
                + String(rhd[0]) + "\t" + String(rhd[1]) + "\t" + String(sqrt(da)) + "\t" + String(da_max) + "\t" + String(sqrt(da_rest)) + "\t"
                + String(vr) + "\t" + String(vref) + "\t" + String(Float64(env.lanes[0].s[2])) + "\t" + String(sqrt(bs)) + "\t"
                + String(env.last_reward[0]) + "\t" + String(Float64(terms[0])) + "\t" + String(Float64(terms[1])) + "\t"
                + String(Float64(terms[2])) + "\t" + String(hz) + "\t0\n"
            )
            for k in range(D.NQ):
                dump += String(Float64(env.lanes[0].s[k])) + "\t"
            for k in range(7):
                dump += String(bx[k]) + "\t"
            for k in range(7):
                dump += String(Float64(orefs[rf + k])) + "\t"
            for k in range(D.NQ):
                dump += String(Float64(lib_rows[fr * D.S + k])) + ("\n" if k == D.NQ - 1 else "\t")
        write_text_atomic(out_dir + "/" + names[c] + ".trace.tsv", trace)
        write_text_atomic(out_dir + "/" + names[c] + ".qpos.tsv", dump)
        print("  ", names[c], "|", step, "steps of", lens[c], "| FELL" if failed else "| stood")
