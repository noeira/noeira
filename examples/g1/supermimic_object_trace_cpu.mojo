# +--------------------------------------------------------------------------+ #
# | SuperMimic's object interface on the CPU: one clip at a time, traced
# +--------------------------------------------------------------------------+ #
"""A DIAGNOSTIC, not a score: the object interface (or the blind tracker) on
named carry clips, one lane on the CPU, with a row per step saying where the
box is against its reference, where the hands are against the box, and how
far the interface's action is from the tracker's own.

    pixi run mojo build -I . examples/g1/supermimic_object_trace_cpu.mojo -o build/sm_obj_trace
    pixi run ./build/sm_obj_trace --ckpt TRACKER.ckpt --store CARRY_50HZ.h5 --object-dir OBJECT_CSV_DIR \\
        --only NAME[,NAME…] --out-dir DIR [--interface INTERFACE.ckpt] [--gain 0.1] [--near 0.6 --far 1.0]
        [--squeeze 40] [--ref-squeeze 0.05] [--ref-lower 1.0] [--trials 8 --seed 0] [--lead-in 50] [--csv OUT.csv]
        [--box-load KG] [--box-mass-scale S --box-mass-weights 1] [--box-lane-mass S]
        [--teacher 1 --ft-squeeze 10 --ft-squeeze-max 60 --ft-mu 0.5 --ft-mass 1 --ft-kp 200 --ft-kd 20 --ft-kr 5 --ft-kw 0.5
         --ft-point 0.11,-0.01,0 --ft-fmax 60 --ft-rel 1 --ft-mmax 1.5
         --ft-kh 300 --ft-hmax 30 --ft-reach-lead 35]

    --csv OUT.csv    one row a run with `supermimic_object_eval.mojo --csv`'s columns (the same account of the
                     carry, `object_task.SMObjCarryTally`, and the same kernels): the laptop's stand-in for the
                     device benchmark. A run's last step is in neither the carry's account nor the kernels here.

    --box-load KG    a constant vertical force on the box, the weight of KG more kilograms (negative: lighter), for
                     the whole run: weight without inertia (`TrackingSim.set_box_load`). Needs a build with
                     `-D SM_XFRC=1`; without it any KG other than 0 raises.

    --box-mass-scale S  the box's mass times S in the simulator's model, set once (`TrackingSim.scale_box_mass`): its
                     mass and inertia, and with `--box-mass-weights 1` (the default) the solver's weights of the box,
                     which together are what an asset built with that mass holds; `--box-mass-weights 0` leaves the
                     weights at the asset's.

    --box-lane-mass S  the box's mass times S through the simulator's per-lane mass (`TrackingSim.set_box_lane_mass`),
                     the path a device batch uses; the model is not touched. Needs a build with `-D SM_LANE_MASS=1`.

    --near A --far B the interface's gain by the box's horizontal distance to the robot's root: whole within A metres,
                     zero beyond B, linear between (the tracker alone walks up to a box it has never seen from afar).
                     0 0, the default: the gain is whole everywhere.
    --squeeze F      the scripted squeeze on the executed actions (`object_task.sm_squeeze_lane`), F newtons
    --ref-squeeze D  the reference's hands D metres toward the box while it is lifted (`sm_squeeze_reference`)
    --teacher 1      the force teacher on the executed actions (`object_task.sm_force_teacher_lane`): the wrench that
                     takes the box to its reference, shared between the two palms; `--ft-*` are its numbers
                     (`--ft-point`: where the LEFT hand's force acts, in the wrist-yaw link's frame, m; `--ft-rel 1`: the
                     box's target seen from the robot's root; `--ft-mmax`: the largest moment a hand gives, N·m;
                     `--ft-kh`: N/m pulling each hand to the reference's place on the box, from `--ft-reach-lead`
                     frames before the reference lifts, at most `--ft-hmax` N)
    --ref-lower S    … and DOWN by the share S of their height above the box's centre (the arms alone)
    The box's reference and the reward's reference grasp stay the ORIGINAL clip's under both edits.
    --lead-in N      N frames of standing at each clip's first pose before the clip (`object_task.sm_lead_in`; 50 = 1 s).
                  Half the carry clips lift the box within a second of their first frame; train and score with the same N.
    --trials K       every clip K times, each from a start perturbed by the training's reset noise (the robot's
                     root and joints; the box stays at its reference), as `sm_obj_benchmark`'s `trials`; the
                     files are then NAME.tI.trace.tsv / NAME.tI.qpos.tsv. The draws are this process's, seeded
                     by --seed: the same rule as the device benchmark's, not the same starts.

Per clip, in `--out-dir`:

    NAME.trace.tsv   one row a step (columns in the header; `k_all` and `k_trunk_legs` are the six tracking
                     kernels of the step over their maximum, `heads.sm_tracking_kernels_lane`; `tau_arm` and
                     `tau_wrist` are the largest share of its torque limit that a joint of the two arms, and one
                     of the four wrist pitch and yaw joints, was given in the step's last substep)
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
from noeira.deep_agents.supermimic.eval import SM_MAX_STEP_REWARD
from noeira.deep_agents.supermimic.heads import SMHeadLayout, sm_tracking_kernels_lane
from noeira.deep_agents.supermimic.kinematics import sm_fk_lane
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import sm_object_track
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_spec import SM_OBJ
from noeira.deep_agents.supermimic.object_task import (
    SM_OBJ_HELD_SHARE, SM_OBJ_LIFT, SM_OBJ_TERMS, SM_OREF, sm_obj_carry_steps, sm_obj_carry_tally, sm_obj_reward_lane,
    sm_obj_reward_params, sm_obj_lean_deg, SM_OBJ_LEVEL, sm_object_ref_rows, sm_squeeze_lane, sm_squeeze_reference, sm_lead_in, sm_squeeze_weights, sm_trunk_legs_mask,
    sm_force_teacher_lane, SM_FT_PARAMS, FT_SQUEEZE, FT_SQUEEZE_MAX, FT_MU, FT_MASS, FT_KP, FT_KD, FT_KR, FT_KW, FT_POINT, FT_FMAX, FT_REL, FT_MMAX, FT_KH, FT_HMAX, SM_FT_OUT,
)
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBoxRobot, SMDims
from noeira.envs.robots.unitree_g1_mjlab_pd import g1m_action_scale, g1m_effort, g1m_kp
from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime HO = get_defined_int["SM_OBJ_H", 1024]()
comptime Robot = G1MjlabBoxRobot
comptime D = SMDims[Robot]
comptime ARM0 = 15              # joints 15..28 are the two arms
comptime CARRY_RISE = SM_OBJ_LIFT
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
    var near = Float64(_opt(args, String("--near"), String("0")))
    var far = Float64(_opt(args, String("--far"), String("0")))
    var squeeze = Float64(_opt(args, String("--squeeze"), String("0")))
    var ref_squeeze = Float64(_opt(args, String("--ref-squeeze"), String("0")))
    var ref_lower = Float64(_opt(args, String("--ref-lower"), String("0")))
    var trials = Int(_opt(args, String("--trials"), String("1")))
    var rng_seed = Int(_opt(args, String("--seed"), String("0")))
    var csv_path = _opt(args, String("--csv"), String(""))
    var box_load = Float64(_opt(args, String("--box-load"), String("0")))
    var box_mass_scale = Float64(_opt(args, String("--box-mass-scale"), String("1")))
    var box_mass_weights = Int(_opt(args, String("--box-mass-weights"), String("1"))) != 0
    var box_lane_mass = Float64(_opt(args, String("--box-lane-mass"), String("0")))
    var teacher = Int(_opt(args, String("--teacher"), String("0"))) != 0
    var ft = List[Scalar[DT]](length=SM_FT_PARAMS, fill=0)
    ft[FT_SQUEEZE] = Scalar[DT](Float64(_opt(args, String("--ft-squeeze"), String("10"))))
    ft[FT_SQUEEZE_MAX] = Scalar[DT](Float64(_opt(args, String("--ft-squeeze-max"), String("60"))))
    ft[FT_MU] = Scalar[DT](Float64(_opt(args, String("--ft-mu"), String("0.5"))))
    ft[FT_MASS] = Scalar[DT](Float64(_opt(args, String("--ft-mass"), String("1"))))
    ft[FT_KP] = Scalar[DT](Float64(_opt(args, String("--ft-kp"), String("200"))))
    ft[FT_KD] = Scalar[DT](Float64(_opt(args, String("--ft-kd"), String("20"))))
    ft[FT_KR] = Scalar[DT](Float64(_opt(args, String("--ft-kr"), String("5"))))
    ft[FT_KW] = Scalar[DT](Float64(_opt(args, String("--ft-kw"), String("0.5"))))
    ft[FT_FMAX] = Scalar[DT](Float64(_opt(args, String("--ft-fmax"), String("60"))))
    ft[FT_REL] = Scalar[DT](Float64(_opt(args, String("--ft-rel"), String("0"))))
    ft[FT_MMAX] = Scalar[DT](Float64(_opt(args, String("--ft-mmax"), String("0"))))
    ft[FT_KH] = Scalar[DT](Float64(_opt(args, String("--ft-kh"), String("0"))))
    ft[FT_HMAX] = Scalar[DT](Float64(_opt(args, String("--ft-hmax"), String("30"))))
    var reach_lead = Int(_opt(args, String("--ft-reach-lead"), String("35")))
    var ft_point = _opt(args, String("--ft-point"), String("0.11,-0.01,0")).split(",")
    if len(ft_point) != 3:
        raise Error("--ft-point takes x,y,z")
    for k in range(3):
        ft[FT_POINT + k] = Scalar[DT](Float64(String(String(ft_point[k]).strip())))
    if teacher:
        var line = String("force teacher:")
        for k in range(SM_FT_PARAMS):
            line += " " + String(Float64(ft[k]))
        print(line, "(squeeze, its ceiling, mu, mass, kp, kd, kr, kw, the left hand's point x y z, the largest force, relative to the root, the largest moment, the hands' pull N/m and its cap)")
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
    var lead = Int(_opt(args, String("--lead-in"), String("0")))
    if lead > 0:
        sm_lead_in[Robot](rows, lens, poses, lead)
        print("lead-in:", lead, "frames of standing at each clip's first pose before it")
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
    var reach_w = sm_squeeze_weights(lens, track, SM_OBJ, reach_lead)
    var ks = List[Scalar[DT]](length=Robot.NDOF, fill=1)
    for j in range(Robot.NDOF):
        ks[j] = Scalar[DT](g1m_kp(j) * g1m_action_scale(j))
    var sq_scratch = List[Scalar[DT]](length=D.S + 2 * D.FK_BLOCK + SM_FT_OUT, fill=0)
    var rw = sm_obj_reward_params()
    var terms = List[Scalar[DT]](length=SM_OBJ_TERMS + 1, fill=0)
    var env = SMTrackingEnv[Robot](
        1, SMMotionLibrary[Robot](rows^, lens, q0), dr=False, obs_noise=False, reset_noise=trials > 1, action_noise=0.0,
        rng_seed=rng_seed, max_episode_steps=0,
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
    var mask_all = List[Scalar[DT]](length=Robot.N_TRACKED, fill=1)
    var mask_tl = sm_trunk_legs_mask[Robot]()
    var rf_row = List[Scalar[DT]](length=D.REF, fill=0)
    var k6 = List[Scalar[DT]](length=6, fill=0)
    var csv = String(
        "clip,name,frames,steps,norm_return,fell,carry_steps,held_steps,max_err_carry_m,mean_err_carry_m,success,trial,"
        + "carried,late_steps,late_up_steps,mean_err_late_m,kernels_all,kernels_trunk_legs,mean_lean_late_deg,level\n"
    )
    var n_level = 0
    var n_bar_level = 0
    var sum_lean = 0.0
    var n_runs = 0
    var n_bar = 0
    var n_carried = 0
    var n_fell = 0
    var sum_carry = 0
    var sum_held = 0
    var sum_hand = 0.0
    var sum_ret = 0.0
    var sum_ka = 0.0
    var sum_kt = 0.0

    if box_mass_scale != 1.0:
        env.lanes[0].sim.scale_box_mass(box_mass_scale, box_mass_weights)
        print("the box's mass times", box_mass_scale, "(the solver's weights", "with it)" if box_mass_weights else "left at the asset's)")
    if box_lane_mass != 0.0:
        env.lanes[0].sim.set_box_lane_mass(box_lane_mass)
        print("the box's mass times", box_lane_mass, "through the per-lane mass")
    for c in range(len(lens)):
        var f0 = env.lib.clip_start[c]
        var clip_carry = sm_obj_carry_steps(orefs, f0, lens[c])
        for trial in range(max(trials, 1)):
            var tally = sm_obj_carry_tally(clip_carry)
            var ret = 0.0
            var ka = 0.0
            var kt = 0.0
            var k_steps = 0
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
            if box_load != 0.0:
                env.lanes[0].sim.set_box_load(box_load)
            var z0 = Float64(track[f0 * SM_OBJ + 2])
            var trace = String(
                "step\tframe\tref_carry\tref_rise\tbox_err\tbox_dz\tbox_dxy\tbox_tilt\tbox_rise\thand0_box\thand1_box\tref_hand0_box\tref_hand1_box\t"
                + "da_arm\tda_arm_max\tda_rest\troot_speed\tref_root_speed\troot_z\tbox_speed\treward\tr_pos\tr_ori\tr_hands\thand_z\tref_hand_z\tk_all\tk_trunk_legs\tft_f0\tft_f1\tft_m\tbox_lean\ttau_arm\ttau_wrist\n"
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
                if far > 0.0:
                    var bx = Float64(o_t.data[0]) - Float64(s_t.data[0])
                    var by = Float64(o_t.data[1]) - Float64(s_t.data[1])
                    var dist = sqrt(bx * bx + by * by)
                    tr.gain = gain * min(max((far - dist) / max(far - near, 1e-9), 0.0), 1.0)
                var a = tr.act[1](hist, goals, s_t, o_t, oref_t)
                var ab = tr.base.act[1](hist, goals)
                if squeeze > 0.0:
                    sm_squeeze_lane[Robot](
                        ARM0, Scalar[DT](squeeze * sq_w[fr]), tr.base.tables.tree_ptr(), tr.base.tables.head_ptr(),
                        ks.unsafe_ptr().as_unsafe_any_origin(), s_t.data.unsafe_ptr().as_unsafe_any_origin(),
                        o_t.data.unsafe_ptr().as_unsafe_any_origin(), sq_scratch.unsafe_ptr().as_unsafe_any_origin(),
                        a.data.unsafe_ptr().as_unsafe_any_origin(),
                    )
                var ft_f0 = 0.0
                var ft_f1 = 0.0
                var ft_m = 0.0
                if teacher:
                    sm_force_teacher_lane[Robot](
                        ARM0, Scalar[DT](sq_w[fr]), Scalar[DT](reach_w[fr]), ft.unsafe_ptr().as_unsafe_any_origin(), tr.base.tables.tree_ptr(), tr.base.tables.head_ptr(),
                        ks.unsafe_ptr().as_unsafe_any_origin(), s_t.data.unsafe_ptr().as_unsafe_any_origin(),
                        lib_rows.unsafe_ptr().unsafe_offset(fr * D.S).as_unsafe_any_origin(),
                        o_t.data.unsafe_ptr().as_unsafe_any_origin(), oref_t.data.unsafe_ptr().as_unsafe_any_origin(),
                        sq_scratch.unsafe_ptr().as_unsafe_any_origin(), a.data.unsafe_ptr().as_unsafe_any_origin(),
                    )
                    var fo = D.S + 2 * D.FK_BLOCK
                    for k in range(3):
                        ft_f0 += Float64(sq_scratch[fo + k]) * Float64(sq_scratch[fo + k])
                        ft_f1 += Float64(sq_scratch[fo + 3 + k]) * Float64(sq_scratch[fo + 3 + k])
                    ft_f0 = sqrt(ft_f0)
                    ft_f1 = sqrt(ft_f1)
                    ft_m = Float64(sq_scratch[fo + 6])
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
                ret += env.last_reward[0]
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
                # the carry's account and the six tracking kernels, as the device benchmark's (`fr`: the frame the reward used)
                var lean = sm_obj_lean_deg(
                    bx[3], bx[4], bx[5], bx[6], Float64(orefs[rf + 3]), Float64(orefs[rf + 4]), Float64(orefs[rf + 5]), Float64(orefs[rf + 6])
                )
                tally.note(rise, bx[2] - z0, sqrt(dx * dx + dy * dy + dz * dz), lean)
                env.lib.reference_row(fr, tr.base.tables, rf_row.unsafe_ptr().as_unsafe_any_origin())
                var ka_s = Float64(sm_tracking_kernels_lane[DT, Robot](
                    tr.base.tables.head_ptr(), pos, rf_row.unsafe_ptr().as_unsafe_any_origin(), mask_all.unsafe_ptr().as_unsafe_any_origin(),
                    k6.unsafe_ptr().as_unsafe_any_origin(),
                )) / SM_MAX_STEP_REWARD
                var kt_s = Float64(sm_tracking_kernels_lane[DT, Robot](
                    tr.base.tables.head_ptr(), pos, rf_row.unsafe_ptr().as_unsafe_any_origin(), mask_tl.unsafe_ptr().as_unsafe_any_origin(),
                    k6.unsafe_ptr().as_unsafe_any_origin(),
                )) / SM_MAX_STEP_REWARD
                ka += ka_s
                kt += kt_s
                k_steps += 1
                # the arms' torques against their limits (joints 15 .. 28; wrist pitch and yaw: 20, 21, 27, 28)
                var tau_arm = 0.0
                var tau_wrist = 0.0
                for j in range(15, 29):
                    var u = abs(env.lanes[0].sim.joint_torque(j)) / g1m_effort(j)
                    tau_arm = max(tau_arm, u)
                    if j == 20 or j == 21 or j == 27 or j == 28:
                        tau_wrist = max(tau_wrist, u)
                trace += (
                    String(step) + "\t" + String(fr - f0) + "\t" + ("1" if rise >= CARRY_RISE else "0") + "\t" + String(rise) + "\t"
                    + String(sqrt(dx * dx + dy * dy + dz * dz)) + "\t" + String(dz) + "\t" + String(sqrt(dx * dx + dy * dy)) + "\t"
                    + String(2.0 * acos(min(abs(dot), 1.0))) + "\t" + String(bx[2] - z0) + "\t" + String(hd[0]) + "\t" + String(hd[1]) + "\t"
                    + String(rhd[0]) + "\t" + String(rhd[1]) + "\t" + String(sqrt(da)) + "\t" + String(da_max) + "\t" + String(sqrt(da_rest)) + "\t"
                    + String(vr) + "\t" + String(vref) + "\t" + String(Float64(env.lanes[0].s[2])) + "\t" + String(sqrt(bs)) + "\t"
                    + String(env.last_reward[0]) + "\t" + String(Float64(terms[0])) + "\t" + String(Float64(terms[1])) + "\t"
                    + String(Float64(terms[2])) + "\t" + String(hz) + "\t0\t" + String(ka_s) + "\t" + String(kt_s) + "\t" + String(ft_f0) + "\t" + String(ft_f1) + "\t" + String(ft_m) + "\t" + String(lean) + "\t" + String(tau_arm) + "\t" + String(tau_wrist) + "\n"
                )
                for k in range(D.NQ):
                    dump += String(Float64(env.lanes[0].s[k])) + "\t"
                for k in range(7):
                    dump += String(bx[k]) + "\t"
                for k in range(7):
                    dump += String(Float64(orefs[rf + k])) + "\t"
                for k in range(D.NQ):
                    dump += String(Float64(lib_rows[fr * D.S + k])) + ("\n" if k == D.NQ - 1 else "\t")
            var stem = out_dir + "/" + names[c] + ((".t" + String(trial)) if trials > 1 else String(""))
            write_text_atomic(stem + ".trace.tsv", trace)
            write_text_atomic(stem + ".qpos.tsv", dump)
            var at_bar = (not failed) and tally.at_bar()
            var was_carried = (not failed) and tally.carried()
            var was_level = was_carried and tally.level()
            var kn = Float64(max(k_steps, 1))
            var nret = ret / (SM_MAX_STEP_REWARD * Float64(max(step, 1)))
            csv += (
                String(c) + "," + names[c] + "," + String(lens[c]) + "," + String(step) + "," + String(nret) + "," + ("1" if failed else "0") + ","
                + String(tally.carry) + "," + String(tally.held) + "," + String(tally.emax) + "," + String(tally.mean_err()) + ","
                + ("1" if at_bar else "0") + "," + String(trial) + "," + ("1" if was_carried else "0") + "," + String(tally.late) + ","
                + String(tally.late_up) + "," + String(tally.mean_err_late()) + "," + String(ka / kn) + "," + String(kt / kn) + ","
                + String(tally.mean_lean_late()) + "," + ("1" if was_level else "0") + "\n"
            )
            n_runs += 1
            sum_carry += tally.carry
            sum_held += tally.held
            sum_ret += nret
            sum_ka += ka / kn
            sum_kt += kt / kn
            if failed:
                n_fell += 1
            if at_bar:
                n_bar += 1
            if was_carried:
                n_carried += 1
                sum_hand += tally.mean_err_late()
                sum_lean += tally.mean_lean_late()
            if was_level:
                n_level += 1
                if at_bar:
                    n_bar_level += 1
            print(
                "  ", names[c], "trial", trial, "|", step, "steps of", lens[c], "| FELL" if failed else "| stood", "| carry steps", tally.carry, "held",
                tally.held, "| AT THE BAR" if at_bar else "| under the bar", "| carried," if was_carried else "| not carried,", tally.mean_err_late(),
                "m once in hand, leaning", tally.mean_lean_late(), "deg,", "level" if was_level else "not level", "| kernels all", ka / kn,
                "trunk and legs", kt / kn,
            )
    var nr = Float64(max(n_runs, 1))
    print(
        "\n", n_runs, "runs | at the bar", n_bar, "(within 10 cm on >=", SM_OBJ_HELD_SHARE, "of the carry steps, no fall) | carried", n_carried,
        "(no fall, the box up on more than half of the carry's second half) | level", n_level, "(carried, lean at most", SM_OBJ_LEVEL,
        "deg over that half) | at the bar and level", n_bar_level, "| lean once in hand", sum_lean / Float64(max(n_carried, 1)),
        "deg | error once in hand", sum_hand / Float64(max(n_carried, 1)),
        "m | fell", n_fell, "| carry steps held", sum_held, "/", sum_carry, "| normalised return", sum_ret / nr, "| six kernels: all bodies",
        sum_ka / nr, ", trunk and legs", sum_kt / nr,
    )
    if csv_path != "":
        write_text_atomic(csv_path, csv)
        print("per-run results:", csv_path)
