"""SuperMimic's object interface: score it on carry clips, each from its first frame.

`deep_agents/supermimic/object_eval.sm_obj_benchmark`: no noise, no
randomisation, every clip to its end or to a fall, and for the box the counts
`docs/OBJECT_INTERACTION_PLAN.md`'s O1.5 asks for (carry steps, steps within
10 cm of the reference, the largest distance while the reference carries).

Four readings of the box, side by side: runs AT THE BAR (within 10 cm on 95 %
of the carry steps, no fall), runs CARRIED (no fall, the box up on more than
half of the carry's second half), the error once in hand (the mean distance
over that half, carried runs), and the box's LEAN over that half (a carried
run is LEVEL at 15 degrees or less; the first three read the box's centre
and pass a box that hangs from the hands). Two of the tracking: the tracker's
normalised return, and the six kernels over the trunk and the legs (O1.5's
criterion, against the same number of a run without `--interface`).

    pixi run -e nvidia mojo build -I . -D SM_PAPER=1 examples/g1/supermimic_object_eval.mojo -o /tmp/sm_obj_eval
    /tmp/sm_obj_eval --ckpt TRACKER.ckpt --store CARRY_50HZ.h5 --object-dir OBJECT_CSV_DIR --clips CLIPS.tsv \\
        [--interface INTERFACE.ckpt] [--split eval] [--gain 0.1] [--csv OUT.csv]
        [--trials 8] [--seed 0] [--honesty 500] [--squeeze 40] [--lead-in 50] [--mass-scale 3] [--w-pos 1.0] [--std-pos 0.2] [--w-ori 0.5] [--std-ori 0.4] [--w-hands 1.0] [--std-hands 0.1]

    --interface  `supermimic_object_train.mojo --out`'s checkpoint. WITHOUT it the object encoder is at
                 its zero start and the score is the BLIND tracker's on the same clips: the number to beat.
    --split      `eval` (the held-out clips, default) or `train`
    --trials     every clip this many times, each start perturbed by the training's reset noise (1: the
                 deterministic benchmark). The summary then gives the share of carry steps held per trial
                 (mean, lowest, highest) and how many runs pass; rank arms on these, not on one trial.
    --mass-scale S  the box weighs the model's mass times S in every lane (`physics3d/dynamics/lane_mass.mojo`); needs a
                 build with `-D SM_LANE_MASS=1`. Not applied to `--honesty`.
    --lead-in N  N frames of standing at each clip's first pose before the clip (`object_task.sm_lead_in`; 50 = 1 s).
                  Half the carry clips lift the box within a second of their first frame; train and score with the same N.
    --squeeze    the benchmark with the scripted squeeze of this many newtons on every lane
                 (`object_agent_gpu.SMObjSqueeze`): a probe of that exploration prior, not a score of the interface
    --honesty    INSTEAD of the benchmark: O1.4's gate, the imagined rollout's return against the simulator's
                 over the same starts, for this many steps (`object_eval.sm_obj_honesty`). The six reward
                 numbers must be the training run's.

Compile-time sizes: `-D SM_PAPER=1` (a paper checkpoint needs it), `-D
SM_LANES=64`, `-D SM_OBJ_H=1024` (the interface checkpoint's object network).

⚠ NVIDIA only (the batched G1 does not build for Metal). mjlab's G1 only.
"""

from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import sm_object_track
from noeira.deep_agents.supermimic.object_eval import SMObjHonesty, sm_obj_benchmark, sm_obj_honesty
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_task import (
    SM_OBJ_HELD_SHARE, SM_OBJ_LEVEL, SM_OSTD_HANDS, SM_OSTD_ORI, SM_OSTD_POS, SM_OW_HANDS, SM_OW_ORI, SM_OW_POS, sm_lead_in, sm_object_ref_rows,
    sm_obj_reward_params,
)
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBoxBatchedSim, G1MjlabBoxRobot, SMDims
from noeira.envs.robots.unitree_g1_mjlab_pd import g1m_action_scale, g1m_kp
from noeira.io.fileio import read_file_bytes
from noeira.nn.constants import DT


comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime LANES = get_defined_int["SM_LANES", 64]()
comptime HO = get_defined_int["SM_OBJ_H", 1024]()
comptime Robot = G1MjlabBoxRobot
comptime Sim = G1MjlabBoxBatchedSim[LANES]
comptime D = SMDims[Robot]
# the widths of `supermimic_object_train.mojo` (a checkpoint loads only into its own config); the batch shapes are unused here
comptime C = SMConfig[
    "gpu", Robot,
    2048 if P else 128, 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    LANES, 64, 16, 32,
]
comptime OC = SMObjConfig[C, HO, 256 if P else 64, 256 if P else 64, 256 if P else 64, 128 if P else 32]


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


def _names(path: String, split: String) raises -> List[String]:
    """The TSV's rows of `split` (`name` and `split` columns first)."""
    var out = List[String]()
    var first = True
    for line in string_from_bytes(read_file_bytes(path)).split("\n"):
        var t = String(String(line).strip())
        if t.byte_length() == 0:
            continue
        var cols = List[String]()
        for c in t.split("\t"):
            cols.append(String(String(c).strip()))
        if first:
            first = False
            if len(cols) < 2 or cols[0] != "name" or cols[1] != "split":
                raise Error(path + ": the header's first two columns must be `name` and `split`")
            continue
        if len(cols) >= 2 and cols[1] == split:
            out.append(cols[0].copy())
    return out^


def _pct(imag: Float64, sim: Float64) -> Float64:
    return 100.0 * (imag - sim) / max(abs(sim), 1e-9)


def _honesty_line(name: String, h: SMObjHonesty):
    if h.windows == 0:
        print("  ", name, ": no window (", h.dropped, "dropped )")
        return
    print(
        "  ", name, ":", h.windows, "windows (", h.dropped, "dropped ) | tracker terms imagined", h.imag_track, "simulated", h.sim_track,
        "(", _pct(h.imag_track, h.sim_track), "% ), mean |gap|", h.gap_track,
        "| object terms pos/ori/hands imagined", h.imag_pos, h.imag_ori, h.imag_hands, "simulated", h.sim_pos, h.sim_ori, h.sim_hands,
        "| all eleven imagined", h.imag_total(), "simulated", h.sim_total(), "(", _pct(h.imag_total(), h.sim_total()), "% ), mean |gap|", h.gap_total,
        "| imagined box off the simulated one by", h.err_1, h.err_8, h.err_16, h.err_h, "m after 1, 8, 16, H steps",
    )


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var interface = _opt(args, String("--interface"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var object_dir = _opt(args, String("--object-dir"), String(""))
    var clips = _opt(args, String("--clips"), String(""))
    var split = _opt(args, String("--split"), String("eval"))
    var csv_path = _opt(args, String("--csv"), String(""))
    var gain = Float64(_opt(args, String("--gain"), String(OC.DEF_GAIN)))
    var honesty = Int(_opt(args, String("--honesty"), String("0")))
    var squeeze = Float64(_opt(args, String("--squeeze"), String("0")))
    var trials = Int(_opt(args, String("--trials"), String("1")))
    var eval_seed = Int(_opt(args, String("--seed"), String("0")))
    var rw = sm_obj_reward_params(
        Float64(_opt(args, String("--w-pos"), String(SM_OW_POS))), Float64(_opt(args, String("--std-pos"), String(SM_OSTD_POS))),
        Float64(_opt(args, String("--w-ori"), String(SM_OW_ORI))), Float64(_opt(args, String("--std-ori"), String(SM_OSTD_ORI))),
        Float64(_opt(args, String("--w-hands"), String(SM_OW_HANDS))), Float64(_opt(args, String("--std-hands"), String(SM_OSTD_HANDS))),
    )
    if ckpt == "" or store == "" or object_dir == "" or clips == "":
        raise Error("give --ckpt TRACKER.ckpt --store STORE.h5 --object-dir DIR --clips CLIPS.tsv [--interface INTERFACE.ckpt]")

    var want = _names(clips, split)
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
        raise Error("the store holds " + String(len(lens)) + " of the " + String(len(want)) + " `" + split + "` clips named")
    var lead = Int(_opt(args, String("--lead-in"), String("0")))
    var mass_scale = Float64(_opt(args, String("--mass-scale"), String("1")))
    if mass_scale != 1.0:
        print("the box's mass: the model's times", mass_scale, "in every lane")
    if lead > 0:
        sm_lead_in[Robot](rows, lens, poses, lead)
        print("lead-in:", lead, "frames of standing at each clip's first pose before it")
    var track = sm_object_track(poses, lens)

    var ctx = DeviceContext()
    var base = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), Optional(ctx))
    var ctr = base.load_state(ckpt)
    var rows_ref = rows.copy()
    var track_ref = track.copy()
    var orefs = sm_object_ref_rows[Robot](base.tables, rows_ref, track_ref)
    var tr = SMObjTrainer[OC](base^)
    tr.gain = gain
    if interface != "":
        var ic = tr.load_state(interface)
        print("interface", interface, "at iteration", ic[0], "(", ic[1], "updates ), gain", gain, "| tracker at iteration", ctr[0])
    else:
        print("NO interface checkpoint: the object encoder is at its zero start, this is the BLIND tracker (iteration", ctr[0], ")")
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    print(len(lens), "`" + split + "` clips,", len(rows) // D.S, "frames,", LANES, "lanes")
    if honesty > 0:
        var hs = sm_obj_honesty[OC, Sim](ctx, tr, SMMotionLibrary[Robot](rows^, lens, q0), track, orefs, rw, Sim.create(ctx), honesty)
        print("imagined against simulated return,", C.HORIZON, "steps a window,", honesty, "simulated steps, sums over the horizon:")
        _honesty_line(String("reference not carrying"), hs[0])
        _honesty_line(String("reference carrying"), hs[1])
        var w = hs[0].windows + hs[1].windows
        if w > 0:
            var a = Float64(hs[0].windows) / Float64(w)
            var b = Float64(hs[1].windows) / Float64(w)
            var it = a * hs[0].imag_track + b * hs[1].imag_track
            var st_ = a * hs[0].sim_track + b * hs[1].sim_track
            var ia = a * hs[0].imag_total() + b * hs[1].imag_total()
            var sa = a * hs[0].sim_total() + b * hs[1].sim_total()
            print(
                "   all", w, "windows | tracker terms imagined", it, "simulated", st_, "(", _pct(it, st_), "% ) | all eleven imagined", ia,
                "simulated", sa, "(", _pct(ia, sa), "% ) | the gate: within 4.2 % (twice the tracker's own 2.1 %)",
            )
        return
    var ks = List[Float64](length=Robot.NDOF, fill=1.0)
    for j in range(Robot.NDOF):
        ks[j] = g1m_kp(j) * g1m_action_scale(j)
    if squeeze > 0.0:
        print("SCRIPTED SQUEEZE of", squeeze, "N on every lane while the reference box is lifted: a probe, not the interface's score")
    var res = sm_obj_benchmark[OC, Sim](
        ctx, tr, SMMotionLibrary[Robot](rows^, lens, q0), track, orefs, Sim.create(ctx), 0, True, squeeze, ks, 15, trials, eval_seed,
        mass_scale=mass_scale,
    )
    var t_carry = List[Int](length=max(trials, 1), fill=0)
    var t_held = List[Int](length=max(trials, 1), fill=0)
    var t_pass = List[Int](length=max(trials, 1), fill=0)
    var t_half = List[Int](length=max(trials, 1), fill=0)
    var t_fell = List[Int](length=max(trials, 1), fill=0)
    var t_carried = List[Int](length=max(trials, 1), fill=0)
    var c_pass = List[Int](length=len(lens), fill=0)

    var fell = 0
    var ok = 0
    var strict = 0
    var no_carry = 0
    var ret = 0.0
    var carry = 0
    var held = 0
    var carried = 0
    var hand_err = 0.0          # Σ over carried runs of the mean distance in the carry's second half
    var hand_lean = 0.0         # … of the mean lean
    var level = 0
    var bar_level = 0
    var k_all = 0.0
    var k_tl = 0.0
    var csv = String(
        "clip,name,frames,steps,norm_return,fell,carry_steps,held_steps,max_err_carry_m,mean_err_carry_m,success,trial,"
        + "carried,late_steps,late_up_steps,mean_err_late_m,kernels_all,kernels_trunk_legs,mean_lean_late_deg,level\n"
    )
    for i in range(len(res)):
        var r = res[i].copy()
        if r.fell:
            fell += 1
        if r.success():
            ok += 1
        if r.strict():
            strict += 1
        if r.carry_steps == 0:
            no_carry += 1
        ret += r.norm_return / Float64(len(res))
        k_all += r.kernels_all / Float64(len(res))
        k_tl += r.kernels_trunk_legs / Float64(len(res))
        if r.carried():
            carried += 1
            hand_err += r.mean_err_late
            hand_lean += r.mean_lean_late
            t_carried[r.trial] += 1
        if r.level():
            level += 1
            if r.success():
                bar_level += 1
        carry += r.carry_steps
        held += r.held_steps
        t_carry[r.trial] += r.carry_steps
        t_held[r.trial] += r.held_steps
        if r.success():
            t_pass[r.trial] += 1
            c_pass[r.clip] += 1
        if 2 * r.held_steps >= r.carry_steps and r.carry_steps > 0:
            t_half[r.trial] += 1
        if r.fell:
            t_fell[r.trial] += 1
        csv += (
            String(r.clip) + "," + names[r.clip] + "," + String(r.frames) + "," + String(r.steps) + "," + String(r.norm_return) + ","
            + ("1" if r.fell else "0") + "," + String(r.carry_steps) + "," + String(r.held_steps) + "," + String(r.max_err_carry) + ","
            + String(r.mean_err_carry) + "," + ("1" if r.success() else "0") + "," + String(r.trial) + ","
            + ("1" if r.carried() else "0") + "," + String(r.late_steps) + "," + String(r.late_up_steps) + "," + String(r.mean_err_late) + ","
            + String(r.kernels_all) + "," + String(r.kernels_trunk_legs) + "," + String(r.mean_lean_late) + "," + ("1" if r.level() else "0") + "\n"
        )
        print(
            "  ", names[r.clip], "| steps", r.steps, "/", r.frames, "| return", r.norm_return, "| FELL" if r.fell else "| stood",
            "| carry steps", r.carry_steps, "held", r.held_steps, "| box off by", r.mean_err_carry, "m mean,", r.max_err_carry, "m at most",
            "| SUCCESS" if r.success() else "| no", "| carried," if r.carried() else "| not carried,", r.mean_err_late, "m once in hand, leaning",
            r.mean_lean_late, "deg,", "level" if r.level() else "not level",
            "| kernels all", r.kernels_all, "trunk and legs", r.kernels_trunk_legs,
        )
    print(
        "\n", len(res), "runs |" if trials > 1 else "clips |", "success", ok, "(box within 10 cm on >=", SM_OBJ_HELD_SHARE, "of the carry steps, no fall) | strict", strict,
        "| fell", fell, "| clips whose reference never carries", no_carry,
        "| carry steps held", held, "/", carry, "| mean normalised return (the tracker's terms)", ret,
    )
    print(
        " CARRIED", carried, "of", len(res), "(no fall, the box up on more than half of the carry's second half) | error once in hand",
        hand_err / Float64(max(carried, 1)), "m (mean over the carried runs) | six kernels over their maximum: all bodies", k_all,
        ", trunk and legs", k_tl, "(O1.5's tracking criterion: >= 0.95 of the blind tracker's on the same clips)",
    )
    print(
        " LEVEL", level, "of", len(res), "(carried, the box's lean at most", SM_OBJ_LEVEL, "deg over that half) | at the bar and level", bar_level,
        "| lean once in hand", hand_lean / Float64(max(carried, 1)), "deg (mean over the carried runs)",
    )
    if trials > 1:
        var sm = 0.0
        var lo = 1.0
        var hi = 0.0
        var pm = 0.0
        var plo = len(lens)
        var phi = 0
        var hm = 0.0
        var cm = 0.0
        for k in range(trials):
            cm += Float64(t_carried[k]) / Float64(trials)
            var sh = Float64(t_held[k]) / Float64(max(t_carry[k], 1))
            sm += sh / Float64(trials)
            lo = min(lo, sh)
            hi = max(hi, sh)
            pm += Float64(t_pass[k]) / Float64(trials)
            plo = min(plo, t_pass[k])
            phi = max(phi, t_pass[k])
            hm += Float64(t_half[k]) / Float64(trials)
        var always = 0
        var mostly = 0
        for c in range(len(lens)):
            if c_pass[c] == trials:
                always += 1
            if 2 * c_pass[c] >= trials:
                mostly += 1
        print(
            " TRIALS", trials, "| share of carry steps held per trial: mean", sm, "lowest", lo, "highest", hi,
            "| clips passing per trial: mean", pm, "lowest", plo, "highest", phi, "| clips held on at least half the carry, mean", hm,
            "| clips passing in every trial", always, ", in at least half the trials", mostly, "of", len(lens),
            "| clips carried per trial: mean", cm,
        )
    if csv_path != "":
        with open(csv_path, "w") as f:
            f.write(csv)
        print("per-clip results:", csv_path)
