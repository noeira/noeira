# +--------------------------------------------------------------------------+ #
# | SuperMimic on mjlab's G1 — a checkpoint scored clip by clip on the CPU env
# +--------------------------------------------------------------------------+ #
"""Play every clip of ONE store once from its first frame on the single-lane
CPU env and print its score: the Appendix F protocol (deterministic policy, no
noise, no domain randomisation, no episode cap, the clip's end or the
termination decides), without a device. For a small library on a laptop;
`supermimic_eval.mojo` is the benchmark of record (the batched float32 env the
checkpoint trained against).

    pixi run mojo build -I . examples/g1/supermimic_eval_cpu.mojo -o sm_eval_cpu
    pixi run ./sm_eval_cpu --ckpt final.ckpt --store clips_50hz.h5 [--out scores.tsv] [--max-clips N]

Per clip: `norm_return` = Σ r / (0.1 · steps) (1 = perfect, as the benchmark
scores a clip) and `failed` (the benchmark's termination). The summary is the
mean normalised return and the failure count.

WITH A BOX (`-D SM_BOX=1`, build to another name): the same robot in
`UnitreeG1MjlabBox`, the simulator that also holds a free box, and
`--object-dir DIR` with one `<clip name>.csv` per clip: the box's reference
pose at 50 Hz (`x,y,z,qw,qx,qy,qz`, a header row; frame f of the clip is row
f). The box starts at the track's first pose; the tracker is the same and does
not see it. Per clip, over the frames where the REFERENCE box is more than
5 cm above its first height ("carried"):

    carry_frames  how many there are
    carried       on how many of them the simulated box is within 10 cm of
                  the reference (frames after a termination count as missed)
    box_err       the mean distance on those the episode reached
    lift_ref / lift_sim   the largest rise of the box, reference / simulated

and the summary counts the clips with `carried / carry_frames ≥ 0.9`.
`--box-friction MU` sets the box geom's sliding friction (the asset's is 0.9;
the box has priority, so this is the friction of every contact it has).
`--squeeze F` is a FEASIBILITY probe, not a policy: while the reference box is
lifted (or about to be, `SQUEEZE_LEAD` frames ahead), each hand near the
simulated box is pushed toward its centre with F newtons, as joint-target
offsets on the arms (`object_task.sm_squeeze_lane`). It reads the simulated
box, as the object interface does. It says whether ANY action of this kind
holds the box in this simulator.
`--ref-squeeze D` is the same question asked through the tracker's LATENT: the
library's reference is edited before the run (`object_task.sm_squeeze_reference`:
each hand moved D metres toward the reference box's centre while that box is
lifted), and the blind tracker tracks the edited reference. The motion encoder reads the edited goal, so a
lift here is a lift the latent can command.
`--box-park` is the control: the box is left at rest 50 m away, so the robot's
scores must be the robot-only build's (the box family adds a body, not a
difference to the robot).

A LOAD AT THE HANDS (`-D SM_BOX=1 -D SM_XFRC=1`, the tracker alone: use it
with `--box-park`). `--hand-down N` puts N newtons straight down on EACH hand,
`--hand-back N` N newtons on each, horizontal, against the robot's heading.
The force acts at the centre of the hand's capsule (0.11 m along the wrist-yaw
link, 1 cm to the side: the model's `*_hand_collision`), is written again at
every control step (`TrackingSim.set_body_force_at`) and held through the
step's substeps. `--load-when lifted` (the default): while the reference box
is lifted, which needs `--object-dir`; `--load-when always`: the whole clip,
and with `--box-park` no `--object-dir` is needed. It ramps over `LOAD_RAMP`
steps when it comes on and when it goes off. `--hand-readings` adds the
columns below without a force (the 0 N cell); a force adds them too. Over the
"loaded steps", the steps of that schedule whatever the force:

    load_steps      how many the episode reached
    hand_drop       the reference height of the capsule's centre minus the
                    simulated one, mean of the two hands and of those steps (m)
    root_dist_end / root_dist_max   the root's horizontal distance to its
                    reference at the last step reached, and its largest (m)
    tau_wrist_med / tau_wrist_sat   the largest share of its torque limit
                    given to a wrist pitch or yaw joint: its median over the
                    loaded steps, and the share of them where it is ≥ 0.95

The two heights come from one kinematics-only simulator given the simulated
and the reference pose in turn. `--only a,b,c` scores those clips alone.

⚠ CPU PHYSICS: the float64 single-lane path. It agrees with the batched env
to 1e-7 over a step (`test_supermimic_env_gpu` [2]) but long clips diverge, so
a clip's number can differ from `supermimic_eval.mojo`'s.
⚠ Paper widths by default (`-D SM_PAPER=0` for a smoke-width checkpoint);
mjlab's G1 only.
"""

from std.math import atan2, cos, sin, sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_task import sm_squeeze_lane, sm_squeeze_reference
from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.robot import G1MjlabBoxRobot, G1MjlabRobot, SMDims, sm_body_ids
from noeira.envs.robots.unitree_g1_mjlab_pd import g1m_action_scale, g1m_effort, g1m_kp
from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.nn.constants import DT
from noeira.physics3d.gpu.constants import GEOM_IDX_FRICTION
from noeira.nn.core.tensor import TensorImpl


comptime P = get_defined_int["SM_PAPER", 1]() != 0
comptime BOX = get_defined_int["SM_BOX", 0]() != 0
comptime Robot = G1MjlabBoxRobot if BOX else G1MjlabRobot
comptime CARRY_RISE = 0.05      # the reference box counts as carried above its first height by this much (m)
comptime CARRY_TOL = 0.10       # the simulated box counts as with it within this distance (m)
comptime SQUEEZE_LEAD = 10      # frames before the reference lifts at which the squeeze starts
comptime SQUEEZE_ARM0 = 15      # the arms' joints: 15..28 (7 left, 7 right)
comptime LOAD_RAMP = 25         # control steps over which a hand load comes on and goes off (0.5 s)
comptime HAND_X = 0.11          # the hand capsule's centre in the wrist-yaw link's frame: fromto (0.07, 0, 0) .. (0.15, ∓0.02, 0)
comptime HAND_Y = 0.01          # left: −HAND_Y, right: +HAND_Y
# the training example's widths (a checkpoint loads only into its own), one lane
comptime C = SMConfig[
    "cpu", Robot,
    2048 if P else 128, 64 if P else 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    1, 64, 16, 32,
]
comptime D = SMDims[Robot]


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _object_track(path: String) raises -> List[Float64]:
    """The box's reference rows [x y z qw qx qy qz] of one clip, flat."""
    var out = List[Float64]()
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
            out.append(Float64(String(String(f).strip())))
            n += 1
        if n != 7:
            raise Error(path + ": a row of " + String(n) + " values, expected 7")
    if len(out) < 14:
        raise Error(path + ": fewer than two rows")
    return out^


def _median(v: List[Float64]) -> Float64:
    """The median of `v` (0 for none), by an insertion sort of a copy."""
    var a = v.copy()
    for i in range(1, len(a)):
        var x = a[i]
        var j = i - 1
        while j >= 0 and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x
    if len(a) == 0:
        return 0.0
    if len(a) % 2 == 1:
        return a[len(a) // 2]
    return 0.5 * (a[len(a) // 2 - 1] + a[len(a) // 2])


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var out = _opt(args, String("--out"), String(""))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var object_dir = _opt(args, String("--object-dir"), String(""))
    var box_mu = Float64(_opt(args, String("--box-friction"), String("-1")))
    var squeeze = Float64(_opt(args, String("--squeeze"), String("0")))
    var ref_squeeze = Float64(_opt(args, String("--ref-squeeze"), String("0")))
    var hand_down = Float64(_opt(args, String("--hand-down"), String("0")))
    var hand_back = Float64(_opt(args, String("--hand-back"), String("0")))
    var load_when = _opt(args, String("--load-when"), String("lifted"))
    var only = _opt(args, String("--only"), String(""))
    var park = False
    var hands_on = hand_down != 0.0 or hand_back != 0.0
    for i in range(len(args)):
        if args[i] == "--box-park":
            park = True
        if args[i] == "--hand-readings":
            hands_on = True
    if load_when != "lifted" and load_when != "always":
        raise Error("--load-when lifted | always, got " + load_when)
    var wanted = List[String]()
    if only != "":
        for w in only.split(","):
            wanted.append(String(String(w).strip()))
    if ckpt == "" or store == "":
        raise Error("give --ckpt PATH --store STORE.h5 [--out scores.tsv] [--max-clips N]")
    comptime if BOX:
        if object_dir == "" and not (park and load_when == "always"):
            raise Error("-D SM_BOX=1 needs --object-dir DIR (one <clip>.csv per clip), except with --box-park --load-when always")
    else:
        if hands_on:
            raise Error("--hand-down, --hand-back and --hand-readings need the box build (-D SM_BOX=1 -D SM_XFRC=1)")

    var st = TrajectoryStore(store.copy())
    var names = List[String]()
    var lens = List[Int]()
    for e in range(st.n_episodes()):
        names.append(st.manifest.task_text(e))
        lens.append(Int(st.episodes.ep_len[e]))
    var refc = st.load_column[DType.float32](String("ref"))
    var rows = List[Scalar[DT]](length=len(refc), fill=0)
    for i in range(len(refc)):
        rows[i] = refc[i]
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    # the reference poses (the first NQ of each library row), kept for the hands' readings
    var ref_q = List[Float64]()
    if hands_on:
        for f in range(len(refc) // D.S):
            for k in range(Robot.NQ):
                ref_q.append(Float64(refc[f * D.S + k]))

    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create())
    var ctr = trainer.load_state(ckpt)
    print("checkpoint at iteration", ctr[0], "(", ctr[1], "updates )")
    comptime if BOX:
        if ref_squeeze > 0.0:
            # the clips scored: their box positions, one row a library frame (a short track's last row repeated)
            var n_sq = len(lens) if max_clips <= 0 else min(max_clips, len(lens))
            var lens_sq = List[Int]()
            var pos_sq = List[Scalar[DT]]()
            for c in range(n_sq):
                var tr_c = _object_track(object_dir + "/" + names[c] + ".csv")
                var rn = len(tr_c) // 7
                lens_sq.append(lens[c])
                for f in range(lens[c]):
                    for k in range(3):
                        pos_sq.append(Scalar[DT](tr_c[min(f, rn - 1) * 7 + k]))
            rows = sm_squeeze_reference[Robot](trainer.tables, rows, lens_sq, pos_sq, 3, ref_squeeze, SQUEEZE_ARM0)
            print("reference edited: hands", ref_squeeze, "m toward the box while it is lifted, on the", n_sq, "clips scored")
    var env = SMTrackingEnv[Robot](
        1, SMMotionLibrary[Robot](rows^, lens, q0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0,
        max_episode_steps=0,
    )
    var n = env.lib.num_clips
    if max_clips > 0 and max_clips < n:
        n = max_clips
    print("library:", env.lib.num_clips, "clips,", env.lib.num_frames, "frames; scoring", n)
    var rb = SMReplayOf[Robot](1, 64)
    comptime if BOX:
        if box_mu > 0.0:
            var bg = env.lanes[0].sim.num_geoms() - 1          # the box is the model's last geom
            env.lanes[0].sim.set_geom_field(bg, GEOM_IDX_FRICTION, box_mu)
            print("box friction set to", env.lanes[0].sim.geom_field(bg, GEOM_IDX_FRICTION))
    var sq_scratch = List[Scalar[DT]](length=D.S + 2 * D.FK_BLOCK + 6, fill=0)
    var sq_ks = List[Scalar[DT]](length=Robot.NDOF, fill=1)
    for j in range(Robot.NDOF):
        sq_ks[j] = Scalar[DT](g1m_kp(j) * g1m_action_scale(j))      # N·m per action unit
    var box_now = List[Scalar[DT]](length=3, fill=0)
    var hist = TensorImpl[DT].alloc(D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(D.GOAL)

    var tsv = String("clip\tname\tframes\tsteps\tnorm_return\tfailed")
    comptime if BOX:
        tsv += "\tcarry_frames\tcarried\tbox_err\tlift_ref\tlift_sim"
        if hands_on:
            tsv += "\tload_steps\thand_drop\troot_dist_end\troot_dist_max\ttau_wrist_med\ttau_wrist_sat"
    tsv += "\n"
    var hand_ids = List[Int]()
    var kin = Robot.SIM.create()              # a kinematics-only simulator, for the hands' readings
    comptime if BOX:
        if hands_on:
            hand_ids = sm_body_ids[Robot](Robot.hand_bodies())
            print("hands: bodies", hand_ids[0], hand_ids[1], "|", hand_down, "N down and", hand_back, "N backward on each, load", load_when, "| ramp", LOAD_RAMP, "steps")
    var n_scored = 0
    var sum_ret = 0.0
    var n_failed = 0
    var total_steps = 0
    var n_carried = 0
    var sum_share = 0.0
    var t0 = perf_counter_ns()
    for c in range(n):
        if len(wanted) > 0:
            var hit = False
            for w in range(len(wanted)):
                if wanted[w] == names[c]:
                    hit = True
            if not hit:
                continue
        n_scored += 1
        var first: List[Int] = [env.lib.clip_start[c]]
        env.restart(first)
        env.terminations = True
        var track = List[Float64]()
        var rows_n = 0
        var carry_frames = 0
        var carried = 0
        var err_sum = 0.0
        var err_n = 0
        var lift_ref = 0.0
        var lift_sim = 0.0
        var ramp = 0.0
        var load_steps = 0
        var drop_sum = 0.0
        var root_end = 0.0
        var root_max = 0.0
        var tw = List[Float64]()
        comptime if BOX:
            if object_dir == "":
                track = [50.0, 50.0, 0.1805, 1.0, 0.0, 0.0, 0.0, 50.0, 50.0, 0.1805, 1.0, 0.0, 0.0, 0.0]      # parked, never lifted
            else:
                track = _object_track(object_dir + "/" + names[c] + ".csv")
            rows_n = len(track) // 7
            if hands_on:
                for h in range(2):
                    env.lanes[0].sim.set_body_force_at(hand_ids[h], HAND_X, (-HAND_Y if h == 0 else HAND_Y), 0.0, 0.0, 0.0, 0.0)
            var bq = List[Float64]()
            for k in range(7):
                bq.append(track[k])
            var bv: List[Float64] = [
                (track[7] - track[0]) * 50.0, (track[8] - track[1]) * 50.0, (track[9] - track[2]) * 50.0, 0.0, 0.0, 0.0,
            ]
            if park:
                bq = [50.0, 50.0, 0.1805, 1.0, 0.0, 0.0, 0.0]
                bv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            env.lanes[0].sim.set_box(bq, bv)
            for f in range(1, rows_n):
                var rz = track[f * 7 + 2] - track[2]
                lift_ref = max(lift_ref, rz)
                if rz > CARRY_RISE:
                    carry_frames += 1
        var ret = 0.0
        var step = 0
        var failed = False
        while True:
            env.next_frame[0] = env.lib.clip_start[c]
            env.policy_inputs(hist, goals)
            var a = trainer.act[1](hist, goals)
            comptime if BOX:
                var fq = min(step, rows_n - 1)
                var fl = min(step + SQUEEZE_LEAD, rows_n - 1)
                if squeeze > 0.0 and (track[fq * 7 + 2] - track[2] > 0.01 or track[fl * 7 + 2] - track[2] > 0.01):
                    for k in range(3):
                        box_now[k] = Scalar[DT](env.lanes[0].sim.box_qpos(k))
                    sm_squeeze_lane[Robot](
                        SQUEEZE_ARM0, Scalar[DT](squeeze), trainer.tables.tree_ptr(), trainer.tables.head_ptr(),
                        sq_ks.unsafe_ptr().as_unsafe_any_origin(), env.lanes[0].s.unsafe_ptr().as_unsafe_any_origin(),
                        box_now.unsafe_ptr().as_unsafe_any_origin(), sq_scratch.unsafe_ptr().as_unsafe_any_origin(),
                        a.data.unsafe_ptr().as_unsafe_any_origin(),
                    )
            var loaded = False
            comptime if BOX:
                if hands_on:
                    var fr = min(step, rows_n - 1)
                    loaded = load_when == "always" or track[fr * 7 + 2] - track[2] > CARRY_RISE
                    ramp = min(1.0, ramp + 1.0 / Float64(LOAD_RAMP)) if loaded else max(0.0, ramp - 1.0 / Float64(LOAD_RAMP))
                    # the heading: the root's yaw (qpos 3..6 = w, x, y, z)
                    var qw = env.lanes[0].sim.qpos(3)
                    var qx = env.lanes[0].sim.qpos(4)
                    var qy = env.lanes[0].sim.qpos(5)
                    var qz = env.lanes[0].sim.qpos(6)
                    var yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    for h in range(2):
                        env.lanes[0].sim.set_body_force_at(
                            hand_ids[h], HAND_X, (-HAND_Y if h == 0 else HAND_Y), 0.0,
                            -hand_back * cos(yaw) * ramp, -hand_back * sin(yaw) * ramp, -hand_down * ramp,
                        )
            _ = env.step(a.data, rb)
            ret += env.last_reward[0]
            step += 1
            if env.lanes[0].ep_len == 0:        # the clip ended; the lane is back at its first frame
                failed = env.last_terminated[0]
                break
            comptime if BOX:
                if hands_on:
                    # the state after `step` steps against the reference's frame `step`
                    var fh = env.lib.clip_start[c] + min(step, lens[c] - 1)
                    var qs = List[Float64]()
                    var qr = List[Float64]()
                    for k in range(Robot.NQ):
                        qs.append(env.lanes[0].sim.qpos(k))
                        qr.append(ref_q[fh * Robot.NQ + k])
                    var rd = sqrt((qs[0] - qr[0]) * (qs[0] - qr[0]) + (qs[1] - qr[1]) * (qs[1] - qr[1]))
                    root_end = rd
                    root_max = max(root_max, rd)
                    if loaded:
                        var zero_v = List[Float64](length=Robot.NV, fill=0.0)
                        var drop = 0.0
                        kin.set_state(qs, zero_v)
                        for h in range(2):
                            drop -= 0.5 * kin.body_point(hand_ids[h], HAND_X, (-HAND_Y if h == 0 else HAND_Y), 0.0)[2]
                        kin.set_state(qr, zero_v)
                        for h in range(2):
                            drop += 0.5 * kin.body_point(hand_ids[h], HAND_X, (-HAND_Y if h == 0 else HAND_Y), 0.0)[2]
                        drop_sum += drop
                        load_steps += 1
                        var u = 0.0
                        for j in [20, 21, 27, 28]:      # wrist pitch and yaw, left and right
                            u = max(u, abs(env.lanes[0].sim.joint_torque(j)) / g1m_effort(j))
                        tw.append(u)
            comptime if BOX:
                var f = min(step, rows_n - 1)
                var dx = env.lanes[0].sim.box_qpos(0) - track[f * 7]
                var dy = env.lanes[0].sim.box_qpos(1) - track[f * 7 + 1]
                var dz = env.lanes[0].sim.box_qpos(2) - track[f * 7 + 2]
                var e = sqrt(dx * dx + dy * dy + dz * dz)
                lift_sim = max(lift_sim, env.lanes[0].sim.box_qpos(2) - track[2])
                if track[f * 7 + 2] - track[2] > CARRY_RISE:
                    err_sum += e
                    err_n += 1
                    if e < CARRY_TOL:
                        carried += 1
        var nr = ret / (0.1 * Float64(step))
        sum_ret += nr
        total_steps += step
        if failed:
            n_failed += 1
        tsv += String(c) + "\t" + names[c] + "\t" + String(lens[c]) + "\t" + String(step) + "\t" + String(nr) + "\t" + (String("1") if failed else String("0"))
        var extra = String("")
        comptime if BOX:
            var share = Float64(carried) / Float64(max(carry_frames, 1))
            var be = err_sum / Float64(max(err_n, 1))
            sum_share += share
            if share >= 0.9:
                n_carried += 1
            tsv += "\t" + String(carry_frames) + "\t" + String(carried) + "\t" + String(be) + "\t" + String(lift_ref) + "\t" + String(lift_sim)
            extra = "  box: carried " + String(carried) + "/" + String(carry_frames) + ", err " + String(be) + " m, lift " + String(lift_sim) + " of " + String(lift_ref)
            if hands_on:
                var sat = 0
                for i in range(len(tw)):
                    if tw[i] >= 0.95:
                        sat += 1
                var hd = drop_sum / Float64(max(load_steps, 1))
                var ts = Float64(sat) / Float64(max(len(tw), 1))
                tsv += "\t" + String(load_steps) + "\t" + String(hd) + "\t" + String(root_end) + "\t" + String(root_max) + "\t" + String(_median(tw)) + "\t" + String(ts)
                extra = "  hands: " + String(load_steps) + " loaded steps, drop " + String(hd) + " m, root " + String(root_end) + " m (max " + String(root_max) + "), wrists " + String(_median(tw)) + " (" + String(ts) + " at the limit)"
        tsv += "\n"
        print("  [" + String(n_scored) + "] " + names[c] + "  " + String(nr) + (String("  FAILED at step ") + String(step) if failed else String("")) + extra)
    var secs = Float64(perf_counter_ns() - t0) * 1e-9
    n = n_scored
    print("mean normalised return", sum_ret / Float64(max(n, 1)), "| failed", n_failed, "of", n, "|", total_steps, "steps in", secs, "s")
    comptime if BOX:
        if squeeze > 0.0:
            print("scripted squeeze of", squeeze, "N while the reference box is lifted: a probe, not a policy")
        print("box carried (within", CARRY_TOL, "m on ≥ 90 % of the carry frames):", n_carried, "of", n, "| mean share", sum_share / Float64(n))
    if out != "":
        write_text_atomic(out, tsv)
        print("wrote", out)
