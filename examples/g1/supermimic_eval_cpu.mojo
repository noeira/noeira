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
`--box-park` is the control: the box is left at rest 50 m away, so the robot's
scores must be the robot-only build's (the box family adds a body, not a
difference to the robot).

⚠ CPU PHYSICS: the float64 single-lane path. It agrees with the batched env
to 1e-7 over a step (`test_supermimic_env_gpu` [2]) but long clips diverge, so
a clip's number can differ from `supermimic_eval.mojo`'s.
⚠ Paper widths by default (`-D SM_PAPER=0` for a smoke-width checkpoint);
mjlab's G1 only.
"""

from std.math import sqrt
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.robot import G1MjlabBoxRobot, G1MjlabRobot, SMDims
from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.nn.constants import DT
from noeira.physics3d.gpu.constants import GEOM_IDX_FRICTION
from noeira.nn.core.tensor import TensorImpl


comptime P = get_defined_int["SM_PAPER", 1]() != 0
comptime BOX = get_defined_int["SM_BOX", 0]() != 0
comptime Robot = G1MjlabBoxRobot if BOX else G1MjlabRobot
comptime CARRY_RISE = 0.05      # the reference box counts as carried above its first height by this much (m)
comptime CARRY_TOL = 0.10       # the simulated box counts as with it within this distance (m)
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
    var park = False
    for i in range(len(args)):
        if args[i] == "--box-park":
            park = True
    if ckpt == "" or store == "":
        raise Error("give --ckpt PATH --store STORE.h5 [--out scores.tsv] [--max-clips N]")
    comptime if BOX:
        if object_dir == "":
            raise Error("-D SM_BOX=1 needs --object-dir DIR (one <clip>.csv per clip)")

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

    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create())
    var ctr = trainer.load_state(ckpt)
    print("checkpoint at iteration", ctr[0], "(", ctr[1], "updates )")
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
    var hist = TensorImpl[DT].alloc(D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(D.GOAL)

    var tsv = String("clip\tname\tframes\tsteps\tnorm_return\tfailed")
    comptime if BOX:
        tsv += "\tcarry_frames\tcarried\tbox_err\tlift_ref\tlift_sim"
    tsv += "\n"
    var sum_ret = 0.0
    var n_failed = 0
    var total_steps = 0
    var n_carried = 0
    var sum_share = 0.0
    var t0 = perf_counter_ns()
    for c in range(n):
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
        comptime if BOX:
            track = _object_track(object_dir + "/" + names[c] + ".csv")
            rows_n = len(track) // 7
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
            _ = env.step(a.data, rb)
            ret += env.last_reward[0]
            step += 1
            if env.lanes[0].ep_len == 0:        # the clip ended; the lane is back at its first frame
                failed = env.last_terminated[0]
                break
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
        tsv += "\n"
        print("  [" + String(c + 1) + "/" + String(n) + "] " + names[c] + "  " + String(nr) + (String("  FAILED at step ") + String(step) if failed else String("")) + extra)
    var secs = Float64(perf_counter_ns() - t0) * 1e-9
    print("mean normalised return", sum_ret / Float64(n), "| failed", n_failed, "of", n, "|", total_steps, "steps in", secs, "s")
    comptime if BOX:
        print("box carried (within", CARRY_TOL, "m on ≥ 90 % of the carry frames):", n_carried, "of", n, "| mean share", sum_share / Float64(n))
    if out != "":
        write_text_atomic(out, tsv)
        print("wrote", out)
