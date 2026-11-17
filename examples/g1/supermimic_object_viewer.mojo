# +--------------------------------------------------------------------------+ #
# | SuperMimic's object interface — watch the G1 pick up and carry a box (CPU, a window)
# +--------------------------------------------------------------------------+ #
"""Watch the object interface (or the blind tracker) on named carry clips: the
benchmark's protocol (each clip from its first frame to its end or a fall, no
noise, no randomisation), one lane of the CPU env with the box, drawn.

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/supermimic_object_viewer.mojo -o build/sm_obj_view
    pixi run ./build/sm_obj_view --ckpt TRACKER.ckpt --store CARRY_50HZ.h5 --object-dir OBJECT_CSV_DIR \\
        --only NAME[,NAME…] [--interface INTERFACE.ckpt] [--gain 0.1] [--fps 50] [--record demo.mp4]
        [--trials 3 --seed 0] [--lead-in 50]

    --interface  `supermimic_object_train.mojo --out`'s checkpoint. WITHOUT it the object encoder is at its
                 zero start: the blind tracker, which goes through the motions and leaves the box on the floor.
    --trials K   every clip K times, each from a start perturbed by the training's reset noise (the robot's
                 root and joints; the box stays at its reference)
    --lead-in N  N frames of standing at each clip's first pose before the clip (`object_task.sm_lead_in`; 50 = 1 s).
                  Half the carry clips lift the box within a second of their first frame; train and score with the same N.
    --record     every drawn frame to PATH (mp4 or gif, needs `ffmpeg`) at the playback rate

The window's text and each run's printed line give what the benchmark counts (`object_eval.sm_obj_benchmark`): the
carry steps (the reference box 5 cm or more above its first height), how many
of them have the simulated box within 10 cm of its reference, and the largest
distance. SPACE pauses; closing the window ends the run.

⚠ CPU PHYSICS, as `supermimic_viewer.mojo`: the float64 single-lane path, where
the checkpoint trained on the float32 batched one. Judge the motion here and
take the score from `supermimic_object_eval.mojo`.
⚠ The robot and the box are drawn; the reference is not (no ghost).
⚠ The carry clips' source states no data terms and a store of SEED motion is
confidential: run on a private machine, never publish the frames.
"""

from std.math import sqrt
from std.sys import argv, get_defined_int

from noeira.core.bytes import string_from_bytes
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import sm_object_track
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_spec import SM_OBJ
from noeira.deep_agents.supermimic.object_task import SM_OREF, sm_lead_in, sm_object_ref_rows
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBoxRobot, SMDims
from noeira.io.fileio import read_file_bytes
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime HO = get_defined_int["SM_OBJ_H", 1024]()
comptime Robot = G1MjlabBoxRobot
comptime D = SMDims[Robot]
comptime CARRY_RISE = 0.05      # `object_eval.SM_OBJ_LIFT`
comptime HELD = 0.10            # `object_eval`'s bar: the box within this of its reference (m)
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
    var gain = Float64(_opt(args, String("--gain"), String(OC.DEF_GAIN)))
    var fps = Int(_opt(args, String("--fps"), String("50")))
    var record = _opt(args, String("--record"), String(""))
    var trials = Int(_opt(args, String("--trials"), String("1")))
    var rng_seed = Int(_opt(args, String("--seed"), String("0")))
    if ckpt == "" or store == "" or object_dir == "" or only == "":
        raise Error("give --ckpt TRACKER.ckpt --store STORE.h5 --object-dir DIR --only NAME[,NAME…] [--interface CKPT] [--record OUT.mp4]")
    var want = List[String]()
    for w in only.split(","):
        want.append(String(String(w).strip()))

    # the named clips, in the order asked: library rows and the box's poses
    var st = TrajectoryStore(store.copy())
    var refc = st.load_column[DType.float32](String("ref"))
    var rows = List[Scalar[DT]]()
    var lens = List[Int]()
    var names = List[String]()
    var poses = List[Float64]()
    for k in range(len(want)):
        var o = 0
        var hit = False
        for e in range(st.n_episodes()):
            var l = Int(st.episodes.ep_len[e])
            if st.manifest.task_text(e) == want[k]:
                hit = True
                for i in range(l * D.S):
                    rows.append(refc[o * D.S + i])
                lens.append(l)
                names.append(want[k].copy())
                var p = _object_poses(object_dir + "/" + want[k] + ".csv", l)
                for i in range(len(p)):
                    poses.append(p[i])
                break
            o += l
        if not hit:
            raise Error("the store holds no clip named " + want[k])
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
    var delay = max(1000 // max(fps, 1), 1)
    var opened = False
    var quit = False

    for c in range(len(lens)):
        if quit:
            break
        var f0 = env.lib.clip_start[c]
        for trial in range(max(trials, 1)):
            if quit:
                break
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
            if not opened:
                if not env.lanes[0].sim.init_renderer():
                    raise Error("no window (the renderer did not open)")
                opened = True
                if record.byte_length() > 0:
                    env.lanes[0].sim.start_recording(record, fps)
                    print("recording to", record)
            print("clip", names[c], "trial", trial, ":", lens[c], "frames")
            var step = 0
            var carry = 0
            var held = 0
            var worst = 0.0
            var failed = False
            while True:
                if env.lanes[0].sim.check_renderer_quit():
                    quit = True
                    break
                if env.lanes[0].sim.renderer_paused():
                    env.lanes[0].sim.render_frame()
                    env.lanes[0].sim.renderer_delay(delay)
                    continue
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
                _ = env.step(a.data, rb)
                step += 1
                if env.lanes[0].ep_len == 0:        # the clip ended (or the robot fell); the lane is back at its first frame
                    failed = env.last_terminated[0]
                    break
                # after the step: the box against the row the step aimed at
                var d2 = 0.0
                for k in range(3):
                    var dd = env.lanes[0].sim.box_qpos(k) - Float64(orefs[fr * SM_OREF + k])
                    d2 += dd * dd
                var carrying = Float64(orefs[fr * SM_OREF + 2]) - z0 >= CARRY_RISE
                if carrying:
                    carry += 1
                    worst = max(worst, sqrt(d2))
                    if sqrt(d2) <= HELD:
                        held += 1
                var hud = List[String]()
                hud.append(String(""))            # the engine's own readout ends on the line above
                hud.append(names[c] + ("  (start " + String(trial) + ")" if trials > 1 else String("")) + "  step " + String(step) + " / " + String(lens[c]))
                hud.append("box " + String(Int(sqrt(d2) * 100.0 + 0.5)) + " cm from its reference" + ("  [the reference is carrying]" if carrying else String("")))
                hud.append("carry steps with the box within 10 cm: " + String(held) + " / " + String(carry))
                env.lanes[0].sim.env.set_hud_extra(hud)
                env.lanes[0].sim.render_frame()
                env.lanes[0].sim.renderer_delay(delay)
            if not quit:
                print(
                    "  ", step, "steps |", "FELL" if failed else "stood", "| carry steps", carry, ", box within 10 cm on", held,
                    "(", (Float64(held) / Float64(max(carry, 1))), ") | farthest", worst, "m",
                )
    if opened:
        if record.byte_length() > 0:
            env.lanes[0].sim.stop_recording()
        env.lanes[0].sim.close_renderer()
