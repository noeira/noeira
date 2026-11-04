# +--------------------------------------------------------------------------+ #
# | SuperMimic on our G1 — watch a checkpoint track clips (CPU, a window)
# +--------------------------------------------------------------------------+ #
"""Watch a SuperMimic checkpoint track library clips: the benchmark's
protocol (Appendix F: from each clip's first frame to its end, no noise, no
domain randomisation, no episode cap), one lane of the CPU env, drawn.

    pixi run mojo build -I . -Xlinker -ld_classic -D SM_PAPER=1 examples/g1/supermimic_viewer.mojo -o /tmp/smview
    pixi run /tmp/smview --ckpt <last.ckpt> --store <store or dir> --clips 0 17 42 --fps 50 [--record demo.mp4]

A paper-width checkpoint trained on the GPU loads into the CPU trainer
bit for bit (`test_supermimic_gpu` [6]). Each clip prints its normalised
return (Σ r / (0.1 · steps), 1 = perfect) and whether it failed, as the
benchmark scores it, so the picture and the number come together. SPACE
pauses; closing the window ends the run. `--record PATH` writes every
drawn frame to PATH (mp4 or gif, needs `ffmpeg`) at the playback rate; V
toggles a recording into the working directory as usual.

⚠ CPU PHYSICS. The env is the float64 single-lane path; the checkpoint
trained against the float32 batched one. The two agree to 1e-7 over a step
(`test_supermimic_env_gpu` [2]), but long clips diverge: judge the motion,
and take the benchmark's numbers from `supermimic_eval.mojo`.
⚠ The robot is drawn; the reference is not (no ghost model yet).
⚠ SEED is confidential: run on a private machine, never publish the frames.
"""

from std.builtin.sort import sort
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import G1Robot, G1Sim, SMDims, TrackingSim, G1BatchedSim, G1MjlabRobot, G1MjlabSim, G1MjlabBatchedSim
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime P = get_defined_int["SM_PAPER", 0]() != 0

# The robot: mjlab's G1 (the reproduction's body) by default; `-D SM_BFM_G1=1`
# selects BFM-Zero's G1 (checkpoints trained before 2026-11-05 are on it).
comptime BFM_G1 = get_defined_int["SM_BFM_G1", 0]() != 0
comptime Robot = G1Robot if BFM_G1 else G1MjlabRobot
comptime RBatched[N: Int] = G1BatchedSim[N] if BFM_G1 else G1MjlabBatchedSim[N]
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


def _ints_after(args: List[String], name: String) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(args)):
        if args[i] == name:
            var j = i + 1
            while j < len(args) and not args[j].startswith("--"):
                out.append(Int(args[j]))
                j += 1
    return out^


def _store_paths(spec: String) raises -> List[String]:
    var out = List[String]()
    if isdir(spec):
        var names = List[String]()
        for n in listdir(spec):
            if String(n).endswith(".h5"):
                names.append(String(n))
        sort(names)
        for k in range(len(names)):
            out.append(spec + "/" + names[k])
    else:
        for part in spec.split(","):
            out.append(String(part))
    return out^


def _library(spec: String, upto: Int) raises -> SMMotionLibrary[Robot]:
    """Clips 0 .. upto of the stores, in order (enough for the chosen ones)."""
    var paths = _store_paths(spec)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    for p in range(len(paths)):
        if len(lens) > upto:
            break
        var st = TrajectoryStore(paths[p])
        var refc = st.load_column[DType.float32](String("ref"))
        var n_ep = min(st.n_episodes(), upto + 1 - len(lens))
        var rows_n = 0
        for e in range(n_ep):
            var l = Int(st.episodes.ep_len[e])
            lens.append(l)
            rows_n += l
        var o = len(rows)
        rows.resize(o + rows_n * 71, 0)
        for i in range(rows_n * 71):
            rows[o + i] = refc[i]
    var q0 = List[Float64](length=G1_N_DOF, fill=0.0)
    for j in range(G1_N_DOF):
        q0[j] = Robot.default_joint_pos(j)
    return SMMotionLibrary[Robot](rows^, lens, q0)


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var fps = Int(_opt(args, String("--fps"), String("50")))
    var clips = _ints_after(args, String("--clips"))
    var record = _opt(args, String("--record"), String(""))
    if ckpt == "" or store == "":
        raise Error("give --ckpt PATH and --store PATH")
    if len(clips) == 0:
        clips.append(0)
    var upto = 0
    for k in range(len(clips)):
        upto = max(upto, clips[k])
    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create())
    var ctr = trainer.load_state(ckpt)
    print("checkpoint at iteration", ctr[0], "(", ctr[1], "updates )")
    var env = SMTrackingEnv[Robot](
        1, _library(store, upto), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0,
    )
    var rb = SMReplayOf[Robot](1, 64)
    var first: List[Int] = [env.lib.clip_start[clips[0]]]
    env.restart(first)
    env.next_frame[0] = env.lib.clip_start[clips[1]] if len(clips) > 1 else env.lib.clip_start[clips[0]]
    if not env.lanes[0].sim.env.init_renderer():
        raise Error("no window (the renderer did not open)")
    if record.byte_length() > 0:
        env.lanes[0].sim.env.start_recording(record, fps)
        print("recording to", record)
    var delay = max(1000 // max(fps, 1), 1)
    var hist = TensorImpl[DT].alloc(D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(D.GOAL)
    var k = 0
    var ret = 0.0
    var steps = 0
    print("clip", clips[0], ":", env.lib.clip_len[clips[0]], "frames")
    while k < len(clips):
        if env.lanes[0].sim.env.check_renderer_quit():
            break
        if env.lanes[0].sim.env.renderer_paused():
            env.lanes[0].sim.env.render_frame()
            env.lanes[0].sim.env.renderer_delay(delay)
            continue
        env.policy_inputs(hist, goals)
        var a = trainer.act[1](hist, goals)
        _ = env.step(a.data, rb)
        ret += env.last_reward[0]
        steps += 1
        env.lanes[0].sim.env.render_frame()
        env.lanes[0].sim.env.renderer_delay(delay)
        if env.lanes[0].ep_len == 0:            # the clip ended (the lane was reset to the next)
            var failed = env.last_terminated[0]
            print("  normalised return", ret / (0.1 * Float64(max(steps, 1))), "|", steps, "steps |", "FAILED" if failed else "completed")
            k += 1
            ret = 0.0
            steps = 0
            if k < len(clips):
                print("clip", clips[k], ":", env.lib.clip_len[clips[k]], "frames")
                env.next_frame[0] = env.lib.clip_start[clips[k + 1]] if k + 1 < len(clips) else env.lib.clip_start[clips[k]]
    if record.byte_length() > 0:
        env.lanes[0].sim.env.stop_recording()
    env.lanes[0].sim.env.close_renderer()
