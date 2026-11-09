# +--------------------------------------------------------------------------+ #
# | SuperMimic on our G1 — the tracker's own style, in simulation
# +--------------------------------------------------------------------------+ #
"""The frozen tracker's style while it tracks the library's clips: the bar the
prompt interface's style compares to, since its decoder is this tracker's
(G4 in `noeira-docs/SUPERMIMIC_BFM_HYBRID_PLAN.md` §0).

Every clip is played once from its first frame, as in the Appendix F benchmark
(`deep_agents/supermimic/eval.mojo`: deterministic policy, no noise, no domain
randomisation, the clip's end decides). From a clip's 50th step on
(`g1_style`'s start-up skip), each simulated step adds the quantities the
prompt eval reads off the state (`deep_agents/supermimic/prompt_eval.mojo`):

    rock     0.5 (wx² + wy²), the base's body-frame roll and pitch rates
    arm_dev  the mean over the 14 arm joints of (q − q_default)²
    arm_qd   the mean over the 14 arm joints of q̇²

to each of the suite's 13 velocity commands whose window holds the step's
velocity (heading frame |vx − c| < .15, |vy − c| < .15, world |wz − c| < .25),
once by the simulated robot's velocity and once by the reference frame's. A
figure is a root mean square; "pooled" weighs the 13 commands equally, as the
prompt eval pools its tasks.

    pixi run -e nvidia mojo build -I . -D SM_PAPER=1 examples/g1/supermimic_tracker_style.mojo -o sm_style
    pixi run ./sm_style --ckpt final.ckpt --store stores/ [--max-clips N]

`-D SM_STYLE_LANES=N` sets the lane count (2048 by default): any count plays
the same clips. ⚠ SEED is confidential: private compute only.
"""

from std.builtin.sort import sort
from std.math import sqrt
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env_gpu import ST_ENDING, SMReplayGpuOf
from noeira.deep_agents.supermimic.eval import sm_benchmark_env
from noeira.deep_agents.supermimic.heads import qapply_inv, yaw_quat
from noeira.deep_agents.supermimic.kinematics import Q4, qapply
from noeira.deep_agents.supermimic.lanes import talloc
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import G1MjlabBatchedSim, G1MjlabRobot, SMDims
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF
from noeira.nn.constants import DT


comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime Robot = G1MjlabRobot
comptime D = SMDims[Robot]
comptime N = get_defined_int["SM_STYLE_LANES", 2048 if P else 64]()
comptime BS = G1MjlabBatchedSim[N]
# the training example's widths: a checkpoint loads only into its own config
comptime C = SMConfig[
    "gpu", Robot,
    2048 if P else 128, 64 if P else 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    4096 if P else 8, 2048 if P else 64, 4096 if P else 16, 32,
]
comptime SKIP = 50                 # g1_style's start-up transient
comptime ARM0 = 15                 # legs 0..11, waist 12..14, arms 15..28
comptime N_CMD = 13
comptime N_SPEED = 7
comptime NAMES = ("vx-0.5", "vx0.0", "vx0.5", "vx1.0", "vx1.5", "vy-0.3", "vy0.3", "vx0.5_vy-0.3", "vx0.5_vy0.3", "wz-1.0", "wz-0.5", "wz0.5", "wz1.0")


def _cmds() -> List[Float64]:
    """The 13 velocity commands, (vx, vy, wz) each, in `NAMES`' order."""
    return [
        -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0,
        0.0, -0.3, 0.0, 0.0, 0.3, 0.0, 0.5, -0.3, 0.0, 0.5, 0.3, 0.0,
        0.0, 0.0, -1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0,
    ]


def _speed_bin(v: Float64) -> Int:
    var edges: List[Float64] = [0.1, 0.3, 0.6, 0.9, 1.2, 1.6]
    for i in range(len(edges)):
        if v < edges[i]:
            return i
    return N_SPEED - 1


struct StyleBins(Movable):
    """Sums of rock, arm_dev, arm_qd and a count per command, and per planar
    speed bin."""
    var cmd: List[Float64]         # [N_CMD · 4]
    var speed: List[Float64]       # [N_SPEED · 4]

    def __init__(out self):
        self.cmd = List[Float64](length=N_CMD * 4, fill=0.0)
        self.speed = List[Float64](length=N_SPEED * 4, fill=0.0)

    def add(mut self, cmds: List[Float64], vx: Float64, vy: Float64, wz: Float64, rock: Float64, dev: Float64, qd: Float64):
        for k in range(N_CMD):
            if abs(vx - cmds[3 * k]) < 0.15 and abs(vy - cmds[3 * k + 1]) < 0.15 and abs(wz - cmds[3 * k + 2]) < 0.25:
                self.cmd[4 * k] += rock
                self.cmd[4 * k + 1] += dev
                self.cmd[4 * k + 2] += qd
                self.cmd[4 * k + 3] += 1.0
        var b = _speed_bin(sqrt(vx * vx + vy * vy))
        self.speed[4 * b] += rock
        self.speed[4 * b + 1] += dev
        self.speed[4 * b + 2] += qd
        self.speed[4 * b + 3] += 1.0

    def report(self, label: String):
        print("--- binned by the", label, "velocity: rock arm_dev arm_qd (RMS) | steps")
        var pooled = List[Float64](length=3, fill=0.0)
        var n_used = 0
        comptime for k in range(N_CMD):
            var n = self.cmd[4 * k + 3]
            if n > 0:
                print(" ", NAMES[k], "|", sqrt(self.cmd[4 * k] / n), sqrt(self.cmd[4 * k + 1] / n), sqrt(self.cmd[4 * k + 2] / n), "|", Int(n))
                for i in range(3):
                    pooled[i] += self.cmd[4 * k + i] / n
                n_used += 1
            else:
                print(" ", NAMES[k], "| no step in the window")
        var u = Float64(max(n_used, 1))
        print("  pooled over", n_used, "commands:", sqrt(pooled[0] / u), sqrt(pooled[1] / u), sqrt(pooled[2] / u))
        var lo: List[Float64] = [0.0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.6]
        for b in range(N_SPEED):
            var n = self.speed[4 * b + 3]
            if n > 0:
                print("  speed from", lo[b], "|", sqrt(self.speed[4 * b] / n), sqrt(self.speed[4 * b + 1] / n), sqrt(self.speed[4 * b + 2] / n), "|", Int(n))


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _q0() -> List[Float64]:
    var q0 = List[Float64](length=G1_N_DOF, fill=0.0)
    for j in range(G1_N_DOF):
        q0[j] = Robot.default_joint_pos(j)
    return q0^


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


def _library(spec: String, max_clips: Int) raises -> SMMotionLibrary[Robot]:
    var paths = _store_paths(spec)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    for p in range(len(paths)):
        if max_clips > 0 and len(lens) >= max_clips:
            break
        var st = TrajectoryStore(paths[p])
        var refc = st.load_column[DType.float32](String("ref"))
        var n_ep = st.n_episodes()
        if max_clips > 0:
            n_ep = min(n_ep, max_clips - len(lens))
        var rows_n = 0
        for e in range(n_ep):
            var l = Int(st.episodes.ep_len[e])
            lens.append(l)
            rows_n += l
        var o = len(rows)
        rows.resize(o + rows_n * D.S, 0)
        for i in range(rows_n * D.S):
            rows[o + i] = refc[i]
    print("  library:", len(lens), "clips,", len(rows) // D.S, "frames from", len(paths), "store(s)")
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    if ckpt == "" or store == "":
        raise Error("give --ckpt PATH and --store PATH")
    var ctx = DeviceContext()
    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), Optional(ctx))
    var ctr = trainer.load_state(ckpt)
    print("tracker style:", ckpt, "at iteration", ctr[0], "|", N, "lanes")
    var lib = _library(store, max_clips)
    var n_clips = lib.num_clips if max_clips <= 0 else min(max_clips, lib.num_clips)
    var env = sm_benchmark_env[Robot, BS](ctx, lib^, BS.create(ctx))
    var rb = SMReplayGpuOf[Robot](ctx, N, 4)          # the env writes rows; nobody samples them
    var cmds = _cmds()
    var by_sim = StyleBins()
    var by_ref = StyleBins()
    var all = List[Float64](length=4, fill=0.0)
    # the benchmark's queue: lane l plays clips l, l + N, l + 2N, ...
    var current = List[Int](length=N, fill=0)
    var nxt = List[Int](length=N, fill=0)
    var active = List[Bool](length=N, fill=False)
    var frames0 = List[Int](length=N, fill=0)
    for l in range(N):
        current[l] = l
        nxt[l] = l + N
        active[l] = l < n_clips
        frames0[l] = env.lib.clip_start[min(l, n_clips - 1)]
    env.restart(ctx, frames0)
    for l in range(N):
        env.next_frame[l] = env.lib.clip_start[min(nxt[l], n_clips - 1)]
    var steps = List[Int](length=N, fill=0)
    var goals = talloc["gpu"](N * D.GOAL, Optional(ctx))
    var rows = env.lib.rows_ptr()
    var remaining = n_clips
    var next_report = 1000
    var done = 0
    while remaining > 0:
        env.policy_inputs(ctx, goals)
        var a = trainer.act[N](env.hist, goals, resident=True)
        _ = env.step(ctx, a, rb)
        env.s.download(ctx)
        for l in range(N):
            if not active[l]:
                continue
            steps[l] += 1
            if (env.h_out_i[2 * l + 1] & ST_ENDING) != 0:
                # reset to its queued clip: the state is that clip's start
                remaining -= 1
                done += 1
                current[l] = nxt[l]
                nxt[l] += N
                active[l] = current[l] < n_clips
                env.next_frame[l] = env.lib.clip_start[min(nxt[l], n_clips - 1)]
                steps[l] = 0
                continue
            if steps[l] <= SKIP:
                continue
            var o = l * D.S
            ref s = env.s.data
            var dev = 0.0
            var qd = 0.0
            for j in range(ARM0, Robot.NDOF):
                var e = Float64(s[o + 7 + j]) - Robot.default_joint_pos(j)
                dev += e * e
                var v = Float64(s[o + D.NQ + 6 + j])
                qd += v * v
            dev /= Float64(Robot.NDOF - ARM0)
            qd /= Float64(Robot.NDOF - ARM0)
            var w0 = Float64(s[o + D.NQ + 3])
            var w1 = Float64(s[o + D.NQ + 4])
            var rock = 0.5 * (w0 * w0 + w1 * w1)
            # the simulated robot: heading-frame velocity, world yaw rate (the prompt eval's `sm_step_velocity`)
            var q = Q4[DT](s[o + 3], s[o + 4], s[o + 5], s[o + 6])
            var vh = qapply_inv(yaw_quat(q), Q4[DT](s[o + D.NQ], s[o + D.NQ + 1], s[o + D.NQ + 2], 0))
            var wz = Float64(qapply(q, Q4[DT](s[o + D.NQ + 3], s[o + D.NQ + 4], s[o + D.NQ + 5], 0))[2])
            by_sim.add(cmds, Float64(vh[0]), Float64(vh[1]), wz, rock, dev, qd)
            # the reference frame the lane tracks: its row holds world velocities
            var f = Int(env.h_out_i[2 * l])
            var r = rows.unsafe_offset(f * D.S)
            var qr = Q4[DT](r[unsafe_offset=3], r[unsafe_offset=4], r[unsafe_offset=5], r[unsafe_offset=6])
            var vr = qapply_inv(yaw_quat(qr), Q4[DT](r[unsafe_offset=D.NQ], r[unsafe_offset=D.NQ + 1], r[unsafe_offset=D.NQ + 2], 0))
            by_ref.add(cmds, Float64(vr[0]), Float64(vr[1]), Float64(r[unsafe_offset=D.NQ + 5]), rock, dev, qd)
            all[0] += rock
            all[1] += dev
            all[2] += qd
            all[3] += 1.0
        if done >= next_report:
            print("  played", done, "/", n_clips, "clips")
            next_report += 1000
    print("\nall steps from step", SKIP, ": rock", sqrt(all[0] / max(all[3], 1.0)), "arm_dev", sqrt(all[1] / max(all[3], 1.0)), "arm_qd", sqrt(all[2] / max(all[3], 1.0)), "|", Int(all[3]), "steps")
    by_sim.report(String("simulated"))
    by_ref.report(String("reference"))
    print("(the prompt interface pools its 13 velocity tasks; G4's bars .46 / .51 / .19 are 1.25 x a teacher on BFM's body)")
