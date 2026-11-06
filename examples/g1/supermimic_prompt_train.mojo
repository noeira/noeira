# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface on a frozen tracker
# +--------------------------------------------------------------------------+ #
"""Train an encoder into a tracker's decoder through the world model (the paper's
Appendix B and G; `noeira-docs/SUPERMIMIC_BFM_HYBRID_PLAN.md` §0). The tracker's
motion encoder and decoder do not move: tracking through them cannot change.

Two interfaces, the same code, different data (`prompt_bank.mojo`):

    default                     a reward-derived z, 32-D, from the tracker's latent
    -D SM_PROMPT_VEL=1          the paper's velocity command: the positive control

Three scales:

    # the loop's logic, on the Mac (CPU env, small networks, a synthetic library)
    pixi run mojo run -I . examples/g1/supermimic_prompt_train.mojo --iters 40

    # paper widths on the CPU env, a real tracker on a few lanes (a wiring run)
    pixi run mojo build -I . -Xlinker -ld_classic -D SM_PAPER=1 -D SM_LANES=8 -D SM_WM_BATCH=32 -D SM_PI_BATCH=16 examples/g1/supermimic_prompt_train.mojo -o smp
    pixi run ./smp --tracker final.ckpt --store seed_d_0024.h5 --max-clips 300 --bank 256 --pool 20000 --iters 120

    # everything on the device: NVIDIA only (the batched G1)
    pixi run -e nvidia mojo build -I . -D SM_GPU=2 -D SM_PAPER=1 examples/g1/supermimic_prompt_train.mojo -o smp_dev
    ./smp_dev --tracker final.ckpt --store stores/ --iters 100000 --out prompt.ckpt

Options
-------
--tracker PATH  the tracker's checkpoint (paper widths; required with -D SM_PAPER=1)
--store PATH    a library store, a comma-separated list or a directory; none: synthetic
--max-clips N   the first N clips (0 = all)
--bank K        command-bank rows (default 4096 at paper scale, 64 at smoke scale)
--pool P        frames of the pool z is computed over (default 131072)
--z-beta B      z retrieves with r^B, the reward stays r (default 1); the eval must match
--z-ess N       each reward sharpens as far as an effective sample size of N (default off)
--no-hold-out   train on the held-out rewards too: G3's comparator, never a result
--split S       lanes below S keep tracking (default half)
--align-warm U  updates at 100× the alignment's weight (default 2000 at paper scale)
--resume PATH   go on from this run's checkpoint, up to --iters in all
--iters N --log N --seed S --ckpt-every N --out PATH --no-noise

⚠ A SEED store is confidential: private compute only, and the pool and bank are
SEED-derived.
"""

from std.builtin.sort import sort
from std.math import sin
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig
from noeira.deep_agents.supermimic.agent import sm_schedule
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.prompt_agent import SMPromptAgent
from noeira.deep_agents.supermimic.prompt_agent_gpu import SMPromptAgentGpu
from noeira.deep_agents.supermimic.prompt_bank import sm_prompt_data
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfig
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBatchedSim, G1MjlabRobot, SMDims
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.envs.robots.unitree_g1_mjlab import UnitreeG1Mjlab
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF
from noeira.nn.constants import DT


comptime MODE = get_defined_int["SM_GPU", 0]()       # 0 CPU, 2 all on the device
comptime TARGET: StaticString = "gpu" if MODE != 0 else "cpu"
comptime PAPER = get_defined_int["SM_PAPER", 0]() != 0
comptime VEL = get_defined_int["SM_PROMPT_VEL", 0]() != 0
comptime Robot = G1MjlabRobot
comptime C = SMConfig[
    TARGET, Robot,
    2048 if PAPER else 128, 64 if PAPER else 64, 2048 if PAPER else 128, 1024 if PAPER else 64, 512 if PAPER else 64, 256 if PAPER else 32,
    2048 if PAPER else 128, 1024 if PAPER else 64, 512 if PAPER else 64, 256 if PAPER else 32, 256 if PAPER else 64, 128 if PAPER else 32,
    get_defined_int["SM_LANES", 4096 if PAPER else 8](), get_defined_int["SM_WM_BATCH", 2048 if PAPER else 16](), get_defined_int["SM_PI_BATCH", 2048 if PAPER else 16](), 32,
]
comptime CMD = 3 if VEL else SMDims[Robot].LATENT
comptime P = SMPromptConfig[C, CMD, 256 if PAPER else 32, 256 if PAPER else 32, 256 if PAPER else 32, 128 if PAPER else 32]


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _flag(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if args[i] == name:
            return True
    return False


def _q0() -> List[Float64]:
    var q0 = List[Float64](length=G1_N_DOF, fill=0.0)
    for j in range(G1_N_DOF):
        q0[j] = Robot.default_joint_pos(j)
    return q0^


def _synthetic() raises -> SMMotionLibrary[Robot]:
    """The G1 settled 0.2 s (mjlab's soft servos drop it by ~1.2 s under zero
    action), then two clips swaying arms and waist while it drifts forward:
    smoke only, but a library whose frames have some velocity."""
    var env = UnitreeG1Mjlab[DType.float64]()
    _ = env.reset()
    var a = ContAction[G1_N_DOF]()
    for _ in range(10):
        _ = env.step(a)
    var lens: List[Int] = [300, 200]
    var rows = List[Scalar[DT]](length=500 * 71, fill=0)
    for f in range(500):
        var t = Float64(f) * 0.02
        for k in range(36):
            rows[f * 71 + k] = Scalar[DT](env.d.qpos.data[k])
        rows[f * 71] += Scalar[DT](0.3 * t)
        rows[f * 71 + 36] = Scalar[DT](0.3 + 0.2 * sin(t))
        for j in range(15, 29):
            var w = 1.5 + 0.1 * Float64(j)
            rows[f * 71 + 7 + j] += Scalar[DT](0.3 * sin(w * t))
            rows[f * 71 + 42 + j] = Scalar[DT](0.3 * w * sin(w * t + 1.5707963267948966))
    return SMMotionLibrary[Robot](rows^, lens, _q0())


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


def _from_store(spec: String, max_clips: Int) raises -> SMMotionLibrary[Robot]:
    comptime S = SMDims[Robot].S
    var paths = _store_paths(spec)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    for p in range(len(paths)):
        var st = TrajectoryStore(paths[p])
        var refc = st.load_column[DType.float32](String("ref"))
        var n_ep = st.n_episodes()
        var n_rows = 0
        for e in range(n_ep):
            if max_clips > 0 and len(lens) >= max_clips:
                break
            var l = Int(st.episodes.ep_len[e])
            lens.append(l)
            n_rows += l
        var o = len(rows)
        rows.resize(o + n_rows * S, 0)
        for i in range(n_rows * S):
            rows[o + i] = refc[i]
        if max_clips > 0 and len(lens) >= max_clips:
            break
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var tracker = _opt(args, String("--tracker"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var iters = Int(_opt(args, String("--iters"), String("40")))
    var log_every = Int(_opt(args, String("--log"), String("10")))
    var seed_v = Int(_opt(args, String("--seed"), String("0")))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var bank = Int(_opt(args, String("--bank"), String("4096" if PAPER else "64")))
    var pool = Int(_opt(args, String("--pool"), String("131072")))
    var z_beta = Float64(_opt(args, String("--z-beta"), String("1")))
    var z_ess = Float64(_opt(args, String("--z-ess"), String("0")))
    var split = Int(_opt(args, String("--split"), String(C.LANES // 2)))
    var ckpt_every = Int(_opt(args, String("--ckpt-every"), String("0")))
    var align_warm = Int(_opt(args, String("--align-warm"), String("2000" if PAPER else "0")))
    var out_path = _opt(args, String("--out"), String(""))
    var resume = _opt(args, String("--resume"), String(""))
    var noise = not _flag(args, String("--no-noise"))
    comptime if PAPER:
        if tracker == "":
            raise Error("paper widths need --tracker PATH")

    var lib = _from_store(store, max_clips) if store != "" else _synthetic()
    print("library:", lib.num_clips, "clips,", lib.num_frames, "frames")
    var ctx: Optional[DeviceContext] = None
    comptime if TARGET == "gpu":
        ctx = DeviceContext()
    var base = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), ctx)
    if tracker != "":
        var c0 = base.load_state(tracker)
        print("tracker at iteration", c0[0])
    var t0 = perf_counter_ns()
    var data = sm_prompt_data[C, CMD](base, lib, bank, pool, z_beta, z_ess, not _flag(args, String("--no-hold-out")))
    print("tables:", data.frames, "frames,", data.bank_n, "bank rows in", Float64(perf_counter_ns() - t0) * 1e-9, "s")
    var tr = SMPromptTrainer[P](base^)
    var at_it = 0
    var at_up = 0
    if resume != "":
        var c1 = tr.load_state(resume)
        at_it = c1[0]
        at_up = c1[1]
        print("resuming at iteration", at_it, ", update", at_up)
    var sched = sm_schedule[C](iters, noise, seed_v)
    print("prompt interface:", CMD, "-D command,", C.LANES - split, "of", C.LANES, "lanes, H =", C.HORIZON, ", batch", C.PI_BATCH)
    comptime if MODE == 2:
        comptime Sim = G1MjlabBatchedSim[C.LANES]
        var agent = SMPromptAgentGpu[P, Sim](ctx.value(), lib^, Sim.create(ctx.value()), tr^, data^, sched^, split)
        agent.align_warm = align_warm
        agent.resume(at_it, at_up)
        agent.train(iters, log_every, ckpt_every, out_path)
    else:
        var agent = SMPromptAgent[P](lib^, tr^, data^, sched^, split)
        agent.align_warm = align_warm
        agent.resume(at_it, at_up)
        agent.train(iters, log_every, out_path)
    print("done")
