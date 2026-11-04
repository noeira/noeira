# +--------------------------------------------------------------------------+ #
# | SuperMimic on our G1 — the training loop, smoke scale
# +--------------------------------------------------------------------------+ #
"""Algorithm A.1 end to end: collect, fit the world model, improve the policy.

Smoke scale by default (`SMSmokeG1`'s widths). The env and
replay run on the CPU; the networks and the four update blocks run on the
CPU, or on the GPU when built with `-D SM_GPU=1`. `-D SM_GPU=2` puts the env
and the replay on the device too (`SuperMimicAgentGpu` over the batched G1:
NVIDIA only); `-D SM_PAPER=1` the paper's widths and 4096 lanes:

    pixi run -e apple mojo build -I . -D SM_GPU=1 examples/g1/supermimic_train.mojo -o sm_train_gpu
    pixi run -e nvidia mojo build -I . -D SM_GPU=2 -D SM_PAPER=1 examples/g1/supermimic_train.mojo -o sm_train_dev

    # a SuperMimic library store (examples/g1/seed_import.mojo's output):
    pixi run mojo run -I . examples/g1/supermimic_train.mojo --store seed_g1_50hz.h5 --iters 400

    # no data: a synthetic library (a settled pose, swaying)
    pixi run mojo run -I . examples/g1/supermimic_train.mojo --iters 200

THIS IS A WIRING RUN, NOT THE PAPER'S. Everything is the paper's recipe
except the scale: 8 lanes instead of 4096, networks of 128 / 64 units
instead of 2048, world-model batches of 64 windows and policy batches of 16
starts instead of 2048 / 4096, all on the CPU. It shows the pieces turn
together (losses fall, nothing diverges), not what the method reaches.
⚠ A SEED store is confidential: run it on private compute only.

Options
-------
--store PATH   a SuperMimic library store (`ref` column, one episode per clip)
--max-clips N  use the first N clips of the store (0 = all)
--iters N      env steps (default 200)
--log N        log every N steps (default 25)
--seed S
--no-noise     no observation / action / reset noise, no DR (debugging)
"""

from std.math import sin
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from noeira.core.cont_action import ContAction
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SMSchedule, SuperMimicAgent
from noeira.deep_agents.supermimic.agent_gpu import SuperMimicAgentGpu
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import G1BatchedSim, G1Robot
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.nn.constants import DT


comptime MODE = get_defined_int["SM_GPU", 0]()       # 0 CPU, 1 nets on the GPU, 2 all on the GPU
comptime TARGET: StaticString = "gpu" if MODE != 0 else "cpu"
comptime P = get_defined_int["SM_PAPER", 0]() != 0
# SMPaperG1's widths with -D SM_PAPER=1, SMSmokeG1's otherwise (one concrete
# SMConfig, so its robot stays G1Robot for the library's type)
comptime C = SMConfig[
    TARGET, G1Robot,
    2048 if P else 128, 64 if P else 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    4096 if P else 8, 2048 if P else 64, 4096 if P else 16, 32,
]


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
        q0[j] = g1_default_pos(j)
    return q0^


def _synthetic() raises -> SMMotionLibrary[G1Robot]:
    """Our G1 settled 2 s under zero action, then two clips of it swaying its
    arms and waist (sinusoids on the reference joints, velocities consistent)."""
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var a = ContAction[G1_N_DOF]()
    for _ in range(100):
        _ = env.step(a)
    var lens: List[Int] = [300, 200]
    var rows = List[Scalar[DT]](length=500 * 71, fill=0)
    for f in range(500):
        var t = Float64(f) * 0.02
        for k in range(36):
            rows[f * 71 + k] = Scalar[DT](env.d.qpos.data[k])
        for j in range(15, 29):   # waist and arms
            var w = 1.5 + 0.1 * Float64(j)
            rows[f * 71 + 7 + j] += Scalar[DT](0.3 * sin(w * t))
            rows[f * 71 + 42 + j] = Scalar[DT](0.3 * w * sin(w * t + 1.5707963267948966))
    return SMMotionLibrary[G1Robot](rows^, lens, _q0())


def _from_store(path: String, max_clips: Int) raises -> SMMotionLibrary[G1Robot]:
    var st = TrajectoryStore(path)
    var refc = st.load_column[DType.float32](String("ref"))
    var n_ep = st.n_episodes()
    if max_clips > 0:
        n_ep = min(n_ep, max_clips)
    var lens = List[Int]()
    var rows_n = 0
    for e in range(n_ep):
        var l = Int(st.episodes.ep_len[e])
        lens.append(l)
        rows_n += l
    var rows = List[Scalar[DT]](length=rows_n * 71, fill=0)
    for i in range(rows_n * 71):      # episodes are contiguous from row 0
        rows[i] = refc[i]
    print("  library:", n_ep, "clips,", rows_n, "frames (", Float64(rows_n) / 50.0 / 3600.0, "h ) from", path)
    return SMMotionLibrary[G1Robot](rows^, lens, _q0())


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var store = _opt(args, String("--store"), String(""))
    var iters = Int(_opt(args, String("--iters"), String("200")))
    var log_every = Int(_opt(args, String("--log"), String("25")))
    var seed_v = Int(_opt(args, String("--seed"), String("0")))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var noise = not _flag(args, String("--no-noise"))

    var lib = _from_store(store, max_clips) if store != "" else _synthetic()
    var sched = SMSchedule(iters, 0.4, 40, 2e-4, 1000, noise, seed_v)
    print("SuperMimic, mode", MODE, "(" + String(TARGET) + "):", C.LANES, "lanes, H =", C.HORIZON, ", WM batch", C.WM_BATCH, ", policy batch", C.PI_BATCH, ", noise", noise)
    var t0 = perf_counter_ns()
    comptime if MODE == 2:
        var dctx = DeviceContext()
        var sim = G1BatchedSim[C.LANES].create(dctx)
        var agent = SuperMimicAgentGpu[C, G1BatchedSim[C.LANES]](dctx, lib^, sim^, sched^)
        agent.train(iters, log_every)
        print("done:", iters, "steps,", agent.updates, "updates in", Float64(perf_counter_ns() - t0) * 1e-9, "s")
    else:
        var ctx: Optional[DeviceContext] = None
        comptime if TARGET == "gpu":
            ctx = DeviceContext()
        var agent = SuperMimicAgent[C](lib^, sched^, ctx)
        agent.train(iters, log_every)
        print("done:", iters, "steps,", agent.updates, "updates in", Float64(perf_counter_ns() - t0) * 1e-9, "s")
