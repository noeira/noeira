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
--store PATH   a SuperMimic library store (`ref` column, one episode per clip),
               a comma-separated list, or a directory of stores
--max-clips N  use the first N clips of the store (0 = all)
--iters N      env steps (default 200)
--log N        log every N steps (default 25)
--seed S
--no-noise     no observation / action / reset noise, no DR (debugging)
--ckpt-every N checkpoint every N steps (default 0: only at the end)
--bench-every N  SM_GPU=2: the Appendix F benchmark on a fixed clip set every N
               steps and at the end (`SMInRunBench`; default 0 = off), logged
               as `bench_norm_return` / `bench_failed`: the unbiased learning
               curve (`avg_reward` averages episodes the reset sampler chose)
--bench-clips K  that set: K clips evenly spaced over the library (default
               the benchmark's lane count, 512 at paper scale: one clip per lane)

THE RUN IS AN OBJECT ON DISK: `runs/<id>/` holds `run.kv`, `metrics.csv` (the
`SMMetrics` bundle every `--log` steps) and the checkpoint `last` (world
model + policy + critic, one v3 file); the monitor gets the same metrics
under the same id when `.env` names it (`NOEIRA_CLOUD_URL`).
"""

from std.math import sin
from std.builtin.sort import sort
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from noeira.core.cont_action import ContAction
from noeira.data.store import TrajectoryStore
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.supermimic import SMConfig, sm_agent, sm_agent_gpu
from noeira.deep_agents.supermimic.eval import SMInRunBench
from noeira.io.artifact_sink import ArtifactSink, sink_for_run
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import G1BatchedSim, G1Robot, G1Sim, G1MjlabRobot, G1MjlabSim, G1MjlabBatchedSim
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.nn.constants import DT


comptime MODE = get_defined_int["SM_GPU", 0]()       # 0 CPU, 1 nets on the GPU, 2 all on the GPU
comptime TARGET: StaticString = "gpu" if MODE != 0 else "cpu"
comptime P = get_defined_int["SM_PAPER", 0]() != 0

# The robot: mjlab's G1 (the reproduction's body) by default; `-D SM_BFM_G1=1`
# selects BFM-Zero's G1 (checkpoints trained before 2026-11-05 are on it).
comptime BFM_G1 = get_defined_int["SM_BFM_G1", 0]() != 0
comptime Robot = G1Robot if BFM_G1 else G1MjlabRobot
comptime RBatched[N: Int] = G1BatchedSim[N] if BFM_G1 else G1MjlabBatchedSim[N]
# SMPaperG1's widths with -D SM_PAPER=1, SMSmokeG1's otherwise (one concrete
# SMConfig, so its robot stays Robot for the library's type)
comptime C = SMConfig[
    TARGET, Robot,
    2048 if P else 128, 64 if P else 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    4096 if P else 8, 2048 if P else 64, 4096 if P else 16, 32,
]
comptime BENCH_LANES = 512 if P else 8         # the in-run benchmark's simulator (memory: see SMInRunBench)
comptime BenchSim = RBatched[BENCH_LANES]


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
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def _store_paths(spec: String) raises -> List[String]:
    """`--store`: one store, a comma-separated list, or a directory (every
    `*.h5` in it, sorted: an import split in shards / chunks)."""
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
    if len(out) == 0:
        raise Error("no store in " + spec)
    return out^


def _from_store(spec: String, max_clips: Int) raises -> SMMotionLibrary[Robot]:
    """The clips of every store, concatenated (each store's episodes are
    contiguous from row 0); `max_clips` > 0 stops after that many."""
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
        rows.resize(o + rows_n * 71, 0)
        for i in range(rows_n * 71):
            rows[o + i] = refc[i]
    var frames = len(rows) // 71
    print("  library:", len(lens), "clips,", frames, "frames (", Float64(frames) / 50.0 / 3600.0, "h ) from", len(paths), "store(s)")
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def _subset(lib: SMMotionLibrary[Robot], k: Int) raises -> SMMotionLibrary[Robot]:
    """K clips evenly spaced over `lib` (clip i · n / K): a fixed set that
    spans every store of a sorted import."""
    var n = lib.num_clips
    var kk = min(max(k, 1), n)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    comptime S = 71
    for i in range(kk):
        var c = i * n // kk
        var f0 = lib.clip_start[c]
        var l = lib.clip_len[c]
        lens.append(l)
        var o = len(rows)
        rows.resize(o + l * S, 0)
        for j in range(l * S):
            rows[o + j] = lib.rows[f0 * S + j]
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def _bench(
    ctx: DeviceContext, var lib: SMMotionLibrary[Robot], every: Int,
) raises -> SMInRunBench[C, BenchSim]:
    print("  in-run benchmark:", lib.num_clips, "clips,", lib.num_frames, "frames, every", every, "steps")
    var sim = BenchSim.create(ctx)
    return SMInRunBench[C, BenchSim](ctx, lib^, sim^, every)


def _train(
    var lib: SMMotionLibrary[Robot], iters: Int, log_every: Int, noise: Bool, seed_v: Int, ckpt_every: Int,
    lp: Pointer[RunLogger, MutAnyOrigin], ckpt: String, artifacts: Optional[ArtifactSink], run_dir: String,
    bench_every: Int, bench_clips: Int,
) raises -> String:
    """Train in the build's mode; returns the run's outcome."""
    var t0 = perf_counter_ns()
    var updates: Int
    var avg: Float64
    comptime if MODE == 2:
        var dctx = DeviceContext()
        var bench_lib = Optional(_subset(lib, (bench_clips if bench_clips > 0 else BENCH_LANES) if bench_every > 0 else 1))
        var agent = sm_agent_gpu[C, RBatched[C.LANES]](dctx, lib^, RBatched[C.LANES].create(dctx), iters, noise, seed_v)
        if bench_every > 0:
            var bench = _bench(dctx, bench_lib.take(), bench_every)
            agent.train[RunLogger, BenchSim](
                iters, log_every, logger=lp, checkpoint_every=ckpt_every, checkpoint_path=ckpt, artifacts=artifacts, run_dir=run_dir,
                bench=Pointer(to=bench).as_unsafe_any_origin(),
            )
            _ = bench^
        else:
            agent.train[RunLogger](iters, log_every, logger=lp, checkpoint_every=ckpt_every, checkpoint_path=ckpt, artifacts=artifacts, run_dir=run_dir)
        updates = agent.updates
        avg = agent.last_avg_reward
    else:
        var ctx: Optional[DeviceContext] = None
        comptime if TARGET == "gpu":
            ctx = DeviceContext()
        var agent = sm_agent[C](lib^, iters, noise, seed_v, ctx=ctx)
        agent.train[RunLogger](iters, log_every, logger=lp, checkpoint_every=ckpt_every, checkpoint_path=ckpt, artifacts=artifacts, run_dir=run_dir)
        updates = agent.updates
        avg = agent.last_avg_reward
    print("done:", iters, "steps,", updates, "updates in", Float64(perf_counter_ns() - t0) * 1e-9, "s")
    return String("avg_reward=") + String(avg) + String(",updates=") + String(updates)


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

    var ckpt_every = Int(_opt(args, String("--ckpt-every"), String("0")))
    var bench_every = Int(_opt(args, String("--bench-every"), String("0")))
    var bench_clips = Int(_opt(args, String("--bench-clips"), String("0")))
    if bench_every > 0 and MODE != 2:
        raise Error("--bench-every needs the device env (-D SM_GPU=2)")

    var lib = _from_store(store, max_clips) if store != "" else _synthetic()
    var mode_s = String("cpu") if MODE == 0 else (String("gpu-nets") if MODE == 1 else String("gpu-all"))
    var run = RunContext(
        project=String("g1"),
        driver=String("examples/g1/supermimic_train.mojo"),
        slug=String("supermimic-") + mode_s + (String("-paper") if P else String("-smoke")),
        env=String("noeira:g1/supermimic_tracking"),
        dataset=String("synthetic") if store == "" else store,
        seed=seed_v,
    )
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "SuperMimic")
    logger.set_config("target", String(TARGET))
    logger.set_config("mode", mode_s)
    logger.set_config("lanes", String(C.LANES))
    logger.set_config("horizon", String(C.HORIZON))
    logger.set_config("wm_width", String(C.WM_H))
    logger.set_config("wm_batch", String(C.WM_BATCH))
    logger.set_config("pi_batch", String(C.PI_BATCH))
    logger.set_config("lr", String(C.DEF_LR))
    logger.set_config("utd", String(C.DEF_UTD))
    logger.set_config("noise", String(noise))
    logger.set_config("library_frames", String(lib.num_frames))
    logger.set_config("library_clips", String(lib.num_clips))
    logger.set_config("bench_every", String(bench_every))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var lp = Pointer(to=logger).as_unsafe_any_origin()
    var ckpt = run.checkpoint_path(String("last"))
    print("SuperMimic, mode", MODE, "(" + String(TARGET) + "):", C.LANES, "lanes, H =", C.HORIZON, ", WM batch", C.WM_BATCH, ", policy batch", C.PI_BATCH, ", noise", noise)
    print("  run", run.dir)
    var outcome = _train(lib^, iters, log_every, noise, seed_v, ckpt_every, lp, ckpt, artifacts, run.dir, bench_every, bench_clips)
    finish_run(run, logger, artifacts, outcome)
    _ = logger  # lifetime extender for lp
