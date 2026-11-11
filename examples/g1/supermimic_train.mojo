# +--------------------------------------------------------------------------+ #
# | SuperMimic on our G1 — the training loop, smoke scale
# +--------------------------------------------------------------------------+ #
"""Algorithm A.1 end to end: collect, fit the world model, improve the policy.

Smoke scale by default (`SMSmokeG1`'s widths). The env and
replay run on the CPU; the networks and the four update blocks run on the
CPU, or on the GPU when built with `-D SM_GPU=1`. `-D SM_GPU=2` puts the env
and the replay on the device too (`SuperMimicAgentGpu` over the batched G1:
NVIDIA only); `-D SM_PAPER=1` the paper's widths and 4096 lanes; `-D
SM_LIGHT=1` half-width networks and 1024 lanes; `-D SM_BHL=1` Berkeley
Humanoid Lite instead of the G1 (its stores: `examples/bhl/retarget_import.mojo`;
`-D BHL_EFFORT_PCT=` / `-D BHL_SPEED_PCT=` scale its torque and speed limits):

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
--bench-list FILE  that set: the clips whose library indices FILE lists, one per
               line (e.g. a benchmark's failures and hardest clips)
--resume PATH  go on from a checkpoint (`last`, or a run's `final.ckpt`): its
               networks and counters; `--iters` stays the run's TOTAL. The
               replay and Adam start empty, the sampler's estimates fresh
--lr-restart R   with --resume, realmimic's warm restart: the lr restarts at
               R x lr (default 0.3) and decays by a second cosine to 0.03 x lr
               over --lr-restart-decay N iterations (default: to --iters);
               0 = go on with the first schedule
--sampler S    the reset sampler's bin estimate: `ema` (the reproduction's,
               default) or `count` (realmimic's: pessimistic for unvisited
               bins, a running mean, clip ends continued)
--sampler-seed CSV  with `count`: start the estimate from a benchmark of the
               resumed policy (`supermimic_eval`'s benchmark.csv on the same
               library): failed clips' bins at 0, the others at G_MAX x their
               return, one visit each; skips the coverage phase of a fine-tune

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

from noeira.data.store import TrajectoryStore
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.supermimic import SMConfig, sm_agent, sm_agent_gpu
from noeira.deep_agents.supermimic.eval import SMInRunBench
from noeira.io.artifact_sink import ArtifactSink, sink_for_run
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary, SMResetSampler
from noeira.deep_agents.supermimic.robot import (
    BhlBatchedSim, BhlRobot, G1BatchedSim, G1MjlabBatchedSim, G1MjlabRobot, G1Robot, SMDims, TrackingSim,
)
from noeira.nn.constants import DT


comptime MODE = get_defined_int["SM_GPU", 0]()       # 0 CPU, 1 nets on the GPU, 2 all on the GPU
comptime TARGET: StaticString = "gpu" if MODE != 0 else "cpu"
comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime LIGHT = get_defined_int["SM_LIGHT", 0]() != 0 and not P

# The robot: mjlab's G1 (the reproduction's body) by default; `-D SM_BFM_G1=1`
# selects BFM-Zero's G1 (checkpoints trained before 2026-11-05 are on it),
# `-D SM_BHL=1` Berkeley Humanoid Lite.
comptime BFM_G1 = get_defined_int["SM_BFM_G1", 0]() != 0
comptime BHL = get_defined_int["SM_BHL", 0]() != 0
comptime Robot = BhlRobot if BHL else (G1Robot if BFM_G1 else G1MjlabRobot)
comptime RBatched[N: Int] = BhlBatchedSim[N] if BHL else (G1BatchedSim[N] if BFM_G1 else G1MjlabBatchedSim[N])
comptime S = SMDims[Robot].S                   # a library row: qpos, then qvel
comptime NDOF = Robot.NDOF
# The synthetic library sways these joints: the G1's waist and arms, BHL's arms.
comptime SWAY0 = 0 if BHL else 15
comptime SWAY1 = 10 if BHL else 29
# SMPaperG1's widths with -D SM_PAPER=1, a quarter of them with -D SM_LIGHT=1
# (1024 lanes, half-width networks: a small library without the hardest
# clips), SMSmokeG1's otherwise (one concrete SMConfig, so its robot stays
# Robot for the library's type)
comptime C = SMConfig[
    TARGET, Robot,
    2048 if P else (1024 if LIGHT else 128), 64,
    2048 if P else (1024 if LIGHT else 128), 1024 if P else (512 if LIGHT else 64),
    512 if P else (256 if LIGHT else 64), 256 if P else (128 if LIGHT else 32),
    2048 if P else (1024 if LIGHT else 128), 1024 if P else (512 if LIGHT else 64),
    512 if P else (256 if LIGHT else 64), 256 if P else (128 if LIGHT else 32),
    256 if P else (128 if LIGHT else 64), 128 if P else (64 if LIGHT else 32),
    4096 if P else (1024 if LIGHT else 8), 2048 if P else (1024 if LIGHT else 64),
    4096 if P else (2048 if LIGHT else 16), 32,
]
comptime BENCH_LANES = 512 if P else (256 if LIGHT else 8)   # the in-run benchmark's simulator (memory: see SMInRunBench)
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
    var q0 = List[Float64](length=NDOF, fill=0.0)
    for j in range(NDOF):
        q0[j] = Robot.default_joint_pos(j)
    return q0^


def _synthetic() raises -> SMMotionLibrary[Robot]:
    """The robot settled 2 s under zero action, then two clips of it swaying
    its arms (and the G1's waist): sinusoids on the reference joints,
    velocities consistent."""
    var sim = Robot.SIM.create()
    var zero = List[Float64](length=NDOF, fill=0.0)
    for _ in range(100):
        sim.control_step(zero, zero)
    var lens: List[Int] = [300, 200]
    var rows = List[Scalar[DT]](length=500 * S, fill=0)
    comptime QD = Robot.NQ + 6      # joint velocities in a row
    for f in range(500):
        var t = Float64(f) * 0.02
        for k in range(Robot.NQ):
            rows[f * S + k] = Scalar[DT](sim.qpos(k))
        for j in range(SWAY0, SWAY1):
            var w = 1.5 + 0.1 * Float64(j)
            rows[f * S + 7 + j] += Scalar[DT](0.3 * sin(w * t))
            rows[f * S + QD + j] = Scalar[DT](0.3 * w * sin(w * t + 1.5707963267948966))
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
        rows.resize(o + rows_n * S, 0)
        for i in range(rows_n * S):
            rows[o + i] = refc[i]
    var frames = len(rows) // S
    print("  library:", len(lens), "clips,", frames, "frames (", Float64(frames) / 50.0 / 3600.0, "h ) from", len(paths), "store(s)")
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def _subset_of(lib: SMMotionLibrary[Robot], clips: List[Int]) raises -> SMMotionLibrary[Robot]:
    """The listed clips of `lib`, in that order."""
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    for i in range(len(clips)):
        var c = clips[i]
        if c < 0 or c >= lib.num_clips:
            raise Error("--bench-list: clip " + String(c) + " is outside the library's " + String(lib.num_clips))
        var f0 = lib.clip_start[c]
        var l = lib.clip_len[c]
        lens.append(l)
        var o = len(rows)
        rows.resize(o + l * S, 0)
        for j in range(l * S):
            rows[o + j] = lib.rows[f0 * S + j]
    return SMMotionLibrary[Robot](rows^, lens, _q0())


def _read_benchmark(path: String, n_clips: Int) raises -> Tuple[List[Float64], List[Bool]]:
    """A benchmark.csv's per-clip normalised return and failure, indexed by its `clip` column."""
    var ret = List[Float64](length=n_clips, fill=-1.0)
    var failed = List[Bool](length=n_clips, fill=False)
    var seen = 0
    with open(path, "r") as f:
        var lines = f.read().split("\n")
        for i in range(1, len(lines)):
            var t = String(lines[i].strip())
            if t.byte_length() == 0:
                continue
            var cols = t.split(",")
            var c = Int(String(cols[0]))
            if c < 0 or c >= n_clips:
                raise Error("--sampler-seed: clip " + String(c) + " is outside the library's " + String(n_clips))
            ret[c] = Float64(String(cols[3]))
            failed[c] = String(cols[4]) == "1"
            seen += 1
    if seen != n_clips:
        raise Error("--sampler-seed: " + String(seen) + " clips in " + path + ", the library has " + String(n_clips))
    return (ret^, failed^)


def _read_ints(path: String) raises -> List[Int]:
    var out = List[Int]()
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            var t = String(line.strip())
            if t.byte_length() > 0:
                out.append(Int(t))
    return out^


def _subset(lib: SMMotionLibrary[Robot], k: Int) raises -> SMMotionLibrary[Robot]:
    """K clips evenly spaced over `lib` (clip i · n / K): a fixed set that
    spans every store of a sorted import."""
    var n = lib.num_clips
    var kk = min(max(k, 1), n)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
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
    bench_every: Int, bench_clips: Int, bench_list: List[Int], resume: String, lr_restart: Float64, lr_restart_decay: Int,
    count: Bool, sampler_seed: String,
) raises -> String:
    """Train in the build's mode; returns the run's outcome."""
    var t0 = perf_counter_ns()
    var updates: Int
    var avg: Float64
    comptime if MODE == 2:
        var dctx = DeviceContext()
        var bench_lib = Optional(_subset_of(lib, bench_list) if len(bench_list) > 0 else _subset(lib, (bench_clips if bench_clips > 0 else BENCH_LANES) if bench_every > 0 else 1))
        var agent = sm_agent_gpu[C, RBatched[C.LANES]](dctx, lib^, RBatched[C.LANES].create(dctx), iters, noise, seed_v)
        if count:
            agent.env.sampler = SMResetSampler(agent.env.lib.num_bins, True)
            if sampler_seed.byte_length() > 0:
                var bm = _read_benchmark(sampler_seed, agent.env.lib.num_clips)
                agent.env.sampler.seed_from_benchmark(agent.env.lib.bin_clip, bm[0], bm[1])
                print("  count estimator seeded from", sampler_seed)
        if resume.byte_length() > 0:
            var ctr = agent.trainer.load_state(resume)
            var decay = lr_restart_decay if lr_restart_decay > 0 else iters - ctr[0]
            agent.resume(ctr[0], ctr[1], lr_restart, decay)
            print("  resumed from", resume, "at iteration", ctr[0], "(", ctr[1], "updates ), to", iters, "| lr restart", lr_restart, "over", decay, "iterations")
        print("  reset sampler:", "count (realmimic)" if count else "ema (the reproduction's)")
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
        if count:
            agent.env.sampler = SMResetSampler(agent.env.lib.num_bins, True)
            if sampler_seed.byte_length() > 0:
                var bm = _read_benchmark(sampler_seed, agent.env.lib.num_clips)
                agent.env.sampler.seed_from_benchmark(agent.env.lib.bin_clip, bm[0], bm[1])
                print("  count estimator seeded from", sampler_seed)
        if resume.byte_length() > 0:
            var ctr = agent.trainer.load_state(resume)
            var decay = lr_restart_decay if lr_restart_decay > 0 else iters - ctr[0]
            agent.resume(ctr[0], ctr[1], lr_restart, decay)
            print("  resumed from", resume, "at iteration", ctr[0], "(", ctr[1], "updates ), to", iters, "| lr restart", lr_restart, "over", decay, "iterations")
        print("  reset sampler:", "count (realmimic)" if count else "ema (the reproduction's)")
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
    var bench_list_path = _opt(args, String("--bench-list"), String(""))
    var resume = _opt(args, String("--resume"), String(""))
    var lr_restart = Float64(_opt(args, String("--lr-restart"), String("0.3")))
    var lr_restart_decay = Int(_opt(args, String("--lr-restart-decay"), String("0")))
    var sampler = _opt(args, String("--sampler"), String("ema"))
    if sampler != "ema" and sampler != "count":
        raise Error("--sampler is ema or count, not " + sampler)
    var sampler_seed = _opt(args, String("--sampler-seed"), String(""))
    if sampler_seed.byte_length() > 0 and sampler != "count":
        raise Error("--sampler-seed seeds the count estimator: give --sampler count")
    var bench_list = _read_ints(bench_list_path) if bench_list_path.byte_length() > 0 else List[Int]()
    if bench_every > 0 and MODE != 2:
        raise Error("--bench-every needs the device env (-D SM_GPU=2)")

    var lib = _from_store(store, max_clips) if store != "" else _synthetic()
    var mode_s = String("cpu") if MODE == 0 else (String("gpu-nets") if MODE == 1 else String("gpu-all"))
    var run = RunContext(
        project=String("bhl") if BHL else String("g1"),
        driver=String("examples/g1/supermimic_train.mojo"),
        slug=String("supermimic-") + mode_s + (String("-paper") if P else (String("-light") if LIGHT else String("-smoke"))),
        env=String("noeira:bhl/supermimic_tracking") if BHL else String("noeira:g1/supermimic_tracking"),
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
    logger.set_config("sampler", sampler + (String(" seeded from ") + sampler_seed if sampler_seed.byte_length() > 0 else String("")))
    logger.set_config("resume", resume)
    logger.set_config("lr_restart", String(lr_restart) if resume.byte_length() > 0 else String("off"))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var lp = Pointer(to=logger).as_unsafe_any_origin()
    var ckpt = run.checkpoint_path(String("last"))
    print("SuperMimic, mode", MODE, "(" + String(TARGET) + "):", C.LANES, "lanes, H =", C.HORIZON, ", WM batch", C.WM_BATCH, ", policy batch", C.PI_BATCH, ", noise", noise)
    print("  run", run.dir)
    var outcome = _train(
        lib^, iters, log_every, noise, seed_v, ckpt_every, lp, ckpt, artifacts, run.dir, bench_every, bench_clips, bench_list, resume,
        lr_restart, lr_restart_decay, sampler == "count", sampler_seed,
    )
    finish_run(run, logger, artifacts, outcome)
    _ = logger  # lifetime extender for lp
