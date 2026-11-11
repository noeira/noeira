# +--------------------------------------------------------------------------+ #
# | SuperMimic on our G1 — the Appendix F benchmark of a checkpoint
# +--------------------------------------------------------------------------+ #
"""Score a SuperMimic checkpoint on every clip of a library (the paper's
Appendix F protocol, `deep_agents/supermimic/eval.mojo`), on the device env
(NVIDIA: the batched G1).

    pixi run -e nvidia mojo build -I . -D SM_PAPER=1 examples/g1/supermimic_eval.mojo -o sm_eval
    pixi run ./sm_eval --ckpt runs/<id>/checkpoints/last.ckpt --store /workspace/seed/stores

The run is a `RunContext` of its own (project g1, slug `supermimic-eval`):
its `run.kv` names the checkpoint, the per-clip CSV is in its `eval/`
directory, and the monitor gets `eval_return` (the mean normalised return),
`eval_return_step_weighted`, `eval_success_rate` (1 − failures / clips) and
the per-term means. ⚠ SEED is confidential: run it on private compute only.

Options
-------
--ckpt PATH      a `SuperMimicTrainer.save_state` checkpoint (the config's widths)
--store PATH     a store, a comma-separated list or a directory of stores
--max-clips N    the first N clips only (0 = all)
"""

from std.builtin.sort import sort
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run, run_logger
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.eval import sm_benchmark_gpu, sm_summarize, sm_term_name
from noeira.deep_agents.supermimic.heads import SM_N_TERMS
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import (
    BhlBatchedSim, BhlRobot, G1BatchedSim, G1MjlabBatchedSim, G1MjlabRobot, G1Robot, SMDims, TrackingSim,
)
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT


comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime LIGHT = get_defined_int["SM_LIGHT", 0]() != 0 and not P

# The robot: mjlab's G1 (the reproduction's body) by default; `-D SM_BFM_G1=1`
# selects BFM-Zero's G1 (checkpoints trained before 2026-11-05 are on it),
# `-D SM_BHL=1` Berkeley Humanoid Lite (with `-D BHL_EFFORT_PCT=` / `-D
# BHL_SPEED_PCT=` to benchmark a checkpoint under other actuator limits).
comptime BFM_G1 = get_defined_int["SM_BFM_G1", 0]() != 0
comptime BHL = get_defined_int["SM_BHL", 0]() != 0
comptime Robot = BhlRobot if BHL else (G1Robot if BFM_G1 else G1MjlabRobot)
comptime RBatched[N: Int] = BhlBatchedSim[N] if BHL else (G1BatchedSim[N] if BFM_G1 else G1MjlabBatchedSim[N])
comptime S = SMDims[Robot].S                   # a library row: qpos, then qvel
comptime NDOF = Robot.NDOF
# the training example's widths (`supermimic_train.mojo`, the same three
# sizes): a checkpoint loads only into its own config
comptime C = SMConfig[
    "gpu", Robot,
    2048 if P else (1024 if LIGHT else 128), 64,
    2048 if P else (1024 if LIGHT else 128), 1024 if P else (512 if LIGHT else 64),
    512 if P else (256 if LIGHT else 64), 256 if P else (128 if LIGHT else 32),
    2048 if P else (1024 if LIGHT else 128), 1024 if P else (512 if LIGHT else 64),
    512 if P else (256 if LIGHT else 64), 256 if P else (128 if LIGHT else 32),
    256 if P else (128 if LIGHT else 64), 128 if P else (64 if LIGHT else 32),
    4096 if P else (1024 if LIGHT else 8), 2048 if P else (1024 if LIGHT else 64),
    4096 if P else (2048 if LIGHT else 16), 32,
]


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _q0() -> List[Float64]:
    var q0 = List[Float64](length=NDOF, fill=0.0)
    for j in range(NDOF):
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
        rows.resize(o + rows_n * S, 0)
        for i in range(rows_n * S):
            rows[o + i] = refc[i]
    print("  library:", len(lens), "clips,", len(rows) // S, "frames from", len(paths), "store(s)")
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
    var run = RunContext(
        project=String("bhl") if BHL else String("g1"), driver=String("examples/g1/supermimic_eval.mojo"),
        slug=String("supermimic-eval"), env=String("noeira:g1/supermimic_tracking"), dataset=store,
    )
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "SuperMimic")
    logger.set_config("protocol", "appendix_f")
    logger.set_config("checkpoint", ckpt)
    logger.set_config("lanes", String(C.LANES))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    print("SuperMimic benchmark:", ckpt, "| run", run.dir)
    var ctx = DeviceContext()
    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), Optional(ctx))
    var ctr = trainer.load_state(ckpt)
    print("  checkpoint at iteration", ctr[0], ",", ctr[1], "updates")
    var lib = _library(store, max_clips)
    var t0 = perf_counter_ns()
    var res = sm_benchmark_gpu[C, RBatched[C.LANES]](ctx, trainer, lib^, RBatched[C.LANES].create(ctx), max_clips)
    var s = sm_summarize(res)
    var secs = Float64(perf_counter_ns() - t0) * 1e-9
    print("\n", s.clips, "clips | failed", s.failed, "(", 100.0 * Float64(s.failed) / Float64(max(s.clips, 1)), "% ) | mean norm return", s.mean_norm_return, "| step-weighted", s.step_weighted_norm_return, "|", secs, "s")
    print("per-step term means (max 0.02 for the body terms, 0.01 for the root terms):")
    for k in range(SM_N_TERMS):
        print("  ", sm_term_name(k), s.terms[k])
    # the 10 worst clips
    var order = List[Int]()
    for i in range(len(res)):
        order.append(i)
    for i in range(min(10, len(order))):
        var best = i
        for j in range(i + 1, len(order)):
            if res[order[j]].norm_return < res[order[best]].norm_return:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp
        var r = res[order[i]].copy()
        print("  worst: clip", r.clip, "ret", r.norm_return, "steps", r.steps, "/", r.frames, "FAILED" if r.failed else "")
    # per-clip CSV
    var path = run.eval_dir() + "/benchmark.csv"
    var csv = String("clip,frames,steps,norm_return,failed")
    for k in range(SM_N_TERMS):
        csv += "," + sm_term_name(k)
    csv += "\n"
    for i in range(len(res)):
        csv += String(res[i].clip) + "," + String(res[i].frames) + "," + String(res[i].steps) + "," + String(res[i].norm_return) + "," + ("1" if res[i].failed else "0")
        for k in range(SM_N_TERMS):
            csv += "," + String(res[i].terms[k])
        csv += "\n"
    with open(path, "w") as f:
        f.write(csv)
    print("  per-clip results:", path)
    logger.log_scalar("eval_return", s.mean_norm_return, ctr[0])
    logger.log_scalar("eval_return_step_weighted", s.step_weighted_norm_return, ctr[0])
    logger.log_scalar("eval_success_rate", 1.0 - Float64(s.failed) / Float64(max(s.clips, 1)), ctr[0])
    logger.log_scalar("eval_episodes", Float64(s.clips), ctr[0])
    for k in range(SM_N_TERMS):
        logger.log_scalar("eval_term_" + sm_term_name(k), s.terms[k], ctr[0])
    finish_run(run, logger, artifacts, String("eval_return=") + String(s.mean_norm_return) + ",failed=" + String(s.failed) + "/" + String(s.clips))
