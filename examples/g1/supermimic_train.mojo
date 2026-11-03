# +--------------------------------------------------------------------------+ #
# | SuperMimic on our G1 — the training loop, CPU smoke scale
# +--------------------------------------------------------------------------+ #
"""Algorithm A.1 end to end on the CPU (`deep_agents/supermimic/loop.mojo`):
collect, fit the world model, improve the policy through it.

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
from std.sys import argv
from std.time import perf_counter_ns

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic.env import SMG1Env
from noeira.deep_agents.supermimic.loop import SMTrainCfg, SMTrainer
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction


comptime NL = 8
comptime BW = 64
comptime BP = 16
comptime H = 32


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


def _synthetic() raises -> SMMotionLibrary:
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
    return SMMotionLibrary(rows^, lens, _q0())


def _from_store(path: String, max_clips: Int) raises -> SMMotionLibrary:
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
    return SMMotionLibrary(rows^, lens, _q0())


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
    var env = SMG1Env(NL, lib^, dr=noise, obs_noise=noise, reset_noise=noise, action_noise=0.1 if noise else 0.0, rng_seed=seed_v)
    var cfg = SMTrainCfg(iters, 0.4, 40, 2e-4, noise, log_every)
    var tr = SMTrainer[128, 64, 128, 64, 64, 32, 128, 64, 64, 32, 64, 32, NL, BW, BP, H](env^, 1000, cfg^)
    print("SuperMimic CPU smoke:", NL, "lanes, H =", H, ", WM batch", BW, ", policy batch", BP, ", noise", noise)
    var t0 = perf_counter_ns()
    var n_up = 0
    var wl = 0.0
    var w1 = 0.0
    var w8 = 0.0
    var ij = 0.0
    var iu = 0.0
    var cl = 0.0
    var vm = 0.0
    var tmn = 0.0
    var gn = 0.0
    for it in range(iters):
        var r = tr.iterate()
        if r[0]:
            n_up += 1
            wl += r[1].loss
            w1 += r[1].nse_1step
            w8 += r[1].nse_kstep
            ij += r[2].imag_objective
            iu += r[2].imag_return_undisc
            cl += r[2].critic_loss
            vm += r[2].critic_value_mean
            tmn += r[2].critic_target_mean
            gn += r[2].policy_grad_norm
        if (it + 1) % log_every == 0:
            var tr_ret = 0.0
            for k in range(len(tr.finished)):
                tr_ret += tr.finished[k]
            var nf = len(tr.finished)
            var d = Float64(max(n_up, 1))
            print(
                "it", it + 1, "| updates", tr.updates, "| train_ret", (tr_ret / Float64(nf)) if nf > 0 else -1.0, "(n=", nf, ")",
                "| wm loss", wl / d, "nse1", w1 / d, "nse8", w8 / d,
                "| imag J", ij / d, "ret", iu / d, "| critic", cl / d, "V", vm / d, "R", tmn / d, "|g|", gn / d,
                "| t sim/wm/pi", tr.t_sim, tr.t_wm, tr.t_pi,
            )
            tr.finished = List[Float64]()
            n_up = 0
            wl = 0.0
            w1 = 0.0
            w8 = 0.0
            ij = 0.0
            iu = 0.0
            cl = 0.0
            vm = 0.0
            tmn = 0.0
            gn = 0.0
    print("done:", iters, "steps,", tr.updates, "updates in", Float64(perf_counter_ns() - t0) * 1e-9, "s")
