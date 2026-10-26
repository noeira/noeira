"""Zero-shot evaluation of an ONLINE-trained FB checkpoint on walker.

Sibling of `fb_eval_walker.mojo` for checkpoints written by
`fb_online_walker_gpu.mojo`. Same protocol — `z` from a task's TRUE reward
relabelled over the SAC store's rows, `pi_z` rolled out PAIRED against the
random policy on identical reset seeds, three tasks from ONE checkpoint —
with one difference that is the whole reason this file exists:

⚠⚠ **The online agent's `B` and actor see dm_control's 24-D observation**
(the batched env's vector), not the store's `[qpos | qvel]` (18-D) that the
offline runs trained on. So every state fed to `B` here goes through
`Phyics3dEnv.obs_at(qpos, qvel)`, and the rollout feeds the env's own obs.
Feeding the 18-D layout to a 24-D net would fail on shape — feeding a
24-D vector assembled in the wrong ORDER would not, and would score as
"the arm did not help". `obs_at` is the one producer of that vector.

⚠ The relabel rows come from `fb_walker_all_sac.h5` even though the online
run never saw it: `z = E_rho[B(s)·r(s)]` needs SOME state distribution with
coverage of the task, and using the same rows as the offline eval keeps the
two arms comparable. §16.5 notes the choice of rows is a real knob.

⚠ `--episodes 64` is the number that makes stand and run decidable
(fb_eval_walker.mojo's power note); use fewer only for a shakeout.

    pixi run mojo run -I . examples/fb/fb_eval_walker_online.mojo <ckpt> \
        [--episodes N] [--rows N]

⚠ This file and `fb_eval_walker.mojo` should become ONE script with an
observation-mode parameter once the A2 sweep has finished running on the
other one. Written separately so the sweep's binary is untouched.
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.sys import argv, get_defined_int

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine

from noeira.data.store import TrajectoryStore
from noeira.data.resident import ResidentColumn
from noeira.data.sampler import UniformSampler

from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

from noeira.nn.core.module import Module
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb import z_from_reward
from noeira.deep_agents.fb.walker_nets import WalkerBNetProj


comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = DMWalkerModel.ACTION_DIM
comptime OBS: Int = DMWalkerModel.OBS_DIM   # 24 — see the header
comptime D: Int = 128
comptime HID: Int = 1024
comptime BATCH: Int = 1024
comptime STORE: StaticString = "fb_walker_all_sac.h5"
comptime Z_CENTER: Bool = True              # fb_eval_walker.mojo's A/B'd default
comptime RELABEL_ROWS_DEFAULT: Int = 4096
comptime EVAL_EPISODES_DEFAULT: Int = 64
comptime EVAL_LEN: Int = 1000
comptime SEED: Int = 20260805

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D
# ⚠ MUST match fb_online_walker_gpu.mojo exactly, norm layer included.
comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
# `-D B_PROJ=1` (mojo BUILD, not run — a stale define-less compile is reused
# by `mojo run`): B reads the walker projection ϑ_w instead of the whole
# observation — the FB-MEBE rung-W arms (`fb_mebe_walker_gpu.mojo`). Every
# other net is the same family, so one eval scores both.
comptime B_PROJ: Int = get_defined_int["B_PROJ", 0]()
comptime TrainerOf[BN: Module] = FBTrainer[FNet, BN, ANet, OBS, NACT, D, BATCH, "cpu"]
comptime ScorerEnv = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[0.0], DType.float64, False
]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _draw(ref s: UniformSampler, n: Int) raises -> List[Int]:
    var d = s.draw(n)
    var out = List[Int]()
    for i in range(n):
        out.append(Int(d.host[i]))
    return out^


def _z_for_task[
    CONFIG: Phyics3dEnvConfig
](
    ref qpos: ResidentColumn[DType.float32],
    ref qvel: ResidentColumn[DType.float32],
    ref action: ResidentColumn[DType.float32],
    ref idx: List[Int],
    ref b_flat: List[Scalar[DT]],
    name: String,
) raises -> Tensor:
    comptime Env = Phyics3dEnv[DMWalkerModel, CONFIG, DType.float64, False]
    var scorer = Env()
    _ = scorer.reset()
    var n = len(idx)
    var rewards = List[Scalar[DT]](length=n, fill=Scalar[DT](0))
    var q = List[Float64](length=NQ, fill=0.0)
    var v = List[Float64](length=NV, fill=0.0)
    var ac = List[Float64](length=NACT, fill=0.0)
    var lo = Float64(1e30)
    var hi = Float64(-1e30)
    var sm = Float64(0)
    for i in range(n):
        var r = idx[i]
        for k in range(NQ):
            q[k] = Float64(qpos.host[r * NQ + k])
        for k in range(NV):
            v[k] = Float64(qvel.host[r * NV + k])
        for k in range(NACT):
            ac[k] = Float64(action.host[r * NACT + k])
        var got = scorer.reward_at(q, v, ac)
        var rv = Float64(got[0])
        rewards[i] = Scalar[DT](rv)
        sm += rv
        if rv < lo:
            lo = rv
        if rv > hi:
            hi = rv
    print(
        "      ", name, "relabelled reward: mean", sm / Float64(n),
        " range [", lo, ",", hi, "]",
    )
    comptime if Z_CENTER:
        var mu = sm / Float64(n)
        for i in range(n):
            rewards[i] = Scalar[DT](Float64(rewards[i]) - mu)
    var zl = z_from_reward[D](b_flat, rewards, n)
    var z = Tensor.alloc(D)
    for k in range(D):
        z.data[k] = zl[k]
    return z^


def _rollout[
    CONFIG: Phyics3dEnvConfig, BN: Module
](
    mut t: TrainerOf[BN], ref z: Tensor, use_policy: Bool, ep_seed: Int,
    mut act_mean_abs: Float64, mut act_sat_frac: Float64,
) raises -> Float64:
    comptime Env = Phyics3dEnv[DMWalkerModel, CONFIG, DType.float64, False]
    seed(ep_seed)
    var env = Env()
    var state = env.reset()
    var obs = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    for k in range(D):
        z1.data[k] = z.data[k]
    var act_out = Tensor()
    var ret = Float64(0)
    var abs_sum = Float64(0)
    var sat = 0
    var n_act = 0
    for _ in range(EVAL_LEN):
        var a = Env.ActionType()
        if use_policy:
            # The env's OWN observation — the vector the online actor trained on.
            for k in range(OBS):
                obs.data[k] = Scalar[DT](Float64(state.data[k]))
            t.act[1](obs, z1, act_out)
            for k in range(NACT):
                var av = Float64(act_out.data[k])
                a.data[k] = av
                abs_sum += abs(av)
                if abs(av) > 0.99:
                    sat += 1
                n_act += 1
        else:
            for k in range(NACT):
                a.data[k] = random_float64() * 2.0 - 1.0
        var out = env.step(a)
        state = out[0]
        ret += Float64(out[1])
    if use_policy and n_act > 0:
        act_mean_abs = abs_sum / Float64(n_act)
        act_sat_frac = Float64(sat) / Float64(n_act)
    return ret


def _eval_task[
    CONFIG: Phyics3dEnvConfig, BN: Module
](mut t: TrainerOf[BN], ref z: Tensor, name: String, episodes: Int) raises:
    var diffs = List[Float64]()
    var rp = Float64(0)
    var rr = Float64(0)
    var mean_abs = Float64(0)
    var sat_frac = Float64(0)
    for ep in range(episodes):
        var ama = Float64(0)
        var asf = Float64(0)
        var a = _rollout[CONFIG, BN](t, z, True, SEED + 1000 + ep, ama, asf)
        var da = Float64(0)
        var db = Float64(0)
        var b = _rollout[CONFIG, BN](t, z, False, SEED + 1000 + ep, da, db)
        mean_abs += ama
        sat_frac += asf
        rp += a
        rr += b
        diffs.append(a - b)
    var n = Float64(episodes)
    var mp = rp / n
    var mr = rr / n
    var md = Float64(0)
    for i in range(len(diffs)):
        md += diffs[i]
    md /= n
    var sq = Float64(0)
    for i in range(len(diffs)):
        var d = diffs[i] - md
        sq += d * d
    var sd = sqrt(sq / (n - 1.0)) if episodes > 1 else 0.0
    var se = sd / sqrt(n)
    var tstat = md / se if se > 1e-12 else 0.0
    print(
        "   ", name, ": pi_z", mp, "  random", mr,
        "  ratio", mp / mr if mr > 1e-9 else 0.0,
    )
    print(
        "            paired diff", md, "+-", se, " (t =", tstat, ")",
        " SIGNAL" if abs(tstat) > 2.0 else " within noise",
    )
    print(
        "            pi_z action: mean|a| =", mean_abs / n,
        " saturated(|a|>0.99) =", sat_frac / n,
    )


def main() raises:
    comptime if B_PROJ == 1:
        print("[B_PROJ=1] B reads the walker projection ϑ_w")
        _main[WalkerBNetProj]()
    else:
        _main[BNet]()


def _main[BN: Module]() raises:
    var av = argv()
    if len(av) < 2 or String(av[1]).startswith("--"):
        raise Error("usage: fb_eval_walker_online.mojo <ckpt> [--episodes N] [--rows N]")
    var ck = String(av[1])
    var episodes = atol(_flag(String("--episodes"), String(EVAL_EPISODES_DEFAULT)))
    var n_relabel = atol(_flag(String("--rows"), String(RELABEL_ROWS_DEFAULT)))
    print("[1] loading checkpoint", ck, "(online arm, OBS =", OBS, ") ...")
    var t = TrainerOf[BN].make(lr=3e-4, ctx=None)
    t.load_state(ck)

    print("[2] loading", STORE, "for relabelling ...")
    var store = TrajectoryStore(String(STORE))
    var n_rows = store.n_rows()
    var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
    var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
    var action = ResidentColumn[DType.float32].load(store, String("action"))
    var sampler = UniformSampler(n_rows)
    seed(SEED)
    var idx = _draw(sampler, n_relabel)

    # B(s) over the relabel rows, with s rebuilt as the 24-D env observation.
    var scorer = ScorerEnv()
    _ = scorer.reset()
    var q = List[Float64](length=NQ, fill=0.0)
    var v = List[Float64](length=NV, fill=0.0)
    var b_in = Tensor.alloc(n_relabel * OBS)
    for i in range(n_relabel):
        var r = idx[i]
        for k in range(NQ):
            q[k] = Float64(qpos.host[r * NQ + k])
        for k in range(NV):
            v[k] = Float64(qvel.host[r * NV + k])
        var o = scorer.obs_at(q, v)
        for k in range(OBS):
            b_in.data[i * OBS + k] = Scalar[DT](Float64(o.data[k]))
    var b_flat = List[Scalar[DT]](length=n_relabel * D, fill=Scalar[DT](0))
    # `backward_embed[N]` is comptime in N; chunk at the default row count.
    var done = 0
    while done < n_relabel:
        var take = RELABEL_ROWS_DEFAULT if n_relabel - done >= RELABEL_ROWS_DEFAULT else n_relabel - done
        if take == RELABEL_ROWS_DEFAULT:
            var chunk = Tensor.alloc(RELABEL_ROWS_DEFAULT * OBS)
            for i in range(RELABEL_ROWS_DEFAULT * OBS):
                chunk.data[i] = b_in.data[done * OBS + i]
            var out = Tensor()
            t.backward_embed[RELABEL_ROWS_DEFAULT](chunk, out)
            for i in range(RELABEL_ROWS_DEFAULT * D):
                b_flat[done * D + i] = out.data[i]
        else:
            # Tail smaller than a chunk: one row at a time (shakeout sizes).
            for r in range(take):
                var one = Tensor.alloc(OBS)
                for k in range(OBS):
                    one.data[k] = b_in.data[(done + r) * OBS + k]
                var out1 = Tensor()
                t.backward_embed[1](one, out1)
                for k in range(D):
                    b_flat[(done + r) * D + k] = out1.data[k]
        done += take

    print("[3] computing z per task from RELABELLED rewards ...")
    var z_stand = _z_for_task[DMWalkerConfig[0.0]](
        qpos, qvel, action, idx, b_flat, String("stand")
    )
    var z_walk = _z_for_task[DMWalkerConfig[1.0]](
        qpos, qvel, action, idx, b_flat, String("walk ")
    )
    var z_run = _z_for_task[DMWalkerConfig[8.0]](
        qpos, qvel, action, idx, b_flat, String("run  ")
    )
    var d_sw = Float64(0)
    var d_wr = Float64(0)
    for k in range(D):
        d_sw += abs(Float64(z_stand.data[k]) - Float64(z_walk.data[k]))
        d_wr += abs(Float64(z_walk.data[k]) - Float64(z_run.data[k]))
    print("      |z_stand - z_walk|_1 =", d_sw, "  |z_walk - z_run|_1 =", d_wr)
    if d_sw < 1e-3 or d_wr < 1e-3:
        print("  ⚠⚠ task z vectors are (nearly) IDENTICAL — z_from_reward is not"
              " responding to its reward; the numbers below are one experiment")

    print("[4] rolling out", episodes, "x", EVAL_LEN, "steps per task, paired ...")
    _eval_task[DMWalkerConfig[0.0], BN](t, z_stand, String("stand"), episodes)
    _eval_task[DMWalkerConfig[1.0], BN](t, z_walk, String("walk "), episodes)
    _eval_task[DMWalkerConfig[8.0], BN](t, z_run, String("run  "), episodes)
    print("")
    print("Offline bar (A2 winner, two seeds, §18.6.2): stand 1.51  walk 1.82  run 1.44")
    print("SAC expert returns in the store:             ~983      ~965      ~720")
    print("A3 run 1 (no act penalty):  stand 2.06 walk 0.71 run 0.48 — 82-90% SATURATED")
    print("A3 run 2 (plain L2 @ 1.0):  stand 0.93 walk 0.75 run 0.96 — NULL policy, mean|a| 0.10")
