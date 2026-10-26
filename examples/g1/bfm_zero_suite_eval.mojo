"""The BFM-Zero G1 on FB-MEBE's base-command suite — the teacher's ceiling
(`noeira-docs/FB_MEBE_G1_PLAN.md` §13.5), on the CPU.

    pixi run mojo run -I . examples/g1/bfm_zero_suite_eval.mojo --reps 3

`mojo run`, not `mojo build`, on the Mac: the build dies in `ld` (a mangled
symbol longer than the linker's limit) and Xcode 27's linker ignores
`-ld_classic`. It takes no `-D`, so the JIT runs the code as written. The
full suite (22 tasks × 3 prompts × 3 reps) runs in about 3.5 min after
the compile.

Before seeding FB-MEBE's ring with this policy's rollouts, measure what it
does on the SAME 22 tasks, the SAME reward (`g1_cmd_reward`, σ_v 0.3, σ_ω
0.2, σ_g 0.1, σ_h 0.05) and the SAME start (stand keyframe, every joint
moved U(−0.1, 0.1)) as `fb_mebe_g1_gpu.mojo`'s eval: 250 steps, return in
[0, 250], a sticky fall below h 0.5. The columns read the 527-D obs at the
slots the suite does (h 64, gravity 58:61, heading-frame vx/vy 341:343,
ωz 436) — the same slots the BFM-Zero joystick reads.

The z for a task is inferred from that task's reward over `POOL` rows of
the LAFAN store (the policy's own training data, strided), three ways:

    raw      z_from_reward(B, r)            the joystick's convention
    centred  z_from_reward(B, r − mean r)   FB-MEBE's eval convention
    wide     z_from_reward(B, r^¼)          every σ × 2 (probe `upright_wide`)

Per task and prompt it prints the return, the fall rate, the mean (vx, vy,
ωz, h) held, |vx − vx*|, and vx over the last 125 steps; per task it also
prints how many pool rows the reward sees (r > 0.5, r > 0.1), which says
whether LAFAN even contains the commanded state.

`--dump-z FILE` also writes each task's z per prompt, one
`z <task> <prompt> v_1 .. v_D` line each — what
`bfm_zero_prior_collect.mojo` rolls the teacher out under on the box, which
has no LAFAN. `--reps 0` writes them without any rollout:

    pixi run mojo run -I . examples/g1/bfm_zero_suite_eval.mojo \
        --ckpt projects/g1/policies/bfm_room.ckpt --reps 0 --prompts centred \
        --dump-z g1_teacher_z_s36000.txt
"""

from std.math import abs
from std.random import random_float64, seed
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.core.cont_action import ContAction
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.z_sampler import z_from_reward
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.deep_agents.fb.g1_mebe import (
    G1Cmd,
    G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_H, G1M_COL_GX, G1M_COL_GY,
    g1_cmd_suite,
    g1_cmd_reward,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import (
    G1_N_DOF, g1_default_pos, g1_pos_lower, g1_pos_upper,
)
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
    UNITREE_G1_PRIV_DIM, ROOT_QPOS_SIZE,
)
from noeira.envs.robots.g1_tracking_eval import G1_D, G1_H, G1_L, G1_HB

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime POOL: Int = 8192
comptime EVAL_LEN: Int = 250
comptime EVAL_FALL_H: Float64 = 0.5
comptime RESET_NOISE: Float64 = 0.1

comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _fmt(x: Float64) -> String:
    """3 decimals, for a table the eye can scan."""
    var s = Float64(Int(abs(x) * 1000.0 + 0.5)) / 1000.0
    var t = String(s)
    return ("-" if x < 0.0 and s > 0.0 else "") + t


def _reward_row(ref c: G1Cmd, ref p: List[Float64], ref o: List[Scalar[DT]], off: Int) -> Float64:
    return g1_cmd_reward(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        Float64(o[off + G1M_COL_VX]), Float64(o[off + G1M_COL_VY]),
        Float64(o[off + G1M_COL_WZ]), Float64(o[off + G1M_COL_GX]),
        Float64(o[off + G1M_COL_GY]), Float64(o[off + G1M_COL_H]),
    )


def _roll[
    FNET: Module, BNET: Module, ANET: Module
](
    mut t: FBTrainer[FNET, BNET, ANET, OBS, ACT, D, BATCH, "cpu"],
    mut env: UnitreeG1[DType.float64],
    ref norm: Optional[ObsNorm[OBS]],
    ref z: List[Scalar[DT]],
    ref c: G1Cmd,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut acc: List[Float64],
) raises:
    """One 250-step eval rollout from the suite's start. Adds into `acc`:
    return, fell, Σvx, Σvy, Σωz, Σh, Σ|vx − vx*|, Σvx over the last half."""
    _ = env.reset()
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    for i in range(NQ):
        qp[i] = Float64(env.d.qpos.data[i])
    for i in range(G1_N_DOF):
        var q = g1_default_pos(i) + (2.0 * random_float64() - 1.0) * RESET_NOISE
        q = max(g1_pos_lower(i), min(g1_pos_upper(i), q))
        qp[ROOT_QPOS_SIZE + i] = q
    env.set_state(qp, qv)
    var aobs = G1ActorObs()
    for k in range(D):
        z1.data[k] = z[k]
    var p = c.params()
    var ret = 0.0
    var fell = 0.0
    for step in range(EVAL_LEN):
        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
        t.act[1](obs_t, z1, act_out)
        var a = ContAction[ACT]()
        for k in range(ACT):
            var v = Float64(act_out.data[k])
            a.data[k] = max(-1.0, min(1.0, v))
        aobs.push(o, act_out)
        _ = env.step(a)
        var o2 = env.get_obs_list()
        var ol = List[Scalar[DT]](length=SP, fill=Scalar[DT](0))
        for k in range(SP):
            ol[k] = Scalar[DT](o2[k])
        ret += _reward_row(c, p, ol, 0)
        var vx = Float64(ol[G1M_COL_VX])
        var hh = Float64(ol[G1M_COL_H])
        if hh < EVAL_FALL_H:
            fell = 1.0
        acc[2] += vx
        acc[3] += Float64(ol[G1M_COL_VY])
        acc[4] += Float64(ol[G1M_COL_WZ])
        acc[5] += hh
        acc[6] += abs(vx - c.vx)
        if step >= EVAL_LEN // 2:
            acc[7] += vx
    acc[0] += ret
    acc[1] += fell


def main() raises:
    var ckpt = _flag(String("--ckpt"), String("projects/g1/policies/bfm.ckpt"))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var reps = atol(_flag(String("--reps"), String(3)))
    var only = _flag(String("--tasks"), String(""))     # comma list, "" = all
    var prompts = _flag(String("--prompts"), String("raw,centred,wide"))
    var dump_z = _flag(String("--dump-z"), String(""))
    var z_lines = String("")
    seed(atol(_flag(String("--seed"), String(1))))

    print("=" * 76)
    print("BFM-Zero G1 on the FB-MEBE base-command suite —", ckpt)
    print("  pool", POOL, "LAFAN rows (strided)   reps", reps, "  prompts", prompts)
    print("=" * 76)

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm next to the checkpoint: running unnormalised")

    # ── the pool: LAFAN rows as the joystick builds them ──────────────────
    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var n_rows = len(st) // UNITREE_G1_STATE_DIM
    var stride = n_rows // POOL
    var pool = Tensor.alloc(POOL * OBS)
    for i in range(POOL * OBS):
        pool.data[i] = Scalar[DT](0.0)
    var raw = List[Scalar[DT]](length=POOL * SP, fill=Scalar[DT](0))
    for i in range(POOL):
        var r = i * stride
        for k in range(UNITREE_G1_STATE_DIM):
            var x = Scalar[DT](st[r * UNITREE_G1_STATE_DIM + k])
            pool.data[i * OBS + k] = x
            raw[i * SP + k] = x
        for k in range(UNITREE_G1_PRIV_DIM):
            var x = Scalar[DT](pv[r * UNITREE_G1_PRIV_DIM + k])
            pool.data[i * OBS + UNITREE_G1_STATE_DIM + k] = x
            raw[i * SP + UNITREE_G1_STATE_DIM + k] = x
    if norm:
        norm.value().apply_rows(pool, POOL)
    var b_pool = Tensor()
    t.backward_embed[POOL](pool, b_pool)
    var b_list = List[Scalar[DT]](length=POOL * D, fill=Scalar[DT](0))
    for i in range(POOL * D):
        b_list[i] = b_pool.data[i]
    print("  pool", POOL, "of", n_rows, "rows encoded")

    var env = UnitreeG1[DType.float64]()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)

    var suite = g1_cmd_suite()
    var pnames = prompts.split(",")
    # family sums per prompt
    var fam = List[Float64](length=len(pnames) * 3, fill=0.0)
    var fam_n = List[Int](length=3, fill=0)
    var t0 = perf_counter_ns()
    print("POOL task  r>0.5  r>0.1  mean_r")
    print("EVAL prompt task  ret  fall   vx  vy  wz  h   |e_vx|  late_vx")
    for k in range(len(suite)):
        var c = suite[k].copy()
        if only.byte_length() > 0:
            var hit = False
            for nm in only.split(","):
                if String(nm) == c.name:
                    hit = True
            if not hit:
                continue
        fam_n[c.family] += 1
        var p = c.params()
        var rew = List[Float64](length=POOL, fill=0.0)
        var n5 = 0
        var n1 = 0
        var mu = 0.0
        for i in range(POOL):
            var r = _reward_row(c, p, raw, i * SP)
            rew[i] = r
            mu += r
            if r > 0.5:
                n5 += 1
            if r > 0.1:
                n1 += 1
        mu /= Float64(POOL)
        print("POOL", c.name, n5, n1, _fmt(mu))
        for v in range(len(pnames)):
            var pn = String(pnames[v])
            var rv = List[Scalar[DT]](length=POOL, fill=Scalar[DT](0))
            for i in range(POOL):
                if pn == "raw":
                    rv[i] = Scalar[DT](rew[i])
                elif pn == "centred":
                    rv[i] = Scalar[DT](rew[i] - mu)
                elif pn == "wide":
                    rv[i] = Scalar[DT](rew[i] ** 0.25 if rew[i] > 0.0 else 0.0)
                else:
                    raise Error("unknown prompt '" + pn + "' (raw, centred, wide)")
            var z = z_from_reward[D](b_list, rv, POOL)
            if dump_z.byte_length() > 0:
                z_lines += "z " + c.name + " " + pn
                for j in range(D):
                    z_lines += " " + String(z[j])
                z_lines += "\n"
            if reps == 0:
                continue
            var acc = List[Float64](length=8, fill=0.0)
            for _ in range(reps):
                _roll[FNet, BNet, ANet](
                    t, env, norm, z, c, obs_t, z1, act_out, acc
                )
            var nr = Float64(reps)
            var ns = Float64(reps * EVAL_LEN)
            var nl = Float64(reps * (EVAL_LEN - EVAL_LEN // 2))
            fam[v * 3 + c.family] += acc[0] / nr
            print("EVAL", pn, c.name, _fmt(acc[0] / nr), _fmt(acc[1] / nr),
                  _fmt(acc[2] / ns), _fmt(acc[3] / ns), _fmt(acc[4] / ns),
                  _fmt(acc[5] / ns), _fmt(acc[6] / ns), _fmt(acc[7] / nl))
    if dump_z.byte_length() > 0:
        with open(dump_z, "w") as fh:
            fh.write(z_lines)
        print("  wrote the prompts' z ->", dump_z)
    print("-" * 76)
    var fam_names: List[String] = ["velocity", "posture", "composite"]
    for v in range(len(pnames)):
        var line = String("[suite ") + String(pnames[v]) + "]"
        for f in range(3):
            if fam_n[f] > 0:
                line += " " + fam_names[f] + " " + _fmt(fam[v * 3 + f] / Float64(fam_n[f]))
        print(line)
    print("  ", Float64(perf_counter_ns() - t0) * 1e-9, "s")
