""""Does CEM improve the JOYSTICK's prompts? A four-command probe (§12.46).

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/bfm_zero_cmd_probe.mojo -o build/g1probe
    ./build/g1probe --ckpt runs/<id>/checkpoints/step_36000.ckpt

## Why a probe and not the table

§12.46 showed CEM over a single `z` buys 37.6 % on POSE reaching. The joystick
prompts differently — `z = E_rho[B(s)r(s)]` for a velocity command — and the
obvious upgrade is to CEM-optimise a grid of commands offline and ship the
table, slerping between grid points at runtime.

That grid is ~60 commands and a couple of hours. This measures FOUR first,
because the knot sweep (§12.47) is a standing reminder that a parameterisation
can be wrong in a way no amount of search budget will reveal: there, CEM
returned exactly 0.0 % and the fault was the knots, not the search. Four
commands is enough to see whether the gain exists at all.

## The objective

Mean |v_achieved - v_commanded| over the last `HOLD` steps of a `--horizon`
rollout, on the three commanded axes, read from the SAME heading-frame slots
the reward is written against (`local_body_vel` body 0, `local_body_ang_vel`
body 0 z). Achieved velocity is what a driver actually feels, and it is the
quantity the reward is a proxy FOR — so optimising it directly is the honest
test of whether the proxy leaves anything on the table.

⚠ The zero-shot column is EXACTLY what the joystick computes today, same pool,
same sigma, same projection. If it were not, the comparison would be against a
strawman rather than against the shipped demo.
"""

from std.math import exp, sqrt, abs, cos as _cos64, log as _log64
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
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
)
from noeira.envs.robots.unitree_g1_priv_obs import (
    G1_PRIV_OFF_VEL, G1_PRIV_OFF_ANGVEL,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
    UNITREE_G1_PRIV_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_D, G1_H, G1_L, G1_HB, g1_project_z,
)

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime POOL: Int = 4096
comptime HOLD: Int = 40

# the SAME offsets and sigmas the joystick uses — see the header
comptime OFF_VX: Int = UNITREE_G1_STATE_DIM + G1_PRIV_OFF_VEL
comptime OFF_WZ: Int = UNITREE_G1_STATE_DIM + G1_PRIV_OFF_ANGVEL + 2
comptime SIGMA_V: Float64 = 0.5
comptime SIGMA_W: Float64 = 0.5

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


def _f3(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 1000.0 + 0.5)
    var f = String(h % 1000)
    while f.byte_length() < 3:
        f = String("0") + f
    var body = String(h // 1000) + String(".") + f
    return (String("-") + body) if neg else body


def _gauss() -> Float64:
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * _log64(u1)) * _cos64(6.283185307179586 * random_float64())


def _roll[
    FNET: Module, BNET: Module, ANET: Module
](
    mut t: FBTrainer[FNET, BNET, ANET, OBS, ACT, D, BATCH, "cpu"],
    mut env: UnitreeG1[DType.float64],
    ref rsi: G1RsiTable,
    ref norm: Optional[ObsNorm[OBS]],
    start_row: Int,
    ref z: List[Scalar[DT]],
    z_off: Int,
    cx: Float64, cy: Float64, cw: Float64,
    horizon: Int,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut qp: List[Float64],
    mut qv: List[Float64],
) raises -> Float64:
    """Mean |v_achieved - v_commanded| over the last HOLD steps.

    ⚠ Identical reset for every candidate (`set_state` + `G1ActorObs.reset()`)
    or CEM ranks start states rather than prompts — the actor's 401-dim history
    is part of the state.
    """
    var base = start_row * (G1_RSI_NQ + G1_RSI_NV)
    for i in range(NQ):
        qp[i] = Float64(rsi.rows.data[base + i])
    for i in range(NV):
        qv[i] = Float64(rsi.rows.data[base + G1_RSI_NQ + i])
    env.set_state(qp, qv)
    var aobs = G1ActorObs()
    for k in range(D):
        z1.data[k] = z[z_off + k]

    var acc = 0.0
    var n = 0
    for step in range(horizon):
        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
        t.act[1](obs_t, z1, act_out)
        var a = ContAction[ACT]()
        for k in range(ACT):
            var v = Float64(act_out.data[k])
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            a.data[k] = v
        aobs.push(o, act_out)
        _ = env.step(a)
        if step >= horizon - HOLD:
            var o2 = env.get_obs_list()
            acc += (
                abs(Float64(o2[OFF_VX + 0]) - cx)
                + abs(Float64(o2[OFF_VX + 1]) - cy)
                + abs(Float64(o2[OFF_WZ]) - cw)
            ) / 3.0
            n += 1
    return acc / Float64(n)


def main() raises:
    seed(31)
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var horizon = atol(_flag(String("--horizon"), String(150)))
    var iters = atol(_flag(String("--cem-iters"), String(8)))
    var pop = atol(_flag(String("--cem-pop"), String(16)))
    var elites = atol(_flag(String("--cem-elites"), String(4)))
    var sigma0 = Float64(String(_flag(String("--cem-sigma"), String("0.30"))))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 76)
    print("BFM-Zero G1 — does CEM improve the JOYSTICK's prompts? (4 commands)")
    print("=" * 76)
    print("  objective: mean |v_achieved - v_cmd| over the last", HOLD,
          "of", horizon, "steps")
    print("  CEM", iters, "x", pop, ",", elites, "elites — NETWORK FROZEN")

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")

    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var rsi = G1RsiTable.from_store(store)
    var n_rows = len(st) // UNITREE_G1_STATE_DIM
    var stride = n_rows // POOL

    # the joystick's pool, built exactly as the joystick builds it
    var pool = Tensor.alloc(POOL * OBS)
    for i in range(POOL * OBS):
        pool.data[i] = Scalar[DT](0.0)
    var pvx = List[Float64](length=POOL, fill=0.0)
    var pvy = List[Float64](length=POOL, fill=0.0)
    var pwz = List[Float64](length=POOL, fill=0.0)
    for i in range(POOL):
        var r = i * stride
        for k in range(UNITREE_G1_STATE_DIM):
            pool.data[i * OBS + k] = Scalar[DT](
                st[r * UNITREE_G1_STATE_DIM + k]
            )
        for k in range(UNITREE_G1_PRIV_DIM):
            pool.data[i * OBS + UNITREE_G1_STATE_DIM + k] = Scalar[DT](
                pv[r * UNITREE_G1_PRIV_DIM + k]
            )
        pvx[i] = Float64(pool.data[i * OBS + OFF_VX + 0])
        pvy[i] = Float64(pool.data[i * OBS + OFF_VX + 1])
        pwz[i] = Float64(pool.data[i * OBS + OFF_WZ])
    if norm:
        norm.value().apply_rows(pool, POOL)
    var b_pool = Tensor()
    t.backward_embed[POOL](pool, b_pool)
    var b_list = List[Scalar[DT]](length=POOL * D, fill=Scalar[DT](0))
    for i in range(POOL * D):
        b_list[i] = b_pool.data[i]
    print("  pool", POOL, "rows encoded")

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var rewards = List[Scalar[DT]](length=POOL, fill=Scalar[DT](0))
    var start_row = Int(rsi.ep_offset.data[13])

    var cxs = List[Float64]()
    var cys = List[Float64]()
    var cws = List[Float64]()
    var names = List[String]()
    cxs.append(0.8); cys.append(0.0); cws.append(0.0); names.append(String("walk    "))
    cxs.append(-0.5); cys.append(0.0); cws.append(0.0); names.append(String("back    "))
    cxs.append(0.0); cys.append(0.0); cws.append(1.2); names.append(String("spin    "))
    cxs.append(0.5); cys.append(0.4); cws.append(0.0); names.append(String("diagonal"))

    var sum_z = 0.0
    var sum_c = 0.0
    var sum_r = 0.0
    var n_better = 0
    var t0 = perf_counter_ns()
    print("-" * 76)
    print("  command    random   zero-shot(shipped)     CEM     gain")
    for g in range(len(cxs)):
        var cx = cxs[g]
        var cy = cys[g]
        var cw = cws[g]
        # zero-shot: EXACTLY the joystick's prompt
        for i in range(POOL):
            var dx = pvx[i] - cx
            var dy = pvy[i] - cy
            var dw = pwz[i] - cw
            rewards[i] = Scalar[DT](
                exp(-(dx * dx + dy * dy) / (SIGMA_V * SIGMA_V))
                * exp(-(dw * dw) / (SIGMA_W * SIGMA_W))
            )
        var z0 = z_from_reward[D](b_list, rewards, POOL)
        var e_zs = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, z0, 0, cx, cy, cw, horizon,
            obs_t, z1, act_out, qp, qv,
        )
        # positive control
        var zr = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        var zt = Tensor.alloc(D)
        for k in range(D):
            zt.data[k] = Scalar[DT](_gauss())
        g1_project_z[D](zt, 0)
        for k in range(D):
            zr[k] = zt.data[k]
        var e_rnd = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zr, 0, cx, cy, cw, horizon,
            obs_t, z1, act_out, qp, qv,
        )
        # CEM from the zero-shot prompt
        var mean = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        for k in range(D):
            mean[k] = z0[k]
        var sigma = sigma0
        var best = e_zs
        var cand = List[Scalar[DT]](length=pop * D, fill=Scalar[DT](0))
        var score = List[Float64](length=pop, fill=0.0)
        var order = List[Int](length=pop, fill=0)
        for _it in range(iters):
            var ct = Tensor.alloc(pop * D)
            for c in range(pop):
                for k in range(D):
                    ct.data[c * D + k] = Scalar[DT](
                        Float64(mean[k]) + sigma * _gauss()
                    )
                g1_project_z[D](ct, c)
            for c in range(pop):
                for k in range(D):
                    cand[c * D + k] = ct.data[c * D + k]
                score[c] = _roll[FNet, BNet, ANet](
                    t, env, rsi, norm, start_row, cand, c * D,
                    cx, cy, cw, horizon, obs_t, z1, act_out, qp, qv,
                )
                if score[c] < best:
                    best = score[c]
            for c in range(pop):
                order[c] = c
            for i in range(elites):
                var m = i
                for j in range(i + 1, pop):
                    if score[order[j]] < score[order[m]]:
                        m = j
                var tmp = order[i]
                order[i] = order[m]
                order[m] = tmp
            var mt = Tensor.alloc(D)
            for k in range(D):
                var acc = 0.0
                for e in range(elites):
                    acc += Float64(cand[order[e] * D + k])
                mt.data[k] = Scalar[DT](acc / Float64(elites))
            g1_project_z[D](mt, 0)
            for k in range(D):
                mean[k] = mt.data[k]
            sigma *= 0.85
        sum_z += e_zs
        sum_c += best
        sum_r += e_rnd
        if best < e_zs - 1e-9:
            n_better += 1
        print("  ", names[g], " ", _f3(e_rnd), "     ", _f3(e_zs),
              "        ", _f3(best), "  ",
              _f3(100.0 * (e_zs - best) / e_zs) + String("%"))

    var el = Float64(perf_counter_ns() - t0) * 1e-9
    var mz = sum_z / Float64(len(cxs))
    var mc = sum_c / Float64(len(cxs))
    var mr = sum_r / Float64(len(cxs))
    print("-" * 76)
    print("  MEAN  random", _f3(mr), "  zero-shot", _f3(mz), "  CEM", _f3(mc))
    print("  CEM improved", n_better, "of", len(cxs), "commands;",
          _f3(100.0 * (mz - mc) / mz) + String("%"), "mean gain")
    print("  ", el, "s")
    if mr <= mz * 1.05:
        print("  ⚠⚠ CONTROL FAILED: a random z tracks the command as well as"
              " the prompt does, so neither column above is evidence. The"
              " objective cannot separate policies at this horizon.")
    else:
        print("  control OK: a random z is", _f3(mr / mz) + String("x"),
              "the prompt's error")
