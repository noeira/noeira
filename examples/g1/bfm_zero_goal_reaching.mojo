""""Pose reaching, and what CEM over `z` adds on top — §12.45's next rung.

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/bfm_zero_goal_reaching.mojo -o /tmp/g1goal
    pixi run /tmp/g1goal --ckpt runs/<id>/checkpoints/step_36000.ckpt
    pixi run /tmp/g1goal --ckpt <path> --goals 6 --cem-iters 12 --cem-pop 24

## What this measures, and why it is the rung after the joystick

§12.45 lists three prompt modes. Tracking is the eval; reward optimisation is
the joystick; **goal reaching is the one nothing here had ever exercised** —
`z_from_b` has been in the tree since the walker era with no caller that
scored it.

It also answers the question the whole track keeps deferring: BFM-Zero's
"few-shot optimization-based adaptation" is CEM over `z` with the network
FROZEN, and the paper reports it as the first rung of post-training. So this
runs both and prints them side by side:

  * **zero-shot** `z = project(B(s_goal))` — one forward pass, no search;
  * **CEM**       `z` refined by `--cem-iters` rounds of sample-and-refit,
    starting FROM the zero-shot answer, network untouched.

The gap between those two columns is what post-training is worth on this
model, in this repo's own units — not a number quoted from a paper.

## ⚠ THE METRIC, stated because it decides the answer

Mean absolute joint error over the 29 actuated DoF, averaged over the LAST
`HOLD` steps of a `--horizon` rollout. Averaging over a held window rather
than reading the final frame is deliberate: a policy that swings through the
pose and keeps going would score well on one frame and badly on a held
window, and "reached it" should mean the second thing.

⚠ THE ROOT IS EXCLUDED. `qpos[0:7]` is the floating base; a pose prompt does
not say where in the world to stand, so scoring position would punish a
correct pose reached two metres away. That also means this metric cannot see
the robot falling over WHILE holding a good joint configuration — the height
column is printed beside it so that case is visible rather than hidden.

## ⚠ THE POSITIVE CONTROL IS NOT OPTIONAL

A random `z` on the sphere is scored for every goal. Without it a table of
"zero-shot 0.31, CEM 0.28" says nothing: it could be that any `z` scores ~0.3
on this metric because the G1's joint ranges are small and a standing robot is
never far from any pose. The random column is what makes the other two
readable, and the run prints a loud warning if it is not clearly worse.
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
from noeira.deep_agents.fb.z_sampler import z_from_b
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
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
comptime HOLD: Int = 20
"""Steps at the end of the rollout the error is averaged over."""

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
    var ip = h // 1000
    var fp = h % 1000
    var f = String(fp)
    while f.byte_length() < 3:
        f = String("0") + f
    var body = String(ip) + String(".") + f
    return (String("-") + body) if neg else body


def _gauss() -> Float64:
    """Box-Muller. `z_sampler` rolls its own for the same reason: the stdlib
    has no gaussian and a uniform perturbation would bias CEM toward the
    corners of the cube rather than the sphere it is searching."""
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    var u2 = random_float64()
    return sqrt(-2.0 * _log64(u1)) * _cos64(6.283185307179586 * u2)


struct Reach(Copyable, Movable):
    var err: Float64
    """Mean |dq| over the 29 DoF, averaged over the last HOLD steps."""
    var height: Float64
    """Mean pelvis z over the same window — the fall detector the joint
    metric is blind to."""

    def __init__(out self, err: Float64, height: Float64):
        self.err = err
        self.height = height


def _rollout[
    FNET: Module, BNET: Module, ANET: Module
](
    mut t: FBTrainer[FNET, BNET, ANET, OBS, ACT, D, BATCH, "cpu"],
    mut env: UnitreeG1[DType.float64],
    ref rsi: G1RsiTable,
    ref norm: Optional[ObsNorm[OBS]],
    start_row: Int,
    ref goal_dof: List[Float64],
    ref z: List[Scalar[DT]],
    z_off: Int,
    horizon: Int,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut qp: List[Float64],
    mut qv: List[Float64],
) raises -> Reach:
    """One scored rollout under a fixed `z`. Deterministic given `start_row`.

    ⚠ EVERY CANDIDATE MUST START IDENTICALLY or CEM is ranking start states.
    `set_state` + `G1ActorObs.reset()` is the pair that guarantees it — the
    actor's 401 is part of the state, and a stale history would carry the
    previous candidate's actions into this one's first steps.
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
    var acc_h = 0.0
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
            var e = 0.0
            for k in range(ACT):
                e += abs(Float64(env.d.qpos.data[7 + k]) - goal_dof[k])
            acc += e / Float64(ACT)
            acc_h += Float64(env.d.qpos.data[2])
            n += 1
    return Reach(acc / Float64(n), acc_h / Float64(n))


def main() raises:
    seed(11)
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var n_goals = atol(_flag(String("--goals"), String(6)))
    var horizon = atol(_flag(String("--horizon"), String(120)))
    var cem_iters = atol(_flag(String("--cem-iters"), String(12)))
    var cem_pop = atol(_flag(String("--cem-pop"), String(24)))
    var cem_elites = atol(_flag(String("--cem-elites"), String(6)))
    var sigma0 = Float64(String(_flag(String("--cem-sigma"), String("0.35"))))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 74)
    print("BFM-Zero G1 — POSE REACHING: zero-shot z=B(s_goal) vs CEM over z")
    print("=" * 74)
    print("  horizon", horizon, "steps, error averaged over the last", HOLD)
    print("  CEM:", cem_iters, "iters x", cem_pop, "pop,", cem_elites,
          "elites, sigma0", sigma0, " — THE NETWORK IS FROZEN")

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm sidecar: RAW inputs")

    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var qpos_col = store.load_column[DType.float32](String("qpos"))
    var rsi = G1RsiTable.from_store(store)
    var n_rows = len(st) // UNITREE_G1_STATE_DIM

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var goal_in = Tensor.alloc(OBS)
    var goal_b = Tensor()

    # Every goal is reached FROM THE SAME START, so the columns compare the
    # prompt and nothing else. Clip 13 row 0 is a standing frame of a walk.
    var start_row = Int(rsi.ep_offset.data[13])

    var sum_zs = 0.0
    var sum_cem = 0.0
    var sum_rnd = 0.0
    var n_better = 0
    var t0 = perf_counter_ns()

    print("-" * 74)
    print("  goal      random      zero-shot         CEM     gain   height(zs/cem)")
    for g in range(n_goals):
        # goals spread across the store, avoiding the first rows of a clip
        # (which are all the same neutral stance and would make the task
        # trivially easy in a way the mean would hide)
        var grow = ((2 * g + 1) * n_rows) // (2 * n_goals)
        var goal_dof = List[Float64](length=ACT, fill=0.0)
        for k in range(ACT):
            goal_dof[k] = Float64(qpos_col[grow * NQ + 7 + k])

        # ── zero-shot: z = project(B(s_goal)) ──────────────────────────
        for i in range(OBS):
            goal_in.data[i] = Scalar[DT](0.0)
        for k in range(UNITREE_G1_STATE_DIM):
            goal_in.data[k] = Scalar[DT](st[grow * UNITREE_G1_STATE_DIM + k])
        for k in range(UNITREE_G1_PRIV_DIM):
            goal_in.data[UNITREE_G1_STATE_DIM + k] = Scalar[DT](
                pv[grow * UNITREE_G1_PRIV_DIM + k]
            )
        if norm:
            norm.value().apply_rows(goal_in, 1)
        t.backward_embed[1](goal_in, goal_b)
        var bl = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        for k in range(D):
            bl[k] = goal_b.data[k]
        var z0 = z_from_b[D](bl, 1)
        var r_zs = _rollout[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, goal_dof, z0, 0, horizon,
            obs_t, z1, act_out, qp, qv,
        )

        # ── the positive control: a random z on the same sphere ────────
        var zr = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        for k in range(D):
            zr[k] = Scalar[DT](_gauss())
        var zr_t = Tensor.alloc(D)
        for k in range(D):
            zr_t.data[k] = zr[k]
        g1_project_z[D](zr_t, 0)
        for k in range(D):
            zr[k] = zr_t.data[k]
        var r_rnd = _rollout[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, goal_dof, zr, 0, horizon,
            obs_t, z1, act_out, qp, qv,
        )

        # ── CEM over z, FROM the zero-shot answer, network frozen ──────
        var mean = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        for k in range(D):
            mean[k] = z0[k]
        var sigma = sigma0
        var best = r_zs.err
        var best_h = r_zs.height
        var cand = List[Scalar[DT]](length=cem_pop * D, fill=Scalar[DT](0))
        var score = List[Float64](length=cem_pop, fill=0.0)
        var order = List[Int](length=cem_pop, fill=0)
        for _it in range(cem_iters):
            var ct = Tensor.alloc(cem_pop * D)
            for c in range(cem_pop):
                for k in range(D):
                    ct.data[c * D + k] = Scalar[DT](
                        Float64(mean[k]) + sigma * _gauss()
                    )
                # ⚠ BACK ONTO THE SPHERE, every candidate, every iteration.
                # A perturbation leaves it; `π_z` off the sphere is queried at
                # a point training never reached and its score is meaningless
                # — the search would then optimise the radius, not the task.
                g1_project_z[D](ct, c)
            for c in range(cem_pop):
                for k in range(D):
                    cand[c * D + k] = ct.data[c * D + k]
                var rc = _rollout[FNet, BNet, ANet](
                    t, env, rsi, norm, start_row, goal_dof, cand, c * D,
                    horizon, obs_t, z1, act_out, qp, qv,
                )
                score[c] = rc.err
                if rc.err < best:
                    best = rc.err
                    best_h = rc.height
            # elites by selection sort on the score (pop is tens, not
            # thousands — a real sort would be more code than it saves)
            for c in range(cem_pop):
                order[c] = c
            for i in range(cem_elites):
                var m = i
                for j in range(i + 1, cem_pop):
                    if score[order[j]] < score[order[m]]:
                        m = j
                var tmp = order[i]
                order[i] = order[m]
                order[m] = tmp
            var nm = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
            for i in range(cem_elites):
                var c = order[i]
                for k in range(D):
                    nm[k] = Scalar[DT](
                        Float64(nm[k]) + Float64(cand[c * D + k])
                    )
            var mt = Tensor.alloc(D)
            for k in range(D):
                mt.data[k] = Scalar[DT](Float64(nm[k]) / Float64(cem_elites))
            g1_project_z[D](mt, 0)
            for k in range(D):
                mean[k] = mt.data[k]
            sigma *= 0.85

        sum_zs += r_zs.err
        sum_cem += best
        sum_rnd += r_rnd.err
        if best < r_zs.err - 1e-9:
            n_better += 1
        print("  ", g, "      ", _f3(r_rnd.err), "     ", _f3(r_zs.err),
              "     ", _f3(best), "   ",
              _f3(100.0 * (r_zs.err - best) / r_zs.err) + String("%"),
              "  ", _f3(r_zs.height) + String("/") + _f3(best_h))

    var el = Float64(perf_counter_ns() - t0) * 1e-9
    var m_zs = sum_zs / Float64(n_goals)
    var m_cem = sum_cem / Float64(n_goals)
    var m_rnd = sum_rnd / Float64(n_goals)
    print("-" * 74)
    print("  MEAN   random", _f3(m_rnd), "  zero-shot", _f3(m_zs),
          "  CEM", _f3(m_cem))
    print("  CEM improved", n_better, "of", n_goals, "goals;",
          _f3(100.0 * (m_zs - m_cem) / m_zs) + String("%"), "mean gain")
    print("  ", el, "s")

    # ⚠ THE CONTROL, READ OUT LOUD. If a random z scores like the prompt, the
    # metric cannot tell policies apart and NEITHER column above means
    # anything — that is the failure this run exists to make impossible to
    # miss, not a footnote.
    if m_rnd <= m_zs * 1.05:
        print("  ⚠⚠ THE POSITIVE CONTROL FAILED: a random z scores",
              _f3(m_rnd), "against the prompt's", _f3(m_zs),
              "— this metric cannot separate policies, so the zero-shot and"
              " CEM columns above are NOT evidence of anything. Lengthen"
              " --horizon or pick goals further from the start pose.")
    else:
        print("  control OK: a random z is",
              _f3(m_rnd / m_zs) + String("x"), "the prompt's error")
