""""Optimise the PROMPT SEQUENCE for tracking — §12.46's next rung.

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/bfm_zero_traj_opt.mojo -o /tmp/g1traj
    pixi run /tmp/g1traj --ckpt runs/<id>/checkpoints/step_36000.ckpt
    pixi run /tmp/g1traj --ckpt <path> --clips 0 12 13 --knots 9 --iters 8 --pop 16

## The question

§12.46 showed CEM over a SINGLE `z` buys 37.6 % on pose reaching with the
network frozen. BFM-Zero's other adaptation mode optimises the prompt
SEQUENCE `z_{t:t+H-1}` instead, and the paper reports +29.1 % on a leaping
motion. That one lands on TRACKING, which is the number this whole track is
judged by: 1.191 mean EMD against the reference's ~1.0 (§12.45).

If latent search closes part of that with no retraining, then the residual is
not "our policy is worse" but "our prompt is worse" — a different problem with
a much cheaper fix. That is worth knowing before anyone spends another GPU
week on the policy.

## ⚠ KNOTS, NOT 499 INDEPENDENT VECTORS

A segment is `G1_SEG_ROWS = 499` steps. Optimising a `z` per step is
499 x 256 = 127 744 parameters against one scalar objective, which CEM cannot
search at any population this side of a cluster. So the free parameters are
`--knots` prompts spread over the segment, **slerped** along the sphere
between them (`g1_slerp_z`).

That is a modelling choice and it cuts both ways. It matches how the prompt
actually varies — `B` over adjacent rows of a smooth 50 Hz motion is nearly
parallel, which §12.45 measured — so the coarse sequence can represent the
baseline well. But it cannot express a prompt that must change abruptly, and a
motion with a real discontinuity (a footfall, a catch) is exactly where the
dense version would win. A null here is therefore a null about KNOTS, not
about trajectory optimisation, and the doc must say so.

## ⚠ THE BASELINE IS THE BEST PROMPT WE HAVE, NOT THE DEFAULT

`--z-horizon` defaults to 8 here, not to 1. §12.45 measured H=8 as the better
prompt (1.1910 vs 1.1991), and starting the search from H=1 would let it
re-earn a gain we already know about and bank it twice.

## ⚠ THE CONTROL

The mean score of the FIRST CEM generation is reported beside the result. Those
are perturbations of the baseline that have been scored but not selected, so if
the "optimised" column is not clearly better than that mean, the search found
noise. A perturbation that improves on the baseline by luck is expected; a
whole population that does is a broken objective.
"""

from std.math import sqrt, abs, cos as _cos64, log as _log64
from std.random import random_float64, seed
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_SEG_ROWS, G1_D, G1_H, G1_L, G1_HB,
    g1_n_segments, g1_segment_row, g1_segment_pick, g1_score_segment,
    g1_build_prompt, g1_project_z, g1_slerp_z,
)

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime T: Int = G1_SEG_ROWS

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


def _flag_ints(name: String) raises -> List[Int]:
    var out = List[Int]()
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            var j = i + 1
            while j < len(av) and not String(av[j]).startswith("--"):
                out.append(atol(String(av[j])))
                j += 1
            break
    return out^


def _f4(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 10000.0 + 0.5)
    var ip = h // 10000
    var fp = h % 10000
    var f = String(fp)
    while f.byte_length() < 4:
        f = String("0") + f
    var body = String(ip) + String(".") + f
    return (String("-") + body) if neg else body


def _gauss() -> Float64:
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * _log64(u1)) * _cos64(6.283185307179586 * random_float64())


def _expand(
    ref knots: Tensor, kbase: Int, n_knots: Int, mut z_seg: Tensor
):
    """Slerp `n_knots` prompts out to the segment's `T` steps.

    Knot `i` sits at step `round(i * (T-1) / (n_knots-1))`, so the first and
    last knots are pinned to the segment's ends and every step between two
    knots is a geodesic blend of them.
    """
    var span = Float64(T - 1) / Float64(n_knots - 1)
    for j in range(T):
        var f = Float64(j) / span
        var i0 = Int(f)
        if i0 > n_knots - 2:
            i0 = n_knots - 2
        var u = f - Float64(i0)
        if u < 0.0:
            u = 0.0
        if u > 1.0:
            u = 1.0
        g1_slerp_z[D](
            knots, kbase + i0, kbase + i0 + 1, u, z_seg, j
        )


def main() raises:
    seed(23)
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var clips = _flag_ints(String("--clips"))
    var n_seg = atol(_flag(String("--segments"), String(1)))
    var n_knots = atol(_flag(String("--knots"), String(9)))
    var iters = atol(_flag(String("--iters"), String(8)))
    var pop = atol(_flag(String("--pop"), String(16)))
    var elites = atol(_flag(String("--elites"), String(4)))
    var sigma0 = Float64(String(_flag(String("--sigma"), String("0.25"))))
    var zh = atol(_flag(String("--z-horizon"), String(8)))
    # ⚠ MEASURE THE PARAMETERISATION BEFORE SEARCHING IT. The first smoke
    # ran CEM at 5 knots and the re-described prompt scored 1.9813 against
    # the derived 1.4221 — a 39 % LOSS before the search had sampled
    # anything, so no amount of CEM could have shown a gain. This mode skips
    # the search and prints the re-description cost alone, which is the
    # question "how fast does the prompt actually have to vary?".
    var sweep = _flag_ints(String("--knot-sweep"))
    # ⚠ SMOOTH PERTURBATIONS, and the knot sweep is why. Independent noise
    # per knot destroys exactly the property the sweep showed is doing the
    # work: at 65 knots the SMOOTHED prompt already beats the derived one by
    # 1.5 %, so a search that jitters each knot on its own is sampling away
    # from the thing that helps — and it shows, the 5-knot smoke's first
    # generation averaged 1.96 against a 1.42 baseline.
    #
    # A candidate is therefore a LOW-FREQUENCY random field along the knot
    # index: `--modes` random directions in R^D, each modulated by a cosine
    # in the knot index. That keeps every sample smooth AND cuts the search
    # space from n_knots*D to modes*D — 65*256 = 16 640 down to 1 536 at the
    # default, which is the difference between a search and a lottery.
    # `--modes 0` restores independent white noise so the two are comparable.
    var modes = atol(_flag(String("--modes"), String(6)))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")
    if len(clips) == 0:
        clips.append(0)
        clips.append(12)
        clips.append(13)

    print("=" * 76)
    print("BFM-Zero G1 — TRAJECTORY OPTIMISATION over the prompt sequence")
    print("=" * 76)
    print("  knots", n_knots, "slerped over", T, "steps  ·  CEM", iters,
          "x", pop, ",", elites, "elites, sigma0", sigma0)
    print("  perturbation:",
          "white noise per knot" if modes <= 0 else
          (String("smooth, ") + String(modes) + String(" cosine modes")))
    print("  baseline prompt: z_horizon", zh, "(§12.45's best, NOT the default 1)")
    print("  THE NETWORK IS FROZEN — this searches the prompt only")

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

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var b_in = Tensor.alloc(T * OBS)
    var b_out = Tensor()
    var z_seg = Tensor.alloc(T * D)
    var ach = List[Float64](length=T * ACT, fill=0.0)
    var tgt = List[Float64](length=T * ACT, fill=0.0)
    var base_knots = Tensor.alloc(n_knots * D)
    var cand = Tensor.alloc(pop * n_knots * D)
    var mean = Tensor.alloc(n_knots * D)
    var gdir = Tensor.alloc((modes if modes > 0 else 1) * D)
    var score = List[Float64](length=pop, fill=0.0)
    var order = List[Int](length=pop, fill=0)

    var sum_b = 0.0
    var sum_o = 0.0
    var sum_g0 = 0.0
    var n = 0
    var n_better = 0
    var t0 = perf_counter_ns()

    # ── knot-density sweep: re-description cost, no search ────────────
    if len(sweep) > 0:
        print("-" * 76)
        print("  A knot expansion can only HURT: it re-describes a prompt the")
        print("  derived sequence already gives exactly. This is the floor any")
        print("  search at that knot count starts from.")
        print("-" * 76)
        print("  clip seg  derived     knots ->  score      loss")
        for ci in range(len(clips)):
            var clip = clips[ci]
            var n_avail = g1_n_segments(Int(rsi.ep_len.data[clip]))
            var take = n_avail if n_avail < n_seg else n_seg
            for k in range(take):
                var seg = g1_segment_pick(n_avail, take, k)
                var r0 = g1_segment_row(Int(rsi.ep_offset.data[clip]), seg)
                var sc_d = g1_score_segment[
                    FNet, BNet, ANet, OBS, ACT, D, BATCH
                ](
                    t, env, rsi, st, pv, qpos_col, norm, r0,
                    ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
                    z_horizon=zh,
                )
                # ⚠ the derived prompt must be REBUILT per knot count: the
                # expansion overwrites `z_seg` in place, so sampling knots off
                # a previously expanded sequence would compound the loss and
                # make every row after the first look worse than it is.
                for si in range(len(sweep)):
                    var nk = sweep[si]
                    if nk < 2 or nk > T:
                        continue
                    g1_build_prompt[BNet, FNet, ANet, OBS, ACT, D, BATCH](
                        t, st, pv, norm, r0, b_in, b_out, z_seg, zh
                    )
                    var kt = Tensor.alloc(nk * D)
                    var sp = Float64(T - 1) / Float64(nk - 1)
                    for i in range(nk):
                        var j = Int(Float64(i) * sp + 0.5)
                        if j > T - 1:
                            j = T - 1
                        for d in range(D):
                            kt.data[i * D + d] = z_seg.data[j * D + d]
                        g1_project_z[D](kt, i)
                    _expand(kt, 0, nk, z_seg)
                    var sc_e = g1_score_segment[
                        FNet, BNet, ANet, OBS, ACT, D, BATCH
                    ](
                        t, env, rsi, st, pv, qpos_col, norm, r0,
                        ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
                        z_given=True,
                    )
                    print("  ", clip, " ", seg, " ", _f4(sc_d.emd),
                          "   ", nk, " -> ", _f4(sc_e.emd), "  ",
                          _f4(100.0 * (sc_e.emd - sc_d.emd) / sc_d.emd)
                          + String("%"))
        print("-" * 76)
        return

    print("-" * 76)
    print("  clip seg   baseline    gen-0 mean    optimised     gain")
    for ci in range(len(clips)):
        var clip = clips[ci]
        var n_avail = g1_n_segments(Int(rsi.ep_len.data[clip]))
        var take = n_avail if n_avail < n_seg else n_seg
        for k in range(take):
            var seg = g1_segment_pick(n_avail, take, k)
            var r0 = g1_segment_row(Int(rsi.ep_offset.data[clip]), seg)

            # ── baseline: the derived prompt at z_horizon ──────────────
            var sc_b = g1_score_segment[FNet, BNet, ANet, OBS, ACT, D, BATCH](
                t, env, rsi, st, pv, qpos_col, norm, r0,
                ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
                z_horizon=zh,
            )
            # `z_seg` now holds the baseline prompt; sample the knots off it so
            # the search STARTS from the best prompt we have rather than from
            # noise, which is what makes the gain attributable to the search.
            var span = Float64(T - 1) / Float64(n_knots - 1)
            for i in range(n_knots):
                var j = Int(Float64(i) * span + 0.5)
                if j > T - 1:
                    j = T - 1
                for d in range(D):
                    base_knots.data[i * D + d] = z_seg.data[j * D + d]
                g1_project_z[D](base_knots, i)

            # ⚠ THE KNOT BASELINE IS NOT THE DERIVED BASELINE. Expanding the
            # sampled knots back out is a LOSSY re-description of the prompt,
            # so `sc_b` above is not the score the search starts from. Score
            # the expansion too, or a "gain" could be the search merely
            # climbing back to where the derived prompt already was.
            _expand(base_knots, 0, n_knots, z_seg)
            var sc_k = g1_score_segment[FNet, BNet, ANet, OBS, ACT, D, BATCH](
                t, env, rsi, st, pv, qpos_col, norm, r0,
                ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
                z_given=True,
            )

            for i in range(n_knots * D):
                mean.data[i] = base_knots.data[i]
            var best = sc_k.emd if sc_k.emd < sc_b.emd else sc_b.emd
            var sigma = sigma0
            var gen0 = 0.0

            for it in range(iters):
                for c in range(pop):
                    var kb = c * n_knots
                    if modes <= 0:
                        for i in range(n_knots):
                            for d in range(D):
                                cand.data[(kb + i) * D + d] = Scalar[DT](
                                    Float64(mean.data[i * D + d])
                                    + sigma * _gauss()
                                )
                            g1_project_z[D](cand, kb + i)
                    else:
                        # one draw of `modes` directions, shared by every knot
                        # and modulated along the sequence
                        for m in range(modes):
                            for d in range(D):
                                gdir.data[m * D + d] = Scalar[DT](_gauss())
                        var inv = 1.0 / sqrt(Float64(modes))
                        for i in range(n_knots):
                            var u = Float64(i) / Float64(n_knots - 1)
                            for d in range(D):
                                var acc = 0.0
                                for m in range(modes):
                                    acc += (
                                        _cos64(3.141592653589793
                                               * Float64(m) * u)
                                        * Float64(gdir.data[m * D + d])
                                    )
                                cand.data[(kb + i) * D + d] = Scalar[DT](
                                    Float64(mean.data[i * D + d])
                                    + sigma * acc * inv
                                )
                            g1_project_z[D](cand, kb + i)
                    _expand(cand, kb, n_knots, z_seg)
                    var sc = g1_score_segment[
                        FNet, BNet, ANet, OBS, ACT, D, BATCH
                    ](
                        t, env, rsi, st, pv, qpos_col, norm, r0,
                        ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
                        z_given=True,
                    )
                    score[c] = sc.emd
                    if sc.emd < best:
                        best = sc.emd
                if it == 0:
                    for c in range(pop):
                        gen0 += score[c]
                    gen0 /= Float64(pop)
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
                for i in range(n_knots):
                    for d in range(D):
                        var acc = 0.0
                        for e in range(elites):
                            acc += Float64(
                                cand.data[
                                    (order[e] * n_knots + i) * D + d
                                ]
                            )
                        mean.data[i * D + d] = Scalar[DT](
                            acc / Float64(elites)
                        )
                    g1_project_z[D](mean, i)
                sigma *= 0.85

            sum_b += sc_b.emd
            sum_o += best
            sum_g0 += gen0
            n += 1
            if best < sc_b.emd - 1e-9:
                n_better += 1
            print("  ", clip, " ", seg, "  ", _f4(sc_b.emd),
                  "  (knots ", _f4(sc_k.emd), ") ", _f4(gen0), "  ",
                  _f4(best), "  ",
                  _f4(100.0 * (sc_b.emd - best) / sc_b.emd) + String("%"))

    var el = Float64(perf_counter_ns() - t0) * 1e-9
    var mb = sum_b / Float64(n)
    var mo = sum_o / Float64(n)
    var mg = sum_g0 / Float64(n)
    print("-" * 76)
    print("  MEAN  baseline", _f4(mb), "  gen-0", _f4(mg),
          "  optimised", _f4(mo))
    print("  improved", n_better, "of", n, "segments;",
          _f4(100.0 * (mb - mo) / mb) + String("%"), "mean gain")
    print("  ", el, "s")
    if mo >= mg:
        print("  ⚠⚠ THE SEARCH FOUND NOTHING: the optimised score is no better"
              " than the mean of the FIRST generation, which is unselected"
              " perturbations of the baseline. Either the objective is flat in"
              " the knots or sigma is far off.")
    else:
        print("  control OK: the search beat its own first generation by",
              _f4(100.0 * (mg - mo) / mg) + String("%"))
