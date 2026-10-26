"""`fb/density.mojo` — FB-MEBE's inverse-density draw, on distributions whose
answer is known. CPU only, no device. `noeira-docs/FB_MEBE_G1_PLAN.md` §2 B2.

  [1] the KDE is a density: it integrates to 1 on a grid in 1-D and 2-D
      (noise-free — a missing normaliser or a wrong h^m power fails), and on
      20 000 draws of N(0, 1) its log density at 0, ±1 is within 0.06 of its
      expectation N(0, 1 + h²).
  [2] β moves the draw the way eq. 10 says, on a 90/10 mixture of two
      clusters: β = 0 picks the minor cluster at its BUFFER share (~0.10,
      the ablation leg), β = 1 lifts it well above that (~uniform over the
      support), β = 2 lifts it further (the frontier over-weight).
  [3] validity: an invalid candidate is NEVER returned, and invalid
      references do not enter the KDE (a far cloud of invalid refs leaves
      the density at its location as low as if it were absent).
  [4] scale invariance: φ × 10 gives the SAME draws from the same seed and
      an entropy larger by exactly m·log 10 — the leg that fails if the
      standardisation is removed (the bandwidth would then live in raw units).
  [5] entropy of U[0,1]^2 is 0; the resubstitution estimate lands within
      0.2 (boundary bias makes it read slightly high).
  [6] no valid candidate RAISES rather than drawing from nothing.

Run: pixi run mojo run -I . tests/deep_agents/test_fb_density.mojo
"""

from std.math import abs, exp, log, sqrt, pi, cos
from std.testing import assert_true

from noeira.deep_agents.fb.density import (
    SplitMix64,
    InverseDensityConfig,
    InverseDensityReport,
    inverse_density_draw,
    kde_log_density,
)


def _gauss(mut rng: SplitMix64) -> Float64:
    """Box-Muller, one value."""
    var u1 = rng.uniform()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = rng.uniform()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _two_clusters(n: Int, mut rng: SplitMix64) -> List[Float64]:
    """1-D: 90 % N(0, 1), 10 % N(6, 0.5), in that order (first 0.9 n major)."""
    var x = List[Float64](length=n, fill=0.0)
    var n_major = (n * 9) // 10
    for i in range(n):
        if i < n_major:
            x[i] = _gauss(rng)
        else:
            x[i] = 6.0 + 0.5 * _gauss(rng)
    return x^


def _minor_share(ref x: List[Float64], ref idx: List[Int]) -> Float64:
    var c = 0
    for t in range(len(idx)):
        if x[idx[t]] > 3.0:
            c += 1
    return Float64(c) / Float64(len(idx))


def test_kde_is_a_density() raises:
    print("[1] KDE: integrates to 1 (m = 1 and m = 2), and matches N(0,1+h²) ...")
    var rng = SplitMix64(UInt64(11))
    comptime N = 20000
    var r = List[Float64](length=N, fill=0.0)
    for i in range(N):
        r[i] = _gauss(rng)
    var valid = List[Bool](length=N, fill=True)
    var h = Float64(N) ** (-1.0 / 5.0)
    # (a) ∫ q dx = 1 on a fine grid — noise-free: a missing normaliser or a
    # wrong bandwidth power fails it whatever the sample
    comptime NG = 1600
    var grid = List[Float64](length=NG, fill=0.0)
    for g in range(NG):
        grid[g] = -8.0 + 0.01 * (Float64(g) + 0.5)
    var lg = List[Float64](length=NG, fill=0.0)
    kde_log_density(grid, NG, r, valid, N, 1, h, lg)
    var integral = Float64(0)
    for g in range(NG):
        integral += exp(lg[g]) * 0.01
    print("      m=1  ∫q =", integral)
    assert_true(abs(integral - 1.0) < 1e-3, "1-D KDE does not integrate to 1")
    # the same in 2-D (the h^m power)
    comptime N2 = 1000
    var r2 = List[Float64](length=N2 * 2, fill=0.0)
    for i in range(N2 * 2):
        r2[i] = _gauss(rng)
    var v2 = List[Bool](length=N2, fill=True)
    comptime NG2 = 160
    var grid2 = List[Float64](length=NG2 * NG2 * 2, fill=0.0)
    for a in range(NG2):
        for b in range(NG2):
            grid2[(a * NG2 + b) * 2] = -8.0 + 0.1 * (Float64(a) + 0.5)
            grid2[(a * NG2 + b) * 2 + 1] = -8.0 + 0.1 * (Float64(b) + 0.5)
    var lg2 = List[Float64](length=NG2 * NG2, fill=0.0)
    var h2 = Float64(N2) ** (-1.0 / 6.0)
    kde_log_density(grid2, NG2 * NG2, r2, v2, N2, 2, h2, lg2)
    var integral2 = Float64(0)
    for g in range(NG2 * NG2):
        integral2 += exp(lg2[g]) * 0.01
    print("      m=2  ∫q =", integral2)
    assert_true(abs(integral2 - 1.0) < 2e-3, "2-D KDE does not integrate to 1")
    # (b) the shape in the bulk: its expectation is N(0, 1 + h²)
    var xs: List[Float64] = [0.0, 1.0, -1.0]
    var lq = List[Float64](length=3, fill=0.0)
    kde_log_density(xs, 3, r, valid, N, 1, h, lq)
    var v = 1.0 + h * h
    for i in range(3):
        var truth = -0.5 * xs[i] * xs[i] / v - 0.5 * log(2.0 * pi * v)
        print("      x", xs[i], " kde", lq[i], " analytic N(0,1+h²)", truth)
        assert_true(abs(lq[i] - truth) < 0.06, "KDE log density off by > 0.06")


def test_beta_moves_the_draw() raises:
    print("[2] β on a 90/10 two-cluster buffer ...")
    var rng = SplitMix64(UInt64(22))
    comptime NC = 4000
    comptime NR = 2000
    var cand = _two_clusters(NC, rng)
    var refs = _two_clusters(NR, rng)
    var cv = List[Bool](length=NC, fill=True)
    var rv = List[Bool](length=NR, fill=True)
    var shares = List[Float64]()
    var betas: List[Float64] = [0.0, 1.0, 2.0]
    for b in range(3):
        var cfg = InverseDensityConfig(beta=betas[b], eps_rel=0.01, bw_scale=1.0)
        var draw_rng = SplitMix64(UInt64(33))
        var idx = List[Int]()
        var rep = InverseDensityReport()
        inverse_density_draw(
            cand, cv, NC, refs, rv, NR, 1, 8000, cfg, draw_rng, idx, rep
        )
        var s = _minor_share(cand, idx)
        shares.append(s)
        print("      β", betas[b], " minor-cluster share", s,
              " ess", rep.ess_frac, " max share", rep.max_w_share)
    assert_true(abs(shares[0] - 0.10) < 0.02, "β=0 is not uniform buffer sampling")
    # support widths ~ 2·(3σ): major ~6, minor ~3 → ~1/3 under β = 1
    assert_true(shares[1] > 0.25, "β=1 did not lift the rare cluster")
    assert_true(shares[2] > shares[1] + 0.02, "β=2 not above β=1")


def test_validity() raises:
    print("[3] invalid candidates never drawn; invalid refs not in the KDE ...")
    var rng = SplitMix64(UInt64(44))
    comptime NC = 1000
    comptime NR = 1000
    var cand = _two_clusters(NC, rng)
    var refs = _two_clusters(NR, rng)
    var cv = List[Bool](length=NC, fill=True)
    # every minor-cluster candidate invalid: with β = 2 they would dominate
    for i in range(NC):
        if cand[i] > 3.0:
            cv[i] = False
    var rv = List[Bool](length=NR, fill=True)
    var cfg = InverseDensityConfig(beta=2.0, eps_rel=0.01, bw_scale=1.0)
    var drng = SplitMix64(UInt64(55))
    var idx = List[Int]()
    var rep = InverseDensityReport()
    inverse_density_draw(cand, cv, NC, refs, rv, NR, 1, 5000, cfg, drng, idx, rep)
    var bad = 0
    for t in range(len(idx)):
        if not cv[idx[t]]:
            bad += 1
    print("      invalid picks", bad, "/", len(idx), " valid cands", rep.n_valid_cand)
    assert_true(bad == 0, "an invalid candidate was drawn")

    # invalid refs: add a far cloud at x=50, flagged invalid; the density
    # there must stay as low as without it
    var refs2 = refs.copy()
    var rv2 = rv.copy()
    for _ in range(500):
        refs2.append(50.0 + 0.1 * _gauss(rng))
        rv2.append(False)
    var probe: List[Float64] = [50.0]
    var lq_a = List[Float64](length=1, fill=0.0)
    var lq_b = List[Float64](length=1, fill=0.0)
    kde_log_density(probe, 1, refs, rv, NR, 1, 0.3, lq_a)
    kde_log_density(probe, 1, refs2, rv2, NR + 500, 1, 0.3, lq_b)
    print("      log q(50): without the cloud", lq_a[0], " with it (invalid)", lq_b[0])
    assert_true(abs(lq_a[0] - lq_b[0]) < 1e-9, "an invalid reference entered the KDE")


def test_scale_invariance() raises:
    print("[4] φ x 10: same draws, entropy + m·log 10 ...")
    var rng = SplitMix64(UInt64(66))
    comptime NC = 1500
    comptime NR = 800
    comptime M = 2
    var cand = List[Float64](length=NC * M, fill=0.0)
    var refs = List[Float64](length=NR * M, fill=0.0)
    for i in range(NC * M):
        cand[i] = _gauss(rng) * (1.0 if i % 2 == 0 else 3.0)
    for i in range(NR * M):
        refs[i] = _gauss(rng) * (1.0 if i % 2 == 0 else 3.0)
    var cand10 = List[Float64](length=NC * M, fill=0.0)
    var refs10 = List[Float64](length=NR * M, fill=0.0)
    for i in range(NC * M):
        cand10[i] = cand[i] * 10.0
    for i in range(NR * M):
        refs10[i] = refs[i] * 10.0
    var cv = List[Bool](length=NC, fill=True)
    var rv = List[Bool](length=NR, fill=True)
    var cfg = InverseDensityConfig.default()
    var r1 = SplitMix64(UInt64(77))
    var r2 = SplitMix64(UInt64(77))
    var i1 = List[Int]()
    var i2 = List[Int]()
    var p1 = InverseDensityReport()
    var p2 = InverseDensityReport()
    inverse_density_draw(cand, cv, NC, refs, rv, NR, M, 2000, cfg, r1, i1, p1)
    inverse_density_draw(cand10, cv, NC, refs10, rv, NR, M, 2000, cfg, r2, i2, p2)
    var diff = 0
    for t in range(2000):
        if i1[t] != i2[t]:
            diff += 1
    var dh = p2.entropy - p1.entropy
    print("      differing draws", diff, "  ΔH", dh, " expected", Float64(M) * log(10.0))
    assert_true(diff == 0, "scaling φ changed the draw — standardisation missing")
    assert_true(abs(dh - Float64(M) * log(10.0)) < 1e-6, "entropy not shift-by-log-det")


def test_uniform_entropy() raises:
    print("[5] entropy of U[0,1]^2 ~ 0 ...")
    var rng = SplitMix64(UInt64(88))
    comptime N = 3000
    var r = List[Float64](length=N * 2, fill=0.0)
    for i in range(N * 2):
        r[i] = rng.uniform()
    var v = List[Bool](length=N, fill=True)
    var cfg = InverseDensityConfig.default()
    var idx = List[Int]()
    var rep = InverseDensityReport()
    var drng = SplitMix64(UInt64(99))
    inverse_density_draw(r, v, N, r, v, N, 2, 10, cfg, drng, idx, rep)
    print("      H", rep.entropy)
    assert_true(abs(rep.entropy) < 0.2, "uniform-square entropy far from 0")


def test_no_valid_raises() raises:
    print("[6] no valid candidate raises ...")
    var c: List[Float64] = [0.0, 1.0, 2.0]
    var cv = List[Bool](length=3, fill=False)
    var rv = List[Bool](length=3, fill=True)
    var cfg = InverseDensityConfig.default()
    var rng = SplitMix64(UInt64(1))
    var idx = List[Int]()
    var rep = InverseDensityReport()
    var raised = False
    try:
        inverse_density_draw(c, cv, 3, c, rv, 3, 1, 4, cfg, rng, idx, rep)
    except:
        raised = True
    assert_true(raised, "an all-invalid candidate set did not raise")


def main() raises:
    test_kde_is_a_density()
    test_beta_moves_the_draw()
    test_validity()
    test_scale_invariance()
    test_uniform_entropy()
    test_no_valid_raises()
    print("ALL PASS — fb density")
