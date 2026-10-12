"""Inverse-density state sampling — FB-MEBE's exploration rule, host side.

`noeira-docs/FB_MEBE_G1_PLAN.md` §2 B2; Hu, Armengol Urpí, Cheng, Coros,
"Maximum Entropy Behavior Exploration for Sim2Real Zero-Shot RL" (RLJ 2026,
arXiv 2603.25464), eqs. 10-11:

    s_j ~ (q(φ(s)) + ε)^-β   over states of the replay buffer,  z_j = B(s_j)

so the lanes explore toward the behaviours the buffer holds LEAST of, and the
achieved distribution is pushed toward uniform on the projection φ.

What this file is: the pure arithmetic between "here are φ for some
candidate states and some reference states" and "here are the candidates to
explore toward". No device, no ring, no network. The agent gathers φ from
its ring (`fb/mebe.mojo`), hands it here, and turns the returned indices
into `z = B(s)` — so this half can be gated on the CPU with distributions
whose answer is known (`tests/deep_agents/test_fb_density.mojo`).

## Deviations from the released code, each deliberate

The authors' code (github.com/MATH-286-Pro/FB-MEBE, `density_estimator/`)
fits a 10-coupling RealNVP to the last 10 k accepted states every 1000
updates and weights a 100 k sample buffer by `(exp(log_prob) + eps)^-beta`.

  * DENSITY = a Gaussian KDE, not a flow. φ is 2-4 dimensional here; a KDE
    needs no training, is deterministic given its reference set, and its
    answer on a synthetic mixture is closed-form — so it can be GATED. The
    paper's claim is about the weighting, not about the estimator.
  * φ IS STANDARDISED by the valid references' mean and std before the KDE,
    and `eps` is RELATIVE: `eps = eps_rel · mean(q)` over the valid
    candidates. The release's `eps = 1e-2` sits next to a density in φ's
    RAW units (m/s for base velocity), so its meaning changes with the
    robot's speed range; a relative floor means the same thing on a walker
    and on a G1 (`InverseDensityConfig.eps_rel`).
  * VALIDITY is the caller's mask. The release keeps a state only when the
    base is upright (`g_z <= -0.93`, i.e. tilt <= ~21.5°) and its (vx, vy)
    lie in a fixed box — falls carry large velocities that have nothing to
    do with locomotion, and an inverse-density sampler would chase exactly
    them. Invalid candidates get weight 0; invalid references do not enter
    the KDE.

## The weights, and why β matters

With `w = (q + ε)^-β` and candidates drawn from the buffer (density q), a
state is SELECTED with density `∝ q · (q + ε)^-β`: β = 1 makes the selection
~uniform over the support where q >> ε, β = 2 over-weights the frontier
(the paper's main setting; the release's config runs β = 1). β = 0 is plain
uniform buffer sampling — the paper's `FB-MEBE (β = 0)` ablation, and the
gate's control leg.
"""

from std.math import exp, log, sqrt, pi


# ══════════════════════════════════════════════════════════════════════
# A small deterministic RNG (no global state: the agent owns one)
# ══════════════════════════════════════════════════════════════════════


struct SplitMix64(Copyable, Movable):
    """SplitMix64 — enough for index draws and candidate picks; owned per
    sampler so two samplers never share (or reseed) a global stream."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_u64(mut self) -> UInt64:
        self.state += UInt64(0x9E3779B97F4A7C15)
        var z = self.state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

    def uniform(mut self) -> Float64:
        """[0, 1) with 53 bits."""
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def below(mut self, n: Int) -> Int:
        """Uniform integer in [0, n). n >= 1."""
        var i = Int(self.uniform() * Float64(n))
        if i >= n:
            i = n - 1
        return i


# ══════════════════════════════════════════════════════════════════════
# Configuration and report
# ══════════════════════════════════════════════════════════════════════


@fieldwise_init
struct InverseDensityConfig(Copyable, Movable):
    """`beta`: the inverse-weighting strength (paper 2.0, release 1.0, 0 =
    uniform buffer sampling). `eps_rel`: the density floor as a fraction of
    the mean candidate density — the largest weight is then bounded by
    `(eps_rel · mean q)^-beta`, so a handful of isolated outliers cannot
    take the whole draw. `bw_scale`: multiplies Scott's bandwidth
    `n^(-1/(m+4))` (in standardised units)."""

    var beta: Float64
    var eps_rel: Float64
    var bw_scale: Float64

    @staticmethod
    def default() -> Self:
        return Self(beta=2.0, eps_rel=0.1, bw_scale=1.0)


struct InverseDensityReport(Copyable, Movable):
    """What a draw looked like. `entropy` is the resubstitution estimate
    `-mean_ref log q(ref)` in φ's RAW units (the standardisation's log-det
    added back), comparable across refreshes as the paper's entropy plot.
    `ess` is the effective sample size `(Σw)² / Σw²` over the candidates, as
    a fraction of the valid candidates: 1 = uniform, → 0 = a few states
    take the whole draw."""

    var n_valid_cand: Int
    var n_valid_ref: Int
    var mean_q: Float64
    var eps: Float64
    var entropy: Float64
    var ess_frac: Float64
    var max_w_share: Float64

    def __init__(out self):
        self.n_valid_cand = 0
        self.n_valid_ref = 0
        self.mean_q = 0.0
        self.eps = 0.0
        self.entropy = 0.0
        self.ess_frac = 0.0
        self.max_w_share = 0.0


# ══════════════════════════════════════════════════════════════════════
# The KDE and the draw
# ══════════════════════════════════════════════════════════════════════


def kde_log_density(
    x_std: List[Float64],
    n_x: Int,
    r_std: List[Float64],
    r_valid: List[Bool],
    n_r: Int,
    m: Int,
    h: Float64,
    mut out_log_q: List[Float64],
):
    """`log q(x_i)` for a Gaussian KDE with isotropic bandwidth `h` over the
    VALID reference rows, everything in standardised units. Row-major `[n, m]`
    inputs. A log-sum-exp per row, so a point far from every reference gets a
    very negative log density rather than log(0)."""
    var n_valid = 0
    for j in range(n_r):
        if r_valid[j]:
            n_valid += 1
    var log_norm = (
        -log(Float64(max(n_valid, 1)))
        - Float64(m) * log(h)
        - 0.5 * Float64(m) * log(2.0 * pi)
    )
    var inv_2h2 = 1.0 / (2.0 * h * h)
    for i in range(n_x):
        # pass 1: the smallest squared distance, for the log-sum-exp shift
        var dmin = Float64(1e300)
        for j in range(n_r):
            if not r_valid[j]:
                continue
            var d2 = Float64(0)
            for k in range(m):
                var dv = x_std[i * m + k] - r_std[j * m + k]
                d2 += dv * dv
            if d2 < dmin:
                dmin = d2
        var acc = Float64(0)
        for j in range(n_r):
            if not r_valid[j]:
                continue
            var d2 = Float64(0)
            for k in range(m):
                var dv = x_std[i * m + k] - r_std[j * m + k]
                d2 += dv * dv
            acc += exp(-(d2 - dmin) * inv_2h2)
        if n_valid == 0:
            out_log_q[i] = -1e300
        else:
            out_log_q[i] = log_norm - dmin * inv_2h2 + log(acc)


def inverse_density_draw(
    cand: List[Float64],
    cand_valid: List[Bool],
    n_cand: Int,
    refs: List[Float64],
    ref_valid: List[Bool],
    n_ref: Int,
    m: Int,
    k_draw: Int,
    cfg: InverseDensityConfig,
    mut rng: SplitMix64,
    mut out_idx: List[Int],
    mut report: InverseDensityReport,
) raises:
    """Draw `k_draw` candidate indices with probability `∝ (q + ε)^-β` over
    the VALID candidates, with replacement (the release's
    `torch.multinomial(..., replacement=True)`).

    `cand` `[n_cand, m]` and `refs` `[n_ref, m]` are φ in RAW units; both are
    standardised here by the valid references' mean and std (a constant
    dimension keeps std 1 so it drops out instead of dividing by zero).

    RAISES when no candidate is valid: an exploration step with nothing to
    explore toward is a configuration error (a validity box that excludes
    the whole buffer), not something to paper over with a uniform draw.
    """
    if m < 1:
        raise Error("inverse_density_draw: m must be >= 1")
    if len(cand) < n_cand * m or len(refs) < n_ref * m:
        raise Error("inverse_density_draw: φ arrays shorter than n * m")
    # ── standardisation from the valid references ───────────────────
    var mean = List[Float64](length=m, fill=0.0)
    var std = List[Float64](length=m, fill=1.0)
    var nr = 0
    for j in range(n_ref):
        if ref_valid[j]:
            nr += 1
            for k in range(m):
                mean[k] += refs[j * m + k]
    report.n_valid_ref = nr
    if nr < 2:
        raise Error(
            "inverse_density_draw: fewer than 2 valid reference states — the"
            " validity box excludes (almost) the whole buffer"
        )
    for k in range(m):
        mean[k] /= Float64(nr)
    var var_ = List[Float64](length=m, fill=0.0)
    for j in range(n_ref):
        if ref_valid[j]:
            for k in range(m):
                var dv = refs[j * m + k] - mean[k]
                var_[k] += dv * dv
    var log_det = Float64(0)
    for k in range(m):
        var s = sqrt(var_[k] / Float64(nr - 1))
        if s > 1e-12:
            std[k] = s
            log_det += log(s)
    var c_std = List[Float64](length=n_cand * m, fill=0.0)
    for i in range(n_cand):
        for k in range(m):
            c_std[i * m + k] = (cand[i * m + k] - mean[k]) / std[k]
    var r_std = List[Float64](length=n_ref * m, fill=0.0)
    for j in range(n_ref):
        for k in range(m):
            r_std[j * m + k] = (refs[j * m + k] - mean[k]) / std[k]

    # ── KDE (Scott's rule in standardised units) ────────────────────
    var h = cfg.bw_scale * (Float64(nr) ** (-1.0 / Float64(m + 4)))
    var lq_c = List[Float64](length=n_cand, fill=0.0)
    kde_log_density(c_std, n_cand, r_std, ref_valid, n_ref, m, h, lq_c)
    # resubstitution entropy over the references (raw units: + log det)
    var lq_r = List[Float64](length=n_ref, fill=0.0)
    kde_log_density(r_std, n_ref, r_std, ref_valid, n_ref, m, h, lq_r)
    var hsum = Float64(0)
    for j in range(n_ref):
        if ref_valid[j]:
            hsum += -lq_r[j]
    report.entropy = hsum / Float64(nr) + log_det

    # ── weights ──────────────────────────────────────────────────────
    var nc = 0
    var qsum = Float64(0)
    for i in range(n_cand):
        if cand_valid[i]:
            nc += 1
            qsum += exp(lq_c[i])
    report.n_valid_cand = nc
    if nc == 0:
        raise Error(
            "inverse_density_draw: no valid candidate — the validity box"
            " excludes every sampled state"
        )
    var mean_q = qsum / Float64(nc)
    var eps = cfg.eps_rel * mean_q
    report.mean_q = mean_q
    report.eps = eps
    var w = List[Float64](length=n_cand, fill=0.0)
    var wsum = Float64(0)
    var w2sum = Float64(0)
    var wmax = Float64(0)
    for i in range(n_cand):
        if not cand_valid[i]:
            continue
        var wi: Float64
        if cfg.beta == 0.0:
            wi = 1.0
        else:
            # (q + eps)^-beta in log space: q may underflow, eps keeps it finite
            wi = exp(-cfg.beta * log(exp(lq_c[i]) + eps))
        w[i] = wi
        wsum += wi
        w2sum += wi * wi
        if wi > wmax:
            wmax = wi
    report.ess_frac = (wsum * wsum / w2sum) / Float64(nc)
    report.max_w_share = wmax / wsum

    # ── the draw: inverse CDF over the cumulative weights ────────────
    var cdf = List[Float64](length=n_cand, fill=0.0)
    var run = Float64(0)
    for i in range(n_cand):
        run += w[i]
        cdf[i] = run
    if len(out_idx) < k_draw:
        out_idx.resize(k_draw, 0)
    for t in range(k_draw):
        var u = rng.uniform() * run
        # first i with cdf[i] > u (binary search); zero-weight rows have
        # cdf equal to their predecessor's, so they are never returned
        var lo = 0
        var hi = n_cand - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cdf[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        out_idx[t] = lo
