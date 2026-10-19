"""Statistical quality tests for PhiloxRandom (CPU + GPU kernel).

Motivated by EZ-V2 Pendulum convergence slowing down after
`noeira/envs/pendulum/pendulum_v2.mojo` switched its CPU reset RNG from
the global `std.random.random_float64` to per-env `PhiloxRandom`. Goal:
rule out PhiloxRandom itself as the cause.

Hypotheses tested:
  H1 — PhiloxRandom is weak in isolation (uniformity, moments, autocorr,
       bit balance).
  H2 — Near-by seeds produce correlated streams (the consecutive-seed
       concern when each env picks `seed=base+env_id`).
  H3 — Per-env independent streams shift the joint pendulum-reset
       distribution vs. the single global stream of `random_float64`.
  H4 — The GPU seed formula `step*2654435761 + env*12345` clusters seeds.

All checks have ≈5σ thresholds (per-check false-positive ≈ 1e-6).
A failure means the RNG is genuinely suspect, not statistical noise.

Run:
    pixi run -e apple  mojo run -I . tests/test_philox_random_quality.mojo
    pixi run -e nvidia mojo run -I . tests/test_philox_random_quality.mojo
"""

from std.math import sqrt, cos, sin, pi
from std.random.philox import Random as PhiloxRandom
from std.random import random_float64, seed as global_seed
from noeira.core.tiles import Tile, tile
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics2d import dtype, TPB


# ============================================================================
# Configuration
# ============================================================================

comptime N_UNIFORMITY: Int = 200_000
comptime N_BINS_1D: Int = 200

comptime N_BITS_TESTED: Int = 22  # lower 22 bits of u*2^24 — drops the
                                  # two LSBs which can quantize under fp32

comptime N_AUTOCORR: Int = 200_000

comptime N_NEARBY_SEEDS: Int = 50_000
comptime N_NEARBY_BINS: Int = 200

comptime PENDULUM_N_ENVS: Int = 32
comptime PENDULUM_N_RESETS_PER_ENV: Int = 500
comptime PENDULUM_GRID: Int = 20

comptime GPU_BATCH: Int = 8192
comptime GPU_BINS: Int = 100
comptime GPU_STEPS: Int = 64   # simulated training steps for seed-formula sweep

# χ² thresholds at ~5σ in Cornish–Fisher (df, threshold)
#   df=199 → 5σ ≈ 199 + 4.75·√398 ≈ 294 ; use 320 for headroom
#   df=99  → 5σ ≈ 99  + 4.75·√198 ≈ 166 ; use 190
#   df=399 → 5σ ≈ 399 + 4.75·√798 ≈ 533 ; use 580
comptime CHI2_TH_200: Float64 = 320.0
comptime CHI2_TH_100: Float64 = 190.0
comptime CHI2_TH_400: Float64 = 580.0

# Moment tolerances at ~5σ for N=200k samples of U[0,1]
#   SE(mean) ≈ √((1/12)/N) ≈ 6.45e-4 → 5σ ≈ 3.2e-3
#   SE(var)  ≈ √(2/N)·var ≈ 2.6e-4   → 5σ ≈ 1.3e-3
comptime MEAN_TOL: Float64 = 5e-3
comptime VAR_TOL: Float64 = 2e-3

# Autocorrelation: SE = 1/√N ≈ 2.24e-3 → 5σ ≈ 1.1e-2
comptime AUTOCORR_TOL: Float64 = 1.5e-2

# Bit-balance: SE = √(0.25/N) ≈ 1.12e-3 → 5σ ≈ 5.6e-3
comptime BIT_BALANCE_TOL: Float64 = 8e-3


# ============================================================================
# Statistical helpers
# ============================================================================

def chi2_uniform(counts: List[Int], n_total: Int) -> Float64:
    var k = len(counts)
    var expected = Float64(n_total) / Float64(k)
    var stat = Float64(0.0)
    for i in range(k):
        var d = Float64(counts[i]) - expected
        stat += d * d / expected
    return stat


def sample_mean(samples: List[Float64]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(samples)):
        s += samples[i]
    return s / Float64(len(samples))


def sample_var(samples: List[Float64], m: Float64) -> Float64:
    var s = Float64(0.0)
    for i in range(len(samples)):
        var d = samples[i] - m
        s += d * d
    return s / Float64(len(samples) - 1)


def lag_k_autocorr(samples: List[Float64], k: Int) -> Float64:
    var n = len(samples)
    var m = sample_mean(samples)
    var num = Float64(0.0)
    var den = Float64(0.0)
    for i in range(n - k):
        num += (samples[i] - m) * (samples[i + k] - m)
    for i in range(n):
        var d = samples[i] - m
        den += d * d
    if den == 0.0:
        return Float64(0.0)
    return num / den


def abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def banner(name: String):
    print("")
    print("======================================================================")
    print("  " + name)
    print("======================================================================")


def check_lt(label: String, value: Float64, threshold: Float64) raises:
    var passed = value < threshold
    var tag = "PASS" if passed else "FAIL"
    print("  [" + tag + "] " + label + " =", value, "  (threshold <", threshold, ")")
    if not passed:
        raise Error("CHECK FAILED: " + label)


def check_abs_lt(label: String, value: Float64, threshold: Float64) raises:
    var av = abs_f64(value)
    var passed = av < threshold
    var tag = "PASS" if passed else "FAIL"
    print("  [" + tag + "] " + label + " =", value, "  (|·| <", threshold, ")")
    if not passed:
        raise Error("CHECK FAILED: " + label)


# ============================================================================
# Test 1 — Uniformity (single stream, 200k samples, 200-bin χ²)
# ============================================================================

def test_uniformity() raises:
    banner("Test 1 — Uniformity (single stream, 200-bin χ²)")

    # PhiloxRandom: fresh instance per call, offset 0..N-1, slot 0
    var counts_p = List[Int](capacity=N_BINS_1D)
    for _ in range(N_BINS_1D):
        counts_p.append(0)
    for i in range(N_UNIFORMITY):
        var rng = PhiloxRandom(seed=UInt64(2026), offset=UInt64(i))
        var u = Float64(rng.step_uniform()[0])
        var b = Int(u * Float64(N_BINS_1D))
        if b < 0:
            b = 0
        elif b >= N_BINS_1D:
            b = N_BINS_1D - 1
        counts_p[b] += 1
    var chi2_p = chi2_uniform(counts_p, N_UNIFORMITY)
    check_lt("Philox(seed=2026, offset=i)[0] χ²", chi2_p, CHI2_TH_200)

    # PhiloxRandom: SAME instance, advance internal counter via 4-block batches
    var counts_p2 = List[Int](capacity=N_BINS_1D)
    for _ in range(N_BINS_1D):
        counts_p2.append(0)
    var rng_serial = PhiloxRandom(seed=UInt64(2026), offset=UInt64(0))
    var quartet_idx = 0
    var quartet = rng_serial.step_uniform()
    for _ in range(N_UNIFORMITY):
        if quartet_idx == 4:
            quartet = rng_serial.step_uniform()
            quartet_idx = 0
        var u = Float64(quartet[quartet_idx])
        quartet_idx += 1
        var b = Int(u * Float64(N_BINS_1D))
        if b < 0:
            b = 0
        elif b >= N_BINS_1D:
            b = N_BINS_1D - 1
        counts_p2[b] += 1
    var chi2_p2 = chi2_uniform(counts_p2, N_UNIFORMITY)
    check_lt("Philox single-stream serial (advance internal counter) χ²",
             chi2_p2, CHI2_TH_200)

    # random_float64 baseline (process-global) — sanity reference
    global_seed(2026)
    var counts_r = List[Int](capacity=N_BINS_1D)
    for _ in range(N_BINS_1D):
        counts_r.append(0)
    for _ in range(N_UNIFORMITY):
        var u = random_float64()
        var b = Int(u * Float64(N_BINS_1D))
        if b < 0:
            b = 0
        elif b >= N_BINS_1D:
            b = N_BINS_1D - 1
        counts_r[b] += 1
    var chi2_r = chi2_uniform(counts_r, N_UNIFORMITY)
    check_lt("random_float64() baseline χ²", chi2_r, CHI2_TH_200)


# ============================================================================
# Test 2 — Sample moments (mean, variance vs U[0,1] theoretical)
# ============================================================================

def test_moments() raises:
    banner("Test 2 — Sample moments (mean → 0.5, var → 1/12)")

    var samples_p = List[Float64](capacity=N_UNIFORMITY)
    for i in range(N_UNIFORMITY):
        var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(i))
        samples_p.append(Float64(rng.step_uniform()[0]))
    var mp = sample_mean(samples_p)
    var vp = sample_var(samples_p, mp)
    check_abs_lt("Philox sample_mean - 0.5", mp - 0.5, MEAN_TOL)
    check_abs_lt("Philox sample_var  - 1/12", vp - (1.0 / 12.0), VAR_TOL)

    global_seed(7)
    var samples_r = List[Float64](capacity=N_UNIFORMITY)
    for _ in range(N_UNIFORMITY):
        samples_r.append(random_float64())
    var mr = sample_mean(samples_r)
    var vr = sample_var(samples_r, mr)
    check_abs_lt("random_float64 sample_mean - 0.5", mr - 0.5, MEAN_TOL)
    check_abs_lt("random_float64 sample_var  - 1/12", vr - (1.0 / 12.0), VAR_TOL)


# ============================================================================
# Test 3 — Autocorrelation (lag 1, 2, 5)
# ============================================================================

def test_autocorrelation() raises:
    banner("Test 3 — Autocorrelation (lag 1, 2, 5 ≈ 0)")

    # Serial Philox stream (advance internal counter via 4-block batches)
    var samples = List[Float64](capacity=N_AUTOCORR)
    var rng = PhiloxRandom(seed=UInt64(31), offset=UInt64(0))
    var qi = 0
    var q = rng.step_uniform()
    for _ in range(N_AUTOCORR):
        if qi == 4:
            q = rng.step_uniform()
            qi = 0
        samples.append(Float64(q[qi]))
        qi += 1
    var r1 = lag_k_autocorr(samples, 1)
    var r2 = lag_k_autocorr(samples, 2)
    var r5 = lag_k_autocorr(samples, 5)
    check_abs_lt("Philox lag-1 autocorr", r1, AUTOCORR_TOL)
    check_abs_lt("Philox lag-2 autocorr", r2, AUTOCORR_TOL)
    check_abs_lt("Philox lag-5 autocorr", r5, AUTOCORR_TOL)

    # Same test on the offset-i pattern (most-used in this repo)
    var samples_o = List[Float64](capacity=N_AUTOCORR)
    for i in range(N_AUTOCORR):
        var r = PhiloxRandom(seed=UInt64(31), offset=UInt64(i))
        samples_o.append(Float64(r.step_uniform()[0]))
    var ro1 = lag_k_autocorr(samples_o, 1)
    var ro2 = lag_k_autocorr(samples_o, 2)
    var ro5 = lag_k_autocorr(samples_o, 5)
    check_abs_lt("Philox (offset=i)[0] lag-1 autocorr", ro1, AUTOCORR_TOL)
    check_abs_lt("Philox (offset=i)[0] lag-2 autocorr", ro2, AUTOCORR_TOL)
    check_abs_lt("Philox (offset=i)[0] lag-5 autocorr", ro5, AUTOCORR_TOL)


# ============================================================================
# Test 4 — Bit balance (each bit position of int(u·2^24) is ~50/50)
# ============================================================================

def test_bit_balance() raises:
    banner("Test 4 — Bit balance (per-bit fraction ≈ 0.5)")

    var n = N_UNIFORMITY
    var counts = List[Int](capacity=N_BITS_TESTED)
    for _ in range(N_BITS_TESTED):
        counts.append(0)

    for i in range(n):
        var rng = PhiloxRandom(seed=UInt64(101), offset=UInt64(i))
        var u = Float64(rng.step_uniform()[0])
        # Map to a 24-bit integer; test lower N_BITS_TESTED bits.
        var uint24 = Int(u * 16777216.0)
        if uint24 < 0:
            uint24 = 0
        elif uint24 >= 16777216:
            uint24 = 16777215
        for b in range(N_BITS_TESTED):
            if (uint24 >> b) & 1 == 1:
                counts[b] += 1

    var worst_dev = Float64(0.0)
    var worst_bit = -1
    for b in range(N_BITS_TESTED):
        var frac = Float64(counts[b]) / Float64(n)
        var dev = abs_f64(frac - 0.5)
        if dev > worst_dev:
            worst_dev = dev
            worst_bit = b
    print("  worst bit", worst_bit, " |fraction - 0.5| =", worst_dev)
    check_lt("Philox worst bit |fraction - 0.5|", worst_dev, BIT_BALANCE_TOL)


# ============================================================================
# Test 5 — Near-by seed independence
# ============================================================================
# For seeds s, s+1, s+2, ..., s+N-1 each at offset=0, slot=0, check the
# resulting sample is uniform (per-seed χ²) AND uncorrelated across the seed
# axis (lag-1 autocorrelation on the seed-indexed sequence).
# This is the direct test of the "consecutive-seed correlation" hypothesis.

def test_nearby_seed_independence() raises:
    banner("Test 5 — Near-by seed independence")

    var seed_samples = List[Float64](capacity=N_NEARBY_SEEDS)
    var counts = List[Int](capacity=N_NEARBY_BINS)
    for _ in range(N_NEARBY_BINS):
        counts.append(0)
    for s in range(N_NEARBY_SEEDS):
        var rng = PhiloxRandom(seed=UInt64(2026 + s), offset=UInt64(0))
        var u = Float64(rng.step_uniform()[0])
        seed_samples.append(u)
        var b = Int(u * Float64(N_NEARBY_BINS))
        if b < 0:
            b = 0
        elif b >= N_NEARBY_BINS:
            b = N_NEARBY_BINS - 1
        counts[b] += 1

    var chi2 = chi2_uniform(counts, N_NEARBY_SEEDS)
    check_lt("Near-by seeds first-sample χ² (200 bins)", chi2, CHI2_TH_200)

    # Autocorr across the seed axis — does seed s correlate with seed s+1?
    var ac_tol = 1.5 / sqrt(Float64(N_NEARBY_SEEDS))  # ~5σ for this N
    var r1 = lag_k_autocorr(seed_samples, 1)
    var r2 = lag_k_autocorr(seed_samples, 2)
    print("  seed-axis lag-1 = ", r1, "  lag-2 = ", r2, "  tol ≈", ac_tol)
    check_abs_lt("seed-axis lag-1 autocorr", r1, 6.0 * ac_tol)
    check_abs_lt("seed-axis lag-2 autocorr", r2, 6.0 * ac_tol)

    # Joint 2D check: pairs (u(s, 0), u(s+1, 0)) on a 100-bin grid (10x10)
    # 100 cells, df=99, χ² threshold = CHI2_TH_100
    var pair_counts = List[Int](capacity=100)
    for _ in range(100):
        pair_counts.append(0)
    for s in range(N_NEARBY_SEEDS - 1):
        var a = seed_samples[s]
        var b = seed_samples[s + 1]
        var ai = Int(a * 10.0)
        var bi = Int(b * 10.0)
        if ai < 0:
            ai = 0
        elif ai > 9:
            ai = 9
        if bi < 0:
            bi = 0
        elif bi > 9:
            bi = 9
        pair_counts[ai * 10 + bi] += 1
    var chi2_pair = chi2_uniform(pair_counts, N_NEARBY_SEEDS - 1)
    check_lt("Consecutive-seed pair χ² (10x10 grid)", chi2_pair, CHI2_TH_100)


# ============================================================================
# Test 6 — Pendulum-reset 2D distribution (V1 vs V2 A/B)
# ============================================================================
# Exact replication of the env reset logic for V2 (per-env Philox stream,
# `seed=base+env_id`, counter starting at 0) and V1 (global random_float64).
# Both should produce a uniform 2D distribution on [-π,π] × [-1,1].
# Within-env serial correlation is also tested — this is where the two
# protocols differ in principle, even if marginal distributions agree.

def test_pendulum_reset_distribution() raises:
    banner("Test 6 — Pendulum-reset 2D distribution (V1 vs V2)")

    var total = PENDULUM_N_ENVS * PENDULUM_N_RESETS_PER_ENV
    var n_cells = PENDULUM_GRID * PENDULUM_GRID

    # ----- V2 path: per-env PhiloxRandom -----
    var v2_counts = List[Int](capacity=n_cells)
    for _ in range(n_cells):
        v2_counts.append(0)
    # also collect per-env theta sequence for serial-correlation check
    var v2_within_env_corr_sum = Float64(0.0)
    var v2_within_env_corr_count = 0

    for env_id in range(PENDULUM_N_ENVS):
        var rng_seed = UInt64(2026 + env_id)
        var thetas = List[Float64](capacity=PENDULUM_N_RESETS_PER_ENV)
        for reset_idx in range(PENDULUM_N_RESETS_PER_ENV):
            var rng = PhiloxRandom(seed=rng_seed, offset=UInt64(reset_idx))
            var rv = rng.step_uniform()
            var u0 = Float64(rv[0])
            var u1 = Float64(rv[1])
            var theta = (u0 * 2.0 - 1.0) * pi
            var theta_dot = u1 * 2.0 - 1.0
            thetas.append(theta)
            # Map (theta, theta_dot) ∈ [-π,π]×[-1,1] to grid cell
            var ti = Int(((theta + pi) / (2.0 * pi)) * Float64(PENDULUM_GRID))
            var di = Int(((theta_dot + 1.0) / 2.0) * Float64(PENDULUM_GRID))
            if ti < 0:
                ti = 0
            elif ti >= PENDULUM_GRID:
                ti = PENDULUM_GRID - 1
            if di < 0:
                di = 0
            elif di >= PENDULUM_GRID:
                di = PENDULUM_GRID - 1
            v2_counts[ti * PENDULUM_GRID + di] += 1
        # per-env serial autocorrelation
        var r = lag_k_autocorr(thetas, 1)
        v2_within_env_corr_sum += r * r
        v2_within_env_corr_count += 1

    var chi2_v2 = chi2_uniform(v2_counts, total)
    check_lt("V2 (per-env Philox) reset 2D χ² (20x20)", chi2_v2, CHI2_TH_400)
    var v2_rms_within_env = sqrt(v2_within_env_corr_sum / Float64(v2_within_env_corr_count))
    # Per-env N = 500 → SE ≈ 1/√500 ≈ 4.5e-2; average over 32 envs → ~8e-3.
    # 5σ on the RMS ≈ 5 * 0.045 / sqrt(32) ≈ 4e-2. Use 6e-2 for safety.
    check_lt("V2 within-env RMS lag-1 autocorr", v2_rms_within_env, 6e-2)

    # ----- V1 path: global random_float64 -----
    global_seed(2026)
    var v1_counts = List[Int](capacity=n_cells)
    for _ in range(n_cells):
        v1_counts.append(0)
    var v1_within_env_corr_sum = Float64(0.0)
    var v1_within_env_corr_count = 0

    # V1 protocol: there's no per-env stream — env resets are interleaved
    # in the global stream. Simulate the realistic pattern: round-robin
    # across envs (env 0 reset 0, env 1 reset 0, ..., env 0 reset 1, ...).
    var v1_thetas_by_env = List[List[Float64]]()
    for _ in range(PENDULUM_N_ENVS):
        v1_thetas_by_env.append(List[Float64]())

    for _ in range(PENDULUM_N_RESETS_PER_ENV):
        for env_id in range(PENDULUM_N_ENVS):
            var u0 = random_float64()
            var u1 = random_float64()
            var theta = (u0 * 2.0 - 1.0) * pi
            var theta_dot = u1 * 2.0 - 1.0
            v1_thetas_by_env[env_id].append(theta)
            var ti = Int(((theta + pi) / (2.0 * pi)) * Float64(PENDULUM_GRID))
            var di = Int(((theta_dot + 1.0) / 2.0) * Float64(PENDULUM_GRID))
            if ti < 0:
                ti = 0
            elif ti >= PENDULUM_GRID:
                ti = PENDULUM_GRID - 1
            if di < 0:
                di = 0
            elif di >= PENDULUM_GRID:
                di = PENDULUM_GRID - 1
            v1_counts[ti * PENDULUM_GRID + di] += 1

    var chi2_v1 = chi2_uniform(v1_counts, total)
    check_lt("V1 (global random_float64) reset 2D χ² (20x20)", chi2_v1, CHI2_TH_400)

    for env_id in range(PENDULUM_N_ENVS):
        var r = lag_k_autocorr(v1_thetas_by_env[env_id], 1)
        v1_within_env_corr_sum += r * r
        v1_within_env_corr_count += 1
    var v1_rms_within_env = sqrt(v1_within_env_corr_sum / Float64(v1_within_env_corr_count))
    check_lt("V1 within-env RMS lag-1 autocorr", v1_rms_within_env, 6e-2)

    # Direct A/B: the per-cell counts of V1 and V2 should be statistically
    # indistinguishable. Compute a two-sample χ² (Pearson form).
    # H0: both samples share a common distribution.
    var twosample_chi2 = Float64(0.0)
    for i in range(n_cells):
        var c1 = Float64(v1_counts[i])
        var c2 = Float64(v2_counts[i])
        var rowsum = c1 + c2
        if rowsum > 0.0:
            var expected_each = rowsum / 2.0
            var d1 = c1 - expected_each
            var d2 = c2 - expected_each
            twosample_chi2 += d1 * d1 / expected_each
            twosample_chi2 += d2 * d2 / expected_each
    # df = (n_cells - 1) * (2 - 1) = n_cells - 1 = 399
    print("  V1 vs V2 two-sample χ²(df=399) =", twosample_chi2)
    check_lt("V1 vs V2 two-sample χ²", twosample_chi2, CHI2_TH_400)


# ============================================================================
# Test 7 — GPU seed-formula diffusion (CPU-side replication)
# ============================================================================
# Sweep (training_step, env_id) ∈ [0, GPU_STEPS) × [0, GPU_BATCH); apply the
# GPU formula `Int(step) * 2654435761 + env * 12345` (same as
# `_reset_env_gpu`); check the derived first-sample distribution is uniform
# and uncorrelated along both axes.

def test_gpu_seed_formula() raises:
    banner("Test 7 — GPU seed-formula diffusion")

    var total = GPU_STEPS * GPU_BATCH
    var counts = List[Int](capacity=N_BINS_1D)
    for _ in range(N_BINS_1D):
        counts.append(0)

    # We'll also collect samples along the env-axis (fixed step) for one step,
    # and along the step-axis (fixed env) for one env, to test diffusion in
    # both directions.
    var env_axis_samples = List[Float64](capacity=GPU_BATCH)
    var step_axis_samples = List[Float64](capacity=GPU_STEPS)

    for step in range(GPU_STEPS):
        for env in range(GPU_BATCH):
            var combined = Int(step) * 2654435761 + env * 12345
            var rng = PhiloxRandom(seed=UInt64(combined), offset=UInt64(0))
            var u = Float64(rng.step_uniform()[0])
            var b = Int(u * Float64(N_BINS_1D))
            if b < 0:
                b = 0
            elif b >= N_BINS_1D:
                b = N_BINS_1D - 1
            counts[b] += 1
            if step == 0:
                env_axis_samples.append(u)
            if env == 0:
                step_axis_samples.append(u)

    var chi2 = chi2_uniform(counts, total)
    check_lt("GPU seed-formula χ² (200 bins)", chi2, CHI2_TH_200)

    var r_env_1 = lag_k_autocorr(env_axis_samples, 1)
    var r_env_2 = lag_k_autocorr(env_axis_samples, 2)
    var env_tol = 6.0 / sqrt(Float64(GPU_BATCH))  # ~6σ for this N
    print("  env-axis lag-1 =", r_env_1, " lag-2 =", r_env_2, " tol ≈", env_tol)
    check_abs_lt("env-axis lag-1 autocorr", r_env_1, env_tol)
    check_abs_lt("env-axis lag-2 autocorr", r_env_2, env_tol)

    var r_step_1 = lag_k_autocorr(step_axis_samples, 1)
    var step_tol = 6.0 / sqrt(Float64(GPU_STEPS))  # very small N → loose
    print("  step-axis lag-1 =", r_step_1, " tol ≈", step_tol)
    check_abs_lt("step-axis lag-1 autocorr", r_step_1, step_tol)


# ============================================================================
# GPU kernel test — call PhiloxRandom from a real device kernel
# ============================================================================
# Replicates `_reset_env_gpu`'s exact pattern. Writes 4 samples per thread
# into a [BATCH, 4] device buffer, copies back, runs uniformity + per-env
# χ² checks.

def test_gpu_kernel() raises:
    banner("GPU Kernel Test — on-device PhiloxRandom usage")

    var ctx = DeviceContext()
    var samples_dbuf = ctx.enqueue_create_buffer[dtype](GPU_BATCH * 4)
    var samples_t = tile[GPU_BATCH, 4](samples_dbuf.unsafe_ptr().as_unsafe_any_origin())

    comptime BLOCKS = (GPU_BATCH + TPB - 1) // TPB

    @always_inline
    def gen_kernel(
        dst: Tile[dtype, GPU_BATCH, 4],
        seed_v: Scalar[dtype],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= GPU_BATCH:
            return
        var combined = Int(seed_v) * 2654435761 + env * 12345
        var rng = PhiloxRandom(seed=UInt64(combined), offset=UInt64(0))
        var v = rng.step_uniform()
        dst[env, 0] = v[0]
        dst[env, 1] = v[1]
        dst[env, 2] = v[2]
        dst[env, 3] = v[3]

    ctx.enqueue_function[gen_kernel](
        samples_t,
        Scalar[dtype](42.0),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[dtype](GPU_BATCH * 4)
    ctx.enqueue_copy(host, samples_dbuf)
    ctx.synchronize()

    # Pool all 4*BATCH samples; check uniformity
    var counts = List[Int](capacity=GPU_BINS)
    for _ in range(GPU_BINS):
        counts.append(0)
    var hp = host.unsafe_ptr().as_unsafe_any_origin()
    var total = GPU_BATCH * 4
    var samples = List[Float64](capacity=total)
    for i in range(total):
        var u = Float64(hp[i])
        samples.append(u)
        var b = Int(u * Float64(GPU_BINS))
        if b < 0:
            b = 0
        elif b >= GPU_BINS:
            b = GPU_BINS - 1
        counts[b] += 1

    var chi2 = chi2_uniform(counts, total)
    check_lt("GPU on-device samples χ² (100 bins)", chi2, CHI2_TH_100)

    var m = sample_mean(samples)
    # SE ≈ √((1/12)/total) ≈ 5e-4 for total=32k; 5σ ≈ 2.5e-3. Use 5e-3 for safety.
    check_abs_lt("GPU on-device mean - 0.5", m - 0.5, 5e-3)

    # Cross-env slot correlation: within a single thread, slots 0-3 of
    # step_uniform() should be independent. Check correlation between
    # slots 0 and 1 across the batch.
    var s0 = Float64(0.0)
    var s1 = Float64(0.0)
    for env in range(GPU_BATCH):
        s0 += Float64(hp[env * 4 + 0])
        s1 += Float64(hp[env * 4 + 1])
    var m0 = s0 / Float64(GPU_BATCH)
    var m1 = s1 / Float64(GPU_BATCH)
    var num = Float64(0.0)
    var d0sq = Float64(0.0)
    var d1sq = Float64(0.0)
    for env in range(GPU_BATCH):
        var x = Float64(hp[env * 4 + 0]) - m0
        var y = Float64(hp[env * 4 + 1]) - m1
        num += x * y
        d0sq += x * x
        d1sq += y * y
    var corr_01 = num / sqrt(d0sq * d1sq)
    var corr_tol = 6.0 / sqrt(Float64(GPU_BATCH))
    print("  slot 0 vs slot 1 corr =", corr_01, "  tol ≈", corr_tol)
    check_abs_lt("GPU step_uniform slot 0 vs slot 1 corr", corr_01, corr_tol)


# ============================================================================
# Main
# ============================================================================

def main() raises:
    print("PhiloxRandom statistical quality test suite")
    print("N_UNIFORMITY    =", N_UNIFORMITY)
    print("N_NEARBY_SEEDS  =", N_NEARBY_SEEDS)
    print("PENDULUM_N_ENVS =", PENDULUM_N_ENVS)
    print("PENDULUM_RESETS =", PENDULUM_N_RESETS_PER_ENV)
    print("GPU_BATCH       =", GPU_BATCH)

    test_uniformity()
    test_moments()
    test_autocorrelation()
    test_bit_balance()
    test_nearby_seed_independence()
    test_pendulum_reset_distribution()
    test_gpu_seed_formula()
    test_gpu_kernel()

    print("")
    print("======================================================================")
    print("  ALL TESTS PASSED")
    print("======================================================================")
