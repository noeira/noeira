"""Planner-side GPU kernels for MPPI.

These are the planner-generic pieces of the MPPI pipeline — sampling
actions, accumulating returns, softmax weighting (top-K elites),
mean/std refit, action selection. They contain **no agent-specific
logic** (no Q-decoding, no policy heads, no categorical bins): the
``RolloutCallbackGPU`` adapter is responsible for everything between
"take z and an action" and "produce z_next + scalar reward".

Canonical home of the kernels — agents reach in here, not the other
way around. The reverse (TD-MPC2 owning shared kernels) was the
transitional state after Phase 2 landed; this file took ownership
during the Phase 2 cleanup.

Kernels:
  * ``mppi_broadcast_z0_zero_returns_batched_kernel`` — fused init
    of (z_all = z0 broadcast, returns = 0) across all envs.
  * ``mppi_sample_actions_batched_kernel`` — per-(env, sample,
    horizon-step) Gaussian sampling around per-env (mean, std);
    policy-warm-start trajectories (``local_s < NUM_PI_TRAJS``) use
    ``pi_out`` as the mean with smaller noise.
  * ``mppi_accum_reward_scalar_kernel`` — accumulate pre-decoded
    scalar reward into returns. Trait-friendly: the callback decodes
    the reward (categorical → scalar where applicable) and writes
    the result into ``reward_step``; this kernel multiplies by the
    discount and adds to running returns.
  * ``mppi_copy_z_kernel`` — copy z_next → z between horizon steps.
  * ``mppi_add_terminal_value_kernel`` — add ``discount^H * V(z_H)``
    to running returns, with a NaN/Inf guard to keep softmax from
    poisoning.
  * ``mppi_softmax_weights_kernel`` — per-env top-K elite softmax
    over returns, normalizing the weights. Non-elite weights are
    zeroed. Matches the reference TD-MPC2 recipe
    (``tdmpc2.py:186``).
  * ``mppi_weighted_mean_std_kernel`` — per-env weighted mean and
    std refit over the elite-weighted action distribution. Std is
    clamped to ``[0.05, 2.0]``.
  * ``mppi_select_action_kernel`` — gumbel-max sampling from elite
    softmax + optional per-axis exploration noise + clamp. One
    block per env; one kernel call selects the per-env action and
    writes it directly to an on-device output buffer.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.math import exp, log, sqrt, cos, isnan, isinf
from std.random.philox import Random as PhiloxRandom
from noeira.core.tiles import Tile, shared_tile


# =============================================================================
# Setup: fused z0 broadcast + returns zero-init (one kernel per call)
# =============================================================================


@always_inline
def mppi_broadcast_z0_zero_returns_batched_kernel[
    dtype: DType,
    BATCH_TOTAL: Int,
    N_ENVS: Int,
    TOTAL_SAMPLES: Int,
    LATENT_DIM: Int,
](
    z0: Tile[dtype, N_ENVS, LATENT_DIM],
    z_all: Tile[dtype, BATCH_TOTAL, LATENT_DIM],
    returns: Tile[dtype, BATCH_TOTAL],
) where dtype.is_floating_point():
    """Fused: broadcast per-env ``z0`` to every candidate row + zero
    the returns accumulator. One kernel launch replaces the legacy
    two-kernel pair.

    ⚠ ONE THREAD PER ELEMENT — same fix, and same reason, as
    ``mppi_copy_z_kernel``: the row-per-thread form ran a private
    512-float loop per thread on a 9-block grid, measured at 92 us x
    800 launches = 2.4% of GPU time on an RTX 5090 to write 4.4 MB.
    The ``returns`` zeroing rides along on the ``k == 0`` threads so
    this stays a single launch.
    """
    comptime N = BATCH_TOTAL * LATENT_DIM
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N:
        return

    var row = i // LATENT_DIM
    var k = i % LATENT_DIM
    z_all[row, k] = z0[row // TOTAL_SAMPLES, k]
    if k == 0:
        returns[row] = Scalar[dtype](0.0)


# =============================================================================
# Per-horizon-step sampling + accumulation
# =============================================================================


@always_inline
def mppi_sample_actions_batched_kernel[
    dtype: DType,
    BATCH_TOTAL: Int,
    N_ENVS: Int,
    TOTAL_SAMPLES: Int,
    NUM_PI_TRAJS: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    POL_OUT: Int = ACTION_DIM * 2,
](
    pi_out: Tile[dtype, BATCH_TOTAL, POL_OUT],
    mean: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
    std: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
    act_step: Tile[dtype, BATCH_TOTAL, ACTION_DIM],
    all_actions: Tile[dtype, BATCH_TOTAL * HORIZON * ACTION_DIM],
    # ⚠ Int32, NOT Int: `Int`/`UInt` do not conform to `DevicePassable` since
    # Mojo 1.0.0rc2, so an `Int` here fails to compile AT THE LAUNCH SITE with
    # a constraint error inside `SIMD`. The call site must pass `Int32(t)` too
    # — a bare `t` re-introduces the same failure through implicit conversion.
    step: Int32,
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Sample actions for every env's MPPI candidates at one horizon
    step.

    Thread ``i`` is the global sample index. ``env_idx = i //
    TOTAL_SAMPLES``, ``local_s = i % TOTAL_SAMPLES``. Policy-warm
    trajectories (``local_s < NUM_PI_TRAJS``) sample around
    ``pi_out[i]`` with std=0.1; MPPI trajectories sample around the
    refit Gaussian ``(mean, std)``.

    ``POL_OUT`` defaults to ``2 * ACTION_DIM`` (TDMPC2's
    mean+log_std actor output) but the kernel only reads the first
    ``ACTION_DIM`` columns of ``pi_out`` — pass ``POL_OUT =
    ACTION_DIM`` when the adapter feeds the mean directly.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH_TOTAL:
        return

    var env_idx = i // TOTAL_SAMPLES
    var local_s = i % TOTAL_SAMPLES
    var step_i = Int(step)

    for j in range(ACTION_DIM):
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(i) * UInt64(ACTION_DIM) + UInt64(j),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var u1 = Scalar[DType.float32](rand_vals[0]) + 1e-8
        var u2 = Scalar[DType.float32](rand_vals[1])
        var mag = sqrt(Float32(-2.0) * log(u1))
        var noise = Scalar[dtype](
            mag * cos(u2 * Scalar[DType.float32](6.283185307179586))
        )

        var act: Scalar[dtype]
        if local_s < NUM_PI_TRAJS:
            var pi_mean = Scalar[dtype](pi_out[i, j][0])
            act = pi_mean + noise * Scalar[dtype](0.1)
        else:
            var mean_idx = (
                env_idx * HORIZON * ACTION_DIM + step_i * ACTION_DIM + j
            )
            var mu = Scalar[dtype](mean[mean_idx][0])
            var sigma = Scalar[dtype](std[mean_idx][0])
            act = mu + sigma * noise

        if act < Scalar[dtype](-1.0):
            act = Scalar[dtype](-1.0)
        if act > Scalar[dtype](1.0):
            act = Scalar[dtype](1.0)

        act_step[i, j] = act
        all_actions[i * HORIZON * ACTION_DIM + step_i * ACTION_DIM + j] = act


@always_inline
def mppi_accum_reward_scalar_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
](
    reward_step: Tile[dtype, TOTAL_SAMPLES],
    returns: Tile[dtype, TOTAL_SAMPLES],
    discount: Scalar[dtype],
) where dtype.is_floating_point():
    """Accumulate pre-decoded per-step rewards into discounted returns.

    ``returns[i] += discount * reward_step[i]``. The callback's
    ``rollout_step_gpu`` is expected to have already produced a
    scalar reward per batch row — this is the planner-side half of
    the trait split (the callback owns reward decoding).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return
    returns[i] = returns[i] + discount * Scalar[dtype](reward_step[i][0])


@always_inline
def mppi_copy_z_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
    LATENT_DIM: Int,
](
    dst: Tile[dtype, TOTAL_SAMPLES, LATENT_DIM],
    src: Tile[dtype, TOTAL_SAMPLES, LATENT_DIM],
) where dtype.is_floating_point():
    """Copy latent states: ``dst[i] = src[i]``.

    Used between horizon steps to roll ``z_next`` back into ``z``
    for the next step's input.

    ⚠ ONE THREAD PER ELEMENT, not per row. The row-per-thread version
    (``for k in range(LATENT_DIM)``) gave each thread a 512-float
    private run, so adjacent threads touched addresses 512 floats
    apart — fully uncoalesced — and the grid was only
    ``ceil(BATCH_TOTAL / TPB)`` = 9 blocks on a card with ~170 SMs.
    Measured on an RTX 5090 at BATCH_TOTAL=2144: 104 us per launch,
    2400 launches, **8.0% of all GPU time** to move 8.8 MB, i.e.
    ~84 GB/s on a ~1.5 TB/s part. Flat indexing makes the access
    contiguous across a warp and the grid 4288 blocks.
    """
    comptime N = TOTAL_SAMPLES * LATENT_DIM
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N:
        return
    dst[i // LATENT_DIM, i % LATENT_DIM] = src[
        i // LATENT_DIM, i % LATENT_DIM
    ]


# =============================================================================
# Terminal bootstrap
# =============================================================================


@always_inline
def mppi_add_terminal_value_kernel[
    dtype: DType,
    TOTAL_SAMPLES: Int,
](
    q_min: Tile[dtype, TOTAL_SAMPLES],
    returns: Tile[dtype, TOTAL_SAMPLES],
    discount: Scalar[dtype],
) where dtype.is_floating_point():
    """Add discounted terminal bootstrap value to returns, with NaN/
    Inf guard.

    ``returns[i] = nan_to_num(returns[i] + discount * q_min[i])``.
    Without the guard, a single NaN sample from saturating
    activations early in training pollutes the entire env's softmax
    (because ``exp(NaN) = NaN`` propagates). Matches reference TD-MPC2
    (``tdmpc2.py:185``, ``value.nan_to_num(0)``).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= TOTAL_SAMPLES:
        return

    var v = Scalar[dtype](returns[i][0]) + discount * Scalar[dtype](q_min[i][0])
    if isnan(v) or isinf(v):
        v = Scalar[dtype](0.0)
    returns[i] = v


# =============================================================================
# Top-K elite softmax + weighted mean/std refit
# =============================================================================


@always_inline
def mppi_softmax_weights_kernel[
    dtype: DType,
    N_ENVS: Int,
    TOTAL_SAMPLES: Int,
    NUM_ELITES: Int,
    BLOCK_SIZE: Int,
](
    returns: Tile[dtype, N_ENVS * TOTAL_SAMPLES],
    weights: Tile[dtype, N_ENVS * TOTAL_SAMPLES],
    temperature: Scalar[dtype],
) where dtype.is_floating_point():
    """Per-env top-K elite softmax over returns → normalized weights.

    Grid: ``N_ENVS`` blocks. Block: ``BLOCK_SIZE`` threads.
    Each block handles one env's ``TOTAL_SAMPLES`` returns.

    Reference TD-MPC2 (``tdmpc2.py:186``) selects ``num_elites=64`` of
    ``num_samples=536`` by value, then computes softmax over only
    those elites. Without the elite filter the bottom-fraction
    trajectories still contribute small-but-nonzero weight, biasing
    the MPPI mean update toward averaging-in bad samples — was the
    likely cause of the HalfCheetah training plateau at ~-800 reward
    before this kernel was added.

    Algorithm:
      1. Reduce max return per env (numerical stability shift).
      2. For each sample ``s``, compute ``rank = #{k : returns[k] >
         returns[s] OR (returns[k] == returns[s] AND k < s)}``. Sample
         is elite iff ``rank < NUM_ELITES``.
      3. ``weights[s] = exp(temperature * (returns[s] - max))`` if
         elite, else 0.
      4. Reduce sum + normalize over elites.
    """
    var env_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = env_idx * TOTAL_SAMPLES

    var smem = shared_tile[dtype, BLOCK_SIZE]()

    # Pass 1: find max return.
    var local_max = Scalar[dtype](-1e30)
    var s = tid
    while s < TOTAL_SAMPLES:
        var v = Scalar[dtype](returns[base + s][0])
        if v > local_max:
            local_max = v
        s += BLOCK_SIZE
    smem[tid] = local_max

    barrier()

    var stride = BLOCK_SIZE >> 1
    while stride > 0:
        if tid < stride:
            var other = smem[tid + stride]
            var mine = smem[tid]
            if other > mine:
                smem[tid] = other
        barrier()
        stride >>= 1

    var max_ret = smem[0]
    barrier()

    # Pass 2: per-sample rank + masked exp weight + local sum.
    var local_sum: smem.ElementType = 0.0
    s = tid
    while s < TOTAL_SAMPLES:
        var v_s = Scalar[dtype](returns[base + s][0])
        var rank = 0
        for k in range(TOTAL_SAMPLES):
            var v_k = Scalar[dtype](returns[base + k][0])
            if v_k > v_s:
                rank += 1
            elif v_k == v_s and k < s:
                rank += 1
        if rank < NUM_ELITES:
            var w = exp(temperature * (v_s - max_ret))
            weights[base + s] = w
            local_sum += w
        else:
            weights[base + s] = Scalar[dtype](0.0)
        s += BLOCK_SIZE
    smem[tid] = local_sum

    barrier()

    stride = BLOCK_SIZE >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] = rebind[Scalar[dtype]](smem[tid]) + rebind[
                Scalar[dtype]
            ](smem[tid + stride])
        barrier()
        stride >>= 1

    var total_sum = rebind[Scalar[dtype]](smem[0])
    if total_sum < Scalar[dtype](1e-10):
        total_sum = Scalar[dtype](1e-10)
    barrier()

    # Pass 3: normalize.
    var inv_sum = Scalar[dtype](1.0) / total_sum
    s = tid
    while s < TOTAL_SAMPLES:
        weights[base + s] = Scalar[dtype](weights[base + s][0]) * inv_sum
        s += BLOCK_SIZE


@always_inline
def mppi_weighted_mean_std_kernel[
    dtype: DType,
    N_ENVS: Int,
    TOTAL_SAMPLES: Int,
    HORIZON: Int,
    ACTION_DIM: Int,
    BLOCK_SIZE: Int,
](
    weights: Tile[dtype, N_ENVS * TOTAL_SAMPLES],
    all_actions: Tile[dtype, N_ENVS * TOTAL_SAMPLES * HORIZON * ACTION_DIM],
    mean_out: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
    std_out: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
) where dtype.is_floating_point():
    """Refit ``(mean, std)`` per env using elite-weighted actions.

    ONE BLOCK per ``(env, t, a)`` triplet, reducing over ``TOTAL_SAMPLES``
    inside the block. Std is clamped to ``[STD_MIN=0.05, STD_MAX=2.0]``
    (TD-MPC2 reference defaults). The ``+1e-8`` inside ``sqrt`` keeps gradients
    well-defined at near-zero variance — safe because elite weights are
    guaranteed non-degenerate by ``mppi_softmax_weights_kernel``.

    ⚠ THE OUTPUT COUNT IS SMALLER THAN ONE BLOCK, so a thread-per-output grid
    cannot be widened. At the walker's dims ``TOTAL_DIMS = 8*3*6 = 144``, and
    the previous one-thread-per-output form launched
    ``ceil(144 / 256) = 1 BLOCK`` — one SM of ~170 on an RTX 5090 — with each
    thread running two serial 268-iteration passes. Measured at 53.3 us x 800
    launches = 42.6 ms, **1.4% of all GPU time** to produce 144 numbers.
    Widening the grid is not an option here; the parallelism has to come from
    the REDUCTION, hence block-per-output. 144 blocks x BLOCK_SIZE threads,
    each thread covering ``ceil(268 / BLOCK_SIZE)`` samples.

    ⚠ The tree reduction changes the fp32 SUMMATION ORDER versus the old
    serial loop, so refit means/stds move by ~1 ulp and a long MPPI rollout
    will diverge from previously recorded numbers. That is a reordering, not a
    regression — but it does mean this kernel is not bit-comparable across the
    change.
    """
    comptime TOTAL_DIMS = N_ENVS * HORIZON * ACTION_DIM
    var out_idx = Int(block_idx.x)
    if out_idx >= TOTAL_DIMS:
        return
    var tid = Int(thread_idx.x)

    var env_idx = out_idx // (HORIZON * ACTION_DIM)
    var rem = out_idx % (HORIZON * ACTION_DIM)
    var t = rem // ACTION_DIM
    var a = rem % ACTION_DIM

    var w_off = env_idx * TOTAL_SAMPLES
    # Fold the fixed (t, a) offset in once — the per-sample stride is then a
    # flat `HORIZON * ACTION_DIM`.
    var act_off = (
        env_idx * TOTAL_SAMPLES * HORIZON * ACTION_DIM
        + t * ACTION_DIM
        + a
    )
    comptime ASTRIDE = HORIZON * ACTION_DIM

    var smem = shared_tile[dtype, BLOCK_SIZE]()

    # ── pass 1: weighted mean ────────────────────────────────────────
    var acc = Scalar[dtype](0.0)
    var s = tid
    while s < TOTAL_SAMPLES:
        acc += Scalar[dtype](weights[w_off + s][0]) * Scalar[dtype](
            all_actions[act_off + s * ASTRIDE][0]
        )
        s += BLOCK_SIZE
    smem[tid] = acc
    barrier()

    var stride = BLOCK_SIZE >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] = smem[tid] + smem[tid + stride]
        barrier()
        stride >>= 1

    var wm = Scalar[dtype](smem[0][0])
    # ⚠ Every thread must READ smem[0] before pass 2 overwrites smem[tid].
    barrier()

    # ── pass 2: weighted variance about that mean ────────────────────
    var acc2 = Scalar[dtype](0.0)
    s = tid
    while s < TOTAL_SAMPLES:
        var diff = Scalar[dtype](
            all_actions[act_off + s * ASTRIDE][0]
        ) - wm
        acc2 += Scalar[dtype](weights[w_off + s][0]) * diff * diff
        s += BLOCK_SIZE
    smem[tid] = acc2
    barrier()

    stride = BLOCK_SIZE >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] = smem[tid] + smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        mean_out[out_idx] = wm
        var std_val = sqrt(Scalar[dtype](smem[0][0]) + Scalar[dtype](1e-8))
        if std_val < Scalar[dtype](0.05):
            std_val = Scalar[dtype](0.05)
        if std_val > Scalar[dtype](2.0):
            std_val = Scalar[dtype](2.0)
        std_out[out_idx] = std_val


# =============================================================================
# Action selection (Gumbel-max + per-axis noise + clamp)
# =============================================================================


@always_inline
def mppi_select_action_kernel[
    dtype: DType,
    N_ENVS: Int,
    TOTAL_SAMPLES: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    BLOCK_SIZE: Int,
](
    weights: Tile[dtype, N_ENVS * TOTAL_SAMPLES],
    all_actions: Tile[dtype, N_ENVS * TOTAL_SAMPLES * HORIZON * ACTION_DIM],
    std: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
    out_action: Tile[dtype, N_ENVS * ACTION_DIM],
    rng_seed: Scalar[DType.uint32],
    action_scale: Scalar[dtype],
    deterministic: UInt32,
) where dtype.is_floating_point():
    """Per-env action selection via Gumbel-max trick + Gaussian noise.

    Algorithm matches reference TD-MPC2 (``tdmpc2.py:201-207``):
      1. Sample ``idx ~ Categorical(weights)`` via Gumbel-max:
         ``idx = argmax_s (log(weights[s]) + Gumbel(0,1)[s])``
         where weights are post-softmax (already normalized in
         ``mppi_softmax_weights_kernel``; non-elite weights are 0).
      2. Take ``action = all_actions[env, idx, t=0, :]``.
      3. If not deterministic: ``action += std[env, 0, :] * N(0, 1)``.
      4. ``action = clamp(action * action_scale, ±action_scale)``.

    Grid: ``N_ENVS`` blocks. Block: ``BLOCK_SIZE`` threads. Block-wide
    argmax via shared-memory tree-reduction.
    """
    var env_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var w_off = env_idx * TOTAL_SAMPLES
    var act_off = env_idx * TOTAL_SAMPLES * HORIZON * ACTION_DIM

    var smem_val = shared_tile[dtype, BLOCK_SIZE]()
    var smem_idx = shared_tile[DType.uint32, BLOCK_SIZE]()

    # Phase 1: per-thread local argmax of (log(w) + gumbel).
    var local_max = Scalar[dtype](-1e30)
    var local_idx: UInt32 = 0
    var s = tid
    while s < TOTAL_SAMPLES:
        var w = Scalar[dtype](weights[w_off + s][0])
        if w > Scalar[dtype](1e-20):
            var philox = PhiloxRandom(
                seed=UInt64(rng_seed)
                + UInt64(env_idx) * UInt64(TOTAL_SAMPLES)
                + UInt64(s),
                offset=0,
            )
            var rand_vals = philox.step_uniform()
            var u = (
                Scalar[DType.float32](rand_vals[0])
                + Scalar[DType.float32](1e-12)
            )
            var g = -log(-log(u) + Scalar[DType.float32](1e-12))
            var score = log(w) + Scalar[dtype](g)
            if score > local_max:
                local_max = score
                local_idx = UInt32(s)
        s += BLOCK_SIZE
    smem_val[tid] = local_max
    smem_idx[tid] = local_idx

    barrier()

    # Phase 2: tree-reduce argmax across block.
    var stride = BLOCK_SIZE >> 1
    while stride > 0:
        if tid < stride:
            var other_v = Scalar[dtype](smem_val[tid + stride][0])
            var mine_v = Scalar[dtype](smem_val[tid][0])
            if other_v > mine_v:
                smem_val[tid] = other_v
                smem_idx[tid] = smem_idx[tid + stride]
        barrier()
        stride >>= 1

    var selected_s = Int(smem_idx[0])
    barrier()

    # Phase 3: thread 0..ACTION_DIM copies action, adds noise, clamps.
    if tid < ACTION_DIM:
        var a = tid
        var act_idx = (
            act_off
            + selected_s * HORIZON * ACTION_DIM
            + 0 * ACTION_DIM  # t = 0
            + a
        )
        var act_val = Scalar[dtype](all_actions[act_idx][0])

        if deterministic == UInt32(0):
            var noise_philox = PhiloxRandom(
                seed=(
                    UInt64(rng_seed)
                    + UInt64(0xA5A5A5A5)
                    + UInt64(env_idx) * UInt64(ACTION_DIM)
                    + UInt64(a)
                ),
                offset=0,
            )
            var noise_vals = noise_philox.step_uniform()
            var u1 = (
                Scalar[DType.float32](noise_vals[0])
                + Scalar[DType.float32](1e-8)
            )
            var u2 = Scalar[DType.float32](noise_vals[1])
            var mag = sqrt(Float32(-2.0) * log(u1))
            var z = mag * cos(u2 * Scalar[DType.float32](6.283185307179586))
            var std_val = Scalar[dtype](
                std[env_idx * HORIZON * ACTION_DIM + 0 * ACTION_DIM + a][0]
            )
            act_val = act_val + std_val * Scalar[dtype](z)

        act_val = act_val * action_scale
        if act_val > action_scale:
            act_val = action_scale
        if act_val < -action_scale:
            act_val = -action_scale
        out_action[env_idx * ACTION_DIM + a] = act_val
