"""Phase 2 planners: MPPIGPUBatched goal-reach test.

GPU counterpart to ``test_mppi_goal_reach.mojo``. Same stub world
(``IdentityDynamics`` + ``GoalReachReward``, ``z' = z + a``,
``r = -‖z' - goal‖²``) lifted to the ``RolloutCallbackGPU`` trait:
trait methods are implemented by small inline kernels rather than
trait-CPU list operations.

The point is to validate the **full GPU pipeline** (per-iter helper
+ all planner kernels + callback's three trait methods + GPU
multinomial action selection) against a closed-form-optimal target
on real device hardware. Confirms:
  * sample → rollout → accumulate → softmax → refit cycle is wired
    correctly end-to-end on device
  * the trait-bound LayoutTensor types unify across planner and
    callback (the ``rebind`` machinery in ``_run_mppi_iteration``)
  * episode-reset state (``planner.env_t0_flags``) round-trips

The single-env case (``N_ENVS = 1``) is the primary test. A small
multi-env case (``N_ENVS = 2``) exercises the truly-batched code
path: two envs with different goals planned in one kernel grid.

Usage:
    pixi run -e apple  mojo run -I . tests/planners/trajectory/test_mppi_gpu_goal_reach.mojo
    pixi run -e nvidia mojo run -I . tests/planners/trajectory/test_mppi_gpu_goal_reach.mojo
"""

from std.math import abs as math_abs
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random import seed as _set_seed
from std.testing import assert_true
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype, TPB
from noeira.planners.trajectory import MPPIGPUBatched, RolloutCallbackGPU


# Comptime sizes for this test — chosen so MPPI converges tightly
# and symmetrically across dims. 128 samples × 6 iters is too few:
# both CPU and GPU show dim-asymmetric convergence (spread > 0.20)
# at that budget, which made earlier diagnostics look like a GPU
# bug when it's actually MPPI's natural variance at low sample
# counts. With 512 samples × 12 iters all 3 dims land in (1.0 ±
# 0.15) consistently on both backends.
comptime LATENT_DIM: Int = 3
comptime ACTION_DIM: Int = 3
comptime HORIZON: Int = 3
comptime NUM_SAMPLES: Int = 512
comptime NUM_PI_TRAJS: Int = 0
comptime NUM_ELITES: Int = 64
comptime NUM_ITERATIONS: Int = 12


# =============================================================================
# Stub callback kernels
# =============================================================================


@always_inline
def _zero_action_kernel[
    dtype: DType, B: Int, ACT_DIM: Int
](
    action_out: Tile[dtype, B, ACT_DIM],
) where dtype.is_floating_point():
    """Write zeros into ``action_out`` — the stub has no learned policy
    so ``policy_action_gpu`` degenerates to zero."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= B:
        return
    for j in range(ACT_DIM):
        action_out[i, j] = Scalar[dtype](0.0)


@always_inline
def _goal_reach_rollout_step_kernel[
    dtype: DType,
    B: Int,
    DIM: Int,  # = LATENT_DIM == ACTION_DIM (IdentityDynamics requires equality)
    TOTAL_SAMPLES: Int,
    N_ENVS: Int,
](
    z: Tile[dtype, B, DIM],
    a: Tile[dtype, B, DIM],
    z_next_out: Tile[dtype, B, DIM],
    r_out: Tile[dtype, B],
    goals: Tile[dtype, N_ENVS, DIM],
) where dtype.is_floating_point():
    """One-step IdentityDynamics + per-env goal-reach reward.

    Each batch row ``i`` belongs to env ``i // TOTAL_SAMPLES`` (matches
    the planner's row layout). The reward uses that env's goal vector
    so the multi-env case has env-specific optima.

    Requires LATENT_DIM == ACTION_DIM (collapsed to ``DIM`` here)
    because IdentityDynamics ``z' = z + a`` only makes geometric sense
    when the two have equal dimensionality.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= B:
        return
    var env_idx = i // TOTAL_SAMPLES
    var sq_dist = Scalar[dtype](0.0)
    for k in range(DIM):
        var zk = Scalar[dtype](z[i, k][0]) + Scalar[dtype](a[i, k][0])
        z_next_out[i, k] = zk
        var diff = zk - Scalar[dtype](goals[env_idx, k][0])
        sq_dist = sq_dist + diff * diff
    r_out[i] = -sq_dist


@always_inline
def _zero_terminal_kernel[
    dtype: DType, B: Int
](
    v_out: Tile[dtype, B],
) where dtype.is_floating_point():
    """No bootstrap in the stub — write zeros into ``v_out``."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= B:
        return
    v_out[i] = Scalar[dtype](0.0)


# =============================================================================
# Stub callback struct
# =============================================================================


@fieldwise_init
struct GoalReachGPUCallback[
    LATENT_DIM_PARAM: Int,
    ACTION_DIM_PARAM: Int,
    N_ENVS: Int,
    TOTAL_SAMPLES: Int,
](Movable, Deinitable, RolloutCallbackGPU):
    """``RolloutCallbackGPU`` against IdentityDynamics + GoalReachReward.

    Holds a device buffer of per-env goals (shape
    ``(N_ENVS, LATENT_DIM)``). Caller initializes it before calling
    ``planner.plan_gpu``.
    """

    comptime LATENT_DIM: Int = Self.LATENT_DIM_PARAM
    comptime ACTION_DIM: Int = Self.ACTION_DIM_PARAM

    var goals_buf: DeviceBuffer[dtype]
    """`(N_ENVS, LATENT_DIM)` — per-env goal vectors. Caller owns
    initialization."""

    def policy_action_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        action_out: Tile[dtype, B, Self.ACTION_DIM],
    ) raises:
        comptime kernel = _zero_action_kernel[dtype, B, Self.ACTION_DIM]
        comptime BLOCKS = (B + TPB - 1) // TPB
        ctx.enqueue_function[kernel](
            action_out,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    def rollout_step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        a: Tile[dtype, B, Self.ACTION_DIM],
        z_next_out: Tile[dtype, B, Self.LATENT_DIM],
        r_out: Tile[dtype, B],
    ) raises:
        comptime assert Self.LATENT_DIM == Self.ACTION_DIM, (
            "GoalReachGPUCallback requires LATENT_DIM == ACTION_DIM"
        )
        comptime kernel = _goal_reach_rollout_step_kernel[
            dtype,
            B,
            Self.LATENT_DIM,
            Self.TOTAL_SAMPLES,
            Self.N_ENVS,
        ]
        comptime BLOCKS = (B + TPB - 1) // TPB
        var goals_view = tile[Self.N_ENVS, Self.LATENT_DIM](self.goals_buf.unsafe_ptr().as_unsafe_any_origin())
        ctx.enqueue_function[kernel](
            z,
            a,
            z_next_out,
            r_out,
            goals_view,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    def terminal_value_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        v_out: Tile[dtype, B],
        seed: UInt32,
    ) raises:
        comptime kernel = _zero_terminal_kernel[dtype, B]
        comptime BLOCKS = (B + TPB - 1) // TPB
        ctx.enqueue_function[kernel](
            v_out,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


# =============================================================================
# Single-env test
# =============================================================================


comptime N_ENVS_1: Int = 1
comptime TOTAL_SAMPLES_1: Int = NUM_SAMPLES + NUM_PI_TRAJS


def test_mppi_gpu_converges_to_goal() raises:
    """N_ENVS=1, goal=(1,1,1), z0=(0,0,0). MPPI's selected first
    action (downloaded back to host) should land close to
    ``a* = goal - z0 = (1, 1, 1)``.

    Tolerance is 0.25 per component — looser than the CPU goal-reach
    test (0.20) because GPU action selection is gumbel-max over the
    elite softmax (``mppi_select_action_kernel``) and adds one
    additional layer of sample variance on top of the converged
    distribution. The companion diagnostic test
    (``test_mppi_gpu_mean_converges_to_goal``) verifies the
    underlying Gaussian mean converges tightly (within 0.06 of
    optimum, symmetric across dims) — so any failure here at 0.25
    indicates the gumbel-max sampler is picking from a much wider
    tail than expected.
    """
    _set_seed(0xCAFE)
    var ctx = DeviceContext()

    # Goals: single env, goal = (1, 1, 1).
    var goals_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    var goals_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    goals_host[0] = Scalar[dtype](1.0)
    goals_host[1] = Scalar[dtype](1.0)
    goals_host[2] = Scalar[dtype](1.0)
    ctx.enqueue_copy(goals_buf, goals_host)

    var cb = GoalReachGPUCallback[
        LATENT_DIM, ACTION_DIM, N_ENVS_1, TOTAL_SAMPLES_1
    ](goals_buf=goals_buf^)

    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS_1,
    ](ctx)

    # z0 on device: (0, 0, 0).
    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS_1 * LATENT_DIM)
    var z0_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    for k in range(N_ENVS_1 * LATENT_DIM):
        z0_host[k] = Scalar[dtype](0.0)
    ctx.enqueue_copy(z0_buf, z0_host)
    var z0_tensor = tile[N_ENVS_1, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())

    # Output action buffer.
    var out_act_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_1 * ACTION_DIM
    )
    var out_act_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * ACTION_DIM
    )
    var out_act_tensor = tile[N_ENVS_1 * ACTION_DIM](out_act_buf.unsafe_ptr().as_unsafe_any_origin())

    planner.plan_gpu(
        ctx,
        cb,
        z0_tensor,
        out_act_tensor,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
        rng_base_seed=UInt32(0xC0FFEE),
    )
    ctx.enqueue_copy(out_act_host, out_act_buf)
    ctx.synchronize()

    for i in range(ACTION_DIM):
        var act = Float64(out_act_host[i])
        var err = math_abs(act - 1.0)
        assert_true(
            err < 0.25,
            "GPU MPPI first action[" + String(i) + "] = "
            + String(act)
            + " not within 0.25 of optimal 1.0 (err = "
            + String(err)
            + ")",
        )


def test_mppi_gpu_mean_converges_to_goal() raises:
    """**Diagnostic** test — cross-checks that the *converged
    distribution mean* (not the gumbel-max-sampled action) lands
    tightly on the optimum (1, 1, 1).

    Why this test exists: ``test_mppi_gpu_converges_to_goal`` uses a
    loose 0.5 tolerance because GPU action selection is gumbel-max
    over the elite softmax — high single-sample variance even when
    the underlying distribution has converged. This test bypasses
    that by reading ``planner.env_prev_means[0]`` directly after
    ``plan_gpu``, which holds the post-iteration converged Gaussian
    mean. If MPPI is working correctly the step-0 mean should be
    within ~0.1 of (1, 1, 1) — comparable to the CPU tolerance.

    A failure here means the GPU optimization itself is broken
    (kernel bug, RNG bias, refit error). A passing here together
    with a "loose" action assertion means the variance is purely
    from sample selection, not from the underlying distribution.
    """
    _set_seed(0xD1A6)
    var ctx = DeviceContext()

    var goals_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    var goals_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    for k in range(LATENT_DIM):
        goals_host[k] = Scalar[dtype](1.0)
    ctx.enqueue_copy(goals_buf, goals_host)

    var cb = GoalReachGPUCallback[
        LATENT_DIM, ACTION_DIM, N_ENVS_1, TOTAL_SAMPLES_1
    ](goals_buf=goals_buf^)

    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS_1,
    ](ctx)

    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS_1 * LATENT_DIM)
    var z0_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    for k in range(N_ENVS_1 * LATENT_DIM):
        z0_host[k] = Scalar[dtype](0.0)
    ctx.enqueue_copy(z0_buf, z0_host)
    var z0_tensor = tile[N_ENVS_1, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())

    var out_act_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_1 * ACTION_DIM
    )
    var out_act_tensor = tile[N_ENVS_1 * ACTION_DIM](out_act_buf.unsafe_ptr().as_unsafe_any_origin())

    planner.plan_gpu(
        ctx,
        cb,
        z0_tensor,
        out_act_tensor,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
        rng_base_seed=UInt32(0xD1A6),
    )
    ctx.synchronize()

    # After plan_gpu, planner.env_prev_means[0] holds the converged
    # mean for env 0. Layout: (HORIZON, ACTION_DIM) flat.
    # The step-0 mean (first ACTION_DIM entries) is what MPPI
    # iteratively optimized — should be close to (1, 1, 1).
    #
    # Cross-check 1: by symmetry, all 3 dims should converge to the
    # same value (the setup has goal=(1,1,1), z0=(0,0,0), identical
    # init mean/std for all dims). A spread > 0.15 across dims
    # signals dim-asymmetric bias — a real bug.
    var means: List[Float64] = [0.0, 0.0, 0.0]
    for k in range(ACTION_DIM):
        means[k] = planner.env_prev_means[0][k]
        print(
            "  diagnostic: mean[step=0, dim=" + String(k) + "] = "
            + String(means[k])
        )
    # Spread across dims (max - min) — should be small by symmetry.
    var mmin = means[0]
    var mmax = means[0]
    for k in range(1, ACTION_DIM):
        if means[k] < mmin:
            mmin = means[k]
        if means[k] > mmax:
            mmax = means[k]
    var spread = mmax - mmin
    print("  diagnostic: cross-dim spread = " + String(spread))
    assert_true(
        spread < 0.10,
        "Converged-mean spread across dims = " + String(spread)
        + " > 0.10. By symmetry (goal=(1,1,1), z0=(0,0,0)) all dims"
        + " should converge identically. With 512×12 budget the"
        + " observed spread is ~0.03; > 0.10 indicates a structural"
        + " RNG / kernel bias.",
    )
    # Cross-check 2: all dims close to optimum.
    for k in range(ACTION_DIM):
        var err = math_abs(means[k] - 1.0)
        assert_true(
            err < 0.10,
            "converged mean[step=0, dim=" + String(k) + "] = "
            + String(means[k])
            + " is not within 0.10 of optimum 1.0 (err = "
            + String(err) + ").",
        )


def test_mppi_gpu_at_goal_stays() raises:
    """N_ENVS=1, z0 = goal = (1,1,1). Optimal action is the zero
    vector; assert each component has |a| ≤ 0.30, matching the CPU
    at-goal tolerance. With 512 samples × 12 iters the converged
    mean concentrates around 0, and the gumbel-max sample stays in
    a tight ±0.30 band.
    """
    _set_seed(0x60A1)
    var ctx = DeviceContext()

    var goals_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    var goals_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    for k in range(LATENT_DIM):
        goals_host[k] = Scalar[dtype](1.0)
    ctx.enqueue_copy(goals_buf, goals_host)

    var cb = GoalReachGPUCallback[
        LATENT_DIM, ACTION_DIM, N_ENVS_1, TOTAL_SAMPLES_1
    ](goals_buf=goals_buf^)

    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS_1,
    ](ctx)

    # z0 = goal = (1, 1, 1).
    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS_1 * LATENT_DIM)
    var z0_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * LATENT_DIM
    )
    for k in range(N_ENVS_1 * LATENT_DIM):
        z0_host[k] = Scalar[dtype](1.0)
    ctx.enqueue_copy(z0_buf, z0_host)
    var z0_tensor = tile[N_ENVS_1, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())

    var out_act_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_1 * ACTION_DIM
    )
    var out_act_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_1 * ACTION_DIM
    )
    var out_act_tensor = tile[N_ENVS_1 * ACTION_DIM](out_act_buf.unsafe_ptr().as_unsafe_any_origin())

    planner.plan_gpu(
        ctx,
        cb,
        z0_tensor,
        out_act_tensor,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
        rng_base_seed=UInt32(0xBEEF),
    )
    ctx.enqueue_copy(out_act_host, out_act_buf)
    ctx.synchronize()

    for i in range(ACTION_DIM):
        var act = Float64(out_act_host[i])
        assert_true(
            math_abs(act) < 0.30,
            "GPU MPPI at-goal action[" + String(i) + "] = "
            + String(act)
            + " should be near 0",
        )


# =============================================================================
# Multi-env test — exercises the truly-batched code path
# =============================================================================


comptime N_ENVS_2: Int = 2
comptime TOTAL_SAMPLES_2: Int = NUM_SAMPLES + NUM_PI_TRAJS


def test_mppi_gpu_two_envs_distinct_goals() raises:
    """Two envs with different goals planned in one kernel grid.

    env 0: z0=(0,0,0), goal=( 1, 1, 1) → optimal first action ≈ ( 1, 1, 1)
    env 1: z0=(0,0,0), goal=(-1,-1,-1) → optimal first action ≈ (-1,-1,-1)

    Exercises the batched code path properly: ``BATCH_TOTAL = N_ENVS *
    TOTAL_SAMPLES`` rows in the rollout kernels, two distinct softmax
    reductions per ``mppi_softmax_weights_kernel`` block, two distinct
    refits in ``mppi_weighted_mean_std_kernel``.
    """
    _set_seed(0xBA7CE)
    var ctx = DeviceContext()

    var goals_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_2 * LATENT_DIM
    )
    var goals_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_2 * LATENT_DIM
    )
    # env 0: goal = (+1, +1, +1)
    for k in range(LATENT_DIM):
        goals_host[k] = Scalar[dtype](1.0)
    # env 1: goal = (-1, -1, -1)
    for k in range(LATENT_DIM):
        goals_host[LATENT_DIM + k] = Scalar[dtype](-1.0)
    ctx.enqueue_copy(goals_buf, goals_host)

    var cb = GoalReachGPUCallback[
        LATENT_DIM, ACTION_DIM, N_ENVS_2, TOTAL_SAMPLES_2
    ](goals_buf=goals_buf^)

    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS_2,
    ](ctx)

    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS_2 * LATENT_DIM)
    var z0_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_2 * LATENT_DIM
    )
    for k in range(N_ENVS_2 * LATENT_DIM):
        z0_host[k] = Scalar[dtype](0.0)
    ctx.enqueue_copy(z0_buf, z0_host)
    var z0_tensor = tile[N_ENVS_2, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())

    var out_act_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS_2 * ACTION_DIM
    )
    var out_act_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS_2 * ACTION_DIM
    )
    var out_act_tensor = tile[N_ENVS_2 * ACTION_DIM](out_act_buf.unsafe_ptr().as_unsafe_any_origin())

    planner.plan_gpu(
        ctx,
        cb,
        z0_tensor,
        out_act_tensor,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
        rng_base_seed=UInt32(0xBABE2024),
    )
    ctx.enqueue_copy(out_act_host, out_act_buf)
    ctx.synchronize()

    # env 0: each component close to +1.0 (goal direction)
    for i in range(ACTION_DIM):
        var act = Float64(out_act_host[i])
        assert_true(
            math_abs(act - 1.0) < 0.25,
            "env 0 action[" + String(i) + "] = " + String(act)
            + " not within 0.25 of +1.0",
        )

    # env 1: each component close to -1.0 (goal direction)
    for i in range(ACTION_DIM):
        var act = Float64(out_act_host[ACTION_DIM + i])
        assert_true(
            math_abs(act - (-1.0)) < 0.25,
            "env 1 action[" + String(i) + "] = " + String(act)
            + " not within 0.25 of -1.0",
        )


def test_mppi_gpu_start_episode_resets_warm_start() raises:
    """``planner.start_episode(env_idx)`` should reset that env's
    warm-start flag. Verifies the contract directly via
    ``planner.env_t0_flags`` without re-running a plan.
    """
    var ctx = DeviceContext()
    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS_2,
    ](ctx)

    # Freshly constructed: both envs t0=True.
    assert_true(
        planner.env_t0_flags[0] and planner.env_t0_flags[1],
        "freshly-constructed planner should have all env_t0=True",
    )

    # Simulate post-first-plan state: both flags False.
    planner.env_t0_flags[0] = False
    planner.env_t0_flags[1] = False

    # Reset env 0 only.
    planner.start_episode(0)
    assert_true(planner.env_t0_flags[0], "start_episode(0) → env 0 t0=True")
    assert_true(
        not planner.env_t0_flags[1],
        "start_episode(0) should not touch env 1",
    )


def main() raises:
    print("=== Phase 2 planners: MPPIGPUBatched goal-reach ===")
    test_mppi_gpu_converges_to_goal()
    print(
        "  PASS GPU MPPI converges to goal"
        " (IdentityDynamics + GoalReachReward, N_ENVS=1)"
    )
    test_mppi_gpu_mean_converges_to_goal()
    print(
        "  PASS GPU MPPI converged-mean diagnostic: mean tight"
        " around (1, 1, 1)"
    )
    test_mppi_gpu_at_goal_stays()
    print("  PASS GPU MPPI at goal selects near-zero action")
    test_mppi_gpu_two_envs_distinct_goals()
    print(
        "  PASS GPU MPPI batched: 2 envs, distinct goals,"
        " correct per-env action"
    )
    test_mppi_gpu_start_episode_resets_warm_start()
    print("  PASS start_episode(env_idx) resets only that env's flag")
    print("OK")
