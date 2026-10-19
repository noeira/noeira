"""Phase 2 planners: MPPIGPUBatched compile smoke test.

This test exists ONLY to verify that ``MPPIGPUBatched`` and its
``RolloutCallbackGPU`` trait wiring compile end-to-end. It uses a
stub callback that no-ops the kernel calls (writes zeros). The full
parity / numerics tests live in ``test_tdmpc2_mppi_parity.mojo``
(coming with task 38).

Why a stub instead of jumping straight to TDMPC2: lets us validate
the planner trait machinery in isolation. If TDMPC2's MPPI parity
breaks later, we can disambiguate planner-bug-vs-callback-bug.
"""

from max.gpu.host import DeviceContext
from std.memory import alloc
from std.testing import assert_true
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.planners.trajectory import (
    MPPIGPUBatched,
    RolloutCallbackGPU,
)


@fieldwise_init
struct ZeroRolloutCallback(
    Movable, Deinitable, RolloutCallbackGPU
):
    """Trivial GPU callback that writes zeros everywhere — exists
    only to validate that the trait surface is implementable.
    """

    comptime LATENT_DIM: Int = 2
    comptime ACTION_DIM: Int = 1

    def policy_action_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        action_out: Tile[dtype, B, Self.ACTION_DIM],
    ) raises:
        # No-op stub: leave action_out as-is (zero initialized).
        pass

    def rollout_step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        a: Tile[dtype, B, Self.ACTION_DIM],
        z_next_out: Tile[dtype, B, Self.LATENT_DIM],
        r_out: Tile[dtype, B],
    ) raises:
        pass

    def terminal_value_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        v_out: Tile[dtype, B],
        seed: UInt32,
    ) raises:
        pass


comptime LATENT_DIM: Int = 2
comptime ACTION_DIM: Int = 1
comptime HORIZON: Int = 3
comptime NUM_SAMPLES: Int = 8
comptime NUM_PI_TRAJS: Int = 2
comptime NUM_ELITES: Int = 4
comptime NUM_ITERATIONS: Int = 2
comptime N_ENVS: Int = 1


def test_mppi_gpu_constructs() raises:
    """MPPIGPUBatched should construct against the stub callback
    without errors."""
    var ctx = DeviceContext()
    var planner = MPPIGPUBatched[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ELITES,
        NUM_ITERATIONS,
        N_ENVS,
    ](ctx)
    assert_true(planner.env_t0_flags[0], "freshly-constructed env_t0 should be True")


def test_mppi_gpu_plan_runs() raises:
    """``plan_gpu`` end-to-end runs without throwing — exercises the
    full per-iter helper and every kernel in the pipeline. Returned
    actions are arbitrary (zero callback), the point is the pipeline
    compiles and dispatches.
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
        N_ENVS,
    ](ctx)
    var callback = ZeroRolloutCallback()

    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * LATENT_DIM)
    var out_act_buf = ctx.enqueue_create_buffer[dtype](
        N_ENVS * ACTION_DIM
    )
    var z0_tensor = tile[N_ENVS, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())
    var out_act_tensor = tile[N_ENVS * ACTION_DIM](out_act_buf.unsafe_ptr().as_unsafe_any_origin())

    planner.plan_gpu(
        ctx,
        callback,
        z0_tensor,
        out_act_tensor,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
        rng_base_seed=UInt32(42),
    )
    ctx.synchronize()
    # No assertions on values — stub callback makes returns
    # degenerate to zero. The point of this test is "doesn't crash".
    assert_true(True, "plan_gpu completed without error")


def main() raises:
    print("=== Phase 2 planners: MPPIGPUBatched compile smoke ===")
    test_mppi_gpu_constructs()
    print("  PASS MPPIGPUBatched constructs")
    test_mppi_gpu_plan_runs()
    print("  PASS plan_gpu runs end-to-end with stub callback")
    print("OK")
