"""Phase 4 planners: ILQRGPUBatched smoke test.

Compile-and-construct: instantiates ``ILQRGPUBatched`` against a
``LinearQuadratic1DILQRCallback`` adapter that implements
``RolloutJacobianCallbackGPU``. Asserts plan_gpu dispatches without
raising. Numerical correctness comes later — see
``test_ilqr_gpu_lqr_oracle.mojo``.

If a GPU is not available, the test exits silently (CI on CPU-only
machines should still pass).
"""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.planners.trajectory import (
    ILQRGPUBatched,
    RolloutJacobianCallbackGPU,
)


comptime LATENT_DIM: Int = 1
comptime ACTION_DIM: Int = 1
comptime HORIZON: Int = 4
comptime N_ENVS: Int = 2


@fieldwise_init
struct DummyGPUCB(
    Copyable, Movable, Deinitable, RolloutJacobianCallbackGPU
):
    """Trivial gradient-free wrapper to satisfy
    ``RolloutJacobianCallbackGPU`` for a compile-time smoke test.

    Forwards through identity dynamics + zero cost + zero gradients.
    Not numerically meaningful — just exercises the trait surface.
    """

    comptime LATENT_DIM: Int = LATENT_DIM
    comptime ACTION_DIM: Int = ACTION_DIM

    var _placeholder: Int

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        u: Tile[dtype, B, Self.ACTION_DIM],
        z_next_out: Tile[dtype, B, Self.LATENT_DIM],
        cost_out: Tile[dtype, B],
    ) raises:
        pass

    def linearize_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        u: Tile[dtype, B, Self.ACTION_DIM],
        A_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        B_out: Tile[dtype, B, Self.LATENT_DIM, Self.ACTION_DIM],
        l_z_out: Tile[dtype, B, Self.LATENT_DIM],
        l_u_out: Tile[dtype, B, Self.ACTION_DIM],
        l_zz_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        l_uu_out: Tile[dtype, B, Self.ACTION_DIM, Self.ACTION_DIM],
        l_zu_out: Tile[dtype, B, Self.LATENT_DIM, Self.ACTION_DIM],
    ) raises:
        pass

    def terminal_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        V_z_out: Tile[dtype, B, Self.LATENT_DIM],
        V_zz_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        cost_out: Tile[dtype, B],
    ) raises:
        pass


def main() raises:
    comptime if not has_accelerator():
        print("=== ILQRGPUBatched smoke: no GPU, skipping ===")
        return

    print("=== ILQRGPUBatched smoke (compile + construct + dispatch) ===")
    var ctx = DeviceContext()
    var planner = ILQRGPUBatched[
        LATENT_DIM, ACTION_DIM, HORIZON, N_ENVS
    ](ctx, n_iters=2)

    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * LATENT_DIM)
    z0_buf.enqueue_fill(0.0)
    var z0 = tile[N_ENVS, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())

    var cb = DummyGPUCB(_placeholder=0)
    planner.plan_gpu(ctx, cb, z0)
    ctx.synchronize()
    print("  PASS plan_gpu dispatch completes without raising")
    print("OK")
