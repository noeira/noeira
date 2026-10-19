"""Phase 4 planners: ILQRGPUBatched on LinearQuadratic1D — oracle test.

Numerical correctness check for the batched GPU iLQR. Drives
``ILQRGPUBatched[LATENT=1, ACTION=1, HORIZON, N_ENVS=2]`` against a
hand-written GPU callback adapter for the same 1-D LQ problem the CPU
oracle uses, then asserts every env's first action ``U[0]`` matches
``-K_0·z_0`` from the finite-horizon Riccati recursion to high
precision.

The callback adapter writes its dynamics + cost Jacobians directly
from constants via tiny one-thread-per-batch-row GPU kernels — no
neural network involved. iLQR's job is to reduce the trajectory cost;
on an LQ problem it should converge in **one outer iteration** at
``α = 1`` because the quadratic approximation is exact.

If a GPU is not available, the test exits silently (skips on
CPU-only machines).
"""

from std.sys import has_accelerator
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import abs as math_abs
from std.testing import assert_true
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.planners.trajectory import (
    ILQRGPUBatched,
    RolloutJacobianCallbackGPU,
)


comptime LATENT_DIM: Int = 1
comptime ACTION_DIM: Int = 1
comptime HORIZON: Int = 8
comptime N_ENVS: Int = 2


# ── 1D LQ step kernel: z_next = A·z + B·u, cost = Q·z² + R·u² ─────────


def lq_step_kernel[
    dtype: DType,
    B: Int,
](
    z: Tile[dtype, B, LATENT_DIM],
    u: Tile[dtype, B, ACTION_DIM],
    z_next_out: Tile[dtype, B, LATENT_DIM],
    cost_out: Tile[dtype, B],
    A_c: Scalar[dtype],
    B_c: Scalar[dtype],
    Q_c: Scalar[dtype],
    R_c: Scalar[dtype],
):
    var i = Int(block_idx.x * thread_idx.x + thread_idx.x)
    if i >= B:
        return
    var z_v = rebind[Scalar[dtype]](z[i, 0])
    var u_v = rebind[Scalar[dtype]](u[i, 0])
    z_next_out[i, 0] = rebind[z_next_out.ElementType](A_c * z_v + B_c * u_v)
    cost_out[i] = rebind[cost_out.ElementType](
        Q_c * z_v * z_v + R_c * u_v * u_v
    )


# ── 1D LQ linearize kernel: constants for A/B/l_zz/l_uu, l_z/l_u at z/u ─


def lq_linearize_kernel[
    dtype: DType,
    B: Int,
](
    z: Tile[dtype, B, LATENT_DIM],
    u: Tile[dtype, B, ACTION_DIM],
    A_out: Tile[dtype, B, LATENT_DIM, LATENT_DIM],
    B_out: Tile[dtype, B, LATENT_DIM, ACTION_DIM],
    l_z_out: Tile[dtype, B, LATENT_DIM],
    l_u_out: Tile[dtype, B, ACTION_DIM],
    l_zz_out: Tile[dtype, B, LATENT_DIM, LATENT_DIM],
    l_uu_out: Tile[dtype, B, ACTION_DIM, ACTION_DIM],
    l_zu_out: Tile[dtype, B, LATENT_DIM, ACTION_DIM],
    A_c: Scalar[dtype],
    B_c: Scalar[dtype],
    Q_c: Scalar[dtype],
    R_c: Scalar[dtype],
):
    var i = Int(block_idx.x * thread_idx.x + thread_idx.x)
    if i >= B:
        return
    var z_v = rebind[Scalar[dtype]](z[i, 0])
    var u_v = rebind[Scalar[dtype]](u[i, 0])
    A_out[i, 0, 0] = rebind[A_out.ElementType](A_c)
    B_out[i, 0, 0] = rebind[B_out.ElementType](B_c)
    l_z_out[i, 0] = rebind[l_z_out.ElementType](Scalar[dtype](2.0) * Q_c * z_v)
    l_u_out[i, 0] = rebind[l_u_out.ElementType](Scalar[dtype](2.0) * R_c * u_v)
    l_zz_out[i, 0, 0] = rebind[l_zz_out.ElementType](Scalar[dtype](2.0) * Q_c)
    l_uu_out[i, 0, 0] = rebind[l_uu_out.ElementType](Scalar[dtype](2.0) * R_c)
    l_zu_out[i, 0, 0] = rebind[l_zu_out.ElementType](Scalar[dtype](0.0))


# ── 1D LQ terminal kernel: cost = Q_T·z², V_z = 2·Q_T·z, V_zz = 2·Q_T ──


def lq_terminal_kernel[
    dtype: DType,
    B: Int,
](
    z: Tile[dtype, B, LATENT_DIM],
    V_z_out: Tile[dtype, B, LATENT_DIM],
    V_zz_out: Tile[dtype, B, LATENT_DIM, LATENT_DIM],
    cost_out: Tile[dtype, B],
    Q_T_c: Scalar[dtype],
):
    var i = Int(block_idx.x * thread_idx.x + thread_idx.x)
    if i >= B:
        return
    var z_v = rebind[Scalar[dtype]](z[i, 0])
    V_z_out[i, 0] = rebind[V_z_out.ElementType](
        Scalar[dtype](2.0) * Q_T_c * z_v
    )
    V_zz_out[i, 0, 0] = rebind[V_zz_out.ElementType](
        Scalar[dtype](2.0) * Q_T_c
    )
    cost_out[i] = rebind[cost_out.ElementType](Q_T_c * z_v * z_v)


# ── Callback ──


@fieldwise_init
struct LQ1DGPUCallback(
    Copyable, Movable, Deinitable, RolloutJacobianCallbackGPU
):
    """Hand-rolled GPU adapter for 1-D LQ + iLQR oracle test."""

    comptime LATENT_DIM: Int = LATENT_DIM
    comptime ACTION_DIM: Int = ACTION_DIM

    var A: Float64
    var B: Float64
    var Q: Float64
    var R: Float64
    var Q_T: Float64

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        u: Tile[dtype, B, Self.ACTION_DIM],
        z_next_out: Tile[dtype, B, Self.LATENT_DIM],
        cost_out: Tile[dtype, B],
    ) raises:
        ctx.enqueue_function[lq_step_kernel[dtype, B]](
            z,
            u,
            z_next_out,
            cost_out,
            Scalar[dtype](self.A),
            Scalar[dtype](self.B),
            Scalar[dtype](self.Q),
            Scalar[dtype](self.R),
            grid_dim=1,
            block_dim=B,
        )

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
        ctx.enqueue_function[lq_linearize_kernel[dtype, B]](
            z,
            u,
            A_out,
            B_out,
            l_z_out,
            l_u_out,
            l_zz_out,
            l_uu_out,
            l_zu_out,
            Scalar[dtype](self.A),
            Scalar[dtype](self.B),
            Scalar[dtype](self.Q),
            Scalar[dtype](self.R),
            grid_dim=1,
            block_dim=B,
        )

    def terminal_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        V_z_out: Tile[dtype, B, Self.LATENT_DIM],
        V_zz_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        cost_out: Tile[dtype, B],
    ) raises:
        ctx.enqueue_function[lq_terminal_kernel[dtype, B]](
            z,
            V_z_out,
            V_zz_out,
            cost_out,
            Scalar[dtype](self.Q_T),
            grid_dim=1,
            block_dim=B,
        )


def finite_horizon_first_gain(
    A: Float64,
    B: Float64,
    Q: Float64,
    R: Float64,
    Q_T: Float64,
    T: Int,
) -> Float64:
    """Closed-form first-action LQR gain (cf. CPU LinearQuadratic1DILQRCallback).
    """
    var P = Q_T
    for _ in range(T - 1):
        var denom = R + B * B * P
        P = Q + A * A * P - (A * B * P) * (A * B * P) / denom
    var denom = R + B * B * P
    return (B * P * A) / denom


def test_ilqr_gpu_matches_lqr_first_action() raises:
    var ctx = DeviceContext()

    var A: Float64 = 0.9
    var B: Float64 = 1.0
    var Q: Float64 = 1.0
    var R: Float64 = 0.1
    var Q_T: Float64 = 1.0

    var planner = ILQRGPUBatched[
        LATENT_DIM, ACTION_DIM, HORIZON, N_ENVS
    ](ctx, n_iters=3, mu_init=1e-3)

    # Per-env z0 = [1.0, 0.5] so we exercise non-trivial state across envs.
    var z0_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * LATENT_DIM)
    z0_host[0] = 1.0
    z0_host[1] = 0.5
    var z0_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * LATENT_DIM)
    ctx.enqueue_copy(z0_buf, z0_host)
    var z0 = tile[N_ENVS, LATENT_DIM](z0_buf.unsafe_ptr().as_unsafe_any_origin())

    var cb = LQ1DGPUCallback(A=A, B=B, Q=Q, R=R, Q_T=Q_T)
    planner.plan_gpu(ctx, cb, z0)
    ctx.synchronize()

    # Read back the optimized controls U[0, *, 0] (first timestep per env).
    var U_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * HORIZON * ACTION_DIM
    )
    ctx.enqueue_copy(U_host, planner.U_buf)
    ctx.synchronize()

    var K0 = finite_horizon_first_gain(A, B, Q, R, Q_T, HORIZON)
    # Timestep-major layout: U[t=0, e, 0] lives at offset e * ACTION_DIM.
    var z0_per_env = List[Float64](length=N_ENVS, fill=0.0)
    z0_per_env[0] = 1.0
    z0_per_env[1] = 0.5
    for e in range(N_ENVS):
        var got = Float64(U_host[e * ACTION_DIM + 0])
        var expected = -K0 * z0_per_env[e]
        var err = math_abs(got - expected)
        assert_true(
            err < 1e-4,
            "env "
            + String(e)
            + ": iLQR GPU U[0] = "
            + String(got)
            + " ≠ LQR -K0*z0 = "
            + String(expected)
            + " (err "
            + String(err)
            + ")",
        )


def main() raises:
    comptime if not has_accelerator():
        print("=== ILQRGPUBatched LQ oracle: no GPU, skipping ===")
        return
    print("=== Phase 4 planners: ILQRGPUBatched on LinearQuadratic1D ===")
    test_ilqr_gpu_matches_lqr_first_action()
    print("  PASS iLQR GPU U[0] matches LQR -K0*z0 (all envs) within 1e-4")
    print("OK")
