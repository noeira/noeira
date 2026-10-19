"""Fused GPU kernel for the multi-body (Box2D-faithful) car step.

One thread per environment runs the entire `CarDynamicsMB.step_single_env`
(steering motors + tire forces + integrate + iterative revolute solve, all
sub-stepped) for its car. The car-vs-car work is embarrassingly parallel: no
cross-thread communication, identical to the legacy `CarPhysicsKernel` structure
but driving the multi-body dynamics instead of the single-body model.

Usage:
    CarMBPhysicsKernel.step_gpu[
        BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, JOINTS_OFFSET,
        ROLLING_OFFSET, CONTROLS_OFFSET,
    ](ctx, state_buf, friction_limit, dt)
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from ..constants import dtype, TPB
from .car_multibody import CarDynamicsMB
from noeira.core.tiles import Tile, tile


struct CarMBPhysicsKernel:
    """Fused GPU kernel: one multi-body car step per thread."""

    @always_inline
    @staticmethod
    def _step_kernel[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ROLLING_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        SUBSTEPS: Int,
        VEL_ITERS: Int,
        POS_ITERS: Int,
    ](
        state: Tile[dtype, BATCH, STATE_SIZE],
        friction_limit: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return
        CarDynamicsMB.step_single_env[
            BATCH,
            STATE_SIZE,
            BODIES_OFFSET,
            FORCES_OFFSET,
            JOINTS_OFFSET,
            ROLLING_OFFSET,
            CONTROLS_OFFSET,
            SUBSTEPS,
            VEL_ITERS,
            POS_ITERS,
        ](env, state, friction_limit, dt)

    @staticmethod
    def step_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ROLLING_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        SUBSTEPS: Int = CarDynamicsMB.DEFAULT_SUBSTEPS,
        VEL_ITERS: Int = CarDynamicsMB.DEFAULT_VEL_ITERS,
        POS_ITERS: Int = CarDynamicsMB.DEFAULT_POS_ITERS,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        friction_limit: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch one multi-body car step for all BATCH environments.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE] with the car sub-blocks
                (bodies/forces/joints/rolling/controls) at the given offsets.
            friction_limit: Surface friction limit (FRICTION_LIMIT * road/grass).
            dt: Frame time step.
        """
        var state = tile[BATCH, STATE_SIZE](state_buf)

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
            state: Tile[dtype, BATCH, STATE_SIZE],
            friction_limit: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            CarMBPhysicsKernel._step_kernel[
                BATCH,
                STATE_SIZE,
                BODIES_OFFSET,
                FORCES_OFFSET,
                JOINTS_OFFSET,
                ROLLING_OFFSET,
                CONTROLS_OFFSET,
                SUBSTEPS,
                VEL_ITERS,
                POS_ITERS,
            ](state, friction_limit, dt)

        ctx.enqueue_function[kernel_wrapper](
            state,
            friction_limit,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
