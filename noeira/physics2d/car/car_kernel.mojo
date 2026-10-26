"""Fused GPU kernel for CarRacing physics step.

This module provides a fully fused physics step kernel that runs:
1. Steering joint updates
2. Friction zone lookup for each wheel
3. Wheel force computation (slip-based friction)
4. Hull dynamics integration
5. Observation update

ALL IN ONE KERNEL LAUNCH.

Unlike the impulse-based PhysicsStepKernel used for LunarLander, this kernel
is specific to top-down car physics with slip-based tire friction.

Example usage:
    ```mojo
    CarPhysicsKernel.step_gpu[BATCH, Layout](
        ctx, state_buf, tiles_buf, num_active_tiles, dt
    )
    ```
"""

from layout import Layout
from noeira.core.tiles import Tile, tile
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from .constants import (
    TILE_DATA_SIZE,
    MAX_TRACK_TILES,
    CAR_DT,
)
from .layout import CarRacingLayout
from .car_dynamics import CarDynamics

from ..constants import dtype, TPB, erase_origin


struct CarPhysicsKernel:
    """Fused GPU kernel for CarRacing physics step.

    This kernel performs the entire car physics simulation in one launch:
    1. Update steering joints (front wheels)
    2. Lookup friction limits from track tiles
    3. Compute wheel forces (slip-based friction)
    4. Integrate hull dynamics
    5. Update observation vector

    Each thread processes one environment independently.
    """

    # =========================================================================
    # Fused Physics Step Kernel
    # =========================================================================

    @always_inline
    @staticmethod
    def _step_kernel[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        state: Tile[dtype, BATCH, STATE_SIZE],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """GPU kernel that runs the entire car physics step.

        Each thread processes one environment.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        # Full physics step with observation update
        CarDynamics.step_with_obs[
            BATCH,
            STATE_SIZE,
            OBS_OFFSET,
            OBS_DIM,
            HULL_OFFSET,
            WHEELS_OFFSET,
            CONTROLS_OFFSET,
            MAX_TILES,
        ](env, state, tiles, num_active_tiles, dt)

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    def step_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        tiles_buf: DeviceBuffer[dtype],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ) raises:
        """Launch car physics step kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            tiles_buf: Track tiles buffer [MAX_TILES * TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles in buffer.
            dt: Time step.
        """
        var state = tile[BATCH, STATE_SIZE](erase_origin(state_buf.unsafe_ptr()))

        var tiles = tile[MAX_TILES, TILE_DATA_SIZE](erase_origin(tiles_buf.unsafe_ptr()))

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
            state: Tile[dtype, BATCH, STATE_SIZE],
            tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
            num_active_tiles_arg: Int64,
            dt: Scalar[dtype],
        ):
            # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
            # a fixed-width `Int64` and re-binds the original name here.
            var num_active_tiles = Int(num_active_tiles_arg)
            CarPhysicsKernel._step_kernel[
                BATCH,
                STATE_SIZE,
                OBS_OFFSET,
                OBS_DIM,
                HULL_OFFSET,
                WHEELS_OFFSET,
                CONTROLS_OFFSET,
                MAX_TILES,
            ](state, tiles, num_active_tiles, dt)

        ctx.enqueue_function[kernel_wrapper](
            state,
            tiles,
            Int64(num_active_tiles),
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Layout-Based API (Convenience)
    # =========================================================================

    @staticmethod
    def step_gpu_with_layout[
        BATCH: Int,
        Layout: CarRacingLayout,
        MAX_TILES: Int = MAX_TRACK_TILES,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        tiles_buf: DeviceBuffer[dtype],
        num_active_tiles: Int,
        dt: Scalar[dtype] = Scalar[dtype](CAR_DT),
    ) raises:
        """Launch car physics step using CarRacingLayout.

        This is a convenience method that extracts layout constants automatically.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * Layout.STATE_SIZE].
            tiles_buf: Track tiles buffer [MAX_TILES * TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles.
            dt: Time step (defaults to CAR_DT = 0.02).
        """
        CarPhysicsKernel.step_gpu[
            BATCH,
            Layout.STATE_SIZE,
            Layout.OBS_OFFSET,
            Layout.OBS_DIM,
            Layout.HULL_OFFSET,
            Layout.WHEELS_OFFSET,
            Layout.CONTROLS_OFFSET,
            MAX_TILES,
        ](ctx, state_buf, tiles_buf, num_active_tiles, dt)


struct CarPhysicsKernelParallel:
    """Parallel version using multiple threads per environment.

    For environments with many track tiles, we can parallelize the
    tile collision lookup across multiple threads.

    NOTE: This is a more complex optimization that may be added later.
    For now, the single-thread-per-env version is sufficient for most
    batch sizes (tile lookup is O(num_tiles) per wheel, ~1200 lookups
    total per env which is fast on modern GPUs).
    """

    pass  # TODO: Implement if needed for very large tracks
