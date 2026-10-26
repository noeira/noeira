"""Integrator trait for physics simulation.

Integrators handle velocity and position updates for rigid bodies.
They implement both CPU and GPU execution paths following the nn pattern.
"""

from ..constants import dtype, BODY_STATE_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.core.tiles import Tile


trait Integrator(ImplicitlyCopyable):
    """Trait for velocity and position integration methods.

    Integrators are stateless - they describe the integration algorithm but
    don't store state. All body state is managed externally via LayoutTensor.

    The integration is split into two phases:
    1. integrate_velocities: Apply forces and gravity to update velocities
    2. integrate_positions: Use new velocities to update positions

    This split allows constraint solving between the two phases.

    Layout:
    - bodies: [BATCH, NUM_BODIES, BODY_STATE_SIZE] - row-major, batch-first
    - forces: [BATCH, NUM_BODIES, 3] - (fx, fy, torque) per body
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    def integrate_velocities[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        self,
        bodies: Tile[dtype, BATCH, NUM_BODIES, BODY_STATE_SIZE],
        forces: Tile[dtype, BATCH, NUM_BODIES, 3],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate velocities: v' = v + (F/m + g) * dt.

        Args:
            bodies: Body state tensor [BATCH, NUM_BODIES, BODY_STATE_SIZE].
            forces: Force tensor [BATCH, NUM_BODIES, 3] - (fx, fy, torque).
            gravity_x: Gravity x component.
            gravity_y: Gravity y component.
            dt: Time step.
        """
        ...

    def integrate_positions[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        self,
        bodies: Tile[dtype, BATCH, NUM_BODIES, BODY_STATE_SIZE],
        dt: Scalar[dtype],
    ):
        """Integrate positions: x' = x + v' * dt.

        Uses the NEW velocity (after integrate_velocities) for semi-implicit Euler.

        Args:
            bodies: Body state tensor [BATCH, NUM_BODIES, BODY_STATE_SIZE].
            dt: Time step.
        """
        ...

    # =========================================================================
    # GPU Methods (Static - called without instance)
    # =========================================================================

    @staticmethod
    @staticmethod
    def integrate_velocities_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """GPU kernel launcher for velocity integration.

        Args:
            ctx: GPU device context.
            state_buf: Body state buffer on GPU.
            gravity_x: Gravity x component.
            gravity_y: Gravity y component.
            dt: Time step.
        """
        ...

    @staticmethod
    def integrate_positions_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises:
        """GPU kernel launcher for position integration.

        Args:
            ctx: GPU device context.
            state_buf: Body state buffer on GPU.
            dt: Time step.
        """
        ...
