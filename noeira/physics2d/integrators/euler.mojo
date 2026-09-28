"""Semi-Implicit Euler integrator implementation.

This integrator follows Box2D's integration order:
1. v(t+dt) = v(t) + a(t) * dt
2. x(t+dt) = x(t) + v(t+dt) * dt  <- uses NEW velocity

This order is crucial for energy conservation in constrained systems.
"""

from std.math import cos, sin, pi, sqrt
from layout import LayoutTensor, Layout
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    B2_MAX_TRANSLATION,
    B2_MAX_ROTATION,
)
from ..traits.integrator import Integrator


struct SemiImplicitEuler(Integrator):
    """Semi-implicit (symplectic) Euler integrator.

    This is the workhorse integrator for game physics, matching Box2D.
    It provides good stability and energy conservation for constrained systems.

    Integration order:
    1. v' = v + (F/m + g) * dt   (velocity update)
    2. x' = x + v' * dt          (position update using NEW velocity)
    """

    def __init__(out self):
        """Initialize the integrator (stateless, nothing to store)."""
        pass

    # =========================================================================
    # CPU Implementation
    # =========================================================================

    def integrate_velocities[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        forces: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate velocities: v' = v + (F/m + g) * dt."""
        for env in range(BATCH):
            for body in range(NUM_BODIES):
                # Get inverse mass (0 = static body)
                var inv_mass = bodies[env, body, IDX_INV_MASS]
                var inv_inertia = bodies[env, body, IDX_INV_INERTIA]

                # Skip static bodies
                if inv_mass == Scalar[dtype](0):
                    continue

                # Read current velocities
                var vx = bodies[env, body, IDX_VX]
                var vy = bodies[env, body, IDX_VY]
                var omega = bodies[env, body, IDX_OMEGA]

                # Read forces
                var fx = forces[env, body, 0]
                var fy = forces[env, body, 1]
                var tau = forces[env, body, 2]

                # Integrate: v' = v + a * dt where a = F/m + g
                vx = vx + (fx * inv_mass + gravity_x) * dt
                vy = vy + (fy * inv_mass + gravity_y) * dt
                omega = omega + tau * inv_inertia * dt

                # Write back
                bodies[env, body, IDX_VX] = vx
                bodies[env, body, IDX_VY] = vy
                bodies[env, body, IDX_OMEGA] = omega

    def integrate_positions[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """Integrate positions: x' = x + v' * dt (using NEW velocity). The
        bodies layout is the flat state layout with BODIES_OFFSET = 0, so
        this routes to `integrate_positions_single_env` (ONE rule)."""
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies.ptr)
        for env in range(BATCH):
            Self.integrate_positions_single_env[
                BATCH, NUM_BODIES, NUM_BODIES * BODY_STATE_SIZE, 0
            ](env, state, dt)

    # =========================================================================
    # Strided GPU Kernels for 2D State Layout
    # =========================================================================
    #
    # These methods work with 2D [BATCH, STATE_SIZE] layout where physics
    # data is packed per-environment with offsets.
    #
    # Memory layout: state[env, OFFSET + body * BODY_STATE_SIZE + field]
    # This enables integration with GPUDiscreteEnv trait.
    # =========================================================================

    # =========================================================================
    # Single-Environment Methods (can be called from fused kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    def integrate_velocities_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate velocities for a single environment.

        This is the core logic, extracted to be callable from:
        - integrate_velocities_kernel (standalone kernel)
        - PhysicsStepKernel (fused kernel)
        """
        comptime for body in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE
            var force_off = FORCES_OFFSET + body * 3

            var inv_mass = state[env, body_off + IDX_INV_MASS]
            var inv_inertia = state[env, body_off + IDX_INV_INERTIA]

            # Skip static bodies
            if inv_mass == Scalar[dtype](0):
                continue

            var vx = state[env, body_off + IDX_VX]
            var vy = state[env, body_off + IDX_VY]
            var omega = state[env, body_off + IDX_OMEGA]

            var fx = state[env, force_off + 0]
            var fy = state[env, force_off + 1]
            var tau = state[env, force_off + 2]

            vx = vx + (fx * inv_mass + gravity_x) * dt
            vy = vy + (fy * inv_mass + gravity_y) * dt
            omega = omega + tau * inv_inertia * dt

            state[env, body_off + IDX_VX] = vx
            state[env, body_off + IDX_VY] = vy
            state[env, body_off + IDX_OMEGA] = omega

    @always_inline
    @staticmethod
    def integrate_positions_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """Integrate positions for a single environment.

        This is the core logic, extracted to be callable from:
        - integrate_positions_kernel (standalone kernel)
        - PhysicsStepKernel (fused kernel)
        """
        comptime for body in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE

            var inv_mass = state[env, body_off + IDX_INV_MASS]

            if inv_mass == Scalar[dtype](0):
                continue

            var x = state[env, body_off + IDX_X]
            var y = state[env, body_off + IDX_Y]
            var angle = state[env, body_off + IDX_ANGLE]
            var vx = state[env, body_off + IDX_VX]
            var vy = state[env, body_off + IDX_VY]
            var omega = state[env, body_off + IDX_OMEGA]

            # b2Island::Solve: cap the per-step translation / rotation by
            # scaling the velocity itself. Angles are NOT wrapped (Box2D
            # never does; a wrapped angle breaks joint angles aB - aA).
            var tx = vx * dt
            var ty = vy * dt
            var t2 = tx * tx + ty * ty
            var max_t = Scalar[dtype](B2_MAX_TRANSLATION)
            if t2 > max_t * max_t:
                var ratio = max_t / sqrt(t2)
                vx = vx * ratio
                vy = vy * ratio
            var rot = omega * dt
            var max_r = Scalar[dtype](B2_MAX_ROTATION)
            if rot * rot > max_r * max_r:
                omega = omega * (max_r / abs(rot))
            state[env, body_off + IDX_VX] = vx
            state[env, body_off + IDX_VY] = vy
            state[env, body_off + IDX_OMEGA] = omega

            state[env, body_off + IDX_X] = x + vx * dt
            state[env, body_off + IDX_Y] = y + vy * dt
            state[env, body_off + IDX_ANGLE] = angle + omega * dt

    @always_inline
    @staticmethod
    def integrate_velocities_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """GPU kernel for velocity integration with 2D strided layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        SemiImplicitEuler.integrate_velocities_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](env, state, gravity_x, gravity_y, dt)

    @always_inline
    @staticmethod
    def integrate_positions_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """GPU kernel for position integration with 2D strided layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        SemiImplicitEuler.integrate_positions_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, dt)

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
        """Launch strided velocity integration kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            gravity_x: Gravity X component.
            gravity_y: Gravity Y component.
            dt: Time step.
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE)
        ](state_buf)  # mut=True view from `mut state_buf` (written in place)

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
            state: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, STATE_SIZE),
                MutAnyOrigin,
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler.integrate_velocities_kernel[
                BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
            ](state, gravity_x, gravity_y, dt)

        ctx.enqueue_function[kernel_wrapper](
            state,
            gravity_x,
            gravity_y,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

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
        """Launch strided position integration kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            dt: Time step.
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE)
        ](state_buf)  # mut=True view from `mut state_buf` (written in place)

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
            state: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, STATE_SIZE),
                MutAnyOrigin,
            ],
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler.integrate_positions_kernel[
                BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
            ](state, dt)

        ctx.enqueue_function[kernel_wrapper](
            state,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
