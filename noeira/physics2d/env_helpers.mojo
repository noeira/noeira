"""PhysicsEnvHelpers - Environment setup utilities for physics-based envs.

This module provides helper functions that reduce boilerplate for initializing
physics-based environments. Works with the strided 2D layout [BATCH, STATE_SIZE].

Functions include:
- Body initialization (position, velocity, mass)
- Force application
- Joint creation
- Terrain setup
- State extraction

These helpers work with both CPU LayoutTensors and GPU DeviceBuffers.

Example:
    ```mojo
    from physics_gpu.env_helpers import PhysicsEnvHelpers
    from physics_gpu.layout import LunarLanderLayout

    comptime Layout = LunarLanderLayout

    # Initialize a body
    PhysicsEnvHelpers.init_body[BATCH, Layout.NUM_BODIES, Layout.STATE_SIZE, Layout.BODIES_OFFSET](
        states, env=0, body=0,
        x=10.0, y=5.0, angle=0.0, mass=5.0, inertia=2.0, shape_idx=0
    )
    ```
"""

from std.math import sqrt
from layout import LayoutTensor, Layout

from .joints.revolute import RevoluteJointSolver
from .constants import (
    dtype,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_FLAGS,
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
)


struct PhysicsEnvHelpers:
    """Helper functions for physics environment setup.

    This struct provides static methods that work with the strided 2D layout.
    All methods take compile-time layout parameters for flexibility.
    """

    # =========================================================================
    # Body Initialization
    # =========================================================================

    @staticmethod
    @always_inline
    def init_body[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
        x: Float64,
        y: Float64,
        angle: Float64,
        mass: Float64,
        inertia: Float64,
        shape_idx: Int,
    ):
        """Initialize a body's state.

        Sets position, angle, mass properties, and shape reference.
        Velocities are initialized to zero.

        Args:
            states: State tensor [BATCH, STATE_SIZE].
            env: Environment index.
            body: Body index.
            x: X position.
            y: Y position.
            angle: Rotation angle (radians).
            mass: Body mass (0 for static bodies).
            inertia: Moment of inertia.
            shape_idx: Index into shapes buffer.
        """
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE

        # Position and angle
        states[env, off + IDX_X] = Scalar[dtype](x)
        states[env, off + IDX_Y] = Scalar[dtype](y)
        states[env, off + IDX_ANGLE] = Scalar[dtype](angle)

        # Velocities (zero)
        states[env, off + IDX_VX] = Scalar[dtype](0)
        states[env, off + IDX_VY] = Scalar[dtype](0)
        states[env, off + IDX_OMEGA] = Scalar[dtype](0)

        # Mass properties
        states[env, off + IDX_MASS] = Scalar[dtype](mass)
        if mass > 0:
            states[env, off + IDX_INV_MASS] = Scalar[dtype](1.0 / mass)
            if inertia > 0:
                states[env, off + IDX_INV_INERTIA] = Scalar[dtype](
                    1.0 / inertia
                )
            else:
                states[env, off + IDX_INV_INERTIA] = Scalar[dtype](0)
        else:
            # Static body
            states[env, off + IDX_INV_MASS] = Scalar[dtype](0)
            states[env, off + IDX_INV_INERTIA] = Scalar[dtype](0)

        # Shape reference
        states[env, off + IDX_SHAPE] = Scalar[dtype](shape_idx)

    @staticmethod
    @always_inline
    def init_body_full[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
        x: Float64,
        y: Float64,
        angle: Float64,
        vx: Float64,
        vy: Float64,
        omega: Float64,
        mass: Float64,
        inertia: Float64,
        shape_idx: Int,
    ):
        """Initialize a body's full state including velocities."""
        Self.init_body[BATCH, STATE_SIZE, BODIES_OFFSET](
            states, env, body, x, y, angle, mass, inertia, shape_idx
        )

        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        states[env, off + IDX_VX] = Scalar[dtype](vx)
        states[env, off + IDX_VY] = Scalar[dtype](vy)
        states[env, off + IDX_OMEGA] = Scalar[dtype](omega)

    # =========================================================================
    # Body State Manipulation
    # =========================================================================

    @staticmethod
    @always_inline
    def set_body_position[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
        x: Float64,
        y: Float64,
    ):
        """Set body position."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        states[env, off + IDX_X] = Scalar[dtype](x)
        states[env, off + IDX_Y] = Scalar[dtype](y)

    @staticmethod
    @always_inline
    def set_body_velocity[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
        vx: Float64,
        vy: Float64,
        omega: Float64,
    ):
        """Set body linear and angular velocity."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        states[env, off + IDX_VX] = Scalar[dtype](vx)
        states[env, off + IDX_VY] = Scalar[dtype](vy)
        states[env, off + IDX_OMEGA] = Scalar[dtype](omega)

    # =========================================================================
    # Body State Reading
    # =========================================================================

    @staticmethod
    @always_inline
    def get_body_x[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body X position."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return rebind[Scalar[dtype]](states[env, off + IDX_X])

    @staticmethod
    @always_inline
    def get_body_y[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body Y position."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return rebind[Scalar[dtype]](states[env, off + IDX_Y])

    @staticmethod
    @always_inline
    def get_body_angle[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body angle."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return rebind[Scalar[dtype]](states[env, off + IDX_ANGLE])

    @staticmethod
    @always_inline
    def get_body_vx[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body X velocity."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return rebind[Scalar[dtype]](states[env, off + IDX_VX])

    @staticmethod
    @always_inline
    def get_body_vy[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body Y velocity."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return rebind[Scalar[dtype]](states[env, off + IDX_VY])

    @staticmethod
    @always_inline
    def get_body_omega[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Scalar[dtype]:
        """Get body angular velocity."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return rebind[Scalar[dtype]](states[env, off + IDX_OMEGA])

    @staticmethod
    @always_inline
    def get_body_state[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
    ) -> Tuple[
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
    ]:
        """Get full body state (x, y, angle, vx, vy, omega)."""
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        return (
            rebind[Scalar[dtype]](states[env, off + IDX_X]),
            rebind[Scalar[dtype]](states[env, off + IDX_Y]),
            rebind[Scalar[dtype]](states[env, off + IDX_ANGLE]),
            rebind[Scalar[dtype]](states[env, off + IDX_VX]),
            rebind[Scalar[dtype]](states[env, off + IDX_VY]),
            rebind[Scalar[dtype]](states[env, off + IDX_OMEGA]),
        )

    # =========================================================================
    # Force Application
    # =========================================================================

    @staticmethod
    @always_inline
    def apply_force[
        BATCH: Int,
        STATE_SIZE: Int,
        FORCES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
        fx: Float64,
        fy: Float64,
        torque: Float64,
    ):
        """Apply force and torque to a body (accumulated until step).

        Forces are added to existing accumulated forces.
        """
        var off = FORCES_OFFSET + body * 3
        states[env, off + 0] = states[env, off + 0] + Scalar[dtype](fx)
        states[env, off + 1] = states[env, off + 1] + Scalar[dtype](fy)
        states[env, off + 2] = states[env, off + 2] + Scalar[dtype](torque)

    @staticmethod
    @always_inline
    def set_force[
        BATCH: Int,
        STATE_SIZE: Int,
        FORCES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body: Int,
        fx: Float64,
        fy: Float64,
        torque: Float64,
    ):
        """Set force directly (replaces existing forces)."""
        var off = FORCES_OFFSET + body * 3
        states[env, off + 0] = Scalar[dtype](fx)
        states[env, off + 1] = Scalar[dtype](fy)
        states[env, off + 2] = Scalar[dtype](torque)

    @staticmethod
    @always_inline
    def clear_forces[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        FORCES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Clear all forces for an environment."""
        for body in range(NUM_BODIES):
            var off = FORCES_OFFSET + body * 3
            states[env, off + 0] = Scalar[dtype](0)
            states[env, off + 1] = Scalar[dtype](0)
            states[env, off + 2] = Scalar[dtype](0)

    # =========================================================================
    # Joint Management
    # =========================================================================

    @staticmethod
    @always_inline
    def get_joint_count[
        BATCH: Int,
        STATE_SIZE: Int,
        JOINT_COUNT_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ) -> Int:
        """Get number of active joints."""
        return Int(states[env, JOINT_COUNT_OFFSET])

    @staticmethod
    @always_inline
    def set_joint_count[
        BATCH: Int,
        STATE_SIZE: Int,
        JOINT_COUNT_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        count: Int,
    ):
        """Set joint count."""
        states[env, JOINT_COUNT_OFFSET] = Scalar[dtype](count)

    @staticmethod
    @always_inline
    def add_revolute_joint[
        BATCH: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        JOINT_COUNT_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        body_a: Int,
        body_b: Int,
        anchor_ax: Float64,
        anchor_ay: Float64,
        anchor_bx: Float64,
        anchor_by: Float64,
        lower_limit: Float64 = 0.0,
        upper_limit: Float64 = 0.0,
        enable_limit: Bool = False,
        enable_motor: Bool = False,
        motor_speed: Float64 = 0.0,
        max_motor_torque: Float64 = 0.0,
        reference_angle: Optional[Float64] = None,
    ) -> Int:
        """Add a revolute joint between two bodies.

        Returns:
            Joint index, or -1 if max joints reached.
        """
        var joint_idx = Self.get_joint_count[
            BATCH, STATE_SIZE, JOINT_COUNT_OFFSET
        ](states, env)
        if joint_idx >= MAX_JOINTS:
            return -1

        var ref = Scalar[dtype](0)
        if reference_angle:
            ref = Scalar[dtype](reference_angle.value())
        else:
            ref = Self.get_body_angle[BATCH, STATE_SIZE, BODIES_OFFSET](
                states, env, body_b
            ) - Self.get_body_angle[BATCH, STATE_SIZE, BODIES_OFFSET](
                states, env, body_a
            )
        RevoluteJointSolver.write_joint[BATCH, STATE_SIZE](
            states,
            env,
            JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE,
            body_a=body_a,
            body_b=body_b,
            anchor_ax=Scalar[dtype](anchor_ax),
            anchor_ay=Scalar[dtype](anchor_ay),
            anchor_bx=Scalar[dtype](anchor_bx),
            anchor_by=Scalar[dtype](anchor_by),
            reference_angle=ref,
            lower_limit=Scalar[dtype](lower_limit),
            upper_limit=Scalar[dtype](upper_limit),
            enable_limit=enable_limit,
            enable_motor=enable_motor,
            motor_speed=Scalar[dtype](motor_speed),
            max_motor_torque=Scalar[dtype](max_motor_torque),
        )

        # Increment count
        Self.set_joint_count[BATCH, STATE_SIZE, JOINT_COUNT_OFFSET](
            states, env, joint_idx + 1
        )

        return joint_idx

    # =========================================================================
    # Terrain Management
    # =========================================================================

    @staticmethod
    @always_inline
    def get_edge_count[
        BATCH: Int,
        STATE_SIZE: Int,
        EDGE_COUNT_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ) -> Int:
        """Get number of active terrain edges."""
        return Int(states[env, EDGE_COUNT_OFFSET])

    @staticmethod
    @always_inline
    def set_edge_count[
        BATCH: Int,
        STATE_SIZE: Int,
        EDGE_COUNT_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        count: Int,
    ):
        """Set edge count."""
        states[env, EDGE_COUNT_OFFSET] = Scalar[dtype](count)

    @staticmethod
    @always_inline
    def set_edge[
        BATCH: Int,
        STATE_SIZE: Int,
        EDGES_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        edge: Int,
        x0: Float64,
        y0: Float64,
        x1: Float64,
        y1: Float64,
        nx: Float64,
        ny: Float64,
    ):
        """Set edge data directly."""
        var off = EDGES_OFFSET + edge * 6
        states[env, off + 0] = Scalar[dtype](x0)
        states[env, off + 1] = Scalar[dtype](y0)
        states[env, off + 2] = Scalar[dtype](x1)
        states[env, off + 3] = Scalar[dtype](y1)
        states[env, off + 4] = Scalar[dtype](nx)
        states[env, off + 5] = Scalar[dtype](ny)

    @staticmethod
    @always_inline
    def set_flat_terrain[
        BATCH: Int,
        STATE_SIZE: Int,
        EDGES_OFFSET: Int,
        EDGE_COUNT_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        ground_y: Float64,
        x_min: Float64,
        x_max: Float64,
    ):
        """Set a single flat terrain edge."""
        # Single horizontal edge with upward normal
        Self.set_edge[BATCH, STATE_SIZE, EDGES_OFFSET](
            states, env, 0, x_min, ground_y, x_max, ground_y, 0.0, 1.0
        )
        Self.set_edge_count[BATCH, STATE_SIZE, EDGE_COUNT_OFFSET](
            states, env, 1
        )

    # =========================================================================
    # Metadata Access
    # =========================================================================

    @staticmethod
    @always_inline
    def get_metadata[
        BATCH: Int,
        STATE_SIZE: Int,
        METADATA_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        field: Int,
    ) -> Scalar[dtype]:
        """Get a metadata field value."""
        return rebind[Scalar[dtype]](states[env, METADATA_OFFSET + field])

    @staticmethod
    @always_inline
    def set_metadata[
        BATCH: Int,
        STATE_SIZE: Int,
        METADATA_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        field: Int,
        value: Float64,
    ):
        """Set a metadata field value."""
        states[env, METADATA_OFFSET + field] = Scalar[dtype](value)

    # =========================================================================
    # Observation Access
    # =========================================================================

    @staticmethod
    @always_inline
    def get_observation[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        idx: Int,
    ) -> Scalar[dtype]:
        """Get an observation value."""
        return rebind[Scalar[dtype]](states[env, OBS_OFFSET + idx])

    @staticmethod
    @always_inline
    def set_observation[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        idx: Int,
        value: Float64,
    ):
        """Set an observation value."""
        states[env, OBS_OFFSET + idx] = Scalar[dtype](value)
