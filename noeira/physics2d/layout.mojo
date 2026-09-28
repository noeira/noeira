"""PhysicsLayout - Compile-time layout for flat 2D strided buffers.

This module provides compile-time computation of buffer sizes and offsets
for the flat 2D [BATCH, STATE_SIZE] layout required by GPUDiscreteEnv.

PhysicsLayout packs all physics data per-environment in a single row:

    state[env, OFFSET + body * BODY_STATE_SIZE + field]

This enables efficient GPU access patterns and compatibility with the GPUDiscreteEnv trait.

Example:
    ```mojo
    from physics_gpu.layout import PhysicsLayout

    # Define layout for LunarLander: 3 bodies, 2 joints, 16 terrain edges
    comptime Layout = PhysicsLayout[
        NUM_BODIES=3,
        MAX_CONTACTS=8,
        MAX_JOINTS=2,
        MAX_TERRAIN_EDGES=16,
        OBS_DIM=8,
        METADATA_SIZE=4,
    ]

    # Access computed sizes and offsets at compile time
    comptime state_size = Layout.STATE_SIZE  # Total floats per environment
    comptime bodies_off = Layout.BODIES_OFFSET  # Offset to body data
    ```
"""

from .constants import (
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    MAX_JOINTS_PER_ENV,
)


struct PhysicsLayout[
    NUM_BODIES: Int,
    MAX_CONTACTS: Int = 16,
    MAX_JOINTS: Int = MAX_JOINTS_PER_ENV,
    MAX_TERRAIN_EDGES: Int = 16,
    OBS_DIM: Int = 8,
    METADATA_SIZE: Int = 4,
    NUM_SHAPES: Int = 3,  # Number of shape definitions (shared across envs)
]:
    """Compile-time layout calculator for flat 2D strided buffers.

    This struct computes buffer sizes and offsets at compile time for physics
    data stored in a flat [BATCH, STATE_SIZE] layout. All state for one
    environment is packed contiguously in a single row.

    Parameters:
        NUM_BODIES: Number of rigid bodies per environment.
        MAX_CONTACTS: Maximum contact points per environment.
        MAX_JOINTS: Maximum joints per environment.
        MAX_TERRAIN_EDGES: Maximum terrain edge segments per environment.
        OBS_DIM: Observation dimension (typically 8 for LunarLander).
        METADATA_SIZE: Size of metadata (step_count, total_reward, etc.).
        NUM_SHAPES: Number of shape definitions (shared across all envs).

    Layout (offsets are cumulative):
        [observation | bodies | forces | joints | joint_count | edges | edge_count | metadata]
    """

    # =========================================================================
    # Component Sizes
    # =========================================================================

    # Bodies: NUM_BODIES * 13 floats per body
    # Layout per body: [x, y, angle, vx, vy, omega, fx, fy, tau, mass, inv_mass, inv_inertia, shape_idx]
    comptime BODIES_SIZE: Int = Self.NUM_BODIES * BODY_STATE_SIZE

    # Forces: NUM_BODIES * 3 floats (fx, fy, torque)
    # Stored separately from bodies for efficient clearing
    comptime FORCES_SIZE: Int = Self.NUM_BODIES * 3

    # Joints: MAX_JOINTS * 17 floats per joint
    comptime JOINTS_SIZE: Int = Self.MAX_JOINTS * JOINT_DATA_SIZE

    # Terrain edges: MAX_TERRAIN_EDGES * 6 floats per edge (x0, y0, x1, y1, nx, ny)
    comptime EDGES_SIZE: Int = Self.MAX_TERRAIN_EDGES * 6

    # Shapes: Shared across all environments (separate buffer)
    comptime SHAPES_SIZE: Int = Self.NUM_SHAPES * SHAPE_MAX_SIZE

    # Contacts: Workspace for collision detection (separate buffer)
    comptime CONTACTS_SIZE: Int = Self.MAX_CONTACTS * CONTACT_DATA_SIZE

    # =========================================================================
    # Offsets within Environment State (cumulative)
    # =========================================================================

    # Observation at the start
    comptime OBS_OFFSET: Int = 0

    # Bodies follow observation
    comptime BODIES_OFFSET: Int = Self.OBS_DIM

    # Forces follow bodies
    comptime FORCES_OFFSET: Int = Self.BODIES_OFFSET + Self.BODIES_SIZE

    # Joints follow forces
    comptime JOINTS_OFFSET: Int = Self.FORCES_OFFSET + Self.FORCES_SIZE

    # Joint count (single value)
    comptime JOINT_COUNT_OFFSET: Int = Self.JOINTS_OFFSET + Self.JOINTS_SIZE

    # Terrain edges follow joint count
    comptime EDGES_OFFSET: Int = Self.JOINT_COUNT_OFFSET + 1

    # Edge count (single value)
    comptime EDGE_COUNT_OFFSET: Int = Self.EDGES_OFFSET + Self.EDGES_SIZE

    # Metadata at the end
    comptime METADATA_OFFSET: Int = Self.EDGE_COUNT_OFFSET + 1

    # =========================================================================
    # Total State Size
    # =========================================================================

    # Total size per environment
    comptime STATE_SIZE: Int = Self.METADATA_OFFSET + Self.METADATA_SIZE

    # =========================================================================
    # Utility Methods - Body Access
    # =========================================================================

    @staticmethod
    @always_inline
    def body_offset(body: Int) -> Int:
        """Compute offset for a specific body within environment state.

        Args:
            body: Body index (0 to NUM_BODIES-1).

        Returns:
            Offset to body's first field (x position).
        """
        return Self.BODIES_OFFSET + body * BODY_STATE_SIZE

    @staticmethod
    @always_inline
    def body_field_offset(body: Int, field: Int) -> Int:
        """Compute offset for a specific field of a body.

        Args:
            body: Body index.
            field: Field index (IDX_X, IDX_Y, etc.).

        Returns:
            Offset to the specific field.
        """
        return Self.BODIES_OFFSET + body * BODY_STATE_SIZE + field

    # =========================================================================
    # Utility Methods - Force Access
    # =========================================================================

    @staticmethod
    @always_inline
    def force_offset(body: Int) -> Int:
        """Compute offset for a body's forces within environment state.

        Args:
            body: Body index.

        Returns:
            Offset to body's force data (fx).
        """
        return Self.FORCES_OFFSET + body * 3

    # =========================================================================
    # Utility Methods - Joint Access
    # =========================================================================

    @staticmethod
    @always_inline
    def joint_offset(joint: Int) -> Int:
        """Compute offset for a specific joint within environment state.

        Args:
            joint: Joint index.

        Returns:
            Offset to joint's first field.
        """
        return Self.JOINTS_OFFSET + joint * JOINT_DATA_SIZE

    @staticmethod
    @always_inline
    def joint_field_offset(joint: Int, field: Int) -> Int:
        """Compute offset for a specific field of a joint.

        Args:
            joint: Joint index.
            field: Field index (JOINT_TYPE, JOINT_BODY_A, etc.).

        Returns:
            Offset to the specific field.
        """
        return Self.JOINTS_OFFSET + joint * JOINT_DATA_SIZE + field

    # =========================================================================
    # Utility Methods - Edge Access
    # =========================================================================

    @staticmethod
    @always_inline
    def edge_offset(edge: Int) -> Int:
        """Compute offset for a specific edge within environment state.

        Args:
            edge: Edge index.

        Returns:
            Offset to edge's first field (x0).
        """
        return Self.EDGES_OFFSET + edge * 6

    # =========================================================================
    # Utility Methods - Metadata Access
    # =========================================================================

    @staticmethod
    @always_inline
    def metadata_offset(field: Int) -> Int:
        """Compute offset for a metadata field.

        Args:
            field: Metadata field index.

        Returns:
            Offset to the metadata field.
        """
        return Self.METADATA_OFFSET + field


# =============================================================================
# Common Layout Aliases
# =============================================================================

# LunarLander: 3 bodies (lander + 2 legs), 2 joints, 16 terrain edges
comptime LunarLanderLayout = PhysicsLayout[
    NUM_BODIES=3,
    MAX_CONTACTS=8,
    MAX_JOINTS=2,
    MAX_TERRAIN_EDGES=16,
    OBS_DIM=8,
    METADATA_SIZE=5,  # step_count, total_reward, prev_shaping, done, sleep_time
    NUM_SHAPES=3,
]

# CartPole: 2 bodies (cart + pole), 1 joint, flat terrain
comptime CartPoleLayout = PhysicsLayout[
    NUM_BODIES=2,
    MAX_CONTACTS=4,
    MAX_JOINTS=1,
    MAX_TERRAIN_EDGES=2,
    OBS_DIM=4,
    METADATA_SIZE=2,
    NUM_SHAPES=2,
]

# Acrobot: 2 bodies (2 links), 1 joint, no terrain
comptime AcrobotLayout = PhysicsLayout[
    NUM_BODIES=2,
    MAX_CONTACTS=0,
    MAX_JOINTS=1,
    MAX_TERRAIN_EDGES=0,
    OBS_DIM=6,
    METADATA_SIZE=2,
    NUM_SHAPES=2,
]

# BipedalWalker: 5 bodies (hull + 4 legs), 4 joints, 128 terrain edges
comptime BipedalWalkerLayout = PhysicsLayout[
    NUM_BODIES=5,
    MAX_CONTACTS=20,
    MAX_JOINTS=4,
    MAX_TERRAIN_EDGES=128,
    OBS_DIM=24,
    METADATA_SIZE=8,
    NUM_SHAPES=5,
]

# HalfCheetah: 7 bodies (torso + 2x thigh, shin, foot), 6 joints, flat terrain
comptime HalfCheetahLayout = PhysicsLayout[
    NUM_BODIES=7,
    MAX_CONTACTS=14,
    MAX_JOINTS=6,
    MAX_TERRAIN_EDGES=2,
    OBS_DIM=17,
    METADATA_SIZE=8,
    NUM_SHAPES=7,
]
