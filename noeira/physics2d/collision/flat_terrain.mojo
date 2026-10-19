"""Flat terrain collision detection for LunarLander-like environments.

This is an optimized collision system for environments with:
- Flat ground at a fixed y coordinate
- Simple polygon/circle bodies
- No body-body collision (handled via joints)

Perfect for LunarLander where we need fast ground contact detection.
"""

from std.math import cos, sin
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_SHAPE,
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    MAX_POLYGON_VERTS,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
)
from ..traits.collision import CollisionSystem
from noeira.core.tiles import Tile, ImmTile, tile


struct FlatTerrainCollision(CollisionSystem):
    """Collision detection against flat terrain at y=ground_y.

    Optimized for LunarLander-like environments:
    - Ground is always at fixed y coordinate
    - Only checks body vertices/circles against ground
    - No body-body collision (handled via joints)

    This is much faster than general SAT collision for these simple cases.
    """

    comptime MAX_CONTACTS_PER_ENV: Int = 16  # Enough for 2 bodies * 8 vertices

    var ground_y: Scalar[dtype]

    def __init__(out self, ground_y: Float64):
        """Initialize with ground height.

        Args:
            ground_y: Y coordinate of the flat terrain.
        """
        self.ground_y = Scalar[dtype](ground_y)

    # =========================================================================
    # CPU Implementation
    # =========================================================================

    def detect[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int = 16,
    ](
        self,
        bodies: Tile[dtype, BATCH, NUM_BODIES, BODY_STATE_SIZE],
        shapes: Tile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
        contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
        contact_counts: Tile[dtype, BATCH],
    ):
        """Detect ground contacts for all bodies in all environments."""
        for env in range(BATCH):
            var count = 0

            for body_idx in range(NUM_BODIES):
                # Get body transform
                var body_x = bodies[env, body_idx, IDX_X]
                var body_y = bodies[env, body_idx, IDX_Y]
                var body_angle = bodies[env, body_idx, IDX_ANGLE]

                # Get shape info
                var shape_idx = Int(bodies[env, body_idx, IDX_SHAPE])
                var shape_type = Int(shapes[shape_idx, 0])

                var cos_a = cos(body_angle)
                var sin_a = sin(body_angle)

                if shape_type == SHAPE_POLYGON:
                    # Check each vertex against ground
                    var n_verts = Int(shapes[shape_idx, 1])

                    for v in range(n_verts):
                        if v >= MAX_POLYGON_VERTS:
                            break

                        # Local vertex position (in shape data after type and n_verts)
                        var local_x = shapes[shape_idx, 2 + v * 2]
                        var local_y = shapes[shape_idx, 3 + v * 2]

                        # Transform to world coordinates
                        var world_x = body_x + local_x * cos_a - local_y * sin_a
                        var world_y = body_y + local_x * sin_a + local_y * cos_a

                        # Check penetration (ground normal points up)
                        var penetration = self.ground_y - world_y
                        if (
                            penetration > Scalar[dtype](0)
                            and count < MAX_CONTACTS
                        ):
                            # Store contact
                            contacts[env, count, CONTACT_BODY_A] = Scalar[
                                dtype
                            ](body_idx)
                            contacts[env, count, CONTACT_BODY_B] = Scalar[
                                dtype
                            ](
                                -1
                            )  # -1 = ground
                            contacts[env, count, CONTACT_POINT_X] = world_x
                            contacts[env, count, CONTACT_POINT_Y] = world_y
                            contacts[env, count, CONTACT_NORMAL_X] = Scalar[
                                dtype
                            ](0)
                            contacts[env, count, CONTACT_NORMAL_Y] = Scalar[
                                dtype
                            ](
                                1
                            )  # Up
                            contacts[env, count, CONTACT_DEPTH] = penetration
                            contacts[
                                env, count, CONTACT_NORMAL_IMPULSE
                            ] = Scalar[dtype](0)
                            contacts[
                                env, count, CONTACT_TANGENT_IMPULSE
                            ] = Scalar[dtype](0)
                            count += 1

                elif shape_type == SHAPE_CIRCLE:
                    # Circle: check if bottom touches ground
                    var radius = shapes[shape_idx, 1]
                    var center_offset_x = shapes[shape_idx, 2]
                    var center_offset_y = shapes[shape_idx, 3]

                    # Transform center to world
                    var center_world_x = (
                        body_x
                        + center_offset_x * cos_a
                        - center_offset_y * sin_a
                    )
                    var center_world_y = (
                        body_y
                        + center_offset_x * sin_a
                        + center_offset_y * cos_a
                    )

                    # Check penetration
                    var penetration = self.ground_y - (center_world_y - radius)
                    if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                        contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                            body_idx
                        )
                        contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                        contacts[env, count, CONTACT_POINT_X] = center_world_x
                        contacts[env, count, CONTACT_POINT_Y] = (
                            center_world_y - radius
                        )
                        contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](
                            0
                        )
                        contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](
                            1
                        )
                        contacts[env, count, CONTACT_DEPTH] = penetration
                        contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        count += 1

            contact_counts[env] = Scalar[dtype](count)

    # =========================================================================
    # GPU Kernel Implementation
    # =========================================================================

    @always_inline
    @staticmethod
    def detect_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
    ](
        bodies: Tile[dtype, BATCH, NUM_BODIES, BODY_STATE_SIZE],
        shapes: ImmTile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
        contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
        contact_counts: Tile[dtype, BATCH],
        ground_y: Scalar[dtype],
    ):
        """GPU kernel: one thread per environment."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var count = 0

        comptime for body_idx in range(NUM_BODIES):
            var body_x = bodies[env, body_idx, IDX_X]
            var body_y = bodies[env, body_idx, IDX_Y]
            var body_angle = bodies[env, body_idx, IDX_ANGLE]
            var shape_idx = Int(bodies[env, body_idx, IDX_SHAPE])
            var shape_type = Int(shapes[shape_idx, 0])

            var cos_a = cos(body_angle)
            var sin_a = sin(body_angle)

            if shape_type == SHAPE_POLYGON:
                var n_verts = Int(shapes[shape_idx, 1])

                comptime for v in range(MAX_POLYGON_VERTS):
                    if v >= n_verts:
                        break

                    var local_x = shapes[shape_idx, 2 + v * 2]
                    var local_y = shapes[shape_idx, 3 + v * 2]
                    var world_x = body_x + local_x * cos_a - local_y * sin_a
                    var world_y = body_y + local_x * sin_a + local_y * cos_a

                    var penetration = ground_y - world_y
                    if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                        contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                            body_idx
                        )
                        contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                        contacts[env, count, CONTACT_POINT_X] = world_x
                        contacts[env, count, CONTACT_POINT_Y] = world_y
                        contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](
                            0
                        )
                        contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](
                            1
                        )
                        contacts[env, count, CONTACT_DEPTH] = penetration
                        contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        count += 1

            elif shape_type == SHAPE_CIRCLE:
                var radius = shapes[shape_idx, 1]
                var center_offset_x = shapes[shape_idx, 2]
                var center_offset_y = shapes[shape_idx, 3]

                var center_world_x = (
                    body_x + center_offset_x * cos_a - center_offset_y * sin_a
                )
                var center_world_y = (
                    body_y + center_offset_x * sin_a + center_offset_y * cos_a
                )

                var penetration = ground_y - (center_world_y - radius)
                if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                    contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                        body_idx
                    )
                    contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                    contacts[env, count, CONTACT_POINT_X] = center_world_x
                    contacts[env, count, CONTACT_POINT_Y] = (
                        center_world_y - radius
                    )
                    contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](0)
                    contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](1)
                    contacts[env, count, CONTACT_DEPTH] = penetration
                    contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                        dtype
                    ](0)
                    contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                        dtype
                    ](0)
                    count += 1

        contact_counts[env] = Scalar[dtype](count)

    # =========================================================================
    # Strided GPU Kernels for 2D State Layout
    # =========================================================================
    #
    # These methods work with 2D [BATCH, STATE_SIZE] layout for bodies.
    # Contacts are kept in standard layout as workspace (not persisted).
    # Memory layout: state[env, BODIES_OFFSET + body * BODY_STATE_SIZE + field]
    # =========================================================================

    @always_inline
    @staticmethod
    def detect_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: Tile[dtype, BATCH, STATE_SIZE],
        shapes: ImmTile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
        contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
        contact_counts: Tile[dtype, BATCH],
        ground_y: Scalar[dtype],
    ):
        """GPU kernel for flat terrain collision with 2D strided state layout.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var count = 0

        comptime for body_idx in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body_idx * BODY_STATE_SIZE

            var body_x = state[env, body_off + IDX_X]
            var body_y = state[env, body_off + IDX_Y]
            var body_angle = state[env, body_off + IDX_ANGLE]
            var shape_idx = Int(state[env, body_off + IDX_SHAPE])
            var shape_type = Int(shapes[shape_idx, 0])

            var cos_a = cos(body_angle)
            var sin_a = sin(body_angle)

            if shape_type == SHAPE_POLYGON:
                var n_verts = Int(shapes[shape_idx, 1])

                comptime for v in range(MAX_POLYGON_VERTS):
                    if v >= n_verts:
                        break

                    var local_x = shapes[shape_idx, 2 + v * 2]
                    var local_y = shapes[shape_idx, 3 + v * 2]
                    var world_x = body_x + local_x * cos_a - local_y * sin_a
                    var world_y = body_y + local_x * sin_a + local_y * cos_a

                    var penetration = ground_y - world_y
                    if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                        contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                            body_idx
                        )
                        contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                        contacts[env, count, CONTACT_POINT_X] = world_x
                        contacts[env, count, CONTACT_POINT_Y] = world_y
                        contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](
                            0
                        )
                        contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](
                            1
                        )
                        contacts[env, count, CONTACT_DEPTH] = penetration
                        contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        count += 1

            elif shape_type == SHAPE_CIRCLE:
                var radius = shapes[shape_idx, 1]
                var center_offset_x = shapes[shape_idx, 2]
                var center_offset_y = shapes[shape_idx, 3]

                var center_world_x = (
                    body_x + center_offset_x * cos_a - center_offset_y * sin_a
                )
                var center_world_y = (
                    body_y + center_offset_x * sin_a + center_offset_y * cos_a
                )

                var penetration = ground_y - (center_world_y - radius)
                if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                    contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                        body_idx
                    )
                    contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                    contacts[env, count, CONTACT_POINT_X] = center_world_x
                    contacts[env, count, CONTACT_POINT_Y] = (
                        center_world_y - radius
                    )
                    contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](0)
                    contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](1)
                    contacts[env, count, CONTACT_DEPTH] = penetration
                    contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                        dtype
                    ](0)
                    contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                        dtype
                    ](0)
                    count += 1

        contact_counts[env] = Scalar[dtype](count)

    @always_inline
    @staticmethod
    def detect_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        shapes: Tile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
        ground_y: Scalar[dtype],
        contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
        contact_counts: Tile[dtype, BATCH],
    ):
        """Detect collisions for a single environment against flat terrain.

        This is the core logic, extracted to be callable from:
        - detect_kernel (standalone kernel)
        - Fused physics step kernels

        Args:
            env: Environment index.
            state: State tensor [BATCH, STATE_SIZE].
            shapes: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            ground_y: Y coordinate of ground plane.
            contacts: Contact buffer to write to.
            contact_counts: Contact count per environment.
        """
        var count = 0

        for body_idx in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body_idx * BODY_STATE_SIZE

            var body_x = state[env, body_off + IDX_X]
            var body_y = state[env, body_off + IDX_Y]
            var body_angle = state[env, body_off + IDX_ANGLE]
            var shape_idx = Int(state[env, body_off + IDX_SHAPE])
            var shape_type = Int(shapes[shape_idx, 0])

            var cos_a = cos(body_angle)
            var sin_a = sin(body_angle)

            if shape_type == SHAPE_POLYGON:
                var n_verts = Int(shapes[shape_idx, 1])

                for v in range(MAX_POLYGON_VERTS):
                    if v >= n_verts:
                        break

                    var local_x = shapes[shape_idx, 2 + v * 2]
                    var local_y = shapes[shape_idx, 3 + v * 2]
                    var world_x = body_x + local_x * cos_a - local_y * sin_a
                    var world_y = body_y + local_x * sin_a + local_y * cos_a

                    var penetration = ground_y - world_y
                    if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                        contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                            body_idx
                        )
                        contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                        contacts[env, count, CONTACT_POINT_X] = world_x
                        contacts[env, count, CONTACT_POINT_Y] = world_y
                        contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](
                            0
                        )
                        contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](
                            1
                        )
                        contacts[env, count, CONTACT_DEPTH] = penetration
                        contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                            dtype
                        ](0)
                        count += 1

            elif shape_type == SHAPE_CIRCLE:
                var radius = shapes[shape_idx, 1]
                var center_offset_x = shapes[shape_idx, 2]
                var center_offset_y = shapes[shape_idx, 3]

                var center_world_x = (
                    body_x + center_offset_x * cos_a - center_offset_y * sin_a
                )
                var center_world_y = (
                    body_y + center_offset_x * sin_a + center_offset_y * cos_a
                )

                var penetration = ground_y - (center_world_y - radius)
                if penetration > Scalar[dtype](0) and count < MAX_CONTACTS:
                    contacts[env, count, CONTACT_BODY_A] = Scalar[dtype](
                        body_idx
                    )
                    contacts[env, count, CONTACT_BODY_B] = Scalar[dtype](-1)
                    contacts[env, count, CONTACT_POINT_X] = center_world_x
                    contacts[env, count, CONTACT_POINT_Y] = (
                        center_world_y - radius
                    )
                    contacts[env, count, CONTACT_NORMAL_X] = Scalar[dtype](0)
                    contacts[env, count, CONTACT_NORMAL_Y] = Scalar[dtype](1)
                    contacts[env, count, CONTACT_DEPTH] = penetration
                    contacts[env, count, CONTACT_NORMAL_IMPULSE] = Scalar[
                        dtype
                    ](0)
                    contacts[env, count, CONTACT_TANGENT_IMPULSE] = Scalar[
                        dtype
                    ](0)
                    count += 1

        contact_counts[env] = Scalar[dtype](count)

    @staticmethod
    def detect_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        ground_y: Scalar[dtype],
    ) raises:
        """Launch strided flat terrain collision detection kernel."""
        # state/shapes read-only here (mut=False views -> ImmutAnyOrigin kernel
        # params); contacts/contact_counts written (mut=True).
        var state = tile[BATCH, STATE_SIZE](state_buf)
        var shapes = tile[NUM_SHAPES, SHAPE_MAX_SIZE](shapes_buf)
        var contacts = tile[BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE](contacts_buf)
        var contact_counts = tile[BATCH](contact_counts_buf)

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        def kernel_wrapper(
            state: Tile[dtype, BATCH, STATE_SIZE],
            shapes: ImmTile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
            contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
            contact_counts: Tile[dtype, BATCH],
            ground_y: Scalar[dtype],
        ):
            FlatTerrainCollision.detect_kernel[
                BATCH,
                NUM_BODIES,
                NUM_SHAPES,
                MAX_CONTACTS,
                STATE_SIZE,
                BODIES_OFFSET,
            ](state, shapes, contacts, contact_counts, ground_y)

        ctx.enqueue_function[kernel_wrapper](
            state,
            shapes,
            contacts,
            contact_counts,
            ground_y,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
