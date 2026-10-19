"""Track tile collision for friction zone lookup.

This module provides point-in-quad collision detection to determine which
track tile (if any) a position is over. This determines the friction
coefficient at that position (road vs grass).

Algorithm: Cross-product test for convex quad
- A point is inside a convex quad if it's on the same side of all 4 edges
- For each edge (v0→v1), compute cross product: (v1-v0) × (p-v0)
- If all cross products have the same sign, point is inside

Track tiles are stored as:
    [v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y, friction]

Where vertices are in CCW order around the quad.
"""

from std.math import sqrt

from .constants import (
    TILE_DATA_SIZE,
    TILE_V0_X,
    TILE_V0_Y,
    TILE_V1_X,
    TILE_V1_Y,
    TILE_V2_X,
    TILE_V2_Y,
    TILE_V3_X,
    TILE_V3_Y,
    TILE_FRICTION,
    MAX_TRACK_TILES,
    GRASS_FRICTION,
    FRICTION_LIMIT,
    NUM_WHEELS,
    # Wheel positions for world coordinate calculation
    WHEEL_POS_FL_X,
    WHEEL_POS_FL_Y,
    WHEEL_POS_FR_X,
    WHEEL_POS_FR_Y,
    WHEEL_POS_RL_X,
    WHEEL_POS_RL_Y,
    WHEEL_POS_RR_X,
    WHEEL_POS_RR_Y,
)
from .wheel_friction import WheelFriction

from ..constants import dtype
from noeira.core.tiles import Tile


struct TileCollision:
    """Point-in-convex-quad collision for track tile friction lookup.

    Track tiles are convex quads that define road segments. Each tile has
    a friction coefficient (1.0 for road, 0.6 for grass edges, etc.).

    This struct provides stateless collision detection methods.
    """

    # =========================================================================
    # Point-in-Quad Test
    # =========================================================================

    @staticmethod
    @always_inline
    def point_in_quad(
        px: Scalar[dtype],
        py: Scalar[dtype],
        v0x: Scalar[dtype],
        v0y: Scalar[dtype],
        v1x: Scalar[dtype],
        v1y: Scalar[dtype],
        v2x: Scalar[dtype],
        v2y: Scalar[dtype],
        v3x: Scalar[dtype],
        v3y: Scalar[dtype],
    ) -> Bool:
        """Test if point (px, py) is inside convex quad v0-v1-v2-v3.

        Uses cross-product test: inside if same side of all 4 edges.

        Args:
            px: Point x-coordinate to test.
            py: Point y-coordinate to test.
            v0x: Quad vertex x-coordinate 0.
            v0y: Quad vertex y-coordinate 0.
            v1x: Quad vertex x-coordinate 1.
            v1y: Quad vertex y-coordinate 1.
            v2x: Quad vertex x-coordinate 2.
            v2y: Quad vertex y-coordinate 2.
            v3x: Quad vertex x-coordinate 3.
            v3y: Quad vertex y-coordinate 3.

        Returns:
            True if point is inside the quad.
        """
        var zero = Scalar[dtype](0.0)

        # Cross products for each edge
        # c = (v_next - v_curr) × (p - v_curr)
        # c = (dx * (py - vy)) - (dy * (px - vx))

        # Edge 0→1
        var c0 = (v1x - v0x) * (py - v0y) - (v1y - v0y) * (px - v0x)

        # Edge 1→2
        var c1 = (v2x - v1x) * (py - v1y) - (v2y - v1y) * (px - v1x)

        # Edge 2→3
        var c2 = (v3x - v2x) * (py - v2y) - (v3y - v2y) * (px - v2x)

        # Edge 3→0
        var c3 = (v0x - v3x) * (py - v3y) - (v0y - v3y) * (px - v3x)

        # Inside if all same sign (all positive or all negative)
        var all_positive = (
            c0 >= zero and c1 >= zero and c2 >= zero and c3 >= zero
        )
        var all_negative = (
            c0 <= zero and c1 <= zero and c2 <= zero and c3 <= zero
        )

        return all_positive or all_negative

    # =========================================================================
    # Friction Lookup
    # =========================================================================

    @staticmethod
    @always_inline
    def get_friction_at[
        MAX_TILES: Int,
    ](
        x: Scalar[dtype],
        y: Scalar[dtype],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
    ) -> Scalar[dtype]:
        """Get friction coefficient at position (x, y).

        Searches through track tiles to find which tile (if any) contains
        the point. Returns the tile's friction, or GRASS_FRICTION if not
        on any tile.

        Args:
            x: Position x-coordinate to query.
            y: Position y-coordinate to query.
            tiles: Track tile data [MAX_TILES, TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles in the buffer.

        Returns:
            Friction coefficient (GRASS_FRICTION if not on road).
        """
        # Search through all active tiles
        for i in range(num_active_tiles):
            var v0x = rebind[Scalar[dtype]](tiles[i, TILE_V0_X])
            var v0y = rebind[Scalar[dtype]](tiles[i, TILE_V0_Y])
            var v1x = rebind[Scalar[dtype]](tiles[i, TILE_V1_X])
            var v1y = rebind[Scalar[dtype]](tiles[i, TILE_V1_Y])
            var v2x = rebind[Scalar[dtype]](tiles[i, TILE_V2_X])
            var v2y = rebind[Scalar[dtype]](tiles[i, TILE_V2_Y])
            var v3x = rebind[Scalar[dtype]](tiles[i, TILE_V3_X])
            var v3y = rebind[Scalar[dtype]](tiles[i, TILE_V3_Y])

            if TileCollision.point_in_quad(
                x, y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
            ):
                return rebind[Scalar[dtype]](tiles[i, TILE_FRICTION])

        return Scalar[dtype](GRASS_FRICTION)

    @staticmethod
    @always_inline
    def get_friction_limit_at[
        MAX_TILES: Int,
    ](
        x: Scalar[dtype],
        y: Scalar[dtype],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
    ) -> Scalar[dtype]:
        """Get friction limit at position (x, y).

        Combines the surface friction multiplier with FRICTION_LIMIT.

        Args:
            x: Position x-coordinate to query.
            y: Position y-coordinate to query.
            tiles: Track tile data.
            num_active_tiles: Number of valid tiles.

        Returns:
            Friction limit = FRICTION_LIMIT * surface_friction.
        """
        var surface_friction = TileCollision.get_friction_at[MAX_TILES](
            x, y, tiles, num_active_tiles
        )
        return Scalar[dtype](FRICTION_LIMIT) * surface_friction

    # =========================================================================
    # Multi-Wheel Friction Lookup
    # =========================================================================

    @staticmethod
    @always_inline
    def get_wheel_friction_limits[
        MAX_TILES: Int,
    ](
        hull_x: Scalar[dtype],
        hull_y: Scalar[dtype],
        hull_angle: Scalar[dtype],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
    ) -> Array[Scalar[dtype], NUM_WHEELS]:
        """Get friction limits for all 4 wheels.

        Computes wheel world positions from hull state and looks up
        friction for each wheel position.

        Args:
            hull_x: Hull center position x-coordinate.
            hull_y: Hull center position y-coordinate.
            hull_angle: Hull orientation.
            tiles: Track tile data.
            num_active_tiles: Number of valid tiles.

        Returns:
            Array of friction limits [FL, FR, RL, RR].
        """
        from std.math import cos, sin

        var cos_a = cos(hull_angle)
        var sin_a = sin(hull_angle)

        # Wheel local positions
        var local_fl = WheelFriction.get_wheel_local_pos(0)  # FL
        var local_fr = WheelFriction.get_wheel_local_pos(1)  # FR
        var local_rl = WheelFriction.get_wheel_local_pos(2)  # RL
        var local_rr = WheelFriction.get_wheel_local_pos(3)  # RR

        # Transform to world coordinates
        var fl_x = hull_x + local_fl[0] * cos_a - local_fl[1] * sin_a
        var fl_y = hull_y + local_fl[0] * sin_a + local_fl[1] * cos_a

        var fr_x = hull_x + local_fr[0] * cos_a - local_fr[1] * sin_a
        var fr_y = hull_y + local_fr[0] * sin_a + local_fr[1] * cos_a

        var rl_x = hull_x + local_rl[0] * cos_a - local_rl[1] * sin_a
        var rl_y = hull_y + local_rl[0] * sin_a + local_rl[1] * cos_a

        var rr_x = hull_x + local_rr[0] * cos_a - local_rr[1] * sin_a
        var rr_y = hull_y + local_rr[0] * sin_a + local_rr[1] * cos_a

        # Get friction limits
        var fl_limit = TileCollision.get_friction_limit_at[MAX_TILES](
            fl_x, fl_y, tiles, num_active_tiles
        )
        var fr_limit = TileCollision.get_friction_limit_at[MAX_TILES](
            fr_x, fr_y, tiles, num_active_tiles
        )
        var rl_limit = TileCollision.get_friction_limit_at[MAX_TILES](
            rl_x, rl_y, tiles, num_active_tiles
        )
        var rr_limit = TileCollision.get_friction_limit_at[MAX_TILES](
            rr_x, rr_y, tiles, num_active_tiles
        )

        var result = Array[Scalar[dtype], NUM_WHEELS](
            fill=Scalar[dtype](0)
        )
        result[0] = fl_limit
        result[1] = fr_limit
        result[2] = rl_limit
        result[3] = rr_limit
        return result^

    # =========================================================================
    # Tile Visitation Tracking
    # =========================================================================

    @staticmethod
    @always_inline
    def check_tile_visited[
        MAX_TILES: Int,
    ](
        x: Scalar[dtype],
        y: Scalar[dtype],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
    ) -> Int:
        """Check if position is on any tile and return tile index.

        Used for reward calculation - visiting new tiles gives reward.

        Args:
            x: Position x-coordinate to check.
            y: Position y-coordinate to check.
            tiles: Track tile data.
            num_active_tiles: Number of valid tiles.

        Returns:
            Tile index if on a tile, -1 if on grass.
        """
        for i in range(num_active_tiles):
            var v0x = rebind[Scalar[dtype]](tiles[i, TILE_V0_X])
            var v0y = rebind[Scalar[dtype]](tiles[i, TILE_V0_Y])
            var v1x = rebind[Scalar[dtype]](tiles[i, TILE_V1_X])
            var v1y = rebind[Scalar[dtype]](tiles[i, TILE_V1_Y])
            var v2x = rebind[Scalar[dtype]](tiles[i, TILE_V2_X])
            var v2y = rebind[Scalar[dtype]](tiles[i, TILE_V2_Y])
            var v3x = rebind[Scalar[dtype]](tiles[i, TILE_V3_X])
            var v3y = rebind[Scalar[dtype]](tiles[i, TILE_V3_Y])

            if TileCollision.point_in_quad(
                x, y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
            ):
                return i

        return -1

    # =========================================================================
    # Embedded Track Methods (track stored in per-env state buffer)
    # =========================================================================

    @staticmethod
    @always_inline
    def get_friction_at_embedded[
        BATCH: Int,
        STATE_SIZE: Int,
        TRACK_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        x: Scalar[dtype],
        y: Scalar[dtype],
        states: Tile[dtype, BATCH, STATE_SIZE],
        num_active_tiles: Int,
    ) -> Scalar[dtype]:
        """Get friction coefficient at position using embedded track data.

        Reads track tiles from per-env state buffer instead of separate tiles buffer.

        Args:
            env: Environment index.
            x: Position x-coordinate to query.
            y: Position y-coordinate to query.
            states: State tensor with embedded track at TRACK_OFFSET.
            num_active_tiles: Number of valid tiles.

        Returns:
            Friction coefficient (GRASS_FRICTION if not on road).
        """
        # Search through all active tiles
        for i in range(num_active_tiles):
            var tile_off = TRACK_OFFSET + i * TILE_DATA_SIZE
            var v0x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V0_X])
            var v0y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V0_Y])
            var v1x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V1_X])
            var v1y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V1_Y])
            var v2x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V2_X])
            var v2y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V2_Y])
            var v3x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V3_X])
            var v3y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V3_Y])

            if TileCollision.point_in_quad(
                x, y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
            ):
                return rebind[Scalar[dtype]](
                    states[env, tile_off + TILE_FRICTION]
                )

        return Scalar[dtype](GRASS_FRICTION)

    @staticmethod
    @always_inline
    def get_friction_limit_at_embedded[
        BATCH: Int,
        STATE_SIZE: Int,
        TRACK_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        x: Scalar[dtype],
        y: Scalar[dtype],
        states: Tile[dtype, BATCH, STATE_SIZE],
        num_active_tiles: Int,
    ) -> Scalar[dtype]:
        """Get friction limit at position using embedded track data.

        Args:
            env: Environment index.
            x: Position x-coordinate to query.
            y: Position y-coordinate to query.
            states: State tensor with embedded track.
            num_active_tiles: Number of valid tiles.

        Returns:
            Friction limit = FRICTION_LIMIT * surface_friction.
        """
        var surface_friction = TileCollision.get_friction_at_embedded[
            BATCH, STATE_SIZE, TRACK_OFFSET, MAX_TILES
        ](env, x, y, states, num_active_tiles)
        return Scalar[dtype](FRICTION_LIMIT) * surface_friction

    @staticmethod
    @always_inline
    def check_tile_visited_embedded[
        BATCH: Int,
        STATE_SIZE: Int,
        TRACK_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        x: Scalar[dtype],
        y: Scalar[dtype],
        states: Tile[dtype, BATCH, STATE_SIZE],
        num_active_tiles: Int,
    ) -> Int:
        """Check if position is on any tile using embedded track data.

        Args:
            env: Environment index.
            x: Position x-coordinate to check.
            y: Position y-coordinate to check.
            states: State tensor with embedded track.
            num_active_tiles: Number of valid tiles.

        Returns:
            Tile index if on a tile, -1 if on grass.
        """
        for i in range(num_active_tiles):
            var tile_off = TRACK_OFFSET + i * TILE_DATA_SIZE
            var v0x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V0_X])
            var v0y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V0_Y])
            var v1x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V1_X])
            var v1y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V1_Y])
            var v2x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V2_X])
            var v2y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V2_Y])
            var v3x = rebind[Scalar[dtype]](states[env, tile_off + TILE_V3_X])
            var v3y = rebind[Scalar[dtype]](states[env, tile_off + TILE_V3_Y])

            if TileCollision.point_in_quad(
                x, y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
            ):
                return i

        return -1

    @staticmethod
    @always_inline
    def check_and_mark_visited_embedded[
        BATCH: Int,
        STATE_SIZE: Int,
        TRACK_OFFSET: Int,
        VISITED_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        x: Scalar[dtype],
        y: Scalar[dtype],
        states: Tile[dtype, BATCH, STATE_SIZE],
        num_active_tiles: Int,
    ) -> Tuple[Int, Bool]:
        """Check tile visit and mark as visited if new.

        Combines tile lookup and visited flag update for efficiency.

        Args:
            env: Environment index.
            x: Position x-coordinate to check.
            y: Position y-coordinate to check.
            states: State tensor with embedded track and visited flags.
            num_active_tiles: Number of valid tiles.

        Returns:
            Tuple of (tile_index, is_newly_visited).
            tile_index is -1 if not on any tile.
        """
        var tile_idx = TileCollision.check_tile_visited_embedded[
            BATCH, STATE_SIZE, TRACK_OFFSET, MAX_TILES
        ](env, x, y, states, num_active_tiles)

        if tile_idx < 0:
            return (tile_idx, False)

        # Check if already visited
        var visited_off = VISITED_OFFSET + tile_idx
        var already_visited = rebind[Scalar[dtype]](
            states[env, visited_off]
        ) > Scalar[dtype](0.5)

        if already_visited:
            return (tile_idx, False)

        # Mark as visited
        states[env, visited_off] = Scalar[dtype](1.0)
        return (tile_idx, True)
