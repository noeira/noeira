"""Track generation for CarRacing.

This module provides procedural track generation for the CarRacing environment.
The track is represented as a list of quad tiles that define the road surface.

Algorithm (matching Gymnasium car_racing.py):
1. Generate random checkpoints around a circle
2. Trace a path from start, navigating towards each checkpoint
3. Find closed loop by detecting start line crossings
4. Create quad tiles from path points with TRACK_WIDTH offset
"""

from std.math import sin, cos, sqrt, pi, atan2
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom

from noeira.physics2d.car.constants import (
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
    ROAD_FRICTION,
)
from noeira.physics2d import dtype

from .constants import CRConstants
from noeira.core.tiles import Tile


# =============================================================================
# Track RNG using PhiloxRandom (GPU/CPU compatible)
# =============================================================================


struct TrackRNG:
    """PhiloxRandom-based RNG for reproducible track generation (GPU/CPU compatible).
    """

    var seed: Int
    var counter: Int

    def __init__(out self, seed: UInt64):
        self.seed = Int(seed) if seed != 0 else 1
        self.counter = 0

    def next(mut self) -> UInt64:
        """Generate next random value."""
        self.counter += 1
        var rng = PhiloxRandom(
            seed=UInt64(self.seed), offset=UInt64(self.counter)
        )
        var vals = rng.step_uniform()
        # Convert float to UInt64-like range
        return UInt64(Float64(vals[0]) * Float64(UInt64.MAX))

    def uniform(mut self, low: Float64, high: Float64) -> Float64:
        """Generate uniform random float in [low, high)."""
        self.counter += 1
        var rng = PhiloxRandom(
            seed=UInt64(self.seed), offset=UInt64(self.counter)
        )
        var vals = rng.step_uniform()
        var normalized = Float64(vals[0])
        return low + normalized * (high - low)

    def uniform_scalar[
        DTYPE: DType
    ](mut self, low: Float64, high: Float64) -> Scalar[DTYPE]:
        """Generate uniform random Scalar in [low, high)."""
        return Scalar[DTYPE](self.uniform(low, high))


# =============================================================================
# Track Point (intermediate representation)
# =============================================================================


struct TrackPoint[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """A point on the track centerline with direction."""

    var alpha: Scalar[Self.DTYPE]  # Angle from center (for lap tracking)
    var beta: Scalar[Self.DTYPE]  # Track direction (tangent angle)
    var x: Scalar[Self.DTYPE]
    var y: Scalar[Self.DTYPE]

    def __init__(
        out self,
        alpha: Scalar[Self.DTYPE],
        beta: Scalar[Self.DTYPE],
        x: Scalar[Self.DTYPE],
        y: Scalar[Self.DTYPE],
    ):
        self.alpha = alpha
        self.beta = beta
        self.x = x
        self.y = y

    def __init__(out self, *, copy: Self):
        self.alpha = copy.alpha
        self.beta = copy.beta
        self.x = copy.x
        self.y = copy.y

    def __init__(out self, *, deinit move: Self):
        self.alpha = move.alpha
        self.beta = move.beta
        self.x = move.x
        self.y = move.y


# =============================================================================
# Checkpoint
# =============================================================================


struct Checkpoint[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """A checkpoint on the track."""

    var alpha: Scalar[Self.DTYPE]  # Angle from center
    var x: Scalar[Self.DTYPE]
    var y: Scalar[Self.DTYPE]

    def __init__(
        out self,
        alpha: Scalar[Self.DTYPE],
        x: Scalar[Self.DTYPE],
        y: Scalar[Self.DTYPE],
    ):
        self.alpha = alpha
        self.x = x
        self.y = y

    def __init__(out self, *, copy: Self):
        self.alpha = copy.alpha
        self.x = copy.x
        self.y = copy.y

    def __init__(out self, *, deinit move: Self):
        self.alpha = move.alpha
        self.x = move.x
        self.y = move.y


struct TrackTile[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """A single track tile with quad vertices and friction.

    The tile is stored as 4 vertices in CCW order forming a convex quad.
    Each tile also stores its center point and direction for car placement.
    """

    var v0_x: Scalar[Self.DTYPE]
    var v0_y: Scalar[Self.DTYPE]
    var v1_x: Scalar[Self.DTYPE]
    var v1_y: Scalar[Self.DTYPE]
    var v2_x: Scalar[Self.DTYPE]
    var v2_y: Scalar[Self.DTYPE]
    var v3_x: Scalar[Self.DTYPE]
    var v3_y: Scalar[Self.DTYPE]

    var friction: Scalar[Self.DTYPE]
    var visited: Bool
    var idx: Int

    # Center and direction for car placement
    var center_x: Scalar[Self.DTYPE]
    var center_y: Scalar[Self.DTYPE]
    var direction: Scalar[Self.DTYPE]  # Tangent angle

    def __init__(out self):
        self.v0_x = 0.0
        self.v0_y = 0.0
        self.v1_x = 0.0
        self.v1_y = 0.0
        self.v2_x = 0.0
        self.v2_y = 0.0
        self.v3_x = 0.0
        self.v3_y = 0.0
        self.friction = 1.0
        self.visited = False
        self.idx = 0
        self.center_x = 0.0
        self.center_y = 0.0
        self.direction = 0.0

    def __init__(out self, *, copy: Self):
        self.v0_x = copy.v0_x
        self.v0_y = copy.v0_y
        self.v1_x = copy.v1_x
        self.v1_y = copy.v1_y
        self.v2_x = copy.v2_x
        self.v2_y = copy.v2_y
        self.v3_x = copy.v3_x
        self.v3_y = copy.v3_y
        self.friction = copy.friction
        self.visited = copy.visited
        self.idx = copy.idx
        self.center_x = copy.center_x
        self.center_y = copy.center_y
        self.direction = copy.direction

    def __init__(out self, *, deinit move: Self):
        self.v0_x = move.v0_x
        self.v0_y = move.v0_y
        self.v1_x = move.v1_x
        self.v1_y = move.v1_y
        self.v2_x = move.v2_x
        self.v2_y = move.v2_y
        self.v3_x = move.v3_x
        self.v3_y = move.v3_y
        self.friction = move.friction
        self.visited = move.visited
        self.idx = move.idx
        self.center_x = move.center_x
        self.center_y = move.center_y
        self.direction = move.direction

    def to_buffer[
        MAX_TILES: Int
    ](
        self,
        # by-value (not `mut`): a concrete-origin view widens into MutAnyOrigin
        # by value (as kernel params do); setitem writes through the pointer.
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        idx: Int,
    ):
        """Write tile data to GPU buffer."""
        tiles[idx, TILE_V0_X] = Scalar[dtype](self.v0_x)
        tiles[idx, TILE_V0_Y] = Scalar[dtype](self.v0_y)
        tiles[idx, TILE_V1_X] = Scalar[dtype](self.v1_x)
        tiles[idx, TILE_V1_Y] = Scalar[dtype](self.v1_y)
        tiles[idx, TILE_V2_X] = Scalar[dtype](self.v2_x)
        tiles[idx, TILE_V2_Y] = Scalar[dtype](self.v2_y)
        tiles[idx, TILE_V3_X] = Scalar[dtype](self.v3_x)
        tiles[idx, TILE_V3_Y] = Scalar[dtype](self.v3_y)
        tiles[idx, TILE_FRICTION] = Scalar[dtype](self.friction)


struct TrackGenerator[DTYPE: DType](Copyable, Movable):
    """Procedural track generator for CarRacing.

    Generates a closed-loop track around random checkpoints.
    Uses Bezier smoothing for natural-looking curves.
    """

    var track: List[TrackTile[Self.DTYPE]]
    var track_length: Int

    def __init__(out self):
        self.track = List[TrackTile[Self.DTYPE]]()
        self.track_length = 0

    def __init__(out self, *, copy: Self):
        self.track = copy.track.copy()
        self.track_length = copy.track_length

    def __init__(out self, *, deinit move: Self):
        self.track = move.track^
        self.track_length = move.track_length

    def generate_simple_track(mut self):
        """Generate a simple circular track.

        This is a reliable fallback that always produces a valid track.
        """
        comptime assert (
            Self.DTYPE.is_floating_point()
        ), "DTYPE must be a floating point type"

        self.track.clear()

        var num_tiles: Int = 100
        var rad = Scalar[Self.DTYPE](CRConstants.TRACK_RAD)
        var width = Scalar[Self.DTYPE](CRConstants.TRACK_WIDTH)

        for i in range(num_tiles):
            var alpha1 = (
                2.0 * pi * Scalar[Self.DTYPE](i) / Scalar[Self.DTYPE](num_tiles)
            )
            var alpha2 = (
                2.0
                * pi
                * Scalar[Self.DTYPE](i + 1)
                / Scalar[Self.DTYPE](num_tiles)
            )

            # Center line points
            var x1 = rad * cos(alpha1)
            var y1 = rad * sin(alpha1)
            var x2 = rad * cos(alpha2)
            var y2 = rad * sin(alpha2)

            # Create quad tile
            var tile = TrackTile[Self.DTYPE]()

            # Inner edge
            tile.v0_x = x1 - width * cos(alpha1)
            tile.v0_y = y1 - width * sin(alpha1)
            tile.v1_x = x2 - width * cos(alpha2)
            tile.v1_y = y2 - width * sin(alpha2)

            # Outer edge
            tile.v2_x = x2 + width * cos(alpha2)
            tile.v2_y = y2 + width * sin(alpha2)
            tile.v3_x = x1 + width * cos(alpha1)
            tile.v3_y = y1 + width * sin(alpha1)

            # Metadata
            tile.friction = Scalar[Self.DTYPE](CRConstants.ROAD_FRICTION)
            tile.visited = False
            tile.idx = i
            tile.center_x = (x1 + x2) / 2.0
            tile.center_y = (y1 + y2) / 2.0
            tile.direction = alpha1 + pi / 2.0  # Tangent direction

            self.track.append(tile^)

        self.track_length = len(self.track)

    def generate_random_track(
        mut self, seed: UInt64 = 42, verbose: Bool = False
    ):
        """Generate a random procedural track matching Gymnasium's algorithm.

        Algorithm:
        1. Create random checkpoints around a circle
        2. Trace path from start, navigating towards checkpoints
        3. Find closed loop by detecting start line crossings
        4. Create quad tiles from path points

        Falls back to simple track if generation fails after max attempts.
        """
        var rng = TrackRNG(seed)

        # Try multiple times with different seeds
        for attempt in range(20):
            var success = self._try_generate_track(rng, verbose)
            if success:
                if verbose:
                    print(
                        "Track generated successfully on attempt", attempt + 1
                    )
                    print("  Track length:", self.track_length)
                return
            # Try different seed on failure
            _ = rng.next()

        # Fallback to simple track
        if verbose:
            print("Falling back to simple circular track")
        self.generate_simple_track()

    def _try_generate_track(
        mut self, mut rng: TrackRNG, verbose: Bool = False
    ) -> Bool:
        """Attempt to generate a random track. Returns True on success."""
        self.track.clear()

        comptime assert (
            Self.DTYPE.is_floating_point()
        ), "DTYPE must be a floating point type"

        var two_pi = Scalar[Self.DTYPE](2.0 * pi)
        var track_rad = Scalar[Self.DTYPE](CRConstants.TRACK_RAD)
        var track_width = Scalar[Self.DTYPE](CRConstants.TRACK_WIDTH)
        var detail_step = Scalar[Self.DTYPE](CRConstants.TRACK_DETAIL_STEP)
        var turn_rate = Scalar[Self.DTYPE](CRConstants.TRACK_TURN_RATE)
        var scale = Scalar[Self.DTYPE](CRConstants.SCALE)
        var num_checkpoints = CRConstants.NUM_CHECKPOINTS

        # =====================================================================
        # Step 1: Create random checkpoints
        # =====================================================================

        var checkpoints = List[Checkpoint[Self.DTYPE]]()

        for c in range(num_checkpoints):
            var c_flt = Scalar[Self.DTYPE](c)
            var n_flt = Scalar[Self.DTYPE](num_checkpoints)

            # Add noise to angle (except first and last)
            var noise = rng.uniform_scalar[Self.DTYPE](
                0.0, 2.0 * pi / Float64(num_checkpoints)
            )
            var alpha = two_pi * c_flt / n_flt + noise

            # Random radius
            var rad = rng.uniform_scalar[Self.DTYPE](
                Float64(CRConstants.TRACK_RAD) / 3.0,
                Float64(CRConstants.TRACK_RAD),
            )

            # First checkpoint: fixed position
            if c == 0:
                alpha = Scalar[Self.DTYPE](0.0)
                rad = Scalar[Self.DTYPE](1.5) * track_rad

            # Last checkpoint: smooth transition back to start
            if c == num_checkpoints - 1:
                alpha = two_pi * c_flt / n_flt
                rad = Scalar[Self.DTYPE](1.5) * track_rad

            var x = rad * cos(alpha)
            var y = rad * sin(alpha)
            checkpoints.append(Checkpoint[Self.DTYPE](alpha, x, y))

        # Start alpha for detecting closed loop
        var start_alpha = (
            two_pi
            * Scalar[Self.DTYPE](-0.5)
            / Scalar[Self.DTYPE](num_checkpoints)
        )

        # =====================================================================
        # Step 2: Trace path from checkpoint to checkpoint
        # =====================================================================

        var track_points = List[TrackPoint[Self.DTYPE]]()
        var x = Scalar[Self.DTYPE](1.5) * track_rad
        var y = Scalar[Self.DTYPE](0.0)
        var beta = Scalar[Self.DTYPE](0.0)  # Track direction

        var dest_i = 0
        var laps = 0
        var no_freeze = 2500
        var visited_other_side = False

        while no_freeze > 0:
            no_freeze -= 1

            # Current angle from origin (raw, can be negative)
            var alpha_raw = atan2(y, x)

            # Track lap completion (using raw alpha)
            if visited_other_side and alpha_raw > Scalar[Self.DTYPE](0.0):
                laps += 1
                visited_other_side = False
            if alpha_raw < Scalar[Self.DTYPE](0.0):
                visited_other_side = True

            # Normalized alpha for checkpoint finding (always positive)
            var alpha = alpha_raw
            if alpha < Scalar[Self.DTYPE](0.0):
                alpha = alpha + two_pi

            # Find current destination checkpoint
            var failed = True
            for _ in range(num_checkpoints + 1):
                var dest = checkpoints[dest_i % num_checkpoints]
                if alpha <= dest.alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % num_checkpoints == 0:
                    break

            if failed:
                # Wrap alpha and try again
                alpha = alpha - two_pi
                for _ in range(num_checkpoints + 1):
                    var dest = checkpoints[dest_i % num_checkpoints]
                    if alpha <= dest.alpha:
                        break
                    dest_i += 1

            var dest = checkpoints[dest_i % num_checkpoints]
            var dest_x = dest.x
            var dest_y = dest.y

            # Direction vectors
            var r1x = cos(beta)
            var r1y = sin(beta)
            var p1x = -r1y  # Perpendicular (forward direction)
            var p1y = r1x

            # Vector towards destination
            var dest_dx = dest_x - x
            var dest_dy = dest_y - y

            # Project destination on radial direction
            var proj = r1x * dest_dx + r1y * dest_dy

            # Normalize beta relative to alpha
            var pi_1_5 = Scalar[Self.DTYPE](1.5 * pi)
            while beta - alpha > pi_1_5:
                beta = beta - two_pi
            while beta - alpha < -pi_1_5:
                beta = beta + two_pi

            var prev_beta = beta

            # Adjust beta based on projection (steering towards checkpoint)
            proj = proj * scale
            var threshold = Scalar[Self.DTYPE](0.3)
            var proj_abs = proj if proj >= Scalar[Self.DTYPE](0.0) else -proj

            if proj > threshold:
                var adjust = Scalar[Self.DTYPE](0.001) * proj_abs
                if adjust > turn_rate:
                    adjust = turn_rate
                beta = beta - adjust

            if proj < -threshold:
                var adjust = Scalar[Self.DTYPE](0.001) * proj_abs
                if adjust > turn_rate:
                    adjust = turn_rate
                beta = beta + adjust

            # Move forward
            x = x + p1x * detail_step
            y = y + p1y * detail_step

            # Record track point (use raw alpha for start line detection)
            var avg_beta = prev_beta * Scalar[Self.DTYPE](0.5) + beta * Scalar[
                Self.DTYPE
            ](0.5)
            track_points.append(
                TrackPoint[Self.DTYPE](alpha_raw, avg_beta, x, y)
            )

            # Exit if enough laps completed
            if laps > 4:
                break

        # =====================================================================
        # Step 3: Find closed loop (i1..i2)
        # =====================================================================

        if verbose:
            print("  Track points generated:", len(track_points), "laps:", laps)

        if len(track_points) < 10:
            if verbose:
                print("  Failed: too few track points")
            return False

        var i1 = -1
        var i2 = -1
        var i = len(track_points)

        while i > 1:
            i -= 1
            var curr_alpha = track_points[i].alpha
            var prev_alpha = track_points[i - 1].alpha

            # Check if crossing start line
            var pass_through_start = (
                curr_alpha > start_alpha and prev_alpha <= start_alpha
            )

            if pass_through_start:
                if i2 == -1:
                    i2 = i
                elif i1 == -1:
                    i1 = i
                    break

        if verbose:
            print("  Loop indices: i1=", i1, "i2=", i2)

        if i1 == -1 or i2 == -1 or i2 <= i1:
            if verbose:
                print("  Failed: invalid loop indices")
            return False

        # Extract the closed loop segment
        var loop_length = i2 - i1 - 1
        if verbose:
            print("  Loop length:", loop_length)

        if loop_length < 20:
            if verbose:
                print("  Failed: loop too short")
            return False

        # =====================================================================
        # Step 4: Validate loop closure
        # =====================================================================

        var first_point = track_points[i1]
        var last_point = track_points[i2 - 1]
        var first_beta = first_point.beta
        var first_perp_x = cos(first_beta)
        var first_perp_y = sin(first_beta)

        var dx = first_point.x - last_point.x
        var dy = first_point.y - last_point.y
        var proj_x = first_perp_x * dx
        var proj_y = first_perp_y * dy
        var well_glued = sqrt(proj_x * proj_x + proj_y * proj_y)

        if verbose:
            print("  Well glued:", well_glued, "vs", detail_step)

        if well_glued > detail_step:
            if verbose:
                print("  Failed: loop not well glued")
            return False

        # =====================================================================
        # Step 5: Create track tiles
        # =====================================================================

        for j in range(loop_length):
            var idx1 = i1 + j
            var idx2 = i1 + j + 1
            if idx2 >= i2:
                idx2 = i1  # Wrap around

            var p1 = track_points[idx1]
            var p2 = track_points[idx2]

            # Create tile with vertices offset by track width
            var tile = TrackTile[Self.DTYPE]()

            # Left edge (inner)
            tile.v0_x = p1.x - track_width * cos(p1.beta)
            tile.v0_y = p1.y - track_width * sin(p1.beta)
            tile.v1_x = p2.x - track_width * cos(p2.beta)
            tile.v1_y = p2.y - track_width * sin(p2.beta)

            # Right edge (outer)
            tile.v2_x = p2.x + track_width * cos(p2.beta)
            tile.v2_y = p2.y + track_width * sin(p2.beta)
            tile.v3_x = p1.x + track_width * cos(p1.beta)
            tile.v3_y = p1.y + track_width * sin(p1.beta)

            # Metadata
            tile.friction = Scalar[Self.DTYPE](CRConstants.ROAD_FRICTION)
            tile.visited = False
            tile.idx = j
            tile.center_x = (p1.x + p2.x) / Scalar[Self.DTYPE](2.0)
            tile.center_y = (p1.y + p2.y) / Scalar[Self.DTYPE](2.0)
            # Store direction consistent with simple track (beta + pi/2)
            # get_start_position subtracts pi/2 to get car angle
            tile.direction = p1.beta + Scalar[Self.DTYPE](pi / 2.0)

            self.track.append(tile^)

            # Limit track length
            if len(self.track) >= CRConstants.MAX_TRACK_TILES:
                break

        self.track_length = len(self.track)
        return self.track_length >= 20

    def to_buffer[
        MAX_TILES: Int
    ](
        self,
        # by-value (not `mut`): see per-tile `to_buffer` above.
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
    ):
        """Copy track data to GPU buffer."""
        for i in range(min(self.track_length, MAX_TILES)):
            self.track[i].to_buffer[MAX_TILES](tiles, i)

    def reset_visited(mut self):
        """Mark all tiles as unvisited."""
        for i in range(self.track_length):
            self.track[i].visited = False

    def get_start_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get starting position (x, y, angle) for the car."""
        if self.track_length > 0:
            var tile = self.track[0]
            return (tile.center_x, tile.center_y, tile.direction - pi / 2.0)
        return (
            Scalar[Self.DTYPE](1.5 * CRConstants.TRACK_RAD),
            Scalar[Self.DTYPE](0.0),
            Scalar[Self.DTYPE](0.0),
        )

    def mark_tile_visited(mut self, tile_idx: Int) -> Bool:
        """Mark a tile as visited. Returns True if this is a new visit."""
        if tile_idx >= 0 and tile_idx < self.track_length:
            if not self.track[tile_idx].visited:
                self.track[tile_idx].visited = True
                return True
        return False

    def count_visited(self) -> Int:
        """Count number of visited tiles."""
        var count = 0
        for i in range(self.track_length):
            if self.track[i].visited:
                count += 1
        return count
