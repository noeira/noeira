"""Tile Coding for Linear Function Approximation.

Tile coding is a form of coarse coding that uses multiple overlapping tilings
to create binary feature vectors. Each tiling partitions the state space into
tiles, and the active tiles form a sparse feature representation.

References:
- Sutton & Barto, Chapter 9.5: "Tile Coding"
- http://incompleteideas.net/tiles/tiles3.html

Example usage:
    # Create tile coding for CartPole (4D continuous state)
    var tc = TileCoding(
        num_tilings=8,
        tiles_per_dim=List[Int](8, 8, 8, 8),
        state_low=List[Scalar[Self.DTYPE]](-2.4, -3.0, -0.21, -3.0),
        state_high=List[Scalar[Self.DTYPE]](2.4, 3.0, 0.21, 3.0),
    )

    # Get active tile indices for a state
    var state = List[Scalar[Self.DTYPE]](0.1, -0.5, 0.05, 0.2)
    var tiles = tc.get_tiles(state)  # Returns num_tilings active tile indices
"""

from std.random import random_float64
from std.math import floor


struct TileCoding[DTYPE: DType]:
    """Multi-dimensional tile coding with multiple tilings.

    Creates num_tilings overlapping grids, each offset by a different amount.
    For a state, returns num_tilings active tile indices (one per tiling).

    Total number of tiles = num_tilings * product(tiles_per_dim)
    """

    var num_tilings: Int
    var num_dims: Int
    var tiles_per_dim: List[Int]
    var state_low: List[Scalar[Self.DTYPE]]
    var state_high: List[Scalar[Self.DTYPE]]
    var tile_widths: List[Scalar[Self.DTYPE]]
    var offsets: List[List[Scalar[Self.DTYPE]]]  # [tiling][dim] offset values
    var tiles_per_tiling: Int
    var total_tiles: Int

    def __init__(
        out self,
        num_tilings: Int,
        var tiles_per_dim: List[Int],
        var state_low: List[Scalar[Self.DTYPE]],
        var state_high: List[Scalar[Self.DTYPE]],
    ):
        """Initialize tile coding.

        Args:
            num_tilings: Number of overlapping tilings (typically 8-32).
            tiles_per_dim: Number of tiles per dimension (e.g., [8, 8, 8, 8] for 4D).
            state_low: Lower bounds for each state dimension.
            state_high: Upper bounds for each state dimension.
        """
        self.num_tilings = num_tilings
        self.num_dims = len(tiles_per_dim)

        # Calculate tile widths for each dimension
        self.tile_widths = List[Scalar[Self.DTYPE]]()
        for i in range(self.num_dims):
            var width = (state_high[i] - state_low[i]) / Scalar[Self.DTYPE](
                tiles_per_dim[i]
            )
            self.tile_widths.append(width)

        # Calculate tiles per tiling (product of tiles_per_dim)
        self.tiles_per_tiling = 1
        for i in range(self.num_dims):
            self.tiles_per_tiling *= tiles_per_dim[i]

        self.tiles_per_dim = tiles_per_dim^
        self.state_low = state_low^
        self.state_high = state_high^

        self.total_tiles = num_tilings * self.tiles_per_tiling

        # Generate asymmetric offsets for each tiling
        # Using displacement vectors that are coprime to avoid aliasing
        self.offsets = List[List[Scalar[Self.DTYPE]]]()
        for t in range(num_tilings):
            var tiling_offsets = List[Scalar[Self.DTYPE]]()
            for d in range(self.num_dims):
                # Offset pattern: t * (2*d + 1) / num_tilings gives asymmetric offsets
                var offset_fraction = (
                    Scalar[Self.DTYPE](t)
                    * Scalar[Self.DTYPE](2 * d + 1)
                    / Scalar[Self.DTYPE](num_tilings)
                )
                # Wrap to [0, 1) and scale by tile width
                offset_fraction = offset_fraction - floor(offset_fraction)
                var offset = offset_fraction * self.tile_widths[d]
                tiling_offsets.append(offset)
            self.offsets.append(tiling_offsets^)

    def get_tiles(self, state: List[Scalar[Self.DTYPE]]) -> List[Int]:
        """Get active tile indices for a given state.

        Args:
            state: Continuous state values (one per dimension).

        Returns:
            List of num_tilings active tile indices.
        """
        var active_tiles = List[Int]()

        for t in range(self.num_tilings):
            var tile_index = self._get_tile_in_tiling(state, t)
            # Add offset for this tiling's tile space
            active_tiles.append(t * self.tiles_per_tiling + tile_index)

        return active_tiles^

    def _get_tile_in_tiling(
        self, state: List[Scalar[Self.DTYPE]], tiling: Int
    ) -> Int:
        """Get tile index within a specific tiling.

        Args:
            state: Continuous state values
            tiling: Which tiling (0 to num_tilings-1)

        Returns:
            Tile index within this tiling (0 to tiles_per_tiling-1)
        """
        var flat_index = 0
        var multiplier = 1

        for d in range(self.num_dims):
            # Apply offset for this tiling
            var adjusted_value = (
                state[d] - self.state_low[d] + self.offsets[tiling][d]
            )

            # Calculate bin index
            var bin_idx = Int(floor(adjusted_value / self.tile_widths[d]))

            # Clamp to valid range
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= self.tiles_per_dim[d]:
                bin_idx = self.tiles_per_dim[d] - 1

            flat_index += bin_idx * multiplier
            multiplier *= self.tiles_per_dim[d]

        return flat_index

    def get_tiles_simd4(self, state: SIMD[Self.DTYPE, 4]) -> List[Int]:
        """Get active tile indices for a 4D SIMD state (optimized for CartPole).

        Args:
            state: 4D continuous state as SIMD vector.

        Returns:
            List of num_tilings active tile indices.
        """
        var state_list = List[Scalar[Self.DTYPE]]()
        state_list.append(state[0])
        state_list.append(state[1])
        state_list.append(state[2])
        state_list.append(state[3])
        return self.get_tiles(state_list)

    def get_tiles_simd2(self, state: SIMD[Self.DTYPE, 2]) -> List[Int]:
        """Get active tile indices for a 2D SIMD state (optimized for MountainCar).

        Args:
            state: 2D continuous state as SIMD vector.

        Returns:
            List of num_tilings active tile indices.
        """
        var state_list = List[Scalar[Self.DTYPE]]()
        state_list.append(state[0])
        state_list.append(state[1])
        return self.get_tiles(state_list)

    def get_num_tiles(self) -> Int:
        """Return total number of tiles across all tilings."""
        return self.total_tiles

    def get_num_tilings(self) -> Int:
        """Return number of tilings."""
        return self.num_tilings


struct TiledWeights[DTYPE: DType]:
    """Weight storage for tile-coded linear function approximation.

    Stores weights for each action, indexed by tile.
    Q(s, a) = sum of weights for active tiles in action a.
    """

    var weights: List[List[Scalar[Self.DTYPE]]]  # [action][tile]
    var num_actions: Int
    var num_tiles: Int

    def __init__(
        out self,
        num_tiles: Int,
        num_actions: Int,
        init_value: Scalar[Self.DTYPE] = 0.0,
    ):
        """Initialize weights.

        Args:
            num_tiles: Total number of tiles from TileCoding.
            num_actions: Number of discrete actions.
            init_value: Initial weight value (default 0.0, can use optimistic init).
        """
        self.num_tiles = num_tiles
        self.num_actions = num_actions
        self.weights = List[List[Scalar[Self.DTYPE]]]()

        for _ in range(num_actions):
            var action_weights = List[Scalar[Self.DTYPE]]()
            for _ in range(num_tiles):
                action_weights.append(init_value)
            self.weights.append(action_weights^)

    def get_value(self, tiles: List[Int], action: Int) -> Scalar[Self.DTYPE]:
        """Get Q-value for state (represented by active tiles) and action.

        Args:
            tiles: List of active tile indices from TileCoding.get_tiles().
            action: Action index.

        Returns:
            Q(s, a) = sum of weights for active tiles.
        """
        var value: Scalar[Self.DTYPE] = 0.0
        for i in range(len(tiles)):
            value += self.weights[action][tiles[i]]
        return value

    def get_all_values(self, tiles: List[Int]) -> List[Scalar[Self.DTYPE]]:
        """Get Q-values for all actions given a state.

        Args:
            tiles: List of active tile indices.

        Returns:
            List of Q-values, one per action.
        """
        var values = List[Scalar[Self.DTYPE]]()
        for a in range(self.num_actions):
            values.append(self.get_value(tiles, a))
        return values^

    def get_best_action(self, tiles: List[Int]) -> Int:
        """Get action with highest Q-value.

        Args:
            tiles: List of active tile indices.

        Returns:
            Action index with highest value (ties broken by first occurrence).
        """
        var best_action = 0
        var best_value = self.get_value(tiles, 0)

        for a in range(1, self.num_actions):
            var value = self.get_value(tiles, a)
            if value > best_value:
                best_value = value
                best_action = a

        return best_action

    def update(
        mut self,
        tiles: List[Int],
        action: Int,
        target: Scalar[Self.DTYPE],
        learning_rate: Scalar[Self.DTYPE],
    ):
        """Update weights using gradient descent.

        For linear function approximation with tile coding:
        w_i += α * (target - Q(s,a)) * ∇Q(s,a)

        Since ∇Q(s,a) = 1 for active tiles, this simplifies to:
        w_i += α * (target - Q(s,a)) / num_active_tiles

        Args:
            tiles: List of active tile indices.
            action: Action taken.
            target: TD target (e.g., r + γ * max_a' Q(s', a')).
            learning_rate: Learning rate (α).
        """
        var current_value = self.get_value(tiles, action)
        var td_error = target - current_value

        # Divide learning rate by number of active tiles
        # This ensures the total update magnitude is controlled
        var step_size = learning_rate / Scalar[Self.DTYPE](len(tiles))

        for i in range(len(tiles)):
            var tile_idx = tiles[i]
            self.weights[action][tile_idx] += step_size * td_error

    def update_eligibility(
        mut self,
        traces: List[List[Scalar[Self.DTYPE]]],
        td_error: Scalar[Self.DTYPE],
        learning_rate: Scalar[Self.DTYPE],
    ):
        """Update weights using eligibility traces.

        w += α * δ * e

        Args:
            traces: Eligibility trace values [action][tile].
            td_error: TD error (δ).
            learning_rate: Learning rate (α).
        """
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                self.weights[a][t] += learning_rate * td_error * traces[a][t]
