"""Linear Function Approximation for Q-Learning.

This module provides infrastructure for Q-Learning with arbitrary feature vectors,
generalizing beyond tile coding to support any feature representation.

Key components:
- LinearWeights: Weight storage for linear Q-function approximation
- FeatureVector: Dense feature representation for state-action pairs

The Q-value is computed as a linear combination:
    Q(s, a) = w[a]^T * φ(s) = Σ w[a][i] * φ(s)[i]

where φ(s) is the feature vector for state s.

Example usage:
    # Create weights for 10-dimensional features and 2 actions
    var weights = LinearWeights(num_features=10, num_actions=2)

    # Get Q-value for a state-action pair
    var features = List[Float64]()  # Your feature extraction
    var q_value = weights.get_value(features, action=0)

    # Update using TD target
    weights.update(features, action=0, target=td_target, learning_rate=0.01)

References:
- Sutton & Barto, Chapter 9: "On-policy Prediction with Approximation"
- Sutton & Barto, Chapter 10: "On-policy Control with Approximation"
"""

from std.math import sqrt, exp
from std.random import random_float64


trait FeatureExtractor:
    """Trait for feature extractors that convert continuous states to feature vectors.
    """

    comptime DTYPE: DType

    def get_features_simd4(
        self, state: SIMD[Self.DTYPE, 4]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract features from 4D SIMD state (e.g., CartPole)."""
        ...

    def get_features(
        self, state: List[Scalar[Self.DTYPE]]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract features from state."""
        ...

    def get_num_features(self) -> Int:
        """Return the number of features."""
        ...


struct LinearWeights[DTYPE: DType]:
    """Weight storage for linear function approximation.

    Stores a weight vector for each action. Q-values are computed as:
        Q(s, a) = w[a]^T * φ(s)

    where φ(s) is the feature vector for state s.

    Unlike tile coding (which uses sparse binary features), this supports
    dense real-valued feature vectors.
    """

    var weights: List[List[Scalar[Self.DTYPE]]]  # [action][feature]
    var num_actions: Int
    var num_features: Int

    def __init__(
        out self,
        num_features: Int,
        num_actions: Int,
        init_std: Scalar[Self.DTYPE] = 0.01,
    ):
        """Initialize weights with small random values.

        Args:
            num_features: Dimensionality of feature vectors.
            num_actions: Number of discrete actions.
            init_std: Standard deviation for weight initialization (use 0.0 for zero init).
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.weights = List[List[Scalar[Self.DTYPE]]]()

        for _ in range(num_actions):
            var action_weights = List[Scalar[Self.DTYPE]]()
            for _ in range(num_features):
                if init_std > 0.0:
                    # Small random initialization centered at 0
                    var rand_val = (
                        Scalar[Self.DTYPE](random_float64() - 0.5)
                        * 2.0
                        * init_std
                    )
                    action_weights.append(rand_val)
                else:
                    action_weights.append(0.0)
            self.weights.append(action_weights^)

    def get_value(
        self, features: List[Scalar[Self.DTYPE]], action: Int
    ) -> Scalar[Self.DTYPE]:
        """Compute Q-value for a state-action pair.

        Q(s, a) = w[a]^T * φ(s) = Σ w[a][i] * features[i]

        Args:
            features: Feature vector φ(s) for the state.
            action: Action index.

        Returns:
            Q(s, a) as the dot product of weights and features.
        """
        var value: Scalar[Self.DTYPE] = 0.0
        for i in range(self.num_features):
            value += self.weights[action][i] * features[i]
        return value

    def get_all_values(
        self, features: List[Scalar[Self.DTYPE]]
    ) -> List[Scalar[Self.DTYPE]]:
        """Compute Q-values for all actions given a state.

        Args:
            features: Feature vector φ(s).

        Returns:
            List of Q(s, a) for all actions.
        """
        var values = List[Scalar[Self.DTYPE]]()
        for a in range(self.num_actions):
            values.append(self.get_value(features, a))
        return values^

    def get_best_action(self, features: List[Scalar[Self.DTYPE]]) -> Int:
        """Get action with highest Q-value.

        Args:
            features: Feature vector φ(s).

        Returns:
            Action index with highest Q-value.
        """
        var best_action = 0
        var best_value = self.get_value(features, 0)

        for a in range(1, self.num_actions):
            var value = self.get_value(features, a)
            if value > best_value:
                best_value = value
                best_action = a

        return best_action

    def get_max_value(
        self, features: List[Scalar[Self.DTYPE]]
    ) -> Scalar[Self.DTYPE]:
        """Get maximum Q-value over all actions.

        Args:
            features: Feature vector φ(s).

        Returns:
            Max_a Q(s, a).
        """
        var best_action = self.get_best_action(features)
        return self.get_value(features, best_action)

    def update(
        mut self,
        features: List[Scalar[Self.DTYPE]],
        action: Int,
        target: Scalar[Self.DTYPE],
        learning_rate: Scalar[Self.DTYPE],
    ):
        """Update weights using gradient descent.

        For linear function approximation:
            w[a] += α * (target - Q(s,a)) * ∇_w Q(s,a)

        Since Q(s,a) = w[a]^T * φ(s), we have ∇_w Q(s,a) = φ(s), so:
            w[a][i] += α * (target - Q(s,a)) * φ(s)[i] / num_features

        We divide by num_features to normalize the total update magnitude,
        similar to how tile coding divides by the number of active tiles.

        Args:
            features: Feature vector φ(s).
            action: Action taken.
            target: TD target (e.g., r + γ * max_a' Q(s', a')).
            learning_rate: Learning rate α.
        """
        var current_value = self.get_value(features, action)
        var td_error = target - current_value

        # Divide learning rate by number of features (like tile coding)
        var step_size = learning_rate / Scalar[Self.DTYPE](self.num_features)

        for i in range(self.num_features):
            self.weights[action][i] += step_size * td_error * features[i]

    def update_with_eligibility(
        mut self,
        traces: List[List[Scalar[Self.DTYPE]]],
        td_error: Scalar[Self.DTYPE],
        learning_rate: Scalar[Self.DTYPE],
    ):
        """Update weights using eligibility traces.

        w[a][i] += α * δ * e[a][i]

        Args:
            traces: Eligibility trace values [action][feature].
            td_error: TD error δ.
            learning_rate: Learning rate α.
        """
        for a in range(self.num_actions):
            for i in range(self.num_features):
                self.weights[a][i] += learning_rate * td_error * traces[a][i]


struct PolynomialFeatures[dtype: DType](FeatureExtractor):
    """Polynomial feature extractor for continuous state spaces.

    Generates polynomial features up to a specified degree with state normalization.
    For a 2D state [x, y] with degree 2:
        φ(s) = [1, x_norm, y_norm, x_norm², x_norm*y_norm, y_norm²]

    State normalization maps each dimension to [-1, 1] range, which is critical
    for stable learning with polynomial features.

    This is useful for environments like MountainCar or CartPole where
    polynomial combinations of state variables can capture value function structure.
    """

    comptime DTYPE = Self.dtype

    var state_dim: Int
    var degree: Int
    var num_features: Int
    var include_bias: Bool
    var state_low: List[Scalar[Self.DTYPE]]
    var state_high: List[Scalar[Self.DTYPE]]
    var normalize: Bool

    def __init__(
        out self,
        state_dim: Int,
        degree: Int = 2,
        include_bias: Bool = True,
        var state_low: List[Scalar[Self.DTYPE]] = List[Scalar[Self.DTYPE]](),
        var state_high: List[Scalar[Self.DTYPE]] = List[Scalar[Self.DTYPE]](),
    ):
        """Initialize polynomial feature extractor.

        Args:
            state_dim: Dimensionality of the state space.
            degree: Maximum polynomial degree (default 2).
            include_bias: Whether to include a bias term (constant 1).
            state_low: Lower bounds for state normalization (optional).
            state_high: Upper bounds for state normalization (optional).
        """
        self.state_dim = state_dim
        self.degree = degree
        self.include_bias = include_bias

        # Check if normalization bounds are provided
        self.normalize = (
            len(state_low) == state_dim and len(state_high) == state_dim
        )
        self.state_low = state_low^
        self.state_high = state_high^

        # Calculate number of features inline
        var count = 0
        if include_bias:
            count = 1  # Bias term

        if state_dim == 1:
            count += degree
        elif state_dim == 2:
            # For 2D: degree 1 has 2, degree 2 adds 3 (x², xy, y²), etc.
            for d in range(1, degree + 1):
                count += d + 1
        elif state_dim == 3:
            # For 3D: more complex enumeration
            for d in range(1, degree + 1):
                count += (d + 1) * (d + 2) // 2
        elif state_dim == 4:
            # For 4D: even more complex
            for d in range(1, degree + 1):
                count += (d + 1) * (d + 2) * (d + 3) // 6
        else:
            # Fallback: just use degree 1 features
            count += state_dim

        self.num_features = count

    def _normalize_state(
        self, state: List[Scalar[Self.DTYPE]]
    ) -> List[Scalar[Self.DTYPE]]:
        """Normalize state to [-1, 1] range."""
        var normalized = List[Scalar[Self.DTYPE]]()
        if self.normalize:
            for i in range(self.state_dim):
                var range_size = self.state_high[i] - self.state_low[i]
                if range_size > 0:
                    # Map to [0, 1] then to [-1, 1]
                    var norm_val = (
                        2.0 * (state[i] - self.state_low[i]) / range_size - 1.0
                    )
                    # Clamp to [-1, 1]
                    if norm_val < -1.0:
                        norm_val = -1.0
                    elif norm_val > 1.0:
                        norm_val = 1.0
                    normalized.append(norm_val)
                else:
                    normalized.append(0.0)
        else:
            for i in range(self.state_dim):
                normalized.append(state[i])
        return normalized^

    def get_features(
        self, state: List[Scalar[Self.DTYPE]]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract polynomial features from state.

        Args:
            state: Raw state vector (will be normalized if bounds provided).

        Returns:
            Polynomial feature vector.
        """
        # Normalize state first if bounds are provided
        var norm_state = self._normalize_state(state)

        var features = List[Scalar[Self.DTYPE]]()

        # Bias term
        if self.include_bias:
            features.append(1.0)

        if self.state_dim == 1:
            # 1D case: [x, x², x³, ...]
            var x = norm_state[0]
            var power: Scalar[Self.DTYPE] = x
            for _ in range(self.degree):
                features.append(power)
                power *= x

        elif self.state_dim == 2:
            # 2D case: explicit enumeration
            var x = norm_state[0]
            var y = norm_state[1]

            for total_deg in range(1, self.degree + 1):
                for x_deg in range(total_deg, -1, -1):
                    var y_deg = total_deg - x_deg
                    var term = self._power(x, x_deg) * self._power(y, y_deg)
                    features.append(term)

        elif self.state_dim == 3:
            # 3D case
            var x = norm_state[0]
            var y = norm_state[1]
            var z = norm_state[2]

            for total_deg in range(1, self.degree + 1):
                for x_deg in range(total_deg, -1, -1):
                    for y_deg in range(total_deg - x_deg, -1, -1):
                        var z_deg = total_deg - x_deg - y_deg
                        var term = (
                            self._power(x, x_deg)
                            * self._power(y, y_deg)
                            * self._power(z, z_deg)
                        )
                        features.append(term)

        elif self.state_dim == 4:
            # 4D case (e.g., CartPole)
            var x0 = norm_state[0]
            var x1 = norm_state[1]
            var x2 = norm_state[2]
            var x3 = norm_state[3]

            for total_deg in range(1, self.degree + 1):
                for d0 in range(total_deg, -1, -1):
                    for d1 in range(total_deg - d0, -1, -1):
                        for d2 in range(total_deg - d0 - d1, -1, -1):
                            var d3 = total_deg - d0 - d1 - d2
                            var term = (
                                self._power(x0, d0)
                                * self._power(x1, d1)
                                * self._power(x2, d2)
                                * self._power(x3, d3)
                            )
                            features.append(term)
        else:
            # Fallback: just linear features
            for i in range(self.state_dim):
                features.append(norm_state[i])

        return features^

    def get_features_simd2(
        self, state: SIMD[Self.DTYPE, 2]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract features from 2D SIMD state (e.g., MountainCar)."""
        var state_list = List[Scalar[Self.DTYPE]]()
        state_list.append(state[0])
        state_list.append(state[1])
        return self.get_features(state_list)

    def get_features_simd4(
        self, state: SIMD[Self.DTYPE, 4]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract features from 4D SIMD state (e.g., CartPole)."""
        var state_list = List[Scalar[Self.DTYPE]]()
        state_list.append(state[0])
        state_list.append(state[1])
        state_list.append(state[2])
        state_list.append(state[3])
        return self.get_features(state_list)

    def get_num_features(self) -> Int:
        """Return the number of features."""
        return self.num_features

    @staticmethod
    def _power(x: Scalar[Self.DTYPE], n: Int) -> Scalar[Self.DTYPE]:
        """Compute x^n."""
        if n == 0:
            return 1.0
        var result: Scalar[Self.DTYPE] = 1.0
        for _ in range(n):
            result *= x
        return result


struct RBFFeatures[dtype: DType](FeatureExtractor):
    """Radial Basis Function (RBF) feature extractor.

    Creates features based on distance to fixed centers:
        φ_i(s) = exp(-||s - c_i||² / (2σ²))

    RBF features provide smooth generalization and are useful for
    continuous state spaces where locality matters.
    """

    comptime DTYPE = Self.dtype

    var centers: List[List[Scalar[Self.DTYPE]]]  # [center_idx][dim]
    var sigma: Scalar[Self.DTYPE]
    var num_features: Int
    var state_dim: Int

    def __init__(
        out self,
        var centers: List[List[Scalar[Self.DTYPE]]],
        sigma: Scalar[Self.DTYPE] = 1.0,
    ):
        """Initialize RBF feature extractor.

        Args:
            centers: List of center positions, each of dimension state_dim.
            sigma: RBF width parameter (standard deviation).
        """
        self.num_features = len(centers)
        self.sigma = sigma
        if self.num_features > 0:
            self.state_dim = len(centers[0])
        else:
            self.state_dim = 0
        self.centers = centers^

    def get_features(
        self, state: List[Scalar[Self.DTYPE]]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract RBF features from state.

        Args:
            state: Raw state vector.

        Returns:
            RBF feature vector with one feature per center.
        """
        var features = List[Scalar[Self.DTYPE]]()
        var two_sigma_sq = 2.0 * self.sigma * self.sigma

        for c in range(self.num_features):
            # Compute squared distance to center
            var dist_sq: Scalar[Self.DTYPE] = 0.0
            for d in range(self.state_dim):
                var diff = state[d] - self.centers[c][d]
                dist_sq += diff * diff

            # RBF activation
            var activation = self._exp(-dist_sq / two_sigma_sq)
            features.append(activation)

        return features^

    def get_features_simd2(
        self, state: SIMD[Self.DTYPE, 2]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract features from 2D SIMD state."""
        var state_list = List[Scalar[Self.DTYPE]]()
        state_list.append(state[0])
        state_list.append(state[1])
        return self.get_features(state_list)

    def get_features_simd4(
        self, state: SIMD[Self.DTYPE, 4]
    ) -> List[Scalar[Self.DTYPE]]:
        """Extract features from 4D SIMD state."""
        var state_list = List[Scalar[Self.DTYPE]]()
        state_list.append(state[0])
        state_list.append(state[1])
        state_list.append(state[2])
        state_list.append(state[3])
        return self.get_features(state_list)

    def get_num_features(self) -> Int:
        """Return number of RBF features."""
        return self.num_features

    @staticmethod
    def _exp(x: Scalar[Self.DTYPE]) -> Scalar[Self.DTYPE]:
        """Compute e^x using Taylor series approximation."""
        # For x < 0 (which is always the case for RBF), we can use
        # exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
        # But for better accuracy, we'll use the math module
        comptime assert (
            Self.DTYPE.is_floating_point()
        ), "DTYPE must be floating point"

        return exp(x)


def make_grid_rbf_centers[
    DTYPE: DType
](
    state_low: List[Scalar[DTYPE]],
    state_high: List[Scalar[DTYPE]],
    num_centers_per_dim: Int,
) -> List[List[Scalar[DTYPE]]]:
    """Create a grid of RBF centers covering the state space.

    Args:
        state_low: Lower bounds for each state dimension.
        state_high: Upper bounds for each state dimension.
        num_centers_per_dim: Number of centers along each dimension.

    Returns:
        List of center positions.
    """
    var state_dim = len(state_low)
    var centers = List[List[Scalar[DTYPE]]]()

    if state_dim == 1:
        for i in range(num_centers_per_dim):
            var c = List[Scalar[DTYPE]]()
            var t = (
                Scalar[DTYPE](i)
                / Scalar[DTYPE](num_centers_per_dim - 1) if num_centers_per_dim
                > 1 else 0.5
            )
            c.append(state_low[0] + t * (state_high[0] - state_low[0]))
            centers.append(c^)

    elif state_dim == 2:
        for i in range(num_centers_per_dim):
            for j in range(num_centers_per_dim):
                var c = List[Scalar[DTYPE]]()
                var ti = (
                    Scalar[DTYPE](i)
                    / Scalar[DTYPE](
                        num_centers_per_dim - 1
                    ) if num_centers_per_dim
                    > 1 else 0.5
                )
                var tj = (
                    Scalar[DTYPE](j)
                    / Scalar[DTYPE](
                        num_centers_per_dim - 1
                    ) if num_centers_per_dim
                    > 1 else 0.5
                )
                c.append(state_low[0] + ti * (state_high[0] - state_low[0]))
                c.append(state_low[1] + tj * (state_high[1] - state_low[1]))
                centers.append(c^)

    elif state_dim == 3:
        for i in range(num_centers_per_dim):
            for j in range(num_centers_per_dim):
                for k in range(num_centers_per_dim):
                    var c = List[Scalar[DTYPE]]()
                    var ti = (
                        Scalar[DTYPE](i)
                        / Scalar[DTYPE](
                            num_centers_per_dim - 1
                        ) if num_centers_per_dim
                        > 1 else 0.5
                    )
                    var tj = (
                        Scalar[DTYPE](j)
                        / Scalar[DTYPE](
                            num_centers_per_dim - 1
                        ) if num_centers_per_dim
                        > 1 else 0.5
                    )
                    var tk = (
                        Scalar[DTYPE](k)
                        / Scalar[DTYPE](
                            num_centers_per_dim - 1
                        ) if num_centers_per_dim
                        > 1 else 0.5
                    )
                    c.append(state_low[0] + ti * (state_high[0] - state_low[0]))
                    c.append(state_low[1] + tj * (state_high[1] - state_low[1]))
                    c.append(state_low[2] + tk * (state_high[2] - state_low[2]))
                    centers.append(c^)

    elif state_dim == 4:
        # For 4D, use fewer centers per dim to avoid explosion
        var n = min(num_centers_per_dim, 4)  # Cap at 4^4 = 256 centers
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        var c = List[Scalar[DTYPE]]()
                        var ti = (
                            Scalar[DTYPE](i) / Scalar[DTYPE](n - 1) if n
                            > 1 else 0.5
                        )
                        var tj = (
                            Scalar[DTYPE](j) / Scalar[DTYPE](n - 1) if n
                            > 1 else 0.5
                        )
                        var tk = (
                            Scalar[DTYPE](k) / Scalar[DTYPE](n - 1) if n
                            > 1 else 0.5
                        )
                        var tl = (
                            Scalar[DTYPE](l) / Scalar[DTYPE](n - 1) if n
                            > 1 else 0.5
                        )
                        c.append(
                            state_low[0] + ti * (state_high[0] - state_low[0])
                        )
                        c.append(
                            state_low[1] + tj * (state_high[1] - state_low[1])
                        )
                        c.append(
                            state_low[2] + tk * (state_high[2] - state_low[2])
                        )
                        c.append(
                            state_low[3] + tl * (state_high[3] - state_low[3])
                        )
                        centers.append(c^)

    return centers^


def make_mountain_car_poly_features[
    DTYPE: DType
](degree: Int = 3) -> PolynomialFeatures[DTYPE]:
    """Create polynomial features for MountainCar (2D state) with normalization.

    MountainCar state: [position, velocity]
    - position: [-1.2, 0.6]
    - velocity: [-0.07, 0.07]

    Args:
        degree: Maximum polynomial degree.

    Returns:
        PolynomialFeatures extractor configured for MountainCar with normalization.
    """
    var state_low = List[Scalar[DTYPE]]()
    state_low.append(-1.2)  # position min
    state_low.append(-0.07)  # velocity min

    var state_high = List[Scalar[DTYPE]]()
    state_high.append(0.6)  # position max
    state_high.append(0.07)  # velocity max

    return PolynomialFeatures[DTYPE](
        state_dim=2,
        degree=degree,
        include_bias=True,
        state_low=state_low^,
        state_high=state_high^,
    )
