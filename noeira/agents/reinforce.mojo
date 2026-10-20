"""REINFORCE (Monte Carlo Policy Gradient) Agent.

REINFORCE is the simplest policy gradient algorithm. It learns a parameterized
policy directly by performing gradient ascent on the expected return.

The policy is represented using a softmax over linear features (tile coding):
    π(a|s) = exp(θ_a · φ(s)) / Σ_a' exp(θ_a' · φ(s))

where φ(s) are the active tiles for state s.

Update rule (at end of episode):
    θ += α * G_t * ∇log π(a_t|s_t)

For softmax policy with linear features:
    ∇log π(a|s) = φ(s) - Σ_a' π(a'|s) * φ(s) = φ(s) - E[φ(s)]

Since φ(s) is the same for all actions (tile coding), this simplifies to:
    ∇log π(a|s)[a] = 1 - π(a|s)  (for chosen action)
    ∇log π(a|s)[a'] = -π(a'|s)   (for other actions)

References:
- Sutton & Barto, Chapter 13: "Policy Gradient Methods"
- Williams (1992): "Simple Statistical Gradient-Following Algorithms"

Example usage:
    from noeira.core.tile_coding import make_cartpole_tile_coding
    from noeira.agents.reinforce import REINFORCEAgent

    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)
    var agent = REINFORCEAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=0.001,
    )

    # Training loop
    var tiles = tc.get_tiles_simd4(obs)
    var action = agent.select_action(tiles)
    # ... environment step ...
    agent.store_transition(tiles, action, reward)
    if done:
        agent.update_from_episode()
"""

from std.math import exp, log
from std.random import random_float64
from noeira.core.tile_coding import TileCoding
from noeira.core import BoxDiscreteActionEnv, RenderableEnv, TrainingMetrics
from noeira.core.utils.softmax import softmax, sample_from_probs, argmax_probs


struct REINFORCEAgent(Copyable, ImplicitlyCopyable, Movable):
    """REINFORCE agent with tile coding function approximation.

    Uses softmax policy over tile-coded features.
    Updates policy parameters at the end of each episode using
    Monte Carlo returns.
    """

    # Policy parameters: θ[action][tile]
    var theta: List[List[Float64]]
    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64

    # Episode storage for Monte Carlo update
    var episode_tiles: List[List[Int]]
    var episode_actions: List[Int]
    var episode_rewards: List[Float64]

    # Optional baseline for variance reduction
    var use_baseline: Bool
    var baseline_weights: List[Float64]  # V(s) approximation
    var baseline_lr: Float64

    def __init__(
        out self,
        tile_coding: TileCoding[DType.float64],
        num_actions: Int,
        learning_rate: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        use_baseline: Bool = True,
        baseline_lr: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize REINFORCE agent.

        Args:
            tile_coding: TileCoding instance defining the feature representation.
            num_actions: Number of discrete actions.
            learning_rate: Policy learning rate α (default 0.001).
            discount_factor: Discount factor γ (default 0.99).
            use_baseline: Whether to use a learned value baseline (default True).
            baseline_lr: Learning rate for baseline (default 0.01).
            init_value: Initial parameter value (default 0.0).
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.use_baseline = use_baseline
        self.baseline_lr = baseline_lr

        # Initialize policy parameters θ
        self.theta = List[List[Float64]]()
        for _ in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize baseline (state value function approximation)
        self.baseline_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.baseline_weights.append(0.0)

        # Initialize episode storage
        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    def __init__(out self, *, copy: Self):
        self.num_actions = copy.num_actions
        self.num_tiles = copy.num_tiles
        self.num_tilings = copy.num_tilings
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.use_baseline = copy.use_baseline
        self.baseline_lr = copy.baseline_lr
        self.theta = List[List[Float64]]()
        for a in range(copy.num_actions):
            var action_params = List[Float64]()
            for t in range(copy.num_tiles):
                action_params.append(copy.theta[a][t])
            self.theta.append(action_params^)
        self.baseline_weights = List[Float64]()
        for t in range(copy.num_tiles):
            self.baseline_weights.append(copy.baseline_weights[t])
        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.num_actions = move.num_actions
        self.num_tiles = move.num_tiles
        self.num_tilings = move.num_tilings
        self.learning_rate = move.learning_rate
        self.discount_factor = move.discount_factor
        self.use_baseline = move.use_baseline
        self.baseline_lr = move.baseline_lr
        self.theta = move.theta^
        self.baseline_weights = move.baseline_weights^
        self.episode_tiles = move.episode_tiles^
        self.episode_actions = move.episode_actions^
        self.episode_rewards = move.episode_rewards^

    def _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences (logits) for given state.

        h(s, a) = θ_a · φ(s) = sum of θ[a][tile] for active tiles

        Args:
            tiles: Active tile indices from tile coding

        Returns:
            List of preferences, one per action
        """
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    def get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities for given state.

        Args:
            tiles: Active tile indices.

        Returns:
            Probability distribution over actions.
        """
        var prefs = self._get_action_preferences(tiles)
        return softmax(prefs)

    def select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy π(a|s).

        Args:
            tiles: Active tile indices from TileCoding.get_tiles().

        Returns:
            Sampled action index.
        """
        var probs = self.get_action_probs(tiles)
        return sample_from_probs(probs)

    def get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action (highest probability).

        Args:
            tiles: Active tile indices.

        Returns:
            Action with highest probability.
        """
        var probs = self.get_action_probs(tiles)
        return argmax_probs(probs)

    def _get_baseline_value(self, tiles: List[Int]) -> Float64:
        """Get baseline value estimate V(s).

        Args:
            tiles: Active tile indices.

        Returns:
            Estimated state value.
        """
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.baseline_weights[tiles[i]]
        return value

    def _get_baseline_value_idx(self, episode_idx: Int) -> Float64:
        """Get baseline value for episode step by index."""
        var value: Float64 = 0.0
        var num_tiles = len(self.episode_tiles[episode_idx])
        for i in range(num_tiles):
            var tile_idx = self.episode_tiles[episode_idx][i]
            value += self.baseline_weights[tile_idx]
        return value

    def _get_action_probs_idx(self, episode_idx: Int) -> List[Float64]:
        """Get action probabilities for episode step by index."""
        var preferences = List[Float64]()
        var num_tiles = len(self.episode_tiles[episode_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.episode_tiles[episode_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return softmax(preferences)

    def store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
    ):
        """Store transition for end-of-episode update.

        Args:
            tiles: Active tiles for current state.
            action: Action taken.
            reward: Reward received.
        """
        # Copy tiles since we need to store them
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.episode_tiles.append(tiles_copy^)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def update_from_episode(mut self):
        """Update policy parameters using REINFORCE.

        Called at end of episode. Computes returns and updates
        θ += α * G_t * ∇log π(a_t|s_t)
        """
        var num_steps = len(self.episode_tiles)
        if num_steps == 0:
            return

        # Compute returns (discounted sum of future rewards)
        var returns = List[Float64]()
        for _ in range(num_steps):
            returns.append(0.0)

        var g: Float64 = 0.0
        for t in range(num_steps - 1, -1, -1):
            g = self.episode_rewards[t] + self.discount_factor * g
            returns[t] = g

        # Update policy for each timestep
        var step_size = self.learning_rate / Float64(self.num_tilings)

        for t in range(num_steps):
            var action = self.episode_actions[t]
            var g_t = returns[t]
            var num_tiles_t = len(self.episode_tiles[t])

            # Optionally subtract baseline for variance reduction
            var advantage = g_t
            if self.use_baseline:
                var baseline = self._get_baseline_value_idx(t)
                advantage = g_t - baseline

                # Update baseline towards return
                var baseline_step = self.baseline_lr / Float64(self.num_tilings)
                var baseline_error = g_t - baseline
                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.baseline_weights[tile_idx] += (
                        baseline_step * baseline_error
                    )

            # Compute action probabilities
            var probs = self._get_action_probs_idx(t)

            # Update θ using policy gradient
            # ∇log π(a|s) for softmax:
            #   For chosen action a: 1 - π(a|s)
            #   For other actions a': -π(a'|s)
            for a in range(self.num_actions):
                var grad: Float64
                if a == action:
                    grad = 1.0 - probs[a]
                else:
                    grad = -probs[a]

                # θ_a += α * G_t * ∇log π(a|s)
                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.theta[a][tile_idx] += step_size * advantage * grad

        # Clear episode storage
        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def reset(mut self):
        """Reset episode storage for new episode."""
        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute entropy of policy at given state.

        H(π) = -Σ_a π(a|s) log π(a|s)

        Higher entropy = more exploration.

        Args:
            tiles: Active tile indices.

        Returns:
            Policy entropy (in nats).
        """
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:  # Avoid log(0)
                entropy -= probs[a] * log(probs[a])
        return entropy

    def train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        tile_coding: TileCoding[DType.float64],
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment using REINFORCE.

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="REINFORCE",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Clear episode storage
            var obs_f64 = _reinforce_obs_to_f64(env.reset_obs_list())
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles(obs_f64)
                var action = self.select_action(tiles)

                var result = env.step_obs(action)
                var reward = result[1]
                var done = result[2]

                self.store_transition(tiles, action, Float64(reward))

                total_reward += Float64(reward)
                steps += 1
                obs_f64 = _reinforce_obs_to_f64(result[0])

                if done:
                    break

            # REINFORCE updates at end of episode
            self.update_from_episode()
            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    def evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        tile_coding: TileCoding[DType.float64],
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment visually.
            frame_delay_ms: Delay in milliseconds between frames when rendering.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for _ in range(num_episodes):
            if quit_requested:
                break

            var obs_f64 = _reinforce_obs_to_f64(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                var tiles = tile_coding.get_tiles(obs_f64)
                var action = self.get_best_action(tiles)

                var result = env.step_obs(action)
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += Float64(reward)
                obs_f64 = _reinforce_obs_to_f64(result[0])

                if done:
                    break

            total_reward += episode_reward

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)


struct REINFORCEWithEntropyAgent(Copyable, ImplicitlyCopyable, Movable):
    """REINFORCE with entropy regularization for better exploration.

    Adds entropy bonus to encourage exploration:
    J(θ) = E[Σ_t (G_t * log π(a_t|s_t) + β * H(π(·|s_t)))]

    where β is the entropy coefficient.
    """

    var theta: List[List[Float64]]
    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var entropy_coef: Float64

    var episode_tiles: List[List[Int]]
    var episode_actions: List[Int]
    var episode_rewards: List[Float64]

    var use_baseline: Bool
    var baseline_weights: List[Float64]
    var baseline_lr: Float64

    def __init__(
        out self,
        tile_coding: TileCoding[DType.float64],
        num_actions: Int,
        learning_rate: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        entropy_coef: Float64 = 0.01,
        use_baseline: Bool = True,
        baseline_lr: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize REINFORCE with entropy regularization.

        Args:
            tile_coding: TileCoding instance.
            num_actions: Number of discrete actions.
            learning_rate: Policy learning rate.
            discount_factor: Discount factor γ.
            entropy_coef: Entropy bonus coefficient β.
            use_baseline: Whether to use learned baseline.
            baseline_lr: Baseline learning rate.
            init_value: Initial parameter value.
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.baseline_lr = baseline_lr

        self.theta = List[List[Float64]]()
        for _ in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        self.baseline_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.baseline_weights.append(0.0)

        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    def __init__(out self, *, copy: Self):
        self.num_actions = copy.num_actions
        self.num_tiles = copy.num_tiles
        self.num_tilings = copy.num_tilings
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.entropy_coef = copy.entropy_coef
        self.use_baseline = copy.use_baseline
        self.baseline_lr = copy.baseline_lr
        self.theta = List[List[Float64]]()
        for a in range(copy.num_actions):
            var action_params = List[Float64]()
            for t in range(copy.num_tiles):
                action_params.append(copy.theta[a][t])
            self.theta.append(action_params^)
        self.baseline_weights = List[Float64]()
        for t in range(copy.num_tiles):
            self.baseline_weights.append(copy.baseline_weights[t])
        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.num_actions = move.num_actions
        self.num_tiles = move.num_tiles
        self.num_tilings = move.num_tilings
        self.learning_rate = move.learning_rate
        self.discount_factor = move.discount_factor
        self.entropy_coef = move.entropy_coef
        self.use_baseline = move.use_baseline
        self.baseline_lr = move.baseline_lr
        self.theta = move.theta^
        self.baseline_weights = move.baseline_weights^
        self.episode_tiles = move.episode_tiles^
        self.episode_actions = move.episode_actions^
        self.episode_rewards = move.episode_rewards^

    def _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences."""
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    def get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities."""
        var prefs = self._get_action_preferences(tiles)
        return softmax(prefs)

    def select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy."""
        var probs = self.get_action_probs(tiles)
        return sample_from_probs(probs)

    def get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        var probs = self.get_action_probs(tiles)
        return argmax_probs(probs)

    def _get_baseline_value(self, tiles: List[Int]) -> Float64:
        """Get baseline value estimate."""
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.baseline_weights[tiles[i]]
        return value

    def _get_baseline_value_idx(self, episode_idx: Int) -> Float64:
        """Get baseline value for episode step by index."""
        var value: Float64 = 0.0
        var num_tiles = len(self.episode_tiles[episode_idx])
        for i in range(num_tiles):
            var tile_idx = self.episode_tiles[episode_idx][i]
            value += self.baseline_weights[tile_idx]
        return value

    def _get_action_probs_idx(self, episode_idx: Int) -> List[Float64]:
        """Get action probabilities for episode step by index."""
        var preferences = List[Float64]()
        var num_tiles = len(self.episode_tiles[episode_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.episode_tiles[episode_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return softmax(preferences)

    def store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
    ):
        """Store transition."""
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])
        self.episode_tiles.append(tiles_copy^)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def update_from_episode(mut self):
        """Update with entropy regularization."""
        var num_steps = len(self.episode_tiles)
        if num_steps == 0:
            return

        var returns = List[Float64]()
        for _ in range(num_steps):
            returns.append(0.0)

        var g: Float64 = 0.0
        for t in range(num_steps - 1, -1, -1):
            g = self.episode_rewards[t] + self.discount_factor * g
            returns[t] = g

        var step_size = self.learning_rate / Float64(self.num_tilings)

        for t in range(num_steps):
            var action = self.episode_actions[t]
            var g_t = returns[t]
            var num_tiles_t = len(self.episode_tiles[t])

            var advantage = g_t
            if self.use_baseline:
                var baseline = self._get_baseline_value_idx(t)
                advantage = g_t - baseline
                var baseline_step = self.baseline_lr / Float64(self.num_tilings)
                var baseline_error = g_t - baseline
                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.baseline_weights[tile_idx] += (
                        baseline_step * baseline_error
                    )

            var probs = self._get_action_probs_idx(t)

            # Policy gradient with entropy bonus
            # ∇J = ∇log π(a|s) * A + β * ∇H(π)
            # ∇H(π) = -∇(Σ_a π log π) = -(1 + log π) for each action
            for a in range(self.num_actions):
                var policy_grad: Float64
                if a == action:
                    policy_grad = 1.0 - probs[a]
                else:
                    policy_grad = -probs[a]

                # Entropy gradient: ∂H/∂θ_a = -π(a)(1 + log π(a)) * ∂π/∂θ_a
                # For softmax: ∂π_a/∂θ_a = π_a(1 - π_a)
                # Simplified: encourages uniform distribution
                var entropy_grad: Float64 = 0.0
                if probs[a] > 1e-10:
                    entropy_grad = (
                        -probs[a]
                        * (1.0 + log(probs[a]))
                        * probs[a]
                        * (1.0 - probs[a])
                    )

                var total_grad = (
                    advantage * policy_grad + self.entropy_coef * entropy_grad
                )

                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.theta[a][tile_idx] += step_size * total_grad

        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def reset(mut self):
        """Reset episode storage."""
        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy."""
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy

    def train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        tile_coding: TileCoding[DType.float64],
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment.

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="REINFORCE + Entropy",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Clear episode storage
            var obs_f64 = _reinforce_obs_to_f64(env.reset_obs_list())
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles(obs_f64)
                var action = self.select_action(tiles)

                var result = env.step_obs(action)
                var reward = result[1]
                var done = result[2]

                self.store_transition(tiles, action, Float64(reward))

                total_reward += Float64(reward)
                steps += 1
                obs_f64 = _reinforce_obs_to_f64(result[0])

                if done:
                    break

            # REINFORCE updates at end of episode
            self.update_from_episode()
            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    def evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        tile_coding: TileCoding[DType.float64],
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment visually.
            frame_delay_ms: Delay in milliseconds between frames when rendering.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for _ in range(num_episodes):
            if quit_requested:
                break

            var obs_f64 = _reinforce_obs_to_f64(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                var tiles = tile_coding.get_tiles(obs_f64)
                var action = self.get_best_action(tiles)

                var result = env.step_obs(action)
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += Float64(reward)
                obs_f64 = _reinforce_obs_to_f64(result[0])

                if done:
                    break

            total_reward += episode_reward

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)


def _reinforce_obs_to_f64[
    DTYPE: DType
](obs: List[Scalar[DTYPE]]) -> List[Scalar[DType.float64]]:
    """Convert observation list to Float64."""
    var result = List[Scalar[DType.float64]](capacity=len(obs))
    for i in range(len(obs)):
        result.append(Scalar[DType.float64](obs[i]))
    return result^
