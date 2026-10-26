"""Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).

PPO is a state-of-the-art policy gradient algorithm that improves training
stability through clipped surrogate objectives. Combined with GAE for
advantage estimation, it provides a good balance of sample efficiency and
stability.

Key components:
1. **GAE (Generalized Advantage Estimation)**: Computes advantages using
   exponentially-weighted average of TD residuals:
   A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)

2. **PPO Clipping**: Constrains policy updates to prevent large changes:
   L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
   where r(θ) = π_θ(a|s) / π_θ_old(a|s)

References:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Schulman et al. (2016): "High-Dimensional Continuous Control Using GAE"

Example usage:
    from noeira.core.tile_coding import make_cartpole_tile_coding
    from noeira.agents.ppo import PPOAgent

    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)
    var agent = PPOAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=0.0003,
        critic_lr=0.001,
        clip_epsilon=0.2,
        gae_lambda=0.95,
    )

    # Collect rollout
    for step in range(rollout_length):
        var tiles = tc.get_tiles_simd4(obs)
        var action = agent.select_action(tiles)
        var log_prob = agent.get_log_prob(tiles, action)
        # ... environment step ...
        agent.store_transition(tiles, action, reward, log_prob)

    # Update at end of rollout
    agent.update(next_tiles, done)
"""

from std.math import exp, log
from std.random import random_float64
from noeira.core.tile_coding import TileCoding
from noeira.core import BoxDiscreteActionEnv, RenderableEnv, TrainingMetrics
from noeira.core.utils.gae import compute_gae, compute_returns_from_advantages
from noeira.core.utils.softmax import softmax, sample_from_probs, argmax_probs
from noeira.core.utils.normalization import normalize
from noeira.core.utils.shuffle import shuffle_indices


struct PPOAgent(Copyable, ImplicitlyCopyable, Movable):
    """Proximal Policy Optimization agent with tile coding.

    Uses clipped surrogate objective for stable policy updates and
    GAE for advantage estimation.
    """

    # Actor (policy) parameters: θ[action][tile]
    var theta: List[List[Float64]]

    # Critic (value) parameters: w[tile]
    var critic_weights: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var normalize_advantages: Bool

    # Rollout buffer
    var buffer_tiles: List[List[Int]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Float64]
    var buffer_log_probs: List[Float64]  # Old log probabilities
    var buffer_values: List[Float64]  # Old value estimates

    def __init__(
        out self,
        tile_coding: TileCoding[DType.float64],
        num_actions: Int,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        normalize_advantages: Bool = True,
        init_value: Float64 = 0.0,
    ):
        """Initialize PPO agent.

        Args:
            tile_coding: TileCoding instance defining feature representation.
            num_actions: Number of discrete actions.
            actor_lr: Actor (policy) learning rate (default 0.0003).
            critic_lr: Critic (value) learning rate (default 0.001).
            discount_factor: Discount factor γ (default 0.99).
            gae_lambda: GAE parameter λ (default 0.95).
            clip_epsilon: PPO clipping parameter ε (default 0.2).
            entropy_coef: Entropy bonus coefficient (default 0.01).
            value_loss_coef: Value loss coefficient (default 0.5).
            num_epochs: Number of optimization epochs per update (default 4).
            normalize_advantages: Whether to normalize advantages (default True).
            init_value: Initial parameter value (default 0.0).
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.normalize_advantages = normalize_advantages

        # Initialize actor parameters
        self.theta = List[List[Float64]]()
        for _ in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize critic parameters
        self.critic_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)

        # Initialize rollout buffer
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    def __init__(out self, *, copy: Self):
        self.num_actions = copy.num_actions
        self.num_tiles = copy.num_tiles
        self.num_tilings = copy.num_tilings
        self.actor_lr = copy.actor_lr
        self.critic_lr = copy.critic_lr
        self.discount_factor = copy.discount_factor
        self.gae_lambda = copy.gae_lambda
        self.clip_epsilon = copy.clip_epsilon
        self.entropy_coef = copy.entropy_coef
        self.value_loss_coef = copy.value_loss_coef
        self.num_epochs = copy.num_epochs
        self.normalize_advantages = copy.normalize_advantages
        self.theta = List[List[Float64]]()
        for a in range(copy.num_actions):
            var action_params = List[Float64]()
            for t in range(copy.num_tiles):
                action_params.append(copy.theta[a][t])
            self.theta.append(action_params^)
        self.critic_weights = List[Float64]()
        for t in range(copy.num_tiles):
            self.critic_weights.append(copy.critic_weights[t])
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.num_actions = move.num_actions
        self.num_tiles = move.num_tiles
        self.num_tilings = move.num_tilings
        self.actor_lr = move.actor_lr
        self.critic_lr = move.critic_lr
        self.discount_factor = move.discount_factor
        self.gae_lambda = move.gae_lambda
        self.clip_epsilon = move.clip_epsilon
        self.entropy_coef = move.entropy_coef
        self.value_loss_coef = move.value_loss_coef
        self.num_epochs = move.num_epochs
        self.normalize_advantages = move.normalize_advantages
        self.theta = move.theta^
        self.critic_weights = move.critic_weights^
        self.buffer_tiles = move.buffer_tiles^
        self.buffer_actions = move.buffer_actions^
        self.buffer_rewards = move.buffer_rewards^
        self.buffer_log_probs = move.buffer_log_probs^
        self.buffer_values = move.buffer_values^

    def _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences (logits).

        h(s, a) = θ_a · φ(s)
        """
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    def get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities π(·|s).

        Args:
            tiles: Active tile indices.

        Returns:
            Probability distribution over actions.
        """
        var prefs = self._get_action_preferences(tiles)
        return softmax(prefs)

    def get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate V(s).

        Args:
            tiles: Active tile indices.

        Returns:
            Estimated state value.
        """
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    def get_log_prob(self, tiles: List[Int], action: Int) -> Float64:
        """Get log probability of action under current policy.

        Args:
            tiles: Active tile indices.
            action: Action index.

        Returns:
            Log π(a|s).
        """
        var probs = self.get_action_probs(tiles)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0  # log(1e-10) ≈ -23

    def select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy π(a|s).

        Args:
            tiles: Active tile indices.

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

    def store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        log_prob: Float64,
        value: Float64,
    ):
        """Store transition in rollout buffer.

        Args:
            tiles: Active tiles for current state.
            action: Action taken.
            reward: Reward received.
            log_prob: Log probability of action (from old policy).
            value: Value estimate (from old critic).
        """
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.buffer_tiles.append(tiles_copy^)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    def _get_value_idx(self, buffer_idx: Int) -> Float64:
        """Get current value estimate for buffer step."""
        var value: Float64 = 0.0
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for i in range(num_tiles):
            var tile_idx = self.buffer_tiles[buffer_idx][i]
            value += self.critic_weights[tile_idx]
        return value

    def _get_action_probs_idx(self, buffer_idx: Int) -> List[Float64]:
        """Get current action probabilities for buffer step."""
        var preferences = List[Float64]()
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.buffer_tiles[buffer_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return softmax(preferences)

    def _get_log_prob_idx(self, buffer_idx: Int, action: Int) -> Float64:
        """Get current log probability for buffer step."""
        var probs = self._get_action_probs_idx(buffer_idx)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0

    def update(
        mut self,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update policy and value function using PPO.

        Performs multiple epochs of optimization on collected rollout.

        Args:
            next_tiles: Tiles for next state (for bootstrap).
            done: Whether episode terminated.
        """
        var buffer_len = len(self.buffer_tiles)
        if buffer_len == 0:
            return

        # Get bootstrap value
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.get_value(next_tiles)

        # Compute GAE advantages
        var advantages = compute_gae(
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            done,
            self.discount_factor,
            self.gae_lambda,
        )

        # Compute returns for value function
        var returns = compute_returns_from_advantages(
            advantages, self.buffer_values
        )

        # Normalize advantages (optional but recommended)
        if self.normalize_advantages and buffer_len > 1:
            normalize(advantages)

        # Multiple epochs of optimization
        var actor_step = self.actor_lr / Float64(self.num_tilings)
        var critic_step = self.critic_lr / Float64(self.num_tilings)

        for _ in range(self.num_epochs):
            for t in range(buffer_len):
                var action = self.buffer_actions[t]
                var old_log_prob = self.buffer_log_probs[t]
                var advantage = advantages[t]
                var return_t = returns[t]
                var num_tiles_t = len(self.buffer_tiles[t])

                # Current policy probability
                var new_log_prob = self._get_log_prob_idx(t, action)

                # Probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                var ratio = exp(new_log_prob - old_log_prob)

                # Clipped surrogate objective
                # var surr1 = ratio * advantage
                # var clipped_ratio: Float64
                # if advantage >= 0:
                #     clipped_ratio = min(ratio, 1.0 + self.clip_epsilon)
                # else:
                #     clipped_ratio = max(ratio, 1.0 - self.clip_epsilon)
                # var surr2 = clipped_ratio * advantage

                # Note: We don't use policy_loss directly since we compute gradients
                # manually below. The clipping check uses surr1 vs surr2.

                # Value function update (gradient of squared error)
                var current_value = self._get_value_idx(t)
                # Note: value_loss = (return_t - current_value)^2, gradient is 2*(current - return)

                # Get current probabilities for entropy and gradient
                var probs = self._get_action_probs_idx(t)

                # Entropy bonus (negative entropy because we minimize loss)
                var entropy: Float64 = 0.0
                for a in range(self.num_actions):
                    if probs[a] > 1e-10:
                        entropy -= probs[a] * log(probs[a])

                # Update critic (minimize value loss)
                var value_grad = 2.0 * (current_value - return_t)
                for i in range(num_tiles_t):
                    var tile_idx = self.buffer_tiles[t][i]
                    self.critic_weights[tile_idx] -= (
                        critic_step * self.value_loss_coef * value_grad
                    )

                # Update actor (minimize policy loss - entropy bonus)
                # Gradient of clipped objective is:
                # - If not clipped: advantage * ∇log π(a|s)
                # - If clipped: 0 (no gradient)
                var clipped = (ratio < 1.0 - self.clip_epsilon) or (
                    ratio > 1.0 + self.clip_epsilon
                )

                if not clipped:
                    for a in range(self.num_actions):
                        # Policy gradient for softmax
                        var policy_grad: Float64
                        if a == action:
                            policy_grad = 1.0 - probs[a]
                        else:
                            policy_grad = -probs[a]

                        # Entropy gradient: encourages uniform distribution
                        var entropy_grad: Float64 = 0.0
                        if probs[a] > 1e-10:
                            # ∂H/∂θ_a for softmax
                            entropy_grad = (
                                -probs[a]
                                * (1.0 + log(probs[a]))
                                * probs[a]
                                * (1.0 - probs[a])
                            )

                        # Combined gradient (advantage * policy grad - entropy_coef * entropy grad)
                        # We maximize: advantage * log_prob + entropy_coef * entropy
                        # So gradient is: advantage * policy_grad + entropy_coef * entropy_grad
                        var total_grad = (
                            advantage * policy_grad
                            + self.entropy_coef * entropy_grad
                        )

                        for i in range(num_tiles_t):
                            var tile_idx = self.buffer_tiles[t][i]
                            self.theta[a][tile_idx] += actor_step * total_grad

        # Clear buffer
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    def reset(mut self):
        """Reset buffer for new episode."""
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    def get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy at given state.

        Args:
            tiles: Active tile indices.

        Returns:
            Policy entropy (in nats).
        """
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
        rollout_length: Int = 2048,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the PPO agent on a continuous-state environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            rollout_length: Steps to collect before each PPO update.
            verbose: Whether to print progress.
            print_every: Print frequency when verbose.
            environment_name: Name for logging.

        Returns:
            Training metrics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="PPO",
            environment_name=environment_name,
        )

        var total_steps = 0
        for episode in range(num_episodes):
            var obs_f64 = _ppo_obs_to_f64(env.reset_obs_list())
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles(obs_f64)
                var action = self.select_action(tiles)
                var log_prob = self.get_log_prob(tiles, action)
                var value = self.get_value(tiles)

                var result = env.step_obs(action)
                var next_obs_f64 = _ppo_obs_to_f64(result[0])
                var reward = result[1]
                var done = result[2]

                self.store_transition(
                    tiles, action, Float64(reward), log_prob, value
                )
                total_reward += Float64(reward)
                steps += 1
                total_steps += 1

                # Update at rollout boundary or episode end
                if total_steps % rollout_length == 0 or done:
                    var next_tiles = tile_coding.get_tiles(next_obs_f64)
                    self.update(next_tiles, done)

                obs_f64 = next_obs_f64^
                if done:
                    break

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
        num_episodes: Int = 100,
        max_steps_per_episode: Int = 500,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the PPO agent using greedy policy.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            render: Whether to render the environment visually.
            frame_delay_ms: Delay in milliseconds between frames when rendering.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for _ in range(num_episodes):
            if quit_requested:
                break

            var obs_f64 = _ppo_obs_to_f64(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
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
                obs_f64 = _ppo_obs_to_f64(result[0])
                if done:
                    break

            total_reward += episode_reward

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)


struct PPOAgentWithMinibatch(Copyable, ImplicitlyCopyable, Movable):
    """PPO Agent with minibatch updates for larger rollouts.

    Extends PPOAgent with minibatch sampling during updates,
    which is more efficient for longer rollouts.
    """

    # Actor (policy) parameters
    var theta: List[List[Float64]]

    # Critic (value) parameters
    var critic_weights: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var normalize_advantages: Bool

    # Rollout buffer
    var buffer_tiles: List[List[Int]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Float64]
    var buffer_log_probs: List[Float64]
    var buffer_values: List[Float64]

    def __init__(
        out self,
        tile_coding: TileCoding[DType.float64],
        num_actions: Int,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        normalize_advantages: Bool = True,
        init_value: Float64 = 0.0,
    ):
        """Initialize PPO agent with minibatch updates.

        Args:
            tile_coding: TileCoding instance.
            num_actions: Number of discrete actions.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
            discount_factor: Discount factor γ.
            gae_lambda: GAE parameter λ.
            clip_epsilon: PPO clipping parameter ε.
            entropy_coef: Entropy bonus coefficient.
            value_loss_coef: Value loss coefficient.
            num_epochs: Number of optimization epochs.
            minibatch_size: Size of minibatches for updates.
            normalize_advantages: Whether to normalize advantages.
            init_value: Initial parameter value.
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantages = normalize_advantages

        # Initialize actor parameters
        self.theta = List[List[Float64]]()
        for _ in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize critic parameters
        self.critic_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)

        # Initialize rollout buffer
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    def __init__(out self, *, copy: Self):
        self.num_actions = copy.num_actions
        self.num_tiles = copy.num_tiles
        self.num_tilings = copy.num_tilings
        self.actor_lr = copy.actor_lr
        self.critic_lr = copy.critic_lr
        self.discount_factor = copy.discount_factor
        self.gae_lambda = copy.gae_lambda
        self.clip_epsilon = copy.clip_epsilon
        self.entropy_coef = copy.entropy_coef
        self.value_loss_coef = copy.value_loss_coef
        self.num_epochs = copy.num_epochs
        self.minibatch_size = copy.minibatch_size
        self.normalize_advantages = copy.normalize_advantages
        self.theta = List[List[Float64]]()
        for a in range(copy.num_actions):
            var action_params = List[Float64]()
            for t in range(copy.num_tiles):
                action_params.append(copy.theta[a][t])
            self.theta.append(action_params^)
        self.critic_weights = List[Float64]()
        for t in range(copy.num_tiles):
            self.critic_weights.append(copy.critic_weights[t])
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.num_actions = move.num_actions
        self.num_tiles = move.num_tiles
        self.num_tilings = move.num_tilings
        self.actor_lr = move.actor_lr
        self.critic_lr = move.critic_lr
        self.discount_factor = move.discount_factor
        self.gae_lambda = move.gae_lambda
        self.clip_epsilon = move.clip_epsilon
        self.entropy_coef = move.entropy_coef
        self.value_loss_coef = move.value_loss_coef
        self.num_epochs = move.num_epochs
        self.minibatch_size = move.minibatch_size
        self.normalize_advantages = move.normalize_advantages
        self.theta = move.theta^
        self.critic_weights = move.critic_weights^
        self.buffer_tiles = move.buffer_tiles^
        self.buffer_actions = move.buffer_actions^
        self.buffer_rewards = move.buffer_rewards^
        self.buffer_log_probs = move.buffer_log_probs^
        self.buffer_values = move.buffer_values^

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

    def get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate."""
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    def get_log_prob(self, tiles: List[Int], action: Int) -> Float64:
        """Get log probability of action."""
        var probs = self.get_action_probs(tiles)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0

    def select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy."""
        var probs = self.get_action_probs(tiles)
        return sample_from_probs(probs)

    def get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        var probs = self.get_action_probs(tiles)
        return argmax_probs(probs)

    def store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        log_prob: Float64,
        value: Float64,
    ):
        """Store transition in buffer."""
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.buffer_tiles.append(tiles_copy^)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    def _get_value_idx(self, buffer_idx: Int) -> Float64:
        """Get current value for buffer step."""
        var value: Float64 = 0.0
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for i in range(num_tiles):
            var tile_idx = self.buffer_tiles[buffer_idx][i]
            value += self.critic_weights[tile_idx]
        return value

    def _get_action_probs_idx(self, buffer_idx: Int) -> List[Float64]:
        """Get current action probabilities for buffer step."""
        var preferences = List[Float64]()
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.buffer_tiles[buffer_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return softmax(preferences)

    def _get_log_prob_idx(self, buffer_idx: Int, action: Int) -> Float64:
        """Get current log probability for buffer step."""
        var probs = self._get_action_probs_idx(buffer_idx)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0

    def update(
        mut self,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update using minibatch PPO.

        Args:
            next_tiles: Tiles for next state.
            done: Whether episode terminated.
        """
        var buffer_len = len(self.buffer_tiles)
        if buffer_len == 0:
            return

        # Get bootstrap value
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.get_value(next_tiles)

        # Compute GAE advantages
        var advantages = compute_gae(
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            done,
            self.discount_factor,
            self.gae_lambda,
        )

        # Compute returns
        var returns = compute_returns_from_advantages(
            advantages, self.buffer_values
        )

        # Normalize advantages
        if self.normalize_advantages and buffer_len > 1:
            normalize(advantages)

        var actor_step = self.actor_lr / Float64(self.num_tilings)
        var critic_step = self.critic_lr / Float64(self.num_tilings)

        # Multiple epochs with minibatch updates
        for _ in range(self.num_epochs):
            var indices = List[Int]()
            shuffle_indices(buffer_len, indices)

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = min(
                    batch_start + self.minibatch_size, buffer_len
                )

                # Process minibatch
                for b in range(batch_start, batch_end):
                    var t = indices[b]
                    var action = self.buffer_actions[t]
                    var old_log_prob = self.buffer_log_probs[t]
                    var advantage = advantages[t]
                    var return_t = returns[t]
                    var num_tiles_t = len(self.buffer_tiles[t])

                    # Current policy probability
                    var new_log_prob = self._get_log_prob_idx(t, action)

                    # Probability ratio
                    var ratio = exp(new_log_prob - old_log_prob)

                    # Clipping
                    var clipped = (ratio < 1.0 - self.clip_epsilon) or (
                        ratio > 1.0 + self.clip_epsilon
                    )

                    # Update critic
                    var current_value = self._get_value_idx(t)
                    var value_grad = 2.0 * (current_value - return_t)
                    for i in range(num_tiles_t):
                        var tile_idx = self.buffer_tiles[t][i]
                        self.critic_weights[tile_idx] -= (
                            critic_step * self.value_loss_coef * value_grad
                        )

                    # Update actor if not clipped
                    if not clipped:
                        var probs = self._get_action_probs_idx(t)

                        for a in range(self.num_actions):
                            var policy_grad: Float64
                            if a == action:
                                policy_grad = 1.0 - probs[a]
                            else:
                                policy_grad = -probs[a]

                            var entropy_grad: Float64 = 0.0
                            if probs[a] > 1e-10:
                                entropy_grad = (
                                    -probs[a]
                                    * (1.0 + log(probs[a]))
                                    * probs[a]
                                    * (1.0 - probs[a])
                                )

                            var total_grad = (
                                advantage * policy_grad
                                + self.entropy_coef * entropy_grad
                            )

                            for i in range(num_tiles_t):
                                var tile_idx = self.buffer_tiles[t][i]
                                self.theta[a][tile_idx] += (
                                    actor_step * total_grad
                                )

                batch_start = batch_end

        # Clear buffer
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    def reset(mut self):
        """Reset buffer."""
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

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
        rollout_length: Int = 2048,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the PPO agent with minibatch updates.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            rollout_length: Steps to collect before each PPO update.
            verbose: Whether to print progress.
            print_every: Print frequency when verbose.
            environment_name: Name for logging.

        Returns:
            Training metrics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="PPO (Minibatch)",
            environment_name=environment_name,
        )

        var total_steps = 0
        for episode in range(num_episodes):
            var obs_f64 = _ppo_obs_to_f64(env.reset_obs_list())
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles(obs_f64)
                var action = self.select_action(tiles)
                var log_prob = self.get_log_prob(tiles, action)
                var value = self.get_value(tiles)

                var result = env.step_obs(action)
                var next_obs_f64 = _ppo_obs_to_f64(result[0])
                var reward = result[1]
                var done = result[2]

                self.store_transition(
                    tiles, action, Float64(reward), log_prob, value
                )
                total_reward += Float64(reward)
                steps += 1
                total_steps += 1

                # Update at rollout boundary or episode end
                if total_steps % rollout_length == 0 or done:
                    var next_tiles = tile_coding.get_tiles(next_obs_f64)
                    self.update(next_tiles, done)

                obs_f64 = next_obs_f64^
                if done:
                    break

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
        num_episodes: Int = 100,
        max_steps_per_episode: Int = 500,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the PPO agent using greedy policy.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            render: Whether to render the environment visually.
            frame_delay_ms: Delay in milliseconds between frames when rendering.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for _ in range(num_episodes):
            if quit_requested:
                break

            var obs_f64 = _ppo_obs_to_f64(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
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
                obs_f64 = _ppo_obs_to_f64(result[0])
                if done:
                    break

            total_reward += episode_reward

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)


def _ppo_obs_to_f64[
    DTYPE: DType
](obs: List[Scalar[DTYPE]]) -> List[Scalar[DType.float64]]:
    """Convert observation list to Float64."""
    var result = List[Scalar[DType.float64]](capacity=len(obs))
    for i in range(len(obs)):
        result.append(Scalar[DType.float64](obs[i]))
    return result^
