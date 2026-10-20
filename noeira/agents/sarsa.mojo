from std.random import random_si64, random_float64
from .qlearning import QTable
from noeira.core import (
    train_tabular,
    evaluate_tabular,
    TabularAgent,
    DiscreteEnv,
    RenderableEnv,
    TrainingMetrics,
)


struct SARSAAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """Tabular SARSA agent with epsilon-greedy exploration.

    SARSA is on-policy: uses Q(s',a') instead of max Q(s',a').
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

    def __init__(out self, *, copy: Self):
        self.q_table = copy.q_table
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.epsilon = copy.epsilon
        self.epsilon_decay = copy.epsilon_decay
        self.epsilon_min = copy.epsilon_min
        self.num_actions = copy.num_actions

    def __init__(out self, *, deinit move: Self):
        self.q_table = move.q_table^
        self.learning_rate = move.learning_rate
        self.discount_factor = move.discount_factor
        self.epsilon = move.epsilon
        self.epsilon_decay = move.epsilon_decay
        self.epsilon_min = move.epsilon_min
        self.num_actions = move.num_actions

    def __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
    ):
        self.q_table = QTable(num_states, num_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions

    def select_action(self, state_idx: Int) -> Int:
        var rand = random_float64()
        if rand < self.epsilon:
            return Int(random_si64(0, Int64(self.num_actions - 1)))
        else:
            return self.q_table.get_best_action(state_idx)

    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """TabularAgent interface (Q-learning style for generic training)."""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(
                next_state_idx
            )
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    def update_sarsa(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        next_action: Int,
        done: Bool,
    ):
        """True SARSA: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))."""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get(
                next_state_idx, next_action
            )
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    def decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> Float64:
        return self.epsilon

    def get_best_action(self, state_idx: Int) -> Int:
        return self.q_table.get_best_action(state_idx)

    def train[
        E: DiscreteEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 100,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent using true on-policy SARSA updates.

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="SARSA",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var state = env.reset()
            var state_idx = env.state_to_index(state)
            var action_idx = self.select_action(state_idx)
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var action = env.action_from_index(action_idx)
                var result = env.step(action)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                var next_state_idx = env.state_to_index(next_state)
                var next_action_idx = self.select_action(next_state_idx)

                # True SARSA update with next action
                self.update_sarsa(
                    state_idx,
                    action_idx,
                    Float64(reward),
                    next_state_idx,
                    next_action_idx,
                    done,
                )

                total_reward += Float64(reward)
                steps += 1
                state_idx = next_state_idx
                action_idx = next_action_idx

                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, self.epsilon)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    def evaluate[
        E: DiscreteEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Greedy eval — delegates to the shared `evaluate_tabular` loop
        (core/tabular_training.mojo)."""
        return evaluate_tabular(
            self,
            env,
            num_episodes=num_episodes,
            render=render,
            frame_delay_ms=frame_delay_ms,
        )
