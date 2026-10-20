"""Shared train / evaluate loops for tabular agents.

ONE copy of the episode loop that every tabular agent previously carried
as a byte-identical method pair (~125 lines × 11 agents — verified
identical modulo the `algorithm_name` string before extraction):

  * `train_tabular` — the epsilon-greedy episode loop: select → step →
    `agent.update(s, a, r, s', done)` → decay epsilon per episode →
    `TrainingMetrics` logging. Algorithm-specific behaviour lives inside
    the agent's `update` (Q-learning target, expected-SARSA expectation,
    Monte-Carlo trajectory accumulation, Dyna-Q planning, …) and the
    per-episode `begin_episode` hook (PER's beta anneal). SARSA is the
    one agent whose LOOP genuinely differs (on-policy: the next action
    is selected before the update) — it keeps its own `train`.
  * `evaluate_tabular` — the greedy (no-exploration) eval loop with
    optional env-owned rendering.

The agents keep thin `train` / `evaluate` methods delegating here, so
the public API (`agent.train(env, …)` in the solve_* examples) is
unchanged.
"""

from .tabular_agent import TabularAgent
from .env_traits import DiscreteEnv, RenderableEnv
from .metrics import TrainingMetrics


def train_tabular[
    A: TabularAgent, E: DiscreteEnv
](
    mut agent: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 100,
    verbose: Bool = False,
    print_every: Int = 100,
    algorithm_name: String = "Tabular",
    environment_name: String = "Environment",
) -> TrainingMetrics:
    """Train `agent` on `env` for `num_episodes` episodes.

    Args:
        agent: Any `TabularAgent`.
        env: The discrete environment to train on.
        num_episodes: Number of episodes to train.
        max_steps_per_episode: Maximum steps per episode.
        verbose: Whether to print progress.
        print_every: Print progress every N episodes (if verbose).
        algorithm_name: Algorithm label for metrics.
        environment_name: Environment label for metrics.

    Returns:
        TrainingMetrics object with episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    for episode in range(num_episodes):
        var state = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0

        # Per-episode hook (default no-op) — e.g. PER anneals its IS beta
        # towards 1.0 over training here.
        agent.begin_episode(episode, num_episodes)

        for _ in range(max_steps_per_episode):
            var state_idx = env.state_to_index(state)
            var action_idx = agent.select_action(state_idx)
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state_idx = env.state_to_index(next_state)
            agent.update(
                state_idx, action_idx, Float64(reward), next_state_idx, done
            )

            total_reward += Float64(reward)
            steps += 1
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        metrics.log_episode(episode, total_reward, steps, agent.get_epsilon())

        if verbose and (episode + 1) % print_every == 0:
            metrics.print_progress(episode, window=100)

    return metrics^


def evaluate_tabular[
    A: TabularAgent, E: DiscreteEnv & RenderableEnv
](
    agent: A,
    mut env: E,
    *,
    num_episodes: Int = 10,
    render: Bool = False,
    frame_delay_ms: Int = 16,
) raises -> Float64:
    """Greedy (no exploration) evaluation of `agent` on `env`.

    Args:
        agent: Any `TabularAgent` (read-only — greedy actions only).
        env: The discrete environment to evaluate on.
        num_episodes: Number of evaluation episodes.
        render: Whether to render the environment.
        frame_delay_ms: Delay between frames in milliseconds.

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
        var state = env.reset()
        var episode_reward: Float64 = 0.0

        for _ in range(1000):  # Max steps for evaluation
            var state_idx = env.state_to_index(state)
            var action_idx = agent.get_best_action(state_idx)
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            if render:
                env.render_frame()
                env.renderer_delay(frame_delay_ms)
                if env.check_renderer_quit():
                    quit_requested = True
                    break

            episode_reward += Float64(reward)
            state = next_state

            if done:
                break

        total_reward += episode_reward

    if render:
        env.close_renderer()

    return total_reward / Float64(num_episodes)
