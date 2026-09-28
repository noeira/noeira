"""PPODiscreteAgent — facade over PPODiscreteTrainer + discrete driver.

On-policy categorical PPO. Single-env training entry point
(`train`) routes to `run_onpolicy_discrete_train`; `eval` routes to
`run_onpolicy_discrete_eval`. Mirrors the continuous `PPOAgent`
(see ppo/agent.mojo) but for `BoxDiscreteActionEnv`.

Usage:

    comptime ActorNet = Sequential[
        Linear[OBS, H], Tanh[H], Linear[H, H], Tanh[H], Linear[H, N_ACTIONS],
    ]
    comptime CriticNet = Sequential[
        Linear[OBS, H], Tanh[H], Linear[H, H], Tanh[H], Linear[H, 1],
    ]
    var agent = PPODiscreteAgent[
        "cpu", ActorNet, CriticNet, OBS, N_ACTIONS, ROLLOUT, MINIBATCH, EPOCHS,
    ](actor_lr=3e-4, critic_lr=1e-3, entropy_coef=0.01)
    var ep_returns = agent.train(env, total_timesteps=50_000)

For GPU: `PPODiscreteAgent["gpu", ...](ctx=ctx, ...)`.
"""

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.env_traits import BoxDiscreteActionEnv

from ..training.batched_env import BatchedEnv
from ..training.driver_onpolicy_discrete import (
    run_onpolicy_discrete_train,
    run_onpolicy_discrete_eval,
    run_onpolicy_discrete_train_batched,
)

from ..ppo.metrics import PPOMetrics
from ...io.artifact_sink import ArtifactSink
from .trainer import PPODiscreteTrainer


struct PPODiscreteAgent[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    N_ACTIONS: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](Movable & Deinitable):
    """Thin facade over `PPODiscreteTrainer` + the discrete on-policy
    driver."""

    comptime TrainerT = PPODiscreteTrainer[
        Self.train_target, Self.ACTOR, Self.CRITIC,
        Self.OBS_DIM, Self.N_ACTIONS, Self.ROLLOUT_LEN, Self.MINIBATCH,
        Self.N_EPOCHS, Self.N_ENVS,
    ]

    var trainer: Self.TrainerT

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        gae_lambda: Scalar[DT] = 0.95,
        clip_eps: Scalar[DT] = 0.2,
        entropy_coef: Scalar[DT] = 0.01,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = 0.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Construct a PPODiscreteAgent. Forwards every kwarg to
        `PPODiscreteTrainer.make`. `ctx` is required for
        `train_target='gpu'`."""
        self.trainer = Self.TrainerT.make(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            ctx=ctx,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BoxDiscreteActionEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        print_every: Int = 1_000,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = "",
        checkpoint_every: Int = 0,
    ) raises -> List[Scalar[DT]]:
        """Single-env discrete on-policy training via
        `run_onpolicy_discrete_train`. Covers `(env=cpu, train=cpu)`
        and `(env=cpu, train=gpu)`."""
        return run_onpolicy_discrete_train[Self.TrainerT, E, L](
            self.trainer,
            env,
            total_timesteps,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
        )

    def train_batched[
        E: BatchedEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        ctx: Optional[DeviceContext],
        mut env: E,
        total_env_steps: Int,
        *,
        rng_seed: UInt64 = UInt64(42),
        print_every: Int = 5_000,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = "",
        checkpoint_every: Int = 0,
        base_step: Int = 0,
        stop_at_mean_return: Optional[Scalar[DT]] = None,
        stop_min_episodes: Int = 0,
    ) raises -> List[Scalar[DT]]:
        """N_ENVS-wide discrete on-policy training via
        `run_onpolicy_discrete_train_batched`. Covers same-target
        `(env=cpu, train=cpu)` and `(env=gpu, train=gpu)` at the
        compile-time `N_ENVS`. For single-env / cross-target use `train`.

        `stop_at_mean_return` stops early once the windowed mean return
        reaches it with >= `stop_min_episodes` episodes done (see
        `_run_onpolicy_batched_body`)."""
        return run_onpolicy_discrete_train_batched[Self.TrainerT, E, L](
            ctx,
            self.trainer,
            env,
            total_env_steps,
            rng_seed=rng_seed,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
            base_step=base_step,
            stop_at_mean_return=stop_at_mean_return,
            stop_min_episodes=stop_min_episodes,
        )

    # ─── Evaluation ─────────────────────────────────────────────────────

    def eval[
        E: BoxDiscreteActionEnv,
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        verbose: Bool = False,
    ) raises -> Scalar[DT]:
        """Greedy eval — argmax over logits, no sampling."""
        return run_onpolicy_discrete_eval[Self.TrainerT, E](
            self.trainer,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
        )

    # ─── Single-step inference (host-list interface) ───────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        step_idx: Int,
    ) raises -> Int:
        """Categorical sample from the softmax policy."""
        return self.trainer.select_action(obs, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic argmax over logits."""
        return self.trainer.select_greedy_action(obs)

    # ─── Episode-tracker accessors ─────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.trainer.mean_return()

    def ep_count(self) -> Int:
        return self.trainer.ep_count()

    # ─── Metrics / logging passthrough ─────────────────────────────────

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        """Drain trainer accumulators into a PPOMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        return self.trainer.flush_timer_log()

    # ─── Checkpointing ──────────────────────────────────────────────────

    def save(mut self, path: String) raises:
        """Passthrough to `trainer.save_state(path)` — actor, critic,
        their opts. Rollout buffer NOT included (on-policy resume
        re-rolls)."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. No target nets in PPO."""
        self.trainer.load_state(path)
