"""PPOAgent — user-facing facade over PPOTrainer + on-policy drivers.

On-policy continuous PPO. Two training entry points mirror the off-
policy agents:

  * `train(env, ...)`       — `BatchedEnv`, same-target (cpu+cpu /
                              gpu+gpu) at any `N_ENVS >= 1`. Routes to
                              `run_onpolicy_train_batched`.
  * `train_single(env, ...)` — `BoxContinuousActionEnv`, single-env.
                               Routes to `run_onpolicy_train`. Use this
                               for cross-target (cpu env + gpu trainer)
                               or for the simplest single-env loop.

Usage:

    var agent = PPOAgent[
        "cpu", ActorNet, CriticNet,
        OBS_DIM, ACT_DIM, ROLLOUT_LEN, MINIBATCH, N_EPOCHS,
    ](actor_lr=3e-4, critic_lr=1e-3, log_std_init=-0.5)
    var ep_returns = agent.train_single(env, total_timesteps=200_000)
"""

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.env_traits import BoxContinuousActionEnv

from ..training.batched_env import BatchedEnv
from ...io.artifact_sink import ArtifactSink
from ..training.driver_onpolicy import (
    run_onpolicy_train,
    run_onpolicy_train_batched,
)

from .metrics import PPOMetrics
from .trainer import PPOTrainer


struct PPOAgent[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](Movable & Deinitable):
    """Thin facade over `PPOTrainer` + on-policy drivers."""

    var trainer: PPOTrainer[
        Self.train_target, Self.ACTOR, Self.CRITIC,
        Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        Self.N_EPOCHS, Self.N_ENVS,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        gae_lambda: Scalar[DT] = 0.95,
        clip_eps: Scalar[DT] = 0.2,
        entropy_coef: Scalar[DT] = 0.0,
        action_scale: Scalar[DT] = 1.0,
        log_std_init: Scalar[DT] = -0.5,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1600.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Construct a PPOAgent. Forwards every kwarg to `PPOTrainer.make`.

        `max_grad_norm` is the canonical PPO L2 grad-norm clip (Schulman
        2017 default 0.5; distinct from `clip_eps`, the policy ratio
        surrogate clip). 0.0 disables clipping — bit-identical to the
        pre-clip code path.
        """
        self.trainer = PPOTrainer[
            Self.train_target, Self.ACTOR, Self.CRITIC,
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
            Self.N_EPOCHS, Self.N_ENVS,
        ].make(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            action_scale=action_scale,
            log_std_init=log_std_init,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            ctx=ctx,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BatchedEnv,
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        rng_seed: UInt64 = 42,
        print_every: Int = 5_000,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = "",
        checkpoint_every: Int = 0,
    ) raises -> List[Scalar[DT]]:
        """On-policy training via `run_onpolicy_train_batched`.

        Covers same-target (cpu+cpu, gpu+gpu) at the trainer's
        compile-time `N_ENVS`. For single-env / cross-target, use
        `train_single()` instead.

        See `SACAgent.train_single` for `diag_every` / `checkpoint_*`
        semantics.
        """
        var ctx = self.trainer.state.ctx
        return run_onpolicy_train_batched[
            PPOTrainer[
                Self.train_target, Self.ACTOR, Self.CRITIC,
                Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN,
                Self.MINIBATCH, Self.N_EPOCHS, Self.N_ENVS,
            ],
            E,
            L,
        ](
            ctx,
            self.trainer,
            env,
            total_timesteps,
            rng_seed=rng_seed,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
        )

    def train_single[
        E: BoxContinuousActionEnv,
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
        """Single-env on-policy training via `run_onpolicy_train`. Covers
        `(env=cpu, train=cpu)` and `(env=cpu, train=gpu)` cross-target.

        See `SACAgent.train_single` for `diag_every` / `checkpoint_*`
        semantics."""
        return run_onpolicy_train[
            PPOTrainer[
                Self.train_target, Self.ACTOR, Self.CRITIC,
                Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN,
                Self.MINIBATCH, Self.N_EPOCHS, Self.N_ENVS,
            ],
            E,
            L,
        ](
            self.trainer,
            env,
            total_timesteps,
            obs_dim=Self.OBS_DIM,
            act_dim=Self.ACT_DIM,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
        )

    # ─── Single-step inference (host-list interface) ───────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Stochastic action sample from the Gaussian policy."""
        self.trainer.select_action(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic action — actor's mean (no Gaussian sample)."""
        self.trainer.select_greedy_action(obs, action_out)

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

    # ─── Checkpointing (CPU only) ──────────────────────────────────────

    def save(mut self, path: String) raises:
        """Thin passthrough to `trainer.save_state(path)`. Writes ONE
        file (v3 `storage-ckpt` file) with prefixed sections for
        actor, critic, actor_opt, critic_opt. Rollout buffer NOT
        included (on-policy resume re-rolls). CPU-only."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. No target nets in PPO; nothing to hard-copy."""
        self.trainer.load_state(path)
