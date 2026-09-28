"""REDQAgent — user-facing facade over `REDQTrainer` + off-policy drivers.

Mirrors `SACAgent` shape one-for-one (so call sites stay consistent
across the off-policy continuous family). The trainer owns the
algorithm (parameters, optimizers, replay buffer, training pipeline);
the off-policy drivers own the env loop. This agent blends the two so
users write:

    var agent = REDQAgent["cpu", SAMPLE, ACTOR, CRITIC,
                          N, N_MIN, UTD, POLICY_DELAY, MODE](actor_lr=...)
    var ep_returns = agent.train_single(env, total_timesteps=10_000)
    var mean_eval  = agent.eval(env, num_episodes=10)
    agent.save("/tmp/redq.bin")

instead of wiring `REDQTrainer.make(...)` + free-function
`run_offpolicy_train(trainer, env, ...)` by hand.

The wrapped trainer remains accessible as `agent.trainer` for power
users who want to compose custom training loops (multi-seed sweeps,
manual gradient inspection, etc.).

Two training entry points:

  * `train_single(env, ...)`  — single-env `BoxContinuousActionEnv`.
                                Routes to `run_offpolicy_train`. Used
                                in the R.4 / R.5 Pendulum smokes.
  * `eval(env, ...)`          — greedy `run_offpolicy_eval`.

R.5 ships single-env support; the batched `train()` entry point that
SAC has (over `BatchedEnv` via `run_offpolicy_train_batched`) is left
as a follow-up — it needs validation across the (env_target ×
N_ENVS) matrix that R.4 didn't touch.
"""

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.env_traits import BoxContinuousActionEnv

from ..training.blocks import SampleBlock
from ...io.artifact_sink import ArtifactSink
from ..training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_eval,
)

from .metrics import REDQMetrics
from .trainer import REDQTrainer


struct REDQAgent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
    USE_TRAIN_CUDA_GRAPH: Bool = False,
](Movable & Deinitable):
    """Thin facade over `REDQTrainer` + off-policy drivers.

    Comptime params mirror `REDQTrainer` one-for-one. Dimensions
    (OBS / ACT / BATCH) are derived from `SAMPLE`.
    """

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    var trainer: REDQTrainer[
        Self.train_target,
        Self.SAMPLE,
        Self.ACTOR,
        Self.CRITIC,
        Self.N,
        Self.N_MIN,
        Self.UTD,
        Self.POLICY_DELAY,
        Self.Q_MODE,
        Self.USE_TRAIN_CUDA_GRAPH,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 3e-4,
        alpha_lr: Scalar[DT] = 3e-4,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        init_alpha: Scalar[DT] = 0.2,
        target_entropy: Scalar[DT] = -1.0,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
        max_grad_norm: Scalar[DT] = 0.0,
    ) raises:
        """Forward every kwarg to `REDQTrainer.make`."""
        self.trainer = REDQTrainer[
            Self.train_target,
            Self.SAMPLE,
            Self.ACTOR,
            Self.CRITIC,
            Self.N,
            Self.N_MIN,
            Self.UTD,
            Self.POLICY_DELAY,
            Self.Q_MODE,
            Self.USE_TRAIN_CUDA_GRAPH,
        ].make(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
        )

    # ─── Training entry point ──────────────────────────────────────────

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
        """Single-env off-policy training via `run_offpolicy_train`.
        Covers `(env=cpu, train=cpu)` and `(env=cpu, train=gpu)`."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train[
            REDQTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
                Self.N,
                Self.N_MIN,
                Self.UTD,
                Self.POLICY_DELAY,
                Self.Q_MODE,
                Self.USE_TRAIN_CUDA_GRAPH,
            ],
            E,
            L,
        ](
            self.trainer,
            env,
            total_timesteps,
            ctx=ctx,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
        )

    # ─── Evaluation ────────────────────────────────────────────────────

    def eval[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        verbose: Bool = False,
    ) raises -> Scalar[DT]:
        """Greedy eval via `run_offpolicy_eval`."""
        return run_offpolicy_eval[
            REDQTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
                Self.N,
                Self.N_MIN,
                Self.UTD,
                Self.POLICY_DELAY,
                Self.Q_MODE,
                Self.USE_TRAIN_CUDA_GRAPH,
            ],
            E,
        ](
            self.trainer,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
        )

    # ─── Single-step inference ─────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Stochastic action sample. `step_idx` decides warmup vs
        policy path inside the trainer."""
        self.trainer.select_action(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic action — uses the actor's mean (no Gaussian
        sample). Use this at inference time after training."""
        self.trainer.select_greedy_action(obs, action_out)

    # ─── Episode-tracker accessors ─────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.trainer.mean_return()

    def ep_count(self) -> Int:
        return self.trainer.ep_count()

    # ─── Metrics / logging passthroughs ────────────────────────────────

    def flush_metrics[
        L: Logger = NoOpLogger,
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> REDQMetrics:
        """Drain trainer accumulators into a REDQMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        """Per-section wall-time report (resets accumulators)."""
        return self.trainer.flush_timer_log()

    # ─── Checkpointing (delegates to trainer.save_state / load_state) ─

    def save(mut self, path: String) raises:
        """One v3 `storage-ckpt` file: actor + N online critics, with α's
        optimizer and the step counter as `K` scalars. Targets reconstructed
        via hard-copy on load. Network optimizers, replay buffer and episode
        tracker not saved."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. Target critics are hard-copied from their
        online twins after the online params are restored."""
        self.trainer.load_state(path)
