"""DDPGAgent — user-facing facade over DDPGTrainer + off-policy drivers.

Mirrors `SACAgent` (see sac/agent.mojo) but specialised for DDPG:

  * Single critic (no twin-critic min); deterministic actor with Gaussian
    exploration noise; no entropy temperature.
  * CPU + GPU — `train_target` is the first comptime param (Phase 4.1).

Usage:

    var agent = DDPGAgent["cpu", SAMPLE, ACTOR, CRITIC](
        actor_lr=1e-4, critic_lr=1e-3, noise_scale=0.1,
    )
    var ep_returns = agent.train(env, total_timesteps=30_000)
    var mean_eval = agent.eval(env, num_episodes=10)

For GPU: `DDPGAgent["gpu", GPU_SAMPLE, ACTOR, CRITIC](ctx=ctx, ...)`.

Dimensions (OBS / ACT / BATCH) are derived from `SAMPLE` so they appear
in one place at instantiation. `agent.trainer` remains exposed for power
users.
"""

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.env_traits import BoxContinuousActionEnv

from ..training.blocks import SampleBlock
from ..training.batched_env import BatchedEnv
from ...io.artifact_sink import ArtifactSink
from ..training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
    run_offpolicy_eval,
)

from .metrics import DDPGMetrics
from .trainer import DDPGTrainer


struct DDPGAgent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](Movable & Deinitable):
    """Thin facade over `DDPGTrainer` + off-policy drivers. Comptime
    parameters mirror `DDPGTrainer` one-for-one; dimensions derive from
    `SAMPLE`."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    var trainer: DDPGTrainer[
        Self.train_target,
        Self.SAMPLE,
        Self.ACTOR,
        Self.CRITIC,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 1e-4,
        critic_lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        noise_scale: Scalar[DT] = 0.1,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
        max_grad_norm: Scalar[DT] = 0.0,
        use_bf16: Bool = False,
    ) raises:
        """Construct a DDPGAgent. Forwards every kwarg to `DDPGTrainer.make`.
        `ctx` is required for `train_target='gpu'`; `use_bf16` (GPU) enables
        mixed-precision training."""
        self.trainer = DDPGTrainer[
            Self.train_target,
            Self.SAMPLE,
            Self.ACTOR,
            Self.CRITIC,
        ].make(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            noise_scale=noise_scale,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            use_bf16=use_bf16,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BatchedEnv,
        N_ENVS: Int = 1,
        NS: Int = 1,
        L: Logger = NoOpLogger,
        # DDPGTrainer does not implement train_device_kernels — the capture
        # path would raise at the first train step, so default OFF (the env
        # capture below is independent and safe).
        USE_TRAIN_CUDA_GRAPH: Bool = False,
        USE_ENV_CUDA_GRAPH: Bool = True,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        rng_seed: UInt64 = 42,
        updates_per_step: Int = 1,
        print_every: Int = 5_000,
        verbose: Bool = True,
        nstep_gamma: Scalar[DT] = 0.99,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        episode_sync_every: Int = 1,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = "",
        checkpoint_every: Int = 0,
    ) raises -> List[Scalar[DT]]:
        """Off-policy training via `run_offpolicy_train_batched`. Covers
        same-target (cpu+cpu, gpu+gpu) at any `N_ENVS >= 1`. See
        `SACAgent.train` for `diag_every` / `episode_sync_every` /
        `checkpoint_*` and the CUDA-graph capture flags (GPU + uniform
        replay only; off by default, no-op on non-NVIDIA)."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train_batched[
            DDPGTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
            ],
            E,
            N_ENVS,
            NS,
            L,
            # CUDA-graph capture is a GPU-only path (the driver asserts
            # gpu); gate the flags by target so the True defaults mean
            # "capture when on GPU, no-op on CPU" instead of failing to
            # compile on a cpu agent.
            USE_TRAIN_CUDA_GRAPH and Self.train_target == "gpu",
            USE_ENV_CUDA_GRAPH and Self.train_target == "gpu",
        ](
            ctx,
            self.trainer,
            env,
            total_timesteps,
            rng_seed=rng_seed,
            updates_per_step=updates_per_step,
            print_every=print_every,
            verbose=verbose,
            nstep_gamma=nstep_gamma,
            logger=logger,
            diag_every=diag_every,
            episode_sync_every=episode_sync_every,
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
        """Single-env off-policy training via `run_offpolicy_train`.
        Covers `(env=cpu, train=cpu)` and `(env=cpu, train=gpu)`."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train[
            DDPGTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
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

    # ─── Evaluation ─────────────────────────────────────────────────────

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
        """Greedy eval — actor mean action, no exploration noise."""
        return run_offpolicy_eval[
            DDPGTrainer[
                Self.train_target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
            ],
            E,
        ](
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
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.trainer.select_action(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
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
    ) raises -> DDPGMetrics:
        """Drain trainer accumulators into a DDPGMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        return self.trainer.flush_timer_log()

    # ─── Checkpointing ──────────────────────────────────────────────────

    def save(mut self, path: String) raises:
        """Thin passthrough to `trainer.save_state(path)`. Writes ONE
        file (v3 `storage-ckpt` file) with prefixed sections for actor,
        critic, actor_opt, critic_opt. Replay buffer + episode tracker
        NOT included."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. Target nets hard-copied from their online
        twins after the online params are restored."""
        self.trainer.load_state(path)
