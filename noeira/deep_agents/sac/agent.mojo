"""SACAgent — user-facing facade over SACTrainer + off-policy drivers.

The trainer owns the algorithm (parameters, optimizers, replay buffer,
training step pipeline); the off-policy drivers own the env loop. This
agent blends the two so users write:

    var agent = SACAgent["cpu", SAMPLE, ACTOR, CRITIC].make(actor_lr=...)
    var ep_returns = agent.train(env, total_timesteps=30_000)
    var mean_eval = agent.eval(env, num_episodes=10)

instead of wiring `SACTrainer.make(...)` + free-function
`run_offpolicy_train_batched[type, type, N_ENVS](trainer, env, ...)`
by hand.

The wrapped `SACTrainer` remains accessible as `agent.trainer` for
power users who want to compose custom training loops (e.g. multi-seed
sweeps, save/resume integration, manual gradient inspection).

Two training entry points:

  * `train(env, ...)`     — `BatchedEnv` (CPU+CPU or GPU+GPU at any
                            N_ENVS ≥ 1). Routes to
                            `run_offpolicy_train_batched`.
  * `train_single(env, ...)` — `BoxContinuousActionEnv` (single-env,
                               typically used for cross-target CPU env
                               + GPU trainer). Routes to
                               `run_offpolicy_train`.

Eval is a single method (`eval`) bound on `BoxContinuousActionEnv`,
matching the existing `run_offpolicy_eval`.

Bit-identity: this agent is a pure facade — it forwards every call
unchanged. `pendulum_sac_nn_driver.mojo` continues to produce
`mean_ret(10) = -169.04118` at 30k steps seed=42.
"""

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.env_traits import BoxContinuousActionEnv, RenderableEnv

from ..training.blocks import SampleBlock
from ..training.batched_env import BatchedEnv
from ...io.artifact_sink import ArtifactSink
from ..training.driver_offpolicy import (
    run_offpolicy_train,
    run_offpolicy_train_batched,
    run_offpolicy_eval,
    run_offpolicy_eval_render,
)

from .metrics import SACMetrics
from .trainer import SACTrainer


struct SACAgent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](Movable & Deinitable):
    """Thin facade over `SACTrainer` + off-policy drivers.

    Comptime parameters mirror `SACTrainer` one-for-one. Dimensions
    (OBS / ACT / BATCH) are derived from `SAMPLE` so they appear in
    one place at instantiation.
    """

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    var trainer: SACTrainer[
        Self.train_target,
        Self.SAMPLE,
        Self.ACTOR,
        Self.CRITIC,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 1e-3,
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
        per_alpha: Scalar[DT] = 0.6,
        per_beta: Scalar[DT] = 0.4,
        per_epsilon: Scalar[DT] = 1e-6,
        use_bf16: Bool = False,
        use_ere: Bool = False,
        ere_eta: Scalar[DT] = 0.996,
        ere_c_min: Int = 1,
        ere_k_max: Int = 1000,
    ) raises:
        """Construct a SACAgent. Forwards every kwarg to `SACTrainer.make`."""

        self.trainer = SACTrainer[
            Self.train_target,
            Self.SAMPLE,
            Self.ACTOR,
            Self.CRITIC,
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
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_epsilon=per_epsilon,
            use_bf16=use_bf16,
            use_ere=use_ere,
            ere_eta=ere_eta,
            ere_c_min=ere_c_min,
            ere_k_max=ere_k_max,
        )

    # ─── Training entry points ─────────────────────────────────────────

    def train[
        E: BatchedEnv,
        N_ENVS: Int = 1,
        NS: Int = 1,
        L: Logger = NoOpLogger,
        USE_TRAIN_CUDA_GRAPH: Bool = True,
        USE_ENV_CUDA_GRAPH: Bool = True,
        EE: BatchedEnv = E,
        EVAL_ENVS: Int = N_ENVS,
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
        base_step: Int = 0,
        eval_env: Optional[Pointer[EE, MutAnyOrigin]] = None,
        eval_every: Int = 0,
        eval_episodes: Int = 16,
        eval_max_steps: Int = 1_000,
    ) raises -> List[Scalar[DT]]:
        """Off-policy training via `run_offpolicy_train_batched`.

        Covers same-target (cpu+cpu, gpu+gpu) at any `N_ENVS >= 1`. Pass
        `N_ENVS` as a comptime method param when wrapping multiple envs,
        e.g. `agent.train[N_ENVS=8](env, total_timesteps=200_000)`.

        Pass `logger=Optional[Pointer[L, MutAnyOrigin]](
        Pointer(to=my_logger))` to stream `env/mean_ret` and
        `env/ep_count` at `print_every` cadence. Default `L=NoOpLogger`
        comptime-elides the emit path entirely (bit-identical no-op).

        Set `diag_every > 0` to also drain the full SAC metric bundle
        (`actor_loss` / `critic_loss` / `alpha` / `mean_q` / `mean_reward`
        / `train_steps` / …) through the logger every `diag_every`
        env-steps — the GPU multi-env counterpart of `train_single`'s
        diag cadence. Default 0 keeps only the `avg_reward` / `episodes`
        stream.

        Set `USE_TRAIN_CUDA_GRAPH=True` (GPU + uniform replay only; no-op on
        non-NVIDIA) to capture the per-update device kernel sequence into a
        CUDA graph and replay it — removing per-kernel launch overhead from
        the train step. Pair it with `episode_sync_every > 1` to also batch
        the per-iteration reward/done readback, so the host stops serializing
        the GPU pipeline every iteration (otherwise that sync negates the
        capture win). Returns stay exact at every print/diag boundary.

        Set `checkpoint_every > 0` + `checkpoint_path` to auto-save the
        trainer's one-file v3 binary checkpoint (actor + the two online
        critics; optimizer moments and α are NOT saved — a resume re-warms
        them) every `checkpoint_every` env-steps and
        one final time at the end — the batched GPU counterpart of
        `train_single`'s checkpoint cadence. The save runs in host code
        between iterations (D2H of live params on the GPU target) so it is
        CUDA-graph-capture safe. The replay buffer / episode tracker are NOT
        persisted, so resume starts with a fresh replay.

        ⚠ `checkpoint_path` is OVERWRITTEN on every save, so a single `train`
        call cannot produce a LADDER of policies (random → expert). To build
        one, call `train` in segments on the SAME agent — the nets, replay
        buffer and optimizer state all persist across calls — saving to a
        step-stamped path after each, and pass `base_step` so the logger's
        x-axis stays continuous instead of restarting per segment. That is
        what `examples/dm_control/sac_dm_walker_training_gpu.mojo` does; the
        ladder is what the FB dataset is collected from.

        Set `eval_every > 0` AND pass an ISOLATED `eval_env` (a second
        `BatchedGpuEnv[..., EVAL_ENVS, ...]` — NOT the training env) to run a
        periodic GPU-parallel DETERMINISTIC (greedy, no exploration noise)
        eval every `eval_every` env-steps and log the true policy quality as
        `eval/mean_return`. This is the deployable-policy signal; the always-on
        `avg_reward` is a stochastic rollout that under-reports SAC by the
        entropy term. `EVAL_ENVS` (comptime, default N_ENVS) must equal the
        eval env's struct batch size; `eval_episodes <= EVAL_ENVS` completes in
        one `eval_max_steps` window. Eval touches no replay/optimizer state."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train_batched[
            SACTrainer[
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
            EE,
            EVAL_ENVS,
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
            base_step=base_step,
            eval_env=eval_env,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_max_steps=eval_max_steps,
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
        Covers `(env=cpu, train=cpu)` and `(env=cpu, train=gpu)` cross-
        target. For batched same-target, use `train()` instead.

        Auto-flush + auto-save (matches the legacy `DeepSACAgent.train`):
          * `diag_every` (env steps, default 0 = off): when > 0, the
            driver calls `trainer.flush_metrics_through_logger(logger,
            step)` inline every `diag_every` env-steps — emits the full
            `SACMetrics` bundle (`actor_loss`, `critic_loss`, `alpha`,
            `mean_q`, `mean_target`, `mean_reward`, `mean_done`,
            `mean_abs_action`, `train_steps`, `n_updates`).
          * `checkpoint_every` (env steps, default 0 = off) +
            `checkpoint_path`: when both set, the driver calls
            `trainer.save_state(checkpoint_path)` inline every
            `checkpoint_every` env-steps and one final time at the end.

        The env-loop is NEVER chunked — same single pass as the bespoke
        `run_offpolicy_train` invocation, just with extra `if`s at the
        cadence boundaries. Same bit-identity profile as before."""
        var ctx = self.trainer.ctx
        return run_offpolicy_train[
            SACTrainer[
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
        """Greedy eval via `run_offpolicy_eval`. Returns mean episode
        return across `num_episodes`. Does not touch the replay buffer,
        optimizers, or episode tracker."""
        return run_offpolicy_eval[
            SACTrainer[
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

    def eval_render[
        E: BoxContinuousActionEnv & RenderableEnv,
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        *,
        max_steps_per_episode: Int = 1_000,
        frame_delay_ms: Int = 16,
        verbose: Bool = True,
    ) raises -> Scalar[DT]:
        """Greedy eval with live env-owned rendering via
        `run_offpolicy_eval_render`. Same non-mutating greedy loop as
        `eval` plus the `RenderableEnv` init/per-frame-render/quit/close
        handling — replaces the hand-rolled render loop in the eval
        example scripts. Falls back to headless if no renderer is
        available. Returns the mean episode return."""
        return run_offpolicy_eval_render[
            SACTrainer[
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
            frame_delay_ms=frame_delay_ms,
            verbose=verbose,
        )

    # ─── Single-step inference (host-list interface) ───────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Stochastic action sample. `step_idx` decides warmup vs policy
        path inside the trainer."""
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
        """Mean episodic return over the last `window_size` episodes."""
        return self.trainer.mean_return()

    def ep_count(self) -> Int:
        """Total completed episodes since training began."""
        return self.trainer.ep_count()

    # ─── Metrics / logging passthrough ─────────────────────────────────

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain trainer accumulators into a SACMetrics bundle. Pass an
        Pointer-to-logger to also emit one `log_scalar` call per
        metric field. Resets accumulators on every call."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        """Per-section wall-time report (resets accumulators)."""
        return self.trainer.flush_timer_log()

    # ─── Checkpointing (CPU only — delegates to trainer.save_state) ───

    def save(mut self, path: String) raises:
        """Thin passthrough to `trainer.save_state(path)`. Writes ONE
        file (v3 `storage-ckpt` file) with prefixed sections for
        actor, critic1, critic2 (the online nets). Optimizer moments, α,
        replay buffer and episode tracker are NOT included — resume
        re-warms. CPU-only."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. Target critics are hard-copied from their
        online twins after the online params are restored."""
        self.trainer.load_state(path)
