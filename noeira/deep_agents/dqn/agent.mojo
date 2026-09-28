"""DQNAgent — user-facing facade over DQNTrainer + discrete off-policy drivers.

Discrete-action DQN. Returns integer actions (argmax over Q) rather than
the continuous-action float vector that SAC/DDPG/TD3/PPO produce.

Usage:

    var agent = DQNAgent[
        "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
    ](lr=1e-3, gamma=0.99, epsilon=1.0)
    var ep_returns = agent.train(env, total_timesteps=10_000)
    var mean_eval = agent.eval(env, num_episodes=10)

`DOUBLE=True` enables Double-DQN target-Y (van Hasselt et al. 2016).
"""

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.env_traits import BoxDiscreteActionEnv

from ..training.blocks import SampleBlock
from ..training.batched_env import BatchedEnv
from ...io.artifact_sink import ArtifactSink
from ..training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_train_gpu_batched,
    run_offpolicy_discrete_eval,
)

from .metrics import DQNMetrics
from .trainer import DQNTrainer


struct DQNAgent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    DOUBLE: Bool = False,
](Movable & Deinitable):
    """Thin facade over `DQNTrainer` + discrete off-policy drivers."""

    var trainer: DQNTrainer[
        Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = 1e-3,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        epsilon: Scalar[DT] = 1.0,
        epsilon_decay: Scalar[DT] = 0.995,
        epsilon_min: Scalar[DT] = 0.01,
        learning_starts: Int = 1_000,
        target_update_freq: Int = 0,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = 0.0,
        max_grad_norm: Scalar[DT] = 0.0,
        per_alpha: Scalar[DT] = 0.6,
        per_beta: Scalar[DT] = 0.4,
        per_epsilon: Scalar[DT] = 1e-6,
        nstep: Int = 1,
    ) raises:
        """Construct a DQNAgent. Forwards every kwarg to `DQNTrainer.make`.

        `per_*` configure prioritized replay and `nstep` the N-step
        return horizon; both are no-ops when the `SAMPLE` block is
        uniform / single-step (the trainer's `configure_per` /
        `configure_gamma` default to no-op for those backends). They are
        wired through here so the `DQNPER` / N-step configs in
        `config.mojo` can reach them."""
        self.trainer = DQNTrainer[
            Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
        ].make(
            ctx=ctx,
            lr=lr,
            gamma=gamma,
            tau=tau,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            learning_starts=learning_starts,
            target_update_freq=target_update_freq,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_epsilon=per_epsilon,
            nstep=nstep,
        )

    # ─── Training entry point ──────────────────────────────────────────

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
        """Single-env discrete off-policy training via
        `run_offpolicy_discrete_train`. Covers `(env=cpu, train=cpu)` and
        `(env=cpu, train=gpu)` — env is always CPU-side for discrete.

        See `SACAgent.train_single` for `diag_every` / `checkpoint_*`
        semantics; the discrete driver wires them the same way."""
        var ctx = self.trainer.ctx
        return run_offpolicy_discrete_train[
            DQNTrainer[
                Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
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

    def train_gpu_batched[
        E: BatchedEnv,
        N_ENVS: Int,
        NS: Int = 1,
        L: Logger = NoOpLogger,
        USE_TRAIN_CUDA_GRAPH: Bool = True,
    ](
        mut self,
        mut env: E,
        total_env_steps: Int,
        *,
        rng_seed: UInt64 = UInt64(42),
        updates_per_step: Int = 1,
        print_every: Int = 5_000,
        verbose: Bool = True,
        nstep_gamma: Scalar[DT] = Scalar[DT](0.99),
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        base_step: Int = 0,
        diag_every: Int = 0,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = "",
        eval_env: Optional[Pointer[E, MutAnyOrigin]] = None,
        eval_every: Int = 0,
        eval_episodes: Int = 16,
        eval_max_iters: Int = 20_000,
        progress_label: String = "dqn",
        episode_sync_every: Int = 1,
    ) raises -> List[Scalar[DT]]:
        """GPU-batched training via `run_offpolicy_discrete_train_gpu_batched`:
        steps `N_ENVS` device-resident envs (`BatchedGpuDiscreteEnv`) in
        parallel while the Q-net trains on the same device. Requires a
        `"gpu"`-target agent (raises if no DeviceContext). `NS` must match the
        SAMPLE block's N-step (`nstep_gamma` its discount); `eval_env` enables
        periodic noise-off greedy eval on a separate env instance.

        `USE_TRAIN_CUDA_GRAPH` (default True; no-op on non-NVIDIA) captures the
        per-update device kernel sequence into a CUDA graph and replays it,
        removing per-kernel launch overhead from the train step. Pair it with
        `episode_sync_every > 1` to batch the reward/done readback. Capture is
        bit-identical to the non-captured path at every flush boundary. For PER
        (e.g. RainbowDQN), the IS-β anneal freezes at capture time — `set_beta`
        no longer takes effect on the captured graph (benign)."""
        var ctx = self.trainer.ctx
        if not ctx:
            raise Error(
                "DQNAgent.train_gpu_batched: gpu-target agent required"
                " (no DeviceContext)"
            )
        return run_offpolicy_discrete_train_gpu_batched[
            DQNTrainer[
                Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
            ],
            E,
            N_ENVS,
            NS,
            L,
            USE_TRAIN_CUDA_GRAPH,
        ](
            ctx.value(),
            self.trainer,
            env,
            total_env_steps,
            rng_seed=rng_seed,
            updates_per_step=updates_per_step,
            print_every=print_every,
            verbose=verbose,
            nstep_gamma=nstep_gamma,
            logger=logger,
            base_step=base_step,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
            eval_env=eval_env,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_max_iters=eval_max_iters,
            progress_label=progress_label,
            episode_sync_every=episode_sync_every,
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
        """Greedy eval — argmax Q, no epsilon. Returns mean episode return."""
        return run_offpolicy_discrete_eval[
            DQNTrainer[
                Self.train_target, Self.SAMPLE, Self.Q_NET, Self.DOUBLE,
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

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Argmax-Q action (no epsilon). Returns the integer action index."""
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
    ) raises -> DQNMetrics:
        """Drain trainer accumulators into a DQNMetrics bundle."""
        return self.trainer.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        return self.trainer.flush_timer_log()

    # ─── Checkpointing (CPU only) ──────────────────────────────────────

    def save(mut self, path: String) raises:
        """Thin passthrough to `trainer.save_state(path)`. Writes ONE
        file (v3 `storage-ckpt` file) with prefixed sections for q_net
        and q_opt. Replay buffer + episode tracker NOT included.
        CPU-only — GPU trainer raises with a helpful message."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. Target net hard-copied from the online net
        after the online params are restored."""
        self.trainer.load_state(path)
