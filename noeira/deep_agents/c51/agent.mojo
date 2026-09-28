"""C51Agent — user-facing facade over C51Trainer + discrete off-policy drivers.

Distributional DQN (Bellemare et al. 2017). Returns integer actions
(argmax over expected-Q, i.e. argmax_a Σ_k softmax(logits[a])_k · z_k)
rather than the continuous-action float vector that SAC/DDPG/TD3/PPO
produce.

Usage:

    var agent = C51Agent[
        "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], C51QNet,
        N_ATOMS=51, NUM_ACTIONS=2,
    ](lr=1e-4, gamma=0.99, v_min=-10.0, v_max=10.0)
    var ep_returns = agent.train(env, total_timesteps=30_000)
    var mean_eval = agent.eval(env, num_episodes=10)

`Q_NET.OUT_DIM` must equal `NUM_ACTIONS · N_ATOMS` (per-atom logits).
`DOUBLE=True` enables Double-DQN target-Y (van Hasselt et al. 2016).
Rainbow is C51 with `DOUBLE=True` + a PER + N-step SAMPLE block +
a dueling/noisy `Q_NET`; configure those via the SAMPLE block and net.
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
    run_offpolicy_discrete_train_cpu_env_gpu_agent,
    run_offpolicy_discrete_eval,
)

from .metrics import C51Metrics
from .trainer import C51Trainer


struct C51Agent[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    N_ATOMS: Int = 51,
    NUM_ACTIONS: Int = 2,
    DOUBLE: Bool = False,
](Movable & Deinitable):
    """Thin facade over `C51Trainer` + discrete off-policy drivers."""

    var trainer: C51Trainer[
        Self.train_target,
        Self.SAMPLE,
        Self.Q_NET,
        Self.N_ATOMS,
        Self.NUM_ACTIONS,
        Self.DOUBLE,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = 1e-4,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        epsilon: Scalar[DT] = 1.0,
        epsilon_decay: Scalar[DT] = 0.995,
        epsilon_min: Scalar[DT] = 0.05,
        learning_starts: Int = 1_000,
        target_update_freq: Int = 500,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = 0.0,
        max_grad_norm: Scalar[DT] = 0.0,
        per_alpha: Scalar[DT] = 0.6,
        per_beta: Scalar[DT] = 0.4,
        per_epsilon: Scalar[DT] = 1e-6,
        nstep: Int = 1,
        v_min: Scalar[DT] = -10.0,
        v_max: Scalar[DT] = 10.0,
    ) raises:
        """Construct a C51Agent. Forwards every kwarg to `C51Trainer.make`."""
        self.trainer = C51Trainer[
            Self.train_target,
            Self.SAMPLE,
            Self.Q_NET,
            Self.N_ATOMS,
            Self.NUM_ACTIONS,
            Self.DOUBLE,
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
            v_min=v_min,
            v_max=v_max,
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
        base_step: Int = 0,
        eval_env: Optional[Pointer[E, MutAnyOrigin]] = None,
        eval_every: Int = 0,
        eval_episodes: Int = 10,
        eval_max_steps: Int = 20_000,
        progress_label: String = "c51",
    ) raises -> List[Scalar[DT]]:
        """Single-env discrete off-policy training via
        `run_offpolicy_discrete_train`. Covers `(env=cpu, train=cpu)` and
        `(env=cpu, train=gpu)` — env is always CPU-side for discrete.

        See `DQNAgent.train` for `diag_every` / `checkpoint_*`
        semantics; the discrete driver wires them the same way.
        `eval_env` enables periodic noise-off greedy eval on a separate
        env instance. All kwargs forward to the driver unchanged."""
        var ctx = self.trainer.ctx
        return run_offpolicy_discrete_train[
            C51Trainer[
                Self.train_target,
                Self.SAMPLE,
                Self.Q_NET,
                Self.N_ATOMS,
                Self.NUM_ACTIONS,
                Self.DOUBLE,
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
            base_step=base_step,
            eval_env=eval_env,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_max_steps=eval_max_steps,
            progress_label=progress_label,
        )

    def train_gpu_batched[
        E: BatchedEnv,
        N_ENVS: Int,
        NS: Int = 1,
        L: Logger = NoOpLogger,
        USE_TRAIN_CUDA_GRAPH: Bool = False,
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
        progress_label: String = "c51",
        episode_sync_every: Int = 1,
    ) raises -> List[Scalar[DT]]:
        """GPU-batched training via `run_offpolicy_discrete_train_gpu_batched`:
        steps `N_ENVS` device-resident envs (`BatchedGpuDiscreteEnv`) in
        parallel while the Q-net trains on the same device. Requires a
        `"gpu"`-target agent (raises if no DeviceContext). `NS` must match
        the SAMPLE block's N-step (`nstep_gamma` its discount); `eval_env`
        enables periodic noise-off greedy eval on a separate env instance.
        All kwargs forward to the driver unchanged.

        `USE_TRAIN_CUDA_GRAPH` (default False; no-op on non-NVIDIA) captures the
        per-update device kernel sequence into a CUDA graph and replays it,
        removing per-kernel launch overhead from the train step. Pair it with
        `episode_sync_every > 1` to batch the reward/done readback so the host
        doesn't serialize the GPU pipeline every iteration. Capture is
        bit-identical to the non-captured path at every flush boundary. For
        Rainbow (PER), the IS-β anneal freezes at capture time — `set_beta`
        no longer takes effect on the captured graph (benign; β stays at its
        post-warmup value)."""
        var ctx = self.trainer.ctx
        if not ctx:
            raise Error(
                "C51Agent.train_gpu_batched: gpu-target agent required"
                " (no DeviceContext)"
            )
        return run_offpolicy_discrete_train_gpu_batched[
            C51Trainer[
                Self.train_target,
                Self.SAMPLE,
                Self.Q_NET,
                Self.N_ATOMS,
                Self.NUM_ACTIONS,
                Self.DOUBLE,
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

    def train_cpu_batched[
        E: BatchedEnv,
        N_ENVS: Int,
        NS: Int = 1,
        L: Logger = NoOpLogger,
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
        progress_label: String = "c51",
    ) raises -> List[Scalar[DT]]:
        """CPU-env / GPU-agent hybrid training via
        `run_offpolicy_discrete_train_cpu_env_gpu_agent`: steps `N_ENVS`
        host-side envs per iteration (e.g. Atari emulators in a
        `BatchedCpuDiscreteEnv`, multi-core) while the Q-net selects
        actions and trains on the GPU. Requires a `"gpu"`-target agent.
        `NS` must match the SAMPLE block's N-step (`nstep_gamma` its
        discount); replay ratio = updates_per_step / N_ENVS. All kwargs
        forward to the driver unchanged."""
        var ctx = self.trainer.ctx
        if not ctx:
            raise Error(
                "C51Agent.train_cpu_batched: gpu-target agent required"
                " (no DeviceContext)"
            )
        return run_offpolicy_discrete_train_cpu_env_gpu_agent[
            C51Trainer[
                Self.train_target,
                Self.SAMPLE,
                Self.Q_NET,
                Self.N_ATOMS,
                Self.NUM_ACTIONS,
                Self.DOUBLE,
            ],
            E,
            N_ENVS,
            NS,
            L,
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
        """Greedy eval — expected-Q argmax, no epsilon. Returns mean
        episode return."""
        return run_offpolicy_discrete_eval[
            C51Trainer[
                Self.train_target,
                Self.SAMPLE,
                Self.Q_NET,
                Self.N_ATOMS,
                Self.NUM_ACTIONS,
                Self.DOUBLE,
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
        """Expected-Q argmax action (no epsilon). Returns the integer
        action index."""
        return self.trainer.select_greedy_action(obs)

    # ─── PER beta annealing (Rainbow) ──────────────────────────────────

    def set_beta(mut self, beta: Scalar[DT]):
        """Update the prioritized-replay importance-sampling exponent.
        No-op when the SAMPLE block is non-prioritized."""
        self.trainer.set_beta(beta)

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
    ) raises -> C51Metrics:
        """Drain trainer accumulators into a C51Metrics bundle."""
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
