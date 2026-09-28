"""Off-policy discrete-action training + eval drivers.

Parallel to `driver_offpolicy.mojo` (continuous actions), this module
provides training drivers for discrete-action agents (DQN family:
DQN, Double DQN, Dueling DQN, DQN+PER, Noisy DQN, C51, Rainbow).

Key differences from the continuous off-policy driver:
  - Env bound: `BoxDiscreteActionEnv` (`step_obs` takes Int action)
  - Action is a single Int index per env, not ACT_DIM floats
  - No actor network / log_prob — no ao_scratch / alp_scratch
  - Epsilon-greedy exploration managed by the agent internally

  env_target | train_target | N_ENVS | driver
  -----------|--------------|--------|----------------------------------
  cpu        | cpu          | 1      | run_offpolicy_discrete_train
  cpu        | gpu          | 1      | run_offpolicy_discrete_train

Trait surface
  - `OffPolicyDiscreteAgent` — minimal: select_action_batched[N_ENVS],
    record, train_step, episode tracker accessors, batched CPU record
    + add_complete_return.

Storage: action indices stored as Scalar[DT] in driver scratches and
replay buffers (cast to/from Int at the env boundary). This keeps the
pointer-based API consistent with the continuous driver and avoids a
separate Int-buffer path.

Batched driver (Tier-3) deferred until a consumer needs it.
"""

from std.time import perf_counter_ns
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.utils.progress import IntervalProgress
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.cuda import CUDAGraph, maybe_capture_replay
from .driver_scratch import DriverScratch
from .batched_env import BatchedEnv
from .blocks.episode_readback import (
    EpisodeReturnRing,
    accumulate_episode_returns,
)
from .blocks.cadence import DriverCadence
from ..data.n_step_replay import GPUNStepBuffer
from .checkpoint import announce_checkpoint
from ...io.artifact_sink import ArtifactSink


# ──────────────────────────────────────────────────────────────────────
# OffPolicyDiscreteAgent — trait for the discrete off-policy drivers.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyDiscreteAgent(Deinitable, Movable):
    """Single-trait surface for the discrete off-policy drivers.

    Mirrors `OffPolicyAgent` (continuous) but adapted for discrete
    action spaces:

      - `select_action_batched` writes action INDICES (as Scalar[DT])
        into `action_ptr`, not continuous action vectors.
      - `record` takes `action_idx: Int`, not `action: List[Scalar[DT]]`.
      - No ao_scratch / alp_scratch parameters (no actor network).

    DQN / Double DQN / Dueling / PER / Noisy / C51 / Rainbow all
    conform. Epsilon-greedy is agent-internal — the driver does not
    manage exploration."""

    comptime AGENT_TRAIN_TARGET: StaticString
    comptime AGENT_OBS_DIM: Int
    comptime AGENT_NUM_ACTIONS: Int

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Write N_ENVS discrete action indices into action_ptr.

        obs_ptr: N_ENVS * AGENT_OBS_DIM scalars on train_target.
        action_ptr: N_ENVS scalars on train_target (one index per env,
                    stored as Scalar[DT]).

        The agent handles epsilon-greedy / noisy-net exploration
        internally based on step_idx."""
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic, exploration-free action selection for eval.

        Host-list signature; trainers dispatch internally on
        AGENT_TRAIN_TARGET (CPU trainers run native, GPU trainers
        H2D the obs and D2H the Q-values under the hood)."""
        ...

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        """Store one transition. Action is an Int index; the trainer
        converts to Scalar[DT] internally for replay storage."""
        ...

    def end_episode(mut self):
        ...

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...

    def total_train_steps(self) -> Int:
        """Cumulative gradient-update count for the inter-log progress bar's
        `Train:` field. Default 0 for trainers that don't track it; DQN-family
        trainers may override."""
        return 0

    # ─── Batched support (for future Tier-3 driver) ─────────────────

    def add_complete_return(mut self, ret: Scalar[DT]):
        """Push an externally-tracked complete-episode return into the
        trainer's episode tracker. For future N_ENVS batched drivers."""
        ...

    def record_batch_cpu[
        N_ENVS: Int
    ](
        mut self,
        prev_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push N transitions from host pointer slabs. action_ptr
        stores N_ENVS action indices as Scalar[DT]. Does NOT update
        the episode tracker."""
        ...

    # ─── Optional cadence hooks (default no-op) ──────────────────────
    #
    # Mirror the continuous `OffPolicyAgent` surface so the discrete
    # driver can call into the trainer at `diag_every` / `checkpoint_every`
    # cadences. `flush_metrics_through_logger` has a `pass` default and
    # `save_state` a raising one; DQNTrainer overrides both with real bodies
    # that drain its `DQNMetrics` bundle and write a one-file v3 checkpoint.

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        pass

    def save_state(mut self, path: String) raises:
        """⚠ RAISES BY DEFAULT, on purpose. Drivers call this only when a
        `checkpoint_path` was given; a `pass` default meant a trainer that did
        not override it produced no file and no error, and the run looked
        checkpointed until someone tried to resume it."""
        raise Error(
            "save_state: this trainer does not implement checkpointing, and a"
            " checkpoint was requested at " + path
        )

    def set_noise_scale(mut self, scale: Scalar[DT]) raises:
        """Toggle Noisy-net exploration magnitude on the acting net:
        1.0 = train/explore, 0.0 = deterministic mean weights. Used to
        bracket a greedy-eval rollout (CPU single-env or GPU batched).
        Default no-op — agents without Noisy layers ignore it."""
        pass


# ──────────────────────────────────────────────────────────────────────
# OffPolicyDiscreteAgentGpu — discrete sibling of `OffPolicyAgentGpu`.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyDiscreteAgentGpu(OffPolicyDiscreteAgent):
    """Extends `OffPolicyDiscreteAgent` with the GPU-batched device-record
    surfaces the GPU-batched-env discrete driver needs.

    Mirrors the continuous `OffPolicyAgentGpu`: the env owns the
    obs/action/reward/done device buffers, and these methods push a whole
    `N_ENVS` minibatch of device transitions into the trainer's replay in
    one call (no per-env host round trip). The discrete action is a single
    index per env, so the n-step buffer's ACT dimension is 1.

    Only GPU-target discrete trainers whose SAMPLE block supports device
    batched adds (GPU uniform / PER) conform; the driver comptime-requires
    `env_target == train_target == "gpu"`, so these are never invoked on a
    CPU trainer."""

    # Discrete action arity at the replay / n-step boundary — always 1 (a
    # single action index). Declared (rather than hard-coded to `1` in the
    # `record_batch_gpu_nstep` signature) so it aliases the conformer's
    # `SAMPLE.ACT`; the inner `store_via_block_gpu` needs that exact symbol
    # for compile-time type unification (a literal `1` won't unify).
    comptime AGENT_ACT_DIM: Int

    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """1-step batched device record. `action_dev` holds N_ENVS action
        indices as Scalar[DT]; `done_dev` should carry `terminated` (the TD
        bootstrap is dropped on natural termination, kept on truncation)."""
        ...

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """N-step batched device record. Caller owns the
        `GPUNStepBuffer[NS, OBS, 1, N_ENVS]` (N_ENVS is method-comptime).
        Keep `NS` aligned with the trainer's target-Y γ^N bootstrap."""
        ...

    # ─── Greedy-eval surface (batched, noise-off deterministic rollout) ──

    def select_greedy_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Pure greedy action selection for N_ENVS envs — argmax (expected-)Q,
        no epsilon, no warmup gate. Writes N_ENVS action indices (as
        Scalar[DT]) into `action_ptr`. Used by the batched greedy eval."""
        ...

    # ─── CUDA-graph capture surface ───────────────────────────────────
    #
    # Default bodies so agents that don't support train-step capture still
    # conform. Only invoked on the `USE_TRAIN_CUDA_GRAPH=True` driver path; the
    # defaults are never reached for capture-capable agents (DQN / C51 /
    # Rainbow), which override all three.

    def train_device_kernels(mut self) raises:
        """Pure device-kernel train step (no host work) — the body captured
        into the CUDA graph. Default raises: capture unsupported."""
        raise Error(
            "train_device_kernels: CUDA-graph capture not supported by"
            " this agent (USE_TRAIN_CUDA_GRAPH must stay False)"
        )

    def train_device_kernels_on(mut self, gctx: DeviceContext) raises:
        """`train_device_kernels`, but enqueueing on `gctx` — the body RECORDED
        into a `DeviceGraph` by `noeira.cuda.maybe_record_replay`.

        ⚠ THE ARGUMENT IS NOT DECORATION. Recording happens because operations
        are enqueued through the *recording* context, so an override that
        reaches for its own `self.ctx` records NOTHING and runs eagerly —
        silently, since the work is still correct. That includes every helper
        the body calls that CACHES a context of its own (`DeviceMeanAccum`,
        `TrainState`, optimizer scratch): each must be pointed at `gctx` first,
        or its kernels run during the build pass and are skipped on every
        replay. `DeviceMeanAccum.set_ctx` documents that trap in full.

        Default raises, deliberately: an agent that has not been migrated must
        fail LOUDLY when the DeviceGraph backend is selected, never silently
        record a partial step."""
        raise Error(
            "train_device_kernels_on: DeviceGraph recording not supported by"
            " this agent (use NOEIRA_GRAPH_BACKEND=stream, or migrate it)"
        )

    def note_train_update(mut self):
        """Advance one logical update's host bookkeeping (counters / metric
        accumulators). Default no-op."""
        pass

    def learning_starts_count(self) -> Int:
        """Cumulative env-step threshold after which training begins — the
        driver gates the capture path on this. Default 0."""
        return 0


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_train — single-env, env_target="cpu".
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_train[
    A: OffPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
    L: Logger = NoOpLogger,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    ctx: Optional[DeviceContext] = None,
    print_every: Int = 1_000,
    verbose: Bool = True,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    artifacts: Optional[ArtifactSink] = None,
    run_dir: String = "",
    base_step: Int = 0,
    eval_env: Optional[Pointer[E, MutAnyOrigin]] = None,
    eval_every: Int = 0,
    eval_episodes: Int = 10,
    eval_max_steps: Int = 20_000,
    progress_label: String = "dqn",
) raises -> List[Scalar[DT]]:
    """Single-env discrete off-policy training driver.

    Covers (env_target=cpu, train_target=cpu) and
    (env_target=cpu, train_target=gpu). The env is always CPU-side
    (`BoxDiscreteActionEnv`); when train_target="gpu" the driver
    stages obs H2D and action D2H around the agent's
    `select_action_batched` call.

    Loop: one env step + one `train_step` per iteration.

    Args:
        trainer: Any nn discrete off-policy trainer (DQN family).
        env: Any `BoxDiscreteActionEnv`.
        total_timesteps: Number of env steps to run.
        ctx: Required for train_target="gpu"; ignored for "cpu".
        print_every: Verbose status-line cadence (env steps). 0 disables.
        verbose: Print a per-cadence status line.
        logger: Optional logger instance.
        diag_every: Diagnostic logging cadence (env-steps). 0 disables.
        checkpoint_every: Checkpoint writing cadence (env-steps). 0 disables.
        checkpoint_path: Path to write checkpoints to.
        artifacts: Sink each written checkpoint is offered to (None: not offered).
        run_dir: Run directory the offered checkpoint path is made relative to.
        base_step: Base step counter for the training loop.
        eval_env: Optional separate env used for greedy evaluation.
        eval_every: Greedy-eval cadence (env steps; 0 disables).
        eval_episodes: Episodes per greedy-eval pass.
        eval_max_steps: Max env iterations per eval episode (safety cap).
        progress_label: Label for the progress bar.

    Returns:
        List of `trainer.mean_return()` snapshots at each completed
        episode boundary.
    """
    comptime env_target: StaticString = "cpu"
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM

    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "run_offpolicy_discrete_train: train_target must be 'cpu' or 'gpu'"
    comptime if train_target == "gpu":
        if not ctx:
            raise Error(
                "run_offpolicy_discrete_train[train_target='gpu']:"
                " ctx required for env→trainer H2D/D2H staging"
            )

    comptime needs_boundary_copy: Bool = env_target != train_target
    var obs_scratch = DriverScratch["obs", 1, OBS].make[train_target](
        ctx=ctx,
        with_host_mirror=needs_boundary_copy,
    )
    var action_scratch = DriverScratch["action", 1, 1].make[train_target](
        ctx=ctx,
        with_host_mirror=needs_boundary_copy,
    )

    var obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var env_obs = env.reset_obs_list()

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    # In-place progress bar between log lines (pure CPU, no GPU sync).
    var prog = IntervalProgress(
        print_every, label=progress_label, enabled=verbose
    )
    while step < total_timesteps:
        var obs_scratch_h = obs_scratch.host_ptr()
        for d in range(OBS):
            var v = Scalar[DT](env_obs[d])
            obs_list[d] = v
            obs_scratch_h[unsafe_offset=d] = v

        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(obs_scratch.dev.value(), obs_scratch_h)

        # `base_step + step` — cumulative env-step counter for the
        # trainer's warmup gating. Equivalent to `step` when base_step=0.
        trainer.select_action_batched[1](
            obs_scratch.target_ptr[train_target](),
            action_scratch.target_ptr[train_target](),
            base_step + step,
        )

        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(
                action_scratch.host_ptr(), action_scratch.dev.value()
            )
            c.synchronize()

        var action_idx = Int(action_scratch.host_ptr()[unsafe_offset=0])

        var step_res = env.step_obs(action_idx)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        # `done` (terminated OR truncated) drives reset/episode tracking; the
        # replay buffer stores `terminated` ONLY so the DQN/C51 TD bootstrap
        # `(1 − done)·γ·max Q'` is kept on time-limit truncation and dropped
        # on natural termination.
        var terminated = env.was_terminated()
        for d in range(OBS):
            next_obs_list[d] = Scalar[DT](nxt[d])

        trainer.record(
            obs_list,
            action_idx,
            Scalar[DT](reward),
            next_obs_list,
            Scalar[DT](1.0) if terminated else Scalar[DT](0.0),
        )

        if done:
            trainer.end_episode()
            env_obs = env.reset_obs_list()
            var new_ep_count = trainer.ep_count()
            if new_ep_count > current_ep_count:
                ep_returns.append(trainer.mean_return())
                current_ep_count = new_ep_count
        else:
            env_obs = nxt^

        step += 1
        _ = trainer.train_step(base_step + step)

        var abs_step = base_step + step

        prog.tick(abs_step, trainer.total_train_steps())

        if verbose and print_every > 0 and abs_step % print_every == 0:
            prog.clear()
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                abs_step,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )

        # Logger emit at the same cadence. Comptime-elided when
        # L=NoOpLogger (default).
        comptime if L.ENABLED:
            if print_every > 0 and abs_step % print_every == 0 and Bool(logger):
                logger.value()[].log_scalar(
                    "avg_reward",
                    Float64(trainer.mean_return()),
                    abs_step,
                )
                logger.value()[].log_scalar(
                    "episodes",
                    Float64(trainer.ep_count()),
                    abs_step,
                )
                # No forced flush — `log_scalar` auto-flushes when the
                # logger's buffer fills; user controls cadence via
                # `buffer_size`. Final residual sent by `logger.close()`.

        # `diag_every` — drain the trainer's metric bundle through the
        # logger at its own cadence. Default trait impl is no-op for
        # trainers that haven't wired this up yet.
        comptime if L.ENABLED:
            if diag_every > 0 and abs_step % diag_every == 0 and Bool(logger):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        # `checkpoint_every` — overwrite `checkpoint_path` with the
        # trainer's one-file v3 checkpoint. The trait default raises.
        if (
            checkpoint_every > 0
            and abs_step % checkpoint_every == 0
            and checkpoint_path.byte_length() > 0
        ):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)

        # `eval_every` — deterministic (noise-off) greedy eval on a SEPARATE
        # env. For ε=0 Noisy nets the training rollout IS the noisy argmax, so
        # `avg_reward` under-reports the learned policy; bracket a greedy
        # rollout with set_noise_scale(0)/(1) to log the true policy quality.
        if eval_every > 0 and Bool(eval_env) and abs_step % eval_every == 0:
            trainer.set_noise_scale(Scalar[DT](0.0))
            var eval_ret = run_offpolicy_discrete_eval[A, E](
                trainer,
                eval_env.value()[],
                eval_episodes,
                max_steps_per_episode=eval_max_steps,
                verbose=False,
            )
            trainer.set_noise_scale(Scalar[DT](1.0))
            if verbose:
                prog.clear()
                print(
                    "[step ",
                    abs_step,
                    "] eval/mean_return = ",
                    eval_ret,
                )
            comptime if L.ENABLED:
                if Bool(logger):
                    logger.value()[].log_scalar(
                        "eval_return", Float64(eval_ret), abs_step
                    )
                    logger.value()[].flush()

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_eval — single-env greedy eval.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_eval[
    A: OffPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Non-mutating greedy eval driver for discrete agents.

    Uses `select_greedy_action` (argmax Q, no epsilon). Does not
    touch the trainer's replay buffer, optimizers, or episode tracker.

    Returns:
        Mean return over `num_episodes` evaluation episodes.
    """
    comptime OBS = A.AGENT_OBS_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var total_return = Scalar[DT](0.0)
    var t_start = perf_counter_ns()

    for ep in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        for _ in range(max_steps_per_episode):
            for d in range(OBS):
                obs[d] = Scalar[DT](obs_list[d])
            var action_idx = trainer.select_greedy_action(obs)
            var step_res = env.step_obs(action_idx)
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            ep_return += Scalar[DT](reward)
            ep_steps += 1
            if done:
                break
            obs_list = nxt^

        total_return += ep_return
        if verbose:
            print(
                "  [eval ep ",
                ep + 1,
                "/",
                num_episodes,
                "] return=",
                ep_return,
                " steps=",
                ep_steps,
            )

    var mean = total_return / Scalar[DT](num_episodes)
    if verbose:
        var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
        print(
            "eval: mean_return=",
            mean,
            " (",
            num_episodes,
            " episodes, ",
            elapsed,
            " s)",
        )
    return mean


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_train_gpu_batched — Tier-3, GPU-batched envs.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_train_gpu_batched[
    A: OffPolicyDiscreteAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int,
    NS: Int = 1,
    L: Logger = NoOpLogger,
    USE_TRAIN_CUDA_GRAPH: Bool = False,
](
    ctx: DeviceContext,
    mut trainer: A,
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
    """GPU-batched discrete off-policy training driver (Tier-3).

    The discrete analogue of `run_offpolicy_train_batched`'s GPU-env path.
    Steps `N_ENVS` GPU-resident environments in parallel (Pong / Breakout /
    SpaceInvaders / CartPole-GPU via `BatchedGpuDiscreteEnv`) while the
    Q-network trains on the same device — both `env` and `trainer` must be
    `"gpu"` target. Unlike the single-env `run_offpolicy_discrete_train`
    (CPU env stepping), this keeps the entire collect→train loop on device,
    so it matches the legacy `train_gpu` throughput the deep_agents
    discrete path previously lacked.

    Loop per iteration (advances `step_idx` by `N_ENVS`):
      1. snapshot env.obs_ptr() → prev_obs (D→D copy, driver-owned)
      2. trainer.select_action_batched[N_ENVS] → env.action_ptr() (device)
      3. env.step_batch[N_ENVS] → env obs/reward/done/terminated
      4. record_batch_gpu (NS==1) OR record_batch_gpu_nstep (NS>1) over the
         env's device pointers — replay stores `terminated`, so the TD
         bootstrap survives truncation and drops on natural termination
      5. per-env return accumulation (D2H reward+done) + add_complete_return
      6. env.selective_reset_batch[N_ENVS]
      7. updates_per_step × trainer.train_step

    Args:
        ctx: Device context (same one the trainer + env were built with).
        trainer: A GPU discrete off-policy trainer (C51 / Rainbow).
        env: A GPU `BatchedEnv` (e.g. `BatchedGpuDiscreteEnv`).
        total_env_steps: Total env transitions (counts each of the N_ENVS
            per iteration).
        rng_seed: Base seed for env step/reset RNG streams.
        updates_per_step: Gradient updates per env iteration (replay ratio
            ≈ updates_per_step / N_ENVS).
        print_every: Env-step cadence for progress prints.
        verbose: Whether to print progress lines.
        nstep_gamma: Discount for the device n-step accumulator (NS>1).
        logger: Optional logger to flush metrics through.
        base_step: Cumulative env-step offset (threaded across chunked calls).
        diag_every: Diagnostics flush cadence (env steps; 0 disables).
        checkpoint_every: Checkpoint cadence (env steps; 0 disables).
        checkpoint_path: Destination path for periodic checkpoints.
        artifacts: Sink each written checkpoint is offered to (None: not offered).
        run_dir: Run directory the offered checkpoint path is made relative to.
        eval_env: Optional separate env used for greedy evaluation.
        eval_every: Greedy-eval cadence (env steps; 0 disables).
        eval_episodes: Episodes per greedy-eval pass.
        eval_max_iters: Max env iterations per eval episode (safety cap).
        progress_label: Label for the progress bar.
        episode_sync_every: Batch the per-iteration reward+done D2H: copies.

    Returns:
        List of `trainer.mean_return()` snapshots at episode boundaries.
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM

    comptime assert env_target == "gpu", (
        "run_offpolicy_discrete_train_gpu_batched: env must be a GPU"
        " BatchedEnv (env_target == 'gpu')"
    )
    comptime assert train_target == "gpu", (
        "run_offpolicy_discrete_train_gpu_batched: trainer must be GPU"
        " (train_target == 'gpu')"
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert NS > 0, "NS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS
    ), "BatchedEnv OBS_DIM must match trainer AGENT_OBS_DIM"

    # Device n-step accumulator (NS > 1 only). N_ENVS is method-comptime on
    # the trainer's record path, so the driver owns the buffer.
    var nstep_buf: Optional[
        GPUNStepBuffer[NS, OBS, A.AGENT_ACT_DIM, N_ENVS]
    ] = None
    if NS > 1:
        nstep_buf = Optional(
            GPUNStepBuffer[NS, OBS, A.AGENT_ACT_DIM, N_ENVS].new(
                ctx, gamma=nstep_gamma
            )
        )

    var prev_obs = DriverScratch["prev_obs", N_ENVS, OBS].make["gpu"](ctx=ctx)

    # `episode_sync_every` batches the per-iteration reward+done D2H: copies
    # land async into a host ring (one slot per pending iteration) and the
    # stream is drained with ONE `synchronize` only when the ring fills or a
    # log/diag/ckpt/eval/end boundary is imminent (see `EpisodeReturnRing`;
    # `episode_sync_every == 1` reproduces the original per-iteration sync
    # exactly). Mirrors the SAC GPU-env driver. The selective_reset uses the
    # DEVICE done buffer, so it is unaffected.
    var ep_ring = EpisodeReturnRing[N_ENVS].make(ctx, episode_sync_every)

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    # Shared threshold-counter cadence state (see blocks/cadence.mojo).
    # Deterministic greedy-eval cadence: the training `avg_reward` is the
    # NOISY rollout (for ε=0 Noisy nets the acting policy IS the noisy
    # argmax), so it systematically under-reports the learned greedy policy.
    # When an isolated `eval_env` is supplied, run a noise-off greedy rollout
    # and log the true policy quality as `eval/mean_return`.
    var cad = DriverCadence.make(
        print_every,
        min_stride=N_ENVS,
        label=progress_label,
        verbose=verbose,
        diag_every=diag_every,
        checkpoint_every=checkpoint_every,
        ckpt_enabled=checkpoint_path.byte_length() > 0,
        eval_every=eval_every,
        eval_enabled=Bool(eval_env),
    )

    # Lazily-captured train-step graph (USE_TRAIN_CUDA_GRAPH). Declared at
    # function scope so the binding survives across loop iterations; a no-op on
    # non-NVIDIA where `CUDAGraph` itself no-ops.
    var train_graph: Optional[CUDAGraph] = None

    while step_idx < total_env_steps:
        # ── 1. Snapshot prev_obs (D→D) from the env's obs buffer.
        var env_obs_view = DeviceBuffer[DT](
            ctx, env.obs_ptr(), N_ENVS * OBS, owning=False
        )
        ctx.enqueue_copy(prev_obs.dev.value(), env_obs_view)

        # ── 2. Trainer writes action indices directly into env.action_ptr().
        trainer.select_action_batched[N_ENVS](
            env.obs_ptr(),
            env.action_ptr(),
            base_step + step_idx,
        )

        # ── 3. Env step (writes env-internal obs/reward/done/terminated).
        env.step_batch[N_ENVS](
            ctx=ctx, rng_seed=rng_seed + UInt64(iter_idx + 1)
        )

        # ── 4. Replay push over the env's device pointers. `terminated`
        # (NOT the combined `done`) is stored so the TD bootstrap is kept on
        # time-limit truncation and dropped on natural termination.
        # Discrete action buffer: one Scalar[DT] index per env.
        var action_buf = DeviceBuffer[DT](
            ctx, env.action_ptr(), N_ENVS, owning=False
        )
        var reward_buf = DeviceBuffer[DT](
            ctx, env.reward_ptr(), N_ENVS, owning=False
        )
        var obs_buf = DeviceBuffer[DT](
            ctx, env.obs_ptr(), N_ENVS * OBS, owning=False
        )
        var term_buf = DeviceBuffer[DT](
            ctx, env.terminated_ptr(), N_ENVS, owning=False
        )
        comptime if NS > 1:
            trainer.record_batch_gpu_nstep[N_ENVS, NS](
                ctx,
                nstep_buf.value(),
                prev_obs.dev.value(),
                action_buf,
                reward_buf,
                obs_buf,
                term_buf,
            )
        else:
            trainer.record_batch_gpu[N_ENVS](
                ctx,
                prev_obs.dev.value(),
                action_buf,
                reward_buf,
                obs_buf,
                term_buf,
            )

        # ── 5. Per-env episode tracking. Enqueue the small reward+done D2H
        # into the next ring slot WITHOUT synchronizing; drain (one sync) only
        # when the ring fills or a log/diag/ckpt/eval/end boundary is imminent.
        # `step_idx` is pre-increment here, so the upcoming post-increment value
        # the cadence blocks test is `step_idx + N_ENVS`.
        ep_ring.enqueue(ctx, env.reward_ptr(), env.done_ptr())

        var emit_now = cad.emit_boundary_imminent(
            step_idx + N_ENVS, total_env_steps
        )
        if ep_ring.due(emit_now):
            var completed = ep_ring.drain(ctx)
            for i in range(len(completed)):
                trainer.add_complete_return(completed[i])
                ep_returns.append(trainer.mean_return())

        # ── 6. Selective reset of the done envs.
        env.selective_reset_batch[N_ENVS](
            ctx=ctx, rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7)
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 7. Trainer updates.
        # Gated on NVIDIA to honour the 'NVIDIA only' contract this flag
        # has always claimed. The continuous driver's ENV-graph twin was
        # MEASURED to freeze the env on Apple/Metal rather than no-op
        # (see driver_offpolicy.mojo, 2026-08-07); this train-side branch
        # was not independently measured, and is gated for consistency.
        comptime if USE_TRAIN_CUDA_GRAPH and has_nvidia_gpu_accelerator():
            # Capture path: once the buffer is warm, the per-update device
            # kernel sequence (`train_device_kernels`) is captured into
            # `train_graph` on first call and replayed thereafter — host
            # bookkeeping advances via `note_train_update` so counters stay
            # correct across replays. During warmup we skip training, matching
            # `train_step`'s gating to False. On non-NVIDIA the closure simply
            # runs each call (no-op capture) → identical to the non-captured
            # path. Replay must be uniform or device-PER (DEVICE_TREE_, the
            # default); ERE / host-PER are NOT capture-safe. For PER the IS-β
            # is baked at capture time (anneal freezes — `set_beta` no longer
            # takes effect on the captured graph), which is benign.
            if base_step + step_idx >= trainer.learning_starts_count():
                # Capture ALL `updates_per_step` device-kernel sequences into a
                # SINGLE graph, replayed once per iteration — each captured
                # `train_device_kernels` advances the device RNG counters, so
                # the replayed sequence draws a fresh minibatch per sub-update.
                def _captured_updates() capturing raises -> None:
                    for _ in range(updates_per_step):
                        trainer.train_device_kernels()

                maybe_capture_replay[_captured_updates](train_graph, ctx)
                # Host bookkeeping advances once per logical update.
                for _ in range(updates_per_step):
                    trainer.note_train_update()
        else:
            for _ in range(updates_per_step):
                _ = trainer.train_step(base_step + step_idx)

        var abs_step = base_step + step_idx

        cad.tick(step_idx, trainer.total_train_steps())

        if cad.print_due(step_idx):
            cad.print_status(
                abs_step, trainer.mean_return(), trainer.ep_count()
            )

        # Logger emit WITH a live flush at the (rare) print cadence.
        comptime if L.ENABLED:
            if Bool(logger) and cad.log_due(step_idx):
                cad.log_status[L, True](
                    logger,
                    abs_step,
                    trainer.mean_return(),
                    trainer.ep_count(),
                )

        comptime if L.ENABLED:
            if Bool(logger) and cad.diag_due(step_idx):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        if cad.ckpt_due(step_idx):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)

        # Deterministic greedy eval on the isolated `eval_env` (noise off).
        if cad.eval_due(step_idx):
            var eval_ret = run_offpolicy_discrete_eval_batched[A, E, N_ENVS](
                ctx,
                trainer,
                eval_env.value()[],
                eval_episodes,
                max_iters=eval_max_iters,
                rng_seed=rng_seed + UInt64(step_idx + 1),
            )
            comptime if L.ENABLED:
                if Bool(logger):
                    logger.value()[].log_scalar(
                        "eval_return", Float64(eval_ret), abs_step
                    )
                    logger.value()[].flush()
            if verbose:
                cad.clear()
                print(
                    "[step ",
                    abs_step,
                    "] eval/mean_return = ",
                    eval_ret,
                )

    if cad.ckpt_on:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    # Defensive final drain of any buffered episode readbacks — a no-op
    # (and NO sync) when the last iteration hit an emit boundary.
    var completed_final = ep_ring.drain(ctx)
    for i in range(len(completed_final)):
        trainer.add_complete_return(completed_final[i])
        ep_returns.append(trainer.mean_return())

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_train_cpu_env_gpu_agent — batched CPU envs,
#   GPU trainer (the Atari-emulator row: cpu env / gpu train / N_ENVS).
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_train_cpu_env_gpu_agent[
    A: OffPolicyDiscreteAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int,
    NS: Int = 1,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut trainer: A,
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
) raises -> List[Scalar[DT]]:
    """CPU-env / GPU-agent hybrid discrete off-policy driver.

    The discrete sibling of `run_offpolicy_train_cpu_env_gpu_agent`:
    steps `N_ENVS` host-side envs per iteration (e.g. Atari emulators in
    a `BatchedCpuDiscreteEnv`, which parallelizes the step across CPU
    cores) while the Q-network selects actions and trains on the GPU.
    This is the Stage-1 throughput path from `docs/ATARI_AUDIT.md` — the
    emulator cannot run on GPU, so multi-core batched stepping is the
    scaling lever, and the single-env `run_offpolicy_discrete_train`
    left all but one core idle.

    Per-iteration loop (advances `step_idx` by `N_ENVS`):
      1. H2D current obs (env.obs_ptr → obs_dev) + D→D snapshot into
         prev_obs_dev (the `s_t` of the transition).
      2. `trainer.select_action_batched[N_ENVS]` on device (warmup /
         exploration gated inside the trainer on `base_step + step_idx`).
      3. D2H action indices → env.action_ptr, `synchronize()` so the
         host slab is valid before the CPU envs step.
      4. `env.step_batch[N_ENVS]` on the host (multi-core).
      5. H2D next-obs / reward / terminated, then
         `record_batch_gpu` (NS==1) or `record_batch_gpu_nstep` (NS>1).
         Replay stores `terminated` so the TD bootstrap survives
         time-limit truncation and drops on natural termination.
      6. `synchronize()` so device reads of the staged buffers complete
         before the host mutates env slabs in selective_reset.
      7. Host per-env return accumulation; `add_complete_return` on done.
      8. `env.selective_reset_batch[N_ENVS]` on the host.
      9. `updates_per_step` × `trainer.train_step` (GPU). Replay ratio
         = updates_per_step / N_ENVS.

    `NS` must match the trainer's target-Y γ^N bootstrap (Rainbow N-step);
    `nstep_gamma` is the device n-step accumulator's discount. `eval_env`
    enables periodic noise-off greedy eval on a SEPARATE batched env via
    `run_offpolicy_discrete_eval_cpu_env_gpu_agent`.
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM

    comptime assert env_target == "cpu", (
        "run_offpolicy_discrete_train_cpu_env_gpu_agent: env must be a CPU"
        " BatchedEnv (for a GPU env use"
        " run_offpolicy_discrete_train_gpu_batched)"
    )
    comptime assert train_target == "gpu", (
        "run_offpolicy_discrete_train_cpu_env_gpu_agent: trainer must be GPU"
        " (for a CPU trainer use run_offpolicy_discrete_train)"
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert NS > 0, "NS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS
    ), "BatchedEnv OBS_DIM must match trainer AGENT_OBS_DIM"

    # Device n-step accumulator (NS > 1 only) — same ownership pattern as
    # the GPU-batched discrete driver.
    var nstep_buf: Optional[
        GPUNStepBuffer[NS, OBS, A.AGENT_ACT_DIM, N_ENVS]
    ] = None
    if NS > 1:
        nstep_buf = Optional(
            GPUNStepBuffer[NS, OBS, A.AGENT_ACT_DIM, N_ENVS].new(
                ctx, gamma=nstep_gamma
            )
        )

    # Device staging: obs_dev holds s_t then s_{t+1}; prev_obs_dev keeps
    # the pre-step s_t for the stored transition; reward/term are H2D
    # mirrors of the env's host slabs; action_dev is one index per env.
    var obs_dev = DriverScratch["hd_obs", N_ENVS, OBS].make["gpu"](ctx=ctx)
    var prev_obs_dev = DriverScratch["hd_prev_obs", N_ENVS, OBS].make["gpu"](
        ctx=ctx
    )
    var action_dev = DriverScratch["hd_action", N_ENVS, 1].make["gpu"](ctx=ctx)
    var reward_dev = DriverScratch["hd_reward", N_ENVS, 1].make["gpu"](ctx=ctx)
    var term_dev = DriverScratch["hd_term", N_ENVS, 1].make["gpu"](ctx=ctx)

    var per_env_returns = List[Scalar[DT]](
        length=N_ENVS,
        fill=Scalar[DT](0.0),
    )

    # CPU env: ctx ignored. Wrap in Optional(None) for the trait sig.
    env.reset_batch[N_ENVS](ctx=None, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    # Shared threshold-counter cadence state (see blocks/cadence.mojo).
    var cad = DriverCadence.make(
        print_every,
        min_stride=N_ENVS,
        label=progress_label,
        verbose=verbose,
        diag_every=diag_every,
        checkpoint_every=checkpoint_every,
        ckpt_enabled=checkpoint_path.byte_length() > 0,
        eval_every=eval_every,
        eval_enabled=Bool(eval_env),
    )

    while step_idx < total_env_steps:
        # ── 1. H2D current obs (s_t) + D→D snapshot into prev_obs.
        ctx.enqueue_copy(obs_dev.dev.value(), env.obs_ptr())
        ctx.enqueue_copy(prev_obs_dev.dev.value(), obs_dev.dev.value())

        # ── 2. Action selection on device (exploration inside trainer).
        trainer.select_action_batched[N_ENVS](
            obs_dev.dev_ptr(),
            action_dev.dev_ptr(),
            base_step + step_idx,
        )

        # ── 3. D2H action indices → env host slab; sync before CPU steps.
        ctx.enqueue_copy(env.action_ptr(), action_dev.dev.value())
        ctx.synchronize()

        # ── 4. Step CPU envs (multi-core; writes host obs/reward/done/term).
        env.step_batch[N_ENVS](
            ctx=None,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 5. H2D next-obs / reward / terminated, then GPU replay push.
        ctx.enqueue_copy(obs_dev.dev.value(), env.obs_ptr())
        ctx.enqueue_copy(reward_dev.dev.value(), env.reward_ptr())
        ctx.enqueue_copy(term_dev.dev.value(), env.terminated_ptr())
        comptime if NS > 1:
            trainer.record_batch_gpu_nstep[N_ENVS, NS](
                ctx,
                nstep_buf.value(),
                prev_obs_dev.dev.value(),
                action_dev.dev.value(),
                reward_dev.dev.value(),
                obs_dev.dev.value(),
                term_dev.dev.value(),
            )
        else:
            trainer.record_batch_gpu[N_ENVS](
                ctx,
                prev_obs_dev.dev.value(),
                action_dev.dev.value(),
                reward_dev.dev.value(),
                obs_dev.dev.value(),
                term_dev.dev.value(),
            )

        # ── 6. Sync so device reads of the env host slabs finish before
        # selective_reset (host) mutates env.obs in place.
        ctx.synchronize()

        # ── 7. Per-env episode tracking (host reward + done).
        var completed = List[Scalar[DT]]()
        accumulate_episode_returns[N_ENVS](
            env.reward_ptr(), env.done_ptr(), per_env_returns, completed
        )
        for i in range(len(completed)):
            trainer.add_complete_return(completed[i])
            ep_returns.append(trainer.mean_return())

        # ── 8. Selective env reset (host, multi-core).
        env.selective_reset_batch[N_ENVS](
            ctx=None,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 9. Trainer updates (GPU).
        for _ in range(updates_per_step):
            _ = trainer.train_step(base_step + step_idx)

        var abs_step = base_step + step_idx

        cad.tick(step_idx, trainer.total_train_steps())

        if cad.print_due(step_idx):
            cad.print_status(
                abs_step, trainer.mean_return(), trainer.ep_count()
            )

        # Logger emit WITH a live flush at the (rare) print cadence.
        comptime if L.ENABLED:
            if Bool(logger) and cad.log_due(step_idx):
                cad.log_status[L, True](
                    logger,
                    abs_step,
                    trainer.mean_return(),
                    trainer.ep_count(),
                )

        comptime if L.ENABLED:
            if Bool(logger) and cad.diag_due(step_idx):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        if cad.ckpt_due(step_idx):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)

        # Deterministic greedy eval on the isolated `eval_env` (noise off).
        if cad.eval_due(step_idx):
            var eval_ret = run_offpolicy_discrete_eval_cpu_env_gpu_agent[
                A, E, N_ENVS
            ](
                ctx,
                trainer,
                eval_env.value()[],
                eval_episodes,
                max_iters=eval_max_iters,
                rng_seed=rng_seed + UInt64(step_idx + 1),
            )
            comptime if L.ENABLED:
                if Bool(logger):
                    logger.value()[].log_scalar(
                        "eval_return", Float64(eval_ret), abs_step
                    )
                    logger.value()[].flush()
            if verbose:
                cad.clear()
                print(
                    "[step ",
                    abs_step,
                    "] eval/mean_return = ",
                    eval_ret,
                )

    if cad.ckpt_on:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_eval_cpu_env_gpu_agent — noise-off greedy
#   rollout over batched CPU envs with a GPU trainer.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_eval_cpu_env_gpu_agent[
    A: OffPolicyDiscreteAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int,
](
    ctx: DeviceContext,
    mut trainer: A,
    mut eval_env: E,
    num_episodes: Int,
    *,
    max_iters: Int = 20_000,
    rng_seed: UInt64 = UInt64(123),
) raises -> Scalar[DT]:
    """Deterministic greedy eval over batched CPU envs with a GPU agent.

    The hybrid sibling of `run_offpolicy_discrete_eval_batched`: obs is
    staged H2D each iteration, `select_greedy_action_batched` runs on
    device, action indices come back D2H, the envs step on the host.
    Does NOT touch the trainer's replay/optimizer/episode tracker — use
    a SEPARATE env instance from training.

    Returns the mean completed-episode return (0 if none completed).
    """
    comptime OBS = A.AGENT_OBS_DIM
    comptime assert (
        E.ENV_TARGET == "cpu"
    ), "run_offpolicy_discrete_eval_cpu_env_gpu_agent: env must be CPU"
    comptime assert (
        E.OBS_DIM == OBS
    ), "eval_env OBS_DIM must match trainer AGENT_OBS_DIM"

    var obs_dev = DriverScratch["hde_obs", N_ENVS, OBS].make["gpu"](ctx=ctx)
    var action_dev = DriverScratch["hde_action", N_ENVS, 1].make["gpu"](ctx=ctx)

    trainer.set_noise_scale(Scalar[DT](0.0))  # deterministic mean weights

    eval_env.reset_batch[N_ENVS](ctx=None, rng_seed=rng_seed)

    var per_env = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
    var returns_sum = Scalar[DT](0.0)
    var n_done: Int = 0
    var it: Int = 0
    while n_done < num_episodes and it < max_iters:
        ctx.enqueue_copy(obs_dev.dev.value(), eval_env.obs_ptr())
        trainer.select_greedy_action_batched[N_ENVS](
            obs_dev.dev_ptr(), action_dev.dev_ptr()
        )
        ctx.enqueue_copy(eval_env.action_ptr(), action_dev.dev.value())
        ctx.synchronize()
        eval_env.step_batch[N_ENVS](
            ctx=None, rng_seed=rng_seed + UInt64(it + 1)
        )
        var rew_h = eval_env.reward_ptr()
        var done_h = eval_env.done_ptr()
        for e in range(N_ENVS):
            per_env[e] = per_env[e] + rew_h[unsafe_offset=e]
            if done_h[unsafe_offset=e] > Scalar[DT](0.5):
                returns_sum = returns_sum + per_env[e]
                per_env[e] = Scalar[DT](0.0)
                n_done += 1
        eval_env.selective_reset_batch[N_ENVS](
            ctx=None, rng_seed=rng_seed + UInt64(it + 1) * UInt64(7)
        )
        it += 1

    trainer.set_noise_scale(Scalar[DT](1.0))  # restore noisy exploration

    if n_done == 0:
        return Scalar[DT](0.0)
    return returns_sum / Scalar[DT](n_done)


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_eval_batched — noise-off greedy GPU rollout.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_eval_batched[
    A: OffPolicyDiscreteAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int,
](
    ctx: DeviceContext,
    mut trainer: A,
    mut eval_env: E,
    num_episodes: Int,
    *,
    max_iters: Int = 20_000,
    rng_seed: UInt64 = UInt64(123),
) raises -> Scalar[DT]:
    """Deterministic greedy eval over a GPU-batched discrete env.

    Disables Noisy-net exploration (`set_noise_scale(0)`), rolls the
    greedy policy (`select_greedy_action_batched`) on `N_ENVS` parallel
    envs until `num_episodes` complete (or `max_iters` iterations elapse),
    then restores noise. Does NOT touch the trainer's replay, optimizer,
    or episode tracker — use a SEPARATE env instance from training.

    `max_iters` bounds iterations (each advances all N_ENVS envs one step);
    set it well above `ceil(num_episodes / N_ENVS) · episode_length`.

    Returns the mean completed-episode return (0 if none completed).
    """
    comptime OBS = A.AGENT_OBS_DIM
    comptime assert (
        E.OBS_DIM == OBS
    ), "eval_env OBS_DIM must match trainer AGENT_OBS_DIM"

    trainer.set_noise_scale(Scalar[DT](0.0))  # deterministic mean weights

    eval_env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var per_env = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
    var rew_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
    var done_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))

    var returns_sum = Scalar[DT](0.0)
    var n_done: Int = 0
    var it: Int = 0
    while n_done < num_episodes and it < max_iters:
        trainer.select_greedy_action_batched[N_ENVS](
            eval_env.obs_ptr(), eval_env.action_ptr()
        )
        eval_env.step_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed + UInt64(it + 1))
        var rv = DeviceBuffer[DT](
            ctx, eval_env.reward_ptr(), N_ENVS, owning=False
        )
        var dv = DeviceBuffer[DT](
            ctx, eval_env.done_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(rew_h.unsafe_ptr(), rv)
        ctx.enqueue_copy(done_h.unsafe_ptr(), dv)
        ctx.synchronize()
        for e in range(N_ENVS):
            per_env[e] = per_env[e] + rew_h[e]
            if done_h[e] > Scalar[DT](0.5):
                returns_sum = returns_sum + per_env[e]
                per_env[e] = Scalar[DT](0.0)
                n_done += 1
        eval_env.selective_reset_batch[N_ENVS](
            ctx=ctx, rng_seed=rng_seed + UInt64(it + 1) * UInt64(7)
        )
        it += 1

    trainer.set_noise_scale(Scalar[DT](1.0))  # restore noisy exploration

    if n_done == 0:
        return Scalar[DT](0.0)
    return returns_sum / Scalar[DT](n_done)
