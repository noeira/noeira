"""On-policy training driver — namespace twin of `driver_offpolicy.mojo`.

Mirrors the symbol shape of the off-policy driver so on-policy
trainers (PPO today, possibly A2C later) plug into a consistent surface:

  - `OnPolicyAgent` — N=1 host-list trait for single-env trainers.
  - `OnPolicyAgentBatched` — N_ENVS-wide pointer trait for batched
    trainers (PPOTrainer conforms).
  - `run_onpolicy_train` — single-env on-policy training driver.
  - `run_onpolicy_train_batched` — BatchedEnv driver covering same-
    target (env_target == train_target) combinations × any N_ENVS,
    via host-staging scratches (D2H for GPU env).

Cross-target (cpu env, gpu train) and the degenerate (gpu env, cpu
train) combination are not exposed by `run_onpolicy_train_batched` —
single-env users get `run_onpolicy_train`; everyone else uses one
target consistently across env + trainer.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.utils.progress import IntervalProgress
from noeira.core.env_traits import BoxContinuousActionEnv
from .batched_env import BatchedEnv
from .driver_scratch import DriverScratch
from .blocks.cadence import DriverCadence
from .checkpoint import announce_checkpoint
from ...io.artifact_sink import ArtifactSink


trait OnPolicyCheckpointable(Deinitable, Movable):
    """Shared ROOT surface for ALL on-policy traits (continuous +
    discrete, single-env + batched).

    Everything common to the whole family is declared here ONCE:
    PPOTrainer conforms to both `OnPolicyAgent` AND `OnPolicyAgentBatched`
    (and PPODiscreteTrainer to `OnPolicyDiscreteAgentBatched`, which
    inherits both `OnPolicyDiscreteAgent` and `OnPolicyBatchedCore`). If
    two traits in such a diamond declared the same method independently,
    resolving the concrete override against two unrelated declarations
    recurses ("attempt to resolve a recursive reference to
    `PPOTrainer.save_state`"). Hoisting each shared member into this
    common ancestor gives a single declaration → the diamonds resolve
    cleanly. Off-policy trainers avoid this naturally (single-trait chain
    `OffPolicyAgentGpu(OffPolicyAgent)`)."""

    # Trait-visible aliases of the trainer's struct comptime params —
    # the drivers comptime-gate H2D/D2H staging + size scratches on these.
    comptime AGENT_TRAIN_TARGET: StaticString
    comptime AGENT_OBS_DIM: Int

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """Returns False on most steps; True when a rollout-length
        boundary is hit and the K-epoch minibatch updates fire."""
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...

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

    def total_train_steps(self) -> Int:
        """Cumulative gradient-update count for the inter-log progress bar's
        `Train:` field. Declared on the shared ancestor so BOTH on-policy
        traits (single-env + batched) inherit it. Default 0 for trainers
        that don't track it; real trainers may override."""
        return 0


trait OnPolicyAgent(OnPolicyCheckpointable):
    """Surface every nn on-policy trainer (PPO / future A2C) exposes
    for the on-policy training driver.

    Per-step contract mirrors the off-policy driver so the loop stays
    almost identical (collect transition → record → call `train_step`
    once per env step). The only behavioural difference is that
    on-policy `train_step` returns False on the vast majority of steps
    and True only when a rollout-length boundary is hit and the
    K-epoch minibatch updates fire.

    Internal state ownership: the trainer caches `(unbounded action,
    log_prob, value)` between `select_action` and `record_transition`.
    Callers must invoke them in pairs — same as the off-policy driver's
    select-then-record pattern. The driver does NOT pass log_prob /
    value back to the trainer; the trainer caches them itself.

    `select_action` writes the *env-ready* action (already action-scaled
    and clamped). The trainer's internal cache holds the *unbounded*
    sample used for the log_prob during the upcoming update.
    """

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        ...

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        ...

    def end_episode(mut self):
        ...

    def mark_terminal(mut self) raises:
        """Mark the just-recorded (N=1) transition as a TRUE terminal so GAE
        zeroes its V bootstrap. The driver calls this only when the env
        reports `was_terminated()` — time-limit truncation is left
        unmarked (bootstrap kept)."""
        ...

    # `train_step` / `mean_return` / `ep_count` / `total_train_steps` and
    # the cadence hooks (`flush_metrics_through_logger` / `save_state`)
    # are inherited from `OnPolicyCheckpointable` — declared once so the
    # PPOTrainer override doesn't recurse across two trait declarations.


def run_onpolicy_train[
    A: OnPolicyAgent,
    E: BoxContinuousActionEnv,
    L: Logger = NoOpLogger,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    obs_dim: Int,
    act_dim: Int,
    print_every: Int = 1_000,
    verbose: Bool = True,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    artifacts: Optional[ArtifactSink] = None,
    run_dir: String = "",
    base_step: Int = 0,
    progress_label: String = "on-policy",
) raises -> List[Scalar[DT]]:
    """Step-based on-policy single-env training driver.

    One env step + one `train_step` call per iteration. PPO's rollout
    accumulation and K-epoch update fire inside `trainer.train_step`
    whenever a rollout-length boundary is crossed (most steps return
    False).

    Args:
        trainer: Any nn on-policy trainer (PPO today).
        env: Any `BoxContinuousActionEnv`.
        total_timesteps: Number of env steps to run.
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        print_every: Verbose status-line cadence (env-steps). 0 disables.
        verbose: Print a per-cadence status line.
        logger: Optional logger instance.
        diag_every: Diagnostic logging cadence (env-steps). 0 disables.
        checkpoint_every: Checkpoint writing cadence (env-steps). 0 disables.
        checkpoint_path: Path to write checkpoints to.
        artifacts: Sink each written checkpoint is offered to (None: not offered).
        run_dir: Run directory the offered checkpoint path is made relative to.
        base_step: Base step counter for the training loop.
        progress_label: Label for the progress bar.

    Returns:
        List of `trainer.mean_return()` snapshots taken at each completed
        episode boundary (same shape as the off-policy driver).
    """
    var obs = List[Scalar[DT]](length=obs_dim, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=obs_dim, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=act_dim, fill=Scalar[DT](0.0))

    var obs_list = env.reset_obs_list()
    var action_list = List[Scalar[E.dtype]](capacity=act_dim)
    for _ in range(act_dim):
        action_list.append(Scalar[E.dtype](0.0))

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    # In-place progress bar between log lines (pure CPU, no GPU sync).
    var prog = IntervalProgress(
        print_every, label=progress_label, enabled=verbose
    )
    while step < total_timesteps:
        for d in range(obs_dim):
            obs[d] = Scalar[DT](obs_list[d])
        # `base_step + step` — cumulative env-step counter for the
        # trainer's warmup gating (when chunked through agent wrappers).
        trainer.select_action(obs, action, base_step + step)
        for j in range(act_dim):
            action_list[j] = Scalar[E.dtype](action[j])
        var step_res = env.step_continuous_vec[E.dtype](action_list)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(obs_dim):
            next_obs[d] = Scalar[DT](nxt[d])
        trainer.record_transition(
            obs,
            action,
            Scalar[DT](reward),
            next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        # Mark the just-recorded transition as a TRUE terminal (V(s')=0 in
        # GAE) ONLY on natural termination; time-limit truncation keeps the
        # value bootstrap (CleanRL / Gymnasium terminated-vs-truncated). No-op
        # for non-terminating envs (`was_terminated()` default False) → GAE
        # `term_buf` stays all-zero → bit-identical on Pendulum/HalfCheetah.
        if env.was_terminated():
            trainer.mark_terminal()
        if done:
            trainer.end_episode()
            obs_list = env.reset_obs_list()
            var new_ep_count = trainer.ep_count()
            if new_ep_count > current_ep_count:
                ep_returns.append(trainer.mean_return())
                current_ep_count = new_ep_count
        else:
            obs_list = nxt^
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

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# OnPolicyAgentBatched — trait for the Tier-3 BatchedEnv driver.
# ──────────────────────────────────────────────────────────────────────


trait OnPolicyBatchedCore(OnPolicyCheckpointable):
    """N_ENVS-wide pointer surface shared by the CONTINUOUS and DISCRETE
    batched on-policy drivers — the trait bound of the ONE shared loop
    body `_run_onpolicy_batched_body`. The two batched loops were
    line-identical (the discrete action slot is just ACT ≡ 1 floats), so
    the method surface they consume is declared once here.

    All pointer args are HOST-side. For GPU envs the driver D2Hs
    env-side obs/reward/done into host scratches before calling. The
    trainer is responsible for any internal H2D of obs into device-
    side scratches (PPOTrainer does this inside PPOActStep)."""

    comptime AGENT_N_ENVS: Int

    def select_action_batched(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Reads AGENT_N_ENVS * AGENT_OBS_DIM from `obs_ptr`, writes
        AGENT_N_ENVS action rows into `action_ptr` (ACT_DIM floats for
        continuous trainers; ONE index-as-float for discrete). Caches
        per-env (sample, log_prob, value) internally for the upcoming
        `record_batch_cpu`. Both pointers must be host-side."""
        ...

    def record_batch_cpu(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push AGENT_N_ENVS transitions into the rollout buffer. All
        pointers host-side. Maintains per-env running returns and
        pushes completed episodes into the EpisodeTracker on done."""
        ...

    def mark_terminal_env(mut self, env_idx: Int) raises:
        """Mark the just-recorded transition for `env_idx` as a TRUE terminal
        so GAE zeroes its V bootstrap. The driver calls this only for envs
        whose `terminated_ptr()` is set — time-limit truncation is left
        unmarked (bootstrap kept)."""
        ...


trait OnPolicyAgentBatched(OnPolicyBatchedCore):
    """Continuous batched on-policy trait consumed by
    `run_onpolicy_train_batched` — the batched core plus the continuous
    action width + list-based greedy eval."""

    comptime AGENT_ACT_DIM: Int

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Single-env greedy eval — list-based. Always BATCH=1 even
        when the trainer is configured for N_ENVS > 1."""
        ...


def run_onpolicy_train_batched[
    A: OnPolicyAgentBatched,
    E: BatchedEnv,
    L: Logger = NoOpLogger,
](
    ctx: Optional[DeviceContext],
    mut trainer: A,
    mut env: E,
    total_env_steps: Int,
    *,
    rng_seed: UInt64 = UInt64(42),
    print_every: Int = 5_000,
    verbose: Bool = True,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    artifacts: Optional[ArtifactSink] = None,
    run_dir: String = "",
    base_step: Int = 0,
    progress_label: String = "on-policy",
) raises -> List[Scalar[DT]]:
    """Tier-3 on-policy driver covering same-target combinations.

    Same-target means `env_target == train_target` × any N_ENVS through
    the `BatchedEnv` trait:

      env_target | train_target | N_ENVS | covered
      -----------|--------------|--------|--------
      cpu        | cpu          | >=1    | yes  (via BatchedCpuEnv)
      gpu        | gpu          | >=1    | yes  (via BatchedGpuEnv)

    Cross-target combinations are NOT covered here:
      - (cpu env, gpu train) → use `run_onpolicy_train` (single-env)
      - (gpu env, cpu train) → degenerate (D2H every obs)

    Unlike the off-policy driver, the on-policy trainer always wants
    host-side pointers (PPO's rollout buffer lives host-only on
    both targets; the trainer itself does H2D of obs internally
    inside PPOActStep). The driver therefore stages env outputs
    through host scratches — a no-op pointer alias for CPU envs and
    a D2H copy for GPU envs.

    Loop per iteration (N_ENVS env steps):
      1. Snapshot env.obs_ptr()                  → prev_obs_h
      2. trainer.select_action_batched(prev_obs_h, action_h, step_idx)
      3. (gpu env) H2D action_h → env.action_ptr()
      4. env.step_batch[N_ENVS]
      5. Snapshot env.obs/reward/done             → next_obs_h/reward_h/done_h
      6. trainer.record_batch_cpu(prev_obs_h, reward_h, next_obs_h, done_h)
      7. env.selective_reset_batch[N_ENVS]
      8. trainer.train_step (fires the K-epoch update at rollout
         boundary)
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM
    comptime N_ENVS = A.AGENT_N_ENVS

    comptime assert (
        env_target == "cpu" or env_target == "gpu"
    ), "env_target must be 'cpu' or 'gpu'"
    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "train_target must be 'cpu' or 'gpu'"
    comptime assert env_target == train_target, (
        "run_onpolicy_train_batched: env_target must equal train_target."
        " Cross-target (cpu env, gpu train) → use run_onpolicy_train;"
        " (gpu env, cpu train) → rejected as degenerate."
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS and E.ACT_DIM == ACT
    ), "BatchedEnv dimensions must match trainer dimensions"
    comptime if env_target == "gpu":
        if not ctx:
            raise Error(
                "run_onpolicy_train_batched: ctx required when"
                " env_target is 'gpu'"
            )

    return _run_onpolicy_batched_body[A, E, ACT, L](
        ctx,
        trainer,
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
        progress_label=progress_label,
    )


def _run_onpolicy_batched_body[
    A: OnPolicyBatchedCore,
    E: BatchedEnv,
    ACT: Int,
    L: Logger = NoOpLogger,
](
    ctx: Optional[DeviceContext],
    mut trainer: A,
    mut env: E,
    total_env_steps: Int,
    *,
    rng_seed: UInt64,
    print_every: Int,
    verbose: Bool,
    logger: Optional[Pointer[L, MutAnyOrigin]],
    diag_every: Int,
    checkpoint_every: Int,
    checkpoint_path: String,
    artifacts: Optional[ArtifactSink],
    run_dir: String,
    base_step: Int,
    progress_label: String,
    stop_at_mean_return: Optional[Scalar[DT]] = None,
    stop_min_episodes: Int = 0,
) raises -> List[Scalar[DT]]:
    """ONE loop body behind BOTH batched on-policy drivers (continuous
    `run_onpolicy_train_batched` and discrete
    `run_onpolicy_discrete_train_batched`). The two loops were
    line-identical — the only genuine axis is the action-slot width
    `ACT` (AGENT_ACT_DIM floats vs ONE index-as-float), so it is a
    comptime param supplied by the public wrappers, which also carry
    the per-driver comptime asserts + ctx checks.

    `stop_at_mean_return` ends training early the first time the
    trainer's windowed `mean_return()` reaches it with at least
    `stop_min_episodes` episodes completed (pass the tracker's window
    size so the window holds real returns, not its initial fill) — the
    "solved" exit of a time-to-solve benchmark. The final checkpoint is
    still written."""
    comptime env_target: StaticString = E.ENV_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime N_ENVS = A.AGENT_N_ENVS

    # Host-side staging scratches. The trainer always reads/writes
    # host pointers; on GPU env we D2H env outputs into these.
    var prev_obs_h = DriverScratch["prev_obs", N_ENVS, OBS].make["cpu"](
        ctx=None
    )
    var action_h = DriverScratch["action", N_ENVS, ACT].make["cpu"](ctx=None)
    var next_obs_h = DriverScratch["next_obs", N_ENVS, OBS].make["cpu"](
        ctx=None
    )
    var reward_h = DriverScratch["reward", N_ENVS, 1].make["cpu"](ctx=None)
    var done_h = DriverScratch["done", N_ENVS, 1].make["cpu"](ctx=None)
    # Natural-termination flag (NOT combined done) — used to mark true
    # terminals in the rollout so GAE drops the V bootstrap on termination
    # while keeping it on time-limit truncation.
    var term_h = DriverScratch["term", N_ENVS, 1].make["cpu"](ctx=None)

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    # Shared threshold-counter cadence state (see blocks/cadence.mojo) —
    # the log counter only advances inside `comptime if L.ENABLED`, so
    # bit-identity is preserved when L=NoOpLogger (default).
    var cad = DriverCadence.make(
        print_every,
        min_stride=N_ENVS,
        label=progress_label,
        verbose=verbose,
        diag_every=diag_every,
        checkpoint_every=checkpoint_every,
        ckpt_enabled=checkpoint_path.byte_length() > 0,
    )
    var last_ep_count = trainer.ep_count()

    while step_idx < total_env_steps:
        # ── 1. Snapshot env.obs_ptr() → prev_obs_h.
        var po_p = prev_obs_h.host_ptr()
        comptime if env_target == "cpu":
            var ob_p = env.obs_ptr()
            for k in range(N_ENVS * OBS):
                po_p[unsafe_offset=k] = ob_p[unsafe_offset=k]
        else:
            var c = ctx.value()
            var env_obs_view = DeviceBuffer[DT](
                c,
                env.obs_ptr(),
                N_ENVS * OBS,
                owning=False,
            )
            var po_host = c.enqueue_create_host_buffer[DT](N_ENVS * OBS)
            c.enqueue_copy(po_host, env_obs_view)
            c.synchronize()
            var ph = po_host.unsafe_ptr()
            for k in range(N_ENVS * OBS):
                po_p[unsafe_offset=k] = ph[unsafe_offset=k]

        # ── 2. Trainer writes action into host scratch.
        # `base_step + step_idx` — cumulative env-step counter (see
        # the `base_step` note on `run_offpolicy_train`).
        trainer.select_action_batched(
            po_p,
            action_h.host_ptr(),
            base_step + step_idx,
        )

        # ── 3. (gpu env) H2D action into env.action_ptr().
        comptime if env_target == "gpu":
            var c = ctx.value()
            var env_act_view = DeviceBuffer[DT](
                c,
                env.action_ptr(),
                N_ENVS * ACT,
                owning=False,
            )
            c.enqueue_copy(env_act_view, action_h.host_ptr())
        else:
            # CPU env: copy action_h → env.action_ptr() (same target side).
            var ap = action_h.host_ptr()
            var ea = env.action_ptr()
            for k in range(N_ENVS * ACT):
                ea[unsafe_offset=k] = ap[unsafe_offset=k]

        # ── 4. Env step.
        env.step_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 5. Snapshot env outputs → host scratches.
        var no_p = next_obs_h.host_ptr()
        var rew_p = reward_h.host_ptr()
        var dn_p = done_h.host_ptr()
        var tm_p = term_h.host_ptr()
        comptime if env_target == "cpu":
            var ob_p = env.obs_ptr()
            var er_p = env.reward_ptr()
            var ed_p = env.done_ptr()
            var et_p = env.terminated_ptr()
            for k in range(N_ENVS * OBS):
                no_p[unsafe_offset=k] = ob_p[unsafe_offset=k]
            for e in range(N_ENVS):
                rew_p[unsafe_offset=e] = er_p[unsafe_offset=e]
                dn_p[unsafe_offset=e] = ed_p[unsafe_offset=e]
                tm_p[unsafe_offset=e] = et_p[unsafe_offset=e]
        else:
            var c = ctx.value()
            var env_obs_view = DeviceBuffer[DT](
                c,
                env.obs_ptr(),
                N_ENVS * OBS,
                owning=False,
            )
            var env_rew_view = DeviceBuffer[DT](
                c,
                env.reward_ptr(),
                N_ENVS,
                owning=False,
            )
            var env_done_view = DeviceBuffer[DT](
                c,
                env.done_ptr(),
                N_ENVS,
                owning=False,
            )
            var env_term_view = DeviceBuffer[DT](
                c,
                env.terminated_ptr(),
                N_ENVS,
                owning=False,
            )
            var no_host = c.enqueue_create_host_buffer[DT](N_ENVS * OBS)
            var rew_host = c.enqueue_create_host_buffer[DT](N_ENVS)
            var dn_host = c.enqueue_create_host_buffer[DT](N_ENVS)
            var tm_host = c.enqueue_create_host_buffer[DT](N_ENVS)
            c.enqueue_copy(no_host, env_obs_view)
            c.enqueue_copy(rew_host, env_rew_view)
            c.enqueue_copy(dn_host, env_done_view)
            c.enqueue_copy(tm_host, env_term_view)
            c.synchronize()
            var nh = no_host.unsafe_ptr()
            var rh = rew_host.unsafe_ptr()
            var dh = dn_host.unsafe_ptr()
            var th = tm_host.unsafe_ptr()
            for k in range(N_ENVS * OBS):
                no_p[unsafe_offset=k] = nh[unsafe_offset=k]
            for e in range(N_ENVS):
                rew_p[unsafe_offset=e] = rh[unsafe_offset=e]
                dn_p[unsafe_offset=e] = dh[unsafe_offset=e]
                tm_p[unsafe_offset=e] = th[unsafe_offset=e]

        # ── 6. Trainer push, then mark TRUE terminals (V=0 bootstrap in GAE)
        # — truncation keeps the bootstrap. No-op for non-terminating envs
        # (`term ≡ 0`) → bit-identical.
        trainer.record_batch_cpu(po_p, rew_p, no_p, dn_p)
        for e in range(N_ENVS):
            if tm_p[unsafe_offset=e] > Scalar[DT](0.5):
                trainer.mark_terminal_env(e)

        # ── 7. Selective env reset (env handles per-env done internally).
        env.selective_reset_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 8. Trainer update (returns True at K-epoch boundary).
        _ = trainer.train_step(base_step + step_idx)

        # Snapshot mean_return whenever an episode completes.
        var new_ep_count = trainer.ep_count()
        var reached_target = False
        if new_ep_count > last_ep_count:
            ep_returns.append(trainer.mean_return())
            last_ep_count = new_ep_count
            if stop_at_mean_return and new_ep_count >= stop_min_episodes:
                reached_target = (
                    trainer.mean_return() >= stop_at_mean_return.value()
                )

        var abs_step = base_step + step_idx

        if reached_target:
            if verbose:
                print(
                    "[" + progress_label + "] target mean return reached:",
                    trainer.mean_return(),
                    ">=",
                    stop_at_mean_return.value(),
                    "| step",
                    abs_step,
                    "| episodes",
                    new_ep_count,
                )
            break

        cad.tick(step_idx, trainer.total_train_steps())

        if cad.print_due(step_idx):
            cad.print_status(
                abs_step, trainer.mean_return(), trainer.ep_count()
            )

        # Logger emit at the same cadence (independent of verbose). No
        # forced flush (`buffer_size` auto-flush — see note in
        # run_offpolicy_train). Comptime-elided when L=NoOpLogger (default).
        comptime if L.ENABLED:
            if Bool(logger) and cad.log_due(step_idx):
                cad.log_status[L, False](
                    logger,
                    abs_step,
                    trainer.mean_return(),
                    trainer.ep_count(),
                )

        # `diag_every` — drain the trainer's metric bundle through the
        # logger at its own cadence. Default trait impl is no-op for
        # trainers that haven't wired this up yet.
        comptime if L.ENABLED:
            if Bool(logger) and cad.diag_due(step_idx):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        # `checkpoint_every` — overwrite `checkpoint_path` with the
        # trainer's one-file v3 checkpoint. The trait default raises.
        if cad.ckpt_due(step_idx):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)

    if cad.ckpt_on:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    return ep_returns^
