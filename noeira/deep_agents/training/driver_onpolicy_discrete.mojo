"""On-policy discrete-action training + eval drivers.

Discrete-action sibling of `driver_onpolicy.mojo` (continuous PPO).
Single-env driver for categorical on-policy trainers (discrete PPO
today, possibly discrete A2C later).

Key differences from the continuous on-policy driver:
  - Env bound: `BoxDiscreteActionEnv` (`step_obs` takes an Int action).
  - Action is a single Int index per step, not ACT_DIM floats.
  - `select_action` returns the sampled index; `record_transition`
    takes `action_idx: Int`.

  env_target | train_target | driver
  -----------|--------------|--------------------------------
  cpu        | cpu          | run_onpolicy_discrete_train
  cpu        | gpu          | run_onpolicy_discrete_train (trainer
                              H2Ds obs internally inside the act step)

The batched (Tier-3) discrete on-policy driver is deferred until a
consumer needs it — single-env covers CartPole / classic-control.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.utils.progress import IntervalProgress
from noeira.core.env_traits import BoxDiscreteActionEnv
from .batched_env import BatchedEnv
from .driver_scratch import DriverScratch
from .driver_onpolicy import (
    OnPolicyCheckpointable,
    OnPolicyBatchedCore,
    _run_onpolicy_batched_body,
)
from .checkpoint import announce_checkpoint
from ...io.artifact_sink import ArtifactSink


# ──────────────────────────────────────────────────────────────────────
# OnPolicyDiscreteAgent — trait for the discrete on-policy driver.
# ──────────────────────────────────────────────────────────────────────


trait OnPolicyDiscreteAgent(OnPolicyCheckpointable):
    """Single-env host-list surface for the discrete on-policy driver.

    Mirrors `OnPolicyAgent` (continuous PPO) but adapted for discrete
    action spaces: `select_action` / `select_greedy_action` return an
    Int action index, and `record_transition` takes that index.

    The trainer caches `(action index, log_prob, value)` between
    `select_action` and `record_transition`, exactly like the
    continuous on-policy trainer caches `(unbounded sample, log_prob,
    value)`. Callers invoke the pair in order.

    The family-wide surface (`train_step` / `mean_return` / `ep_count` /
    `total_train_steps` / cadence hooks / `AGENT_TRAIN_TARGET` /
    `AGENT_OBS_DIM`) is inherited from `OnPolicyCheckpointable` — ONE
    declaration so the `OnPolicyDiscreteAgentBatched` diamond (this trait
    + `OnPolicyBatchedCore`) resolves cleanly.
    """

    comptime AGENT_NUM_ACTIONS: Int

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        step_idx: Int,
    ) raises -> Int:
        """Sample an action index from the categorical policy and cache
        the (index, log_prob, value) triple for the upcoming record."""
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic argmax over logits — no sampling, no cache."""
        ...

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        ...

    def mark_terminal(mut self) raises:
        """Mark the just-recorded transition as a TRUE terminal so GAE
        zeroes its V bootstrap. The driver calls this only when the env
        reports `was_terminated()` — time-limit truncation keeps the
        bootstrap."""
        ...

    def end_episode(mut self):
        ...


# ──────────────────────────────────────────────────────────────────────
# run_onpolicy_discrete_train — single-env discrete on-policy training.
# ──────────────────────────────────────────────────────────────────────


def run_onpolicy_discrete_train[
    A: OnPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
    L: Logger = NoOpLogger,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
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
    """Step-based discrete on-policy single-env training driver.

    One env step + one `train_step` call per iteration. The rollout
    accumulation and K-epoch update fire inside `trainer.train_step`
    whenever a rollout-length boundary is crossed (most steps return
    False). Covers (env=cpu, train=cpu) and (env=cpu, train=gpu) —
    the trainer H2Ds obs internally on the GPU path.
    """
    comptime OBS = A.AGENT_OBS_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var obs_list = env.reset_obs_list()

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    # In-place progress bar between log lines (pure CPU, no GPU sync).
    var prog = IntervalProgress(
        print_every, label=progress_label, enabled=verbose
    )
    while step < total_timesteps:
        for d in range(OBS):
            obs[d] = Scalar[DT](obs_list[d])
        var action_idx = trainer.select_action(obs, base_step + step)
        var step_res = env.step_obs(action_idx)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = Scalar[DT](nxt[d])
        trainer.record_transition(
            obs, action_idx, Scalar[DT](reward), next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        # TRUE terminal (V(s')=0 in GAE) only on natural termination;
        # time-limit truncation keeps the value bootstrap.
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
                "[step ", abs_step, "] mean_ret(10)=", trainer.mean_return(),
                " ep=", trainer.ep_count(),
                " elapsed=", elapsed, "s",
            )

        comptime if L.ENABLED:
            if (
                print_every > 0
                and abs_step % print_every == 0
                and Bool(logger)
            ):
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

        comptime if L.ENABLED:
            if (
                diag_every > 0
                and abs_step % diag_every == 0
                and Bool(logger)
            ):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        if (
            checkpoint_every > 0
            and abs_step % checkpoint_every == 0
            and checkpoint_path.byte_length() > 0
        ):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)

    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_onpolicy_discrete_eval — single-env greedy eval.
# ──────────────────────────────────────────────────────────────────────


def run_onpolicy_discrete_eval[
    A: OnPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Non-mutating greedy eval driver for discrete on-policy agents.

    Uses `select_greedy_action` (argmax logits). Does not touch the
    rollout buffer, optimizers, or episode tracker.
    """
    comptime OBS = A.AGENT_OBS_DIM
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var total_return = Scalar[DT](0.0)
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
                "  [eval ep ", ep + 1, "/", num_episodes,
                "] return=", ep_return, " steps=", ep_steps,
            )

    var mean = total_return / Scalar[DT](num_episodes)
    if verbose:
        print("eval: mean_return=", mean, " (", num_episodes, " episodes)")
    return mean


# ──────────────────────────────────────────────────────────────────────
# OnPolicyDiscreteAgentBatched — trait for the Tier-3 BatchedEnv driver.
# ──────────────────────────────────────────────────────────────────────


trait OnPolicyDiscreteAgentBatched(OnPolicyDiscreteAgent, OnPolicyBatchedCore):
    """N_ENVS-wide pointer surface for the batched discrete on-policy
    driver — the single-env `OnPolicyDiscreteAgent` plus the shared
    `OnPolicyBatchedCore` (`AGENT_N_ENVS` / `select_action_batched` /
    `record_batch_cpu` / `mark_terminal_env`). Both parents inherit the
    ONE `OnPolicyCheckpointable` root, so the diamond's shared members
    keep a single declaration and resolve cleanly.

    All pointer args are HOST-side. For GPU envs the driver D2Hs env-side
    obs/reward/done into host scratches before calling. The action slot is
    a single discrete index per env stored as a float (ACT ≡ 1)."""

    pass


# ──────────────────────────────────────────────────────────────────────
# run_onpolicy_discrete_train_batched — Tier-3 BatchedEnv discrete driver.
# ──────────────────────────────────────────────────────────────────────


def run_onpolicy_discrete_train_batched[
    A: OnPolicyDiscreteAgentBatched,
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
    progress_label: String = "on-policy-disc",
    stop_at_mean_return: Optional[Scalar[DT]] = None,
    stop_min_episodes: Int = 0,
) raises -> List[Scalar[DT]]:
    """Discrete-action sibling of `run_onpolicy_train_batched`.

    Same-target only (`env_target == train_target`) × any N_ENVS through
    the `BatchedEnv` trait. The discrete action is a single index per env;
    the driver stages it through a host scratch of width N_ENVS (ACT ≡ 1)
    and copies it into `env.action_ptr()` (the discrete BatchedEnv wrapper
    reads each slot as the action index). Mirrors the continuous batched
    driver's stage/step/record/reset loop otherwise.
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime N_ENVS = A.AGENT_N_ENVS
    comptime ACT = 1  # discrete action index slot

    comptime assert (
        env_target == "cpu" or env_target == "gpu"
    ), "env_target must be 'cpu' or 'gpu'"
    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "train_target must be 'cpu' or 'gpu'"
    comptime assert env_target == train_target, (
        "run_onpolicy_discrete_train_batched: env_target must equal"
        " train_target. Cross-target → use run_onpolicy_discrete_train."
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert E.OBS_DIM == OBS, (
        "BatchedEnv OBS_DIM must match trainer AGENT_OBS_DIM"
    )
    comptime assert E.ACT_DIM == ACT, (
        "discrete BatchedEnv ACT_DIM must be 1 (single action index)"
    )
    comptime if env_target == "gpu":
        if not ctx:
            raise Error(
                "run_onpolicy_discrete_train_batched: ctx required when"
                " env_target is 'gpu'"
            )

    # ONE shared loop body with the continuous batched driver — the
    # discrete action slot is just ACT ≡ 1 (one index-as-float per env).
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
        stop_at_mean_return=stop_at_mean_return,
        stop_min_episodes=stop_min_episodes,
    )
