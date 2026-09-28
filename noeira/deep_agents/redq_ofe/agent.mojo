"""REDQOFEAgent — user-facing facade over `REDQOFETrainer`.

Mirrors `REDQAgent`'s shape with the OFE-specific extensions
(state_branch + action_branch + predictor are exposed as comptime
generics; the trainer composes them internally).

Two surfaces here:

  (a) `train_single(env, total_timesteps, ...)` — single-env CPU
      env loop. Drives `select_action`, `record`, `end_episode`,
      `train_step` in the canonical order with a warmup gate. Returns
      the per-episode return list (matches REDQ's
      `run_offpolicy_train` return value).

  (b) `eval(env, num_episodes, ...)` — greedy eval loop. Each
      episode runs to env-`done` or to `max_steps_per_episode`,
      accumulating return; returns the mean across episodes.

These keep the public API consistent with `REDQAgent.train_single` /
`eval` for swappable comparison runs. The full `OffPolicyAgent` trait
conformance + `run_offpolicy_train` driver wiring is a follow-up;
this facade goes directly to the underlying trainer surface so we
don't need it.

`save` / `load` / `select_action` / `select_greedy_action` /
`mean_return` / `ep_count` are passthrough to the trainer.

`target` is comptime-parametric for forward-compat — only `"cpu"` is
supported today (the underlying trainer is CPU-only); GPU will land
behind a `target == "gpu"` branch when the trainer gains GPU paths.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module

from noeira.core.env_traits import BoxContinuousActionEnv
from noeira.core.logger import Logger, NoOpLogger

from ..training.blocks import SampleBlock
from ...io.artifact_sink import ArtifactSink
from ..training.driver_offpolicy import (
    run_offpolicy_train, run_offpolicy_eval,
)

from .trainer import REDQOFETrainer
from .metrics import REDQOFEMetrics


struct REDQOFEAgent[
    target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
    SB: Module,
    AB: Module,
    PRED: Module,
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
](Movable & Deinitable):
    """Thin facade over `REDQOFETrainer`.

    Dimensions (OBS / ACT / BATCH) are derived from `SAMPLE`. The 5
    OFE networks (ACTOR, CRITIC, SB, AB, PRED) are passed through as
    comptime generics so users can swap them via Design F presets
    without touching the facade.

    `target` is reserved for future GPU support; only `"cpu"` is
    currently implemented (asserted at `make` time).
    """

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    var trainer: REDQOFETrainer[
        Self.target,
        Self.SAMPLE,
        Self.ACTOR,
        Self.CRITIC,
        Self.SB,
        Self.AB,
        Self.PRED,
        Self.N,
        Self.N_MIN,
        Self.UTD,
        Self.POLICY_DELAY,
        Self.Q_MODE,
    ]

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = 3e-4,
        critic_lr: Scalar[DT] = 3e-4,
        ofe_lr: Scalar[DT] = 3e-4,
        alpha_lr: Scalar[DT] = 3e-4,
        gamma: Scalar[DT] = 0.99,
        tau: Scalar[DT] = 0.005,
        action_scale: Scalar[DT] = 1.0,
        init_alpha: Scalar[DT] = 0.2,
        target_entropy: Scalar[DT] = -1.0,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = -1250.0,
    ) raises:
        comptime assert (
            Self.target == "cpu" or Self.target == "gpu"
        ), "REDQOFEAgent: target must be 'cpu' or 'gpu'"
        self.trainer = REDQOFETrainer[
            Self.target,
            Self.SAMPLE,
            Self.ACTOR,
            Self.CRITIC,
            Self.SB,
            Self.AB,
            Self.PRED,
            Self.N,
            Self.N_MIN,
            Self.UTD,
            Self.POLICY_DELAY,
            Self.Q_MODE,
        ].make(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            ofe_lr=ofe_lr,
            alpha_lr=alpha_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
        )

    # ──────────────────────────────────────────────────────────────────
    # Training entry point.
    # ──────────────────────────────────────────────────────────────────

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
        """Single-env CPU off-policy training via the canonical
        `run_offpolicy_train` driver. Trait conformance on
        `REDQOFETrainer` gates this entry — the driver calls
        `select_action_batched[1]`, `record`, `train_step`,
        `end_episode`, `mean_return`, `ep_count`, etc. under the
        hood, all of which `REDQOFETrainer` now implements.

        Returns a list of windowed-mean returns appended at each
        completed episode (same shape as `REDQAgent.train_single`)."""
        # Hoist ctx into a local so the Mojo aliasing checker doesn't
        # see `self.trainer` + `self.trainer.ctx` both passed into the
        # call site.
        var ctx_local = self.trainer.ctx
        return run_offpolicy_train[
            REDQOFETrainer[
                Self.target,
                Self.SAMPLE,
                Self.ACTOR,
                Self.CRITIC,
                Self.SB,
                Self.AB,
                Self.PRED,
                Self.N,
                Self.N_MIN,
                Self.UTD,
                Self.POLICY_DELAY,
                Self.Q_MODE,
            ],
            E,
            L,
        ](
            self.trainer,
            env,
            total_timesteps,
            ctx=ctx_local,
            print_every=print_every,
            verbose=verbose,
            logger=logger,
            diag_every=diag_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run_dir,
        )

    def train_single_manual[
        E: BoxContinuousActionEnv,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        print_every: Int = 1_000,
        verbose: Bool = True,
    ) raises -> List[Scalar[DT]]:
        """Legacy manual env-loop training path. Returns the *per-
        episode complete return* (not the windowed mean — that's
        what the driver-based `train_single` returns).

        Kept for callers that want fine-grained per-episode returns
        without going through the driver. Same final policy outcome
        modulo RNG ordering."""
        var ep_returns = List[Scalar[DT]]()

        # Dtype bridge — the env may use a different dtype than DT
        # (the framework's compute dtype). We maintain TWO action and
        # obs Lists per step: one in DT for the trainer, one in
        # `E.dtype` for the env. Mirrors the canonical driver pattern
        # in `training/driver_offpolicy.mojo:380-396`.
        _ = env.reset()
        var env_obs = env.get_obs_list()
        var obs_list = List[Scalar[DT]](
            length=Self.OBS_DIM, fill=Scalar[DT](0.0),
        )
        var action_list = List[Scalar[DT]](
            length=Self.ACT_DIM, fill=Scalar[DT](0.0),
        )
        var env_action = List[Scalar[E.dtype]](
            length=Self.ACT_DIM, fill=Scalar[E.dtype](0.0),
        )
        var next_obs_list = List[Scalar[DT]](
            length=Self.OBS_DIM, fill=Scalar[DT](0.0),
        )
        for d in range(Self.OBS_DIM):
            obs_list[d] = Scalar[DT](env_obs[d])

        var current_return: Scalar[DT] = Scalar[DT](0.0)
        var warmup = self.trainer.learning_starts

        for step in range(total_timesteps):
            self.trainer.select_action(obs_list, action_list, step)
            for j in range(Self.ACT_DIM):
                env_action[j] = Scalar[E.dtype](action_list[j])

            var result = env.step_continuous_vec[E.dtype](env_action)
            var nxt = result[0].copy()
            var reward = Scalar[DT](result[1])
            var done = result[2]
            for d in range(Self.OBS_DIM):
                next_obs_list[d] = Scalar[DT](nxt[d])
            var done_f = (
                Scalar[DT](1.0) if done else Scalar[DT](0.0)
            )
            self.trainer.record(
                obs_list, action_list, reward, next_obs_list, done_f,
            )
            current_return += reward
            for d in range(Self.OBS_DIM):
                obs_list[d] = next_obs_list[d]

            if done:
                ep_returns.append(current_return)
                current_return = Scalar[DT](0.0)
                self.trainer.end_episode()
                _ = env.reset()
                env_obs = env.get_obs_list()
                for d in range(Self.OBS_DIM):
                    obs_list[d] = Scalar[DT](env_obs[d])
            if step >= warmup:
                _ = self.trainer.train_step(step)
            if verbose and step > 0 and step % print_every == 0:
                print(
                    "  step", step,
                    " mean_ret(window)=", self.trainer.mean_return(),
                    " ep_count=", self.trainer.ep_count(),
                    " train_steps=", self.trainer.total_train_steps(),
                )

        return ep_returns^

    # ──────────────────────────────────────────────────────────────────
    # Evaluation.
    # ──────────────────────────────────────────────────────────────────

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
        """Greedy eval over `num_episodes` episodes. Returns mean
        return."""
        var total: Scalar[DT] = Scalar[DT](0.0)
        var action_list = List[Scalar[DT]](
            length=Self.ACT_DIM, fill=Scalar[DT](0.0),
        )
        var env_action = List[Scalar[E.dtype]](
            length=Self.ACT_DIM, fill=Scalar[E.dtype](0.0),
        )
        var obs_list = List[Scalar[DT]](
            length=Self.OBS_DIM, fill=Scalar[DT](0.0),
        )

        for ep in range(num_episodes):
            _ = env.reset()
            var env_obs = env.get_obs_list()
            for d in range(Self.OBS_DIM):
                obs_list[d] = Scalar[DT](env_obs[d])
            var ep_return: Scalar[DT] = Scalar[DT](0.0)
            for _ in range(max_steps_per_episode):
                self.trainer.select_greedy_action(obs_list, action_list)
                for j in range(Self.ACT_DIM):
                    env_action[j] = Scalar[E.dtype](action_list[j])
                var result = env.step_continuous_vec[E.dtype](env_action)
                ep_return += Scalar[DT](result[1])
                var nxt = result[0].copy()
                for d in range(Self.OBS_DIM):
                    obs_list[d] = Scalar[DT](nxt[d])
                if result[2]:
                    break
            if verbose:
                print("  eval ep", ep, " return =", ep_return)
            total += ep_return

        return total / Scalar[DT](num_episodes)

    # ──────────────────────────────────────────────────────────────────
    # Passthroughs (single-step inference / accessors / checkpoint).
    # ──────────────────────────────────────────────────────────────────

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

    def mean_return(self) -> Scalar[DT]:
        return self.trainer.mean_return()

    def ep_count(self) -> Int:
        return self.trainer.ep_count()

    def total_train_steps(self) -> Int:
        return self.trainer.total_train_steps()

    def flush_metrics(mut self) -> REDQOFEMetrics:
        """Drain trainer accumulators into a snapshot + reset.
        Returns means of (critic_loss, actor_loss, α, log_prob_mean,
        aux_loss) plus update counts. See `REDQOFEMetrics`."""
        return self.trainer.flush_metrics()

    def save(mut self, path: String) raises:
        """One v3 `storage-ckpt` file over actor + N critics + SB + AB +
        PRED, with α's optimizer and the step counter as `K` scalars. The
        network optimizers are not saved."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Inverse of `save`. Target critics are hard-copied from
        their online twins after the online params are restored."""
        self.trainer.load_state(path)
