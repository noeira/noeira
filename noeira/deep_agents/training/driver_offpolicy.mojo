"""Off-policy training + eval drivers — Tier-1 + Tier-3.

Two training driver functions covering all useful (env_target,
train_target, N_ENVS) combinations:

  env_target | train_target | N_ENVS | driver
  -----------|--------------|--------|----------------------------------
  cpu        | cpu          | >=1    | run_offpolicy_train_batched
  gpu        | gpu          | >=1    | run_offpolicy_train_batched
  cpu        | gpu          | 1      | run_offpolicy_train
  cpu        | gpu          | >=1    | run_offpolicy_train_cpu_env_gpu_agent

Plus one eval driver `run_offpolicy_eval` that replaces the
legacy `run_offpolicy_eval_cpu` / `run_offpolicy_eval_gpu` split — the
trainer dispatches CPU vs GPU internally inside
`select_greedy_action`.

The (env=gpu, train=cpu) combination is omitted as degenerate
(D2H every obs back to CPU for training — never useful in practice).
Batched cross-target (cpu env, gpu train, N>=1) — the GPU-agent /
CPU-env hybrid, ported from legacy `gpu_agent_cpu_env_train.mojo`
(Phase 6.1) — is covered by `run_offpolicy_train_cpu_env_gpu_agent`,
which wraps the same `BatchedCpuEnv` adapter and inserts per-step H2D
obs / D2H action / H2D transition-slab boundary copies around the GPU
trainer's `select_action_batched` + `record_batch_gpu`. Lets a GPU
SAC/TD3/DDPG agent train against any CPU-stepped env (e.g. a Python
Gymnasium MuJoCo env) so training failures can be attributed to the
env vs the algorithm.

Trait surface
  - `OffPolicyAgent` — minimal: select_action_batched[N_ENVS],
    record, train_step, episode tracker accessors, batched
    CPU record + add_complete_return.
  - `OffPolicyAgentGpu(OffPolicyAgent)` — adds
    record_batch_gpu / record_batch_gpu_nstep for the gpu-env path.

Storage: all driver-owned buffers live in `DriverScratch[NAME, N, DIM]`
which unifies host `List` and device `DeviceBuffer` backing behind one
type. Env adapters (`BatchedCpuEnv`, `BatchedGpuEnv` in `batched_env.mojo`)
own their obs/action/reward/done buffers and expose pointer accessors
through the `BatchedEnv` trait.
"""

from std.time import perf_counter_ns
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
from noeira.utils.progress import IntervalProgress
from std.os import getenv

from noeira.cuda import (
    CUDAGraph,
    GraphSlot,
    maybe_capture_replay,
    maybe_record_replay,
)
from ..data.n_step_replay import GPUNStepBuffer
from noeira.core.env_traits import BoxContinuousActionEnv, RenderableEnv
from .batched_env import BatchedEnv
from .driver_scratch import DriverScratch
from .episode_tracker import EpisodeTracker
from .blocks.episode_readback import (
    EpisodeReturnRing,
    accumulate_episode_returns,
)
from .blocks.cadence import DriverCadence
from .checkpoint import announce_checkpoint
from ...io.artifact_sink import ArtifactSink


# ──────────────────────────────────────────────────────────────────────
# OffPolicyAgent — trait for the off-policy drivers.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyAgent(Deinitable, Movable):
    """Single-trait surface for the off-policy drivers.
    Exposes `AGENT_TRAIN_TARGET` (so the driver can comptime-gate
    H2D/D2H around the env step) and routes all action selection
    through one `select_action_batched[N_ENVS]` entry instead of the
    historic three `select_action[/_gpu/_gpu_batched]` variants.

    `record` keeps a host-`List` signature for single-env use (env step
    returns Lists). Batched record paths live on the `OffPolicyAgentGpu`
    sub-trait (or, for the CPU env batched path, in `record_batch_cpu`
    here). SAC / MBPO / DDPG / TD3 all conform."""

    # Trait-visible alias of the trainer's struct-comptime `train_target`.
    # SACTrainer exposes this via `AGENT_TRAIN_TARGET = Self.train_target`.
    # Conceptually distinct from the env's `ENV_TARGET` — see the
    # module docstring for the dual-target model.
    comptime AGENT_TRAIN_TARGET: StaticString
    # `AGENT_` prefix avoids clashing with the struct's own
    # OBS_DIM/ACT_DIM comptime params on conforming trainers.
    comptime AGENT_OBS_DIM: Int
    comptime AGENT_ACT_DIM: Int

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        alp_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM + 1), MutAnyOrigin
        ],
        step_idx: Int,
    ) raises:
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic, exploration-free action selection for eval.
        Host-list signature; trainers dispatch internally on
        `AGENT_TRAIN_TARGET` (CPU trainers run native, GPU trainers
        H2D the obs and D2H the action under the hood). Used by
        `run_offpolicy_eval`."""
        ...

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
    ) raises:
        """Batched deterministic (greedy) action selection for GPU-parallel
        eval (`run_offpolicy_eval_batched`). View contract is target-side
        (host for CPU trainers, device for GPU); `ao_scratch` is a
        N_ENVS*2*ACT actor-output scratch.

        DEFAULT = a correct-but-serial per-env fallback that loops
        `select_greedy_action` (which handles its own H2D/D2H at N=1), so
        every off-policy trainer conforms with no extra code. SACTrainer
        OVERRIDES this with a single batched actor forward + greedy kernel."""
        comptime OBS = Self.AGENT_OBS_DIM
        comptime ACT = Self.AGENT_ACT_DIM
        _ = ao_scratch  # the per-env fallback uses select_greedy_action's
        #                  own N=1 scratch; no batched scratch needed.
        var obs_h = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_h = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
        comptime if Self.AGENT_TRAIN_TARGET == "cpu":
            for e in range(N_ENVS):
                for d in range(OBS):
                    obs_h[d] = rebind[Scalar[DT]](obs[e, d])
                self.select_greedy_action(obs_h, act_h)
                for j in range(ACT):
                    action[e, j] = act_h[j]
        else:
            # Serial per-env D2H/H2D: build owning=False device sub-views of
            # row `e` from the LayoutTensor base ptr (`.ptr`).
            var c = ctx.value()
            for e in range(N_ENVS):
                var obs_view = DeviceBuffer[DT](
                    c,
                    obs.ptr.unsafe_offset(e * OBS),
                    OBS,
                    owning=False,
                )
                c.enqueue_copy(obs_h.unsafe_ptr(), obs_view)
                c.synchronize()
                self.select_greedy_action(obs_h, act_h)
                var act_view = DeviceBuffer[DT](
                    c,
                    action.ptr.unsafe_offset(e * ACT),
                    ACT,
                    owning=False,
                )
                c.enqueue_copy(act_view, act_h.unsafe_ptr())
            c.synchronize()

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        ...

    # ─── Episode-tracker plumbing (S6: trait defaults) ───────────────
    #
    # Every off-policy trainer owns a `tracker: EpisodeTracker` and four
    # one-line delegators into it (`end_episode` / `mean_return` /
    # `ep_count` / `add_complete_return`). Those four were byte-identical
    # across SAC/DDPG/TD3/REDQ/REDQ-OFE/MBPO. They are now trait DEFAULTS
    # expressed against ONE required accessor, `_tracker_ptr`, so a
    # conformer supplies only the field handle (one method) instead of
    # re-typing four delegators. The pointer is laundered to `MutAnyOrigin`
    # at the accessor (the codebase's standard field-handle idiom) so the
    # read-receiver defaults (`mean_return`/`ep_count`) and the mut-receiver
    # ones (`end_episode`/`add_complete_return`) can share it.

    def _tracker_ptr(self) -> Pointer[EpisodeTracker, MutAnyOrigin]:
        """Handle to the conformer's `EpisodeTracker` field. The ONE
        required member behind the four delegator defaults below."""
        ...

    def end_episode(mut self):
        self._tracker_ptr()[].end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def total_train_steps(self) -> Int:
        """Cumulative gradient-update count (UTD-aware), monotonic, not
        reset by flushes. The single-env driver keys the `diag_every`
        cadence on THIS rather than env-steps, so one diag point spans
        `diag_every` updates regardless of UTD. (At UTD=40 an env-step
        cadence makes each point span 40×diag_every updates — the
        Q/critic curves collapse to a handful of points.) Default 0 means
        a conformer doesn't track it; every real trainer overrides."""
        return 0

    def mean_return(self) -> Scalar[DT]:
        return self._tracker_ptr()[].mean_return()

    def ep_count(self) -> Int:
        return self._tracker_ptr()[].ep_count

    # ─── Tier-2 additions — batched CPU env support ──────────────────
    #
    # Drivers that step many envs per iteration push complete-episode
    # returns through `add_complete_return` (instead of the single-env
    # `end_episode`/`record` flow which auto-updates the tracker's
    # `current_return`). `record_batch_cpu` pushes N transitions into
    # the trainer's replay without touching the tracker so the driver
    # can manage per-env return accumulators on the host.
    #
    # `add_complete_return` was previously declared on
    # `OffPolicyAgentGpu`; lifted to the parent so both the
    # GPU-env driver (Phase 3.5) and the new CPU-env batched driver
    # (Tier-2) share one source of truth — and so the Gpu sub-trait
    # doesn't re-declare it (which would create diamond ambiguity).

    def add_complete_return(mut self, ret: Scalar[DT]):
        self._tracker_ptr()[].add_complete_return(ret)

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        """Push ONE host-list transition into the trainer's replay. The
        per-lane hook behind the defaulted `record_batch_cpu` staging loop
        below — the ONE genuinely trainer-specific line (SAC/DDPG/TD3/REDQ/
        REDQ-OFE forward to `sample_blk.add`; MBPO to its DualSampleStep's
        `real_add`, which the uniform default can't name since
        DualSampleStep isn't a `SampleBlock`)."""
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
        """Push N transitions from host pointer slabs into the trainer's
        replay buffer. Does NOT update the trainer's episode tracker —
        caller manages per-env returns via `add_complete_return`.

        DEFAULT (S6 follow-on): stage each lane into reused per-lane Lists
        and push via the `_replay_add` hook. This staging loop was
        byte-identical across all 6 off-policy trainers; it now lives here
        once. Only ever instantiated on the same-target cpu-env batched
        driver (cpu-env ⇒ cpu-train), so `AGENT_TRAIN_TARGET == "cpu"`."""
        comptime assert (
            Self.AGENT_TRAIN_TARGET == "cpu"
        ), "record_batch_cpu: trainer's train_target must be 'cpu'"
        comptime OBS = Self.AGENT_OBS_DIM
        comptime ACT = Self.AGENT_ACT_DIM
        # Reused per-lane Lists for the `_replay_add` hook (which takes
        # List args). Re-using across lanes avoids per-lane re-allocation.
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[unsafe_offset=env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[unsafe_offset=env_idx * OBS + d]
            for j in range(ACT):
                act_lane[j] = action_ptr[unsafe_offset=env_idx * ACT + j]
            self._replay_add(
                obs_lane,
                act_lane,
                reward_ptr[unsafe_offset=env_idx],
                nxt_lane,
                done_ptr[unsafe_offset=env_idx],
            )

    # ─── Optional cadence hooks (default no-op) ──────────────────────
    #
    # These let the driver call into the trainer at the user's
    # `diag_every` / `checkpoint_every` cadences without splitting the
    # env-loop into chunks. Each has a `pass` default so existing
    # trainers (DDPG/TD3/PPO/MBPO/DQN) don't need to do anything until
    # their own metric/checkpoint paths are wired up. SACTrainer
    # overrides both with real bodies that call its `flush_metrics` and
    # a one-file v2 `save_state` writer.
    #
    # `flush_metrics_through_logger[L]` discards the typed bundle so
    # the trait can be uniform across algorithms whose `flush_metrics`
    # return different `*Metrics` structs.

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


# ──────────────────────────────────────────────────────────────────────
# OffPolicyAgentGpu — adds GPU-batched methods on top.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyAgentGpu(OffPolicyAgent):
    """Extends `OffPolicyAgent` with the GPU-batched record
    surfaces needed by the GPU-env driver. `add_complete_return`
    is inherited from the parent — single source of truth.

    CPU-only trainers (e.g. MBPOTrainer) conform with raising stubs for
    `record_batch_gpu` / `record_batch_gpu_nstep`; the Tier-3 driver
    comptime-elides the GPU branch when env_target == "cpu" so the
    stubs are never invoked."""

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
        ...

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS,
            Self.AGENT_OBS_DIM,
            Self.AGENT_ACT_DIM,
            N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        ...

    # ─── CUDA-graph capture surface (Slice 7) ─────────────────────────
    #
    # Default bodies so agents that don't support train-step capture still
    # conform. Only invoked on the `USE_TRAIN_CUDA_GRAPH=True` driver path
    # (SAC GPU today), so the defaults are never reached for other agents.

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
        """Advance one logical update's host bookkeeping (counters/metric
        accumulators). Default no-op."""
        pass

    def learning_starts_count(self) -> Int:
        """Cumulative env-step threshold after which training begins — the
        driver gates the capture path on this. Default 0."""
        return 0


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train — single-env, env_target="cpu".
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train[
    A: OffPolicyAgent,
    E: BoxContinuousActionEnv,
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
    progress_label: String = "off-policy",
) raises -> List[Scalar[DT]]:
    """Single-env off-policy training driver bound on the CPU env trait
    (`BoxContinuousActionEnv`). Covers (env_target=cpu, train_target=cpu)
    and (env_target=cpu, train_target=gpu).

    `ctx` is required for `train_target=gpu` (used for H2D obs / D2H
    action staging) and ignored for `train_target=cpu`. Must be the
    SAME `DeviceContext` the trainer was built with — Apple Metal's
    queue pool exhausts if a new context is constructed per call.

    Loop semantics: one env step + one `train_step` per
    iteration. `select_action_batched` consumes RNG in the same order
    on CPU as the legacy single-env CPU path. The GPU branch differs
    in warmup RNG (Philox kernel vs host `random_float64`).
    """
    # ENV_TARGET is implicit "cpu" since E is bound on BoxContinuousActionEnv.
    # Made explicit here so the dual-axis model is visible at the dispatch site.
    comptime env_target: StaticString = "cpu"
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "run_offpolicy_train: train_target must be 'cpu' or 'gpu'"
    comptime if train_target == "gpu":
        if not ctx:
            raise Error(
                "run_offpolicy_train[train_target='gpu']:"
                " ctx required for env→trainer H2D/D2H staging"
            )

    # Driver-owned scratches. Allocated on train_target so the trainer's
    # select_action_batched consumes them natively. When env_target !=
    # train_target (cpu env + gpu trainer), obs + action also need host
    # mirrors for the per-step H2D/D2H around the env step.
    comptime needs_boundary_copy: Bool = env_target != train_target
    var obs_scratch = DriverScratch["obs", 1, OBS].make[train_target](
        ctx=ctx,
        with_host_mirror=needs_boundary_copy,
    )
    var action_scratch = DriverScratch["action", 1, ACT].make[train_target](
        ctx=ctx,
        with_host_mirror=needs_boundary_copy,
    )
    var ao = DriverScratch["ao", 1, 2 * ACT].make[train_target](ctx=ctx)
    var alp = DriverScratch["alp", 1, ACT + 1].make[train_target](ctx=ctx)

    # Host-side Lists for `trainer.record` + env stepping. (`record`'s
    # signature is the legacy host-List form — see trait docstring.)
    var obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action_list = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    var env_obs = env.reset_obs_list()
    var env_action = List[Scalar[E.dtype]](capacity=ACT)
    for _ in range(ACT):
        env_action.append(Scalar[E.dtype](0.0))

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    # In-place progress bar between log lines (pure CPU, no GPU sync).
    var prog = IntervalProgress(
        print_every, label=progress_label, enabled=verbose
    )
    # Diag cadence is keyed on TRAIN steps (gradient updates) via
    # `trainer.total_train_steps()`, not env-steps — see the trait method.
    # `last_diag_bucket` tracks the last emitted `train_steps // diag_every`.
    var last_diag_bucket: Int = 0
    while step < total_timesteps:
        # Copy env obs (E.dtype) into obs_list (DT) for record + into
        # the driver scratch (DT) for the select_action_batched call.
        # When train_target=="cpu", obs_scratch.host_ptr() IS the
        # scratch's only storage; when "gpu" it's the host mirror that
        # will be H2D'd below.
        var obs_scratch_h = obs_scratch.host_ptr()
        for d in range(OBS):
            var v = Scalar[DT](env_obs[d])
            obs_list[d] = v
            obs_scratch_h[unsafe_offset=d] = v

        # Boundary copy: env_target != train_target requires H2D obs.
        # Elided when env_target == train_target.
        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(obs_scratch.dev.value(), obs_scratch_h)

        # `base_step + step` is the cumulative env-step counter — required
        # so the trainer's `learning_starts` warmup gate stays correct
        # across chunked calls (see `SACAgent.train_single` which loops
        # this driver in cadence-sized blocks). Equivalent to `step` when
        # `base_step=0` (single-call usage).
        trainer.select_action_batched[1](
            LayoutTensor[DT, Layout.row_major(1, OBS), MutAnyOrigin](
                obs_scratch.target_ptr[train_target]()
            ),
            LayoutTensor[DT, Layout.row_major(1, ACT), MutAnyOrigin](
                action_scratch.target_ptr[train_target]()
            ),
            LayoutTensor[DT, Layout.row_major(1, 2 * ACT), MutAnyOrigin](
                ao.target_ptr[train_target]()
            ),
            LayoutTensor[DT, Layout.row_major(1, ACT + 1), MutAnyOrigin](
                alp.target_ptr[train_target]()
            ),
            base_step + step,
        )

        # Boundary copy: env_target != train_target requires D2H action.
        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(
                action_scratch.host_ptr(), action_scratch.dev.value()
            )
            c.synchronize()

        var action_h = action_scratch.host_ptr()
        for j in range(ACT):
            var a = action_h[unsafe_offset=j]
            action_list[j] = a
            env_action[j] = Scalar[E.dtype](a)

        # Env step (CPU env).
        var step_res = env.step_continuous_vec[E.dtype](env_action)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        # `done` (terminated OR truncated) drives reset/episode tracking; the
        # replay buffer stores `terminated` ONLY (natural termination) so the
        # SAC TD bootstrap is kept on truncation but dropped on termination.
        var terminated = env.was_terminated()
        for d in range(OBS):
            next_obs_list[d] = Scalar[DT](nxt[d])

        trainer.record(
            obs_list,
            action_list,
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

        # Logger emit at the print cadence. Comptime-elided when
        # L=NoOpLogger (default).
        #
        # NOTE: no forced `.flush()` here — `log_scalar` auto-flushes
        # when the logger's buffer fills (CsvLogger / RemoteLogger
        # check `len(entries) >= buffer_size`). The user controls flush
        # cadence via `buffer_size`. Final residual entries are sent
        # by `logger.close()` at end of training. Synchronous flushes
        # are expensive (~50-200 ms per HTTP POST on a remote
        # endpoint); forcing one every print_every was the dominant
        # logger overhead in profiling.
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

        # `diag_every` — drain the trainer's metric bundle through the
        # logger every `diag_every` TRAIN steps (gradient updates), not
        # env-steps. At UTD>1 the env-step cadence makes each diag point
        # span UTD×diag_every updates (e.g. 200k) → the Q/critic/dyn curves
        # collapse to a few points. Keying on `total_train_steps()` keeps
        # one point per `diag_every` updates at any UTD. Default trait impl
        # of `flush_metrics_through_logger` is a no-op for trainers that
        # haven't wired metrics up yet.
        var diag_due = False
        if diag_every > 0:
            var bucket = trainer.total_train_steps() // diag_every
            if bucket > last_diag_bucket:
                last_diag_bucket = bucket
                diag_due = True

        # Log the diagnostic bundle (Q-values, critic/actor loss, alpha …)
        # against the TRAIN-step counter, not `abs_step` (env interactions).
        # These are per-gradient-update quantities, so at UTD>1 (e.g. REDQ
        # 40:1) the env-step x-axis under-counts by UTD×. `avg_reward` /
        # `episodes` above stay on `abs_step` — those are env-level.
        comptime if L.ENABLED:
            if diag_due and Bool(logger):
                trainer.flush_metrics_through_logger[L](
                    logger, trainer.total_train_steps()
                )

        # Live flush: push whatever was just buffered to the monitoring
        # server at the print/diag cadence so the dashboard updates DURING
        # training (otherwise points only auto-flush once the buffer hits
        # `buffer_size`, leaving long runs blind until `logger.close()`).
        # `flush()` early-returns when the buffer is empty, so this is a
        # no-op on the vast majority of steps.
        comptime if L.ENABLED:
            if Bool(logger) and (
                (print_every > 0 and abs_step % print_every == 0) or diag_due
            ):
                logger.value()[].flush()

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
# run_offpolicy_eval_batched — GPU-parallel deterministic eval.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_eval_batched[
    A: OffPolicyAgentGpu,
    EE: BatchedEnv,
    EVAL_ENVS: Int,
](
    ctx: Optional[DeviceContext],
    mut trainer: A,
    mut eval_env: EE,
    num_episodes: Int,
    *,
    max_steps: Int = 1_000,
    rng_seed: UInt64 = UInt64(123_457),
) raises -> Scalar[DT]:
    """Non-mutating GPU-parallel greedy eval. Drives `eval_env` (an ISOLATED
    BatchedEnv — never the training env) with deterministic actions from
    `trainer.select_greedy_action_batched` (the actor MEAN → tanh → scale, no
    sampling), accumulating per-env returns and auto-resetting on done. Touches
    NO replay buffer / optimizer / episode tracker — it only reads actor weights.

    Runs up to `max_steps` parallel steps, collecting every completed episode,
    and returns the mean over completed episodes (0.0 if none finished). With
    EVAL_ENVS parallel envs, ~EVAL_ENVS full-length episodes complete per
    `max_steps` window, so pass `num_episodes <= EVAL_ENVS` to be satisfied in
    one window.

    `EVAL_ENVS` must equal the eval env's struct `N_ENVS`. It is independent of
    the training driver's `N_ENVS`, so a smaller eval batch can cap VRAM.
    Same-target only (eval env's `ENV_TARGET` == trainer's `AGENT_TRAIN_TARGET`).
    """
    comptime ACT = A.AGENT_ACT_DIM
    comptime OBS = A.AGENT_OBS_DIM
    comptime target = A.AGENT_TRAIN_TARGET

    # Actor-output (mean|log_std) scratch on the train target.
    var ao = DriverScratch["eval_ao", EVAL_ENVS, 2 * ACT].make[target](ctx=ctx)

    eval_env.reset_batch[EVAL_ENVS](ctx=ctx, rng_seed=rng_seed)

    var per_env = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
    var returns = List[Scalar[DT]]()
    # Persistent host staging for the per-step reward/done D2H (GPU env only).
    var rew_h = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
    var done_h = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))

    var step = 0
    while len(returns) < num_episodes and step < max_steps:
        # Greedy action → eval_env.action_ptr() (target-side pointer).
        trainer.select_greedy_action_batched[EVAL_ENVS](
            ctx,
            LayoutTensor[DT, Layout.row_major(EVAL_ENVS, OBS), MutAnyOrigin](
                eval_env.obs_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(EVAL_ENVS, ACT), MutAnyOrigin](
                eval_env.action_ptr()
            ),
            LayoutTensor[
                DT, Layout.row_major(EVAL_ENVS, 2 * ACT), MutAnyOrigin
            ](ao.target_ptr[target]()),
        )
        eval_env.step_batch[EVAL_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(step + 1),
        )

        # Read reward + done host-side, accumulate, record on episode end.
        comptime if EE.ENV_TARGET == "cpu":
            var rp = eval_env.reward_ptr()
            var dp = eval_env.done_ptr()
            for e in range(EVAL_ENVS):
                per_env[e] = per_env[e] + rp[unsafe_offset=e]
                if dp[unsafe_offset=e] > Scalar[DT](0.5):
                    returns.append(per_env[e])
                    per_env[e] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            var rew_view = DeviceBuffer[DT](
                c,
                eval_env.reward_ptr(),
                EVAL_ENVS,
                owning=False,
            )
            var done_view = DeviceBuffer[DT](
                c,
                eval_env.done_ptr(),
                EVAL_ENVS,
                owning=False,
            )
            c.enqueue_copy(rew_h.unsafe_ptr(), rew_view)
            c.enqueue_copy(done_h.unsafe_ptr(), done_view)
            c.synchronize()
            for e in range(EVAL_ENVS):
                per_env[e] = per_env[e] + rew_h[e]
                if done_h[e] > Scalar[DT](0.5):
                    returns.append(per_env[e])
                    per_env[e] = Scalar[DT](0.0)

        eval_env.selective_reset_batch[EVAL_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(step + 1) * UInt64(7),
        )
        step += 1

    if len(returns) == 0:
        return Scalar[DT](0.0)
    var tot = Scalar[DT](0.0)
    for i in range(len(returns)):
        tot = tot + returns[i]
    return tot / Scalar[DT](len(returns))


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train_batched — Tier-3: ONE driver for all
#   (env_target, train_target, N_ENVS) combos via BatchedEnv.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train_batched[
    A: OffPolicyAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int = 1,
    NS: Int = 1,
    L: Logger = NoOpLogger,
    USE_TRAIN_CUDA_GRAPH: Bool = False,
    USE_ENV_CUDA_GRAPH: Bool = False,
    EE: BatchedEnv = E,
    EVAL_ENVS: Int = N_ENVS,
](
    ctx: Optional[DeviceContext],
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
    episode_sync_every: Int = 1,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    artifacts: Optional[ArtifactSink] = None,
    run_dir: String = "",
    eval_env: Optional[Pointer[EE, MutAnyOrigin]] = None,
    eval_every: Int = 0,
    eval_episodes: Int = 16,
    eval_max_steps: Int = 1_000,
    progress_label: String = "off-policy",
) raises -> List[Scalar[DT]]:
    """Tier-3 off-policy driver covering same-target combinations.

    `checkpoint_every` (env steps, default 0 = off) + `checkpoint_path`:
    when both set, the driver calls `trainer.save_state(checkpoint_path)`
    inline every `checkpoint_every` env-steps and one final time at the
    end of the loop — overwriting `checkpoint_path` with the one-file
    v3 `storage-ckpt` file (for SAC: actor + twin critics, with α's
    optimizer and the step counter as `K` scalars; the network optimizers,
    replay buffer and episode tracker are NOT persisted). On the GPU train target the save does a D2H of the live
    params; it runs in host code between iterations, so it is safe to
    combine with `USE_TRAIN_CUDA_GRAPH` / `USE_ENV_CUDA_GRAPH` (the
    captured graphs are per-step, not the whole loop). Default trait
    `save_state` raises, so a trainer without checkpointing fails loudly.

    `episode_sync_every` (GPU-env path only): batch the per-iteration
    reward/done D2H readback used for episode-return bookkeeping over this many
    iterations, draining (one `synchronize`) only when the ring fills OR a
    print/diag boundary is reached OR the loop ends. The default `1` reproduces
    the original per-iteration sync exactly; higher values trade logging
    granularity for far fewer host↔device stalls — important once the train
    step is a captured CUDA graph (the sync would otherwise serialize every
    iteration and negate the capture win). Returns are still drained in order,
    so `mean_return` / `ep_count` are exact at every emit boundary.

    Same-target means `env_target == train_target`. Routed through the
    `BatchedEnv` trait:

      env_target | train_target | N_ENVS | covered
      -----------|--------------|--------|--------
      cpu        | cpu          | >=1    | yes  (via BatchedCpuEnv)
      gpu        | gpu          | >=1    | yes  (via BatchedGpuEnv)

    Cross-target combinations are NOT covered here:
      - (cpu env, gpu train) reachable via `run_offpolicy_train`
        (Tier-1 Phase 3) at N_ENVS=1, or — for N_ENVS>=1 batched —
        via `run_offpolicy_train_cpu_env_gpu_agent` (Phase 6.1), which
        adds the H2D prev_obs/action/reward/obs/terminated boundary
        plumbing before `record_batch_gpu`.
      - (gpu env, cpu train) rejected as degenerate (D2H every obs).

    Bounded on `OffPolicyAgentGpu` because the gpu-env branch
    needs `record_batch_gpu`; the cpu-env branch uses `record_batch_cpu`
    inherited from the parent. The driver comptime-branches on
    `(env_target, N_ENVS)` so each combination compiles only the
    kernels it actually needs.

    `ctx` is required for `env_target == "gpu"`; pass `None` for the
    pure CPU case.

    Loop:
      1. snapshot env.obs_ptr()           → prev_obs (driver-owned)
      2. trainer.select_action_batched[N_ENVS] → env.action_ptr() directly
      3. env.step_batch[N_ENVS]           → env.obs / .reward / .done
      4. trainer.record_batch_cpu OR record_batch_gpu (env-side ptrs)
      5. per-env return accumulation + add_complete_return on done
      6. env.selective_reset_batch[N_ENVS]
      7. updates_per_step × trainer.train_step
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    comptime assert (
        env_target == "cpu" or env_target == "gpu"
    ), "env_target must be 'cpu' or 'gpu'"
    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "train_target must be 'cpu' or 'gpu'"
    comptime assert env_target == train_target, (
        "run_offpolicy_train_batched: env_target must equal train_target."
        " Cross-target combinations: (cpu env, gpu train) → use"
        " run_offpolicy_train (Tier-1, single-env); (gpu env,"
        " cpu train) → rejected as degenerate."
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert NS > 0, "NS must be > 0"
    comptime if USE_TRAIN_CUDA_GRAPH:
        comptime assert train_target == "gpu", (
            "USE_TRAIN_CUDA_GRAPH requires train_target == 'gpu' (CUDA-graph"
            " capture is a GPU-only path; no-op on non-NVIDIA)"
        )
    comptime if USE_ENV_CUDA_GRAPH:
        comptime assert env_target == "gpu", (
            "USE_ENV_CUDA_GRAPH requires env_target == 'gpu' (the captured env"
            " step is GPU physics kernels; no-op on non-NVIDIA). Only valid for"
            " envs whose GPU step is RNG-free / deterministic (physics3d:"
            " RNG lives in reset, which stays eager) — a stochastic-step env"
            " would bake its step seed and must wire a device RNG counter"
            " first."
        )
    comptime assert (
        E.OBS_DIM == OBS and E.ACT_DIM == ACT
    ), "BatchedEnv dimensions must match trainer dimensions"
    comptime if NS > 1:
        comptime assert env_target == "gpu", (
            "run_offpolicy_train_batched[NS>1]: n-step only supported"
            " on GPU env path (GPUNStepBuffer is GPU-only)"
        )
    comptime if env_target == "gpu":
        if not ctx:
            raise Error(
                "run_offpolicy_train_batched: ctx required when"
                " env_target is 'gpu'"
            )

    # n-step buffer (NS > 1 only). Declared as Optional at function
    # level because Mojo nightly's `comptime if` does not bleed
    # bindings to sibling blocks — Optional+`if ctx:` is the bridge.
    # For CPU env (ctx=None), stays None and is never touched
    # (`comptime assert NS == 1` for the CPU path is enforced above).
    var nstep_buf: Optional[
        GPUNStepBuffer[NS, A.AGENT_OBS_DIM, A.AGENT_ACT_DIM, N_ENVS]
    ] = None
    if ctx:
        nstep_buf = Optional(
            GPUNStepBuffer[
                NS,
                A.AGENT_OBS_DIM,
                A.AGENT_ACT_DIM,
                N_ENVS,
            ].new(ctx.value(), gamma=nstep_gamma)
        )

    # Slice 7 — lazily-captured train-step graph (None until first capture
    # past warmup). Declared at function level (Mojo `comptime if` bindings
    # don't bleed to sibling blocks; the slot is read/written inside the
    # loop's capture branch). Unused when USE_TRAIN_CUDA_GRAPH is False, and
    # a no-op on non-NVIDIA where `CUDAGraph` itself is a no-op.
    var train_graph: Optional[CUDAGraph] = None
    # The DeviceGraph slot, used instead of `train_graph` when
    # NOEIRA_GRAPH_BACKEND=device. Both are declared unconditionally because
    # `comptime if` does not bleed bindings to sibling blocks; the unused one
    # costs a null pointer and a Bool.
    var train_slot = GraphSlot()
    # ⚠ READ ONCE, OUTSIDE THE LOOP. `getenv` per iteration would be a syscall
    # on the hot path, and a backend that could change mid-run would leave two
    # half-populated graphs behind.
    #
    # Defaults to "stream" — the LD_PRELOAD shim — so this commit changes NO
    # behaviour. Flip it to "device" to record on MAX's own `DeviceGraph`
    # (no interceptor, no preload); see `noeira/cuda/device_graph.mojo`.
    #
    # ⚠ "device" REQUIRES the agent to override `train_device_kernels_on`, and
    # the trait default RAISES rather than silently recording a partial step.
    # SAC is migrated; others are not yet.
    var use_device_graph = getenv("NOEIRA_GRAPH_BACKEND", "stream") == "device"
    # Env-step graph (env capture). None until first capture; the deterministic
    # physics `step_batch` is captured once and replayed per iteration when
    # `USE_ENV_CUDA_GRAPH` — collapsing the env's per-step eager kernel launches
    # (newton / integrators / collision) into one launch. No-op otherwise.
    var env_graph: Optional[CUDAGraph] = None
    # Env-reset graph (env capture). Sibling of `env_graph` for the selective
    # reset: [increment device RNG counter → selective_reset_kernel →
    # extract_obs]. Captured once and replayed thereafter. The reset randomness
    # is driven by the env's DEVICE-resident RNG counter (bumped inside the
    # captured sequence), so each replay resets done envs to FRESH random
    # states without host intervention — the recapture-free pattern legacy uses
    # (cf. gpu_offpolicy_train.mojo selective_reset in-graph). `record` stays
    # eager between the step and reset graphs (its replay write index is a host
    # scalar — NOT capture-safe), so we capture step and reset as two separate
    # graphs rather than one. No-op otherwise.
    var env_reset_graph: Optional[CUDAGraph] = None

    # All scratches on the single target (env_target == train_target).
    var ao = DriverScratch["ao", N_ENVS, 2 * ACT].make[train_target](ctx=ctx)
    var alp = DriverScratch["alp", N_ENVS, ACT + 1].make[train_target](ctx=ctx)
    var prev_obs = DriverScratch["prev_obs", N_ENVS, OBS].make[env_target](
        ctx=ctx
    )

    var per_env_returns = List[Scalar[DT]](
        length=N_ENVS,
        fill=Scalar[DT](0.0),
    )

    # Deferred episode-tracking readback (GPU env only) — see
    # `EpisodeReturnRing`. Declared Optional at function level because Mojo
    # nightly's `comptime if` bindings don't bleed to sibling blocks;
    # populated only for the GPU-env branch. `episode_sync_every == 1`
    # reproduces the original per-iteration sync exactly.
    var ep_ring: Optional[EpisodeReturnRing[N_ENVS]] = None
    comptime if env_target == "gpu":
        if ctx:
            ep_ring = Optional(
                EpisodeReturnRing[N_ENVS].make(ctx.value(), episode_sync_every)
            )

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    # Shared threshold-counter cadence state (see blocks/cadence.mojo):
    # print/log at `print_every` (log counter only advances inside
    # `comptime if L.ENABLED`), diag bundle at `diag_every`, checkpoint at
    # `checkpoint_every` (needs a non-empty path), deterministic eval at
    # `eval_every` (needs an isolated `eval_env`; first eval fires at
    # `eval_every`, not at step 0 where the policy is random).
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
        # ── 1. Snapshot prev_obs from env.obs_ptr().
        comptime if env_target == "cpu":
            var po_p = prev_obs.host_ptr()
            var ob_p = env.obs_ptr()
            for k in range(N_ENVS * OBS):
                po_p[unsafe_offset=k] = ob_p[unsafe_offset=k]
        else:
            # GPU env: D→D enqueue_copy. Reconstruct DeviceBuffer view
            # over env.obs_ptr() (owning=False — env still owns memory).
            var c = ctx.value()
            var env_obs_view = DeviceBuffer[DT](
                c,
                env.obs_ptr(),
                N_ENVS * OBS,
                owning=False,
            )
            c.enqueue_copy(prev_obs.dev.value(), env_obs_view)

        # ── 2. Trainer writes action directly into env.action_ptr().
        # env_target == train_target so the pointer is on the right side.
        # `base_step + step_idx` is the cumulative env-step counter (see
        # the `base_step` note in `run_offpolicy_train`).
        trainer.select_action_batched[N_ENVS](
            LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
                env.obs_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](
                env.action_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, 2 * ACT), MutAnyOrigin](
                ao.target_ptr[train_target]()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin](
                alp.target_ptr[train_target]()
            ),
            base_step + step_idx,
        )

        # ── 3. Env step (writes env-internal obs/reward/done).
        # ⚠ `and has_nvidia_gpu_accelerator()` added 2026-08-07. The
        # comptime asserts above have always CLAIMED this path is a
        # "no-op on non-NVIDIA" — it was not. On Apple/Metal the capture
        # branch ran and silently FROZE the env: measured ep_count 0 vs 6
        # over an identical 6000-step dm_control walker run with the flag
        # off (3 truncations x 2 lanes is the correct count). Every GPU
        # example in examples/ sets USE_ENV_CUDA_GRAPH=True, so on Apple
        # they were all training against a stopped environment.
        comptime if (
            USE_ENV_CUDA_GRAPH
            and env_target == "gpu"
            and has_nvidia_gpu_accelerator()
        ):
            # Capture the (deterministic) physics `step_batch` into a CUDA graph
            # and replay it once per iteration — collapses the env's dozens of
            # eager per-step kernel launches (newton solver, integrators,
            # collision) into a single launch, keeping the GPU fed at low
            # N_ENVS / high iteration counts (the walker2d N_ENVS=4 / 250k-iter
            # regime). SAFE: the env's GPU step is RNG-free (the `rng_seed` arg
            # is accepted but unused — RNG lives only in reset, which stays
            # eager), so the value baked at capture is never read. The captured
            # kernels read the LIVE action/state device buffers (the eager
            # `select_action` above wrote `action_ptr` this iteration), so each
            # replay advances physics from the current state in-place. First
            # capture happens on iteration 0 (warmup), so the inherent
            # settle+capture double-step only perturbs one warmup transition.
            # TEMP DIAGNOSTIC (rule out the `maybe_capture_replay` closure):
            # inline, legacy-style explicit CUDAGraph capture. Capture once on
            # the first iteration (settle run → begin → re-run → end), replay on
            # the mojo stream thereafter. The node-count print confirms capture
            # actually happened and how many kernels it grabbed (a tiny/zero
            # count means capture silently failed).
            var ce = ctx.value()
            if not env_graph:
                env.step_batch[N_ENVS](
                    ctx=ctx,
                    rng_seed=rng_seed + UInt64(iter_idx + 1),
                )
                ce.synchronize()
                var g = CUDAGraph(ce)
                # ⚠ CHECK is_disabled() BEFORE CAPTURING. `CUDAGraph`
                # disables itself when the interceptor is not hooked into the
                # process — which is the ORDINARY state of a binary you BUILD
                # and then run directly, because LD_PRELOAD comes from pixi's
                # nvidia activation and not from the binary. `begin_capture`
                # RAISES on a disabled graph, so without this guard the whole
                # run dies rather than falling back to running the step
                # directly. `maybe_capture_replay` has always done this; these
                # two env-graph sites did not, and a `mojo build` + direct run
                # killed the process on the first iteration.
                #
                # The settle run above has already executed this iteration's
                # work, so taking the disabled path here is a COMPLETED step,
                # not a skipped one. Storing the disabled graph makes later
                # iterations take the `else` branch instead of reconstructing
                # (and re-printing) every time.
                if not g.is_disabled():
                    g.begin_capture()
                    env.step_batch[N_ENVS](
                        ctx=ctx,
                        rng_seed=rng_seed + UInt64(iter_idx + 1),
                    )
                    g.end_capture()
                    if verbose:
                        print(
                            "[CUDA Graph] Captured ENV STEP with",
                            g.num_nodes(),
                            "nodes",
                        )
                env_graph = g^
            elif env_graph.value().is_disabled():
                env.step_batch[N_ENVS](
                    ctx=ctx,
                    rng_seed=rng_seed + UInt64(iter_idx + 1),
                )
            else:
                env_graph.value().replay_on_mojo_stream()
        else:
            env.step_batch[N_ENVS](
                ctx=ctx,
                rng_seed=rng_seed + UInt64(iter_idx + 1),
            )

        # ── 4. Replay push (env-target-specific).
        comptime if env_target == "cpu":
            # Replay stores `terminated` (natural termination only) so the TD
            # bootstrap is kept on truncation, dropped on termination. `done`
            # (terminated OR truncated) is still used below for episode
            # tracking and selective reset.
            trainer.record_batch_cpu[N_ENVS](
                prev_obs.host_ptr(),
                env.action_ptr(),
                env.reward_ptr(),
                env.obs_ptr(),
                env.terminated_ptr(),
            )
        else:
            # GPU env. Reconstruct non-owning DeviceBuffer views over
            # the env's pointers to pass to record_batch_gpu /
            # record_batch_gpu_nstep.
            var c = ctx.value()
            var action_buf = DeviceBuffer[DT](
                c,
                env.action_ptr(),
                N_ENVS * ACT,
                owning=False,
            )
            var reward_buf = DeviceBuffer[DT](
                c,
                env.reward_ptr(),
                N_ENVS,
                owning=False,
            )
            var obs_buf = DeviceBuffer[DT](
                c,
                env.obs_ptr(),
                N_ENVS * OBS,
                owning=False,
            )
            # Replay stores `terminated` (natural termination only), NOT the
            # combined `done`, so the TD bootstrap is kept on time-limit
            # truncation and dropped on real termination. Episode tracking and
            # selective reset below still use `done_ptr()`.
            var term_buf = DeviceBuffer[DT](
                c,
                env.terminated_ptr(),
                N_ENVS,
                owning=False,
            )
            comptime if NS > 1:
                trainer.record_batch_gpu_nstep[N_ENVS, NS](
                    c,
                    nstep_buf.value(),
                    prev_obs.dev.value(),
                    action_buf,
                    reward_buf,
                    obs_buf,
                    term_buf,
                )
            else:
                trainer.record_batch_gpu[N_ENVS](
                    c,
                    prev_obs.dev.value(),
                    action_buf,
                    reward_buf,
                    obs_buf,
                    term_buf,
                )

        # ── 5. Per-env episode tracking. Needs host-side reward+done.
        comptime if env_target == "cpu":
            var completed = List[Scalar[DT]]()
            accumulate_episode_returns[N_ENVS](
                env.reward_ptr(), env.done_ptr(), per_env_returns, completed
            )
            for i in range(len(completed)):
                trainer.add_complete_return(completed[i])
                ep_returns.append(trainer.mean_return())
        else:
            # GPU env: enqueue the small reward+done D2H into the ring
            # WITHOUT synchronizing; drain (one sync) only when the ring
            # fills or an emit boundary is imminent. `step_idx` is still
            # pre-increment here, so the upcoming post-increment value is
            # `step_idx + N_ENVS` — the same value the print/diag blocks test.
            var c = ctx.value()
            ep_ring.value().enqueue(c, env.reward_ptr(), env.done_ptr())

            var emit_now = cad.emit_boundary_imminent(
                step_idx + N_ENVS, total_env_steps
            )
            if ep_ring.value().due(emit_now):
                var completed = ep_ring.value().drain(c)
                for i in range(len(completed)):
                    trainer.add_complete_return(completed[i])
                    ep_returns.append(trainer.mean_return())

        # ── 8. Selective env reset. Driven by the env's DEVICE-resident RNG
        # counter (bumped inside `selective_reset_batch`), so the captured graph
        # advances reset randomness on every replay — no baked host seed. The
        # `rng_seed` arg is retained for trait/CPU compatibility but the GPU
        # reset ignores it (the device counter is authoritative).
        # ⚠ `and has_nvidia_gpu_accelerator()` added 2026-08-07. The
        # comptime asserts above have always CLAIMED this path is a
        # "no-op on non-NVIDIA" — it was not. On Apple/Metal the capture
        # branch ran and silently FROZE the env: measured ep_count 0 vs 6
        # over an identical 6000-step dm_control walker run with the flag
        # off (3 truncations x 2 lanes is the correct count). Every GPU
        # example in examples/ sets USE_ENV_CUDA_GRAPH=True, so on Apple
        # they were all training against a stopped environment.
        comptime if (
            USE_ENV_CUDA_GRAPH
            and env_target == "gpu"
            and has_nvidia_gpu_accelerator()
        ):
            # TEMP DIAGNOSTIC: inline, legacy-style explicit CUDAGraph capture
            # (mirrors the env-step block above). Reset randomness is driven by
            # the env's DEVICE RNG counter (bumped inside `selective_reset_batch`
            # → captured), so each replay resets done envs to fresh states.
            var cr = ctx.value()
            if not env_reset_graph:
                env.selective_reset_batch[N_ENVS](
                    ctx=ctx,
                    rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
                )
                cr.synchronize()
                var gr = CUDAGraph(cr)
                # ⚠ CHECK is_disabled() BEFORE CAPTURING. `CUDAGraph`
                # disables itself when the interceptor is not hooked into the
                # process — which is the ORDINARY state of a binary you BUILD
                # and then run directly, because LD_PRELOAD comes from pixi's
                # nvidia activation and not from the binary. `begin_capture`
                # RAISES on a disabled graph, so without this guard the whole
                # run dies rather than falling back to running the step
                # directly. `maybe_capture_replay` has always done this; these
                # two env-graph sites did not, and a `mojo build` + direct run
                # killed the process on the first iteration.
                #
                # The settle run above has already executed this iteration's
                # work, so taking the disabled path here is a COMPLETED step,
                # not a skipped one. Storing the disabled graph makes later
                # iterations take the `else` branch instead of reconstructing
                # (and re-printing) every time.
                if not gr.is_disabled():
                    gr.begin_capture()
                    env.selective_reset_batch[N_ENVS](
                        ctx=ctx,
                        rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
                    )
                    gr.end_capture()
                    if verbose:
                        print(
                            "[CUDA Graph] Captured ENV RESET with",
                            gr.num_nodes(),
                            "nodes",
                        )
                env_reset_graph = gr^
            elif env_reset_graph.value().is_disabled():
                env.selective_reset_batch[N_ENVS](
                    ctx=ctx,
                    rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
                )
            else:
                env_reset_graph.value().replay_on_mojo_stream()
        else:
            env.selective_reset_batch[N_ENVS](
                ctx=ctx,
                rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
            )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 9. Trainer updates.
        # Gated on NVIDIA for the same reason as the env branches above
        # (same CUDAGraph API, same 'NVIDIA only' contract). ⚠ Unlike the
        # env path this one was NOT independently measured to misbehave on
        # Metal — it is gated to honour the contract, not to fix a
        # diagnosed defect.
        comptime if USE_TRAIN_CUDA_GRAPH and has_nvidia_gpu_accelerator():
            # Capture path: once the buffer is warm, the per-update device
            # kernel sequence (`train_device_kernels`) is captured into
            # `train_graph` on first call and replayed thereafter — host
            # bookkeeping advances via `note_train_update` so counters stay
            # correct across replays. During warmup we skip training, which
            # matches `train_step` gating to False. On non-NVIDIA the closure
            # simply runs each call (no-op capture) → identical to the
            # non-captured path. Capture requires uniform replay (the
            # caller must not enable PER with this flag).
            if base_step + step_idx >= trainer.learning_starts_count():
                var c = ctx.value()

                # Capture ALL `updates_per_step` device-kernel sequences into a
                # SINGLE graph, replayed once per iteration — vs the old "one
                # graph replay per update" (which made `cuGraphLaunch` =
                # updates_per_step × iters, the dominant CPU/launch cost on
                # NVIDIA). Each captured `train_device_kernels` advances the
                # device RNG / Adam counters, so the replayed sequence draws a
                # fresh minibatch per sub-update — identical correctness to
                # single-update capture, just far fewer launches.
                def _captured_updates() capturing raises -> None:
                    for _ in range(updates_per_step):
                        trainer.train_device_kernels()

                # Same sequence, enqueued on the context the recorder hands us
                # rather than the trainer's own. That argument is the whole
                # difference between the two backends — see
                # `sac/trainer.mojo::train_device_kernels_on` for why a step
                # that reaches for `self.ctx` records nothing, silently.
                def _recorded_updates(
                    gctx: DeviceContext
                ) capturing raises -> None:
                    for _ in range(updates_per_step):
                        trainer.train_device_kernels_on(gctx)

                if use_device_graph:
                    maybe_record_replay[_recorded_updates](train_slot, c)
                else:
                    maybe_capture_replay[_captured_updates](train_graph, c)
                # Host bookkeeping advances once per logical update (the graph
                # replays the device work; counters stay on the host).
                for _ in range(updates_per_step):
                    trainer.note_train_update()
        else:
            for _ in range(updates_per_step):
                _ = trainer.train_step(base_step + step_idx)

        cad.tick(step_idx, trainer.total_train_steps())

        if cad.print_due(step_idx):
            cad.print_status(
                base_step + step_idx,
                trainer.mean_return(),
                trainer.ep_count(),
            )

        # Logger emit at the same cadence (independent of verbose), WITH a
        # live flush at this (rare) print cadence so the always-on
        # avg_reward/episodes stream reaches the dashboard even when
        # `diag_every == 0` — diag metrics batch via `log_scalar`'s
        # `buffer_size` auto-flush instead (see the diag block below).
        # Comptime-elided when L=NoOpLogger (default).
        comptime if L.ENABLED:
            if Bool(logger) and cad.log_due(step_idx):
                cad.log_status[L, True](
                    logger,
                    base_step + step_idx,
                    trainer.mean_return(),
                    trainer.ep_count(),
                )

        # `diag_every` — drain the trainer's full metric bundle (mean_q,
        # critic_loss, alpha, train_steps, …) through the logger at its own
        # cadence, mirroring the single-env `run_offpolicy_train` driver.
        # Default trait impl is a no-op for trainers that haven't wired this
        # up; SACTrainer overrides it.
        #
        # NO forced `flush()` here. `log_scalar` already auto-flushes when the
        # buffer hits `buffer_size`, so the metrics batch into one HTTP POST per
        # ~`buffer_size` points (e.g. ~every 6 diags for an ~11-metric bundle at
        # buffer_size=64). The previous per-diag `flush()` fired a BLOCKING HTTP
        # POST every `diag_every` steps — at diag_every=1000 over a 1M-step run
        # that's ~1000 synchronous POSTs serializing the training loop on HTTP
        # latency. The print-cadence flush above (rare) + `logger.close()` drain
        # the remainder, so nothing is lost.
        # Log the bundle against the TRAIN-step counter, not `step_idx`
        # (env interactions). These are per-gradient-update quantities; at
        # updates_per_step>1 (e.g. REDQ UTD 40:1) the env-step x-axis
        # under-counts by that factor. `avg_reward` / `episodes` above stay
        # on `base_step + step_idx` — those are env-level.
        comptime if L.ENABLED:
            if Bool(logger) and cad.diag_due(step_idx):
                trainer.flush_metrics_through_logger[L](
                    logger, trainer.total_train_steps()
                )

        # ── Checkpoint cadence — overwrite `checkpoint_path` with the
        # trainer's one-file v3 checkpoint. Runs in host code between
        # iterations (D2H of live params on the GPU target), so it is
        # CUDA-graph-capture safe. The default `save_state` raises.
        if cad.ckpt_due(step_idx):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)
            if verbose:
                cad.clear()
                print(
                    "[step ",
                    base_step + step_idx,
                    "] checkpoint → ",
                    checkpoint_path,
                )

        # ── Deterministic eval cadence. `avg_reward` above is a STOCHASTIC
        # rollout signal — for SAC the entropy/exploration noise inflates the
        # failure rate, so it systematically under-reports the learned policy.
        # When an isolated `eval_env` is supplied, run a GPU-parallel greedy
        # rollout (no sampling, no replay/optimizer touch) and log the true
        # policy quality as `eval/mean_return`. Runs in host code between
        # iterations (capture-safe, like the checkpoint block).
        if cad.eval_due(step_idx):
            var eval_ret = run_offpolicy_eval_batched[A, EE, EVAL_ENVS](
                ctx,
                trainer,
                eval_env.value()[],
                eval_episodes,
                max_steps=eval_max_steps,
                rng_seed=rng_seed + UInt64(step_idx + 1),
            )
            comptime if L.ENABLED:
                if Bool(logger):
                    logger.value()[].log_scalar(
                        "eval_return",
                        Float64(eval_ret),
                        base_step + step_idx,
                    )
                    logger.value()[].flush()
            if verbose:
                cad.clear()
                print(
                    "[step ",
                    base_step + step_idx,
                    "] eval/mean_return = ",
                    eval_ret,
                )

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if cad.ckpt_on:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    # Defensive final drain of any buffered episode readbacks. With
    # `sync_every == 1` (or the last iteration hitting an emit boundary)
    # nothing is pending and `drain` no-ops (no sync); this only fires if
    # the loop exited mid-ring.
    comptime if env_target == "gpu":
        if ctx and ep_ring:
            var completed = ep_ring.value().drain(ctx.value())
            for i in range(len(completed)):
                trainer.add_complete_return(completed[i])
                ep_returns.append(trainer.mean_return())

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_train_cpu_env_gpu_agent — Phase 6.1: batched
#   (cpu env, gpu train, N_ENVS>=1) GPU-agent / CPU-env hybrid.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train_cpu_env_gpu_agent[
    A: OffPolicyAgentGpu,
    E: BatchedEnv,
    N_ENVS: Int = 1,
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
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    artifacts: Optional[ArtifactSink] = None,
    run_dir: String = "",
    base_step: Int = 0,
    progress_label: String = "off-policy",
) raises -> List[Scalar[DT]]:
    """GPU-agent / CPU-env hybrid off-policy driver (Phase 6.1).

    Covers the one cross-target combination the other off-policy drivers
    deferred: `env_target == "cpu"` with `train_target == "gpu"` at any
    `N_ENVS >= 1`. Ported from the legacy
    `gpu_agent_cpu_env_train.mojo` (`run_offpolicy_continuous_train_cpu_env_gpu_agent`).

    Lets a GPU SAC/TD3/DDPG agent train against a CPU-stepped env (e.g. a
    Python Gymnasium MuJoCo env wrapped in `BatchedCpuEnv`) so a training
    failure can be attributed to "the env" or "the algorithm". The env
    steps on the host; obs/action/reward/terminated cross the H↔D
    boundary every iteration.

    Per-iteration loop:
      1. H2D current obs (env.obs_ptr → obs_dev) and D→D snapshot into
         prev_obs_dev (the `s_t` of the transition).
      2. `trainer.select_action_batched[N_ENVS]` on device (reads
         obs_dev, writes action_dev). GPU warmup uses the trainer's own
         Philox kernel, gated on `base_step + step_idx` vs
         `learning_starts`.
      3. D2H action (action_dev → env.action_ptr), then `synchronize()`
         so the host action slab is valid before the CPU env steps.
      4. `env.step_batch[N_ENVS]` on the host (writes host
         obs/reward/done/terminated).
      5. H2D next-obs / reward / terminated, then
         `trainer.record_batch_gpu[N_ENVS]`. The replay stores
         `terminated` (natural termination only) so the TD bootstrap is
         kept on time-limit truncation, dropped on real termination —
         the truncation-correct bootstrap Phase 6.1 calls for.
      6. `synchronize()` so all device reads of the env host buffers
         complete before the host mutates them in selective_reset.
      7. Per-env return accumulation; `add_complete_return` on `done`.
      8. `env.selective_reset_batch[N_ENVS]` on the host.
      9. `updates_per_step` × `trainer.train_step` (GPU).

    `ctx` MUST be the same `DeviceContext` the trainer was built with
    (Apple Metal's queue pool exhausts if a fresh context is constructed
    per call). The env's `reset_batch` / `step_batch` /
    `selective_reset_batch` receive `ctx=None` (CPU env ignores it).
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    comptime assert env_target == "cpu", (
        "run_offpolicy_train_cpu_env_gpu_agent: env_target must be 'cpu'"
        " (this is the CPU-env / GPU-agent hybrid; for a GPU env use"
        " run_offpolicy_train_batched)"
    )
    comptime assert train_target == "gpu", (
        "run_offpolicy_train_cpu_env_gpu_agent: train_target must be 'gpu'"
        " (for a CPU trainer use run_offpolicy_train_batched, which"
        " covers same-target cpu/cpu at any N_ENVS)"
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS and E.ACT_DIM == ACT
    ), "BatchedEnv dimensions must match trainer dimensions"

    # All scratches live on the device (train_target == "gpu"). obs_dev
    # holds `s_t` then is overwritten with `s_{t+1}`; prev_obs_dev keeps
    # the pre-step `s_t` for the stored transition. reward_dev /
    # term_dev are H2D mirrors of the env's host reward / terminated
    # slabs. ao / alp are the actor-output + action-log-prob scratch the
    # trainer's select_action_batched writes through.
    var obs_dev = DriverScratch["he_obs", N_ENVS, OBS].make["gpu"](ctx=ctx)
    var prev_obs_dev = DriverScratch["he_prev_obs", N_ENVS, OBS].make["gpu"](
        ctx=ctx
    )
    var action_dev = DriverScratch["he_action", N_ENVS, ACT].make["gpu"](
        ctx=ctx
    )
    var reward_dev = DriverScratch["he_reward", N_ENVS, 1].make["gpu"](ctx=ctx)
    var term_dev = DriverScratch["he_term", N_ENVS, 1].make["gpu"](ctx=ctx)
    var ao = DriverScratch["he_ao", N_ENVS, 2 * ACT].make["gpu"](ctx=ctx)
    var alp = DriverScratch["he_alp", N_ENVS, ACT + 1].make["gpu"](ctx=ctx)

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
    )

    while step_idx < total_env_steps:
        var abs_step = base_step + step_idx

        # ── 1. H2D current obs (s_t) + D→D snapshot into prev_obs.
        ctx.enqueue_copy(obs_dev.dev.value(), env.obs_ptr())
        ctx.enqueue_copy(prev_obs_dev.dev.value(), obs_dev.dev.value())

        # ── 2. Action selection on device (warmup gated inside trainer).
        trainer.select_action_batched[N_ENVS](
            LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
                obs_dev.dev_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](
                action_dev.dev_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, 2 * ACT), MutAnyOrigin](
                ao.dev_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin](
                alp.dev_ptr()
            ),
            abs_step,
        )

        # ── 3. D2H action → env host slab; sync before CPU steps.
        ctx.enqueue_copy(env.action_ptr(), action_dev.dev.value())
        ctx.synchronize()

        # ── 4. Step CPU envs (writes host obs/reward/done/terminated).
        env.step_batch[N_ENVS](
            ctx=None,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 5. H2D next-obs / reward / terminated, then GPU replay push.
        # `terminated` (natural termination only) is stored so the TD
        # bootstrap is kept on truncation, dropped on termination.
        ctx.enqueue_copy(obs_dev.dev.value(), env.obs_ptr())
        ctx.enqueue_copy(reward_dev.dev.value(), env.reward_ptr())
        ctx.enqueue_copy(term_dev.dev.value(), env.terminated_ptr())
        trainer.record_batch_gpu[N_ENVS](
            ctx,
            prev_obs_dev.dev.value(),
            action_dev.dev.value(),
            reward_dev.dev.value(),
            obs_dev.dev.value(),
            term_dev.dev.value(),
        )

        # ── 6. Sync so all device reads of the env host slabs finish
        # before selective_reset (host) mutates env.obs in place.
        ctx.synchronize()

        # ── 7. Per-env episode tracking (host reward + done, already
        # CPU-written by step_batch).
        var completed = List[Scalar[DT]]()
        accumulate_episode_returns[N_ENVS](
            env.reward_ptr(), env.done_ptr(), per_env_returns, completed
        )
        for i in range(len(completed)):
            trainer.add_complete_return(completed[i])
            ep_returns.append(trainer.mean_return())

        # ── 8. Selective env reset (host).
        env.selective_reset_batch[N_ENVS](
            ctx=None,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 9. Trainer updates (GPU).
        for _ in range(updates_per_step):
            _ = trainer.train_step(base_step + step_idx)

        cad.tick(step_idx, trainer.total_train_steps())

        if cad.print_due(step_idx):
            cad.print_status(
                base_step + step_idx,
                trainer.mean_return(),
                trainer.ep_count(),
            )

        # Logger emit at the print cadence (comptime-elided for NoOpLogger).
        # No forced flush — `buffer_size` auto-flush.
        comptime if L.ENABLED:
            if Bool(logger) and cad.log_due(step_idx):
                cad.log_status[L, False](
                    logger,
                    base_step + step_idx,
                    trainer.mean_return(),
                    trainer.ep_count(),
                )

        # `diag_every` — drain the trainer's metric bundle through the
        # logger. Default trait impl is a no-op for trainers that haven't
        # wired this up yet. Logged against the TRAIN-step counter (not the
        # env-interaction `step_idx`) — per-gradient-update quantities at
        # updates_per_step>1 (e.g. REDQ UTD 40:1) would otherwise be plotted
        # on an x-axis under-counted by that factor.
        comptime if L.ENABLED:
            if Bool(logger) and cad.diag_due(step_idx):
                trainer.flush_metrics_through_logger[L](
                    logger, trainer.total_train_steps()
                )

        # `checkpoint_every` — overwrite `checkpoint_path` with the
        # trainer's one-file v3 checkpoint. The trait default raises.
        if cad.ckpt_due(step_idx):
            trainer.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if cad.ckpt_on:
        trainer.save_state(checkpoint_path)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_eval — single-env greedy eval, target-agnostic.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_eval[
    A: OffPolicyAgent,
    E: BoxContinuousActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Non-mutating greedy eval driver — replaces both
    `run_offpolicy_eval_cpu` and `run_offpolicy_eval_gpu`.

    Trainer contract: `OffPolicyAgent.select_greedy_action`
    handles target dispatch internally (CPU trainers run native; GPU
    trainers H2D the obs and D2H the action under the hood). Only that
    method is invoked here — `record` / `train_step` /
    `end_episode` / `add_complete_return` are intentionally skipped so
    eval doesn't touch the trainer's replay buffer, optimizers, or
    episode tracker. `obs_dim` / `act_dim` are read from
    `A.AGENT_OBS_DIM` / `A.AGENT_ACT_DIM`.
    """
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    var action_list = List[Scalar[E.dtype]](capacity=ACT)
    for _ in range(ACT):
        action_list.append(Scalar[E.dtype](0.0))

    var total_return = Scalar[DT](0.0)
    var t_start = perf_counter_ns()

    for ep in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        for _ in range(max_steps_per_episode):
            for d in range(OBS):
                obs[d] = Scalar[DT](obs_list[d])
            trainer.select_greedy_action(obs, action)
            for j in range(ACT):
                action_list[j] = Scalar[E.dtype](action[j])
            var step_res = env.step_continuous_vec[E.dtype](action_list)
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


def run_offpolicy_eval_render[
    A: OffPolicyAgent,
    E: BoxContinuousActionEnv & RenderableEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    frame_delay_ms: Int = 16,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Greedy eval with live env-owned 3D/2D rendering — the render-enabled
    sibling of `run_offpolicy_eval`. Identical greedy loop (only
    `select_greedy_action` touches the trainer; replay / optimizers /
    episode tracker are untouched), plus the `RenderableEnv` calls the
    example eval scripts used to inline: `init_renderer` once, per-step
    `render_frame` + `renderer_delay`, and a `check_renderer_quit` /
    `is_renderer_open` early-out that stops ALL episodes when the user
    closes the window. Falls back to headless (reward-only) if the
    renderer is unavailable. Returns the mean episode return.

    Bound on `BoxContinuousActionEnv & RenderableEnv` so it stays a no-op-
    safe single entry point for any physics env. `frame_delay_ms` paces
    playback (~16 ms ≈ 60 FPS)."""
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))

    var action_list = List[Scalar[E.dtype]](capacity=ACT)
    for _ in range(ACT):
        action_list.append(Scalar[E.dtype](0.0))

    var have_renderer = env.init_renderer()
    if not have_renderer and verbose:
        print(
            "  WARNING: renderer unavailable — running headless (reward only)."
        )

    var total_return = Scalar[DT](0.0)
    var t_start = perf_counter_ns()
    var quit = False

    for ep in range(num_episodes):
        if quit:
            break
        var obs_list = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        for _ in range(max_steps_per_episode):
            for d in range(OBS):
                obs[d] = Scalar[DT](obs_list[d])
            trainer.select_greedy_action(obs, action)
            for j in range(ACT):
                action_list[j] = Scalar[E.dtype](action[j])
            var step_res = env.step_continuous_vec[E.dtype](action_list)
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            ep_return += Scalar[DT](reward)
            ep_steps += 1

            if have_renderer:
                env.render_frame()
                env.renderer_delay(frame_delay_ms)
                if env.check_renderer_quit() or not env.is_renderer_open():
                    quit = True
                    break

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

    if have_renderer:
        env.close_renderer()

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
