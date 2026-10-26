"""BatchedEnv trait + BatchedCpuEnv / BatchedGpuEnv adapters — Tier-3.

The Tier-1 drivers had to remain two functions because env
APIs differ structurally — CPU envs return host Lists, GPU envs take
DeviceBuffer kernel args. Tier-3 lifts a uniform `BatchedEnv` trait
over both via the **env-owns-buffers** pattern:

  trait BatchedEnv:
      comptime ENV_TARGET: StaticString
      comptime OBS_DIM: Int
      comptime ACT_DIM: Int

      def reset_batch[BATCH](mut self, ctx, rng_seed) raises
      def step_batch[BATCH](mut self, ctx, rng_seed) raises
      def selective_reset_batch[BATCH](mut self, ctx, rng_seed) raises

      def obs_ptr(self)    -> Pointer[Scalar[DT], MutAnyOrigin]
      def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]
      def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]
      def done_ptr(self)   -> Pointer[Scalar[DT], MutAnyOrigin]

The env adapter owns the obs/action/reward/done buffers (host List for
CPU, DeviceBuffer for GPU). The driver writes actions into
`env.action_ptr()`, calls `env.step_batch()`, then reads
obs/reward/done via the accessor pointers.

Two adapters:

  - `BatchedCpuEnv[E, N_ENVS, OBS_DIM, ACT_DIM]` wraps any
    `BoxContinuousActionEnv & Copyable & Movable` env. Holds
    `Array[E, N_ENVS]` of independent env instances.
  - `BatchedGpuEnv[E, N_ENVS, OBS_DIM, ACT_DIM, STATE_SIZE]` wraps any
    `GPUContinuousEnv`. Holds internal `DeviceBuffer` fields for
    state/obs/action/reward/done/terminated; dispatches the env's
    static `*_kernel_gpu` methods.

For consumers needing the underlying `DeviceBuffer` (e.g. to call
trainer.record_batch_gpu which takes DeviceBuffer args), the driver
reconstructs non-owning views via
`DeviceBuffer[DT](ctx, env.obs_ptr(), N*OBS, owning=False)`.
"""

# `parallelize` moved out of the stdlib into MAX in Mojo 1.0 — `std.algorithm`
# keeps `vectorize` but no longer carries any parallel primitive. The async
# runtime it is built on is still `std.runtime.asyncrt`.
from max.algorithm import parallelize
from max.gpu import thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.core.env_traits import (
    BoxContinuousActionEnv,
    BoxDiscreteActionEnv,
    GPUContinuousEnv,
    GPUDiscreteEnv,
)


from noeira.nn.core.target_storage import require_ctx
from noeira.core.tiles import Tile, tile


def _increment_env_rng_kernel(
    counter: Tile[DType.uint64, 1],
):
    """Bump the device-resident env RNG counter by 1. Launch grid=(1,),
    block=(1,). Enqueued at the head of the selective-reset sequence so each
    CUDA-graph replay draws a FRESH reset seed without host intervention —
    mirrors legacy `increment_env_rng_kernel`."""
    if Int(thread_idx.x) == 0:
        counter[0] = counter[0] + UInt64(1)


# ──────────────────────────────────────────────────────────────────────
# BatchedEnv trait — uniform env-interaction surface.
# ──────────────────────────────────────────────────────────────────────


trait BatchedEnv(Movable & Deinitable):
    """Uniform N_ENVS-batched env surface. Env owns its
    obs/action/reward/done buffers internally; driver reads/writes via
    pointer accessors. Method comptime is `BATCH` (not `N_ENVS`) so
    impls whose struct already has an N_ENVS comptime can conform
    without name shadowing."""

    comptime ENV_TARGET: StaticString
    comptime OBS_DIM: Int
    comptime ACT_DIM: Int

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        """Initial reset for all N envs. Writes the env's internal
        obs buffer; subsequent `obs_ptr()` reads see the new obs."""
        ...

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        """Step all N envs. Reads the env's `action_ptr()`; overwrites
        `obs_ptr()`, `reward_ptr()`, `done_ptr()` in place."""
        ...

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        """Reset only envs where the last `step_batch` wrote
        `done > 0.5`; obs for reset envs gets refreshed. Reads
        `done_ptr()`."""
        ...

    # ─── Pointer accessors — same return type for CPU and GPU ────────

    def obs_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS, OBS_DIM] obs slab on ENV_TARGET."""
        ...

    def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS, ACT_DIM] action slab. Driver
        writes action into this slab before calling step_batch."""
        ...

    def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS] reward slab."""
        ...

    def done_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS] done slab (1.0 if done else 0.0).

        `done` = terminated OR truncated — drives episode tracking and
        selective reset."""
        ...

    def terminated_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the [N_ENVS] terminated slab (1.0 iff natural
        termination, NOT time-limit truncation).

        Stored into the replay buffer so the off-policy TD bootstrap is kept
        on truncation but dropped on termination. For envs that never
        terminate naturally this is all-zeros (bootstrap always kept)."""
        ...


# ──────────────────────────────────────────────────────────────────────
# BatchedCpuEnv[E, N_ENVS, OBS, ACT] — CPU env adapter.
# ──────────────────────────────────────────────────────────────────────


struct BatchedCpuEnv[
    E: BoxContinuousActionEnv & Copyable & Deinitable,
    N_ENVS: Int,
    OBS_DIM_: Int,
    ACT_DIM_: Int,
](BatchedEnv):
    """Wraps N independent instances of a CPU env via
    `Array[E, N_ENVS]` and owns the obs/action/reward/done host
    buffers. Per-env state independence is verified by the Tier-2
    viability spike.

    Construction:
        var template = PendulumEnv[DT]()  # OBS=3, ACT=1
        var batched = BatchedCpuEnv[PendulumEnv[DT], 4, 3, 1](template)

    `OBS_DIM_` / `ACT_DIM_` are explicit comptime params because
    `BoxContinuousActionEnv` exposes obs/action dimensions only as
    runtime methods, not comptime members — we need them comptime to
    size the internal buffers.
    """

    comptime ENV_TARGET: StaticString = "cpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = Self.ACT_DIM_

    var envs: Array[Self.E, Self.N_ENVS]

    # Internally-owned host buffers — driver reads/writes via accessors.
    var _obs: List[Scalar[DT]]
    var _action: List[Scalar[DT]]
    var _reward: List[Scalar[DT]]
    var _done: List[Scalar[DT]]
    var _terminated: List[Scalar[DT]]

    # Pre-allocated host scratch for the per-env action List we feed
    # into `step_continuous_vec` (avoids allocating a new List every
    # step × env). E.dtype may differ from DT.
    var _action_scratch: List[Scalar[Self.E.dtype]]

    def __init__(out self, template: Self.E):
        self.envs = Array[Self.E, Self.N_ENVS](fill=template)
        self._obs = List[Scalar[DT]](
            length=Self.N_ENVS * Self.OBS_DIM,
            fill=Scalar[DT](0.0),
        )
        self._action = List[Scalar[DT]](
            length=Self.N_ENVS * Self.ACT_DIM,
            fill=Scalar[DT](0.0),
        )
        self._reward = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._done = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._terminated = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._action_scratch = List[Scalar[Self.E.dtype]](
            capacity=Self.ACT_DIM,
        )
        for _ in range(Self.ACT_DIM):
            self._action_scratch.append(Scalar[Self.E.dtype](0.0))

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedCpuEnv: reset_batch BATCH must match struct param"
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            var obs_list = self.envs[env_idx].reset_obs_list()
            for d in range(Self.OBS_DIM):
                self._obs[env_idx * Self.OBS_DIM + d] = Scalar[DT](obs_list[d])

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedCpuEnv: step_batch BATCH must match struct param"
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            # Stage this env's action lane (E.dtype) from the host buf.
            for j in range(Self.ACT_DIM):
                self._action_scratch[j] = Scalar[Self.E.dtype](
                    self._action[env_idx * Self.ACT_DIM + j]
                )
            var step_res = self.envs[env_idx].step_continuous_vec[Self.E.dtype](
                self._action_scratch
            )
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            for d in range(Self.OBS_DIM):
                self._obs[env_idx * Self.OBS_DIM + d] = Scalar[DT](nxt[d])
            self._reward[env_idx] = Scalar[DT](reward)
            self._done[env_idx] = Scalar[DT](1.0) if done else Scalar[DT](0.0)
            self._terminated[env_idx] = Scalar[DT](1.0) if self.envs[
                env_idx
            ].was_terminated() else Scalar[DT](0.0)

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedCpuEnv: selective_reset_batch BATCH must match struct param"
        _ = ctx
        _ = rng_seed
        for env_idx in range(Self.N_ENVS):
            if self._done[env_idx] > Scalar[DT](0.5):
                var obs_list = self.envs[env_idx].reset_obs_list()
                for d in range(Self.OBS_DIM):
                    self._obs[env_idx * Self.OBS_DIM + d] = Scalar[DT](
                        obs_list[d]
                    )

    def obs_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._terminated.unsafe_ptr())


# ──────────────────────────────────────────────────────────────────────
# BatchedCpuDiscreteEnv[E, N_ENVS, OBS] — multi-core CPU discrete envs.
# ──────────────────────────────────────────────────────────────────────


def _splitmix64(x: UInt64) -> UInt64:
    """SplitMix64 finalizer — cheap stateless per-(seed, env, episode)
    hash for the no-op start draw. Wrapping UInt64 arithmetic."""
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


struct BatchedCpuDiscreteEnv[
    E: BoxDiscreteActionEnv & Movable & Deinitable,
    N_ENVS: Int,
    OBS_DIM_: Int,
](BatchedEnv):
    """Discrete-action sibling of `BatchedCpuEnv` that steps its N envs
    in PARALLEL across CPU cores (`max.algorithm.parallelize`) — the
    Stage-1 lever from `docs/ATARI_AUDIT.md` §2. Each env instance is
    fully independent, so per-env work items never share mutable state
    and the parallel step is bit-identical to a serial loop (validated
    by `tests/deep_agents/test_batched_cpu_discrete_env.mojo` on the
    deterministic Atari emulator).

    Differences from the continuous `BatchedCpuEnv`:

      - `E` is bound on `BoxDiscreteActionEnv` (`step_obs(Int)`), action
        slab holds one index per env as `Scalar[DT]` (`ACT_DIM = 1`).
      - Envs are MOVED in via a caller-built `List[E]` instead of copied
        from a template: emulator-style envs (AtariEnv) own heap buffers
        and are move-only; per-instance construction also lets the
        caller share one read-only ROM across instances.
      - Optional ALE-style random no-op starts (`noop_max`): on every
        (selective) reset each env plays `k ~ U[0, noop_max]` no-op
        actions, `k` drawn from a per-(seed, env, episode) hash. Without
        this, N deterministic emulators driven by a near-deterministic
        policy step in lockstep and the batch collapses to ~1 effective
        env. `noop_action` must be the env's NOOP index (0 for the ALE
        minimal action sets).

    Thread-safety contract: `E.step_obs` / `E.reset_obs_list` must not
    touch shared mutable state (e.g. the GLOBAL std.random stream) —
    AtariEnv is fully self-contained; envs that draw reset noise from
    the global RNG keep working but lose run-to-run determinism.

    Construction:
        var envs = List[AtariEnv[1]]()
        for _ in range(8):
            envs.append(AtariEnv[1](game, rom, rom_size))
        var batched = BatchedCpuDiscreteEnv[AtariEnv[1], 8, 28224](
            envs^, noop_max=30
        )
    """

    comptime ENV_TARGET: StaticString = "cpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = 1

    var envs: List[Self.E]
    var noop_max: Int
    var noop_action: Int

    # Internally-owned host buffers — driver reads/writes via accessors.
    var _obs: List[Scalar[DT]]
    var _action: List[Scalar[DT]]
    var _reward: List[Scalar[DT]]
    var _done: List[Scalar[DT]]
    var _terminated: List[Scalar[DT]]
    # Per-env episode counter feeding the no-op start hash (so consecutive
    # episodes of the same env draw different no-op counts).
    var _reset_count: List[Int]

    def __init__(
        out self,
        var envs: List[Self.E],
        *,
        noop_max: Int = 0,
        noop_action: Int = 0,
    ) raises:
        if len(envs) != Self.N_ENVS:
            raise Error(
                "BatchedCpuDiscreteEnv: expected ",
                Self.N_ENVS,
                " envs, got ",
                len(envs),
            )
        self.envs = envs^
        self.noop_max = noop_max
        self.noop_action = noop_action
        self._obs = List[Scalar[DT]](
            length=Self.N_ENVS * Self.OBS_DIM,
            fill=Scalar[DT](0.0),
        )
        self._action = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._reward = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._done = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._terminated = List[Scalar[DT]](
            length=Self.N_ENVS,
            fill=Scalar[DT](0.0),
        )
        self._reset_count = List[Int](length=Self.N_ENVS, fill=0)

    def _reset_lanes(mut self, rng_seed: UInt64, only_done: Bool):
        """Reset env lanes in parallel (all lanes, or only `done` ones),
        apply the optional no-op start, and refresh the obs slab."""
        var envs_ptr = self.envs.unsafe_ptr()
        var obs_ptr = self._obs.unsafe_ptr()
        var done_ptr = self._done.unsafe_ptr()
        var rc_ptr = self._reset_count.unsafe_ptr()
        var noop_max = self.noop_max
        var noop_action = self.noop_action

        def reset_one(env_idx: Int) {imm}:
            if only_done and done_ptr[unsafe_offset=env_idx] <= Scalar[DT](0.5):
                return
            var obs_list = envs_ptr[unsafe_offset=env_idx].reset_obs_list()
            if noop_max > 0:
                rc_ptr[unsafe_offset=env_idx] += 1
                var k = Int(
                    _splitmix64(
                        rng_seed
                        ^ (UInt64(env_idx) * UInt64(0x9E3779B97F4A7C15))
                        ^ (UInt64(rc_ptr[unsafe_offset=env_idx]) * UInt64(0xBF58476D1CE4E5B9))
                    )
                    % UInt64(noop_max + 1)
                )
                for _ in range(k):
                    var res = envs_ptr[unsafe_offset=env_idx].step_obs(noop_action)
                    if res[2]:
                        # Done during the no-op run (degenerate env) —
                        # re-reset and start the episode without no-ops.
                        obs_list = envs_ptr[unsafe_offset=env_idx].reset_obs_list()
                        break
                    obs_list = res[0].copy()
            for d in range(Self.OBS_DIM):
                obs_ptr[unsafe_offset=env_idx * Self.OBS_DIM + d] = Scalar[DT](obs_list[d])

        parallelize(reset_one, Self.N_ENVS)

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedCpuDiscreteEnv: reset_batch BATCH must match struct param"
        _ = ctx
        self._reset_lanes(rng_seed, only_done=False)

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedCpuDiscreteEnv: step_batch BATCH must match struct param"
        _ = ctx
        _ = rng_seed
        var envs_ptr = self.envs.unsafe_ptr()
        var obs_ptr = self._obs.unsafe_ptr()
        var action_ptr = self._action.unsafe_ptr()
        var reward_ptr = self._reward.unsafe_ptr()
        var done_ptr = self._done.unsafe_ptr()
        var term_ptr = self._terminated.unsafe_ptr()

        def step_one(env_idx: Int) {imm}:
            comptime if Self.E.dtype == DT:
                # Allocation-free path: the env writes its obs lane in
                # place (AtariEnv overrides `step_obs_into` to skip the
                # 28K-element List the `step_obs` path materializes).
                var res = envs_ptr[unsafe_offset=env_idx].step_obs_into(
                    Int(action_ptr[unsafe_offset=env_idx]),
                    rebind[Pointer[Scalar[Self.E.dtype], MutAnyOrigin]](
                        obs_ptr.unsafe_offset(env_idx * Self.OBS_DIM)
                    ),
                )
                reward_ptr[unsafe_offset=env_idx] = rebind[Scalar[DT]](res[0])
                done_ptr[unsafe_offset=env_idx] = Scalar[DT](1.0) if res[1] else Scalar[DT](
                    0.0
                )
            else:
                # Dtype-converting fallback (E.dtype != DT).
                var res = envs_ptr[unsafe_offset=env_idx].step_obs(Int(action_ptr[unsafe_offset=env_idx]))
                for d in range(Self.OBS_DIM):
                    obs_ptr[unsafe_offset=env_idx * Self.OBS_DIM + d] = Scalar[DT](res[0][d])
                reward_ptr[unsafe_offset=env_idx] = Scalar[DT](res[1])
                done_ptr[unsafe_offset=env_idx] = Scalar[DT](1.0) if res[2] else Scalar[DT](
                    0.0
                )
            term_ptr[unsafe_offset=env_idx] = Scalar[DT](1.0) if envs_ptr[
                unsafe_offset=env_idx
            ].was_terminated() else Scalar[DT](0.0)

        parallelize(step_one, Self.N_ENVS)

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedCpuDiscreteEnv: selective_reset_batch BATCH must match"
            " struct param"
        )
        _ = ctx
        self._reset_lanes(rng_seed, only_done=True)

    def obs_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._terminated.unsafe_ptr())


# ──────────────────────────────────────────────────────────────────────
# BatchedGpuEnv[E, N_ENVS, OBS, ACT, STATE_SIZE] — GPU env adapter.
# ──────────────────────────────────────────────────────────────────────


struct BatchedGpuEnv[
    E: GPUContinuousEnv,
    N_ENVS: Int,
    OBS_DIM_: Int,
    ACT_DIM_: Int,
](BatchedEnv):
    """Wraps a `GPUContinuousEnv` and owns the per-step device buffers
    (state, obs, action, reward, done, terminated). Dispatches the
    env's static `*_kernel_gpu` methods inside the BatchedEnv methods.

    Construction:
        var ctx = DeviceContext()
        var env = BatchedGpuEnv[PendulumV2[DT], 4, 3, 1](ctx)

    `STATE_SIZE` is derived from `E.STATE_SIZE` — the trait surface
    exposes it directly, so taking it as a separate comptime would
    invite the caller-passes-wrong-value bug (which we hit during
    Tier-3 bring-up when STATE_SIZE=3 was passed for PendulumV2
    whose actual STATE_SIZE=6, sizing the state buffer too small
    and making the kernel read garbage step-counters).

    The adapter is intentionally `E`-agnostic for its buffers — only
    the kernel dispatches use `E`. Driver bounds on `BatchedEnv`
    (uniform trait); the driver reconstructs `DeviceBuffer` views
    from `obs_ptr()` etc. via `(ctx, ptr, size, owning=False)` to
    pass to trainer.record_batch_gpu.
    """

    comptime ENV_TARGET: StaticString = "gpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = Self.ACT_DIM_
    comptime STATE_SIZE: Int = Self.E.STATE_SIZE
    # True when state-prefix obs extraction is safe. False (e.g. a future
    # pixel-obs continuous env) → don't re-extract obs on selective_reset, or
    # the raw trait default would clobber the stepped obs (see the discrete
    # `BatchedGpuDiscreteEnv` for the bug this guards against). No live env
    # hits the False branch today; this is a defensive symmetric guard.
    comptime _OBS_IS_STATE_PREFIX: Bool = Self.OBS_DIM_ <= Self.E.STATE_SIZE

    var _states: DeviceBuffer[DT]
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]
    # Persistent physics step workspace: shared model params + per-env solver
    # scratch (layout [STEP_WS_SHARED | N_ENVS*STEP_WS_PER_ENV]). Allocated and
    # model-initialized ONCE here, then passed to `step_kernel_gpu` every step.
    # Without it, `step_kernel_gpu(workspace_ptr=None)` allocates AND re-uploads
    # the model on EVERY step (large per-step waste) — and, fatally, under
    # CUDA-graph capture the captured kernels reference that per-call buffer
    # after it is freed, so every replay runs physics on garbage model params
    # (silent divergence on NVIDIA; invisible on Apple where capture is a
    # no-op). Mirrors the legacy GPU driver's persistent `workspace_buf`.
    var _workspace: DeviceBuffer[DT]
    # Device-resident env RNG counter (1-elem uint64). `selective_reset_batch`
    # bumps it (via `_increment_env_rng_kernel`) and feeds it to the env's
    # `selective_reset_kernel_gpu` as `rng_counter_ptr`, so reset randomness
    # lives on-device and advances on every CUDA-graph replay (the env-reset
    # graph would otherwise bake a host seed → every episode resets to the SAME
    # initial state, collapsing the start-state distribution). This changes the
    # reset RNG stream vs the old host-seed scheme, so trajectories differ from
    # prior runs (same distribution, different draw) — but Apple-eager and
    # NVIDIA-captured now share the identical device-counter stream.
    var _env_rng_counter: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext) raises:
        # ⚠ `OBS_DIM_` / `ACT_DIM_` are CALLER-SUPPLIED, so nothing stopped them
        # disagreeing with the wrapped env's own dims — the same
        # caller-passes-wrong-value bug this struct's docstring records for
        # STATE_SIZE, left open for the other two. It bites harder than it
        # looks: a driver's `E.ACT_DIM == trainer.ACT` check reads the
        # WRAPPER's declared value, so a wrong one satisfies every assert and
        # the env kernel then strides the action slab wrong — garbage actions,
        # no error. (Wrapping a 1-action Pendulum as 2-action cost a debug
        # cycle on the multi-task MPC gate.)
        #
        # ⚠ RUNTIME `raise`, NOT `comptime assert`: a `comptime assert` in this
        # non-parametric `__init__` is silently never evaluated — verified by
        # putting `comptime assert False` here and watching it compile clean.
        # (The ones in `reset_batch[BATCH]` / `step_batch[BATCH]` do fire;
        # those methods carry their own comptime params.) Construction runs
        # once per env, so a runtime check costs nothing.
        if Self.OBS_DIM_ != Self.E.OBS_DIM:
            raise Error(
                "BatchedGpuEnv: OBS_DIM_ must equal the wrapped env's OBS_DIM"
            )
        if Self.ACT_DIM_ != Self.E.ACTION_DIM:
            raise Error(
                "BatchedGpuEnv: ACT_DIM_ must equal the wrapped env's"
                " ACTION_DIM"
            )
        self._states = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.STATE_SIZE
        )
        self._obs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS_DIM)
        self._action = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.ACT_DIM)
        self._reward = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._terminated = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        ctx.enqueue_memset(self._action, 0)
        ctx.enqueue_memset(self._reward, 0)
        ctx.enqueue_memset(self._done, 0)
        ctx.enqueue_memset(self._terminated, 0)
        # Persistent step workspace (shared model + per-env scratch). Min size 1
        # so envs with no workspace (STEP_WS_SHARED+PER_ENV==0) still hold a
        # valid (unused) buffer. Initialize the shared model portion once.
        comptime WS_TOTAL = (
            Self.E.STEP_WS_SHARED + Self.N_ENVS * Self.E.STEP_WS_PER_ENV
        )
        self._workspace = ctx.enqueue_create_buffer[DT](
            WS_TOTAL if WS_TOTAL > 0 else 1
        )
        comptime if WS_TOTAL > 0:
            Self.E.init_step_workspace_gpu[Self.N_ENVS](ctx, self._workspace)
        # Seed the device RNG counter. Any nonzero start works; reset uses the
        # live counter value, bumped before each selective reset.
        self._env_rng_counter = ctx.enqueue_create_buffer[DType.uint64](1)
        self._env_rng_counter.enqueue_fill(UInt64(42))

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedGpuEnv: reset_batch BATCH must match struct param"
        var c = require_ctx["BatchedGpuEnv.reset_batch"](ctx)
        Self.E.reset_kernel_gpu[Self.N_ENVS, Self.STATE_SIZE](
            c,
            self._states,
            rng_seed=rng_seed,
        )
        Self.E.extract_obs_kernel_gpu[
            Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM
        ](c, self._states, self._obs)

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedGpuEnv: step_batch BATCH must match struct param"
        var c = require_ctx["BatchedGpuEnv.step_batch"](ctx)
        # Pass the PERSISTENT workspace so `step_kernel_gpu` neither
        # re-allocates nor re-uploads the model per step, and — critically —
        # the captured kernels (USE_ENV_CUDA_GRAPH) reference stable memory on
        # every replay. physics3d uses it; envs that don't need a workspace
        # ignore the pointer.
        Self.E.step_kernel_gpu[
            Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM, Self.ACT_DIM
        ](
            c,
            self._states,
            self._action,
            self._reward,
            self._done,
            self._terminated,
            self._obs,
            rng_seed=rng_seed,
            workspace_ptr=mptr(self._workspace.unsafe_ptr()),
        )

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedGpuEnv: selective_reset_batch BATCH must match struct param"
        var c = require_ctx["BatchedGpuEnv.selective_reset_batch"](ctx)
        # `rng_seed` is retained for trait/CPU compatibility but the GPU reset
        # is driven by the DEVICE counter (capture-safe): bump it, then pass it
        # as `rng_counter_ptr` so `selective_reset_kernel_gpu` ignores the host
        # seed. The bump is enqueued on the same stream and is captured into the
        # env-reset graph, so every replay advances the seed.
        _ = rng_seed
        var cnt_t = tile[1](self._env_rng_counter)
        c.enqueue_function[_increment_env_rng_kernel](
            cnt_t, grid_dim=(1,), block_dim=(1,)
        )
        Self.E.selective_reset_kernel_gpu[Self.N_ENVS, Self.STATE_SIZE](
            c,
            self._states,
            self._done,
            rng_seed=rng_seed,
            # Pass the PERSISTENT workspace: like `step_kernel_gpu`,
            # `selective_reset_kernel_gpu` otherwise allocates AND re-uploads the
            # model on EVERY reset (per-reset waste) — and, fatally under
            # capture, the captured kernels would reference that per-call buffer
            # after it is freed (garbage model on replay → divergence). Mirrors
            # legacy's reset call passing `workspace_buf`.
            workspace_ptr=mptr(self._workspace.unsafe_ptr()),
            rng_counter_ptr=mptr(self._env_rng_counter.unsafe_ptr()),
        )
        # Re-derive obs from the (post-step / post-reset) state — correct for
        # state-prefix / derived clean-obs envs (all continuous envs today).
        # Gated so a future pixel-obs continuous env (OBS_DIM > STATE_SIZE)
        # would NOT have its stepped obs clobbered by the raw state-prefix
        # default every iteration (the bug fixed in BatchedGpuDiscreteEnv).
        comptime if Self._OBS_IS_STATE_PREFIX:
            Self.E.extract_obs_kernel_gpu[
                Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM
            ](c, self._states, self._obs)

    def obs_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        # `_terminated` is written by `step_kernel_gpu` (1.0 iff natural
        # termination, NOT truncation) — see GPUContinuousEnv.step_kernel_gpu.
        return mptr(self._terminated.unsafe_ptr())


# ──────────────────────────────────────────────────────────────────────
# BatchedGpuDiscreteEnv[E, N_ENVS, OBS_DIM, ACT_DIM] — GPU discrete env.
# ──────────────────────────────────────────────────────────────────────


struct BatchedGpuDiscreteEnv[
    E: GPUDiscreteEnv,
    N_ENVS: Int,
    OBS_DIM_: Int,
    ACT_DIM_: Int = 1,
](BatchedEnv):
    """Discrete-action sibling of `BatchedGpuEnv`. Wraps a
    `GPUDiscreteEnv` (Pong, Breakout, SpaceInvaders, CartPole-GPU, …) and
    owns the per-step device buffers, dispatching the env's static
    `*_kernel_gpu` methods.

    Differences from the continuous `BatchedGpuEnv`:

      - `E` is bound on `GPUDiscreteEnv`, whose `step_kernel_gpu` takes
        THREE comptime params `[N_ENVS, STATE_SIZE, OBS_DIM]` (no
        `ACTION_DIM`) and reads `_action` as `[N_ENVS]` integer indices
        stored as `Scalar[DT]`. `ACT_DIM_` defaults to 1 (one action
        index per env) and exists only so the buffer/trait dimensions
        line up with the discrete trainer (`SAMPLE.ACT == 1`).

      - Obs seeding after reset: the trait-default `extract_obs_kernel_gpu`
        copies `obs[e] = state[e][0:OBS_DIM]`, which is valid only when
        `OBS_DIM <= STATE_SIZE` (clean-obs envs whose observation is a
        state prefix). For pixel envs (`OBS_DIM = 4·84·84 ≫ STATE_SIZE`)
        that would read out of bounds, so we comptime-skip it and
        zero-fill `_obs` instead — `step_kernel_gpu` renders the real
        pixel observation on the first step (frame-stack warmup
        convention). The very first `prev_obs` per episode is therefore a
        zero / raw-prefix frame; negligible over multi-thousand-step Pong
        episodes and entirely inside the random-action warmup window.

    Construction:
        var ctx = DeviceContext()
        var env = BatchedGpuDiscreteEnv[PongEnv[DT], 256, 6, 1](ctx)
    """

    comptime ENV_TARGET: StaticString = "gpu"
    comptime OBS_DIM: Int = Self.OBS_DIM_
    comptime ACT_DIM: Int = Self.ACT_DIM_
    comptime STATE_SIZE: Int = Self.E.STATE_SIZE
    # True when the trait-default state-prefix obs extraction is safe.
    comptime _OBS_IS_STATE_PREFIX: Bool = Self.OBS_DIM_ <= Self.E.STATE_SIZE

    var _states: DeviceBuffer[DT]
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]
    # Persistent step workspace (e.g. PongPixelEnv frame stacks /
    # framebuffers). Allocated + initialized ONCE; passed to every
    # `step_kernel_gpu` / `selective_reset_kernel_gpu` — see
    # `BatchedGpuEnv` for the per-step-alloc / capture rationale.
    var _workspace: DeviceBuffer[DT]
    var _env_rng_counter: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext) raises:
        self._states = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.STATE_SIZE
        )
        self._obs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS_DIM)
        self._action = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.ACT_DIM)
        self._reward = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._terminated = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        ctx.enqueue_memset(self._obs, 0)
        ctx.enqueue_memset(self._action, 0)
        ctx.enqueue_memset(self._reward, 0)
        ctx.enqueue_memset(self._done, 0)
        ctx.enqueue_memset(self._terminated, 0)
        comptime WS_TOTAL = (
            Self.E.STEP_WS_SHARED + Self.N_ENVS * Self.E.STEP_WS_PER_ENV
        )
        self._workspace = ctx.enqueue_create_buffer[DT](
            WS_TOTAL if WS_TOTAL > 0 else 1
        )
        comptime if WS_TOTAL > 0:
            Self.E.init_step_workspace_gpu[Self.N_ENVS](ctx, self._workspace)
        self._env_rng_counter = ctx.enqueue_create_buffer[DType.uint64](1)
        self._env_rng_counter.enqueue_fill(UInt64(42))

    def _seed_obs(mut self, c: DeviceContext) raises:
        """Seed `_obs` after a (selective) reset. Clean-obs envs use the
        trait-default state-prefix extraction; pixel envs zero-fill and
        let the next `step_batch` render."""
        comptime if Self._OBS_IS_STATE_PREFIX:
            Self.E.extract_obs_kernel_gpu[
                Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM
            ](c, self._states, self._obs)
        else:
            c.enqueue_memset(self._obs, 0)

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedGpuDiscreteEnv: reset_batch BATCH must match struct param"
        var c = require_ctx["BatchedGpuDiscreteEnv.reset_batch"](ctx)
        Self.E.reset_kernel_gpu[Self.N_ENVS, Self.STATE_SIZE](
            c,
            self._states,
            rng_seed=rng_seed,
        )
        self._seed_obs(c)

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert (
            BATCH == Self.N_ENVS
        ), "BatchedGpuDiscreteEnv: step_batch BATCH must match struct param"
        var c = require_ctx["BatchedGpuDiscreteEnv.step_batch"](ctx)
        # Discrete `step_kernel_gpu`: THREE comptime params (no ACTION_DIM).
        # Reads `_action` as [N_ENVS] integer indices; writes obs/reward/
        # done/terminated in place.
        Self.E.step_kernel_gpu[Self.N_ENVS, Self.STATE_SIZE, Self.OBS_DIM](
            c,
            self._states,
            self._action,
            self._reward,
            self._done,
            self._terminated,
            self._obs,
            rng_seed=rng_seed,
            workspace_ptr=mptr(self._workspace.unsafe_ptr()),
        )

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "BatchedGpuDiscreteEnv: selective_reset_batch BATCH must match"
            " struct param"
        )
        if not ctx:
            raise Error(
                "BatchedGpuDiscreteEnv.selective_reset_batch: ctx required"
            )
        var c = ctx.value()
        _ = rng_seed
        var cnt_t = tile[1](self._env_rng_counter)
        c.enqueue_function[_increment_env_rng_kernel](
            cnt_t, grid_dim=(1,), block_dim=(1,)
        )
        Self.E.selective_reset_kernel_gpu[Self.N_ENVS, Self.STATE_SIZE](
            c,
            self._states,
            self._done,
            rng_seed=rng_seed,
            workspace_ptr=mptr(self._workspace.unsafe_ptr()),
            rng_counter_ptr=mptr(self._env_rng_counter.unsafe_ptr()),
        )
        # Re-seed obs ONLY for state-prefix (clean-obs) envs, where extraction
        # reproduces the current obs from state — harmless for non-done envs,
        # correct (reset-start obs) for done envs. For PIXEL envs `_seed_obs`
        # MEMSETS `_obs` to 0; running it here every iteration would zero the
        # driver's `prev_obs` snapshot on the next loop top, corrupting every
        # transition (prev_obs all-zero vs the normalized rendered next_obs)
        # → uniform collapse. Pixel obs already lives in the workspace frame
        # stack and is rewritten by the next `step_batch`, so leave `_obs` as
        # the just-stepped observation. (Done pixel envs carry ~FRAME_STACK
        # stale frames into the new episode — a minor boundary effect, not a
        # training-killer.)
        comptime if Self._OBS_IS_STATE_PREFIX:
            self._seed_obs(c)

    def obs_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._terminated.unsafe_ptr())
