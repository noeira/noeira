"""N-step transition buffer — accumulates rewards across N env steps
and emits compressed transitions `(s_t, a_t, R_n, s_{t+n}, done_any)`
where `R_n = r_t + γ·r_{t+1} + … + γ^{n−1}·r_{t+n−1}`.

Phase C.2 — primitive for n-step replay. Base replay buffers
(`CPUReplay`, `GPUReplay`) store these compressed transitions; the
agent's TD target uses `γ^N` (not `γ`) for the bootstrap term:
`y = R_n + γ^N · Q(s_{t+n})`. Useful for Rainbow-style returns and
SAC variants that benefit from longer-horizon targets.

Mirrors `noeira/deep_agents/core/replay/nstep_buffer.mojo` but
extended for continuous-action support: actions are `[ACT]` vectors
(not scalars), tracked as `Array[Scalar[DT], ACT]` on CPU and
as `[N_ENVS * N * ACT]` device buffer rings on GPU.

Two surfaces:

  - `NStepBuffer[N, OBS, ACT]` — single-env CPU streaming. Caller
    invokes `add(s, a, r, s', done)` and receives an
    `NStepTransition` result; when `result.valid == True`, push it
    into the base replay via the standard `add(...)` API.

  - `GPUNStepBuffer[N, OBS, ACT, N_ENVS]` — per-env parallel
    accumulation on GPU. `process(ctx, obs, act, rew, nobs, done)`
    runs one kernel over all `N_ENVS` envs; the internal `out_*`
    device buffers receive emitted compressed transitions plus an
    `out_valid` flag. `store_into[CAP](ctx, buf)` blindly pushes
    all `N_ENVS` slots into a base `GPUReplay[OBS, ACT, CAP]` via
    `add_batch[N_ENVS]` — invalid slots are zero-padded and get
    overwritten as the circular buffer wraps (matches deep_agents).

Done semantics: when `done` is set, the kernel flushes the current
partial chain immediately and resets the ring. Subsequent steps
start fresh from the new episode.

Trainer integration is left to the caller: nn keeps the primitive
narrow and lets `SACTrainer.record` / `record_batch_gpu` users wrap
the call as needed. A `use_n_step: Bool` Saveable config flag in
`SACConfig` is a future C.2b chunk.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB
from ..training.replay_buffer import ReplayBuffer
from noeira.core.tiles import Tile, tile


# ──────────────────────────────────────────────────────────────────────
# CPU surface: NStepTransition + NStepBuffer
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct NStepTransition[OBS: Int, ACT: Int](Movable & Deinitable):
    """Compressed n-step transition returned by `NStepBuffer.add`.

    `valid=False` means the buffer accumulated but didn't emit (ring
    not yet full and not done). `valid=True` means the caller should
    push this transition into the base replay via the standard
    `add(obs, action, reward, next_obs, done)` API.

    Storage is heap `List`s, NOT `Array` (compile-memory critical):
    with pixel obs (OBS=28224) a by-value Array makes this a ~225 KB
    stack aggregate returned through every `NStepBuffer.add` — the -O2/-O3
    scalarization passes explode superlinearly in OBS on those copies
    (>30 GB compiler footprint on the Rainbow Atari pixel example; -O0 was
    the only build that fit). Lists keep the payload on the heap and the
    struct register-sized. `valid=False` transitions carry EMPTY lists —
    callers must check `valid` before indexing.
    """

    var valid: Bool
    var obs: List[Scalar[DT]]
    var action: List[Scalar[DT]]
    var reward: Scalar[DT]
    var next_obs: List[Scalar[DT]]
    var done: Bool

    @staticmethod
    def empty() -> Self:
        return Self(
            valid=False,
            obs=List[Scalar[DT]](),
            action=List[Scalar[DT]](),
            reward=Scalar[DT](0.0),
            next_obs=List[Scalar[DT]](),
            done=False,
        )


@fieldwise_init
struct NStepBuffer[N: Int, OBS: Int, ACT: Int](Movable & Deinitable):
    """CPU streaming n-step buffer for a single env.

    Accumulates up to `N` transitions; on emit (ring full or `done`),
    computes the discounted n-step return via Horner's rule and
    produces an `NStepTransition` for the caller to push into a base
    replay buffer. `done` flushes the current partial chain
    immediately and resets the ring.

    Storage: flat host `List` allocations sized at construction.
    `count` tracks current ring fill (0..N).
    """

    var obs: List[Scalar[DT]]  # [N * OBS]
    var actions: List[Scalar[DT]]  # [N * ACT]
    var rewards: List[Scalar[DT]]  # [N]
    var gamma: Scalar[DT]
    var count: Int

    @staticmethod
    def new(gamma: Scalar[DT] = Scalar[DT](0.99)) -> Self:
        return Self(
            obs=List[Scalar[DT]](
                length=Self.N * Self.OBS,
                fill=Scalar[DT](0.0),
            ),
            actions=List[Scalar[DT]](
                length=Self.N * Self.ACT,
                fill=Scalar[DT](0.0),
            ),
            rewards=List[Scalar[DT]](
                length=Self.N,
                fill=Scalar[DT](0.0),
            ),
            gamma=gamma,
            count=0,
        )

    def _compute_return(self, n: Int) -> Scalar[DT]:
        """R = r_0 + γ·r_1 + … + γ^{n−1}·r_{n−1} via Horner's rule."""
        var r = Scalar[DT](0.0)
        for i in range(n - 1, -1, -1):
            r = r * self.gamma + self.rewards[i]
        return r

    def _shift_left(mut self):
        """Drop the oldest slot, shift everything left by one."""
        for i in range(Self.N - 1):
            for d in range(Self.OBS):
                self.obs[i * Self.OBS + d] = self.obs[(i + 1) * Self.OBS + d]
            for j in range(Self.ACT):
                self.actions[i * Self.ACT + j] = self.actions[
                    (i + 1) * Self.ACT + j
                ]
            self.rewards[i] = self.rewards[i + 1]

    def add(
        mut self,
        ref obs_p: List[Scalar[DT]],
        ref action_p: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs_p: List[Scalar[DT]],
        done: Bool,
    ) -> NStepTransition[Self.OBS, Self.ACT]:
        """Push one env-step transition. Returns an emitted n-step
        transition (`valid=True`) once ready, otherwise `valid=False`.

        Args:
            obs_p: Current obs list of length OBS.
            action_p: Action list of length ACT.
            reward: Per-step reward.
            next_obs_p: Next obs list of length OBS.
            done: Episode-done flag.
        """
        var c = self.count
        for d in range(Self.OBS):
            self.obs[c * Self.OBS + d] = obs_p[d]
        for j in range(Self.ACT):
            self.actions[c * Self.ACT + j] = action_p[j]
        self.rewards[c] = reward
        c += 1
        self.count = c

        if done or c == Self.N:
            var r_n = self._compute_return(c)
            var s0 = List[Scalar[DT]](length=Self.OBS, fill=Scalar[DT](0.0))
            var a0 = List[Scalar[DT]](length=Self.ACT, fill=Scalar[DT](0.0))
            for d in range(Self.OBS):
                s0[d] = self.obs[d]
            for j in range(Self.ACT):
                a0[j] = self.actions[j]
            var sn = List[Scalar[DT]](length=Self.OBS, fill=Scalar[DT](0.0))
            for d in range(Self.OBS):
                sn[d] = next_obs_p[d]

            if done:
                self.count = 0
            else:
                self._shift_left()
                self.count = Self.N - 1

            return NStepTransition[Self.OBS, Self.ACT](
                valid=True,
                obs=s0^,
                action=a0^,
                reward=r_n,
                next_obs=sn^,
                done=done,
            )
        return NStepTransition[Self.OBS, Self.ACT].empty()

    def reset(mut self):
        """Reset the ring (e.g. on episode boundary if the caller is
        feeding `done=False` transitions and needs an explicit
        flush)."""
        self.count = 0


# ──────────────────────────────────────────────────────────────────────
# GPU surface: kernel + GPUNStepBuffer.
# ──────────────────────────────────────────────────────────────────────


def _nstep_decide_kernel[
    N_ENVS: Int,
    N: Int,
](
    in_rew: Tile[DT, N_ENVS],
    in_done: Tile[DT, N_ENVS],
    ring_rew: Tile[DT, N_ENVS, N],
    counts: Tile[DType.int32, N_ENVS],
    out_rew: Tile[DT, N_ENVS],
    out_done: Tile[DT, N_ENVS],
    out_valid: Tile[DType.int32, N_ENVS],
    proc_slot: Tile[DType.int32, N_ENVS],
    gamma: Scalar[DT],
):
    """Phase 1 of the n-step process split — one thread per env (cheap
    scalar + reward-ring work).

    Appends `in_rew` at ring slot `count`, decides emit (done OR ring
    full), computes the Horner `R_n`, writes `out_rew/out_done/out_valid`
    plus the new `count`, and shifts the reward ring on a non-terminal
    emit. Publishes the append slot (the PRE-increment count) into
    `proc_slot` so the element-parallel copy kernel reads a stable index
    instead of racing on `counts` (which this kernel overwrites). The
    obs/act rings + obs emit copies are handled by `_nstep_copy_kernel`,
    which recomputes the same emit decision from the read-only inputs.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var c = Int(counts[e])
    proc_slot[e] = Int32(c)
    ring_rew[e, c] = in_rew[e]
    var newc = c + 1

    var is_done = in_done[e] > Scalar[DT](0.5)

    if is_done or newc == N:
        var r_n = Scalar[DT](0.0)
        for i in range(newc - 1, -1, -1):
            r_n = r_n * gamma + rebind[Scalar[DT]](ring_rew[e, i])
        out_rew[e] = r_n
        out_done[e] = Scalar[DT](1.0) if is_done else Scalar[DT](0.0)
        out_valid[e] = Int32(1)

        if is_done:
            counts[e] = Int32(0)
        else:
            for i in range(N - 1):
                ring_rew[e, i] = ring_rew[e, i + 1]
            counts[e] = Int32(N - 1)
    else:
        out_valid[e] = Int32(0)
        counts[e] = Int32(newc)


def _nstep_copy_kernel[
    N_ENVS: Int,
    N: Int,
    OBS: Int,
    ACT: Int,
](
    in_obs: Tile[DT, N_ENVS, OBS],
    in_act: Tile[DT, N_ENVS, ACT],
    in_nobs: Tile[DT, N_ENVS, OBS],
    in_done: Tile[DT, N_ENVS],
    ring_obs: Tile[DT, N_ENVS, N * OBS],
    ring_act: Tile[DT, N_ENVS, N * ACT],
    out_obs: Tile[DT, N_ENVS, OBS],
    out_act: Tile[DT, N_ENVS, ACT],
    out_nobs: Tile[DT, N_ENVS, OBS],
    proc_slot: Tile[DType.int32, N_ENVS],
):
    """Phase 2 of the n-step process split — element-parallel over
    (env × OBS element); one thread per `(e, d)`.

    Each thread owns column `d` of env `e`'s ring, so the append, the
    slot-0 emit read, and the in-place left-shift all happen in program
    order *within the thread* — no cross-thread race — while the heavy
    OBS copies run at full occupancy + coalesced (replaces the old
    one-thread-per-env serial OBS loops, ~43% of GPU time on Pong-pixel,
    see `project_rainbow_pong_pixel_replay_gather_bottleneck`).

    Reads the append slot from `proc_slot` (published by the decide
    kernel) and recomputes `is_done`/emit from the read-only inputs, so
    it stays bit-identical to the original fused kernel. The `d < ACT`
    threads ride the OBS grid to carry the action lanes. Requires
    OBS >= ACT (always true for these envs)."""
    comptime assert OBS >= ACT, "_nstep_copy_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= N_ENVS * OBS:
        return
    var e = t // OBS
    var d = t % OBS

    var c = Int(proc_slot[e])
    # Append obs/act at ring slot c (mirrors the decide kernel's rew append).
    ring_obs[e, c * OBS + d] = in_obs[e, d]
    if d < ACT:
        ring_act[e, c * ACT + d] = in_act[e, d]

    var newc = c + 1
    var is_done = in_done[e] > Scalar[DT](0.5)

    if is_done or newc == N:
        # Emit slot-0 (read before the shift below clobbers it).
        out_obs[e, d] = rebind[Scalar[DT]](ring_obs[e, d])
        out_nobs[e, d] = rebind[Scalar[DT]](in_nobs[e, d])
        if d < ACT:
            out_act[e, d] = rebind[Scalar[DT]](ring_act[e, d])

        if not is_done:
            for i in range(N - 1):
                ring_obs[e, i * OBS + d] = rebind[Scalar[DT]](
                    ring_obs[e, (i + 1) * OBS + d]
                )
                if d < ACT:
                    ring_act[e, i * ACT + d] = rebind[Scalar[DT]](
                        ring_act[e, (i + 1) * ACT + d]
                    )


@fieldwise_init
struct GPUNStepBuffer[N: Int, OBS: Int, ACT: Int, N_ENVS: Int](
    Movable & Deinitable
):
    """GPU per-env streaming n-step buffer for `N_ENVS` parallel envs.

    Internal storage:
      * `ring_obs`, `ring_act`, `ring_rew` — per-env rings of N slots.
      * `counts` — per-env step counter (0..N) on device.
      * `out_*` — compressed emit buffers, overwritten each call.
      * `out_valid` — 1 if env emitted this call, 0 otherwise.

    Usage:
      ```
      var ns = GPUNStepBuffer[3, 3, 1, 8].new(ctx, gamma=Scalar[DT](0.99))
      # per loop iter:
      ns.process(ctx, obs, act, rew, nobs, done)
      ns.store_into[CAP](ctx, replay_buf)
      ```

    Blind-store semantics in `store_into`: pushes ALL N_ENVS slots
    regardless of `out_valid`. Slots where the env wasn't ready hold
    zero-padded data; the circular buffer overwrites them as it wraps.
    Matches `deep_agents`' `GPUNStepBuffer.store_into`. For exact
    compaction (skip invalid), the caller can D2H `out_valid` and call
    `buf.add` per-valid; nn keeps the simpler path here.
    """

    var ring_obs: DeviceBuffer[DT]
    var ring_act: DeviceBuffer[DT]
    var ring_rew: DeviceBuffer[DT]
    var counts: DeviceBuffer[DType.int32]
    # Append slot (pre-increment count) published by the decide kernel for
    # the element-parallel copy kernel to read — avoids racing on `counts`.
    var proc_slot: DeviceBuffer[DType.int32]

    var out_obs: DeviceBuffer[DT]
    var out_act: DeviceBuffer[DT]
    var out_rew: DeviceBuffer[DT]
    var out_nobs: DeviceBuffer[DT]
    var out_done: DeviceBuffer[DT]
    var out_valid: DeviceBuffer[DType.int32]

    var gamma: Scalar[DT]

    @staticmethod
    def new(
        ctx: DeviceContext,
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        var ring_obs = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.N * Self.OBS
        )
        var ring_act = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.N * Self.ACT
        )
        var ring_rew = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.N)
        var counts = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        var proc_slot = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        ring_obs.enqueue_fill(Scalar[DT](0.0))
        ring_act.enqueue_fill(Scalar[DT](0.0))
        ring_rew.enqueue_fill(Scalar[DT](0.0))
        counts.enqueue_fill(Int32(0))
        proc_slot.enqueue_fill(Int32(0))

        var out_obs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS)
        var out_act = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.ACT)
        var out_rew = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        var out_nobs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS)
        var out_done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        var out_valid = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        out_obs.enqueue_fill(Scalar[DT](0.0))
        out_act.enqueue_fill(Scalar[DT](0.0))
        out_rew.enqueue_fill(Scalar[DT](0.0))
        out_nobs.enqueue_fill(Scalar[DT](0.0))
        out_done.enqueue_fill(Scalar[DT](0.0))
        out_valid.enqueue_fill(Int32(0))

        return Self(
            ring_obs=ring_obs^,
            ring_act=ring_act^,
            ring_rew=ring_rew^,
            counts=counts^,
            proc_slot=proc_slot^,
            out_obs=out_obs^,
            out_act=out_act^,
            out_rew=out_rew^,
            out_nobs=out_nobs^,
            out_done=out_done^,
            out_valid=out_valid^,
            gamma=gamma,
        )

    def process(
        mut self,
        ctx: DeviceContext,
        obs: DeviceBuffer[DT],
        act: DeviceBuffer[DT],
        rew: DeviceBuffer[DT],
        nobs: DeviceBuffer[DT],
        done: DeviceBuffer[DT],
    ) raises:
        """One kernel: ring update + emit decision for all N_ENVS envs.
        Outputs land in the internal `out_*` device buffers;
        `out_valid[e]` is 1 if env `e` emitted this call."""
        var in_obs_lt = rebind[
            Tile[DT, Self.N_ENVS, Self.OBS]
        ](
            tile[Self.N_ENVS, Self.OBS](obs)
        )
        var in_act_lt = rebind[
            Tile[DT, Self.N_ENVS, Self.ACT]
        ](
            tile[Self.N_ENVS, Self.ACT](act)
        )
        var in_rew_lt = rebind[
            Tile[DT, Self.N_ENVS]
        ](
            tile[Self.N_ENVS](rew)
        )
        var in_nobs_lt = rebind[
            Tile[DT, Self.N_ENVS, Self.OBS]
        ](
            tile[Self.N_ENVS, Self.OBS](nobs)
        )
        var in_done_lt = rebind[
            Tile[DT, Self.N_ENVS]
        ](
            tile[Self.N_ENVS](done)
        )

        var ring_obs_lt = tile[Self.N_ENVS, Self.N * Self.OBS](self.ring_obs)
        var ring_act_lt = tile[Self.N_ENVS, Self.N * Self.ACT](self.ring_act)
        var ring_rew_lt = tile[Self.N_ENVS, Self.N](self.ring_rew)
        var counts_lt = tile[Self.N_ENVS](self.counts)

        var out_obs_lt = tile[Self.N_ENVS, Self.OBS](self.out_obs)
        var out_act_lt = tile[Self.N_ENVS, Self.ACT](self.out_act)
        var out_rew_lt = tile[Self.N_ENVS](self.out_rew)
        var out_nobs_lt = tile[Self.N_ENVS, Self.OBS](self.out_nobs)
        var out_done_lt = tile[Self.N_ENVS](self.out_done)
        var out_valid_lt = tile[Self.N_ENVS](self.out_valid)
        var proc_slot_lt = tile[Self.N_ENVS](self.proc_slot)

        # Phase 1 — decide (one thread per env): rew ring + emit decision +
        # counts, publishing the append slot into proc_slot. Must precede
        # phase 2, which reads proc_slot (enqueue order = execution order).
        comptime n_blocks_decide = (Self.N_ENVS + TPB - 1) // TPB
        comptime decide_kernel = _nstep_decide_kernel[Self.N_ENVS, Self.N]
        ctx.enqueue_function[decide_kernel](
            in_rew_lt,
            in_done_lt,
            ring_rew_lt,
            counts_lt,
            out_rew_lt,
            out_done_lt,
            out_valid_lt,
            proc_slot_lt,
            self.gamma,
            grid_dim=n_blocks_decide,
            block_dim=TPB,
        )

        # Phase 2 — copy (element-parallel over N_ENVS × OBS): obs/act ring
        # append + emit copies + in-place shift, each (e, d) thread owning
        # its ring column.
        comptime n_blocks_copy = (Self.N_ENVS * Self.OBS + TPB - 1) // TPB
        comptime copy_kernel = _nstep_copy_kernel[
            Self.N_ENVS,
            Self.N,
            Self.OBS,
            Self.ACT,
        ]
        ctx.enqueue_function[copy_kernel](
            in_obs_lt,
            in_act_lt,
            in_nobs_lt,
            in_done_lt,
            ring_obs_lt,
            ring_act_lt,
            out_obs_lt,
            out_act_lt,
            out_nobs_lt,
            proc_slot_lt,
            grid_dim=n_blocks_copy,
            block_dim=TPB,
        )

    def store_into[
        S: ReplayBuffer
    ](self, ctx: DeviceContext, mut buf: S,) raises:
        """Blind-store all N_ENVS slots into any device-backed
        `ReplayBuffer` via its `add_batch[N_ENVS]`. Invalid slots
        (`out_valid[e] == 0`) contain zero-padded data and get
        overwritten as the buffer wraps. Matches deep_agents semantics.

        Generic over `S: ReplayBuffer` — `add_batch` is the only
        capability used, and it's a trait method (default-raises for CPU
        backends), so one method covers both `GPUReplay` (uniform) and
        `GPUPrioritizedReplay` (the PER `max_priority^alpha` slot init
        lives inside *its* `add_batch`, not here). Breaking the former
        `n_step_replay → gpu_replay` import edge is what lets the generic
        sample blocks subsume the GPU-only sample blocks.
        """
        comptime assert (
            S.OBS == Self.OBS and S.ACT == Self.ACT
        ), "store_into: buffer OBS/ACT must match the n-step buffer's"
        buf.add_batch[Self.N_ENVS](
            ctx,
            self.out_obs,
            self.out_act,
            self.out_rew,
            self.out_nobs,
            self.out_done,
        )

    def reset(mut self, ctx: DeviceContext) raises:
        """Reset all per-env counts to 0 (e.g. on episode boundary
        when the kernel's auto-reset-on-done path isn't used)."""
        self.counts.enqueue_fill(Int32(0))
