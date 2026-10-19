"""GPUSequenceReplay[OBS, ACT, CAP] — device-resident circular sequence
replay buffer; the GPU sibling of `SequenceReplay`.

Single flat ring on device (same layout / semantics as the CPU buffer, not
the per-env layout of the deep_agents v1 buffer) so CPU and GPU are true
siblings — same start distribution, same window contents, only the RNG
differs (Philox here vs `random_float64` on CPU, so no bit-identity; there
is no GPU bit-identity baseline, mirroring `GPUReplay`).

Stored layout (4 DeviceBuffers, each CAP slots):
  obs  [CAP, OBS] row-major
  act  [CAP, ACT] row-major
  rew  [CAP]
  dne  [CAP]

Surface (conforms to `SequenceReplayBuffer`):
  - `make["gpu"](ctx)`                  — allocate + zero device storage.
  - `record(s, a, r, d)`                — host ptrs → 1-slot staging →
                                          store-one kernel at `pos`.
  - `record_batch[N_ENVS](ctx, …)`      — device sources → one store kernel
                                          writing N_ENVS lockstep slots.
  - `sample_batch_dev[B, T](ctx, …)`    — Philox window sample directly into
                                          caller device buffers (no D2H).
  - `sample_batch[B, T](host ptrs)`     — device sample then D2H copy-out via
                                          the stored `ctx`; lets a GPU buffer
                                          feed the current CPU world-model
                                          pipeline.

RNG: each `sample_batch_dev` bumps `_rng_offset` so back-to-back samples draw
disjoint Philox streams (pattern from `GPUReplay`).

Kernel tiling (Part C of docs/DEVICE_PER_TREE_PLAN.md): store and sample are
element-parallel — one thread per (env × OBS element) for the store, one per
(window × frame × OBS element) for the sample, with the window start drawn
once per window by a tiny pre-kernel so all of a window's threads agree.
This replaces the original one-thread-per-env / one-thread-per-window
kernels whose serial `(T+1)·OBS` copy loops were the same anti-pattern fixed
in gpu_replay/n_step (latent ~1000× bottleneck for image-obs world models).
Same elements, re-tiled → bit-identical.

Obs storage dtype (`OBS_STORE_DT_`, default `DT` — bit-identical): set
`DType.uint8` for pixel obs to quantize on store / dequantize on sample
(4× window capacity; lossless for exact `k/255` inputs; pixel-only —
see `noeira/data/quantize.mojo`).
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT, TPB
from noeira.data.quantize import _obs_quant, _obs_dequant
from .sequence_replay_buffer import SequenceReplayBuffer
from noeira.core.tiles import Tile, tile


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _seq_store_one_kernel[
    OBS: Int, ACT: Int, CAP: Int, SDT: DType = DT
](
    stage_s: Tile[DT, OBS],
    stage_a: Tile[DT, ACT],
    stage_r: Tile[DT, 1],
    stage_d: Tile[DT, 1],
    buf_s: Tile[SDT, CAP, OBS],
    buf_a: Tile[DT, CAP, ACT],
    buf_r: Tile[DT, CAP],
    buf_d: Tile[DT, CAP],
    slot: Int32,
):
    """Element-parallel store: thread d writes stage[d] → buf[slot, d] for obs
    (OBS lanes) and act (ACT lanes); global thread 0 writes the scalar rew / dne.
    Launched grid=(ceil(max(OBS,ACT)/TPB),), block=(TPB,) — grid-strided across
    blocks so OBS may exceed the 1024-thread block cap (pixel obs)."""
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    var s = Int(slot)
    if d < OBS:
        buf_s[s, d] = _obs_quant[SDT](rebind[Scalar[DT]](stage_s[d]))
    if d < ACT:
        buf_a[s, d] = stage_a[d]
    if d == 0:
        buf_r[s] = stage_r[0]
        buf_d[s] = stage_d[0]


def _seq_store_batch_kernel[
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
    CAP: Int,
    SDT: DType = DT,
](
    src_obs: Tile[DT, N_ENVS, OBS],
    src_act: Tile[DT, N_ENVS, ACT],
    src_rew: Tile[DT, N_ENVS],
    src_dne: Tile[DT, N_ENVS],
    buf_s: Tile[SDT, CAP, OBS],
    buf_a: Tile[DT, CAP, ACT],
    buf_r: Tile[DT, CAP],
    buf_d: Tile[DT, CAP],
    start_pos: Int32,
):
    """Element-parallel batched store: one thread per (env × obs element).

    Thread `t` → env `e = t // OBS`, element `d = t % OBS`; writes env e's
    transition into slot `(start_pos + e) % CAP`. The `d < ACT` threads
    carry act; the `d == 0` thread writes the scalar rew/dne. Launched as
    `grid=(ceil(N_ENVS·OBS/TPB),), block=(TPB,)`.

    Replaces the old one-thread-per-env kernel (serial OBS loop) — same
    writes, re-tiled for occupancy + coalescing; bit-identical. Requires
    OBS >= ACT (true for every env here; obs dominates)."""
    comptime assert OBS >= ACT, "_seq_store_batch_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= N_ENVS * OBS:
        return
    var e = t // OBS
    var d = t % OBS
    var slot = (Int(start_pos) + e) % CAP
    buf_s[slot, d] = _obs_quant[SDT](rebind[Scalar[DT]](src_obs[e, d]))
    if d < ACT:
        buf_a[slot, d] = src_act[e, d]
    if d == 0:
        buf_r[slot] = src_rew[e]
        buf_d[slot] = src_dne[e]


def _increment_rng_offset_kernel[
    B: Int
](offset: Tile[DType.uint64, 1],):
    """Bump the device RNG offset by `2 * B` (the host stride). Launch
    grid=(1,), block=(1,); enqueued after the sample kernel so the
    offset sequence is bit-identical to the old host `_rng_offset += 2·B`
    — now device-resident for CUDA-graph capture."""
    if Int(thread_idx.x) == 0:
        offset[0] = offset[0] + UInt64(B * 2)


def _seq_draw_starts_kernel[
    B: Int
](
    starts: Tile[DType.int32, B],
    n_valid: Int32,
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Pre-kernel: thread `b` draws window start `s ∈ [0, n_valid)` via
    Philox into `starts[b]`. The same draw math (seed + b, offset) the old
    fused per-window kernel used, hoisted out so the element-parallel
    gather's threads all read one agreed start per window. Launched
    grid=(ceil(B/TPB),), block=(TPB,)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= B:
        return
    var nv = Int(n_valid)
    if nv < 1:
        nv = 1
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(b), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var s = Int(u * Float32(nv))
    if s >= nv:
        s = nv - 1
    if s < 0:
        s = 0
    starts[b] = Int32(s)


def _seq_set_start_kernel[
    B: Int
](
    starts: Tile[DType.int32, B],
    row: Int32,
    value: Int32,
):
    """Overwrite `starts[row]` with a host-chosen (online-queue) window start.
    Enqueued eagerly between the draw and gather kernels; the Philox draw
    still runs for every row, so the RNG offset sequence is unchanged whether
    or not online rows are served. Launch grid=(1,), block=(1,)."""
    if Int(thread_idx.x) == 0:
        starts[Int(row)] = value


def _seq_sample_kernel[
    B: Int,
    T: Int,
    OBS: Int,
    ACT: Int,
    CAP: Int,
    SDT: DType = DT,
](
    buf_s: Tile[SDT, CAP, OBS],
    buf_a: Tile[DT, CAP, ACT],
    buf_r: Tile[DT, CAP],
    buf_d: Tile[DT, CAP],
    out_obs: Tile[DT, B, T + 1, OBS],
    out_act: Tile[DT, B, T, ACT],
    out_rew: Tile[DT, B, T],
    out_dne: Tile[DT, B, T],
    starts: Tile[DType.int32, B],
    origin: Int32,
):
    """Element-parallel window gather: one thread per
    (window × frame × OBS element), i.e. the `[B, T+1, OBS]` output
    flattened to a 1-D grid. Thread `t` → `(b, k, i)`; reads the window
    start from `starts[b]` (drawn by `_seq_draw_starts_kernel`) and copies
    one obs element from physical slot `(origin + s_b + k) % CAP`. The
    `k < T` frames also carry act (`i < ACT` threads) and the scalar
    rew/dne (`i == 0` thread) — the same ride-along trick as
    `_gather_batch_kernel`.

    Matches the CPU `SequenceReplay.sample_batch` index math exactly; same
    elements as the old one-thread-per-window kernel (which serialised
    `(T+1)·OBS` copies per thread — the audit's HIGH find for image-obs
    world models), only re-tiled → bit-identical. Requires OBS >= ACT.
    Launched grid=(ceil(B·(T+1)·OBS/TPB),), block=(TPB,)."""
    comptime assert OBS >= ACT, "_seq_sample_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= B * (T + 1) * OBS:
        return
    var b = t // ((T + 1) * OBS)
    var rem = t % ((T + 1) * OBS)
    var k = rem // OBS
    var i = rem % OBS

    var s = Int(starts[b])
    var phys = (Int(origin) + s + k) % CAP
    out_obs[b, k, i] = _obs_dequant[SDT](rebind[Scalar[SDT]](buf_s[phys, i]))
    if k < T:
        if i < ACT:
            out_act[b, k, i] = buf_a[phys, i]
        if i == 0:
            out_rew[b, k] = buf_r[phys]
            out_dne[b, k] = buf_d[phys]


def _seq_store_fst_kernel[CAP: Int](
    stage_fst: Tile[DT, 1],
    buf_fst: Tile[DT, CAP],
    pos: Int32,
):
    """Store the staged `is_first` flag into the ring at `pos` (1 thread)."""
    if Int(block_dim.x * block_idx.x + thread_idx.x) != 0:
        return
    buf_fst[Int(pos)] = rebind[Scalar[DT]](stage_fst[0])


def _seq_gather_fst_kernel[B: Int, T: Int, CAP: Int](
    buf_fst: Tile[DT, CAP],
    fst_out: Tile[DT, B, T + 1],
    starts: Tile[DType.int32, B],
    origin: Int32,
):
    """Per-(window × obs-frame) gather of the `is_first` flag — same window
    starts + `phys = (origin + s_b + k) % CAP` as `_seq_sample_kernel`, so the
    `[B, T+1]` fst frames align with the obs frames. Matches the CPU
    `SequenceReplay.sample_batch_fst` index math."""
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= B * (T + 1):
        return
    var b = t // (T + 1)
    var k = t % (T + 1)
    var phys = (Int(origin) + Int(starts[b]) + k) % CAP
    fst_out[b, k] = rebind[Scalar[DT]](buf_fst[phys])


# ──────────────────────────────────────────────────────────────────────
# GPUSequenceReplay struct.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct GPUSequenceReplay[
    OBS_: Int, ACT_: Int, CAP_: Int, OBS_STORE_DT_: DType = DT
](SequenceReplayBuffer):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_
    comptime SDT = Self.OBS_STORE_DT_

    # Device-resident circular storage (obs in OBS_STORE_DT_).
    var obs: DeviceBuffer[Self.SDT]
    var act: DeviceBuffer[DT]
    var rew: DeviceBuffer[DT]
    var dne: DeviceBuffer[DT]
    # Per-frame `is_first` ring [CAP] (device twin of `SequenceReplay.fst`):
    # 1.0 if this slot's obs begins a new episode. Lets the WM key its carry
    # reset on `fst` (frame t+1) instead of `dne`, and lets `record_terminal`
    # store a genuine terminal obs while the post-terminal reset frame stays
    # flagged first. Sampled by `sample_batch_fst_dev`.
    var fst: DeviceBuffer[DT]

    # Single-transition device staging (used by `record`).
    var stage_obs: DeviceBuffer[DT]
    var stage_act: DeviceBuffer[DT]
    var stage_rew: DeviceBuffer[DT]
    var stage_dne: DeviceBuffer[DT]
    var stage_fst: DeviceBuffer[DT]

    # Per-window start scratch for the element-parallel sample (sized to
    # `batch_capacity`, mirroring `GPUReplay.indices`). Written by
    # `_seq_draw_starts_kernel`, read by `_seq_sample_kernel`.
    var starts: DeviceBuffer[DType.int32]
    var batch_capacity: Int

    # Host scratch for scalar staging so `record` issues uniform H2D copies.
    # Owning RAII `List` (was a raw `alloc`'d `MutAnyOrigin` pointer that was
    # never freed — a leak). The `enqueue_copy` sites pass `.unsafe_ptr()` (a
    # plain unpinned host pointer → the EAGER copy overload, capturing the
    # value at enqueue time); a `HostBuffer` here would take the async-DMA
    # overload and race the single-slot reuse in `record`.
    var _h_rew: List[Scalar[DT]]
    var _h_dne: List[Scalar[DT]]
    var _h_fst: List[Scalar[DT]]

    # Stored context — needed by the host `sample_batch` bridge (no ctx arg).
    var ctx: DeviceContext

    # CPU bookkeeping.
    var size: Int
    var pos: Int
    # True ⇒ the next `record`ed frame begins a new episode (fst=1). Starts
    # True; a done=1 `record` arms it; `record_terminal` leaves it armed.
    var pending_first: Bool
    var rng_seed: UInt64
    # Slice 5 — device-resident Philox offset (was a host `UInt64`). The
    # sample kernel reads `_rng_offset_dev[0]`; `_increment_rng_offset_kernel`
    # advances it on-device → CUDA-graph capturable. Offset sequence
    # (k·2·B) unchanged → sampled windows bit-identical to the host path.
    var _rng_offset_dev: DeviceBuffer[DType.uint64]
    # Online queue (reference replay `online: True`) — host-side twin of
    # `SequenceReplay.online_*`: every `online_every`-th appended frame
    # enqueues the newest window's END slot (physical); `sample_batch_dev`
    # serves them into the first batch rows via `_seq_set_start_kernel`
    # before the uniform rows. 0 = off (default; all consumers unchanged).
    # Single-env `record`/`record_terminal` only (`record_batch` unsupported).
    var online_every: Int
    var online_tick: Int
    var online_q: List[Int]

    @staticmethod
    def new(ctx: DeviceContext, batch_capacity: Int = 4096) raises -> Self:
        """Allocate + zero device storage and the 1-slot staging buffers.
        `batch_capacity` caps the per-call `B` of `sample_batch_dev`
        (sizes the `starts` scratch)."""
        var s = ctx.enqueue_create_buffer[Self.SDT](Self.CAP * Self.OBS)
        var a = ctx.enqueue_create_buffer[DT](Self.CAP * Self.ACT)
        var r = ctx.enqueue_create_buffer[DT](Self.CAP)
        var d = ctx.enqueue_create_buffer[DT](Self.CAP)
        var f = ctx.enqueue_create_buffer[DT](Self.CAP)
        s.enqueue_fill(Scalar[Self.SDT](0))
        a.enqueue_fill(Scalar[DT](0.0))
        r.enqueue_fill(Scalar[DT](0.0))
        d.enqueue_fill(Scalar[DT](0.0))
        f.enqueue_fill(Scalar[DT](0.0))

        var stage_s = ctx.enqueue_create_buffer[DT](Self.OBS)
        var stage_a = ctx.enqueue_create_buffer[DT](Self.ACT)
        var stage_r = ctx.enqueue_create_buffer[DT](1)
        var stage_d = ctx.enqueue_create_buffer[DT](1)
        var stage_f = ctx.enqueue_create_buffer[DT](1)

        var starts = ctx.enqueue_create_buffer[DType.int32](batch_capacity)
        starts.enqueue_fill(Int32(0))

        var rng_off = ctx.enqueue_create_buffer[DType.uint64](1)
        rng_off.enqueue_fill(UInt64(0))

        var hr = List[Scalar[DT]](length=1, fill=Scalar[DT](0))
        var hd = List[Scalar[DT]](length=1, fill=Scalar[DT](0))
        var hf = List[Scalar[DT]](length=1, fill=Scalar[DT](0))

        return Self(
            obs=s^,
            act=a^,
            rew=r^,
            dne=d^,
            fst=f^,
            stage_obs=stage_s^,
            stage_act=stage_a^,
            stage_rew=stage_r^,
            stage_dne=stage_d^,
            stage_fst=stage_f^,
            starts=starts^,
            batch_capacity=batch_capacity,
            _h_rew=hr^,
            _h_dne=hd^,
            _h_fst=hf^,
            ctx=ctx,
            size=0,
            pos=0,
            pending_first=True,
            rng_seed=UInt64(0xC0FFEE_DECADE_0042),
            _rng_offset_dev=rng_off^,
            online_every=0,
            online_tick=0,
            online_q=List[Int](),
        )

    # ─── SequenceReplayBuffer trait surface ──────────────────────────────

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Trait factory. `ctx` required (device storage); raises if None."""
        comptime assert target == "gpu", (
            "GPUSequenceReplay is the GPU backend; use SequenceReplay for"
            ' target == "cpu"'
        )
        if not ctx:
            raise Error(
                "GPUSequenceReplay.make: ctx required for device storage"
            )
        return Self.new(ctx.value())

    def count(self) -> Int:
        return self.size

    def can_sample[T: Int](self) -> Bool:
        return self.size >= T + 1

    def set_online(mut self, every: Int):
        """Enable the online queue: each `every`-th appended frame enqueues the
        freshest length-`every` window for guaranteed prompt sampling. Pass the
        training window length T. 0 disables (the default)."""
        self.online_every = every

    def _online_tick_append(mut self):
        """Called after every single-env ring append (device twin of
        `SequenceReplay._online_tick_append`; pure host bookkeeping)."""
        if self.online_every <= 0:
            return
        self.online_tick += 1
        if (
            self.online_tick >= self.online_every
            and self.size >= self.online_every + 1
        ):
            if len(self.online_q) >= 32:  # backlog cap: drop oldest
                _ = self.online_q.pop(0)
            self.online_q.append((self.pos - 1 + Self.CAP) % Self.CAP)
            self.online_tick = 0

    def _online_pop_start(mut self, T: Int) -> Int:
        """Pop the oldest queued fresh window; return its LOGICAL start index,
        or -1 if the queue is empty (or the window was evicted)."""
        var origin = 0 if self.size < Self.CAP else self.pos
        while len(self.online_q) > 0:
            var phys_end = self.online_q.pop(0)
            var end_logical = (phys_end - origin + Self.CAP) % Self.CAP
            var s = end_logical - T
            if s >= 0 and end_logical < self.size:
                return s
        return -1

    def record(
        mut self,
        s: Pointer[Scalar[DT], MutAnyOrigin],
        a: Pointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Stage one host transition to device then store it at `pos`."""
        self._h_rew[0] = r
        self._h_dne[0] = d
        self._h_fst[0] = Scalar[DT](1.0) if self.pending_first else Scalar[DT](0.0)
        self.ctx.enqueue_copy(self.stage_obs, s)
        self.ctx.enqueue_copy(self.stage_act, a)
        self.ctx.enqueue_copy(self.stage_rew, self._h_rew.unsafe_ptr())
        self.ctx.enqueue_copy(self.stage_dne, self._h_dne.unsafe_ptr())
        self.ctx.enqueue_copy(self.stage_fst, self._h_fst.unsafe_ptr())

        var stage_s_lt = tile[Self.OBS](self.stage_obs)
        var stage_a_lt = tile[Self.ACT](self.stage_act)
        var stage_r_lt = tile[1](self.stage_rew)
        var stage_d_lt = tile[1](self.stage_dne)
        var buf_s_lt = tile[Self.CAP, Self.OBS](self.obs)
        var buf_a_lt = tile[Self.CAP, Self.ACT](self.act)
        var buf_r_lt = tile[Self.CAP](self.rew)
        var buf_d_lt = tile[Self.CAP](self.dne)

        # grid-strided: max(OBS,ACT) may exceed the 1024-thread block cap (pixels)
        comptime MAXD_R = Self.OBS if Self.OBS > Self.ACT else Self.ACT
        comptime n_blocks_r = (MAXD_R + TPB - 1) // TPB
        comptime kernel = _seq_store_one_kernel[
            Self.OBS, Self.ACT, Self.CAP, Self.SDT
        ]
        self.ctx.enqueue_function[kernel](
            stage_s_lt,
            stage_a_lt,
            stage_r_lt,
            stage_d_lt,
            buf_s_lt,
            buf_a_lt,
            buf_r_lt,
            buf_d_lt,
            Int32(self.pos),
            grid_dim=n_blocks_r,
            block_dim=TPB,
        )
        # store the `is_first` flag into the ring at `pos`.
        self.ctx.enqueue_function[_seq_store_fst_kernel[Self.CAP]](
            tile[1](self.stage_fst),
            tile[Self.CAP](self.fst),
            Int32(self.pos),
            grid_dim=1,
            block_dim=1,
        )
        # a done frame ends the episode → the *next* recorded frame is first.
        self.pending_first = d >= Scalar[DT](0.5)
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1
        self._online_tick_append()

    def record_terminal(
        mut self,
        s: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Store a genuine TERMINAL observation as its own frame (act=0, rew=0,
        dne=0, fst=0). Called right after a `record(done=1)` so the window's obs
        frame *after* the terminal transition is the real terminal state (→ the
        cont head learns `latent(terminal)→0`). `pending_first` is left armed (by
        the preceding done=1 `record`) so the FOLLOWING reset frame — not this
        terminal frame — is flagged `is_first`. Device twin of
        `SequenceReplay.record_terminal`."""
        self._h_rew[0] = Scalar[DT](0.0)
        self._h_dne[0] = Scalar[DT](0.0)
        self._h_fst[0] = Scalar[DT](0.0)
        self.ctx.enqueue_copy(self.stage_obs, s)
        self.ctx.enqueue_copy(self.stage_rew, self._h_rew.unsafe_ptr())
        self.ctx.enqueue_copy(self.stage_dne, self._h_dne.unsafe_ptr())
        self.ctx.enqueue_copy(self.stage_fst, self._h_fst.unsafe_ptr())
        self.stage_act.enqueue_fill(Scalar[DT](0.0))  # terminal frame: act = 0

        comptime MAXD_T = Self.OBS if Self.OBS > Self.ACT else Self.ACT
        comptime n_blocks_t = (MAXD_T + TPB - 1) // TPB
        self.ctx.enqueue_function[
            _seq_store_one_kernel[Self.OBS, Self.ACT, Self.CAP, Self.SDT]
        ](
            tile[Self.OBS](self.stage_obs),
            tile[Self.ACT](self.stage_act),
            tile[1](self.stage_rew),
            tile[1](self.stage_dne),
            tile[Self.CAP, Self.OBS](self.obs),
            tile[Self.CAP, Self.ACT](self.act),
            tile[Self.CAP](self.rew),
            tile[Self.CAP](self.dne),
            Int32(self.pos),
            grid_dim=n_blocks_t,
            block_dim=TPB,
        )
        self.ctx.enqueue_function[_seq_store_fst_kernel[Self.CAP]](
            tile[1](self.stage_fst),
            tile[Self.CAP](self.fst),
            Int32(self.pos),
            grid_dim=1,
            block_dim=1,
        )
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1
        self._online_tick_append()

    def record_batch[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Store `N_ENVS` device-resident transitions in one kernel launch
        (lockstep multi-env collection); slots `(pos + e) % CAP`."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        var src_obs_lt = rebind[
            Tile[DT, N_ENVS, Self.OBS]
        ](tile[N_ENVS, Self.OBS](src_obs))
        var src_act_lt = rebind[
            Tile[DT, N_ENVS, Self.ACT]
        ](tile[N_ENVS, Self.ACT](src_act))
        var src_rew_lt = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](src_rew))
        var src_dne_lt = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](src_dne))
        var buf_s_lt = tile[Self.CAP, Self.OBS](self.obs)
        var buf_a_lt = tile[Self.CAP, Self.ACT](self.act)
        var buf_r_lt = tile[Self.CAP](self.rew)
        var buf_d_lt = tile[Self.CAP](self.dne)

        # Element-parallel: one thread per (env × OBS element).
        comptime n_blocks = (N_ENVS * Self.OBS + TPB - 1) // TPB
        comptime kernel = _seq_store_batch_kernel[
            N_ENVS, Self.OBS, Self.ACT, Self.CAP, Self.SDT
        ]
        ctx.enqueue_function[kernel](
            src_obs_lt,
            src_act_lt,
            src_rew_lt,
            src_dne_lt,
            buf_s_lt,
            buf_a_lt,
            buf_r_lt,
            buf_d_lt,
            Int32(self.pos),
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        self.pos = (self.pos + N_ENVS) % Self.CAP
        self.size += N_ENVS
        if self.size > Self.CAP:
            self.size = Self.CAP

    def sample_batch_dev[
        B: Int,
        T: Int,
    ](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        act_dev: DeviceBuffer[DT],
        rew_dev: DeviceBuffer[DT],
        dne_dev: DeviceBuffer[DT],
    ) raises:
        """Draw `B` length-`T` windows directly into caller device buffers."""
        comptime assert B > 0 and T > 0, "B and T must be > 0"
        if self.size < T + 1:
            raise Error(
                "GPUSequenceReplay.sample_batch_dev: not enough data for a"
                " length-T window"
            )
        if B > self.batch_capacity:
            raise Error(
                "GPUSequenceReplay.sample_batch_dev[B="
                + String(B)
                + "] exceeds batch_capacity="
                + String(self.batch_capacity)
            )
        var origin = 0 if self.size < Self.CAP else self.pos
        var n_valid = self.size - T
        # Drain pending online windows FIRST (host bookkeeping, before any
        # LayoutTensor views of self exist — keeps the aliasing checker happy).
        var online_rows = List[Int]()
        while len(online_rows) < B:
            var on = self._online_pop_start(T)
            if on < 0:
                break
            online_rows.append(on)

        var buf_s_lt = tile[Self.CAP, Self.OBS](self.obs)
        var buf_a_lt = tile[Self.CAP, Self.ACT](self.act)
        var buf_r_lt = tile[Self.CAP](self.rew)
        var buf_d_lt = tile[Self.CAP](self.dne)
        var out_obs_lt = rebind[
            Tile[DT, B, T + 1, Self.OBS]
        ](tile[B, T + 1, Self.OBS](obs_dev))
        var out_act_lt = rebind[
            Tile[DT, B, T, Self.ACT]
        ](tile[B, T, Self.ACT](act_dev))
        var out_rew_lt = rebind[
            Tile[DT, B, T]
        ](tile[B, T](rew_dev))
        var out_dne_lt = rebind[
            Tile[DT, B, T]
        ](tile[B, T](dne_dev))

        var off_lt = tile[1](self._rng_offset_dev)

        # 1) Draw one start per window (same Philox math the old fused
        #    kernel used per-thread, so the window distribution is
        #    unchanged) into the pre-sized `starts` scratch.
        var starts_lt = tile[B](self.starts)
        comptime n_blocks_draw = (B + TPB - 1) // TPB
        comptime draw_kernel = _seq_draw_starts_kernel[B]
        ctx.enqueue_function[draw_kernel](
            starts_lt,
            Int32(n_valid),
            self.rng_seed,
            off_lt,
            grid_dim=n_blocks_draw,
            block_dim=TPB,
        )

        # 1b) Online rows (reference `online: True`): overwrite the first
        #     rows' starts with the pending fresh windows. Eager (the sample
        #     phase sits outside any captured graph); the Philox draw above
        #     already consumed its stream, so the uniform rows are
        #     bit-identical whether or not online rows are served.
        for i in range(len(online_rows)):
            ctx.enqueue_function[_seq_set_start_kernel[B]](
                starts_lt,
                Int32(i),
                Int32(online_rows[i]),
                grid_dim=1,
                block_dim=1,
            )

        # 2) Element-parallel gather: one thread per (window × frame ×
        #    OBS element).
        comptime n_blocks = (B * (T + 1) * Self.OBS + TPB - 1) // TPB
        comptime kernel = _seq_sample_kernel[
            B, T, Self.OBS, Self.ACT, Self.CAP, Self.SDT
        ]
        ctx.enqueue_function[kernel](
            buf_s_lt,
            buf_a_lt,
            buf_r_lt,
            buf_d_lt,
            out_obs_lt,
            out_act_lt,
            out_rew_lt,
            out_dne_lt,
            starts_lt,
            Int32(origin),
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        # Bump the offset on-device after the sample reads it.
        comptime inc_kernel = _increment_rng_offset_kernel[B]
        ctx.enqueue_function[inc_kernel](
            off_lt,
            grid_dim=1,
            block_dim=1,
        )

    def sample_batch_fst_dev[
        B: Int,
        T: Int,
    ](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        act_dev: DeviceBuffer[DT],
        rew_dev: DeviceBuffer[DT],
        dne_dev: DeviceBuffer[DT],
        fst_dev: DeviceBuffer[DT],   # [B, T+1] per-obs-frame is_first
    ) raises:
        """`sample_batch_dev` + the per-obs-frame `is_first` flags into
        `fst_dev` ([B, T+1]). Reuses the SAME drawn window starts so fst aligns
        with the obs frames. Device twin of `SequenceReplay.sample_batch_fst`."""
        self.sample_batch_dev[B, T](ctx, obs_dev, act_dev, rew_dev, dne_dev)
        var origin = 0 if self.size < Self.CAP else self.pos
        var starts_lt = tile[B](self.starts)
        var buf_fst_lt = tile[Self.CAP](self.fst)
        var out_fst_lt = rebind[
            Tile[DT, B, T + 1]
        ](tile[B, T + 1](fst_dev))
        comptime n_blocks_f = (B * (T + 1) + TPB - 1) // TPB
        ctx.enqueue_function[_seq_gather_fst_kernel[B, T, Self.CAP]](
            buf_fst_lt,
            out_fst_lt,
            starts_lt,
            Int32(origin),
            grid_dim=n_blocks_f,
            block_dim=TPB,
        )

    def sample_batch[
        B: Int,
        T: Int,
    ](
        mut self,
        obs_out: Pointer[Scalar[DT], MutAnyOrigin],
        act_out: Pointer[Scalar[DT], MutAnyOrigin],
        rew_out: Pointer[Scalar[DT], MutAnyOrigin],
        dne_out: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Host-output bridge: sample on-device into temp buffers, then copy
        out via the stored `ctx`. Lets a GPU buffer feed the current CPU
        world-model pipeline. Allocates per call (not the hot path)."""
        # Local copy of the handle: passing `self.ctx` while `self` is the
        # `mut` receiver of `sample_batch_dev` trips the aliasing checker.
        var c = self.ctx
        var tmp_obs = c.enqueue_create_buffer[DT](B * (T + 1) * Self.OBS)
        var tmp_act = c.enqueue_create_buffer[DT](B * T * Self.ACT)
        var tmp_rew = c.enqueue_create_buffer[DT](B * T)
        var tmp_dne = c.enqueue_create_buffer[DT](B * T)
        self.sample_batch_dev[B, T](c, tmp_obs, tmp_act, tmp_rew, tmp_dne)
        c.enqueue_copy(obs_out, tmp_obs)
        c.enqueue_copy(act_out, tmp_act)
        c.enqueue_copy(rew_out, tmp_rew)
        c.enqueue_copy(dne_out, tmp_dne)
        c.synchronize()
