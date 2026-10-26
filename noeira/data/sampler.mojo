# +--------------------------------------------------------------------------+ #
# | Index policies — how rows get chosen
# +--------------------------------------------------------------------------+ #
"""Samplers produce row indices; the store knows nothing about how they were
chosen, and the policy knows nothing about columns.

That split is the point of the whole layer. The legacy
`ReplayBuffer.sample_into(state: TrainerState)` fuses three concerns —
storage, sampling policy, destination layout — which is why every
{policy x backend x column set} combination needed its own struct. Here a
policy emits an `IndexBatch` and `ResidentColumn.gather_*` consumes it, so
adding PER costs one policy rather than N buffers.

**These are ports, not redesigns.** Each policy reproduces its legacy
counterpart's index sequence bit-for-bit under a fixed seed. That was gated
against the legacy buffers directly until they were deleted; the sequences were
captured as goldens first, so the gate is now
`tests/data/test_sampler_golden.mojo`. A sampler that is subtly *differently*
random still trains — just worse, and only visibly so several algorithms
later. So the arithmetic below is copied deliberately, including its clamps:

  * `Int(u * Float64(size))` then clamp to `size-1`, rather than a modulo
  * PER's stratified segments and its `total - 1e-7` upper clamp
  * the sum-tree descent's `<=` on the left child

⚠ **Host and device uniform are DIFFERENT sequences and always were.** The
legacy CPU path draws from the global `random_float64()`; the GPU path runs
one Philox stream per lane keyed on `seed + lane`. Neither is wrong, but
nothing can make them agree, so parity is per-backend against its own legacy
counterpart — never CPU-vs-GPU.

⚠ **n-step is NOT an index policy.** `NStepBuffer`/`GPUNStepBuffer` are
*accumulators*: they hold N pending transitions and emit one aggregated
transition with a discounted return, before storage. The plan listed n-step
alongside uniform/PER/sequence; that was wrong. n-step belongs at write time
(or as a transform over gathered rows), and nothing here needs to change for
it.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import pow as fpow
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.utils import IndexList
from layout import TileTensor

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from .episode_index import EpisodeIndex
from .resident import DYN1, IDX_DT, IndexBatch, rl1


# ══════════════════════════════════════════════════════════════════════════
# Uniform — host
# ══════════════════════════════════════════════════════════════════════════

struct UniformSampler(Movable & Deinitable):
    """Uniform with replacement, host RNG.

    Port of `CPUReplay.sample` (`cpu_replay.mojo:113`):

        idx = Int(random_float64() * Float64(size))
        if idx >= size: idx = size - 1

    The clamp matters: `random_float64()` returning exactly 1.0 would index
    one past the end. Kept rather than "fixed" to a modulo, which would draw a
    different sequence.
    """

    var n_rows: Int

    def __init__(out self, n_rows: Int):
        self.n_rows = n_rows

    def __init__(out self, *, deinit move: Self):
        self.n_rows = move.n_rows

    def draw(self, batch: Int) raises -> IndexBatch:
        if batch <= 0:
            raise Error("UniformSampler.draw: batch must be > 0")
        if self.n_rows <= 0:
            raise Error("UniformSampler.draw: store is empty")
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=batch)
        for k in range(batch):
            var idx = Int(random_float64() * Float64(self.n_rows))
            if idx >= self.n_rows:
                idx = self.n_rows - 1
            h[k] = Scalar[IDX_DT](idx)
        return IndexBatch(h^)


# ══════════════════════════════════════════════════════════════════════════
# Uniform — device (Philox, one stream per lane)
# ══════════════════════════════════════════════════════════════════════════

def _uniform_indices_kernel(
    indices: TileTensor[IDX_DT, DYN1, MutAnyOrigin],
    size: Int32,
    seed: UInt64,
    offset: UInt64,
):
    """Port of `gpu_replay.mojo::_sample_indices_kernel`.

    One Philox stream per lane, seed mixed with the lane index so lanes do not
    correlate. `Float32` (not Float64) and `Int(u * Float32(size))` are copied
    exactly — widening to Float64 here would silently change the draw.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var n = Int(indices.dim[0]())
    if i >= n:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset)
    var u = Float32(philox.step_uniform()[0])
    var sz = Int(size)
    var idx = Int(u * Float32(sz))
    if idx >= sz:
        idx = sz - 1
    if idx < 0:
        idx = 0
    indices[i] = Scalar[IDX_DT](idx)


struct UniformDeviceSampler(Movable & Deinitable):
    """Uniform with replacement, on device.

    Mirrors `GPUReplay`'s RNG bookkeeping: a fixed `seed`, plus an `offset`
    counter advanced by `2 * batch` after every draw so successive batches use
    disjoint Philox streams.
    """

    var n_rows: Int
    var seed: UInt64
    var offset: UInt64

    def __init__(out self, n_rows: Int, seed: UInt64 = 0, offset: UInt64 = 0):
        self.n_rows = n_rows
        self.seed = seed
        self.offset = offset

    def __init__(out self, *, deinit move: Self):
        self.n_rows = move.n_rows
        self.seed = move.seed
        self.offset = move.offset

    def draw_into_device(
        mut self,
        ctx: DeviceContext,
        dev: DeviceBuffer[IDX_DT],
        batch: Int,
    ) raises:
        """Fill a caller-owned device index buffer. No readback, no
        synchronize — this is the training-loop path, where a D2H round trip
        per minibatch would dominate a gather that costs microseconds."""
        if batch <= 0:
            raise Error("UniformDeviceSampler: batch must be > 0")
        if self.n_rows <= 0:
            raise Error("UniformDeviceSampler: store is empty")
        var lt = TileTensor[IDX_DT, DYN1, MutAnyOrigin](
            ptr=mptr(dev.unsafe_ptr()), layout=rl1(batch)
        )
        var n_blocks = (batch + TPB - 1) // TPB
        ctx.enqueue_function[_uniform_indices_kernel](
            lt,
            Int32(self.n_rows),
            self.seed,
            self.offset,
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        # Same advance as GPUReplay's `_increment_rng_offset_kernel`.
        self.offset += UInt64(2 * batch)

    def draw(mut self, ctx: DeviceContext, batch: Int) raises -> IndexBatch:
        """Device draw WITH readback — for gating and inspection, not the
        training loop (see `draw_into_device`)."""
        var dev = ctx.enqueue_create_buffer[IDX_DT](batch)
        self.draw_into_device(ctx, dev, batch)
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=batch)
        ctx.enqueue_copy(h.unsafe_ptr(), dev)
        ctx.synchronize()
        var out = IndexBatch(h^)
        out.dev = dev^
        out.dev_len = batch
        return out^


# ══════════════════════════════════════════════════════════════════════════
# Prioritized — stratified sum-tree
# ══════════════════════════════════════════════════════════════════════════

struct PrioritizedSampler(Movable & Deinitable):
    """Stratified proportional PER over a sum-tree.

    Port of `CPUPrioritizedReplay` (`cpu_per_replay.mojo`), split out of
    storage: the tree indexes rows, and nothing here touches column data.

    `capacity` must be the tree's leaf count — a power of two is NOT required
    by the descent, but it must match the legacy `CAP` for indices to line up,
    since leaves live at `[CAP-1, 2*CAP-1)`.
    """

    var tree: List[Scalar[DT]]
    var capacity: Int
    var size: Int
    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var epsilon: Scalar[DT]
    var max_priority: Scalar[DT]
    var last_indices: List[Int]
    var last_weights: List[Scalar[DT]]

    def __init__(
        out self,
        capacity: Int,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self.tree = List[Scalar[DT]](
            length=2 * capacity - 1, fill=Scalar[DT](0.0)
        )
        self.capacity = capacity
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = Scalar[DT](1.0)
        self.last_indices = List[Int]()
        self.last_weights = List[Scalar[DT]]()

    def __init__(out self, *, deinit move: Self):
        self.tree = move.tree^
        self.capacity = move.capacity
        self.size = move.size
        self.alpha = move.alpha
        self.beta = move.beta
        self.epsilon = move.epsilon
        self.max_priority = move.max_priority
        self.last_indices = move.last_indices^
        self.last_weights = move.last_weights^

    def set_beta(mut self, beta: Scalar[DT]):
        self.beta = beta

    def total(self) -> Scalar[DT]:
        return self.tree[0]

    def _update_leaf(mut self, leaf_idx: Int, priority: Scalar[DT]):
        var tree_idx = leaf_idx + self.capacity - 1
        var diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] = self.tree[tree_idx] + diff

    def _descend(self, u: Scalar[DT]) -> Int:
        """`<=` on the left child, matching the legacy descent exactly. `<`
        would move the tie to the right child and change the sequence."""
        var idx: Int = 0
        var v = u
        while idx < self.capacity - 1:
            var left = 2 * idx + 1
            var right = left + 1
            var left_sum = self.tree[left]
            if v <= left_sum:
                idx = left
            else:
                v = v - left_sum
                idx = right
        return idx - (self.capacity - 1)

    def note_added(mut self, row: Int):
        """Register a newly stored row. Its priority is `max_priority^alpha`,
        so fresh experience enters at the top of the distribution."""
        var p = Scalar[DT](
            fpow(Float64(self.max_priority), Float64(self.alpha))
        )
        self._update_leaf(row, p)
        if row + 1 > self.size:
            self.size = row + 1

    def draw(mut self, batch: Int) raises -> IndexBatch:
        if batch <= 0:
            raise Error("PrioritizedSampler.draw: batch must be > 0")
        if self.size < batch:
            raise Error(
                "PrioritizedSampler.draw: only " + String(self.size)
                + " rows registered, need " + String(batch)
            )
        var total = self.total()
        var segment = total / Scalar[DT](batch)
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=batch)
        self.last_indices = List[Int](length=batch, fill=0)
        self.last_weights = List[Scalar[DT]](
            length=batch, fill=Scalar[DT](1.0)
        )

        var max_w = Scalar[DT](0.0)
        for i in range(batch):
            var lo = segment * Scalar[DT](i)
            var hi = segment * Scalar[DT](i + 1)
            var u = lo + (hi - lo) * Scalar[DT](random_float64())
            if u >= total:
                u = total - Scalar[DT](1e-7)
            if u < Scalar[DT](0.0):
                u = Scalar[DT](0.0)
            var leaf = self._descend(u)
            if leaf >= self.size:
                leaf = self.size - 1
            self.last_indices[i] = leaf
            h[i] = Scalar[IDX_DT](leaf)

            var p_leaf = self.tree[leaf + self.capacity - 1]
            var P = p_leaf / total
            var w = Scalar[DT](
                fpow(Float64(self.size) * Float64(P), Float64(-self.beta))
            )
            self.last_weights[i] = w
            if w > max_w:
                max_w = w

        if max_w <= Scalar[DT](0.0):
            max_w = Scalar[DT](1.0)
        for i in range(batch):
            self.last_weights[i] = self.last_weights[i] / max_w

        return IndexBatch(h^)

    def update_priorities(
        mut self, ref td_residuals: List[Scalar[DT]]
    ) raises:
        """`p = (|TD| + eps)^alpha` for the rows of the most recent `draw`."""
        var n = len(self.last_indices)
        if n == 0:
            raise Error(
                "PrioritizedSampler.update_priorities: no preceding draw"
            )
        if len(td_residuals) < n:
            raise Error(
                "PrioritizedSampler.update_priorities: got "
                + String(len(td_residuals)) + " residuals for a batch of "
                + String(n)
            )
        var new_max = self.max_priority
        for i in range(n):
            var td = td_residuals[i]
            var td_abs = td if td >= Scalar[DT](0.0) else -td
            var raw = td_abs + self.epsilon
            if raw > new_max:
                new_max = raw
            var p = Scalar[DT](fpow(Float64(raw), Float64(self.alpha)))
            self._update_leaf(self.last_indices[i], p)
        self.max_priority = new_max


# ══════════════════════════════════════════════════════════════════════════
# Sequence windows
# ══════════════════════════════════════════════════════════════════════════

struct SequenceWindowSampler(Movable & Deinitable):
    """Draw start rows for length-`span` windows.

    Port of `SequenceReplay.sample_batch_fst`'s start draw
    (`sequence_replay.mojo:308`):

        n_valid = size - T
        s = Int(random_float64() * Float64(n_valid)); clamp to n_valid - 1

    ⚠ **The legacy sampler lets a window SPAN episode boundaries** and relies
    on the per-frame `is_first` flags to reset the model's carry at the seam.
    That is deliberate in DreamerV3, so `within_episode=False` is the default
    and the parity-faithful setting. `within_episode=True` rejects spanning
    draws using `EpisodeIndex.window_fits` — which is what a consumer wanting
    genuinely intra-episode windows (FB, anything treating a window as one
    trajectory) should ask for. Silently "fixing" the default would change
    DreamerV3's training distribution.
    """

    var n_rows: Int
    var span: Int
    var within_episode: Bool

    def __init__(out self, n_rows: Int, span: Int, within_episode: Bool = False):
        self.n_rows = n_rows
        self.span = span
        self.within_episode = within_episode

    def __init__(out self, *, deinit move: Self):
        self.n_rows = move.n_rows
        self.span = move.span
        self.within_episode = move.within_episode

    def n_valid(self) -> Int:
        """Legal start rows, matching the legacy `size - T` (note: NOT
        `size - T + 1` — the legacy window reads `T + 1` obs frames)."""
        return self.n_rows - self.span

    def draw_starts(self, batch: Int) raises -> IndexBatch:
        """Start rows only; `within_episode` is ignored here (see
        `draw_starts_in_episodes`)."""
        if batch <= 0:
            raise Error("SequenceWindowSampler.draw_starts: batch must be > 0")
        var nv = self.n_valid()
        if nv <= 0:
            raise Error(
                "SequenceWindowSampler: store holds " + String(self.n_rows)
                + " rows, too few for span " + String(self.span)
            )
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=batch)
        for b in range(batch):
            var s = Int(random_float64() * Float64(nv))
            if s >= nv:
                s = nv - 1
            h[b] = Scalar[IDX_DT](s)
        return IndexBatch(h^)

    def draw_starts_in_episodes(
        self, ref episodes: EpisodeIndex, batch: Int, max_tries: Int = 100
    ) raises -> IndexBatch:
        """Rejection-sample start rows whose `span` stays inside one episode.

        Rejection rather than an episode-first draw, because an episode-first
        draw is NOT uniform over valid starts — it over-samples short episodes.
        Raises if a start cannot be found, rather than falling back to a
        spanning window, which would reintroduce the bug silently.
        """
        if batch <= 0:
            raise Error("draw_starts_in_episodes: batch must be > 0")
        var nv = self.n_valid()
        if nv <= 0:
            raise Error(
                "SequenceWindowSampler: store holds " + String(self.n_rows)
                + " rows, too few for span " + String(self.span)
            )
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=batch)
        for b in range(batch):
            var found = -1
            for _ in range(max_tries):
                var s = Int(random_float64() * Float64(nv))
                if s >= nv:
                    s = nv - 1
                if episodes.window_fits(s, self.span + 1):
                    found = s
                    break
            if found < 0:
                raise Error(
                    "draw_starts_in_episodes: no within-episode start for span "
                    + String(self.span) + " after " + String(max_tries)
                    + " tries — episodes are probably shorter than the span"
                )
            h[b] = Scalar[IDX_DT](found)
        return IndexBatch(h^)

    def expand_window(
        self, ref starts: IndexBatch, mut out: List[Scalar[IDX_DT]]
    ) raises:
        """Expand `[batch]` starts into `[batch * (span+1)]` row indices, so a
        window gather is an ordinary gather. `span+1` frames, matching the
        legacy window's `T + 1` obs frames."""
        var batch = len(starts.host)
        var frames = self.span + 1
        var need = batch * frames
        if len(out) != need:
            out = List[Scalar[IDX_DT]](unsafe_uninit_length=need)
        for b in range(batch):
            var s = Int(starts.host[b])
            for k in range(frames):
                out[b * frames + k] = Scalar[IDX_DT](s + k)
