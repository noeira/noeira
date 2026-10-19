"""GPUMCTSSequenceReplay — device-obs MuZero trajectory replay (Phase 2).

The device-resident sibling of `MCTSSequenceReplay`: the obs ring lives on the
GPU (uint8-quantized) so the batched self-play driver never D2H's a full
``[N_ENVS, OBS]`` pixel observation per step — `record_obs_meta` stores obs
device→device straight from `env.obs_ptr()`, and `sample_training_batch_dev`
gathers the ``[B, OBS]`` training obs slab device→device into the train step's
own buffer. Only the tiny metadata (action / reward / MCTS policy + value /
done / to_play / legal) stays on the host, where the proven n-step target and
absorbing-padding logic is reused verbatim.

Ring model — **single shared ring, strided by ``N_ENVS``.** All ``N_ENVS`` envs
step in lockstep, so iteration ``it`` writes ``N_ENVS`` steps at absolute
positions ``it·N_ENVS + e`` (ring slot ``pos % CAP``). Env ``e``'s consecutive
steps are therefore ``N_ENVS`` slots apart, and a window of ``off = 0..`` reads
slot ``(ep_start + off·N_ENVS) % CAP`` — the only deviation from the host
buffer's stride-1 reads. ``CAP % N_ENVS == 0`` keeps the stride exact across
wrap. The shared ring packs long and short episodes together (no per-env
capacity split), but because every env advances in lockstep an in-flight episode
of ``L`` steps spans ``L·N_ENVS`` ring positions, so **``CAP`` must exceed
``N_ENVS · max_ep_steps``** or an episode self-overwrites before it closes
(asserted softly via the prune window; size ``CAP`` accordingly).

Recording is two-phase per iteration to match the driver's data availability:
  * `record_obs_meta[N_ENVS]` at the search root (obs + chosen action + MCTS
    policy/value known) — stores obs (kernel) + metadata (host), extends each
    env's open episode.
  * `record_outcome[N_ENVS]` after the env step (reward / done / terminated
    known) — writes reward into each env's just-written slot and, on
    done/time-limit, sets the terminal flag (kept off on truncation → n-step
    bootstrap) and closes the episode.

Reanalyze hooks (`sample_position` / `read_obs` / `update_targets`) mirror the
host buffer; `read_obs` gathers a single obs row device→host.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc, dealloc

from noeira.nn.constants import DT, TPB
from noeira.core.sum_tree import SumTree
from noeira.data.quantize import _obs_quant, _obs_dequant
from .nstep_targets import compute_nstep_value_targets
from noeira.core.tiles import Tile, tile


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Local per-window scratch (w_rew/w_done/…) feeding the shared raw-pointer
    `compute_nstep_value_targets`; alloc'd + freed within one sample call."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (element-parallel, one thread per (item × obs element)).
# ──────────────────────────────────────────────────────────────────────


def _mz_obs_store_batch_kernel[
    N_ENVS: Int, OBS: Int, CAP: Int, SDT: DType,
](
    src_obs: Tile[DT, N_ENVS, OBS],
    buf_s: Tile[SDT, CAP, OBS],
    start_pos: Int32,
):
    """Store ``N_ENVS`` obs rows into the device ring at lockstep slots
    ``(start_pos + e) % CAP``. Thread ``t`` → env ``e = t // OBS``, element
    ``d = t % OBS``; quantizes ``DT → SDT`` on the way in. Launch
    ``grid=(ceil(N_ENVS·OBS/TPB),), block=(TPB,)``."""
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= N_ENVS * OBS:
        return
    var e = t // OBS
    var d = t % OBS
    var slot = (Int(start_pos) + e) % CAP
    buf_s[slot, d] = _obs_quant[SDT](rebind[Scalar[DT]](src_obs[e, d]))


def _mz_obs_gather_kernel[
    B: Int, OBS: Int, CAP: Int, SDT: DType,
](
    slots: Tile[DType.int32, B],
    buf_s: Tile[SDT, CAP, OBS],
    out_obs: Tile[DT, B, OBS],
):
    """Gather ``B`` obs rows from the ring at the caller-supplied ``slots`` into
    ``out_obs[B, OBS]``, dequantizing ``SDT → DT``. Thread ``t`` → window
    ``b = t // OBS``, element ``d = t % OBS``. Launch
    ``grid=(ceil(B·OBS/TPB),), block=(TPB,)``."""
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= B * OBS:
        return
    var b = t // OBS
    var d = t % OBS
    var slot = Int(slots[b])
    out_obs[b, d] = _obs_dequant[SDT](rebind[Scalar[SDT]](buf_s[slot, d]))


# ──────────────────────────────────────────────────────────────────────
# Replay.
# ──────────────────────────────────────────────────────────────────────


struct GPUMCTSSequenceReplay[
    OBS: Int, ACT: Int, CAP: Int, N_ENVS: Int,
    OBS_STORE_DT: DType = DType.uint8,
](Movable, Deinitable):
    """Device-obs MuZero replay (see module docstring). ``OBS_STORE_DT`` is the
    obs ring dtype (default ``uint8``: ``round(x·255)`` store / ``k/255`` read,
    bit-lossless for the arcade pixel pipeline; set ``DT`` for vector obs)."""

    comptime SDT = Self.OBS_STORE_DT
    comptime STRIDE = Self.N_ENVS

    var ctx: DeviceContext
    var obs_dev: DeviceBuffer[Self.SDT]   # [CAP, OBS]

    # host metadata rings (indexed by ring slot = abs_pos % CAP)
    var act: List[Scalar[DT]]    # [CAP]
    var rew: List[Scalar[DT]]    # [CAP]
    var done: List[Scalar[DT]]   # [CAP]
    var val: List[Scalar[DT]]    # [CAP]
    var tp: List[Scalar[DT]]     # [CAP]
    var pol: List[Scalar[DT]]    # [CAP, ACT]
    var legal: List[Scalar[DT]]  # [CAP, ACT]

    # closed-episode index (absolute start positions).
    var ep_start: List[Int]
    var ep_len: List[Int]
    var ep_trunc: List[Bool]
    # per-env open episode (abs start, len); start < 0 ⇒ none open.
    var open_start: List[Int]
    var open_len: List[Int]

    var gtotal: Int   # total absolute steps written (= iterations · N_ENVS)
    var rng: UInt64

    # cached gather staging (lazily sized to B on first sample).
    var d_slots: Optional[DeviceBuffer[DType.int32]]
    var h_slots: List[Int32]
    var slots_n: Int
    # separate cached staging for reanalyze chunk gathers (sized to the chunk =
    # N_ENVS, kept apart from the training-batch slots so the two don't thrash
    # each other's lazily-sized buffers when B != chunk).
    var d_rslots: Optional[DeviceBuffer[DType.int32]]
    var h_rslots: List[Int32]
    var rslots_n: Int

    # ── PER (only maintained when `per` is True; uniform path untouched) ──
    # A per-slot `SumTree` of priorities + a per-slot reverse map
    # (slot → episode astart / offset / length / truncated) backfilled when an
    # episode CLOSES, so the prioritized sampler needs no episode search. New
    # steps clear their slot to 0 on overwrite; a step enters the tree at
    # `max_prio` only once its episode closes (so only closed steps are drawn,
    # matching the uniform sampler's resident-closed-steps semantics).
    var per: Bool
    var tree: SumTree[DT]
    var max_prio: Scalar[DT]
    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var eps: Scalar[DT]
    var slot_astart: List[Int]   # [CAP]
    var slot_off: List[Int]      # [CAP]
    var slot_len: List[Int]      # [CAP] episode length
    var slot_trunc: List[Int]    # [CAP] 0/1 truncated

    def __init__(
        out self,
        ctx: DeviceContext,
        seed: UInt64 = 0,
        per: Bool = False,
        alpha: Scalar[DT] = Scalar[DT](1.0),
        beta: Scalar[DT] = Scalar[DT](1.0),
    ) raises:
        comptime assert Self.CAP % Self.N_ENVS == 0, (
            "GPUMCTSSequenceReplay: CAP must be a multiple of N_ENVS"
        )
        self.ctx = ctx
        self.obs_dev = ctx.enqueue_create_buffer[Self.SDT](Self.CAP * Self.OBS)
        ctx.enqueue_memset(self.obs_dev, 0)
        self.act = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.rew = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.done = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.val = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.tp = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.pol = List[Scalar[DT]](
            length=Self.CAP * Self.ACT, fill=Scalar[DT](0)
        )
        self.legal = List[Scalar[DT]](
            length=Self.CAP * Self.ACT, fill=Scalar[DT](0)
        )
        self.ep_start = List[Int]()
        self.ep_len = List[Int]()
        self.ep_trunc = List[Bool]()
        self.open_start = List[Int]()
        self.open_len = List[Int]()
        for _ in range(Self.N_ENVS):
            self.open_start.append(-1)
            self.open_len.append(0)
        self.gtotal = 0
        self.rng = seed ^ UInt64(0x9E3779B97F4A7C15)
        self.d_slots = None
        self.h_slots = List[Int32](length=1, fill=Int32(0))
        self.slots_n = 0
        self.d_rslots = None
        self.h_rslots = List[Int32](length=1, fill=Int32(0))
        self.rslots_n = 0
        self.per = per
        self.tree = SumTree[DT](Self.CAP)
        self.max_prio = Scalar[DT](1.0)
        self.alpha = alpha
        self.beta = beta
        self.eps = Scalar[DT](1e-6)
        self.slot_astart = List[Int](length=Self.CAP, fill=0)
        self.slot_off = List[Int](length=Self.CAP, fill=0)
        self.slot_len = List[Int](length=Self.CAP, fill=0)
        self.slot_trunc = List[Int](length=Self.CAP, fill=0)

    def _xorshift(mut self) -> UInt64:
        var x = self.rng
        x = x ^ (x << 13); x = x ^ (x >> 7); x = x ^ (x << 17)
        self.rng = x
        return x

    def _slot(self, abs_start: Int, off: Int) -> Int:
        """Ring slot of episode-relative step ``off`` (strided by N_ENVS)."""
        return (abs_start + off * Self.STRIDE) % Self.CAP

    def num_episodes(self) -> Int:
        return len(self.ep_start)

    def num_steps(self) -> Int:
        var t = 0
        for e in range(len(self.ep_len)):
            t += self.ep_len[e]
        return t

    def _prune(mut self):
        """Drop closed episodes whose earliest slot has fallen out of the last
        ``CAP`` absolute positions (mirrors the host buffer's prune rule)."""
        var floor = self.gtotal - Self.CAP
        while len(self.ep_start) > 0 and self.ep_start[0] < floor:
            _ = self.ep_start.pop(0)
            _ = self.ep_len.pop(0)
            _ = self.ep_trunc.pop(0)

    def record_obs_meta(
        mut self,
        src_obs: DeviceBuffer[DT],          # [N_ENVS, OBS]
        h_act: List[Scalar[DT]],            # [N_ENVS]
        h_pol: List[Scalar[DT]],            # [N_ENVS, ACT]
        h_val: List[Scalar[DT]],            # [N_ENVS]
    ) raises:
        """Phase 1 of a step: store the root obs (device→device) + chosen action
        + MCTS policy/value (host) for all ``N_ENVS`` envs, extending each env's
        open episode. ``reward``/``done`` arrive later via `record_outcome`."""
        var base = self.gtotal % Self.CAP
        # `src_obs` is an immutable param, so its pointer now carries an
        # immutable origin; the kernel ABI declares `MutAnyOrigin`. The device
        # allocation is not owned by Mojo's origin system (cf. `enqueue_copy`
        # into immutable `DeviceBuffer` params elsewhere), so the cast is sound
        # — this view is read-only here regardless.
        var src_t = tile[Self.N_ENVS, Self.OBS](src_obs.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
        var buf_t = tile[Self.CAP, Self.OBS](self.obs_dev.unsafe_ptr().as_unsafe_any_origin())
        comptime nb = (Self.N_ENVS * Self.OBS + TPB - 1) // TPB
        self.ctx.enqueue_function[
            _mz_obs_store_batch_kernel[Self.N_ENVS, Self.OBS, Self.CAP, Self.SDT]
        ](src_t, buf_t, Int32(base), grid_dim=nb, block_dim=TPB)

        for e in range(Self.N_ENVS):
            var pos = self.gtotal + e
            var slot = pos % Self.CAP
            self.act[slot] = h_act[e]
            self.val[slot] = h_val[e]
            self.tp[slot] = Scalar[DT](0.0)
            self.done[slot] = Scalar[DT](0.0)
            for a in range(Self.ACT):
                self.pol[slot * Self.ACT + a] = h_pol[e * Self.ACT + a]
                self.legal[slot * Self.ACT + a] = Scalar[DT](1.0)
            if self.open_start[e] < 0:
                self.open_start[e] = pos
                self.open_len[e] = 0
            self.open_len[e] += 1
            # PER: this slot is being (re)written by an OPEN step — clear any
            # stale priority of the previous (now overwritten) occupant so it is
            # unsampleable until this episode closes (→ enters at max_prio), and
            # mark its reverse-map stale (slot_len=0) so a rare 0-priority sample
            # hit on it falls back to a uniform draw instead of reading old meta.
            if self.per:
                self.tree.update(slot, Scalar[DT](0.0))
                self.slot_len[slot] = 0
        self.gtotal += Self.N_ENVS

    def record_outcome(
        mut self,
        h_rew: List[Scalar[DT]],   # [N_ENVS]
        h_done: List[Scalar[DT]],  # [N_ENVS]
        h_term: List[Scalar[DT]],  # [N_ENVS]
        max_ep_steps: Int,
    ):
        """Phase 2 of a step: write each env's reward into its just-written slot
        and, on done / time-limit, set the terminal flag (kept off on truncation
        so the n-step target bootstraps) and close the episode."""
        for e in range(Self.N_ENVS):
            if self.open_start[e] < 0:
                continue
            var last_pos = self.open_start[e] + (self.open_len[e] - 1) * Self.STRIDE
            var last_slot = last_pos % Self.CAP
            self.rew[last_slot] = h_rew[e]
            var done = h_done[e] > Scalar[DT](0.5)
            var terminated = h_term[e] > Scalar[DT](0.5)
            if done or self.open_len[e] >= max_ep_steps:
                self.done[last_slot] = (
                    Scalar[DT](1.0) if terminated else Scalar[DT](0.0)
                )
                var astart = self.open_start[e]
                var L = self.open_len[e]
                var trunc = not terminated
                self.ep_start.append(astart)
                self.ep_len.append(L)
                self.ep_trunc.append(trunc)
                # PER: episode closed → backfill every slot's reverse map and
                # admit it to the tree at max priority (now sampleable).
                if self.per:
                    var tr = 1 if trunc else 0
                    for off in range(L):
                        var sl = (astart + off * Self.STRIDE) % Self.CAP
                        self.slot_astart[sl] = astart
                        self.slot_off[sl] = off
                        self.slot_len[sl] = L
                        self.slot_trunc[sl] = tr
                        self.tree.update(sl, self.max_prio)
                self.open_start[e] = -1
                self.open_len[e] = 0
        self._prune()

    def _sample_step_uniform(mut self) -> Tuple[Int, Int]:
        """Step-uniform resident (episode, offset) — every closed step equally
        likely (mirrors the host buffer)."""
        var tot = 0
        for e in range(len(self.ep_len)):
            tot += self.ep_len[e]
        var u = Int(self._xorshift() % UInt64(tot))
        var e = 0
        while u >= self.ep_len[e]:
            u -= self.ep_len[e]
            e += 1
        return (e, u)

    def _ensure_slots(mut self, n: Int) raises:
        if self.slots_n != n:
            self.h_slots = List[Int32](length=n, fill=Int32(0))
            self.d_slots = self.ctx.enqueue_create_buffer[DType.int32](n)
            self.slots_n = n

    def sample_training_batch_dev[
        B: Int, K: Int, N: Int,
    ](
        mut self,
        gamma: Scalar[DT],
        mut d_obs0: DeviceBuffer[DT],     # [B, OBS] (out)
        mut actions: List[Scalar[DT]],    # [K, B]
        mut policy_tgt: List[Scalar[DT]], # [K+1, B, ACT]
        mut value_tgt: List[Scalar[DT]],  # [K+1, B]
        mut reward_tgt: List[Scalar[DT]], # [K, B]
    ) raises:
        """Device-obs twin of `MCTSSequenceReplay.sample_training_batch`: the
        host metadata + n-step targets are computed exactly as there (strided
        slots), while the ``[B, OBS]`` obs slab is gathered device→device into
        ``d_obs0`` (the train step's own obs buffer). Caller guarantees
        ``num_episodes() > 0``."""
        comptime HV = K + N + 1
        comptime HR = K + N
        var w_rew = _a(HR)
        var w_done = _a(HR)
        var w_val = _a(HV)
        var w_tp = _a(HV)
        var w_vt = _a(K + 1)
        self._ensure_slots(B)

        for b in range(B):
            var pos = self._sample_step_uniform()
            var e = pos[0]
            var s = pos[1]
            var L = self.ep_len[e]
            var astart = self.ep_start[e]
            var lv = K + N + 1
            if self.ep_trunc[e]:
                lv = L - 1 - s

            # obs0 ring slot (gathered on device below).
            self.h_slots[b] = Int32(self._slot(astart, s))

            # reward / done horizon (absorbing past terminal).
            for h in range(HR):
                if s + h >= L:
                    w_rew[unsafe_offset=h] = Scalar[DT](0.0)
                    w_done[unsafe_offset=h] = Scalar[DT](1.0)
                else:
                    var sl = self._slot(astart, s + h)
                    w_rew[unsafe_offset=h] = self.rew[sl]
                    w_done[unsafe_offset=h] = self.done[sl]
            # value / to_play horizon.
            for h in range(HV):
                if s + h >= L:
                    w_val[unsafe_offset=h] = Scalar[DT](0.0)
                    w_tp[unsafe_offset=h] = Scalar[DT](0.0)
                else:
                    var sl = self._slot(astart, s + h)
                    w_val[unsafe_offset=h] = self.val[sl]
                    w_tp[unsafe_offset=h] = self.tp[sl]

            compute_nstep_value_targets[K, N](
                w_rew, w_done, w_val, w_tp, gamma, w_vt, last_valid=lv
            )

            for k in range(K + 1):
                value_tgt[k * B + b] = w_vt[unsafe_offset=k]
                var pbase = k * B * Self.ACT + b * Self.ACT
                if s + k >= L:
                    var u = Scalar[DT](1.0) / Scalar[DT](Self.ACT)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = u
                else:
                    var sl = self._slot(astart, s + k)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = self.pol[sl * Self.ACT + a]
            for k in range(K):
                if s + k >= L:
                    actions[k * B + b] = Scalar[DT](0.0)
                else:
                    actions[k * B + b] = self.act[self._slot(astart, s + k)]
                reward_tgt[k * B + b] = w_rew[unsafe_offset=k]

        # ── gather obs0 device→device into the caller's buffer ──
        self.ctx.enqueue_copy(self.d_slots.value(), self.h_slots.unsafe_ptr())
        var slots_t = tile[B](self.d_slots.value().unsafe_ptr().as_unsafe_any_origin())
        var buf_t = tile[Self.CAP, Self.OBS](self.obs_dev.unsafe_ptr().as_unsafe_any_origin())
        var out_t = tile[B, Self.OBS](d_obs0.unsafe_ptr().as_unsafe_any_origin())
        comptime nb = (B * Self.OBS + TPB - 1) // TPB
        self.ctx.enqueue_function[
            _mz_obs_gather_kernel[B, Self.OBS, Self.CAP, Self.SDT]
        ](slots_t, buf_t, out_t, grid_dim=nb, block_dim=TPB)

        w_rew.unsafe_free(); w_done.unsafe_free(); w_val.unsafe_free(); w_tp.unsafe_free(); w_vt.unsafe_free()

    def sample_training_batch_per_dev[
        B: Int, K: Int, N: Int,
    ](
        mut self,
        gamma: Scalar[DT],
        mut d_obs0: DeviceBuffer[DT],     # [B, OBS] (out)
        mut actions: List[Scalar[DT]],    # [K, B]
        mut policy_tgt: List[Scalar[DT]], # [K+1, B, ACT]
        mut value_tgt: List[Scalar[DT]],  # [K+1, B]
        mut reward_tgt: List[Scalar[DT]], # [K, B]
        mut is_weights: List[Scalar[DT]], # [B] IS weights
        mut sample_slots: List[Int],      # [B] ring slots
        out_prio: Optional[
            Pointer[Scalar[DT], MutAnyOrigin]
        ] = None,  # [B] raw PER priority |ν − z| per sampled root (paper formula)
    ) raises:
        """Prioritized device-obs twin of `sample_training_batch_dev`: window
        starts are drawn ∝ priorityᵅ (stratified over B equal-mass bins) with
        per-sample IS weights ``(N·P_i)^(−β)`` normalized by the batch max; the
        n-step targets + device obs gather are computed exactly as the uniform
        sampler. The root ring slot per sample is the `SumTree` leaf, and its
        episode ``(astart, off, L, trunc)`` come from the slot reverse-map
        backfilled at episode close (no episode search). ``sample_slots[b]``
        records the slot for `update_priorities`. Requires the buffer was built
        with ``per=True`` and ``num_episodes() > 0``."""
        comptime HV = K + N + 1
        comptime HR = K + N
        var w_rew = _a(HR)
        var w_done = _a(HR)
        var w_val = _a(HV)
        var w_tp = _a(HV)
        var w_vt = _a(K + 1)
        self._ensure_slots(B)

        var ns = self.num_steps()
        var total_p = self.tree.total_sum()
        if total_p <= Scalar[DT](0.0):
            total_p = Scalar[DT](1.0)
        var seg = total_p / Scalar[DT](B)

        # First pass: stratified prioritized slot draw + raw IS weights.
        var max_w = Scalar[DT](0.0)
        for b in range(B):
            var u = Float64(self._xorshift() % UInt64(1_000_000)) / 1_000_000.0
            var target = (Scalar[DT](b) + Scalar[DT](u)) * seg
            var slot = self.tree.sample(target)
            sample_slots[b] = slot
            var p = self.tree.get(slot)
            if p <= Scalar[DT](0.0):
                p = self.eps
            var prob = p / total_p
            var w = (Scalar[DT](ns) * prob) ** (-self.beta)
            is_weights[b] = w
            if w > max_w:
                max_w = w
        if max_w <= Scalar[DT](0.0):
            max_w = Scalar[DT](1.0)

        for b in range(B):
            is_weights[b] = is_weights[b] / max_w
            var slot = sample_slots[b]
            var astart = self.slot_astart[slot]
            var s = self.slot_off[slot]
            var L = self.slot_len[slot]
            var trunc = self.slot_trunc[slot] != 0
            if L <= 0:
                # Degenerate (slot holds no closed step) — uniform fallback.
                var pos = self._sample_step_uniform()
                var e = pos[0]
                s = pos[1]
                astart = self.ep_start[e]
                L = self.ep_len[e]
                trunc = self.ep_trunc[e]
                sample_slots[b] = self._slot(astart, s)
            var lv = K + N + 1
            if trunc:
                lv = L - 1 - s

            # obs0 ring slot (gathered on device below).
            self.h_slots[b] = Int32(self._slot(astart, s))

            for h in range(HR):
                if s + h >= L:
                    w_rew[unsafe_offset=h] = Scalar[DT](0.0)
                    w_done[unsafe_offset=h] = Scalar[DT](1.0)
                else:
                    var sl = self._slot(astart, s + h)
                    w_rew[unsafe_offset=h] = self.rew[sl]
                    w_done[unsafe_offset=h] = self.done[sl]
            for h in range(HV):
                if s + h >= L:
                    w_val[unsafe_offset=h] = Scalar[DT](0.0)
                    w_tp[unsafe_offset=h] = Scalar[DT](0.0)
                else:
                    var sl = self._slot(astart, s + h)
                    w_val[unsafe_offset=h] = self.val[sl]
                    w_tp[unsafe_offset=h] = self.tp[sl]

            compute_nstep_value_targets[K, N](
                w_rew, w_done, w_val, w_tp, gamma, w_vt, last_valid=lv
            )

            for k in range(K + 1):
                value_tgt[k * B + b] = w_vt[unsafe_offset=k]
                var pbase = k * B * Self.ACT + b * Self.ACT
                if s + k >= L:
                    var uni = Scalar[DT](1.0) / Scalar[DT](Self.ACT)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = uni
                else:
                    var sl = self._slot(astart, s + k)
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = self.pol[sl * Self.ACT + a]
            # PER priority = MuZero paper p_i = |ν_i − z_i| (|root search value −
            # n-step return|): ν = w_val[0] (stored MCTS root value, current via
            # reanalyze), z = w_vt[0] (n-step target). update_priorities applies
            # (·+eps)^α. Replaces the old value-head soft-CE (not the paper signal).
            if out_prio:
                var praw = w_val[unsafe_offset=0] - w_vt[unsafe_offset=0]
                if praw < Scalar[DT](0.0):
                    praw = -praw
                out_prio.value()[unsafe_offset=b] = praw
            for k in range(K):
                if s + k >= L:
                    actions[k * B + b] = Scalar[DT](0.0)
                else:
                    actions[k * B + b] = self.act[self._slot(astart, s + k)]
                reward_tgt[k * B + b] = w_rew[unsafe_offset=k]

        # ── gather obs0 device→device into the caller's buffer ──
        self.ctx.enqueue_copy(self.d_slots.value(), self.h_slots.unsafe_ptr())
        var slots_t = tile[B](self.d_slots.value().unsafe_ptr().as_unsafe_any_origin())
        var buf_t = tile[Self.CAP, Self.OBS](self.obs_dev.unsafe_ptr().as_unsafe_any_origin())
        var out_t = tile[B, Self.OBS](d_obs0.unsafe_ptr().as_unsafe_any_origin())
        comptime nb = (B * Self.OBS + TPB - 1) // TPB
        self.ctx.enqueue_function[
            _mz_obs_gather_kernel[B, Self.OBS, Self.CAP, Self.SDT]
        ](slots_t, buf_t, out_t, grid_dim=nb, block_dim=TPB)

        w_rew.unsafe_free(); w_done.unsafe_free(); w_val.unsafe_free(); w_tp.unsafe_free(); w_vt.unsafe_free()

    def update_priorities(
        mut self,
        slots: List[Int],              # [n] ring slots
        priorities: List[Scalar[DT]],  # [n] |value error|
        n: Int,
    ):
        """Write back fresh priorities ``(|value error| + eps)ᵅ`` for the sampled
        slots and lift the running max (so newly closed steps enter at it)."""
        for i in range(n):
            var p = priorities[i]
            if p < Scalar[DT](0.0):
                p = -p
            p = (p + self.eps) ** self.alpha
            self.tree.update(slots[i], p)
            if p > self.max_prio:
                self.max_prio = p

    # ── reanalyze hooks (host metadata + single-obs device→host gather) ──

    def _ensure_rslots(mut self, n: Int) raises:
        if self.rslots_n != n:
            self.h_rslots = List[Int32](length=n, fill=Int32(0))
            self.d_rslots = self.ctx.enqueue_create_buffer[DType.int32](n)
            self.rslots_n = n

    def sample_reanalyze_chunk[
        R: Int,
    ](
        mut self,
        mut out_obs: DeviceBuffer[DT],   # [R, OBS] (out, device)
    ) raises -> Tuple[List[Int], List[Int]]:
        """Sample ``R`` resident positions and gather their root obs
        device→device into ``out_obs`` in ONE kernel launch (no per-position
        sync — unlike `read_obs`), returning the ``(ep_idx, offset)`` lists so the
        caller can write fresh MCTS targets back with `update_targets`. This is
        the high-coverage reanalyze primitive: the driver calls it per chunk of
        ``R = N_ENVS`` (the planner's root width) and loops to cover
        ``reanalyze_batch`` positions per iteration. Caller guarantees
        ``num_episodes() > 0``."""
        self._ensure_rslots(R)
        var eps = List[Int]()
        var offs = List[Int]()
        for r in range(R):
            var p = self.sample_position()
            eps.append(p[0])
            offs.append(p[1])
            self.h_rslots[r] = Int32(self._slot(self.ep_start[p[0]], p[1]))
        self.ctx.enqueue_copy(self.d_rslots.value(), self.h_rslots.unsafe_ptr())
        var slots_t = tile[R](self.d_rslots.value().unsafe_ptr().as_unsafe_any_origin())
        var buf_t = tile[Self.CAP, Self.OBS](self.obs_dev.unsafe_ptr().as_unsafe_any_origin())
        var out_t = tile[R, Self.OBS](out_obs.unsafe_ptr().as_unsafe_any_origin())
        comptime nb = (R * Self.OBS + TPB - 1) // TPB
        self.ctx.enqueue_function[
            _mz_obs_gather_kernel[R, Self.OBS, Self.CAP, Self.SDT]
        ](slots_t, buf_t, out_t, grid_dim=nb, block_dim=TPB)
        return (eps^, offs^)

    def sample_position(mut self) -> Tuple[Int, Int]:
        var e = Int(self._xorshift() % UInt64(len(self.ep_start)))
        var o = Int(self._xorshift() % UInt64(self.ep_len[e]))
        return (e, o)

    def read_obs(
        mut self,
        ep_idx: Int,
        offset: Int,
        mut out: List[Scalar[DT]],   # [OBS] host
    ) raises:
        """Copy stored obs at ``(ep_idx, offset)`` device→host (1-row gather)."""
        var slot = self._slot(self.ep_start[ep_idx], offset)
        # one-row gather via tiny throwaway device buffers.
        var d_one = self.ctx.enqueue_create_buffer[DType.int32](1)
        var d_out = self.ctx.enqueue_create_buffer[DT](Self.OBS)
        var hs_a = alloc[Int32]({count = 1})
        var hs = hs_a.unsafe_ptr().as_unsafe_any_origin()
        try:
            hs[unsafe_offset=0] = Int32(slot)
            self.ctx.enqueue_copy(d_one, hs)
            var slots_t = tile[1](d_one.unsafe_ptr().as_unsafe_any_origin())
            var buf_t = tile[Self.CAP, Self.OBS](self.obs_dev.unsafe_ptr().as_unsafe_any_origin())
            var out_t = tile[1, Self.OBS](d_out.unsafe_ptr().as_unsafe_any_origin())
            comptime nb = (Self.OBS + TPB - 1) // TPB
            self.ctx.enqueue_function[
                _mz_obs_gather_kernel[1, Self.OBS, Self.CAP, Self.SDT]
            ](slots_t, buf_t, out_t, grid_dim=nb, block_dim=TPB)
            # `.unsafe_ptr()`: sanctioned D2H-staging boundary (device row → host List).
            self.ctx.enqueue_copy(out.unsafe_ptr(), d_out)
            self.ctx.synchronize()
        finally:
            dealloc(hs_a^)

    def update_targets(
        mut self,
        ep_idx: Int,
        offset: Int,
        new_policy: List[Scalar[DT]],  # [ACT]
        new_value: Scalar[DT],
    ):
        if ep_idx < 0 or ep_idx >= len(self.ep_start):
            return
        if offset < 0 or offset >= self.ep_len[ep_idx]:
            return
        var slot = self._slot(self.ep_start[ep_idx], offset)
        for a in range(Self.ACT):
            self.pol[slot * Self.ACT + a] = new_policy[a]
        self.val[slot] = new_value
