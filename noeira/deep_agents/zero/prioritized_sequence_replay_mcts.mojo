"""Prioritized MuZero/EZv2 trajectory replay — PER over sequence windows.

The prioritized sibling of `MCTSSequenceReplay` (same flat episode rings + the
`sample_training_batch_seq` slab contract the EZv2 unroll consumes), adding
proportional **prioritized experience replay** keyed per stored step:

  * a `SumTree` over the ``CAP`` ring slots holds each step's priority;
  * `sample_training_batch_seq_per_gpu` draws each of ``B`` window-starts ∝
    priorityᵅ (stratified over ``B`` equal mass bins, the standard PER batch
    draw) and emits per-sample importance-sampling weights ``w_i = (N·P_i)^(−β)``
    (β=1, α=1 ⇒ atari.yaml `priority_prob_alpha/beta`), normalized by the batch
    max;
  * `update_priorities` writes back fresh priorities (|TD error|, the value-
    prediction error) after the train step; new steps enter at the running max
    priority so they are sampled at least once.

**Device obs ring (pixel-obs perf).** The observation ring lives on the
**device** (a `DeviceBuffer[SDT]`, `[CAP, OBS]`, uint8 for pixels), NOT on host.
All PER/target bookkeeping (sum-tree, episode index, act/rew/pol/val/done/tp/
legal, n-step targets) stays on the **host** — it is cheap. Sampling computes,
per (k, b), the physical ring slot on the host (no obs copy), then a single
element-parallel **gather kernel** assembles the `[K+1, B, OBS]` training slab
directly in device memory (dequantizing uint8→DT) — so the train step never
H2D-copies the obs slab. This kills the old anti-pattern (a serial host
dequant-build of `(K+1)·B·OBS` elements + a ~680 MB/step H2D) that dominated
runtime once training started — the same image-obs replay bottleneck fixed in
`GPUSequenceReplay`/`gpu_replay` (see those files + the Rainbow pixel profile).
`store_episode` quantizes the episode's host obs and H2D's it into the ring via
≤2 contiguous sub-buffer copies (chunked, wrap-aware); reanalyze obs are gathered
on-device the same way (`gather_obs_for_positions`).

Ring/prune correctness: by the time an episode is pruned (its start has fallen
out of the last ``CAP`` steps) every one of its slots has ALREADY been
overwritten by newer steps — whose `store_episode` reset those slots' priorities
— so a slot's priority always corresponds to its current resident absolute step.
No prune-time priority bookkeeping is needed; the sum-tree stays consistent.

Targets (n-step value, two-player sign flip, absorbing/obs-repeat padding,
truncation bootstrap, consistency boundary mask) are identical to the uniform
replay — only the *which-window* draw and the IS weights differ.
"""

from std.memory import alloc
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB
from noeira.core.sum_tree import SumTree
from .nstep_targets import compute_nstep_value_targets
from noeira.data.quantize import _obs_quant, _obs_dequant
from noeira.core.tiles import Tile, tile


# Host→device store staging chunk (steps). Bounds the transient host uint8
# buffer to CHUNK·OBS; store is per-episode (rare) so a sync per chunk is fine.
comptime STORE_CHUNK = 1024


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Local per-window scratch (w_rew/w_done/…) feeding the shared raw-pointer
    `compute_nstep_value_targets`; alloc'd + freed within one sample call."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


# ──────────────────────────────────────────────────────────────────────
# Element-parallel obs gather: one thread per (row × OBS element). Row m's
# physical ring slot is `slots[m]` (precomputed on host); dequantizes
# uint8→DT. Used for both the training slab (M = (K+1)·B rows) and reanalyze
# (M = REANA_W rows). Mirrors `_seq_sample_kernel` in gpu_sequence_replay.mojo
# but with host-precomputed slots (PER + within-episode window clamping live
# on the host), so it is a pure indexed gather.
# ──────────────────────────────────────────────────────────────────────
def _ez_obs_gather_kernel[
    M: Int, OBS: Int, CAP: Int, SDT: DType
](
    ring: Tile[SDT, CAP, OBS],
    slots: Tile[DType.int32, M],
    dst: Tile[DT, M, OBS],
):
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= M * OBS:
        return
    var m = t // OBS
    var i = t % OBS
    var slot = Int(slots[m])
    dst[m, i] = _obs_dequant[SDT](rebind[Scalar[SDT]](ring[slot, i]))


struct PrioritizedMCTSSequenceReplay[
    OBS: Int, ACT: Int, CAP: Int, OBS_STORE_DT: DType = DT
](Deinitable, Movable):
    """Prioritized ring of MCTS-labelled steps + episode index + a per-slot
    `SumTree`. ``CAP`` = max resident steps. ``OBS_STORE_DT`` mirrors
    `MCTSSequenceReplay` (DT or uint8 pixel storage). ``alpha``/``beta`` are
    the PER exponents (EZ Atari: 1.0 / 1.0). The obs ring is **device-resident**
    (see module docstring); everything else is host."""

    comptime SDT = Self.OBS_STORE_DT

    var obs: DeviceBuffer[Self.SDT]  # [CAP, OBS] on device
    var act: List[Scalar[DT]]
    var rew: List[Scalar[DT]]
    var done: List[Scalar[DT]]
    var pol: List[Scalar[DT]]  # [CAP, ACT]
    var val: List[Scalar[DT]]
    var tp: List[Scalar[DT]]
    var legal: List[Scalar[DT]]  # [CAP, ACT]

    # Host staging for the device-ring obs store (quantize → sub-buffer H2D).
    var _stage_u8: List[Scalar[Self.SDT]]  # [CHUNK, OBS]

    var ep_start: List[Int]
    var ep_len: List[Int]
    var ep_trunc: List[Bool]
    var total: Int
    var rng: UInt64

    var tree: SumTree[DT]  # priority per ring slot
    var max_prio: Scalar[DT]  # running max priority (new-sample init)
    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var eps: Scalar[DT]  # priority floor so nothing is unreachable

    var ctx: DeviceContext  # for device-ring store / gather

    def __init__(
        out self,
        ctx: DeviceContext,
        seed: UInt64 = 0,
        alpha: Scalar[DT] = Scalar[DT](1.0),
        beta: Scalar[DT] = Scalar[DT](1.0),
    ) raises:
        self.ctx = ctx
        self.obs = ctx.enqueue_create_buffer[Self.SDT](Self.CAP * Self.OBS)
        self.obs.enqueue_fill(Scalar[Self.SDT](0))
        self._stage_u8 = List[Scalar[Self.SDT]](
            length=STORE_CHUNK * Self.OBS, fill=Scalar[Self.SDT](0)
        )
        self.act = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.rew = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.done = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.pol = List[Scalar[DT]](
            length=Self.CAP * Self.ACT, fill=Scalar[DT](0)
        )
        self.val = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.tp = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0))
        self.legal = List[Scalar[DT]](
            length=Self.CAP * Self.ACT, fill=Scalar[DT](0)
        )
        self.ep_start = List[Int]()
        self.ep_len = List[Int]()
        self.ep_trunc = List[Bool]()
        self.total = 0
        self.rng = seed ^ UInt64(0x9E3779B97F4A7C15)
        self.tree = SumTree[DT](Self.CAP)
        self.max_prio = Scalar[DT](1.0)
        self.alpha = alpha
        self.beta = beta
        self.eps = Scalar[DT](1e-6)

    def _xorshift(mut self) -> UInt64:
        var x = self.rng
        x = x ^ (x << 13)
        x = x ^ (x >> 7)
        x = x ^ (x << 17)
        self.rng = x
        return x

    def num_episodes(self) -> Int:
        return len(self.ep_start)

    def num_steps(self) -> Int:
        return self.total if self.total < Self.CAP else Self.CAP

    def _ring_lt(
        self,
    ) -> Tile[Self.SDT, Self.CAP, Self.OBS]:
        return rebind[
            Tile[Self.SDT, Self.CAP, Self.OBS]
        ](
            tile[Self.CAP, Self.OBS](self.obs)
        )

    def store_episode(
        mut self,
        ep_obs: List[Scalar[DT]],
        ep_act: List[Scalar[DT]],
        ep_rew: List[Scalar[DT]],
        ep_pol: List[Scalar[DT]],
        ep_val: List[Scalar[DT]],
        ep_tp: List[Scalar[DT]],
        ep_legal: List[Scalar[DT]],
        length: Int,
        truncated: Bool = False,
    ) raises:
        """Append a finished episode; each new step enters the sum-tree at the
        running max priority (standard PER — guarantees it is seen). The host
        label fields are written directly; the obs are quantized into the host
        staging buffer and H2D'd into the **device** ring in ≤2 contiguous
        sub-buffer copies per chunk (wrap-aware). See `MCTSSequenceReplay.
        store_episode` for the absorbing/truncation rules."""
        var start = self.total
        for i in range(length):
            var slot = (self.total) % Self.CAP
            for a in range(Self.ACT):
                self.pol[slot * Self.ACT + a] = ep_pol[i * Self.ACT + a]
            for a in range(Self.ACT):
                self.legal[slot * Self.ACT + a] = ep_legal[i * Self.ACT + a]
            self.act[slot] = ep_act[i]
            self.rew[slot] = ep_rew[i]
            self.val[slot] = ep_val[i]
            self.tp[slot] = ep_tp[i]
            self.done[slot] = Scalar[DT](1.0) if (
                i == length - 1 and not truncated
            ) else Scalar[DT](0.0)
            # new step → max priority (overwrites any stale priority on this slot)
            self.tree.update(slot, self.max_prio)
            self.total += 1

        # ── obs → device ring: quantize chunk → H2D (wrap-split sub-buffers) ──
        var done_steps = 0
        while done_steps < length:
            var m = STORE_CHUNK if (length - done_steps) > STORE_CHUNK else (
                length - done_steps
            )
            for r in range(m):
                var src = (done_steps + r) * Self.OBS
                for j in range(Self.OBS):
                    self._stage_u8[r * Self.OBS + j] = _obs_quant[Self.SDT](
                        ep_obs[src + j]
                    )
            var abs0 = start + done_steps
            var slot0 = abs0 % Self.CAP
            var first = m if (Self.CAP - slot0) > m else (Self.CAP - slot0)
            var sub1 = self.obs.create_sub_buffer[Self.SDT](
                slot0 * Self.OBS, first * Self.OBS
            )
            self.ctx.enqueue_copy(sub1, self._stage_u8.unsafe_ptr())
            if m > first:
                var sub2 = self.obs.create_sub_buffer[Self.SDT](
                    0, (m - first) * Self.OBS
                )
                self.ctx.enqueue_copy(
                    sub2, self._stage_u8.unsafe_ptr().unsafe_offset(first * Self.OBS)
                )
            self.ctx.synchronize()  # staging reused next chunk
            done_steps += m

        self.ep_start.append(start)
        self.ep_len.append(length)
        self.ep_trunc.append(truncated)
        var floor = self.total - Self.CAP
        while len(self.ep_start) > 0 and self.ep_start[0] < floor:
            _ = self.ep_start.pop(0)
            _ = self.ep_len.pop(0)
            _ = self.ep_trunc.pop(0)

    def _slot_to_pos(self, slot: Int) -> Tuple[Int, Int]:
        """Map a ring slot to its current resident ``(episode_idx, offset)``,
        or ``(-1, -1)`` if the slot holds no resident-episode step. ``slot``'s
        resident absolute step is the unique ``a ≡ slot (mod CAP)`` in
        ``[total-num_steps, total)``."""
        var ns = self.num_steps()
        if ns == 0:
            return (-1, -1)
        var lo = self.total - ns
        # largest a <= total-1 with a % CAP == slot
        var a = slot + Self.CAP * ((self.total - 1 - slot) // Self.CAP)
        if a < lo or a >= self.total:
            return (-1, -1)
        # binary search: rightmost episode with ep_start <= a
        var ne = len(self.ep_start)
        if ne == 0:
            return (-1, -1)
        var lo_i = 0
        var hi_i = ne - 1
        var e = -1
        while lo_i <= hi_i:
            var mid = (lo_i + hi_i) // 2
            if self.ep_start[mid] <= a:
                e = mid
                lo_i = mid + 1
            else:
                hi_i = mid - 1
        if e < 0:
            return (-1, -1)
        var off = a - self.ep_start[e]
        if off < 0 or off >= self.ep_len[e]:
            return (-1, -1)
        return (e, off)

    def sample_training_batch_seq_per_gpu[
        B: Int,
        K: Int,
        N: Int,
    ](
        mut self,
        ctx: DeviceContext,
        gamma: Scalar[DT],
        out_obs_dev: DeviceBuffer[DT],  # [(K+1)*B, OBS]
        d_slots: DeviceBuffer[DType.int32],  # [(K+1)*B]
        mut h_slots: List[Int32],  # [(K+1)*B]
        mut actions: List[Scalar[DT]],  # [K, B]
        mut policy_tgt: List[Scalar[DT]],  # [K+1, B, ACT]
        mut value_tgt: List[Scalar[DT]],  # [K+1, B]
        mut reward_tgt: List[Scalar[DT]],  # [K, B]
        mut is_weights: List[Scalar[DT]],  # [B] IS weights
        mut sample_slots: List[Int],  # [B] ring slots
        cons_mask: Optional[
            Pointer[Scalar[DT], MutAnyOrigin]
        ] = None,  # [K, B]
        out_prio: Optional[
            Pointer[Scalar[DT], MutAnyOrigin]
        ] = None,  # [B] raw PER priority |ν − z| per sampled root (paper formula)
    ) raises:
        """Prioritized window sample with **device-side obs gather**: each window
        start is drawn ∝ priorityᵅ (stratified over B equal-mass bins) with
        per-sample IS weights ``(N·P_i)^(−β)`` normalized by the batch max; the
        host computes the per-(k,b) physical ring slot into ``h_slots`` and the
        gather kernel writes the dequantized ``[K+1, B, OBS]`` obs slab directly
        into ``out_obs_dev`` (no host obs copy, no slab H2D). ``sample_slots[b]``
        records the chosen root ring slot for `update_priorities`. Caller
        guarantees num_episodes()>0 and num_steps()>0; ``out_obs_dev`` /
        ``d_slots`` sized ``(K+1)*B*OBS`` / ``(K+1)*B``."""
        comptime HV = K + N + 1
        comptime HR = K + N
        comptime NK = K + 1
        comptime M = NK * B
        var w_rew = _a(HR)
        var w_done = _a(HR)
        var w_val = _a(HV)
        var w_tp = _a(HV)
        var w_vt = _a(K + 1)

        var ns = self.num_steps()
        var total_p = self.tree.total_sum()
        if total_p <= Scalar[DT](0.0):
            total_p = Scalar[DT](1.0)
        var seg = total_p / Scalar[DT](B)

        # First pass: pick slots + raw IS weights; track max weight for norm.
        var max_w = Scalar[DT](0.0)
        for b in range(B):
            # stratified target in bin b
            var u = Float64(self._xorshift() % UInt64(1_000_000)) / 1_000_000.0
            var target = (Scalar[DT](b) + Scalar[DT](u)) * seg
            var slot = self.tree.sample(target)
            sample_slots[b] = slot
            var p = self.tree.get(slot)
            if p <= Scalar[DT](0.0):
                p = self.eps
            var prob = p / total_p
            # w = (N * P)^(-beta)
            var w = (Scalar[DT](ns) * prob) ** (-self.beta)
            is_weights[b] = w
            if w > max_w:
                max_w = w

        if max_w <= Scalar[DT](0.0):
            max_w = Scalar[DT](1.0)

        for b in range(B):
            is_weights[b] = is_weights[b] / max_w  # normalize by batch max
            var pos = self._slot_to_pos(sample_slots[b])
            var e = pos[0]
            var s = pos[1]
            if e < 0:
                # Degenerate (slot not resident) — fall back to a uniform step.
                e = Int(self._xorshift() % UInt64(len(self.ep_start)))
                s = Int(self._xorshift() % UInt64(self.ep_len[e]))
                sample_slots[b] = (self.ep_start[e] + s) % Self.CAP
            var L = self.ep_len[e]
            var lv = K + N + 1
            if self.ep_trunc[e]:
                lv = L - 1 - s

            # record per-(k,b) physical ring slot for the device gather
            for k in range(K + 1):
                var off = s + k
                if off >= L:
                    off = L - 1
                var oslot = (self.ep_start[e] + off) % Self.CAP
                h_slots[k * B + b] = Int32(oslot)
            if cons_mask:
                var cm = cons_mask.value()
                for k in range(K):
                    cm[unsafe_offset=k * B + b] = Scalar[DT](
                        1.0
                    ) if s + k + 1 < L else Scalar[DT](0.0)

            for h in range(HR):
                if s + h >= L:
                    w_rew[unsafe_offset=h] = Scalar[DT](0.0)
                    w_done[unsafe_offset=h] = Scalar[DT](1.0)
                else:
                    var rslot = (self.ep_start[e] + s + h) % Self.CAP
                    w_rew[unsafe_offset=h] = self.rew[rslot]
                    w_done[unsafe_offset=h] = self.done[rslot]
            for h in range(HV):
                if s + h >= L:
                    w_val[unsafe_offset=h] = Scalar[DT](0.0)
                    w_tp[unsafe_offset=h] = Scalar[DT](0.0)
                else:
                    var vslot = (self.ep_start[e] + s + h) % Self.CAP
                    w_val[unsafe_offset=h] = self.val[vslot]
                    w_tp[unsafe_offset=h] = self.tp[vslot]

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
                    var pslot = (self.ep_start[e] + s + k) % Self.CAP
                    for a in range(Self.ACT):
                        policy_tgt[pbase + a] = self.pol[pslot * Self.ACT + a]
            # PER priority = MuZero paper formula p_i = |ν_i − z_i|: |root search
            # value − observed n-step return|. ν = w_val[0] (the stored MCTS root
            # value, kept current by reanalyze); z = w_vt[0] (the n-step target).
            # Caller applies (·+eps)^α via update_priorities. Replaces the old
            # value-head soft-CE, which was not the paper's signal.
            if out_prio:
                var praw = w_val[unsafe_offset=0] - w_vt[unsafe_offset=0]
                if praw < Scalar[DT](0.0):
                    praw = -praw
                out_prio.value()[unsafe_offset=b] = praw
            for k in range(K):
                if s + k >= L:
                    actions[k * B + b] = Scalar[DT](0.0)
                else:
                    var aslot = (self.ep_start[e] + s + k) % Self.CAP
                    actions[k * B + b] = self.act[aslot]
                reward_tgt[k * B + b] = w_rew[unsafe_offset=k]

        w_rew.unsafe_free()
        w_done.unsafe_free()
        w_val.unsafe_free()
        w_tp.unsafe_free()
        w_vt.unsafe_free()

        # ── device gather: H2D the M slot indices, assemble the obs slab ──
        ctx.enqueue_copy(d_slots, h_slots.unsafe_ptr())  # sanctioned H2D-staging boundary
        var ring_lt = self._ring_lt()
        var slots_lt = rebind[
            Tile[DType.int32, M]
        ](tile[M](d_slots))
        var out_lt = rebind[
            Tile[DT, M, Self.OBS]
        ](tile[M, Self.OBS](out_obs_dev))
        comptime n_blocks = (M * Self.OBS + TPB - 1) // TPB
        comptime kernel = _ez_obs_gather_kernel[M, Self.OBS, Self.CAP, Self.SDT]
        ctx.enqueue_function[kernel](
            ring_lt,
            slots_lt,
            out_lt,
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    def gather_obs_for_positions[
        M: Int,
    ](
        mut self,
        ctx: DeviceContext,
        out_obs_dev: DeviceBuffer[DT],  # [M, OBS]
        d_slots: DeviceBuffer[DType.int32],  # [M]
        mut h_slots: List[Int32],  # [M]
        ep_idx: List[Int],
        offset: List[Int],
    ) raises:
        """Gather the obs of ``M`` (episode, offset) positions into
        ``out_obs_dev`` on device (reanalyze path; replaces the old per-position
        host `read_obs` + H2D). Positions past their episode length clamp to the
        last step (mirrors the sample windowing)."""
        for m in range(M):
            var e = ep_idx[m]
            var o = offset[m]
            if e < 0 or e >= len(self.ep_start):
                h_slots[m] = Int32(0)
                continue
            if o >= self.ep_len[e]:
                o = self.ep_len[e] - 1
            if o < 0:
                o = 0
            h_slots[m] = Int32((self.ep_start[e] + o) % Self.CAP)
        ctx.enqueue_copy(d_slots, h_slots.unsafe_ptr())  # sanctioned H2D-staging boundary
        var ring_lt = self._ring_lt()
        var slots_lt = rebind[
            Tile[DType.int32, M]
        ](tile[M](d_slots))
        var out_lt = rebind[
            Tile[DT, M, Self.OBS]
        ](tile[M, Self.OBS](out_obs_dev))
        comptime n_blocks = (M * Self.OBS + TPB - 1) // TPB
        comptime kernel = _ez_obs_gather_kernel[M, Self.OBS, Self.CAP, Self.SDT]
        ctx.enqueue_function[kernel](
            ring_lt,
            slots_lt,
            out_lt,
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    def update_priorities(
        mut self,
        slots: List[Int],  # [n] ring slots
        priorities: List[Scalar[DT]],  # [n] |TD error|
        n: Int,
    ):
        """Write back fresh priorities (``|TD error|`` + eps)ᵅ for the sampled
        slots and update the running max priority (so new steps enter at it)."""
        for i in range(n):
            var p = priorities[i]
            if p < Scalar[DT](0.0):
                p = -p
            p = (p + self.eps) ** self.alpha
            self.tree.update(slots[i], p)
            if p > self.max_prio:
                self.max_prio = p

    # ── reanalyze hooks (mirror the uniform replay) ──

    def update_targets(
        mut self,
        ep_idx: Int,
        offset: Int,
        new_policy: List[Scalar[DT]],
        new_value: Scalar[DT],
    ):
        if ep_idx < 0 or ep_idx >= len(self.ep_start):
            return
        if offset < 0 or offset >= self.ep_len[ep_idx]:
            return
        var slot = (self.ep_start[ep_idx] + offset) % Self.CAP
        for a in range(Self.ACT):
            self.pol[slot * Self.ACT + a] = new_policy[a]
        self.val[slot] = new_value

    def sample_position(mut self) -> Tuple[Int, Int]:
        var e = Int(self._xorshift() % UInt64(len(self.ep_start)))
        var o = Int(self._xorshift() % UInt64(self.ep_len[e]))
        return (e, o)

    def read_legal(self, ep_idx: Int, offset: Int) -> List[Bool]:
        """Legal mask at a stored (episode, offset) — host-resident, so a plain
        read (mirrors `MCTSSequenceReplay.read_legal`). Used by the two-player
        reanalyze path to mask the root search to legal columns."""
        var out = List[Bool]()
        var o = offset
        if o >= self.ep_len[ep_idx]:
            o = self.ep_len[ep_idx] - 1
        if o < 0:
            o = 0
        var slot = (self.ep_start[ep_idx] + o) % Self.CAP
        for a in range(Self.ACT):
            out.append(self.legal[slot * Self.ACT + a] > Scalar[DT](0.5))
        return out^
