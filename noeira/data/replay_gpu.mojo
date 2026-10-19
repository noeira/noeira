# +--------------------------------------------------------------------------+ #
# | StoreReplayGpu — device-resident ring behind the ReplayBuffer seam
# +--------------------------------------------------------------------------+ #
"""GPU counterpart of `StoreReplay`, using the Stage-3 device index policy.

Kernels here are deliberately NOT imported from `gpu_replay.mojo`. Depending
on the module we intend to delete would make the deletion impossible, so this
carries its own store/gather kernels — ports of the legacy ones, gated
bit-identical against them.

Dims are comptime at this seam (`OBS`/`ACT`/`CAP` are trait parameters), so
these use static `Layout.row_major(...)` rather than the runtime-layout idiom
that `resident.mojo` needs for store-metadata dims. Both spellings exist on
purpose: static where the dims are known, runtime where they are not.

Three index policies are covered, each gated bit-identical against its legacy
counterpart: **uniform**, **ERE** (Emphasizing Recent Experience — sample from
a shrinking recent window) and **PER** (device sum-tree). `PRIORITIZED` is a
comptime flag, so ONE struct replaces both `GPUReplay` and
`GPUPrioritizedReplay`.

⚠ **Capture-safety differs by policy, faithfully.** `size` and the Philox
`offset` live in DEVICE buffers for the uniform and PER paths — a baked host
`size` is what the legacy `gpu_replay.mojo` (deleted 4d.3) called "the
catastrophic-divergence bug", since capture would freeze sampling at the warmup
fill. ERE passes `size`/`pos`/`c_k`
as host scalars, exactly as the legacy does: the legacy documents ERE as
deliberately non-capturable (it would need an on-device anneal).

⚠ **ERE and PER cannot both be enabled.** The legacy pairing was
`GPUReplay` + ERE or `GPUPrioritizedReplay` without it, never both;
`configure_ere(enable=True)` on a PER buffer raises rather than silently
dropping one of the two policies.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import pow as fpow
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT, TPB
from noeira.deep_agents.training.replay_buffer import ReplayBuffer
from noeira.deep_agents.training.trainer_block import TrainerState
from .quantize import _obs_dequant, _obs_quant
from .resident import IDX_DT
from .sampler import UniformDeviceSampler
from noeira.core.tiles import Tile, tile



# ── kernels ───────────────────────────────────────────────────────────────

def _store_one_kernel[
    OBS: Int, ACT: Int, CAP: Int, SDT: DType = DT
](
    stage_s: Tile[DT, OBS],
    stage_a: Tile[DT, ACT],
    stage_r: Tile[DT, 1],
    stage_sp: Tile[DT, OBS],
    stage_d: Tile[DT, 1],
    buf_s: Tile[SDT, CAP, OBS],
    buf_a: Tile[DT, CAP, ACT],
    buf_r: Tile[DT, CAP],
    buf_sp: Tile[SDT, CAP, OBS],
    buf_d: Tile[DT, CAP],
    slot: Int32,
):
    """One transition, staged host→device, written into `buf[slot]`."""
    var d = Int(thread_idx.x)
    var s = Int(slot)
    if d < OBS:
        buf_s[s, d] = _obs_quant[SDT](rebind[Scalar[DT]](stage_s[d]))
        buf_sp[s, d] = _obs_quant[SDT](rebind[Scalar[DT]](stage_sp[d]))
    if d < ACT:
        buf_a[s, d] = stage_a[d]
    if d == 0:
        buf_r[s] = stage_r[0]
        buf_d[s] = stage_d[0]


def _store_batch_kernel[
    N_ENVS: Int, OBS: Int, ACT: Int, CAP: Int, SDT: DType = DT
](
    src_s: Tile[DT, N_ENVS, OBS],
    src_a: Tile[DT, N_ENVS, ACT],
    src_r: Tile[DT, N_ENVS],
    src_sp: Tile[DT, N_ENVS, OBS],
    src_d: Tile[DT, N_ENVS],
    buf_s: Tile[SDT, CAP, OBS],
    buf_a: Tile[DT, CAP, ACT],
    buf_r: Tile[DT, CAP],
    buf_sp: Tile[SDT, CAP, OBS],
    buf_d: Tile[DT, CAP],
    start_pos: Int32,
    base: Int32,
):
    """`N_ENVS` transitions written to consecutive ring slots from
    `start_pos`, wrapping. Element-parallel over (env, obs element).

    `base` is the pinned DEMO prefix (`pin_demo_prefix`): rows `[0, base)`
    are never written and the ring wraps inside `[base, CAP)`. 0 when no
    prefix is pinned, which is the old modulo exactly."""
    comptime assert OBS >= ACT, "_store_batch_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= N_ENVS * OBS:
        return
    var e = t // OBS
    var d = t % OBS
    var b = Int(base)
    var slot = b + ((Int(start_pos) - b) + e) % (CAP - b)
    buf_s[slot, d] = _obs_quant[SDT](rebind[Scalar[DT]](src_s[e, d]))
    buf_sp[slot, d] = _obs_quant[SDT](rebind[Scalar[DT]](src_sp[e, d]))
    if d < ACT:
        buf_a[slot, d] = src_a[e, d]
    if d == 0:
        buf_r[slot] = src_r[e]
        buf_d[slot] = src_d[e]


def _gather_batch_kernel[
    BATCH: Int, OBS: Int, ACT: Int, CAP: Int, SDT: DType = DT
](
    mb_s: Tile[DT, BATCH, OBS],
    mb_a: Tile[DT, BATCH, ACT],
    mb_r: Tile[DT, BATCH],
    mb_sp: Tile[DT, BATCH, OBS],
    mb_d: Tile[DT, BATCH],
    buf_s: Tile[SDT, CAP, OBS],
    buf_a: Tile[DT, CAP, ACT],
    buf_r: Tile[DT, CAP],
    buf_sp: Tile[SDT, CAP, OBS],
    buf_d: Tile[DT, CAP],
    indices: Tile[IDX_DT, BATCH],
):
    """One thread per (lane, obs element). Element-parallel rather than
    one-thread-per-lane: a per-lane kernel serialises the row copy and
    launches only BATCH threads, which cost ~73% of GPU time on pixel-wide
    rows (`project_rainbow_pong_pixel_replay_gather_bottleneck`)."""
    comptime assert OBS >= ACT, "_gather_batch_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= BATCH * OBS:
        return
    var i = t // OBS
    var d = t % OBS
    var idx = Int(indices[i])
    mb_s[i, d] = _obs_dequant[SDT](rebind[Scalar[SDT]](buf_s[idx, d]))
    mb_sp[i, d] = _obs_dequant[SDT](rebind[Scalar[SDT]](buf_sp[idx, d]))
    if d < ACT:
        mb_a[i, d] = buf_a[idx, d]
    if d == 0:
        mb_r[i] = buf_r[idx]
        mb_d[i] = buf_d[idx]



# ── device-resident RNG / size bookkeeping ────────────────────────────────
#
# `size` and the Philox `offset` live in DEVICE buffers, not host scalars.
# That is not fussiness: the legacy `gpu_replay.mojo` (deleted 4d.3) documented
# a baked host `size` as "the catastrophic-divergence bug" — under capture it freezes
# sampling to the capture-time fill (≈ warmup) forever. 4b shipped host
# scalars here; this restores capture-safety.

def _uniform_indices_dev_kernel[
    BATCH: Int
](
    indices: Tile[IDX_DT, BATCH],
    size_buf: Tile[DType.int32, 1],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Capture-safe port of `gpu_replay.mojo::_sample_indices_kernel`."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var size = Int(size_buf[0])
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var idx = Int(u * Float32(size))
    if idx >= size:
        idx = size - 1
    if idx < 0:
        idx = 0
    indices[i] = Scalar[IDX_DT](idx)


def _mixed_indices_dev_kernel[
    BATCH: Int
](
    indices: Tile[IDX_DT, BATCH],
    size_buf: Tile[DType.int32, 1],
    demo_buf: Tile[DType.int32, 1],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """RLPD's symmetric sampling over ONE storage: lanes `[0, BATCH/2)` draw
    uniformly from the pinned demo prefix `[0, demo_n)`, the rest from the
    online ring `[demo_n, size)`. With no online rows yet every lane draws
    from the prefix. Capture-safe: `size` and `demo_n` are device words."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var size = Int(size_buf[0])
    var demo_n = Int(demo_buf[0])
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var lo = 0
    var n = demo_n
    if i >= BATCH // 2 and size > demo_n:
        lo = demo_n
        n = size - demo_n
    var idx = lo + Int(u * Float32(n))
    if idx >= lo + n:
        idx = lo + n - 1
    if idx < lo:
        idx = lo
    indices[i] = Scalar[IDX_DT](idx)


def _incr_offset_kernel[BATCH: Int](
    offset_buf: Tile[DType.uint64, 1],
):
    """`offset += 2 * BATCH` on device — the capturable form of the host
    counter bump."""
    if Int(thread_idx.x) != 0:
        return
    offset_buf[0] = rebind[UInt64](offset_buf[0]) + UInt64(2 * BATCH)


def _set_size_kernel(
    size_buf: Tile[DType.int32, 1],
    v: Int32,
):
    if Int(thread_idx.x) != 0:
        return
    size_buf[0] = v



def _ere_indices_kernel[
    BATCH: Int, CAP: Int
](
    indices: Tile[IDX_DT, BATCH],
    size: Int32,
    write_pos: Int32,
    c_k: Int32,
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Emphasizing Recent Experience (Wang & Ross 2019) — sample uniformly
    from the most recent `c_k` rows of the ring rather than all of it.

    Port of `gpu_replay.mojo::_sample_indices_ere_kernel`. ⚠ `size`,
    `write_pos` and `c_k` are HOST scalars here exactly as in the legacy: the
    legacy docstring notes ERE is deliberately NOT CUDA-graph capturable,
    since capture would need device-resident `c_k` and an on-device anneal.
    The uniform and PER paths above ARE capturable; this one is not, and that
    is faithful rather than a regression.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var c = Int(c_k)
    var sz = Int(size)
    if c > sz:
        c = sz
    if c < 1:
        c = 1
    var off = Int(u * Float32(c))
    if off >= c:
        off = c - 1
    if off < 0:
        off = 0
    var idx = (Int(write_pos) - c + off + CAP) % CAP
    if idx < 0:
        idx = idx + CAP
    indices[i] = Scalar[IDX_DT](idx)



def _range_indices_kernel[
    BATCH: Int
](
    indices: Tile[IDX_DT, BATCH],
    lo: Int32,
    hi: Int32,
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Uniform index in `[lo, hi)` — MBPO's dynamics train/holdout split.

    Port of `gpu_replay.mojo::_sample_indices_range_kernel`. The range is a
    host scalar, matching the legacy: that split is not CUDA-graph captured.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var lo_i = Int(lo)
    var hi_i = Int(hi)
    var span = hi_i - lo_i
    if span < 1:
        span = 1
    var idx = lo_i + Int(u * Float32(span))
    if idx >= hi_i:
        idx = hi_i - 1
    if idx < lo_i:
        idx = lo_i
    indices[i] = Scalar[IDX_DT](idx)


# ── PER: device sum-tree ──────────────────────────────────────────────────

def _per_leafset_new_kernel[
    N: Int, CAP: Int
](
    tree: Tile[DT, 2 * CAP - 1],
    max_p: Tile[DT, 1],
    start_pos: Int32,
    alpha: Scalar[DT],
):
    """New rows enter at `max_priority^alpha`. `max_p` is read on device so
    the ceiling tracks updates without host involvement."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N:
        return
    var leaf = (Int(start_pos) + e) % CAP
    tree[CAP - 1 + leaf] = fpow(rebind[Scalar[DT]](max_p[0]), alpha)


def _per_update_leaves_kernel[
    BATCH: Int, CAP: Int
](
    tree: Tile[DT, 2 * CAP - 1],
    indices: Tile[IDX_DT, BATCH],
    td: Tile[DT, BATCH],
    alpha: Scalar[DT],
    epsilon: Scalar[DT],
):
    """`tree[leaf] = (|td| + eps)^alpha`.

    ⚠ Duplicate-leaf determinism: the host loop applies lanes sequentially, so
    the LAST lane touching a leaf wins. Mirrored here by letting only the last
    duplicate write (O(BATCH) scan per thread — negligible). Without this the
    winner is whichever thread happens to run last, and the tree diverges
    non-deterministically from the host path.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var leaf = Int(indices[i])
    for j in range(i + 1, BATCH):
        if Int(indices[j]) == leaf:
            return
    var td_i = rebind[Scalar[DT]](td[i])
    var td_abs = td_i if td_i >= Scalar[DT](0.0) else -td_i
    tree[CAP - 1 + leaf] = fpow(td_abs + epsilon, alpha)


def _per_max_priority_kernel[
    BATCH: Int
](
    max_p: Tile[DT, 1],
    td: Tile[DT, BATCH],
    epsilon: Scalar[DT],
):
    """Single-thread max reduction — BATCH is small and the codebase avoids
    float atomics by convention."""
    if Int(thread_idx.x) != 0:
        return
    var m = rebind[Scalar[DT]](max_p[0])
    for i in range(BATCH):
        var td_i = rebind[Scalar[DT]](td[i])
        var td_abs = td_i if td_i >= Scalar[DT](0.0) else -td_i
        var raw = td_abs + epsilon
        if raw > m:
            m = raw
    max_p[0] = m


def _per_tree_propagate_kernel[
    CAP: Int
](tree: Tile[DT, 2 * CAP - 1],):
    """Rebuild every internal node bottom-up, level by level, in ONE
    single-block launch with a barrier between levels. Full-level recompute
    sidesteps the concurrent ancestor-path race that per-leaf propagation
    would have, and needs no atomics. Work is ~CAP adds total."""
    var tid = Int(thread_idx.x)
    var tpb = Int(block_dim.x)
    var l_start = 0
    while (1 << (l_start + 1)) - 1 <= CAP - 2:
        l_start += 1
    var l = l_start
    while l >= 0:
        var lo = (1 << l) - 1
        var hi = (1 << (l + 1)) - 2
        if hi > CAP - 2:
            hi = CAP - 2
        var node = lo + tid
        while node <= hi:
            tree[node] = rebind[Scalar[DT]](tree[2 * node + 1]) + rebind[
                Scalar[DT]
            ](tree[2 * node + 2])
            node += tpb
        barrier()
        l -= 1


def _per_sample_kernel[
    BATCH: Int, CAP: Int
](
    tree: Tile[DT, 2 * CAP - 1],
    size_buf: Tile[DType.int32, 1],
    out_idx: Tile[IDX_DT, BATCH],
    out_w: Tile[DT, BATCH],
    beta: Scalar[DT],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Stratified draw + iterative root-to-leaf descent, one thread per lane.
    `<=` on the left child, matching the host descent exactly."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var total = rebind[Scalar[DT]](tree[0])
    var size = Int(size_buf[0])
    if total <= Scalar[DT](0.0) or size < 1:
        out_idx[i] = Scalar[IDX_DT](0)
        out_w[i] = Scalar[DT](1.0)
        return
    var segment = total / Scalar[DT](BATCH)
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var r = Scalar[DT](Float32(philox.step_uniform()[0]))
    var u = segment * (Scalar[DT](i) + r)
    if u >= total:
        u = total - Scalar[DT](1e-7)
    if u < Scalar[DT](0.0):
        u = Scalar[DT](0.0)
    var idx = 0
    while idx < CAP - 1:
        var left = 2 * idx + 1
        var left_sum = rebind[Scalar[DT]](tree[left])
        if u <= left_sum:
            idx = left
        else:
            u = u - left_sum
            idx = left + 1
    var leaf = idx - (CAP - 1)
    if leaf >= size:
        leaf = size - 1
    out_idx[i] = Scalar[IDX_DT](leaf)
    var p_leaf = rebind[Scalar[DT]](tree[CAP - 1 + leaf])
    out_w[i] = fpow(Scalar[DT](size) * (p_leaf / total), -beta)


def _per_normalize_weights_kernel[
    BATCH: Int
](w: Tile[DT, BATCH],):
    """Normalize so max sampled weight == 1, matching the host two-pass."""
    if Int(thread_idx.x) != 0:
        return
    var max_w = Scalar[DT](0.0)
    for i in range(BATCH):
        var wi = rebind[Scalar[DT]](w[i])
        if wi > max_w:
            max_w = wi
    if max_w <= Scalar[DT](0.0):
        max_w = Scalar[DT](1.0)
    for i in range(BATCH):
        w[i] = rebind[Scalar[DT]](w[i]) / max_w


def _per_copy_weights_kernel[
    BATCH: Int
](
    dst: Tile[DT, BATCH],
    src: Tile[DT, BATCH],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    dst[i] = src[i]


# ── the buffer ────────────────────────────────────────────────────────────

struct StoreReplayGpu[
    OBS_: Int,
    ACT_: Int,
    CAP_: Int,
    PRIORITIZED: Bool = False,
    OBS_STORE_DT_: DType = DT,
](ReplayBuffer):
    """Device ring + a device index policy. `PRIORITIZED` is a comptime FLAG,
    matching the CPU `StoreReplay`: ONE storage struct serves both policies,
    because the sum-tree is a sampler, not a storage subclass.

    The tree buffers are `Optional`, so a uniform buffer pays nothing for
    them."""
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_
    comptime SDT = Self.OBS_STORE_DT_
    """Obs storage dtype. `DT` stores verbatim; `uint8` quantises — see the
    module's quantisation note."""

    var obs: DeviceBuffer[Self.SDT]
    var act: DeviceBuffer[DT]
    var rew: DeviceBuffer[DT]
    var nxt: DeviceBuffer[Self.SDT]
    var dne: DeviceBuffer[DT]

    var stage_obs: DeviceBuffer[DT]
    var stage_act: DeviceBuffer[DT]
    var stage_rew: DeviceBuffer[DT]
    var stage_nxt: DeviceBuffer[DT]
    var stage_dne: DeviceBuffer[DT]
    var idx_buf: DeviceBuffer[IDX_DT]

    var _h_rew: List[Scalar[DT]]
    var _h_dne: List[Scalar[DT]]

    var size: Int
    var pos: Int
    var sampler: UniformDeviceSampler
    var batch_capacity: Int

    # Device-resident so the sample path stays CUDA-graph capturable.
    var size_dev: DeviceBuffer[DType.int32]
    var offset_dev: DeviceBuffer[DType.uint64]

    # The pinned DEMO prefix — see `pin_demo_prefix`. `demo_n` rows at the
    # front are never overwritten and half of every minibatch is drawn from
    # them (`_mixed_indices_dev_kernel`). 0 = no prefix, the plain ring.
    var demo_n: Int
    var demo_dev: DeviceBuffer[DType.int32]

    # PER state — present only when PRIORITIZED.
    var tree: Optional[DeviceBuffer[DT]]
    var max_p: Optional[DeviceBuffer[DT]]
    var w_buf: Optional[DeviceBuffer[DT]]
    var alpha: Scalar[DT]
    var beta: Scalar[DT]
    var epsilon: Scalar[DT]

    # ERE (Wang & Ross 2019): c_k = clamp(floor(size * eta^k), c_min, size).
    var ere_enabled: Bool
    var ere_eta: Scalar[DT]
    var _ere_k: Int
    var _ere_k_max: Int
    var _ere_eta_pow_k: Scalar[DT]
    var _ere_c_min: Int

    def __init__(
        out self,
        var obs: DeviceBuffer[Self.SDT],
        var act: DeviceBuffer[DT],
        var rew: DeviceBuffer[DT],
        var nxt: DeviceBuffer[Self.SDT],
        var dne: DeviceBuffer[DT],
        var stage_obs: DeviceBuffer[DT],
        var stage_act: DeviceBuffer[DT],
        var stage_rew: DeviceBuffer[DT],
        var stage_nxt: DeviceBuffer[DT],
        var stage_dne: DeviceBuffer[DT],
        var idx_buf: DeviceBuffer[IDX_DT],
        var _h_rew: List[Scalar[DT]],
        var _h_dne: List[Scalar[DT]],
        size: Int,
        pos: Int,
        var sampler: UniformDeviceSampler,
        batch_capacity: Int,
        var size_dev: DeviceBuffer[DType.int32],
        var offset_dev: DeviceBuffer[DType.uint64],
        var demo_dev: DeviceBuffer[DType.int32],
        var tree: Optional[DeviceBuffer[DT]],
        var max_p: Optional[DeviceBuffer[DT]],
        var w_buf: Optional[DeviceBuffer[DT]],
        alpha: Scalar[DT],
        beta: Scalar[DT],
        epsilon: Scalar[DT],
    ):
        self.ere_enabled = False
        self.ere_eta = Scalar[DT](0.996)
        self._ere_k = 0
        self._ere_k_max = 1000
        self._ere_eta_pow_k = Scalar[DT](1.0)
        self._ere_c_min = 1
        self.obs = obs^
        self.act = act^
        self.rew = rew^
        self.nxt = nxt^
        self.dne = dne^
        self.stage_obs = stage_obs^
        self.stage_act = stage_act^
        self.stage_rew = stage_rew^
        self.stage_nxt = stage_nxt^
        self.stage_dne = stage_dne^
        self.idx_buf = idx_buf^
        self._h_rew = _h_rew^
        self._h_dne = _h_dne^
        self.size = size
        self.pos = pos
        self.sampler = sampler^
        self.batch_capacity = batch_capacity
        self.size_dev = size_dev^
        self.offset_dev = offset_dev^
        self.demo_n = 0
        self.demo_dev = demo_dev^
        self.tree = tree^
        self.max_p = max_p^
        self.w_buf = w_buf^
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def __init__(out self, *, deinit move: Self):
        self.obs = move.obs^
        self.act = move.act^
        self.rew = move.rew^
        self.nxt = move.nxt^
        self.dne = move.dne^
        self.stage_obs = move.stage_obs^
        self.stage_act = move.stage_act^
        self.stage_rew = move.stage_rew^
        self.stage_nxt = move.stage_nxt^
        self.stage_dne = move.stage_dne^
        self.idx_buf = move.idx_buf^
        self._h_rew = move._h_rew^
        self._h_dne = move._h_dne^
        self.size = move.size
        self.pos = move.pos
        self.sampler = move.sampler^
        self.batch_capacity = move.batch_capacity
        self.size_dev = move.size_dev^
        self.offset_dev = move.offset_dev^
        self.demo_n = move.demo_n
        self.demo_dev = move.demo_dev^
        self.tree = move.tree^
        self.max_p = move.max_p^
        self.w_buf = move.w_buf^
        self.alpha = move.alpha
        self.beta = move.beta
        self.epsilon = move.epsilon
        self.ere_enabled = move.ere_enabled
        self.ere_eta = move.ere_eta
        self._ere_k = move._ere_k
        self._ere_k_max = move._ere_k_max
        self._ere_eta_pow_k = move._ere_eta_pow_k
        self._ere_c_min = move._ere_c_min

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        if not ctx:
            raise Error("StoreReplayGpu.make: ctx required (GPU backend)")
        var c = ctx.value()
        var s = c.enqueue_create_buffer[Self.SDT](Self.CAP * Self.OBS)
        var a = c.enqueue_create_buffer[DT](Self.CAP * Self.ACT)
        var r = c.enqueue_create_buffer[DT](Self.CAP)
        var sp = c.enqueue_create_buffer[Self.SDT](Self.CAP * Self.OBS)
        var d = c.enqueue_create_buffer[DT](Self.CAP)
        s.enqueue_fill(Scalar[Self.SDT](0))
        a.enqueue_fill(Scalar[DT](0))
        r.enqueue_fill(Scalar[DT](0))
        sp.enqueue_fill(Scalar[Self.SDT](0))
        d.enqueue_fill(Scalar[DT](0))

        var sz_dev = c.enqueue_create_buffer[DType.int32](1)
        sz_dev.enqueue_fill(Int32(0))
        var off_dev = c.enqueue_create_buffer[DType.uint64](1)
        off_dev.enqueue_fill(UInt64(0))
        var demo_dev = c.enqueue_create_buffer[DType.int32](1)
        demo_dev.enqueue_fill(Int32(0))

        var tree_opt = Optional[DeviceBuffer[DT]](None)
        var maxp_opt = Optional[DeviceBuffer[DT]](None)
        var w_opt = Optional[DeviceBuffer[DT]](None)
        comptime if Self.PRIORITIZED:
            var t = c.enqueue_create_buffer[DT](2 * Self.CAP - 1)
            t.enqueue_fill(Scalar[DT](0))
            var mp = c.enqueue_create_buffer[DT](1)
            mp.enqueue_fill(Scalar[DT](1.0))
            var wb = c.enqueue_create_buffer[DT](batch_capacity)
            wb.enqueue_fill(Scalar[DT](1.0))
            tree_opt = t^
            maxp_opt = mp^
            w_opt = wb^

        return Self(
            obs=s^, act=a^, rew=r^, nxt=sp^, dne=d^,
            stage_obs=c.enqueue_create_buffer[DT](Self.OBS),
            stage_act=c.enqueue_create_buffer[DT](Self.ACT),
            stage_rew=c.enqueue_create_buffer[DT](1),
            stage_nxt=c.enqueue_create_buffer[DT](Self.OBS),
            stage_dne=c.enqueue_create_buffer[DT](1),
            idx_buf=c.enqueue_create_buffer[IDX_DT](batch_capacity),
            _h_rew=List[Scalar[DT]](length=1, fill=Scalar[DT](0)),
            _h_dne=List[Scalar[DT]](length=1, fill=Scalar[DT](0)),
            size=0,
            pos=0,
            # Seed matches GPUReplay so a migrated call site draws the SAME
            # sequence — the parity gate depends on it.
            sampler=UniformDeviceSampler(
                0, seed=UInt64(0xC0FFEE_DECADE_0042), offset=UInt64(0)
            ),
            batch_capacity=batch_capacity,
            size_dev=sz_dev^,
            offset_dev=off_dev^,
            demo_dev=demo_dev^,
            tree=tree_opt^,
            max_p=maxp_opt^,
            w_buf=w_opt^,
            alpha=Scalar[DT](0.6),
            beta=Scalar[DT](0.4),
            epsilon=Scalar[DT](1e-6),
        )

    def count(self) -> Int:
        return self.size

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if not ctx:
            raise Error("StoreReplayGpu.add: ctx required")
        var c = ctx.value()
        self._h_rew[0] = r
        self._h_dne[0] = d
        c.enqueue_copy(self.stage_obs, s.unsafe_ptr())
        c.enqueue_copy(self.stage_act, a.unsafe_ptr())
        c.enqueue_copy(self.stage_rew, self._h_rew.unsafe_ptr())
        c.enqueue_copy(self.stage_nxt, sp.unsafe_ptr())
        c.enqueue_copy(self.stage_dne, self._h_dne.unsafe_ptr())

        comptime tpb = Self.OBS if Self.OBS > Self.ACT else Self.ACT
        comptime kern = _store_one_kernel[Self.OBS, Self.ACT, Self.CAP, Self.SDT]
        c.enqueue_function[kern](
            tile[Self.OBS](self.stage_obs),
            tile[Self.ACT](self.stage_act),
            tile[1](self.stage_rew),
            tile[Self.OBS](self.stage_nxt),
            tile[1](self.stage_dne),
            tile[Self.CAP, Self.OBS](self.obs),
            tile[Self.CAP, Self.ACT](self.act),
            tile[Self.CAP](self.rew),
            tile[Self.CAP, Self.OBS](self.nxt),
            tile[Self.CAP](self.dne),
            Int32(self.pos),
            grid_dim=1,
            block_dim=tpb,
        )
        comptime if Self.PRIORITIZED:
            c.enqueue_function[_per_leafset_new_kernel[1, Self.CAP]](
                tile[2 * Self.CAP - 1](self.tree.value()),
                tile[1](self.max_p.value()),
                Int32(self.pos),
                self.alpha,
                grid_dim=1,
                block_dim=1,
            )
            c.enqueue_function[_per_tree_propagate_kernel[Self.CAP]](
                tile[2 * Self.CAP - 1](self.tree.value()),
                grid_dim=1,
                block_dim=TPB,
            )
        self.pos = self._next_pos(1)
        if self.size < Self.CAP:
            self.size += 1
        self._sync_size(c)

    def _next_pos(self, k: Int) -> Int:
        """The write pointer after `k` rows, wrapping INSIDE the online
        region `[demo_n, CAP)` so a pinned prefix is never overwritten."""
        var b = self.demo_n
        return b + ((self.pos - b) + k) % (Self.CAP - b)

    def pin_demo_prefix(
        mut self, n: Int, ctx: Optional[DeviceContext] = None
    ) raises:
        """Declare rows `[0, n)` — everything added so far — as DEMONSTRATIONS.

        RLPD / HIL-SERL keep the demos in a second buffer and sample half of
        every batch from it; here the two buffers are ONE storage and the
        prefix is the demo half. After this call the ring's writes wrap inside
        `[n, CAP)` and `sample_into` draws `BATCH/2` rows from `[0, n)` and
        `BATCH/2` from `[n, size)` (all from the prefix until an online row
        exists). `n == 0` is the plain ring.

        ⚠ CALL ONCE, RIGHT AFTER THE DEMOS ARE ADDED, BEFORE ANY ONLINE ROW:
        `size` must equal `n` (the prefix is what has been added, nothing
        else), and the prefix must leave room for a minibatch of online rows.
        Not combinable with PER or ERE, whose index kernels know one region.
        """
        comptime if Self.PRIORITIZED:
            raise Error("pin_demo_prefix: not available on a prioritized replay")
        if self.ere_enabled:
            raise Error("pin_demo_prefix: not available with ERE enabled")
        if n < 0 or n != self.size:
            raise Error(
                "pin_demo_prefix: n=" + String(n) + " but the buffer holds "
                + String(self.size) + " rows — pin exactly what was added"
            )
        if n >= Self.CAP - self.batch_capacity:
            raise Error(
                "pin_demo_prefix: " + String(n) + " demo rows leave fewer than"
                " batch_capacity=" + String(self.batch_capacity)
                + " online rows in CAP=" + String(Self.CAP)
            )
        if not ctx:
            raise Error("pin_demo_prefix: ctx required (GPU backend)")
        self.demo_n = n
        self.pos = n
        # The device word the sample kernel reads — one launch, no wait.
        ctx.value().enqueue_function[_set_size_kernel](
            tile[1](self.demo_dev),
            Int32(n),
            grid_dim=1,
            block_dim=1,
        )

    def demo_count(self) -> Int:
        return self.demo_n

    def _sync_size(mut self, ctx: DeviceContext) raises:
        """Mirror the host `size` into the device buffer the sample kernels
        read. Device-resident so capture cannot bake a stale fill."""
        ctx.enqueue_function[_set_size_kernel](
            tile[1](self.size_dev),
            Int32(self.size),
            grid_dim=1,
            block_dim=1,
        )

    def add_batch[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Device multi-env store — the GPU-batched driver path."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        # ⚠ The `src_*` buffers arrive as READ-ONLY arguments, so the
        # LayoutTensor built from one carries an immutable origin and does
        # NOT convert to the kernel's `MutAnyOrigin` parameter. The rebind is
        # what legacy `gpu_replay.add_batch` did and is load-bearing: without
        # it this method simply fails to instantiate (and, being generic, it
        # stayed silently uncompiled until the first GPU-batched call site).
        var src_obs_lt = rebind[
            Tile[DT, N_ENVS, Self.OBS]
        ](tile[N_ENVS, Self.OBS](src_obs))
        var src_act_lt = rebind[
            Tile[DT, N_ENVS, Self.ACT]
        ](tile[N_ENVS, Self.ACT](src_act))
        var src_rew_lt = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](src_rew))
        var src_nxt_lt = rebind[
            Tile[DT, N_ENVS, Self.OBS]
        ](tile[N_ENVS, Self.OBS](src_nxt))
        var src_dne_lt = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](src_dne))
        comptime n_blocks = (N_ENVS * Self.OBS + TPB - 1) // TPB
        comptime kern = _store_batch_kernel[
            N_ENVS, Self.OBS, Self.ACT, Self.CAP, Self.SDT
        ]
        ctx.enqueue_function[kern](
            src_obs_lt,
            src_act_lt,
            src_rew_lt,
            src_nxt_lt,
            src_dne_lt,
            tile[Self.CAP, Self.OBS](self.obs),
            tile[Self.CAP, Self.ACT](self.act),
            tile[Self.CAP](self.rew),
            tile[Self.CAP, Self.OBS](self.nxt),
            tile[Self.CAP](self.dne),
            Int32(self.pos),
            Int32(self.demo_n),
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        comptime if Self.PRIORITIZED:
            comptime nb = (N_ENVS + TPB - 1) // TPB
            ctx.enqueue_function[
                _per_leafset_new_kernel[N_ENVS, Self.CAP]
            ](
                tile[2 * Self.CAP - 1](self.tree.value()),
                tile[1](self.max_p.value()),
                Int32(self.pos),
                self.alpha,
                grid_dim=nb,
                block_dim=TPB,
            )
            ctx.enqueue_function[_per_tree_propagate_kernel[Self.CAP]](
                tile[2 * Self.CAP - 1](self.tree.value()),
                grid_dim=1,
                block_dim=TPB,
            )
        self.pos = self._next_pos(N_ENVS)
        self.size = self.size + N_ENVS
        if self.size > Self.CAP:
            self.size = Self.CAP
        self._sync_size(ctx)

    def sample_into[
        BATCH: Int
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        if BATCH > self.batch_capacity:
            raise Error(
                "StoreReplayGpu.sample_into: BATCH=" + String(BATCH)
                + " exceeds batch_capacity=" + String(self.batch_capacity)
            )
        var ctx = state.ctx.value()
        var idx_lt = tile[BATCH](self.idx_buf)
        var size_lt = tile[1](self.size_dev)
        var off_lt = tile[1](self.offset_dev)
        comptime n_idx_blocks = (BATCH + TPB - 1) // TPB

        comptime if Self.PRIORITIZED:
            var tree_lt = tile[2 * Self.CAP - 1](self.tree.value())
            var w_lt = tile[BATCH](self.w_buf.value())
            ctx.enqueue_function[_per_sample_kernel[BATCH, Self.CAP]](
                tree_lt, size_lt, idx_lt, w_lt, self.beta,
                UInt64(0xC0FFEE_DECADE_0042), off_lt,
                grid_dim=n_idx_blocks, block_dim=TPB,
            )
            ctx.enqueue_function[_per_normalize_weights_kernel[BATCH]](
                w_lt, grid_dim=1, block_dim=1,
            )
            ctx.enqueue_function[_per_copy_weights_kernel[BATCH]](
                tile[BATCH](state.mb_w.dev.value()),
                w_lt,
                grid_dim=n_idx_blocks, block_dim=TPB,
            )
            state.has_per = True
        else:
            if self.demo_n > 0:
                ctx.enqueue_function[_mixed_indices_dev_kernel[BATCH]](
                    idx_lt, size_lt,
                    tile[1](self.demo_dev),
                    UInt64(0xC0FFEE_DECADE_0042), off_lt,
                    grid_dim=n_idx_blocks, block_dim=TPB,
                )
            elif self.ere_enabled:
                # c_k = clamp(floor(size * eta^k), c_min, size), host-side —
                # matching the legacy, which is deliberately not capturable.
                var c = Int(Scalar[DT](self.size) * self._ere_eta_pow_k)
                if c < self._ere_c_min:
                    c = self._ere_c_min
                if c > self.size:
                    c = self.size
                if c < 1:
                    c = 1
                ctx.enqueue_function[_ere_indices_kernel[BATCH, Self.CAP]](
                    idx_lt, Int32(self.size), Int32(self.pos), Int32(c),
                    UInt64(0xC0FFEE_DECADE_0042), off_lt,
                    grid_dim=n_idx_blocks, block_dim=TPB,
                )
                self._ere_k = self._ere_k + 1
                self._ere_eta_pow_k = self._ere_eta_pow_k * self.ere_eta
                if self._ere_k >= self._ere_k_max:
                    self._ere_k = 0
                    self._ere_eta_pow_k = Scalar[DT](1.0)
            else:
                ctx.enqueue_function[_uniform_indices_dev_kernel[BATCH]](
                    idx_lt, size_lt, UInt64(0xC0FFEE_DECADE_0042), off_lt,
                    grid_dim=n_idx_blocks, block_dim=TPB,
                )
        ctx.enqueue_function[_incr_offset_kernel[BATCH]](
            off_lt, grid_dim=1, block_dim=1,
        )

        comptime n_blocks = (BATCH * Self.OBS + TPB - 1) // TPB
        comptime kern = _gather_batch_kernel[
            BATCH, Self.OBS, Self.ACT, Self.CAP, Self.SDT
        ]
        ctx.enqueue_function[kern](
            tile[BATCH, Self.OBS](state.mb_s.dev.value()),
            tile[BATCH, Self.ACT](state.mb_a.dev.value()),
            tile[BATCH](state.mb_r.dev.value()),
            tile[BATCH, Self.OBS](state.mb_sp.dev.value()),
            tile[BATCH](state.mb_d.dev.value()),
            tile[Self.CAP, Self.OBS](self.obs),
            tile[Self.CAP, Self.ACT](self.act),
            tile[Self.CAP](self.rew),
            tile[Self.CAP, Self.OBS](self.nxt),
            tile[Self.CAP](self.dne),
            tile[BATCH](self.idx_buf),
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def set_beta(mut self, beta: Scalar[DT]):
        self.beta = beta

    def update_priorities[
        BATCH: Int
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        comptime if Self.PRIORITIZED:
            var ctx = state.ctx.value()
            var td = tile[BATCH](state.td_residuals.dev.value())
            var tree_lt = tile[2 * Self.CAP - 1](self.tree.value())
            var maxp_lt = tile[1](self.max_p.value())
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            ctx.enqueue_function[
                _per_update_leaves_kernel[BATCH, Self.CAP]
            ](
                tree_lt,
                tile[BATCH](self.idx_buf),
                td,
                self.alpha,
                self.epsilon,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            ctx.enqueue_function[_per_max_priority_kernel[BATCH]](
                maxp_lt, td, self.epsilon, grid_dim=1, block_dim=1,
            )
            ctx.enqueue_function[_per_tree_propagate_kernel[Self.CAP]](
                tree_lt, grid_dim=1, block_dim=TPB,
            )

    def sample[
        N: Int
    ](
        mut self,
        ctx: DeviceContext,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Uniform draw into CALLER device buffers.

        `sample_into` targets a `TrainerState`; MBPO needs the raw form for
        its dynamics-model batches and rollout start states.
        """
        var idx_lt = tile[N](self.idx_buf)
        var size_lt = tile[1](self.size_dev)
        var off_lt = tile[1](self.offset_dev)
        comptime nb = (N + TPB - 1) // TPB
        ctx.enqueue_function[_uniform_indices_dev_kernel[N]](
            idx_lt, size_lt, UInt64(0xC0FFEE_DECADE_0042), off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        ctx.enqueue_function[_incr_offset_kernel[N]](
            off_lt, grid_dim=1, block_dim=1,
        )
        self._launch_gather[N](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    def sample_range[
        N: Int
    ](
        mut self,
        ctx: DeviceContext,
        lo: Int,
        hi: Int,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Uniform draw restricted to rows `[lo, hi)` — MBPO's dynamics
        train/holdout split."""
        var idx_lt = tile[N](self.idx_buf)
        var off_lt = tile[1](self.offset_dev)
        comptime nb = (N + TPB - 1) // TPB
        ctx.enqueue_function[_range_indices_kernel[N]](
            idx_lt, Int32(lo), Int32(hi),
            UInt64(0xC0FFEE_DECADE_0042), off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        ctx.enqueue_function[_incr_offset_kernel[N]](
            off_lt, grid_dim=1, block_dim=1,
        )
        self._launch_gather[N](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    def _launch_gather[
        N: Int
    ](
        mut self,
        ctx: DeviceContext,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        # Same immutable-origin rebind as `add_batch` — these are the kernel's
        # DESTINATIONS and arrive read-only, so without it `sample` /
        # `sample_range` (the MBPO raw-buffer path) fail to instantiate.
        var mb_s_lt = rebind[
            Tile[DT, N, Self.OBS]
        ](tile[N, Self.OBS](mb_s))
        var mb_a_lt = rebind[
            Tile[DT, N, Self.ACT]
        ](tile[N, Self.ACT](mb_a))
        var mb_r_lt = rebind[
            Tile[DT, N]
        ](tile[N](mb_r))
        var mb_sp_lt = rebind[
            Tile[DT, N, Self.OBS]
        ](tile[N, Self.OBS](mb_sp))
        var mb_d_lt = rebind[
            Tile[DT, N]
        ](tile[N](mb_d))
        comptime nbg = (N * Self.OBS + TPB - 1) // TPB
        comptime kern = _gather_batch_kernel[
            N, Self.OBS, Self.ACT, Self.CAP, Self.SDT
        ]
        ctx.enqueue_function[kern](
            mb_s_lt,
            mb_a_lt,
            mb_r_lt,
            mb_sp_lt,
            mb_d_lt,
            tile[Self.CAP, Self.OBS](self.obs),
            tile[Self.CAP, Self.ACT](self.act),
            tile[Self.CAP](self.rew),
            tile[Self.CAP, Self.OBS](self.nxt),
            tile[Self.CAP](self.dne),
            tile[N](self.idx_buf),
            grid_dim=nbg, block_dim=TPB,
        )

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        """Enable ERE recency-biased sampling.

        ⚠ Not supported together with PER — the legacy pairing is
        `GPUReplay` + ERE or `GPUPrioritizedReplay` (no ERE), never both.
        Raising beats silently ignoring one of the two policies.
        """
        comptime if Self.PRIORITIZED:
            if enable:
                raise Error(
                    "StoreReplayGpu: ERE and PER cannot both be enabled"
                    " (the legacy buffers never combined them either)."
                )
            return
        self.ere_enabled = enable
        if enable:
            self.ere_eta = eta
            self._ere_c_min = c_min
            self._ere_k_max = k_max
            self._ere_k = 0
            self._ere_eta_pow_k = Scalar[DT](1.0)
