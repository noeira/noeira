"""PPOMinibatchGatherStep — Fisher-Yates shuffle + minibatch gather (STORAGE).

Three methods:
  - `reset_indices[target]` — write [0..ROLLOUT_LEN) into state.indices
    (called once per rollout, BEFORE the K-epoch loop — epoch shuffles
    operate on whatever state the previous epoch left behind).
  - `shuffle_epoch[target]` — in-place Fisher-Yates over state.indices
    (called once per K-epoch, AFTER reset_indices on the first epoch).
  - `gather[target]` — gather the `mb_idx`-th minibatch into mb_obs /
    mb_act / mb_olp / mb_adv / mb_ret, then mean/std normalise mb_adv
    (CleanRL per-minibatch normalisation).

STORAGE migration: the rollout pool + the minibatch staging both index the
storage tensors' host `.data` Lists directly. On GPU the populated mb_* host
mirrors are `upload`ed so the actor/critic train steps read the device buffers.
Indices stay an Int32 raw pointer on `state.indices` (Tensor is DT-only).

## The device path (CUDA-graph capture of the K-epoch update)

`gather` stages each minibatch on the host and `upload`s five tensors, which
reallocates and synchronises twice per tensor: ten device syncs per minibatch,
and a capture blocker. The device path moves the per-minibatch work into
kernels so one minibatch step can be captured once and replayed:

  - `stage_epoch_indices` — after `shuffle_epoch`, copy that epoch's shuffled
    order into `_idx_all` at the epoch's offset, AND each of its minibatches'
    normalised advantages into `_adv_all` — on the host, through the same
    `_normalize_minibatch_adv` the eager `gather` uses, so the two paths feed
    the actor bit-identical advantages (a device kernel's f32 division and
    square root need not round like the host's). The host Fisher-Yates and its
    RNG stream are unchanged, so the minibatches are the eager path's.
  - `upload_device_update[N_EPOCHS]` — once per rollout: the rollout pool,
    every epoch's indices and normalised advantages, `upload_resident` (stable
    pointers), and the device minibatch counter set to 0.
  - `gather_device` — CAPTURED. Gathers minibatch `counter` from the device
    pool (a pure copy).
  - `advance_counter` — CAPTURED, last in the step: `counter += 1`, so each
    replay gathers the next minibatch.
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.random import random_float64
from std.math import sqrt as fsqrt

from noeira.nn.constants import DT, TPB
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.tensor import Tensor
from ...training.onpolicy_state import OnPolicyState


def _ppo_gather_kernel[
    OBS: Int, ACT: Int, MB: Int, RN: Int, N_IDX: Int
](
    counter: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    idx: LayoutTensor[DT, Layout.row_major(N_IDX), MutAnyOrigin],
    obs: LayoutTensor[DT, Layout.row_major(RN * OBS), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(RN * ACT), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(RN), MutAnyOrigin],
    adv_all: LayoutTensor[DT, Layout.row_major(N_IDX), MutAnyOrigin],
    ret: LayoutTensor[DT, Layout.row_major(RN), MutAnyOrigin],
    mb_obs: LayoutTensor[DT, Layout.row_major(MB * OBS), MutAnyOrigin],
    mb_act: LayoutTensor[DT, Layout.row_major(MB * ACT), MutAnyOrigin],
    mb_olp: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    mb_adv: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    mb_ret: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
):
    """One thread per minibatch row: the device twin of `gather`'s copy loop.
    The minibatch number is read from `counter`, not passed, so a captured
    launch gathers a different minibatch on every replay. `adv_all` holds the
    advantages already normalised per minibatch, in minibatch order."""
    var k = Int(global_idx.x)
    if k >= MB:
        return
    var mb = Int(rebind[Scalar[DT]](counter[0]))
    var src = Int(rebind[Scalar[DT]](idx[mb * MB + k]))
    for d in range(OBS):
        mb_obs[k * OBS + d] = obs[src * OBS + d]
    for j in range(ACT):
        mb_act[k * ACT + j] = act[src * ACT + j]
    mb_olp[k] = olp[src]
    mb_adv[k] = adv_all[mb * MB + k]
    mb_ret[k] = ret[src]


def _ppo_counter_advance_kernel(
    counter: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    if Int(global_idx.x) == 0:
        counter[0] = rebind[Scalar[DT]](counter[0]) + Scalar[DT](1.0)


def _normalize_minibatch_adv(
    mut data: List[Scalar[DT]], start: Int, n: Int
):
    """CleanRL per-minibatch advantage normalisation of `data[start:start+n]`
    in place (subtract the mean, divide by std + 1e-8). The ONE implementation
    behind the eager `gather` and the device path's `stage_epoch_indices`, so
    both produce the same floats."""
    var s: Scalar[DT] = 0.0
    for t in range(n):
        s += data[start + t]
    var mean = s / Scalar[DT](n)
    var sq: Scalar[DT] = 0.0
    for t in range(n):
        var d = data[start + t] - mean
        sq += d * d
    var std = fsqrt(sq / Scalar[DT](n))
    for t in range(n):
        data[start + t] = (data[start + t] - mean) / (std + Scalar[DT](1e-8))


struct PPOMinibatchGatherStep[
    OBS_: Int,
    ACT_: Int,
    ROLLOUT_LEN_: Int,
    MINIBATCH_: Int,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime ROLLOUT_LEN = Self.ROLLOUT_LEN_
    comptime MINIBATCH = Self.MINIBATCH_

    # Device path only (see the module header). Indices are stored as DT:
    # exact for a pool below 2^24 rows, asserted in `upload_device_update`.
    var _idx_all: Tensor
    """Every epoch's shuffled order, `N_EPOCHS * ROLLOUT_LEN * N_ENVS`."""
    var _adv_all: Tensor
    """Every epoch's minibatches' normalised advantages, minibatch order,
    `N_EPOCHS * ROLLOUT_LEN * N_ENVS`."""
    var _counter: Tensor
    """The minibatch the next captured `gather_device` reads, `[1]`."""

    def __init__(out self):
        self._idx_all = Tensor()
        self._adv_all = Tensor()
        self._counter = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOMinibatchGatherStep: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def reset_indices[target: StaticString, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
    ) raises:
        """Write [0..ROLLOUT_LEN*N_ENVS) into state.indices. Caller
        invokes this ONCE per rollout before the K-epoch loop.
        Subsequent epoch shuffles operate on whatever state the
        previous epoch left behind — bit-identity-critical (legacy
        resets once per rollout, not once per epoch)."""
        var idx_p = state.indices.value()
        for k in range(Self.ROLLOUT_LEN * N_ENVS):
            idx_p[unsafe_offset=k] = Int32(k)

    def shuffle_epoch[target: StaticString, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
    ) raises:
        """In-place Fisher-Yates over state.indices (length
        ROLLOUT_LEN*N_ENVS). Caller invokes this once at the top of
        each K-epoch (after `reset_indices` on the first epoch)."""
        var n_total = Self.ROLLOUT_LEN * N_ENVS
        var idx_p = state.indices.value()
        for t in range(n_total - 1, 0, -1):
            var j = Int(random_float64() * Float64(t + 1))
            if j > t:
                j = t
            var tmp = idx_p[unsafe_offset=t]
            idx_p[unsafe_offset=t] = idx_p[unsafe_offset=j]
            idx_p[unsafe_offset=j] = tmp

    def gather[target: StaticString, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        mb_idx: Int,
    ) raises:
        """Gather the `mb_idx`-th minibatch from the flat
        ROLLOUT_LEN*N_ENVS pool into mb_obs/mb_act/mb_olp/mb_adv/mb_ret,
        mean/std normalise mb_adv in place, then (on GPU) H2D upload
        the populated mb_* host mirrors.

        At N_ENVS=1 the flat index space equals the per-env time index,
        so the math reduces to the N=1 case bit-identically."""
        # Gather works on the rollout pool + minibatch staging host `.data`
        # Lists directly (no raw pointers). `state.indices` is an Int32 array
        # (Tensor is DT-only) so it stays a raw pointer.
        ref obs = state.obs_buf.data
        ref act = state.act_buf.data
        ref olp = state.olp_buf.data
        ref adv = state.adv_buf.data
        ref ret = state.ret_buf.data
        ref mb_obs = state.mb_obs.data
        ref mb_act = state.mb_act.data
        ref mb_olp = state.mb_olp.data
        ref mb_adv = state.mb_adv.data
        ref mb_ret = state.mb_ret.data
        var idx_p = state.indices.value()
        for k in range(Self.MINIBATCH):
            var src = Int(idx_p[unsafe_offset=mb_idx * Self.MINIBATCH + k])
            for d in range(Self.OBS):
                mb_obs[k * Self.OBS + d] = obs[src * Self.OBS + d]
            for j in range(Self.ACT):
                mb_act[k * Self.ACT + j] = act[src * Self.ACT + j]
            mb_olp[k] = olp[src]
            mb_adv[k] = adv[src]
            mb_ret[k] = ret[src]
        # CleanRL per-minibatch advantage normalisation.
        _normalize_minibatch_adv(mb_adv, 0, Self.MINIBATCH)

        comptime if target == "gpu":
            # H2D the populated minibatch so the train steps read the device
            # buffers (state.mb_*.lt["gpu", ...] / .dev.value()).
            var c = state.ctx.value()
            state.mb_obs.upload(c)
            state.mb_act.upload(c)
            state.mb_olp.upload(c)
            state.mb_adv.upload(c)
            state.mb_ret.upload(c)

    # ── Device path (CUDA-graph capture) ─────────────────────────────

    def stage_epoch_indices[N_EPOCHS: Int, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        epoch: Int,
    ) raises:
        """Copy this epoch's shuffled order (call right after
        `shuffle_epoch`) into `_idx_all`, and its minibatches' normalised
        advantages into `_adv_all`, at the epoch's offset."""
        comptime RN = Self.ROLLOUT_LEN * N_ENVS
        comptime MB = Self.MINIBATCH
        self._idx_all.ensure(N_EPOCHS * RN)
        self._adv_all.ensure(N_EPOCHS * RN)
        var idx_p = state.indices.value()
        ref adv = state.adv_buf.data
        var base = epoch * RN
        for k in range(RN):
            var src = idx_p[unsafe_offset=k]
            self._idx_all.data[base + k] = Scalar[DT](src)
            self._adv_all.data[base + k] = adv[Int(src)]
        for mb in range(RN // MB):
            _normalize_minibatch_adv(self._adv_all.data, base + mb * MB, MB)

    def upload_device_update[N_EPOCHS: Int, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        pool_on_device: Bool = False,
    ) raises:
        """Once per rollout, after GAE and the epochs' `stage_epoch_indices`:
        the rollout pool and the indices to their EXISTING device buffers
        (`upload_resident`: no reallocation, so a captured gather stays
        valid), and the minibatch counter to 0. All enqueued, no sync.
        `pool_on_device` (the device rollout recorded it there): the pool
        upload is skipped — only the indices and advantages go up."""
        comptime RN = Self.ROLLOUT_LEN * N_ENVS
        comptime assert N_EPOCHS * RN < (1 << 24), (
            "PPOMinibatchGatherStep: the device path stores indices as DT;"
            " N_EPOCHS * ROLLOUT_LEN * N_ENVS must stay below 2^24"
        )
        var c = state.ctx.value()
        if not pool_on_device:
            state.obs_buf.upload_resident(c)
            state.act_buf.upload_resident(c)
            state.olp_buf.upload_resident(c)
            state.ret_buf.upload_resident(c)
        self._idx_all.ensure(N_EPOCHS * RN)
        self._idx_all.upload_resident(c)
        self._adv_all.ensure(N_EPOCHS * RN)
        self._adv_all.upload_resident(c)
        self._counter.ensure_gpu(c, 1)
        fill_dev(self._counter.dev.value(), 1, Scalar[DT](0.0), c)

    def gather_device[N_EPOCHS: Int, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
    ) raises:
        """CAPTURE-SAFE device twin of `gather`: minibatch `_counter` from the
        device pool into the device `mb_*` (advantages already normalised by
        `stage_epoch_indices`). The host `mb_*.data` mirrors are NOT written."""
        comptime RN = Self.ROLLOUT_LEN * N_ENVS
        comptime MB = Self.MINIBATCH
        comptime N_IDX = N_EPOCHS * RN
        var c = state.ctx.value()
        c.enqueue_function[
            _ppo_gather_kernel[Self.OBS, Self.ACT, MB, RN, N_IDX]
        ](
            self._counter.lt["gpu", Layout.row_major(1)](),
            self._idx_all.lt["gpu", Layout.row_major(N_IDX)](),
            state.obs_buf.lt["gpu", Layout.row_major(RN * Self.OBS)](),
            state.act_buf.lt["gpu", Layout.row_major(RN * Self.ACT)](),
            state.olp_buf.lt["gpu", Layout.row_major(RN)](),
            self._adv_all.lt["gpu", Layout.row_major(N_IDX)](),
            state.ret_buf.lt["gpu", Layout.row_major(RN)](),
            state.mb_obs.lt["gpu", Layout.row_major(MB * Self.OBS)](),
            state.mb_act.lt["gpu", Layout.row_major(MB * Self.ACT)](),
            state.mb_olp.lt["gpu", Layout.row_major(MB)](),
            state.mb_adv.lt["gpu", Layout.row_major(MB)](),
            state.mb_ret.lt["gpu", Layout.row_major(MB)](),
            grid_dim=(MB + TPB - 1) // TPB,
            block_dim=TPB,
        )

    def advance_counter[N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
    ) raises:
        """CAPTURE-SAFE: `_counter += 1`. Last in the captured step, after
        everything that read this minibatch."""
        var c = state.ctx.value()
        c.enqueue_function[_ppo_counter_advance_kernel](
            self._counter.lt["gpu", Layout.row_major(1)](),
            grid_dim=1,
            block_dim=1,
        )
