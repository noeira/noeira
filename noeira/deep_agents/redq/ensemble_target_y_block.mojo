"""EnsembleTargetYBlock — REDQ target-y computation (STORAGE surface).

Computes the SAC-style entropy-regularized TD target using a randomized
ensemble of N target critics:

    y[b] = r[b] + (1 - term[b]) * γ * (combined_Q[b] - α * log_π(a'|s')[b])

`combined_Q[b]` is the per-sample reduction of N target-Q evaluations selected
by `MODE`:
  - MODE = MIN — min over `subset_idxs[0..N_MIN]` (paper-faithful REDQ)
  - MODE = AVE — mean over all N critics
  - MODE = REM — GPU-only (deferred; not on this surface)

Pipeline (per step):
  1. `actor.forward(s')` → raw [B, 2·ACT] (mean | log_std)
  2. `rsample.forward(raw)` → packed [B, ACT+1] (action | log_prob)
  3. `sa = concat(s', action)` + extract log_prob[b]
  4. For i in 0..N: `target_net_i.forward(sa)` into row i of `_mb_stacked_q`
  5. `redq_ensemble_target_*` → `state.mb_y[b]` (combine + α·logp + γ + mask)

STORAGE migration (Stage 5): legacy `Scratch`/`TargetStorage`/`mptr`/TileTensor
gone — scratch are owned `nn.storage.Tensor`s (alloc on target); the actor +
RSample + critics use the storage Module surface (`forward[target, B](
TensorRefs, mut out, ctx)`). The combine is the storage `kernels.mojo`
functions. Forward-only — actor + target critics are read but never receive
gradient here.

The `step` surface takes `mut state` (reads mb_sp/mb_r/mb_d, writes mb_y) + the
actor + the ensemble, mirroring the SAC storage `TargetYBlock.step`.
"""

from max.gpu import global_idx
from noeira.nn.core.ptr import untracked
from max.gpu.host import DeviceContext
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT, TPB
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.nn.core.call import call_forward
from noeira.nn.primitives.rsample import RSample
from noeira.nn.random.box_muller import advance_rng_offset_kernel

from ..training.trainer_block import TrainerState
from .ensemble import CriticEnsemble
from .kernels import (
    redq_ensemble_target_cpu,
    redq_ensemble_target_gpu,
    redq_ensemble_target_gpu_dev,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
)
from noeira.core.tiles import Tile


# ────────────────────────────────────────────────────────────────────
# Device subset resample — Fisher-Yates partial shuffle on-device.
# One thread; reads the Philox offset from `offset_buf[0]` (advanced by
# `advance_rng_offset_kernel` after the draw), so each call AND each CUDA-graph
# replay draws a fresh N_MIN-from-N subset without host RNG / upload. Mirrors the
# RSample/box_muller device-RNG-counter pattern. GPU only.
# ────────────────────────────────────────────────────────────────────


def _redq_resample_subset_kernel[
    N: Int, N_MIN: Int
](
    subset: Tile[DType.uint32, N_MIN],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    if Int(global_idx.x) != 0:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var picks = Array[Int, N](uninitialized=True)
    for i in range(N):
        picks[i] = i
    for i in range(N_MIN):
        var rng = PhiloxRandom(seed=seed, offset=offset_base + UInt64(i))
        var u = Float32(rng.step_uniform()[0])
        var j = i + Int(u * Float32(N - i))
        if j >= N:
            j = N - 1
        var tmp = picks[i]
        picks[i] = picks[j]
        picks[j] = tmp
        subset[i] = UInt32(picks[i])


# ────────────────────────────────────────────────────────────────────
# GPU helper kernel — concat(sp, action) + extract log_prob from alp.
# `alp` is the packed RSample output [BATCH, ACT+1] = (action | log_prob).
#   sa[b, :OBS]   = sp[b, :]
#   sa[b, OBS:]   = alp[b, :ACT]      (action portion)
#   lp[b]         = alp[b, ACT]       (log_prob column)
# One thread per output element in `sa`; the lp[] write is gated on d==0.
# ────────────────────────────────────────────────────────────────────


def _redq_concat_sa_extract_lp_kernel[
    OBS: Int, ACT: Int, BATCH: Int, SA_DIM: Int, ALP_DIM: Int,
](
    sp: Tile[DT, BATCH, OBS],
    alp: Tile[DT, BATCH, ALP_DIM],
    sa: Tile[DT, BATCH, SA_DIM],
    lp: Tile[DT, BATCH],
):
    var idx = Int(global_idx.x)
    var total = BATCH * SA_DIM
    if idx >= total:
        return
    var b = idx // SA_DIM
    var d = idx % SA_DIM
    if d < OBS:
        sa[b, d] = rebind[Scalar[DT]](sp[b, d])
    else:
        sa[b, d] = rebind[Scalar[DT]](alp[b, d - OBS])
    if d == 0:
        lp[b] = rebind[Scalar[DT]](alp[b, ACT])


def _copy_row_kernel[
    NB: Int, BATCH: Int
](
    src: Tile[DT, BATCH],
    dst: Tile[DT, NB],
    base_arg: Int64,
):
    """`dst[base + b] = src[b]` — write src into the row `base = i*BATCH` of the
    flat [N*BATCH] stacked buffer."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var base = Int(base_arg)
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    dst.ptr[unsafe_offset=base + b] = rebind[Scalar[DT]](src[b])


struct EnsembleTargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    N_: Int,
    BATCH_: Int,
    OBS_: Int,
    ACT_: Int,
    N_MIN_: Int,
    MODE_: Int,
](Movable & Deinitable):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime N_MIN = Self.N_MIN_
    comptime MODE = Self.MODE_
    comptime SA_DIM = Self.OBS + Self.ACT
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    var _mb_ao: Tensor          # [BATCH, 2*ACT]
    var _mb_alp: Tensor         # [BATCH, ACT+1]
    var _mb_sa: Tensor          # [BATCH, SA_DIM]
    var _mb_stacked_q: Tensor   # [N, BATCH] flat
    var _mb_q_i: Tensor         # [BATCH] per-critic forward scratch
    var _mb_lp: Tensor          # [BATCH]

    var subset_idxs: List[Int]
    # GPU mirror of `subset_idxs` — uploaded once per `step["gpu"]`. None on CPU.
    var _subset_dev: TensorImpl[DType.uint32]

    # CUDA-graph device path (GPU + USE_TRAIN_CUDA_GRAPH). When `_device_resample`
    # is set, `step` resamples the subset on-device (Philox + `_subset_offset`)
    # instead of the host Fisher-Yates + upload, and reads alpha from `_alpha_ptr`
    # (the device temperature buffer) — both capture-safe.
    var _device_resample: Bool
    var _subset_offset: TensorImpl[DType.uint64]  # device Philox offset [1]
    var subset_seed: UInt64
    var _alpha_ptr: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Tensor()
        self._mb_alp = Tensor()
        self._mb_sa = Tensor()
        self._mb_stacked_q = Tensor()
        self._mb_q_i = Tensor()
        self._mb_lp = Tensor()
        # Deterministic first subset: [0, 1, …, N_MIN-1]. The trainer
        # will call resample_subset_idxs() each step in production.
        self.subset_idxs = List[Int](length=Self.N_MIN, fill=0)
        for k in range(Self.N_MIN):
            self.subset_idxs[k] = k
        self._subset_dev = TensorImpl[DType.uint32]()
        self._device_resample = False
        self._subset_offset = TensorImpl[DType.uint64]()
        self.subset_seed = UInt64(0x5EED_F00D)
        self._alpha_ptr = None
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.ctx = None

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleTargetYBlock: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleTargetYBlock.make[target='gpu']: ctx required"
                )
        comptime assert (
            Self.MODE == REDQ_TARGET_MIN or Self.MODE == REDQ_TARGET_AVE
        ), (
            "EnsembleTargetYBlock: MODE must be MIN (0) or AVE (1)."
            " REM (random ensemble mixture) is not on this surface."
        )
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS, (
            "EnsembleTargetYBlock: ACTOR.IN_DIM must equal OBS"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleTargetYBlock: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.SA_DIM, (
            "EnsembleTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleTargetYBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        blk.rsample = RSample[Self.ACT].make[target, Zero](ctx=ctx)
        blk.rsample.action_scale = action_scale
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.ctx = ctx
        comptime if target == "cpu":
            blk._mb_ao = Tensor.alloc(Self.BATCH * (2 * Self.ACT))
            blk._mb_alp = Tensor.alloc(Self.BATCH * Self.ALP_DIM)
            blk._mb_sa = Tensor.alloc(Self.BATCH * Self.SA_DIM)
            blk._mb_stacked_q = Tensor.alloc(Self.N * Self.BATCH)
            blk._mb_q_i = Tensor.alloc(Self.BATCH)
            blk._mb_lp = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            blk._mb_ao = Tensor.alloc_gpu(c, Self.BATCH * (2 * Self.ACT))
            blk._mb_alp = Tensor.alloc_gpu(c, Self.BATCH * Self.ALP_DIM)
            blk._mb_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_DIM)
            blk._mb_stacked_q = Tensor.alloc_gpu(c, Self.N * Self.BATCH)
            blk._mb_q_i = Tensor.alloc_gpu(c, Self.BATCH)
            blk._mb_lp = Tensor.alloc_gpu(c, Self.BATCH)
            blk._subset_dev.ensure_gpu(c, Self.N_MIN)
            blk._subset_offset.ensure_gpu(c, 1)
            blk._subset_offset.dev.value().enqueue_fill(UInt64(0))
        return blk^

    def set_alpha_ptr(mut self, p: Pointer[Scalar[DT], MutAnyOrigin]):
        """Wire REDQ's on-device alpha buffer into the combine (GPU device-alpha
        path). After this, `step` on GPU reads alpha from the device buffer
        instead of the `alpha` arg — CUDA-graph capturable."""
        self._alpha_ptr = untracked(p)

    def enable_device_resample(mut self, seed: UInt64 = UInt64(0x5EED_F00D)):
        """Switch the GPU subset draw to the on-device Philox kernel (no host
        RNG / upload). Required for CUDA-graph capture."""
        self._device_resample = True
        self.subset_seed = seed

    def resample_subset_dev(mut self) raises:
        """Enqueue one on-device Fisher-Yates subset draw into `_subset_dev`
        (reads `_subset_offset`, then advances it). GPU + device path only."""
        var c = self.ctx.value()
        c.enqueue_function[_redq_resample_subset_kernel[Self.N, Self.N_MIN]](
            self._subset_dev.tt["gpu", Self.N_MIN](),
            self.subset_seed,
            self._subset_offset.tt["gpu", 1](),
            grid_dim=1,
            block_dim=1,
        )
        c.enqueue_function[advance_rng_offset_kernel[Self.N_MIN]](
            self._subset_offset.tt["gpu", 1](),
            grid_dim=1,
            block_dim=1,
        )

    def set_subset_idxs(mut self, idxs: List[Int]) raises:
        """Pin the MODE=MIN subset deterministically (test hook)."""
        if len(idxs) != Self.N_MIN:
            raise Error(
                "set_subset_idxs: expected length " + String(Self.N_MIN)
                + ", got " + String(len(idxs))
            )
        for k in range(Self.N_MIN):
            if idxs[k] < 0 or idxs[k] >= Self.N:
                raise Error(
                    "set_subset_idxs: index out of range [0, "
                    + String(Self.N) + ")"
                )
            self.subset_idxs[k] = idxs[k]

    def resample_subset_idxs(mut self) raises:
        """Fisher-Yates partial shuffle: pick N_MIN distinct indices from
        {0..N-1}, store into `self.subset_idxs`. CPU host RNG (GPU mirrors it)."""
        var picks = List[Int](length=Self.N, fill=0)
        for i in range(Self.N):
            picks[i] = i
        for i in range(Self.N_MIN):
            var j = i + Int(random_float64() * Float64(Self.N - i))
            if j >= Self.N:
                j = Self.N - 1
            var tmp = picks[i]
            picks[i] = picks[j]
            picks[j] = tmp
            self.subset_idxs[i] = picks[i]

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        alpha: Scalar[DT],
    ) raises:
        """Write `state.mb_y[b] = r[b] + (1 − term[b]) · γ · (combined_Q[b]
        − α · log_prob[b])` in-place. Forward-only."""
        var ctx = state.ctx

        # 1. actor.forward(s') → _mb_ao [BATCH, 2·ACT].
        call_forward[target, Self.BATCH, POLICY=POLICY](
            actor, TensorRefs[Self.ACTOR.ARITY](state.mb_sp), self._mb_ao, ctx
        )

        # 2. rsample → _mb_alp [BATCH, ACT+1] (packed action | log_prob).
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.rsample, TensorRefs[1](self._mb_ao), self._mb_alp, ctx
        )

        # 3. sa = concat(s', action) + extract log_prob.
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for d in range(Self.OBS):
                    self._mb_sa.data[b * Self.SA_DIM + d] = (
                        state.mb_sp.data[b * Self.OBS + d]
                    )
                for j in range(Self.ACT):
                    self._mb_sa.data[b * Self.SA_DIM + Self.OBS + j] = (
                        self._mb_alp.data[b * Self.ALP_DIM + j]
                    )
                self._mb_lp.data[b] = (
                    self._mb_alp.data[b * Self.ALP_DIM + Self.ACT]
                )
        else:
            var c = ctx.value()
            comptime total_sa = Self.BATCH * Self.SA_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _redq_concat_sa_extract_lp_kernel[
                Self.OBS, Self.ACT, Self.BATCH, Self.SA_DIM, Self.ALP_DIM,
            ]
            c.enqueue_function[kernel](
                state.mb_sp.tt["gpu", Self.BATCH, Self.OBS](),
                self._mb_alp.tt["gpu", Self.BATCH, Self.ALP_DIM](),
                self._mb_sa.tt["gpu", Self.BATCH, Self.SA_DIM](),
                self._mb_lp.tt["gpu", Self.BATCH](),
                grid_dim=n_blocks, block_dim=TPB,
            )

        # 4. Loop N target critic forwards. Each writes its [BATCH, 1] output
        # into row i of `_mb_stacked_q` [N, BATCH] via a per-row Tensor view —
        # but storage Module.forward owns the output `ensure`, so we forward
        # into a [BATCH] scratch then copy. To avoid a per-row scratch + copy
        # we forward each critic into a dedicated per-row sub-Tensor by aliasing
        # the stacked buffer; the storage surface only exposes whole-Tensor
        # outputs, so we forward into `_mb_q_i` then copy the row.
        for i in range(Self.N):
            call_forward[target, Self.BATCH, POLICY=POLICY](
                ensemble.pairs[i].target_net,
                TensorRefs[Self.CRITIC.ARITY](self._mb_sa),
                self._mb_q_i,
                ctx,
            )
            comptime if target == "cpu":
                var base = i * Self.BATCH
                for b in range(Self.BATCH):
                    self._mb_stacked_q.data[base + b] = self._mb_q_i.data[b]
            else:
                var c = ctx.value()
                comptime nb = (Self.BATCH + TPB - 1) // TPB
                c.enqueue_function[
                    _copy_row_kernel[Self.N * Self.BATCH, Self.BATCH]
                ](
                    self._mb_q_i.tt["gpu", Self.BATCH](),
                    self._mb_stacked_q.tt["gpu", Self.N * Self.BATCH](),
                    Int64(i * Self.BATCH),
                    grid_dim=nb, block_dim=TPB,
                )

        # 5. Combine + α·lp + γ + terminal mask → state.mb_y.
        comptime if target == "cpu":
            redq_ensemble_target_cpu[
                Self.N, Self.N_MIN, Self.MODE, Self.BATCH,
            ](
                state.mb_r,
                self._mb_stacked_q,
                state.mb_d,
                self._mb_lp,
                self.subset_idxs,
                self.gamma,
                alpha,
                state.mb_y,
            )
        else:
            var c = ctx.value()
            if self._device_resample:
                # CUDA-graph device path: resample subset on-device (no host
                # upload) and read alpha from the device buffer. Capture-safe.
                self.resample_subset_dev()
                redq_ensemble_target_gpu_dev[
                    Self.N, Self.N_MIN, Self.MODE, Self.BATCH,
                ](
                    c,
                    state.mb_y,
                    state.mb_r,
                    self._mb_stacked_q,
                    state.mb_d,
                    self._mb_lp,
                    self._subset_dev.tt["gpu", Self.N_MIN](),
                    self.gamma,
                    self._alpha_ptr.value().as_unsafe_any_origin(),
                )
            else:
                # Upload subset_idxs (host List[Int] → device uint32) per step.
                self._subset_dev.ensure_host(c, Self.N_MIN)
                var hb = self._subset_dev.hbuf.value()
                c.synchronize()
                for k in range(Self.N_MIN):
                    hb[k] = UInt32(self.subset_idxs[k])
                c.enqueue_copy(self._subset_dev.dev.value(), hb)
                redq_ensemble_target_gpu[
                    Self.N, Self.N_MIN, Self.MODE, Self.BATCH,
                ](
                    c,
                    state.mb_y,
                    state.mb_r,
                    self._mb_stacked_q,
                    state.mb_d,
                    self._mb_lp,
                    self._subset_dev.tt["gpu", Self.N_MIN](),
                    self.gamma,
                    alpha,
                )
