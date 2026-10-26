"""EnsembleTargetYBlockOFE — REDQ-OFE target-y computation (STORAGE).

Mirrors `redq.EnsembleTargetYBlock` but reads φ(s') (precomputed by
`OFEFeatureStep`, passed by `mut Tensor` ref) instead of raw next-obs, and
runs the OFE `action_branch` forward to obtain φ(s', a') before the N
target-critic forwards.

Pipeline (per step) — differences from the non-OFE block flagged ▌:

  1.   actor.forward(φ(s')) → raw [B, 2·ACT]                ▌ φ in
  2.   rsample.forward(raw) → packed [B, ACT+1] = (a' | logπ)
  3.   sa_in = concat(φ(s'), a')                            ▌ wider
  4. ▌ action_branch.forward(sa_in) → φ(s', a')              ▌ NEW
  5.   for i in 0..N: target_net_i.forward(φ(s', a')) → row i of stacked Q
  6.   redq_ensemble_target → mb_y[b]

Forward-only — actor + action_branch + target critics are read but never
receive gradient here.

STORAGE migration (Stage 5): scratch are owned `nn.storage.Tensor`s; the
actor + RSample + action_branch + critics use the storage Module surface; the
combine reuses `redq/kernels.mojo`. The concat(φ(s'), a') + log-prob extract
reuses REDQ's `_redq_concat_sa_extract_lp_kernel` on GPU (its "OBS" param is
just the first-input width — passing PHI_S_DIM there is correct).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.random import random_float64

from noeira.nn.constants import DT, TPB
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.nn.core.call import call_forward
from noeira.nn.primitives.rsample import RSample

from ..training.trainer_block import TrainerState
from ..redq.ensemble import CriticEnsemble
from ..redq.ensemble_target_y_block import (
    _redq_concat_sa_extract_lp_kernel, _copy_row_kernel,
)
from ..redq.kernels import (
    redq_ensemble_target_cpu,
    redq_ensemble_target_gpu,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
)


struct EnsembleTargetYBlockOFE[
    ACTOR: Module,          # IN=PHI_S_DIM, OUT=2·ACT
    AB: Module,             # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    CRITIC: Module,         # IN=PHI_SA_DIM, OUT=1
    N_: Int,
    BATCH_: Int,
    OBS_: Int,
    PHI_S_DIM_: Int,
    ACT_: Int,
    N_MIN_: Int,
    MODE_: Int,
](Movable & Deinitable):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime OBS = Self.OBS_
    comptime PHI_S_DIM = Self.PHI_S_DIM_
    comptime ACT = Self.ACT_
    comptime N_MIN = Self.N_MIN_
    comptime MODE = Self.MODE_
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    var _mb_ao: Tensor          # [BATCH, 2*ACT]
    var _mb_alp: Tensor         # [BATCH, ACT+1]
    var _mb_sa_in: Tensor       # [BATCH, SA_IN_DIM]
    var _mb_phi_spap: Tensor    # [BATCH, PHI_SA_DIM]
    var _mb_stacked_q: Tensor   # [N, BATCH] flat
    var _mb_q_i: Tensor         # [BATCH] per-critic forward scratch
    var _mb_lp: Tensor          # [BATCH]

    var subset_idxs: List[Int]
    var _subset_dev: TensorImpl[DType.uint32]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Tensor()
        self._mb_alp = Tensor()
        self._mb_sa_in = Tensor()
        self._mb_phi_spap = Tensor()
        self._mb_stacked_q = Tensor()
        self._mb_q_i = Tensor()
        self._mb_lp = Tensor()
        self.subset_idxs = List[Int](length=Self.N_MIN, fill=0)
        for k in range(Self.N_MIN):
            self.subset_idxs[k] = k
        self._subset_dev = TensorImpl[DType.uint32]()
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
            "EnsembleTargetYBlockOFE: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleTargetYBlockOFE.make[target='gpu']: ctx required"
                )
        comptime assert (
            Self.MODE == REDQ_TARGET_MIN or Self.MODE == REDQ_TARGET_AVE
        ), "EnsembleTargetYBlockOFE: MODE must be MIN (0) or AVE (1)"
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.PHI_S_DIM, (
            "EnsembleTargetYBlockOFE: ACTOR.IN_DIM must equal PHI_S_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleTargetYBlockOFE: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "EnsembleTargetYBlockOFE: AB.IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "EnsembleTargetYBlockOFE: CRITIC.IN_DIM must equal PHI_SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleTargetYBlockOFE: CRITIC.OUT_DIM must equal 1"
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
            blk._mb_sa_in = Tensor.alloc(Self.BATCH * Self.SA_IN_DIM)
            blk._mb_phi_spap = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
            blk._mb_stacked_q = Tensor.alloc(Self.N * Self.BATCH)
            blk._mb_q_i = Tensor.alloc(Self.BATCH)
            blk._mb_lp = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            blk._mb_ao = Tensor.alloc_gpu(c, Self.BATCH * (2 * Self.ACT))
            blk._mb_alp = Tensor.alloc_gpu(c, Self.BATCH * Self.ALP_DIM)
            blk._mb_sa_in = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_IN_DIM)
            blk._mb_phi_spap = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_SA_DIM)
            blk._mb_stacked_q = Tensor.alloc_gpu(c, Self.N * Self.BATCH)
            blk._mb_q_i = Tensor.alloc_gpu(c, Self.BATCH)
            blk._mb_lp = Tensor.alloc_gpu(c, Self.BATCH)
            blk._subset_dev.ensure_gpu(c, Self.N_MIN)
        return blk^

    def set_subset_idxs(mut self, idxs: List[Int]) raises:
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
        mut action_branch: Self.AB,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mut phi_sp: Tensor,
        alpha: Scalar[DT],
    ) raises:
        """Write `state.mb_y[b] = r[b] + (1 − term[b]) · γ · (combined_Q[b]
        − α · log_prob[b])` in-place. Reads φ(s') (= `phi_sp`) + state.mb_r /
        state.mb_d (BATCH-width). Forward-only. The actor input is φ(s')
        (= `phi_sp`), NOT the raw next-obs `state.mb_sp`."""
        var ctx = state.ctx

        # 1. actor.forward(φ(s')) → _mb_ao [BATCH, 2·ACT].
        call_forward[target, Self.BATCH, POLICY=POLICY](
            actor, TensorRefs[Self.ACTOR.ARITY](phi_sp), self._mb_ao, ctx
        )

        # 2. rsample → _mb_alp [BATCH, ACT+1] (packed a' | log_prob).
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.rsample, TensorRefs[1](self._mb_ao), self._mb_alp, ctx
        )

        # 3. sa_in = concat(φ(s'), a') + extract log_prob.
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for d in range(Self.PHI_S_DIM):
                    self._mb_sa_in.data[b * Self.SA_IN_DIM + d] = (
                        phi_sp.data[b * Self.PHI_S_DIM + d]
                    )
                for j in range(Self.ACT):
                    self._mb_sa_in.data[b * Self.SA_IN_DIM + Self.PHI_S_DIM + j] = (
                        self._mb_alp.data[b * Self.ALP_DIM + j]
                    )
                self._mb_lp.data[b] = (
                    self._mb_alp.data[b * Self.ALP_DIM + Self.ACT]
                )
        else:
            var c = ctx.value()
            comptime total_sa = Self.BATCH * Self.SA_IN_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _redq_concat_sa_extract_lp_kernel[
                Self.PHI_S_DIM, Self.ACT, Self.BATCH,
                Self.SA_IN_DIM, Self.ALP_DIM,
            ]
            c.enqueue_function[kernel](
                phi_sp.tt["gpu", Self.BATCH, Self.PHI_S_DIM](),
                self._mb_alp.tt["gpu", Self.BATCH, Self.ALP_DIM](),
                self._mb_sa_in.tt["gpu", Self.BATCH, Self.SA_IN_DIM](),
                self._mb_lp.tt["gpu", Self.BATCH](),
                grid_dim=n_blocks, block_dim=TPB,
            )

        # 4. action_branch.forward(sa_in) → φ(s', a') [BATCH, PHI_SA_DIM].
        call_forward[target, Self.BATCH, POLICY=POLICY](
            action_branch,
            TensorRefs[Self.AB.ARITY](self._mb_sa_in), self._mb_phi_spap, ctx
        )

        # 5. Loop N target critic forwards on φ(s', a') → row i of stacked Q.
        for i in range(Self.N):
            call_forward[target, Self.BATCH, POLICY=POLICY](
                ensemble.pairs[i].target_net,
                TensorRefs[Self.CRITIC.ARITY](self._mb_phi_spap),
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

        # 6. Combine + α·lp + γ + terminal mask → state.mb_y.
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
