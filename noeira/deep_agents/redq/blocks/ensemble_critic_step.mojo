"""EnsembleCriticStep — N-critic gradient step against a shared target y (STORAGE).

Replaces SAC's `TwinCriticStep` for the REDQ ensemble: loops over the N online
critics in `CriticEnsemble[CRITIC, N]` and runs one storage `CriticUpdateBlock.
step` per critic against `state.mb_s` / `state.mb_a` / `state.mb_y`. Sums the
per-critic losses into `state.critic_loss` (matches SAC's `loss1 + loss2`).

ONE `CriticUpdateBlock` instance, reused per critic — the scratch are pure
intermediates (overwritten each `step` call, consumed by `opt.step` before the
next critic). The storage `CriticUpdateBlock` owns the concat(s,a) → forward →
MSE → vjp → opt.step pipeline, so this block carries no scratch of its own.

STORAGE migration (Stage 5): imports the storage `CriticUpdateBlock` (which
already builds (s,a) internally via the storage `mb_*` Tensors) — so the legacy
`_mb_sa` scratch + `concat_sa` here are gone.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs

from ...loss.critic_update_block import CriticUpdateBlock
from ...training.off_policy_critic import concat_sa, concat_sa_gpu
from ...training.trainer_block import TrainerState
from ..ensemble import CriticEnsemble


struct EnsembleCriticStep[
    CRITIC: Module,
    N: Int,
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA_DIM = Self.OBS + Self.ACT

    var member_step: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var _mb_sa: Tensor   # shared concat(s, a) scratch

    def __init__(out self):
        self.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM,
        ]()
        self._mb_sa = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleCriticStep: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        blk.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM,
        ].make[target](ctx=ctx)
        comptime if target == "cpu":
            blk._mb_sa = Tensor.alloc(Self.BATCH * Self.SA_DIM)
        else:
            blk._mb_sa = Tensor.alloc_gpu(ctx.value(), Self.BATCH * Self.SA_DIM)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
    ) raises:
        """One ensemble-wide critic gradient step. Reads `state.mb_s` /
        `state.mb_a` / `state.mb_y`, writes `state.critic_loss = Σᵢ loss_i`.
        Under GPU `ACCUMULATE` the per-critic loss folds into the shared
        `member_step.mse_loss` device accumulator (NO D2H — capture-safe) and
        `state.critic_loss` stays 0 (read at flush via `read_accum`)."""
        var ctx = state.ctx
        # Build the shared concat(s, a) once; pass it to each critic update.
        comptime if target == "cpu":
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                state.mb_s.tt["cpu", Self.BATCH, Self.OBS](),
                state.mb_a.tt["cpu", Self.BATCH, Self.ACT](),
                self._mb_sa.tt["cpu", Self.BATCH, Self.SA_DIM](),
            )
        else:
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                ctx.value(),
                state.mb_s.tt["gpu", Self.BATCH, Self.OBS](),
                state.mb_a.tt["gpu", Self.BATCH, Self.ACT](),
                self._mb_sa.tt["gpu", Self.BATCH, Self.SA_DIM](),
            )

        var loss_sum: Scalar[DT] = Scalar[DT](0.0)
        for i in range(Self.N):
            var loss = self.member_step.step[
                target, POLICY, ACCUMULATE
            ](
                ensemble.pairs[i].online,
                ensemble.opts[i],
                self._mb_sa,
                state.mb_y,
                ctx=ctx,
            )
            loss_sum += loss
        state.critic_loss = loss_sum
