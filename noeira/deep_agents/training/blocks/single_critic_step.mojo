"""SingleCriticStep — single-critic update (sa concat) for DDPG.

Wraps `CriticUpdateBlock[CRITIC, BATCH, SA_DIM]`. Builds sa = concat(s, a)
internally (the critic forward takes the pre-concatenated [B, OBS+ACT]). Owns
the sa scratch since the SA shape is block-specific (TrainerState carries only
OBS/ACT).

STORAGE migration (Stage 5): the single-critic sibling of TwinCriticStep —
storage `Tensor` concat scratch, no TargetStorage/Scratch/TileTensor; passes the
storage TrainerState minibatch Tensors straight into the migrated
`CriticUpdateBlock`. CPU + GPU.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from ...loss.critic_update_block import CriticUpdateBlock
from ...training.off_policy_critic import concat_sa, concat_sa_gpu
from ..trainer_block import TrainerState
from noeira.core.tiles import Tile


struct SingleCriticStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA = Self.OBS + Self.ACT
    comptime Inner = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA]

    var inner: Self.Inner
    var _mb_sa: Tensor

    def __init__(out self):
        self.inner = Self.Inner()
        self._mb_sa = Tensor()

    @staticmethod
    def make[
        target: StaticString = "cpu"
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "SingleCriticStep: target must be 'cpu' or 'gpu'"
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        comptime if target == "cpu":
            b._mb_sa = Tensor.alloc(Self.BATCH * Self.SA)
        else:
            b._mb_sa = Tensor.alloc_gpu(ctx.value(), Self.BATCH * Self.SA)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut critic: Self.CRITIC,
        mut critic_opt: Adam,
    ) raises:
        comptime if target == "cpu":
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                state.mb_s.tt["cpu", Self.BATCH, Self.OBS](),
                state.mb_a.tt["cpu", Self.BATCH, Self.ACT](),
                self._mb_sa.tt["cpu", Self.BATCH, Self.SA](),
            )
        else:
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                state.ctx.value(),
                state.mb_s.tt["gpu", Self.BATCH, Self.OBS](),
                state.mb_a.tt["gpu", Self.BATCH, Self.ACT](),
                self._mb_sa.tt["gpu", Self.BATCH, Self.SA](),
            )

        var weights: Optional[
            Tile[DT, Self.BATCH]
        ] = None
        var td_residuals: Optional[
            Tile[DT, Self.BATCH]
        ] = None
        if state.has_per:
            weights = state.mb_w.tt[target, Self.BATCH]()
            td_residuals = state.td_residuals.tt[target, Self.BATCH]()

        var loss = self.inner.step[target, POLICY, ACCUMULATE](
            critic,
            critic_opt,
            self._mb_sa,
            state.mb_y,
            weights=weights,
            td_residuals=td_residuals,
            ctx=state.ctx,
        )
        state.critic_loss = loss
