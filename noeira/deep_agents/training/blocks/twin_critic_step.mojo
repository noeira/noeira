"""TwinCriticStep — twin-critic gradient step (owns inner TwinCriticUpdateBlock).

Reads state.mb_s, state.mb_a, state.mb_y → writes state.critic_loss.
Reusable across SAC, TD3, MBPO.

STORAGE migration (Stage 5): passes the storage `TrainerState` minibatch Tensors
(mb_s/mb_a/mb_y) straight into the inner block (which builds views internally).
PER weight / TD-residual views are built via `state.mb_*.lt[target, layout]`.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.optimizer.adam import Adam
from ...loss.critic_update_block import TwinCriticUpdateBlock
from ..trainer_block import TrainerState
from noeira.core.tiles import Tile


struct TwinCriticStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = TwinCriticUpdateBlock[
        Self.CRITIC,
        Self.BATCH,
        Self.OBS,
        Self.ACT,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[
        target: StaticString = "cpu"
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut critic1: Self.CRITIC,
        mut critic1_opt: Adam,
        mut critic2: Self.CRITIC,
        mut critic2_opt: Adam,
    ) raises:
        # PER hook: when state.has_per is set, forward IS weights into the
        # update and capture per-sample signed TD residuals. When unset, both
        # stay None and the inner block falls back to the uniform path.
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
            critic1,
            critic1_opt,
            critic2,
            critic2_opt,
            state.mb_s,
            state.mb_a,
            state.mb_y,
            weights=weights,
            td_residuals=td_residuals,
            ctx=state.ctx,
        )
        # With ACCUMULATE (GPU) the per-batch loss is reduced on-device into the
        # critics' accumulators; `loss` is a 0 sentinel here, real metric at flush.
        state.critic_loss = loss
