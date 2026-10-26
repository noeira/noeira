"""DQNQUpdateStep — wraps DQNQUpdateBlock (owns the inner block) (STORAGE).

Reads `state.mb_s / mb_a / mb_y` → writes `state.critic_loss`. The single-Q-net
discrete analogue of `training/blocks/single_critic_step.mojo` (gather instead
of `sa` concat).

STORAGE migration (Stage 5): passes the storage `TrainerState` minibatch
`Tensor`s straight into the migrated `DQNQUpdateBlock`; PER IS-weights /
TD-residual capture are built as `state.mb_w` / `state.td_residuals` `.lt`
views (mirrors `SingleCriticStep`). CPU + GPU.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.optimizer.adam import Adam
from ..q_update_block import DQNQUpdateBlock
from ...training.trainer_block import TrainerState
from noeira.core.tiles import Tile


struct DQNQUpdateStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, NA_: Int, Q_NET: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_  # = 1 for DQN
    comptime BATCH = Self.BATCH_
    comptime NA = Self.NA_
    comptime Inner = DQNQUpdateBlock[
        Self.Q_NET, Self.BATCH, Self.OBS, Self.NA,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
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
        mut q_online: Self.Q_NET,
        mut q_opt: Adam,
    ) raises:
        """PER hook: when `state.has_per`, forward IS weights into the update +
        capture per-sample signed TD residuals so the sample block can refresh
        sum-tree priorities. When unset, both stay None and the inner block
        falls back to the uniform path (bit-identical to pre-PER).

        `ACCUMULATE` (GPU only) forwards to the inner block's device loss
        accumulator (CUDA-graph capture); `state.critic_loss` then holds a 0
        sentinel — read the running loss via `inner.mse_loss.read_accum`."""
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
            q_online,
            q_opt,
            state.mb_s,
            state.mb_a,
            state.mb_y,
            weights=weights,
            td_residuals=td_residuals,
            ctx=state.ctx,
        )
        state.critic_loss = loss
