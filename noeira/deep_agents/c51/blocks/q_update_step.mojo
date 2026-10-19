"""C51QUpdateStep — wraps C51QUpdateBlock (STORAGE).

Reads `state.mb_s / mb_a` + the caller-supplied `mb_m` (target distribution
Tensor, owned by the trainer) → writes `state.critic_loss`. The distributional
analogue of `dqn/blocks/q_update_step.mojo`.

STORAGE migration (Stage 5): passes the storage `TrainerState` minibatch
`Tensor`s + the trainer's `_mb_m` straight into the migrated `C51QUpdateBlock`;
PER IS-weights / per-sample-CE capture are `state.mb_w` / `state.td_residuals`
`.lt` views. CPU + GPU.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from ..q_update_block import C51QUpdateBlock
from ...training.trainer_block import TrainerState
from noeira.core.tiles import Tile


struct C51QUpdateStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, NA_: Int, N_ATOMS_: Int, Q_NET: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime NA = Self.NA_
    comptime N_ATOMS = Self.N_ATOMS_
    comptime Inner = C51QUpdateBlock[
        Self.Q_NET, Self.BATCH, Self.OBS, Self.NA, Self.N_ATOMS,
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
        mut mb_m: Tensor,
    ) raises:
        """PER hook: when `state.has_per`, forward IS weights into the update +
        capture per-sample CE residuals.

        `ACCUMULATE` (GPU only) forwards to the inner block's device loss
        accumulator (CUDA-graph capture); `state.critic_loss` then holds a 0
        sentinel — read the running loss via `inner.ce_loss.read_accum`."""
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
            mb_m,
            weights=weights,
            td_residuals=td_residuals,
            ctx=state.ctx,
        )
        state.critic_loss = loss
