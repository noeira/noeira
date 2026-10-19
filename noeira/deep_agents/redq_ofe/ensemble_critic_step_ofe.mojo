"""EnsembleCriticStepOFE — N-critic gradient step against shared target y (STORAGE).

Mirrors `redq.blocks.EnsembleCriticStep`: loops over the N online critics in
`CriticEnsemble[CRITIC, N]` and runs one storage `CriticUpdateBlock.step` per
critic. The OFE delta is the input:

  - Non-OFE REDQ: critic takes `concat(mb_s, mb_a)` of dim OBS+ACT
  - OFE REDQ:     critic takes `φ(s, a) = action_branch(concat(φ(s), mb_a))`
                  of dim PHI_SA_DIM

So this block runs the action-branch forward ONCE on `concat(φ(s), mb_a)` to
produce `φ(s, a)`, then loops the N critic updates against that same `φ(s, a)`
(every critic sees the same φ(s,a) — matches the legacy REDQ-OFE data flow).

Gradient policy
===============
`CriticUpdateBlock.step` writes `grad_φ(s, a)` into its own scratch and stops
there — it doesn't propagate back into `action_branch` or `state_branch`.
That's correct: OFE params only train via the aux loss path. The action-branch
forward here populates its cache, but no `action_branch.vjp` ever follows it on
the RL path.

STORAGE migration (Stage 5): reuses the storage `CriticUpdateBlock` + the
width-agnostic `concat_sa(_gpu)` glue (its "OBS" param is the first-input width
— passing PHI_S_DIM there is correct). φ(s) comes in by `mut Tensor` ref.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward

from ..loss.critic_update_block import CriticUpdateBlock
from ..training.off_policy_critic import concat_sa, concat_sa_gpu
from ..training.trainer_block import TrainerState
from ..redq.ensemble import CriticEnsemble


struct EnsembleCriticStepOFE[
    AB: Module,            # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    CRITIC: Module,        # IN=PHI_SA_DIM, OUT=1
    N: Int,
    OBS_: Int,
    PHI_S_DIM_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime PHI_S_DIM = Self.PHI_S_DIM_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM

    var member_step: CriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.PHI_SA_DIM,
    ]
    var _mb_sa_in: Tensor   # concat(φ(s), a) scratch  [BATCH, SA_IN_DIM]
    var _mb_phi_sa: Tensor  # action_branch output     [BATCH, PHI_SA_DIM]

    def __init__(out self):
        self.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.PHI_SA_DIM,
        ]()
        self._mb_sa_in = Tensor()
        self._mb_phi_sa = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleCriticStepOFE: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "EnsembleCriticStepOFE: AB.IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "EnsembleCriticStepOFE: CRITIC.IN must equal AB.OUT"
        )
        var blk = Self()
        blk.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.PHI_SA_DIM,
        ].make[target](ctx=ctx)
        comptime if target == "cpu":
            blk._mb_sa_in = Tensor.alloc(Self.BATCH * Self.SA_IN_DIM)
            blk._mb_phi_sa = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
        else:
            var c = ctx.value()
            blk._mb_sa_in = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_IN_DIM)
            blk._mb_phi_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_SA_DIM)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut action_branch: Self.AB,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mut phi_s: Tensor,
    ) raises -> Scalar[DT]:
        """One ensemble-wide critic gradient step. Reads φ(s) (= `phi_s`),
        state.mb_a, and the shared target state.mb_y; returns Σᵢ loss_i."""
        var ctx = state.ctx

        # 1. concat(φ(s), mb_a) → _mb_sa_in [BATCH, PHI_S_DIM + ACT].
        comptime if target == "cpu":
            concat_sa[Self.PHI_S_DIM, Self.ACT, Self.BATCH](
                phi_s.tt["cpu", Self.BATCH, Self.PHI_S_DIM](),
                state.mb_a.tt["cpu", Self.BATCH, Self.ACT](),
                self._mb_sa_in.tt["cpu", Self.BATCH, Self.SA_IN_DIM](),
            )
        else:
            concat_sa_gpu[Self.PHI_S_DIM, Self.ACT, Self.BATCH](
                ctx.value(),
                phi_s.tt["gpu", Self.BATCH, Self.PHI_S_DIM](),
                state.mb_a.tt["gpu", Self.BATCH, Self.ACT](),
                self._mb_sa_in.tt["gpu", Self.BATCH, Self.SA_IN_DIM](),
            )

        # 2. action_branch.forward(sa_in) → φ(s, a) [BATCH, PHI_SA_DIM].
        call_forward[target, Self.BATCH, POLICY=POLICY](
            action_branch,
            TensorRefs[Self.AB.ARITY](self._mb_sa_in), self._mb_phi_sa, ctx
        )

        # 3. Loop N online critics, each gets the same φ(s, a) → mb_y.
        var loss_sum: Scalar[DT] = Scalar[DT](0.0)
        for i in range(Self.N):
            var loss = self.member_step.step[target, POLICY, ACCUMULATE=False](
                ensemble.pairs[i].online,
                ensemble.opts[i],
                self._mb_phi_sa,
                state.mb_y,
                ctx=ctx,
            )
            loss_sum += loss
        state.critic_loss = loss_sum
        return loss_sum
