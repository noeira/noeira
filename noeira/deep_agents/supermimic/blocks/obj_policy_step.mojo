"""ObjActorStep and ObjCriticStep — the object interface's two updates.

`ActorStep` and `CriticFitStep` for the interface of
`docs/OBJECT_INTERACTION_PLAN.md` §4.4: the same clips, Adam settings and
TD(λ) fit, on the object encoder and the interface's own critic. The
optimiser holds the object encoder alone: the motion encoder, the decoder, the
tracker's dynamics and the object network carry the gradient and do not move
here (the object network has its own update, `ObjectModelUpdateStep`).
"""

from std.random import random_ui64

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks.actor_step import ClipValue, SM_CLIP_NORM, SM_CLIP_VALUE
from noeira.deep_agents.supermimic.blocks.critic_fit_step import SMCriticFit, sm_value_fit
from noeira.deep_agents.supermimic.config import SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.object_imag_lanes import SMObjImagDims, sm_oc_in
from noeira.deep_agents.supermimic.object_imagination import SMObjImagBatch, SMObjImagOut, sm_imagine_obj
from noeira.deep_agents.supermimic.object_policy import SMObjConfigT, SMObjCriticOf, SMObjEncOf, SMObjModelOf
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam


struct ObjActorStep[OC: SMObjConfigT](Movable):
    """On "gpu" the optimiser adopts the object encoder at the first step."""
    var opt: Adam
    var adopted: Bool

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.OC.DEF_LR), beta1=Scalar[DT](0.9), beta2=Scalar[DT](0.95))
        self.adopted = False

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted

    def step[target: StaticString, N: Int](
        mut self,
        mut wm: SMWorldModelOf[Self.OC.BASE],
        mut pol: SMPolicyOf[Self.OC.BASE],
        mut om: SMObjModelOf[Self.OC],
        mut enc: SMObjEncOf[Self.OC],
        mut critic: SMObjCriticOf[Self.OC],
        mut tb: SMTables[DT],
        mut ib: SMObjImagBatch[Self.OC.BASE.ROBOT],
        gain: Float64,
        obs_noise: Bool,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Tuple[SMObjImagOut[Self.OC.BASE.ROBOT], Float64]:
        """The rollout and the pre-clip gradient norm of the object encoder."""
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](enc, ctx)
                self.adopted = True
        enc.zero_grad[target](ctx)
        var seed = UInt32(random_ui64(1, 0xFFFFFFFF)) if obs_noise else UInt32(0)
        var o = sm_imagine_obj[Self.OC, N, target](wm, pol, om, enc, critic, tb, ib, Self.OC.BASE.DEF_GAMMA, gain, True, seed, ctx)
        var gn = self.opt.clip_grads[target](enc, Scalar[DT](SM_CLIP_NORM), ctx)
        var cv = ClipValue(Scalar[DT](SM_CLIP_VALUE))
        enc.for_each_param[target](cv, ctx)
        self.opt.step[target](enc, ctx)
        return (o^, Float64(gn))


struct ObjCriticStep[OC: SMObjConfigT](Movable):
    var opt: Adam
    var adopted: Bool

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.OC.DEF_LR))
        self.adopted = False

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted

    def step[target: StaticString, N: Int, H: Int](
        mut self,
        mut critic: SMObjCriticOf[Self.OC],
        mut o: SMObjImagOut[Self.OC.BASE.ROBOT],
        mut ib: SMObjImagBatch[Self.OC.BASE.ROBOT],
        gamma: Float64,
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        ctx: Optional[DeviceContext] = None,
    ) raises -> SMCriticFit:
        """The TD(λ) fit on the rollout's (H+1)·N states (`sm_value_fit`)."""
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](critic, ctx)
                self.adopted = True
        comptime R = Self.OC.BASE.ROBOT
        comptime CIN = SMObjImagDims[R].OC_IN
        var X = talloc[target]((H + 1) * N * CIN, ctx)
        sm_oc_in[target, R](
            ctx, (H + 1) * N, N, tp[target](o.base.states), tp[target](ib.base.goals), tp[target](ib.base.xi), tp[target](o.base.e_root),
            tp[target](o.objs), tp[target](ib.orefs), tp[target](ib.octx), tp[target](X),
        )
        return sm_value_fit[target, CIN, Self.OC.CRIT1, Self.OC.CRIT2, N, H](
            critic, self.opt, X, tp[target](o.base.rewards), tp[target](o.base.dhat), gamma, perms, values_out, targets_out, ctx
        )
