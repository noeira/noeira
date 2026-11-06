"""PromptActorStep and PromptCriticStep — the prompt interface's two updates.

`ActorStep` and `CriticFitStep` for a downstream interface (Appendix B): the
same clips, Adam settings and TD(λ) fit, on the prompt encoder and its own
critic. The optimiser holds the encoder alone: the decoder and the world model
carry the gradient and do not move.
"""

from std.random import random_ui64

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks.actor_step import ClipValue, SM_CLIP_NORM, SM_CLIP_VALUE
from noeira.deep_agents.supermimic.blocks.critic_fit_step import SMCriticFit, sm_value_fit
from noeira.deep_agents.supermimic.config import SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagOut
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch, sm_imagine_prompt
from noeira.deep_agents.supermimic.prompt_lanes import sm_p_critic_in
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfigT, SMPromptCriticOf, SMPromptEncOf
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam


struct PromptActorStep[P: SMPromptConfigT](Movable):
    var opt: Adam
    var adopted: Bool

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.P.DEF_LR), beta1=Scalar[DT](0.9), beta2=Scalar[DT](0.95))
        self.adopted = False

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted

    def step[target: StaticString, N: Int](
        mut self,
        mut wm: SMWorldModelOf[Self.P.BASE],
        mut pol: SMPolicyOf[Self.P.BASE],
        mut enc: SMPromptEncOf[Self.P],
        mut critic: SMPromptCriticOf[Self.P],
        mut tb: SMTables[DT],
        mut ib: SMPromptBatch[Self.P.BASE.ROBOT, Self.P.CMD],
        w_align: Float64,
        obs_noise: Bool,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Tuple[SMImagOut[Self.P.BASE.ROBOT], Float64]:
        """The rollout and the pre-clip gradient norm."""
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](enc, ctx)
                self.adopted = True
        enc.zero_grad[target](ctx)
        var seed = UInt32(random_ui64(1, 0xFFFFFFFF)) if obs_noise else UInt32(0)
        var o = sm_imagine_prompt[Self.P, N, target](wm, pol, enc, critic, tb, ib, Self.P.BASE.DEF_GAMMA, w_align, True, seed, ctx)
        var gn = self.opt.clip_grads[target](enc, Scalar[DT](SM_CLIP_NORM), ctx)
        var cv = ClipValue(Scalar[DT](SM_CLIP_VALUE))
        enc.for_each_param[target](cv, ctx)
        self.opt.step[target](enc, ctx)
        return (o^, Float64(gn))


struct PromptCriticStep[P: SMPromptConfigT](Movable):
    var opt: Adam
    var adopted: Bool

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.P.DEF_LR))
        self.adopted = False

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted

    def step[target: StaticString, N: Int, H: Int](
        mut self,
        mut critic: SMPromptCriticOf[Self.P],
        mut o: SMImagOut[Self.P.BASE.ROBOT],
        mut ib: SMPromptBatch[Self.P.BASE.ROBOT, Self.P.CMD],
        gamma: Float64,
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        ctx: Optional[DeviceContext] = None,
    ) raises -> SMCriticFit:
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](critic, ctx)
                self.adopted = True
        comptime R = Self.P.BASE.ROBOT
        comptime D = SMDims[R]
        comptime CIN = D.S + Self.P.CMD + D.X
        var X = talloc[target]((H + 1) * N * CIN, ctx)
        sm_p_critic_in[target, R, Self.P.CMD](ctx, (H + 1) * N, N, tp[target](o.states), tp[target](ib.cmds), tp[target](ib.xi), tp[target](X))
        return sm_value_fit[target, CIN, Self.P.CRIT1, Self.P.CRIT2, N, H](
            critic, self.opt, X, tp[target](o.rewards), tp[target](o.dhat), gamma, perms, values_out, targets_out, ctx
        )
