"""ActorStep — the policy's update through the frozen world model.

`PolicyTrainer.update`'s actor half: imagine H steps (`imagination.sm_imagine`),
loss = −J / H, `clip_grad_norm(1.0)` then `clip_grad_value(0.1)` over the
policy, Adam(lr, β = (0.9, 0.95)). Returns the rollout (the critic fits on it)
and the pre-clip gradient norm.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch, SMImagOut, sm_imagine
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_GAMMA: Float64 = 0.99
comptime SM_LR: Float64 = 2e-4
comptime SM_CLIP_NORM: Float64 = 1.0
comptime SM_CLIP_VALUE: Float64 = 0.1


struct ClipValueCPU(ParamVisitor):
    """torch's `clip_grad_value_`: every gradient element into [−c, c]."""
    var c: Scalar[DT]

    def __init__(out self, c: Scalar[DT]):
        self.c = c

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            grad.data[i] = min(max(grad.data[i], -self.c), self.c)


struct ActorStep[C: SMConfigT](Movable):
    var opt: Adam

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](SM_LR), beta1=Scalar[DT](0.9), beta2=Scalar[DT](0.95))

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^

    def step[target: StaticString, N: Int](
        mut self,
        mut wm: SMWorldModelOf[Self.C],
        mut pol: SMPolicyOf[Self.C],
        mut critic: SMCriticOf[Self.C],
        mut tb: SMTables[DT],
        mut ib: SMImagBatch[Self.C.ROBOT],
        obs_noise: Bool,
    ) raises -> Tuple[SMImagOut[Self.C.ROBOT], Float64]:
        comptime if target == "gpu":
            raise Error("SuperMimic ActorStep: the GPU path (imagination kernels) is not built yet")
        pol.zero_grad["cpu"](None)
        var o = sm_imagine[Self.C, N](wm, pol, critic, tb, ib, SM_GAMMA, True, obs_noise)
        var gn = clip_grad_norm["cpu"](pol, Scalar[DT](SM_CLIP_NORM), None)
        var cv = ClipValueCPU(Scalar[DT](SM_CLIP_VALUE))
        pol.for_each_param["cpu"](cv, None)
        self.opt.step["cpu"](pol, None)
        return (o^, Float64(gn))
