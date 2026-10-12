"""PPOActorTrainStep — PPO actor gradient step (STORAGE).

Thin wrapper around `PPOActorLoss.forward_backward[target, POLICY]`,
which is dual-target. Reads (mb_obs, mb_act, mb_olp, mb_adv) from
`OnPolicyState` and passes them as storage `Tensor`s. Returns the mean
per-batch loss for logging.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.optimizer.adam import Adam
from ..actor_loss import PPOActorLoss
from ...training.onpolicy_state import OnPolicyState


struct PPOActorTrainStep[
    OBS_: Int,
    ACT_: Int,
    MINIBATCH_: Int,
    ACTOR: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime MINIBATCH = Self.MINIBATCH_
    comptime Inner = PPOActorLoss[Self.ACTOR, Self.MINIBATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOActorTrainStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.inner = Self.Inner.make[target](
            ctx=ctx, clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        return b^

    def set_clip_eps(mut self, value: Scalar[DT]):
        self.inner.set_clip_eps(value)

    def set_entropy_coef(mut self, value: Scalar[DT]):
        self.inner.set_entropy_coef(value)

    def step[
        target: StaticString,
        ROLLOUT_LEN: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
        DEVICE_LOSS: Bool = False,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Scalar[DT]:
        return self.inner.forward_backward[
            target, POLICY=POLICY, DEVICE_LOSS=DEVICE_LOSS
        ](
            actor,
            actor_opt,
            state.mb_obs,
            state.mb_act,
            state.mb_olp,
            state.mb_adv,
            max_grad_norm,
            state.ctx,
        )
