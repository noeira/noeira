"""PPOCriticTrainStep — PPO critic gradient step (STORAGE).

Wraps `critic.forward + MSELoss.forward + MSELoss.vjp + critic.vjp +
critic_opt.step` over the minibatch (s, ret) slot of `OnPolicyState`.

STORAGE migration: critic is a storage `Module`, the loss is a storage
`MSELoss[1]` (make_cpu / make_gpu; vjp recomputes from logits+targets — no
cache). All inputs are passed as storage `Tensor`s; the obs-side grad lands in
`state.mb_gi` (unused downstream but owned by the state). Gradient clipping is
an explicit `clip_grads` call between vjp and step, only when max_grad_norm > 0.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.loss.mse_loss import MSELoss
from noeira.nn.optimizer.adam import Adam
from ...training.onpolicy_state import OnPolicyState


struct PPOCriticTrainStep[
    OBS_: Int,
    MINIBATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime MINIBATCH = Self.MINIBATCH_
    comptime Inner = MSELoss[1]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOCriticTrainStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        comptime if target == "cpu":
            b.inner = Self.Inner.make_cpu()
        else:
            b.inner = Self.Inner.make_gpu(ctx.value())
        return b^

    def step[
        target: StaticString,
        ACT: Int,
        ROLLOUT_LEN: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
        DEVICE_LOSS: Bool = False,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, ACT, ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        mut critic: Self.CRITIC,
        mut critic_opt: Adam,
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Scalar[DT]:
        comptime MB = Self.MINIBATCH
        critic_opt.zero_grad[target, M=Self.CRITIC](critic, state.ctx)
        call_forward[target, MB, POLICY=POLICY](
            critic, TensorRefs[Self.CRITIC.ARITY](state.mb_obs), state.mb_v, state.ctx
        )
        # DEVICE_LOSS (GPU, CUDA-graph capture): the loss folds into the MSE
        # block's device accumulator (`read_device_loss` at flush cadence) and 0
        # is returned; otherwise the per-step host read.
        var loss: Scalar[DT] = 0.0
        comptime if DEVICE_LOSS:
            comptime assert target == "gpu", "DEVICE_LOSS is GPU-only"
            self.inner.forward_accumulate[target, MB](
                state.mb_v, state.mb_ret, state.ctx
            )
        else:
            loss = self.inner.forward[target, MB](
                state.mb_v, state.mb_ret, state.ctx
            )
        self.inner.vjp[target, MB](
            state.mb_v, state.mb_ret, state.mb_gv, state.ctx
        )
        call_vjp[target, MB, POLICY=POLICY](
            critic,
            TensorRefs[Self.CRITIC.ARITY](state.mb_obs),
            state.mb_gv,
            TensorRefs[Self.CRITIC.ARITY](state.mb_gi),
            state.ctx,
        )
        if max_grad_norm > Scalar[DT](0.0):
            comptime if DEVICE_LOSS:
                critic_opt.clip_grads_device[target, M=Self.CRITIC](
                    critic, max_grad_norm, state.ctx
                )
            else:
                _ = critic_opt.clip_grads[target, M=Self.CRITIC](
                    critic, max_grad_norm, state.ctx
                )
        critic_opt.step[target, M=Self.CRITIC](critic, state.ctx)
        return loss

    def read_device_loss(
        mut self, ctx: Optional[DeviceContext]
    ) raises -> Scalar[DT]:
        """Mean of the `DEVICE_LOSS` minibatch losses since the last reset
        (one D2H — flush cadence only)."""
        return self.inner.read_accum["gpu"](ctx)

    def reset_device_loss(mut self) raises:
        self.inner.reset_accum["gpu"]()
