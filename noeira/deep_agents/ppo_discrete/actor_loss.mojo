"""PPODiscreteActorLoss — categorical PPO actor loss as an imperative block.

The discrete sibling of `ppo/actor_loss.mojo`. The actor-side loss is a 2-node
chain (actor → PPODiscreteObjective), so it does NOT use a ComputeGraph (the
storage graph dispatch tops out at arity 3; PPODiscreteObjective is arity 4).
It drives the pieces directly:

    actor.forward(s) → actor_out [B, N_ACTIONS]   (logits)
    loss_per_b = PPODiscreteObjective(actor_out, a_idx, old_log_prob, adv)  [B, 1]
    seed = 1/BATCH ; objective.vjp → grad_actor_out (+ zero rollout grads)
    actor.vjp(grad_actor_out) → actor param grads ; (optional grad-norm clip)
    actor_opt.step

The four objective inputs are staged into an owned `TensorPack[4]` so they share
one origin (§B0) for the `TensorRefs[4]` the leaf consumes; grad_inputs land in a
second pool. The action slot is width 1 (a discrete index stored as a float),
not an ACT_DIM-wide continuous sample. `forward_backward` returns the mean
per-batch loss for logging.
"""

from layout import Layout
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.initializer import Zero
from .objective import PPODiscreteObjective
from ..loss.loss_block import LossBlock
from ..training.device_mean_accum import DeviceMeanAccum


struct PPODiscreteActorLoss[
    ACTOR: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime N_ACTIONS = Self.ACTOR.OUT_DIM

    var objective: PPODiscreteObjective[Self.N_ACTIONS]
    var _in: TensorPack[4]   # [actor_out (N) | a_idx (1) | olp (1) | adv (1)]
    var _gin: TensorPack[4]  # grad_inputs (only slot 0 = grad_actor_out used)
    var _loss_out: Tensor    # [B] loss_per_b
    var _grad_seed: Tensor   # [B] = 1/BATCH backward seed
    var _obs_grad: Tensor    # [B*OBS] unused grad sink for actor.vjp
    var _loss_mean_dev: DeviceMeanAccum
    """GPU: the per-minibatch mean loss, accumulated on the device by the
    `DEVICE_LOSS` path (read at flush cadence) instead of a per-step D2H."""

    def __init__(out self):
        self.objective = PPODiscreteObjective[Self.N_ACTIONS]()
        self._in = TensorPack[4]()
        self._gin = TensorPack[4]()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()
        self._obs_grad = Tensor()
        self._loss_mean_dev = DeviceMeanAccum()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPODiscreteActorLoss: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        blk.objective = PPODiscreteObjective[Self.N_ACTIONS].make[target, Zero](ctx)
        blk.objective.set_attr["clip_eps"](clip_eps)
        blk.objective.set_attr["entropy_coef"](entropy_coef)
        blk._loss_out = Tensor.make[target](Self.BATCH, ctx)
        blk._obs_grad = Tensor.make[target](Self.BATCH * Self.OBS_DIM, ctx)
        # Seed = 1/BATCH in every slot (host-fill → upload on GPU).
        blk._grad_seed = Tensor.alloc(Self.BATCH)
        for b in range(Self.BATCH):
            blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        comptime if target == "gpu":
            blk._grad_seed.upload(ctx.value())
            blk._loss_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        return blk^

    def set_clip_eps(mut self, value: Scalar[DT]):
        self.objective.set_attr["clip_eps"](value)

    def set_entropy_coef(mut self, value: Scalar[DT]):
        self.objective.set_attr["entropy_coef"](value)

    @staticmethod
    def _copy_into[
        target: StaticString
    ](
        mut dst: Tensor, mut src: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        """Copy `n` elements src → dst (host element-loop on CPU; device
        enqueue_copy on GPU). Stages an external input into the §B0 input pool."""
        comptime if target == "cpu":
            dst.ensure(n)
            for i in range(n):
                dst.data[i] = src.data[i]
        else:
            var c = ctx.value()
            dst.ensure_gpu(c, n)
            c.enqueue_copy(dst.dev.value(), src.dev.value())

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        DEVICE_LOSS: Bool = False,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut mb_s: Tensor,
        mut mb_a: Tensor,
        mut mb_olp: Tensor,
        mut mb_adv: Tensor,
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime BB = Self.BATCH

        actor_opt.zero_grad[target, M = Self.ACTOR](actor, ctx)

        # actor.forward(s) → _in[0] (the logit vector).
        call_forward[target, BB, POLICY=POLICY](
            actor, TensorRefs[Self.ACTOR.ARITY](mb_s), self._in[0], ctx
        )
        # Stage the three rollout-time inputs into the §B0 pool (action=width 1).
        Self._copy_into[target](self._in[1], mb_a, BB, ctx)
        Self._copy_into[target](self._in[2], mb_olp, BB, ctx)
        Self._copy_into[target](self._in[3], mb_adv, BB, ctx)

        # loss_per_b = PPODiscreteObjective(...).
        self.objective.forward[target, BB, POLICY=POLICY](
            TensorRefs[4](self._in[0], self._in[1], self._in[2], self._in[3]),
            self._loss_out,
            ctx,
        )

        # Mean loss for logging. DEVICE_LOSS (GPU, CUDA-graph capture): folded
        # into `_loss_mean_dev` on the device, no D2H, and 0 is returned — the
        # caller reads `read_device_loss` at flush cadence. Otherwise the host
        # reduction (D2H on GPU).
        var loss_mean: Scalar[DT] = 0.0
        comptime if DEVICE_LOSS:
            comptime assert target == "gpu", "DEVICE_LOSS is GPU-only"
            self._loss_mean_dev.accumulate_gpu_lt[BB](
                self._loss_out.lt["gpu", Layout.row_major(BB)]()
            )
        else:
            comptime if target == "gpu":
                self._loss_out.download(ctx.value())
            var loss_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += self._loss_out.data[b]
            loss_mean = loss_sum / Scalar[DT](BB)

        # Backward: seed 1/BATCH → grad_actor_out (+ zeroed rollout grads).
        self.objective.vjp[target, BB, POLICY=POLICY](
            TensorRefs[4](self._in[0], self._in[1], self._in[2], self._in[3]),
            self._grad_seed,
            TensorRefs[4](
                self._gin[0], self._gin[1], self._gin[2], self._gin[3]
            ),
            ctx,
        )
        # actor.vjp(grad_actor_out) → actor param grads (obs grad discarded).
        call_vjp[target, BB, POLICY=POLICY](
            actor,
            TensorRefs[Self.ACTOR.ARITY](mb_s),
            self._gin[0],
            TensorRefs[Self.ACTOR.ARITY](self._obs_grad),
            ctx,
        )

        if max_grad_norm > Scalar[DT](0.0):
            # The device clip (no D2H of the norm) under capture.
            comptime if DEVICE_LOSS:
                actor_opt.clip_grads_device[target, M = Self.ACTOR](
                    actor, max_grad_norm, ctx
                )
            else:
                _ = actor_opt.clip_grads[target, M = Self.ACTOR](
                    actor, max_grad_norm, ctx
                )
        actor_opt.step[target, M = Self.ACTOR](actor, ctx)
        return loss_mean

    def read_device_loss(mut self) raises -> Scalar[DT]:
        """Mean of the `DEVICE_LOSS` minibatch losses since the last reset
        (one D2H — flush cadence only)."""
        return self._loss_mean_dev.read["gpu"]()

    def reset_device_loss(mut self) raises:
        self._loss_mean_dev.reset["gpu"]()
