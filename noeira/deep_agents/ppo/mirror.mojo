"""Mirror-symmetry loss for a PPO actor (rsl_rl's `use_mirror_loss`).

    loss_mirror = coeff * mean_{b, j} ( mu(M_o s_b)_j - [M_a mu(s_b)]_j )^2

with the target `M_a mu(s)` detached. `M_o` / `M_a` are index + sign maps:

    (M x)[k] = sign[k] * x[idx[k]]

over the actor's input (the WHOLE observation it is handed — a `Slice` at
the actor's head reads its part) and over the action. A left / right
symmetric robot under a symmetric policy has no reason to prefer one side;
the loss makes the actor so. RoboParty's walker (`rpo_interrupt_agent_cfg.
py`) uses it at 0.2, with data augmentation besides (not ported: it doubles
every minibatch shape).

USE (opt-in, default off — `PPOActorLoss` is unchanged when not enabled):

    agent.trainer.actor_train.inner.enable_mirror["gpu"](
        obs_idx, obs_sign, act_idx, act_sign, coeff=0.2, ctx=ctx)

⚠ ORDER OF THE PASSES. `PPOActorLoss.forward_backward` zeroes the grads,
runs THIS pass (forward on s, forward on M_o s, vjp of the mirror loss —
param grads ACCUMULATE), then its own forward / vjp on s, the clip and the
step. The second forward of this pass leaves the actor's activation caches
on M_o s; the PPO forward rewrites them before its vjp reads them.

⚠ THE LOG-STD GETS NO MIRROR GRADIENT: the loss reads the mean half of the
actor output only (a state-independent log-std is symmetric or not on its
own; RoboParty's mirror loss is on `act_inference`, the mean, too).
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from ..training.device_mean_accum import DeviceMeanAccum

comptime _V[n: Int] = LayoutTensor[DT, Layout.row_major(n), MutAnyOrigin]


def _mirror_rows_k[B: Int, D: Int](
    src: _V[B * D], idx: _V[D], sign: _V[D], dst: _V[B * D],
):
    var t = Int(global_idx.x)
    if t >= B * D:
        return
    var b = t // D
    var k = t % D
    dst[t] = rebind[Scalar[DT]](sign[k]) * rebind[Scalar[DT]](
        src[b * D + Int(rebind[Scalar[DT]](idx[k]))]
    )


def _mirror_grad_k[B: Int, ACT: Int](
    ao: _V[B * 2 * ACT],
    ao_m: _V[B * 2 * ACT],
    aidx: _V[ACT],
    asign: _V[ACT],
    coeff: Scalar[DT],
    grad: _V[B * 2 * ACT],
    loss: _V[B],
):
    """One thread per row: the target `M_a mu(s)` from `ao`, the residual
    against `ao_m`'s mean, the gradient `2 coeff r / (B ACT)` on the mean
    half (0 on the log-std half), and the row's loss sum."""
    var b = Int(global_idx.x)
    if b >= B:
        return
    var inv = Scalar[DT](2.0) * coeff / Scalar[DT](B * ACT)
    var l: Scalar[DT] = 0.0
    for j in range(ACT):
        var tgt = rebind[Scalar[DT]](asign[j]) * rebind[Scalar[DT]](
            ao[b * 2 * ACT + Int(rebind[Scalar[DT]](aidx[j]))]
        )
        var r = rebind[Scalar[DT]](ao_m[b * 2 * ACT + j]) - tgt
        grad[b * 2 * ACT + j] = inv * r
        grad[b * 2 * ACT + ACT + j] = Scalar[DT](0.0)
        l += r * r
    loss[b] = coeff * l / Scalar[DT](ACT)


struct ActorMirror[OBS: Int, ACT: Int, BATCH: Int](Movable):
    """The maps and the minibatch scratch of the mirror pass."""

    var on: Bool
    var coeff: Scalar[DT]
    var obs_idx: Tensor
    var obs_sign: Tensor
    var act_idx: Tensor
    var act_sign: Tensor
    var s_m: Tensor
    var ao: Tensor
    var ao_m: Tensor
    var grad: Tensor
    var loss: Tensor
    var gi: Tensor
    var loss_mean_dev: DeviceMeanAccum

    def __init__(out self):
        self.on = False
        self.coeff = Scalar[DT](0.0)
        self.obs_idx = Tensor()
        self.obs_sign = Tensor()
        self.act_idx = Tensor()
        self.act_sign = Tensor()
        self.s_m = Tensor()
        self.ao = Tensor()
        self.ao_m = Tensor()
        self.grad = Tensor()
        self.loss = Tensor()
        self.gi = Tensor()
        self.loss_mean_dev = DeviceMeanAccum()

    def enable[target: StaticString](
        mut self,
        obs_idx: List[Int],
        obs_sign: List[Float64],
        act_idx: List[Int],
        act_sign: List[Float64],
        coeff: Float64,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Check the maps are involutions with unit signs, then allocate."""
        Self._check(obs_idx, obs_sign, Self.OBS, "obs")
        Self._check(act_idx, act_sign, Self.ACT, "action")
        self.obs_idx = Tensor.make[target](Self.OBS, ctx)
        self.obs_sign = Tensor.make[target](Self.OBS, ctx)
        self.act_idx = Tensor.make[target](Self.ACT, ctx)
        self.act_sign = Tensor.make[target](Self.ACT, ctx)
        self.obs_idx.ensure(Self.OBS)
        self.obs_sign.ensure(Self.OBS)
        self.act_idx.ensure(Self.ACT)
        self.act_sign.ensure(Self.ACT)
        for k in range(Self.OBS):
            self.obs_idx.data[k] = Scalar[DT](obs_idx[k])
            self.obs_sign.data[k] = Scalar[DT](obs_sign[k])
        for j in range(Self.ACT):
            self.act_idx.data[j] = Scalar[DT](act_idx[j])
            self.act_sign.data[j] = Scalar[DT](act_sign[j])
        comptime if target == "gpu":
            var c = ctx.value()
            self.obs_idx.upload_resident(c)
            self.obs_sign.upload_resident(c)
            self.act_idx.upload_resident(c)
            self.act_sign.upload_resident(c)
            self.loss_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        self.s_m = Tensor.make[target](Self.BATCH * Self.OBS, ctx)
        self.ao = Tensor.make[target](Self.BATCH * 2 * Self.ACT, ctx)
        self.ao_m = Tensor.make[target](Self.BATCH * 2 * Self.ACT, ctx)
        self.grad = Tensor.make[target](Self.BATCH * 2 * Self.ACT, ctx)
        self.loss = Tensor.make[target](Self.BATCH, ctx)
        self.gi = Tensor.make[target](Self.BATCH * Self.OBS, ctx)
        self.coeff = Scalar[DT](coeff)
        self.on = True

    @staticmethod
    def _check(idx: List[Int], sign: List[Float64], n: Int, what: String) raises:
        if len(idx) != n or len(sign) != n:
            raise Error("ActorMirror: the " + what + " map has " + String(len(idx))
                        + " entries, expected " + String(n))
        for k in range(n):
            if idx[k] < 0 or idx[k] >= n:
                raise Error("ActorMirror: " + what + " index out of range at " + String(k))
            if sign[k] != 1.0 and sign[k] != -1.0:
                raise Error("ActorMirror: " + what + " sign not +-1 at " + String(k))
            # an involution: mirroring twice is the identity
            if idx[idx[k]] != k or sign[idx[k]] * sign[k] != 1.0:
                raise Error("ActorMirror: the " + what + " map is not an involution at "
                            + String(k))

    def add_grads[target: StaticString, ACTOR: Module](
        mut self, mut actor: ACTOR, mut mb_s: Tensor, ctx: Optional[DeviceContext],
    ) raises:
        """The mirror loss's param gradients, ACCUMULATED into the actor's
        (the caller zeroed them first)."""
        comptime B = Self.BATCH
        comptime D = Self.OBS
        comptime A = Self.ACT
        # M_o s
        comptime if target == "gpu":
            var c = ctx.value()
            c.enqueue_function[_mirror_rows_k[B, D]](
                mb_s.lt["gpu", Layout.row_major(B * D)](),
                self.obs_idx.lt["gpu", Layout.row_major(D)](),
                self.obs_sign.lt["gpu", Layout.row_major(D)](),
                self.s_m.lt["gpu", Layout.row_major(B * D)](),
                grid_dim=(B * D + TPB - 1) // TPB, block_dim=TPB,
            )
        else:
            for b in range(B):
                for k in range(D):
                    self.s_m.data[b * D + k] = self.obs_sign.data[k] * mb_s.data[
                        b * D + Int(self.obs_idx.data[k])
                    ]
        call_forward[target, B](actor, TensorRefs[ACTOR.ARITY](mb_s), self.ao, ctx)
        call_forward[target, B](actor, TensorRefs[ACTOR.ARITY](self.s_m), self.ao_m, ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            c.enqueue_function[_mirror_grad_k[B, A]](
                self.ao.lt["gpu", Layout.row_major(B * 2 * A)](),
                self.ao_m.lt["gpu", Layout.row_major(B * 2 * A)](),
                self.act_idx.lt["gpu", Layout.row_major(A)](),
                self.act_sign.lt["gpu", Layout.row_major(A)](),
                self.coeff,
                self.grad.lt["gpu", Layout.row_major(B * 2 * A)](),
                self.loss.lt["gpu", Layout.row_major(B)](),
                grid_dim=(B + TPB - 1) // TPB, block_dim=TPB,
            )
            self.loss_mean_dev.accumulate_gpu_lt[B](
                self.loss.lt["gpu", Layout.row_major(B)]()
            )
        else:
            var inv = Scalar[DT](2.0) * self.coeff / Scalar[DT](B * A)
            for b in range(B):
                var l: Scalar[DT] = 0.0
                for j in range(A):
                    var tgt = self.act_sign.data[j] * self.ao.data[
                        b * 2 * A + Int(self.act_idx.data[j])
                    ]
                    var r = self.ao_m.data[b * 2 * A + j] - tgt
                    self.grad.data[b * 2 * A + j] = inv * r
                    self.grad.data[b * 2 * A + A + j] = Scalar[DT](0.0)
                    l += r * r
                self.loss.data[b] = self.coeff * l / Scalar[DT](A)
        call_vjp[target, B](
            actor,
            TensorRefs[ACTOR.ARITY](self.s_m),
            self.grad,
            TensorRefs[ACTOR.ARITY](self.gi),
            ctx,
        )

    def host_loss(mut self) -> Float64:
        """CPU: the last minibatch's mirror loss (mean over rows)."""
        var s = 0.0
        for b in range(Self.BATCH):
            s += Float64(self.loss.data[b])
        return s / Float64(Self.BATCH)
