"""EnsembleActorStepOFE — REDQ-OFE actor loss + gradient step (STORAGE).

Port of `redq.EnsembleActorLoss.forward_backward` with the OFE data path
inserted between the concat step and the N critic forwards on the forward side,
and between the critic backwards and the rsample backward on the backward side.

Forward chain (φ-aware delta tagged ▌):
  0.   actor.zero_grad
  1.   actor.forward(φ(s)) → ao [B, 2·ACT]                  ▌ φ in
  2.   rsample.forward(ao) → alp [B, ACT+1] = (a | logπ)
  3.   sa_in = concat(φ(s), a), extract lp[b] = alp[b, ACT] ▌ wider
  4. ▌ action_branch.forward(sa_in) → φ(s, a) [B, PHI_SA]    ▌ NEW
  5.   for i in 0..N: q_i = critic_i(φ(s, a)); q_sum += q_i
  6.   loss = mean_b(α · lp[b] − q_sum[b] / N); lp_mean = mean_b(lp[b])

Backward chain:
  7.   for i in 0..N:
         grad_q_i[b] = −1 / (N · B)
         critic_i.forward(φ(s,a))  (refresh cache); critic_i.vjp → grad_φ(s,a)_i
         grad_φ(s, a)_sum += grad_φ(s, a)_i
  8. ▌ action_branch.vjp(grad_φ(s, a)_sum) → grad_sa_in [B, PHI_S+ACT]  ▌ NEW
  9.   grad_alp[b, :ACT] = grad_sa_in[b, PHI_S:]; grad_alp[b, ACT] = α / B
  10.  rsample.vjp(grad_alp) → grad_ao
  11.  actor.vjp(grad_ao) → grad_φ(s) (discarded); accumulates actor params
  12.  actor_opt.step(actor)

Gradient policy
===============
  - actor: trained on this path (zero_grad → vjp → opt.step).
  - action_branch + critics: their PARAM grads ARE written by `.vjp` (storage
    Module.vjp has no `mode`), but they are DISCARDED — the action_branch grads
    get zeroed by the aux step before `ab_opt.step`, and the critic grads get
    zeroed by the next `EnsembleCriticStepOFE` before its own update. Only the
    actor is stepped here. (Same contract as `redq.EnsembleActorLoss`.)
  - state_branch: NEVER touched here — grad_φ(s) is discarded from both the
    concat split and from actor.vjp; φ(s) is consumed read-only.

STORAGE migration (Stage 5): scratch are owned `nn.storage.Tensor`s; all
forwards/vjps use the storage Module surface over `TensorRefs`; the small
elementwise helper kernels reuse REDQ's actor-loss kernels.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.rsample import RSample

from ..redq.ensemble import CriticEnsemble
from ..redq.ensemble_actor_loss import (
    EnsembleActorLossResult,
    _eal_zero_kernel,
    _eal_add_into_kernel,
    _eal_fill_const_kernel,
    _eal_concat_sa_extract_lp_kernel,
    _eal_build_grad_alp_kernel,
)


struct EnsembleActorStepOFE[
    ACTOR: Module,   # IN=PHI_S_DIM, OUT=2·ACT
    AB: Module,      # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    CRITIC: Module,  # IN=PHI_SA_DIM, OUT=1
    N_: Int,
    BATCH_: Int,
    PHI_S_DIM_: Int,
    ACT_: Int,
](Movable & Deinitable):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime PHI_S_DIM = Self.PHI_S_DIM_
    comptime ACT = Self.ACT_
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    # Forward scratch.
    var _mb_ao: Tensor          # [BATCH, 2*ACT]
    var _mb_alp: Tensor         # [BATCH, ACT+1]
    var _mb_sa_in: Tensor       # [BATCH, SA_IN_DIM]
    var _mb_phi_sa: Tensor      # [BATCH, PHI_SA_DIM]
    var _mb_q_i: Tensor         # [BATCH]
    var _mb_q_sum: Tensor       # [BATCH]
    var _mb_lp: Tensor          # [BATCH] (GPU lp scratch)

    # Backward scratch.
    var _mb_grad_q_i: Tensor        # [BATCH]
    var _mb_grad_phi_sa_i: Tensor   # [BATCH, PHI_SA_DIM]
    var _mb_grad_phi_sa_sum: Tensor # [BATCH, PHI_SA_DIM]
    var _mb_grad_sa_in: Tensor      # [BATCH, SA_IN_DIM]
    var _mb_grad_alp: Tensor        # [BATCH, ACT+1]
    var _mb_grad_ao: Tensor         # [BATCH, 2*ACT]
    var _mb_grad_phi_s: Tensor      # [BATCH, PHI_S_DIM] (discarded)

    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Tensor()
        self._mb_alp = Tensor()
        self._mb_sa_in = Tensor()
        self._mb_phi_sa = Tensor()
        self._mb_q_i = Tensor()
        self._mb_q_sum = Tensor()
        self._mb_lp = Tensor()
        self._mb_grad_q_i = Tensor()
        self._mb_grad_phi_sa_i = Tensor()
        self._mb_grad_phi_sa_sum = Tensor()
        self._mb_grad_sa_in = Tensor()
        self._mb_grad_alp = Tensor()
        self._mb_grad_ao = Tensor()
        self._mb_grad_phi_s = Tensor()
        self.ctx = None

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleActorStepOFE: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleActorStepOFE.make[target='gpu']: ctx required"
                )
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.PHI_S_DIM, (
            "EnsembleActorStepOFE: ACTOR.IN must equal PHI_S_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleActorStepOFE: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "EnsembleActorStepOFE: AB.IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "EnsembleActorStepOFE: CRITIC.IN must equal PHI_SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleActorStepOFE: CRITIC.OUT_DIM must equal 1"
        )
        var b = Self()
        b.rsample = RSample[Self.ACT].make[target, Zero](ctx=ctx)
        b.rsample.action_scale = action_scale
        b.ctx = ctx
        comptime if target == "cpu":
            b._mb_ao = Tensor.alloc(Self.BATCH * (2 * Self.ACT))
            b._mb_alp = Tensor.alloc(Self.BATCH * Self.ALP_DIM)
            b._mb_sa_in = Tensor.alloc(Self.BATCH * Self.SA_IN_DIM)
            b._mb_phi_sa = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
            b._mb_q_i = Tensor.alloc(Self.BATCH)
            b._mb_q_sum = Tensor.alloc(Self.BATCH)
            b._mb_lp = Tensor.alloc(Self.BATCH)
            b._mb_grad_q_i = Tensor.alloc(Self.BATCH)
            b._mb_grad_phi_sa_i = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
            b._mb_grad_phi_sa_sum = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
            b._mb_grad_sa_in = Tensor.alloc(Self.BATCH * Self.SA_IN_DIM)
            b._mb_grad_alp = Tensor.alloc(Self.BATCH * Self.ALP_DIM)
            b._mb_grad_ao = Tensor.alloc(Self.BATCH * (2 * Self.ACT))
            b._mb_grad_phi_s = Tensor.alloc(Self.BATCH * Self.PHI_S_DIM)
        else:
            var c = ctx.value()
            b._mb_ao = Tensor.alloc_gpu(c, Self.BATCH * (2 * Self.ACT))
            b._mb_alp = Tensor.alloc_gpu(c, Self.BATCH * Self.ALP_DIM)
            b._mb_sa_in = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_IN_DIM)
            b._mb_phi_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_SA_DIM)
            b._mb_q_i = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_q_sum = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_lp = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_grad_q_i = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_grad_phi_sa_i = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_SA_DIM)
            b._mb_grad_phi_sa_sum = Tensor.alloc_gpu(
                c, Self.BATCH * Self.PHI_SA_DIM
            )
            b._mb_grad_sa_in = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_IN_DIM)
            b._mb_grad_alp = Tensor.alloc_gpu(c, Self.BATCH * Self.ALP_DIM)
            b._mb_grad_ao = Tensor.alloc_gpu(c, Self.BATCH * (2 * Self.ACT))
            b._mb_grad_phi_s = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_S_DIM)
        return b^

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut action_branch: Self.AB,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mut phi_s: Tensor,
        alpha: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> EnsembleActorLossResult:
        """One actor gradient step over φ(s). Returns (loss, log_prob_mean).
        The N critics' + action_branch's param grads are discarded."""
        comptime BB = Self.BATCH
        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
        var grad_q_val: Scalar[DT] = -inv_n * inv_b
        var grad_lp_val: Scalar[DT] = alpha * inv_b

        # ── Step 0 — zero actor grad slab.
        actor.zero_grad[target](ctx)

        # ── Step 1 — actor.forward(φ(s)) → _mb_ao [B, 2·ACT].
        call_forward[target, BB, POLICY=POLICY](
            actor, TensorRefs[Self.ACTOR.ARITY](phi_s), self._mb_ao, ctx
        )

        # ── Step 2 — rsample.forward(ao) → _mb_alp [B, ACT+1].
        call_forward[target, BB, POLICY=POLICY](
            self.rsample, TensorRefs[1](self._mb_ao), self._mb_alp, ctx
        )

        # ── Step 3 — sa_in = concat(φ(s), a) + extract lp[b] = alp[b, ACT].
        comptime if target == "cpu":
            for b in range(BB):
                for d in range(Self.PHI_S_DIM):
                    self._mb_sa_in.data[b * Self.SA_IN_DIM + d] = (
                        phi_s.data[b * Self.PHI_S_DIM + d]
                    )
                for j in range(Self.ACT):
                    self._mb_sa_in.data[b * Self.SA_IN_DIM + Self.PHI_S_DIM + j] = (
                        self._mb_alp.data[b * Self.ALP_DIM + j]
                    )
        else:
            var c = ctx.value()
            comptime total_sa = BB * Self.SA_IN_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _eal_concat_sa_extract_lp_kernel[
                Self.PHI_S_DIM, Self.ACT, BB, Self.SA_IN_DIM, Self.ALP_DIM,
            ]
            c.enqueue_function[kernel](
                phi_s.tt["gpu", BB, Self.PHI_S_DIM](),
                self._mb_alp.tt["gpu", BB, Self.ALP_DIM](),
                self._mb_sa_in.tt["gpu", BB, Self.SA_IN_DIM](),
                self._mb_lp.tt["gpu", BB](),
                grid_dim=n_blocks, block_dim=TPB,
            )

        # ── Step 4 — action_branch.forward(sa_in) → φ(s, a).
        call_forward[target, BB, POLICY=POLICY](
            action_branch,
            TensorRefs[Self.AB.ARITY](self._mb_sa_in), self._mb_phi_sa, ctx
        )

        # ── Step 5 — loop N online critic forwards; accumulate Σᵢ Qᵢ.
        comptime if target == "cpu":
            for b in range(BB):
                self._mb_q_sum.data[b] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            comptime nbb = (BB + TPB - 1) // TPB
            c.enqueue_function[_eal_zero_kernel[BB]](
                self._mb_q_sum.tt["gpu", BB](),
                grid_dim=nbb, block_dim=TPB,
            )
        for i in range(Self.N):
            call_forward[target, BB, POLICY=POLICY](
                ensemble.pairs[i].online,
                TensorRefs[Self.CRITIC.ARITY](self._mb_phi_sa), self._mb_q_i, ctx
            )
            comptime if target == "cpu":
                for b in range(BB):
                    self._mb_q_sum.data[b] += self._mb_q_i.data[b]
            else:
                var c = ctx.value()
                comptime nbb = (BB + TPB - 1) // TPB
                c.enqueue_function[_eal_add_into_kernel[BB]](
                    self._mb_q_sum.tt["gpu", BB](),
                    self._mb_q_i.tt["gpu", BB](),
                    grid_dim=nbb, block_dim=TPB,
                )

        # ── Step 6 — host-side scalar reduction: loss + log_prob_mean.
        var loss: Scalar[DT] = Scalar[DT](0.0)
        var lp_sum: Scalar[DT] = Scalar[DT](0.0)
        comptime if target == "cpu":
            for b in range(BB):
                var combined = self._mb_q_sum.data[b] * inv_n
                var lp = self._mb_alp.data[b * Self.ALP_DIM + Self.ACT]
                loss += alpha * lp - combined
                lp_sum += lp
        else:
            var c = ctx.value()
            self._mb_q_sum.download(c)
            self._mb_lp.download(c)
            for b in range(BB):
                var combined = self._mb_q_sum.data[b] * inv_n
                var lp = self._mb_lp.data[b]
                loss += alpha * lp - combined
                lp_sum += lp
        loss *= inv_b
        var log_prob_mean = lp_sum * inv_b

        # ── Step 7 — backward seed grad_qᵢ + zero grad_φ(s,a)_sum.
        comptime if target == "cpu":
            for b in range(BB):
                self._mb_grad_q_i.data[b] = grad_q_val
        else:
            var c = ctx.value()
            comptime nbb = (BB + TPB - 1) // TPB
            c.enqueue_function[_eal_fill_const_kernel[BB]](
                self._mb_grad_q_i.tt["gpu", BB](),
                grad_q_val,
                grid_dim=nbb, block_dim=TPB,
            )
        comptime PSA_TOTAL = BB * Self.PHI_SA_DIM
        comptime if target == "cpu":
            for k in range(PSA_TOTAL):
                self._mb_grad_phi_sa_sum.data[k] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            comptime n_blocks_z = (PSA_TOTAL + TPB - 1) // TPB
            c.enqueue_function[_eal_zero_kernel[PSA_TOTAL]](
                self._mb_grad_phi_sa_sum.tt["gpu", PSA_TOTAL](),
                grid_dim=n_blocks_z, block_dim=TPB,
            )
        for i in range(Self.N):
            # Re-forward the SAME φ(s,a) to refresh this critic's vjp cache,
            # then vjp. The critic's param grads accumulate but are DISCARDED.
            call_forward[target, BB, POLICY=POLICY](
                ensemble.pairs[i].online,
                TensorRefs[Self.CRITIC.ARITY](self._mb_phi_sa), self._mb_q_i, ctx
            )
            call_vjp[target, BB, POLICY=POLICY](
                ensemble.pairs[i].online,
                TensorRefs[Self.CRITIC.ARITY](self._mb_phi_sa),
                self._mb_grad_q_i,
                TensorRefs[Self.CRITIC.ARITY](self._mb_grad_phi_sa_i),
                ctx,
            )
            comptime if target == "cpu":
                for k in range(PSA_TOTAL):
                    self._mb_grad_phi_sa_sum.data[k] += (
                        self._mb_grad_phi_sa_i.data[k]
                    )
            else:
                var c = ctx.value()
                comptime n_blocks_a = (PSA_TOTAL + TPB - 1) // TPB
                c.enqueue_function[_eal_add_into_kernel[PSA_TOTAL]](
                    self._mb_grad_phi_sa_sum.tt["gpu", PSA_TOTAL](),
                    self._mb_grad_phi_sa_i.tt["gpu", PSA_TOTAL](),
                    grid_dim=n_blocks_a, block_dim=TPB,
                )

        # ── Step 8 — action_branch.vjp → grad_sa_in (param grads discarded).
        call_vjp[target, BB, POLICY=POLICY](
            action_branch,
            TensorRefs[Self.AB.ARITY](self._mb_sa_in),
            self._mb_grad_phi_sa_sum,
            TensorRefs[Self.AB.ARITY](self._mb_grad_sa_in),
            ctx,
        )

        # ── Step 9 — assemble grad_alp [B, ACT+1].
        comptime if target == "cpu":
            for b in range(BB):
                for j in range(Self.ACT):
                    self._mb_grad_alp.data[b * Self.ALP_DIM + j] = (
                        self._mb_grad_sa_in.data[
                            b * Self.SA_IN_DIM + Self.PHI_S_DIM + j
                        ]
                    )
                self._mb_grad_alp.data[b * Self.ALP_DIM + Self.ACT] = grad_lp_val
        else:
            var c = ctx.value()
            comptime total_galp = BB * Self.ALP_DIM
            comptime n_blocks_g = (total_galp + TPB - 1) // TPB
            comptime build_galp = _eal_build_grad_alp_kernel[
                BB, Self.PHI_S_DIM, Self.ACT, Self.SA_IN_DIM, Self.ALP_DIM,
            ]
            c.enqueue_function[build_galp](
                self._mb_grad_sa_in.tt["gpu", BB, Self.SA_IN_DIM](),
                self._mb_grad_alp.tt["gpu", BB, Self.ALP_DIM](),
                grad_lp_val,
                grid_dim=n_blocks_g, block_dim=TPB,
            )

        # ── Step 10 — rsample.vjp(grad_alp) → grad_ao.
        call_vjp[target, BB, POLICY=POLICY](
            self.rsample,
            TensorRefs[1](self._mb_ao),
            self._mb_grad_alp,
            TensorRefs[1](self._mb_grad_ao),
            ctx,
        )

        # ── Step 11 — actor.vjp(grad_ao) → grad_φ(s) (discarded); accumulates
        # actor param grads.
        call_vjp[target, BB, POLICY=POLICY](
            actor,
            TensorRefs[Self.ACTOR.ARITY](phi_s),
            self._mb_grad_ao,
            TensorRefs[Self.ACTOR.ARITY](self._mb_grad_phi_s),
            ctx,
        )

        # ── Step 12 — actor_opt.step.
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)

        return EnsembleActorLossResult(loss=loss, log_prob_mean=log_prob_mean)
