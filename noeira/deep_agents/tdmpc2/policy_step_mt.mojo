"""TD-MPC2 multi-task policy (actor) update block — storage (CPU + GPU) §14.3.

Clone of `policy_step.mojo` threading a per-row task embedding: the policy graph
gains a `task_emb` input (feeding both `zt = [z|task_emb]` and the detached-Q
`za = [z|action|task_emb]`). After `graph.vjp`, the `task_emb` input-slot grad is
scatter-added into the table (site 3 of the embedding grad flow). The policy
batch `B` here is `PB = (H+1)·B_wm`; the caller passes one task id per row.

Storage migration: all scratch is storage `Tensor`s; the policy + the random Q
pair are threaded as distinct externals (RSample is an internal graph `Node`,
mirroring the single-task `PolicyStep`); `z`/`task_ids` are `Tensor`s.

Action masking note: per-task action masking (zeroing unused action dims) is
applied at acting + replay-record time (env wrapper + `agent_mt.select_action`).
In-graph masking of the actor-loss sampled action is deferred-experimental (a
no-op at MAX_ACT=1) — same spirit as the QP-dropout caveats.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Zero
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch

from .nets_mt import TDMPC2PolicyMT, TDMPC2QNetMT
from .policy_graph_mt import TDMPC2PolicyGraphMT
from .policy_step import _scale_copy_k
from .running_scale import RunningScale
from .task_embedding import TaskEmbedding


struct PolicyStepMT[
    LATENT: Int,
    MAX_ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
](Movable & Deinitable):
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime GraphT = TDMPC2PolicyGraphMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX,
        Self.TASK_EMB, Self.QP,
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]

    var graph: Self.GraphT
    var scale: RunningScale
    # ── PER-TASK policy-loss scale (OFF by default — a DEVIATION) ────────
    # The reference normalizes the policy loss by ONE running scale across
    # every task (`tdmpc2/tdmpc2.py:34`, a single `RunningScale` even for
    # MT80). With `per_task_scale = False` this struct is bit-identical to
    # that, and to every result measured before 2026-08-13.
    #
    # Rationale for the option: on walker stand+walk+run the shared scale was
    # set by the two solved tasks (Q ~98) while run sat at ~16, and run
    # collapsed to the standing floor with a MATCHED gradient budget — see
    # `docs/TDMPC2_MULTITASK_VALIDATION.md`. `RunningScale` normalizes by Q
    # SPREAD, so per-task scaling makes each task's policy gradient invariant
    # to its own spread rather than to the mixed-batch spread.
    var per_task_scale: Bool
    var task_scales: List[RunningScale]
    var max_reweight: Scalar[DT]
    """Clamp on the per-row weight `shared_spread / task_spread`. A task whose
    Q spread collapses toward 0 would otherwise divide by ~0 and swamp the
    batch. Applied symmetrically ([1/max, max])."""
    var entropy_coef: Scalar[DT]
    var q_mean: Scalar[DT]
    var q_min: Scalar[DT]
    var q_max: Scalar[DT]
    # scratch Tensors (allocated once in make, reused every step).
    var tem: Tensor    # [B*TASK_EMB] gathered embeddings (graph input)
    var loss: Tensor   # [B] graph output (loss_per_b)
    var qavg: Tensor   # [B] host-side avg-Q for RunningScale + stats
    var grad: Tensor   # [B] backward seed (1/B)
    var qsub: Tensor   # [B] host scratch: one task's rows, for its percentile

    def __init__(out self):
        self.graph = Self.GraphT()
        self.scale = RunningScale()
        self.per_task_scale = False
        self.task_scales = List[RunningScale]()
        for _ in range(Self.NUM_TASKS):
            self.task_scales.append(RunningScale())
        self.max_reweight = Scalar[DT](10.0)
        self.entropy_coef = Scalar[DT](1e-4)
        self.q_mean = Scalar[DT](0.0)
        self.q_min = Scalar[DT](0.0)
        self.q_max = Scalar[DT](0.0)
        self.tem = Tensor()
        self.loss = Tensor()
        self.qavg = Tensor()
        self.grad = Tensor()
        self.qsub = Tensor()

    def task_scale(self, t: Int) -> Scalar[DT]:
        """Per-task running scale, for logging. Meaningless unless
        `per_task_scale` — the table is not updated when the flag is off."""
        return self.task_scales[t].value

    def _set_q_stats(mut self, n: Int):
        if n <= 0:
            return
        var s: Scalar[DT] = 0.0
        var mn = self.qavg.data[0]
        var mx = self.qavg.data[0]
        for i in range(n):
            var v = self.qavg.data[i]
            s += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        self.q_mean = s / Scalar[DT](n)
        self.q_min = mn
        self.q_max = mx

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PolicyStepMT: target must be 'cpu' or 'gpu'"
        )
        comptime BB = Self.B
        var s = Self()
        s.graph = Self.GraphT.make[target, INIT=Zero](ctx=ctx)
        s.tem = Tensor.make[target](BB * Self.TASK_EMB, ctx)
        s.loss = Tensor.make[target](BB, ctx)
        s.grad = Tensor.make[target](BB, ctx)
        # qavg is host-resident (RunningScale + stats are host reductions).
        s.qavg = Tensor.alloc(BB)
        s.qsub = Tensor.alloc(BB)
        return s^

    def _bind(mut self) raises:
        self.graph.set_node_attr["alpha_lp", "multiplier"](
            self.entropy_coef * Scalar[DT](Self.MAX_ACT)
        )
        self.graph.set_node_attr["qscaled", "multiplier"](
            Scalar[DT](0.5) * self.scale.inv()
        )

    def step[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q_a: Self.QNetT,
        mut q_b: Self.QNetT,
        mut pi_opt: Adam,
        mut task_emb: Self.EmbT,
        mut z: Tensor,         # [B, LATENT] (host or device matching target)
        mut task_ids: Tensor,  # [B] per-row DT ids
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime BB = Self.B
        self._bind()
        pi_opt.zero_grad[target, M=Self.PolicyT](policy, ctx)

        # gather per-row task embeddings into the graph slot.
        task_emb.gather[target, BB](task_ids, self.tem, ctx)

        # seed inputs + forward (policy, q_a, q_b threaded in node order; the
        # RSample `alp` node is graph-internal so its noise cache persists to
        # vjp — mirrors the single-task PolicyStep wiring).
        self.graph.set_input["z", BB](z, ctx)
        self.graph.set_input["task_emb", BB](self.tem, ctx)
        self.graph.forward[BB, target](self.loss, ctx, policy, q_a, q_b)

        # avg-Q = 0.5·qsum; host reduction (CPU reads pool, GPU D2H qsum→qavg).
        comptime if target == "cpu":
            ref qsum = self.graph.node_output["qsum"]()
            for b in range(BB):
                self.qavg.data[b] = qsum.data[b] * Scalar[DT](0.5)
        else:
            var c = ctx.value()
            comptime nb = (BB + TPB - 1) // TPB
            ref qsum = self.graph.node_output["qsum"]()
            c.enqueue_function[_scale_copy_k[BB]](
                qsum.tt["gpu", BB](),
                self.grad.tt["gpu", BB](),
                Scalar[DT](0.5),
                grid_dim=nb, block_dim=TPB,
            )
            self.grad.download(c)
            for b in range(BB):
                self.qavg.data[b] = self.grad.data[b]

        # loss mean (CPU reads pool; GPU D2H the loss vector).
        var loss_mean: Scalar[DT]
        comptime if target == "cpu":
            var loss_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += self.loss.data[b]
            loss_mean = loss_sum / Scalar[DT](BB)
        else:
            self.loss.download(ctx.value())
            var loss_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += self.loss.data[b]
            loss_mean = loss_sum / Scalar[DT](BB)

        self.scale.update_from(self.qavg, BB)
        self._set_q_stats(BB)

        # ── per-task scales (only when enabled) ─────────────────────────
        # Each task's spread is estimated from ITS OWN rows in this batch.
        # A task with <2 rows this step is skipped rather than updated from a
        # degenerate sample; its previous value carries.
        if self.per_task_scale:
            for t in range(Self.NUM_TASKS):
                var n = 0
                for b in range(BB):
                    if Int(task_ids.data[b]) == t:
                        self.qsub.data[n] = self.qavg.data[b]
                        n += 1
                if n > 1:
                    self.task_scales[t].update_from(self.qsub, n)

        # backward seed + policy step. Grad flows through the Q heads (param
        # grads discarded) into the policy (stepped).
        if self.per_task_scale:
            # Per-row reweight `shared_spread / task_spread`, which converts
            # the graph's Q multiplier (bound from the SHARED scale) into a
            # per-task one.
            #
            # ⚠ APPROXIMATION: this scales the row's WHOLE loss, so the
            # entropy term is reweighted too, where the exact formulation
            # would rescale only Q. With `entropy_coef = 1e-4` that term is
            # ~0.1-1% of the row loss, so the error is second-order. Doing it
            # exactly needs a per-row multiplier INSIDE the graph — a new
            # binary-multiply primitive plus a second graph shape, and Mojo
            # cannot conditionally alias between two graph types
            # (`feedback_mojo_conditional_type_alias_blocked`), so the flag
            # could not gate it.
            var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)
            var lo = Scalar[DT](1.0) / self.max_reweight
            for b in range(BB):
                var t = Int(task_ids.data[b])
                var ts = self.task_scales[t].value
                var w = Scalar[DT](1.0)
                if ts > Scalar[DT](1e-8):
                    w = self.scale.value / ts
                if w > self.max_reweight:
                    w = self.max_reweight
                if w < lo:
                    w = lo
                self.grad.data[b] = inv_b * w
            comptime if target == "gpu":
                self.grad.upload_resident(ctx.value())
        else:
            seed_grad_inv_batch[target, BB](
                self.grad.tt[target, BB, 1](), ctx=ctx
            )
        self.graph.vjp[BB, target](self.grad, ctx, policy, q_a, q_b)

        # site 3: actor-loss grad w.r.t. the task embedding → table.
        task_emb.accumulate[target, BB, Self.TASK_EMB, 0](
            task_ids, self.graph.grad_input["task_emb"](), ctx
        )

        pi_opt.step[target, M=Self.PolicyT](policy, ctx)
        return loss_mean
