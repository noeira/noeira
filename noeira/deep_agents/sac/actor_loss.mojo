"""SACActorLoss — SAC actor loss on the storage ComputeGraph (named DX).

  loss_per_b[b] = α·log π(a|s) − min(Q1(s,a), Q2(s,a)),   a ~ π(·|s)
  loss          = mean_b(loss_per_b);   ∂loss/∂loss_per_b = 1/BATCH

STORAGE migration (Stage 5): the graph is declared with the legacy NAME-wired DX
(`InputSlot`/`Node`/`ExternalNode`, edges by predecessor name — no runtime edge
list). The online actor + the two ONLINE critics are `ExternalNode`s threaded as
tracked `mut` refs into `graph.forward`/`vjp` (the actor accumulates param grads
and is stepped; the critics' grads are computed-then-discarded — the critic
block zero_grads before its own update). The `Scale` node's runtime `multiplier`
carries the moving α (host scalar). Mean loss + mean log_prob are host reductions
(per-step D2H on GPU; cheap at SAC scales).

  graph: s → actor → rsample → {action, logp} ; (s, action) → concat →
         q1, q2 → min_q ;  α·logp = Scale(logp) ;  loss_q = α·logp − min_q
         + a BEHAVIOUR-CLONING term on the demo half (HIL-SERL / RLPD runs):
         mu_t = tanh(mu) ; bc = L1(mu_t, a_batch) masked to the demo rows ;
         loss = loss_q + λ·bc                                      (output)

## THE BC TERM (`set_bc`)

RLPD's symmetric sampling alone did not pull the actor onto the expert's
manifold here: `so101_tower_lift_brick` with 7677 expert rows pinned as half
of every batch plateaued at the parked-on-the-brick return (eval 356 at 25k
and 355 at 50k, 2026-09-20) — the critic learned the demos' value but the
actor never entered the pinch states from where its own rollouts sit. The
TD3+BC / DAPG remedy is a per-row behaviour-cloning penalty on the DEMO rows
only: `λ · (1/ACT) Σ_j |tanh(mu_j(s)) − a_j|` for rows `[0, n_demo)` of the
batch and 0 elsewhere. The demo rows are the first `BATCH/2` rows by the
replay's construction (`_mixed_indices_dev_kernel`), so the mask is a
constant `[B, 1]` tensor set once. `λ = 0` (the default) leaves the graph's
answer bit-identical to the plain SAC loss: the extra nodes multiply by 0.
Scale: with Q ~ 100 and the L1 in [0, 2], TD3+BC's `α = 2.5` normalisation
puts λ around `mean|Q| / 2.5` ≈ 40.
"""

from max.gpu import thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB_REDUCE
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Zero
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.rsample import RSample
from noeira.nn.primitives.slice import Slice
from noeira.nn.primitives.concat import Concat2
from noeira.nn.primitives.scale import Scale
from noeira.nn.primitives.binary_elementwise import (
    BinaryElemMin, BinarySub, BinaryElementwise,
)
from noeira.nn.primitives.ops.binary_add_op import BinaryAddOp
from noeira.nn.primitives.activations import Tanh
from noeira.nn.primitives.l1_masked_per_sample import L1MaskedPerSample

comptime BinaryAdd[DIM: Int] = BinaryElementwise[DIM, BinaryAddOp]
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node, ExternalNode
from ..loss.loss_block import LossBlock


# ── Device reductions (single-block; launch grid=1, block=TPB_REDUCE) ──
# `reduce_mean_write_kernel` writes mean(src) -> dst[0,0] (overwrite); the
# device ScalarAdam reads it as the entropy grad this same step (no D2H).
# `reduce_mean_acc_kernel` accumulates (Sum-of-means, count) into acc[0,0]/
# acc[1,0]; the host drains it once per flush.
def reduce_mean_write_kernel[
    B: Int
](
    src: LayoutTensor[DT, Layout.row_major(B, 1), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(1, 1), MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < B:
        my_sum += rebind[Scalar[DT]](src[k, 0])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        dst[0, 0] = total[0] / Scalar[DT](B)


def reduce_mean_acc_kernel[
    B: Int
](
    src: LayoutTensor[DT, Layout.row_major(B, 1), MutAnyOrigin],
    acc: LayoutTensor[DT, Layout.row_major(1, 2), MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < B:
        my_sum += rebind[Scalar[DT]](src[k, 0])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0, 0] = rebind[Scalar[DT]](acc[0, 0]) + total[0] / Scalar[DT](B)
        acc[0, 1] = rebind[Scalar[DT]](acc[0, 1]) + Scalar[DT](1.0)


@fieldwise_init
struct SACActorLossOut(Movable & Deinitable):
    var loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]


struct SACActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime ALP_DIM = Self.ACT_DIM + 1
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    comptime Graph = ComputeGraph[
        InputSlot["s", Self.OBS_DIM],                        # s
        ExternalNode["actor", Self.ACTOR, "s"],              # → [mu|ls]
        Node["rsample", RSample[Self.ACT_DIM], "actor"],     # → [a|logp]
        Node["action", Slice[Self.ALP_DIM, 0, Self.ACT_DIM], "rsample"],
        Node["logp", Slice[Self.ALP_DIM, Self.ACT_DIM, Self.ALP_DIM], "rsample"],
        Node["concat", Concat2[Self.OBS_DIM, Self.ACT_DIM], "s", "action"],
        ExternalNode["q1", Self.CRITIC, "concat"],
        ExternalNode["q2", Self.CRITIC, "concat"],
        Node["min_q", BinaryElemMin[1], "q1", "q2"],
        Node["alogp", Scale[1], "logp"],                     # α·logp
        Node["loss_q", BinarySub[1], "alogp", "min_q"],      # α·logp − min_q
        # ── the BC term on the demo half — see `set_bc` ──
        InputSlot["a_demo", Self.ACT_DIM],                   # the batch's actions
        InputSlot["bc_mask", 1],                             # 1 on demo rows
        Node["mu", Slice[2 * Self.ACT_DIM, 0, Self.ACT_DIM], "actor"],
        Node["mu_t", Tanh[Self.ACT_DIM], "mu"],              # the greedy action
        Node["bc", L1MaskedPerSample[1, Self.ACT_DIM], "mu_t", "a_demo", "bc_mask"],
        Node["bc_w", Scale[1], "bc"],                        # λ·bc
        Node["loss_qw", Scale[1], "loss_q"],                 # w·(α·logp − min_q), see `set_q_weight`
        Node["loss", BinaryAdd[1], "loss_qw", "bc_w"],       # loss_per_b (output)
    ]

    var graph: Self.Graph
    var _loss_out: Tensor   # [B] loss_per_b (graph output)
    var _grad_seed: Tensor  # [B] = 1/BATCH (backward seed)
    # Device-α path (GPU only): `_lp_mean` [1] holds mean(log_prob), read by
    # the device ScalarAdam this same step; `_loss_acc` [2] = (Σmean, count)
    # actor-loss metric accumulator drained at flush cadence. Empty on CPU.
    var _lp_mean: Tensor
    var _loss_acc: Tensor
    var _bc_mask: Tensor    # [B] 1.0 on the demo rows, 0 elsewhere — `set_bc`
    var _bc_w_dev: Tensor   # [1] the BC weight the `bc_w` Scale reads ON-DEVICE
    var bc_weight: Scalar[DT]

    def __init__(out self):
        self.graph = Self.Graph()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()
        self._lp_mean = Tensor()
        self._loss_acc = Tensor()
        self._bc_mask = Tensor()
        self._bc_w_dev = Tensor()
        self.bc_weight = Scalar[DT](0.0)

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: Optional[DeviceContext] = None,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "SACActorLoss: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM
        ), "SACActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "SACActorLoss: CRITIC.IN_DIM must equal OBS_DIM + ACT_DIM"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "SACActorLoss: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.Graph.make[target, Zero](ctx)
        blk.graph.set_node_attr["rsample", "action_scale"](action_scale)
        # BC off: a zero mask and a zero weight. Scale by 0 kills the
        # gradient; the zero mask keeps the L1 itself at 0.
        blk.graph.set_node_attr["bc_w", "multiplier"](Scalar[DT](0.0))
        blk.graph.set_node_attr["loss_qw", "multiplier"](Scalar[DT](1.0))
        blk._bc_mask = Tensor.make[target](Self.BATCH, ctx)
        # the host copy `set_bc` writes and uploads (zeros until then)
        blk._bc_mask.ensure(Self.BATCH)
        comptime if target == "cpu":
            blk._loss_out = Tensor.alloc(Self.BATCH)
            blk._grad_seed = Tensor.alloc(Self.BATCH)
            for b in range(Self.BATCH):
                blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        else:
            var c = ctx.value()
            blk._loss_out = Tensor.alloc_gpu(c, Self.BATCH)
            blk._grad_seed = Tensor.alloc(Self.BATCH)
            for b in range(Self.BATCH):
                blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            blk._grad_seed.upload(c)
            blk._lp_mean = Tensor.alloc_gpu(c, 1)
            blk._lp_mean.dev.value().enqueue_fill(Scalar[DT](0))
            blk._loss_acc = Tensor.alloc_gpu(c, 2)
            blk._loss_acc.dev.value().enqueue_fill(Scalar[DT](0))
            # The BC weight lives in a device word the `bc_w` Scale reads at
            # every launch, like α: a host multiplier is BAKED into a captured
            # CUDA graph, so `set_bc_weight` after the first train step (the
            # mean-|Q| tracking of `set_bc_q_ratio`) would be a silent no-op
            # there. One H2D of one word per flush instead.
            blk._bc_w_dev = Tensor.alloc_gpu(c, 1)   # device word, zeroed
            blk._bc_w_dev.ensure(1)                  # + its host copy for `upload`
            blk.graph.set_node_attr_buf["bc_w", "multiplier"](
                blk._bc_w_dev.dev.value()
            )
        return blk^

    def set_bc(
        mut self, weight: Scalar[DT], n_demo_rows: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Turn the behaviour-cloning term on: weight λ, applied to the first
        `n_demo_rows` rows of every batch (the replay's pinned-demo half).
        See the header. `weight` 0 turns it off again."""
        if n_demo_rows < 0 or n_demo_rows > Self.BATCH:
            raise Error("set_bc: n_demo_rows out of [0, BATCH]")
        self.bc_weight = weight
        for b in range(Self.BATCH):
            self._bc_mask.data[b] = Scalar[DT](1.0) if b < n_demo_rows else Scalar[DT](0.0)
        if self._bc_mask.dev:
            self._bc_mask.upload(ctx.value())
        self.set_bc_weight(weight, ctx)

    def set_bc_weight(
        mut self, weight: Scalar[DT], ctx: Optional[DeviceContext] = None
    ) raises:
        """Change λ alone; the mask stays. Safe after the first train step on
        every target (the GPU weight is a device word, see `make`) — this is
        what `SACTrainer.flush_metrics` calls to track mean|Q|."""
        self.bc_weight = weight
        self.graph.set_node_attr["bc_w", "multiplier"](weight)
        if self._bc_w_dev.dev:
            self._bc_w_dev.data[0] = weight
            # ⚠ IN PLACE. `upload` REALLOCATES the device buffer, and the
            # `bc_w` Scale node holds the handle wired in `make` — an upload
            # would leave it reading the original zeroed word for the whole
            # run (it did: the first --bc-q-ratio run trained with λ = 0).
            self._bc_w_dev.upload_resident(ctx.value())

    def set_q_weight(mut self, weight: Scalar[DT]) raises:
        """The multiplier on the SAC half of the loss, `α·logp − min_q`: 1 by
        default; 0 makes the actor loss the BC term alone (a behaviour-cloning
        phase inside the same trainer — the critics keep training on the
        imitator's own rollouts, the actor ignores them). ⚠ 0 ALSO SILENCES
        THE ENTROPY TERM: nothing then moves `log_std` but the BC gradient
        through `mu`, so the exploration noise stays where the init put it."""
        self.graph.set_node_attr["loss_qw", "multiplier"](weight)

    # ── Device-α accessors (GPU only) ────────────────────────────────
    def lp_mean_dev(mut self) -> DeviceBuffer[DT]:
        """The device `lp_mean` [1] buffer — the device ScalarAdam reads it as
        the per-step entropy grad."""
        return self._lp_mean.dev.value()

    def set_alpha_buf(mut self, buf: DeviceBuffer[DT]):
        """One-time GPU wiring: point the `alogp` Scale node at SAC's on-device
        α buffer. After this the actor-loss forward/vjp read α on-device, so
        `forward_backward` skips the per-step host α bake."""
        self.graph.set_node_attr_buf["alogp", "multiplier"](buf)

    def reset_loss_accum(mut self) raises:
        """Zero the device (Σmean, count) loss accumulator — flush cadence."""
        self._loss_acc.dev.value().enqueue_fill(Scalar[DT](0))

    def read_loss_accum(mut self, ctx: DeviceContext) raises -> Scalar[DT]:
        """D2H the device loss accumulator once (flush cadence) and return its
        window mean (Σmean / count). 0 if no steps."""
        self._loss_acc.download(ctx)
        var s = self._loss_acc.data[0]
        var n = self._loss_acc.data[1]
        if n == Scalar[DT](0.0):
            return Scalar[DT](0.0)
        return s / n

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mut mb_s: Tensor,
        mut mb_a: Tensor,
        alpha: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> SACActorLossOut:
        comptime BB = Self.BATCH
        actor.zero_grad[target](ctx)
        # CPU bakes the host α scalar into the `alogp` Scale node per step. On
        # GPU α lives on-device (wired once at make via `set_alpha_buf`) and is
        # refreshed by the device ScalarAdam — no per-step host work.
        comptime if target == "cpu":
            self.graph.set_node_attr["alogp", "multiplier"](alpha)  # α

        # Seed the graph input slot with s (a COPY into the graph pool), then
        # forward (actor + online critics threaded as tracked refs).
        self.graph.set_input["s", BB](mb_s, ctx)
        # The BC inputs: the batch's stored actions and the demo-row mask.
        # Seeded every step (a D2D copy on GPU, capture-safe); with the mask
        # and weight at zero they cost two copies and nothing else.
        self.graph.set_input["a_demo", BB](mb_a, ctx)
        self.graph.set_input["bc_mask", BB](self._bc_mask, ctx)
        self.graph.forward[BB, target, POLICY](
            self._loss_out, ctx, actor, critic1, critic2
        )

        var loss_mean: Scalar[DT] = 0.0
        var lp_mean: Scalar[DT] = 0.0
        comptime if target == "cpu":
            # CPU: host reduction over the graph buffers (bit-identity path).
            var loss_sum: Scalar[DT] = 0.0
            var lp_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += self._loss_out.data[b]
                lp_sum += self.graph.node_output["logp"]().data[b]
            var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)
            loss_mean = loss_sum * inv_b
            lp_mean = lp_sum * inv_b
        else:
            # GPU: device-reduce both, NO D2H. lp_mean → `_lp_mean` (read by the
            # device ScalarAdam this step); loss → `_loss_acc` (drained at flush).
            # Returned host scalars stay 0 sentinels — the trainer drains the
            # device buffers instead.
            var c = ctx.value()
            # Materialize the detached (MutAnyOrigin) device views first, so no
            # `self` field borrow is held across the launch.
            var logp_v = self.graph.node_output["logp"]().lt[
                "gpu", Layout.row_major(BB, 1)
            ]()
            var lp_mean_v = self._lp_mean.lt["gpu", Layout.row_major(1, 1)]()
            var loss_v = self._loss_out.lt["gpu", Layout.row_major(BB, 1)]()
            var loss_acc_v = self._loss_acc.lt["gpu", Layout.row_major(1, 2)]()
            c.enqueue_function[reduce_mean_write_kernel[BB]](
                logp_v,
                lp_mean_v,
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            c.enqueue_function[reduce_mean_acc_kernel[BB]](
                loss_v,
                loss_acc_v,
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )

        # backward (seed = 1/BATCH) + actor step. Grad flows through the
        # critics (param grads discarded) into the actor (stepped).
        self.graph.vjp[BB, target, POLICY](
            self._grad_seed, ctx, actor, critic1, critic2
        )
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)

        return SACActorLossOut(loss=loss_mean, log_prob_mean=lp_mean)
