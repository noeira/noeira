"""DDPGActorLoss — deterministic policy gradient on the storage ComputeGraph.

Maximizes E_s[critic(s, π_φ(s))] via the per-batch loss

  loss_per_b[b] = -critic(s_b, π_φ(s_b)) ;  loss = mean_b loss_per_b
  ∂loss/∂loss_per_b = 1/BATCH

STORAGE migration: mirrors `SACActorLoss` minus the rsample/entropy machinery.
The deterministic actor (Tanh-bounded, ACT-wide) feeds straight into the critic;
the `Scale[-1]` node turns Q into the negated loss so the seed is the same
`1/BATCH` as SAC. The actor + the online critic are `ExternalNode`s threaded as
tracked `mut` refs into `graph.forward`/`vjp`: the actor accumulates param grads
and is stepped; the critic's grads are computed-then-discarded (the critic block
zero_grads before its own next update, exactly as in SAC — so the legacy
`vjp[mode="input_only"]` is unnecessary).

  graph: s → actor → action ; (s, action) → concat → q1 ; loss = -q1 (output)

CPU returns the scalar loss; GPU reduces `-mean q` on-device into a `[Σ, count]`
accumulator drained at flush (no per-step D2H — capture-friendly). Shared by TD3
(it uses critic1 only; identical math).
"""

from max.gpu import thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB_REDUCE
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Zero
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.concat import Concat2
from noeira.nn.primitives.scale import Scale
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node, ExternalNode
from ..loss.loss_block import LossBlock
from noeira.core.tiles import Tile


# ── Device reduction (single block; grid=1, block=TPB_REDUCE) ──────────
# `acc[0] += mean(src); acc[1] += 1`. src = loss_per_b = -q, so the running
# mean IS the actor loss. The host drains `acc` once per flush. Mirrors
# SACActorLoss's `reduce_mean_acc_kernel`.
def _ddpg_loss_mean_acc_kernel[
    B: Int
](
    src: Tile[DT, B, 1],
    acc: Tile[DT, 1, 2],
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


struct DDPGActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    comptime Graph = ComputeGraph[
        InputSlot["s", Self.OBS_DIM],                          # s
        ExternalNode["actor", Self.ACTOR, "s"],                # → action [B,ACT]
        Node["concat", Concat2[Self.OBS_DIM, Self.ACT_DIM], "s", "actor"],
        ExternalNode["q1", Self.CRITIC, "concat"],             # → [B, 1]
        Node["loss", Scale[1], "q1"],                          # -q1 (output)
    ]

    var graph: Self.Graph
    var _loss_out: Tensor   # [B] loss_per_b = -q1 (graph output)
    var _grad_seed: Tensor  # [B] = 1/BATCH (backward seed)
    # Device loss accumulator (GPU only): [Σ(-mean q), count], drained at flush.
    var _loss_acc: Tensor

    def __init__(out self):
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "DDPGActorLoss: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGActorLoss: CRITIC.OUT_DIM must equal 1"
        self.graph = Self.Graph()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()
        self._loss_acc = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "DDPGActorLoss: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.graph = Self.Graph.make[target, Zero](ctx)
        # loss_per_b = -q1: the Scale node's constant multiplier (set once).
        b.graph.set_node_attr["loss", "multiplier"](Scalar[DT](-1.0))
        comptime if target == "cpu":
            b._loss_out = Tensor.alloc(Self.BATCH)
            b._grad_seed = Tensor.alloc(Self.BATCH)
            for i in range(Self.BATCH):
                b._grad_seed.data[i] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        else:
            var c = ctx.value()
            b._loss_out = Tensor.alloc_gpu(c, Self.BATCH)
            b._grad_seed = Tensor.alloc(Self.BATCH)
            for i in range(Self.BATCH):
                b._grad_seed.data[i] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            b._grad_seed.upload(c)
            b._loss_acc = Tensor.alloc_gpu(c, 2)
            b._loss_acc.dev.value().enqueue_fill(Scalar[DT](0))
        return b^

    # ── GPU loss-accumulator accessors (flush cadence) ───────────────
    def reset_loss_accum(mut self) raises:
        """Zero the device (Σ, count) loss accumulator. GPU only."""
        self._loss_acc.dev.value().enqueue_fill(Scalar[DT](0))

    def read_loss_accum(mut self, ctx: DeviceContext) raises -> Scalar[DT]:
        """D2H the device loss accumulator once (flush cadence); return its
        window mean (Σ / count). 0 if no steps. GPU only."""
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
        mut critic: Self.CRITIC,
        mut mb_s: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """One DPG update step. Returns the scalar actor loss (= -mean_b q) on
        CPU; a 0 sentinel on GPU (the real metric is drained from
        `read_loss_accum` at flush). The caller holds `mut critic`; its params
        are NOT stepped here (grads discarded by the next critic update)."""
        comptime BB = Self.BATCH
        actor.zero_grad[target](ctx)

        # Seed the named input slot with s (a COPY into the graph pool), then
        # forward (actor + online critic threaded as tracked refs).
        self.graph.set_input["s", BB](mb_s, ctx)
        self.graph.forward[BB, target, POLICY](
            self._loss_out, ctx, actor, critic
        )

        var loss_mean: Scalar[DT] = 0.0
        comptime if target == "cpu":
            var loss_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += self._loss_out.data[b]
            loss_mean = loss_sum / Scalar[DT](BB)
        else:
            # Device-reduce -mean q into the accumulator (NO D2H).
            var c = ctx.value()
            var loss_v = self._loss_out.tt["gpu", BB, 1]()
            var acc_v = self._loss_acc.tt["gpu", 1, 2]()
            c.enqueue_function[_ddpg_loss_mean_acc_kernel[BB]](
                loss_v, acc_v, grid_dim=1, block_dim=TPB_REDUCE,
            )

        # backward (seed = 1/BATCH) + actor step. Grad flows through the
        # critic (param grads discarded) into the actor (stepped).
        self.graph.vjp[BB, target, POLICY](
            self._grad_seed, ctx, actor, critic
        )
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)
        return loss_mean
