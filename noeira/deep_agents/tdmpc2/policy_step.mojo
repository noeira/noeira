"""TD-MPC2 policy (actor) update block — storage framework (CPU + GPU).

Drives `TDMPC2PolicyGraph`: threads the trainer-owned policy + RSample + two
(random) detached Q heads as externals (node order: pi_out, alp, q1, q2),
sets the per-step Scale multipliers (`0.5/scale` for Q, `entropy_coef·ACT`
for the entropy term), forwards, updates RunningScale from this step's avg-Q,
backprops into the policy only (the Q-head grads are computed-then-discarded;
the WM/TD steps zero_grad the Q nets before their own update), and steps the
pi optimizer.

RunningScale uses a one-step lag: the multiplier for step t uses the value
from step t−1's Q, then is updated from step t's Q for t+1. The EMA
(tau=0.01) drifts slowly so the lag is negligible — and it keeps the scale
out of the autograd graph (reference detaches it anyway).

The policy + Q heads + pi optimizer are passed by ref (trainer-owned); the
agent's comptime dispatch supplies the random Q pair as DISTINCT fields.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Zero
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch

from .nets import TDMPC2Policy, TDMPC2QNet
from .policy_graph import TDMPC2PolicyGraph
from .running_scale import RunningScale
from noeira.core.tiles import Tile


def _scale_copy_k[N: Int](
    src: Tile[DT, N],
    dst: Tile[DT, N],
    f: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i]) * f


struct PolicyStep[
    LATENT: Int,
    ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    QP: Float64 = 0.0,
](Movable & Deinitable):
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime GraphT = TDMPC2PolicyGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX, Self.QP,
    ]

    var graph: Self.GraphT
    var scale: RunningScale
    var entropy_coef: Scalar[DT]
    # last-step Q diagnostics (avg-of-2 decoded Q at the policy's actions).
    var q_mean: Scalar[DT]
    var q_min: Scalar[DT]
    var q_max: Scalar[DT]
    # scratch Tensors (allocated once in make, reused every step).
    var loss: Tensor   # [B] graph output (loss_per_b)
    var qavg: Tensor   # [B] host-side avg-Q for RunningScale + stats
    var grad: Tensor   # [B] backward seed (1/B)
    var z_in: Tensor   # [B*LATENT] input slot seed

    def __init__(out self):
        self.graph = Self.GraphT()
        self.scale = RunningScale()
        self.entropy_coef = Scalar[DT](1e-4)
        self.q_mean = Scalar[DT](0.0)
        self.q_min = Scalar[DT](0.0)
        self.q_max = Scalar[DT](0.0)
        self.loss = Tensor()
        self.qavg = Tensor()
        self.grad = Tensor()
        self.z_in = Tensor()

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
            "PolicyStep: target must be 'cpu' or 'gpu'"
        )
        comptime BB = Self.B
        var s = Self()
        s.graph = Self.GraphT.make[target, INIT=Zero](ctx=ctx)
        # loss + grad seed live where the graph reads/writes them.
        s.loss = Tensor.make[target](BB, ctx)
        s.grad = Tensor.make[target](BB, ctx)
        # qavg is host-resident (RunningScale + stats are host reductions).
        s.qavg = Tensor.alloc(BB)
        s.z_in = Tensor.make[target](BB * Self.LATENT, ctx)
        return s^

    def _bind(
        mut self,
        mut policy: Self.PolicyT,
    ) raises:
        self.graph.set_node_attr["alpha_lp", "multiplier"](
            self.entropy_coef * Scalar[DT](Self.ACT)
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
        mut z: Tensor,   # [B, LATENT] (host or device matching target)
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime BB = Self.B
        self._bind(policy)
        pi_opt.zero_grad[target, M=Self.PolicyT](policy, ctx)

        # seed input + forward (policy, q_a, q_b threaded in node order; the
        # RSample `alp` node is graph-internal so its noise cache persists to
        # vjp — mirrors the SAC actor-loss internal-rsample wiring).
        self.graph.set_input["z", BB](z, ctx)
        self.graph.forward[BB, target](self.loss, ctx, policy, q_a, q_b)

        # avg-Q = 0.5·qsum; host reduction (CPU reads pool, GPU D2H qsum→qavg).
        comptime if target == "cpu":
            ref qsum = self.graph.node_output["qsum"]()
            for b in range(BB):
                self.qavg.data[b] = qsum.data[b] * Scalar[DT](0.5)
        else:
            var c = ctx.value()
            comptime nb = (BB + TPB - 1) // TPB
            # scale qsum by 0.5 in place of qavg's device staging, then D2H.
            ref qsum = self.graph.node_output["qsum"]()
            # qsum lives in the graph pool; copy·0.5 into the loss-sized device
            # scratch via grad (reused before vjp seeds it).
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

        # backward (seed = 1/B) + policy step. Grad flows through the Q heads
        # (param grads discarded) into the policy (stepped).
        seed_grad_inv_batch[target, BB](
            self.grad.tt[target, BB, 1](), ctx=ctx
        )
        self.graph.vjp[BB, target](self.grad, ctx, policy, q_a, q_b)
        pi_opt.step[target, M=Self.PolicyT](policy, ctx)
        return loss_mean
