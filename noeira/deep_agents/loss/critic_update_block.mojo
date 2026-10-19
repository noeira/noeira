"""CriticUpdateBlock / TwinCriticUpdateBlock — critic-update LossBlocks (storage).

Single Module forward + MSELoss + backward + opt step (no fan-out, no DAG — a
ComputeGraph would just add overhead; the win is scratch ownership). The critic
`M` is a plain storage `Module` taking the PRE-CONCATENATED `[B, OBS+ACT]` input
(config = `Sequential[LinearReLU[OBS+ACT,H], …, Linear[H,1]]`); the (s,a) concat
happens here via `concat_sa` into the owned `_mb_sa` Tensor.

STORAGE migration (Stage 5):
  - scratch `Scratch` fields -> owned storage `Tensor`s (alloc on target).
  - `TargetStorage`/`assert_tag_for`/`init_scratch_auto`/`mptr`/TileTensor gone.
  - `critic.forward/vjp` use the storage surface: `forward[target,B](TensorRefs,
    mut out, ctx)`, `vjp[target,B](TensorRefs(input), mut grad_out,
    TensorRefs(grad_in), ctx)` — vjp takes the forward input explicitly (no cache).
  - storage `MSELoss[1]` (forward/forward_accumulate/vjp/read_accum, no POLICY).
  - optimizer step via storage `Adam.step[target, M](critic, ctx)`.

CPU + GPU.

Surface:
    CriticUpdateBlock[CRITIC, BATCH, SA_DIM]
        step[target, POLICY, ACCUMULATE](mut critic, mut opt, mut sa, mut y,
             weights=None, td_residuals=None, ctx) -> Scalar[DT]
    TwinCriticUpdateBlock[CRITIC, BATCH, OBS, ACT]
        step[target, POLICY, ACCUMULATE](mut c1, mut c1_opt, mut c2, mut c2_opt,
             mut mb_s, mut mb_a, mut mb_y, weights=None, td_residuals=None, ctx)
             -> Scalar[DT]
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.optimizer.adam import Adam
from noeira.nn.loss.mse_loss import MSELoss
from .loss_block import LossBlock
from ..training.off_policy_critic import concat_sa, concat_sa_gpu
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# IS-weighted MSE gradient scaling + TD-residual capture kernels (PER).
# ──────────────────────────────────────────────────────────────────────


def _scale_grad_by_weights_kernel[
    BATCH: Int
](
    mb_grad_q: Tile[DT, BATCH, 1],
    weights: Tile[DT, BATCH],
):
    """Per-lane in-place scaling `mb_grad_q[i,0] *= weights[i]` — the PER
    correction `grad_θ ∝ Σ_b w_b·(Q_b−y_b)·∂Q/∂θ` (Schaul et al. §3.4)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    mb_grad_q[i, 0] = mb_grad_q[i, 0] * rebind[Scalar[DT]](weights[i])


def _capture_td_residuals_kernel[
    BATCH: Int
](
    mb_grad_q: Tile[DT, BATCH, 1],
    out_residuals: Tile[DT, BATCH],
):
    """Recover the unscaled signed TD residual `Q−y = mb_grad_q·BATCH` (the
    value MSE.vjp wrote) for PER priority refresh. Run BEFORE IS scaling."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    out_residuals[i] = rebind[Scalar[DT]](mb_grad_q[i, 0]) * Scalar[DT](BATCH)


struct CriticUpdateBlock[
    CRITIC: Module,
    BATCH: Int,
    SA_DIM: Int,
](LossBlock):
    """Single-critic MSE update step. Owns all intermediate scratch."""

    var mse_loss: MSELoss[1]
    var _mb_q: Tensor
    var _mb_grad_q: Tensor
    var _mb_grad_sa: Tensor

    def __init__(out self):
        self.mse_loss = MSELoss[1]()
        self._mb_q = Tensor()
        self._mb_grad_q = Tensor()
        self._mb_grad_sa = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "CriticUpdateBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "CriticUpdateBlock: CRITIC.IN_DIM must equal SA_DIM"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "CriticUpdateBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        comptime if target == "cpu":
            blk.mse_loss = MSELoss[1].make_cpu()
            blk._mb_q = Tensor.alloc(Self.BATCH)
            blk._mb_grad_q = Tensor.alloc(Self.BATCH)
            blk._mb_grad_sa = Tensor.alloc(Self.BATCH * Self.SA_DIM)
        else:
            var c = ctx.value()
            blk.mse_loss = MSELoss[1].make_gpu(c)
            blk._mb_q = Tensor.alloc_gpu(c, Self.BATCH)
            blk._mb_grad_q = Tensor.alloc_gpu(c, Self.BATCH)
            blk._mb_grad_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_DIM)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut critic: Self.CRITIC,
        mut opt: Adam,
        mut sa: Tensor,
        mut y: Tensor,
        weights: Optional[
            Tile[DT, Self.BATCH]
        ] = None,
        td_residuals: Optional[
            Tile[DT, Self.BATCH]
        ] = None,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """`zero_grad` → critic.forward → MSE.forward(+accum) → MSE.vjp →
        [PER td capture + IS-weight scale] → critic.vjp → opt.step. Returns
        the scalar loss (0 sentinel under GPU ACCUMULATE; read at flush)."""
        critic.zero_grad[target](ctx)
        call_forward[target, Self.BATCH, POLICY=POLICY](
            critic, TensorRefs[Self.CRITIC.ARITY](sa), self._mb_q, ctx
        )

        var loss: Scalar[DT]
        comptime if target == "gpu" and ACCUMULATE:
            self.mse_loss.forward_accumulate[target, Self.BATCH](
                self._mb_q, y, ctx
            )
            loss = Scalar[DT](0.0)
        else:
            loss = self.mse_loss.forward[target, Self.BATCH](
                self._mb_q, y, ctx
            )

        self.mse_loss.vjp[target, Self.BATCH](self._mb_q, y, self._mb_grad_q, ctx)

        # PER residual capture (raw signed TD `Q−y`), BEFORE IS scaling.
        if td_residuals:
            var td = td_residuals.value()
            comptime if target == "cpu":
                for i in range(Self.BATCH):
                    td[i] = self._mb_grad_q.data[i] * Scalar[DT](Self.BATCH)
            else:
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                ctx.value().enqueue_function[
                    _capture_td_residuals_kernel[Self.BATCH]
                ](
                    self._mb_grad_q.tt["gpu", Self.BATCH, 1](),
                    td,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

        # IS-weight scaling (PER), gated on Optional sentinel.
        if weights:
            var w = weights.value()
            comptime if target == "cpu":
                for i in range(Self.BATCH):
                    self._mb_grad_q.data[i] = (
                        self._mb_grad_q.data[i] * rebind[Scalar[DT]](w[i])
                    )
            else:
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                ctx.value().enqueue_function[
                    _scale_grad_by_weights_kernel[Self.BATCH]
                ](
                    self._mb_grad_q.tt["gpu", Self.BATCH, 1](),
                    w,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

        call_vjp[target, Self.BATCH, POLICY=POLICY](
            critic,
            TensorRefs[Self.CRITIC.ARITY](sa),
            self._mb_grad_q,
            TensorRefs[Self.CRITIC.ARITY](self._mb_grad_sa),
            ctx,
        )
        opt.step[target, M=Self.CRITIC](critic, ctx)
        return loss


struct TwinCriticUpdateBlock[
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    """Twin-critic update against shared target `y`. Owns two
    `CriticUpdateBlock`s + a shared `_mb_sa` concat scratch."""

    comptime SA_DIM = Self.OBS + Self.ACT

    var c1: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var c2: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var _mb_sa: Tensor

    def __init__(out self):
        self.c1 = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]()
        self.c2 = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]()
        self._mb_sa = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TwinCriticUpdateBlock: target must be 'cpu' or 'gpu'"
        var blk = Self()
        blk.c1 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target](ctx=ctx)
        blk.c2 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target](ctx=ctx)
        comptime if target == "cpu":
            blk._mb_sa = Tensor.alloc(Self.BATCH * Self.SA_DIM)
        else:
            blk._mb_sa = Tensor.alloc_gpu(ctx.value(), Self.BATCH * Self.SA_DIM)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut critic1: Self.CRITIC,
        mut critic1_opt: Adam,
        mut critic2: Self.CRITIC,
        mut critic2_opt: Adam,
        mut mb_s: Tensor,
        mut mb_a: Tensor,
        mut mb_y: Tensor,
        weights: Optional[
            Tile[DT, Self.BATCH]
        ] = None,
        td_residuals: Optional[
            Tile[DT, Self.BATCH]
        ] = None,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """`concat` (s,a) → c1.step + c2.step against shared `mb_y`. `td_residuals`
        captured from critic1 only (canonical PER proxy)."""
        comptime if target == "cpu":
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                mb_s.tt["cpu", Self.BATCH, Self.OBS](),
                mb_a.tt["cpu", Self.BATCH, Self.ACT](),
                self._mb_sa.tt["cpu", Self.BATCH, Self.SA_DIM](),
            )
        else:
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                ctx.value(),
                mb_s.tt["gpu", Self.BATCH, Self.OBS](),
                mb_a.tt["gpu", Self.BATCH, Self.ACT](),
                self._mb_sa.tt["gpu", Self.BATCH, Self.SA_DIM](),
            )

        var loss1 = self.c1.step[target, POLICY, ACCUMULATE](
            critic1, critic1_opt, self._mb_sa, mb_y,
            weights=weights, td_residuals=td_residuals, ctx=ctx,
        )
        var loss2 = self.c2.step[target, POLICY, ACCUMULATE](
            critic2, critic2_opt, self._mb_sa, mb_y,
            weights=weights, ctx=ctx,
        )
        return loss1 + loss2
