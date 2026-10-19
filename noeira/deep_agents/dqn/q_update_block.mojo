"""DQNQUpdateBlock — DQN online-Q gradient step (STORAGE).

DQN's single-Q analogue of `loss/critic_update_block.mojo`'s
`CriticUpdateBlock`. Differences from the SAC critic block:

  - Q-net takes observation only (`mb_s`, OBS dims), not `sa = concat(s,a)`.
  - Gather step between Q.forward and MSE: extract `Q(s, a_taken)` from
    `[B, NA]` via the action indices in `mb_a` ([B, 1]). Done with the
    storage `GatherCols[NA].forward` — on-device, no CPU shim.
  - Scatter step before Q.vjp: only the `a_taken` slot of `grad_q_all`
    has gradient, the rest must be zero. Done via a block-owned kernel
    `_scatter_action_grad` that takes `mb_a` directly — cheaper than going
    through `GatherCols.vjp` which doesn't have access to indices on the
    standard `Module.vjp(grad_output, *grad_inputs)` surface.

STORAGE migration (Stage 5):
  - `Scratch`/`TargetStorage`/`assert_tag_for`/`init_scratch_auto`/TileTensor
    gone — scratch are owned `nn.storage.Tensor`s (alloc on target).
  - `Q_NET.forward/vjp` use the storage surface: `forward[target,B,POLICY=](
    TensorRefs, mut out, ctx)`, `vjp[target,B,POLICY=](TensorRefs(input),
    mut grad_out, TensorRefs(grad_in), ctx)`.
  - `GatherCols.forward` takes `TensorRefs[2]` from a block-owned `TensorPack[2]`
    (q_all + an mb_a copy → ONE origin, the §B0 constraint).
  - storage `MSELoss[1]` (forward / forward_accumulate / vjp / read_accum).
  - optimizer step via storage `Adam.step[target, M](q_online, ctx)`.

CPU + GPU.

PER hooks (gated on non-null `weights`/`td_residuals` LayoutTensor sentinels;
null = uniform replay, bit-identical to pre-PER). Mirrors the storage
`CriticUpdateBlock` PER surface exactly.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Zero
from noeira.nn.optimizer.adam import Adam
from noeira.nn.loss.mse_loss import MSELoss
from noeira.nn.primitives.gather_cols import GatherCols
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# Block-owned kernels (PER scaling/capture + action-grad scatter).
# ──────────────────────────────────────────────────────────────────────


def _capture_td_residuals_kernel[
    BATCH: Int
](
    mb_grad_q: Tile[DT, BATCH, 1],
    out_residuals: Tile[DT, BATCH],
):
    """Recover the raw signed TD residual `Q − y = mb_grad_q·BATCH` (the value
    MSE.vjp wrote) for PER priority refresh. Run BEFORE IS-weight scaling."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    out_residuals[i] = rebind[Scalar[DT]](mb_grad_q[i, 0]) * Scalar[DT](BATCH)


def _scale_grad_by_weights_kernel[
    BATCH: Int
](
    mb_grad_q: Tile[DT, BATCH, 1],
    weights: Tile[DT, BATCH],
):
    """In-place PER IS-weight scaling `mb_grad_q[i,0] *= weights[i]`."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    mb_grad_q[i, 0] = mb_grad_q[i, 0] * rebind[Scalar[DT]](weights[i])


def _scatter_action_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_q_gath: Tile[DT, BATCH, 1],
    mb_a: Tile[DT, BATCH, 1],
    grad_q_all: Tile[DT, BATCH, NA],
):
    """`grad_q_all[b, k] = grad_q_gath[b, 0]` if `k == Int(mb_a[b, 0])` else `0`.
    One thread per (BATCH * NA) element; per-row write is unique (no atomics)."""
    var lin = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * NA
    if lin < total:
        var b = lin // NA
        var k = lin % NA
        var a = Int(rebind[Scalar[DT]](mb_a[b, 0]))
        if k == a:
            grad_q_all[b, k] = rebind[Scalar[DT]](grad_q_gath[b, 0])
        else:
            grad_q_all[b, k] = Scalar[DT](0.0)


struct DQNQUpdateBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
](Defaultable & Movable & Deinitable):
    var mse_loss: MSELoss[1]
    var gather_cols: GatherCols[Self.NA]

    # GatherCols inputs share ONE origin (§B0): slot[0] = Q(s) [B, NA] (the
    # Q.forward output writes here directly), slot[1] = an mb_a copy [B].
    var _gather_in: TensorPack[2]

    var _mb_q_gath: Tensor       # [B] — Q(s, a_taken)
    var _mb_grad_q: Tensor       # [B] — d(loss)/dQ_gathered
    var _mb_grad_q_all: Tensor   # [B*NA] — scattered into the a_taken slot
    var _mb_grad_obs: Tensor     # [B*OBS] — Q.vjp grad-input (discarded)

    def __init__(out self):
        self.mse_loss = MSELoss[1]()
        self.gather_cols = GatherCols[Self.NA]()
        self._gather_in = TensorPack[2]()
        self._mb_q_gath = Tensor()
        self._mb_grad_q = Tensor()
        self._mb_grad_q_all = Tensor()
        self._mb_grad_obs = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DQNQUpdateBlock: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.Q_NET.IN_DIMS[0] == Self.OBS, (
            "DQNQUpdateBlock: Q_NET.IN_DIM must equal OBS"
        )
        comptime assert Self.Q_NET.OUT_DIM == Self.NA, (
            "DQNQUpdateBlock: Q_NET.OUT_DIM must equal NA"
        )
        var b = Self()
        b.gather_cols = GatherCols[Self.NA].make[target, INIT=Zero](ctx=ctx)
        comptime if target == "cpu":
            b.mse_loss = MSELoss[1].make_cpu()
            b._gather_in[0].ensure(Self.BATCH * Self.NA)
            b._gather_in[1].ensure(Self.BATCH)
            b._mb_q_gath = Tensor.alloc(Self.BATCH)
            b._mb_grad_q = Tensor.alloc(Self.BATCH)
            b._mb_grad_q_all = Tensor.alloc(Self.BATCH * Self.NA)
            b._mb_grad_obs = Tensor.alloc(Self.BATCH * Self.OBS)
        else:
            var c = ctx.value()
            b.mse_loss = MSELoss[1].make_gpu(c)
            b._gather_in[0].ensure_gpu(c, Self.BATCH * Self.NA)
            b._gather_in[1].ensure_gpu(c, Self.BATCH)
            b._mb_q_gath = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_grad_q = Tensor.alloc_gpu(c, Self.BATCH)
            b._mb_grad_q_all = Tensor.alloc_gpu(c, Self.BATCH * Self.NA)
            b._mb_grad_obs = Tensor.alloc_gpu(c, Self.BATCH * Self.OBS)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut q_online: Self.Q_NET,
        mut q_opt: Adam,
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
        """`zero_grad` → Q.forward → gather → MSE forward(+accum) → MSE.vjp →
        [PER td capture + IS-weight scale] → scatter → Q.vjp → opt.step.
        Returns the scalar loss (0 sentinel under GPU ACCUMULATE; read at flush
        via `mse_loss.read_accum`)."""
        # 1. Zero grads.
        q_online.zero_grad[target](ctx)

        # 2. Q_online(s) → _gather_in[0] ([B, NA]).
        call_forward[target, Self.BATCH, POLICY=POLICY](
            q_online, TensorRefs[Self.Q_NET.ARITY](mb_s), self._gather_in[0], ctx
        )

        # 3. Stage mb_a into the gather pool (so both GatherCols inputs share
        # ONE origin — the §B0 constraint), then gather Q(s, a_taken).
        comptime if target == "cpu":
            for k in range(Self.BATCH):
                self._gather_in[1].data[k] = mb_a.data[k]
        else:
            ctx.value().enqueue_copy(
                self._gather_in[1].dev.value(), mb_a.dev.value()
            )
        self.gather_cols.forward[target, Self.BATCH, POLICY=POLICY](
            TensorRefs[2](self._gather_in[0], self._gather_in[1]),
            self._mb_q_gath,
            ctx,
        )

        # 4. MSE(q_gath, y). On GPU+ACCUMULATE the reduction stays on device
        # (no D2H → capturable); host reads it at flush via read_accum.
        var loss: Scalar[DT]
        comptime if target == "gpu" and ACCUMULATE:
            self.mse_loss.forward_accumulate[target, Self.BATCH](
                self._mb_q_gath, mb_y, ctx
            )
            loss = Scalar[DT](0.0)
        else:
            loss = self.mse_loss.forward[target, Self.BATCH](
                self._mb_q_gath, mb_y, ctx
            )

        # 5. MSE.vjp → grad_q ([B]).
        self.mse_loss.vjp[target, Self.BATCH](
            self._mb_q_gath, mb_y, self._mb_grad_q, ctx
        )

        # 5a. PER residual capture (raw signed TD `Q−y`), BEFORE IS scaling.
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

        # 5b. PER IS-weight scaling.
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

        # 6. Scatter grad_q into grad_q_all at the taken-action slot.
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                var a = Int(mb_a.data[b])
                for k in range(Self.NA):
                    if k == a:
                        self._mb_grad_q_all.data[b * Self.NA + k] = (
                            self._mb_grad_q.data[b]
                        )
                    else:
                        self._mb_grad_q_all.data[b * Self.NA + k] = Scalar[DT](
                            0.0
                        )
        else:
            comptime total = Self.BATCH * Self.NA
            comptime n_blocks = (total + TPB - 1) // TPB
            ctx.value().enqueue_function[
                _scatter_action_grad_kernel[Self.BATCH, Self.NA]
            ](
                self._mb_grad_q.tt["gpu", Self.BATCH, 1](),
                mb_a.tt["gpu", Self.BATCH, 1](),
                self._mb_grad_q_all.tt["gpu", Self.BATCH, Self.NA](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

        # 7. Q.vjp(grad_q_all) → grad_obs (discarded).
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            q_online,
            TensorRefs[Self.Q_NET.ARITY](mb_s),
            self._mb_grad_q_all,
            TensorRefs[Self.Q_NET.ARITY](self._mb_grad_obs),
            ctx,
        )

        # 8. opt.step.
        q_opt.step[target, M=Self.Q_NET](q_online, ctx)

        return loss
