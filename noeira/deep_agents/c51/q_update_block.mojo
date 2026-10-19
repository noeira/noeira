"""C51QUpdateBlock — distributional Q-net gradient step (cross-entropy) (STORAGE).

Mirrors `dqn/q_update_block.mojo` but the loss is a cross-entropy against the
per-batch target distribution `m [B, N_ATOMS]` from `C51TargetYBlock` instead of
MSE against a scalar y.

Pipeline:
  1. q.zero_grad
  2. Q_online(s) → _logits_all                       [B, NA · N_ATOMS]
  3. GatherActionSlice(_logits_all, mb_a) → _logits_a [B, N_ATOMS]
  4. CrossEntropyLoss[N_ATOMS](_logits_a, m) → scalar loss
  5. CE.vjp(_logits_a, m, grad_logits_a) → (softmax(_logits_a) − m)/BATCH
  6. Scatter grad_logits_a into _grad_logits_all at slot `a_taken · N_ATOMS`
  7. Q_online.vjp(_grad_logits_all) → _grad_obs (discarded)
  8. opt.step

PER: same sentinel pattern as `DQNQUpdateBlock`. PER signal = per-sample
cross-entropy (captured before IS scaling); IS weights scale grad_logits_a
per-row after capture.

STORAGE migration (Stage 5): `Scratch`/`TargetStorage`/`init_scratch_auto`/
TileTensor gone; scratch are owned `Tensor`s; the gather slice feeds from a
block-owned `TensorPack[2]` (§B0); storage `CrossEntropyLoss` + `Adam`. CPU +
GPU. PER weights/td_residuals are `Optional[LayoutTensor]` (mirror
`CriticUpdateBlock`/`DQNQUpdateBlock`).
"""

from std.math import exp as fexp, log as flog
from max.gpu import global_idx
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
from noeira.nn.loss.cross_entropy import CrossEntropyLoss
from noeira.nn.primitives.gather_action_slice import GatherActionSlice
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level.
# ──────────────────────────────────────────────────────────────────────


def _c51_per_residual_kernel[
    BATCH: Int, N_ATOMS: Int
](
    logits_a: Tile[DT, BATCH, N_ATOMS],
    m: Tile[DT, BATCH, N_ATOMS],
    td_out: Tile[DT, BATCH],
):
    """Per-row cross-entropy used as the PER priority signal."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var mx = rebind[Scalar[DT]](logits_a[b, 0])
        for i in range(1, N_ATOMS):
            var v = rebind[Scalar[DT]](logits_a[b, i])
            if v > mx:
                mx = v
        var s_exp: Scalar[DT] = Scalar[DT](0.0)
        for i in range(N_ATOMS):
            s_exp = s_exp + fexp(rebind[Scalar[DT]](logits_a[b, i]) - mx)
        var lse = mx + flog(s_exp)
        var ce: Scalar[DT] = Scalar[DT](0.0)
        for i in range(N_ATOMS):
            var log_p = rebind[Scalar[DT]](logits_a[b, i]) - lse
            if log_p < Scalar[DT](-20.0):
                log_p = Scalar[DT](-20.0)
            ce = ce - rebind[Scalar[DT]](m[b, i]) * log_p
        td_out[b] = ce


def _c51_per_scale_kernel[
    BATCH: Int, N_ATOMS: Int
](
    grad_logits_a: Tile[DT, BATCH, N_ATOMS],
    weights: Tile[DT, BATCH],
):
    """`grad_logits_a[b, i] *= weights[b]`."""
    var idx = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if idx < total:
        var b = idx // N_ATOMS
        var i = idx % N_ATOMS
        grad_logits_a[b, i] = rebind[Scalar[DT]](grad_logits_a[b, i]) * rebind[
            Scalar[DT]
        ](weights[b])


def _c51_scatter_grad_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int
](
    grad_logits_a: Tile[DT, BATCH, N_ATOMS],
    mb_a: Tile[DT, BATCH],
    grad_logits_all: Tile[DT, BATCH, NA * N_ATOMS],
):
    """`grad_logits_all[b, c] = 0`, then `grad_logits_all[b, a·N + i] =
    grad_logits_a[b, i]` for `i ∈ [0, N_ATOMS)` and `a = int(mb_a[b])`."""
    var idx = Int(global_idx.x)
    comptime ROW = NA * N_ATOMS
    var total = BATCH * ROW
    if idx < total:
        var b = idx // ROW
        var c = idx % ROW
        var a = Int(rebind[Scalar[DT]](mb_a[b]))
        var lo = a * N_ATOMS
        var hi = lo + N_ATOMS
        if c >= lo and c < hi:
            grad_logits_all[b, c] = rebind[Scalar[DT]](grad_logits_a[b, c - lo])
        else:
            grad_logits_all[b, c] = Scalar[DT](0.0)


struct C51QUpdateBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
    N_ATOMS: Int,
](Defaultable & Movable & Deinitable):
    var ce_loss: CrossEntropyLoss[Self.N_ATOMS]
    var gather_slice: GatherActionSlice[Self.NA, Self.N_ATOMS]

    # GatherActionSlice inputs share ONE origin (§B0): slot[0] = Q(s) logits
    # [B, NA·N_ATOMS] (Q.forward writes here), slot[1] = an mb_a copy [B].
    var _gather_in: TensorPack[2]

    var _logits_a: Tensor          # [B · N_ATOMS]
    var _grad_logits_a: Tensor     # [B · N_ATOMS]
    var _grad_logits_all: Tensor   # [B · NA · N_ATOMS]
    var _grad_obs: Tensor          # [B · OBS]

    def __init__(out self):
        self.ce_loss = CrossEntropyLoss[Self.N_ATOMS]()
        self.gather_slice = GatherActionSlice[Self.NA, Self.N_ATOMS]()
        self._gather_in = TensorPack[2]()
        self._logits_a = Tensor()
        self._grad_logits_a = Tensor()
        self._grad_logits_all = Tensor()
        self._grad_obs = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "C51QUpdateBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "C51QUpdateBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA * Self.N_ATOMS
        ), "C51QUpdateBlock: Q_NET.OUT_DIM must equal NA · N_ATOMS"
        var b = Self()
        b.gather_slice = GatherActionSlice[Self.NA, Self.N_ATOMS].make[
            target, INIT=Zero,
        ](ctx=ctx)
        comptime ROW = Self.NA * Self.N_ATOMS
        comptime if target == "cpu":
            b.ce_loss = CrossEntropyLoss[Self.N_ATOMS].make_cpu()
            b._gather_in[0].ensure(Self.BATCH * ROW)
            b._gather_in[1].ensure(Self.BATCH)
            b._logits_a = Tensor.alloc(Self.BATCH * Self.N_ATOMS)
            b._grad_logits_a = Tensor.alloc(Self.BATCH * Self.N_ATOMS)
            b._grad_logits_all = Tensor.alloc(Self.BATCH * ROW)
            b._grad_obs = Tensor.alloc(Self.BATCH * Self.OBS)
        else:
            var c = ctx.value()
            b.ce_loss = CrossEntropyLoss[Self.N_ATOMS].make_gpu(c)
            b._gather_in[0].ensure_gpu(c, Self.BATCH * ROW)
            b._gather_in[1].ensure_gpu(c, Self.BATCH)
            b._logits_a = Tensor.alloc_gpu(c, Self.BATCH * Self.N_ATOMS)
            b._grad_logits_a = Tensor.alloc_gpu(c, Self.BATCH * Self.N_ATOMS)
            b._grad_logits_all = Tensor.alloc_gpu(c, Self.BATCH * ROW)
            b._grad_obs = Tensor.alloc_gpu(c, Self.BATCH * Self.OBS)
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
        mut mb_m: Tensor,
        weights: Optional[
            Tile[DT, Self.BATCH]
        ] = None,
        td_residuals: Optional[
            Tile[DT, Self.BATCH]
        ] = None,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """`zero_grad` → Q.forward → gather slice → CE forward(+accum) → CE.vjp →
        [PER capture + IS scale] → scatter → Q.vjp → opt.step. Returns the
        scalar loss (0 sentinel under GPU ACCUMULATE; read at flush)."""
        comptime ROW = Self.NA * Self.N_ATOMS

        # 1. Zero grads.
        q_online.zero_grad[target](ctx)

        # 2. Q_online(s) → _gather_in[0] ([B, NA·N_ATOMS]).
        call_forward[target, Self.BATCH, POLICY=POLICY](
            q_online, TensorRefs[Self.Q_NET.ARITY](mb_s), self._gather_in[0], ctx
        )

        # 3. Stage mb_a into the gather pool (§B0), gather slice at a_taken.
        comptime if target == "cpu":
            for k in range(Self.BATCH):
                self._gather_in[1].data[k] = mb_a.data[k]
        else:
            ctx.value().enqueue_copy(
                self._gather_in[1].dev.value(), mb_a.dev.value()
            )
        self.gather_slice.forward[target, Self.BATCH, POLICY=POLICY](
            TensorRefs[2](self._gather_in[0], self._gather_in[1]),
            self._logits_a,
            ctx,
        )

        # 4. CE(logits_a, m) → scalar loss.
        var loss: Scalar[DT]
        comptime if target == "gpu" and ACCUMULATE:
            self.ce_loss.forward_accumulate[target, Self.BATCH](
                self._logits_a, mb_m, ctx
            )
            loss = Scalar[DT](0.0)
        else:
            loss = self.ce_loss.forward[target, Self.BATCH](
                self._logits_a, mb_m, ctx
            )

        # 5. CE.vjp → grad_logits_a = (softmax(logits_a) − m) / BATCH.
        self.ce_loss.vjp[target, Self.BATCH](
            self._logits_a, mb_m, self._grad_logits_a, ctx
        )

        # 5a. PER residual capture (per-sample CE), BEFORE IS scaling.
        if td_residuals:
            var td = td_residuals.value()
            comptime if target == "cpu":
                for b in range(Self.BATCH):
                    var off = b * Self.N_ATOMS
                    var mx = self._logits_a.data[off]
                    for i in range(1, Self.N_ATOMS):
                        if self._logits_a.data[off + i] > mx:
                            mx = self._logits_a.data[off + i]
                    var s_exp: Scalar[DT] = 0.0
                    for i in range(Self.N_ATOMS):
                        s_exp = s_exp + fexp(self._logits_a.data[off + i] - mx)
                    var lse = mx + flog(s_exp)
                    var ce: Scalar[DT] = 0.0
                    for i in range(Self.N_ATOMS):
                        var log_p = self._logits_a.data[off + i] - lse
                        if log_p < Scalar[DT](-20.0):
                            log_p = Scalar[DT](-20.0)
                        ce = ce - mb_m.data[off + i] * log_p
                    td[b] = ce
            else:
                comptime n_blocks_pr = (Self.BATCH + TPB - 1) // TPB
                ctx.value().enqueue_function[
                    _c51_per_residual_kernel[Self.BATCH, Self.N_ATOMS]
                ](
                    self._logits_a.tt["gpu", Self.BATCH, Self.N_ATOMS](),
                    mb_m.tt["gpu", Self.BATCH, Self.N_ATOMS](),
                    td,
                    grid_dim=n_blocks_pr, block_dim=TPB,
                )

        # 5b. PER IS-weight scaling on grad_logits_a (per-row).
        if weights:
            var w = weights.value()
            comptime if target == "cpu":
                for b in range(Self.BATCH):
                    for i in range(Self.N_ATOMS):
                        self._grad_logits_a.data[b * Self.N_ATOMS + i] = (
                            self._grad_logits_a.data[b * Self.N_ATOMS + i]
                            * rebind[Scalar[DT]](w[b])
                        )
            else:
                comptime total_scl = Self.BATCH * Self.N_ATOMS
                comptime n_blocks_ps = (total_scl + TPB - 1) // TPB
                ctx.value().enqueue_function[
                    _c51_per_scale_kernel[Self.BATCH, Self.N_ATOMS]
                ](
                    self._grad_logits_a.tt["gpu", Self.BATCH, Self.N_ATOMS](),
                    w,
                    grid_dim=n_blocks_ps, block_dim=TPB,
                )

        # 6. Scatter grad_logits_a → grad_logits_all at the a_taken slot.
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                var a = Int(mb_a.data[b])
                var dst_base = b * ROW
                for c in range(ROW):
                    self._grad_logits_all.data[dst_base + c] = Scalar[DT](0.0)
                var src_base = b * Self.N_ATOMS
                var dst_slice = dst_base + a * Self.N_ATOMS
                for i in range(Self.N_ATOMS):
                    self._grad_logits_all.data[dst_slice + i] = (
                        self._grad_logits_a.data[src_base + i]
                    )
        else:
            comptime total_sc = Self.BATCH * ROW
            comptime n_blocks_sc = (total_sc + TPB - 1) // TPB
            ctx.value().enqueue_function[
                _c51_scatter_grad_kernel[Self.BATCH, Self.NA, Self.N_ATOMS]
            ](
                self._grad_logits_a.tt["gpu", Self.BATCH, Self.N_ATOMS](),
                mb_a.tt["gpu", Self.BATCH](),
                self._grad_logits_all.tt["gpu", Self.BATCH, ROW](),
                grid_dim=n_blocks_sc, block_dim=TPB,
            )

        # 7. Q_online.vjp(grad_logits_all) → grad_obs (discarded).
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            q_online,
            TensorRefs[Self.Q_NET.ARITY](mb_s),
            self._grad_logits_all,
            TensorRefs[Self.Q_NET.ARITY](self._grad_obs),
            ctx,
        )

        # 8. opt.step.
        q_opt.step[target, M=Self.Q_NET](q_online, ctx)

        return loss
