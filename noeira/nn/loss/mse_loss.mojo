"""MSELoss[DIM] — stateful MSE loss with device-resident accumulation (SAC).

The critic loss SAC reads each step. Transformed from legacy `nn.loss.MSELoss`;
SAC convention (NOT the mean-MSE free functions in `mse.mojo`):

    loss = (1/BATCH) Σ_b Σ_j 0.5·(logit_{b,j} - target_{b,j})²
    grad = (logit - target) / BATCH

Accumulation (capture-friendly metric readout, no per-step D2H):
  - `forward_accumulate` reduces on device and adds (mean_loss, +1) into the
    [2] `loss_acc` buffer ([sum_of_means, count]); CPU mirrors with host scalars.
  - `read_accum` D2Hs once at flush cadence; `reset_accum` zeroes.

STORAGE simplification: `vjp` takes `logits` explicitly (the caller keeps the
critic output around), so there is NO `cache_logits` field — unlike legacy,
whose trait `backward(targets, grad)` forced caching the logits on forward.
"""

from max.gpu import global_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from ..core.tensor import Tensor
from noeira.core.tiles import Tile


def _mse_fwd_kernel[
    BATCH: Int, DIM: Int
](
    logits: Tile[DT, BATCH, DIM],
    targets: Tile[DT, BATCH, DIM],
    partial: Tile[DT, BATCH],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var s: Scalar[DT] = 0.0
        for j in range(DIM):
            var d = rebind[Scalar[DT]](logits[b, j]) - rebind[Scalar[DT]](
                targets[b, j]
            )
            s += Scalar[DT](0.5) * d * d
        partial[b] = s


def _mse_reduce_kernel[
    BATCH: Int
](
    partial: Tile[DT, BATCH],
    acc: Tile[DT, 2],
):
    """block.sum(partial)/BATCH → acc[0] += mean ; acc[1] += 1. No D2H."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += rebind[Scalar[DT]](partial[k])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = rebind[Scalar[DT]](acc[0]) + total[0] / Scalar[DT](BATCH)
        acc[1] = rebind[Scalar[DT]](acc[1]) + Scalar[DT](1.0)


def _mse_bwd_kernel[
    BATCH: Int, DIM: Int
](
    logits: Tile[DT, BATCH, DIM],
    targets: Tile[DT, BATCH, DIM],
    grad: Tile[DT, BATCH, DIM],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * DIM:
        var b = idx // DIM
        var j = idx % DIM
        grad[b, j] = (
            rebind[Scalar[DT]](logits[b, j]) - rebind[Scalar[DT]](targets[b, j])
        ) / Scalar[DT](BATCH)


struct MSELoss[DIM_: Int](Movable & Deinitable):
    var partial: Tensor  # GPU [BATCH] per-row partials (lazy)
    var loss_acc: Tensor  # GPU [2] = [sum_of_means, count]
    var _acc_sum: Scalar[DT]
    var _acc_n: Int

    def __init__(out self):
        self.partial = Tensor()
        self.loss_acc = Tensor()
        self._acc_sum = Scalar[DT](0.0)
        self._acc_n = 0

    @staticmethod
    def make_cpu() raises -> Self:
        return Self()

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var l = Self()
        l.loss_acc.ensure_gpu(ctx, 2)
        l.loss_acc.dev.value().enqueue_fill(Scalar[DT](0))
        return l^

    def _mean_loss_cpu[
        B: Int
    ](self, ref logits: Tensor, ref targets: Tensor) -> Scalar[DT]:
        var s: Scalar[DT] = 0.0
        for i in range(B * Self.DIM_):
            var d = logits.data[i] - targets.data[i]
            s += Scalar[DT](0.5) * d * d
        return s / Scalar[DT](B)

    def forward[
        target: StaticString, B: Int
    ](
        mut self,
        mut logits: Tensor,
        mut targets: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime if target == "cpu":
            return self._mean_loss_cpu[B](logits, targets)
        else:
            var c = ctx.value()
            self.partial.ensure_gpu(c, B)
            comptime nblk = (B + TPB_REDUCE - 1) // TPB_REDUCE
            c.enqueue_function[_mse_fwd_kernel[B, Self.DIM_]](
                logits.tt["gpu", B, Self.DIM_](),
                targets.tt["gpu", B, Self.DIM_](),
                self.partial.tt["gpu", B](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
            self.partial.download(c)
            var s: Scalar[DT] = 0.0
            for b in range(B):
                s += self.partial.data[b]
            return s / Scalar[DT](B)

    def forward_accumulate[
        target: StaticString, B: Int
    ](
        mut self,
        mut logits: Tensor,
        mut targets: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self._acc_sum += self._mean_loss_cpu[B](logits, targets)
            self._acc_n += 1
        else:
            var c = ctx.value()
            self.partial.ensure_gpu(c, B)
            comptime nblk = (B + TPB_REDUCE - 1) // TPB_REDUCE
            c.enqueue_function[_mse_fwd_kernel[B, Self.DIM_]](
                logits.tt["gpu", B, Self.DIM_](),
                targets.tt["gpu", B, Self.DIM_](),
                self.partial.tt["gpu", B](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
            c.enqueue_function[_mse_reduce_kernel[B]](
                self.partial.tt["gpu", B](),
                self.loss_acc.tt["gpu", 2](),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )

    def reset_accum[target: StaticString](mut self) raises:
        comptime if target == "cpu":
            self._acc_sum = Scalar[DT](0.0)
            self._acc_n = 0
        else:
            self.loss_acc.dev.value().enqueue_fill(Scalar[DT](0))

    def read_accum[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext] = None) raises -> Scalar[DT]:
        comptime if target == "cpu":
            if self._acc_n == 0:
                return Scalar[DT](0.0)
            return self._acc_sum / Scalar[DT](self._acc_n)
        else:
            self.loss_acc.download(ctx.value())
            var s = self.loss_acc.data[0]
            var n = self.loss_acc.data[1]
            if n == Scalar[DT](0.0):
                return Scalar[DT](0.0)
            return s / n

    def vjp[
        target: StaticString, B: Int
    ](
        mut self,
        mut logits: Tensor,
        mut targets: Tensor,
        mut grad: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        comptime if target == "cpu":
            grad.ensure(M)
            for i in range(M):
                grad.data[i] = (logits.data[i] - targets.data[i]) / Scalar[DT](
                    B
                )
        else:
            var c = ctx.value()
            grad.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_mse_bwd_kernel[B, Self.DIM_]](
                logits.tt["gpu", B, Self.DIM_](),
                targets.tt["gpu", B, Self.DIM_](),
                grad.tt["gpu", B, Self.DIM_](),
                grid_dim=nblk,
                block_dim=TPB,
            )
