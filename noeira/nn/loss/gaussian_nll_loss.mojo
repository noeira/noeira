"""GaussianNLLLoss[DIM] — diagonal Gaussian NLL (storage surface).

Transformed from legacy `nn.loss.GaussianNLLLoss`. The probabilistic head
used by MBPO's dynamics ensemble: each member predicts `(µ, raw_logvar)`
per output dim, and training maximises the log-likelihood of the observed
target under that diagonal Gaussian. Mirrors MSELoss/CrossEntropyLoss's
device-resident accumulator (forward_accumulate/read_accum/reset_accum).

STORAGE simplification: `vjp` RECOMPUTES (µ-y), σ⁻², in_clamp from `logits`
+ `targets` (passed explicitly) — NO cache_diff/cache_inv_var/cache_in_clamp
fields (unlike legacy, whose `Loss` trait forced caching on forward).

Tensor conventions:
  - logits   shape `BATCH × (2*DIM)` — first DIM cols means, next DIM raw logvars.
  - targets  shape `BATCH × DIM` — observed values.
  - grad     shape `BATCH × (2*DIM)` — first DIM = d/dµ, next DIM = d/d_raw_logvar.

Math (per-row, summed over DIM, averaged over BATCH):
  σ²       = exp(clamp(raw_logvar, [LOGVAR_MIN, LOGVAR_MAX]))
  loss_row = Σᵢ ½·(µᵢ - yᵢ)² · σ⁻²ᵢ + ½·clamped_logvarᵢ
  loss     = (1/BATCH) · Σ_row loss_row
  d/d_µ_i          = (µᵢ - yᵢ) · σ⁻²ᵢ / BATCH
  d/d_raw_logvar_i = [½ - ½·(µᵢ - yᵢ)²·σ⁻²ᵢ] / BATCH · in_clamp
"""

from std.math import exp
from max.gpu import global_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from ..core.tensor import Tensor
from noeira.core.tiles import Tile


def _gnll_fwd_kernel[
    BATCH: Int, DIM: Int, LV_MIN: Float64, LV_MAX: Float64
](
    logits: Tile[DT, BATCH, 2 * DIM],
    targets: Tile[DT, BATCH, DIM],
    partial: Tile[DT, BATCH],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var lv_min = Scalar[DT](LV_MIN)
        var lv_max = Scalar[DT](LV_MAX)
        var row_total: Scalar[DT] = 0.0
        for i in range(DIM):
            var mu = rebind[Scalar[DT]](logits[b, i])
            var raw_lv = rebind[Scalar[DT]](logits[b, DIM + i])
            var y = rebind[Scalar[DT]](targets[b, i])
            var lv = raw_lv
            if lv > lv_max:
                lv = lv_max
            elif lv < lv_min:
                lv = lv_min
            var inv_var = exp(-lv)
            var d = mu - y
            row_total += Scalar[DT](0.5) * d * d * inv_var + Scalar[DT](0.5) * lv
        partial[b] = row_total


def _gnll_reduce_kernel[
    BATCH: Int
](
    partial: Tile[DT, BATCH],
    acc: Tile[DT, 2],
):
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


def _gnll_bwd_kernel[
    BATCH: Int, DIM: Int, LV_MIN: Float64, LV_MAX: Float64
](
    logits: Tile[DT, BATCH, 2 * DIM],
    targets: Tile[DT, BATCH, DIM],
    grad: Tile[DT, BATCH, 2 * DIM],
):
    # one thread per element: recompute diff/inv_var/in_clamp from logits.
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var i = idx % DIM
        var lv_min = Scalar[DT](LV_MIN)
        var lv_max = Scalar[DT](LV_MAX)
        var inv_b: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
        var mu = rebind[Scalar[DT]](logits[b, i])
        var raw_lv = rebind[Scalar[DT]](logits[b, DIM + i])
        var y = rebind[Scalar[DT]](targets[b, i])
        var ic = Scalar[DT](1.0)
        var lv = raw_lv
        if lv > lv_max:
            lv = lv_max
            ic = Scalar[DT](0.0)
        elif lv < lv_min:
            lv = lv_min
            ic = Scalar[DT](0.0)
        var inv_v = exp(-lv)
        var d = mu - y
        # d_loss/d_µ = (µ-y) · σ⁻² / BATCH.
        grad[b, i] = d * inv_v * inv_b
        # d_loss/d_raw_lv = (½ - ½·d²·σ⁻²) / BATCH · in_clamp.
        grad[b, DIM + i] = (
            Scalar[DT](0.5) - Scalar[DT](0.5) * d * d * inv_v
        ) * inv_b * ic


struct GaussianNLLLoss[
    DIM_: Int,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](Movable & Deinitable):
    var partial: Tensor  # GPU [BATCH] per-row losses (lazy)
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
        var lv_min = Scalar[DT](Self.LOGVAR_MIN)
        var lv_max = Scalar[DT](Self.LOGVAR_MAX)
        var total: Scalar[DT] = 0.0
        for b in range(B):
            var lo = b * (2 * Self.DIM_)
            var to = b * Self.DIM_
            for i in range(Self.DIM_):
                var mu = logits.data[lo + i]
                var raw_lv = logits.data[lo + Self.DIM_ + i]
                var y = targets.data[to + i]
                var lv = raw_lv
                if lv > lv_max:
                    lv = lv_max
                elif lv < lv_min:
                    lv = lv_min
                var inv_var = exp(-lv)
                var d = mu - y
                total += Scalar[DT](0.5) * d * d * inv_var + Scalar[DT](
                    0.5
                ) * lv
        return total / Scalar[DT](B)

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
            c.enqueue_function[
                _gnll_fwd_kernel[
                    B, Self.DIM_, Self.LOGVAR_MIN, Self.LOGVAR_MAX
                ]
            ](
                logits.tt["gpu", B, 2 * Self.DIM_](),
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
            c.enqueue_function[
                _gnll_fwd_kernel[
                    B, Self.DIM_, Self.LOGVAR_MIN, Self.LOGVAR_MAX
                ]
            ](
                logits.tt["gpu", B, 2 * Self.DIM_](),
                targets.tt["gpu", B, Self.DIM_](),
                self.partial.tt["gpu", B](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
            c.enqueue_function[_gnll_reduce_kernel[B]](
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
        comptime M = B * 2 * Self.DIM_
        comptime if target == "cpu":
            grad.ensure(M)
            var lv_min = Scalar[DT](Self.LOGVAR_MIN)
            var lv_max = Scalar[DT](Self.LOGVAR_MAX)
            var inv_b = Scalar[DT](1.0) / Scalar[DT](B)
            for b in range(B):
                var lo = b * (2 * Self.DIM_)
                var to = b * Self.DIM_
                for i in range(Self.DIM_):
                    var mu = logits.data[lo + i]
                    var raw_lv = logits.data[lo + Self.DIM_ + i]
                    var y = targets.data[to + i]
                    var ic = Scalar[DT](1.0)
                    var lv = raw_lv
                    if lv > lv_max:
                        lv = lv_max
                        ic = Scalar[DT](0.0)
                    elif lv < lv_min:
                        lv = lv_min
                        ic = Scalar[DT](0.0)
                    var inv_v = exp(-lv)
                    var d = mu - y
                    # d_loss/d_µ = (µ-y) · σ⁻² / BATCH.
                    grad.data[lo + i] = d * inv_v * inv_b
                    # d_loss/d_raw_lv = (½ - ½·d²·σ⁻²) / BATCH · in_clamp.
                    grad.data[lo + Self.DIM_ + i] = (
                        Scalar[DT](0.5) - Scalar[DT](0.5) * d * d * inv_v
                    ) * inv_b * ic
        else:
            var c = ctx.value()
            grad.ensure_gpu(c, M)
            comptime total = B * Self.DIM_
            comptime nblk = (total + TPB - 1) // TPB
            c.enqueue_function[
                _gnll_bwd_kernel[
                    B, Self.DIM_, Self.LOGVAR_MIN, Self.LOGVAR_MAX
                ]
            ](
                logits.tt["gpu", B, 2 * Self.DIM_](),
                targets.tt["gpu", B, Self.DIM_](),
                grad.tt["gpu", B, 2 * Self.DIM_](),
                grid_dim=nblk,
                block_dim=TPB,
            )
