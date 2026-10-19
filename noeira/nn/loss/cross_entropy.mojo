"""CrossEntropyLoss[N_CLASSES] — softmax + cross-entropy (storage surface).

Transformed from legacy `nn.loss.CrossEntropyLoss`. Numerically stable
(max-shift + log-sum-exp). Mirrors MSELoss's device-resident accumulator
(forward_accumulate/read_accum/reset_accum). STORAGE simplification: vjp
RECOMPUTES the softmax from `logits` (passed explicitly) — no softmax cache.

  forward:  loss = (1/B)·Σ_b Σ_c -target[b,c]·(logit[b,c] - lse[b])
  backward: grad[b,c] = (softmax[b,c] - target[b,c]) / B
"""

from std.math import exp, log
from max.gpu import global_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from ..core.tensor import Tensor
from noeira.core.tiles import Tile


def _ce_fwd_kernel[
    BATCH: Int, NC: Int
](
    logits: Tile[DT, BATCH, NC],
    targets: Tile[DT, BATCH, NC],
    partial: Tile[DT, BATCH],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var m = rebind[Scalar[DT]](logits[b, 0])
        for c in range(1, NC):
            var v = rebind[Scalar[DT]](logits[b, c])
            if v > m:
                m = v
        var sum_exp: Scalar[DT] = 0.0
        for c in range(NC):
            sum_exp += exp(rebind[Scalar[DT]](logits[b, c]) - m)
        var lse = m + log(sum_exp)
        var s: Scalar[DT] = 0.0
        for c in range(NC):
            s += -rebind[Scalar[DT]](targets[b, c]) * (
                rebind[Scalar[DT]](logits[b, c]) - lse
            )
        partial[b] = s


def _ce_reduce_kernel[
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


def _ce_bwd_kernel[
    BATCH: Int, NC: Int
](
    logits: Tile[DT, BATCH, NC],
    targets: Tile[DT, BATCH, NC],
    grad: Tile[DT, BATCH, NC],
):
    # one thread per row: recompute softmax, then grad = (softmax - target)/B
    var b = Int(global_idx.x)
    if b < BATCH:
        var m = rebind[Scalar[DT]](logits[b, 0])
        for c in range(1, NC):
            var v = rebind[Scalar[DT]](logits[b, c])
            if v > m:
                m = v
        var sum_exp: Scalar[DT] = 0.0
        for c in range(NC):
            sum_exp += exp(rebind[Scalar[DT]](logits[b, c]) - m)
        var lse = m + log(sum_exp)
        var inv_b: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
        for c in range(NC):
            var sm = exp(rebind[Scalar[DT]](logits[b, c]) - lse)
            grad[b, c] = (sm - rebind[Scalar[DT]](targets[b, c])) * inv_b


struct CrossEntropyLoss[NC_: Int](Movable & Deinitable):
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
        var s: Scalar[DT] = 0.0
        for b in range(B):
            var m = logits.data[b * Self.NC_]
            for c in range(1, Self.NC_):
                var v = logits.data[b * Self.NC_ + c]
                if v > m:
                    m = v
            var sum_exp: Scalar[DT] = 0.0
            for c in range(Self.NC_):
                sum_exp += exp(logits.data[b * Self.NC_ + c] - m)
            var lse = m + log(sum_exp)
            for c in range(Self.NC_):
                s += -targets.data[b * Self.NC_ + c] * (
                    logits.data[b * Self.NC_ + c] - lse
                )
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
            c.enqueue_function[_ce_fwd_kernel[B, Self.NC_]](
                logits.tt["gpu", B, Self.NC_](),
                targets.tt["gpu", B, Self.NC_](),
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
            c.enqueue_function[_ce_fwd_kernel[B, Self.NC_]](
                logits.tt["gpu", B, Self.NC_](),
                targets.tt["gpu", B, Self.NC_](),
                self.partial.tt["gpu", B](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
            c.enqueue_function[_ce_reduce_kernel[B]](
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
        comptime M = B * Self.NC_
        comptime if target == "cpu":
            grad.ensure(M)
            for b in range(B):
                var m = logits.data[b * Self.NC_]
                for c in range(1, Self.NC_):
                    var v = logits.data[b * Self.NC_ + c]
                    if v > m:
                        m = v
                var sum_exp: Scalar[DT] = 0.0
                for c in range(Self.NC_):
                    sum_exp += exp(logits.data[b * Self.NC_ + c] - m)
                var lse = m + log(sum_exp)
                for c in range(Self.NC_):
                    var sm = exp(logits.data[b * Self.NC_ + c] - lse)
                    grad.data[b * Self.NC_ + c] = (
                        sm - targets.data[b * Self.NC_ + c]
                    ) / Scalar[DT](B)
        else:
            var c = ctx.value()
            grad.ensure_gpu(c, M)
            comptime nblk = (B + TPB_REDUCE - 1) // TPB_REDUCE
            c.enqueue_function[_ce_bwd_kernel[B, Self.NC_]](
                logits.tt["gpu", B, Self.NC_](),
                targets.tt["gpu", B, Self.NC_](),
                grad.tt["gpu", B, Self.NC_](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
