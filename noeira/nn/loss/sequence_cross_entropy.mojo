"""SequenceCrossEntropyLoss[SEQ_LEN, VOCAB] — per-token softmax + CE (storage).

Transformed from legacy `nn.loss.SequenceCrossEntropyLoss`. For LM/sequence
heads whose output is `SEQ_LEN * VOCAB` logits per sample (e.g. nn `GPT`):
treats the `(BATCH, SEQ_LEN*VOCAB)` slab as `(BATCH*SEQ_LEN, VOCAB)` and
applies an independent softmax + cross-entropy at each token position,
averaging over all `BATCH*SEQ_LEN` positions.

Reuses the storage `CrossEntropyLoss` kernels (`_ce_fwd_kernel`,
`_ce_reduce_kernel`, `_ce_bwd_kernel`) with effective batch `BT=BATCH*SEQ_LEN`.
Like CrossEntropyLoss, `vjp` RECOMPUTES the softmax from `logits` (passed
explicitly) — no softmax cache. Mirrors the device-resident accumulator
(forward_accumulate/read_accum/reset_accum).

  forward:  loss = (1/BT)·Σ_r Σ_c -target[r,c]·(logit[r,c] - lse[r])
  backward: grad[r,c] = (softmax[r,c] - target[r,c]) / BT
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from std.math import exp, log
from ..core.tensor import Tensor
from .cross_entropy import _ce_fwd_kernel, _ce_reduce_kernel, _ce_bwd_kernel


struct SequenceCrossEntropyLoss[SEQ_LEN: Int, VOCAB: Int](
    Movable & Deinitable
):
    var partial: Tensor  # GPU [BT] per-row losses (lazy)
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
        BT: Int
    ](self, ref logits: Tensor, ref targets: Tensor) -> Scalar[DT]:
        var s: Scalar[DT] = 0.0
        for r in range(BT):
            var base = r * Self.VOCAB
            var m = logits.data[base]
            for c in range(1, Self.VOCAB):
                var v = logits.data[base + c]
                if v > m:
                    m = v
            var sum_exp: Scalar[DT] = 0.0
            for c in range(Self.VOCAB):
                sum_exp += exp(logits.data[base + c] - m)
            var lse = m + log(sum_exp)
            for c in range(Self.VOCAB):
                s += -targets.data[base + c] * (logits.data[base + c] - lse)
        return s / Scalar[DT](BT)

    def forward[
        target: StaticString, B: Int
    ](
        mut self,
        mut logits: Tensor,
        mut targets: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime BT = B * Self.SEQ_LEN
        comptime if target == "cpu":
            return self._mean_loss_cpu[BT](logits, targets)
        else:
            var c = ctx.value()
            self.partial.ensure_gpu(c, BT)
            comptime nblk = (BT + TPB_REDUCE - 1) // TPB_REDUCE
            c.enqueue_function[_ce_fwd_kernel[BT, Self.VOCAB]](
                logits.tt["gpu", BT, Self.VOCAB](),
                targets.tt["gpu", BT, Self.VOCAB](),
                self.partial.tt["gpu", BT](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
            self.partial.download(c)
            var s: Scalar[DT] = 0.0
            for r in range(BT):
                s += self.partial.data[r]
            return s / Scalar[DT](BT)

    def forward_accumulate[
        target: StaticString, B: Int
    ](
        mut self,
        mut logits: Tensor,
        mut targets: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime BT = B * Self.SEQ_LEN
        comptime if target == "cpu":
            self._acc_sum += self._mean_loss_cpu[BT](logits, targets)
            self._acc_n += 1
        else:
            var c = ctx.value()
            self.partial.ensure_gpu(c, BT)
            comptime nblk = (BT + TPB_REDUCE - 1) // TPB_REDUCE
            c.enqueue_function[_ce_fwd_kernel[BT, Self.VOCAB]](
                logits.tt["gpu", BT, Self.VOCAB](),
                targets.tt["gpu", BT, Self.VOCAB](),
                self.partial.tt["gpu", BT](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
            c.enqueue_function[_ce_reduce_kernel[BT]](
                self.partial.tt["gpu", BT](),
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
        comptime BT = B * Self.SEQ_LEN
        comptime M = BT * Self.VOCAB
        comptime if target == "cpu":
            grad.ensure(M)
            for r in range(BT):
                var base = r * Self.VOCAB
                var m = logits.data[base]
                for c in range(1, Self.VOCAB):
                    var v = logits.data[base + c]
                    if v > m:
                        m = v
                var sum_exp: Scalar[DT] = 0.0
                for c in range(Self.VOCAB):
                    sum_exp += exp(logits.data[base + c] - m)
                var lse = m + log(sum_exp)
                for c in range(Self.VOCAB):
                    var sm = exp(logits.data[base + c] - lse)
                    grad.data[base + c] = (
                        sm - targets.data[base + c]
                    ) / Scalar[DT](BT)
        else:
            var c = ctx.value()
            grad.ensure_gpu(c, M)
            comptime nblk = (BT + TPB_REDUCE - 1) // TPB_REDUCE
            c.enqueue_function[_ce_bwd_kernel[BT, Self.VOCAB]](
                logits.tt["gpu", BT, Self.VOCAB](),
                targets.tt["gpu", BT, Self.VOCAB](),
                grad.tt["gpu", BT, Self.VOCAB](),
                grid_dim=nblk,
                block_dim=TPB_REDUCE,
            )
