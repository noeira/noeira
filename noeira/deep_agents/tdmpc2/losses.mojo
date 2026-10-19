"""TD-MPC2 loss ops — storage ComputeGraph nodes (Module ABI).

Four ops the world-model / policy `ComputeGraph`s attach as nodes so the
framework routes their gradient back to the upstream predictions automatically:

  * `MSELossPlain[DIM]`  (latent consistency) — inputs (pred, target) → [B,1];
        loss = Σ_k (pred − target)² ; grad = 2·(pred − target) to `pred` only
        (target detached). NO symlog (unlike DreamerV3's `SymlogMSELoss`):
        TD-MPC2's consistency loss is a plain MSE in SimNorm-latent space.

  * `BCEWithLogitsLoss`  (termination head) — inputs (logit[B,1], target[B,1])
        → [B,1]; stable BCE-with-logits, target detached.

  * `TDMPC2TwoHotLoss[BINS, VMIN, VMAX]`  (reward + value) — inputs
        (logits[B,BINS], target[B,1]) → [B,1]; two-hot soft cross-entropy with
        **linear bins in symlog space** (`linspace(VMIN, VMAX, BINS)`) and the
        target symlog-compressed inside. Delegates to the reusable bin-agnostic
        two-hot CE in `nn.storage.loss.two_hot` (shared with DreamerV3).

  * `TwoHotDecode[BINS, VMIN, VMAX]`  (policy loss) — logits → scalar value
        (ARITY=1); value = symexp(Σ softmax·bins), differentiable w.r.t. logits.
        Delegates to the reusable decode fwd/bwd in `nn.storage.loss.two_hot`.

STORAGE migration: the legacy `TensorPack`/`TileTensor`/`mptr`/`TargetStorage`
ABI is gone — these are plain storage `Module`s (forward over `TensorRefs` →
`Tensor`, vjp recomputed from `forward_input`, no cached pointers). The two-hot
math now lives once in `loss/two_hot.mojo`; only MSE / BCE keep private kernels
here. No trainable params → inherit the no-op `for_each_param`/`zero_grad`.
"""

from std.math import exp, log
from max.gpu import global_idx

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP
from max.gpu.host import DeviceContext

from noeira.nn.loss.two_hot import (
    fill_bins,
    decode_value_batch,
    two_hot_decode_batch,
    two_hot_ce_loss_batch,
    two_hot_ce_backward_batch,
    decode_value_backward_batch,
    two_hot_ce_fwd_kernel,
    two_hot_ce_bwd_kernel,
    decode_value_fwd_kernel,
    decode_value_bwd_kernel,
)
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# MSELossPlain[DIM] — latent consistency. inputs (pred[B,DIM], target[B,DIM]).
# loss[b] = Σ_k (pred-tgt)² ; grad_pred = 2·(pred-tgt)·go ; grad_tgt = 0.
# ──────────────────────────────────────────────────────────────────────


def _mse_fwd_kernel[B: Int, DIM: Int](
    pred: Tile[DT, B * DIM],
    tgt: Tile[DT, B * DIM],
    o: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        var s: Scalar[DT] = 0.0
        for k in range(DIM):
            var d = rebind[Scalar[DT]](pred[b * DIM + k]) - rebind[Scalar[DT]](
                tgt[b * DIM + k]
            )
            s += d * d
        o[b] = s


def _mse_bwd_kernel[B: Int, DIM: Int](
    go: Tile[DT, B],
    pred: Tile[DT, B * DIM],
    tgt: Tile[DT, B * DIM],
    gp: Tile[DT, B * DIM],
    gt: Tile[DT, B * DIM],
):
    var b = Int(global_idx.x)
    if b < B:
        var up = rebind[Scalar[DT]](go[b])
        for k in range(DIM):
            var idx = b * DIM + k
            gp[idx] = up * Scalar[DT](2.0) * (
                rebind[Scalar[DT]](pred[idx]) - rebind[Scalar[DT]](tgt[idx])
            )
            gt[idx] = 0.0


struct MSELossPlain[DIM: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.DIM)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var s: Scalar[DT] = 0.0
                for k in range(Self.DIM):
                    var d = pred.data[b * Self.DIM + k] - tgt.data[
                        b * Self.DIM + k
                    ]
                    s += d * d
                out.data[b] = s
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_mse_fwd_kernel[B, Self.DIM]](
                pred.tt["gpu", B * Self.DIM](),
                tgt.tt["gpu", B * Self.DIM](),
                out.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = forward_input[0]
        ref tgt = forward_input[1]
        ref gp = grad_inputs[0]
        ref gt = grad_inputs[1]
        comptime if target == "cpu":
            gp.ensure(B * Self.DIM)
            gt.ensure(B * Self.DIM)
            for b in range(B):
                var up = grad_output.data[b]
                for k in range(Self.DIM):
                    var idx = b * Self.DIM + k
                    gp.data[idx] = up * Scalar[DT](2.0) * (
                        pred.data[idx] - tgt.data[idx]
                    )
                    gt.data[idx] = 0.0
        else:
            var c = ctx.value()
            gp.ensure_gpu(c, B * Self.DIM)
            gt.ensure_gpu(c, B * Self.DIM)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_mse_bwd_kernel[B, Self.DIM]](
                grad_output.tt["gpu", B](),
                pred.tt["gpu", B * Self.DIM](),
                tgt.tt["gpu", B * Self.DIM](),
                gp.tt["gpu", B * Self.DIM](),
                gt.tt["gpu", B * Self.DIM](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# BCEWithLogitsLoss — termination head. inputs (logit[B,1], target[B,1]).
#   loss = max(x,0) − x·y + log(1 + exp(−|x|));   d/dx = sigmoid(x) − y
# ──────────────────────────────────────────────────────────────────────


@always_inline
def _bce_with_logits(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
    var ax = x if x >= Scalar[DT](0.0) else -x
    var mx = x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)
    return mx - x * y + log(Scalar[DT](1.0) + exp(-ax))


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


def _bce_fwd_kernel[B: Int](
    lg: Tile[DT, B],
    tgt: Tile[DT, B],
    o: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        o[b] = _bce_with_logits(
            rebind[Scalar[DT]](lg[b]), rebind[Scalar[DT]](tgt[b])
        )


def _bce_bwd_kernel[B: Int](
    go: Tile[DT, B],
    lg: Tile[DT, B],
    tgt: Tile[DT, B],
    gl: Tile[DT, B],
    gt: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        gl[b] = rebind[Scalar[DT]](go[b]) * (
            _sigmoid(rebind[Scalar[DT]](lg[b])) - rebind[Scalar[DT]](tgt[b])
        )
        gt[b] = 0.0


struct BCEWithLogitsLoss(Module):
    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=1)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref lg = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                out.data[b] = _bce_with_logits(lg.data[b], tgt.data[b])
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_bce_fwd_kernel[B]](
                lg.tt["gpu", B](),
                tgt.tt["gpu", B](),
                out.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref lg = forward_input[0]
        ref tgt = forward_input[1]
        ref gl = grad_inputs[0]
        ref gt = grad_inputs[1]
        comptime if target == "cpu":
            gl.ensure(B)
            gt.ensure(B)
            for b in range(B):
                gl.data[b] = grad_output.data[b] * (
                    _sigmoid(lg.data[b]) - tgt.data[b]
                )
                gt.data[b] = 0.0
        else:
            var c = ctx.value()
            gl.ensure_gpu(c, B)
            gt.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_bce_bwd_kernel[B]](
                grad_output.tt["gpu", B](),
                lg.tt["gpu", B](),
                tgt.tt["gpu", B](),
                gl.tt["gpu", B](),
                gt.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# TDMPC2TwoHotLoss[BINS, VMIN, VMAX] — reward + value heads.
# inputs (logits[B,BINS], target[B,1]); linear bins in [VMIN,VMAX] (symlog
# space); target symlog'd inside. Delegates to the reusable two-hot CE.
# ──────────────────────────────────────────────────────────────────────


struct TDMPC2TwoHotLoss[BINS: Int, VMIN: Int, VMAX: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = 1

    @staticmethod
    def _mk_in_dims() -> Array[Int, 2]:
        var d = Array[Int, 2](fill=1)
        d[0] = Self.BINS
        return d^

    var bins: Tensor  # [BINS] linspace(VMIN, VMAX); host + device

    def __init__(out self):
        self.bins = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TDMPC2TwoHotLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bins = Tensor.alloc(Self.BINS)
        fill_bins[Self.BINS](Scalar[DT](Self.VMIN), Scalar[DT](Self.VMAX), m.bins)
        comptime if target == "gpu":
            if not ctx:
                raise Error("TDMPC2TwoHotLoss.make[gpu]: ctx required")
            m.bins.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_ce_loss_batch[B, Self.BINS, True](inputs, self.bins, out)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[two_hot_ce_fwd_kernel[B, Self.BINS, True]](
                inputs[0].tt["gpu", B * Self.BINS](),
                inputs[1].tt["gpu", B](),
                self.bins.tt["gpu", Self.BINS](),
                out.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_ce_backward_batch[B, Self.BINS, True](
                forward_input, self.bins, grad_output, grad_inputs
            )
        else:
            var c = ctx.value()
            ref g_lg = grad_inputs[0]
            ref g_tgt = grad_inputs[1]
            g_lg.ensure_gpu(c, B * Self.BINS)
            g_tgt.ensure_gpu(c, B)
            g_tgt.dev.value().enqueue_fill(Scalar[DT](0))
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[two_hot_ce_bwd_kernel[B, Self.BINS, True]](
                grad_output.tt["gpu", B](),
                forward_input[0].tt["gpu", B * Self.BINS](),
                forward_input[1].tt["gpu", B](),
                self.bins.tt["gpu", Self.BINS](),
                g_lg.tt["gpu", B * Self.BINS](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# TwoHotDecode[BINS, VMIN, VMAX] — logits → scalar value (ARITY=1).
# value = symexp(Σ softmax·bins), linear bins in [VMIN,VMAX]. Differentiable
# w.r.t. logits. Delegates to the reusable decode fwd/bwd.
# ──────────────────────────────────────────────────────────────────────


struct TwoHotDecode[BINS: Int, VMIN: Int, VMAX: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.BINS)
    comptime OUT_DIM = 1

    var bins: Tensor  # [BINS] linspace(VMIN, VMAX); host + device

    def __init__(out self):
        self.bins = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TwoHotDecode: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bins = Tensor.alloc(Self.BINS)
        fill_bins[Self.BINS](Scalar[DT](Self.VMIN), Scalar[DT](Self.VMAX), m.bins)
        comptime if target == "gpu":
            if not ctx:
                raise Error("TwoHotDecode.make[gpu]: ctx required")
            m.bins.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_decode_batch[B, Self.BINS](inputs, self.bins, out)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[decode_value_fwd_kernel[B, Self.BINS]](
                inputs[0].tt["gpu", B * Self.BINS](),
                self.bins.tt["gpu", Self.BINS](),
                out.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            decode_value_backward_batch[B, Self.BINS](
                forward_input, self.bins, grad_output, grad_inputs
            )
        else:
            var c = ctx.value()
            ref g_lg = grad_inputs[0]
            g_lg.ensure_gpu(c, B * Self.BINS)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[decode_value_bwd_kernel[B, Self.BINS]](
                grad_output.tt["gpu", B](),
                forward_input[0].tt["gpu", B * Self.BINS](),
                self.bins.tt["gpu", Self.BINS](),
                g_lg.tt["gpu", B * Self.BINS](),
                grid_dim=nb, block_dim=TPB,
            )
