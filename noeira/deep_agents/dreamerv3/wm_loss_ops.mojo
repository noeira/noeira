"""World-model loss ops — storage ComputeGraph nodes (Module ABI).

The WM-loss `ComputeGraph` needs the recon / reward / cont losses to attach
as graph nodes so the framework routes their gradient to the upstream
logits/preds automatically (the same role `OneHotKLLoss` plays for dyn/rep).
Each op:

  * `SymlogMSELoss[OBS]`  (recon) — inputs (pred, target) → [B,1];
        loss = Σ_o (pred − symlog(target))² ; grad = 2·(pred − symlog(t))
        to `pred` only (target detached).
  * `TwoHotLoss[BINS]`    (reward) — inputs (logits, target) → [B,1];
        twohot cross-entropy against symexp-twohot bins (value space, so the
        raw target is bracketed directly — NO symlog). Delegates to the
        reusable bin-agnostic two-hot CE in `nn.storage.loss.two_hot`
        (shared with TD-MPC2), called with `SYMLOG=False`; bins owned by the
        op (symexp grid from `make`).
  * `BinaryLoss`          (cont)  — inputs (logit[1], target[1]) → [B,1];
        loss = softplus(x) − t·x ; grad = sigmoid(x) − target.

STORAGE migration: the legacy `TensorPack`/`TileTensor`/`mptr`/`TargetStorage`
ABI is gone — these are plain storage `Module`s (forward over `TensorRefs` →
`Tensor`, vjp recomputed from `forward_input`, no cached pointers). The two-hot
math now lives once in `loss/two_hot.mojo`; only SymlogMSE / Binary keep private
kernels here. No trainable params → inherit the no-op `for_each_param`/
`zero_grad`.
"""

from std.math import exp, log, log1p
from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP

from noeira.nn.loss.two_hot import (
    two_hot_ce_loss_batch,
    two_hot_ce_backward_batch,
    two_hot_ce_fwd_kernel,
    two_hot_ce_bwd_kernel,
)
from .twohot import symexp_twohot_bins, DREAMER_REWARD_GRID_LO
from noeira.core.tiles import Tile


@always_inline
def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    """`sign(x) * log(1 + |x|)` — used by the CPU path AND the GPU kernels.

    ⚠⚠ THESE WERE TWO DIFFERENT FUNCTIONS NINE LINES APART, AND THE CPU AND GPU
    PATHS OF THE SAME LOSS USED ONE EACH. `_symlog` was `s * log1p(a)`
    (`SymlogMSELoss.forward` / `.vjp`) and `_symk` was `s * log(1 + a)`
    (`_symmse_fwd_kernel` / `_symmse_bwd_kernel`). They are the same function
    mathematically and NOT the same function numerically: `std.math.log1p`
    carries up to 1.01e-06 RELATIVE error for |x| in [0.05, 0.42] — rising
    smoothly to a peak at ~0.404 and collapsing by ~0.424, the signature of a
    branch cutover set too high — where libm is 1e-16.

    So a recon target in that band was symlogged differently on the two paths,
    and `test_wm_loss_ops_storage.test_symmse` compares them at a tolerance of
    1e-4, which is 100x too loose to notice. It read as GPU numerics.

    ⚠ THE SURVIVING SPELLING IS `log(1 + a)`, THE GPU ONE, ON PURPOSE — it
    leaves the GPU kernels bit-identical and moves only the CPU path, by at
    most ~1e-06 relative, toward it. Making BOTH accurate (the
    `2*atanh(a/(2+a))` identity, 2.8e-14 where these are 1e-09 and 1e-06) is a
    real improvement and a SEPARATE change: it moves every DreamerV3 number
    and needs its own before/after, exactly as the dm_control touch sensors got
    in `dm_control/dtype_math.log1p_dt`.

    ⚠ `noeira/nn/primitives/ops/symlog_math.mojo` calls itself the "single
    source of truth (audit L4)" for this formula and this file does not use it,
    which is how the drift happened in the first place. Routing through it is
    the right follow-up; it is a cross-package import, not a one-line change.
    """
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log(Scalar[DT](1.0) + a)


@always_inline
def _symk(x: Scalar[DT]) -> Scalar[DT]:
    """Kept as the GPU kernels' name for `_symlog`, which they now share."""
    return _symlog(x)


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ──────────────────────────────────────────────────────────────────────
# SymlogMSELoss[OBS] — recon head. inputs (pred[B,OBS], target[B,OBS]).
# loss[b] = Σ_k (pred − symlog(tgt))² ; grad_pred = 2·(pred − symlog(tgt))·go ;
# grad_tgt = 0 (target detached).
# ──────────────────────────────────────────────────────────────────────


def _symmse_fwd_kernel[B: Int, OBS: Int, SIGMOID: Bool](
    pred: Tile[DT, B * OBS],
    tgt: Tile[DT, B * OBS],
    o: Tile[DT, B],
):
    # ONE BLOCK per batch element; threads stride over the OBS pixels and the
    # block reduces. (Was grid=(B/TPB) → only B threads active, each looping all
    # OBS=C*H*W pixels serially — catastrophic for pixel obs. This parallelizes
    # the per-b reduction across the whole block.)
    # SIGMOID=True → reference recon: loss = (sigmoid(pred) − tgt)², tgt raw
    # [0,1]. False → symlog recon: loss = (pred − symlog(tgt))².
    var b = Int(block_idx.x)
    if b < B:
        var t = Int(thread_idx.x)
        var s: Scalar[DT] = 0.0
        var k = t
        while k < OBS:
            var p = rebind[Scalar[DT]](pred[b * OBS + k])
            var tg = rebind[Scalar[DT]](tgt[b * OBS + k])
            var d: Scalar[DT]
            comptime if SIGMOID:
                d = _sigmoid(p) - tg
            else:
                d = p - _symk(tg)
            s += d * d
            k += TPB_REDUCE
        var tot = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s)
        if t == 0:
            o[b] = tot[0]


def _symmse_bwd_kernel[B: Int, OBS: Int, SIGMOID: Bool](
    go: Tile[DT, B],
    pred: Tile[DT, B * OBS],
    tgt: Tile[DT, B * OBS],
    gp: Tile[DT, B * OBS],
    gt: Tile[DT, B * OBS],
):
    # Pure elementwise over ALL B*OBS — one thread per element (grid-strided),
    # not one thread per batch row. (Was grid=(B/TPB) → B threads each looping
    # OBS serially.)
    var i = Int(global_idx.x)
    if i < B * OBS:
        var up = rebind[Scalar[DT]](go[i // OBS])
        var p = rebind[Scalar[DT]](pred[i])
        var tg = rebind[Scalar[DT]](tgt[i])
        comptime if SIGMOID:
            var sg = _sigmoid(p)
            gp[i] = up * Scalar[DT](2.0) * (sg - tg) * sg * (Scalar[DT](1.0) - sg)
        else:
            gp[i] = up * Scalar[DT](2.0) * (p - _symk(tg))
        gt[i] = 0.0


struct SymlogMSELoss[OBS: Int, SIGMOID: Bool = False](Module):
    # SIGMOID=False (default): symlog recon — loss = (pred − symlog(tgt))² (the
    # right choice for unbounded vector obs). SIGMOID=True: reference pixel recon
    # — loss = (sigmoid(pred) − tgt)², tgt raw [0,1] (decode = sigmoid(pred)).
    comptime ARITY: Int = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.OBS)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("SigmoidMSE") if Self.SIGMOID else String("SymlogMSE")

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
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var s: Scalar[DT] = 0.0
                for k in range(Self.OBS):
                    var p = pred.data[b * Self.OBS + k]
                    var tg = tgt.data[b * Self.OBS + k]
                    var d: Scalar[DT]
                    comptime if Self.SIGMOID:
                        d = _sigmoid(p) - tg
                    else:
                        d = p - _symlog(tg)
                    s += d * d
                out.data[b] = s
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            # one block per batch element (block-reduce over OBS).
            c.enqueue_function[_symmse_fwd_kernel[B, Self.OBS, Self.SIGMOID]](
                pred.tt["gpu", B * Self.OBS](),
                tgt.tt["gpu", B * Self.OBS](),
                out.tt["gpu", B](),
                grid_dim=B, block_dim=TPB_REDUCE,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = forward_input[0]
        ref tgt = forward_input[1]
        ref gp = grad_inputs[0]
        ref gt = grad_inputs[1]
        comptime if target == "cpu":
            gp.ensure(B * Self.OBS)
            gt.ensure(B * Self.OBS)
            for b in range(B):
                var up = grad_output.data[b]
                for k in range(Self.OBS):
                    var idx = b * Self.OBS + k
                    var p = pred.data[idx]
                    var tg = tgt.data[idx]
                    comptime if Self.SIGMOID:
                        var sg = _sigmoid(p)
                        gp.data[idx] = up * Scalar[DT](2.0) * (sg - tg) * sg * (
                            Scalar[DT](1.0) - sg
                        )
                    else:
                        gp.data[idx] = up * Scalar[DT](2.0) * (p - _symlog(tg))
                    gt.data[idx] = 0.0
        else:
            var c = ctx.value()
            gp.ensure_gpu(c, B * Self.OBS)
            gt.ensure_gpu(c, B * Self.OBS)
            # one thread per (b, pixel) element across all B*OBS.
            comptime nb = (B * Self.OBS + TPB - 1) // TPB
            c.enqueue_function[_symmse_bwd_kernel[B, Self.OBS, Self.SIGMOID]](
                grad_output.tt["gpu", B](),
                pred.tt["gpu", B * Self.OBS](),
                tgt.tt["gpu", B * Self.OBS](),
                gp.tt["gpu", B * Self.OBS](),
                gt.tt["gpu", B * Self.OBS](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# TwoHotLoss[BINS] — reward head. inputs (logits[B,BINS], target[B,1]).
# Bins owned by the op (symexp_twohot grid; value space). The two-hot CE math
# is delegated to the reusable `two_hot_ce_*` helpers with SYMLOG=False (the
# bins already live in value space, so the raw target is bracketed directly —
# matching the legacy `twohot_loss`/`twohot_loss_backward`).
# ──────────────────────────────────────────────────────────────────────


struct TwoHotLoss[BINS: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("TwoHot")

    @staticmethod
    def _mk_in_dims() -> Array[Int, 2]:
        var d = Array[Int, 2](fill=1)
        d[0] = Self.BINS
        return d^

    var bins: Tensor  # [BINS] symexp_twohot grid; host + device

    def __init__(out self):
        self.bins = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TwoHotLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bins = Tensor.alloc(Self.BINS)
        # This grid MUST match the grid the reward is read back on in
        # imagination / imag_loss (DreamerV3Trainer.bins). Both now read the
        # SAME `DREAMER_REWARD_GRID_LO` constant (S4) so they can't diverge — a
        # past -9-vs-(-20)-default split made the head learn the right bin INDEX
        # but decode it on the wrong value grid → predictions ~5× off, poisoning
        # imagined returns. The narrow grid also keeps bin values bounded
        # (≈8102) so `Σ softmax·bins` stays CPU↔GPU bit-stable.
        symexp_twohot_bins[Self.BINS](
            m.bins.data.unsafe_ptr(),
            lo=Scalar[DT](DREAMER_REWARD_GRID_LO),
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("TwoHotLoss.make[gpu]: ctx required")
            m.bins.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_ce_loss_batch[B, Self.BINS, False](inputs, self.bins, out)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[two_hot_ce_fwd_kernel[B, Self.BINS, False]](
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
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_ce_backward_batch[B, Self.BINS, False](
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
            c.enqueue_function[two_hot_ce_bwd_kernel[B, Self.BINS, False]](
                grad_output.tt["gpu", B](),
                forward_input[0].tt["gpu", B * Self.BINS](),
                forward_input[1].tt["gpu", B](),
                self.bins.tt["gpu", Self.BINS](),
                g_lg.tt["gpu", B * Self.BINS](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# BinaryLoss — cont head. inputs (logit[B,1], target[B,1]).
# loss = softplus(x) − t·x ; grad = sigmoid(x) − target.
# ──────────────────────────────────────────────────────────────────────


def _binary_fwd_kernel[B: Int](
    lo: Tile[DT, B],
    tg: Tile[DT, B],
    o: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        var x = rebind[Scalar[DT]](lo[b])
        var ax = x if x >= Scalar[DT](0.0) else -x
        var sp = (x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)) + log(
            Scalar[DT](1.0) + exp(-ax)
        )
        o[b] = sp - rebind[Scalar[DT]](tg[b]) * x


def _binary_bwd_kernel[B: Int](
    go: Tile[DT, B],
    lo: Tile[DT, B],
    tg: Tile[DT, B],
    gl: Tile[DT, B],
    gt: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        var x = rebind[Scalar[DT]](lo[b])
        var sig = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
        gl[b] = rebind[Scalar[DT]](go[b]) * (sig - rebind[Scalar[DT]](tg[b]))
        gt[b] = 0.0


struct BinaryLoss(Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Array[Int, 2](fill=1)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("Binary")

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
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref lo = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var x = lo.data[b]
                # softplus stable: max(x,0) + log(1+exp(-|x|))
                var ax = x if x >= Scalar[DT](0.0) else -x
                var sp = (x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)) + log(
                    Scalar[DT](1.0) + exp(-ax)
                )
                out.data[b] = sp - tgt.data[b] * x
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_binary_fwd_kernel[B]](
                lo.tt["gpu", B](),
                tgt.tt["gpu", B](),
                out.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref lo = forward_input[0]
        ref tgt = forward_input[1]
        ref gl = grad_inputs[0]
        ref gt = grad_inputs[1]
        comptime if target == "cpu":
            gl.ensure(B)
            gt.ensure(B)
            for b in range(B):
                gl.data[b] = grad_output.data[b] * (
                    _sigmoid(lo.data[b]) - tgt.data[b]
                )
                gt.data[b] = 0.0
        else:
            var c = ctx.value()
            gl.ensure_gpu(c, B)
            gt.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_binary_bwd_kernel[B]](
                grad_output.tt["gpu", B](),
                lo.tt["gpu", B](),
                tgt.tt["gpu", B](),
                gl.tt["gpu", B](),
                gt.tt["gpu", B](),
                grid_dim=nb, block_dim=TPB,
            )
