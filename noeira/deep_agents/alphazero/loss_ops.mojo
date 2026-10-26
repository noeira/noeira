"""AlphaZero loss op — graph-Module wrapper (ARITY=2) for the AZ training loss.

One node captures the whole AlphaZero objective so the ComputeGraph routes its
gradient straight back into the prediction net's logits/value:

    inputs (pred[B, ACT+1], target[B, ACT+1]) → loss[B, 1]

where ``pred = [policy_logits(ACT) | raw_value(1)]`` (the net output) and
``target = [mcts_policy(ACT) | z(1)]`` (the packed self-play target). The loss is

    loss_b = −Σ_a π_a · log_softmax(logits)_a   +   (tanh(raw_value) − z)²

i.e. soft cross-entropy of the policy against the MCTS visit-count distribution
plus value MSE on the **tanh-squashed** value head (AlphaZero value ∈ [-1,1]).
The value squash lives here (not in the net) so the same raw value head feeds
the MCTS expand kernel's ``VALUE_SQUASH``.

Gradients (target detached, ``grad_target = 0``):
  * policy: ``grad_logits_a = up · (softmax(logits)_a − π_a)``  (Σπ = 1)
  * value:  ``grad_raw = up · 2·(tanh(raw) − z)·(1 − tanh(raw)²)``

No trainable params (inherits the no-op param walkers). Storage surface: `vjp`
receives `forward_input` (no cached input pointers), `ctx` is a method arg (no
`TargetStorage`); CPU indexes `.data`, GPU builds device views via `.lt`.
"""

from std.math import exp, log, tanh
from max.gpu.host import DeviceContext
from max.gpu import global_idx

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


# ── GPU kernels (one thread per batch row). ───────────────────────────────
def _az_loss_fwd_kernel[B: Int, ACT: Int](
    pred: Tile[DT, B * (ACT + 1)],
    tgt: Tile[DT, B * (ACT + 1)],
    o: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * (ACT + 1)
        var zmax = rebind[Scalar[DT]](pred[base])
        for c in range(1, ACT):
            var v = rebind[Scalar[DT]](pred[base + c])
            if v > zmax:
                zmax = v
        var ssum: Scalar[DT] = 0.0
        for c in range(ACT):
            ssum += exp(rebind[Scalar[DT]](pred[base + c]) - zmax)
        var lse = zmax + log(ssum)
        var ce: Scalar[DT] = 0.0
        for c in range(ACT):
            ce += rebind[Scalar[DT]](tgt[base + c]) * (
                rebind[Scalar[DT]](pred[base + c]) - lse
            )
        var tv = tanh(rebind[Scalar[DT]](pred[base + ACT]))
        var d = tv - rebind[Scalar[DT]](tgt[base + ACT])
        o[b] = -ce + d * d


def _az_loss_bwd_kernel[B: Int, ACT: Int](
    go: Tile[DT, B],
    pred: Tile[DT, B * (ACT + 1)],
    tgt: Tile[DT, B * (ACT + 1)],
    gp: Tile[DT, B * (ACT + 1)],
    gt: Tile[DT, B * (ACT + 1)],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * (ACT + 1)
        var up = rebind[Scalar[DT]](go[b])
        var zmax = rebind[Scalar[DT]](pred[base])
        for c in range(1, ACT):
            var v = rebind[Scalar[DT]](pred[base + c])
            if v > zmax:
                zmax = v
        var ssum: Scalar[DT] = 0.0
        for c in range(ACT):
            ssum += exp(rebind[Scalar[DT]](pred[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        for c in range(ACT):
            var sm = exp(rebind[Scalar[DT]](pred[base + c]) - zmax) * inv
            gp[base + c] = up * (sm - rebind[Scalar[DT]](tgt[base + c]))
            gt[base + c] = 0.0
        var tv = tanh(rebind[Scalar[DT]](pred[base + ACT]))
        var d = tv - rebind[Scalar[DT]](tgt[base + ACT])
        gp[base + ACT] = up * Scalar[DT](2.0) * d * (Scalar[DT](1.0) - tv * tv)
        gt[base + ACT] = 0.0


struct AZLossOp[ACT: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.ACT + 1)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "AZLossOp: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime W = Self.ACT + 1
        ref pred = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var base = b * W
                var zmax = pred.data[base]
                for c in range(1, Self.ACT):
                    if pred.data[base + c] > zmax:
                        zmax = pred.data[base + c]
                var ssum: Scalar[DT] = 0.0
                for c in range(Self.ACT):
                    ssum += exp(pred.data[base + c] - zmax)
                var lse = zmax + log(ssum)
                var ce: Scalar[DT] = 0.0
                for c in range(Self.ACT):
                    ce += tgt.data[base + c] * (pred.data[base + c] - lse)
                var tv = tanh(pred.data[base + Self.ACT])
                var d = tv - tgt.data[base + Self.ACT]
                out.data[b] = -ce + d * d
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            var predl = pred.tt["gpu", B * W]()
            var tgtl = tgt.tt["gpu", B * W]()
            var ol = out.tt["gpu", B]()
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_az_loss_fwd_kernel[B, Self.ACT]](
                predl, tgtl, ol, grid_dim=nb, block_dim=TPB
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime W = Self.ACT + 1
        ref pred = forward_input[0]
        ref tgt = forward_input[1]
        ref g_pred = grad_inputs[0]
        ref g_tgt = grad_inputs[1]
        comptime if target == "cpu":
            g_pred.ensure(B * W)
            g_tgt.ensure(B * W)
            for b in range(B):
                var base = b * W
                var up = grad_output.data[b]
                var zmax = pred.data[base]
                for c in range(1, Self.ACT):
                    if pred.data[base + c] > zmax:
                        zmax = pred.data[base + c]
                var ssum: Scalar[DT] = 0.0
                for c in range(Self.ACT):
                    ssum += exp(pred.data[base + c] - zmax)
                var inv = Scalar[DT](1.0) / ssum
                for c in range(Self.ACT):
                    var sm = exp(pred.data[base + c] - zmax) * inv
                    g_pred.data[base + c] = up * (sm - tgt.data[base + c])
                    g_tgt.data[base + c] = 0.0
                var tv = tanh(pred.data[base + Self.ACT])
                var d = tv - tgt.data[base + Self.ACT]
                g_pred.data[base + Self.ACT] = (
                    up * Scalar[DT](2.0) * d * (Scalar[DT](1.0) - tv * tv)
                )
                g_tgt.data[base + Self.ACT] = 0.0
        else:
            var c = ctx.value()
            g_pred.ensure_gpu(c, B * W)
            g_tgt.ensure_gpu(c, B * W)
            var gol = grad_output.tt["gpu", B]()
            var predl = pred.tt["gpu", B * W]()
            var tgtl = tgt.tt["gpu", B * W]()
            var gpl = g_pred.tt["gpu", B * W]()
            var gtl = g_tgt.tt["gpu", B * W]()
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_az_loss_bwd_kernel[B, Self.ACT]](
                gol, predl, tgtl, gpl, gtl, grid_dim=nb, block_dim=TPB
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
