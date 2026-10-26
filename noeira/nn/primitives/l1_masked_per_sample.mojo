"""L1MaskedPerSample[K, D] — ACT's reconstruction loss, per sample.

    inputs   pred   [BATCH, K*D]
             target [BATCH, K*D]
             valid  [BATCH, K]        1.0 = a real step, 0.0 = padding
    output          [BATCH, 1]

    out[b, 0] = (1 / (K*D)) * sum_{t,j} |pred - target| * valid[b, t]

ARITY=3, no params. `policy.py:31`:

    all_l1 = F.l1_loss(actions, a_hat, reduction='none')
    l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

⚠⚠ **The denominator is `K*D`, not the number of VALID entries.** `.mean()` runs
over the whole tensor; masked positions contribute 0 to the numerator and still
count in the denominator. So a chunk that is 90% padding produces a loss ~10x
smaller than the same errors on a full chunk. That reads like a bug and is not
one to "fix": it down-weights end-of-episode samples, and matching it is the
difference between reproducing ACT and reproducing something near it. Dividing
by the valid count instead would change the effective learning rate per sample
in a way no shape or gate would catch.

The mask multiplies |pred - target|, so it is a gate on the gradient too:

    d/dpred[b,t,j]   =  grad_out[b] * valid[b,t] * sign(pred - target) / (K*D)
    d/dtarget        = -d/dpred
    d/dvalid         =  0   (data, not a parameter)

⚠ `sign(0) = 0`. L1 is not differentiable at zero; both PyTorch and this leaf
take the subgradient 0 there. It matters only for an exact tie, but a gate that
compares gradients against autograd on synthetic data can hit one.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


def _l1m_forward_kernel[
    BATCH: Int, K: Int, D: Int
](
    pred: Tile[DT, BATCH, K * D],
    tgt: Tile[DT, BATCH, K * D],
    valid: Tile[DT, BATCH, K],
    o: Tile[DT, BATCH, 1],
):
    var r = Int(global_idx.x)
    if r >= BATCH:
        return
    var s = Scalar[DT](0)
    for t in range(K):
        var m = rebind[Scalar[DT]](valid[r, t])
        for j in range(D):
            var d = rebind[Scalar[DT]](pred[r, t * D + j]) - rebind[
                Scalar[DT]
            ](tgt[r, t * D + j])
            var a = d if d >= Scalar[DT](0) else -d
            s += a * m
    o[r, 0] = s / Scalar[DT](K * D)


def _l1m_backward_kernel[
    BATCH: Int, K: Int, D: Int
](
    go: Tile[DT, BATCH, 1],
    pred: Tile[DT, BATCH, K * D],
    tgt: Tile[DT, BATCH, K * D],
    valid: Tile[DT, BATCH, K],
    gp: Tile[DT, BATCH, K * D],
    gt: Tile[DT, BATCH, K * D],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * K * D:
        return
    var r = idx // (K * D)
    var i = idx % (K * D)
    var t = i // D
    var c = rebind[Scalar[DT]](go[r, 0]) / Scalar[DT](K * D)
    var d = rebind[Scalar[DT]](pred[r, i]) - rebind[Scalar[DT]](tgt[r, i])
    var sgn = Scalar[DT](0)
    if d > Scalar[DT](0):
        sgn = Scalar[DT](1)
    elif d < Scalar[DT](0):
        sgn = Scalar[DT](-1)
    var v = c * sgn * rebind[Scalar[DT]](valid[r, t])
    gp[r, i] = v
    gt[r, i] = -v


def _l1m_zero_kernel[
    N: Int
](g: Tile[DT, N]):
    var i = Int(global_idx.x)
    if i < N:
        g.ptr[unsafe_offset=i] = Scalar[DT](0.0)


struct L1MaskedPerSample[K: Int, D: Int](Module):
    comptime ARITY: Int = 3
    comptime IN_DIMS = _l1m_in_dims[Self.K, Self.D]()
    comptime OUT_DIM: Int = 1
    comptime N: Int = Self.K * Self.D

    def __init__(out self):
        comptime assert Self.K > 0 and Self.D > 0, (
            "L1MaskedPerSample: K, D must be > 0"
        )

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "L1MaskedPerSample: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[3, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = inputs[0]
        ref tgt = inputs[1]
        ref valid = inputs[2]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var s = Scalar[DT](0)
                for t in range(Self.K):
                    var m = valid.data[b * Self.K + t]
                    for j in range(Self.D):
                        var i = t * Self.D + j
                        var d = (
                            pred.data[b * Self.N + i]
                            - tgt.data[b * Self.N + i]
                        )
                        var a = d if d >= Scalar[DT](0) else -d
                        s += a * m
                out.data[b] = s / Scalar[DT](Self.N)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            c.enqueue_function[_l1m_forward_kernel[B, Self.K, Self.D]](
                pred.tt["gpu", B, Self.N](),
                tgt.tt["gpu", B, Self.N](),
                valid.tt["gpu", B, Self.K](),
                out.tt["gpu", B, 1](),
                grid_dim=(B + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[3, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[3, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = forward_input[0]
        ref tgt = forward_input[1]
        ref valid = forward_input[2]
        ref gp = grad_inputs[0]
        ref gt = grad_inputs[1]
        ref gv = grad_inputs[2]
        comptime if target == "cpu":
            gp.ensure(B * Self.N)
            gt.ensure(B * Self.N)
            gv.ensure(B * Self.K)
            for b in range(B):
                var c = grad_output.data[b] / Scalar[DT](Self.N)
                for t in range(Self.K):
                    var m = valid.data[b * Self.K + t]
                    # The mask is data, not a parameter — zero, and zero
                    # EXPLICITLY: the caller's grad slot is reused between
                    # nodes and an accumulating graph would add stale values.
                    gv.data[b * Self.K + t] = Scalar[DT](0)
                    for j in range(Self.D):
                        var i = t * Self.D + j
                        var d = (
                            pred.data[b * Self.N + i]
                            - tgt.data[b * Self.N + i]
                        )
                        var sgn = Scalar[DT](0)
                        if d > Scalar[DT](0):
                            sgn = Scalar[DT](1)
                        elif d < Scalar[DT](0):
                            sgn = Scalar[DT](-1)
                        var v = c * sgn * m
                        gp.data[b * Self.N + i] = v
                        gt.data[b * Self.N + i] = -v
        else:
            var c = ctx.value()
            gp.ensure_gpu(c, B * Self.N)
            gt.ensure_gpu(c, B * Self.N)
            gv.ensure_gpu(c, B * Self.K)
            c.enqueue_function[_l1m_backward_kernel[B, Self.K, Self.D]](
                grad_output.tt["gpu", B, 1](),
                pred.tt["gpu", B, Self.N](),
                tgt.tt["gpu", B, Self.N](),
                valid.tt["gpu", B, Self.K](),
                gp.tt["gpu", B, Self.N](),
                gt.tt["gpu", B, Self.N](),
                grid_dim=(B * Self.N + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[_l1m_zero_kernel[B * Self.K]](
                gv.tt["gpu", B * Self.K](),
                grid_dim=(B * Self.K + TPB - 1) // TPB,
                block_dim=TPB,
            )


def _l1m_in_dims[K: Int, D: Int]() -> Array[Int, 3]:
    """`Array` has no variadic-element literal in Mojo 1.0; mirrors
    `concat.mojo`'s comptime helpers."""
    var a = Array[Int, 3](fill=K * D)
    a[2] = K
    return a^
