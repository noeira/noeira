"""Freeze or unfreeze every parameter of a model.

A frozen parameter is one whose gradient buffer is EMPTY (`Param.freeze`). It
keeps its value, its name and its place in every walk, so checkpoints, the
safetensors maps and the torch-name maps see exactly what they saw before;
it has no gradient and no optimizer moments. The optimizers, the gradient
clippers, the optimizer arena and the ZeRO wrappers skip it, and the
checkpoint readers do not give it moments back.

    var n = freeze_params["gpu"](model, ctx)     # every parameter of `model`
    _ = unfreeze_params["gpu"](model, ctx)       # zeroed gradients again

A struct that walks its parameters without being a `ParamWalkable` (SmolVLA's
text tower) takes the visitors directly:

    var f = FreezeParams()
    tower.for_each_param["gpu"](f, ctx)

⚠ FREEZE BEFORE AN OPTIMIZER ADOPTS THE MODEL. Adopting points every
trainable gradient into the optimizer's arena and leaves frozen ones out. A
parameter frozen afterwards keeps its slot: the grouped step goes on applying
weight decay to its value.

⚠ WHICH BACKWARDS SUPPORT A FROZEN PARAMETER. `Linear` (every path) and
`RMSNorm` form no gradient for a frozen parameter and compute their input
gradient unchanged. Every other leaf still writes its parameter gradient in
`vjp` and aborts on an empty buffer. Freeze those only where their backward
never runs: SmolVLA's vision tower, connector and token embedding are upstream
of nothing trained.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.param import ParamVisitor, ParamVisitorRT, ParamWalkable, walk_params


struct FreezeParams(Movable, ParamVisitor, ParamVisitorRT):
    """Releases each visited parameter's gradient and moments, host and
    device. `values` counts the values that were trainable before."""

    var values: Int

    def __init__(out self):
        self.values = 0

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n > 0:
            self.values += n
        grad.release()
        m.release()
        v.release()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


struct UnfreezeParams(Movable, ParamVisitor, ParamVisitorRT):
    """Gives each frozen parameter a zeroed gradient again, allocated the way
    `Param.make` allocates one. Moments come back lazily, on the optimizer's
    next step. `values` counts the values that were frozen before."""

    var values: Int

    def __init__(out self):
        self.values = 0

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n > 0:
            return
        grad = Tensor.alloc(n)
        comptime if target == "gpu":
            var c = ctx.value()
            grad.ensure_gpu(c, n)
            grad.dev.value().enqueue_fill(Scalar[DT](0))
        self.values += n

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


def freeze_params[target: StaticString, M: ParamWalkable](
    mut model: M, ctx: Optional[DeviceContext] = None
) raises -> Int:
    """Freeze every parameter of `model`. Returns the number of values that
    were trainable and are now frozen."""
    var f = FreezeParams()
    walk_params[target](model, f, ctx)
    return f.values


def unfreeze_params[target: StaticString, M: ParamWalkable](
    mut model: M, ctx: Optional[DeviceContext] = None
) raises -> Int:
    """Unfreeze every parameter of `model`. Returns the number of values that
    were frozen and are now trainable."""
    var u = UnfreezeParams()
    walk_params[target](model, u, ctx)
    return u.values
