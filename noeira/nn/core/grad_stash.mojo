"""Hold a model's gradient buffers out of it for one backward pass.

A Param whose gradient buffer is EMPTY is frozen (`Param.freeze`), and the
layers that own weights (`Linear`, `LayerNorm`) then form grad_input only: no
dW GEMM, no bias or γ/β reduction. `suspend_grads` empties every gradient of a
model by MOVING the buffers into a `GradStash`; `resume_grads` moves the same
buffers back. Nothing is released or reallocated, so the device pointers a
captured CUDA graph or an optimizer's pointer table holds stay valid.

The use is a backward pass THROUGH a network whose parameters are not being
trained by it, e.g. the actor loss of an actor-critic method, which needs
dQ/da and nothing else:

    var held = suspend_grads[T](critic, ctx)
    call_vjp[T, B](critic, inputs, cotangent, grad_inputs, ctx)
    resume_grads[T](critic, held^, ctx)

grad_input is the same bits as without the stash (`Linear.vjp` and
`LayerNorm.vjp` compute it without reading dW / dγ / dβ); the gradient
buffers come back holding exactly what they held before.

⚠ A layer that owns a Param and does NOT honour the empty-gradient rule
would write through an empty buffer. `Linear` and `LayerNorm` honour it;
check any other parameterised layer before stashing a model that holds one.
"""

from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.module import Module


struct GradStash(ParamVisitor, Movable):
    """The gradient buffers of one model, in its param-walk order."""

    var held: List[Tensor]
    var restoring: Bool
    var next: Int

    def __init__(out self):
        self.held = List[Tensor]()
        self.restoring = False
        self.next = 0

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.restoring:
            if len(self.held) == 0:
                raise Error("resume_grads: the model has more params than were stashed")
            grad = self.held.pop(0)
            self.next += 1
        else:
            self.held.append(grad^)
            grad = Tensor()


def suspend_grads[
    target: StaticString, M: Module
](mut model: M, ctx: Optional[DeviceContext]) raises -> GradStash:
    """Empty every gradient buffer of `model` (its params read as frozen) and
    return the buffers. Give them back with `resume_grads`."""
    var s = GradStash()
    model.for_each_param[target](s, ctx)
    return s^


def resume_grads[
    target: StaticString, M: Module
](mut model: M, var stash: GradStash, ctx: Optional[DeviceContext]) raises:
    """Put back the buffers `suspend_grads` took from the same `model`."""
    stash.restoring = True
    stash.next = 0
    model.for_each_param[target](stash, ctx)
    if len(stash.held) != 0:
        raise Error("resume_grads: the model has fewer params than were stashed")
