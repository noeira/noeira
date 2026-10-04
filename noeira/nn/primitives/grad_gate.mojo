"""GradGate[DIM] — identity forward, a runtime-switchable stop-gradient.

Forward:  out = in
Backward: grad_in = grad_out          (default)
          grad_in = 0                 (`set_attr["stop_grad"](1)`)

`StopGrad` severs the gradient for good (its `ElementOp` backward is static);
GradGate lets one graph serve both regimes — e.g. LeWM's training loss has NO
stop-gradient on its target embeddings, while AdaJEPA's test-time adaptation
detaches them. Off, the vjp copies `grad_out` exactly (×1), so a graph that
gains a GradGate node trains bit-identically. No params, no cache: both paths
are `Scale`'s CPU SIMD loop and GPU kernel, multiplier 1 forward and 1 / 0
backward.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .scale import Scale


struct GradGate[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    @staticmethod
    def display_label() -> String:
        return String("GradGate")

    var fwd: Scale[Self.DIM_]
    """Multiplier 1: the identity forward."""
    var bwd: Scale[Self.DIM_]
    """Multiplier 1 (pass) or 0 (stop): the backward."""

    def __init__(out self):
        self.fwd = Scale[Self.DIM_]()
        self.bwd = Scale[Self.DIM_]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """INIT accepted for `make[target, INIT]` uniformity, ignored."""
        var g = Self()
        return g^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`stop_grad` (0/1): 1 zeroes the gradient to the input."""
        comptime if ATTR == "stop_grad":
            self.bwd.multiplier = Scalar[DT](0) if value != Scalar[DT](0) else Scalar[DT](1)

    def stops(self) -> Bool:
        return self.bwd.multiplier == Scalar[DT](0)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.fwd.forward[target, B](inputs, out, ctx)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.bwd.vjp[target, B](forward_input, grad_output, grad_inputs, ctx)
