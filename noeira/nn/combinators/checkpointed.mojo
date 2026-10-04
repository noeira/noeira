"""Checkpointed[Inner] — activation checkpointing (recompute in the vjp).

With checkpointing ON (`set_attr["checkpoint"](1)`):

  forward: Inner.forward(x) -> out, then `Inner.release_buffers()` — every
           activation, padded copy and scratch slab inside Inner goes back to
           the device pool; only `x` (owned by the caller) and `out` survive.
  vjp:     Inner.forward(x) AGAIN (into a discarded `redo`), Inner.vjp, then
           release again.

So a stack of N wrapped blocks holds N block inputs plus ONE block's internals
at a time, for one extra forward of each block. OFF (the default) it is a
plain pass-through — every existing model is unchanged until it opts in.

Parameter names are TRANSPARENT: no path segment is added, so a checkpoint
or a converter keyed on the unwrapped names reads the wrapped model as is.

⚠ The recompute must reproduce the first forward exactly. Do NOT wrap a
module whose forward has side effects or draws randomness: BatchNorm in train
mode (its running stats would update twice), dropout (a fresh mask), SIGReg
with `resample` (a fresh projection). The gradients are those of the SECOND
forward, so a deterministic Inner gives the same gradients as unwrapped.

⚠ Memory is only returned by modules that override `release_buffers`
(`Module`'s default keeps everything). Measure with `NOEIRA_ALLOC_TRACE=1`.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.amp import AMPPolicy, NoAMP


struct Checkpointed[Inner: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    comptime ACT_DT = Self.Inner.ACT_DT

    var inner: Self.Inner
    var redo: TensorImpl[Self.ACT_DT]
    """The recomputed forward's output in the vjp: discarded (the caller's
    `out` already holds the first one)."""
    var enabled: Bool

    def __init__(out self):
        comptime assert Self.Inner.ARITY == 1, "Checkpointed wraps a unary module"
        self.inner = Self.Inner()
        self.redo = TensorImpl[Self.ACT_DT]()
        self.enabled = False

    def __init__[
        target: StaticString, INIT: Initializer
    ](out self, *, ctx: Optional[DeviceContext]) raises:
        comptime assert Self.Inner.ARITY == 1, "Checkpointed wraps a unary module"
        self.inner = Self.Inner.make[target, INIT](ctx)
        self.redo = TensorImpl[Self.ACT_DT]()
        self.enabled = False

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self.__init__[target, INIT](ctx=ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`checkpoint` (0/1) switches recomputation; every attr also goes to
        Inner (a nested `Checkpointed` follows the same switch)."""
        comptime if ATTR == "checkpoint":
            self.enabled = value != Scalar[DT](0)
        self.inner.set_attr[ATTR](value)

    def set_attr_buf[ATTR: StaticString](mut self, buf: DeviceBuffer[DT]):
        self.inner.set_attr_buf[ATTR](buf)

    def release_buffers(mut self):
        self.redo.release()
        self.inner.release_buffers()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        self.inner.forward[target, B, POLICY=POLICY](
            child_refs[cn, ci](inputs[0]), rebind[TensorImpl[ci]](out), ctx
        )
        if self.enabled:
            self.inner.release_buffers()

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        if self.enabled:
            self.inner.forward[target, B, POLICY=POLICY](
                child_refs[cn, ci](forward_input[0]),
                rebind[TensorImpl[ci]](self.redo),
                ctx,
            )
        self.inner.vjp[target, B, POLICY=POLICY](
            child_refs[cn, ci](forward_input[0]),
            rebind[TensorImpl[ci]](grad_output),
            child_refs[cn, ci](grad_inputs[0]),
            ctx,
        )
        if self.enabled:
            self.inner.release_buffers()
            self.redo.release()

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
