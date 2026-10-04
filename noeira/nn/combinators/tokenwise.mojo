"""Tokenwise[SEQ_LEN, Inner] — apply a shared-weight Module per token.

A sequence sample is laid out `(SEQ_LEN, Inner.IN)` row-major. Tokenwise
reinterprets the `(BATCH, SEQ_LEN*Inner.IN)` slab as `(BATCH*SEQ_LEN, Inner.IN)`
and runs `Inner` once over that flattened batch — same weights at every position.
The reshape is pure index reinterpretation (row-major flat index is identical),
so there's NO mid-slab and NO extra kernel: forward/vjp delegate straight to
`Inner` at batch `BATCH*SEQ_LEN`.

  IN_DIM  = SEQ_LEN * Inner.IN_DIMS[0]
  OUT_DIM = SEQ_LEN * Inner.OUT_DIM
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP


struct Tokenwise[SEQ_LEN: Int, Inner: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](
        fill=Self.SEQ_LEN * Self.Inner.IN_DIMS[0]
    )
    comptime OUT_DIM = Self.SEQ_LEN * Self.Inner.OUT_DIM
    # Reshape-only wrapper — activation dtype is the wrapped child's.
    comptime ACT_DT = Self.Inner.ACT_DT

    var inner: Self.Inner

    def __init__(out self):
        comptime assert Self.SEQ_LEN >= 1, "Tokenwise requires SEQ_LEN >= 1"
        self.inner = Self.Inner()


    def __init__[
        target: StaticString, INIT: Initializer
    ](out self, *, ctx: Optional[DeviceContext]) raises:
        """Build the children IN PLACE, straight from their `make`.

        Same reason as `Sequential.__init__[target, INIT]`: `make` used to
        default-construct `Self` (recursively, down to every leaf's empty
        Params) and then move each made child over its default. Inlined,
        that construct-then-move chain was the weight of the constructors on
        the ACT trainer (docs/COMPILE_TIME_PROFILING.md §3.2).
        """
        comptime assert Self.SEQ_LEN >= 1, "Tokenwise requires SEQ_LEN >= 1"
        self.inner = Self.Inner.make[target, INIT](ctx)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self.__init__[target, INIT](ctx=ctx)

    def release_buffers(mut self):
        self.inner.release_buffers()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Forward runtime attrs into the wrapped module. Without this the
        `Module` trait's no-op default swallowed them here — BatchNorm
        `training` never reached the LeWM encoder projector / PredProj BNs
        (both live under a Tokenwise), so "eval-mode" planning forwards
        kept updating the running stats."""
        self.inner.set_attr[ATTR](value)

    def set_attr_buf[ATTR: StaticString](mut self, buf: DeviceBuffer[DT]):
        """Forward device-buffer attrs into the wrapped module (same
        swallowed-default hazard as `set_attr`)."""
        self.inner.set_attr_buf[ATTR](buf)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Buffers are typed at Self.ACT_DT (== Inner.ACT_DT, distinct to the
        # checker); bridge the input pack with `child_refs` and the mut output
        # with `rebind[TensorImpl[ci]]`.
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        self.inner.forward[target, B * Self.SEQ_LEN, POLICY=POLICY](
            child_refs[cn, ci](inputs[0]), rebind[TensorImpl[ci]](out), ctx
        )

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
        self.inner.vjp[target, B * Self.SEQ_LEN, POLICY=POLICY](
            child_refs[cn, ci](forward_input[0]),
            rebind[TensorImpl[ci]](grad_output),
            child_refs[cn, ci](grad_inputs[0]),
            ctx,
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_param[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_state[target](
            visitor, ctx, join_name(prefix, String(0))
        )

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
