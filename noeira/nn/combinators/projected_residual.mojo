"""ProjectedResidual[Inner, Skip] — y = Inner(x) + Skip(x) (storage surface).

Like `Residual[Inner]` but the skip path is its own parameterised module (so it
can PROJECT the input to match `Inner.OUT_DIM` when shapes change) — the ResNet
downsampling block (`Inner` = 3×3-s2 → BN → ReLU → 3×3-s1 → BN main path,
`Skip` = 1×1-s2 → BN projection; the external ReLU on the sum is applied by
wrapping this in `Sequential[ProjectedResidual[...], ReLU]`).

Constraints: `Inner.IN_DIMS[0] == Skip.IN_DIMS[0]`, `Inner.OUT_DIM == Skip.OUT_DIM`.
Forward sums the two branch outputs; backward feeds BOTH branches the full
grad_output and sums their grad-inputs. (Branch vjps must not mutate grad_output
— true for all real main/skip paths, which end in BN/Conv/Linear, not a gate.)
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from .residual import _resid_add_kernel


struct ProjectedResidual[Inner: Module, Skip: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    comptime IN = Self.Inner.IN_DIMS[0]
    # Both branches share one activation dtype (asserted in __init__); the sum is
    # element-wise so it's the same dtype on output.
    comptime ACT_DT = Self.Inner.ACT_DT

    var inner: Self.Inner
    var skip: Self.Skip
    var inner_out: TensorImpl[Self.ACT_DT]
    var skip_out: TensorImpl[Self.ACT_DT]
    var gi_inner: TensorImpl[Self.ACT_DT]
    var gi_skip: TensorImpl[Self.ACT_DT]

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Skip.IN_DIMS[0]
        ), "ProjectedResidual requires Inner.IN_DIMS[0] == Skip.IN_DIMS[0]"
        comptime assert (
            Self.Inner.OUT_DIM == Self.Skip.OUT_DIM
        ), "ProjectedResidual requires Inner.OUT_DIM == Skip.OUT_DIM"
        comptime assert (
            Self.Skip.ACT_DT == Self.ACT_DT
        ), "ProjectedResidual requires Inner.ACT_DT == Skip.ACT_DT"
        self.inner = Self.Inner()
        self.skip = Self.Skip()
        self.inner_out = TensorImpl[Self.ACT_DT]()
        self.skip_out = TensorImpl[Self.ACT_DT]()
        self.gi_inner = TensorImpl[Self.ACT_DT]()
        self.gi_skip = TensorImpl[Self.ACT_DT]()


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
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Skip.IN_DIMS[0]
        ), "ProjectedResidual requires Inner.IN_DIMS[0] == Skip.IN_DIMS[0]"
        comptime assert (
            Self.Inner.OUT_DIM == Self.Skip.OUT_DIM
        ), "ProjectedResidual requires Inner.OUT_DIM == Skip.OUT_DIM"
        comptime assert (
            Self.Skip.ACT_DT == Self.ACT_DT
        ), "ProjectedResidual requires Inner.ACT_DT == Skip.ACT_DT"
        self.inner = Self.Inner.make[target, INIT](ctx)
        self.skip = Self.Skip.make[target, INIT](ctx)
        self.inner_out = TensorImpl[Self.ACT_DT]()
        self.skip_out = TensorImpl[Self.ACT_DT]()
        self.gi_inner = TensorImpl[Self.ACT_DT]()
        self.gi_skip = TensorImpl[Self.ACT_DT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self.__init__[target, INIT](ctx=ctx)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Bridge each unary branch input via `child_refs`; the branch output
        # buffers are typed at Self.ACT_DT, rebound to the child's (ci).
        comptime ii = Self.Inner.ACT_DT
        comptime in_ = Self.Inner.ARITY
        comptime si = Self.Skip.ACT_DT
        comptime sn = Self.Skip.ARITY
        ref in0 = inputs[0]
        self.inner.forward[target, B, POLICY=POLICY](
            child_refs[in_, ii](in0), rebind[TensorImpl[ii]](self.inner_out), ctx
        )
        self.skip.forward[target, B, POLICY=POLICY](
            child_refs[sn, si](in0), rebind[TensorImpl[si]](self.skip_out), ctx
        )
        comptime N = B * Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(N)
            var op = out.data.unsafe_ptr()
            var ap = self.inner_out.data.unsafe_ptr()
            var bp = self.skip_out.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= N:
                op.unsafe_store(k, ap.unsafe_load[width=W](k) + bp.unsafe_load[width=W](k))
                k += W
            while k < N:
                op[unsafe_offset=k] = ap[unsafe_offset=k] + bp[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            c.enqueue_function[_resid_add_kernel[N, Self.ACT_DT]](
                self.inner_out.tt["gpu", N](),
                self.skip_out.tt["gpu", N](),
                out.tt["gpu", N](),
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=TPB,
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
        comptime ii = Self.Inner.ACT_DT
        comptime in_ = Self.Inner.ARITY
        comptime si = Self.Skip.ACT_DT
        comptime sn = Self.Skip.ARITY
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        self.inner.vjp[target, B, POLICY=POLICY](
            child_refs[in_, ii](fin),
            rebind[TensorImpl[ii]](grad_output),
            child_refs[in_, ii](self.gi_inner),
            ctx,
        )
        self.skip.vjp[target, B, POLICY=POLICY](
            child_refs[sn, si](fin),
            rebind[TensorImpl[si]](grad_output),
            child_refs[sn, si](self.gi_skip),
            ctx,
        )
        comptime NIN = B * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            var gp = gin.data.unsafe_ptr()
            var ap = self.gi_inner.data.unsafe_ptr()
            var bp = self.gi_skip.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= NIN:
                gp.unsafe_store(k, ap.unsafe_load[width=W](k) + bp.unsafe_load[width=W](k))
                k += W
            while k < NIN:
                gp[unsafe_offset=k] = ap[unsafe_offset=k] + bp[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, NIN)
            c.enqueue_function[_resid_add_kernel[NIN, Self.ACT_DT]](
                self.gi_inner.tt["gpu", NIN](),
                self.gi_skip.tt["gpu", NIN](),
                gin.tt["gpu", NIN](),
                grid_dim=(NIN + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_param[target](
            visitor, ctx, join_name(prefix, String(0))
        )
        self.skip.for_each_param[target](
            visitor, ctx, join_name(prefix, String(1))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_state[target](
            visitor, ctx, join_name(prefix, String(0))
        )
        self.skip.for_each_state[target](
            visitor, ctx, join_name(prefix, String(1))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)
        self.skip.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
        self.skip.polyak_from[target](src.skip, tau, ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)
        self.skip.set_attr[ATTR](value)
