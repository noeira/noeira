"""Residual[Inner] — y = inner(x) + x (storage-passing, CPU + GPU).

The storage twin of legacy `nn.combinators.Residual`. Requires
`Inner.IN_DIMS[0] == Inner.OUT_DIM`. Owns one `mid` scratch Tensor: forward runs
`inner.forward → mid` then `out = mid + x` (SIMD on CPU, one add kernel on GPU);
vjp runs `inner.vjp → mid` (mid = grad wrt inner's input) then
`grad_input = mid + grad_output` (the identity skip's grad). The legacy
`TargetStorage`/`POLICY`/`mode`/name-keyed-param machinery is gone — the storage
surface (`ref`/`mut Tensor`, `TensorRefs`) handles both targets uniformly.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP


def _resid_add_kernel[
    N: Int, ADT: DType = DT
](
    a: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    b: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
):
    # The skip ADD runs on the combinator's ACT_DT activation buffers; `ADT`
    # defaults to DT so NoAMP-callers (passing the default) are unchanged.
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[ADT]](a[i]) + rebind[Scalar[ADT]](b[i])


struct Residual[Inner: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    comptime DIM = Self.Inner.OUT_DIM
    # The skip path is element-wise — no dtype change — so the residual's
    # activation dtype is just the wrapped module's.
    comptime ACT_DT = Self.Inner.ACT_DT

    var inner: Self.Inner
    var mid: TensorImpl[Self.ACT_DT]  # inner's output (fwd) / inner's grad-input (bwd)

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Residual requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.inner = Self.Inner()
        self.mid = TensorImpl[Self.ACT_DT]()


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
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Residual requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.inner = Self.Inner.make[target, INIT](ctx)
        self.mid = TensorImpl[Self.ACT_DT]()

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
        # Buffers are typed at Self.ACT_DT; bridge to the inner child's
        # (ARITY, ACT_DT) via `child_refs` (input pack) and `rebind` (mut output).
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        ref in0 = inputs[0]
        self.inner.forward[target, B, POLICY=POLICY](
            child_refs[cn, ci](in0), rebind[TensorImpl[ci]](self.mid), ctx
        )
        comptime N = B * Self.DIM
        comptime if target == "cpu":
            out.ensure(N)
            var op = out.data.unsafe_ptr()
            var mp = self.mid.data.unsafe_ptr()
            var ip = in0.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= N:
                op.unsafe_store(k, mp.unsafe_load[width=W](k) + ip.unsafe_load[width=W](k))
                k += W
            while k < N:
                op[unsafe_offset=k] = mp[unsafe_offset=k] + ip[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            c.enqueue_function[_resid_add_kernel[N, Self.ACT_DT]](
                self.mid.lt["gpu", Layout.row_major(N)](),
                in0.lt["gpu", Layout.row_major(N)](),
                out.lt["gpu", Layout.row_major(N)](),
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
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        # mid := grad wrt inner's input; then grad_input = mid + grad_output.
        self.inner.vjp[target, B, POLICY=POLICY](
            child_refs[cn, ci](fin),
            rebind[TensorImpl[ci]](grad_output),
            child_refs[cn, ci](self.mid),
            ctx,
        )
        comptime N = B * Self.DIM
        comptime if target == "cpu":
            gin.ensure(N)
            var gp = gin.data.unsafe_ptr()
            var mp = self.mid.data.unsafe_ptr()
            var gop = grad_output.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= N:
                gp.unsafe_store(k, mp.unsafe_load[width=W](k) + gop.unsafe_load[width=W](k))
                k += W
            while k < N:
                gp[unsafe_offset=k] = mp[unsafe_offset=k] + gop[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            c.enqueue_function[_resid_add_kernel[N, Self.ACT_DT]](
                self.mid.lt["gpu", Layout.row_major(N)](),
                grad_output.lt["gpu", Layout.row_major(N)](),
                gin.lt["gpu", Layout.row_major(N)](),
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=TPB,
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

    def release_buffers(mut self):
        """`mid` (the inner output / inner grad-input, both transient), then
        the inner module."""
        self.mid.release()
        self.inner.release_buffers()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)
