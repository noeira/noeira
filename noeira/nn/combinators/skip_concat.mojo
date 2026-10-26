"""SkipConcat[Inner] — y = concat(x, inner(x)) (storage surface, CPU + GPU).

DenseNet/U-Net-style skip-concat: the output is the input column-stacked with the
inner module's output. Mirrors `Residual` structurally (one child) but the merge
is concat instead of add.

  IN_DIMS[0] = Inner.IN_DIMS[0]                  (passthrough)
  OUT_DIM    = Inner.IN_DIMS[0] + Inner.OUT_DIM  (input ++ inner output)

Forward:  out = [x | inner(x)]
Backward: grad_input = grad_output[:, :IN] + inner.vjp(grad_output[:, IN:])
(the input feeds both the passthrough column block and inner — its grads sum).
Reuses Parallel's concat/split kernels + Residual's add kernel.

The concat is a column move (no recast), so every activation buffer (inner_out,
go_*, gi_inner, out, grads) is `TensorImpl[Self.ACT_DT]` and the shared kernels
run at that dtype (`ADT` defaults to DT → NoAMP unchanged).
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
from .parallel import _par_concat_kernel, _par_split_kernel
from .residual import _resid_add_kernel


struct SkipConcat[Inner: Module](Module):
    comptime ARITY = 1
    comptime IN = Self.Inner.IN_DIMS[0]
    comptime OINNER = Self.Inner.OUT_DIM
    comptime IN_DIMS = Array[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.IN_DIMS[0] + Self.Inner.OUT_DIM
    # The skip is a column concat (no dtype change) — activation dtype is just
    # the wrapped module's.
    comptime ACT_DT = Self.Inner.ACT_DT

    var inner: Self.Inner
    var inner_out: TensorImpl[Self.ACT_DT]
    var go_pass: TensorImpl[Self.ACT_DT]  # grad_output[:, :IN]   (bwd)
    var go_inner: TensorImpl[Self.ACT_DT]  # grad_output[:, IN:]   (bwd, fed to inner)
    var gi_inner: TensorImpl[Self.ACT_DT]  # inner's grad-input

    def __init__(out self):
        self.inner = Self.Inner()
        self.inner_out = TensorImpl[Self.ACT_DT]()
        self.go_pass = TensorImpl[Self.ACT_DT]()
        self.go_inner = TensorImpl[Self.ACT_DT]()
        self.gi_inner = TensorImpl[Self.ACT_DT]()


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
        self.inner = Self.Inner.make[target, INIT](ctx)
        self.inner_out = TensorImpl[Self.ACT_DT]()
        self.go_pass = TensorImpl[Self.ACT_DT]()
        self.go_inner = TensorImpl[Self.ACT_DT]()
        self.gi_inner = TensorImpl[Self.ACT_DT]()

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
            child_refs[cn, ci](in0),
            rebind[TensorImpl[ci]](self.inner_out),
            ctx,
        )
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            for b in range(B):
                var ob = b * Self.OUT_DIM
                for j in range(Self.IN):
                    out.data[ob + j] = in0.data[b * Self.IN + j]
                for j in range(Self.OINNER):
                    out.data[ob + Self.IN + j] = self.inner_out.data[
                        b * Self.OINNER + j
                    ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            c.enqueue_function[
                _par_concat_kernel[B, Self.IN, Self.OINNER, Self.ACT_DT]
            ](
                in0.tt["gpu", B, Self.IN](),
                self.inner_out.tt["gpu", B, Self.OINNER](),
                out.tt["gpu", B, Self.OUT_DIM](),
                grid_dim=(B * Self.OUT_DIM + TPB - 1) // TPB,
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
        # split grad_output → go_pass [:IN], go_inner [IN:]
        comptime if target == "cpu":
            self.go_pass.ensure(B * Self.IN)
            self.go_inner.ensure(B * Self.OINNER)
            for b in range(B):
                var gb = b * Self.OUT_DIM
                for j in range(Self.IN):
                    self.go_pass.data[b * Self.IN + j] = grad_output.data[
                        gb + j
                    ]
                for j in range(Self.OINNER):
                    self.go_inner.data[b * Self.OINNER + j] = grad_output.data[
                        gb + Self.IN + j
                    ]
        else:
            var c = ctx.value()
            self.go_pass.ensure_gpu(c, B * Self.IN)
            self.go_inner.ensure_gpu(c, B * Self.OINNER)
            c.enqueue_function[
                _par_split_kernel[B, Self.IN, Self.OINNER, Self.ACT_DT]
            ](
                grad_output.tt["gpu", B, Self.OUT_DIM](),
                self.go_pass.tt["gpu", B, Self.IN](),
                self.go_inner.tt["gpu", B, Self.OINNER](),
                grid_dim=(B * Self.OUT_DIM + TPB - 1) // TPB,
                block_dim=TPB,
            )
        # inner's vjp: gradient flows go_inner → gi_inner at the inner child's
        # (ARITY, ACT_DT) — bridge fin/go_inner/gi_inner like forward.
        self.inner.vjp[target, B, POLICY=POLICY](
            child_refs[cn, ci](fin),
            rebind[TensorImpl[ci]](self.go_inner),
            child_refs[cn, ci](self.gi_inner),
            ctx,
        )
        # grad_input = go_pass + gi_inner
        comptime NIN = B * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            var gp = gin.data.unsafe_ptr()
            var pp = self.go_pass.data.unsafe_ptr()
            var ip = self.gi_inner.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= NIN:
                gp.unsafe_store(k, pp.unsafe_load[width=W](k) + ip.unsafe_load[width=W](k))
                k += W
            while k < NIN:
                gp[unsafe_offset=k] = pp[unsafe_offset=k] + ip[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, NIN)
            c.enqueue_function[_resid_add_kernel[NIN, Self.ACT_DT]](
                self.go_pass.tt["gpu", NIN](),
                self.gi_inner.tt["gpu", NIN](),
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
