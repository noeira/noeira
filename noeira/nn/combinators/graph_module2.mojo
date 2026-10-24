"""GraphModule2[IN0, IN1, OUT, GRAPH] — an ARITY=2 `Module` over any two-input
`ComputeGraph`.

`ComputeGraph` is not a `Module`: it is driven by name (`set_input["x", B]`,
`forward[B, target]`) rather than by an input pack, so it cannot be a child of
`Sequential` / `Repeat` / `RepeatConditional`. `primitives/decoder_block.mojo`
bridges that gap for ONE graph by hand-writing the wrapper — seed both slots,
run, copy the two named input-grads back out, and forward every walker into the
graph.

That wrapper is entirely mechanical and its only graph-specific parts are the
two input dims, the output dim, and the graph type. This is the same code with
those four as parameters, so a new two-input block is an alias rather than
another 120 lines in which one of the walkers can be quietly omitted.

    comptime MyBlock[D: Int] = GraphModule2[D, D, D, MyGraph[D]]

Requirements on `GRAPH`: exactly two `InputSlot`s, named **"x"** (the residual
stream, dim `IN0`) and **"c"** (the conditioning, dim `IN1`), with the terminal
node's width equal to `OUT`. Those names are the convention `RepeatConditional`
already implies; a mismatch is a comptime error inside `ComputeGraph`.

`IN0 == OUT` additionally makes the block stackable by
`RepeatConditional[N, Block]`, which is how the DETR encoder/decoder stacks are
built — but it is not required here (a block may change width).

CPU + GPU, following `decoder_block.mojo` exactly, including the size-exact
sub-buffer copies on the GPU grad path (the caller's temporary grad slots are
reused across nodes and may be LARGER than this block's `B*DIM`; a whole-buffer
copy errors on the mismatch).
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP



trait TwoInputGraph(Defaultable & Movable & Deinitable):
    """The `ComputeGraph` surface `GraphModule2` drives.

    `ComputeGraph` is not a `Module` — it is addressed by slot NAME rather than
    by an input pack — so a generic wrapper needs a bound of its own. This
    declares exactly the members used below and nothing else; `ComputeGraph`
    conforms structurally and the conformance is declared on its struct.
    """

    comptime ACT_DT: DType
    comptime OUT_DIM: Int

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        ...

    def set_input[
        slot_name: StaticString, B: Int
    ](
        mut self,
        mut src: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def forward[
        B: Int,
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
        *EXT: Module,
    ](
        mut self,
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
        mut *externals: *EXT,
    ) raises:
        ...

    def vjp[
        B: Int,
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
        *EXT: Module,
    ](
        mut self,
        mut grad_out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
        mut *externals: *EXT,
    ) raises:
        ...

    def grad_input[
        name: StaticString
    ](mut self) raises -> ref [MutAnyOrigin] TensorImpl[Self.ACT_DT]:
        ...

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        ...

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        ...

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        ...

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        ...

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        ...


trait NamedNodeGraph(TwoInputGraph):
    """`TwoInputGraph` plus the by-NAME node surface a trainer drives: a node's
    attribute, a node's output, and whether a node exists. For a trainer that
    is generic over several graphs of one family (the SO-101 world models,
    `experimental/lewm/so101_trainer.mojo`); `ComputeGraph` conforms."""

    @staticmethod
    def has_node[name: StaticString]() -> Bool:
        """True when a decl is named `name` (comptime-folded): guards a
        `node_output` / `set_input` the graph may not have."""
        ...

    def set_node_attr[
        NAME: StaticString, ATTR: StaticString
    ](mut self, value: Scalar[DT]):
        ...

    def node_output[
        name: StaticString
    ](mut self) raises -> ref [MutAnyOrigin] TensorImpl[Self.ACT_DT]:
        ...


struct GraphModule2[
    IN0: Int, IN1: Int, OUT: Int, GRAPH: TwoInputGraph
](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = _gm2_in_dims[Self.IN0, Self.IN1]()
    comptime OUT_DIM: Int = Self.OUT

    var graph: Self.GRAPH

    def __init__(out self):
        self.graph = Self.GRAPH()

    def __init__(out self, *, deinit move: Self):
        self.graph = move.graph^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "GraphModule2: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.graph = Self.GRAPH.make[target, INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime gd = Self.GRAPH.ACT_DT
        comptime assert gd == DT, (
            "GraphModule2: only fp32-flow graphs are supported"
        )
        ref x = inputs[0]
        ref c = inputs[1]
        # `rebind` because the compiler will not unify the child's OPAQUE
        # `GRAPH.ACT_DT` with `DT` even though the assert above pins them equal
        # — the same limitation `core/tensor_refs.mojo::child_refs` documents.
        # The graph COPIES into its own pool, so no pointer outlives this call.
        self.graph.set_input["x", B](rebind[TensorImpl[gd]](x), ctx)
        self.graph.set_input["c", B](rebind[TensorImpl[gd]](c), ctx)
        self.graph.forward[B, target, POLICY=POLICY](
            rebind[TensorImpl[gd]](out), ctx
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
        comptime gd = Self.GRAPH.ACT_DT
        comptime assert gd == DT, (
            "GraphModule2: only fp32-flow graphs are supported"
        )
        ref gx = grad_inputs[0]
        ref gc = grad_inputs[1]
        self.graph.vjp[B, target, POLICY=POLICY](
            rebind[TensorImpl[gd]](grad_output), ctx
        )
        comptime N0 = B * Self.IN0
        comptime N1 = B * Self.IN1
        comptime if target == "cpu":
            gx.ensure(N0)
            gc.ensure(N1)
            ref sx = rebind[Tensor](self.graph.grad_input["x"]())
            for q in range(N0):
                gx.data[q] = sx.data[q]
            ref sc = rebind[Tensor](self.graph.grad_input["c"]())
            for q in range(N1):
                gc.data[q] = sc.data[q]
        else:
            var cc = ctx.value()
            gx.ensure_gpu(cc, N0)
            gc.ensure_gpu(cc, N1)
            # Size-exact: the caller's temporary grad slots are reused across
            # nodes and may be larger than N0/N1, and a whole-buffer copy
            # errors on the mismatch. Mirrors decoder_block.mojo.
            var gx_src = self.graph.grad_input["x"]().dev.value(
            ).create_sub_buffer[DT](0, N0)
            var gx_dst = gx.dev.value().create_sub_buffer[DT](0, N0)
            cc.enqueue_copy(gx_dst, gx_src)
            var gc_src = self.graph.grad_input["c"]().dev.value(
            ).create_sub_buffer[DT](0, N1)
            var gc_dst = gc.dev.value().create_sub_buffer[DT](0, N1)
            cc.enqueue_copy(gc_dst, gc_src)

    # ── walkers: everything must reach the graph, or the optimizer, the
    # checkpoint and `zero_grad` each silently skip this block's weights.

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.graph.zero_grad[target](ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Forwarded so `set_attr["training"](0.0)` reaches every Dropout /
        BatchNorm inside the graph."""
        self.graph.set_attr[ATTR](value)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.graph.polyak_from[target](src.graph, tau, ctx)


def _gm2_in_dims[IN0: Int, IN1: Int]() -> Array[Int, 2]:
    """`Array` has no variadic-element literal constructor in Mojo 1.0
    (`Array` is not `ImplicitlyCopyable`), so `IN_DIMS` is built by a comptime
    helper — the same shape as `concat.mojo`'s `_total_dim`."""
    var a = Array[Int, 2](fill=IN0)
    a[1] = IN1
    return a^
