"""Sequential[*MODULES] — storage-passing orchestrator (N-ary, CPU + GPU).

Threads `comptime target` + `ctx` to its unary children. Inter-module buffers
live in two owning `TensorPack`s; the children lazily allocate their slabs on
the active target (`out.ensure`/`ensure_gpu`). `TensorPack.__getitem__` returns
a `MutAnyOrigin` ref (load-bearing pin, §7.12); each child input is wrapped in
a borrowing `TensorRefs[1]`. Slice scope: N >= 2.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from ..core.graph_visitor import DisplayStep


struct Sequential[*MODULES: Module](Module):
    comptime ARITY = 1
    comptime N = Self.MODULES.length
    comptime IN_DIMS = Array[Int, 1](fill=Self.MODULES[0].IN_DIMS[0])
    comptime OUT_DIM = Self.MODULES[Self.N - 1].OUT_DIM
    # All children share one activation dtype (asserted in __init__); the chain's
    # ACT_DT is the last child's (= the chain output dtype). Inter-module buffers
    # are stored at this dtype.
    comptime ACT_DT = Self.MODULES[Self.N - 1].ACT_DT

    @staticmethod
    def display_label() -> String:
        return String("Sequential")

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        """Expand the chain — one step per child (its display label + output
        width) — so `ComputeGraph.describe` exporters open a Sequential node
        instead of showing one opaque box."""
        var steps = List[DisplayStep]()
        comptime for i in range(Self.N):
            steps.append(
                DisplayStep(
                    Self.MODULES[i].display_label(),
                    Self.MODULES[i].OUT_DIM,
                )
            )
        return steps^

    var children: Tuple[*Self.MODULES]
    var act: TensorPack[Self.N, Self.ACT_DT]
    var grd: TensorPack[Self.N, Self.ACT_DT]

    def __init__(out self):
        comptime assert Self.N >= 2, "Sequential slice requires N >= 2"
        comptime for i in range(Self.N):
            comptime assert (
                Self.MODULES[i].ARITY == 1
            ), "Sequential chains UNARY children only (ARITY == 1)"
        comptime for i in range(Self.N - 1):
            comptime assert (
                Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIMS[0]
            ), "Sequential: adjacent child dims must match"
        comptime for i in range(Self.N):
            comptime assert (
                Self.MODULES[i].ACT_DT == Self.ACT_DT
            ), "Sequential chains children of ONE activation dtype (ACT_DT)"
        self.children = Tuple[*Self.MODULES]()
        self.act = TensorPack[Self.N, Self.ACT_DT]()
        self.grd = TensorPack[Self.N, Self.ACT_DT]()

    def __init__[
        target: StaticString, INIT: Initializer
    ](out self, *, ctx: Optional[DeviceContext]) raises:
        """Build every child IN PLACE, straight from its `make`.

        The default `__init__` above default-constructs the whole child
        tuple (recursively: a nested Sequential default-constructs its own
        children, down to every leaf's empty Params), and `make` used to
        start from that and then move each made child over the default one.
        Inlined, that construct-then-move chain was most of the weight of
        `Sequential::make` on the ACT trainer (249 KB, with a 113 KB
        `__init__` beside it; docs/COMPILE_TIME_PROFILING.md §3.2). Here the
        tuple storage is marked live and each slot is written once with the
        made child — the same idiom `Tuple.__init__(var *args)` uses.

        ⚠ If a child's `make` raises part-way (a device allocation failing),
        the slots after it are never written and the tuple's destructor
        would run over them. The default path did not have that hole; a
        `make` that raises is an allocation failure the run does not survive
        anyway, which is why this is accepted rather than guarded.
        """
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.children)
        )
        comptime for i in range(Self.N):
            Pointer(to=self.children[i]).unsafe_write(
                Self.MODULES[i].make[target, INIT](ctx)
            )
        self.act = TensorPack[Self.N, Self.ACT_DT]()
        self.grd = TensorPack[Self.N, Self.ACT_DT]()

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
        # Buffers are typed at Self.ACT_DT; each child's forward wants the CHILD's
        # ACT_DT (== Self.ACT_DT, asserted, but distinct to the checker). Bridge:
        # `child_refs[cn, ci]` for the input pack, `rebind[TensorImpl[ci]]` for the
        # mut output buffer (sound — only pointers are reinterpreted).
        comptime for i in range(Self.N):
            comptime ci = Self.MODULES[i].ACT_DT
            comptime cn = Self.MODULES[i].ARITY
            comptime if i == 0:
                self.children[i].forward[target, B, POLICY=POLICY](
                    child_refs[cn, ci](inputs[0]),
                    rebind[TensorImpl[ci]](self.act[0]),
                    ctx,
                )
            elif i == Self.N - 1:
                self.children[i].forward[target, B, POLICY=POLICY](
                    child_refs[cn, ci](self.act[Self.N - 2]),
                    rebind[TensorImpl[ci]](out),
                    ctx,
                )
            else:
                self.children[i].forward[target, B, POLICY=POLICY](
                    child_refs[cn, ci](self.act[i - 1]),
                    rebind[TensorImpl[ci]](self.act[i]),
                    ctx,
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
        comptime for j in range(Self.N):
            comptime i = Self.N - 1 - j
            comptime ci = Self.MODULES[i].ACT_DT
            comptime cn = Self.MODULES[i].ARITY
            comptime if i == Self.N - 1:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    child_refs[cn, ci](self.act[Self.N - 2]),
                    rebind[TensorImpl[ci]](grad_output),
                    child_refs[cn, ci](self.grd[Self.N - 2]),
                    ctx,
                )
            elif i == 0:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    child_refs[cn, ci](forward_input[0]),
                    rebind[TensorImpl[ci]](self.grd[0]),
                    child_refs[cn, ci](grad_inputs[0]),
                    ctx,
                )
            else:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    child_refs[cn, ci](self.act[i - 1]),
                    rebind[TensorImpl[ci]](self.grd[i]),
                    child_refs[cn, ci](self.grd[i - 1]),
                    ctx,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)

    def release_buffers(mut self):
        """The inter-child slabs, then every child (`Module.release_buffers`)."""
        self.act.release()
        self.grd.release()
        comptime for i in range(Self.N):
            self.children[i].release_buffers()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Recurse a named runtime attribute (e.g. BatchNorm's "training") into
        every child — the plain-Sequential analog of ComputeGraph's
        `set_node_attr`. Leaves without the attr no-op."""
        comptime for i in range(Self.N):
            self.children[i].set_attr[ATTR](value)
