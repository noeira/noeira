"""Repeat[N, Inner] — chain N independent copies of `Inner` (storage surface).

`y = Inner_{N-1}(… Inner_1(Inner_0(x)) …)`, each copy with its OWN params. This
is exactly `Sequential` over N homogeneous children — same `TensorPack` mid-slab
wiring (whose `__getitem__` MutAnyOrigin pin lets adjacent slabs alias safely),
just stored in a `List[Inner]` indexed by the comptime stage. Requires
`Inner.IN_DIMS[0] == Inner.OUT_DIM`. `shared=True` is not supported (would need
per-application caches); pass the default `shared=False` (independent copies).
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


struct Repeat[N: Int, Inner: Module, shared: Bool = False](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    # The chain's activation dtype IS the repeated child's (all N copies share
    # one dtype). Inter-stage buffers are stored here.
    comptime ACT_DT = Self.Inner.ACT_DT

    var children: List[Self.Inner]
    var act: TensorPack[Self.N, Self.ACT_DT]
    var grd: TensorPack[Self.N, Self.ACT_DT]

    def __init__(out self):
        comptime assert Self.N >= 1, "Repeat requires N >= 1"
        comptime assert (
            not Self.shared
        ), "Repeat: shared=True not supported — use shared=False"
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Repeat requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.children = List[Self.Inner]()
        self.act = TensorPack[Self.N, Self.ACT_DT]()
        self.grd = TensorPack[Self.N, Self.ACT_DT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        for _ in range(Self.N):
            r.children.append(Self.Inner.make[target, INIT](ctx))
        return r^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Buffers are typed at Self.ACT_DT; the child wants Inner's ACT_DT (==
        # Self.ACT_DT, but distinct to the checker). Bridge via `child_refs[cn,
        # ci]` for the input pack and `rebind[TensorImpl[ci]]` for the mut output.
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        comptime if Self.N == 1:
            self.children[0].forward[target, B, POLICY=POLICY](
                child_refs[cn, ci](inputs[0]),
                rebind[TensorImpl[ci]](out),
                ctx,
            )
        else:
            comptime for i in range(Self.N):
                comptime if i == 0:
                    self.children[0].forward[target, B, POLICY=POLICY](
                        child_refs[cn, ci](inputs[0]),
                        rebind[TensorImpl[ci]](self.act[0]),
                        ctx,
                    )
                elif i == Self.N - 1:
                    self.children[Self.N - 1].forward[target, B, POLICY=POLICY](
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
        # Same child-edge bridge as forward: `child_refs[cn, ci]` for input/grad
        # packs, `rebind[TensorImpl[ci]]` for the mut grad buffers.
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        comptime if Self.N == 1:
            self.children[0].vjp[target, B, POLICY=POLICY](
                child_refs[cn, ci](forward_input[0]),
                rebind[TensorImpl[ci]](grad_output),
                child_refs[cn, ci](grad_inputs[0]),
                ctx,
            )
        else:
            comptime for j in range(Self.N):
                comptime i = Self.N - 1 - j
                comptime if i == Self.N - 1:
                    self.children[Self.N - 1].vjp[target, B, POLICY=POLICY](
                        child_refs[cn, ci](self.act[Self.N - 2]),
                        rebind[TensorImpl[ci]](grad_output),
                        child_refs[cn, ci](self.grd[Self.N - 2]),
                        ctx,
                    )
                elif i == 0:
                    self.children[0].vjp[target, B, POLICY=POLICY](
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
        for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)

    def release_buffers(mut self):
        """The inter-stage slabs, then every copy (`Module.release_buffers`)."""
        self.act.release()
        self.grd.release()
        for i in range(Self.N):
            self.children[i].release_buffers()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        for i in range(Self.N):
            self.children[i].set_attr[ATTR](value)
