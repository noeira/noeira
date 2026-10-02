"""Module — the storage-passing Module trait (N-ary, CPU + GPU).

forward/vjp take a borrowing `TensorRefs[ARITY, o]` input pack (origin-tracked,
inferred via `o`) + a single `Tensor` output/grad. The pack's origin `o` is the
ONE origin parameter on the surface — TRACKED (not the wildcard), threaded so
the referenced inputs stay live across the call. Inside, CPU views are tracked
`TileTensor`s over `.data`; GPU views are `lt_gpu` device tensors whose only
erasure is the kernel-arg `MutAnyOrigin` (the ABI). `grad_output` is `mut` so
the GPU view can be built. `ctx` is the GPU `DeviceContext` (ignored on CPU).
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from .initializer import Initializer
from .tensor import Tensor, TensorImpl
from .tensor_refs import TensorRefs
from .param import ParamVisitor, ParamWalkable
from .walkers import for_each_param_auto, zero_grad_auto
from .state import for_each_state_auto
from .amp import AMPPolicy, NoAMP
from .graph_visitor import DisplayStep


trait Module(ParamWalkable & Defaultable):
    comptime ARITY: Int
    comptime IN_DIMS: Array[Int, Self.ARITY]
    comptime OUT_DIM: Int
    # Activation-flow dtype. DEFAULTS to `DT` (fp32) so every existing leaf —
    # which never declares it — is an fp32 module unchanged (`TensorImpl[DT]` IS
    # `Tensor`, `TensorRefs[N,o,DT]` IS `TensorRefs[N,o]`). bf16-flow modules set
    # `comptime ACT_DT = bfloat16`; combinators derive theirs from their children
    # and type their inter-module buffers `TensorImpl[ACT_DT]` (activations STORED
    # at the flow dtype — the AMP memory win). [[POLICY is being subsumed by this]]
    comptime ACT_DT: DType = DT

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU (impls
        raise at runtime if missing). Params are allocated AND initialized with
        `INIT` at construction (no separate reinit pass); combinators thread
        `[target, INIT]` to their children."""
        ...

    def forward[
        target: StaticString, B: Int, o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Activations (`inputs`, `out`) are stored at `Self.ACT_DT` — fp32 by
        default (then `TensorImpl[Self.ACT_DT]` IS `Tensor`, unchanged), bf16 for
        a bf16-flow module. `POLICY` is the legacy mixed-precision policy (being
        subsumed by `ACT_DT`); matmul leaves still comptime-branch on it for the
        cast-around path until the per-leaf bf16-flow migration."""
        ...

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Default: reflection-walk every `IsParam` field of the concrete
        leaf and dispatch the visitor. Param-less leaves reflect to a no-op;
        Param-bearing leaves no longer need to override (forgetting it can no
        longer silently skip params in checkpoint/optimizer walks).
        Combinators + wrapper leaves (children are Module-typed, not IsParam)
        still override to recurse into children. `prefix` (default empty) is
        the dotted path so far; the walker composes `prefix.<param_name>`."""
        for_each_param_auto[Self, V, target](self, visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """Default: reflection-walk every `IsParam` field and zero its grad.
        Param-less leaves reflect to a no-op; combinators override to recurse."""
        zero_grad_auto[Self, target](self, ctx)

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Default: reflection-walk every `IsState` field of the concrete leaf
        and dispatch the visitor. The checkpoint path runs this right after
        `for_each_param`, so State fields (e.g. BatchNorm running stats) are
        persisted; the optimizer path (`for_each_param`) never reaches them.
        State-less leaves reflect to a no-op; combinators override to recurse."""
        for_each_state_auto[Self, V, target](self, visitor, ctx, prefix)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update this (target) module's params toward `src` (online):
        `p_self = tau·p_src + (1-tau)·p_self`. Default no-op (param-less
        leaves + leaves whose polyak isn't exercised yet); param leaves
        override on their `Param`s, combinators recurse into children."""
        pass

    def release_buffers(mut self):
        """Drop every buffer this module REBUILDS on its next forward / vjp —
        cached activations, padded copies, scratch, inter-child slabs — and
        keep what must persist: params, state (BN running stats), caches
        gated on a param version (`Linear.w_pad`), workspaces sized once
        (`Linear.sk_ws`). Combinators release theirs and recurse.

        For activation checkpointing (`Checkpointed`): a block's whole
        forward state is freed between its forward and its recomputed
        forward in the vjp. Default no-op: a module without an override keeps
        its buffers, which costs memory, never correctness."""
        pass

    def set_attr_buf[
        ATTR: StaticString
    ](mut self, buf: DeviceBuffer[DT]):
        """Point a named runtime scalar attribute at a device buffer (e.g. a
        `Scale` node's `multiplier` = SAC's on-device alpha). The `DeviceBuffer`
        carries device-residency in the type (no raw pointer). Default no-op;
        the few leaves with a device-resident scalar source override and branch
        on `ATTR`. Dispatched via `ComputeGraph.set_node_attr_buf[NAME, ATTR]`."""
        pass

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Set a named runtime scalar attribute on this module (e.g. a `Scale`
        node's `multiplier` = the moving SAC α, or an `RSample` node's
        `action_scale`). Default no-op; the few leaves with a tunable scalar
        override and branch on `ATTR`. `ComputeGraph.set_node_attr[NAME, ATTR]`
        dispatches this to the named node — the name-wired replacement for
        reaching into `graph.children[i].op.<field>`."""
        pass

    # Display surface — read by `ComputeGraph.describe` exporters. Both carry
    # defaults, so existing conformers need no change; leaves override
    # `display_label` with their type name, and containers (`Sequential`)
    # override `display_steps` to expand into their children.
    @staticmethod
    def display_label() -> String:
        """Short display name for graph exporters. Default generic; leaves
        override with their type name (e.g. "Linear")."""
        return String("module")

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        """Inner display steps for container modules — one per child, each
        `(child_label, child_out_dim)`. Default empty = atomic leaf;
        `Sequential` overrides to expand its chain."""
        return List[DisplayStep]()
