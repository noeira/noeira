"""ComputeGraph[*DECLS] — typed, NAME-wired multi-input DAG (storage design).

The keystone orchestrator for agent loss/target graphs (SAC actor-loss,
target-y, twin critic). Nodes are declared by NAME with their predecessors'
NAMES baked in (`InputSlot` / `Node` / `ExternalNode`, see graph_decl.mojo);
the graph resolves every edge to a pool-slot index at COMPILE TIME — no runtime
`List[List[Int]]` edge list to hand-build and keep in sync.

```
comptime G = ComputeGraph[
    InputSlot["s", OBS],                     # slot 0 — external input
    ExternalNode["actor", ACTOR, "s"],       # slot 1 — supplied at forward
    Node["rs", RSample[ACT], "actor"],       # slot 2
    Node["concat", Concat2[OBS, ACT], "s", "act"],  # reads "s" + "act" by name
    ...
]
g.set_input["s", B](state_tensor)            # seed inputs (by COPY)
g.forward[B, target](out, ctx, actor, q1, q2)  # externals threaded in node order
var logp = g.node_output["logp"]()           # read an intermediate by name
g.vjp[B, target](grad_out, ctx, actor, q1, q2)
var ds = g.grad_input["s"]()                 # input grad by name
```

Internals (unchanged from the index-based draft — only edge resolution moved to
comptime + inputs folded into the decl list):
  - One owning `TensorPack` POOL with one slot per decl (slot `i` ↔ decl `i`);
    ALL activations live there → every edge is same-origin (§B0), no raw
    pointers (cf. legacy `GraphNode`'s out_ptr/grad_out_ptr raw-pointer model).
  - `*EXT` externals are threaded as TRACKED `mut *externals` refs (NOT stored)
    — the load-bearing GPU fix (a wildcard pointer field disables arg-
    exclusivity and miscompiles the delegated matmul). Each `ExternalNode`
    (KIND==2 / `IsExternal`) slot dispatches to `externals[ei]` by node order.
  - Backward walks reverse-topo; each node's `vjp` scatters its grad_inputs into
    `tmp`, ACCUMULATED into the fed slots' grad buffers (fan-out summation).
    Input-slot (KIND==0) grad buffers hold the final input gradient, read via
    `grad_input[NAME]`.

CPU + GPU (the leaves + the pool-seed / grad-accumulate run on device).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import Pointer

from noeira.nn.constants import DT, TPB
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.fill import fill_dev
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor, ParamWalkable
from ..core.graph_visitor import GraphVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from .graph_module2 import TwoInputGraph
from .graph_decl import GraphDecl


def _cg_accum_kernel_rt[
    ADT: DType = DT
](
    dst: Pointer[Scalar[ADT], MutAnyOrigin],
    src: Pointer[Scalar[ADT], MutAnyOrigin],
    n_arg: Int64,
):
    """`dst[i] += src[i]` over `[0, n)` with the length at RUNTIME: one
    kernel for every fan-out edge instead of one instantiation per size."""
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + src[unsafe_offset=i]


struct ComputeGraph[*DECLS: GraphDecl](TwoInputGraph & ParamWalkable):
    # `TwoInputGraph` already implies `Defaultable & Movable & Deinitable`. It
    # is the bound `GraphModule2` needs to drive a graph generically — see
    # `combinators/graph_module2.mojo` for why `Module` cannot serve.
    comptime N = Self.DECLS.length
    comptime OUT_DIM = Self.DECLS[Self.N - 1].OUT_DIM
    comptime MAXARITY = Self._max_arity()
    # The graph's activation dtype = the output node's; all COMPUTE nodes
    # (KIND>0) share it (asserted in __init__). Input slots (KIND==0) have no
    # wrapped module — their pool slot is still stored at this ACT_DT (seeded by
    # `set_input`); their own `ACT_DT` label (DT) is irrelevant (forward no-ops).
    comptime ACT_DT = Self.DECLS[Self.N - 1].ACT_DT
    var children: Tuple[*Self.DECLS]
    var pool: TensorPack[Self.N, Self.ACT_DT]  # one slot per decl (slot i ↔ decl i)
    var gpool: TensorPack[Self.N, Self.ACT_DT]  # grads (zeroed + accumulated)
    var tmp: TensorPack[Self.MAXARITY, Self.ACT_DT]  # grad_inputs scratch
    var slot_n: List[Int]  # logical elem count per slot (forward)

    @staticmethod
    def _max_arity() -> Int:
        """The largest `ARITY` over all decls — the width of the grad-input
        scratch `tmp` pool and the per-node input-ref array. Comptime-folded
        (mirrors `_ext_before`). ≥1 so `tmp` always has at least one slot."""
        var m = 1
        comptime for j in range(Self.N):
            comptime aj = Self.DECLS[j].ARITY
            if aj > m:
                m = aj
        return m

    def __init__(out self):
        comptime assert Self.N >= 1, "ComputeGraph needs >= 1 node"
        comptime assert Self.DECLS[Self.N - 1].KIND > 0, (
            "ComputeGraph: the last decl must be a compute/external node"
            " (the graph output), not an InputSlot"
        )
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].KIND > 0:
                comptime assert Self.DECLS[i].ACT_DT == Self.ACT_DT, (
                    "ComputeGraph: all compute nodes must share one ACT_DT"
                )
        self.children = Tuple[*Self.DECLS]()
        self.pool = TensorPack[Self.N, Self.ACT_DT]()
        self.gpool = TensorPack[Self.N, Self.ACT_DT]()
        self.tmp = TensorPack[Self.MAXARITY, Self.ACT_DT]()
        self.slot_n = List[Int](length=Self.N, fill=0)


    def __init__[
        target: StaticString, INIT: Initializer
    ](out self, *, ctx: Optional[DeviceContext]) raises:
        """Build the children IN PLACE, straight from their `make` — same
        idiom and same reason as `Sequential.__init__[target, INIT]`
        (docs/COMPILE_TIME_PROFILING.md §3.2).
        """
        comptime assert Self.N >= 1, "ComputeGraph needs >= 1 node"
        comptime assert Self.DECLS[Self.N - 1].KIND > 0, (
            "ComputeGraph: the last decl must be a compute/external node"
            " (the graph output), not an InputSlot"
        )
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].KIND > 0:
                comptime assert Self.DECLS[i].ACT_DT == Self.ACT_DT, (
                    "ComputeGraph: all compute nodes must share one ACT_DT"
                )
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.children)
        )
        comptime for i in range(Self.N):
            Pointer(to=self.children[i]).unsafe_write(
                Self.DECLS[i].make[target, INIT](ctx)
            )
        self.pool = TensorPack[Self.N, Self.ACT_DT]()
        self.gpool = TensorPack[Self.N, Self.ACT_DT]()
        self.tmp = TensorPack[Self.MAXARITY, Self.ACT_DT]()
        self.slot_n = List[Int](length=Self.N, fill=0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self.__init__[target, INIT](ctx=ctx)

    @staticmethod
    def _slot_of[nm: StaticString]() -> Int:
        """The pool-slot index of the decl named `nm` (slot ↔ decl index).
        Comptime-folded; the named-edge resolution that replaces the runtime
        edge list. Returns -1 if no decl matches (a wiring bug — would index
        out of range)."""
        var s = -1
        comptime for j in range(Self.N):
            comptime if Self.DECLS[j].NAME == nm:
                s = j
        return s

    def describe[
        V: GraphVisitor
    ](mut self, graph_name: String, mut visitor: V) raises:
        """Walk the comptime topology into a pluggable `GraphVisitor` sink:
        `begin`, then per decl a `node` call (op `display_label`), one
        `node_inner` per inner display step (container ops expand their
        children), and one `edge` per input (resolved to the source decl by
        NAME), then `end`. Pure topology — no buffers, no device. Drives the
        Text / Mermaid / FusionReport exporters."""
        visitor.begin(graph_name, Self.N)
        comptime for i in range(Self.N):
            comptime kind = Self.DECLS[i].KIND
            var name = String(Self.DECLS[i].NAME)
            visitor.node(
                i,
                name,
                Self.DECLS[i].display_label_via(),
                kind,
                Self.DECLS[i].OUT_DIM,
            )
            var steps = Self.DECLS[i].display_steps_via()
            for s in range(len(steps)):
                visitor.node_inner(name, s, steps[s].label, steps[s].out_dim)
            comptime for k in range(Self.DECLS[i].ARITY):
                comptime nm_k = Self.DECLS[i].IN_NAMES_L[k]
                comptime sk = Self._slot_of[nm_k]()
                visitor.edge(
                    name,
                    String(nm_k),
                    k,
                    Self.DECLS[sk].OUT_DIM,
                )
        visitor.end()

    @staticmethod
    def _ext_before[upto: Int]() -> Int:
        """Count `ExternalNode` (KIND==2) decls in `DECLS[0:upto]` — the index
        of the threaded external arg that supplies node `upto` (externals are
        passed in node order). Comptime-folded."""
        var c = 0
        comptime for j in range(upto):
            comptime if Self.DECLS[j].KIND == 2:
                c += 1
        return c

    @staticmethod
    def _accumulate_grads_rt[
        target: StaticString
    ](
        mut gpool: TensorPack[Self.N, Self.ACT_DT],
        mut tmp: TensorPack[Self.MAXARITY, Self.ACT_DT],
        ctx: Optional[DeviceContext],
        na: Int,
        sizes: Array[Int, Self.MAXARITY],
        slots: Array[Int, Self.MAXARITY],
    ) raises:
        """Fan-out-accumulate `tmp[k]` into `gpool[slots[k]]` for `k < na`,
        every index a RUNTIME value. One instantiation per (graph, target):
        the comptime-indexed `_accumulate_grads[B, target, NA, i]` was one
        per NODE per branch — 81 copies and 81 per-size kernels on the ACT
        trainer (docs/COMPILE_TIME_PROFILING.md §3.7)."""
        for k in range(na):
            var ak = sizes[k]
            var sk = slots[k]
            comptime if target == "cpu":
                for q in range(ak):
                    gpool[sk].data[q] += tmp[k].data[q]
            else:
                var c = ctx.value()
                c.enqueue_function[_cg_accum_kernel_rt[Self.ACT_DT]](
                    gpool[sk].dev.value(),
                    tmp[k].dev.value(),
                    Int64(ak),
                    grid_dim=(ak + TPB - 1) // TPB,
                    block_dim=TPB,
                )

    @staticmethod
    @always_inline
    def _accumulate_grads[
        B: Int, target: StaticString, NA: Int, i: Int
    ](
        mut gpool: TensorPack[Self.N, Self.ACT_DT],
        mut tmp: TensorPack[Self.MAXARITY, Self.ACT_DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Node `i`'s fan-out, resolved at compile time into two small index
        arrays and handed to the runtime-indexed walk above."""
        var sizes = Array[Int, Self.MAXARITY](fill=0)
        var slots = Array[Int, Self.MAXARITY](fill=0)
        comptime for k in range(NA):
            comptime ak = B * Self.DECLS[i].IN_DIMS_L[k]
            comptime sk = Self._slot_of[Self.DECLS[i].IN_NAMES_L[k]]()
            sizes[k] = ak
            slots[k] = sk
        Self._accumulate_grads_rt[target](gpool, tmp, ctx, NA, sizes, slots)

    def set_input[
        slot_name: StaticString, B: Int
    ](mut self, mut src: TensorImpl[Self.ACT_DT],
      ctx: Optional[DeviceContext] = None) raises:
        """Seed the named input slot's pool entry with `src` (a COPY — no cached
        pointer). Call once per input before each `forward`. CPU copies element-
        wise; GPU does a device-to-device `enqueue_copy`."""
        comptime slot = Self._slot_of[slot_name]()
        comptime assert slot >= 0, "set_input: no InputSlot named " + slot_name
        comptime assert Self.DECLS[slot].KIND == 0, (
            "set_input: '" + slot_name + "' is not an InputSlot"
        )
        var n = B * Self.DECLS[slot].OUT_DIM
        self.slot_n[slot] = n
        if ctx:
            var c = ctx.value()
            self.pool[slot].ensure_gpu(c, n)
            # Copy EXACTLY `n` elements (the logical B·OUT_DIM), not buffer-to-
            # buffer. The pool buffer grows monotonically (`ensure_gpu` never
            # shrinks), so a graph SHARED across two batch sizes (e.g. DreamerV3
            # rew/con driven at B by WM and at NS=T·B by AC) leaves an NS-sized
            # pool that a plain `enqueue_copy` would mismatch against a B-sized
            # src (and vice-versa). Sub-buffer windows of `n` on both sides make
            # the copy size-exact — matching the CPU branch's element-wise `n`
            # loop — so the larger buffer's tail is simply left untouched (the
            # node's `forward[B]` reads only the first `n`).
            var src_sub = src.dev.value().create_sub_buffer[Self.ACT_DT](0, n)
            var pool_sub = self.pool[slot].dev.value().create_sub_buffer[
                Self.ACT_DT
            ](0, n)
            c.enqueue_copy(pool_sub, src_sub)
        else:
            self.pool[slot].ensure(n)
            for q in range(n):
                self.pool[slot].data[q] = src.data[q]

    def set_node_attr[
        NAME: StaticString, ATTR: StaticString
    ](mut self, value: Scalar[DT]):
        """Set a runtime scalar attribute `ATTR` on the named node's inner op
        (e.g. `set_node_attr["alogp", "multiplier"](alpha)`). Dispatches via the
        uniform `Module.set_attr[ATTR]` trait method (no-op on ops without that
        attr). The name-wired replacement for `graph.children[i].op.<field> = …`
        (which a runtime tuple subscript can't express — it erases to the
        existential)."""
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].NAME == NAME:
                self.children[i].set_attr[ATTR](value)

    def set_node_attr_buf[
        NAME: StaticString, ATTR: StaticString
    ](mut self, buf: DeviceBuffer[DT]):
        """Point a runtime scalar attribute `ATTR` on the named node's inner op
        at a device buffer (e.g. `set_node_attr_buf["alogp", "multiplier"](buf)`
        wires SAC's on-device alpha into the actor-loss Scale node). Dispatches
        via the uniform `Module.set_attr_buf[ATTR]` trait method (no-op on ops
        without that attr). The device-alpha capture-wiring boundary."""
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].NAME == NAME:
                self.children[i].set_attr_buf[ATTR](buf)

    def node_output[
        name: StaticString
    ](mut self) raises -> ref[MutAnyOrigin] TensorImpl[Self.ACT_DT]:
        """The forward output of the named node — for diagnostics / reading an
        intermediate (e.g. `log_prob`). Comptime name → slot."""
        comptime slot = Self._slot_of[name]()
        comptime assert slot >= 0, "node_output: no node named " + name
        return self.pool[slot]

    def grad_input[
        name: StaticString
    ](mut self) raises -> ref[MutAnyOrigin] TensorImpl[Self.ACT_DT]:
        """The accumulated gradient flowing back to the named input slot (read
        after `vjp`). Comptime name → grad-pool slot."""
        comptime slot = Self._slot_of[name]()
        comptime assert slot >= 0, "grad_input: no node named " + name
        return self.gpool[slot]

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
        # `externals` supplies the `ExternalNode` (IsExternal) slots, in node
        # order, as TRACKED `mut` refs threaded from the trainer (which owns the
        # actor/critics). Load-bearing: storing them as wildcard pointers
        # miscompiles the GPU matmul (see external_ref.mojo / graph_decl.mojo).
        # Inputs are pre-seeded via set_input — KIND==0 slots are skipped here.
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].KIND == 0:
                pass  # input slot — already seeded by set_input
            else:
                self.slot_n[i] = B * Self.DECLS[i].OUT_DIM
                # Build this node's input-ref array generically (any arity):
                # each slot's pool entry is fed via the §B0 wildcard subscript,
                # so all refs share MutAnyOrigin (same as the unrolled form).
                # The external branch keys its arity off `EXT[ei].ARITY` (the
                # threaded module's), which the type system does not unify with
                # the provably-equal `DECLS[i].ARITY` — hence the split.
                comptime if Self.DECLS[i].KIND == 2:
                    # External slot — dispatch to the threaded module.
                    comptime ei = Self._ext_before[i]()
                    comptime AE = EXT[ei].ARITY
                    comptime cei = EXT[ei].ACT_DT
                    var inrefs = Array[
                        Pointer[TensorImpl[Self.ACT_DT], MutAnyOrigin], AE
                    ](uninitialized=True)
                    comptime for k in range(AE):
                        comptime sk = Self._slot_of[Self.DECLS[i].IN_NAMES_L[k]]()
                        inrefs[k] = Pointer(to=self.pool[sk])
                    # pool buffers are stored at Self.ACT_DT; the external module
                    # wants its own (==Self.ACT_DT) — rebind the pack + out slot.
                    externals[ei].forward[target, B, POLICY=POLICY](
                        rebind[TensorRefs[AE, MutAnyOrigin, cei]](
                            TensorRefs[AE, MutAnyOrigin, Self.ACT_DT](inrefs^)
                        ),
                        rebind[TensorImpl[cei]](self.pool[i]),
                        ctx,
                    )
                else:
                    comptime A = Self.DECLS[i].ARITY
                    comptime ci = Self.DECLS[i].ACT_DT
                    var inrefs = Array[
                        Pointer[TensorImpl[Self.ACT_DT], MutAnyOrigin], A
                    ](uninitialized=True)
                    comptime for k in range(A):
                        comptime sk = Self._slot_of[Self.DECLS[i].IN_NAMES_L[k]]()
                        inrefs[k] = Pointer(to=self.pool[sk])
                    self.children[i].forward[target, B, POLICY=POLICY](
                        rebind[TensorRefs[A, MutAnyOrigin, ci]](
                            TensorRefs[A, MutAnyOrigin, Self.ACT_DT](inrefs^)
                        ),
                        rebind[TensorImpl[ci]](self.pool[i]),
                        ctx,
                    )
        var dN = B * Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(dN)
            for q in range(dN):
                out.data[q] = self.pool[Self.N - 1].data[q]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, dN)
            # Size-exact (`dN`) sub-buffer copy: the output node's pool buffer
            # may be larger than `dN` if this graph was previously driven at a
            # larger batch (monotone `ensure_gpu`), so copy only the logical dN.
            var src_sub = self.pool[Self.N - 1].dev.value().create_sub_buffer[
                Self.ACT_DT
            ](0, dN)
            var out_sub = out.dev.value().create_sub_buffer[Self.ACT_DT](0, dN)
            c.enqueue_copy(out_sub, src_sub)

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
        # Size + ZERO every grad slot (forward-recorded slot_n), then seed the
        # output slot's grad = grad_out.
        for k in range(Self.N):
            var dk = self.slot_n[k]
            comptime if target == "cpu":
                self.gpool[k].ensure(dk)
                for q in range(dk):
                    self.gpool[k].data[q] = 0
            else:
                self.gpool[k].ensure_gpu(ctx.value(), dk)
                # ⚠ NOT `enqueue_fill`: on Metal that is a synchronize per slot,
                # N of them per backward (`nn/core/fill.mojo`).
                fill_dev(self.gpool[k].dev.value(), dk, Scalar[Self.ACT_DT](0), ctx.value())
        var dN = B * Self.OUT_DIM
        comptime if target == "cpu":
            for q in range(dN):
                self.gpool[Self.N - 1].data[q] = grad_out.data[q]
        else:
            # Size-exact (`dN`) seed copy (see `forward`): the output grad slot
            # may be larger than dN if the graph was driven at a larger batch.
            var c = ctx.value()
            var go_sub = grad_out.dev.value().create_sub_buffer[Self.ACT_DT](
                0, dN
            )
            var gp_sub = self.gpool[Self.N - 1].dev.value().create_sub_buffer[
                Self.ACT_DT
            ](0, dN)
            c.enqueue_copy(gp_sub, go_sub)
        # Reverse-topo: each node scatters grad_inputs into tmp; the graph
        # ACCUMULATES tmp into the fed slots' grad buffers (fan-out sum). Input
        # slots (KIND==0) have no vjp — their grad slot is the accumulator.
        comptime for jj in range(Self.N):
            comptime i = Self.N - 1 - jj
            comptime if Self.DECLS[i].KIND == 0:
                pass
            else:
                # Generic reverse step (any arity): ensure the tmp grad-input
                # slots, build the forward-input + grad-input ref arrays from
                # the pool / tmp (all §B0 wildcard MutAnyOrigin), dispatch the
                # node's vjp, then fan-out-accumulate each tmp[k] into the fed
                # slot's grad buffer. External branch keys arity off
                # `EXT[ei].ARITY` (see the forward note).
                comptime if Self.DECLS[i].KIND == 2:
                    comptime ei = Self._ext_before[i]()
                    comptime AE = EXT[ei].ARITY
                    comptime cei = EXT[ei].ACT_DT
                    var firefs = Array[
                        Pointer[TensorImpl[Self.ACT_DT], MutAnyOrigin], AE
                    ](uninitialized=True)
                    var girefs = Array[
                        Pointer[TensorImpl[Self.ACT_DT], MutAnyOrigin], AE
                    ](uninitialized=True)
                    comptime for k in range(AE):
                        comptime ak = B * Self.DECLS[i].IN_DIMS_L[k]
                        comptime sk = Self._slot_of[Self.DECLS[i].IN_NAMES_L[k]]()
                        comptime if target == "cpu":
                            self.tmp[k].ensure(ak)
                        else:
                            self.tmp[k].ensure_gpu(ctx.value(), ak)
                        firefs[k] = Pointer(to=self.pool[sk])
                        girefs[k] = Pointer(to=self.tmp[k])
                    externals[ei].vjp[target, B, POLICY=POLICY](
                        rebind[TensorRefs[AE, MutAnyOrigin, cei]](
                            TensorRefs[AE, MutAnyOrigin, Self.ACT_DT](firefs^)
                        ),
                        rebind[TensorImpl[cei]](self.gpool[i]),
                        rebind[TensorRefs[AE, MutAnyOrigin, cei]](
                            TensorRefs[AE, MutAnyOrigin, Self.ACT_DT](girefs^)
                        ),
                        ctx,
                    )
                    Self._accumulate_grads[B, target, AE, i](
                        self.gpool, self.tmp, ctx
                    )
                else:
                    comptime A = Self.DECLS[i].ARITY
                    comptime ci = Self.DECLS[i].ACT_DT
                    var firefs = Array[
                        Pointer[TensorImpl[Self.ACT_DT], MutAnyOrigin], A
                    ](uninitialized=True)
                    var girefs = Array[
                        Pointer[TensorImpl[Self.ACT_DT], MutAnyOrigin], A
                    ](uninitialized=True)
                    comptime for k in range(A):
                        comptime ak = B * Self.DECLS[i].IN_DIMS_L[k]
                        comptime sk = Self._slot_of[Self.DECLS[i].IN_NAMES_L[k]]()
                        comptime if target == "cpu":
                            self.tmp[k].ensure(ak)
                        else:
                            self.tmp[k].ensure_gpu(ctx.value(), ak)
                        firefs[k] = Pointer(to=self.pool[sk])
                        girefs[k] = Pointer(to=self.tmp[k])
                    self.children[i].vjp[target, B, POLICY=POLICY](
                        rebind[TensorRefs[A, MutAnyOrigin, ci]](
                            TensorRefs[A, MutAnyOrigin, Self.ACT_DT](firefs^)
                        ),
                        rebind[TensorImpl[ci]](self.gpool[i]),
                        rebind[TensorRefs[A, MutAnyOrigin, ci]](
                            TensorRefs[A, MutAnyOrigin, Self.ACT_DT](girefs^)
                        ),
                        ctx,
                    )
                    Self._accumulate_grads[B, target, A, i](
                        self.gpool, self.tmp, ctx
                    )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(Self.DECLS[i].NAME))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(Self.DECLS[i].NAME))
            )

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Propagate a named attribute (e.g. BatchNorm "training") to every
        node's module — the graph counterpart of the combinators' set_attr."""
        comptime for i in range(Self.N):
            self.children[i].set_attr[ATTR](value)

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
        """Soft-update every node toward `src`'s matching node (target ← online).
        """
        comptime for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)
