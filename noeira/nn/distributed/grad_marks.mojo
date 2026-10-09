"""GradReady — mark the point in the backward where a module's gradients are done.

Overlapping the gradient allreduce with the backward needs to know, while the
backward runs, which part of the gradient arena is final. noeira's backward is
one composed `vjp` call with no per-layer hook, so this adds one, as a module:

    Repeat[L, GradReady[Block, ACTIVE]]     # one mark after each block's vjp

`GradReady[INNER, ACTIVE]` is a transparent wrapper (`InitWith`'s shape: same
dims, dtype, param names, walks). With `ACTIVE = False` it compiles to its
inner module exactly. With `ACTIVE = True`, after the inner `vjp` it records
two things into the process-wide `GradMarks` registry, when one is active:

  - the byte range spanned by the inner module's gradients (they sit in the
    optimizer's arena, so the range is an arena slice);
  - a device event recorded on the vjp's stream at that point.

The host does NOT launch anything mid-backward. It enqueues the whole
backward, then each bucket's allreduce on a second stream, each waiting on the
event of the mark that completes it (`DataParallel.enable_overlap`). On the
device a bucket's allreduce starts as soon as the backward passes its mark.

Rule for correctness: a wrapped module's gradients must be final when its vjp
returns. A parameter written later by another module (a tied weight) must not
be inside a wrapped module. Gradients outside every wrapped module are treated
as final only at the end of the backward.

Events are pooled by mark slot and reused every step. That is safe because
CUDA's wait captures the event's most recent record at the time the wait is
enqueued, and the host enqueues step s's waits before step s+1 re-records.

Unit hooks (ZeRO-2/3, `zero_sharded.mojo`). The same wrapper is a sharding
unit: a sharded driver registers one callback (`UnitHooks`) that the wrapper
calls before its forward, before its vjp and after its vjp, with its unit
index and rank. Those are assigned by an ASSIGNING parameter walk: while
`UnitHooks.assigning` is set, each wrapper takes the next unit index and
reports it as `cur_unit` to the visitor for the parameters it holds, so one
walk gives the driver both the unit of every parameter and numbered
wrappers. With no driver registered the wrapper behaves as above.
"""

from std.ffi import _get_global_or_null, external_call
from std.sys import size_of, has_nvidia_gpu_accelerator
from std.memory.alloc import Layout as AllocLayout
from max.gpu.host import DeviceContext, DeviceBuffer, DeviceEvent

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.graph_visitor import DisplayStep


comptime _REGISTRY = "NOEIRA_GRAD_MARKS"

comptime MARK_EVENTS = has_nvidia_gpu_accelerator()
"""Marks record a device event at their point of the backward. Off on Metal,
where MAX implements no `ctx.stream()`: there a bucket waits for the whole
backward (no overlap, same results)."""


@fieldwise_init
struct GradMark(Copyable, Movable):
    var lo: Int
    """Byte address of the module's first gradient element."""
    var hi: Int
    """One past its last gradient byte."""
    var slot: Int
    """Index into `GradMarks.events`; -1 when no event was recorded."""


struct GradMarks(Movable):
    """Marks recorded during one backward, in enqueue order (rank by rank)."""

    var active: Bool
    var marks: List[GradMark]
    var events: List[DeviceEvent]
    var event_dev: List[Int]
    """Device id each pooled event was created on."""

    def __init__(out self):
        self.active = False
        self.marks = List[GradMark]()
        self.events = List[DeviceEvent]()
        self.event_dev = List[Int]()

    def begin(mut self):
        self.marks.clear()
        self.active = True

    def end(mut self):
        self.active = False

    def mark(mut self, lo: Int, hi: Int, ctx: DeviceContext) raises:
        comptime if not MARK_EVENTS:
            # No per-point event (Metal has no `ctx.stream()`): slot -1, and
            # the bucket waits for the whole backward instead.
            self.marks.append(GradMark(lo, hi, -1))
            return
        var slot = len(self.marks)
        var dev = Int(ctx.id())
        if slot == len(self.events):
            self.events.append(ctx.create_event())
            self.event_dev.append(dev)
        elif self.event_dev[slot] != dev:
            self.events[slot] = ctx.create_event()
            self.event_dev[slot] = dev
        ctx.stream().record_event(self.events[slot])
        self.marks.append(GradMark(lo, hi, slot))

    def event(self, slot: Int) -> DeviceEvent:
        return self.events[slot]


comptime GradMarksPtr = Pointer[GradMarks, UntrackedOrigin[mut=True]]


def grad_marks() raises -> GradMarksPtr:
    """The process-wide registry, created on first use (process lifetime,
    like the per-context BLAS handles)."""
    var g = _get_global_or_null(_REGISTRY)
    if not g:
        var p = alloc(AllocLayout[GradMarks].single()).unsafe_leak()
        p.unsafe_write(GradMarks())
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_REGISTRY), p.unsafe_bitcast[NoneType]()
        )
        g = _get_global_or_null(_REGISTRY)
    return g.value().unsafe_bitcast[GradMarks]()


# ── unit hooks (sharded drivers) ─────────────────────────────────────────────

comptime HOOK_PRE_FORWARD = 0
comptime HOOK_PRE_VJP = 1
comptime HOOK_POST_VJP = 2

comptime UnitHookFn = def(Int, Int, Int, Int, DeviceContext) raises thin
"""`(state, event, unit, rank, ctx)`: `state` is the driver's address."""


def _no_hook(
    state: Int, event: Int, unit: Int, rank: Int, ctx: DeviceContext
) raises:
    pass


comptime _UNIT_HOOKS = "NOEIRA_UNIT_HOOKS"


struct UnitHooks(Movable):
    """Process-wide: the sharded driver whose hooks are live, and the state
    of an assigning walk."""

    var assigning: Bool
    var assign_rank: Int
    var next_unit: Int
    var cur_unit: Int
    """Unit whose parameters the walk is visiting; -1 outside every wrapper."""
    var active: Bool
    var state: Int
    var call: UnitHookFn

    def __init__(out self):
        self.assigning = False
        self.assign_rank = -1
        self.next_unit = 0
        self.cur_unit = -1
        self.active = False
        self.state = 0
        self.call = _no_hook

    def begin_assign(mut self, rank: Int):
        self.assigning = True
        self.assign_rank = rank
        self.next_unit = 0
        self.cur_unit = -1

    def end_assign(mut self):
        self.assigning = False
        self.cur_unit = -1

    def register(mut self, state: Int, call: UnitHookFn) raises:
        if self.active:
            raise Error("UnitHooks: another sharded driver is active")
        self.state = state
        self.call = call
        self.active = True

    def unregister(mut self):
        self.active = False
        self.state = 0
        self.call = _no_hook


comptime UnitHooksPtr = Pointer[UnitHooks, UntrackedOrigin[mut=True]]


def unit_hooks() raises -> UnitHooksPtr:
    """The process-wide unit hook registry, created on first use."""
    var g = _get_global_or_null(_UNIT_HOOKS)
    if not g:
        var p = alloc(AllocLayout[UnitHooks].single()).unsafe_leak()
        p.unsafe_write(UnitHooks())
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_UNIT_HOOKS), p.unsafe_bitcast[NoneType]()
        )
        g = _get_global_or_null(_UNIT_HOOKS)
    return g.value().unsafe_bitcast[UnitHooks]()


struct GradSpan(ParamVisitor):
    """Byte range spanned by the device gradients a walk visits."""

    var lo: Int
    var hi: Int

    def __init__(out self):
        self.lo = -1
        self.hi = -1

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if not grad.dev:
            return
        var a = Int(grad.dev.value().unsafe_ptr())
        var b = a + N * size_of[Scalar[DT]]()
        if self.lo < 0 or a < self.lo:
            self.lo = a
        if b > self.hi:
            self.hi = b


struct GradReady[INNER: Module, ACTIVE: Bool = True](Module):
    comptime ARITY = Self.INNER.ARITY
    comptime IN_DIMS = Self.INNER.IN_DIMS
    comptime OUT_DIM = Self.INNER.OUT_DIM
    comptime ACT_DT = Self.INNER.ACT_DT

    var inner: Self.INNER
    var unit: Int
    """Unit index (forward order), set by an assigning walk; -1 before."""
    var rank: Int
    """Rank of the replica this wrapper belongs to, set with `unit`."""

    def __init__(out self):
        self.inner = Self.INNER()
        self.unit = -1
        self.rank = -1

    def __init__[
        target: StaticString, INIT: Initializer
    ](out self, *, ctx: Optional[DeviceContext]) raises:
        self.inner = Self.INNER.make[target, INIT](ctx)
        self.unit = -1
        self.rank = -1

    def _hook(self, event: Int, ctx: DeviceContext) raises:
        var h = unit_hooks()
        if h[].active:
            if self.unit < 0:
                raise Error(
                    "GradReady: unit not assigned (the sharded driver's"
                    " assigning walk did not reach this wrapper)"
                )
            h[].call(h[].state, event, self.unit, self.rank, ctx)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self.__init__[target, INIT](ctx=ctx)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if Self.ACTIVE and target != "cpu":
            if ctx:
                self._hook(HOOK_PRE_FORWARD, ctx.value())
        self.inner.forward[target, B, POLICY=POLICY](inputs, out, ctx)

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
        comptime if Self.ACTIVE and target != "cpu":
            if ctx:
                self._hook(HOOK_PRE_VJP, ctx.value())
        self.inner.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )
        comptime if Self.ACTIVE and target != "cpu":
            if not ctx:
                return
            if unit_hooks()[].active:
                self._hook(HOOK_POST_VJP, ctx.value())
                return
            var reg = grad_marks()
            if not reg[].active:
                return
            var span = GradSpan()
            self.inner.for_each_param[target](span, ctx)
            if span.lo >= 0:
                reg[].mark(span.lo, span.hi, ctx.value())

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime if Self.ACTIVE:
            var h = unit_hooks()
            if h[].assigning:
                if h[].cur_unit != -1:
                    raise Error("GradReady: nested units are not supported")
                self.unit = h[].next_unit
                self.rank = h[].assign_rank
                h[].next_unit += 1
                h[].cur_unit = self.unit
                self.inner.for_each_param[target](visitor, ctx, prefix)
                h[].cur_unit = -1
                return
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

    def release_buffers(mut self):
        self.inner.release_buffers()

    def set_attr_buf[ATTR: StaticString](mut self, buf: DeviceBuffer[DT]):
        self.inner.set_attr_buf[ATTR](buf)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)

    @staticmethod
    def display_label() -> String:
        return Self.INNER.display_label()

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        return Self.INNER.display_steps()
